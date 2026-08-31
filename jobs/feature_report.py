"""M44 verify - what the feature layer actually produced, and whether it moved.

    python jobs/feature_report.py --null-rates --psi     the module's verify command
    python jobs/feature_report.py --date 2026-08-31 --null-rates
    python jobs/feature_report.py --null-rates --strict  exit non-zero on a breach

Two questions, and they fail in opposite directions.

**Null rates** catch a feature that is not being computed. A feature stuck at
None is the quietest failure in the layer: the vector still has the right shape,
the model still trains, and the column contributes nothing. Nobody notices,
because nothing raised.

**PSI** catches the opposite - a feature that is being computed perfectly and
now means something different from what it meant when the model was fitted.
Population stability index over quantile bins taken from the reference period;
the conventional reading is < 0.10 stable, 0.10-0.25 a shift worth knowing
about, > 0.25 a distribution the model was not trained on.

**A null rate is not automatically a fault, and this report refuses to pretend
otherwise.** Group C reads the option chain. On a day with no chain recorded,
every one of those features is None for the whole session and that is the
correct answer to the question asked - the input was not there. Reporting it as
a failure would train whoever reads this to ignore the failures. So a day's
chain count is checked first, and features that had no input to work from are
marked `no input` and left out of the gate. What the gate is for is the other
case: a chain was recorded, and the feature still produced nothing.

The warm-up exclusion is a session-level parameter rather than per-feature. A
feature declaring a window longer than ``--warmup`` will show a null rate that
is warm-up rather than fault, which is why the window is printed beside it.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_data.recorder import DATA_ROOT
from libs.ms_data.replay import recorded_days
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.registry import REGISTRY
from libs.ms_features.sources import load_day

# A bin that is empty on one side makes the log ratio infinite. Flooring both
# sides is the standard remedy and it caps how large a single empty bin can push
# the index - which is a limitation of the statistic, not of this floor: PSI
# cannot distinguish "this bin emptied" from "this bin nearly emptied".
PSI_FLOOR = 1e-4
PSI_SHIFT = 0.10
PSI_BREAK = 0.25

# Cross-asset series live in this reserved instrument range and share the Candle
# schema - see `config/crossasset.yaml`. Group D reads them, so a day with none
# recorded is a day group D had no input on, exactly as a chainless day is for
# group C.
CROSS_ASSET_ID_FLOOR = 1000

# Which recorded stream each group needs before its nulls mean anything. A group
# absent from this map is always gated: it reads the session's own candles,
# which a day without them would not have got this far without.
GROUP_INPUTS = {"C": "chain snapshots", "D": "cross-asset series"}


class DayReport:
    def __init__(self, day: date, instrument_id: int) -> None:
        self.day = day
        self.instrument_id = instrument_id
        self.instants = 0
        self.chain_snapshots = 0
        self.cross_asset_series = 0
        self.values: dict[str, list[float | None]] = defaultdict(list)

    @property
    def had_chain(self) -> bool:
        return self.chain_snapshots > 0

    def null_rate(self, name: str) -> float | None:
        series = self.values.get(name)
        if not series:
            return None
        return sum(1 for v in series if v is None) / len(series)

    def has_no_input(self, name: str) -> bool:
        """True when the feature's data source was absent for the whole day.

        It is a statement about the *recording*, and it must never be inferred
        from the values alone - a feature that is None everywhere because it is
        broken looks identical from the values, and that case is exactly what
        the gate is for.

        Keyed by group rather than hard-coded to the chain. When group D was
        added, every one of its four features came up FAIL at 100% on a day with
        no cross-asset collector running - which is the report failing on a
        condition nobody can fix, the thing its own docstring says makes a gate
        get ignored.
        """
        group = REGISTRY.get(name).group
        if group == "C":
            return not self.had_chain
        if group == "D":
            return self.cross_asset_series == 0
        return False


def build_day(
    day: date, instrument_id: int, *, root: Path, warmup_minutes: int
) -> DayReport | None:
    """Compute every registered feature at every completed minute of one day."""
    loaded = load_day(day, root=root)
    report = DayReport(day, instrument_id)
    report.chain_snapshots = sum(1 for c in loaded.chains if c.underlying_id == instrument_id)
    report.cross_asset_series = len(
        {c.instrument_id for c in loaded.candles if c.instrument_id >= CROSS_ASSET_ID_FLOOR}
    )

    bars = sorted(
        (c for c in loaded.candles if c.instrument_id == instrument_id and c.tf == 1),
        key=lambda c: c.ts,
    )
    if not bars:
        return None

    source = loaded.source()
    open_ts = bars[0].ts
    names = REGISTRY.names()
    for bar in bars:
        # One minute past the bar's stamp: the bar is complete, and the context
        # excludes the bucket containing `as_of` anyway. Asking at the bar's own
        # timestamp would silently drop it from every window.
        as_of = bar.ts + timedelta(minutes=1)
        if (as_of - open_ts) < timedelta(minutes=warmup_minutes):
            continue
        values = REGISTRY.compute(source, instrument_id, as_of, underlying_id=instrument_id)
        report.instants += 1
        for name in names:
            report.values[name].append(values.get(name))
    return report if report.instants else None


def psi(reference: list[float], current: list[float], *, bins: int = 10) -> float | None:
    """Population stability index of ``current`` against ``reference``.

    Bin edges are the reference quantiles, so the reference is uniform across
    bins by construction and any departure is the current period's. None when
    either side is too small to bin, or when the reference is constant - a
    constant reference has no distribution to be stable against, and a number
    computed from one would be an artefact of the floor rather than a measurement.
    """
    ref = sorted(reference)
    if len(ref) < 50 or len(current) < 50:
        return None
    # Distinct values, not distinct edges. A constant reference still yields one
    # quantile edge, which bins into two buckets - one of them empty on both
    # sides - and the index comes out a confident 0.000 for a feature that never
    # moved. The emptiness of the guard was the bug: `if not edges` never fired.
    if len(set(ref)) < 2:
        return None
    edges = sorted({ref[(i * len(ref)) // bins] for i in range(1, bins)})
    if not edges:
        return None

    def histogram(values: list[float]) -> list[int]:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
        return counts

    ref_counts, cur_counts = histogram(ref), histogram(current)
    ref_total, cur_total = sum(ref_counts), sum(cur_counts)
    total = 0.0
    for a, b in zip(ref_counts, cur_counts, strict=True):
        p = max(a / ref_total, PSI_FLOOR)
        q = max(b / cur_total, PSI_FLOOR)
        total += (q - p) * math.log(q / p)
    return total


def render_null_rates(report: DayReport, threshold: float) -> tuple[str, list[str]]:
    lines = [
        f"  {report.day}  instrument {report.instrument_id}  "
        f"{report.instants} instants, {report.chain_snapshots} chain snapshots, "
        f"{report.cross_asset_series} cross-asset series",
        f"  {'feature':<28} {'win':>4} {'null%':>7}  status",
    ]
    breaches: list[str] = []
    for definition in REGISTRY.all():
        name = definition.name
        rate = report.null_rate(name)
        if rate is None:
            continue
        if report.has_no_input(name):
            status = f"no input: {GROUP_INPUTS[definition.group]} were not recorded"
        elif rate > threshold:
            status = "FAIL"
            breaches.append(f"{name} null {rate:.1%} on {report.day}")
        else:
            status = "ok"
        lines.append(f"  {name:<28} {definition.window_minutes:>4} {rate:>6.1%}  {status}")
    return "\n".join(lines), breaches


def render_psi(reports: list[DayReport], threshold: float) -> tuple[str, list[str]]:
    """Compare the most recent day against every day before it, pooled."""
    if len(reports) < 2:
        return (
            "  PSI needs two recorded days to compare and there is one. Nothing is\n"
            "  reported rather than comparing the day against itself.",
            [],
        )
    *earlier, latest = reports
    lines = [
        f"  reference: {earlier[0].day}..{earlier[-1].day} ({len(earlier)} days)  "
        f"current: {latest.day}",
        f"  {'feature':<28} {'PSI':>8}  reading",
    ]
    breaches: list[str] = []
    for definition in REGISTRY.all():
        name = definition.name
        reference = [v for r in earlier for v in r.values.get(name, []) if v is not None]
        current = [v for v in latest.values.get(name, []) if v is not None]
        score = psi(reference, current)
        if score is None:
            continue
        if score > PSI_BREAK:
            reading = "MOVED - outside anything the model was fitted on"
        elif score > PSI_SHIFT:
            reading = "shifted"
        else:
            reading = "stable"
        if score > threshold:
            breaches.append(f"{name} PSI {score:.3f} on {latest.day}")
        lines.append(f"  {name:<28} {score:>8.3f}  {reading}")
    return "\n".join(lines), breaches


def main(argv: list[str] | None = None) -> int:
    configure()
    p = argparse.ArgumentParser(description="Feature null rates and stability (M44)")
    p.add_argument("--null-rates", action="store_true", help="per-feature null rate per day")
    p.add_argument("--psi", action="store_true", help="stability of the latest day vs the rest")
    p.add_argument("--date", help="one day (YYYY-MM-DD); default every recorded day")
    p.add_argument("--instrument", type=int, default=1, help="instrument id (default 1, NIFTY)")
    p.add_argument("--root", default=str(DATA_ROOT), help="recorded data root")
    p.add_argument(
        "--warmup",
        type=int,
        default=35,
        help="minutes after the open to exclude; the longest strict window is 31",
    )
    p.add_argument("--null-threshold", type=float, default=0.01, help="null rate gate (0.01 = 1%%)")
    p.add_argument("--psi-threshold", type=float, default=PSI_BREAK, help="PSI gate")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any breach")
    args = p.parse_args(argv)

    if not args.null_rates and not args.psi:
        args.null_rates = True  # the bare invocation should still say something

    root = Path(args.root)
    days = [date.fromisoformat(args.date)] if args.date else recorded_days(root, "candles")
    if not days:
        print(f"  no recorded days under {root} - nothing to report on.")
        print("  record a session first: python -m services.ingest_market --source angel --chain")
        return 0

    reports: list[DayReport] = []
    for day in days:
        report = build_day(day, args.instrument, root=root, warmup_minutes=args.warmup)
        if report is None:
            print(f"  {day}: no candles for instrument {args.instrument} - skipped")
            continue
        reports.append(report)

    if not reports:
        print("  nothing computable in the recorded days.")
        return 0

    breaches: list[str] = []
    if args.null_rates:
        print("\n  NULL RATES")
        for report in reports:
            text, found = render_null_rates(report, args.null_threshold)
            print(text)
            print()
            breaches.extend(found)

    if args.psi:
        print("\n  STABILITY (PSI)")
        text, found = render_psi(reports, args.psi_threshold)
        print(text)
        print()
        breaches.extend(found)

    if breaches:
        print(f"  {len(breaches)} breach(es):")
        for line in breaches:
            print(f"    - {line}")
        return 1 if args.strict else 0
    print(f"  {len(reports)} day(s) reported, nothing over threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
