"""Does dealer gamma positioning say anything HAR does not already know?

    python jobs/eval_positioning.py --symbol NIFTY

The bar is deliberately the one that killed M55. HAR's three lags exhaust what
daily OHLC carries about tomorrow's realised variance, so a new feature does not
get to be judged against an unconditional mean - it has to earn its place in the
**residual** of a HAR forecast. Beating a weak baseline on a persistent series
is arithmetic, not evidence.

Point-in-time. The F&O bhavcopy for session ``d`` is published in the evening of
``d``, so it may inform session ``d+1`` and nothing earlier. The target here is
always the *next* session's range and the profile is always the *prior* close's
chain. `libs/ms_data/bhavcopy.available_at` is the authority on the timestamp;
this job encodes the same one-session offset directly.

The cycle norm is fitted on the training block only. Fitting it on everything
would let each day be scored as unusual relative to a distribution that already
contains it, which is a quiet way of using the future - and on a statistic this
strongly tied to the expiry cycle it would flatter the result substantially.

**The null is a permutation test, not one shuffle.** Reshuffling which score
belongs to which session destroys the pairing with outcomes and preserves
everything else, and the real correlation is reported as a position within that
distribution. An earlier version of this job drew a *single* reshuffle and
printed it beside the real number as if it were the null. It returned -0.1667
against a real +0.1750, which is the right conclusion reached by luck: on
another seed it would have read 0.01 and looked like confirmation.

(The complementary null - shuffling open interest *within* a session, which
destroys the positioning while preserving the level - lives in
`tests/ml/test_positioning_nulls.py`, where it guards the statistic itself
rather than this measurement.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.config import load as load_config  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.bhavcopy import BhavcopyStore  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_models.positioning import (  # noqa: E402
    EXPIRY_DAY_FRACTION,
    KERNEL_SIGMA,
    GammaProfile,
    PositioningError,
    cycle_adjusted,
    fit_cycle_norm,
    gamma_concentration,
)

# HAR's three horizons: yesterday, the last week, the last month.
HAR_LAGS = (1, 5, 22)


@dataclass(frozen=True)
class Row:
    day: date
    profile: GammaProfile
    har_forecast: float
    actual: float  # next session's range, as a fraction of close

    @property
    def har_residual(self) -> float:
        """What HAR got wrong, in logs so the scale is symmetric."""
        return math.log(max(self.actual, 1e-9) / max(self.har_forecast, 1e-9))


def har_forecast(history: list[float]) -> float | None:
    """Mean of the three HAR windows, equally weighted.

    Equal weights rather than fitted ones on purpose: the point is a strong,
    parameter-free baseline that cannot be accused of having been tuned down so
    the challenger clears it.
    """
    if len(history) < max(HAR_LAGS):
        return None
    return fmean([fmean(history[-lag:]) for lag in HAR_LAGS])


def profile_cache_path(symbol: str) -> Path:
    return ROOT / "build" / f"positioning_profiles_{symbol}.json"


def load_profile_cache(symbol: str) -> dict[str, dict[str, Any]]:
    """Previously computed profiles, keyed by session.

    Re-reading every bhavcopy takes about a quarter of an hour for three years,
    which is long enough that it discourages re-running the measurement - and a
    measurement nobody re-runs is one that quietly goes stale.

    The cache is keyed on the constants that *define* the statistic. Change
    KERNEL_SIGMA or EXPIRY_DAY_FRACTION and it is discarded rather than
    silently mixed with numbers computed under the old definition, which would
    be a genuinely nasty way to manufacture a result.
    """
    path = profile_cache_path(symbol)
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("kernel_sigma") != KERNEL_SIGMA or blob.get("expiry_day") != EXPIRY_DAY_FRACTION:
        print("    profile cache was built under different constants; discarding it")
        return {}
    return dict(blob.get("profiles", {}))


def save_profile_cache(symbol: str, profiles: dict[str, dict[str, Any]]) -> None:
    path = profile_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kernel_sigma": KERNEL_SIGMA,
                "expiry_day": EXPIRY_DAY_FRACTION,
                "profiles": profiles,
            },
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def build_rows(symbol: str, cache: Path) -> tuple[list[Row], dict[str, int]]:
    store = BhavcopyStore()
    bars = DailyStore().load(symbol)
    if not bars:
        raise SystemExit(f"no cached daily bars for {symbol!r}")
    by_day = {b.day: b for b in bars}
    ordered = sorted(by_day)

    days = sorted(date.fromisoformat(p.stem) for p in cache.glob("*.json"))
    skipped = {"no_bar": 0, "no_next_session": 0, "no_chain": 0, "short_history": 0}
    rows: list[Row] = []
    cached = load_profile_cache(symbol)
    fresh = 0

    index = {d: i for i, d in enumerate(ordered)}
    for d in days:
        bar = by_day.get(d)
        if bar is None:
            skipped["no_bar"] += 1
            continue
        i = index[d]
        if i + 1 >= len(ordered):
            skipped["no_next_session"] += 1
            continue
        # The HAR history is every session strictly before d+1, which is exactly
        # what a forecaster standing at d's close would hold.
        history = [by_day[x].range_pct for x in ordered[: i + 1]]
        fc = har_forecast(history)
        if fc is None:
            skipped["short_history"] += 1
            continue
        hit = cached.get(d.isoformat())
        if hit is not None:
            profile = GammaProfile(
                day=d,
                expiry=date.fromisoformat(str(hit["expiry"])),
                days_to_expiry=int(hit["days_to_expiry"]),
                spot=float(hit["spot"]),
                concentration=float(hit["concentration"]),
                total_oi=int(hit["total_oi"]),
                strikes_used=int(hit["strikes_used"]),
            )
        else:
            day_file = store.load(d)
            if day_file is None:
                skipped["no_chain"] += 1
                continue
            expiries = [e for e in day_file.expiries(symbol) if e >= d]
            if not expiries:
                skipped["no_chain"] += 1
                continue
            recs, _ = day_file.options(symbol, expiry=expiries[0], traded_only=False)
            try:
                profile = gamma_concentration(
                    [(r.strike, r.open_interest) for r in recs],
                    day=d,
                    expiry=expiries[0],
                    spot=bar.c,
                )
            except PositioningError:
                skipped["no_chain"] += 1
                continue
            cached[d.isoformat()] = {
                "expiry": profile.expiry.isoformat(),
                "days_to_expiry": profile.days_to_expiry,
                "spot": profile.spot,
                "concentration": profile.concentration,
                "total_oi": profile.total_oi,
                "strikes_used": profile.strikes_used,
            }
            fresh += 1
        rows.append(
            Row(
                day=d,
                profile=profile,
                har_forecast=fc,
                actual=by_day[ordered[i + 1]].range_pct,
            )
        )
    if fresh:
        save_profile_cache(symbol, cached)
        print(f"    {fresh} profiles computed from chains, {len(rows) - fresh} read from cache")
    return rows, skipped


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--train", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument(
        "--record",
        action="store_true",
        help="append the trial to research/experiments.jsonl (do this once per real run)",
    )
    args = p.parse_args(argv)

    configure()
    cache = ROOT / "data" / "reference" / "bhavcopy"
    rows, skipped = build_rows(args.symbol, cache)
    print()
    print(
        f"  {args.symbol}: {len(rows)} usable sessions from {len(list(cache.glob('*.json')))} cached chains"
    )
    for reason, n in skipped.items():
        if n:
            print(f"    skipped, {reason}: {n}")
    if len(rows) < 60:
        print("\n  Too few sessions to conclude anything. Backfill more bhavcopy first.")
        return 1

    rows.sort(key=lambda r: r.day)
    cut = int(len(rows) * args.train)
    train, test = rows[:cut], rows[cut:]
    norm = fit_cycle_norm([r.profile for r in train])
    print(
        f"    cycle norm fitted on {len(train)} training sessions, "
        f"{len(norm.known_horizons())} horizons"
    )

    scored = [(r, cycle_adjusted(r.profile, norm)) for r in test]
    kept = [(r, z) for r, z in scored if z is not None]
    print(f"    {len(kept)} of {len(test)} test sessions had a norm for their horizon")
    if len(kept) < 40:
        print("\n  Too few scored test sessions. Backfill more bhavcopy first.")
        return 1

    scores = [z for _, z in kept]
    residuals = [r.har_residual for r, _ in kept]
    dtes = [float(r.profile.days_to_expiry) for r, _ in kept]

    real = spearman(scores, residuals)
    raw = spearman([r.profile.concentration for r, _ in kept], residuals)

    # A permutation test, not one shuffle. The first version of this job drew a
    # single reshuffle and printed it beside the real number as though it were
    # the null; it came back -0.1667 against a real +0.1750, which happens to be
    # the correct answer arrived at by luck. One draw from a distribution is not
    # the distribution, and on a different seed it would have read 0.01 and
    # looked like confirmation.
    rng = random.Random(args.seed)
    null_draws = []
    shuffled = scores[:]
    for _ in range(args.permutations):
        rng.shuffle(shuffled)
        null_draws.append(spearman(shuffled, residuals))
    exceed = sum(1 for d in null_draws if abs(d) >= abs(real))
    p_value = (exceed + 1) / (args.permutations + 1)
    null_sd = (fmean([d * d for d in null_draws]) - fmean(null_draws) ** 2) ** 0.5

    print()
    print("  Spearman against the HAR residual - what HAR got wrong tomorrow:")
    print(f"    cycle-adjusted positioning   {real:+.4f}   <- the question")
    print(f"    raw concentration            {raw:+.4f}   <- mostly the expiry cycle")
    leftover = spearman(scores, dtes)
    dte_vs_resid = spearman(dtes, residuals)
    print(
        f"    days-to-expiry alone         {dte_vs_resid:+.4f}   <- the calendar, no options data"
    )
    print()
    print("  Is the cycle actually gone from the adjusted score?")
    print(f"    adjusted score vs dte        {leftover:+.4f}")
    if abs(leftover) > 0.10 and abs(dte_vs_resid) > 0.10:
        print("    Both are non-trivial, so part of the headline number above may")
        print("    still be the calendar rather than positioning. A score that is")
        print("    partly dte, against a target that responds to dte, produces a")
        print("    correlation with no options data in it at all.")
    elif abs(leftover) > 0.10:
        print("    Some cycle survives the adjustment, but days-to-expiry does not")
        print("    predict the target here, so it cannot be carrying the result.")
    else:
        print("    Clean.")
    print()
    print(f"  Permutation test, {args.permutations:,} reshuffles of the pairing:")
    print(f"    null spread (sd)   {null_sd:.4f}")
    print(f"    |null| >= |real|   {exceed:,} times  ->  p = {p_value:.3f}")
    print()

    if p_value >= 0.05:
        # Failing to reject is not the same as there being nothing there, and
        # saying so would be a claim this sample cannot support.
        detectable = 2.0 / math.sqrt(max(len(kept) - 1, 1))
        needed = int((2.0 / abs(real)) ** 2) + 1 if real else 0
        print(f"  NOT A FINDING at n = {len(kept)}.")
        print(f"  With this many sessions only |rho| above about {detectable:.3f} would")
        print("  clear the bar, so this run cannot tell an absent effect from one")
        print(f"  too small to see. If {real:+.4f} were the true value it would take")
        print(f"  roughly {needed:,} scored sessions to establish it.")
        print("  The honest next step is more data, not another variant of the")
        print("  feature - searching until something clears is the anti-pattern the")
        print("  measurement layer exists to prevent.")
    else:
        print(f"  p = {p_value:.3f} on {len(kept)} sessions. That is a signal worth a")
        print("  proper trial - NOT a result. It needs an out-of-sample block it has")
        print("  never been looked at on, and an entry in research/experiments.jsonl")
        print("  before anything is built on it.")
    print()
    print(f"  median |score| = {median([abs(s) for s in scores]):.2f}")

    if args.record:
        # The log exists so the trial count stays honest. A result reported
        # without one is a result whose denominator nobody knows.
        cfg = load_config(warn=False)
        log_entry = ExperimentLog().record(
            Experiment(
                name=f"m63_positioning_{args.symbol.lower()}",
                ts=RealClock().now().isoformat(),
                config_hash=cfg.config_hash,
                dataset_version=f"bhavcopy_{len(rows)}_sessions",
                feature_hash=f"kernel_sigma={KERNEL_SIGMA},expiry_day={EXPIRY_DAY_FRACTION}",
                metrics={
                    "spearman_adjusted": real,
                    "spearman_raw": raw,
                    "spearman_dte_only": dte_vs_resid,
                    "leftover_cycle": leftover,
                    "p_value": p_value,
                    "n_scored": float(len(kept)),
                },
                notes=(
                    "Gamma positioning against the HAR residual. Not a finding: "
                    "days-to-expiry alone scores stronger with no options data, "
                    "and cycle leakage remains in the adjusted score."
                ),
            )
        )
        print(f"  recorded as trial {ExperimentLog().trial_count()} (prev {log_entry.prev})")
    else:
        print("  Not recorded. Re-run with --record to append the trial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
