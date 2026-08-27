"""M19 - Data quality gates.

Every check from the blueprint's quality table, in one place, each returning a
verdict *and a number*. A check that only says "bad" cannot be tuned, trended,
or argued with.

Two rules this module is built around:

**A FAIL blocks downstream work.** Research on a day with 200 candles instead of
375 produces a result that looks like a finding and is an artefact. The gate
exists so that never silently happens.

**Every check is tested by injecting the corruption it looks for.** A quality
check nobody has watched fail is not a check, it is a comment. The tests in
``tests/unit/test_quality.py`` take a known-good day and break it in one specific
way each time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, OptionType, RawNewsItem, Tick

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
QC_DIR = REPO_ROOT / "build" / "qc"


class Verdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"  # nothing to check - not the same as passing


@dataclass
class Check:
    name: str
    verdict: Verdict
    value: float
    threshold: float | None = None
    detail: str = ""

    @property
    def blocking(self) -> bool:
        return self.verdict is Verdict.FAIL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def line(self) -> str:
        thr = f" (threshold {self.threshold:g})" if self.threshold is not None else ""
        return f"{self.verdict.value:<4} {self.name:<28} {self.value:>12,.4g}{thr}  {self.detail}"


@dataclass
class QualityReport:
    day: date
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        METRICS.gauge("qc_check_value", check.value, check=check.name)
        METRICS.inc("qc_verdict_total", check=check.name, verdict=check.verdict.value)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is Verdict.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is Verdict.WARN]

    @property
    def ok(self) -> bool:
        """True when nothing blocks. Warnings are visible, not blocking."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "ok": self.ok,
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
        }

    def save(self, directory: Path = QC_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.day.isoformat()}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def render(self) -> str:
        head = f"  data quality - {self.day}  [{'OK' if self.ok else 'BLOCKED'}]"
        return head + "\n" + "\n".join("  " + c.line() for c in self.checks)


# --------------------------------------------------------------------------- #
# Market data checks
# --------------------------------------------------------------------------- #
def check_tick_sequence_gaps(ticks: Sequence[Tick]) -> Check:
    """Any gap at all is worth surfacing.

    A detected gap is an annotation on one minute of data; an undetected one is a
    candle that quietly lies about what happened.
    """
    last: dict[int, int] = {}
    missing = 0
    seen = 0
    for t in ticks:
        if t.seq is None:
            continue
        seen += 1
        prev = last.get(t.instrument_id)
        last[t.instrument_id] = t.seq
        if prev is not None and t.seq > prev + 1:
            missing += t.seq - prev - 1
    if not seen:
        return Check(
            "tick_sequence_gaps",
            Verdict.SKIP,
            0.0,
            detail="feed supplies no sequence numbers - falling back to candle counts",
        )
    verdict = Verdict.PASS if missing == 0 else Verdict.WARN
    return Check("tick_sequence_gaps", verdict, float(missing), 0.0, f"{seen:,} sequenced ticks")


def check_candle_count(
    candles: Sequence[Candle], calendar: SessionCalendar, *, tf: int = 1
) -> Check:
    """A continuous session must contain exactly the expected number of candles.

    Anything else is a feed problem or a calendar problem, and both need a human
    before research runs on that day.
    """
    if not candles:
        return Check("candle_count", Verdict.FAIL, 0.0, detail="no candles at all")
    expected = calendar.expected_candle_count(tf)
    per_instrument: dict[int, set[datetime]] = {}
    for c in candles:
        if c.tf == tf:
            per_instrument.setdefault(c.instrument_id, set()).add(c.ts)

    worst_id, worst_n = None, expected
    for iid, stamps in per_instrument.items():
        if len(stamps) < worst_n:
            worst_id, worst_n = iid, len(stamps)

    if worst_id is None:
        return Check("candle_count", Verdict.PASS, float(expected), float(expected))
    verdict = Verdict.PASS if worst_n == expected else Verdict.FAIL
    return Check(
        "candle_count",
        verdict,
        float(worst_n),
        float(expected),
        f"{len(per_instrument)} instrument(s); worst is {worst_id}",
    )


def check_synthetic_share(candles: Sequence[Candle], *, max_share: float = 0.30) -> Check:
    """A session that is mostly synthetic is a dead feed wearing a complete grid.

    The candle count check passes in that case - the grid is regular - which is
    exactly why this second check exists.
    """
    if not candles:
        return Check("synthetic_share", Verdict.SKIP, 0.0)
    share = sum(1 for c in candles if c.synthetic) / len(candles)
    verdict = Verdict.PASS if share <= max_share else Verdict.FAIL
    return Check("synthetic_share", verdict, share, max_share, f"{len(candles):,} candles")


def check_candle_consistency(candles: Sequence[Candle]) -> Check:
    """High must be highest, low lowest. A violation means the builder is wrong."""
    if not candles:
        return Check("candle_consistency", Verdict.SKIP, 0.0)
    bad = [c for c in candles if not c.is_consistent()]
    verdict = Verdict.PASS if not bad else Verdict.FAIL
    detail = (
        "" if not bad else f"first at {bad[0].ts.isoformat()} instrument {bad[0].instrument_id}"
    )
    return Check("candle_consistency", verdict, float(len(bad)), 0.0, detail)


def check_price_jumps(candles: Sequence[Candle], *, max_move: float = 0.10) -> Check:
    """A single-minute move beyond a sane bound is almost always bad data.

    A real ten percent minute would be a circuit breaker, which the session
    calendar would already have told us about.
    """
    by_instrument: dict[int, list[Candle]] = {}
    for c in sorted(candles, key=lambda c: c.ts):
        by_instrument.setdefault(c.instrument_id, []).append(c)

    # Track "did we compare anything" separately from "what was the worst move".
    # A flat session has a worst move of exactly zero, and reporting SKIP for it
    # would mean the gate silently stops covering perfectly valid days.
    compared = 0
    worst, where = 0.0, ""
    for iid, series in by_instrument.items():
        for a, b in pairwise(series):
            if a.c <= 0:
                continue
            compared += 1
            move = abs(b.c / a.c - 1.0)
            if move > worst:
                worst, where = move, f"instrument {iid} at {b.ts.isoformat()}"
    if compared == 0:
        return Check("price_jumps", Verdict.SKIP, 0.0, detail="fewer than two candles")
    verdict = Verdict.PASS if worst <= max_move else Verdict.FAIL
    return Check("price_jumps", verdict, worst, max_move, where or f"{compared:,} pairs")


# --------------------------------------------------------------------------- #
# Option chain checks
# --------------------------------------------------------------------------- #
def check_chain_completeness(
    snapshots: Sequence[ChainSnapshot], *, min_completeness: float = 0.95
) -> Check:
    if not snapshots:
        return Check("chain_completeness", Verdict.SKIP, 0.0)
    mean = sum(s.completeness for s in snapshots) / len(snapshots)
    verdict = Verdict.PASS if mean >= min_completeness else Verdict.FAIL
    return Check(
        "chain_completeness",
        verdict,
        mean,
        min_completeness,
        f"{len(snapshots):,} snapshots",
    )


def check_iv_sanity(
    snapshots: Sequence[ChainSnapshot], *, lo: float = 0.03, hi: float = 3.0
) -> Check:
    """Implied volatilities outside a plausible band mean a stale or crossed quote.

    Note the denominator: this is a fraction of quotes that *had* an implied
    volatility. Quotes where inversion declined to answer are not failures - that
    refusal is the correct behaviour.
    """
    values = [q.iv for s in snapshots for q in s.quotes if q.iv is not None]
    if not values:
        return Check("iv_sanity", Verdict.SKIP, 0.0, detail="no implied volatilities computed")
    bad = sum(1 for v in values if not (lo < v < hi))
    share = bad / len(values)
    verdict = Verdict.PASS if share == 0 else (Verdict.WARN if share < 0.01 else Verdict.FAIL)
    return Check("iv_sanity", verdict, share, 0.0, f"{bad}/{len(values):,} outside [{lo}, {hi}]")


def check_crossed_books(snapshots: Sequence[ChainSnapshot]) -> Check:
    # Materialise the narrowed pair rather than filtering and re-reading the
    # optional fields: the type checker cannot carry the narrowing across a
    # comprehension boundary, and neither can a reader.
    pairs = [
        (q.bid, q.ask)
        for s in snapshots
        for q in s.quotes
        if q.bid is not None and q.ask is not None
    ]
    if not pairs:
        return Check("crossed_books", Verdict.SKIP, 0.0)
    bad = sum(1 for bid, ask in pairs if bid > ask)
    share = bad / len(pairs)
    verdict = Verdict.PASS if bad == 0 else Verdict.FAIL
    return Check("crossed_books", verdict, share, 0.0, f"{bad}/{len(pairs):,} crossed")


def check_put_call_parity(
    snapshots: Sequence[ChainSnapshot], *, max_violation_share: float = 0.02
) -> Check:
    """Parity is the sharpest staleness detector on a chain.

    Rather than pricing every strike, this compares the *shape*: for calls and
    puts at the same strike, ``C - P`` must decrease as strike increases. A stale
    leg breaks that ordering and nothing else does.
    """
    violations = 0
    compared = 0
    for snap in snapshots:
        by_expiry: dict[datetime, dict[float, dict[str, float]]] = {}
        for q in snap.quotes:
            mid = q.mid
            if mid is None:
                continue
            by_expiry.setdefault(q.expiry, {}).setdefault(q.strike, {})[q.opt_type.value] = mid
        for strikes in by_expiry.values():
            rows = sorted(
                (k, v[OptionType.CE.value] - v[OptionType.PE.value])
                for k, v in strikes.items()
                if OptionType.CE.value in v and OptionType.PE.value in v
            )
            for (_ka, da), (_kb, db) in pairwise(rows):
                compared += 1
                if db > da + 1e-6:
                    violations += 1
    if not compared:
        return Check("put_call_parity", Verdict.SKIP, 0.0)
    share = violations / compared
    verdict = Verdict.PASS if share <= max_violation_share else Verdict.FAIL
    return Check(
        "put_call_parity", verdict, share, max_violation_share, f"{violations:,}/{compared:,} pairs"
    )


# --------------------------------------------------------------------------- #
# News checks
# --------------------------------------------------------------------------- #
def check_news_timestamps(items: Sequence[RawNewsItem], *, max_uncertain: float = 0.10) -> Check:
    """Publish time must precede ingest time, and be knowable.

    Timestamp quality is the whole basis of the point-in-time archive: an item
    whose publish time we guessed is an item whose latency we cannot measure.
    """
    if not items:
        return Check("news_timestamps", Verdict.SKIP, 0.0)
    impossible = sum(1 for i in items if i.t_event > i.t_ingest)
    uncertain = sum(1 for i in items if i.ts_uncertain)
    if impossible:
        return Check(
            "news_timestamps",
            Verdict.FAIL,
            float(impossible),
            0.0,
            f"{impossible} item(s) published after we ingested them",
        )
    share = uncertain / len(items)
    verdict = Verdict.PASS if share <= max_uncertain else Verdict.WARN
    return Check(
        "news_timestamps", verdict, share, max_uncertain, f"{uncertain}/{len(items)} uncertain"
    )


def check_duplicate_rate(items: Sequence[RawNewsItem], *, max_rate: float = 0.03) -> Check:
    if not items:
        return Check("duplicate_rate", Verdict.SKIP, 0.0)
    dupes = sum(1 for i in items if i.duplicate_of)
    rate = dupes / len(items)
    verdict = Verdict.PASS if rate <= max_rate else Verdict.WARN
    return Check(
        "duplicate_rate", verdict, rate, max_rate, f"{dupes}/{len(items)} marked duplicate"
    )


# --------------------------------------------------------------------------- #
# Feature checks
# --------------------------------------------------------------------------- #
def check_null_rate(values: dict[str, Sequence[float | None]], *, max_rate: float = 0.01) -> Check:
    """Per-feature null rate. A feature that is often missing cannot be trained on."""
    if not values:
        return Check("feature_null_rate", Verdict.SKIP, 0.0)
    worst_name, worst_rate = "", 0.0
    for name, series in values.items():
        if not series:
            continue
        rate = sum(1 for v in series if v is None) / len(series)
        if rate > worst_rate:
            worst_name, worst_rate = name, rate
    verdict = Verdict.PASS if worst_rate <= max_rate else Verdict.FAIL
    return Check(
        "feature_null_rate", verdict, worst_rate, max_rate, f"worst: {worst_name or 'none'}"
    )


def population_stability_index(
    baseline: Sequence[float], current: Sequence[float], *, bins: int = 10
) -> float:
    """PSI between two distributions. Above 0.2 conventionally means real drift.

    Used by the drift monitor (MA3) and by the train/serve skew gate: a feature
    whose live distribution has moved away from its training distribution will
    quietly degrade a model long before anyone notices in the P&L.
    """
    if not baseline or not current:
        return 0.0
    ordered = sorted(baseline)
    edges = [ordered[int(i * (len(ordered) - 1) / bins)] for i in range(1, bins)]

    def bucket(xs: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for x in xs:
            idx = 0
            while idx < len(edges) and x > edges[idx]:
                idx += 1
            counts[idx] += 1
        total = len(xs) or 1
        # Floor at a small epsilon so an empty bucket cannot produce infinity.
        return [max(c / total, 1e-6) for c in counts]

    import math

    b, c = bucket(baseline), bucket(current)
    return sum((ci - bi) * math.log(ci / bi) for bi, ci in zip(b, c, strict=True))


def check_train_serve_skew(
    baseline: dict[str, Sequence[float]],
    current: dict[str, Sequence[float]],
    *,
    max_psi: float = 0.2,
) -> Check:
    shared = sorted(set(baseline) & set(current))
    if not shared:
        return Check("train_serve_skew", Verdict.SKIP, 0.0)
    worst_name, worst = "", 0.0
    for name in shared:
        psi = population_stability_index(baseline[name], current[name])
        if psi > worst:
            worst_name, worst = name, psi
    verdict = Verdict.PASS if worst <= max_psi else Verdict.FAIL
    return Check("train_serve_skew", verdict, worst, max_psi, f"worst: {worst_name}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_market_checks(
    day: date,
    calendar: SessionCalendar,
    *,
    ticks: Iterable[Tick] = (),
    candles: Iterable[Candle] = (),
    chain: Iterable[ChainSnapshot] = (),
    news: Iterable[RawNewsItem] = (),
) -> QualityReport:
    """Every gate for one day, in one report."""
    report = QualityReport(day=day)
    tick_list, candle_list = list(ticks), list(candles)
    chain_list, news_list = list(chain), list(news)

    report.add(check_tick_sequence_gaps(tick_list))
    report.add(check_candle_count(candle_list, calendar))
    report.add(check_candle_consistency(candle_list))
    report.add(check_synthetic_share(candle_list))
    report.add(check_price_jumps(candle_list))
    report.add(check_chain_completeness(chain_list))
    report.add(check_iv_sanity(chain_list))
    report.add(check_crossed_books(chain_list))
    report.add(check_put_call_parity(chain_list))
    report.add(check_news_timestamps(news_list))
    report.add(check_duplicate_rate(news_list))

    if report.failures:
        log.error(
            "data quality FAILED - downstream jobs are blocked",
            fields={"day": day.isoformat(), "failed": [c.name for c in report.failures]},
        )
    return report
