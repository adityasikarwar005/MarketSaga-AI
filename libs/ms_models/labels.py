"""M50 - Triple-barrier labelling.

"Was the next candle green?" is the wrong question, because nobody can trade the
answer. The triple-barrier method asks the question a trader actually faces:

    If I enter now with this stop and this target, which do I hit first?

Three barriers: an upper one, a lower one, and a deadline. The label is which was
touched first. Stop-loss and target are therefore *encoded in the label*, so a
model trained on it is answering the question you care about rather than an
academic one about candle colour.

Four things here are easy to get wrong and expensive:

**Barriers scale with current volatility.** A fixed 50-point barrier is a coin
flip in a calm hour and unreachable in a quiet one. ``k * sigma_t`` makes the
label mean the same thing all day - and ``sigma_t`` is estimated from bars
strictly *before* entry, or the label knows its own answer.

**Intrabar ambiguity is real.** When both barriers fall inside one candle, OHLC
cannot say which was touched first. Guessing favourably is how a backtest gains
several points of accuracy that do not exist. Such samples are flagged and
resolved *against* the trade.

**Barriers never cross the session close.** An intraday position is flat by
15:20 whether or not a barrier was hit, so the deadline is clamped.

**Overlapping outcomes are not independent observations.** Two labels whose
windows overlap share most of their price path. Counting them as two independent
samples inflates every significance test built on them, which is why each
carries a uniqueness weight.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import IntEnum
from itertools import pairwise

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle

log = get_logger(__name__)


class Outcome(IntEnum):
    DOWN = -1
    TIMEOUT = 0
    UP = 1


@dataclass(frozen=True)
class BarrierSpec:
    """How far the barriers sit and how long the trade may run."""

    k_up: float = 1.5
    k_down: float = 1.5
    horizon_minutes: int = 15
    vol_lookback_minutes: int = 30
    min_vol: float = 1e-6
    # Minutes before the close after which no new label is started. A position
    # opened at 15:28 cannot reach a fifteen-minute barrier; labelling it anyway
    # manufactures a pile of TIMEOUTs that teach the model nothing.
    no_entry_before_close: int = 20

    def __post_init__(self) -> None:
        if self.k_up <= 0 or self.k_down <= 0:
            raise ValueError("barrier multiples must be positive")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon must be positive")


@dataclass
class Label:
    t0: datetime
    instrument_id: int
    entry: float
    upper: float
    lower: float
    deadline: datetime
    outcome: Outcome
    exit_ts: datetime
    exit_price: float
    sigma: float
    ambiguous: bool = False
    uniqueness: float = 1.0
    event_id: str | None = None

    @property
    def ret(self) -> float:
        return (self.exit_price / self.entry - 1.0) if self.entry else 0.0

    @property
    def holding_minutes(self) -> float:
        return (self.exit_ts - self.t0).total_seconds() / 60.0

    @property
    def hit_barrier(self) -> bool:
        return self.outcome is not Outcome.TIMEOUT


# --------------------------------------------------------------------------- #
# Volatility at entry
# --------------------------------------------------------------------------- #
def trailing_sigma(bars: Sequence[Candle], spec: BarrierSpec) -> float | None:
    """Per-minute return volatility from bars strictly before entry.

    The caller is responsible for passing only prior bars. This function does
    not know what time it is, and that is deliberate - it cannot leak what it
    was never given.
    """
    if len(bars) < 5:
        return None
    rets = [math.log(b.c / a.c) for a, b in pairwise(bars) if a.c > 0 and b.c > 0]
    if len(rets) < 4:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    sigma = math.sqrt(var)
    return sigma if sigma > spec.min_vol else None


# --------------------------------------------------------------------------- #
# Building labels
# --------------------------------------------------------------------------- #
def build_labels(
    candles: Sequence[Candle],
    spec: BarrierSpec,
    calendar: SessionCalendar,
    *,
    instrument_id: int | None = None,
    entry_every: int = 1,
    event_ids: dict[datetime, str] | None = None,
) -> list[Label]:
    """Label every eligible bar in one session.

    ``entry_every`` thins the entries: labelling every minute produces maximally
    overlapping windows, which is legitimate but makes the uniqueness weights
    very small. Thinning trades sample count for independence.
    """
    bars = sorted(
        (c for c in candles if instrument_id is None or c.instrument_id == instrument_id),
        key=lambda c: c.ts,
    )
    if len(bars) < spec.vol_lookback_minutes + 2:
        return []

    iid = bars[0].instrument_id
    day = calendar.local(bars[0].ts).date()
    _, session_close = calendar.session_bounds(day)
    last_entry = session_close - timedelta(minutes=spec.no_entry_before_close)

    labels: list[Label] = []
    for i in range(spec.vol_lookback_minutes, len(bars) - 1):
        if (i - spec.vol_lookback_minutes) % entry_every:
            continue
        bar = bars[i]
        if bar.ts >= last_entry:
            break

        # Volatility from bars strictly before entry - never including the bar
        # we are about to enter on.
        sigma = trailing_sigma(bars[i - spec.vol_lookback_minutes : i], spec)
        if sigma is None:
            continue

        entry = bar.c
        upper = entry * math.exp(spec.k_up * sigma * math.sqrt(spec.horizon_minutes))
        lower = entry * math.exp(-spec.k_down * sigma * math.sqrt(spec.horizon_minutes))
        deadline = min(bar.ts + timedelta(minutes=spec.horizon_minutes), session_close)

        label = _resolve(bars[i + 1 :], entry, upper, lower, deadline)
        if label is None:
            continue
        outcome, exit_ts, exit_price, ambiguous = label
        labels.append(
            Label(
                t0=bar.ts,
                instrument_id=iid,
                entry=entry,
                upper=upper,
                lower=lower,
                deadline=deadline,
                outcome=outcome,
                exit_ts=exit_ts,
                exit_price=exit_price,
                sigma=sigma,
                ambiguous=ambiguous,
                event_id=(event_ids or {}).get(bar.ts),
            )
        )

    apply_uniqueness(labels)
    METRICS.inc("labels_built_total", value=float(len(labels)))
    return labels


def _resolve(
    future: Sequence[Candle], entry: float, upper: float, lower: float, deadline: datetime
) -> tuple[Outcome, datetime, float, bool] | None:
    """Walk forward until a barrier or the deadline.

    When both barriers fall inside one candle the outcome is resolved
    **against** the trade. OHLC cannot say which came first, and choosing
    favourably is exactly the assumption that buys a backtest several points of
    accuracy it will not have live.
    """
    for bar in future:
        if bar.ts > deadline:
            break
        hit_up = bar.h >= upper
        hit_down = bar.low <= lower
        if hit_up and hit_down:
            return Outcome.DOWN, bar.ts, lower, True
        if hit_up:
            return Outcome.UP, bar.ts, upper, False
        if hit_down:
            return Outcome.DOWN, bar.ts, lower, False

    in_window = [b for b in future if b.ts <= deadline]
    if not in_window:
        return None
    last = in_window[-1]
    return Outcome.TIMEOUT, last.ts, last.c, False


# --------------------------------------------------------------------------- #
# Uniqueness
# --------------------------------------------------------------------------- #
def apply_uniqueness(labels: Sequence[Label]) -> None:
    """Weight each label by how much of its window it does not share.

    Two labels whose outcome windows overlap observe mostly the same price path.
    Treating them as two independent samples inflates every significance test
    downstream - the effective sample size is a fraction of the row count, and
    without this weight nobody ever finds that out.
    """
    if not labels:
        return
    edges: set[datetime] = set()
    for label in labels:
        edges.add(label.t0)
        edges.add(label.exit_ts)
    grid = sorted(edges)
    if len(grid) < 2:
        for label in labels:
            label.uniqueness = 1.0
        return

    concurrency = [0] * (len(grid) - 1)
    spans: list[list[int]] = []
    for label in labels:
        cells = [
            idx for idx, (a, b) in enumerate(pairwise(grid)) if a >= label.t0 and b <= label.exit_ts
        ]
        spans.append(cells)
        for idx in cells:
            concurrency[idx] += 1

    for label, cells in zip(labels, spans, strict=True):
        if not cells:
            label.uniqueness = 1.0
            continue
        label.uniqueness = sum(1.0 / concurrency[i] for i in cells) / len(cells)


def effective_sample_size(labels: Sequence[Label]) -> float:
    """Sum of uniqueness weights - what the row count actually means.

    Reported next to any statistic computed on these labels. A thousand rows at
    0.05 uniqueness is fifty observations, and a significance test that thinks
    it has a thousand will find an edge in noise.
    """
    return sum(label.uniqueness for label in labels)


# --------------------------------------------------------------------------- #
# Meta-labelling
# --------------------------------------------------------------------------- #
@dataclass
class MetaLabel:
    """Should the primary model's call have been taken?"""

    t0: datetime
    side: int  # +1 long, -1 short
    take: bool
    realised: float
    uniqueness: float
    event_id: str | None = None


def build_meta_labels(
    labels: Sequence[Label], sides: dict[datetime, int], *, cost: float = 0.0
) -> list[MetaLabel]:
    """Turn primary-model calls into take/skip labels.

    The second model's job is to say **no**. It is trained on whether acting on
    the first model's side would have paid *after costs*, which is why ``cost``
    is a parameter and not an afterthought: a call that is right by less than the
    spread is a call that should have been skipped.
    """
    out: list[MetaLabel] = []
    for label in labels:
        side = sides.get(label.t0)
        if side is None or side == 0:
            continue
        realised = side * label.ret - cost
        out.append(
            MetaLabel(
                t0=label.t0,
                side=side,
                take=realised > 0,
                realised=realised,
                uniqueness=label.uniqueness,
                event_id=label.event_id,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass
class LabelReport:
    total: int = 0
    up: int = 0
    down: int = 0
    timeout: int = 0
    ambiguous: int = 0
    effective_n: float = 0.0
    mean_holding_minutes: float = 0.0
    by_day: dict[date, int] = field(default_factory=dict)

    @property
    def balance(self) -> float:
        """Fraction of barrier hits that were up. Near 0.5 is expected and healthy."""
        hits = self.up + self.down
        return self.up / hits if hits else 0.5

    @property
    def timeout_share(self) -> float:
        return self.timeout / self.total if self.total else 0.0

    def render(self) -> str:
        return (
            f"  {self.total:,} labels  up={self.up:,} down={self.down:,} "
            f"timeout={self.timeout:,} ({self.timeout_share:.1%})\n"
            f"  balance {self.balance:.3f}  ambiguous {self.ambiguous:,}  "
            f"effective n {self.effective_n:,.0f} "
            f"({self.effective_n / self.total:.1%} of rows)\n"
            f"  mean holding {self.mean_holding_minutes:.1f} min"
        )


def summarise(labels: Sequence[Label], calendar: SessionCalendar | None = None) -> LabelReport:
    report = LabelReport(total=len(labels))
    if not labels:
        return report
    for label in labels:
        if label.outcome is Outcome.UP:
            report.up += 1
        elif label.outcome is Outcome.DOWN:
            report.down += 1
        else:
            report.timeout += 1
        if label.ambiguous:
            report.ambiguous += 1
        if calendar is not None:
            day = calendar.local(label.t0).date()
            report.by_day[day] = report.by_day.get(day, 0) + 1
    report.effective_n = effective_sample_size(labels)
    report.mean_holding_minutes = sum(x.holding_minutes for x in labels) / len(labels)
    return report


def build_many(
    per_day: Iterable[Sequence[Candle]],
    spec: BarrierSpec,
    calendar: SessionCalendar,
    **kwargs: object,
) -> list[Label]:
    """Label several sessions.

    Uniqueness is applied per session on purpose: two labels on different days
    share no price path, so pooling them would dilute the weights and overstate
    independence in exactly the opposite direction.
    """
    out: list[Label] = []
    for candles in per_day:
        out.extend(build_labels(candles, spec, calendar, **kwargs))  # type: ignore[arg-type]
    return out
