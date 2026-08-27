"""M30 - Abnormal returns and the time-of-day baseline.

"Nifty moved 0.3% after the headline" is not a finding until you know what Nifty
does anyway at that time of day. Intraday volatility is strongly U-shaped: a
0.3% move in the first ten minutes is unremarkable and the same move at 13:30 is
a two-sigma event. Comparing raw returns across times of day silently rewards
every event that happens to cluster near the open.

So every impact number in this project is an *abnormal* return:

    AR_h = r_h - E[r_h | minute-of-session, volatility regime, day-type]

Two things this module refuses to do:

**Estimate a baseline from too few sessions.** With five days of history the
per-minute mean is mostly noise, and subtracting noise from a measurement makes
it worse rather than better. The profile reports how many sessions it saw and
declines below a floor.

**Use a normal approximation for significance.** Intraday returns are fat-tailed
and a t-test on them reports impressive p-values for nothing at all. Significance
is bootstrapped against matched controls - real windows from the same minute of
the day, in the same regime, on days when nothing happened.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from statistics import fmean, pstdev

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import Candle

log = get_logger(__name__)

# Two different thresholds, because they answer two different questions.
#
# How many SESSIONS before the profile as a whole is worth using:
MIN_SESSIONS_FOR_PROFILE = 20
#
# How many OBSERVATIONS a single slot needs. Splitting by regime divides the
# history three ways, so a per-minute slot is always thin - which is why slots
# pool their neighbouring minutes. The U-shape is smooth, so minutes either side
# are a fair observation of the same part of the day; a minute from a different
# part of the day is not.
MIN_SAMPLES_PER_SLOT = 100
SLOT_POOL_MINUTES = 5


class Regime(StrEnum):
    """Coarse volatility regime. Deliberately three buckets, not thirty.

    Finer buckets mean fewer observations per bucket, and the whole purpose of
    the baseline is to have enough observations to be worth subtracting.
    """

    CALM = "calm"
    NORMAL = "normal"
    STRESSED = "stressed"


@dataclass
class MinuteStats:
    n: int = 0
    mean: float = 0.0
    sigma: float = 0.0
    samples: list[float] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        """Whether this exact minute has enough on its own. Usually it does not -
        see MIN_SAMPLES_PER_SLOT and the pooling in TimeOfDayProfile.expected."""
        return self.n >= MIN_SAMPLES_PER_SLOT


@dataclass
class TimeOfDayProfile:
    """What the market does anyway, minute by minute.

    Keyed on (regime, minute-of-session, horizon). Splitting by regime matters:
    the same minute of the day behaves differently in a calm week and a stressed
    one, and pooling them produces a baseline that fits neither.
    """

    horizon_minutes: int
    stats: dict[tuple[Regime, int], MinuteStats] = field(default_factory=dict)
    sessions: set[date] = field(default_factory=set)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def reliable(self) -> bool:
        return self.n_sessions >= MIN_SESSIONS_FOR_PROFILE

    def get(self, regime: Regime, minute: int) -> MinuteStats | None:
        return self.stats.get((regime, minute))

    def pooled(
        self, regime: Regime, minute: int, *, window: int = SLOT_POOL_MINUTES
    ) -> list[float]:
        """Samples from this minute and its neighbours, in this regime."""
        out: list[float] = []
        for offset in range(-window, window + 1):
            slot = self.stats.get((regime, minute + offset))
            if slot:
                out.extend(slot.samples)
        return out

    def expected(
        self, regime: Regime, minute: int, *, window: int = SLOT_POOL_MINUTES
    ) -> tuple[float, float] | None:
        """(mean, sigma) for this slot, or None when there is not enough of it.

        Returning None rather than a shaky number is the point: the caller then
        reports "unknown", which is recoverable, instead of an abnormal return
        computed against a baseline built from four observations.
        """
        samples = self.pooled(regime, minute, window=window)
        if len(samples) < MIN_SAMPLES_PER_SLOT:
            return None
        return fmean(samples), (pstdev(samples) if len(samples) > 1 else 0.0)

    def coverage(self) -> float:
        """Fraction of slots that have enough observations once pooled."""
        if not self.stats:
            return 0.0
        keys = list(self.stats)
        enough = sum(1 for regime, minute in keys if self.expected(regime, minute) is not None)
        return enough / len(keys)

    def render(self) -> str:
        return (
            f"  time-of-day profile: horizon {self.horizon_minutes}m, "
            f"{self.n_sessions} sessions, {len(self.stats)} slots, "
            f"{self.coverage():.0%} reliable"
            + ("" if self.reliable else f"  [UNRELIABLE - need {MIN_SESSIONS_FOR_PROFILE}]")
        )


# --------------------------------------------------------------------------- #
# Regime classification
# --------------------------------------------------------------------------- #
def session_volatility(bars: Sequence[Candle]) -> float:
    """Close-to-close realised volatility for one session, annualised."""
    closes = [b.c for b in sorted(bars, key=lambda c: c.ts) if b.c > 0]
    if len(closes) < 10:
        return 0.0
    rets = [math.log(b / a) for a, b in pairwise(closes)]
    return pstdev(rets) * math.sqrt(252 * 375) if len(rets) > 1 else 0.0


def classify_regimes(
    per_session: Mapping[date, Sequence[Candle]],
    *,
    calm_pct: float = 0.33,
    stressed_pct: float = 0.67,
) -> dict[date, Regime]:
    """Bucket sessions by their own realised volatility.

    Thresholds are *percentiles of the observed history*, not absolute numbers.
    An absolute threshold ages badly - what counted as stressed two years ago is
    an ordinary Tuesday now - and a model conditioned on a stale definition of
    "stressed" is conditioned on nothing.
    """
    vols = {day: session_volatility(bars) for day, bars in per_session.items()}
    ranked = sorted(v for v in vols.values() if v > 0)
    if len(ranked) < 3:
        return dict.fromkeys(vols, Regime.NORMAL)

    lo = ranked[int(calm_pct * (len(ranked) - 1))]
    hi = ranked[int(stressed_pct * (len(ranked) - 1))]
    out: dict[date, Regime] = {}
    for day, v in vols.items():
        if v <= 0:
            out[day] = Regime.NORMAL
        elif v <= lo:
            out[day] = Regime.CALM
        elif v >= hi:
            out[day] = Regime.STRESSED
        else:
            out[day] = Regime.NORMAL
    return out


# --------------------------------------------------------------------------- #
# Building the profile
# --------------------------------------------------------------------------- #
def forward_return(bars: Sequence[Candle], idx: int, horizon: int) -> float | None:
    """Log return from bar ``idx`` close to ``horizon`` bars later."""
    if idx + horizon >= len(bars):
        return None
    a, b = bars[idx].c, bars[idx + horizon].c
    if a <= 0 or b <= 0:
        return None
    return math.log(b / a)


def build_profile(
    per_session: Mapping[date, Sequence[Candle]],
    calendar: SessionCalendar,
    *,
    horizon_minutes: int = 15,
    regimes: dict[date, Regime] | None = None,
    exclude: set[datetime] | None = None,
) -> TimeOfDayProfile:
    """Learn the ordinary behaviour of each minute of the session.

    ``exclude`` drops windows around known events. Without it the baseline
    absorbs the very moves it is supposed to measure against, and every event
    study run on top of it understates its own effect.
    """
    profile = TimeOfDayProfile(horizon_minutes=horizon_minutes)
    regime_map = regimes or classify_regimes(per_session)
    skip = exclude or set()

    for day, raw in per_session.items():
        bars = sorted(raw, key=lambda c: c.ts)
        if len(bars) < horizon_minutes + 5:
            continue
        profile.sessions.add(day)
        regime = regime_map.get(day, Regime.NORMAL)

        for i, bar in enumerate(bars):
            if bar.ts in skip:
                continue
            r = forward_return(bars, i, horizon_minutes)
            if r is None:
                continue
            minute = calendar.minutes_since_open(bar.ts)
            key = (regime, minute)
            slot = profile.stats.setdefault(key, MinuteStats())
            slot.samples.append(r)

    for slot in profile.stats.values():
        slot.n = len(slot.samples)
        if slot.n:
            slot.mean = fmean(slot.samples)
            slot.sigma = pstdev(slot.samples) if slot.n > 1 else 0.0

    log.info(
        "time-of-day profile built",
        fields={
            "sessions": profile.n_sessions,
            "slots": len(profile.stats),
            "coverage": round(profile.coverage(), 3),
        },
    )
    return profile


# --------------------------------------------------------------------------- #
# Abnormal returns
# --------------------------------------------------------------------------- #
@dataclass
class AbnormalReturn:
    t0: datetime
    horizon_minutes: int
    raw: float
    expected: float
    abnormal: float
    sigma: float
    regime: Regime
    minute_of_session: int
    reliable: bool

    @property
    def z(self) -> float:
        """Abnormal return in units of what is normal for this slot."""
        return self.abnormal / self.sigma if self.sigma > 0 else 0.0

    def render(self) -> str:
        flag = "" if self.reliable else "  [baseline unreliable]"
        return (
            f"    {self.t0:%H:%M} +{self.horizon_minutes}m  raw {self.raw:+.4%}  "
            f"expected {self.expected:+.4%}  abnormal {self.abnormal:+.4%} "
            f"({self.z:+.2f} sigma){flag}"
        )


def abnormal_return(
    bars: Sequence[Candle],
    t0: datetime,
    profile: TimeOfDayProfile,
    calendar: SessionCalendar,
    *,
    regime: Regime = Regime.NORMAL,
) -> AbnormalReturn | None:
    """The move, minus what that minute of the day does anyway."""
    ordered = sorted(bars, key=lambda c: c.ts)
    idx = next((i for i, b in enumerate(ordered) if b.ts >= t0), None)
    if idx is None:
        return None
    raw = forward_return(ordered, idx, profile.horizon_minutes)
    if raw is None:
        return None

    minute = calendar.minutes_since_open(ordered[idx].ts)
    expectation = profile.expected(regime, minute)
    if expectation is None:
        # No usable baseline. Report the raw move and say so, rather than
        # subtracting a number nobody should trust.
        slot = profile.get(regime, minute)
        return AbnormalReturn(
            t0=ordered[idx].ts,
            horizon_minutes=profile.horizon_minutes,
            raw=raw,
            expected=0.0,
            abnormal=raw,
            sigma=slot.sigma if slot else 0.0,
            regime=regime,
            minute_of_session=minute,
            reliable=False,
        )

    mean, sigma = expectation
    return AbnormalReturn(
        t0=ordered[idx].ts,
        horizon_minutes=profile.horizon_minutes,
        raw=raw,
        expected=mean,
        abnormal=raw - mean,
        sigma=sigma,
        regime=regime,
        minute_of_session=minute,
        reliable=True,
    )


# --------------------------------------------------------------------------- #
# Matched controls and bootstrap significance
# --------------------------------------------------------------------------- #
def matched_controls(
    profile: TimeOfDayProfile, regime: Regime, minute: int, *, window: int = 5
) -> list[float]:
    """Real returns from the same minute of the day, in the same regime.

    A window of neighbouring minutes is pooled because a single minute rarely
    has enough observations - the U-shape is smooth, so minutes either side are
    a fair comparison, whereas a minute from a different part of the day is not.
    """
    return profile.pooled(regime, minute, window=window)


@dataclass
class Significance:
    observed: float
    n_controls: int
    p_value: float
    control_mean: float
    control_sigma: float
    percentile: float

    @property
    def significant(self) -> bool:
        """Two-sided at 5%. Not a discovery on its own - see the caveat below."""
        return self.p_value < 0.05

    def render(self) -> str:
        verdict = "significant" if self.significant else "not distinguishable from normal"
        return (
            f"    observed {self.observed:+.4%} vs {self.n_controls:,} matched controls: "
            f"p={self.p_value:.4f} ({self.percentile:.1f}th pct) - {verdict}"
        )


def bootstrap_significance(
    observed: float, controls: Sequence[float], *, n_boot: int = 10_000, seed: int = 11
) -> Significance | None:
    """How unusual is ``observed`` against real windows from the same slot?

    Bootstrapped rather than assumed normal. Intraday returns have heavy tails,
    and a t-test on them reports impressive p-values for nothing at all.

    The p-value is the fraction of controls at least as extreme in absolute
    terms. With a handful of controls it cannot be small however striking the
    observation looks, which is the correct behaviour and the reason
    ``n_controls`` is returned alongside it.
    """
    if len(controls) < 30:
        return None
    rng = random.Random(seed)
    centred = [c - fmean(controls) for c in controls]
    extreme = sum(1 for c in centred if abs(c) >= abs(observed - fmean(controls)))
    p_direct = (extreme + 1) / (len(centred) + 1)

    # Bootstrap the mean to get a stable percentile even with a skewed sample.
    n = len(centred)
    draws = sorted(
        fmean([centred[rng.randrange(n)] for _ in range(n)]) for _ in range(min(n_boot, 2000))
    )
    below = sum(1 for d in draws if d < observed - fmean(controls))
    percentile = 100.0 * below / len(draws)

    return Significance(
        observed=observed,
        n_controls=len(controls),
        p_value=p_direct,
        control_mean=fmean(controls),
        control_sigma=pstdev(controls) if len(controls) > 1 else 0.0,
        percentile=percentile,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def group_by_session(
    candles: Iterable[Candle], calendar: SessionCalendar, *, instrument_id: int | None = None
) -> dict[date, list[Candle]]:
    out: dict[date, list[Candle]] = {}
    for c in candles:
        if instrument_id is not None and c.instrument_id != instrument_id:
            continue
        out.setdefault(calendar.local(c.ts).date(), []).append(c)
    for bars in out.values():
        bars.sort(key=lambda c: c.ts)
    return out


def event_windows(events: Iterable[datetime], *, before: int = 5, after: int = 30) -> set[datetime]:
    """Minute stamps to exclude from the baseline around each event."""
    out: set[datetime] = set()
    for t in events:
        base = t.replace(second=0, microsecond=0)
        for m in range(-before, after + 1):
            out.add(base + timedelta(minutes=m))
    return out
