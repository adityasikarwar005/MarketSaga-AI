"""M31 - Event study, and M32 - hierarchical Bayesian impact priors.

The impact library answers one question, and answers it as a distribution:

    Given an event of class C at severity S, what has the index done next?

Not "what will it do" - that is the prediction models' job. This is the
empirical prior they update, and the thing the rules baseline consults before it
is willing to have an opinion at all.

Two decisions define this module.

**Everything is a distribution with a sample size attached.** The original
design computed "Nifty will drop 1.5%" and that single number, with no error
bar, is the root cause of over-sizing. What is stored instead is a median, a
spread, and ``n`` - and ``n`` is what the risk gate reads when deciding how much
to trust it.

**Rare classes borrow from their family.** Most interesting events are rare.
With four historical instances the empirical mean is noise; a hierarchical model
shrinks the class estimate toward its family, so a class with n=3 is dominated
by the family prior. That is not a limitation being worked around, it is the
correct answer - and when there is no family either, the honest output is
UNKNOWN rather than a number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from itertools import pairwise
from statistics import fmean, median, pstdev

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import Candle
from libs.ms_impact.baseline import (
    Regime,
    TimeOfDayProfile,
    abnormal_return,
    bootstrap_significance,
    matched_controls,
)

log = get_logger(__name__)

DEFAULT_HORIZONS = (1, 5, 15, 30, 60)

# Below this a class-specific estimate is not reported. The number is not
# arbitrary: with fewer than eight observations the credible interval is wider
# than any plausible effect, so the estimate cannot change a decision.
MIN_SAMPLE_FOR_CLASS = 8

# How many standard errors the severity exponent must clear before it is
# believed. Below this the neutral 1.0 is returned instead - see
# severity_response, where a zero slope used to come back as 0.5.
GAMMA_MIN_T = 1.96


# --------------------------------------------------------------------------- #
# One observation
# --------------------------------------------------------------------------- #
@dataclass
class EventObservation:
    """What actually happened after one event."""

    event_id: str
    event_class: str
    t0: datetime
    severity: float
    regime: Regime
    abnormal: dict[int, float] = field(default_factory=dict)  # horizon -> AR
    raw: dict[int, float] = field(default_factory=dict)
    mfe: float = 0.0  # max favourable excursion
    mae: float = 0.0  # max adverse excursion
    vol_ratio: float = 1.0  # realised vol after / before
    time_to_peak_minutes: float | None = None
    half_life_minutes: float | None = None
    reliable: bool = True

    @property
    def family(self) -> str:
        return self.event_class.split(".", 1)[0]


def _realised_vol(bars: Sequence[Candle]) -> float:
    closes = [b.c for b in bars if b.c > 0]
    if len(closes) < 4:
        return 0.0
    rets = [math.log(b / a) for a, b in pairwise(closes)]
    return pstdev(rets) if len(rets) > 1 else 0.0


def observe_event(
    bars: Sequence[Candle],
    t0: datetime,
    event_id: str,
    event_class: str,
    severity: float,
    profile: TimeOfDayProfile,
    calendar: SessionCalendar,
    *,
    regime: Regime = Regime.NORMAL,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    pre_window: int = 30,
) -> EventObservation | None:
    """Measure one event against the time-of-day baseline.

    The excursion statistics matter as much as the returns: max favourable and
    max adverse excursion are where a target and a stop belong, derived from
    evidence rather than from a round number someone liked.
    """
    ordered = sorted(bars, key=lambda c: c.ts)
    idx = next((i for i, b in enumerate(ordered) if b.ts >= t0), None)
    if idx is None or idx + 2 >= len(ordered):
        return None

    obs = EventObservation(
        event_id=event_id,
        event_class=event_class,
        t0=ordered[idx].ts,
        severity=severity,
        regime=regime,
    )

    for h in horizons:
        # Each horizon needs its own baseline: what is normal over one minute is
        # not one fifteenth of what is normal over fifteen.
        sub = TimeOfDayProfile(horizon_minutes=h, stats=profile.stats, sessions=profile.sessions)
        ar = abnormal_return(ordered, t0, sub, calendar, regime=regime)
        if ar is None:
            continue
        obs.abnormal[h] = ar.abnormal
        obs.raw[h] = ar.raw
        obs.reliable = obs.reliable and ar.reliable

    entry = ordered[idx].c
    longest = max(horizons)
    after = [b for b in ordered[idx + 1 : idx + 1 + longest]]
    if after and entry > 0:
        highs = [(b.h / entry - 1.0) for b in after]
        lows = [(b.low / entry - 1.0) for b in after]
        obs.mfe = max(highs)
        obs.mae = min(lows)

        moves = [abs(b.c / entry - 1.0) for b in after]
        peak_idx = moves.index(max(moves))
        obs.time_to_peak_minutes = float(peak_idx + 1)
        peak = moves[peak_idx]
        # Half-life: how long until the move has decayed to half its peak. A
        # short half-life means the move mean-reverts and the trade must be
        # quick; a long one means it persists.
        for j in range(peak_idx + 1, len(moves)):
            if moves[j] <= peak / 2:
                obs.half_life_minutes = float(j - peak_idx)
                break

    before = ordered[max(0, idx - pre_window) : idx]
    v_before, v_after = _realised_vol(before), _realised_vol(after)
    obs.vol_ratio = (v_after / v_before) if v_before > 0 else 1.0
    return obs


# --------------------------------------------------------------------------- #
# Aggregating a class
# --------------------------------------------------------------------------- #
@dataclass
class ClassStats:
    """The empirical response of one event class at one horizon."""

    event_class: str
    horizon: int
    n: int
    median: float
    mean: float
    sigma: float
    q25: float
    q75: float
    p_sign_correct: float
    mean_mfe: float
    mean_mae: float
    mean_vol_ratio: float
    median_time_to_peak: float | None
    median_half_life: float | None
    p_value: float | None = None

    @property
    def usable(self) -> bool:
        return self.n >= MIN_SAMPLE_FOR_CLASS

    def render(self) -> str:
        sig = f"  p={self.p_value:.3f}" if self.p_value is not None else ""
        flag = "" if self.usable else f"  [n<{MIN_SAMPLE_FOR_CLASS}, not usable alone]"
        return (
            f"    {self.event_class:<32} {self.horizon:>3}m  n={self.n:<4} "
            f"median {self.median:+.3%}  IQR [{self.q25:+.3%}, {self.q75:+.3%}]  "
            f"sign {self.p_sign_correct:.2f}{sig}{flag}"
        )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = q * (len(ordered) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def aggregate_class(
    observations: Sequence[EventObservation],
    horizon: int,
    *,
    profile: TimeOfDayProfile | None = None,
    calendar: SessionCalendar | None = None,
) -> ClassStats | None:
    """Summarise one class at one horizon, with significance where possible."""
    rows = [o for o in observations if horizon in o.abnormal]
    if not rows:
        return None
    values = [o.abnormal[horizon] for o in rows]
    med = median(values)
    direction = 1.0 if med > 0 else -1.0
    agree = sum(1 for v in values if (v > 0) == (direction > 0)) / len(values)

    p_value = None
    if profile is not None and calendar is not None and rows:
        controls = matched_controls(
            profile, rows[0].regime, calendar.minutes_since_open(rows[0].t0)
        )
        sig = bootstrap_significance(med, controls)
        p_value = sig.p_value if sig else None

    peaks = [o.time_to_peak_minutes for o in rows if o.time_to_peak_minutes is not None]
    halves = [o.half_life_minutes for o in rows if o.half_life_minutes is not None]

    return ClassStats(
        event_class=rows[0].event_class,
        horizon=horizon,
        n=len(rows),
        median=med,
        mean=fmean(values),
        sigma=pstdev(values) if len(values) > 1 else 0.0,
        q25=_quantile(values, 0.25),
        q75=_quantile(values, 0.75),
        p_sign_correct=agree,
        mean_mfe=fmean([o.mfe for o in rows]),
        mean_mae=fmean([o.mae for o in rows]),
        mean_vol_ratio=fmean([o.vol_ratio for o in rows]),
        median_time_to_peak=median(peaks) if peaks else None,
        median_half_life=median(halves) if halves else None,
        p_value=p_value,
    )


# --------------------------------------------------------------------------- #
# M32 - hierarchical shrinkage
# --------------------------------------------------------------------------- #
@dataclass
class Posterior:
    """A class-level estimate that knows how much of itself is borrowed."""

    event_class: str
    horizon: int
    mean: float
    sigma: float
    n: int
    n_eff: int
    shrinkage: float  # 0 = all data, 1 = all family prior
    family_mean: float
    credible_low: float
    credible_high: float
    is_unknown: bool = False

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero, and means it.

        Requires a usable sample and a non-zero spread. With one observation the
        spread computes to zero, which produces a zero-width interval that
        technically excludes zero and asserts perfect certainty from a single
        data point - precisely the false confidence this module exists to
        prevent.
        """
        if not self.usable or self.sigma <= 0:
            return False
        return (self.credible_low > 0) or (self.credible_high < 0)

    @property
    def usable(self) -> bool:
        return not self.is_unknown and self.n_eff >= MIN_SAMPLE_FOR_CLASS

    def render(self) -> str:
        if self.is_unknown:
            return f"    {self.event_class:<32} {self.horizon:>3}m  UNKNOWN (no usable history)"
        star = " *" if self.excludes_zero else "  "
        return (
            f"    {self.event_class:<32} {self.horizon:>3}m  n={self.n:<4} "
            f"posterior {self.mean:+.3%}  CI [{self.credible_low:+.3%}, "
            f"{self.credible_high:+.3%}]{star} shrink {self.shrinkage:.2f}"
        )


def family_prior(stats: Sequence[ClassStats]) -> tuple[float, float]:
    """Pool every class in a family into one prior: (mean, between-class spread).

    The between-class spread is what controls how much a rare class borrows. A
    family whose classes all behave alike lends heavily; one whose classes
    disagree lends little, because there is no coherent family behaviour to lend.

    The spread of the observed class means is *not* that quantity. Each observed
    mean is a true class effect plus its own sampling error, so their variance
    is the sum of both, and using it directly reports a family that disagrees
    more than it really does. That makes rare classes borrow less than they
    should and keep more of their own noise - under-shrinkage, in the direction
    that flatters thin classes.

    Measured on 20 classes of 8 observations drawn from one identical
    distribution - where the true between-class spread is exactly zero - the raw
    version returned 0.001367 against a pure sampling-noise floor of 0.001276,
    and shrank only 0.46 of the way to the family mean instead of essentially
    all the way.

    So the average within-class sampling variance is subtracted off. When that
    leaves nothing, the honest answer is zero spread: the classes are not
    distinguishable, and every one of them should fall back on the family.
    """
    if not stats:
        return 0.0, 0.0
    means = [s.mean for s in stats]
    grand = fmean(means)
    if len(means) < 2:
        return grand, 0.0

    observed_var = pstdev(means) ** 2
    # Average variance of a class mean, which is what sampling alone contributes.
    sampling = [((s.sigma**2) / s.n) for s in stats if s.n >= 2 and s.sigma > 0]
    noise_var = fmean(sampling) if sampling else 0.0

    return grand, math.sqrt(max(0.0, observed_var - noise_var))


def shrink(
    class_stat: ClassStats,
    family_mean: float,
    between_sigma: float,
    *,
    credible_z: float = 1.96,
    family_size: int = 1,
) -> Posterior:
    """Precision-weighted blend of the class estimate and its family prior.

    With n=3 the posterior is dominated by the family - which is correct, not a
    compromise. An estimate from three observations should not be allowed to
    move a position size, and this is the mechanism that stops it.
    """
    n = class_stat.n
    if n == 0:
        return Posterior(
            event_class=class_stat.event_class,
            horizon=class_stat.horizon,
            mean=0.0,
            sigma=0.0,
            n=0,
            n_eff=0,
            shrinkage=1.0,
            family_mean=family_mean,
            credible_low=0.0,
            credible_high=0.0,
            is_unknown=True,
        )

    between_var = between_sigma**2
    # With fewer than two observations the within-class variance is not
    # estimable - `sigma` is zero because there is nothing to vary, not because
    # the class is precise. Treating that zero as precision is how a single
    # observation acquires a zero-width credible interval.
    estimable = n >= 2 and class_stat.sigma > 0
    within_var = (class_stat.sigma**2) / n if estimable else None

    if within_var is None:
        if between_var > 0:
            # Fall back entirely on the family: the estimate is borrowed, and
            # its uncertainty is the family's spread rather than nothing at all.
            weight = 0.0
            posterior_var = between_var
        else:
            # No within-class information and no family to borrow from.
            return Posterior(
                event_class=class_stat.event_class,
                horizon=class_stat.horizon,
                mean=class_stat.mean,
                sigma=0.0,
                n=n,
                n_eff=n,
                shrinkage=1.0,
                family_mean=family_mean,
                credible_low=0.0,
                credible_high=0.0,
                is_unknown=True,
            )
    elif between_var <= 0:
        # No estimated spread between the classes in this family. Two very
        # different situations arrive here and they need opposite answers.
        #
        # A family of one has no siblings to borrow from, so the class can only
        # stand on its own. But a family whose classes are *indistinguishable*
        # is the strongest possible case for borrowing: if nothing separates
        # them, the family mean is the best estimate for every one of them, and
        # letting each keep its own noisy average is how a class with eight
        # observations acquires an opinion it has not earned.
        #
        # The old code took the first branch for both, so removing sampling
        # noise from the family spread - which correctly drives it to zero -
        # would have turned maximum borrowing into none at all.
        if family_size <= 1:
            weight = 1.0
            posterior_var = within_var
        else:
            weight = 0.0
            posterior_var = within_var
    else:
        weight = between_var / (between_var + within_var)
        posterior_var = (within_var * between_var) / (within_var + between_var)

    posterior_mean = weight * class_stat.mean + (1 - weight) * family_mean
    posterior_sigma = math.sqrt(max(0.0, posterior_var))

    # Effective sample size: how many observations this estimate is really worth
    # once borrowing is accounted for. This is the number the risk gate reads.
    n_eff = round(n * weight) if weight < 1.0 else n

    return Posterior(
        event_class=class_stat.event_class,
        horizon=class_stat.horizon,
        mean=posterior_mean,
        sigma=posterior_sigma,
        n=n,
        n_eff=max(n_eff, 1),
        shrinkage=1.0 - weight,
        family_mean=family_mean,
        credible_low=posterior_mean - credible_z * posterior_sigma,
        credible_high=posterior_mean + credible_z * posterior_sigma,
    )


def severity_response(observations: Sequence[EventObservation], horizon: int) -> float:
    """Fit the exponent in ``g(s) = (s/5)^gamma``.

    Impact is convex in severity - a 9 is much more than twice a 4.5 - and the
    exponent is estimated rather than assumed. Returns 1.0 (linear, the neutral
    assumption) whenever the data does not support a different answer.

    The clamp used to be the whole safety story, and it hid the one case that
    matters. When severity carries no information the slope is zero, and
    ``max(0.5, ...)`` turned that zero into 0.5 - so "severity does not matter"
    came back as "impact scales as the square root of severity", which is a
    claim, not a fallback. Measured on pure noise the old version returned
    exactly 0.500 in 80 of 80 trials.

    So the slope is now tested before it is trusted. An estimate that is not
    distinguishable from zero returns the neutral 1.0, and only a slope that
    survives its own standard error is clamped and returned.
    """
    rows = [
        (o.severity, abs(o.abnormal[horizon]))
        for o in observations
        if horizon in o.abnormal and o.severity > 0 and abs(o.abnormal[horizon]) > 0
    ]
    if len(rows) < 8:
        return 1.0
    xs = [math.log(s / 5.0) for s, _ in rows]
    ys = [math.log(a) for _, a in rows]
    if pstdev(xs) < 1e-9:
        return 1.0
    mx, my = fmean(xs), fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        return 1.0
    gamma = cov / var

    # Is the slope distinguishable from zero at all?
    intercept = my - gamma * mx
    resid = [y - (intercept + gamma * x) for x, y in zip(xs, ys, strict=True)]
    dof = len(rows) - 2
    if dof <= 0:
        return 1.0
    sse = sum(r * r for r in resid)
    stderr = math.sqrt(sse / dof / var) if sse > 0 else 0.0
    if stderr <= 0 or abs(gamma) < GAMMA_MIN_T * stderr:
        return 1.0

    return max(0.5, min(3.0, gamma))


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #
@dataclass
class ImpactLibrary:
    """Every class, every horizon, with provenance."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    posteriors: dict[tuple[str, int], Posterior] = field(default_factory=dict)
    stats: dict[tuple[str, int], ClassStats] = field(default_factory=dict)
    gamma: dict[int, float] = field(default_factory=dict)
    built_from: int = 0

    def get(self, event_class: str, horizon: int) -> Posterior | None:
        return self.posteriors.get((event_class, horizon))

    def usable_classes(self, horizon: int) -> list[str]:
        return sorted(c for (c, h), p in self.posteriors.items() if h == horizon and p.usable)

    def significant_classes(self, horizon: int) -> list[str]:
        return sorted(
            c
            for (c, h), p in self.posteriors.items()
            if h == horizon and p.usable and p.excludes_zero
        )

    def render(self, horizon: int = 15) -> str:
        rows = sorted(
            (p for (_c, h), p in self.posteriors.items() if h == horizon),
            key=lambda p: -abs(p.mean),
        )
        head = (
            f"  impact library: {self.built_from:,} observations, "
            f"{len(self.usable_classes(horizon))} usable classes at {horizon}m, "
            f"{len(self.significant_classes(horizon))} with CI excluding zero"
        )
        return head + "\n" + "\n".join(p.render() for p in rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "horizons": list(self.horizons),
            "built_from": self.built_from,
            "gamma": self.gamma,
            "posteriors": {f"{c}|{h}": asdict(p) for (c, h), p in self.posteriors.items()},
        }


def build_library(
    observations: Sequence[EventObservation],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    profile: TimeOfDayProfile | None = None,
    calendar: SessionCalendar | None = None,
) -> ImpactLibrary:
    """Aggregate observations into shrunk, class-level posteriors."""
    library = ImpactLibrary(horizons=tuple(horizons), built_from=len(observations))

    by_class: dict[str, list[EventObservation]] = {}
    for obs in observations:
        by_class.setdefault(obs.event_class, []).append(obs)

    for horizon in horizons:
        class_stats: dict[str, ClassStats] = {}
        for event_class, rows in by_class.items():
            stat = aggregate_class(rows, horizon, profile=profile, calendar=calendar)
            if stat is not None:
                class_stats[event_class] = stat
                library.stats[(event_class, horizon)] = stat

        by_family: dict[str, list[ClassStats]] = {}
        for event_class, stat in class_stats.items():
            by_family.setdefault(event_class.split(".", 1)[0], []).append(stat)

        for event_class, stat in class_stats.items():
            family = event_class.split(".", 1)[0]
            f_mean, f_sigma = family_prior(by_family[family])
            library.posteriors[(event_class, horizon)] = shrink(
                stat, f_mean, f_sigma, family_size=len(by_family[family])
            )

        library.gamma[horizon] = severity_response(observations, horizon)

    log.info(
        "impact library built",
        fields={
            "observations": len(observations),
            "classes": len(by_class),
            "horizons": list(horizons),
        },
    )
    return library


def prior_for_baseline(
    library: ImpactLibrary, event_class: str, horizon: int = 15
) -> dict[str, object]:
    """Shape the M0 rules baseline expects.

    Returns ``n_eff=0`` for an unknown class, which the baseline reads as "no
    usable prior" and answers with no direction at all. That silence is the
    intended behaviour, not a gap to be filled in later.
    """
    posterior = library.get(event_class, horizon)
    if posterior is None or not posterior.usable:
        return {"prior_return": None, "n_eff": 0}
    return {"prior_return": posterior.mean, "n_eff": posterior.n_eff}


def synthetic_events(
    per_session: Mapping[date, Sequence[Candle]],
    calendar: SessionCalendar,
    *,
    classes: Sequence[str] = ("macro.inflation_release", "geopolitical.military_action"),
    per_session_count: int = 2,
    seed: int = 5,
) -> list[tuple[str, str, datetime, float]]:
    """Deterministic fake events for exercising the machinery.

    Not a stand-in for real events - it cannot be, since the whole point of an
    event study is the relationship between a real classification and a real
    move. It exists so the aggregation, shrinkage and reporting code can be
    tested and reviewed before the news pipeline (L2) exists.
    """
    import random

    rng = random.Random(seed)
    out: list[tuple[str, str, datetime, float]] = []
    for day, bars in sorted(per_session.items()):
        ordered = sorted(bars, key=lambda c: c.ts)
        if len(ordered) < 120:
            continue
        for k in range(per_session_count):
            idx = rng.randrange(60, len(ordered) - 61)
            cls = classes[k % len(classes)]
            out.append(
                (
                    f"{day.isoformat()}-{k}",
                    cls,
                    ordered[idx].ts,
                    round(rng.uniform(5.0, 9.5), 1),
                )
            )
    return out
