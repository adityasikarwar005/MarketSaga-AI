"""M63 - where dealer gamma sits, with the expiry cycle divided out.

Every model in this project that has been measured on real data has lost: M35
to a one-variable baseline, M55 to a three-term one, M54 to sigma * z_alpha,
M53 to a rules baseline indistinguishable from a coin flip. They have one thing
in common. Each fed **price history** into a model and asked a question **about
price**. Daily OHLC is four numbers a session, HAR already exhausts what they
say about variance, and nothing is left over.

Open interest by strike is not in the price series. It is a map of where the
options that must be hedged are sitting. When a lot of gamma sits near spot the
hedging flow it generates is mechanical and large relative to the book; when it
sits far away there is nothing to hedge. That is a statement about the
*character* of tomorrow's path - whether moves get damped or amplified - and it
is direction-free, which matters because direction is the thing four models have
now failed to predict.

**The confound this module exists to remove.** On 64 real NIFTY sessions,
corr(concentration, days_to_expiry) = -0.83. A study regressing realised
volatility on raw concentration would find a strong effect and it would be, to
first order, the effect of expiry proximity - already known, already in the
calendar, needing no options data to see. The signal, if there is one, is the
part that is *unusual for this point in the cycle*, which is what
:func:`cycle_adjusted` returns.

The mechanism is worth stating correctly, because the obvious explanation is
wrong and was written here first. "Gamma climbs as expiry approaches, so the sum
climbs" is **not** it: gamma is a density in strike, its sum over a fixed strike
grid is conserved as time to expiry falls, and on a synthetic chain with uniform
open interest the statistic is flat across the cycle to five decimal places.
What actually drives it is the interaction of two things - open interest is
peaked at the money, and the kernel narrows onto that peak as expiry nears. Far
from expiry the kernel averages open interest over a wide band; close to it, the
kernel sees little but the peak. Dividing the level of open interest out leaves
corr = -0.82, so the shape effect carries almost all of it and the concurrent
build-up of total open interest into expiry (corr -0.72) is the smaller half.

That distinction matters downstream: a normalisation that divided by total open
interest would remove the smaller confound and leave the larger one in place,
while looking like it had handled the problem.

**The kernel is a weighting function, not a Greek.** :func:`gamma_kernel` uses
one fixed volatility across all strikes. This is deliberate and should not be
"improved" by feeding it per-strike implied vols from M70: that would make a
positioning statistic depend on a surface fit which has itself been measured
negative on a real chain, and would change what the number means without
changing its name. The statistic asks where open interest sits relative to
spot. It does not price anything. :func:`concentration_is_robust_to_sigma`
exists so that claim is checked rather than asserted.

**No dealer sign is applied.** The published literature multiplies call and put
open interest by an assumed dealer position - long calls, short puts, or the
reverse depending on who is writing - and that assumption is unobservable and
does all the work. This module sums the magnitudes and leaves the sign to be
estimated from outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import median

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Definitional constants of the statistic, not tunables.
#
# KERNEL_SIGMA sets how far from spot open interest still counts. It is not
# fitted and must not be: a bandwidth chosen to make a result appear is the
# defect this repository's null tests exist to catch. What protects it is that
# the *ranking* of days by concentration barely moves across a plausible range -
# see `concentration_is_robust_to_sigma`, which measures that rather than
# assuming it.
KERNEL_SIGMA = 0.13

# Expiry day is not zero time to expiry. A contract expiring this afternoon has
# hours of gamma left, and the first cut of this module returned exactly 0.0 for
# every expiry session because T was days/365 and days was 0. The number looked
# plausible in a column of numbers, flipped the correlation with days-to-expiry
# from -0.82 to +0.12, and raised no error.
EXPIRY_DAY_FRACTION = 0.35

# Fewer strikes than this and the sum is a handful of contracts, not a profile.
MIN_STRIKES = 8


class PositioningError(RuntimeError):
    """Raised when a profile cannot be computed honestly."""


@dataclass(frozen=True)
class GammaProfile:
    """Gamma-weighted open interest for one underlying, one expiry, one day."""

    day: date
    expiry: date
    days_to_expiry: int
    spot: float
    concentration: float
    total_oi: int
    strikes_used: int

    @property
    def year_fraction(self) -> float:
        return year_fraction(self.days_to_expiry)


@dataclass(frozen=True)
class CycleNorm:
    """The typical concentration at each point in the expiry cycle.

    Built from medians and median absolute deviations rather than means and
    standard deviations. Concentration is heavy-tailed - a single expiry session
    can sit several times above the rest - and a mean would let those sessions
    set the norm they are then measured against.
    """

    centre: dict[int, float]
    scale: dict[int, float]
    n: dict[int, int]

    def known_horizons(self) -> list[int]:
        return sorted(self.centre)


def year_fraction(days_to_expiry: int) -> float:
    """Time to expiry in years, floored so expiry day is not zero.

    One definition, used by both :func:`gamma_concentration` and
    ``GammaProfile.year_fraction``. They computed it separately at first, which
    meant a test pinning the property could pass while the path that actually
    weights the strikes used a different number.
    """
    return max(float(days_to_expiry), EXPIRY_DAY_FRACTION) / 365.0


def gamma_kernel(spot: float, strike: float, t: float, sigma: float = KERNEL_SIGMA) -> float:
    """Black-Scholes gamma at a single flat volatility.

    Raises rather than returning 0.0 on a non-positive ``t``. A zero here is
    indistinguishable from "no open interest near spot" once it has been summed,
    and that is exactly how the expiry-day bug survived its first reading.
    """
    if t <= 0.0:
        raise PositioningError(
            f"time to expiry must be positive, got {t}. Expiry day has hours "
            "left, not zero - see EXPIRY_DAY_FRACTION."
        )
    if spot <= 0.0 or strike <= 0.0:
        raise PositioningError(f"spot and strike must be positive, got {spot} and {strike}")
    v = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + 0.5 * v * v) / v
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) / (spot * v)


def gamma_concentration(
    records: Sequence[tuple[float, int]],
    *,
    day: date,
    expiry: date,
    spot: float,
    sigma: float = KERNEL_SIGMA,
) -> GammaProfile:
    """Sum of kernel-weighted open interest over the strikes of one expiry.

    ``records`` is ``(strike, open_interest)``. Calls and puts are summed
    together and no dealer sign is applied.
    """
    if expiry < day:
        raise PositioningError(f"expiry {expiry} precedes the session {day}")
    strikes = {k for k, _ in records}
    if len(strikes) < MIN_STRIKES:
        raise PositioningError(
            f"{len(strikes)} distinct strikes on {day} for {expiry}; below the "
            f"{MIN_STRIKES} needed before this is a profile rather than a handful "
            "of contracts"
        )
    dte = (expiry - day).days
    t = year_fraction(dte)
    total = sum(gamma_kernel(spot, k, t, sigma) * oi for k, oi in records)
    return GammaProfile(
        day=day,
        expiry=expiry,
        days_to_expiry=dte,
        spot=spot,
        concentration=total,
        total_oi=sum(oi for _, oi in records),
        strikes_used=len(strikes),
    )


def fit_cycle_norm(profiles: Iterable[GammaProfile], *, min_per_horizon: int = 5) -> CycleNorm:
    """What concentration normally looks like at each days-to-expiry.

    Horizons with fewer than ``min_per_horizon`` observations are left out
    rather than estimated from two points. :func:`cycle_adjusted` then refuses
    those days, which is the honest outcome: no norm means no deviation from it.
    """
    by_dte: dict[int, list[float]] = {}
    for p in profiles:
        by_dte.setdefault(p.days_to_expiry, []).append(p.concentration)
    centre: dict[int, float] = {}
    scale: dict[int, float] = {}
    n: dict[int, int] = {}
    for dte, values in by_dte.items():
        if len(values) < min_per_horizon:
            continue
        med = median(values)
        mad = median([abs(v - med) for v in values])
        centre[dte] = med
        # 1.4826 makes a MAD comparable to a standard deviation on Gaussian
        # data, so the returned score reads on a familiar scale.
        scale[dte] = 1.4826 * mad
        n[dte] = len(values)
    if not centre:
        raise PositioningError(
            f"no days-to-expiry horizon had {min_per_horizon} observations; "
            "there is not enough history here to say what is normal"
        )
    log.info("cycle norm fitted", fields={"horizons": len(centre), "rows": sum(n.values())})
    return CycleNorm(centre=centre, scale=scale, n=n)


def cycle_adjusted(profile: GammaProfile, norm: CycleNorm) -> float | None:
    """How unusual this profile is *for its point in the expiry cycle*.

    ``None`` when the horizon has no norm, or when the norm has no spread. A
    zero-spread horizon means every observation of it was identical, and
    dividing by that would manufacture an enormous score out of rounding.
    """
    dte = profile.days_to_expiry
    if dte not in norm.centre:
        return None
    spread = norm.scale.get(dte, 0.0)
    if spread <= 0.0:
        return None
    return (profile.concentration - norm.centre[dte]) / spread


def concentration_is_robust_to_sigma(
    records_by_day: Sequence[tuple[date, date, float, Sequence[tuple[float, int]]]],
    *,
    sigmas: Sequence[float] = (0.09, 0.13, 0.20),
) -> float:
    """Lowest rank correlation between day-orderings under different sigmas.

    The kernel bandwidth is not fitted, so the claim that it is not a tuning
    knob needs evidence. If reordering the days by concentration is insensitive
    to sigma then no choice of sigma buys a different answer. A number near 1
    supports the claim; a low one means the bandwidth *is* doing work and the
    module docstring is wrong.
    """
    orderings = []
    for sigma in sigmas:
        vals = [
            gamma_concentration(recs, day=d, expiry=e, spot=s, sigma=sigma).concentration
            for d, e, s, recs in records_by_day
        ]
        orderings.append(_ranks(vals))
    worst = 1.0
    for i in range(len(orderings)):
        for j in range(i + 1, len(orderings)):
            worst = min(worst, _pearson(orderings[i], orderings[j]))
    return worst


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = float(rank)
    return out


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)
