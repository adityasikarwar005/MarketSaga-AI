"""M71 - Option pricing, Greeks and implied volatility.

Black-76 on the forward, which is the right model for European index options:
the forward already carries the cost of carry, so no dividend assumption has to
be invented. A spot-based helper is provided for the case where only spot and a
rate are available.

Two rules that matter more than the maths:

**Price from mid, never from last-traded.** A stale LTP on an illiquid strike
produces an implied volatility that is not wrong so much as fictional, and it
will happily propagate into a surface fit, a feature, and a trade.

**Implied volatility must fail loudly.** When a quote is below intrinsic, or the
solver does not converge, the answer is ``None`` - not a number that looks
plausible. Every downstream consumer treats ``None`` as "no opinion", which is
recoverable; a fabricated 0.42 is not.

Dependency-free on purpose: ``math`` only. These functions sit in the hot path
(Monte Carlo repricing, M73) and are called millions of times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

# Trading-time constants. Volatility is quoted annualised, so any intraday
# horizon has to be expressed as a fraction of a trading year - and using
# calendar time here instead of trading time is a classic, silent mispricing.
TRADING_DAYS = 252
MINUTES_PER_SESSION = 375
MINUTES_PER_YEAR = TRADING_DAYS * MINUTES_PER_SESSION

SQRT_2PI = math.sqrt(2.0 * math.pi)


class OptType(StrEnum):
    CALL = "CE"
    PUT = "PE"


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function - accurate to double precision."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def years_from_minutes(minutes: float) -> float:
    """Convert an intraday horizon to a fraction of a trading year."""
    return minutes / MINUTES_PER_YEAR


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float  # per 1.00 of volatility (divide by 100 for "per vol point")
    theta: float  # per year
    rho: float

    @property
    def vega_per_point(self) -> float:
        """Vega per one volatility *point*, which is how traders actually quote it."""
        return self.vega / 100.0

    @property
    def theta_per_day(self) -> float:
        return self.theta / TRADING_DAYS

    def theta_per_minutes(self, minutes: float) -> float:
        return self.theta * years_from_minutes(minutes)


def _d1_d2(f: float, k: float, sigma: float, t: float) -> tuple[float, float]:
    v = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / v
    return d1, d1 - v


def black76_price(f: float, k: float, sigma: float, t: float, r: float, kind: OptType) -> float:
    """European option on a forward.

    Degenerate inputs (expiry reached, zero volatility) return intrinsic value
    rather than raising: an option at expiry is a perfectly well-defined thing,
    and the caller should not have to special-case the last minute of the day.
    """
    if t <= 0.0 or sigma <= 0.0 or f <= 0.0 or k <= 0.0:
        intrinsic = max(0.0, f - k) if kind is OptType.CALL else max(0.0, k - f)
        return math.exp(-r * max(t, 0.0)) * intrinsic
    d1, d2 = _d1_d2(f, k, sigma, t)
    df = math.exp(-r * t)
    if kind is OptType.CALL:
        return df * (f * norm_cdf(d1) - k * norm_cdf(d2))
    return df * (k * norm_cdf(-d2) - f * norm_cdf(-d1))


def black76_greeks(f: float, k: float, sigma: float, t: float, r: float, kind: OptType) -> Greeks:
    price = black76_price(f, k, sigma, t, r, kind)
    if t <= 0.0 or sigma <= 0.0 or f <= 0.0 or k <= 0.0:
        # At expiry delta is a step function; gamma, vega and theta vanish.
        itm = (f > k) if kind is OptType.CALL else (f < k)
        delta = (1.0 if kind is OptType.CALL else -1.0) if itm else 0.0
        return Greeks(price=price, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(f, k, sigma, t)
    df = math.exp(-r * t)
    sqrt_t = math.sqrt(t)
    pdf_d1 = norm_pdf(d1)

    gamma = df * pdf_d1 / (f * sigma * sqrt_t)
    vega = df * f * pdf_d1 * sqrt_t

    if kind is OptType.CALL:
        delta = df * norm_cdf(d1)
        theta = -df * f * pdf_d1 * sigma / (2.0 * sqrt_t) - r * df * (
            f * norm_cdf(d1) - k * norm_cdf(d2)
        )
        rho = -t * price
    else:
        delta = -df * norm_cdf(-d1)
        theta = -df * f * pdf_d1 * sigma / (2.0 * sqrt_t) - r * df * (
            k * norm_cdf(-d2) - f * norm_cdf(-d1)
        )
        rho = -t * price

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def forward_from_spot(spot: float, r: float, t: float, q: float = 0.0) -> float:
    """Forward implied by spot, rate and carry. Prefer a traded future when there is one."""
    return spot * math.exp((r - q) * max(t, 0.0))


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #
MIN_VOL = 1e-4
MAX_VOL = 5.0


def intrinsic(f: float, k: float, t: float, r: float, kind: OptType) -> float:
    df = math.exp(-r * max(t, 0.0))
    return df * (max(0.0, f - k) if kind is OptType.CALL else max(0.0, k - f))


def implied_vol(
    price: float,
    f: float,
    k: float,
    t: float,
    r: float,
    kind: OptType,
    *,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float | None:
    """Invert Black-76 for volatility, or return ``None``.

    Newton first (it converges in a handful of steps when vega is healthy), with
    a bisection fallback for the deep wings where vega collapses and Newton
    either overshoots or stalls.

    ``None`` is returned - deliberately, rather than a clamped guess - when:

    * the quote is at or below intrinsic value (an arbitrage or, far more often,
      a stale quote),
    * the quote exceeds the no-arbitrage upper bound,
    * the option has expired,
    * or the solver fails to converge.

    Roughly one quote in twenty on a live chain hits one of these. Silently
    inventing a number for them is how a volatility surface acquires a smile
    that does not exist.
    """
    if t <= 0.0 or price <= 0.0 or f <= 0.0 or k <= 0.0:
        return None

    df = math.exp(-r * t)
    lower = intrinsic(f, k, t, r, kind)
    upper = df * (f if kind is OptType.CALL else k)
    if price <= lower + 1e-12 or price >= upper:
        return None

    # Brenner-Subrahmanyam style starting point: decent near the money,
    # harmless elsewhere because bisection will take over.
    sigma = max(MIN_VOL, math.sqrt(2.0 * math.pi / t) * price / (df * f))
    sigma = min(sigma, MAX_VOL)

    for _ in range(max_iter):
        g = black76_greeks(f, k, sigma, t, r, kind)
        diff = g.price - price
        if abs(diff) < tol:
            return sigma
        if g.vega < 1e-10:
            break  # vega has collapsed; Newton cannot make progress
        step = diff / g.vega
        nxt = sigma - step
        if not (MIN_VOL < nxt < MAX_VOL) or not math.isfinite(nxt):
            break
        sigma = nxt
    else:
        return None

    lo, hi = MIN_VOL, MAX_VOL
    if black76_price(f, k, lo, t, r, kind) > price:
        return None
    if black76_price(f, k, hi, t, r, kind) < price:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        pm = black76_price(f, k, mid, t, r, kind)
        if abs(pm - price) < tol:
            return mid
        if pm < price:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid if abs(black76_price(f, k, mid, t, r, kind) - price) < 1e-4 else None


# --------------------------------------------------------------------------- #
# Chain-level quantities
# --------------------------------------------------------------------------- #
def implied_move(atm_call: float, atm_put: float, spot: float) -> float:
    """Straddle-implied move as a fraction of spot.

    The denominator of the trade filter in the blueprint: a forecast move must
    exceed what the market has already priced, or there is nothing to buy.
    """
    if spot <= 0:
        return 0.0
    return (atm_call + atm_put) / spot


def implied_move_from_vol(sigma: float, minutes: float) -> float:
    """Expected absolute move over an intraday horizon, as a fraction of spot."""
    return sigma * math.sqrt(years_from_minutes(minutes))


def put_call_parity_gap(call: float, put: float, f: float, k: float, t: float, r: float) -> float:
    """``C - P - D(F - K)``. Zero in theory; in practice it measures staleness.

    A gap wider than the two spreads combined means at least one leg is not a
    real price right now. Useful as a data quality gate before the surface fit,
    where a single stale quote can distort the whole smile.
    """
    return (call - put) - math.exp(-r * t) * (f - k)
