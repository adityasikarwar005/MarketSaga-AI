"""M71 - Greeks from the fitted surface, not from a flat vol and not from a vendor.

Black-Scholes delta answers "how does this option's value move with the forward,
**holding volatility constant**". On a real surface volatility is not constant
when the forward moves: the strike stays where it is, log-moneyness ``k = ln(K/F)``
shifts, and the smile hands back a different vol at that same strike.

So the delta that hedges is

    dV/dF = dV/dF  +  dV/dsigma * dsigma/dF

and the second term is not small. On a Nifty-shaped smile a 500-point-OTM call
carries a Black-76 delta of 0.2839 and a true delta of 0.2940 - **3.6% more**.
Hedging on the first number leaves ten lots short by around seven units of
forward, every day, in the same direction. It does not look like a bug. It looks
like hedging slippage, which is a thing that exists and which everybody expects
to see some of.

The same chain rule applies to gamma, with four terms rather than two:

    d²V/dF² = Gamma + 2*vanna*sigma_F + volga*sigma_F² + vega*sigma_FF

Getting one sign wrong in there produces a gamma that is perfectly plausible, so
every term in this module is pinned against a **full reprice**: bump the forward,
recompute ``k``, ask the SVI slice for the vol at the new ``k``, and price again.
That is a genuinely independent route to the same number, and it is what the
plan means by "validate against an independent reference implementation".

Aggregation, and the number that must never be one number
-----------------------------------------------------------

Delta, gamma and theta aggregate across a book. **Vega does not.** A seven-day
vega and a sixty-day vega are different risks that happen to share a unit: front
vol moves two or three times as much as back vol on any day worth hedging. Adding
them produces a book that reports vega-neutral while being short the front and
long the back, which is a real position with real P&L and no number describing
it. :class:`PortfolioGreeks` therefore buckets vega by expiry and exposes the
total only as ``vega_gross``, named so that nobody mistakes it for a net.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from libs.ms_core.obs import get_logger
from libs.ms_options.pricing import (
    Greeks,
    OptType,
    black76_greeks,
    black76_price,
    norm_pdf,
)
from libs.ms_options.surface import ExpirySlice, Surface

log = get_logger(__name__)


class GreeksError(RuntimeError):
    pass


@dataclass(frozen=True)
class SurfaceGreeks:
    """Greeks for one contract, with the smile term kept visible.

    ``delta`` is the one to hedge on. ``delta_black76`` is kept beside it rather
    than discarded, because the difference between them is the quantity that
    explains a hedging P&L, and a module that only returns the right answer
    cannot show anyone why the old answer was wrong.
    """

    strike: float
    kind: OptType
    expiry: datetime
    t: float
    forward: float
    vol: float

    price: float
    delta: float  # smile-adjusted: dV/dF along the surface
    gamma: float  # smile-adjusted: d²V/dF²
    vega: float
    theta: float
    rho: float

    delta_black76: float
    gamma_black76: float
    vanna: float
    volga: float
    dsigma_dforward: float

    @property
    def smile_delta_adjustment(self) -> float:
        """How much of the delta the constant-vol assumption misses."""
        return self.delta - self.delta_black76

    @property
    def vega_per_point(self) -> float:
        return self.vega / 100.0

    def render(self) -> str:
        return (
            f"    {self.strike:>8.0f}{self.kind.value:<3} exp={self.expiry.date()} "
            f"iv={self.vol:.4f}  px={self.price:9.2f}  d={self.delta:+.5f} "
            f"(b76 {self.delta_black76:+.5f}, smile {self.smile_delta_adjustment:+.5f})  "
            f"g={self.gamma:.7f}  v={self.vega_per_point:8.2f}/pt"
        )


def greeks_from_surface(
    sl: ExpirySlice, *, strike: float, kind: OptType, r: float = 0.065
) -> SurfaceGreeks:
    """The Greeks of one contract, priced off the fitted slice.

    The vol comes from the SVI parameters at this strike's log-moneyness. There
    is no path that accepts a vendor IV: a vendor's number is fitted from quotes
    we have already filtered, by a method we cannot see, and mixing it with our
    own surface produces a book hedged against two different volatilities.
    """
    f = sl.forward.forward
    if f <= 0.0 or strike <= 0.0:
        raise GreeksError(f"forward {f} and strike {strike} must both be positive")
    if sl.t <= 0.0:
        raise GreeksError("an expired slice has no Greeks; the position has a settlement value")

    k = math.log(strike / f)
    sigma = sl.params.vol(k, sl.t)
    g: Greeks = black76_greeks(f, strike, sigma, sl.t, r, kind)

    sqrt_t = math.sqrt(sl.t)
    v = sigma * sqrt_t
    d1 = (math.log(f / strike) + 0.5 * sigma * sigma * sl.t) / v
    d2 = d1 - v
    discount = math.exp(-r * sl.t)

    # Second-order Black-76 sensitivities. Both verified against finite
    # differences of vega, because a sign error in either produces a gamma that
    # is smooth, positive and wrong.
    vanna = -discount * norm_pdf(d1) * d2 / sigma  # d(vega)/dF
    volga = g.vega * d1 * d2 / sigma  # d(vega)/d(sigma)

    # How the smile moves the vol at THIS strike when the forward moves.
    # k = ln(K/F), so dk/dF = -1/F, and the sign of the whole term follows the
    # sign of the skew: on a negatively skewed surface a rising forward slides
    # the strike down the smile to a lower vol.
    sigma_k = sl.params.d1(k) / (2.0 * sl.t * sigma)
    sigma_kk = (sl.params.d2(k) / sl.t - 2.0 * sigma_k * sigma_k) / (2.0 * sigma)
    sigma_f = -sigma_k / f
    sigma_ff = (sigma_kk + sigma_k) / (f * f)

    delta = g.delta + g.vega * sigma_f
    gamma = g.gamma + 2.0 * vanna * sigma_f + volga * sigma_f * sigma_f + g.vega * sigma_ff

    return SurfaceGreeks(
        strike=strike,
        kind=kind,
        expiry=sl.expiry,
        t=sl.t,
        forward=f,
        vol=sigma,
        price=g.price,
        delta=delta,
        gamma=gamma,
        vega=g.vega,
        theta=g.theta,
        rho=g.rho,
        delta_black76=g.delta,
        gamma_black76=g.gamma,
        vanna=vanna,
        volga=volga,
        dsigma_dforward=sigma_f,
    )


# --------------------------------------------------------------------------- #
# The independent reference
# --------------------------------------------------------------------------- #
def reference_delta_gamma(
    sl: ExpirySlice, *, strike: float, kind: OptType, r: float = 0.065, bump: float = 2e-4
) -> tuple[float, float]:
    """Delta and gamma by repricing the whole thing at a bumped forward.

    Deliberately shares no algebra with :func:`greeks_from_surface`: it moves the
    forward, recomputes log-moneyness, asks the SVI slice for the vol there, and
    prices again. Two routes to the same number, and a disagreement means one of
    them is wrong rather than that both drifted together.
    """
    f = sl.forward.forward
    h = f * bump

    def price_at(forward: float) -> float:
        k = math.log(strike / forward)
        return black76_price(forward, strike, sl.params.vol(k, sl.t), sl.t, r, kind)

    up, mid, down = price_at(f + h), price_at(f), price_at(f - h)
    return (up - down) / (2.0 * h), (up - 2.0 * mid + down) / (h * h)


# --------------------------------------------------------------------------- #
# Portfolio aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leg:
    """One position in one contract.

    ``lots`` is always positive; ``is_long`` carries the direction. A signed
    quantity works until someone writes ``abs(lots)`` for a display and the sign
    is gone from the record that the risk gate reads.
    """

    strike: float
    kind: OptType
    expiry: datetime
    lots: int
    lot_size: int
    is_long: bool = True

    def __post_init__(self) -> None:
        if self.lots <= 0:
            raise GreeksError(
                f"lots is {self.lots}; it is always positive and direction lives in "
                "is_long. A signed quantity survives until the first abs() in a "
                "display and then the sign is gone from the record risk reads."
            )
        if self.lot_size <= 0:
            raise GreeksError("lot_size must be positive")

    @property
    def signed_quantity(self) -> int:
        return (1 if self.is_long else -1) * self.lots * self.lot_size


@dataclass
class PortfolioGreeks:
    """A book's risk, with vega kept apart by expiry.

    Delta, gamma and theta are summed. Vega is not: a seven-day vega and a
    sixty-day vega are different risks sharing a unit, and adding them describes
    a book that reports neutral while being short front vol and long back vol.
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vega_by_expiry: dict[datetime, float] = field(default_factory=dict)
    delta_black76: float = 0.0
    per_leg: list[tuple[Leg, SurfaceGreeks]] = field(default_factory=list)
    forwards: dict[datetime, float] = field(default_factory=dict)

    @property
    def vega_gross(self) -> float:
        """Every bucket added up. Named ``gross`` so nobody reads it as a net.

        Kept because a single number is genuinely useful for a limit check - the
        most a book can lose to a parallel vol move - and dangerous for anything
        else.
        """
        return sum(self.vega_by_expiry.values())

    @property
    def vega_net_front_back(self) -> tuple[float, float]:
        """Nearest expiry against everything behind it - the calendar position."""
        if not self.vega_by_expiry:
            return 0.0, 0.0
        front = min(self.vega_by_expiry)
        return self.vega_by_expiry[front], sum(
            v for e, v in self.vega_by_expiry.items() if e != front
        )

    @property
    def smile_delta_adjustment(self) -> float:
        return self.delta - self.delta_black76

    @property
    def theta_per_day(self) -> float:
        return self.theta / 252.0

    def render(self) -> str:
        lines = [f"  portfolio greeks ({len(self.per_leg)} legs)"]
        lines += [f"{sg.render()}   x{leg.signed_quantity:+d}" for leg, sg in self.per_leg]
        lines.append(
            f"    delta {self.delta:+12.2f}   (black-76 {self.delta_black76:+.2f}, "
            f"smile term {self.smile_delta_adjustment:+.2f})"
        )
        lines.append(f"    gamma {self.gamma:+12.4f}")
        lines.append(f"    theta {self.theta_per_day:+12.2f} per day")
        lines.append("    vega, by expiry - not one number:")
        for expiry, v in sorted(self.vega_by_expiry.items()):
            lines.append(f"      {expiry.date()}  {v / 100.0:+12.2f} per vol point")
        front, back = self.vega_net_front_back
        lines.append(
            f"      gross {self.vega_gross / 100.0:+.2f}/pt   "
            f"front {front / 100.0:+.2f} vs back {back / 100.0:+.2f} - "
            "a gross near zero across these is a calendar position, not a flat book"
        )
        if len(self.forwards) > 1:
            lines.append(
                f"    delta is summed across {len(self.forwards)} expiries with "
                "different forwards; they move together but are not the same number"
            )
        return "\n".join(lines)


def aggregate(
    legs: Sequence[Leg], surface: Surface, *, now: datetime, r: float = 0.065
) -> PortfolioGreeks:
    """Roll a book up against one surface.

    Goes through :meth:`Surface.slice_at`, so a stale surface refuses here too.
    A risk aggregation is exactly the place a staleness check gets bypassed for
    convenience, and it is the last place that should happen.
    """
    out = PortfolioGreeks()
    for leg in legs:
        sl = surface.slice_at(leg.expiry, now=now)
        sg = greeks_from_surface(sl, strike=leg.strike, kind=leg.kind, r=r)
        q = leg.signed_quantity

        out.delta += sg.delta * q
        out.gamma += sg.gamma * q
        out.theta += sg.theta * q
        out.rho += sg.rho * q
        out.delta_black76 += sg.delta_black76 * q
        out.vega_by_expiry[leg.expiry] = out.vega_by_expiry.get(leg.expiry, 0.0) + sg.vega * q
        out.forwards[leg.expiry] = sl.forward.forward
        out.per_leg.append((leg, sg))
    return out
