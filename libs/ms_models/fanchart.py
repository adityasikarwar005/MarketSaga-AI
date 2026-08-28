"""M61 - the fan chart, and the ratio that decides whether there is a trade.

Per-minute quantiles across the sampled paths, an implied-move band from the
surface laid over them, and one number out the end:

    edge_ratio = forecast |move| / implied |move|

Above 1 the model expects more movement than the options are charging for.
Below 1 it expects less. That single ratio is what turns a distribution into a
direction to lean, so it is worth being careful about what goes into it.

The two moves must be measured in the same year
------------------------------------------------

This is the whole hazard of the module, and it is already latent in the
codebase. ``pricing.MINUTES_PER_YEAR`` is **94 500** - 252 sessions of 375
minutes, trading time. ``surface.fit_surface`` computes its year fraction as
``(expiry - ts) / (365 * 24 * 3600)`` - **525 600** minutes, calendar time.

The two differ by 5.56x, so a move differs by **2.358x**. A 13-vol name over
sixty minutes implies a 0.3276% move on trading time and 0.1389% on calendar
time, and both are defensible numbers to print.

An ``edge_ratio`` that takes its implied side from a calendar-fitted IV and its
horizon from trading minutes is wrong by that factor. It does not look wrong.
It looks like *the model sees far more movement than the options are pricing*,
which is a specific, actionable, confident instruction to buy volatility.

So :class:`TimeBasis` has no default anywhere in this module. Every function
that turns a volatility into a move takes it explicitly, and the surface's basis
is CALENDAR because that is what M70 fitted with. Whether calendar time is the
right convention for an intraday horizon is a real question - options do not
decay over a weekend the way it implies - but it has to be answered by changing
what the surface fits, not by disagreeing with it here.

The tint has to be measured against the entry
-----------------------------------------------

``P(up)`` is the share of paths above **the level you entered at**, not above
the ensemble's own median. Measured against the median it is 50% at every
minute by construction, and the chart gets a tint that varies only with
sampling noise while looking exactly like information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

import numpy as np

from libs.ms_core.obs import get_logger
from libs.ms_models.paths import PathSet
from libs.ms_models.quantile import DEFAULT_QUANTILES

log = get_logger(__name__)

# 252 sessions x 375 minutes. What pricing.years_from_minutes uses.
TRADING_MINUTES_PER_YEAR = 252 * 375
# What surface.fit_surface uses for its year fractions.
CALENDAR_MINUTES_PER_YEAR = 365 * 24 * 60


class TimeBasis(StrEnum):
    """Which year a volatility is quoted against.

    Never defaulted. The two differ by 5.56x, so any move computed from a
    volatility differs by 2.358x depending on which one was meant, and the wrong
    one produces a confident instruction rather than an obvious error.
    """

    TRADING = "trading"
    CALENDAR = "calendar"

    @property
    def minutes_per_year(self) -> int:
        return TRADING_MINUTES_PER_YEAR if self is TimeBasis.TRADING else CALENDAR_MINUTES_PER_YEAR


# The basis M70 fits its surfaces on. Named rather than inlined so that changing
# the surface's convention breaks in one place instead of drifting apart.
SURFACE_BASIS = TimeBasis.CALENDAR


class FanChartError(RuntimeError):
    pass


def implied_move(iv: float, horizon_minutes: float, basis: TimeBasis) -> float:
    """One standard deviation of log-return over the horizon.

    ``basis`` is positional and required. A default here is the entire bug this
    module is written around.
    """
    if iv < 0.0:
        raise FanChartError("implied volatility cannot be negative")
    if horizon_minutes <= 0.0:
        raise FanChartError("horizon must be positive")
    return iv * math.sqrt(horizon_minutes / basis.minutes_per_year)


# --------------------------------------------------------------------------- #
# The fan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FanMinute:
    """One candle. Body is the interquartile range, wick the 5-95 band.

    These are cross-sectional across paths at a single minute, not an
    intra-minute OHLC: with one sample per path per minute there is no
    within-minute high or low to observe, and inventing one from the endpoints
    would understate the range by exactly the amount that matters to a stop.
    """

    minute: int
    quantiles: dict[float, float]  # log-return level, by alpha
    p_up: float

    def body(self) -> tuple[float, float]:
        return self.quantiles[0.25], self.quantiles[0.75]

    def wick(self) -> tuple[float, float]:
        return self.quantiles[0.05], self.quantiles[0.95]

    @property
    def median(self) -> float:
        return self.quantiles[0.50]

    def prices(self, spot: float) -> dict[float, float]:
        """The same candle in price terms. Levels are log-returns."""
        return {a: spot * math.exp(v) for a, v in self.quantiles.items()}


@dataclass
class ImpliedBand:
    """The option market's own view of the move, for the same horizon."""

    iv: float
    basis: TimeBasis
    horizon_minutes: int
    move: float

    def at_minute(self, minute: int) -> float:
        """One sd out to ``minute``, scaled as sqrt of time."""
        if minute <= 0:
            return 0.0
        return self.iv * math.sqrt(minute / self.basis.minutes_per_year)

    def render(self) -> str:
        return (
            f"    implied: iv {self.iv:.4f} on {self.basis.value} time, "
            f"{self.horizon_minutes}min move {self.move * 100:.4f}%"
        )


@dataclass
class FanChart:
    minutes: list[FanMinute]
    spot: float
    horizon_minutes: int
    implied: ImpliedBand | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def terminal(self) -> FanMinute:
        return self.minutes[-1]

    @property
    def forecast_move(self) -> float:
        """Half the terminal 5-95 band, in log-return terms.

        A half-band rather than a standard deviation, because the fan is built
        from empirical quantiles and a distribution with the tails this one has
        does not have a standard deviation worth quoting. Compared against 1.645
        sigma on the implied side so the two are the same width of the same
        thing.
        """
        lo, hi = self.terminal.wick()
        return (hi - lo) / 2.0

    @property
    def edge_ratio(self) -> float | None:
        """Forecast move over implied move. ``None`` if no band is attached.

        Both sides are 1.645 standard deviations: the forecast side is half the
        5-95 empirical band, and the implied side is the one-sd move scaled up
        by the same factor. Comparing a 90% band against a one-sd move gives a
        ratio of 1.645 on a model that agrees exactly with the market.
        """
        if self.implied is None:
            return None
        implied_half_band = self.implied.move * 1.6448536269514722
        if implied_half_band <= 0.0:
            return None
        return self.forecast_move / implied_half_band

    def monotone_at(self, minute: int) -> bool:
        q = self.minutes[minute].quantiles
        values = [q[a] for a in sorted(q)]
        return values == sorted(values)

    @property
    def widths(self) -> list[float]:
        return [m.quantiles[0.95] - m.quantiles[0.05] for m in self.minutes]

    def render(self) -> str:
        lines = [
            f"  fan chart: {self.horizon_minutes} minutes, spot {self.spot:.2f}, "
            f"{len(self.minutes)} candles"
        ]
        step = max(len(self.minutes) // 6, 1)
        lines.append(
            f"    {'min':>5} {'q05':>10} {'q25':>10} {'q50':>10} {'q75':>10} {'q95':>10} {'P(up)':>7}"
        )
        for m in [*self.minutes[::step], self.terminal]:
            q = m.quantiles
            lines.append(
                f"    {m.minute:>5} {q[0.05]:>+10.5f} {q[0.25]:>+10.5f} {q[0.50]:>+10.5f} "
                f"{q[0.75]:>+10.5f} {q[0.95]:>+10.5f} {m.p_up:>6.1%}"
            )
        if self.implied is not None:
            lines.append(self.implied.render())
            ratio = self.edge_ratio
            lines.append(
                f"    edge_ratio {ratio:.3f}  "
                + (
                    "model expects MORE move than options price"
                    if ratio and ratio > 1.0
                    else "model expects LESS move than options price"
                )
            )
        else:
            lines.append(
                "    no implied band attached - edge_ratio is unavailable, which is "
                "the honest state rather than a ratio against an assumed volatility"
            )
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def build_fan(
    paths: PathSet,
    *,
    spot: float,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    minute_of: Sequence[int] | None = None,
) -> FanChart:
    """Cross-sectional quantiles at every step of the sampled paths.

    ``P(up)`` counts paths strictly above the **entry** level, which is zero in
    log-return terms. Against the ensemble median it would be 50% at every
    minute by construction and the tint would carry nothing.
    """
    if spot <= 0.0:
        raise FanChartError("spot must be positive")
    alphas = sorted(quantiles)
    levels = paths.levels
    n_steps = levels.shape[1]
    steps = list(minute_of) if minute_of is not None else list(range(n_steps))
    if len(steps) != n_steps:
        raise FanChartError(f"minute_of has {len(steps)} entries for {n_steps} steps")

    out: list[FanMinute] = []
    for i, minute in enumerate(steps):
        column = levels[:, i]
        qs = {a: float(np.quantile(column, a)) for a in alphas}
        # Every path starts exactly at entry, so at i=0 "strictly above" is
        # zero paths and the first candle would tint as a certain fall. There
        # is no information at t=0, and 0.5 says so.
        p_up = 0.5 if i == 0 else float((column > 0.0).mean())
        out.append(FanMinute(minute=minute, quantiles=qs, p_up=p_up))

    fan = FanChart(minutes=out, spot=spot, horizon_minutes=steps[-1])

    if not all(fan.monotone_at(i) for i in range(len(out))):
        raise FanChartError(
            "quantiles cross at some minute. Empirical quantiles of one sample "
            "cannot cross, so this means the paths were assembled wrongly - and a "
            "crossed candle is a body with negative height."
        )

    widths = fan.widths
    shrinking = sum(1 for a, b in pairwise(widths) if b < a - 1e-12)
    if shrinking > len(widths) * 0.2:
        fan.notes.append(
            f"the band narrows at {shrinking} of {len(widths) - 1} steps. A fan that "
            "closes says the future gets more certain the further out you look, "
            "which for a random walk means the paths are not one."
        )
    return fan


def attach_implied(
    fan: FanChart, *, iv: float, basis: TimeBasis, horizon_minutes: int | None = None
) -> FanChart:
    """Lay the option market's move over the fan.

    ``basis`` is required and there is no sensible default: the surface fits on
    calendar time and ``pricing.years_from_minutes`` uses trading time, and
    picking the wrong one scales the whole comparison by 2.358x.
    """
    horizon = horizon_minutes if horizon_minutes is not None else fan.horizon_minutes
    fan.implied = ImpliedBand(
        iv=iv,
        basis=basis,
        horizon_minutes=horizon,
        move=implied_move(iv, horizon, basis),
    )
    ratio = fan.edge_ratio
    log.info(
        "implied band attached",
        fields={"iv": iv, "basis": basis.value, "edge_ratio": ratio},
    )
    return fan


def edge_ratio_from_surface(
    fan: FanChart, *, iv: float, horizon_minutes: int | None = None
) -> float | None:
    """Convenience for the common path: an IV that came from M70.

    Uses :data:`SURFACE_BASIS`, so a caller cannot pick the wrong one by
    accident, and so that changing what the surface fits breaks here rather than
    silently rescaling every edge estimate.
    """
    attach_implied(fan, iv=iv, basis=SURFACE_BASIS, horizon_minutes=horizon_minutes)
    return fan.edge_ratio
