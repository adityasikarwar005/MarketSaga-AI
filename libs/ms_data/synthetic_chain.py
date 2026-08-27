"""Synthetic option chain generator.

Prices every strike off a real Black-76 model with a volatility smile, so the
chain pipeline - snapshotter, implied-volatility inversion, quality gates, and
later the surface fit - can be built and tested before a broker account exists.

It is not a stub to be deleted. Realistic option data is the hardest thing to
obtain in this project, and a generator that reproduces the *structure* of a
chain (skew, term structure, wider spreads in the wings, open interest
concentrated near the money) is what keeps those components honest in CI, every
commit, forever.

The smile is deliberately asymmetric: equity index options carry a persistent
put skew, and a generator without it would make every skew feature in the
feature catalogue test vacuously.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.schemas import OptionType, Tick
from libs.ms_data.chain import OptionRegistry, minutes_to_expiry
from libs.ms_options.pricing import (
    MINUTES_PER_YEAR,
    OptType,
    black76_price,
    forward_from_spot,
)


@dataclass(frozen=True)
class SmileParams:
    """A crude but structurally honest volatility surface.

    ``skew`` is negative for equity indices: downside strikes trade at higher
    implied volatility than upside ones, because the demand for crash protection
    is real and persistent.
    """

    atm_vol: float = 0.14
    skew: float = -0.35  # d(vol)/d(log-moneyness)
    curvature: float = 0.60  # smile convexity
    term_slope: float = 0.02  # longer expiries priced slightly higher

    def vol(self, strike: float, forward: float, t: float) -> float:
        if strike <= 0 or forward <= 0:
            return self.atm_vol
        m = math.log(strike / forward)
        vol = self.atm_vol + self.skew * m + self.curvature * m * m
        vol += self.term_slope * math.sqrt(max(t, 0.0))
        return max(0.03, min(2.5, vol))


class SyntheticChainFeed:
    """Emits option ticks consistent with a given spot path.

    Drive it from the same spot series the synthetic index feed produces and the
    chain will move coherently with the underlying, which is what makes an
    end-to-end test meaningful rather than decorative.
    """

    def __init__(
        self,
        registry: OptionRegistry,
        calendar: SessionCalendar,
        *,
        underlying_id: int,
        smile: SmileParams | None = None,
        rate: float = 0.065,
        seed: int = 7,
        tick_size: float = 0.05,
        base_spread_ticks: int = 2,
    ) -> None:
        self._registry = registry
        self._cal = calendar
        self._underlying_id = underlying_id
        self._smile = smile or SmileParams()
        self._rate = rate
        self._rng = random.Random(seed)
        self._tick_size = tick_size
        self._base_spread_ticks = base_spread_ticks

    def _round(self, x: float) -> float:
        return round(round(x / self._tick_size) * self._tick_size, 2)

    def quotes_for(self, spot: float, now: datetime) -> Iterator[Tick]:
        """One tick per contract in the registry, priced off ``spot``."""
        today = self._cal.local(now).date()
        for contract in self._registry.for_underlying(self._underlying_id):
            if contract.expiry < today:
                continue
            minutes = minutes_to_expiry(self._cal, now, contract.expiry)
            t = minutes / MINUTES_PER_YEAR
            if t <= 0:
                continue

            fwd = forward_from_spot(spot, self._rate, t)
            vol = self._smile.vol(contract.strike, fwd, t)
            kind = OptType.CALL if contract.opt_type is OptionType.CE else OptType.PUT
            fair = black76_price(fwd, contract.strike, vol, t, self._rate, kind)
            if fair < self._tick_size:
                continue  # worthless wings do not print

            # Relative spread is driven mainly by PRICE, not by distance from the
            # money: a market maker quotes a roughly fixed absolute spread, so a
            # 700-rupee option trades at half a percent while a 5-rupee wing
            # trades at ten. This is the single most important microstructure
            # fact for strike selection, and a generator that got it wrong would
            # let the liquidity filter pass every strike in testing and none in
            # production.
            moneyness = abs(math.log(contract.strike / fwd))
            rel_spread = min(0.30, 0.004 + 0.6 / fair) * (1.0 + 4.0 * moneyness)
            half = max(self._tick_size / 2.0, rel_spread * fair / 2.0)

            mid = self._round(fair)
            bid = max(self._tick_size, self._round(mid - half))
            ask = self._round(mid + half)

            # Open interest concentrates sharply near the money, and round
            # strikes carry disproportionately more of it - writers anchor on
            # them. Both effects are load-bearing later: max pain and dealer
            # gamma (feature group C) are computed from this distribution, and a
            # generator with flat OI would make those features meaningless.
            oi_scale = math.exp(-1500.0 * moneyness * moneyness)
            if contract.strike % 500 == 0:
                oi_scale *= 1.8
            elif contract.strike % 100 == 0:
                oi_scale *= 1.25
            oi = int(50_000 * oi_scale) + self._rng.randint(0, 500)

            yield Tick(
                ts=now,
                instrument_id=contract.instrument_id,
                ltp=mid,
                qty=self._rng.randint(0, 20) * 25,
                bid=bid,
                ask=ask,
                bid_qty=self._rng.randint(1, 30) * 25,
                ask_qty=self._rng.randint(1, 30) * 25,
                oi=oi,
                src=0,
            )

    def true_vol(self, strike: float, spot: float, now: datetime, expiry: date) -> float:
        """The volatility actually used to price - the answer IV inversion must recover."""
        minutes = minutes_to_expiry(self._cal, now, expiry)
        t = minutes / MINUTES_PER_YEAR
        fwd = forward_from_spot(spot, self._rate, t)
        return self._smile.vol(strike, fwd, t)


def strike_grid(spot: float, *, step: float = 50.0, n_per_side: int = 20) -> list[float]:
    """A regular strike ladder around spot, as exchanges actually list them."""
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-n_per_side, n_per_side + 1)]


def weekly_expiries(
    calendar: SessionCalendar, start: date, count: int, weekday: int = 3
) -> list[date]:
    """The next ``count`` weekly expiries on the given weekday.

    Falls back to the previous trading day when an expiry lands on a holiday,
    which is what exchanges do and what the historical data will show.
    """
    from datetime import timedelta

    out: list[date] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() == weekday and cursor >= start:
            day = cursor
            for _ in range(5):
                try:
                    trading = calendar.is_trading_day(day)
                except LookupError:
                    trading = day.weekday() < 5
                if trading:
                    break
                day -= timedelta(days=1)
            if day not in out:
                out.append(day)
        cursor += timedelta(days=1)
    return out
