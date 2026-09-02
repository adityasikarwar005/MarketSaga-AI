"""The projection drawn on `chart.html`, and the three ways it could quietly be wrong.

The arithmetic is closed form, which makes it checkable rather than merely
plausible - so it is checked here against the two things that can settle it: a
Monte Carlo of the same process, and Black-76 itself.

Each of these is a defect class this repository has already been bitten by once:

* **a sign that only shows up on one side.** A put is decreasing in the forward,
  so its price quantiles come from the forward's *opposite* quantiles. Mapping
  both the same way produces an inverted band on puts alone - calls keep looking
  right, and the theta sign error in M63 hid under eighty-two passing tests for
  exactly this reason.
* **lookahead.** `_forward_at` must never return a forward from after the moment
  being priced. This is the rule the whole point-in-time layer exists to enforce.
* **a forecast of a market that is shut.** Projecting an hour past 15:30 puts
  bars where nothing can trade, and they set the chart's vertical scale.
"""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta
from math import exp, sqrt
from zoneinfo import ZoneInfo

import pytest

from jobs.chart_history import FORECAST_STEPS, _anchor, _forward_at, quantiles
from jobs.option_history import strike_step
from libs.ms_core.schemas import Candle
from libs.ms_options.pricing import MINUTES_PER_YEAR, OptType, black76_price

IST = ZoneInfo("Asia/Kolkata")
DAY = datetime(2026, 9, 2, 9, 15, tzinfo=IST)
CLOSE = datetime(2026, 9, 2, 15, 30, tzinfo=IST)

F0 = 24000.0
K = 24000.0
SIGMA = 0.12
T0 = 6.0 / 365.0
R = 0.065


def _bar(minute: int, close: float = 100.0) -> Candle:
    ts = DAY + timedelta(minutes=minute)
    return Candle(ts=ts, instrument_id=1, tf=1, o=close, h=close, low=close, c=close)


class TestQuantiles:
    def test_the_median_matches_a_monte_carlo_of_the_same_process(self) -> None:
        """The closed form is only worth having if it is the same answer.

        `jobs/option_forecast.py` simulates because it re-reads a smile at the
        new moneyness and so loses monotonicity in the forward. Here volatility
        is flat, the price is monotone in the forward, and the quantile maps
        straight through - which is an argument, not a measurement, until it is
        checked against the thing it replaces.
        """
        minutes = 60.0
        got = quantiles(F0, SIGMA, K, T0, minutes, OptType.CALL, R)
        assert got is not None

        dt = minutes / MINUTES_PER_YEAR
        t1 = T0 - dt
        rng = random.Random(4242)
        drift = -0.5 * SIGMA * SIGMA * dt
        step = SIGMA * sqrt(dt)
        prices = [
            black76_price(
                F0 * exp(drift + step * rng.gauss(0.0, 1.0)), K, SIGMA, t1, R, OptType.CALL
            )
            for _ in range(40000)
        ]
        simulated = statistics.median(prices)

        assert got["q50"] == pytest.approx(simulated, rel=0.01), (
            "the closed-form median disagrees with a Monte Carlo of the same "
            "process. One of them is wrong, and the chart draws the closed form."
        )

    def test_a_put_takes_the_forwards_other_tail(self) -> None:
        """The reflection, checked on the only side where getting it wrong shows.

        A call's q90 price comes from the forward's q90; a put's q90 price comes
        from the forward's q10. If both were mapped the same way the put's band
        would be inverted - q10 above q90 - which draws as a perfectly ordinary
        box because the renderer sorts them.
        """
        minutes = 60.0
        dt = minutes / MINUTES_PER_YEAR
        t1 = T0 - dt
        drift = -0.5 * SIGMA * SIGMA * dt
        step = SIGMA * sqrt(dt)

        put = quantiles(F0, SIGMA, K, T0, minutes, OptType.PUT, R)
        assert put is not None

        f_low = F0 * exp(drift + step * -1.2816)
        expected_high = black76_price(f_low, K, SIGMA, t1, R, OptType.PUT)
        assert put["q90"] == pytest.approx(expected_high, rel=1e-9), (
            "a put's upper price quantile was not taken from the forward's lower "
            "one. The put is decreasing in the forward; mapping it like a call "
            "inverts the band and nothing downstream can tell."
        )
        assert put["q10"] < put["q50"] < put["q90"]

    def test_both_sides_are_ordered(self) -> None:
        for kind in (OptType.CALL, OptType.PUT):
            got = quantiles(F0, SIGMA, K, T0, 30.0, kind, R)
            assert got is not None
            assert (
                got["q10"] <= got["q25"] <= got["q50"] <= got["q75"] <= got["q90"]
            ), f"{kind.value} quantiles came back out of order"

    def test_the_band_widens_with_the_horizon(self) -> None:
        near = quantiles(F0, SIGMA, K, T0, 5.0, OptType.CALL, R)
        far = quantiles(F0, SIGMA, K, T0, 60.0, OptType.CALL, R)
        assert near is not None and far is not None
        assert (far["q90"] - far["q10"]) > (near["q90"] - near["q10"])

    def test_it_refuses_rather_than_guessing(self) -> None:
        """Past expiry, and at zero volatility, there is no distribution."""
        assert quantiles(F0, SIGMA, K, T0, T0 * MINUTES_PER_YEAR + 1, OptType.CALL, R) is None
        assert quantiles(F0, 0.0, K, T0, 30.0, OptType.CALL, R) is None


class TestForwardAt:
    def test_it_never_returns_a_forward_from_the_future(self) -> None:
        """Rule 2: point-in-time or it does not exist."""
        future = [_bar(i, 24000.0 + i) for i in range(0, 60, 5)]
        asked = DAY + timedelta(minutes=22)
        got = _forward_at(future, asked)
        assert got == 24020.0, (
            "the forward used to price a bar at 09:37 came from a later minute. "
            "A price from after the decision is lookahead, and it is invisible "
            "in every downstream number."
        )

    def test_a_minute_the_future_did_not_trade_falls_back_not_through(self) -> None:
        """A dict lookup would return nothing here and drop the contract."""
        future = [_bar(0, 24000.0), _bar(10, 24010.0)]
        assert _forward_at(future, DAY + timedelta(minutes=7)) == 24000.0

    def test_before_the_first_bar_there_is_no_forward(self) -> None:
        assert _forward_at([_bar(10, 24010.0)], DAY) is None


class TestAnchor:
    def test_a_finished_day_is_projected_from_inside_it(self) -> None:
        """Otherwise every projected bar lands in a market that is shut.

        A forecast made at the close cannot be checked and cannot be traded, and
        an hour of diffusion past 15:30 was setting the chart's whole vertical
        scale - the widest thing on the page was the part that could not happen.
        """
        bars = [_bar(i) for i in range(0, 376)]  # 09:15 through 15:30
        got = _anchor(bars, CLOSE)
        assert got.ts <= CLOSE - timedelta(minutes=max(FORECAST_STEPS))
        assert got.ts.astimezone(IST).strftime("%H:%M") == "14:30"

    def test_a_live_session_is_projected_from_its_last_bar(self) -> None:
        bars = [_bar(i) for i in range(0, 120)]  # up to 11:15, hours left
        assert _anchor(bars, CLOSE) is bars[-1]

    def test_no_projected_bar_lands_past_the_close(self) -> None:
        bars = [_bar(i) for i in range(0, 376)]
        anchor = _anchor(bars, CLOSE)
        drawn = [anchor.ts + timedelta(minutes=s) for s in FORECAST_STEPS]
        assert all(ts <= CLOSE for ts in drawn), (
            "a projected bar was placed after the session close, where nothing " "can trade into it"
        )


class TestStrikeStep:
    def test_it_reads_the_grid_rather_than_assuming_one(self) -> None:
        assert strike_step([24000.0, 24050.0, 24100.0, 24150.0]) == 50.0
        assert strike_step([57000.0, 57100.0, 57200.0]) == 100.0

    def test_a_sparse_wing_does_not_move_it(self) -> None:
        """Exchanges list the wings coarsely, and the mean gap would follow."""
        grid = [23000.0, 24000.0] + [24000.0 + 50 * i for i in range(1, 20)] + [26000.0]
        assert strike_step(grid) == 50.0
