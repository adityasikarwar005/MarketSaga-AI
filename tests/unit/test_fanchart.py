"""M61 - the fan chart, and the ratio that decides whether there is a trade.

    pytest tests/unit/test_fanchart.py -k "monotone_quantiles or implied_overlay_present"

The verify command names the two structural properties. The one it does not
name is the one that costs money: the forecast move and the implied move have
to be measured against the same year. They are not, by default, anywhere in
this codebase - `pricing.MINUTES_PER_YEAR` is trading time and the surface fits
on calendar time - and getting it wrong scales `edge_ratio` by 2.358x without
anything looking wrong.
"""

from __future__ import annotations

import math

import pytest

from libs.ms_models.fanchart import (
    CALENDAR_MINUTES_PER_YEAR,
    SURFACE_BASIS,
    TRADING_MINUTES_PER_YEAR,
    FanChartError,
    TimeBasis,
    attach_implied,
    build_fan,
    edge_ratio_from_surface,
    implied_move,
)
from libs.ms_models.paths import calibrate, simulate
from libs.ms_models.quantile import DEFAULT_QUANTILES, QuantileForecast
from libs.ms_options.pricing import MINUTES_PER_YEAR

SPOT = 24000.0
HORIZON = 60
VAR0 = (0.0021 / 3.29) ** 2 / HORIZON
Z90 = 1.6448536269514722


def sample(seed: int = 5, steps: int = HORIZON, n: int = 5000):
    return simulate(n_paths=n, steps=steps, variance0=VAR0, seed=seed)


def fan(**kw):
    return build_fan(sample(), spot=SPOT, **kw)


# --------------------------------------------------------------------------- #
# monotone_quantiles
# --------------------------------------------------------------------------- #
def test_monotone_quantiles_at_every_minute():
    """A crossed candle is a body with negative height, and the body is what a
    reader sizes off."""
    f = fan()
    for i in range(len(f.minutes)):
        assert f.monotone_at(i), f"quantiles cross at minute {f.minutes[i].minute}"


def test_monotone_quantiles_are_checked_at_construction_not_at_render():
    """The chart is built once and read many times. Catching a crossing when
    somebody prints it means the wrong numbers already went downstream."""
    import inspect

    src = inspect.getsource(build_fan)
    assert "monotone_at" in src
    assert "FanChartError" in src


def test_monotone_quantiles_hold_for_the_body_and_the_wick_separately():
    f = fan()
    for m in f.minutes:
        lo_body, hi_body = m.body()
        lo_wick, hi_wick = m.wick()
        assert lo_wick <= lo_body <= hi_body <= hi_wick


def test_the_fan_starts_closed():
    """Every path begins at the entry price, so minute zero has no spread.
    A fan open at t=0 is showing uncertainty that has not happened yet."""
    first = fan().minutes[0]
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in first.quantiles.values())


def test_the_fan_opens_with_time():
    widths = fan().widths
    assert widths[0] == pytest.approx(0.0, abs=1e-12)
    assert widths[-1] > widths[len(widths) // 2] > widths[1]


def test_a_narrowing_fan_is_reported_rather_than_drawn_quietly():
    """A band that closes says the future gets more certain the further out you
    look. On a random walk that means the paths are not one."""
    p = sample()
    # Squeeze the back half progressively, so the band closes step after step
    # rather than dropping once and resuming its climb - one narrowing step is
    # ordinary sampling noise and should not raise a note.
    half = p.steps // 2
    for i in range(half, p.steps + 1):
        p.levels[:, i] *= 1.0 - 0.9 * (i - half) / (p.steps - half)

    f = build_fan(p, spot=SPOT)
    assert any("band narrows" in n for n in f.notes)
    assert not any("band narrows" in n for n in fan().notes)


# --------------------------------------------------------------------------- #
# implied_overlay_present
# --------------------------------------------------------------------------- #
def test_implied_overlay_present_after_attaching():
    f = fan()
    assert f.implied is None
    assert f.edge_ratio is None

    attach_implied(f, iv=0.13, basis=TimeBasis.CALENDAR)
    assert f.implied is not None
    assert f.edge_ratio is not None
    assert "implied" in f.render()


def test_implied_overlay_present_is_absent_honestly_when_it_is_not():
    """No band means no ratio, said out loud - rather than a ratio computed
    against an assumed volatility that nobody chose."""
    text = fan().render()
    assert "no implied band attached" in text
    assert "edge_ratio" not in text.split("no implied band")[0]


def test_implied_overlay_present_scales_as_root_time():
    band = attach_implied(fan(), iv=0.13, basis=TimeBasis.CALENDAR).implied
    assert band is not None
    assert band.at_minute(0) == 0.0
    assert band.at_minute(60) == pytest.approx(band.at_minute(15) * 2.0, rel=1e-9)


def test_implied_overlay_present_uses_the_surface_basis_by_construction():
    """`edge_ratio_from_surface` exists so a caller cannot pick the wrong basis
    by accident, and so a change to what M70 fits breaks here rather than
    silently rescaling every edge estimate."""
    assert SURFACE_BASIS is TimeBasis.CALENDAR

    f = fan()
    ratio = edge_ratio_from_surface(f, iv=0.13)
    assert f.implied is not None and f.implied.basis is TimeBasis.CALENDAR

    direct = attach_implied(fan(), iv=0.13, basis=TimeBasis.CALENDAR)
    assert ratio == pytest.approx(direct.edge_ratio)


# --------------------------------------------------------------------------- #
# the basis hazard - what the verify command does not name
# --------------------------------------------------------------------------- #
def test_the_two_year_conventions_really_do_differ_by_the_factor_claimed():
    """Pinned so the docstring's 2.358 cannot drift away from the constants."""
    assert TRADING_MINUTES_PER_YEAR == MINUTES_PER_YEAR == 94_500
    assert CALENDAR_MINUTES_PER_YEAR == 525_600

    ratio = math.sqrt(CALENDAR_MINUTES_PER_YEAR / TRADING_MINUTES_PER_YEAR)
    assert ratio == pytest.approx(2.358, abs=0.001)


def test_the_basis_alone_can_flip_the_decision():
    """Not merely rescale it. With a forecast band sitting between the two
    implied moves, calendar time says the model expects more movement than the
    options are charging for and trading time says less - buy against sell, on
    the same fan, the same IV and the same horizon.
    """
    f = fan()
    # An IV chosen so the calendar-basis ratio lands at 1.5.
    iv = f.forecast_move / (1.5 * Z90 * math.sqrt(HORIZON / CALENDAR_MINUTES_PER_YEAR))

    cal = attach_implied(fan(), iv=iv, basis=TimeBasis.CALENDAR).edge_ratio
    trd = attach_implied(fan(), iv=iv, basis=TimeBasis.TRADING).edge_ratio

    assert cal is not None and trd is not None
    assert cal > 1.0 > trd, f"calendar {cal:.3f}, trading {trd:.3f}"
    assert cal / trd == pytest.approx(2.358, abs=0.005)


def test_implied_move_has_no_default_basis():
    """The default is the bug. Every call site has to say which year it means."""
    import inspect

    sig = inspect.signature(implied_move)
    assert sig.parameters["basis"].default is inspect.Parameter.empty

    sig2 = inspect.signature(attach_implied)
    assert sig2.parameters["basis"].default is inspect.Parameter.empty


def test_implied_move_matches_the_closed_form():
    for basis in (TimeBasis.TRADING, TimeBasis.CALENDAR):
        assert implied_move(0.13, 60, basis) == pytest.approx(
            0.13 * math.sqrt(60 / basis.minutes_per_year)
        )


def test_a_negative_volatility_is_refused():
    with pytest.raises(FanChartError, match="cannot be negative"):
        implied_move(-0.1, 60, TimeBasis.CALENDAR)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(FanChartError, match="horizon must be positive"):
        implied_move(0.13, 0, TimeBasis.CALENDAR)


# --------------------------------------------------------------------------- #
# edge_ratio
# --------------------------------------------------------------------------- #
def test_edge_ratio_is_one_when_the_model_agrees_with_the_market():
    """Both sides are 1.645 standard deviations: half the empirical 5-95 band
    against the one-sd implied move scaled by the same factor. Comparing a 90%
    band against a one-sd move would give 1.645 for a model that agrees exactly,
    and somebody would read that as a large edge."""
    f = fan()
    agreeing_iv = f.forecast_move / (Z90 * math.sqrt(HORIZON / CALENDAR_MINUTES_PER_YEAR))

    attach_implied(f, iv=agreeing_iv, basis=TimeBasis.CALENDAR)
    assert f.edge_ratio == pytest.approx(1.0, abs=1e-9)


def test_edge_ratio_rises_when_the_market_charges_less():
    f_cheap = attach_implied(fan(), iv=0.05, basis=TimeBasis.CALENDAR)
    f_dear = attach_implied(fan(), iv=0.50, basis=TimeBasis.CALENDAR)
    assert f_cheap.edge_ratio > f_dear.edge_ratio


def test_edge_ratio_is_none_rather_than_infinite_at_zero_volatility():
    f = attach_implied(fan(), iv=0.0, basis=TimeBasis.CALENDAR)
    assert f.edge_ratio is None


def test_the_forecast_move_is_the_half_band_not_a_standard_deviation():
    """The fan is built from empirical quantiles of a fat-tailed sample, and a
    distribution with this kurtosis does not have a standard deviation worth
    quoting."""
    f = fan()
    lo, hi = f.terminal.wick()
    assert f.forecast_move == pytest.approx((hi - lo) / 2.0)


# --------------------------------------------------------------------------- #
# the tint
# --------------------------------------------------------------------------- #
def test_p_up_is_measured_against_the_entry_not_the_median():
    """Against the ensemble median it is 50% at every minute by construction,
    and the chart gets a tint that varies only with sampling noise while looking
    exactly like information."""
    p = sample()
    # Push every path up by a constant: P(up) against entry must go to ~1,
    # while against the median it would not move at all.
    p.levels += 0.01
    p.levels[:, 0] = 0.0

    f = build_fan(p, spot=SPOT)
    assert f.terminal.p_up > 0.99


def test_p_up_is_a_half_at_entry_rather_than_zero():
    """Nothing is strictly above the entry at minute zero, so a naive count
    tints the first candle as a certain fall."""
    assert fan().minutes[0].p_up == 0.5


def test_p_up_tracks_a_drift():
    p = sample()
    drift = [0.00002 * i for i in range(p.steps + 1)]
    p.levels += drift

    f = build_fan(p, spot=SPOT)
    assert f.terminal.p_up > f.minutes[len(f.minutes) // 2].p_up > 0.5


# --------------------------------------------------------------------------- #
# candles and prices
# --------------------------------------------------------------------------- #
def test_the_candle_is_cross_sectional_not_intra_minute():
    """With one sample per path per minute there is no within-minute high or
    low to observe, and inventing one from the endpoints would understate the
    range by exactly the amount that matters to a stop."""
    from libs.ms_models.fanchart import FanMinute

    assert FanMinute.__doc__ is not None
    assert "intra-minute" in FanMinute.__doc__


def test_prices_convert_from_log_returns():
    m = fan().terminal
    prices = m.prices(SPOT)
    assert prices[0.5] == pytest.approx(SPOT * math.exp(m.quantiles[0.5]))
    assert prices[0.95] > prices[0.05]


def test_the_fan_can_carry_real_clock_minutes():
    """A chart drawn against session minutes needs the actual minute labels, not
    step indices, or the implied band is laid over the wrong x axis."""
    p = sample(steps=10, n=1000)
    labels = [375 - 10 + i for i in range(p.steps + 1)]
    f = build_fan(p, spot=SPOT, minute_of=labels)

    assert [m.minute for m in f.minutes] == labels
    assert f.horizon_minutes == 375


def test_a_mismatched_minute_labelling_is_refused():
    p = sample(steps=10, n=1000)
    with pytest.raises(FanChartError, match="entries for"):
        build_fan(p, spot=SPOT, minute_of=[1, 2, 3])


def test_a_non_positive_spot_is_refused():
    with pytest.raises(FanChartError, match="spot must be positive"):
        build_fan(sample(steps=5, n=500), spot=0.0)


# --------------------------------------------------------------------------- #
# with the calibrated paths it will actually be given
# --------------------------------------------------------------------------- #
def test_the_fan_terminal_matches_the_quantile_heads_it_was_calibrated_to():
    """End to end: the heads set the terminal distribution, the paths carry the
    route, and the fan's last candle has to agree with the heads or the chart is
    drawing a different forecast from the one the model made."""
    target = QuantileForecast(
        quantiles={0.05: -0.0086, 0.25: -0.0050, 0.50: 0.0005, 0.75: 0.0063, 0.95: 0.0129}
    )
    calibrated = calibrate(sample(), target)
    f = build_fan(calibrated, spot=SPOT)

    for alpha in sorted(DEFAULT_QUANTILES):
        assert f.terminal.quantiles[alpha] == pytest.approx(
            target.quantiles[alpha], abs=2e-5
        ), f"a={alpha}"
