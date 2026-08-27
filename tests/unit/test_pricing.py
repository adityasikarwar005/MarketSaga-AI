"""M71 verify - pricing, Greeks, implied volatility.

Tested against identities that must hold for any correct implementation, not
against numbers copied from another library. If put-call parity holds, delta
bounds hold, and implied volatility round-trips, the model is right.
"""

from __future__ import annotations

import math

import pytest

from libs.ms_options.pricing import (
    MAX_VOL,
    MINUTES_PER_YEAR,
    Greeks,
    OptType,
    black76_greeks,
    black76_price,
    forward_from_spot,
    implied_move,
    implied_move_from_vol,
    implied_vol,
    intrinsic,
    norm_cdf,
    put_call_parity_gap,
    years_from_minutes,
)

F, K, T, R, VOL = 24_800.0, 24_800.0, 7 / 252, 0.065, 0.14


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


def test_put_call_parity_holds():
    """The identity every option model must satisfy."""
    c = black76_price(F, K, VOL, T, R, OptType.CALL)
    p = black76_price(F, K, VOL, T, R, OptType.PUT)
    assert put_call_parity_gap(c, p, F, K, T, R) == pytest.approx(0.0, abs=1e-8)


def test_parity_holds_away_from_the_money():
    for strike in (23_000.0, 24_000.0, 25_500.0, 27_000.0):
        c = black76_price(F, strike, VOL, T, R, OptType.CALL)
        p = black76_price(F, strike, VOL, T, R, OptType.PUT)
        assert put_call_parity_gap(c, p, F, strike, T, R) == pytest.approx(0.0, abs=1e-8)


def test_price_is_above_intrinsic_before_expiry():
    itm = black76_price(F, 23_000.0, VOL, T, R, OptType.CALL)
    assert itm > intrinsic(F, 23_000.0, T, R, OptType.CALL)


def test_price_is_monotone_in_volatility():
    prices = [black76_price(F, K, v, T, R, OptType.CALL) for v in (0.05, 0.10, 0.20, 0.40)]
    assert prices == sorted(prices)


def test_calls_decrease_in_strike_puts_increase():
    strikes = [23_500.0, 24_000.0, 24_500.0, 25_000.0, 25_500.0]
    calls = [black76_price(F, k, VOL, T, R, OptType.CALL) for k in strikes]
    puts = [black76_price(F, k, VOL, T, R, OptType.PUT) for k in strikes]
    assert calls == sorted(calls, reverse=True)
    assert puts == sorted(puts)


def test_expiry_returns_intrinsic_not_an_error():
    """An option at expiry is well defined. The caller should not special-case it."""
    assert black76_price(F, 24_000.0, VOL, 0.0, R, OptType.CALL) == pytest.approx(800.0)
    assert black76_price(F, 25_000.0, VOL, 0.0, R, OptType.CALL) == pytest.approx(0.0)
    assert black76_price(F, 25_000.0, VOL, 0.0, R, OptType.PUT) == pytest.approx(200.0)


def test_zero_volatility_returns_intrinsic():
    assert black76_price(F, 24_000.0, 0.0, T, R, OptType.CALL) == pytest.approx(
        intrinsic(F, 24_000.0, T, R, OptType.CALL)
    )


# ------------------------------------------------------------------- Greeks
def test_delta_bounds_and_signs():
    for k in (23_000.0, 24_800.0, 26_500.0):
        c = black76_greeks(F, k, VOL, T, R, OptType.CALL)
        p = black76_greeks(F, k, VOL, T, R, OptType.PUT)
        assert 0.0 <= c.delta <= 1.0
        assert -1.0 <= p.delta <= 0.0


def test_atm_delta_is_near_half():
    g = black76_greeks(F, F, VOL, T, R, OptType.CALL)
    assert 0.45 < g.delta < 0.55


def test_gamma_and_vega_are_non_negative_and_peak_atm():
    strikes = [22_000.0, 23_500.0, 24_800.0, 26_000.0, 27_500.0]
    gammas = [black76_greeks(F, k, VOL, T, R, OptType.CALL).gamma for k in strikes]
    vegas = [black76_greeks(F, k, VOL, T, R, OptType.CALL).vega for k in strikes]
    assert all(g >= 0 for g in gammas)
    assert all(v >= 0 for v in vegas)
    assert gammas.index(max(gammas)) == 2  # ATM
    assert vegas.index(max(vegas)) == 2


def test_call_and_put_share_gamma_and_vega():
    """Parity implies identical second-order Greeks. A classic sign-error catcher."""
    c = black76_greeks(F, K, VOL, T, R, OptType.CALL)
    p = black76_greeks(F, K, VOL, T, R, OptType.PUT)
    assert c.gamma == pytest.approx(p.gamma)
    assert c.vega == pytest.approx(p.vega)


def test_delta_matches_a_numerical_derivative():
    eps = 0.01
    up = black76_price(F + eps, K, VOL, T, R, OptType.CALL)
    dn = black76_price(F - eps, K, VOL, T, R, OptType.CALL)
    numeric = (up - dn) / (2 * eps)
    assert black76_greeks(F, K, VOL, T, R, OptType.CALL).delta == pytest.approx(numeric, rel=1e-4)


def test_vega_matches_a_numerical_derivative():
    eps = 1e-5
    up = black76_price(F, K, VOL + eps, T, R, OptType.CALL)
    dn = black76_price(F, K, VOL - eps, T, R, OptType.CALL)
    numeric = (up - dn) / (2 * eps)
    assert black76_greeks(F, K, VOL, T, R, OptType.CALL).vega == pytest.approx(numeric, rel=1e-4)


def test_theta_is_negative_for_long_options():
    assert black76_greeks(F, K, VOL, T, R, OptType.CALL).theta < 0
    assert black76_greeks(F, K, VOL, T, R, OptType.PUT).theta < 0


def test_vega_per_point_and_theta_per_day_conversions():
    g: Greeks = black76_greeks(F, K, VOL, T, R, OptType.CALL)
    assert g.vega_per_point == pytest.approx(g.vega / 100.0)
    assert g.theta_per_day == pytest.approx(g.theta / 252)
    assert g.theta_per_minutes(375) == pytest.approx(g.theta * years_from_minutes(375))


def test_greeks_at_expiry_do_not_explode():
    g = black76_greeks(F, 24_000.0, VOL, 0.0, R, OptType.CALL)
    assert g.delta == 1.0 and g.gamma == 0.0 and g.vega == 0.0


# --------------------------------------------------------- implied volatility
def test_implied_vol_round_trips():
    for k in (23_000.0, 24_000.0, 24_800.0, 25_500.0, 26_500.0):
        for kind in (OptType.CALL, OptType.PUT):
            price = black76_price(F, k, VOL, T, R, kind)
            recovered = implied_vol(price, F, k, T, R, kind)
            assert recovered is not None, f"failed at {k} {kind}"
            assert recovered == pytest.approx(VOL, abs=1e-5)


def test_implied_vol_round_trips_across_a_wide_vol_range():
    for vol in (0.05, 0.10, 0.25, 0.50, 1.00):
        price = black76_price(F, K, vol, T, R, OptType.CALL)
        assert implied_vol(price, F, K, T, R, OptType.CALL) == pytest.approx(vol, abs=1e-5)


def test_implied_vol_works_in_the_deep_wings_where_vega_collapses():
    """Newton stalls here; the bisection fallback must take over."""
    k = 30_000.0
    price = black76_price(F, k, 0.35, T, R, OptType.CALL)
    recovered = implied_vol(price, F, k, T, R, OptType.CALL)
    assert recovered is not None
    assert recovered == pytest.approx(0.35, abs=1e-3)


def test_implied_vol_returns_none_below_intrinsic():
    """A stale quote must produce no opinion, not a fabricated number."""
    k = 23_000.0
    below = intrinsic(F, k, T, R, OptType.CALL) * 0.5
    assert implied_vol(below, F, k, T, R, OptType.CALL) is None


def test_implied_vol_returns_none_above_the_arbitrage_bound():
    assert implied_vol(F * 2, F, K, T, R, OptType.CALL) is None


def test_implied_vol_returns_none_at_expiry_and_for_junk_input():
    assert implied_vol(100.0, F, K, 0.0, R, OptType.CALL) is None
    assert implied_vol(0.0, F, K, T, R, OptType.CALL) is None
    assert implied_vol(-5.0, F, K, T, R, OptType.CALL) is None


def test_implied_vol_never_returns_out_of_range_values():
    for price in (0.05, 1.0, 50.0, 500.0, 2000.0):
        iv = implied_vol(price, F, K, T, R, OptType.CALL)
        if iv is not None:
            assert 0 < iv < MAX_VOL


# ------------------------------------------------------------- chain helpers
def test_forward_exceeds_spot_with_a_positive_rate():
    assert forward_from_spot(24_800.0, 0.065, T) > 24_800.0
    assert forward_from_spot(24_800.0, 0.0, T) == pytest.approx(24_800.0)


def test_implied_move_from_a_straddle():
    c = black76_price(F, K, VOL, T, R, OptType.CALL)
    p = black76_price(F, K, VOL, T, R, OptType.PUT)
    move = implied_move(c, p, F)
    # A 14% annual vol over seven trading days is roughly 2.3% of spot,
    # and the straddle is worth about 0.8 of that.
    assert 0.01 < move < 0.04


def test_implied_move_scales_with_the_square_root_of_time():
    a = implied_move_from_vol(0.14, 15)
    b = implied_move_from_vol(0.14, 60)
    assert b / a == pytest.approx(2.0, rel=1e-9)


def test_intraday_horizon_uses_trading_time_not_calendar_time():
    """375 minutes is one session, so one session is 1/252 of a year."""
    assert years_from_minutes(MINUTES_PER_YEAR) == pytest.approx(1.0)
    assert years_from_minutes(375) == pytest.approx(1 / 252)


def test_parity_gap_detects_a_stale_leg():
    c = black76_price(F, K, VOL, T, R, OptType.CALL)
    p = black76_price(F, K, VOL, T, R, OptType.PUT)
    assert abs(put_call_parity_gap(c, p, F, K, T, R)) < 1e-8
    assert abs(put_call_parity_gap(c, p * 0.8, F, K, T, R)) > 1.0  # one leg stale


def test_no_nan_or_inf_anywhere_on_a_realistic_grid():
    for k in (20_000.0, 24_800.0, 30_000.0):
        for t in (1 / MINUTES_PER_YEAR, 1 / 252, 30 / 252):
            for v in (0.05, 0.20, 0.80):
                g = black76_greeks(F, k, v, t, R, OptType.CALL)
                for value in (g.price, g.delta, g.gamma, g.vega, g.theta, g.rho):
                    assert math.isfinite(value), f"non-finite at k={k} t={t} v={v}"
