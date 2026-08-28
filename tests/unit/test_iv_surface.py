"""M70 - the IV surface, and the four ways it goes wrong without erroring.

The verify command names them:

    pytest tests/unit/test_iv_surface.py -k "no_butterfly_arb or no_calendar_arb or rejects_stale or mid_not_ltp"

Every one of those produces a surface that fits well, prices every individual
option at a positive number, and is wrong. Butterfly arbitrage shows up as a
negative probability density that no single price reveals. Calendar arbitrage is
a calendar spread worth less than nothing. A stale surface is a photograph of a
volatility that has stopped existing. And a surface built from last-traded
prices fits *better* than one built from mid, because stale prices are smooth.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_options.pricing import OptType, black76_price
from libs.ms_options.surface import (
    MAX_SURFACE_AGE,
    MIN_QUOTES_PER_EXPIRY,
    ArbitrageError,
    ExpirySlice,
    FilterStats,
    Observation,
    RejectReason,
    StaleSurfaceError,
    Surface,
    SurfaceError,
    SVIParams,
    butterfly_violations,
    calendar_violations,
    durrleman_g,
    fit_rmse,
    fit_surface,
    fit_svi,
    iv_from_quote,
    synthetic_forward,
)

TS = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
EXPIRY = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
SPOT = 24000.0
R = 0.065

# A well-behaved smile: mild negative skew, positive curvature, arbitrage-free.
TRUE = SVIParams(a=0.010, b=0.055, rho=-0.42, m=0.015, sigma=0.115)


def year_fraction(ts: datetime = TS, expiry: datetime = EXPIRY) -> float:
    return (expiry - ts).total_seconds() / (365.0 * 24 * 3600)


def observations(
    params: SVIParams = TRUE,
    *,
    t: float | None = None,
    n: int = 25,
    lo: float = -0.30,
    hi: float = 0.30,
) -> list[Observation]:
    t = t or year_fraction()
    out = []
    for i in range(n):
        k = lo + (hi - lo) * i / (n - 1)
        w = params.total_variance(k)
        out.append(
            Observation(
                strike=SPOT * math.exp(k),
                k=k,
                iv=math.sqrt(w / t),
                total_variance=w,
                kind=OptType.CALL,
                delta=0.5,
                weight=1.0,
            )
        )
    return out


def quote(
    strike: float,
    opt: OptionType,
    *,
    bid: float | None,
    ask: float | None,
    ltp: float | None = None,
    oi: int = 5000,
    expiry: datetime = EXPIRY,
) -> OptionQuote:
    return OptionQuote(expiry=expiry, strike=strike, opt_type=opt, bid=bid, ask=ask, ltp=ltp, oi=oi)


def synthetic_chain(
    params: SVIParams = TRUE,
    *,
    ts: datetime = TS,
    expiry: datetime = EXPIRY,
    forward: float | None = None,
    half_spread_pct: float = 0.01,
    n: int = 21,
) -> ChainSnapshot:
    """A chain priced off a known surface, quoted two-sided around the true value."""
    t = (expiry - ts).total_seconds() / (365.0 * 24 * 3600)
    f = forward if forward is not None else SPOT * math.exp(R * t)
    quotes = []
    for i in range(n):
        k = -0.25 + 0.50 * i / (n - 1)
        strike = round(SPOT * math.exp(k) / 50.0) * 50.0
        kk = math.log(strike / f)
        vol = math.sqrt(params.total_variance(kk) / t)
        for opt, kind in ((OptionType.CE, OptType.CALL), (OptionType.PE, OptType.PUT)):
            mid = black76_price(f, strike, vol, t, R, kind)
            if mid <= 0.05:
                continue
            h = mid * half_spread_pct
            quotes.append(quote(strike, opt, bid=mid - h, ask=mid + h, ltp=mid, expiry=expiry))
    return ChainSnapshot(ts=ts, underlying_id=1, spot=SPOT, quotes=quotes)


# --------------------------------------------------------------------------- #
# mid_not_ltp
# --------------------------------------------------------------------------- #
def test_mid_not_ltp_rejects_a_quote_with_only_a_last_price():
    """A surface built from LTPs is built from prices that were never
    simultaneously available. It fits beautifully - stale prices are smooth."""
    stats = FilterStats()
    q = quote(24000.0, OptionType.CE, bid=None, ask=None, ltp=310.0)

    assert iv_from_quote(q, forward=24100.0, t=0.08, r=R, stats=stats) is None
    assert stats.rejected == {RejectReason.NO_TWO_SIDED_MARKET: 1}
    assert stats.kept == 0


def test_mid_not_ltp_rejects_a_zero_bid_even_with_an_ask():
    """`bid > 0` from the plan's filter list. A zero bid means nobody is buying
    it at any price, and the ask alone is not a market."""
    stats = FilterStats()
    q = quote(26000.0, OptionType.CE, bid=0.0, ask=1.20, ltp=0.60)

    assert iv_from_quote(q, forward=24100.0, t=0.08, r=R, stats=stats) is None
    assert RejectReason.NO_TWO_SIDED_MARKET in stats.rejected


def test_mid_not_ltp_uses_the_midpoint_when_both_sides_exist():
    """And the IV it returns is the one implied by the mid, not by either side."""
    t = year_fraction()
    f = SPOT * math.exp(R * t)
    true_vol = 0.16
    mid = black76_price(f, 24000.0, true_vol, t, R, OptType.CALL)
    stats = FilterStats()

    obs = iv_from_quote(
        quote(24000.0, OptionType.CE, bid=mid * 0.99, ask=mid * 1.01, ltp=mid * 1.6),
        forward=f,
        t=t,
        r=R,
        stats=stats,
    )
    assert obs is not None
    assert obs.iv == pytest.approx(true_vol, rel=1e-4)
    assert stats.kept == 1


def test_mid_not_ltp_a_wildly_wrong_ltp_changes_nothing():
    """The structural guarantee: there is no code path that reads ltp."""
    t = year_fraction()
    f = SPOT * math.exp(R * t)
    mid = black76_price(f, 24000.0, 0.16, t, R, OptType.CALL)

    a = iv_from_quote(
        quote(24000.0, OptionType.CE, bid=mid * 0.99, ask=mid * 1.01, ltp=1.0),
        forward=f,
        t=t,
        r=R,
        stats=FilterStats(),
    )
    b = iv_from_quote(
        quote(24000.0, OptionType.CE, bid=mid * 0.99, ask=mid * 1.01, ltp=99999.0),
        forward=f,
        t=t,
        r=R,
        stats=FilterStats(),
    )
    assert a is not None and b is not None
    assert a.iv == b.iv


# --------------------------------------------------------------------------- #
# no_butterfly_arb
# --------------------------------------------------------------------------- #
def test_no_butterfly_arb_on_a_clean_fit():
    params = fit_svi(observations(), t=year_fraction())
    assert butterfly_violations(params) == []


def test_no_butterfly_arb_durrleman_agrees_with_the_implied_density():
    """The check that `durrleman_g` is the right formula rather than a plausible
    one. Where g < 0 the second derivative of the call price in strike - the
    discounted risk-neutral density - must actually be negative."""
    bad = SVIParams(a=0.001, b=0.9, rho=-0.9, m=0.0, sigma=0.02)
    violations = butterfly_violations(bad)
    assert violations, "the fixture is supposed to be arbitrageable"
    worst_k, _ = min(violations, key=lambda kg: kg[1])

    t, f = 0.05, 24000.0
    strike = f * math.exp(worst_k)

    def call(k_strike: float) -> float:
        w = bad.total_variance(math.log(k_strike / f))
        return black76_price(f, k_strike, math.sqrt(w / t), t, 0.0, OptType.CALL)

    dk = 30.0
    density = (call(strike + dk) - 2 * call(strike) + call(strike - dk)) / (dk * dk)
    assert density < 0.0, (
        f"g={durrleman_g(bad, worst_k):.4f} says arbitrage but the density is "
        f"{density:.3e}; the two must agree or g is measuring something else"
    )


def test_no_butterfly_arb_a_clean_surface_has_a_positive_density_everywhere():
    """The other direction. A formula that returned g >= 0 always would pass the
    test above and be useless."""
    t, f = year_fraction(), SPOT
    for k in (-0.4, -0.15, 0.0, 0.2, 0.45):
        assert durrleman_g(TRUE, k) > 0.0
        strike = f * math.exp(k)

        def call(x: float) -> float:
            w = TRUE.total_variance(math.log(x / f))
            return black76_price(f, x, math.sqrt(w / t), t, 0.0, OptType.CALL)

        dk = strike * 0.002
        assert (call(strike + dk) - 2 * call(strike) + call(strike - dk)) / (dk * dk) > 0.0


def test_no_butterfly_arb_the_surface_refuses_to_serve_a_violating_fit():
    """The enforcement that matters. A penalty inside the objective steers the
    optimiser; whatever survives it is checked again and refused. Checking only
    inside the objective looks identical until someone trades on it."""
    bad = SVIParams(a=0.001, b=0.9, rho=-0.9, m=0.0, sigma=0.02)
    surface = Surface(ts=TS, underlying_id=1, spot=SPOT)
    surface.slices[EXPIRY] = ExpirySlice(
        expiry=EXPIRY,
        t=year_fraction(),
        forward=synthetic_forward([], spot=SPOT, r=R, t=year_fraction()),
        params=bad,
        n_quotes=20,
        rmse=0.001,
        stats=FilterStats(kept=20),
    )
    report = surface.arbitrage_report()
    assert not report.clean
    assert "density is negative" in report.render()

    with pytest.raises(ArbitrageError, match="admits arbitrage"):
        surface.require_arbitrage_free()


def test_no_butterfly_arb_a_good_rmse_does_not_make_a_surface_safe():
    """The whole reason for the second check: fit quality and arbitrage-freedom
    are independent, and the reassuring number is the one people look at."""
    bad = SVIParams(a=0.001, b=0.9, rho=-0.9, m=0.0, sigma=0.02)
    obs = observations(bad, n=25)
    assert fit_rmse(bad, obs, t=year_fraction()) == pytest.approx(0.0, abs=1e-9)
    assert butterfly_violations(bad), "perfect RMSE, and arbitrageable"


def test_no_butterfly_arb_is_scanned_on_a_grid_not_at_one_point():
    """g is not convex in k. A solver that finds one stationary point can call a
    surface clean while a second trough is deeply negative."""
    bad = SVIParams(a=0.001, b=0.9, rho=-0.9, m=0.0, sigma=0.02)
    ks = [k for k, _ in butterfly_violations(bad)]
    assert len(ks) > 1
    assert max(ks) - min(ks) > 0.01


# --------------------------------------------------------------------------- #
# no_calendar_arb
# --------------------------------------------------------------------------- #
def test_no_calendar_arb_when_variance_grows_with_maturity():
    near = SVIParams(a=0.004, b=0.030, rho=-0.40, m=0.0, sigma=0.10)
    far = SVIParams(a=0.012, b=0.070, rho=-0.40, m=0.0, sigma=0.10)
    assert calendar_violations(near, far) == []


def test_no_calendar_arb_catches_a_far_expiry_cheaper_than_a_near_one():
    """Total variance falling with maturity means a calendar spread has negative
    value - you are paid to hold optionality for longer."""
    near = SVIParams(a=0.020, b=0.060, rho=-0.30, m=0.0, sigma=0.10)
    far = SVIParams(a=0.005, b=0.030, rho=-0.30, m=0.0, sigma=0.10)

    violations = calendar_violations(near, far)
    assert violations
    _k, w1, w2 = violations[0]
    assert w2 < w1


def test_no_calendar_arb_is_checked_between_adjacent_expiries_in_time_order():
    """Ordered by T, not by dict insertion. Checking an arbitrary pair order
    reports violations that are only an artefact of the iteration."""
    surface = Surface(ts=TS, underlying_id=1, spot=SPOT)
    far_expiry = TS + timedelta(days=60)
    near_expiry = TS + timedelta(days=7)

    def sl(expiry: datetime, params: SVIParams, t: float) -> ExpirySlice:
        return ExpirySlice(
            expiry=expiry,
            t=t,
            forward=synthetic_forward([], spot=SPOT, r=R, t=t),
            params=params,
            n_quotes=20,
            rmse=0.001,
            stats=FilterStats(kept=20),
        )

    # Inserted far-first, so a naive pairwise walk would compare them backwards.
    surface.slices[far_expiry] = sl(far_expiry, SVIParams(0.012, 0.070, -0.4, 0.0, 0.10), 60 / 365)
    surface.slices[near_expiry] = sl(near_expiry, SVIParams(0.004, 0.030, -0.4, 0.0, 0.10), 7 / 365)
    assert surface.arbitrage_report().clean


def test_no_calendar_arb_violation_is_reported_with_the_expiry_pair():
    surface = Surface(ts=TS, underlying_id=1, spot=SPOT)
    e1, e2 = TS + timedelta(days=7), TS + timedelta(days=60)

    def sl(expiry: datetime, params: SVIParams, t: float) -> ExpirySlice:
        return ExpirySlice(
            expiry=expiry,
            t=t,
            forward=synthetic_forward([], spot=SPOT, r=R, t=t),
            params=params,
            n_quotes=20,
            rmse=0.001,
            stats=FilterStats(kept=20),
        )

    surface.slices[e1] = sl(e1, SVIParams(0.020, 0.060, -0.3, 0.0, 0.10), 7 / 365)
    surface.slices[e2] = sl(e2, SVIParams(0.005, 0.030, -0.3, 0.0, 0.10), 60 / 365)

    report = surface.arbitrage_report()
    assert (e1, e2) in report.calendar
    assert "negative value" in report.render()


# --------------------------------------------------------------------------- #
# rejects_stale
# --------------------------------------------------------------------------- #
def test_rejects_stale_surface():
    """Refusing rather than warning. A warning gets logged and the position gets
    sized against a volatility that stopped existing four minutes ago."""
    surface = fit_surface(synthetic_chain(), now=TS)
    late = TS + MAX_SURFACE_AGE + timedelta(seconds=1)

    assert surface.is_stale(late)
    with pytest.raises(StaleSurfaceError, match="no longer exists"):
        surface.slice_at(EXPIRY, now=late)


def test_rejects_stale_but_serves_inside_the_bound():
    surface = fit_surface(synthetic_chain(), now=TS)
    assert surface.slice_at(EXPIRY, now=TS + MAX_SURFACE_AGE - timedelta(seconds=1))


def test_rejects_stale_features_too():
    """The feature path must not be a way around the staleness check - it is the
    path a model actually uses."""
    surface = fit_surface(synthetic_chain(), now=TS)
    with pytest.raises(StaleSurfaceError):
        surface.features(EXPIRY, now=TS + timedelta(minutes=10))


def test_rejects_stale_bound_is_configurable_per_surface():
    surface = fit_surface(synthetic_chain(), now=TS, max_age=timedelta(seconds=5))
    with pytest.raises(StaleSurfaceError):
        surface.slice_at(EXPIRY, now=TS + timedelta(seconds=6))


# --------------------------------------------------------------------------- #
# the forward
# --------------------------------------------------------------------------- #
def test_the_forward_comes_from_put_call_parity_not_from_the_spot():
    """`S·e^(rT)` is wrong for an index by the dividend yield you do not know,
    and the error appears as a stable, entirely manufactured skew."""
    t = year_fraction()
    true_forward = SPOT * math.exp(R * t) - 180.0  # dividends the model cannot see
    chain = synthetic_chain(forward=true_forward)

    est = synthetic_forward(chain.quotes, spot=SPOT, r=R, t=t)
    assert est.method == "put-call parity"
    assert est.forward == pytest.approx(true_forward, rel=1e-6)
    assert abs(SPOT * math.exp(R * t) - true_forward) > 100.0


def test_the_forward_says_so_when_it_had_to_fall_back():
    """A caller that cannot tell which forward it got will not notice the
    surface acquiring a skew that is not there."""
    t = year_fraction()
    calls_only = [q for q in synthetic_chain().quotes if q.opt_type is OptionType.CE]

    est = synthetic_forward(calls_only, spot=SPOT, r=R, t=t)
    assert "NO parity strike" in est.method
    assert est.forward == pytest.approx(SPOT * math.exp(R * t))


def test_the_forward_uses_the_strike_nearest_the_money():
    """Parity holds at every strike; only the near-the-money ones have two
    liquid sides."""
    est = synthetic_forward(synthetic_chain().quotes, spot=SPOT, r=R, t=year_fraction())
    assert abs(est.strike_used - SPOT) < 600.0


# --------------------------------------------------------------------------- #
# quote filters
# --------------------------------------------------------------------------- #
def test_a_wide_spread_is_rejected_and_counted():
    stats = FilterStats()
    t = year_fraction()
    f = SPOT * math.exp(R * t)
    mid = black76_price(f, 24000.0, 0.16, t, R, OptType.CALL)

    assert (
        iv_from_quote(
            quote(24000.0, OptionType.CE, bid=mid * 0.5, ask=mid * 1.5),
            forward=f,
            t=t,
            r=R,
            stats=stats,
        )
        is None
    )
    assert stats.rejected == {RejectReason.SPREAD_TOO_WIDE: 1}


def test_low_open_interest_is_rejected():
    stats = FilterStats()
    t = year_fraction()
    f = SPOT * math.exp(R * t)
    mid = black76_price(f, 24000.0, 0.16, t, R, OptType.CALL)

    assert (
        iv_from_quote(
            quote(24000.0, OptionType.CE, bid=mid * 0.99, ask=mid * 1.01, oi=3),
            forward=f,
            t=t,
            r=R,
            stats=stats,
        )
        is None
    )
    assert stats.rejected == {RejectReason.OI_TOO_LOW: 1}


def test_a_five_delta_wing_is_rejected():
    """A 2-delta option worth 0.35 with a 0.30 spread carries an implied vol that
    moves twenty points when the quote moves one tick."""
    stats = FilterStats()
    t = year_fraction()
    f = SPOT * math.exp(R * t)
    far = 27000.0  # delta ~0.006, and still worth ~2.0 so the quote is real
    mid = black76_price(f, far, 0.16, t, R, OptType.CALL)

    assert (
        iv_from_quote(
            quote(far, OptionType.CE, bid=mid * 0.99, ask=mid * 1.01),
            forward=f,
            t=t,
            r=R,
            stats=stats,
        )
        is None
    )
    assert RejectReason.DELTA_OUT_OF_BAND in stats.rejected


def test_the_reject_rate_is_reported():
    """A chain that suddenly rejects 60% of its quotes is a feed problem, and
    the surface fitted from the survivors looks entirely normal."""
    stats = FilterStats(kept=30)
    stats.reject(RejectReason.SPREAD_TOO_WIDE)
    stats.reject(RejectReason.SPREAD_TOO_WIDE)
    stats.reject(RejectReason.OI_TOO_LOW)

    assert stats.seen == 33
    assert stats.reject_rate == pytest.approx(3 / 33)
    assert "spread_too_wide" in stats.render()


def test_an_expired_quote_is_rejected_rather_than_priced():
    stats = FilterStats()
    assert (
        iv_from_quote(
            quote(24000.0, OptionType.CE, bid=1.0, ask=1.1),
            forward=24000.0,
            t=0.0,
            r=R,
            stats=stats,
        )
        is None
    )
    assert stats.rejected == {RejectReason.EXPIRED: 1}


# --------------------------------------------------------------------------- #
# the fit itself
# --------------------------------------------------------------------------- #
def test_the_fit_recovers_known_parameters():
    """Pinned against a closed form: data generated from a known SVI must fit
    back to that SVI. A fitter that cannot do this on noiseless data is not
    measuring the smile on noisy data either."""
    got = fit_svi(observations(), t=year_fraction())
    for name in ("a", "b", "rho", "m", "sigma"):
        assert getattr(got, name) == pytest.approx(getattr(TRUE, name), abs=1e-6), name


def test_the_fit_is_deterministic():
    """Two fits of the same chain must be identical, or a recommendation cannot
    be reproduced from its inputs."""
    obs = observations()
    assert fit_svi(obs, t=year_fraction()) == fit_svi(obs, t=year_fraction())


def test_a_flat_smile_fits_flat():
    """The null test. Constant vol across strikes must produce no skew and no
    curvature - a fitter that manufactures a smile here manufactures one
    everywhere, and the smile is what sizing reads."""
    t = year_fraction()
    flat = 0.15
    w = flat * flat * t
    obs = [
        Observation(
            strike=SPOT * math.exp(k),
            k=k,
            iv=flat,
            total_variance=w,
            kind=OptType.CALL,
            delta=0.5,
        )
        for k in [-0.3 + 0.6 * i / 24 for i in range(25)]
    ]
    params = fit_svi(obs, t=t)

    assert params.vol(0.0, t) == pytest.approx(flat, abs=1e-4)
    assert abs(params.d1(0.0)) < 1e-4, f"invented a skew of {params.d1(0.0):.6f} out of flat data"
    assert abs(params.d2(0.0)) < 1e-3, f"invented curvature of {params.d2(0.0):.6f}"


def test_the_fit_refuses_too_few_quotes():
    """Five parameters fitted to fewer points than that is a curve drawn through
    noise, and it looks like a smile."""
    with pytest.raises(SurfaceError, match="drawn through noise"):
        fit_svi(observations(n=MIN_QUOTES_PER_EXPIRY - 1), t=year_fraction())


def test_rmse_is_reported_in_vol_points_not_variance():
    """Total-variance RMSE changes scale with maturity, so the same quality of
    fit reports a different number for a weekly and a monthly."""
    obs = observations()
    t = year_fraction()
    assert fit_rmse(TRUE, obs, t=t) == pytest.approx(0.0, abs=1e-9)

    off = SVIParams(TRUE.a + 0.001, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
    rmse = fit_rmse(off, obs, t=t)
    assert 0.0005 < rmse < 0.05, f"{rmse} does not read as vol points"


def test_a_surface_that_would_give_negative_variance_refuses_to_price():
    bad = SVIParams(a=-1.0, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ArbitrageError, match="must not be used to price"):
        bad.vol(0.0, 0.1)


# --------------------------------------------------------------------------- #
# end to end, and the features
# --------------------------------------------------------------------------- #
def test_a_whole_chain_fits_and_is_arbitrage_free():
    surface = fit_surface(synthetic_chain(), now=TS)
    assert EXPIRY in surface.slices
    surface.require_arbitrage_free()

    sl = surface.slices[EXPIRY]
    assert sl.rmse < 0.005, f"rmse {sl.rmse} on noiseless quotes"
    assert sl.stats.reject_rate < 0.5


def test_the_features_are_level_slope_and_curvature():
    surface = fit_surface(synthetic_chain(), now=TS)
    features = surface.features(EXPIRY, now=TS)

    assert set(features) >= {"iv_level", "iv_slope", "iv_curvature"}
    assert 0.05 < features["iv_level"] < 0.60
    assert features["iv_slope"] < 0.0, "the fixture has negative skew"
    assert features["iv_curvature"] > 0.0


def test_parameters_are_stored_and_the_features_are_derived():
    """Storing the derived features instead would make a surface impossible to
    re-price from, and rebuilding a past recommendation is the point."""
    surface = fit_surface(synthetic_chain(), now=TS)
    sl = surface.slices[EXPIRY]

    rebuilt = SVIParams(sl.params.a, sl.params.b, sl.params.rho, sl.params.m, sl.params.sigma)
    assert rebuilt.total_variance(0.0) == sl.atm_total_variance


def test_one_bad_expiry_does_not_take_the_surface_down():
    """A weekly with four quotes on a holiday-shortened week is normal, and
    refusing the whole surface for it would take the monthlies with it."""
    good = synthetic_chain()
    thin_expiry = TS + timedelta(days=3)
    thin = [
        quote(24000.0, OptionType.CE, bid=100.0, ask=101.0, expiry=thin_expiry),
        quote(24050.0, OptionType.PE, bid=100.0, ask=101.0, expiry=thin_expiry),
    ]
    chain = ChainSnapshot(ts=TS, underlying_id=1, spot=SPOT, quotes=[*good.quotes, *thin])
    surface = fit_surface(chain, now=TS)

    assert EXPIRY in surface.slices
    assert thin_expiry not in surface.slices


def test_an_expired_leg_in_the_chain_is_skipped():
    past = TS - timedelta(days=1)
    chain = ChainSnapshot(
        ts=TS,
        underlying_id=1,
        spot=SPOT,
        quotes=[
            *synthetic_chain().quotes,
            quote(24000.0, OptionType.CE, bid=10.0, ask=11.0, expiry=past),
        ],
    )
    assert past not in fit_surface(chain, now=TS).slices


def test_the_render_shows_the_arbitrage_verdict():
    text = fit_surface(synthetic_chain(), now=TS).render()
    assert "no-arbitrage" in text
    assert "rmse" in text


def test_no_butterfly_arb_needs_the_second_derivative_term():
    """The term that a mutation test found unprotected.

    Durrleman's g is `(1 - k·w'/2w)² - (w'²/4)(1/w + 1/4) + w''/2`, and dropping
    the last term leaves a formula that still says "arbitrage" on obviously
    broken surfaces and "clean" on obviously fine ones. It goes wrong in the
    middle, which is where every real fitted surface lives.

    This smile is genuinely arbitrage-free — the minimum of the numerically
    differentiated call price over k in [-1.2, 1.2] is +1.4e-07, positive
    everywhere. Without the w''/2 term the same smile is reported as violating
    at 22 grid points, and the surface would be refused for an arbitrage that
    is not there. A refusal is safe; a refusal that happens on ordinary smiles
    gets the check switched off.
    """
    steep = SVIParams(a=0.005, b=0.4, rho=0.0, m=0.0, sigma=0.08)
    assert steep.parameters_admissible
    assert butterfly_violations(steep) == [], (
        "a smile with a positive density everywhere was reported as arbitrageable; "
        "the most likely cause is a missing or wrong w'' term in durrleman_g"
    )

    t, f = 0.25, 24000.0

    def call(strike: float) -> float:
        w = steep.total_variance(math.log(strike / f))
        return black76_price(f, strike, math.sqrt(w / t), t, 0.0, OptType.CALL)

    for k in (-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0):
        strike = f * math.exp(k)
        dk = strike * 0.001
        density = (call(strike + dk) - 2 * call(strike) + call(strike - dk)) / (dk * dk)
        assert density > 0.0, f"the fixture is not arbitrage-free at k={k}"
