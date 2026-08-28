"""M71 - Greeks off the fitted surface.

    pytest tests/unit/test_greeks.py -k "matches_reference_within_tol or portfolio_aggregation"

The failure this module exists to prevent is not a crash. It is a book hedged on
Black-Scholes delta - which holds volatility constant - against a surface where
volatility moves every time the forward does. The resulting hedge is short in
the same direction every day, by a few percent, and it presents as slippage.

So every derivative here is pinned twice: against a finite difference of the
Black-76 formula, and against a **full reprice** along the surface, which shares
no algebra with the analytic path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_options.greeks import (
    GreeksError,
    Leg,
    PortfolioGreeks,
    aggregate,
    greeks_from_surface,
    reference_delta_gamma,
)
from libs.ms_options.pricing import OptType, black76_greeks, black76_price
from libs.ms_options.surface import (
    ExpirySlice,
    FilterStats,
    ForwardEstimate,
    StaleSurfaceError,
    Surface,
    SVIParams,
)

TS = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
NEAR = TS + timedelta(days=7)
MID = TS + timedelta(days=28)
FAR = TS + timedelta(days=56)
SPOT = 24000.0
R = 0.065

# A Nifty-shaped smile: ~13% at the money with real negative skew.
SMILE = SVIParams(a=0.0006, b=0.0057, rho=-0.42, m=0.015, sigma=0.115)


def make_slice(
    expiry: datetime, *, scale: float = 1.0, forward: float | None = None
) -> ExpirySlice:
    t = (expiry - TS).total_seconds() / (365.0 * 24 * 3600)
    f = forward if forward is not None else SPOT * math.exp(R * t)
    params = SVIParams(
        a=SMILE.a * scale, b=SMILE.b * scale, rho=SMILE.rho, m=SMILE.m, sigma=SMILE.sigma
    )
    return ExpirySlice(
        expiry=expiry,
        t=t,
        forward=ForwardEstimate(
            forward=f, strike_used=SPOT, n_candidates=9, method="put-call parity"
        ),
        params=params,
        n_quotes=25,
        rmse=0.0004,
        stats=FilterStats(kept=25),
    )


def make_surface(*, ts: datetime = TS) -> Surface:
    s = Surface(ts=ts, underlying_id=1, spot=SPOT)
    for expiry, scale in ((NEAR, 0.45), (MID, 1.0), (FAR, 1.7)):
        s.slices[expiry] = make_slice(expiry, scale=scale)
    return s


# --------------------------------------------------------------------------- #
# matches_reference_within_tol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strike", [21500.0, 23000.0, 24000.0, 24500.0, 26000.0])
@pytest.mark.parametrize("kind", [OptType.CALL, OptType.PUT])
def test_delta_matches_reference_within_tol(strike: float, kind: OptType):
    """The analytic smile delta against a full reprice along the surface.

    The reference shares no algebra with the analytic path: it bumps the
    forward, recomputes log-moneyness, asks the SVI slice for the vol at the new
    k, and prices again. Two routes agreeing is evidence; one route agreeing
    with itself is not.
    """
    sl = make_slice(MID)
    sg = greeks_from_surface(sl, strike=strike, kind=kind, r=R)
    ref_delta, _ = reference_delta_gamma(sl, strike=strike, kind=kind, r=R)

    assert sg.delta == pytest.approx(
        ref_delta, abs=1e-6
    ), f"analytic {sg.delta:.8f} vs reprice {ref_delta:.8f} at K={strike} {kind}"


@pytest.mark.parametrize("strike", [22000.0, 24000.0, 25500.0])
def test_gamma_matches_reference_within_tol(strike: float):
    """Gamma carries four chain-rule terms - Gamma, vanna, volga and vega*sigma_FF - and
    a sign error in any of them produces a number that is smooth, positive and
    wrong."""
    sl = make_slice(MID)
    sg = greeks_from_surface(sl, strike=strike, kind=OptType.CALL, r=R)
    _, ref_gamma = reference_delta_gamma(sl, strike=strike, kind=OptType.CALL, r=R)

    assert sg.gamma == pytest.approx(
        ref_gamma, rel=2e-4
    ), f"analytic {sg.gamma:.10f} vs reprice {ref_gamma:.10f} at K={strike}"


def test_vanna_matches_reference_within_tol():
    """vanna = d(vega)/dF, checked against a finite difference of vega itself."""
    sl = make_slice(MID)
    strike = 24500.0
    sg = greeks_from_surface(sl, strike=strike, kind=OptType.CALL, r=R)

    f, h = sl.forward.forward, sl.forward.forward * 1e-5
    up = black76_greeks(f + h, strike, sg.vol, sl.t, R, OptType.CALL).vega
    down = black76_greeks(f - h, strike, sg.vol, sl.t, R, OptType.CALL).vega

    assert sg.vanna == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_volga_matches_reference_within_tol():
    """volga = d(vega)/dsigma."""
    sl = make_slice(MID)
    strike = 24500.0
    sg = greeks_from_surface(sl, strike=strike, kind=OptType.CALL, r=R)

    h = 1e-5
    up = black76_greeks(sl.forward.forward, strike, sg.vol + h, sl.t, R, OptType.CALL).vega
    down = black76_greeks(sl.forward.forward, strike, sg.vol - h, sl.t, R, OptType.CALL).vega

    assert sg.volga == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_dsigma_dforward_matches_reference_within_tol():
    """The term the whole module turns on: how the smile moves the vol at a
    fixed strike when the forward moves."""
    sl = make_slice(MID)
    strike = 24500.0
    sg = greeks_from_surface(sl, strike=strike, kind=OptType.CALL, r=R)

    f, h = sl.forward.forward, sl.forward.forward * 1e-5

    def vol_at(forward: float) -> float:
        return sl.params.vol(math.log(strike / forward), sl.t)

    assert sg.dsigma_dforward == pytest.approx((vol_at(f + h) - vol_at(f - h)) / (2 * h), rel=1e-5)


def test_price_matches_reference_within_tol():
    sl = make_slice(MID)
    sg = greeks_from_surface(sl, strike=24500.0, kind=OptType.CALL, r=R)
    assert sg.price == pytest.approx(
        black76_price(sl.forward.forward, 24500.0, sg.vol, sl.t, R, OptType.CALL)
    )


def test_the_reference_and_the_analytic_path_share_no_algebra():
    """If the reference used the analytic Greeks it would agree with anything.
    A deliberately wrong slice must make the two disagree."""
    sl = make_slice(MID)
    strike = 24500.0
    ref_delta, _ = reference_delta_gamma(sl, strike=strike, kind=OptType.CALL, r=R)

    flat = ExpirySlice(
        expiry=sl.expiry,
        t=sl.t,
        forward=sl.forward,
        params=SVIParams(a=sl.params.total_variance(0.0), b=0.0, rho=0.0, m=0.0, sigma=0.1),
        n_quotes=sl.n_quotes,
        rmse=sl.rmse,
        stats=sl.stats,
    )
    flat_ref, _ = reference_delta_gamma(flat, strike=strike, kind=OptType.CALL, r=R)
    assert abs(flat_ref - ref_delta) > 1e-4


# --------------------------------------------------------------------------- #
# the smile term itself
# --------------------------------------------------------------------------- #
def test_the_smile_delta_differs_from_black76_by_a_material_amount():
    """The reason this module exists. Hedging on the Black-76 number leaves the
    book short in the same direction every day, and it presents as slippage."""
    sl = make_slice(MID)
    sg = greeks_from_surface(sl, strike=24500.0, kind=OptType.CALL, r=R)

    assert sg.delta != sg.delta_black76
    relative = abs(sg.smile_delta_adjustment / sg.delta_black76)
    assert relative > 0.02, f"the smile term is only {relative:.2%}; check the fixture has skew"


def test_a_flat_surface_makes_the_smile_term_vanish():
    """The null test. No skew, no curvature, so the correction must be exactly
    zero - a module that returns a correction here is inventing one everywhere."""
    t = (MID - TS).total_seconds() / (365.0 * 24 * 3600)
    flat = ExpirySlice(
        expiry=MID,
        t=t,
        forward=ForwardEstimate(SPOT, SPOT, 9, "put-call parity"),
        params=SVIParams(a=0.13 * 0.13 * t, b=0.0, rho=0.0, m=0.0, sigma=0.1),
        n_quotes=25,
        rmse=0.0,
        stats=FilterStats(kept=25),
    )
    sg = greeks_from_surface(flat, strike=24500.0, kind=OptType.CALL, r=R)

    assert sg.dsigma_dforward == pytest.approx(0.0, abs=1e-15)
    assert sg.delta == pytest.approx(sg.delta_black76, abs=1e-12)
    assert sg.gamma == pytest.approx(sg.gamma_black76, abs=1e-12)


def test_the_sign_of_the_smile_term_follows_the_skew():
    """On a negatively skewed surface a rising forward slides a fixed strike
    down the smile to a lower vol, so a call's true delta exceeds Black-76's."""
    sl = make_slice(MID)
    assert sl.params.d1(0.0) < 0.0, "the fixture is supposed to be negatively skewed"

    sg = greeks_from_surface(sl, strike=24500.0, kind=OptType.CALL, r=R)
    assert sg.dsigma_dforward > 0.0
    assert sg.delta > sg.delta_black76


def test_black76_greeks_are_kept_beside_the_corrected_ones():
    """A module that only returns the right answer cannot show anyone why the
    old answer was wrong, and the difference is what explains a hedging P&L."""
    sg = greeks_from_surface(make_slice(MID), strike=24500.0, kind=OptType.CALL, r=R)
    assert sg.delta_black76 == pytest.approx(
        black76_greeks(sg.forward, 24500.0, sg.vol, sg.t, R, OptType.CALL).delta
    )


def test_the_vol_comes_from_the_surface_not_from_an_argument():
    """There is no vendor-IV parameter. A vendor's number is fitted from quotes
    we already filtered, by a method we cannot see, and mixing it with our own
    surface hedges the book against two different volatilities."""
    import inspect

    params = inspect.signature(greeks_from_surface).parameters
    assert "vol" not in params and "iv" not in params and "sigma" not in params


def test_greeks_of_an_expired_slice_are_refused():
    expired = ExpirySlice(
        expiry=TS,
        t=0.0,
        forward=ForwardEstimate(SPOT, SPOT, 9, "put-call parity"),
        params=SMILE,
        n_quotes=25,
        rmse=0.0,
        stats=FilterStats(kept=25),
    )
    with pytest.raises(GreeksError, match="settlement value"):
        greeks_from_surface(expired, strike=24000.0, kind=OptType.CALL, r=R)


# --------------------------------------------------------------------------- #
# portfolio_aggregation
# --------------------------------------------------------------------------- #
def test_portfolio_aggregation_sums_delta_across_legs():
    surface = make_surface()
    legs = [
        Leg(strike=24000.0, kind=OptType.CALL, expiry=MID, lots=2, lot_size=75),
        Leg(strike=25000.0, kind=OptType.CALL, expiry=MID, lots=2, lot_size=75, is_long=False),
    ]
    book = aggregate(legs, surface, now=TS, r=R)

    long_leg = greeks_from_surface(surface.slices[MID], strike=24000.0, kind=OptType.CALL, r=R)
    short_leg = greeks_from_surface(surface.slices[MID], strike=25000.0, kind=OptType.CALL, r=R)
    assert book.delta == pytest.approx((long_leg.delta - short_leg.delta) * 150)


def test_portfolio_aggregation_respects_direction_and_lot_size():
    surface = make_surface()
    one = aggregate([Leg(24000.0, OptType.CALL, MID, lots=1, lot_size=75)], surface, now=TS, r=R)
    short_three = aggregate(
        [Leg(24000.0, OptType.CALL, MID, lots=3, lot_size=75, is_long=False)],
        surface,
        now=TS,
        r=R,
    )
    assert short_three.delta == pytest.approx(-3 * one.delta)
    assert short_three.gamma == pytest.approx(-3 * one.gamma)


def test_portfolio_aggregation_keeps_vega_apart_by_expiry():
    """The number that must never be one number. A seven-day vega and a
    sixty-day vega are different risks sharing a unit."""
    surface = make_surface()
    legs = [
        Leg(24000.0, OptType.CALL, NEAR, lots=4, lot_size=75, is_long=False),
        Leg(24000.0, OptType.CALL, FAR, lots=1, lot_size=75),
    ]
    book = aggregate(legs, surface, now=TS, r=R)

    assert set(book.vega_by_expiry) == {NEAR, FAR}
    assert book.vega_by_expiry[NEAR] < 0.0
    assert book.vega_by_expiry[FAR] > 0.0


def test_portfolio_aggregation_a_near_zero_gross_vega_is_a_calendar_not_a_flat_book():
    """The failure the bucketing prevents: a book that reports vega-neutral
    while short front vol and long back vol. It is a real position with real
    P&L and, summed, no number describing it."""
    surface = make_surface()
    front = greeks_from_surface(surface.slices[NEAR], strike=24000.0, kind=OptType.CALL, r=R)
    back = greeks_from_surface(surface.slices[FAR], strike=24000.0, kind=OptType.CALL, r=R)
    lots_front = 8
    lot_size = 75
    # Choose the back size so the gross very nearly cancels.
    lots_back = max(round(front.vega * lots_front / back.vega), 1)

    book = aggregate(
        [
            Leg(24000.0, OptType.CALL, NEAR, lots=lots_front, lot_size=lot_size, is_long=False),
            Leg(24000.0, OptType.CALL, FAR, lots=lots_back, lot_size=lot_size),
        ],
        surface,
        now=TS,
        r=R,
    )
    f, b = book.vega_net_front_back

    assert abs(book.vega_gross) < abs(f) * 0.5, "the fixture was meant to nearly cancel"
    assert f < 0.0 < b, "and it is a calendar position, not a flat book"
    assert "not one number" in book.render()


def test_portfolio_aggregation_reports_the_smile_term_for_the_whole_book():
    surface = make_surface()
    book = aggregate(
        [
            Leg(24500.0, OptType.CALL, MID, lots=5, lot_size=75),
            Leg(23500.0, OptType.PUT, MID, lots=5, lot_size=75),
        ],
        surface,
        now=TS,
        r=R,
    )
    assert book.delta != book.delta_black76
    assert book.smile_delta_adjustment == pytest.approx(book.delta - book.delta_black76)


def test_portfolio_aggregation_of_an_empty_book_is_zero_not_an_error():
    book = aggregate([], make_surface(), now=TS, r=R)
    assert (book.delta, book.gamma, book.vega_gross) == (0.0, 0.0, 0.0)
    assert book.vega_net_front_back == (0.0, 0.0)


def test_portfolio_aggregation_goes_through_the_staleness_check():
    """Risk aggregation is exactly where a staleness check gets bypassed for
    convenience, and the last place that should happen."""
    surface = make_surface()
    with pytest.raises(StaleSurfaceError):
        aggregate(
            [Leg(24000.0, OptType.CALL, MID, lots=1, lot_size=75)],
            surface,
            now=TS + timedelta(minutes=10),
            r=R,
        )


def test_portfolio_aggregation_says_when_deltas_span_different_forwards():
    """They move together and are not the same number, and a book summed across
    three expiries should say so rather than imply one forward."""
    surface = make_surface()
    book = aggregate(
        [
            Leg(24000.0, OptType.CALL, NEAR, lots=1, lot_size=75),
            Leg(24000.0, OptType.CALL, MID, lots=1, lot_size=75),
            Leg(24000.0, OptType.CALL, FAR, lots=1, lot_size=75),
        ],
        surface,
        now=TS,
        r=R,
    )
    assert len(book.forwards) == 3
    assert "different forwards" in book.render()


def test_a_signed_lot_count_is_refused():
    """Direction lives in is_long. A signed quantity survives until the first
    abs() in a display, and then the sign is gone from the record risk reads."""
    with pytest.raises(GreeksError, match="always positive"):
        Leg(24000.0, OptType.CALL, MID, lots=-2, lot_size=75)


def test_a_straddle_is_delta_light_and_gamma_heavy():
    """A sanity check on the aggregation as a whole, in terms someone would
    recognise: an at-the-money straddle has little delta and a lot of gamma."""
    surface = make_surface()
    f = surface.slices[MID].forward.forward
    atm = round(f / 50.0) * 50.0
    book = aggregate(
        [
            Leg(atm, OptType.CALL, MID, lots=1, lot_size=75),
            Leg(atm, OptType.PUT, MID, lots=1, lot_size=75),
        ],
        surface,
        now=TS,
        r=R,
    )
    single = greeks_from_surface(surface.slices[MID], strike=atm, kind=OptType.CALL, r=R)

    assert abs(book.delta) < abs(single.delta * 75)
    assert book.gamma > 0.0


def test_theta_is_reported_per_day_as_well_as_per_year():
    surface = make_surface()
    book = aggregate([Leg(24000.0, OptType.CALL, MID, lots=1, lot_size=75)], surface, now=TS, r=R)
    assert book.theta_per_day == pytest.approx(book.theta / 252.0)
    assert book.theta < 0.0, "a long option decays"


def test_the_render_shows_both_deltas():
    surface = make_surface()
    text = aggregate(
        [Leg(24500.0, OptType.CALL, MID, lots=3, lot_size=75)], surface, now=TS, r=R
    ).render()
    assert "black-76" in text
    assert "smile term" in text


def test_an_empty_portfolio_renders_without_crashing():
    assert "0 legs" in PortfolioGreeks().render()
