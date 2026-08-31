"""M14 verify - option chain snapshotter."""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.schemas import OptionType, Tick
from libs.ms_data.chain import (
    ChainBuilder,
    ChainSpec,
    OptionInstrument,
    OptionRegistry,
    atm_straddle,
    build_registry_from_specs,
    check_chain,
    liquid_quotes,
    minutes_to_expiry,
)
from libs.ms_data.synthetic_chain import (
    SmileParams,
    SyntheticChainFeed,
    strike_grid,
    weekly_expiries,
)
from tests.conftest import TEST_DAY, ist

UNDERLYING = 1
SPOT = 24_800.0
EXPIRIES = [TEST_DAY + timedelta(days=2), TEST_DAY + timedelta(days=9)]
STRIKES = strike_grid(SPOT, step=50.0, n_per_side=20)


@pytest.fixture
def registry():
    return build_registry_from_specs(UNDERLYING, EXPIRIES, STRIKES)


@pytest.fixture
def spec():
    return ChainSpec(underlying_id=UNDERLYING, n_expiries=2, strikes_per_side=15)


@pytest.fixture
def chain_feed(registry, calendar):
    return SyntheticChainFeed(
        registry, calendar, underlying_id=UNDERLYING, smile=SmileParams(), seed=3
    )


@pytest.fixture
def builder(spec, registry, calendar):
    return ChainBuilder(spec, registry, calendar)


def feed_a_snapshot(builder, chain_feed, now, spot=SPOT):
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=spot))
    for tick in chain_feed.quotes_for(spot, now):
        builder.on_tick(tick)
    return builder.snapshot(now)


# ------------------------------------------------------------------ registry
def test_registry_maps_ids_to_contracts(registry):
    assert len(registry) == len(EXPIRIES) * len(STRIKES) * 2
    some = registry.for_underlying(UNDERLYING)[0]
    assert registry.get(some.instrument_id) == some


def test_unknown_id_is_recorded_not_guessed(registry):
    """Mis-assigning a strike corrupts a whole expiry. Better to drop and count."""
    assert registry.get(999_999) is None
    assert 999_999 in registry.unknown_ids


def test_registry_lists_expiries_and_strikes(registry):
    assert registry.expiries(UNDERLYING) == sorted(EXPIRIES)
    assert registry.strikes(UNDERLYING, EXPIRIES[0]) == sorted(STRIKES)


# --------------------------------------------------------------- time to expiry
def test_minutes_to_expiry_same_day_counts_down_to_the_close(calendar):
    cal = calendar.synthetic_variant()
    assert minutes_to_expiry(cal, ist(9, 15), TEST_DAY) == pytest.approx(375.0)
    assert minutes_to_expiry(cal, ist(15, 0), TEST_DAY) == pytest.approx(30.0)
    assert minutes_to_expiry(cal, ist(15, 30), TEST_DAY) == pytest.approx(0.0)


def test_minutes_to_expiry_skips_the_weekend(calendar):
    """Calendar time overstates an option's value across a weekend."""
    cal = calendar.synthetic_variant()
    friday = TEST_DAY + timedelta(days=2)  # 2026-08-28
    monday = TEST_DAY + timedelta(days=5)  # 2026-08-31
    assert friday.weekday() == 4 and monday.weekday() == 0

    to_friday = minutes_to_expiry(cal, ist(9, 15), friday)
    to_monday = minutes_to_expiry(cal, ist(9, 15), monday)
    # One extra trading session, not three calendar days' worth.
    assert to_monday - to_friday == pytest.approx(375.0)


def test_expired_option_has_no_time_left(calendar):
    cal = calendar.synthetic_variant()
    assert minutes_to_expiry(cal, ist(9, 15), TEST_DAY - timedelta(days=1)) == 0.0


# ----------------------------------------------------------------- snapshots
def test_snapshot_requires_spot(builder, chain_feed):
    """A chain with no underlying is not a chain."""
    now = ist(9, 20)
    for tick in chain_feed.quotes_for(SPOT, now):
        builder.on_tick(tick)
    assert builder.snapshot(now) is None


def test_snapshot_carries_spot_captured_with_it(builder, chain_feed):
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    assert snap.spot == SPOT
    assert snap.ts == ist(9, 20)


def test_snapshot_covers_the_intended_window(builder, chain_feed, spec):
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    strikes = {q.strike for q in snap.quotes}
    # 15 either side of the money plus the ATM strike
    assert len(strikes) <= 2 * spec.strikes_per_side + 1
    assert snap.atm_strike() == pytest.approx(24_800.0)


def test_cadence_controls_emission(builder, chain_feed, spec):
    now = ist(9, 20)
    assert feed_a_snapshot(builder, chain_feed, now) is not None
    assert builder.maybe_snapshot(now + timedelta(seconds=1)) is None
    later = now + timedelta(seconds=spec.snapshot_seconds)
    builder.on_tick(Tick(ts=later, instrument_id=UNDERLYING, ltp=SPOT))
    assert builder.maybe_snapshot(later) is not None


def test_stale_quotes_are_excluded_not_carried_forward(builder, chain_feed, spec):
    """A price from forty seconds ago is not what the market is showing now."""
    now = ist(9, 20)
    first = feed_a_snapshot(builder, chain_feed, now)
    assert first is not None and first.completeness > 0.9

    much_later = now + timedelta(seconds=spec.max_quote_age_seconds + 30)
    builder.on_tick(Tick(ts=much_later, instrument_id=UNDERLYING, ltp=SPOT))
    stale_snap = builder.snapshot(much_later)
    assert stale_snap is not None
    assert stale_snap.quotes == []
    assert stale_snap.completeness == 0.0


def test_incomplete_snapshot_is_stored_and_flagged(builder, chain_feed, registry):
    """Discarding it leaves a hole indistinguishable from an outage."""
    now = ist(9, 20)
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=SPOT))
    ticks = list(chain_feed.quotes_for(SPOT, now))
    for tick in ticks[: len(ticks) // 3]:  # only a third arrived
        builder.on_tick(tick)

    snap = builder.snapshot(now)
    assert snap is not None
    assert snap.quotes, "partial data must still be stored"
    assert snap.completeness < 0.95


def test_ticks_for_other_underlyings_are_dropped(builder, calendar):
    other = OptionRegistry([OptionInstrument(777, 99, EXPIRIES[0], 24_800.0, OptionType.CE)])
    _ = other
    builder.on_tick(Tick(ts=ist(9, 20), instrument_id=777, ltp=100.0))
    assert builder.dropped_unknown == 1


# ------------------------------------------------------- implied volatility
def test_iv_inversion_recovers_the_generating_volatility(builder, chain_feed):
    """The snapshotter's IV must match what the generator used to price."""
    now = ist(9, 20)
    snap = feed_a_snapshot(builder, chain_feed, now)
    assert snap is not None

    priced = [q for q in snap.quotes if q.iv is not None]
    assert len(priced) > 20, "IV inversion failed on too many strikes"

    near_atm = [q for q in priced if abs(q.strike - SPOT) <= 200]
    assert near_atm
    for q in near_atm:
        expected = chain_feed.true_vol(q.strike, SPOT, now, q.expiry.date())
        # Bid-ask rounding means mid is not exactly fair value, so allow a band.
        assert q.iv == pytest.approx(expected, abs=0.02), f"strike {q.strike}"


def test_the_smile_has_a_put_skew(builder, chain_feed):
    """Equity index options carry persistent downside skew. A flat generator
    would make every skew feature test vacuously."""
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    by_strike = {
        q.strike: q.iv for q in snap.quotes if q.iv is not None and q.expiry.date() == EXPIRIES[0]
    }
    downside = [v for k, v in by_strike.items() if k < SPOT - 300]
    upside = [v for k, v in by_strike.items() if k > SPOT + 300]
    assert downside and upside
    assert sum(downside) / len(downside) > sum(upside) / len(upside)


def test_one_sided_book_yields_no_iv(spec, registry, calendar):
    b = ChainBuilder(spec, registry, calendar)
    now = ist(9, 20)
    contract = registry.for_underlying(UNDERLYING)[0]
    b.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=SPOT))
    b.on_tick(Tick(ts=now, instrument_id=contract.instrument_id, ltp=100.0))  # no bid/ask
    snap = b.snapshot(now)
    assert snap is not None
    assert all(q.iv is None for q in snap.quotes)


# --------------------------------------------------------------- quality gates
def test_a_clean_chain_passes(builder, chain_feed, calendar):
    now = ist(9, 20)
    snap = feed_a_snapshot(builder, chain_feed, now)
    assert snap is not None
    report = check_chain(
        snap,
        minutes_to_expiry_hint=minutes_to_expiry(calendar.synthetic_variant(), now, EXPIRIES[0]),
    )
    assert not report.by_kind().get("crossed_book")
    assert not report.by_kind().get("iv_out_of_range")
    assert not report.by_kind().get("non_monotone")


def test_crossed_book_is_detected(builder, chain_feed, registry):
    now = ist(9, 20)
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=SPOT))
    ticks = list(chain_feed.quotes_for(SPOT, now))
    # Must be a strike the spec actually captures - a corrupted quote outside
    # the captured window is dropped and would prove nothing.
    atm_id = next(
        c.instrument_id
        for c in registry.for_underlying(UNDERLYING)
        if c.strike == 24_800.0 and c.opt_type is OptionType.CE and c.expiry == EXPIRIES[0]
    )
    for t in ticks:
        builder.on_tick(
            t.model_copy(update={"bid": 200.0, "ask": 100.0}) if t.instrument_id == atm_id else t
        )

    snap = builder.snapshot(now)
    assert snap is not None
    assert check_chain(snap).by_kind().get("crossed_book", 0) >= 1


def test_non_monotone_prices_are_detected(builder, chain_feed):
    """A call worth more than a lower strike is a stale quote or a bad fill."""
    now = ist(9, 20)
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=SPOT))
    ticks = list(chain_feed.quotes_for(SPOT, now))
    for t in ticks:
        builder.on_tick(t)
    snap = builder.snapshot(now)
    assert snap is not None

    calls = sorted(
        (q for q in snap.quotes if q.opt_type is OptionType.CE and q.mid),
        key=lambda q: (q.expiry, q.strike),
    )
    # Pick a strike with a lower neighbour in the SAME expiry, otherwise
    # inflating it breaks nothing: prices decrease in strike, so corrupting the
    # first strike of an expiry leaves the ordering intact.
    first_expiry = [q for q in calls if q.expiry == calls[0].expiry]
    target = first_expiry[len(first_expiry) // 2]
    corrupted = snap.model_copy(
        update={
            "quotes": [
                q.model_copy(update={"bid": 9_000.0, "ask": 9_100.0}) if q is target else q
                for q in snap.quotes
            ]
        }
    )
    assert check_chain(corrupted).by_kind().get("non_monotone", 0) >= 1


# ------------------------------------------------------------------ helpers
def test_atm_straddle_is_found(builder, chain_feed):
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    pair = atm_straddle(snap)
    assert pair is not None
    call, put = pair
    assert call.strike == put.strike == snap.atm_strike()
    assert call.opt_type is OptionType.CE and put.opt_type is OptionType.PE


def test_liquidity_filter_excludes_untradeable_strikes(builder, chain_feed):
    """A theoretically optimal strike that cannot be exited is a trap."""
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    tight = liquid_quotes(snap, max_spread_pct=0.05)
    loose = liquid_quotes(snap, max_spread_pct=0.50)
    assert len(tight) < len(loose) <= len(snap.quotes)

    # Derive the threshold from the data rather than hard-coding one: a magic
    # number here would silently stop testing anything the moment the generator
    # or the strike window changed.
    ois = sorted(q.oi for q in snap.quotes if q.oi)
    median_oi = ois[len(ois) // 2]
    with_oi = liquid_quotes(snap, min_oi=median_oi, max_spread_pct=0.50)
    assert 0 < len(with_oi) < len(loose)


def test_spreads_widen_away_from_the_money(builder, chain_feed):
    snap = feed_a_snapshot(builder, chain_feed, ist(9, 20))
    assert snap is not None
    near = [
        q.spread_pct_of_mid
        for q in snap.quotes
        if q.spread_pct_of_mid and abs(q.strike - SPOT) <= 100
    ]
    far = [
        q.spread_pct_of_mid
        for q in snap.quotes
        if q.spread_pct_of_mid and abs(q.strike - SPOT) >= 600
    ]
    assert near and far
    assert sum(far) / len(far) > sum(near) / len(near)


def test_weekly_expiries_avoid_non_trading_days(calendar):
    cal = calendar.synthetic_variant()
    out = weekly_expiries(cal, TEST_DAY, count=3, weekday=3)
    assert len(out) == 3
    assert all(cal.is_trading_day(d) for d in out)
    assert out == sorted(out)


def test_expected_quote_count_matches_the_spec(spec):
    assert spec.expected_quote_count() == 2 * (2 * 15 + 1) * 2


def test_snapshot_is_deterministic(spec, registry, calendar):
    def run():
        b = ChainBuilder(spec, registry, calendar)
        f = SyntheticChainFeed(registry, calendar, underlying_id=UNDERLYING, seed=99)
        return feed_a_snapshot(b, f, ist(9, 20))

    a, b = run(), run()
    assert a is not None and b is not None
    # trace_id is per-message provenance, not data - exclude it at every level.
    without_ids = {"trace_id": True, "quotes": {"__all__": {"trace_id"}}}
    assert a.model_dump_json(exclude=without_ids) == b.model_dump_json(exclude=without_ids)


# ----------------------------------------------- the forward the book quotes
BASIS_SPOT = 24_000.0
BASIS_FORWARD = 24_040.0  # about twenty points above the model forward, as the live book was
BASIS_VOL = 0.13


def _basis_chain(calendar, now, expiry, *, spot=BASIS_SPOT, forward=BASIS_FORWARD):
    """A book priced at `forward`, which is deliberately not `spot * exp(r*t)`.

    The synthetic chain feed prices off `forward_from_spot`, so a book built by
    it agrees with the model forward by construction and cannot show this class
    of error at all. That is why these quotes are built by hand.
    """
    from libs.ms_data.chain import MINUTES_PER_YEAR
    from libs.ms_options.pricing import OptType, black76_price

    strikes = [spot + 50.0 * i for i in range(-8, 9)]
    contracts, ticks = [], []
    t = minutes_to_expiry(calendar, now, expiry) / MINUTES_PER_YEAR
    token = 9000
    for strike in strikes:
        for opt, kind in ((OptionType.CE, OptType.CALL), (OptionType.PE, OptType.PUT)):
            token += 1
            contracts.append(OptionInstrument(token, UNDERLYING, expiry, strike, opt))
            fair = black76_price(forward, strike, BASIS_VOL, t, 0.065, kind)
            ticks.append(
                Tick(
                    ts=now,
                    instrument_id=token,
                    ltp=fair,
                    bid=fair - 0.05,
                    ask=fair + 0.05,
                    oi=1000,
                )
            )
    return OptionRegistry(contracts), ticks


def test_a_call_and_its_put_agree_on_volatility(calendar):
    """The invariance that would have caught this, and did not exist.

    Put-call parity is not a modelling opinion: at one strike and one expiry a
    call and a put are the same volatility seen twice. If the two disagree, the
    forward used to invert them is wrong, and nothing else can cause it.

    On the first live NIFTY session every call came out above its own put - 1.7
    volatility points at the money, 10.8 in the wings - because the builder
    priced against `spot * exp(r*t)` while the book was quoting a forward
    twenty points higher. Every downstream reader saw that as skew.
    """
    now = ist(9, 20)
    expiry = TEST_DAY + timedelta(days=2)
    registry, ticks = _basis_chain(calendar, now, expiry)
    spec = ChainSpec(underlying_id=UNDERLYING, n_expiries=1, strikes_per_side=8)
    builder = ChainBuilder(spec, registry, calendar)

    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=BASIS_SPOT))
    for tick in ticks:
        builder.on_tick(tick)
    snap = builder.snapshot(now)
    assert snap is not None

    pairs = {}
    for q in snap.quotes:
        if q.iv is not None:
            pairs.setdefault(q.strike, {})[q.opt_type] = q.iv
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    assert len(both) >= 8, f"only {len(both)} strikes inverted on both sides"

    for strike, sides in sorted(both.items()):
        call, put = sides[OptionType.CE], sides[OptionType.PE]
        assert call == pytest.approx(put, abs=0.005), (
            f"strike {strike:.0f}: call IV {call:.4f} against put IV {put:.4f}. "
            "A call and a put at one strike are the same volatility; a gap here "
            "means the forward they were inverted against is wrong."
        )


def test_the_recovered_volatility_is_the_one_the_book_was_priced_with(calendar):
    """Agreeing with each other is not enough - they must agree with the truth.

    Two IVs inverted against the same wrong forward can still match one another
    at the money while both being wrong, so the level is pinned as well as the
    symmetry.
    """
    now = ist(9, 20)
    expiry = TEST_DAY + timedelta(days=2)
    registry, ticks = _basis_chain(calendar, now, expiry)
    builder = ChainBuilder(
        ChainSpec(underlying_id=UNDERLYING, n_expiries=1, strikes_per_side=8), registry, calendar
    )
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=BASIS_SPOT))
    for tick in ticks:
        builder.on_tick(tick)
    snap = builder.snapshot(now)
    assert snap is not None

    near = [q.iv for q in snap.quotes if q.iv is not None and abs(q.strike - BASIS_FORWARD) <= 200]
    assert near
    for iv in near:
        assert iv == pytest.approx(BASIS_VOL, abs=0.01)


def test_the_model_forward_is_what_splits_them(calendar):
    """A detector nobody has watched fire is not a detector.

    Inverting the same book against `spot * exp(r*t)` - what the builder did
    before - has to reproduce the split, or the test above would pass for
    reasons unrelated to the fix.
    """
    from libs.ms_data.chain import MINUTES_PER_YEAR
    from libs.ms_options.pricing import OptType, black76_price, forward_from_spot, implied_vol

    now = ist(9, 20)
    expiry = TEST_DAY + timedelta(days=2)
    t = minutes_to_expiry(calendar, now, expiry) / MINUTES_PER_YEAR
    model_fwd = forward_from_spot(BASIS_SPOT, 0.065, t)
    assert model_fwd < BASIS_FORWARD - 20, "the fixture must have a basis worth detecting"

    strike = BASIS_SPOT
    call_mid = black76_price(BASIS_FORWARD, strike, BASIS_VOL, t, 0.065, OptType.CALL)
    put_mid = black76_price(BASIS_FORWARD, strike, BASIS_VOL, t, 0.065, OptType.PUT)
    call_iv = implied_vol(call_mid, model_fwd, strike, t, 0.065, OptType.CALL)
    put_iv = implied_vol(put_mid, model_fwd, strike, t, 0.065, OptType.PUT)
    assert call_iv is not None and put_iv is not None
    assert call_iv - put_iv > 0.01, (
        "inverting against the model forward should split call from put by more than "
        f"a volatility point; got {call_iv:.4f} and {put_iv:.4f}"
    )


def test_a_one_sided_book_falls_back_to_the_model_forward(calendar):
    """Parity needs both legs. With calls only there is no forward to read off
    the book, and the honest answer is the model forward rather than no IV at
    all - a chain of calls is still worth inverting."""
    now = ist(9, 20)
    expiry = TEST_DAY + timedelta(days=2)
    registry, ticks = _basis_chain(calendar, now, expiry)
    calls_only = [
        t_
        for t_ in ticks
        if registry.get(t_.instrument_id).opt_type is OptionType.CE  # type: ignore[union-attr]
    ]
    builder = ChainBuilder(
        ChainSpec(underlying_id=UNDERLYING, n_expiries=1, strikes_per_side=8), registry, calendar
    )
    builder.on_tick(Tick(ts=now, instrument_id=UNDERLYING, ltp=BASIS_SPOT))
    for tick in calls_only:
        builder.on_tick(tick)
    snap = builder.snapshot(now)
    assert snap is not None
    assert [q for q in snap.quotes if q.iv is not None], "calls alone must still invert"
