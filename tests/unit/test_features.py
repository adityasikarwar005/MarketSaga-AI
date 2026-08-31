"""M40 verify - feature registry, context and catalogue."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from libs.ms_core.schemas import Candle, ChainSnapshot, OptionQuote, OptionType, Tick
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.registry import (
    REGISTRY,
    AccessLog,
    FeatureContext,
    FeatureError,
    FeatureRegistry,
    feature,
)
from libs.ms_features.sources import OfflineSource, OnlineSource
from tests.conftest import ist


def candles(n: int = 60, start_hour: int = 9, start_min: int = 15, price: float = 100.0):
    out = []
    p = price
    for i in range(n):
        o = p
        p = p * (1 + 0.0004 * ((i % 7) - 3))
        out.append(
            Candle(
                ts=ist(start_hour, start_min) + timedelta(minutes=i),
                instrument_id=1,
                tf=1,
                o=round(o, 2),
                h=round(max(o, p) * 1.0008, 2),
                l=round(min(o, p) * 0.9992, 2),
                c=round(p, 2),
                v=1000 + i,
                vwap=round((o + p) / 2, 4),
            )
        )
    return out


@pytest.fixture
def source():
    return OfflineSource(candles(120))


# ------------------------------------------------------------------- context
def test_the_in_progress_candle_is_excluded(source):
    """At 09:20:30 the 09:20 candle is half formed. Including it leaks thirty
    seconds of the future into every feature built on it."""
    ctx = FeatureContext(source, 1, ist(9, 20, 30))
    bars = ctx.candles(minutes=5)
    assert bars, "expected some history"
    assert all(b.ts < ist(9, 20) for b in bars)
    assert bars[-1].ts == ist(9, 19)


def test_window_is_exactly_the_requested_length(source):
    ctx = FeatureContext(source, 1, ist(10, 0))
    assert len(ctx.candles(minutes=5)) == 5
    assert len(ctx.candles(minutes=15)) == 15


def test_context_cannot_reach_the_future(source):
    """Structural, not conventional: there is no argument that returns a later row."""
    ctx = FeatureContext(source, 1, ist(9, 40))
    for minutes in (1, 5, 30, 375):
        assert all(b.ts < ist(9, 40) for b in ctx.candles(minutes=minutes))


def test_ticks_window_is_right_open():
    ticks = [Tick(ts=ist(9, 20, s), instrument_id=1, ltp=100.0 + s) for s in range(60)]
    ctx = FeatureContext(OfflineSource(ticks=ticks), 1, ist(9, 20, 30))
    got = ctx.ticks(seconds=30)
    assert got and all(t.ts < ist(9, 20, 30) for t in got)


def test_chain_lookup_takes_the_latest_at_or_before_as_of():
    snaps = [
        ChainSnapshot(
            ts=ist(9, 20) + timedelta(seconds=5 * i),
            underlying_id=1,
            spot=100.0 + i,
            quotes=[],
        )
        for i in range(10)
    ]
    src = OfflineSource(chains=snaps)
    ctx = FeatureContext(src, 1, ist(9, 20, 22))
    snap = ctx.chain()
    assert snap is not None
    assert snap.ts == ist(9, 20, 20)  # not 09:20:25


def test_snapshot_stamped_exactly_at_as_of_is_excluded():
    """Right-open, like every other window.

    Tempting to include - it exists at that instant - but the online path cannot
    promise the snapshot has arrived by then: at a tied timestamp the merge may
    deliver the candle first. A boundary rule only one side can keep is
    train/serve skew wearing a reasonable-sounding argument.
    """
    snaps = [ChainSnapshot(ts=ist(9, 20), underlying_id=1, spot=100.0, quotes=[])]
    src = OfflineSource(chains=snaps)
    assert FeatureContext(src, 1, ist(9, 20)).chain() is None
    assert FeatureContext(src, 1, ist(9, 20, 1)).chain() is not None


def test_access_log_records_what_was_touched(source):
    log = AccessLog()
    ctx = FeatureContext(source, 1, ist(9, 40), access_log=log)
    ctx.candles(minutes=10)
    assert log.calls == 1
    assert log.max_ts is not None and log.max_ts < ist(9, 40)


# ------------------------------------------------------------------ registry
def test_feature_must_take_exactly_one_argument():
    reg = FeatureRegistry()
    with pytest.raises(FeatureError, match="exactly one argument"):

        @feature(name="bad", version=1, registry=reg)
        def bad(ctx, extra_data):  # type: ignore[misc]
            return 1.0


def test_re_registering_the_same_version_is_rejected():
    """An in-place edit silently invalidates every model that used the feature."""
    reg = FeatureRegistry()

    @feature(name="x", version=1, registry=reg)
    def x1(ctx):
        return 1.0

    with pytest.raises(FeatureError, match="version bump"):

        @feature(name="x", version=1, registry=reg)
        def x2(ctx):
            return 2.0


def test_version_bump_replaces_the_definition():
    reg = FeatureRegistry()

    @feature(name="x", version=1, registry=reg)
    def x1(ctx):
        return 1.0

    @feature(name="x", version=2, registry=reg)
    def x2(ctx):
        return 2.0

    assert reg.get("x").version == 2


def test_dependencies_are_computed_first(source):
    reg = FeatureRegistry()

    @feature(name="base", version=1, registry=reg)
    def base(ctx):
        return 10.0

    @feature(name="derived", version=1, deps=("base",), registry=reg)
    def derived(ctx):
        return (ctx.value("base") or 0) * 2

    values = reg.compute(source, 1, ist(9, 40))
    assert values["derived"] == 20.0


def test_undeclared_dependency_is_a_clear_error(source):
    reg = FeatureRegistry()

    @feature(name="a", version=1, registry=reg)
    def a(ctx):
        return 1.0

    @feature(name="b", version=1, registry=reg)  # forgot deps=("a",)
    def b(ctx):
        return ctx.value("a")

    with pytest.raises(FeatureError, match="declare it in `deps`"):
        reg.compute(source, 1, ist(9, 40), names=["b"])


def test_dependency_cycle_is_rejected():
    reg = FeatureRegistry()

    @feature(name="a", version=1, deps=("b",), registry=reg)
    def a(ctx):
        return 1.0

    @feature(name="b", version=1, deps=("a",), registry=reg)
    def b(ctx):
        return 1.0

    with pytest.raises(FeatureError, match="dependency cycle"):
        reg.resolution_order()


def test_nan_is_rejected(source):
    """NaN propagates silently into a model and is never noticed."""
    reg = FeatureRegistry()

    @feature(name="broken", version=1, registry=reg)
    def broken(ctx):
        return float("nan")

    with pytest.raises(FeatureError, match="finite number or None"):
        reg.get("broken").compute(FeatureContext(source, 1, ist(9, 40)))


def test_one_broken_feature_does_not_break_the_vector(source):
    reg = FeatureRegistry()

    @feature(name="fine", version=1, registry=reg)
    def fine(ctx):
        return 1.0

    @feature(name="explodes", version=1, registry=reg)
    def explodes(ctx):
        raise ZeroDivisionError("boom")

    values = reg.compute(source, 1, ist(9, 40))
    assert values["fine"] == 1.0
    assert values["explodes"] is None


def test_vector_hash_changes_with_version():
    reg = FeatureRegistry()

    @feature(name="a", version=1, registry=reg)
    def a1(ctx):
        return 1.0

    first = reg.vector_hash()

    @feature(name="a", version=2, registry=reg)
    def a2(ctx):
        return 1.0

    assert reg.vector_hash() != first


# ----------------------------------------------------------------- catalogue
def test_catalogue_is_registered():
    assert len(REGISTRY) >= 20
    assert "ret_5m" in REGISTRY
    assert "tod_vol_multiplier" in REGISTRY


def test_the_whole_catalogue_computes(source):
    values = REGISTRY.compute(source, 1, ist(10, 30))
    assert len(values) == len(REGISTRY)
    # Option features have no chain in this fixture, so some nulls are expected.
    computed = {k: v for k, v in values.items() if v is not None}
    assert len(computed) >= 15


def test_returns_have_the_right_sign(source):
    ctx = FeatureContext(source, 1, ist(10, 0))
    bars = ctx.candles(minutes=5)
    expected = math.log(bars[-1].c / bars[0].o)
    assert REGISTRY.get("ret_5m").compute(ctx) == pytest.approx(expected)


def test_range_position_is_bounded(source):
    for minute in (30, 60, 90):
        ctx = FeatureContext(source, 1, ist(9, 15) + timedelta(minutes=minute))
        value = REGISTRY.get("range_position").compute(ctx)
        assert value is None or 0.0 <= value <= 1.0


def test_time_of_day_multiplier_is_u_shaped(source):
    def at(hh, mm):
        return REGISTRY.compute(source, 1, ist(hh, mm), names=["tod_vol_multiplier"])[
            "tod_vol_multiplier"
        ]

    open_v, mid_v, close_v = at(9, 20), at(12, 30), at(15, 20)
    assert open_v and mid_v and close_v
    assert open_v > mid_v and close_v > mid_v


def test_session_flags(source):
    def at(hh, mm, name):
        return REGISTRY.compute(source, 1, ist(hh, mm), names=[name])[name]

    assert at(9, 20, "is_opening_range") == 1.0
    assert at(11, 0, "is_opening_range") == 0.0
    assert at(15, 25, "is_closing_range") == 1.0
    assert at(11, 0, "is_closing_range") == 0.0


def test_insufficient_history_returns_none_not_a_guess():
    """Two candles is not a thirty-minute window, and pretending otherwise
    produces a number that looks like the real thing."""
    src = OfflineSource(candles(2))
    ctx = FeatureContext(src, 1, ist(9, 20))
    assert REGISTRY.get("ret_30m").compute(ctx) is None
    assert REGISTRY.get("efficiency_ratio_30m").compute(ctx) is None


def test_option_features_use_the_chain():
    quotes = []
    for strike in (96.0, 98.0, 100.0, 102.0, 104.0):
        for opt, iv in ((OptionType.CE, 0.14), (OptionType.PE, 0.16)):
            quotes.append(
                OptionQuote(
                    expiry=ist(15, 30),
                    strike=strike,
                    opt_type=opt,
                    bid=1.0,
                    ask=1.1,
                    iv=iv + (100.0 - strike) * 0.004,
                )
            )
    snap = ChainSnapshot(
        ts=ist(9, 19), underlying_id=1, spot=100.0, quotes=quotes, completeness=0.98
    )
    src = OfflineSource(candles(60), chains=[snap])
    values = REGISTRY.compute(src, 1, ist(9, 20))

    assert values["atm_iv"] is not None
    assert values["chain_completeness"] == pytest.approx(0.98)
    assert values["chain_age_seconds"] == pytest.approx(60.0)
    assert values["iv_skew_25d"] is not None and values["iv_skew_25d"] > 0  # put skew


def test_chain_age_is_never_negative():
    """A negative age would mean the chain is from the future."""
    snap = ChainSnapshot(ts=ist(9, 19), underlying_id=1, spot=100.0, quotes=[])
    src = OfflineSource(candles(30), chains=[snap])
    for minute in range(20, 30):
        values = REGISTRY.compute(src, 1, ist(9, minute), names=["chain_age_seconds"])
        age = values["chain_age_seconds"]
        assert age is None or age >= 0


# -------------------------------------------------------------- online source
def test_online_source_reports_a_short_window():
    """Silent truncation produces a different number from the offline twin, and
    a value mismatch is far harder to diagnose than an explicit short read."""
    online = OnlineSource(max_minutes=10)
    for c in candles(10):
        online.on_candle(c)
    ctx = FeatureContext(online, 1, ist(9, 30))
    ctx.candles(minutes=60)
    assert online.short_windows
    assert "candles" in str(online.short_windows[0])


def test_online_source_trims_by_time():
    online = OnlineSource(max_minutes=10)
    for c in candles(60):
        online.on_candle(c)
    cov = online.coverage(1)
    assert cov.count <= 11


def test_warm_reports_readiness():
    """A service must not emit features before this is true, or its first
    predictions of the session are computed on a partial history."""
    online = OnlineSource(max_minutes=60)
    for c in candles(5):
        online.on_candle(c)
    assert not online.warm(1, minutes=30)
    for c in candles(40):
        online.on_candle(c)
    assert online.warm(1, minutes=30)


# ------------------------------------------------- group C: term structure etc.
SPOT_C = 24_000.0


def _chain_quotes(
    expiry,
    *,
    strikes,
    oi_at,
    iv_at=lambda k: 0.15,
    book_at=None,
    ltp_at=lambda k, o: None,
):
    """Quotes for one expiry. ``book_at`` returns (bid, ask) or None for no book."""
    out = []
    for strike in strikes:
        for opt in (OptionType.CE, OptionType.PE):
            book = book_at(strike, opt) if book_at else None
            out.append(
                OptionQuote(
                    expiry=expiry,
                    strike=strike,
                    opt_type=opt,
                    bid=None if book is None else book[0],
                    ask=None if book is None else book[1],
                    ltp=ltp_at(strike, opt) if book is None else (book[0] + book[1]) / 2,
                    oi=oi_at(strike, opt),
                    iv=iv_at(strike),
                )
            )
    return out


def _ladder(n_per_side: int = 6, step: float = 50.0):
    return [SPOT_C + step * i for i in range(-n_per_side, n_per_side + 1)]


def _snapshot(quotes, *, ts=None, spot=SPOT_C):
    return ChainSnapshot(
        ts=ts or ist(9, 19), underlying_id=1, spot=spot, quotes=quotes, completeness=1.0
    )


def _values(snap, names=None, as_of=None):
    src = OfflineSource(candles(60, price=SPOT_C), chains=[snap])
    return REGISTRY.compute(src, 1, as_of or ist(9, 20), names=names)


def test_days_to_expiry_counts_calendar_days():
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(_chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 1000))
    assert _values(snap)["days_to_expiry"] == pytest.approx(2.0)


def test_front_expiry_is_chosen_by_date_not_by_file_order():
    """A snapshot that still carries a settled series must not point at it.

    The expired quotes are listed first here on purpose. Taking `quotes[0].expiry`
    or `min()` over every expiry present would give a negative days-to-expiry and
    a gamma kernel evaluated at a floored time it was never meant to see.
    """
    stale = ist(15, 30) - timedelta(days=3)
    live = ist(15, 30) + timedelta(days=4)
    quotes = _chain_quotes(stale, strikes=_ladder(), oi_at=lambda k, o: 999)
    quotes += _chain_quotes(live, strikes=_ladder(), oi_at=lambda k, o: 1000)
    assert _values(_snapshot(quotes))["days_to_expiry"] == pytest.approx(4.0)


def test_naive_expiry_is_refused_rather_than_guessed():
    """A naive timestamp read as machine-local is wrong by five and a half hours,
    which moves an expiry across a day boundary without raising anything."""
    from libs.ms_features.catalogue import _ist_date

    with pytest.raises(ValueError, match="naive"):
        _ist_date(datetime(2026, 8, 31, 15, 30))


def test_iv_term_slope_matches_the_difference_of_the_two_atm_ivs():
    front = ist(15, 30) + timedelta(days=2)
    back = ist(15, 30) + timedelta(days=9)
    quotes = _chain_quotes(front, strikes=_ladder(), oi_at=lambda k, o: 10, iv_at=lambda k: 0.12)
    quotes += _chain_quotes(back, strikes=_ladder(), oi_at=lambda k, o: 10, iv_at=lambda k: 0.17)
    assert _values(_snapshot(quotes))["iv_term_slope"] == pytest.approx(0.05)


def test_a_single_expiry_gives_no_term_slope_rather_than_a_flat_one():
    """Null test. Zero is a flat term structure - a real and rare market state.
    Returning it when only one expiry was captured teaches a model that a gap in
    the recording looks exactly like an unusual day."""
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(_chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 10))
    assert _values(snap)["iv_term_slope"] is None


def test_implied_move_is_the_atm_straddle_over_spot():
    expiry = ist(15, 30) + timedelta(days=2)

    def book(strike, opt):
        if strike != SPOT_C:
            return (10.0, 11.0)
        return (145.0, 155.0) if opt is OptionType.CE else (125.0, 135.0)

    snap = _snapshot(_chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 10, book_at=book))
    # mids are 150 and 130; the straddle is 280 against a spot of 24,000.
    assert _values(snap)["implied_move_pct"] == pytest.approx(280.0 / SPOT_C)


def test_implied_move_is_none_without_a_book():
    """Null test, and the one that matters most for the archive.

    Eight years of bhavcopy carry a settlement price and no bid or ask. Marking
    the straddle off two last-traded prices would produce a plausible number for
    every one of those sessions, and it would not be a straddle price. None here
    is M70's finding restated: only a recorded intraday chain can support this.
    """
    expiry = ist(15, 30) + timedelta(days=2)
    quotes = _chain_quotes(
        expiry,
        strikes=_ladder(),
        oi_at=lambda k, o: 10,
        book_at=None,
        ltp_at=lambda k, o: 150.0,  # a settlement mark, as the archive supplies it
    )
    assert all(q.ltp is not None and q.bid is None for q in quotes)
    assert _values(_snapshot(quotes))["implied_move_pct"] is None


def test_gamma_concentration_rises_when_open_interest_hugs_spot():
    expiry = ist(15, 30) + timedelta(days=2)
    tight = _snapshot(
        _chain_quotes(
            expiry,
            strikes=_ladder(),
            oi_at=lambda k, o: 100_000 if abs(k - SPOT_C) <= 50 else 500,
        )
    )
    spread = _snapshot(_chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 20_000))
    near = _values(tight)["gamma_concentration_shape"]
    far = _values(spread)["gamma_concentration_shape"]
    assert near is not None and far is not None
    assert near > far, (
        f"open interest pinned at the money scored {near:.4f}, uniform open interest "
        f"{far:.4f}. The statistic is meant to measure where open interest sits; if "
        "concentrating it does not raise the number, the kernel is not weighting by "
        "distance from spot."
    )


def test_gamma_concentration_is_free_of_the_index_level():
    """The normalisation, pinned against its own algebra.

    Black-Scholes gamma carries a 1/S, so a raw kernel-weighted sum falls by a
    third between NIFTY 8,000 and NIFTY 24,000 on an identically-shaped book -
    a model trained on the last decade would read that drift as a signal about
    volatility regimes. Multiplying by spot cancels it exactly, and this test is
    the only thing standing between that claim and a plausible-looking number.
    """
    expiry = ist(15, 30) + timedelta(days=2)
    out = {}
    for spot in (8_000.0, 24_000.0):
        step = spot / 480.0  # the same ladder in moneyness terms at both levels
        rungs = list(range(-6, 7))
        strikes = [spot + step * i for i in rungs]
        # Keyed by rung, never by comparing a strike against a width. At 8,000 the
        # step is 16.666... and `abs(k - spot) <= step` falls on the other side of
        # one ulp than it does at 24,000, which silently gave the two levels
        # different books and made this test fail for a reason it does not test.
        oi = {k: (100_000 if abs(i) <= 1 else 5_000) for i, k in zip(rungs, strikes, strict=True)}
        quotes = _chain_quotes(expiry, strikes=strikes, oi_at=lambda k, o, m=oi: m[k])
        out[spot] = _values(_snapshot(quotes, spot=spot))["gamma_concentration_shape"]
    assert out[8_000.0] is not None
    assert out[24_000.0] == pytest.approx(out[8_000.0], rel=1e-9), (
        f"the same book shape scored {out[8_000.0]:.6f} at 8,000 and "
        f"{out[24_000.0]:.6f} at 24,000. The index level is leaking into a "
        "statistic that is supposed to describe shape alone."
    )


def test_gamma_concentration_refuses_a_handful_of_contracts():
    """Null test: too few strikes is not a thin profile, it is no profile."""
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(
        _chain_quotes(expiry, strikes=_ladder(n_per_side=2), oi_at=lambda k, o: 10_000)
    )
    assert _values(snap)["gamma_concentration_shape"] is None


def test_gamma_concentration_is_none_when_nothing_is_open():
    """A chain of contracts with no open interest is not a book at zero
    concentration; it is a book nobody has taken a position in."""
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(_chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 0))
    assert _values(snap)["gamma_concentration_shape"] is None


def test_put_call_oi_ratio_pins_to_the_open_interest_it_was_given():
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(
        _chain_quotes(
            expiry,
            strikes=_ladder(),
            oi_at=lambda k, o: 3_000 if o is OptionType.PE else 2_000,
        )
    )
    assert _values(snap)["put_call_oi_ratio"] == pytest.approx(1.5)


def test_put_call_ratio_counts_only_the_front_expiry():
    """Summing every expiry would make the ratio a slow monthly average and hide
    exactly the positioning shift the feature exists to see."""
    front = ist(15, 30) + timedelta(days=2)
    back = ist(15, 30) + timedelta(days=9)
    quotes = _chain_quotes(
        front, strikes=_ladder(), oi_at=lambda k, o: 3_000 if o is OptionType.PE else 2_000
    )
    quotes += _chain_quotes(
        back, strikes=_ladder(), oi_at=lambda k, o: 1_000 if o is OptionType.PE else 9_000
    )
    assert _values(_snapshot(quotes))["put_call_oi_ratio"] == pytest.approx(1.5)


def test_vrp_is_implied_minus_realised_in_the_same_units():
    expiry = ist(15, 30) + timedelta(days=2)
    snap = _snapshot(
        _chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 10, iv_at=lambda k: 0.20)
    )
    values = _values(snap, as_of=ist(10, 0))  # rv_15m needs fifteen completed candles
    assert values["rv_15m"] is not None
    assert values["vrp"] == pytest.approx(values["atm_iv"] - values["rv_15m"])


def test_vrp_is_none_when_either_side_is_missing():
    """Null test. An implied volatility with no realised counterpart is not a
    premium of zero, and a subtraction against a default would look like one."""
    expiry = ist(15, 30) + timedelta(days=2)
    quotes = _chain_quotes(expiry, strikes=_ladder(), oi_at=lambda k, o: 10, iv_at=lambda k: 0.20)
    src = OfflineSource(candles(3, price=SPOT_C), chains=[_snapshot(quotes)])
    values = REGISTRY.compute(src, 1, ist(9, 18))
    assert values["rv_15m"] is None
    assert values["vrp"] is None


def test_group_c_is_silent_without_a_chain():
    """Every option feature returns None rather than a default when no chain has
    arrived. A zero here is a market observation; there is no observation."""
    src = OfflineSource(candles(60, price=SPOT_C))
    values = REGISTRY.compute(src, 1, ist(10, 30))
    for name in [d.name for d in REGISTRY.by_group("C")]:
        assert values[name] is None, f"{name} produced {values[name]!r} with no chain at all"


# ------------------------------------------------------- group D: cross-asset
USDINR_ID = 1006
ASIA_IDX_ID = 1008


def _bars(instrument_id: int, stamps_and_prices):
    """Candles for a cross-asset series at exactly the minutes given, from 09:15."""
    out = []
    for minute, price in stamps_and_prices:
        out.append(
            Candle(
                ts=ist(9, 15) + timedelta(minutes=minute),
                instrument_id=instrument_id,
                tf=1,
                o=price,
                h=price,
                l=price,
                c=price,
                v=100,
                vwap=price,
            )
        )
    return out


def test_a_cross_asset_return_pins_to_its_own_bars():
    """One instrument's window must not pick up another's bars."""
    fx = _bars(USDINR_ID, [(39, 88.0), (40, 88.0), (41, 88.0), (42, 88.0), (43, 88.44)])
    src = OfflineSource(candles(60) + fx)
    ctx = FeatureContext(src, 1, ist(9, 59))
    assert REGISTRY.get("usdinr_ret_5m").compute(ctx) == pytest.approx(math.log(88.44 / 88.0))


def test_a_sparse_window_that_still_spans_the_period_is_accepted():
    """A foreign series arrives sparsely and that is normal, not missing data.

    Four bars across half an hour is a thirty-minute return; the gate is on the
    span they cover, not on there being thirty of them.
    """
    fx = _bars(USDINR_ID, [(15, 88.0), (26, 88.1), (37, 88.2), (44, 88.3)])
    src = OfflineSource(candles(60) + fx)
    ctx = FeatureContext(src, 1, ist(10, 0))
    assert REGISTRY.get("usdinr_ret_30m").compute(ctx) == pytest.approx(math.log(88.3 / 88.0))


def test_five_bars_four_minutes_apart_are_not_a_thirty_minute_return():
    """The null test this group was designed around.

    Group A can demand one bar per minute; a foreign series cannot be held to
    that, so the temptation is to accept any few bars and divide. That computes
    a four-minute move and labels it `ret_30m` - the same mislabelling `_window`
    exists to prevent, wearing a different disguise.
    """
    fx = _bars(USDINR_ID, [(m, 88.0 + 0.1 * m) for m in (40, 41, 42, 43, 44)])
    src = OfflineSource(candles(60) + fx)
    ctx = FeatureContext(src, 1, ist(10, 0))
    assert REGISTRY.get("usdinr_ret_30m").compute(ctx) is None
    # The same bars genuinely are a five-minute window, and that one is allowed.
    assert REGISTRY.get("usdinr_ret_5m").compute(ctx) is not None


def test_a_single_bar_is_never_a_return():
    fx = _bars(USDINR_ID, [(43, 88.0)])
    src = OfflineSource(candles(60) + fx)
    ctx = FeatureContext(src, 1, ist(10, 0))
    assert REGISTRY.get("usdinr_ret_5m").compute(ctx) is None


def test_group_d_is_silent_without_any_cross_asset_data():
    """Null test. The collector being down is not a flat rupee."""
    src = OfflineSource(candles(120))
    values = REGISTRY.compute(src, 1, ist(10, 30))
    for name in [d.name for d in REGISTRY.by_group("D")]:
        assert values[name] is None, f"{name} produced {values[name]!r} with no cross-asset feed"


def test_staleness_grows_when_the_series_freezes():
    """A frozen collector does not raise; it reports that the world stopped
    moving. The only way a model can discount that is by being told."""
    fx = _bars(USDINR_ID, [(m, 88.0) for m in range(20)])  # 09:15 .. 09:34
    src = OfflineSource(candles(120) + fx)
    fresh = REGISTRY.compute(src, 1, ist(9, 36), names=["usdinr_staleness_minutes"])
    stale = REGISTRY.compute(src, 1, ist(10, 30), names=["usdinr_staleness_minutes"])
    assert fresh["usdinr_staleness_minutes"] == pytest.approx(2.0)
    assert stale["usdinr_staleness_minutes"] == pytest.approx(56.0)


def test_staleness_is_never_below_one_minute():
    """Right-open at the bucket containing as_of: the newest visible bar is
    always the previous completed minute, so anything under a minute would mean
    the window had leaked into the present."""
    fx = _bars(USDINR_ID, [(m, 88.0) for m in range(40)])
    src = OfflineSource(candles(120) + fx)
    for minute in range(20, 50):
        values = REGISTRY.compute(
            src, 1, ist(9, 15) + timedelta(minutes=minute), names=["usdinr_staleness_minutes"]
        )
        age = values["usdinr_staleness_minutes"]
        assert age is None or age >= 1.0, f"staleness {age} at minute {minute}"


def test_a_closed_asian_market_goes_absent_rather_than_frozen():
    """Tokyo shuts at 11:30 IST. The composite must stop reporting, not hold its
    last print - a frozen value looks like an Asian market that stopped moving,
    which is a very different statement from one that has closed."""
    asia = _bars(ASIA_IDX_ID, [(m, 3_180.0 + m) for m in range(60)])  # 09:15 .. 10:14
    src = OfflineSource(candles(240) + asia)
    assert (
        REGISTRY.compute(src, 1, ist(10, 10), names=["asia_idx_ret_30m"])["asia_idx_ret_30m"]
        is not None
    )
    assert (
        REGISTRY.compute(src, 1, ist(12, 30), names=["asia_idx_ret_30m"])["asia_idx_ret_30m"]
        is None
    )


def test_cross_asset_windows_cannot_reach_past_as_of():
    """The context filters by instrument and by time independently; a bug in
    either would show up here as a value computed from bars that had not
    happened yet."""
    fx = _bars(USDINR_ID, [(m, 88.0 + m) for m in range(60)])
    src = OfflineSource(candles(120) + fx)
    log = AccessLog()
    REGISTRY.compute(src, 1, ist(9, 45), access_log=log)
    assert log.max_ts is not None and log.max_ts < ist(9, 45)
