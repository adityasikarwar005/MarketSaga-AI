"""M40 verify - feature registry, context and catalogue."""

from __future__ import annotations

import math
from datetime import timedelta

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
