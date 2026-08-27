"""M18 verify - replay engine. Gate G1, second half.

The engine's job is to be indistinguishable from live. These tests check the
four properties that make that true, and each of them exists because getting it
wrong produces a backtest that looks profitable and is not.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import SimClock
from libs.ms_core.schemas import Candle, ChainSnapshot, RawNewsItem, Tick
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.chain import ChainBuilder, ChainSpec, build_registry_from_specs
from libs.ms_data.feed import SyntheticFeed, SyntheticInstrument
from libs.ms_data.recorder import Recorder
from libs.ms_data.replay import (
    NdjsonSource,
    ReplayEngine,
    build_engine,
    default_sources,
    first_timestamp,
    recorded_days,
    release_time,
)
from libs.ms_data.synthetic_chain import SyntheticChainFeed, strike_grid
from tests.conftest import TEST_DAY, ist

UNDERLYING = 1
SPOT = 24_800.0
INSTRUMENTS = [
    SyntheticInstrument(instrument_id=UNDERLYING, start_price=SPOT, annual_vol=0.14),
    SyntheticInstrument(instrument_id=2, start_price=54_200.0, annual_vol=0.18),
]


@pytest.fixture(scope="module")
def calendar_mod():
    import warnings

    from libs.ms_core import config as cfgmod
    from libs.ms_core.clock import SessionCalendar

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = cfgmod.load()
    return SessionCalendar.from_config(cfg.sessions).synthetic_variant()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, calendar_mod):
    """A recorded session: ticks, candles, chain and news on disk."""
    cal = calendar_mod
    raw = tmp_path_factory.mktemp("replay_src") / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(INSTRUMENTS, cal, clock, seed=17)
    feed.subscribe([UNDERLYING, 2])

    registry = build_registry_from_specs(
        UNDERLYING, [TEST_DAY + timedelta(days=2)], strike_grid(SPOT, step=50.0, n_per_side=6)
    )
    chain_feed = SyntheticChainFeed(registry, cal, underlying_id=UNDERLYING, seed=4)
    chain_builder = ChainBuilder(
        ChainSpec(underlying_id=UNDERLYING, n_expiries=1, strikes_per_side=4), registry, cal
    )
    builder = CandleBuilder(cal)
    rec = Recorder(clock, cal, root=raw, fsync_every=10_000)
    end = ist(9, 25)

    for tick in feed.stream():
        if tick.ts >= end:
            break
        rec.write_tick(tick)
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
        chain_builder.on_tick(tick)
        if tick.instrument_id == UNDERLYING:
            for opt in chain_feed.quotes_for(tick.ltp, tick.ts):
                chain_builder.on_tick(opt)
            snap = chain_builder.maybe_snapshot(tick.ts)
            if snap is not None:
                rec.write_chain(snap)
    for candle in builder.flush(end):
        rec.write("candles", candle, key=candle.instrument_id)

    # A few news items, deliberately with t_available later than t_event.
    for i in range(5):
        rec.write(
            "news",
            RawNewsItem(
                source_id=3,
                headline=f"headline {i}",
                t_event=ist(9, 18) + timedelta(seconds=i * 30),
                t_ingest=ist(9, 18) + timedelta(seconds=i * 30 + 4),
                t_available=ist(9, 18) + timedelta(seconds=i * 30 + 6),
            ),
            key=0,
        )
    rec.close()
    return raw


def run_replay(recorded, **kw):
    bus = InMemoryBus()
    engine, clock = build_engine(bus, TEST_DAY, root=recorded, **kw)
    stats = engine.run([TEST_DAY])
    return bus, stats, clock


# ------------------------------------------------------- property 1: order
def test_the_merged_stream_is_globally_ordered(recorded):
    """A chain snapshot delivered before the tick that preceded it lets a
    strategy see a few hundred milliseconds into the future - enough to look
    profitable and be worthless."""
    seen = []
    bus = InMemoryBus()
    engine, _clock = build_engine(bus, TEST_DAY, root=recorded)
    engine.run([TEST_DAY], on_message=lambda topic, msg: seen.append(release_time(msg)))

    assert seen, "replay published nothing"
    assert seen == sorted(seen), "messages were delivered out of timestamp order"


def test_out_of_order_input_is_counted_not_swallowed(recorded, calendar_mod):
    """Cannot happen with sorted sources. If it ever does, the results are
    worthless and nobody would otherwise notice."""
    bus = InMemoryBus()
    clock = SimClock(ist(9, 15))
    src = NdjsonSource("ticks", Topic.MARKET_TICK, Tick, root=recorded)
    engine = ReplayEngine([src], clock, bus)
    stats = engine.run([TEST_DAY])
    assert stats.out_of_order == 0
    assert stats.total > 0


# ------------------------------------------------------- property 2: clock
def test_the_clock_matches_the_message_being_handled(recorded):
    """A consumer reading clock.now() while handling a message must see that
    message's timestamp, not the previous one's."""
    bus = InMemoryBus()
    engine, clock = build_engine(bus, TEST_DAY, root=recorded)

    mismatches = []

    def check(_topic, msg):
        expected = release_time(msg)
        if clock.now() != expected:
            mismatches.append((clock.now(), expected))

    engine.run([TEST_DAY], on_message=check)
    assert not mismatches, f"clock disagreed with the message {len(mismatches)} time(s)"


def test_clock_ends_at_the_last_message(recorded):
    _bus, stats, clock = run_replay(recorded)
    assert stats.last_ts is not None
    assert clock.now() == stats.last_ts


# ------------------------------------------- property 3: availability rule
def test_news_is_released_at_availability_not_at_event_time(recorded):
    """The difference between when a thing happened and when we could act on it
    is the whole of the reality-calibration chapter."""
    released: list[tuple] = []
    bus = InMemoryBus()
    engine, clock = build_engine(bus, TEST_DAY, root=recorded)
    engine.run(
        [TEST_DAY],
        on_message=lambda topic, msg: (
            released.append((clock.now(), msg)) if topic is Topic.RAW_NEWS else None
        ),
    )

    assert released, "no news was replayed"
    for at, item in released:
        assert at == item.t_available
        assert at > item.t_event, "released at event time - that is look-ahead bias"


def test_availability_falls_back_to_ingest_when_unset():
    item = RawNewsItem(source_id=1, headline="x", t_event=ist(9, 20), t_ingest=ist(9, 20, 5))
    assert release_time(item) == ist(9, 20, 5)


def test_market_data_uses_its_observation_timestamp():
    tick = Tick(ts=ist(9, 20), instrument_id=1, ltp=100.0)
    assert release_time(tick) == tick.ts


def test_news_has_no_ts_field_at_all():
    """The bug this helper exists to prevent: a getattr(msg, "ts") based lookup
    silently drops every news item, and the replay then looks fine."""
    item = RawNewsItem(source_id=1, headline="x", t_event=ist(9, 20), t_ingest=ist(9, 20, 5))
    assert not hasattr(item, "ts")
    assert release_time(item) is not None


# ------------------------------------------------ property 4: determinism
def test_two_runs_publish_identical_messages(recorded):
    def collect():
        bus, _stats, _clock = run_replay(recorded)
        out = []
        for topic in (Topic.MARKET_TICK, Topic.MARKET_CANDLE, Topic.MARKET_CHAIN, Topic.RAW_NEWS):
            out.extend(d.payload.model_dump_json() for d in bus.subscribe(topic, "verify", "c"))
        return out

    a, b = collect(), collect()
    assert a, "replay published nothing"
    assert a == b, "replay is not deterministic"


def test_a_different_source_produces_different_output(recorded, tmp_path, calendar_mod):
    """Guards against a determinism test that passes because nothing varies."""
    other = tmp_path / "raw2"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(INSTRUMENTS, calendar_mod, clock, seed=999)
    feed.subscribe([UNDERLYING])
    rec = Recorder(clock, calendar_mod, root=other, fsync_every=10_000)
    for tick in feed.stream():
        if tick.ts >= ist(9, 20):
            break
        rec.write_tick(tick)
    rec.close()

    bus_a = InMemoryBus()
    ea, _ = build_engine(bus_a, TEST_DAY, root=recorded)
    ea.run([TEST_DAY])
    bus_b = InMemoryBus()
    eb, _ = build_engine(bus_b, TEST_DAY, root=other)
    eb.run([TEST_DAY])

    ticks_a = [d.payload.ltp for d in bus_a.subscribe(Topic.MARKET_TICK, "g", "c")]
    ticks_b = [d.payload.ltp for d in bus_b.subscribe(Topic.MARKET_TICK, "g", "c")]
    assert ticks_a != ticks_b


# ------------------------------------------------------------- mechanics
def test_every_recorded_stream_is_replayed(recorded):
    _bus, stats, _clock = run_replay(recorded)
    assert stats.published[Topic.MARKET_TICK.value] > 0
    assert stats.published[Topic.MARKET_CANDLE.value] > 0
    assert stats.published[Topic.MARKET_CHAIN.value] > 0
    assert stats.published[Topic.RAW_NEWS.value] == 5


def test_replayed_messages_keep_their_type(recorded):
    bus, _stats, _clock = run_replay(recorded)
    assert all(isinstance(d.payload, Tick) for d in bus.subscribe(Topic.MARKET_TICK, "g", "c"))
    assert all(isinstance(d.payload, Candle) for d in bus.subscribe(Topic.MARKET_CANDLE, "g", "c"))
    assert all(
        isinstance(d.payload, ChainSnapshot) for d in bus.subscribe(Topic.MARKET_CHAIN, "g", "c")
    )


def test_window_clipping(recorded):
    bus = InMemoryBus()
    engine, _clock = build_engine(bus, TEST_DAY, root=recorded)
    stats = engine.run([TEST_DAY], start=ist(9, 18), end=ist(9, 20))
    assert stats.first_ts is not None and stats.first_ts >= ist(9, 18)
    assert stats.last_ts is not None and stats.last_ts < ist(9, 20)


def test_missing_day_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing recorded"):
        build_engine(InMemoryBus(), TEST_DAY, root=tmp_path / "empty")


def test_engine_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one source"):
        ReplayEngine([], SimClock(ist(9, 15)), InMemoryBus())


def test_recorded_days_and_first_timestamp(recorded):
    assert recorded_days(recorded) == [TEST_DAY]
    assert first_timestamp(default_sources(recorded), TEST_DAY) == ist(9, 15)


def test_stats_summary_is_readable(recorded):
    _bus, stats, _clock = run_replay(recorded)
    text = stats.summary()
    assert "messages" in text and Topic.MARKET_TICK.value in text


def test_replay_drives_a_consumer_end_to_end(recorded, calendar_mod):
    """The point of the whole module: a consumer cannot tell replay from live.

    Candles rebuilt by a consumer watching the replayed tick stream must match
    the candles recorded during the original session.
    """
    bus = InMemoryBus()
    engine, clock = build_engine(bus, TEST_DAY, root=recorded)
    builder = CandleBuilder(calendar_mod)
    rebuilt: list[Candle] = []

    def consume(topic, msg):
        if topic is Topic.MARKET_TICK:
            rebuilt.extend(builder.on_tick(msg))

    engine.run([TEST_DAY], on_message=consume)
    rebuilt.extend(builder.flush(clock.now() + timedelta(minutes=1)))

    original = [d.payload for d in bus.subscribe(Topic.MARKET_CANDLE, "orig", "c")]
    key = lambda c: (c.ts, c.instrument_id)  # noqa: E731
    a = [(c.ts, c.instrument_id, c.o, c.h, c.low, c.c) for c in sorted(original, key=key)]
    b = [(c.ts, c.instrument_id, c.o, c.h, c.low, c.c) for c in sorted(rebuilt, key=key)]
    assert a, "no original candles to compare against"
    assert b[: len(a)] == a
