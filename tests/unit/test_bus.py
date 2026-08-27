"""M04 verify - event bus."""

from __future__ import annotations

import pytest

from libs.ms_core.bus import (
    DROPPABLE,
    TOPIC_SCHEMA,
    Delivery,
    InMemoryBus,
    SchemaMismatch,
    Topic,
)
from libs.ms_core.schemas import Candle, Tick
from tests.conftest import ist


def tick(i: int, iid: int = 1) -> Tick:
    return Tick(ts=ist(9, 20, i % 60), instrument_id=iid, ltp=100.0 + i, seq=i)


@pytest.fixture
def bus():
    return InMemoryBus()


def test_ordering_is_insertion_order(bus):
    """Determinism here is what lets the golden-day replay assert a fixed hash."""
    for i in range(10):
        bus.publish(Topic.MARKET_TICK, tick(i))
    got = [d.typed(Tick).ltp for d in bus.subscribe(Topic.MARKET_TICK, "g", "c")]
    assert got == [100.0 + i for i in range(10)]


def test_schema_mismatch_rejected_at_publish():
    """Fail at the boundary, not three stages downstream in an innocent module."""
    b = InMemoryBus()
    candle = Candle(ts=ist(9, 20), instrument_id=1, tf=1, o=1, h=2, l=1, c=2)
    with pytest.raises(SchemaMismatch, match=r"market\.tick"):
        b.publish(Topic.MARKET_TICK, candle)


def test_every_data_topic_declares_a_schema():
    for topic in Topic:
        if topic.value.startswith("audit."):
            continue
        assert topic in TOPIC_SCHEMA, f"{topic} has no declared schema"


def test_consumer_group_resumes_where_it_left_off(bus):
    for i in range(5):
        bus.publish(Topic.MARKET_TICK, tick(i))
    first = list(bus.subscribe(Topic.MARKET_TICK, "g", "c"))
    assert len(first) == 5

    for i in range(5, 8):
        bus.publish(Topic.MARKET_TICK, tick(i))
    second = list(bus.subscribe(Topic.MARKET_TICK, "g", "c"))
    assert [d.typed(Tick).seq for d in second] == [5, 6, 7]


def test_two_groups_each_see_everything(bus):
    for i in range(4):
        bus.publish(Topic.MARKET_TICK, tick(i))
    a = list(bus.subscribe(Topic.MARKET_TICK, "group-a", "c"))
    b = list(bus.subscribe(Topic.MARKET_TICK, "group-b", "c"))
    assert len(a) == len(b) == 4


def test_unacked_messages_are_pending(bus):
    for i in range(3):
        bus.publish(Topic.MARKET_TICK, tick(i))
    deliveries = list(bus.subscribe(Topic.MARKET_TICK, "g", "c"))
    assert bus.pending_count(Topic.MARKET_TICK, "g") == 3
    for d in deliveries:
        bus.ack(Topic.MARKET_TICK, "g", d.msg_id)
    assert bus.pending_count(Topic.MARKET_TICK, "g") == 0


def test_restart_redelivers_unacked(bus):
    """A consumer that crashes mid-processing must not lose the message."""
    for i in range(3):
        bus.publish(Topic.MARKET_TICK, tick(i))
    list(bus.subscribe(Topic.MARKET_TICK, "g", "c"))  # consumed, never acked
    bus.reset_group(Topic.MARKET_TICK, "g")
    again = list(bus.subscribe(Topic.MARKET_TICK, "g", "c"))
    assert len(again) == 3


def test_replay_by_time_window(bus):
    for i in range(60):
        bus.publish(Topic.MARKET_TICK, Tick(ts=ist(9, 20 + i // 10), instrument_id=1, ltp=100.0))
    window = list(bus.replay(Topic.MARKET_TICK, ist(9, 21), ist(9, 23)))
    assert window and all(ist(9, 21) <= d.payload.ts < ist(9, 23) for d in window)


def test_ticks_are_droppable_orders_are_not():
    """Losing a tick costs a data point. Losing an order intent costs money."""
    assert Topic.MARKET_TICK in DROPPABLE
    assert Topic.ORDER_INTENT not in DROPPABLE
    assert Topic.TRADE_RECO not in DROPPABLE
    assert Topic.ORDER_FILL not in DROPPABLE


def test_backpressure_drops_ticks_but_refuses_to_drop_orders():
    b = InMemoryBus(max_len=5)
    for i in range(20):
        b.publish(Topic.MARKET_TICK, tick(i))
    assert b.depth(Topic.MARKET_TICK) == 5  # oldest dropped, loudly counted

    from libs.ms_core.schemas import Order, Side

    b2 = InMemoryBus(max_len=2)
    for _ in range(2):
        b2.publish(
            Topic.ORDER_INTENT,
            Order(reco_id="r", ts=ist(9, 20), instrument_id=1, side=Side.BUY, qty_lots=1),
        )
    with pytest.raises(BufferError, match="not droppable"):
        b2.publish(
            Topic.ORDER_INTENT,
            Order(reco_id="r", ts=ist(9, 20), instrument_id=1, side=Side.BUY, qty_lots=1),
        )


def test_typed_accessor_guards_the_consumer(bus):
    bus.publish(Topic.MARKET_TICK, tick(1))
    d: Delivery = next(iter(bus.subscribe(Topic.MARKET_TICK, "g", "c")))
    assert d.typed(Tick).instrument_id == 1
    with pytest.raises(SchemaMismatch):
        d.typed(Candle)
