"""M04 - Event bus.

Two implementations behind one interface:

``InMemoryBus``
    Deterministic, dependency-free. Used by tests and by the replay simulator.
    Because the simulator drives the *same* engine code through the *same* bus
    interface, a backtest exercises the production code path - which is the only
    reason a backtest result means anything (blueprint Ch.14).

``RedisStreamsBus``
    Production. Consumer groups, acknowledgement, replay by timestamp range.
    Redis Streams rather than Kafka: Redis is already required for the online
    feature store, and Kafka's operational weight is not justified before
    multi-user (blueprint Ch.4).

Both validate messages against their schema on publish **and** on consume.
A malformed message must fail at the boundary, not three stages downstream where
the stack trace points at the wrong module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar, cast

from pydantic import ValidationError

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import (
    Candle,
    ChainSnapshot,
    ClassifiedEvent,
    FeatureSnapshot,
    Fill,
    Msg,
    Order,
    Outcome,
    Prediction,
    RawNewsItem,
    Recommendation,
    Tick,
)

log = get_logger(__name__)
M = TypeVar("M", bound=Msg)


class Topic(StrEnum):
    RAW_NEWS = "raw.news"
    EVENT_CLASSIFIED = "event.classified"
    MARKET_TICK = "market.tick"
    MARKET_CANDLE = "market.candle"
    MARKET_CHAIN = "market.chain"
    FEATURE_SNAPSHOT = "feature.snapshot"
    SIGNAL_PRED = "signal.pred"
    TRADE_RECO = "trade.reco"
    ORDER_INTENT = "order.intent"
    ORDER_FILL = "order.fill"
    OUTCOME = "outcome.resolved"
    AUDIT_GAP = "audit.gap"
    AUDIT_HALT = "audit.halt"


# One schema per topic. A consumer never has to guess what it received, and a
# producer cannot quietly start publishing something else.
TOPIC_SCHEMA: dict[Topic, type[Msg]] = {
    Topic.RAW_NEWS: RawNewsItem,
    Topic.EVENT_CLASSIFIED: ClassifiedEvent,
    Topic.MARKET_TICK: Tick,
    Topic.MARKET_CANDLE: Candle,
    Topic.MARKET_CHAIN: ChainSnapshot,
    Topic.FEATURE_SNAPSHOT: FeatureSnapshot,
    Topic.SIGNAL_PRED: Prediction,
    Topic.TRADE_RECO: Recommendation,
    Topic.ORDER_INTENT: Order,
    Topic.ORDER_FILL: Fill,
    Topic.OUTCOME: Outcome,
}

# Topics that may be dropped under backpressure, and those that may never be.
# Losing a tick costs one data point; losing an order intent loses money.
DROPPABLE: frozenset[Topic] = frozenset({Topic.MARKET_TICK, Topic.RAW_NEWS})


class SchemaMismatch(TypeError):
    """Raised when a message does not match the schema declared for its topic."""


@dataclass(frozen=True)
class Delivery:
    """One message plus the handle needed to acknowledge it."""

    topic: Topic
    msg_id: str
    payload: Msg

    def typed(self, cls: type[M]) -> M:
        if not isinstance(self.payload, cls):
            raise SchemaMismatch(f"expected {cls.__name__}, got {type(self.payload).__name__}")
        return self.payload


class Bus(ABC):
    """Ordered, durable, replayable transport."""

    def _validate(self, topic: Topic, msg: Msg) -> None:
        expected = TOPIC_SCHEMA.get(topic)
        if expected is None:
            return  # audit.* topics accept any Msg
        if not isinstance(msg, expected):
            raise SchemaMismatch(
                f"topic {topic.value!r} carries {expected.__name__}, "
                f"got {type(msg).__name__}. Publish to the right topic or bump the schema."
            )

    @abstractmethod
    def publish(self, topic: Topic, msg: Msg) -> str: ...

    @abstractmethod
    def subscribe(
        self, topic: Topic, group: str, consumer: str, *, block_ms: int = 1000
    ) -> Iterator[Delivery]: ...

    @abstractmethod
    def ack(self, topic: Topic, group: str, msg_id: str) -> None: ...

    @abstractmethod
    def replay(self, topic: Topic, start: datetime, end: datetime) -> Iterator[Delivery]: ...

    @abstractmethod
    def pending_count(self, topic: Topic, group: str) -> int: ...


# --------------------------------------------------------------------------- #
# In-memory
# --------------------------------------------------------------------------- #
class InMemoryBus(Bus):
    """Deterministic bus for tests and the simulator.

    Delivery order is insertion order, always. That determinism is what makes
    ``test_golden_day`` able to assert a byte-identical output hash.
    """

    def __init__(self, max_len: int | None = None) -> None:
        self._streams: dict[Topic, list[tuple[str, Msg]]] = defaultdict(list)
        self._cursors: dict[tuple[Topic, str], int] = defaultdict(int)
        self._pending: dict[tuple[Topic, str], set[str]] = defaultdict(set)
        self._seq = 0
        self._max_len = max_len

    def publish(self, topic: Topic, msg: Msg) -> str:
        self._validate(topic, msg)
        self._seq += 1
        msg_id = f"{self._seq:012d}"
        stream = self._streams[topic]
        stream.append((msg_id, msg))
        if self._max_len is not None and len(stream) > self._max_len:
            if topic in DROPPABLE:
                del stream[: len(stream) - self._max_len]
                METRICS.inc("bus_dropped_total", topic=topic.value)
            else:
                raise BufferError(
                    f"topic {topic.value!r} exceeded max_len and is not droppable - "
                    "a slow consumer here means lost orders, not lost data points"
                )
        METRICS.inc("bus_published_total", topic=topic.value)
        return msg_id

    def subscribe(
        self, topic: Topic, group: str, consumer: str, *, block_ms: int = 1000
    ) -> Iterator[Delivery]:
        key = (topic, group)
        while True:
            idx = self._cursors[key]
            stream = self._streams[topic]
            if idx >= len(stream):
                return  # in-memory bus never blocks; it ends
            msg_id, payload = stream[idx]
            self._cursors[key] = idx + 1
            self._pending[key].add(msg_id)
            METRICS.inc("bus_consumed_total", topic=topic.value, group=group)
            yield Delivery(topic=topic, msg_id=msg_id, payload=payload)

    def ack(self, topic: Topic, group: str, msg_id: str) -> None:
        self._pending[(topic, group)].discard(msg_id)

    def replay(self, topic: Topic, start: datetime, end: datetime) -> Iterator[Delivery]:
        for msg_id, payload in self._streams[topic]:
            ts = getattr(payload, "ts", None) or getattr(payload, "t_event", None)
            if ts is not None and start <= ts < end:
                yield Delivery(topic=topic, msg_id=msg_id, payload=payload)

    def pending_count(self, topic: Topic, group: str) -> int:
        return len(self._pending[(topic, group)])

    # -- test helpers -------------------------------------------------------
    def reset_group(self, topic: Topic, group: str) -> None:
        """Rewind a consumer group. Simulates a consumer restart."""
        self._cursors[(topic, group)] = 0
        self._pending[(topic, group)].clear()

    def depth(self, topic: Topic) -> int:
        return len(self._streams[topic])


# --------------------------------------------------------------------------- #
# Redis Streams
# --------------------------------------------------------------------------- #
class RedisStreamsBus(Bus):
    """Production bus. ``redis`` is imported lazily so it stays an optional dep."""

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        maxlen: int = 1_000_000,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._r = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "RedisStreamsBus needs the infra extra: pip install -e '.[infra]'"
                ) from exc
            # decode_responses=True makes every reply str rather than bytes,
            # which is the contract the rest of this class is written against.
            # The annotation says so out loud: without it a checker widens every
            # read back to `bytes | str | Any` and the code below looks wrong
            # while being right.
            self._r = cast("Any", redis.Redis.from_url(url, decode_responses=True))
        self._maxlen = maxlen

    @staticmethod
    def _stream(topic: Topic) -> str:
        return f"ms:{topic.value}"

    def publish(self, topic: Topic, msg: Msg) -> str:
        self._validate(topic, msg)
        payload = {"t": type(msg).__name__, "d": msg.model_dump_json(by_alias=True)}
        msg_id: str = self._r.xadd(
            self._stream(topic), payload, maxlen=self._maxlen, approximate=True
        )
        METRICS.inc("bus_published_total", topic=topic.value)
        return msg_id

    def _decode(self, topic: Topic, raw: dict[str, str]) -> Msg:
        cls = TOPIC_SCHEMA.get(topic)
        if cls is None:
            raise SchemaMismatch(f"no schema registered for topic {topic.value!r}")
        try:
            return cls.model_validate(json.loads(raw["d"]))
        except (ValidationError, KeyError, json.JSONDecodeError) as exc:
            # Fail at the boundary. A malformed message that slips through here
            # surfaces three stages later pointing at an innocent module.
            METRICS.inc("bus_decode_error_total", topic=topic.value)
            raise SchemaMismatch(f"undecodable message on {topic.value!r}: {exc}") from exc

    def _ensure_group(self, topic: Topic, group: str) -> None:
        try:
            self._r.xgroup_create(self._stream(topic), group, id="0", mkstream=True)
        except Exception as exc:  # BUSYGROUP - already exists
            if "BUSYGROUP" not in str(exc):
                raise

    def subscribe(
        self, topic: Topic, group: str, consumer: str, *, block_ms: int = 1000
    ) -> Iterator[Delivery]:
        self._ensure_group(topic, group)
        stream = self._stream(topic)
        # ">" delivers new messages; "0" first drains anything this consumer was
        # given but never acknowledged, so a crash mid-processing loses nothing.
        pointer = "0"
        while True:
            resp = self._r.xreadgroup(group, consumer, {stream: pointer}, count=100, block=block_ms)
            if not resp:
                pointer = ">"
                continue
            for _stream_name, entries in resp:
                if not entries and pointer == "0":
                    pointer = ">"
                    continue
                for msg_id, raw in entries:
                    yield Delivery(topic=topic, msg_id=msg_id, payload=self._decode(topic, raw))

    def ack(self, topic: Topic, group: str, msg_id: str) -> None:
        self._r.xack(self._stream(topic), group, msg_id)

    def replay(self, topic: Topic, start: datetime, end: datetime) -> Iterator[Delivery]:
        lo = f"{int(start.timestamp() * 1000)}-0"
        hi = f"{int(end.timestamp() * 1000)}-0"
        for msg_id, raw in self._r.xrange(self._stream(topic), min=lo, max=hi):
            yield Delivery(topic=topic, msg_id=msg_id, payload=self._decode(topic, raw))

    def pending_count(self, topic: Topic, group: str) -> int:
        self._ensure_group(topic, group)
        info = self._r.xpending(self._stream(topic), group)
        return int(info["pending"]) if info else 0
