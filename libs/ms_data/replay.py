"""M18 - Replay engine.

Turns recorded history back into a stream that is indistinguishable from live.
This is the module that makes everything downstream testable: because the
simulator drives the *same* engine code through the *same* bus, a backtest
exercises the production code path rather than a parallel implementation of it.
A backtest that runs different code is not evidence about anything.

Four properties it must have, in order of how expensive they are to get wrong:

**Timestamp order across sources.** Ticks, chain snapshots and news are merged
into one stream sorted by time. A chain snapshot delivered before the tick that
preceded it lets a strategy see the future by a few hundred milliseconds - which
is exactly enough to look profitable and be worthless.

**The clock is set before the message is published.** A consumer that reads
``clock.now()`` while handling a message must see that message's timestamp, not
the previous one's.

**Availability, not occurrence.** Records are released at ``t_available`` - when
we could have acted - never at ``t_event``. The difference is the whole of
Chapter 3.

**Determinism.** Same input, same output, byte for byte. Enforced by
``tests/replay/test_determinism.py`` on every commit.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from libs.ms_core.bus import Bus, Topic
from libs.ms_core.clock import UTC, SessionCalendar, SimClock
from libs.ms_core.obs import METRICS, get_logger, trace
from libs.ms_core.schemas import (
    Candle,
    ChainSnapshot,
    Msg,
    RawNewsItem,
    Tick,
    Timed,
    deterministic_ids,
)
from libs.ms_data.recorder import DATA_ROOT, read_stream

log = get_logger(__name__)

# One replay item: when it becomes actionable, which topic, and the message.
ReplayItem = tuple[datetime, str, Topic, Msg]


class ReplaySource:
    """A source of recorded messages, yielded in timestamp order.

    Each source is responsible only for its own ordering; the engine merges them.
    ``name`` breaks ties deterministically when two sources produce the same
    instant, so two runs cannot interleave differently.
    """

    def __init__(self, name: str, topic: Topic) -> None:
        self.name = name
        self.topic = topic

    def items(self, day: date) -> Iterator[ReplayItem]:  # pragma: no cover - interface
        raise NotImplementedError


def release_time(msg: Msg) -> datetime | None:
    """When this message enters the replay stream, or None if it cannot.

    Market data is released at its observation timestamp; anything carrying the
    three timestamps is released at ``t_available``. News has no ``ts`` field at
    all, which is exactly the case an over-eager ``getattr(msg, "ts")`` drops on
    the floor without a word.
    """
    if isinstance(msg, Timed):
        return msg.t_available or msg.t_ingest
    if isinstance(msg, Candle):
        # A candle stamped 09:20 covers 09:20:00-09:21:00 and cannot be known
        # until 09:21:00. Releasing it at its own timestamp hands a consumer a
        # completed bar a full minute early - a leak large enough to make any
        # model built on candle features look excellent and be worthless.
        return msg.ts + timedelta(minutes=msg.tf)
    ts = getattr(msg, "ts", None)
    return ts if isinstance(ts, datetime) else None


class NdjsonSource(ReplaySource):
    """Replays one stream of one day straight from the recorder's own files."""

    def __init__(
        self,
        name: str,
        topic: Topic,
        cls: type[Msg],
        *,
        root: Path = DATA_ROOT,
        stream: str | None = None,
    ) -> None:
        super().__init__(name, topic)
        self._cls = cls
        self._root = root
        self._stream = stream or name

    def items(self, day: date) -> Iterator[ReplayItem]:
        partition = self._root / self._stream / f"date={day.isoformat()}"
        if not partition.exists():
            return
        rows: list[ReplayItem] = []
        for path in sorted(partition.glob("*.ndjson")):
            for msg in read_stream(path, self._cls):
                when = release_time(msg)
                if when is None:
                    # Not silently dropped: a message with no usable timestamp
                    # cannot be placed in the stream, and pretending otherwise
                    # would put it at an arbitrary point in the past.
                    log.warning(
                        "replay skipped a message with no usable timestamp",
                        fields={"stream": self._stream, "type": type(msg).__name__},
                    )
                    continue
                rows.append((when, self.name, self.topic, msg))
        # Files are per-instrument, so the concatenation is not globally sorted.
        # Sorting here rather than trusting the caller is cheap insurance against
        # the single most damaging class of replay bug.
        rows.sort(key=lambda r: (r[0], r[1], _tiebreak(r[3])))
        yield from rows


def _tiebreak(msg: Msg) -> tuple[int, int]:
    """Stable ordering for messages sharing an instant within one source."""
    instrument = getattr(msg, "instrument_id", None) or getattr(msg, "underlying_id", 0)
    seq = getattr(msg, "seq", None)
    return (int(instrument or 0), int(seq or 0))


@dataclass
class ReplayStats:
    published: dict[str, int] = field(default_factory=dict)
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    out_of_order: int = 0
    wall_seconds: float = 0.0

    @property
    def total(self) -> int:
        return sum(self.published.values())

    def summary(self) -> str:
        span = ""
        if self.first_ts and self.last_ts:
            span = f" {self.first_ts.isoformat()} .. {self.last_ts.isoformat()}"
        parts = ", ".join(f"{k}={v:,}" for k, v in sorted(self.published.items()))
        return f"{self.total:,} messages ({parts}){span}"


class ReplayEngine:
    """Drives a :class:`SimClock` and publishes recorded messages to a bus.

    ``speed`` controls pacing: ``inf`` runs as fast as the machine allows (what a
    backtest wants), ``1.0`` runs in real time (what a demo wants), ``10.0`` runs
    ten times faster than the market did.
    """

    def __init__(
        self,
        sources: Sequence[ReplaySource],
        clock: SimClock,
        bus: Bus,
        *,
        speed: float = float("inf"),
        calendar: SessionCalendar | None = None,
    ) -> None:
        if not sources:
            raise ValueError("replay needs at least one source")
        self._sources = list(sources)
        self._clock = clock
        self._bus = bus
        self._speed = speed
        self._cal = calendar

    def _merged(self, days: Sequence[date]) -> Iterator[ReplayItem]:
        """Merge every source into one timestamp-ordered stream.

        ``heapq.merge`` keeps memory bounded: sources stay as generators, so a
        month of ticks never has to be resident at once.
        """

        # Bind `source` per stream. A generator expression inside a list
        # comprehension captures the loop variable by reference, so the obvious
        # one-liner makes every stream replay the LAST source - and because the
        # streams still merge and still look ordered, the result is plausible
        # and completely wrong.
        def stream_for(src: ReplaySource) -> Iterator[ReplayItem]:
            for day in days:
                yield from src.items(day)

        streams = [stream_for(source) for source in self._sources]
        # Key on (availability, source name, message tiebreak) so two runs with
        # the same inputs interleave identically.
        yield from heapq.merge(*streams, key=lambda r: (r[0], r[1], _tiebreak(r[3])))

    def run(
        self,
        days: Sequence[date],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        on_message: Any | None = None,
    ) -> ReplayStats:
        """Replay ``days``, optionally clipped to ``[start, end)``."""
        stats = ReplayStats()
        started = time.monotonic()
        previous: datetime | None = None
        wall_anchor: float | None = None
        sim_anchor: datetime | None = None

        # Deterministic ids for the whole run: a byte-identical re-run is
        # impossible if every message carries a fresh uuid.
        with deterministic_ids(prefix="rp"):
            for ts, _source, topic, msg in self._merged(days):
                if start is not None and ts < start:
                    continue
                if end is not None and ts >= end:
                    break

                if previous is not None and ts < previous:
                    # Cannot happen with sorted sources; counted rather than
                    # silently accepted, because if it ever does the results
                    # downstream are worthless and nobody would otherwise notice.
                    stats.out_of_order += 1
                    METRICS.inc("replay_out_of_order_total")
                    continue

                if self._speed != float("inf"):
                    if wall_anchor is None or sim_anchor is None:
                        wall_anchor, sim_anchor = time.monotonic(), ts
                    else:
                        sim_elapsed = (ts - sim_anchor).total_seconds() / self._speed
                        lag = sim_elapsed - (time.monotonic() - wall_anchor)
                        if lag > 0:
                            time.sleep(lag)

                # The clock is set BEFORE publishing, so a consumer reading
                # clock.now() while handling this message sees its timestamp.
                self._clock.set(ts)
                with trace(msg.trace_id, stage="replay"):
                    self._bus.publish(topic, msg)
                    if on_message is not None:
                        on_message(topic, msg)

                stats.published[topic.value] = stats.published.get(topic.value, 0) + 1
                stats.first_ts = stats.first_ts or ts
                stats.last_ts = ts
                previous = ts
                METRICS.inc("replay_published_total", topic=topic.value)

        stats.wall_seconds = time.monotonic() - started
        log.info("replay finished", fields={"summary": stats.summary(), "speed": self._speed})
        return stats


# --------------------------------------------------------------------------- #
# Convenience construction
# --------------------------------------------------------------------------- #
def default_sources(root: Path = DATA_ROOT) -> list[ReplaySource]:
    """The streams the recorder writes, in the order a live session produces them."""
    return [
        NdjsonSource("ticks", Topic.MARKET_TICK, Tick, root=root),
        NdjsonSource("candles", Topic.MARKET_CANDLE, Candle, root=root),
        NdjsonSource("chain", Topic.MARKET_CHAIN, ChainSnapshot, root=root),
        NdjsonSource("news", Topic.RAW_NEWS, RawNewsItem, root=root),
    ]


def build_engine(
    bus: Bus,
    day: date,
    *,
    root: Path = DATA_ROOT,
    speed: float = float("inf"),
    calendar: SessionCalendar | None = None,
    sources: Sequence[ReplaySource] | None = None,
) -> tuple[ReplayEngine, SimClock]:
    """Engine primed at the first available message of ``day``."""
    srcs = list(sources) if sources else default_sources(root)
    first = first_timestamp(srcs, day)
    if first is None:
        raise FileNotFoundError(
            f"nothing recorded for {day}. Check data/raw/, or run: python tasks.py record"
        )
    clock = SimClock(first)
    return ReplayEngine(srcs, clock, bus, speed=speed, calendar=calendar), clock


def first_timestamp(sources: Iterable[ReplaySource], day: date) -> datetime | None:
    earliest: datetime | None = None
    for source in sources:
        for ts, _n, _t, _m in source.items(day):
            if earliest is None or ts < earliest:
                earliest = ts
            break  # sources are sorted, so the first item is the earliest
    return earliest


def recorded_days(root: Path = DATA_ROOT, stream: str = "ticks") -> list[date]:
    partition_root = root / stream
    if not partition_root.exists():
        return []
    out: list[date] = []
    for part in partition_root.glob("date=*"):
        try:
            out.append(date.fromisoformat(part.name.removeprefix("date=")))
        except ValueError:
            continue
    return sorted(out)


def session_window(calendar: SessionCalendar, day: date) -> tuple[datetime, datetime]:
    lo, hi = calendar.session_bounds(day)
    return lo, hi


def clip_to_minutes(start: datetime, minutes: int) -> tuple[datetime, datetime]:
    return start.astimezone(UTC), (start + timedelta(minutes=minutes)).astimezone(UTC)
