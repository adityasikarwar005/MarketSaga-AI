"""M11 - Market feed: protocol, health, gap detection, synthetic source.

The adapter is deliberately thin and the *protocol* is what the rest of the
system depends on. Swapping a broker later must not touch a single consumer.

Three things live here that are easy to skip and expensive to skip:

``GapDetector``
    Vendors drop messages. A gap that is detected is a small annotation on one
    minute of data; a gap that is not detected is a candle that silently lies.

``FeedHealth``
    Silence is ambiguous - a quiet market and a dead socket look identical from
    the outside. The heartbeat turns that ambiguity into a decision: after
    ``staleness_halt_seconds`` of silence during market hours, halt.

``SyntheticFeed``
    Deterministic, seeded, offline. It exists so the whole pipeline - recorder,
    candle builder, quality gates - can be built and tested before any broker
    credential exists, and so CI can run the pipeline end to end forever.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from libs.ms_core.clock import Clock, SessionCalendar, SessionState, SimClock
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Tick

log = get_logger(__name__)


class FeedState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    STALE = "STALE"
    HALTED = "HALTED"


# --------------------------------------------------------------------------- #
# Gap detection
# --------------------------------------------------------------------------- #
@dataclass
class Gap:
    instrument_id: int
    expected_seq: int
    received_seq: int
    ts: datetime

    @property
    def size(self) -> int:
        return self.received_seq - self.expected_seq


class GapDetector:
    """Tracks per-instrument sequence numbers and reports what went missing.

    Not every vendor supplies a sequence number. When it is absent this is a
    no-op rather than a fabricated guarantee - the data quality gates (M19) will
    then fall back to candle-count checks, which are weaker but honest.
    """

    def __init__(self) -> None:
        self._last: dict[int, int] = {}
        self.gaps: list[Gap] = []

    def check(self, tick: Tick) -> Gap | None:
        if tick.seq is None:
            return None
        prev = self._last.get(tick.instrument_id)
        self._last[tick.instrument_id] = tick.seq
        if prev is None or tick.seq == prev + 1:
            return None
        if tick.seq <= prev:
            # Out-of-order or duplicate delivery, not a gap.
            METRICS.inc("feed_out_of_order_total", instrument=str(tick.instrument_id))
            return None
        gap = Gap(tick.instrument_id, prev + 1, tick.seq, tick.ts)
        self.gaps.append(gap)
        METRICS.inc("feed_gap_total", instrument=str(tick.instrument_id))
        log.warning(
            "sequence gap",
            fields={
                "instrument_id": tick.instrument_id,
                "missing": gap.size,
                "expected": gap.expected_seq,
                "received": gap.received_seq,
            },
        )
        return gap

    def reset(self, instrument_id: int | None = None) -> None:
        """Call after a reconnect: the vendor may restart its sequence."""
        if instrument_id is None:
            self._last.clear()
        else:
            self._last.pop(instrument_id, None)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@dataclass
class FeedHealth:
    """Turns 'no messages' into a decision instead of a guess."""

    clock: Clock
    calendar: SessionCalendar
    staleness_halt_seconds: float
    last_message_at: datetime | None = None
    state: FeedState = FeedState.DISCONNECTED
    reconnects: int = 0

    def on_message(self) -> None:
        self.last_message_at = self.clock.now()
        if (
            self.state in (FeedState.STALE, FeedState.CONNECTING)
            or self.state is FeedState.DISCONNECTED
        ):
            self.state = FeedState.LIVE

    def on_reconnect(self) -> None:
        self.reconnects += 1
        self.state = FeedState.CONNECTING
        METRICS.inc("feed_reconnect_total")

    def silence_seconds(self) -> float | None:
        if self.last_message_at is None:
            return None
        return (self.clock.now() - self.last_message_at).total_seconds()

    def evaluate(self) -> FeedState:
        """Silence only means something while the market is open."""
        now = self.clock.now()
        try:
            open_now = self.calendar.state(now) is SessionState.OPEN
        except LookupError:
            # Calendar not loaded for this year - refuse to infer health rather
            # than assert a state we cannot justify.
            return self.state
        if not open_now:
            return self.state
        silence = self.silence_seconds()
        if silence is None:
            return self.state
        if silence > self.staleness_halt_seconds:
            if self.state is not FeedState.HALTED:
                log.error(
                    "feed stale during market hours - halting",
                    fields={"silence_s": round(silence, 2)},
                )
            self.state = FeedState.HALTED
        elif silence > self.staleness_halt_seconds / 2:
            self.state = FeedState.STALE
        METRICS.gauge("feed_silence_seconds", silence)
        return self.state

    @property
    def healthy(self) -> bool:
        return self.state is FeedState.LIVE


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
class MarketFeed(ABC):
    """What every market data source must provide. Consumers depend on this only."""

    @abstractmethod
    def subscribe(self, instrument_ids: Sequence[int]) -> None:
        """Register interest. May be called again to extend the subscription."""

    @abstractmethod
    def stream(self) -> Iterator[Tick]:
        """Yield ticks until :meth:`close` is called or the source ends."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    # Subscription capacity is finite and vendor-specific. Budget it explicitly
    # rather than discovering the ceiling when the option chain silently stops
    # updating halfway through an expiry morning.
    max_instruments: int = 3000


# --------------------------------------------------------------------------- #
# Synthetic feed
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticInstrument:
    instrument_id: int
    start_price: float
    annual_vol: float = 0.15
    tick_size: float = 0.05
    spread_ticks: int = 2


class SyntheticFeed(MarketFeed):
    """Deterministic, offline tick generator.

    Not a toy: it produces the microstructure the downstream code actually has
    to cope with - a two-sided book, a U-shaped intraday volatility profile,
    volume clustering, and optional injected gaps. That means the recorder,
    candle builder and quality gates can be finished and trusted before a broker
    account exists, and CI can exercise the full pipeline on every commit.

    Given the same seed it emits exactly the same ticks, which is what lets the
    determinism test assert a byte-identical result.
    """

    def __init__(
        self,
        instruments: Sequence[SyntheticInstrument],
        calendar: SessionCalendar,
        clock: SimClock,
        *,
        seed: int = 7,
        tick_interval_seconds: float = 1.0,
        gap_probability: float = 0.0,
    ) -> None:
        self._instruments = {i.instrument_id: i for i in instruments}
        self._calendar = calendar
        self._clock = clock
        self._rng = random.Random(seed)
        self._interval = tick_interval_seconds
        self._gap_probability = gap_probability
        self._subscribed: list[int] = []
        self._price: dict[int, float] = {i.instrument_id: i.start_price for i in instruments}
        self._seq: dict[int, int] = dict.fromkeys(self._price, 0)
        self._closed = False

    @property
    def name(self) -> str:
        return "synthetic"

    def subscribe(self, instrument_ids: Sequence[int]) -> None:
        unknown = [i for i in instrument_ids if i not in self._instruments]
        if unknown:
            raise KeyError(f"synthetic feed has no instruments {unknown}")
        if len(set(self._subscribed) | set(instrument_ids)) > self.max_instruments:
            raise ValueError(f"subscription exceeds {self.max_instruments} instruments")
        for i in instrument_ids:
            if i not in self._subscribed:
                self._subscribed.append(i)

    def close(self) -> None:
        self._closed = True

    # -- generation ---------------------------------------------------------
    def _vol_multiplier(self, ts: datetime) -> float:
        """U-shaped intraday volatility.

        This is not decoration. Time-of-day normalisation is one of the most
        important features in the whole model (blueprint Ch.8), and if the
        synthetic data were flat, every test of it would pass vacuously.
        """
        m = self._calendar.minutes_since_open(ts)
        total = self._calendar.expected_continuous_minutes
        if m < 0 or m > total:
            return 0.5
        x = m / total
        return 0.7 + 1.6 * (math.exp(-8 * x) + math.exp(-8 * (1 - x)))

    def _next_tick(self, instrument_id: int, ts: datetime) -> Tick:
        spec = self._instruments[instrument_id]
        per_tick_vol = spec.annual_vol / math.sqrt(252 * 375 * 60 / self._interval)
        shock = self._rng.gauss(0.0, 1.0) * per_tick_vol * self._vol_multiplier(ts)
        price = max(spec.tick_size, self._price[instrument_id] * math.exp(shock))
        price = round(price / spec.tick_size) * spec.tick_size
        self._price[instrument_id] = price

        half = spec.spread_ticks * spec.tick_size / 2
        self._seq[instrument_id] += 1
        if self._gap_probability and self._rng.random() < self._gap_probability:
            self._seq[instrument_id] += self._rng.randint(1, 4)  # injected gap

        return Tick(
            ts=ts,
            instrument_id=instrument_id,
            ltp=round(price, 2),
            qty=self._rng.randint(1, 50) * 25,
            bid=round(price - half, 2),
            ask=round(price + half, 2),
            bid_qty=self._rng.randint(1, 40) * 25,
            ask_qty=self._rng.randint(1, 40) * 25,
            seq=self._seq[instrument_id],
            src=0,
        )

    def stream(self) -> Iterator[Tick]:
        if not self._subscribed:
            raise RuntimeError("subscribe() before stream()")
        while not self._closed:
            ts = self._clock.now()
            if self._calendar.state(ts) is SessionState.OPEN:
                for iid in self._subscribed:
                    tick = self._next_tick(iid, ts)
                    METRICS.inc("feed_tick_total", source=self.name)
                    yield tick
            self._clock.advance(self._interval)

    def generate_session(self, day_start: datetime, day_end: datetime) -> list[Tick]:
        """Produce one bounded session of ticks. Convenience for tests and fixtures."""
        out: list[Tick] = []
        self._clock.set(day_start)
        for tick in self.stream():
            if tick.ts >= day_end:
                break
            out.append(tick)
            if self._clock.now() >= day_end:
                break
        return out


# --------------------------------------------------------------------------- #
# Subscription budgeting
# --------------------------------------------------------------------------- #
@dataclass
class SubscriptionPlan:
    """Explicit accounting of a finite resource.

    Vendors cap how many instruments one connection may carry. An option chain
    eats that budget fast - two indices, two expiries and a wide strike range
    runs into the hundreds. Planning it up front is the difference between a
    known trade-off and a chain that quietly stops updating at 09:40.
    """

    max_per_connection: int
    max_connections: int
    assignments: dict[int, list[int]] = field(default_factory=dict)

    @property
    def capacity(self) -> int:
        return self.max_per_connection * self.max_connections

    def plan(self, instrument_ids: Sequence[int]) -> dict[int, list[int]]:
        ids = list(dict.fromkeys(instrument_ids))  # de-dup, keep order
        if len(ids) > self.capacity:
            raise ValueError(
                f"{len(ids)} instruments requested but capacity is {self.capacity} "
                f"({self.max_connections} x {self.max_per_connection}). "
                "Narrow the strike range or drop an expiry - do not silently truncate."
            )
        self.assignments = {
            conn: ids[conn * self.max_per_connection : (conn + 1) * self.max_per_connection]
            for conn in range(math.ceil(len(ids) / self.max_per_connection) or 1)
        }
        return self.assignments

    def utilisation(self) -> float:
        used = sum(len(v) for v in self.assignments.values())
        return used / self.capacity if self.capacity else 0.0


def option_chain_token_estimate(n_underlyings: int, n_expiries: int, strikes_per_side: int) -> int:
    """Tokens needed for a chain: both option types, both sides of the money."""
    return n_underlyings * n_expiries * (2 * strikes_per_side) * 2


def default_session_window(calendar: SessionCalendar, day: datetime) -> tuple[datetime, datetime]:
    lo, hi = calendar.session_bounds(day.date())
    return lo, hi + timedelta(seconds=0)
