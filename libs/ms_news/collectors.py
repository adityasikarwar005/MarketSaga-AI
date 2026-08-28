"""M21 - the collector framework: one per source, and none of them can take the others down.

A collector is a small thing that fetches from somebody else's server and puts
`RawNewsItem` on the bus. The framework around it exists for three reasons, and
all three are about what happens when a source misbehaves rather than when it
works.

Isolation is the whole point
------------------------------

One collector per source, and a failure in one **never** affects another. That
sounds obvious and is easy to lose: a single loop that calls each collector in
turn and lets an exception escape stops every source after the one that threw.
The morning that happens is the morning one provider changes their HTML, and the
symptom is that the *other five* went quiet - which reads as a network problem
and sends the investigation somewhere else entirely.

:func:`run_once` catches per collector, records the failure against that source,
and carries on. A collector that raises is a broken collector, not a broken
pipeline.

Backoff is per source and it decays
-------------------------------------

A source that is failing gets polled less often, and the interval doubles up to
a ceiling. Two details matter more than the doubling.

It must **reset on success**, or a source that failed twice at breakfast is
still on a five-minute interval at noon - and the backoff, meant to be a
courtesy, has quietly become an outage of our own making.

And it needs jitter. Without it, N collectors that all failed during the same
provider outage come back in lockstep the moment it ends and re-create the load
that caused it. The jitter is deterministic given a seed, because a test that
cannot reproduce a backoff schedule cannot check one.

A heartbeat is about silence, not errors
------------------------------------------

The dangerous state is not a collector reporting failures - that is visible. It
is one that returns zero items and no errors, which is exactly what a working
collector looks like on a quiet morning. The two are indistinguishable from the
inside, so :class:`Health` reports **age since last success** and lets the
operator decide: forty minutes of silence from an exchange feed at 10 a.m. is an
outage, and the same forty minutes at 3 a.m. is a Sunday.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from libs.ms_core.bus import Bus, Topic
from libs.ms_core.clock import Clock
from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.sources import Registry

log = get_logger(__name__)

BASE_BACKOFF = timedelta(seconds=15)
MAX_BACKOFF = timedelta(minutes=10)
JITTER_SHARE = 0.25  # +/- this fraction of the interval

# How long a source may be silent before the heartbeat calls it stale. Not a
# failure on its own - see the module docstring - but the number an operator
# needs in front of them.
DEFAULT_STALE_AFTER = timedelta(minutes=20)


class CollectorError(RuntimeError):
    """Raised by a collector. Caught by the framework, never by the caller."""


class Collector(ABC):
    """One source. Fetches, and knows nothing about any other collector."""

    def __init__(self, source_id: int, *, name: str = "") -> None:
        self.source_id = source_id
        self.name = name or f"collector-{source_id}"

    @abstractmethod
    def poll(self, *, clock: Clock) -> Sequence[RawNewsItem]:
        """Fetch whatever is new. May raise; the framework will handle it.

        Returning an empty sequence is normal and is not an error - most polls
        of most sources find nothing, which is why silence needs its own signal
        rather than being inferred from the error count.
        """

    def stream(self, *, clock: Clock) -> Iterator[RawNewsItem]:
        """Long-lived push subscription, for sources that offer one.

        Defaults to a single poll so that a source with no streaming endpoint
        satisfies the interface without a collector having to implement a fake
        one, and so the scheduler can treat both kinds identically.
        """
        yield from self.poll(clock=clock)


# --------------------------------------------------------------------------- #
# Per-source state
# --------------------------------------------------------------------------- #
@dataclass
class Health:
    """What one collector has been doing, and how long since it last worked."""

    source_id: int
    polls: int = 0
    items: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_success: datetime | None = None
    last_error: datetime | None = None
    last_error_text: str = ""
    first_poll: datetime | None = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.polls if self.polls else 0.0

    def items_per_minute(self, now: datetime) -> float:
        if self.first_poll is None:
            return 0.0
        minutes = (now - self.first_poll).total_seconds() / 60.0
        return self.items / minutes if minutes > 0 else 0.0

    def silence(self, now: datetime) -> timedelta | None:
        """Age since the last *successful* poll, or None if there never was one.

        Since success, not since the last attempt. A collector erroring every
        thirty seconds is not silent, and one succeeding-but-empty is - and it
        is the second that looks healthy.
        """
        return None if self.last_success is None else now - self.last_success

    def is_stale(self, now: datetime, *, after: timedelta = DEFAULT_STALE_AFTER) -> bool:
        gap = self.silence(now)
        return True if gap is None else gap > after

    def render(self, now: datetime) -> str:
        gap = self.silence(now)
        quiet = "never succeeded" if gap is None else f"{gap.total_seconds() / 60:.0f}m ago"
        flag = "  STALE" if self.is_stale(now) else ""
        return (
            f"    source {self.source_id:<4} polls={self.polls:<5} items={self.items:<6} "
            f"err={self.error_rate:>5.1%}  {self.items_per_minute(now):>6.2f}/min  "
            f"last ok {quiet}{flag}"
        )


@dataclass
class Backoff:
    """Per source, doubling, capped, jittered, and reset by success."""

    base: timedelta = BASE_BACKOFF
    ceiling: timedelta = MAX_BACKOFF
    jitter_share: float = JITTER_SHARE
    failures: int = 0
    next_allowed: datetime | None = None

    def interval(self, rng: random.Random) -> timedelta:
        if self.failures <= 0:
            return timedelta(0)
        raw = min(self.base * (2 ** (self.failures - 1)), self.ceiling)
        # Jitter, so N collectors that failed during one provider outage do not
        # all return the instant it ends and re-create the load that caused it.
        spread = raw.total_seconds() * self.jitter_share
        return timedelta(seconds=raw.total_seconds() + rng.uniform(-spread, spread))

    def on_failure(self, now: datetime, rng: random.Random) -> None:
        self.failures += 1
        self.next_allowed = now + self.interval(rng)

    def on_success(self) -> None:
        """Reset. Without this a source that failed twice at breakfast is still
        throttled at noon, and the backoff has become an outage of our own."""
        self.failures = 0
        self.next_allowed = None

    def ready(self, now: datetime) -> bool:
        return self.next_allowed is None or now >= self.next_allowed


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
@dataclass
class PollResult:
    source_id: int
    items: int = 0
    error: str = ""
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.skipped


@dataclass
class CollectorRunner:
    """Every collector, run in isolation, publishing to ``raw.news``."""

    bus: Bus
    registry: Registry
    collectors: dict[int, Collector] = field(default_factory=dict)
    health: dict[int, Health] = field(default_factory=dict)
    backoff: dict[int, Backoff] = field(default_factory=dict)
    seed: int = 0
    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, collector: Collector) -> Collector:
        if collector.source_id in self.collectors:
            raise CollectorError(
                f"source {collector.source_id} already has a collector. Two collectors "
                "on one source double every item and each looks like corroboration "
                "for the other."
            )
        self.collectors[collector.source_id] = collector
        self.health[collector.source_id] = Health(source_id=collector.source_id)
        self.backoff[collector.source_id] = Backoff()
        return collector

    def run_once(self, *, clock: Clock) -> list[PollResult]:
        """Poll every collector. One failing never stops another.

        The exception handler is the feature. A single loop that lets an error
        escape stops every source after the one that threw, and the symptom is
        that the *others* went quiet - which reads as a network problem and
        sends the investigation somewhere else.
        """
        now = clock.now()
        results: list[PollResult] = []

        for source_id, collector in sorted(self.collectors.items()):
            health = self.health[source_id]
            backoff = self.backoff[source_id]

            if not backoff.ready(now):
                results.append(PollResult(source_id, skipped="backing off"))
                continue
            if not self.registry.may_poll(source_id, clock=clock):
                results.append(PollResult(source_id, skipped="disabled or out of budget"))
                continue

            health.polls += 1
            if health.first_poll is None:
                health.first_poll = now

            try:
                items = list(collector.poll(clock=clock))
            except Exception as exc:
                health.errors += 1
                health.consecutive_errors += 1
                health.last_error = now
                health.last_error_text = f"{type(exc).__name__}: {exc}"
                backoff.on_failure(now, self._rng)
                log.info(
                    "collector failed",
                    fields={"source_id": source_id, "error": health.last_error_text[:120]},
                )
                results.append(PollResult(source_id, error=health.last_error_text))
                continue

            published = 0
            for item in items:
                if item.source_id != source_id:
                    raise CollectorError(
                        f"collector for source {source_id} emitted an item stamped "
                        f"{item.source_id}. A mislabelled item inherits the wrong "
                        "credibility, and credibility is what decides whether it can "
                        "open a position."
                    )
                self.bus.publish(Topic.RAW_NEWS, item)
                published += 1

            health.items += published
            health.consecutive_errors = 0
            health.last_success = now
            backoff.on_success()
            results.append(PollResult(source_id, items=published))

        return results

    def stale(self, *, clock: Clock, after: timedelta = DEFAULT_STALE_AFTER) -> list[int]:
        now = clock.now()
        return sorted(sid for sid, h in self.health.items() if h.is_stale(now, after=after))

    def render(self, *, clock: Clock, after: timedelta = DEFAULT_STALE_AFTER) -> str:
        now = clock.now()
        lines = [f"  collectors: {len(self.collectors)}"]
        lines += [self.health[sid].render(now) for sid in sorted(self.health)]

        stale = self.stale(clock=clock, after=after)
        if stale:
            lines.append(
                f"    {len(stale)} source(s) silent for over {after}: {stale}. Silence is "
                "not an error - a working collector on a quiet morning looks the same "
                "from the inside - so this is a number for a human, not a halt."
            )
        backing_off = [sid for sid, b in self.backoff.items() if b.failures > 0]
        if backing_off:
            lines.append(f"    backing off: {sorted(backing_off)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# A collector that needs no network
# --------------------------------------------------------------------------- #
class ReplayCollector(Collector):
    """Serves pre-built items, for tests and for replaying a recorded day.

    Exists so the framework can be exercised end to end without a network, and
    so a recorded day can be pushed through the real pipeline rather than
    through a second one written for testing - which is the arrangement where
    the tested path and the live path drift apart.
    """

    def __init__(
        self, source_id: int, batches: Sequence[Sequence[RawNewsItem]], *, name: str = ""
    ) -> None:
        super().__init__(source_id, name=name or f"replay-{source_id}")
        self._batches = list(batches)
        self._at = 0

    def poll(self, *, clock: Clock) -> Sequence[RawNewsItem]:
        if self._at >= len(self._batches):
            return []
        batch = self._batches[self._at]
        self._at += 1
        return batch


class FailingCollector(Collector):
    """Fails on purpose, for the isolation and backoff tests.

    In the library rather than in the test file because a drill against a
    component that always works proves the drill runs, not that the handling
    does - the same reason the kill drill ships with a broker that fights back.
    """

    def __init__(self, source_id: int, *, fail_times: int = 1, then: Sequence[RawNewsItem] = ()):
        super().__init__(source_id, name=f"failing-{source_id}")
        self.fail_times = fail_times
        self._then = list(then)
        self.calls = 0

    def poll(self, *, clock: Clock) -> Sequence[RawNewsItem]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise CollectorError(f"source {self.source_id} is having a morning")
        return self._then
