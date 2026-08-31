"""The feed watchdog - health evaluated on a timer, not on tick arrival.

`health.evaluate()` used to be called from inside the ingest loop, once every
5,000 ticks. That is the one place it cannot work: the condition it detects is
*no ticks arriving*, and with none arriving the counter never advances.

On the first live session the socket dropped at 10:02, the vendor library
retried until 10:08 and gave up, and the recorder then sat in its poll loop
drawing no CPU for twenty-four minutes while the market traded. The staleness
halt was configured, correct, and unreachable.
"""

from __future__ import annotations

import time

import pytest

from libs.ms_core.clock import SimClock
from libs.ms_data.feed import FeedHealth, FeedState
from services.ingest_market.__main__ import _start_feed_watchdog
from tests.conftest import ist

HALT_AFTER = 30.0


class FakeFeed:
    """Only the part of the feed protocol the watchdog touches."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def health(calendar):
    clock = SimClock(ist(11, 0))
    h = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=HALT_AFTER)
    h.on_message()
    return h


def _settle(predicate, *, timeout: float = 3.0) -> bool:
    """Wait briefly for the watchdog thread to act."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_silent_feed_is_closed_even_though_no_tick_ever_arrives(health):
    """The whole point. Nothing calls into the watchdog; it has to notice alone."""
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=health.clock, every=0.02)
    health.clock.advance(seconds=HALT_AFTER + 5)
    assert _settle(lambda: feed.closed), "the watchdog never closed a feed silent past its halt"


def test_a_live_feed_is_left_alone(health):
    """A watchdog that fires on a healthy feed is worse than none: it would turn
    every quiet minute of a real session into a lost recording."""
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=health.clock, every=0.02)
    for _ in range(6):
        health.clock.advance(seconds=HALT_AFTER / 3)
        health.on_message()
        time.sleep(0.05)
    assert not feed.closed


def test_it_closes_the_feed_rather_than_setting_a_stop_flag(health):
    """`_stop` is only read inside the ingest loop's body, which a silent feed
    never enters - so a flag would be set and never noticed. Closing the feed
    ends the generator itself, which is what lets the `finally` block run and
    the session's buffers reach the disk."""
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=health.clock, every=0.02)
    health.clock.advance(seconds=HALT_AFTER + 5)
    assert _settle(lambda: feed.closed)
    assert health.evaluate() is FeedState.HALTED


def test_silence_outside_the_session_is_not_a_fault(calendar):
    """Before the open and after the close there is nothing to be silent about,
    and halting then would close a recording that had not started."""
    clock = SimClock(ist(7, 30))
    health = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=HALT_AFTER)
    health.on_message()
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=clock, every=0.02)
    clock.advance(seconds=HALT_AFTER * 10)
    time.sleep(0.2)
    assert not feed.closed


def test_a_feed_that_never_delivered_anything_is_not_halted(calendar):
    """No first message means no silence to measure. That is the state at
    startup, and halting there would kill every session before it began."""
    clock = SimClock(ist(11, 0))
    health = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=HALT_AFTER)
    assert health.silence_seconds() is None
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=clock, every=0.02)
    clock.advance(seconds=HALT_AFTER * 5)
    time.sleep(0.2)
    assert not feed.closed


# ------------------------------------------- the chain has its own heartbeat
class FakeChainBuilder:
    """Only the two properties the watchdog reads."""

    snapshot_seconds = 5.0

    def __init__(self, last=None) -> None:
        self.last_snapshot_at = last


def test_a_feed_alive_in_aggregate_but_dead_in_the_underlying_is_caught(health):
    """The second stall of 2026-08-31, and the one feed health could not see.

    The socket dropped at 10:58, the vendor library kept retrying, and 1,864
    ticks trickled through over the next twenty-one minutes - enough to keep
    `on_message` fresh past a three-second threshold most of the time. The index
    had stopped, so not one chain snapshot was produced in those twenty-one
    minutes. Aggregate liveness said the feed was fine; the recording was dead.
    """
    feed = FakeFeed()
    chain = FakeChainBuilder(last=health.clock.now())
    _start_feed_watchdog(feed, health, clock=health.clock, chain_builder=chain, every=0.02)

    # A trickle of messages keeps feed health satisfied throughout.
    for _ in range(8):
        health.clock.advance(seconds=1)
        health.on_message()
        time.sleep(0.03)
    assert health.evaluate() is not FeedState.HALTED, "feed health should still be content"
    assert not feed.closed, "nothing is wrong yet"

    health.clock.advance(seconds=12 * FakeChainBuilder.snapshot_seconds + 5)
    health.on_message()  # the trickle continues, and must not save it
    assert _settle(lambda: feed.closed), "a chain that stopped advancing was not caught"


def test_a_chain_keeping_its_cadence_is_left_alone(health):
    feed = FakeFeed()
    chain = FakeChainBuilder(last=health.clock.now())
    _start_feed_watchdog(feed, health, clock=health.clock, chain_builder=chain, every=0.02)
    for _ in range(6):
        health.clock.advance(seconds=FakeChainBuilder.snapshot_seconds)
        chain.last_snapshot_at = health.clock.now()
        health.on_message()
        time.sleep(0.04)
    assert not feed.closed


def test_a_session_with_no_snapshot_yet_is_not_halted(health):
    """Before the first snapshot there is no cadence to have missed. Halting
    there would kill every session in its opening seconds."""
    feed = FakeFeed()
    _start_feed_watchdog(
        feed, health, clock=health.clock, chain_builder=FakeChainBuilder(last=None), every=0.02
    )
    health.clock.advance(seconds=12 * FakeChainBuilder.snapshot_seconds + 30)
    health.on_message()
    time.sleep(0.2)
    assert not feed.closed


def test_a_run_without_a_chain_still_watches_the_feed(health):
    """`--chain` is optional for other sources. Losing the chain check must not
    lose the staleness check with it."""
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, clock=health.clock, chain_builder=None, every=0.02)
    health.clock.advance(seconds=HALT_AFTER + 5)
    assert _settle(lambda: feed.closed)
