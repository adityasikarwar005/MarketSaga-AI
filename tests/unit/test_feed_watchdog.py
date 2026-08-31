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
    _start_feed_watchdog(feed, health, every=0.02)
    health.clock.advance(seconds=HALT_AFTER + 5)
    assert _settle(lambda: feed.closed), "the watchdog never closed a feed silent past its halt"


def test_a_live_feed_is_left_alone(health):
    """A watchdog that fires on a healthy feed is worse than none: it would turn
    every quiet minute of a real session into a lost recording."""
    feed = FakeFeed()
    _start_feed_watchdog(feed, health, every=0.02)
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
    _start_feed_watchdog(feed, health, every=0.02)
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
    _start_feed_watchdog(feed, health, every=0.02)
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
    _start_feed_watchdog(feed, health, every=0.02)
    clock.advance(seconds=HALT_AFTER * 5)
    time.sleep(0.2)
    assert not feed.closed
