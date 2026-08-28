"""M21 - the collector framework.

    pytest tests/unit/test_collectors.py -k "isolation or backoff or heartbeat"

The three names in the verify command are the three things that only matter when
a source misbehaves.

Isolation, because a loop that lets one exception escape stops every source
after it - and the symptom is that the *others* went quiet, which reads as a
network problem. Backoff, because one that never resets turns a courtesy into an
outage of our own. And the heartbeat, because a collector returning zero items
and no errors is indistinguishable from a working one on a quiet morning.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import FrozenClock, SimClock
from libs.ms_core.config import Source, SourcesConfig
from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.collectors import (
    BASE_BACKOFF,
    DEFAULT_STALE_AFTER,
    MAX_BACKOFF,
    Backoff,
    Collector,
    CollectorError,
    CollectorRunner,
    FailingCollector,
    Health,
    ReplayCollector,
)
from libs.ms_news.sources import RateLimit, Registry

T0 = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)


def item(source_id: int, headline: str = "something happened", ts: datetime = T0) -> RawNewsItem:
    return RawNewsItem(source_id=source_id, headline=headline, t_event=ts, t_ingest=ts)


def src(sid: int, *, enabled: bool = True) -> Source:
    return Source(
        id=sid,
        name=f"source-{sid}",
        source_class=f"class-{sid}",
        credibility=0.90,
        typical_lag_seconds=5,
        may_trigger_directional=True,
        tos_note="reviewed",
        enabled=enabled,
    )


def runner(*sids: int, enabled: bool = True, seed: int = 0) -> CollectorRunner:
    cfg = SourcesConfig(sources=[src(s, enabled=enabled) for s in sids])
    generous = {
        s: RateLimit(per_window=10_000, window=timedelta(hours=1), min_interval=timedelta(0))
        for s in sids
    }
    return CollectorRunner(
        bus=InMemoryBus(), registry=Registry.from_config(cfg, limits=generous), seed=seed
    )


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #
def test_isolation_one_failing_collector_does_not_stop_the_others():
    """The symptom of getting this wrong is that the *other* sources went quiet,
    which reads as a network problem and sends the investigation elsewhere."""
    r = runner(1, 2, 3)
    r.add(ReplayCollector(1, [[item(1)]]))
    r.add(FailingCollector(2, fail_times=99))
    r.add(ReplayCollector(3, [[item(3), item(3)]]))

    results = r.run_once(clock=FrozenClock(T0))

    assert [res.source_id for res in results] == [1, 2, 3]
    assert results[0].items == 1
    assert results[1].error and not results[1].ok
    assert results[2].items == 2, "source 3 was skipped because source 2 threw"


def test_isolation_holds_when_the_first_collector_is_the_broken_one():
    """Ordering matters: a loop that dies on the first item never reaches any."""
    r = runner(1, 2)
    r.add(FailingCollector(1, fail_times=99))
    r.add(ReplayCollector(2, [[item(2)]]))

    results = r.run_once(clock=FrozenClock(T0))
    assert results[1].items == 1


def test_isolation_records_the_failure_against_the_right_source():
    r = runner(1, 2)
    r.add(FailingCollector(1, fail_times=99))
    r.add(ReplayCollector(2, [[item(2)]]))
    r.run_once(clock=FrozenClock(T0))

    assert r.health[1].errors == 1
    assert r.health[2].errors == 0
    assert "having a morning" in r.health[1].last_error_text


def test_isolation_still_publishes_what_the_working_collectors_found():
    r = runner(1, 2)
    r.add(FailingCollector(1, fail_times=99))
    r.add(ReplayCollector(2, [[item(2), item(2)]]))
    r.run_once(clock=FrozenClock(T0))

    assert r.bus.depth(Topic.RAW_NEWS) == 2


def test_a_mislabelled_item_is_refused_rather_than_published():
    """An item stamped with the wrong source inherits the wrong credibility,
    and credibility is what decides whether it can open a position."""
    r = runner(1)
    r.add(ReplayCollector(1, [[item(99)]]))

    with pytest.raises(CollectorError, match="open a position"):
        r.run_once(clock=FrozenClock(T0))


def test_two_collectors_on_one_source_are_refused():
    """Double every item, and each copy looks like corroboration for the other."""
    r = runner(1)
    r.add(ReplayCollector(1, []))
    with pytest.raises(CollectorError, match="already has a collector"):
        r.add(ReplayCollector(1, []))


# --------------------------------------------------------------------------- #
# backoff
# --------------------------------------------------------------------------- #
def test_backoff_doubles_with_consecutive_failures():
    rng = random.Random(1)
    b = Backoff(jitter_share=0.0)

    intervals = []
    now = T0
    for _ in range(4):
        b.on_failure(now, rng)
        intervals.append(b.interval(rng))

    assert intervals[1] == intervals[0] * 2
    assert intervals[2] == intervals[0] * 4
    assert intervals == sorted(intervals)


def test_backoff_is_capped():
    rng = random.Random(1)
    b = Backoff(jitter_share=0.0)
    for _ in range(30):
        b.on_failure(T0, rng)
    assert b.interval(rng) == MAX_BACKOFF


def test_backoff_resets_on_success():
    """Without this a source that failed twice at breakfast is still throttled
    at noon, and the courtesy has become an outage of our own making."""
    rng = random.Random(1)
    b = Backoff()
    b.on_failure(T0, rng)
    b.on_failure(T0, rng)
    assert b.failures == 2
    assert not b.ready(T0)

    b.on_success()
    assert b.failures == 0
    assert b.ready(T0)


def test_backoff_resets_through_the_runner():
    r = runner(1)
    r.add(FailingCollector(1, fail_times=1, then=[item(1)]))
    clock = SimClock(T0)

    r.run_once(clock=clock)
    assert r.backoff[1].failures == 1

    clock.set(T0 + MAX_BACKOFF)
    r.run_once(clock=clock)
    assert r.backoff[1].failures == 0
    assert r.health[1].items == 1


def test_backoff_skips_the_poll_rather_than_making_it():
    """The point is not to touch the provider, not merely to record that we did."""
    r = runner(1)
    collector = FailingCollector(1, fail_times=99)
    r.add(collector)
    clock = SimClock(T0)

    r.run_once(clock=clock)
    calls_after_first = collector.calls

    clock.set(T0 + timedelta(seconds=1))
    results = r.run_once(clock=clock)

    assert collector.calls == calls_after_first, "the provider was polled while backing off"
    assert results[0].skipped == "backing off"


def test_backoff_is_jittered_so_collectors_do_not_return_in_lockstep():
    """N collectors that failed during one provider outage returning the instant
    it ends re-create the load that caused it."""
    rng = random.Random(7)
    b = Backoff()
    b.failures = 3

    draws = {b.interval(rng).total_seconds() for _ in range(20)}
    assert len(draws) > 15, "the interval never varied"

    raw = (BASE_BACKOFF * 4).total_seconds()
    assert all(abs(d - raw) <= raw * 0.26 for d in draws)


def test_the_jitter_is_reproducible_from_a_seed():
    """A test that cannot reproduce a backoff schedule cannot check one."""
    a = CollectorRunner(bus=InMemoryBus(), registry=Registry(), seed=42)
    b = CollectorRunner(bus=InMemoryBus(), registry=Registry(), seed=42)
    assert [a._rng.random() for _ in range(5)] == [b._rng.random() for _ in range(5)]


def test_backoff_is_per_source():
    r = runner(1, 2)
    r.add(FailingCollector(1, fail_times=99))
    r.add(ReplayCollector(2, [[item(2)], [item(2)]]))
    clock = SimClock(T0)

    r.run_once(clock=clock)
    clock.set(T0 + timedelta(seconds=1))
    results = r.run_once(clock=clock)

    assert results[0].skipped == "backing off"
    assert results[1].items == 1, "a healthy source was throttled by another's failure"


# --------------------------------------------------------------------------- #
# heartbeat
# --------------------------------------------------------------------------- #
def test_heartbeat_measures_silence_since_the_last_success():
    """Since success, not since the last attempt. A collector erroring every
    thirty seconds is not silent; one succeeding-but-empty is, and it is the
    second that looks healthy."""
    h = Health(source_id=1)
    assert h.silence(T0) is None

    h.last_success = T0
    assert h.silence(T0 + timedelta(minutes=5)) == timedelta(minutes=5)


def test_heartbeat_treats_a_collector_that_never_succeeded_as_stale():
    """Not "unknown". A source that has never worked is the loudest case."""
    assert Health(source_id=1).is_stale(T0)


def test_heartbeat_does_not_reset_on_a_failed_poll():
    r = runner(1)
    r.add(FailingCollector(1, fail_times=99))
    clock = SimClock(T0)
    r.run_once(clock=clock)

    assert r.health[1].last_success is None
    assert r.health[1].is_stale(T0)


def test_an_empty_but_successful_poll_counts_as_alive():
    """Most polls of most sources find nothing. Treating empty as failure would
    put every quiet source into backoff."""
    r = runner(1)
    r.add(ReplayCollector(1, [[]]))
    r.run_once(clock=FrozenClock(T0))

    assert r.health[1].items == 0
    assert r.health[1].last_success == T0
    assert not r.health[1].is_stale(T0)
    assert r.backoff[1].failures == 0


def test_heartbeat_reports_staleness_as_a_number_not_a_halt():
    """Forty minutes of silence from an exchange feed at 10 a.m. is an outage;
    the same forty minutes at 3 a.m. is a Sunday. The framework cannot tell."""
    r = runner(1)
    r.add(ReplayCollector(1, [[item(1)]]))
    clock = SimClock(T0)
    r.run_once(clock=clock)

    clock.set(T0 + DEFAULT_STALE_AFTER + timedelta(minutes=1))
    assert r.stale(clock=clock) == [1]
    text = r.render(clock=clock)
    assert "not an error" in text and "for a human, not a halt" in text


def test_the_health_line_carries_the_per_source_metrics():
    """items/min, error rate, last-success age - the four M21 asks for."""
    r = runner(1)
    r.add(FailingCollector(1, fail_times=1, then=[item(1), item(1)]))
    clock = SimClock(T0)
    r.run_once(clock=clock)
    clock.set(T0 + MAX_BACKOFF)
    r.run_once(clock=clock)

    line = r.health[1].render(clock.now())
    assert "polls=2" in line
    assert "items=2" in line
    assert "err=" in line
    assert "/min" in line
    assert "last ok" in line


def test_items_per_minute_is_measured_from_the_first_poll():
    h = Health(source_id=1, items=30, first_poll=T0)
    assert h.items_per_minute(T0 + timedelta(minutes=10)) == pytest.approx(3.0)
    assert Health(source_id=1).items_per_minute(T0) == 0.0


# --------------------------------------------------------------------------- #
# the registry gates the poll
# --------------------------------------------------------------------------- #
def test_a_disabled_source_is_not_polled():
    r = runner(1, enabled=False)
    collector = ReplayCollector(1, [[item(1)]])
    r.add(collector)

    results = r.run_once(clock=FrozenClock(T0))
    assert results[0].skipped == "disabled or out of budget"
    assert r.health[1].polls == 0


def test_an_exhausted_budget_stops_the_poll():
    cfg = SourcesConfig(sources=[src(1)])
    tight = {1: RateLimit(per_window=1, window=timedelta(hours=1))}
    r = CollectorRunner(bus=InMemoryBus(), registry=Registry.from_config(cfg, limits=tight))
    r.add(ReplayCollector(1, [[item(1)], [item(1)]]))
    clock = FrozenClock(T0)

    assert r.run_once(clock=clock)[0].items == 1
    assert r.run_once(clock=clock)[0].skipped == "disabled or out of budget"


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
def test_stream_defaults_to_a_poll():
    """A source with no streaming endpoint satisfies the interface without a
    collector having to implement a fake one."""
    c = ReplayCollector(1, [[item(1), item(1)]])
    assert len(list(c.stream(clock=FrozenClock(T0)))) == 2


def test_a_collector_must_implement_poll():
    with pytest.raises(TypeError):

        class Incomplete(Collector):  # type: ignore[abstract]
            pass

        Incomplete(1)  # type: ignore[abstract]


def test_the_failing_collector_ships_in_the_library_not_the_tests():
    """Same reasoning as the kill drill's stubborn broker: a drill against a
    component that always works proves the drill runs, not that the handling
    does."""
    import libs.ms_news.collectors as mod

    assert mod.FailingCollector.__module__ == "libs.ms_news.collectors"
    assert mod.FailingCollector.__doc__ is not None
    assert "always works" in mod.FailingCollector.__doc__
