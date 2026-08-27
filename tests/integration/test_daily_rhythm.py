"""MA0 + MA1 - the daily rhythm, and the one job that cannot be skipped."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from libs.ms_core.clock import SimClock
from libs.ms_ops.rhythm import (
    EXCHANGE_TZ,
    RHYTHM,
    DayLog,
    JobKind,
    blocked_by,
    critical_jobs,
    jobs_for,
    missed_windows,
)
from libs.ms_ops.runner import Runner, next_job, render_schedule
from libs.ms_risk.alerts import Alerter, Channel

DAY = date(2026, 3, 4)


def ist(hour: int, minute: int = 0) -> datetime:
    """Exchange-local. The rhythm's times mean nothing in any other zone."""
    return datetime(2026, 3, 4, hour, minute, tzinfo=EXCHANGE_TZ)


def clock_at(hour: int, minute: int = 0) -> SimClock:
    # The clock reports UTC; the schedule is IST. Converting here rather than
    # comparing naive values is the whole point of the fix below.
    return SimClock(ist(hour, minute).astimezone(UTC))


def runner(
    *,
    at: tuple[int, int] = (23, 59),
    fails: set[JobKind] | None = None,
    open_positions: int = 0,
    sent: list | None = None,
) -> Runner:
    failing = fails or set()
    calls: list[JobKind] = []

    def make(kind: JobKind):
        def handler(day: date) -> None:
            calls.append(kind)
            if kind in failing:
                raise RuntimeError(f"{kind.value} blew up")

        return handler

    alerter = None
    if sent is not None:
        alerter = Alerter(send=lambda a, ch: sent.append((a.kind, ch)))

    r = Runner(
        clock=clock_at(*at),
        handlers={j.kind: make(j.kind) for j in RHYTHM},
        alerter=alerter,
        positions_open=lambda: open_positions,
    )
    r.calls = calls  # type: ignore[attr-defined]
    return r


# --------------------------------------------------------------- the schedule
def test_the_rhythm_matches_the_blueprint():
    times = {j.kind: j.at for j in RHYTHM}
    assert times[JobKind.BRIEFING] == time(6, 30)
    assert times[JobKind.PREFLIGHT] == time(8, 45)
    assert times[JobKind.SESSION] == time(9, 15)
    assert times[JobKind.FORCED_FLATTEN] == time(15, 20)
    assert times[JobKind.POST_SESSION] == time(15, 45)
    assert times[JobKind.NIGHTLY] == time(18, 0)


def test_a_holiday_runs_no_session_jobs():
    """is_trading_day comes from the calendar, not the weekday - a holiday is a
    Tuesday with no session and the weekday says nothing about it."""
    trading = {j.kind for j in jobs_for(DAY, is_trading_day=True)}
    closed = {j.kind for j in jobs_for(DAY, is_trading_day=False)}

    assert JobKind.SESSION in trading
    assert JobKind.SESSION not in closed
    assert JobKind.WEEKEND_RETRAIN in closed


def test_preflight_has_no_retries():
    """A retried preflight is a preflight nobody read: the first failure is the
    signal, and succeeding on the second attempt hides that the feed took two
    tries on the morning it later dropped mid-session."""
    preflight = next(j for j in RHYTHM if j.kind is JobKind.PREFLIGHT)
    assert preflight.retries == 0


def test_the_flatten_retries_because_the_broker_failing_is_expected():
    flatten = next(j for j in RHYTHM if j.kind is JobKind.FORCED_FLATTEN)
    assert flatten.retries >= 3


# ------------------------------------------------------------- the critical one
def test_only_the_forced_flatten_is_critical():
    assert [j.kind for j in critical_jobs()] == [JobKind.FORCED_FLATTEN]


def test_the_flatten_runs_even_when_everything_before_it_failed():
    """A broken preflight is a reason to hold no position, not a reason to keep
    one."""
    r = runner(fails={JobKind.PREFLIGHT, JobKind.BRIEFING})
    log = r.run_day(DAY)

    assert JobKind.PREFLIGHT in log.failed
    assert log.flatten_ran, "the flatten was skipped because an earlier job failed"


def test_the_flatten_waits_for_nothing():
    flatten = next(j for j in RHYTHM if j.kind is JobKind.FORCED_FLATTEN)
    assert blocked_by(flatten, completed=[]) == []


def test_a_failed_flatten_is_an_incident_not_a_failed_job():
    sent: list = []
    r = runner(fails={JobKind.FORCED_FLATTEN}, open_positions=2, sent=sent)
    log = r.run_day(DAY)

    assert log.incidents
    assert not log.clean
    assert ("risk_breach", Channel.SMS) in sent, "a failed flatten did not page anyone"


def test_the_flatten_cannot_be_removed_from_the_rhythm():
    """A flag that turns it off is a flag someone sets during a debugging
    session on a Tuesday."""
    for trading in (True, False):
        jobs = jobs_for(DAY, is_trading_day=trading)
        if trading:
            assert any(j.critical for j in jobs)


# ------------------------------------------------------------- missed windows
def test_a_missed_window_is_reported_not_silently_skipped():
    missed = missed_windows(DAY, now=ist(16, 0), completed=[])
    kinds = {j.kind for j in missed}
    assert JobKind.FORCED_FLATTEN in kinds
    assert JobKind.NIGHTLY not in kinds, "18:00 has not passed yet at 16:00"


def test_the_grace_period_is_respected():
    just_after = ist(15, 22)
    assert JobKind.FORCED_FLATTEN not in {
        j.kind for j in missed_windows(DAY, now=just_after, completed=[])
    }


def test_a_missed_flatten_with_positions_open_is_overnight_exposure():
    """A naive catch-up either reruns it into a shut market where the order will
    not fill, or skips it and reports a clean run while a position sits open."""
    sent: list = []
    r = runner(at=(16, 30), open_positions=3, sent=sent)
    log = r.catch_up(DAY)

    assert log.incidents
    assert "overnight exposure nobody chose" in log.incidents[0]
    assert ("risk_breach", Channel.SMS) in sent


def test_a_missed_flatten_with_a_flat_book_is_still_recorded():
    """A flatten that keeps not running is a fault even on the days it happens
    not to matter."""
    sent: list = []
    r = runner(at=(16, 30), open_positions=0, sent=sent)
    log = r.catch_up(DAY)

    assert not log.incidents, "an empty book was reported as exposure"
    assert JobKind.FORCED_FLATTEN in log.skipped
    assert any(kind == "system_health" for kind, _ in sent)


def test_catch_up_does_not_rerun_a_missed_non_critical_job():
    """Running the 06:30 briefing at 16:30 produces a pre-market briefing after
    the market closed."""
    r = runner(at=(16, 30))
    log = r.catch_up(DAY)

    assert JobKind.BRIEFING in log.skipped
    assert "not worth running late" in log.skipped[JobKind.BRIEFING]
    assert JobKind.BRIEFING not in r.calls  # type: ignore[attr-defined]


# --------------------------------------------------------------- dependencies
def test_post_session_waits_for_the_flatten_not_for_the_clock():
    """It computes outcomes for positions the flatten closed. Encoding the order
    as a time means a slow flatten produces attribution over positions that are
    still open, and the numbers look fine."""
    post = next(j for j in RHYTHM if j.kind is JobKind.POST_SESSION)
    assert JobKind.FORCED_FLATTEN in post.depends_on
    assert blocked_by(post, completed=[]) == [JobKind.FORCED_FLATTEN]
    assert blocked_by(post, completed=[JobKind.FORCED_FLATTEN]) == []


def test_the_session_will_not_start_without_a_preflight():
    r = runner(fails={JobKind.PREFLIGHT})
    log = r.run_day(DAY)

    assert JobKind.SESSION in log.skipped
    assert JobKind.SESSION not in r.calls  # type: ignore[attr-defined]


def test_a_skipped_job_is_not_counted_as_a_failure():
    """Counting it as a failure buries the one thing that actually broke under
    the six that followed it, and the post-mortem starts in the wrong place."""
    r = runner(fails={JobKind.PREFLIGHT})
    log = r.run_day(DAY)

    assert list(log.failed) == [JobKind.PREFLIGHT]
    assert JobKind.SESSION in log.skipped
    assert JobKind.SESSION not in log.failed


# ------------------------------------------------------------------- retries
def test_a_transient_failure_is_retried_within_its_budget():
    attempts = {"n": 0}

    def flaky(day: date) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("broker busy")

    r = Runner(clock=clock_at(23), handlers={JobKind.FORCED_FLATTEN: flaky})
    log = DayLog(day=DAY)
    flatten = next(j for j in RHYTHM if j.kind is JobKind.FORCED_FLATTEN)

    assert r.run_job(flatten, DAY, log) is True
    assert attempts["n"] == 3
    assert JobKind.FORCED_FLATTEN in log.completed


def test_a_job_with_no_handler_is_skipped_not_failed():
    r = Runner(clock=clock_at(23), handlers={})
    log = DayLog(day=DAY)
    briefing = next(j for j in RHYTHM if j.kind is JobKind.BRIEFING)

    assert r.run_job(briefing, DAY, log) is False
    assert JobKind.BRIEFING in log.skipped
    assert not log.failed


def test_a_failing_job_alerts_on_the_right_channel():
    """A briefing that fails is a system-health note. A flatten that fails is a
    page."""
    sent: list = []
    runner(fails={JobKind.BRIEFING}, sent=sent).run_day(DAY)
    kinds = {k for k, _ in sent}
    assert "system_health" in kinds
    assert "risk_breach" not in kinds


# ------------------------------------------------------------------ the report
def test_a_clean_day_says_so():
    log = runner().run_day(DAY)
    assert log.clean
    assert log.flatten_ran
    assert "FAILED" not in log.render()


def test_the_report_names_the_missing_flatten_explicitly():
    r = Runner(clock=clock_at(23), handlers={}, positions_open=lambda: 0)
    log = r.run_day(DAY)
    assert "carries a position overnight" in log.render()


def test_the_schedule_renders_for_a_human_to_check():
    text = render_schedule(DAY)
    assert "15:20" in text and "CRITICAL" in text
    assert "no retries" in text  # the preflight
    assert "after forced_flatten" in text


def test_next_job_walks_the_day():
    at_seven = ist(7, 0)
    nxt = next_job(DAY, now=at_seven, completed=[JobKind.BRIEFING])
    assert nxt is not None and nxt.kind is JobKind.PREFLIGHT

    after_close = ist(23, 0)
    assert next_job(DAY, now=after_close, completed=[]) is None


# ------------------------------------------------------------ the adapter
def test_the_rhythm_needs_no_workflow_engine():
    """The semantics live in plain Python. Swapping Prefect for cron, systemd or
    a while-loop must not change a single behaviour - and the suite proves it by
    never importing Prefect."""
    import sys

    runner().run_day(DAY)
    assert "prefect" not in sys.modules


def test_the_prefect_adapter_says_what_it_is_for_when_absent():
    from libs.ms_ops.runner import prefect_flow

    try:
        import prefect  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="runs without it"):
            prefect_flow(runner(), is_trading_day=lambda d: True)


def test_scheduled_times_are_exchange_local_and_aware():
    """A naive 15:20 compared against a UTC clock is a TypeError on a good day
    and a five-and-a-half-hour error on a bad one - firing the forced flatten at
    20:50 IST, hours after the market shut."""
    flatten = next(j for j in RHYTHM if j.kind is JobKind.FORCED_FLATTEN)
    at = flatten.scheduled_for(DAY)

    assert at.tzinfo is not None, "a naive schedule cannot be compared to a clock"
    assert at.utcoffset() == timedelta(hours=5, minutes=30)
    assert at.astimezone(UTC).hour == 9 and at.astimezone(UTC).minute == 50


def test_every_job_schedules_to_an_aware_instant():
    for job in RHYTHM:
        assert job.scheduled_for(DAY).tzinfo is not None, job.kind


def test_a_utc_clock_and_an_ist_schedule_compare_correctly():
    """The comparison that raised. 16:30 IST is 11:00 UTC, and the flatten at
    15:20 IST is behind both."""
    r = runner(at=(16, 30))
    assert r.clock.now().astimezone(UTC).hour == 11
    missed = missed_windows(DAY, now=r.clock.now(), completed=[])
    assert JobKind.FORCED_FLATTEN in {j.kind for j in missed}
