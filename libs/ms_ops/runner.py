"""MA0 - orchestration.

The plan says "Prefect flows for nightly/weekly jobs; retries; alerting on
failure", and this deviates from it deliberately, with a reason worth stating.

The part of orchestration that has correctness requirements is **which job runs,
after what, and what happens when one is missed** - and a missed forced flatten
carries a position overnight. That logic must be testable without standing up a
workflow engine, must run identically under replay, and must not change behaviour
when the engine is upgraded.

So the semantics live here, in plain Python with plain tests, and Prefect becomes
a deployment adapter that calls :meth:`Runner.run_day` on a schedule. The
workflow engine gets the job it is good at - cron, UI, history - and does not get
the job of deciding whether it is safe to skip a flatten.

Retries are per job and bounded
-------------------------------

Preflight has **zero** retries, on purpose. A retried preflight is a preflight
nobody read: the first failure is the signal, and quietly succeeding on the
second attempt hides that the feed took two tries to connect on the morning it
later dropped mid-session.

The forced flatten has three, because there the retry is the point - the broker
returning an error is the expected case, not the exceptional one.

Failure does not cascade silently
---------------------------------

A job whose dependency failed is **skipped with a reason**, not failed. Counting
it as a failure buries the one thing that actually broke under the six that
followed it, and the post-mortem then starts in the wrong place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from libs.ms_core.clock import Clock
from libs.ms_core.obs import get_logger
from libs.ms_ops.rhythm import DEFAULT_GRACE, DayLog, Job, JobKind, blocked_by, jobs_for
from libs.ms_risk.alerts import Alerter, build

log = get_logger(__name__)

# A callable that does the work and raises on failure. Returning a bool would
# make "it failed" and "it returned False" the same thing to the runner.
JobFn = Callable[[date], None]


@dataclass
class Runner:
    """Runs a day's rhythm, retries what is worth retrying, alerts on the rest.

    ``positions_open`` is injected because the runner has to answer one question
    it cannot answer itself: after a missed flatten, is anything actually open?
    Assuming yes cries wolf every weekend; assuming no is how a position spends
    the night unnoticed.
    """

    clock: Clock
    handlers: dict[JobKind, JobFn]
    alerter: Alerter | None = None
    positions_open: Callable[[], int] = lambda: 0
    grace: timedelta = DEFAULT_GRACE
    logs: dict[date, DayLog] = field(default_factory=dict)

    # ------------------------------------------------------------------ alerts
    def _alert(self, kind: str, message: str) -> None:
        if self.alerter is None:
            return
        self.alerter.emit(build(kind, message, ts=self.clock.now()))

    # -------------------------------------------------------------------- one
    def run_job(self, job: Job, day: date, log_: DayLog) -> bool:
        """Attempt one job, honouring its own retry budget."""
        blocked = blocked_by(job, completed=log_.completed)
        if blocked:
            reason = f"waiting on {', '.join(b.value for b in blocked)}"
            log_.record_skip(job.kind, reason)
            log.info("job skipped", fields={"job": job.kind.value, "reason": reason})
            return False

        handler = self.handlers.get(job.kind)
        if handler is None:
            log_.record_skip(job.kind, "no handler registered")
            return False

        last_error = ""
        for attempt in range(job.retries + 1):
            try:
                handler(day)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.info(
                    "job failed",
                    fields={
                        "job": job.kind.value,
                        "attempt": attempt + 1,
                        "of": job.retries + 1,
                        "error": last_error,
                    },
                )
            else:
                log_.record_success(job.kind)
                return True

        log_.record_failure(job.kind, last_error)
        if job.critical:
            log_.incidents.append(
                f"{job.kind.value} failed after {job.retries + 1} attempts: {last_error}"
            )
            self._alert(
                "risk_breach",
                f"CRITICAL job {job.kind.value} failed on {day}: {last_error}",
            )
        else:
            self._alert("system_health", f"{job.kind.value} failed on {day}: {last_error}")
        return False

    # -------------------------------------------------------------------- day
    def run_day(self, day: date, *, is_trading_day: bool = True) -> DayLog:
        """Run everything scheduled for the day, in order."""
        log_ = self.logs.setdefault(day, DayLog(day=day))
        for job in jobs_for(day, is_trading_day=is_trading_day):
            self.run_job(job, day, log_)

        if is_trading_day and not log_.flatten_ran:
            self._raise_flatten_incident(day, log_, "the forced flatten did not complete")
        return log_

    # ---------------------------------------------------------------- catch-up
    def catch_up(self, day: date, *, is_trading_day: bool = True) -> DayLog:
        """Handle a restart: what was missed, and does it still matter?

        The hard case, and the reason this is not a plain scheduler. A naive
        catch-up either reruns everything - squaring off at 16:00 into a shut
        market, where the order will not fill - or skips it and reports a clean
        run while a position sits open overnight.

        Neither is safe alone, so a missed critical job becomes an alert **plus**
        a position check, which is the only response that distinguishes "the
        moment passed harmlessly" from "something is still open".
        """
        from libs.ms_ops.rhythm import missed_windows

        log_ = self.logs.setdefault(day, DayLog(day=day))
        missed = missed_windows(
            day,
            now=self.clock.now(),
            completed=log_.completed,
            is_trading_day=is_trading_day,
            grace=self.grace,
        )
        for job in missed:
            if job.critical:
                self._raise_flatten_incident(
                    day, log_, f"{job.kind.value} never ran - its window closed"
                )
            else:
                log_.record_skip(job.kind, "window missed; not worth running late")
                log.info("missed window", fields={"job": job.kind.value})
        return log_

    def _raise_flatten_incident(self, day: date, log_: DayLog, what: str) -> None:
        """A missed flatten is judged by the account, not by the schedule."""
        open_now = self.positions_open()
        if open_now:
            message = (
                f"{what} and {open_now} position(s) are still open. This is not a "
                "missed job, it is overnight exposure nobody chose."
            )
            log_.incidents.append(message)
            self._alert("risk_breach", f"{day}: {message}")
        else:
            # The window passed and nothing is exposed. Still recorded - a
            # flatten that keeps not running is a fault even on the days it
            # happens not to matter.
            log_.record_skip(JobKind.FORCED_FLATTEN, f"{what}, but the book was already flat")
            self._alert(
                "system_health",
                f"{day}: {what}. Nothing was open, so no exposure - but the job "
                "still did not run.",
            )


# --------------------------------------------------------------------------- #
# Prefect adapter
# --------------------------------------------------------------------------- #
def prefect_flow(runner: Runner, *, is_trading_day: Callable[[date], bool]) -> object:
    """Wrap :meth:`Runner.run_day` as a Prefect flow, if Prefect is installed.

    Deliberately thin. Everything that decides *whether* something is safe lives
    in the runner; this only decides *when* it is invoked. Swapping Prefect for
    cron, systemd timers or a while-loop must not change a single behaviour, and
    the test suite proves it by never importing Prefect at all.
    """
    try:
        from prefect import flow  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError(
            "Prefect is not installed. The rhythm runs without it - "
            "Runner.run_day(day) is the whole interface - and this adapter only "
            "adds scheduling, retries at the flow level, and a UI."
        ) from exc

    @flow(name="marketsage-daily-rhythm")
    def daily(day: date | None = None) -> DayLog:
        target = day or runner.clock.now().date()
        return runner.run_day(target, is_trading_day=is_trading_day(target))

    return daily


def render_schedule(day: date, *, is_trading_day: bool = True) -> str:
    """The day's plan, for a human to read before trusting it."""
    lines = [f"  rhythm for {day}" + ("" if is_trading_day else "  (no session)")]
    for job in jobs_for(day, is_trading_day=is_trading_day):
        marks = []
        if job.critical:
            marks.append("CRITICAL")
        if job.depends_on:
            marks.append("after " + ", ".join(d.value for d in job.depends_on))
        if job.retries == 0:
            marks.append("no retries")
        suffix = f"  [{'; '.join(marks)}]" if marks else ""
        lines.append(f"    {job.at.strftime('%H:%M')}  {job.kind.value:<16}{suffix}")
        lines.append(f"             {job.description}")
    return "\n".join(lines)


def next_job(day: date, *, now: datetime, completed: Sequence[JobKind]) -> Job | None:
    """The next thing due. None when the day is done."""
    done = set(completed)
    upcoming = [
        j
        for j in jobs_for(day, is_trading_day=True)
        if j.kind not in done and j.scheduled_for(day) >= now
    ]
    return min(upcoming, key=lambda j: j.at, default=None)
