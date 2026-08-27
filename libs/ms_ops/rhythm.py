"""MA1 - the daily operating rhythm.

Seven jobs at seven times. Six of them can fail and be retried tomorrow. One
cannot, and most of this module exists because of that one.

The forced flatten is not a scheduled job
-----------------------------------------

15:20 squares off every intraday position, no exceptions. Treating it as one
entry in a job list makes it as skippable as the nightly archive, and the failure
modes are not comparable: a missed archive costs a rerun, a missed flatten
carries a position overnight through a gap nobody modelled.

So :attr:`Job.critical` marks it, and three rules follow that nothing else gets:

* it runs even when every earlier job failed - a broken preflight is a reason to
  hold no position, not a reason to keep one;
* a **missed window is an incident**, not a skipped job. On restart the runner
  checks whether positions are actually open rather than assuming the moment
  passed harmlessly;
* it cannot be disabled by configuration. A flag that turns it off is a flag
  someone sets during a debugging session on a Tuesday.

Missed windows are the interesting case
---------------------------------------

If the process was down from 15:00 to 16:00, the flatten never fired. A naive
scheduler does one of two things and both are wrong: run it late, squaring off at
16:00 when the market is shut and the order will not fill; or skip it, leaving
the position open overnight and reporting a clean run.

:func:`missed_windows` returns what did not fire, and the runner treats a missed
critical job as an alert plus a position check - the only honest response, since
neither running nor skipping is safe on its own.

Ordering is a dependency, not a time
------------------------------------

Post-session must run after the flatten because it computes outcomes for
positions the flatten closed, not because 15:45 is later than 15:20. Encoding it
as a time means a clock change or a slow flatten produces attribution over
positions that are still open, and the numbers look fine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# How late a job may start and still be considered on time. Beyond this it is a
# missed window, which for a critical job is an incident.
DEFAULT_GRACE = timedelta(minutes=5)

# These times are exchange-local, and saying so is load-bearing rather than
# pedantic. The Clock reports timezone-aware UTC; combining a naive 15:20 with a
# date and comparing it against that is either a TypeError - which is what
# happened - or, worse, a silent five-and-a-half-hour error that fires the
# forced flatten at 20:50 IST, hours after the market shut.
EXCHANGE_TZ = ZoneInfo("Asia/Kolkata")


class JobKind(StrEnum):
    BRIEFING = "briefing"
    PREFLIGHT = "preflight"
    PRE_OPEN = "pre_open"
    SESSION = "session"
    FORCED_FLATTEN = "forced_flatten"
    POST_SESSION = "post_session"
    NIGHTLY = "nightly"
    WEEKEND_RETRAIN = "weekend_retrain"


@dataclass(frozen=True)
class Job:
    """One thing that happens at one time, and what it needs first."""

    kind: JobKind
    at: time
    description: str
    depends_on: tuple[JobKind, ...] = ()
    critical: bool = False
    retries: int = 1
    weekend_only: bool = False

    def scheduled_for(self, day: date, tz: ZoneInfo = EXCHANGE_TZ) -> datetime:
        """When this runs, as an instant rather than a wall-clock reading.

        Timezone-aware, because the only thing it is ever compared against is a
        Clock that reports UTC. A naive return value makes that comparison a
        TypeError on a good day and a five-and-a-half-hour error on a bad one.
        """
        return datetime.combine(day, self.at, tzinfo=tz)


# The rhythm, as data. IST throughout - the times are exchange times and mean
# nothing in any other zone.
RHYTHM: tuple[Job, ...] = (
    Job(
        JobKind.BRIEFING,
        time(6, 30),
        "collect overnight events, compute the gap forecast, publish the briefing",
        retries=2,
    ),
    Job(
        JobKind.PREFLIGHT,
        time(8, 45),
        "feed, clock sync, model version, margin, calendar. Fails loudly.",
        retries=0,  # a retried preflight is a preflight nobody read
    ),
    Job(
        JobKind.PRE_OPEN,
        time(9, 0),
        "observe only; no signals",
        depends_on=(JobKind.PREFLIGHT,),
    ),
    Job(
        JobKind.SESSION,
        time(9, 15),
        "hot and cold lanes active, risk gate armed",
        depends_on=(JobKind.PREFLIGHT,),
    ),
    Job(
        JobKind.FORCED_FLATTEN,
        time(15, 20),
        "every intraday position squared off, no exceptions",
        critical=True,
        retries=3,
    ),
    Job(
        JobKind.POST_SESSION,
        time(15, 45),
        "resolve outcomes, update the ledger, run attribution",
        # A dependency, not a time. It computes outcomes for positions the
        # flatten closed, and running it against still-open positions produces
        # attribution that looks fine and means nothing.
        depends_on=(JobKind.FORCED_FLATTEN,),
        retries=2,
    ),
    Job(
        JobKind.NIGHTLY,
        time(18, 0),
        "feature materialisation, quality gates, archive, backup verification",
        depends_on=(JobKind.POST_SESSION,),
        retries=2,
    ),
    Job(
        JobKind.WEEKEND_RETRAIN,
        time(9, 0),
        "incremental retrain, walk-forward, champion/challenger, calibration",
        weekend_only=True,
        retries=1,
    ),
)


def jobs_for(day: date, *, is_trading_day: bool) -> list[Job]:
    """What runs on this day.

    ``is_trading_day`` comes from the session calendar rather than from the
    weekday, because a holiday is a Tuesday with no session and the weekday says
    nothing about it.
    """
    if not is_trading_day:
        return [j for j in RHYTHM if j.weekend_only]
    return [j for j in RHYTHM if not j.weekend_only]


def critical_jobs() -> list[Job]:
    return [j for j in RHYTHM if j.critical]


def missed_windows(
    day: date,
    *,
    now: datetime,
    completed: Sequence[JobKind],
    is_trading_day: bool = True,
    grace: timedelta = DEFAULT_GRACE,
) -> list[Job]:
    """Jobs whose time has passed without them running.

    Returned rather than acted on. What to do about a missed window depends on
    which job it was - a missed briefing is a shrug, a missed flatten is an
    incident - and that decision belongs with the caller who can see the
    positions.
    """
    done = set(completed)
    return [
        j
        for j in jobs_for(day, is_trading_day=is_trading_day)
        if j.kind not in done and now > j.scheduled_for(day) + grace
    ]


def blocked_by(job: Job, *, completed: Sequence[JobKind]) -> list[JobKind]:
    """Which dependencies have not run.

    A critical job has no dependencies it will wait for. A broken preflight is a
    reason to hold no position, not a reason to keep one - so the flatten runs
    regardless of what came before it.
    """
    if job.critical:
        return []
    done = set(completed)
    return [d for d in job.depends_on if d not in done]


# --------------------------------------------------------------------------- #
# The day's state
# --------------------------------------------------------------------------- #
@dataclass
class DayLog:
    """What ran, what failed, and what never got the chance."""

    day: date
    completed: list[JobKind] = field(default_factory=list)
    failed: dict[JobKind, str] = field(default_factory=dict)
    skipped: dict[JobKind, str] = field(default_factory=dict)
    incidents: list[str] = field(default_factory=list)

    def record_success(self, kind: JobKind) -> None:
        self.completed.append(kind)
        self.failed.pop(kind, None)

    def record_failure(self, kind: JobKind, reason: str) -> None:
        self.failed[kind] = reason

    def record_skip(self, kind: JobKind, reason: str) -> None:
        self.skipped[kind] = reason

    @property
    def flatten_ran(self) -> bool:
        return JobKind.FORCED_FLATTEN in self.completed

    @property
    def clean(self) -> bool:
        """A day is clean when nothing failed and no incident was raised.

        Skips are not failures - a job whose dependency did not run was
        correctly not run, and counting that as a failure would bury the one
        that actually broke under the ones that followed it.
        """
        return not self.failed and not self.incidents

    def render(self) -> str:
        lines = [
            f"  {self.day}: {len(self.completed)} ran, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped"
        ]
        lines += [f"    FAILED {k.value}: {v}" for k, v in self.failed.items()]
        lines += [f"    skipped {k.value}: {v}" for k, v in self.skipped.items()]
        lines += [f"    INCIDENT: {i}" for i in self.incidents]
        if not self.flatten_ran:
            lines.append(
                "    the forced flatten did not run - this is the one job whose "
                "absence carries a position overnight"
            )
        return "\n".join(lines)
