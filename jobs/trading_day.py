"""Run one trading day unattended: preflight, record from the open, close down.

    python jobs/trading_day.py                 today, if today is a trading day
    python jobs/trading_day.py --dry-run       say what it would do and stop
    python jobs/trading_day.py --no-post       skip the post-session jobs

Meant to be started by the operating system's scheduler a little before the
open and then left alone. Everything it does is decided from the repository's
own configuration, so the scheduler needs to know nothing except "run this".

**It refuses to run on a day the exchange is shut.** The holiday list is the
project's, not the scheduler's, so a task set to fire every weekday does the
right thing on Ganesh Chaturthi without anybody remembering to disable it. A
scheduler that knew the calendar separately would be a second copy of it, and
the copy that is wrong is always the one nobody is looking at.

**It supervises rather than launches.** The Angel socket dropped four times in
one morning on 2026-08-31, and each drop cost between fourteen and twenty-four
minutes - not because the recorder is slow to restart, which takes about a
second once the session and the scrip master are cached, but because a person
had to notice. The loop removes that wait. It is not a substitute for
reconnecting inside the adapter, which is the real fix.

**Short runs back off.** A recorder that dies immediately is failing to start,
not dropping a socket, and restarting it at full speed against a broker turns
one bad credential into a login flood.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock, SessionCalendar  # noqa: E402
from libs.ms_data.holidays import HolidayStore  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = ROOT / "build" / "sessions"

# Start early enough to absorb a slow scrip-master download and a login, and
# late enough that a machine which woke at 09:00 is not holding a socket open
# for half an hour before there is anything on it.
WARMUP_MINUTES = 5

TOO_SHORT_SECONDS = 20.0
BACKOFF_START = 5.0
BACKOFF_MAX = 120.0


def _calendar() -> SessionCalendar:
    cfg = cfgmod.load(warn=False)
    return SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )


def _say(message: str, log: Path | None = None) -> None:
    line = f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} IST  {message}"
    print(line, flush=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _run(args: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n----- {' '.join(args)} at {datetime.now(IST).isoformat()} -----\n")
        fh.flush()
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT
        ).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Record one trading day, unattended")
    p.add_argument("--date", type=date.fromisoformat, help="override today (for a dry run)")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    p.add_argument("--no-post", action="store_true", help="skip QC and the sheets afterwards")
    args = p.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    clock = RealClock()
    today = args.date or clock.now().astimezone(IST).date()
    log = LOG_DIR / f"session_{today.isoformat()}.log"

    calendar = _calendar()
    try:
        trading = calendar.is_trading_day(today)
    except LookupError as exc:
        _say(f"REFUSING: {exc}", log)
        _say("Populate config/sessions.yaml from the exchange circular first.", log)
        return 2
    if not trading:
        _say(f"{today} is not a trading day. Nothing to record.", log)
        return 0

    opened, closed = calendar.session_bounds(today)
    start_at = opened - timedelta(minutes=WARMUP_MINUTES)
    _say(f"{today} is a trading day.", log)
    _say(f"  session {opened.astimezone(IST):%H:%M} - {closed.astimezone(IST):%H:%M} IST", log)
    _say(f"  starting the recorder at {start_at.astimezone(IST):%H:%M} IST", log)

    if args.dry_run:
        _say("dry run - stopping here.", log)
        return 0

    # The scrip master is public and needs no account. Refreshing it before the
    # open is the difference between a session that resolves its tokens and one
    # that resolves yesterday's expiries.
    _say("refreshing the scrip master ...", log)
    if _run(["jobs/angel_preflight.py", "--refresh-master"], log) != 0:
        _say("preflight FAILED - see the log. Not starting the recorder.", log)
        return 1
    _say("preflight clean.", log)

    while clock.now() < start_at:
        time.sleep(min(30.0, (start_at - clock.now()).total_seconds()))

    runs, backoff = 0, BACKOFF_START
    while clock.now() < closed:
        runs += 1
        _say(f"recorder run {runs} starting", log)
        began = time.monotonic()
        rc = _run(["tasks.py", "record", "angel", "--chain"], log)
        lasted = time.monotonic() - began

        if clock.now() >= closed:
            _say(f"run {runs} ended at the close after {lasted / 60:.0f} min (rc={rc}).", log)
            break
        if lasted < TOO_SHORT_SECONDS:
            _say(
                f"run {runs} exited after {lasted:.0f}s (rc={rc}) - that is a failure to "
                f"start, not a dropped socket. Waiting {backoff:.0f}s.",
                log,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
        else:
            _say(f"run {runs} ended after {lasted / 60:.1f} min (rc={rc}) - restarting.", log)
            backoff = BACKOFF_START

    _say(f"session over. {runs} recorder run(s).", log)

    if args.no_post:
        return 0

    # Quality first, then the sheets. A day that failed its gates still gets its
    # sheets written - the report is what says whether to trust them, and
    # withholding the data would only mean somebody reads it from somewhere else.
    _say("running the quality gates ...", log)
    qc = _run(["jobs/qc.py", "--date", today.isoformat()], log)
    _say(f"quality gates exited {qc} (non-zero means the day is blocked).", log)
    _say("writing the sheets ...", log)
    _run(["jobs/option_sheet.py", "--date", today.isoformat()], log)
    _say(f"done. log: {log}", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
