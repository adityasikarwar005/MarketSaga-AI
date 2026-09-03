"""Close out a trading day from history, whether or not anything watched it live.

    python jobs/close_out.py                    today
    python jobs/close_out.py --date 2026-09-02
    python jobs/close_out.py --dry-run          say what it would do and stop

`jobs/trading_day.py` has to be awake at 09:10 and stay awake until 15:30, and
that has been the whole problem: the machine was off at 09:05 on both mornings
this project has tried, so the recorder started at 09:31 and at 12:07 and missed
the opening auction each time. A schedule that depends on a laptop being awake
for six hours is not a schedule.

This does not depend on that. Angel's historical endpoint holds the completed
session - every contract, every minute - so one run **after the close** produces
the same charts as a recorder that ran all day. If the machine was asleep until
the evening, this still gets the whole day. Nothing is lost by being late.

**After the close is now the better time, not merely an acceptable one.** The
default is every live expiry and every listed strike - 2450 contracts across the
two indices - and each is its own request at about nine a minute, so a sweep
takes four to five hours. Run during the session, a sweep that long reads the
first rung at 10:00 and the last at 15:00 and the board is genuinely ragged.
Run after 15:30, every contract returns the whole day and the raggedness is
gone. The page reports the read window either way rather than pretending to an
instant it never had.

**It is not a replacement for the recorder**, and the difference matters. History
returns last-traded candles; the recorder captures the chain with both sides of
the book quoted at one instant, which is what a volatility surface may be fitted
on and what candles can never support - see
:class:`~libs.ms_options.surface.PriceBasis`. So the recorder still holds the
only data here that cannot be fetched again. This holds the part that can.

**It publishes nothing.** Deploying the result to a public Space is a separate
decision and a separate command.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs import chart_history, option_history  # noqa: E402
from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core import dotenv  # noqa: E402
from libs.ms_core import env as envmod  # noqa: E402
from libs.ms_core.clock import Clock, RealClock, SessionCalendar  # noqa: E402
from libs.ms_data.angelone import AngelCredentials, load_scrip_master, login  # noqa: E402
from libs.ms_data.holidays import HolidayStore  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = ROOT / "build" / "sessions"

UNDERLYINGS = ("NIFTY", "BANKNIFTY")

# None means every listed strike and every live expiry, which is what the page
# is now for. It is also what makes this job long: 1568 NIFTY contracts and 882
# BankNifty ones are 2450 separate requests - the endpoint has no batch form -
# and it answers about nine a minute under sustained load, so a full sweep runs
# for four to five hours. Run after the close that is fine and is in fact the
# better time: every contract then returns the whole session, so the board is
# not ragged the way a sweep taken during trading hours necessarily is.
STRIKES_EACH_SIDE: int | None = None
EXPIRIES: int | None = None


def _say(message: str, log: Path | None = None) -> None:
    line = f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} IST  {message}"
    print(line, flush=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _calendar() -> SessionCalendar:
    cfg = cfgmod.load(warn=False)
    return SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )


def main(argv: list[str] | None = None, *, clock: Clock | None = None) -> int:
    """``clock`` is injectable for the same reason everything else here is.

    The test that this job warns about a session still running has to compare a
    close against a now, and with the clock built in here the only now available
    was the real one - so the test pinned a date, passed on the day it was
    written, and began failing the next morning for a reason that had nothing to
    do with the behaviour it was checking. A time-dependent test that goes red
    on a calendar boundary teaches everyone to ignore it.
    """
    p = argparse.ArgumentParser(description="Build a finished day's charts from history")
    p.add_argument("--date", type=date.fromisoformat, help="override today")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    p.add_argument(
        "--strikes",
        type=option_history.count_or_all,
        default=STRIKES_EACH_SIDE,
        help="each side of the money, or 'all' (default)",
    )
    p.add_argument(
        "--expiries",
        type=option_history.count_or_all,
        default=EXPIRIES,
        help="nearest N expiries, or 'all' (default)",
    )
    p.add_argument("--underlying", nargs="*", default=list(UNDERLYINGS))
    args = p.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    clock = clock or RealClock()
    today = args.date or clock.now().astimezone(IST).date()
    log = LOG_DIR / f"closeout_{today.isoformat()}.log"

    # Before anything reaches for a credential. The runner that skipped this
    # cost two trading days; see libs/ms_core/dotenv.py.
    loaded = dotenv.load()
    try:
        environment = envmod.current().value
    except envmod.EnvironmentError_ as exc:
        _say(f"REFUSING: {exc}", log)
        return 2
    _say(f"environment is {environment!r}, {len(loaded)} name(s) read from .env.", log)

    calendar = _calendar()
    try:
        trading = calendar.is_trading_day(today)
    except LookupError as exc:
        _say(f"REFUSING: {exc}", log)
        return 2
    if not trading:
        _say(f"{today} is not a trading day. Nothing to close out.", log)
        return 0

    _, closed = calendar.session_bounds(today)
    now = clock.now()
    if now < closed:
        # Not an error - a partial day charts fine - but it must be said, or a
        # half-session gets filed as the session.
        left = (closed - now).total_seconds() / 60.0
        _say(f"WARNING: {today} is still open for {left:.0f} more minutes.", log)
        _say("  What is fetched will be the day so far, not the day.", log)

    if args.dry_run:
        _say(f"dry run - would close out {today} for {', '.join(args.underlying)}.", log)
        return 0

    master = load_scrip_master(clock, refresh=False)
    creds = AngelCredentials.from_env()
    session = login(creds, clock)
    _say(f"logged in; scrip master fetched {master.fetched_on}.", log)

    fetched: list[str] = []
    for name in args.underlying:
        try:
            option_history.collect(
                session,
                creds.api_key,
                master,
                name,
                today,
                n_strikes=args.strikes,
                n_expiries=args.expiries,
                root=option_history.HISTORY_ROOT,
                clock=clock,
            )
        except SystemExit as exc:
            # One underlying failing must not take the other down with it: a
            # half-built page is worth more than none, and the log says which
            # half is missing.
            _say(f"{name}: {exc}", log)
            continue
        fetched.append(name)

    if not fetched:
        _say("nothing was fetched. Not rewriting the chart with an empty day.", log)
        return 1

    boards = {
        name: chart_history.build(today, name, option_history.HISTORY_ROOT) for name in fetched
    }
    # Yesterday's expiries are not today's, and a leftover payload is a valid
    # file defining a valid global that the page would load and date from its
    # own contents.
    for old in chart_history.OUT_DIR.glob("chart_*_*.js"):
        old.unlink()
    written = chart_history.write(boards, today, chart_history.OUT_DIR)
    for name, group in boards.items():
        bars = sum(len(s["recorded"]) for b in group for s in b["series"])
        charted = sum(len(b["series"]) for b in group)
        traded = sum(len(b["chain"]) for b in group)
        _say(
            f"{name}: {traded} contracts traded across {len(group)} expiries, "
            f"{charted} chartable, {bars:,} real bars.",
            log,
        )
        for b in group:
            _say(f"  {b['expiry']}: {b['score']['verdict']}", log)
    _say(
        f"written to services/dashboard: {len(written)} files "
        f"({written[0].name} plus one per expiry). Nothing published.",
        log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
