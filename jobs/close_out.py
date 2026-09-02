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
from libs.ms_core.clock import RealClock, SessionCalendar  # noqa: E402
from libs.ms_data.angelone import AngelCredentials, load_scrip_master, login  # noqa: E402
from libs.ms_data.holidays import HolidayStore  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = ROOT / "build" / "sessions"

UNDERLYINGS = ("NIFTY", "BANKNIFTY")
STRIKES_EACH_SIDE = 5


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a finished day's charts from history")
    p.add_argument("--date", type=date.fromisoformat, help="override today")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    p.add_argument("--strikes", type=int, default=STRIKES_EACH_SIDE)
    p.add_argument("--underlying", nargs="*", default=list(UNDERLYINGS))
    args = p.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    clock = RealClock()
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
                root=option_history.HISTORY_ROOT,
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

    blocks = [chart_history.build(today, name, option_history.HISTORY_ROOT) for name in fetched]
    written = chart_history.write(blocks, today, chart_history.OUT_DIR)
    for b in blocks:
        bars = sum(len(s["recorded"]) for s in b["series"])
        _say(
            f"{b['name']}: {len(b['series'])} contracts, {bars:,} real bars, "
            f"forward {b['forward']['close']:,.2f}. {b['score']['verdict']}",
            log,
        )
    names = ", ".join(w.name for w in written)
    _say(f"written to services/dashboard: {names}. Nothing published.", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
