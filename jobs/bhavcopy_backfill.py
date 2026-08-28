"""Backfill NSE F&O bhavcopy into the local cache.

    python tasks.py bhavcopy --days 30
    python tasks.py bhavcopy --from 2024-01-01 --to 2024-12-31

One request per session, cached, and never re-fetched. Be deliberate about the
range: eight years is roughly two thousand requests to a free public archive
that owes nobody a service level. The default is thirty days and there is a
pause between requests, which is not politeness for its own sake - a source that
starts refusing you is a source you have lost.

Weekends are skipped without asking. Holidays cannot be skipped without a
holiday list, and the calendar refuses to guess a year it has not loaded (run
`python tasks.py holidays` first). Without one, every holiday shows up here as
a failure - which is honest, and noisy, and the reason the summary separates
"could not fetch" from "not a trading day" rather than reporting one number.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import CalendarNotLoadedError, SessionCalendar  # noqa: E402
from libs.ms_data.bhavcopy import BhavcopyStore, NoSessionData  # noqa: E402
from libs.ms_data.holidays import HolidayStore  # noqa: E402

PAUSE_SECONDS = 1.0


def run(start: date, end: date, *, symbol: str, pause: float) -> int:
    # from_config alone carries only the manual pins, which are normally empty -
    # the discovered calendar lives in the HolidayStore and has to be merged in.
    # Without this the calendar cannot answer, every holiday reads as a data gap,
    # and the summary below reports a hole that is not there.
    cal = SessionCalendar.from_config(cfgmod.load(warn=False).sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )
    store = BhavcopyStore(calendar=cal)

    fetched = cached = holidays = failed = 0
    traded_total = dropped_total = 0
    failures: list[str] = []

    day = start
    while day <= end:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue

        already = store.load(day) is not None
        try:
            got = store.get(day)
        except NoSessionData as exc:
            text = str(exc)
            if "not a trading day" in text:
                holidays += 1
            else:
                failed += 1
                failures.append(f"    {day}  {text.splitlines()[0][:100]}")
            day += timedelta(days=1)
            continue

        if already:
            cached += 1
        else:
            fetched += 1
            time.sleep(pause)

        kept, dropped = got.options(symbol)
        traded_total += len(kept)
        dropped_total += dropped
        day += timedelta(days=1)

    sessions = fetched + cached
    print(f"\n  {start} to {end}")
    print(f"    sessions in cache   {sessions:,}   ({fetched:,} fetched, {cached:,} already there)")
    print(f"    holidays skipped    {holidays:,}")
    print(f"    could not fetch     {failed:,}")

    if sessions:
        total = traded_total + dropped_total
        print(f"\n    {symbol} contracts across those sessions: {total:,}")
        print(f"      traded                 {traded_total:>10,}  ({traded_total/total:.0%})")
        print(
            f"      never traded, dropped  {dropped_total:>10,}  ({dropped_total/total:.0%})\n"
            "      Those carry an exchange-computed close AND settlement that\n"
            "      disagree by a median 16%. Neither is a price anybody paid, and\n"
            "      fitting a surface to them fits NSE's model, not the market."
        )

    if failures:
        print(f"\n  failures ({len(failures)}):")
        for line in failures[:20]:
            print(line)
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        print(
            "\n  A failure on a real trading day is a gap in the data, not a\n"
            "  holiday. Do not build a series across one without saying so."
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--days", type=int, default=30, help="calendar days back from today (default 30)"
    )
    p.add_argument("--from", dest="start", help="start date, YYYY-MM-DD")
    p.add_argument("--to", dest="end", help="end date, YYYY-MM-DD (default: today)")
    p.add_argument("--symbol", default="NIFTY", help="symbol for the summary counts")
    p.add_argument(
        "--pause",
        type=float,
        default=PAUSE_SECONDS,
        help="seconds between fetches; lower it only if you have a reason",
    )
    args = p.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)
    if start > end:
        raise SystemExit(f"  start {start} is after end {end}")

    try:
        return run(start, end, symbol=args.symbol, pause=args.pause)
    except CalendarNotLoadedError as exc:
        raise SystemExit(
            f"  {exc}\n"
            "  Run `python tasks.py holidays` first. Without a holiday list every\n"
            "  holiday in the range looks like a missing session, and a run that\n"
            "  cannot tell those apart cannot report a real gap."
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
