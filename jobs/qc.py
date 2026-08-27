"""Nightly data quality gates.

    python jobs/qc.py --date 2026-08-26          check one day
    python jobs/qc.py --all                       every recorded day
    python jobs/qc.py --date 2026-08-26 --strict  exit non-zero on any FAIL

Runs at 18:00, before the archive. A FAIL means research on that day would
produce an artefact rather than a finding, so the gate blocks rather than warns.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import configure
from libs.ms_core.schemas import Candle, ChainSnapshot, RawNewsItem, Tick
from libs.ms_data.holidays import HolidayStore
from libs.ms_data.quality import run_market_checks
from libs.ms_data.recorder import DATA_ROOT, read_stream
from libs.ms_data.replay import recorded_days


def _load(stream: str, day: date, cls: type):
    partition = DATA_ROOT / stream / f"date={day.isoformat()}"
    if not partition.exists():
        return []
    out = []
    for path in sorted(partition.glob("*.ndjson")):
        out.extend(read_stream(path, cls))
    return out


def check_day(day: date, calendar: SessionCalendar, *, save: bool = True) -> bool:
    report = run_market_checks(
        day,
        calendar,
        ticks=_load("ticks", day, Tick),
        candles=_load("candles", day, Candle),
        chain=_load("chain", day, ChainSnapshot),
        news=_load("news", day, RawNewsItem),
    )
    print(report.render())
    if save:
        path = report.save()
        print(f"  -> {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return report.ok


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Data quality gates (M19)")
    p.add_argument("--date", help="one day (YYYY-MM-DD)")
    p.add_argument("--all", action="store_true", help="every recorded day")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any FAIL")
    p.add_argument("--no-save", action="store_true", help="do not write the JSON report")
    args = p.parse_args()

    cfg = cfgmod.load(warn=False)
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )

    if args.date:
        days = [date.fromisoformat(args.date)]
    elif args.all:
        days = recorded_days()
    else:
        p.print_help()
        return 1

    if not days:
        print("  nothing recorded yet. Start the recorder: python tasks.py record")
        return 1

    failed = []
    for day in days:
        # A day the exchange was closed has no candles by definition, and running
        # the gates on it would produce a failure that means nothing.
        try:
            if not calendar.is_trading_day(day):
                print(f"  {day}: not a trading day - skipped")
                continue
        except LookupError:
            pass  # no calendar for that year; check it anyway rather than skip
        if not check_day(day, calendar, save=not args.no_save):
            failed.append(day)
        print()

    if failed:
        print(f"  BLOCKED: {len(failed)} day(s) failed - {', '.join(str(d) for d in failed)}")
        return 1 if args.strict else 0
    print(f"  {len(days)} day(s) checked, all clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
