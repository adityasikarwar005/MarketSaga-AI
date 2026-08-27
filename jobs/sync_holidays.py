"""Discover and refresh the exchange holiday calendar.

    python jobs/sync_holidays.py --years 2025 2026        fetch and reconcile
    python jobs/sync_holidays.py --show                    what is known, and why
    python jobs/sync_holidays.py --why 2026-10-02          one day, explained

Run it monthly and after any exchange circular. The calendar refuses to guess
(see libs/ms_core/clock.py), so this job is what turns that refusal into an
answer.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import RealClock
from libs.ms_core.obs import configure
from libs.ms_data.holidays import (
    Confidence,
    HolidayStore,
    InferredHolidayProvider,
    ManualHolidayProvider,
    NseHolidayProvider,
    describe_day,
    next_holiday,
    observed_trading_days_from_recordings,
    staleness_days,
)
from libs.ms_data.recorder import DATA_ROOT


def sync(years: list[int], *, strict: bool) -> int:
    configure()
    clock = RealClock()
    cfg = cfgmod.load(warn=False)
    store = HolidayStore()

    observed = observed_trading_days_from_recordings(DATA_ROOT)
    providers = [
        ManualHolidayProvider(dict(cfg.sessions.holidays)),
        NseHolidayProvider(),
        InferredHolidayProvider(observed),
    ]
    if observed:
        print(f"  cross-checking against {len(observed)} recorded trading days")
    else:
        print("  no recorded days yet - the published list cannot be cross-checked")

    failures = 0
    for year in years:
        try:
            entry = store.sync(year, clock, providers=providers, strict=strict)
        except Exception as exc:
            print(f"  {year}: FAILED - {exc}")
            failures += 1
            continue
        confirmed = sum(1 for h in entry.holidays if h.confidence is Confidence.CONFIRMED)
        print(
            f"  {year}: {len(entry.holidays)} holidays "
            f"({confirmed} confirmed by 2+ sources) -> {store.path(year)}"
        )
        for h in entry.disputed:
            print(f"      DISPUTED {h.day} {h.reason} - {h.note}")
    return 1 if failures else 0


def show() -> int:
    store = HolidayStore()
    clock = RealClock()
    years = store.years()
    if not years:
        print("  no holiday calendar on disk. Run: python jobs/sync_holidays.py --years 2026")
        return 1

    for year in years:
        entry = store.load(year)
        assert entry is not None
        age = staleness_days(entry, clock.now())
        stale = "  (STALE - refresh)" if age > 90 else ""
        print(f"\n  {year}  fetched {age}d ago{stale}")
        for h in entry.holidays:
            mark = {
                Confidence.CONFIRMED: "  ",
                Confidence.SINGLE: " ?",
                Confidence.DISPUTED: " !",
            }[h.confidence]
            src = ",".join(s.value for s in h.sources)
            print(f"   {mark} {h.day}  {h.reason:<42} [{src}]")

    today = clock.now().date()
    nxt = next_holiday(store, today)
    print()
    if nxt:
        days = (nxt.day - today).days
        when = "tomorrow" if days == 1 else f"in {days} days"
        print(f"  next closure: {nxt.day} ({when}) - {nxt.reason}")
    else:
        print("  no upcoming closure within the loaded horizon")
    return 0


def why(day_str: str) -> int:
    print("  " + describe_day(HolidayStore(), date.fromisoformat(day_str)))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Discover the exchange holiday calendar")
    p.add_argument("--years", nargs="*", type=int, help="years to fetch and reconcile")
    p.add_argument("--show", action="store_true", help="print what is known and its provenance")
    p.add_argument("--why", metavar="YYYY-MM-DD", help="explain one specific day")
    p.add_argument("--strict", action="store_true", help="fail if any provider errors")
    args = p.parse_args()

    if args.why:
        return why(args.why)
    if args.years:
        rc = sync(args.years, strict=args.strict)
        show()
        return rc
    return show()


if __name__ == "__main__":
    sys.exit(main())
