"""Nightly archive: NDJSON to Parquet, plus dataset versions.

    python jobs/archive.py --date 2026-08-26      one day
    python jobs/archive.py --all                  every unarchived day
    python jobs/archive.py --coverage             what the research layer can see
    python jobs/archive.py --dataset v1 --from 2026-08-01 --to 2026-08-31

Runs at 18:00 as part of the daily rhythm. The recorder keeps writing NDJSON;
this converts it for research and leaves the source untouched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.clock import RealClock
from libs.ms_core.obs import configure
from libs.ms_data.archive import (
    PARQUET_ROOT,
    Research,
    archive_all,
    available_days,
    build_dataset_version,
    verify_dataset_version,
)

STREAMS = ("ticks", "candles", "chain")


def archive_one(day: date) -> int:
    results = archive_all(day)
    total = 0
    for r in results:
        if r.rows:
            print(f"  {r.stream:<8} {day} {r.rows:>8,} rows -> {len(r.files)} file(s)")
            total += r.rows
        elif r.source_files:
            print(f"  {r.stream:<8} {day} source files present but empty")
    if not total:
        print(f"  {day}: nothing to archive")
    return total


def archive_everything() -> int:
    days: set[date] = set()
    for stream in STREAMS:
        days.update(available_days(stream))
    if not days:
        print("  no recorded days found. Start the recorder: python tasks.py record")
        return 1
    total = 0
    for day in sorted(days):
        total += archive_one(day)
    print(f"\n  {len(days)} day(s), {total:,} rows")
    return 0


def coverage() -> int:
    if not PARQUET_ROOT.exists():
        print("  nothing archived yet. Run: python jobs/archive.py --all")
        return 1
    with Research() as r:
        print(r.coverage().to_string(index=False))
        sessions = r.sessions("candles")
        if sessions:
            print(f"\n  candle sessions: {len(sessions)}  ({sessions[0]} .. {sessions[-1]})")
    return 0


def make_dataset(name: str, start: date, end: date) -> int:
    version = build_dataset_version(name, start, end, RealClock().now())
    print(f"  dataset {name!r}: {len(version.files)} files, digest {version.digest}")
    print(f"  {start} .. {end}")
    ok, problems = verify_dataset_version(name)
    print("  verify: OK" if ok else "  verify: FAILED\n    " + "\n    ".join(problems))
    return 0 if ok else 1


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Archive recorded data to Parquet (M17)")
    p.add_argument("--date", help="archive one day (YYYY-MM-DD)")
    p.add_argument("--all", action="store_true", help="archive every recorded day")
    p.add_argument("--coverage", action="store_true", help="what research can see")
    p.add_argument("--dataset", metavar="NAME", help="build a dataset version")
    p.add_argument("--from", dest="start", help="dataset start (YYYY-MM-DD)")
    p.add_argument("--to", dest="end", help="dataset end (YYYY-MM-DD)")
    args = p.parse_args()

    if args.dataset:
        if not (args.start and args.end):
            p.error("--dataset needs --from and --to")
        return make_dataset(
            args.dataset, date.fromisoformat(args.start), date.fromisoformat(args.end)
        )
    if args.coverage:
        return coverage()
    if args.date:
        archive_one(date.fromisoformat(args.date))
        return 0
    if args.all:
        return archive_everything()

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
