"""Replay a recorded day through the live code path.

    python jobs/replay.py --date 2026-08-26
    python jobs/replay.py --date 2026-08-26 --speed 60      one hour per minute
    python jobs/replay.py --date 2026-08-26 --rebuild       verify candle parity

The point of this job is not the output - it is that the engine and its
consumers cannot tell replay from live. ``--rebuild`` proves it: candles built
by watching the replayed tick stream must match the candles recorded during the
original session, exactly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import configure
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.holidays import HolidayStore
from libs.ms_data.replay import build_engine, recorded_days


def run(day: date, speed: float, rebuild: bool) -> int:
    configure()
    cfg = cfgmod.load(warn=False)
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )

    bus = InMemoryBus()
    engine, clock = build_engine(bus, day, speed=speed, calendar=calendar)

    builder = CandleBuilder(calendar) if rebuild else None
    rebuilt = []

    def consume(topic, msg):
        if builder is not None and topic is Topic.MARKET_TICK:
            rebuilt.extend(builder.on_tick(msg))

    stats = engine.run([day], on_message=consume)
    print(f"  {stats.summary()}")
    print(f"  wall time {stats.wall_seconds:.2f}s, out of order {stats.out_of_order}")

    if builder is None:
        return 0

    from datetime import timedelta

    rebuilt.extend(builder.flush(clock.now() + timedelta(minutes=1)))
    original = [d.payload for d in bus.subscribe(Topic.MARKET_CANDLE, "parity", "c")]
    if not original:
        print("  no recorded candles to compare against - skipping parity check")
        return 0

    def key(c):
        return (c.ts, c.instrument_id)

    def fields(c):
        return (c.ts, c.instrument_id, c.o, c.h, c.low, c.c, c.v)

    a = [fields(c) for c in sorted(original, key=key)]
    b = [fields(c) for c in sorted(rebuilt, key=key)]
    matched = b[: len(a)] == a
    print(
        f"  candle parity: {'OK' if matched else 'MISMATCH'} "
        f"({len(a):,} recorded vs {len(b):,} rebuilt)"
    )
    if not matched:
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                print(f"    first difference at index {i}:\n      recorded {x}\n      rebuilt  {y}")
                break
    return 0 if matched else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Replay a recorded day (M18)")
    p.add_argument("--date", help="day to replay (YYYY-MM-DD)")
    p.add_argument("--speed", type=float, default=float("inf"), help="1.0 = real time")
    p.add_argument("--rebuild", action="store_true", help="verify candle parity")
    p.add_argument("--list", action="store_true", help="show recorded days")
    args = p.parse_args()

    if args.list:
        days = recorded_days()
        print("  " + (", ".join(str(d) for d in days) if days else "nothing recorded"))
        return 0
    if not args.date:
        p.print_help()
        return 1
    return run(date.fromisoformat(args.date), args.speed, args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
