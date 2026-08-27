"""Cross-asset ingest - runs 24x7, unlike the market feed.

    python -m services.ingest_crossasset --source synthetic --hours 12
    python -m services.ingest_crossasset --window 2026-08-27

A separate service from ``ingest_market`` on purpose. The Indian market feed
runs 09:15-15:30; this one must be running at 02:00 when US futures move, which
is precisely when the overnight gap is being formed. Tying the two together
would mean the most informative hours of the day go unrecorded.
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import FrameType

from libs.ms_core import config as cfgmod
from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import UTC, Clock, RealClock, SessionCalendar, SimClock
from libs.ms_core.obs import configure, get_logger, stage, trace
from libs.ms_data.crossasset import (
    CrossAssetCollector,
    SyntheticCrossAssetProvider,
    overnight_moves,
    overnight_window,
    registry_from_config,
)
from libs.ms_data.holidays import HolidayStore
from libs.ms_data.recorder import DATA_ROOT, Recorder

log = get_logger("ingest_crossasset")
_stop = False


def _handle_signal(_sig: int, _frame: FrameType | None) -> None:
    global _stop
    log.info("shutdown signal received - sealing files")
    _stop = True


def _calendar() -> SessionCalendar:
    cfg = cfgmod.load(warn=False)
    return SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )


def show_window(session: date) -> int:
    """Print what a gap forecast for ``session`` is allowed to see."""
    cfg = cfgmod.load(warn=False)
    calendar = _calendar()
    registry = registry_from_config(cfg.crossasset.series)
    hh, _, mm = cfg.crossasset.gap_cutoff_ist.partition(":")
    from datetime import time as dtime

    window = overnight_window(calendar, session, cutoff=dtime(int(hh), int(mm)))
    print(f"  session          : {window.session}")
    print(f"  previous session : {window.previous_session}")
    print(
        f"  window           : {window.start.astimezone(calendar.tz):%Y-%m-%d %H:%M} "
        f"-> {window.end.astimezone(calendar.tz):%Y-%m-%d %H:%M} IST"
    )
    print(
        f"  span             : {window.hours:.1f} hours"
        f"{'  (spans a weekend or holiday)' if window.spans_multiple_days else ''}"
    )

    provider = SyntheticCrossAssetProvider()
    bars = [b for s in registry.all() for b in provider.fetch(s, window.start, window.end)]
    moves = overnight_moves(bars, window, registry)
    if not moves:
        print("\n  no cross-asset bars in the window")
        return 0
    print(f"\n  {'series':<12} {'bars':>6} {'move':>9} {'range':>9}")
    for symbol in sorted(moves):
        m = moves[symbol]
        print(f"  {symbol:<12} {m.bars:>6,} {m.ret:>8.3%} {m.range_pct:>8.3%}")
    return 0


def collect(args: argparse.Namespace) -> int:
    configure()
    cfg = cfgmod.load(warn=False)
    calendar = _calendar()
    registry = registry_from_config(cfg.crossasset.series)

    clock: Clock = (
        SimClock(_start_of(args, calendar)) if args.source == "synthetic" else RealClock()
    )
    provider = SyntheticCrossAssetProvider(seed=args.seed)
    collector = CrossAssetCollector(registry, provider, clock)
    recorder = Recorder(clock, calendar, root=Path(args.root))
    bus = InMemoryBus(max_len=500_000)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "cross-asset ingest starting",
        fields={"series": len(registry), "enabled": len(registry.enabled()), "root": args.root},
    )

    start = clock.now()
    end = start + timedelta(hours=args.hours)
    cursor = start
    written = 0

    while cursor < end and not _stop:
        chunk_end = min(cursor + timedelta(minutes=args.chunk_minutes), end)
        if isinstance(clock, SimClock):
            clock.set(chunk_end)
        with stage("crossasset_poll"):
            bars = collector.poll(cursor, chunk_end)
        for bar in bars:
            with trace(bar.trace_id, stage="crossasset"):
                bus.publish(Topic.MARKET_CANDLE, bar)
                recorder.write("crossasset", bar, key=bar.instrument_id)
                written += 1
        stale = collector.stale_series(chunk_end)
        if stale:
            log.warning("cross-asset series went quiet", fields={"symbols": stale})
        cursor = chunk_end

    manifests = recorder.close()
    log.info(
        "cross-asset ingest stopped",
        fields={"bars": written, "files": len(manifests)},
    )
    print(f"  {written:,} bars across {len(manifests)} file(s)")
    for symbol, health in sorted(collector.health.items()):
        if health.bars:
            print(f"    {symbol:<12} {health.bars:>7,} bars, last {health.last_bar_at:%H:%M} UTC")
    return 0


def _start_of(args: argparse.Namespace, calendar: SessionCalendar) -> datetime:
    day = date.fromisoformat(args.date) if args.date else date(2026, 8, 26)
    # Start in the evening: the overnight window is the point of this service.
    return (
        datetime.combine(day, datetime.min.time(), tzinfo=calendar.tz)
        .replace(hour=args.start_hour)
        .astimezone(UTC)
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Cross-asset ingest, 24x7 (M15)")
    p.add_argument("--source", choices=("synthetic",), default="synthetic")
    p.add_argument("--window", metavar="YYYY-MM-DD", help="show the overnight window and exit")
    p.add_argument("--date", help="synthetic only: start date")
    p.add_argument("--start-hour", type=int, default=16, help="synthetic only: IST start hour")
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--chunk-minutes", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--root", default=str(DATA_ROOT))
    args = p.parse_args()

    if args.window:
        return show_window(date.fromisoformat(args.window))
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
