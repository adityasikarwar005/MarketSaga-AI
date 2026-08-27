"""Market ingest service - feed to bus to recorder to candles.

Runs against either source:

    python -m services.ingest_market --source synthetic --minutes 30
    python -m services.ingest_market --source kite

The synthetic source is not a placeholder to be deleted later. It stays
forever, because it is what lets CI exercise this exact service on every commit,
and what lets the pipeline be debugged at 2am without a live market.
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import timedelta
from pathlib import Path
from types import FrameType

from libs.ms_core import config as cfgmod
from libs.ms_core import env as envmod
from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import Clock, RealClock, SessionCalendar, SimClock
from libs.ms_core.obs import METRICS, configure, get_logger, stage, trace
from libs.ms_data.candles import CandleBuilder, verify_session
from libs.ms_data.chain import ChainBuilder, ChainSpec, build_registry_from_specs, check_chain
from libs.ms_data.feed import FeedHealth, MarketFeed, SyntheticFeed, SyntheticInstrument
from libs.ms_data.holidays import HolidayStore, describe_day, next_holiday
from libs.ms_data.recorder import DATA_ROOT, Recorder
from libs.ms_data.synthetic_chain import SyntheticChainFeed, strike_grid, weekly_expiries

log = get_logger("ingest_market")
_stop = False


def _handle_signal(_sig: int, _frame: FrameType | None) -> None:
    """Shut down cleanly: seal the files and write the manifests.

    A recorder killed mid-write leaves a partition with no manifest, which the
    QC job then reports as an unclean shutdown. Handling the signal is what
    turns "the machine rebooted" into a non-event.
    """
    global _stop
    log.info("shutdown signal received - sealing files")
    _stop = True


def build_synthetic_feed(calendar: SessionCalendar, clock: SimClock, seed: int) -> SyntheticFeed:
    feed = SyntheticFeed(
        [
            SyntheticInstrument(instrument_id=1, start_price=24_800, annual_vol=0.14),
            SyntheticInstrument(instrument_id=2, start_price=54_200, annual_vol=0.18),
            SyntheticInstrument(instrument_id=21, start_price=13.5, annual_vol=0.80),
        ],
        calendar,
        clock,
        seed=seed,
    )
    feed.subscribe([1, 2, 21])
    return feed


def build_kite_feed(clock: Clock) -> MarketFeed:
    from libs.ms_broker.kite import KiteFeed, require_session

    envmod.require_broker_credentials()
    session = require_session(clock)  # raises with instructions if the token is stale
    feed = KiteFeed(session, clock)
    log.info("kite session ready", fields={"user_id": session.user_id})
    return feed


def run(args: argparse.Namespace) -> int:
    configure()
    cfg = cfgmod.load(warn=True)
    # Config pins first, then whatever the holiday sync discovered (M10).
    store = HolidayStore()
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(store.as_calendar_dict())

    if args.source == "synthetic":
        # Synthetic runs use a guessed calendar so a demo works before the real
        # holiday circular is loaded. Real sources never get this.
        calendar = calendar.synthetic_variant()
        clock: Clock = SimClock(_synthetic_start(calendar, args))
        feed: MarketFeed = build_synthetic_feed(calendar, clock, args.seed)  # type: ignore[arg-type]
        end = clock.now() + timedelta(minutes=args.minutes)
    else:
        clock = RealClock()
        feed = build_kite_feed(clock)
        end = None

    chain_builder, chain_feed = _build_chain(args, calendar, clock)

    bus = InMemoryBus(max_len=1_000_000)
    builder = CandleBuilder(calendar)
    recorder = Recorder(clock, calendar, root=Path(args.root))
    health = FeedHealth(
        clock=clock,
        calendar=calendar,
        staleness_halt_seconds=cfg.risk_limits.feed_staleness_halt_seconds,
    )

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    _announce_calendar(store, clock)

    log.info(
        "ingest starting",
        fields={
            "source": feed.name,
            "env": envmod.current().value,
            "root": str(args.root),
            "config_hash": cfg.config_hash,
        },
    )

    ticks = candles = chains = 0
    try:
        for tick in feed.stream():
            if _stop or (end is not None and tick.ts >= end):
                break
            # One trace id per tick, carried through every stage below, so a
            # candle can always be traced back to the ticks that formed it.
            with trace(tick.trace_id, stage="ingest"):
                health.on_message()
                with stage("bus_publish"):
                    bus.publish(Topic.MARKET_TICK, tick)
                with stage("record"):
                    recorder.write_tick(tick)
                with stage("candle_build"):
                    for candle in builder.on_tick(tick):
                        bus.publish(Topic.MARKET_CANDLE, candle)
                        # Candles are recorded too, not only published. The bus is
                        # in-memory: a process restart would otherwise lose them,
                        # and rebuilding a day from ticks is slow enough that the
                        # research layer should never have to.
                        recorder.write("candles", candle, key=candle.instrument_id)
                        candles += 1
                if chain_builder is not None:
                    with stage("chain"):
                        chain_builder.on_tick(tick)
                        if chain_feed is not None and tick.instrument_id == args.chain_underlying:
                            for opt in chain_feed.quotes_for(tick.ltp, tick.ts):
                                chain_builder.on_tick(opt)
                        snap = chain_builder.maybe_snapshot(tick.ts)
                        if snap is not None:
                            bus.publish(Topic.MARKET_CHAIN, snap)
                            recorder.write_chain(snap)
                            chains += 1
            ticks += 1
            if ticks % 5_000 == 0:
                health.evaluate()
                log.info("progress", fields={"ticks": ticks, "candles": candles})
    finally:
        for candle in builder.flush(end):
            bus.publish(Topic.MARKET_CANDLE, candle)
            recorder.write("candles", candle, key=candle.instrument_id)
            candles += 1
        manifests = recorder.close()
        feed.close()

    log.info(
        "ingest stopped",
        fields={
            "ticks": ticks,
            "candles": candles,
            "chains": chains,
            "files": len(manifests),
            "rows": sum(m.rows for m in manifests),
        },
    )

    if args.verify:
        built = [d.payload for d in bus.subscribe(Topic.MARKET_CANDLE, "verify", "c")]
        for r in verify_session(built, calendar):  # type: ignore[arg-type]
            status = "OK " if r.ok else "BAD"
            print(
                f"  {status} instrument={r.instrument_id:<5} "
                f"candles={r.actual}/{r.expected} synthetic={r.synthetic_pct:.1%}"
            )
    if args.verify and chain_builder is not None:
        snaps = [d.payload for d in bus.subscribe(Topic.MARKET_CHAIN, "verify", "c")]
        if snaps:
            last = snaps[-1]
            rep = check_chain(last)  # type: ignore[arg-type]
            status = "OK " if rep.completeness >= 0.95 else "BAD"
            print(
                f"  {status} chain quotes={rep.quotes} "
                f"completeness={rep.completeness:.1%} issues={rep.by_kind() or 'none'}"
            )

    if args.metrics:
        print("\n" + "-" * 60)
        for name, h in METRICS.snapshot()["histograms"].items():
            print(f"  {name:<45} p50={h['p50'] * 1000:7.3f}ms p99={h['p99'] * 1000:7.3f}ms")
    return 0


def _build_chain(args, calendar, clock):
    """Wire up the chain snapshotter (M14).

    Historical option chains cannot be bought back cheaply, so the chain is
    captured from the first session, alongside ticks - not added later once the
    models need it.
    """
    if not args.chain:
        return None, None

    spot_guess = float(args.chain_spot)
    today = calendar.local(clock.now()).date()
    expiries = weekly_expiries(calendar, today, count=2)
    strikes = strike_grid(spot_guess, step=50.0, n_per_side=25)
    registry = build_registry_from_specs(args.chain_underlying, expiries, strikes)
    spec = ChainSpec(
        underlying_id=args.chain_underlying,
        n_expiries=2,
        strikes_per_side=args.strikes_per_side,
        snapshot_seconds=args.chain_seconds,
    )
    builder = ChainBuilder(spec, registry, calendar)

    feed = None
    if args.source == "synthetic":
        feed = SyntheticChainFeed(
            registry, calendar, underlying_id=args.chain_underlying, seed=args.seed
        )
    print(
        f"  chain: {len(registry)} contracts, {len(expiries)} expiries, "
        f"+/-{spec.strikes_per_side} strikes, every {spec.snapshot_seconds:g}s"
    )
    return builder, feed


def _announce_calendar(store: HolidayStore, clock: Clock) -> None:
    """Say what the calendar knows before the session, not after it fails.

    A closure discovered at 09:15 is an outage investigation. The same closure
    announced at start-up is a normal day off.
    """
    today = clock.now().date()
    print("  " + describe_day(store, today))
    nxt = next_holiday(store, today)
    if nxt is None:
        return
    days = (nxt.day - today).days
    when = "tomorrow" if days == 1 else f"in {days} days"
    flag = "  [DISPUTED - verify]" if nxt.confidence.value == "disputed" else ""
    print(f"  next closure: {nxt.day} ({when}) - {nxt.reason}{flag}")


def _synthetic_start(calendar: SessionCalendar, args: argparse.Namespace):
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    day = date.fromisoformat(args.date) if args.date else date(2026, 8, 26)
    tz = ZoneInfo(str(calendar.tz))
    return datetime.combine(day, calendar.continuous_start, tzinfo=tz)


def main() -> int:
    p = argparse.ArgumentParser(description="MarketSage market data ingest (M11 + M12 + M13)")
    p.add_argument("--source", choices=("synthetic", "kite"), default="synthetic")
    p.add_argument("--minutes", type=int, default=30, help="synthetic only: session length")
    p.add_argument("--date", help="synthetic only: session date (YYYY-MM-DD)")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--chain", action="store_true", help="also capture the option chain")
    p.add_argument("--chain-underlying", type=int, default=1)
    p.add_argument("--chain-spot", type=float, default=24_800.0, help="strike ladder centre")
    p.add_argument("--chain-seconds", type=float, default=5.0, help="snapshot cadence")
    p.add_argument("--strikes-per-side", type=int, default=15)
    p.add_argument("--verify", action="store_true", help="print the session candle report")
    p.add_argument("--metrics", action="store_true", help="print stage latencies on exit")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
