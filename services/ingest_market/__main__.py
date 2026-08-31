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
import threading
from datetime import timedelta
from pathlib import Path
from time import sleep
from types import FrameType

from libs.ms_core import config as cfgmod
from libs.ms_core import env as envmod
from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import Clock, RealClock, SessionCalendar, SimClock
from libs.ms_core.obs import METRICS, configure, get_logger, stage, trace
from libs.ms_data.candles import CandleBuilder, verify_session
from libs.ms_data.chain import (
    ChainBuilder,
    ChainSpec,
    OptionInstrument,
    OptionRegistry,
    build_registry_from_specs,
    check_chain,
)
from libs.ms_data.feed import (
    FeedHealth,
    MarketFeed,
    SyntheticFeed,
    SyntheticInstrument,
)
from libs.ms_data.holidays import HolidayStore, describe_day, next_holiday
from libs.ms_data.recorder import DATA_ROOT, Recorder
from libs.ms_data.synthetic_chain import SyntheticChainFeed, strike_grid, weekly_expiries

log = get_logger("ingest_market")

# The strike ladder in strikes, which is the unit its failure is actually
# measured in. A first attempt expressed the tolerance as a percentage of the
# index and set it at 5%; the 625-point miss that motivated the check is 2.58%
# of NIFTY, so the guard sailed past the one case it was written for. The
# condition below is derived from the geometry instead of picked.
STRIKE_STEP = 50.0
REGISTRY_STRIKES_PER_SIDE = 25
MAX_CLOSE_STALENESS_DAYS = 5
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


# Angel quotes an index on the cash segment and its options on the F&O segment.
# ChainBuilder needs both from one stream: a spot joined afterwards carries a lag
# that cannot be undone and reads as edge in every options backtest.
ANGEL_INDEX_SEGMENT = "NSE_CM"
ANGEL_OPTION_SEGMENT = "NFO"


def build_angel_chain(
    args: argparse.Namespace, calendar: SessionCalendar, clock: Clock, master=None
):
    """Chain builder and token plan, from the real Angel scrip master.

    Returns ``(builder, index_token, option_tokens)``. Unlike the synthetic path
    this invents no instrument ids: every id here is the vendor's own token,
    which is what its ticks will arrive stamped with.
    """
    from libs.ms_data.angelone import load_scrip_master

    # Injected in tests. A default of "download it" is right for the service and
    # wrong for a test, which would then need a network and today's publication.
    if master is None:
        master = load_scrip_master(clock)
    today = calendar.local(clock.now()).date()

    index = master.resolve(args.chain_name, as_of=today)
    rows = master.option_contracts(
        args.chain_name, as_of=today, n_expiries=args.n_expiries, exch_seg=ANGEL_OPTION_SEGMENT
    )
    contracts = []
    for r in rows:
        right = r.opt_type
        if right is None or r.expiry is None:
            continue
        contracts.append(
            OptionInstrument(
                instrument_id=r.token,
                underlying_id=index.token,
                expiry=r.expiry,
                strike=r.strike,
                opt_type=right,
            )
        )
    registry = OptionRegistry(contracts)
    spec = ChainSpec(
        underlying_id=index.token,
        n_expiries=args.n_expiries,
        strikes_per_side=args.strikes_per_side,
        snapshot_seconds=args.chain_seconds,
    )
    builder = ChainBuilder(spec, registry, calendar)
    expiries = sorted({c.expiry for c in contracts})
    print(
        f"  chain: {args.chain_name} index token {index.token}, {len(contracts)} contracts, "
        f"{len(expiries)} expiries ({', '.join(str(e) for e in expiries)}), "
        f"+/-{spec.strikes_per_side} strikes at snapshot, every {spec.snapshot_seconds:g}s"
    )
    return builder, index.token, [c.instrument_id for c in contracts]


def build_angel_feed(clock: Clock, *, index_token: int, option_tokens: list[int]) -> MarketFeed:
    from libs.ms_data.angelone import AngelCredentials, AngelOneFeed, login

    envmod.require_broker_credentials()
    creds = AngelCredentials.from_env()
    session = login(creds, clock)
    feed = AngelOneFeed(session, creds.api_key, exchange=ANGEL_INDEX_SEGMENT)
    feed.subscribe_on(ANGEL_INDEX_SEGMENT, [index_token])
    feed.subscribe_on("NSE_FO", option_tokens)
    log.info(
        "angel session ready",
        fields={"client": session.client_code, "tokens": len(option_tokens) + 1},
    )
    return feed


# How many snapshots the chain may miss before the recording is called dead.
# Twelve at the default five-second cadence is a minute - derived from the
# cadence rather than picked, so changing the cadence carries it along.
MISSED_SNAPSHOTS_BEFORE_HALT = 12


def _session_is_open(health: FeedHealth) -> bool:
    """Whether the exchange is trading, off the health object's own calendar."""
    from libs.ms_core.clock import SessionState

    try:
        return health.calendar.state(health.clock.now()) is SessionState.OPEN
    except LookupError:
        # No calendar for that year. `FeedHealth.evaluate` refuses to infer a
        # state it cannot justify and so does this: a watchdog that halts on a
        # missing holiday file would kill every session of an unloaded year.
        return False


def _start_feed_watchdog(
    feed: MarketFeed,
    health: FeedHealth,
    *,
    clock: Clock,
    chain_builder: ChainBuilder | None = None,
    every: float = 15.0,
):
    """Evaluate feed health on a timer, not on tick arrival.

    `health.evaluate()` was called from inside the ingest loop, every 5,000
    ticks. That is the one place it cannot do its job: the condition it exists
    to detect is *no ticks arriving*, and with none arriving the counter never
    advances and the check never runs.

    It happened on the first live session. The Angel socket dropped at 10:02,
    the vendor library retried until 10:08 and gave up, and the process then sat
    in `stream()`'s poll loop drawing no CPU for twenty-four minutes while the
    market traded. Nothing raised. The staleness halt was configured, correct,
    and unreachable.

    The watchdog closes the feed rather than setting `_stop`, and the difference
    matters: `_stop` is only tested inside the `for` loop body, which a silent
    feed never enters. `feed.close()` ends the generator itself, so the `finally`
    block runs and the session's buffers reach the disk.

    **Read the measurement, never the latched state.** `FeedHealth.state` does
    not come back from HALTED - `on_message` recovers STALE, CONNECTING and
    DISCONNECTED and deliberately not that one, because a halt is meant to
    require an explicit decision to lift. Correct for a trading halt, and a trap
    for a watchdog: polling `evaluate()` turned the first three-second lull of a
    session into a permanent kill. Run 2 of the supervised recording died after
    5.8 minutes reporting `silence_seconds: 0.0`, which is what a one-way latch
    looks like from the outside. The latch is not the bug and must not be
    unlatched to suit this loop; reading `silence_seconds()` is the fix.

    **Feed health alone is not enough, and the second stall proved it.** It asks
    whether *any* message arrived. When the socket dropped at 10:58 the vendor
    library kept retrying and 1,864 ticks trickled through over the next
    twenty-one minutes - enough to keep `on_message` fresh past a three-second
    threshold most of the time, so the halt only fired once the library gave up
    entirely. The index had stopped, so not one chain snapshot was produced in
    those twenty-one minutes. A feed can be alive in aggregate and dead in the
    one instrument the recording is about, and only the chain's own cadence can
    tell the difference.
    """

    stall_after = (
        MISSED_SNAPSHOTS_BEFORE_HALT * chain_builder.snapshot_seconds
        if chain_builder is not None
        else None
    )

    def chain_is_stalled() -> float | None:
        """Seconds since the last snapshot, once that is past the threshold."""
        if chain_builder is None or stall_after is None:
            return None
        last = chain_builder.last_snapshot_at
        if last is None:
            return None  # the session has not produced its first snapshot yet
        behind = (clock.now() - last).total_seconds()
        return behind if behind > stall_after else None

    def loop() -> None:
        while not _stop:
            sleep(every)
            behind = chain_is_stalled()
            if behind is not None:
                log.error(
                    "no chain snapshot past the threshold - closing the recording",
                    fields={"behind_seconds": round(behind, 1), "stall_after": stall_after},
                )
                print(
                    f"\n  HALTED: no option chain snapshot for {behind:.0f}s while the "
                    "session is open, though the feed is still delivering something. "
                    "The recording has been closed cleanly; restart it."
                )
                feed.close()
                return
            silence = health.silence_seconds()
            health.evaluate()  # keeps the metric and the log line, decides nothing here
            # Silence only means something while the market is open. `evaluate`
            # checked that and this loop stopped calling it for its verdict, so
            # the check has to be made here - dropping it fired the watchdog
            # before the open, which a test caught immediately.
            if not _session_is_open(health):
                continue
            if silence is not None and silence > health.staleness_halt_seconds:
                log.error(
                    "feed silent past the halt threshold - closing the recording",
                    fields={
                        "silence_seconds": round(silence, 1),
                        "halt_after": health.staleness_halt_seconds,
                    },
                )
                print(
                    f"\n  HALTED: no market data for "
                    f"{round(health.silence_seconds() or 0.0):.0f}s while the session is "
                    "open. The recording has been closed cleanly; restart it."
                )
                feed.close()
                return

    thread = threading.Thread(target=loop, name="feed-watchdog", daemon=True)
    thread.start()
    return thread


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
    elif args.source == "angel":
        clock = RealClock()
        end = None
    else:
        clock = RealClock()
        feed = build_kite_feed(clock)
        end = None

    if args.source == "angel":
        # The registry is built before the feed, not after: its tokens are what
        # the subscription is made of, and they come from the vendor's master
        # rather than being invented the way the synthetic path can afford to.
        if not args.chain:
            return _refuse_angel_without_chain()
        chain_builder, index_token, option_tokens = build_angel_chain(args, calendar, clock)
        chain_feed = None
        feed = build_angel_feed(clock, index_token=index_token, option_tokens=option_tokens)
        args.chain_underlying = index_token
    else:
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

    # Synthetic runs drive the clock from their own ticks, so silence there is
    # the replay ending rather than a feed failing, and a watchdog would only
    # ever fire spuriously.
    if args.source != "synthetic":
        _start_feed_watchdog(feed, health, clock=clock, chain_builder=chain_builder)

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


def _refuse_angel_without_chain() -> int:
    """The one combination that would waste a session.

    Ticks alone can be replayed into candles later; a chain snapshot cannot be
    reconstructed from them, because the book at a given instant is not in the
    tick stream once the instant has passed. Running the live feed without
    --chain spends a morning collecting the recoverable half.
    """
    print("  refusing: --source angel without --chain.")
    print("  The chain is the part that cannot be rebuilt afterwards - a bhavcopy")
    print("  gives daily settlement marks and no book, which is why M70 cannot be")
    print("  validated from the archive. Add --chain.")
    return 2


def _ladder_centre(args, today, *, store=None) -> float | None:
    """Where to centre the strike ladder, and why it is not a constant.

    This defaulted to 24,800 for a while. NIFTY closed at 24,175.65 on
    2026-08-28, so a session started without the flag would have built its
    ladder 625 points - twelve strikes - off centre, and half of it would have
    been so far out of the money that nothing usable came back. Nothing would
    have raised: the recorder would have run all day and written a full file of
    contracts nobody can price. That is a magic number doing exactly the damage
    R5 exists to prevent, in the one place where the data cannot be recaptured.

    So the default is the last close this repository actually holds, and the
    three ways it can go wrong are each refused rather than guessed at:
    no daily history, history too stale to centre on, and an explicit override
    that disagrees with the close by more than a ladder's width.
    """
    from libs.ms_data.daily import DailyStore

    bars = (store or DailyStore()).load(args.chain_name)
    last = bars[-1] if bars else None

    if args.chain_spot is not None:
        if last is not None:
            drift = abs(args.chain_spot - last.c) / STRIKE_STEP
            if drift >= REGISTRY_STRIKES_PER_SIDE:
                print(
                    f"  refusing: --chain-spot {args.chain_spot:,.0f} is {drift:.0f} strikes "
                    f"from the last close ({last.c:,.2f} on {last.day}), so the money is "
                    f"outside a grid that reaches +/-{REGISTRY_STRIKES_PER_SIDE}. The "
                    "recorder would run all day and write a file of unpriceable wings."
                )
                return None
            clipped = drift + args.strikes_per_side - REGISTRY_STRIKES_PER_SIDE
            if clipped > 0:
                print(
                    f"  WARNING: --chain-spot {args.chain_spot:,.0f} is {drift:.0f} strikes "
                    f"from the last close ({last.c:,.2f}). The ladder still covers the money "
                    f"but is clipped: roughly {clipped:.0f} of your {args.strikes_per_side} "
                    "strikes on one side fall outside the grid and will not be captured."
                )
        print(f"  chain: ladder centred on --chain-spot {args.chain_spot:,.2f}")
        return float(args.chain_spot)

    if last is None:
        print(
            f"  refusing: no daily history for {args.chain_name}, so there is nothing to "
            "centre the strike ladder on. Run `python jobs/gap_backfill.py --fetch` first, "
            "or pass --chain-spot explicitly."
        )
        return None

    staleness = (today - last.day).days
    if staleness > MAX_CLOSE_STALENESS_DAYS:
        print(
            f"  refusing: the newest {args.chain_name} close is {last.day}, {staleness} days "
            f"before {today}. Centring on it is no better than the constant this replaced. "
            "Refresh it with `python jobs/gap_backfill.py --fetch`, or pass --chain-spot "
            "explicitly."
        )
        return None

    print(f"  chain: ladder centred on {args.chain_name} close {last.c:,.2f} ({last.day})")
    return float(last.c)


def _build_chain(args, calendar, clock):
    """Wire up the chain snapshotter (M14).

    Historical option chains cannot be bought back cheaply, so the chain is
    captured from the first session, alongside ticks - not added later once the
    models need it.
    """
    if not args.chain:
        return None, None

    today = calendar.local(clock.now()).date()
    spot_guess = _ladder_centre(args, today)
    if spot_guess is None:
        return None, None
    expiries = weekly_expiries(calendar, today, count=2)
    strikes = strike_grid(spot_guess, step=STRIKE_STEP, n_per_side=REGISTRY_STRIKES_PER_SIDE)
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
    p.add_argument("--source", choices=("synthetic", "kite", "angel"), default="synthetic")
    p.add_argument("--minutes", type=int, default=30, help="synthetic only: session length")
    p.add_argument("--date", help="synthetic only: session date (YYYY-MM-DD)")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--chain", action="store_true", help="also capture the option chain")
    p.add_argument("--chain-underlying", type=int, default=1)
    p.add_argument(
        "--chain-spot",
        type=float,
        default=None,
        help="strike ladder centre; default is the last daily close, not a constant",
    )
    p.add_argument("--chain-seconds", type=float, default=5.0, help="snapshot cadence")
    p.add_argument("--strikes-per-side", type=int, default=15)
    p.add_argument("--chain-name", default="NIFTY", help="angel only: underlying name")
    p.add_argument("--n-expiries", type=int, default=2, help="angel only: expiries to subscribe")
    p.add_argument("--verify", action="store_true", help="print the session candle report")
    p.add_argument("--metrics", action="store_true", help="print stage latencies on exit")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
