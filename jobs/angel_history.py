"""M11 - Angel historical candles: cross-check the scale, then backfill.

    python jobs/angel_history.py --cross-check
    python jobs/angel_history.py --symbol NIFTY --interval FIVE_MINUTE --days 30

**Why the cross-check exists.** The websocket sends prices in paise and this
endpoint sends them in rupees. Nothing in the adapter can catch a mix-up between
the two: NIFTY at 24,180 and the same figure read as paise, 241.80, are both
inside the plausibility band, and every downstream volatility, moneyness and
strike distance stays internally consistent while being wrong by two orders of
magnitude. That is this project's recurring failure shape, and the guard that
usually catches it is blind here.

What is available instead is a second opinion. This repository already holds ten
years of NIFTY daily bars from an unrelated provider. Fetching Angel's own daily
series over the same span and comparing them is the only evidence available that
the scale is what the docstring claims. It costs one request and settles the
question once rather than leaving it assumed forever.

**Why intraday matters more than the scale question.** M53 is an intraday
direction model and there is no intraday history, so it has only ever been
measured on daily analogues - which is not what G5 asks. Free intraday is a
rolling window: roughly 30 days at one minute, 60 at three, 100 at five. It is
not recoverable later, which makes it the one thing here worth collecting
before it is needed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.angelone import (  # noqa: E402
    AngelCredentials,
    CandleInterval,
    InstrumentMaster,
    historical_candles,
    load_scrip_master,
    login,
)
from libs.ms_data.daily import DailyStore  # noqa: E402

# Below this the two series are the same numbers; above it they are not, and
# nothing between is a rounding difference between two honest providers.
SCALE_TOLERANCE = 0.02


def resolve_token(master: InstrumentMaster, symbol: str, *, as_of) -> int:
    row = master.resolve(symbol, as_of=as_of)
    return row.token


def cross_check(session, api_key: str, *, symbol: str, token: int, days: int) -> int:
    """Angel's daily closes against the ones already on disk.

    Matched on date, so a holiday either side simply drops out of the
    comparison rather than shifting one series against the other.
    """
    ours = {b.day: b.c for b in DailyStore().load(symbol)}
    if not ours:
        print(f"  no local daily bars for {symbol!r} to check against.")
        print("  Fetch them first: python jobs/gap_backfill.py --fetch")
        return 1

    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    theirs = historical_candles(
        session,
        api_key,
        token=token,
        exchange="NSE_CM",
        interval=CandleInterval.ONE_DAY,
        start=start,
        end=end,
    )
    if not theirs:
        print("  Angel returned no daily candles for that range.")
        return 1

    pairs = [(c, ours[c.ts.date()]) for c in theirs if c.ts.date() in ours]
    print(
        f"  {len(theirs):,} Angel daily candles, {len(ours):,} local bars, "
        f"{len(pairs):,} dates in both"
    )
    if not pairs:
        print("  no overlapping dates - nothing to compare.")
        return 1

    ratios = [c.c / local for c, local in pairs if local]
    median = statistics.median(ratios)
    worst = max(ratios, key=lambda r: abs(r - 1.0))
    print(f"  close ratio Angel/local: median {median:.4f}, furthest from 1 {worst:.4f}")
    print()

    if abs(median - 1.0) <= SCALE_TOLERANCE:
        print("  SCALE CONFIRMED: this endpoint returns rupees.")
        print("  Two independent providers agree on the level, which is the only")
        print("  evidence available that the paise/rupees question is settled -")
        print("  the plausibility band cannot tell 24,180 from 241.80.")
        rc = 0
    elif abs(median - 100.0) <= 100 * SCALE_TOLERANCE:
        print("  SCALE WRONG: the series is 100x the local one, i.e. PAISE.")
        print("  HISTORICAL_PRICES_ARE_RUPEES in libs/ms_data/angelone.py is false")
        print("  and every candle fetched through it needs dividing by 100.")
        rc = 1
    else:
        print(f"  SCALE UNEXPLAINED: median ratio {median:.4f} is neither 1 nor 100.")
        print("  Do not backfill through this path until it is understood; a")
        print("  consistent wrong factor is the shape that survives every")
        print("  downstream check.")
        rc = 1

    sample = pairs[-1]
    print()
    print(
        f"  most recent matched date {sample[0].ts.date()}: "
        f"Angel {sample[0].c:,.2f}  local {sample[1]:,.2f}"
    )
    return rc


def backfill(
    session, api_key: str, *, symbol: str, token: int, interval: CandleInterval, days: int
) -> int:
    end = datetime.now(UTC)
    start = end - timedelta(days=min(days, interval.max_days))
    if days > interval.max_days:
        print(f"  {days} days requested; {interval.value} allows {interval.max_days}.")
        print("  Narrowed rather than silently truncated - the API answers an")
        print("  over-long range with fewer bars, which reads as a quiet period.")
    candles = historical_candles(
        session,
        api_key,
        token=token,
        exchange="NSE_CM",
        interval=interval,
        start=start,
        end=end,
    )
    if not candles:
        print("  no candles returned for that range.")
        return 1

    span = candles[-1].ts - candles[0].ts
    print(
        f"  {len(candles):,} candles, {candles[0].ts.isoformat()} .. "
        f"{candles[-1].ts.isoformat()} ({span.days} days)"
    )
    print(
        f"  first: o={candles[0].o:,.2f} h={candles[0].h:,.2f} "
        f"l={candles[0].low:,.2f} c={candles[0].c:,.2f} v={candles[0].v:,}"
    )
    print(
        f"  last:  o={candles[-1].o:,.2f} h={candles[-1].h:,.2f} "
        f"l={candles[-1].low:,.2f} c={candles[-1].c:,.2f} v={candles[-1].v:,}"
    )
    print()
    print("  Not written to disk by this job. The recorder owns the raw store and")
    print("  its partitioning; this is the fetch path proving it works.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--cross-check", action="store_true", help="validate the price scale")
    p.add_argument("--interval", default="FIVE_MINUTE", choices=[i.value for i in CandleInterval])
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args(argv)

    configure()
    cfgmod.load(warn=False)
    clock = RealClock()

    creds = AngelCredentials.from_env()
    session = login(creds, clock)
    master = load_scrip_master(clock)
    today = clock.now().date()
    token = resolve_token(master, args.symbol, as_of=today)

    print()
    print(f"  {args.symbol} token {token}, client {session.client_code}")
    print()

    if args.cross_check:
        return cross_check(
            session, creds.api_key, symbol=args.symbol, token=token, days=max(args.days, 60)
        )
    return backfill(
        session,
        creds.api_key,
        symbol=args.symbol,
        token=token,
        interval=CandleInterval(args.interval),
        days=args.days,
    )


if __name__ == "__main__":
    raise SystemExit(main())
