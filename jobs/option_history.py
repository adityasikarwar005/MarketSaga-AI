"""Fetch real one-minute option candles from Angel's historical endpoint, and cache them.

    python jobs/option_history.py --date 2026-09-02
    python jobs/option_history.py --date 2026-09-02 --underlying NIFTY --strikes 8

Why this exists rather than reading the recorder's own files: the recorder has
run for exactly one session, on 2026-08-31, on NIFTY only. This endpoint holds a
rolling ~30 days of one-minute history for **every** listed contract, which is
where BankNifty comes from and where any day the recorder missed comes from -
including 2026-09-01 and 2026-09-02, which it missed entirely.

**It is history, not a chain.** What comes back is one series per contract, each
bar being that contract's own prints inside that minute. Two contracts' bars
carrying the same timestamp were not quoted at the same instant, and an illiquid
strike's close can be minutes stale. That is the exact hazard
:class:`~libs.ms_options.surface.PriceBasis` documents: 36.9% of strike triples
built from same-day last-traded marks price a negative butterfly. So this cache
is for **drawing one contract at a time**, and for per-contract arithmetic that
never sets two strikes against each other. Do not fit a smile on it.

The front-month future is fetched alongside, and it is the useful one: it gives a
genuine continuous forward series, so a contract's implied volatility can be
inverted against the forward from *the same minute* rather than against a spot
joined in from somewhere else.

**Everything is cached and nothing is re-fetched.** The endpoint's throttle
answers with an empty body and stays angry for half an hour, so a run that gets
blocked must not lose the contracts it already had. Re-running is free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libs.ms_core import dotenv  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.schemas import Candle  # noqa: E402
from libs.ms_data.angelone import (  # noqa: E402
    AngelAuthError,
    AngelCredentials,
    CandleInterval,
    ScripRow,
    historical_candles,
    load_scrip_master,
    login,
)

IST = ZoneInfo("Asia/Kolkata")
HISTORY_ROOT = ROOT / "data" / "raw" / "history"

# The throttle is undocumented, answers with silence, and is stricter than any
# steady pace worth waiting for: 0.5s blocked inside six requests and 2.5s
# blocked inside seven. Pacing alone does not solve it.
#
# What is true, and is not in the vendor's documentation either, is that the
# block *lifts*. Both blocks above cleared within seconds - the half-hour outage
# recorded in docs/model_audit.md followed four logins rather than a run of
# candle requests, so logging in is the expensive act and this reuses a cached
# session. So the strategy is a modest pace plus a real backoff, and the thing
# that makes backoff safe is that every completed contract is already on disk.
PACE_SECONDS = 1.5
BACKOFF_START = 20.0
BACKOFF_MAX = 120.0
BACKOFF_ATTEMPTS = 5

SESSION_OPEN = (9, 15)
SESSION_CLOSE = (15, 30)


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    base = datetime.combine(day, datetime.min.time(), tzinfo=IST)
    return (
        base.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1]),
        base.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1]),
    )


def strike_step(strikes: list[float]) -> float:
    """The grid's own spacing, taken as the commonest gap.

    Not a constant: NIFTY lists on 50s and BankNifty on 100s, and both change.
    The commonest gap survives the wings, where exchanges list sparsely.
    """
    gaps = Counter(round(b - a, 4) for a, b in pairwise(strikes) if b > a)
    return float(gaps.most_common(1)[0][0]) if gaps else 50.0


def front_future(master: Any, name: str, day: date) -> ScripRow:
    futs = sorted(
        (
            r
            for r in master.rows
            if r.name == name
            and r.instrument_type == "FUTIDX"
            and r.expiry is not None
            and r.expiry >= day
        ),
        key=lambda r: (r.expiry, r.token),
    )
    if not futs:
        raise KeyError(f"no live {name} index future listed on or after {day}")
    return futs[0]


def _cached(path: Path) -> list[Candle] | None:
    if not path.exists():
        return None
    out: list[Candle] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Candle.model_validate_json(line))
    return out


def _write(path: Path, bars: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(b.model_dump_json(by_alias=True) + "\n" for b in bars), encoding="utf-8"
    )


def fetch_series(
    session: Any,
    api_key: str,
    row: ScripRow,
    day: date,
    out_dir: Path,
    *,
    pace: float = PACE_SECONDS,
) -> tuple[list[Candle], bool]:
    """One contract's minute bars, from the cache if it is there.

    Returns the bars and whether the network was touched, so a caller can report
    how much of a run was free.
    """
    path = out_dir / f"{row.token}.ndjson"
    hit = _cached(path)
    if hit is not None:
        return hit, False

    start, end = _day_bounds(day)
    bars = historical_candles(
        session,
        api_key,
        token=int(row.token),
        exchange=row.exch_seg,
        interval=CandleInterval.ONE_MINUTE,
        start=start,
        end=end,
    )
    # Written even when empty. A listed strike that never traded is a fact about
    # the day, and re-asking the endpoint for it every run is how a cache turns
    # into a rate limit.
    _write(path, bars)
    time.sleep(pace)
    return bars, True


def fetch_with_backoff(
    session: Any,
    api_key: str,
    row: ScripRow,
    day: date,
    out_dir: Path,
    *,
    attempts: int = BACKOFF_ATTEMPTS,
) -> tuple[list[Candle], bool]:
    """:func:`fetch_series`, waiting out the throttle rather than abandoning the run.

    Backing off is only safe because the cache makes a retry idempotent: nothing
    already on disk is re-requested, so a blocked run resumes where it stopped
    instead of starting the whole band again and re-earning the block.

    A non-throttle failure is re-raised untouched. Waiting is the response to
    being told to wait, not a general-purpose way of handling errors.
    """
    delay = BACKOFF_START
    for attempt in range(1, attempts + 1):
        try:
            return fetch_series(session, api_key, row, day, out_dir)
        except AngelAuthError as exc:
            if "rate limiting" not in str(exc) or attempt == attempts:
                raise
            print(f"    throttled on {row.symbol}; waiting {delay:.0f}s (attempt {attempt})")
            time.sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX)
    raise AssertionError("unreachable")


def collect(
    session: Any,
    api_key: str,
    master: Any,
    name: str,
    day: date,
    *,
    n_strikes: int,
    root: Path,
) -> dict[str, Any]:
    """The front future and a band of contracts around the money, cached to disk."""
    out_dir = root / f"date={day.isoformat()}" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fut = front_future(master, name, day)
    fut_bars, _ = fetch_series(session, api_key, fut, day, out_dir)
    if not fut_bars:
        raise SystemExit(
            f"  {name}: the front future returned no bars for {day}. Without a "
            "forward series there is nothing to invert a volatility against, and "
            "the index itself has no candles on this endpoint."
        )
    level = fut_bars[-1].c

    contracts = master.option_contracts(name, as_of=day, n_expiries=1)
    expiry = contracts[0].expiry
    grid = sorted({r.strike for r in contracts})
    step = strike_step(grid)
    atm = min(grid, key=lambda k: abs(k - level))
    lo, hi = atm - n_strikes * step, atm + n_strikes * step
    wanted = {k for k in grid if lo - 1e-6 <= k <= hi + 1e-6}

    kept: list[dict[str, Any]] = []
    fetched = 0
    for row in contracts:
        if row.strike not in wanted:
            continue
        try:
            bars, did = fetch_with_backoff(session, api_key, row, day, out_dir)
        except AngelAuthError as exc:
            print(f"  {name}: giving up on {row.symbol}, keeping what is cached - {exc}")
            break
        fetched += int(did)
        if not bars:
            continue  # listed but never traded; a fact, not a failure
        kept.append(
            {
                "token": int(row.token),
                "symbol": row.symbol,
                "strike": row.strike,
                "right": "CE" if row.symbol.endswith("CE") else "PE",
                "bars": len(bars),
            }
        )

    manifest = {
        "underlying": name,
        "day": day.isoformat(),
        "expiry": expiry.isoformat() if expiry else None,
        "future": {
            "token": int(fut.token),
            "symbol": fut.symbol,
            "expiry": fut.expiry.isoformat() if fut.expiry else None,
            "bars": len(fut_bars),
            "close": round(level, 2),
        },
        "strike_step": step,
        "atm": atm,
        "contracts": sorted(kept, key=lambda c: (c["strike"], c["right"])),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"  {name}: front future {fut.symbol} closed {level:,.2f}; expiry "
        f"{manifest['expiry']}; {len(kept)} contracts with bars "
        f"({fetched} fetched, {len(kept) - fetched} already cached); strike step {step:g}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cache real option candles from Angel history")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--underlying", nargs="*", default=["NIFTY", "BANKNIFTY"])
    p.add_argument("--strikes", type=int, default=5, help="each side of the money")
    p.add_argument("--root", default=str(HISTORY_ROOT))
    args = p.parse_args(argv)

    dotenv.load()
    clock = RealClock()
    master = load_scrip_master(clock, refresh=False)
    creds = AngelCredentials.from_env()
    session = login(creds, clock)

    print(f"\n  {args.date}: {2 * args.strikes + 1} strikes each side of the money\n")
    for name in args.underlying:
        collect(
            session,
            creds.api_key,
            master,
            name,
            args.date,
            n_strikes=args.strikes,
            root=Path(args.root),
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
