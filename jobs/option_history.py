"""Fetch real one-minute option candles from Angel's historical endpoint, and cache them.

    python jobs/option_history.py --date 2026-09-03
    python jobs/option_history.py --date 2026-09-03 --underlying NIFTY --strikes 8 --expiries 2

**The default is every listed contract.** Not a band around the money and not
the front expiry: every call and every put that is live on ``--date``, across
all eighteen NIFTY expiries out to 2031 and all six of BankNifty's. That is
1568 and 882 contracts respectively, and each one is its own request - there is
no batch form of this endpoint - so a full run is on the wire for about two
hours. It is resumable, and re-running is free for everything already cached.

``--strikes`` and ``--expiries`` narrow it again when a quick run is wanted.

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
from libs.ms_core.clock import Clock, RealClock  # noqa: E402
from libs.ms_core.schemas import Candle  # noqa: E402
from libs.ms_data.angelone import (  # noqa: E402
    AngelCredentials,
    CandleInterval,
    ScripRow,
    historical_candles,
    load_scrip_master,
    login,
)

IST = ZoneInfo("Asia/Kolkata")
HISTORY_ROOT = ROOT / "data" / "raw" / "history"

# The throttle is undocumented, answers with silence, and - measured rather than
# assumed this time - is **not a function of the pace at all**. Four blocks of 25
# requests, spaced 0.0s, 0.4s, 0.8s and 1.2s apart on a live session, returned
# 11, 16, 14 and 12 successes. Waiting longer between requests bought nothing;
# it just made the run take longer to collect the same rejections.
#
# That changes what the right strategy is. This is not a budget that refills
# over tens of seconds - it is a per-request rejection that clears immediately,
# so the answer is to ask again promptly rather than to stand back. The old
# policy stood back 20 seconds on the first throttle, which at an observed ~50%
# rejection rate is 20s of waiting per contract: 1568 contracts would have been
# nine hours. Retrying at 0.5s gets the same contracts in about 2.8s each.
#
# What has *not* changed is why backing off is safe at all: every completed
# contract is already on disk, so an interrupted run resumes rather than
# restarts. And the half-hour outage in docs/model_audit.md followed four
# logins, not a run of candle requests - logging in is the expensive act, and
# this reuses a cached session.
PACE_SECONDS = 0.4

# Two waits, because these are two different problems and one policy serves
# neither. Both are now set from the measurement above rather than from a guess
# about what a throttle is.
#
# The throttle clears at once, so it gets many quick attempts: at a ~50%
# rejection rate, two attempts is the common case and eleven is a contract that
# is genuinely not coming. The ceiling stays low on purpose - the whole point of
# the change is that standing back does not help, so standing back further
# cannot help more. A dropped connection is the opposite: it usually clears on
# the next attempt but occasionally needs a real pause, so it keeps the growing
# wait. Each kind keeps its own budget, so a run that meets one blip and one
# throttle has not used up either allowance.
#
#   kind        first wait  ceiling  attempts
BACKOFF = {
    "throttled": (0.5, 8.0, 12),
    "network": (2.0, 30.0, 4),
}

# How long a cached series for a session that is **still running** stays good.
# The cache exists so a re-run does not re-earn the block, and for a finished
# day that is unconditional: the bars cannot change. For today they can - the
# day is still being written - and a cache with no expiry would freeze the whole
# board at whatever minute the first run happened to reach it, which looks
# exactly like a market that stopped trading.
TODAY_CACHE_SECONDS = 300.0

# A contract that will not come is skipped, not fatal - the chain is worth more
# with a hole in it than not at all. But contract after contract failing is not
# bad luck with individual contracts, it is the session: an expired token, a
# dropped link, a venue that has stopped answering. Skipping past that is how a
# job reports success having fetched nothing.
CONSECUTIVE_FAILURE_LIMIT = 5

SESSION_OPEN = (9, 15)
SESSION_CLOSE = (15, 30)


def _day_bounds(day: date, clock: Clock | None = None) -> tuple[datetime, datetime]:
    """The window to ask for: the open, and the close or now, whichever is sooner.

    Asking a live session for bars up to 15:30 is not an error - the endpoint
    returns what exists - but it makes every contract's request the same shape
    whether the day is over or not, and that is what lets a run started at 10:30
    and one started at 14:00 be told apart by nothing but their contents.
    """
    base = datetime.combine(day, datetime.min.time(), tzinfo=IST)
    opened = base.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1])
    closed = base.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1])
    if clock is not None:
        now = clock.now().astimezone(IST)
        if now < closed:
            return opened, max(opened, now)
    return opened, closed


def strike_step(strikes: list[float]) -> float:
    """The grid's own spacing, taken as the commonest gap.

    Not a constant: NIFTY lists on 50s and BankNifty on 100s, and both change.
    The commonest gap survives the wings, where exchanges list sparsely.
    """
    gaps = Counter(round(b - a, 4) for a, b in pairwise(strikes) if b > a)
    return float(gaps.most_common(1)[0][0]) if gaps else 50.0


def index_futures(master: Any, name: str, day: date) -> list[ScripRow]:
    """Every listed index future for one underlying, nearest expiry first.

    The front one used to be enough because only the front option expiry was
    fetched. Across eighteen expiries it is not: a December option's forward is
    not September's future, and pricing it against September understates the
    forward by the whole carry - which comes back as an implied volatility that
    is wrong in a direction that looks like a smile.
    """
    return sorted(
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


def front_future(master: Any, name: str, day: date) -> ScripRow:
    futs = index_futures(master, name, day)
    if not futs:
        raise KeyError(f"no live {name} index future listed on or after {day}")
    return futs[0]


def forward_for(futures: list[ScripRow], expiry: date) -> tuple[ScripRow, str]:
    """The future to price one option expiry against, and how honest the match is.

    ``same-cycle`` is a future expiring on the option's own day - the forward is
    then read straight off a traded price and nothing is assumed. ``bracketing``
    is the next future after the option, which is the ordinary case for a weekly:
    the carry between them is days, and small.

    ``carried`` is the one to be careful with. NIFTY lists options out to 2031
    and futures three months out, so a 2029 option has no future behind it at
    all and its forward has to be extended from the last one at the risk-free
    rate. That is an assumption about dividends as much as about rates, and over
    three years it is a large one. It is labelled rather than hidden, and the
    page declines to draw a projection on a carried forward - the candles are
    real either way, and those are what was asked for.
    """
    if not futures:
        raise KeyError("no index future to price against")
    same = [f for f in futures if f.expiry == expiry]
    if same:
        return same[0], "same-cycle"
    after = [f for f in futures if f.expiry and f.expiry >= expiry]
    if after:
        return after[0], "bracketing"
    return futures[-1], "carried"


def _stale(path: Path, day: date, clock: Clock | None, *, max_age: float) -> bool:
    """Whether a cached series has to be asked for again.

    The question is not "is the session over now" - it is **"was the session over
    when this file was written"**. Those look like the same question and are not,
    and getting them confused leaves a day permanently half-read.

    A file written after its own session's close holds every bar that session
    will ever have. Nothing can change it and re-asking for it is how a cache
    turns into a rate limit, so it is never stale - whatever day it is for, and
    whenever it is read.

    A file written *during* the session it describes is a partial day by
    construction. While that session is still running it goes stale on age, so a
    sweep taken at 11:00 does not freeze the whole board at 11:00. Once the
    session has closed it is stale unconditionally: the day it holds three hours
    of is now a day with six and a quarter, and a post-close sweep exists exactly
    to replace it. Without that clause a board swept during trading hours stays
    ragged for good, because every one of its files became "a finished day" the
    moment the bell rang.

    The age is the **file's**, not the last bar's. A contract that has not traded
    since 09:20 and a file last written at 09:20 hold identical bars; one is a
    stale read and the other is a quiet strike, and nothing in the series tells
    them apart.
    """
    if clock is None:
        return False
    now = clock.now().astimezone(IST)
    _, close = _day_bounds(day)
    written = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
    if written >= close:
        return False
    if now >= close:
        return True
    return (now.timestamp() - path.stat().st_mtime) > max_age


def _read_at(path: Path) -> str | None:
    """The minute a cached series was written, as exchange-local clock time."""
    if not path.exists():
        return None
    at = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
    return at.strftime("%H:%M")


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
    clock: Clock | None = None,
    max_age: float = TODAY_CACHE_SECONDS,
) -> tuple[list[Candle], bool]:
    """One contract's minute bars, from the cache if it is there and still good.

    Returns the bars and whether the network was touched, so a caller can report
    how much of a run was free.
    """
    path = out_dir / f"{row.token}.ndjson"
    hit = _cached(path)
    if hit is not None and not _stale(path, day, clock, max_age=max_age):
        return hit, False

    start, end = _day_bounds(day, clock)
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


def transient(exc: BaseException) -> str | None:
    """What kind of "try again" this is, or None if it is not one.

    Two things interrupt a long fetch and neither is a reason to lose it. The
    throttle is the expected one. The other is the network: a run of 454
    contracts spends half an hour on the wire and a single `ConnectTimeout`
    killed one at contract 247, discarding nothing already cached but ending the
    run in a traceback - which under the scheduler is a morning with no charts.

    Matched on the message and the exception's name rather than by importing the
    vendor SDK's exception tree, which is not ours and has changed before.
    Anything not recognised is re-raised untouched: waiting is the answer to
    being told to wait, not a general-purpose way of handling errors.
    """
    text = str(exc).lower()
    if "rate limiting" in text or "exceeding access rate" in text:
        return "throttled"
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or "timed out" in text:
        return "network"
    return None


def fetch_with_backoff(
    session: Any,
    api_key: str,
    row: ScripRow,
    day: date,
    out_dir: Path,
    *,
    policy: dict[str, tuple[float, float, int]] | None = None,
    sleep: Any = time.sleep,
    clock: Clock | None = None,
) -> tuple[list[Candle], bool]:
    """:func:`fetch_series`, waiting out an interruption rather than abandoning the run.

    Backing off is only safe because the cache makes a retry idempotent: nothing
    already on disk is re-requested, so an interrupted run resumes where it
    stopped instead of starting the whole band again and re-earning the block.

    Each kind of interruption keeps its own attempt count and its own delay, so
    meeting a throttle does not spend the network's allowance or start it at the
    throttle's much longer wait. Anything :func:`transient` does not recognise is
    raised immediately - waiting is the answer to being told to wait, and a
    retry that cannot tell a wrong token from a slow one turns a clear failure
    into a slow one.
    """
    rules = BACKOFF if policy is None else policy
    tries: dict[str, int] = {}
    delay: dict[str, float] = {}
    while True:
        try:
            return fetch_series(session, api_key, row, day, out_dir, clock=clock)
        except Exception as exc:
            kind = transient(exc)
            if kind is None or kind not in rules:
                raise
            first, ceiling, limit = rules[kind]
            tries[kind] = tries.get(kind, 0) + 1
            if tries[kind] >= limit:
                raise
            delay[kind] = first if kind not in delay else min(delay[kind] * 2, ceiling)
            if delay[kind] >= 2.0:
                # Sub-second retries are the expected case now, not an event.
                # Printing all eleven of them for each of 1568 contracts buries
                # the run's actual failures under thirty thousand lines.
                print(
                    f"    {kind} on {row.symbol}; waiting {delay[kind]:.0f}s "
                    f"({tries[kind]} of {limit - 1})"
                )
            sleep(delay[kind])


def _selected(
    contracts: list[ScripRow],
    expiries: list[date],
    level: float,
    *,
    n_strikes: int | None,
) -> list[tuple[date, float, float, list[ScripRow]]]:
    """Group the universe by expiry, with each expiry's own grid and money.

    The strike grid is taken **per expiry**, not once. They genuinely differ: the
    front weekly lists 86 strikes on 50s from 22150, and the September monthly
    lists 123 on a grid that widens out at 34500. One step taken from the front
    expiry and applied to all of them puts the far months' wings on rungs that
    were never listed.
    """
    by_expiry: dict[date, list[ScripRow]] = {}
    for r in contracts:
        if r.expiry is not None:
            by_expiry.setdefault(r.expiry, []).append(r)

    out: list[tuple[date, float, float, list[ScripRow]]] = []
    for e in expiries:
        rows = by_expiry.get(e, [])
        if not rows:
            continue
        grid = sorted({r.strike for r in rows})
        step = strike_step(grid)
        atm = min(grid, key=lambda k: abs(k - level))
        if n_strikes is None:
            wanted = set(grid)
        else:
            lo, hi = atm - n_strikes * step, atm + n_strikes * step
            wanted = {k for k in grid if lo - 1e-6 <= k <= hi + 1e-6}
        keep = sorted(
            (r for r in rows if r.strike in wanted),
            key=lambda r: (r.strike, r.symbol),
        )
        out.append((e, atm, step, keep))
    return out


def collect(
    session: Any,
    api_key: str,
    master: Any,
    name: str,
    day: date,
    *,
    n_strikes: int | None,
    n_expiries: int | None,
    root: Path,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Every live call and put on one underlying, cached to disk, expiry by expiry.

    Ordered nearest expiry first and, inside an expiry, outward from the money,
    because a run this long will sometimes be cut short and what it should have
    finished first is what somebody is going to look at first. A run that stops
    halfway through 2027 has still delivered everything anybody traded today.
    """
    out_dir = root / f"date={day.isoformat()}" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = index_futures(master, name, day)
    if not futures:
        raise SystemExit(f"  {name}: no live index future listed on or after {day}.")

    fut_series: dict[int, list[Candle]] = {}
    fut_meta: list[dict[str, Any]] = []

    def refresh_futures() -> None:
        """Re-read the forward series, and do it again between expiries.

        Fetched once at the top of a sweep, the forward stops where the sweep
        started. That was harmless when a run was twenty minutes; over four
        hours it is not. An option bar at 14:00 would be inverted against the
        last forward bar at or before it - which would be the 10:30 close, three
        hours and however many index points stale - and the implied volatility
        that comes back from that is wrong by the whole move, on every contract
        in every expiry the sweep reached after the first.

        Three requests an expiry is nothing against 2450, and on a finished day
        it costs nothing at all: the cache is unconditional there, so this
        returns immediately without touching the network.
        """
        fut_meta.clear()
        for f in futures:
            try:
                bars, _ = fetch_with_backoff(session, api_key, f, day, out_dir, clock=clock)
            except Exception as exc:
                print(f"  {name}: future {f.symbol} did not come - {exc}")
                continue
            fut_series[int(f.token)] = bars
            if bars:
                fut_meta.append(
                    {
                        "token": int(f.token),
                        "symbol": f.symbol,
                        "expiry": f.expiry.isoformat() if f.expiry else None,
                        "bars": len(bars),
                        "close": round(bars[-1].c, 2),
                    }
                )

    refresh_futures()

    front = futures[0]
    if not fut_series.get(int(front.token)):
        raise SystemExit(
            f"  {name}: the front future returned no bars for {day}. Without a "
            "forward series there is nothing to invert a volatility against, and "
            "the index itself has no candles on this endpoint."
        )
    level = fut_series[int(front.token)][-1].c

    contracts = master.option_contracts(name, as_of=day, n_expiries=n_expiries or 10_000)
    expiries = sorted({r.expiry for r in contracts if r.expiry})
    groups = _selected(contracts, expiries, level, n_strikes=n_strikes)
    total = sum(len(rows) for _, _, _, rows in groups)
    print(
        f"  {name}: front future {front.symbol} at {level:,.2f}; "
        f"{total} contracts across {len(groups)} expiries"
    )

    blocks: list[dict[str, Any]] = []
    done = 0
    consecutive = 0
    stopped = False

    def checkpoint() -> dict[str, Any]:
        """Write the manifest for what is finished so far.

        Called after each expiry rather than once at the end. A sweep of every
        listed contract runs for hours at this endpoint's pace, and a job that
        produces its only output on the last line is a job whose morning cannot
        be looked at until the afternoon. Expiries are fetched nearest-first, so
        a manifest written after the third one already describes everything
        anybody is trading today, and the packer downstream reads it happily -
        it has no opinion about how many expiries a manifest ought to contain.
        """
        at = max((b[-1].ts for b in fut_series.values() if b), default=None)
        m = {
            "underlying": name,
            "day": day.isoformat(),
            "as_of": at.astimezone(IST).strftime("%H:%M") if at else None,
            "complete": not stopped and len(blocks) == len(groups),
            "futures": fut_meta,
            "expiries": blocks,
        }
        (out_dir / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m

    for expiry, atm, step, rows in groups:
        if blocks:
            refresh_futures()  # see refresh_futures: the forward must keep up
        # Outward from the money. See the docstring: this order is what a run
        # that gets cut short is worth.
        ordered = sorted(rows, key=lambda r: (abs(r.strike - atm), r.strike, r.symbol))
        kept: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        fetched = 0
        for row in ordered:
            if stopped:
                break
            try:
                bars, did = fetch_with_backoff(session, api_key, row, day, out_dir, clock=clock)
            except Exception as exc:
                # One contract that will not come is a hole in the chain, not the
                # end of it. The run continues, the failure is recorded in the
                # manifest so the page can say the chain is short, and the next
                # contract gets its own chance - the first version of this
                # abandoned four hundred contracts because the two hundred and
                # forty-seventh timed out.
                consecutive += 1
                failed.append(
                    {"token": int(row.token), "symbol": row.symbol, "why": str(exc)[:180]}
                )
                print(f"  {name}: {row.symbol} did not come, moving on - {exc}")
                if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                    print(
                        f"  {name}: {consecutive} in a row. That is the session, not "
                        "the contracts. Stopping with what is cached."
                    )
                    stopped = True
                continue
            consecutive = 0
            fetched += int(did)
            done += 1
            if done % 100 == 0:
                print(f"    {name}: {done}/{total} contracts")
            if not bars:
                continue  # listed but never traded; a fact, not a failure
            kept.append(
                {
                    "token": int(row.token),
                    "symbol": row.symbol,
                    "strike": row.strike,
                    "right": "CE" if row.symbol.endswith("CE") else "PE",
                    "bars": len(bars),
                    # When this contract was read, which is not when it last
                    # traded and is not recoverable from its bars. A far wing
                    # that printed once at 09:16 and never again has a last bar
                    # at 09:16 whether it was read at 09:20 or at 14:00, and a
                    # header that took the earliest last-bar for "when the
                    # sweep started" reported 09:15 on a sweep that began at
                    # 11:00. Taken from the file rather than the clock so a
                    # contract served from the cache reports the run that
                    # actually fetched it.
                    "read_at": _read_at(out_dir / f"{row.token}.ndjson"),
                }
            )

        fwd, basis = forward_for(futures, expiry)
        fwd_bars = fut_series.get(int(fwd.token)) or []
        blocks.append(
            {
                "expiry": expiry.isoformat(),
                "atm": atm,
                "strike_step": step,
                "listed": len(ordered),
                "forward": {
                    "token": int(fwd.token),
                    "symbol": fwd.symbol,
                    "expiry": fwd.expiry.isoformat() if fwd.expiry else None,
                    "close": round(fwd_bars[-1].c, 2) if fwd_bars else None,
                    "bars": len(fwd_bars),
                    # See forward_for: "carried" means no future stands behind
                    # this expiry at all and its forward is an extrapolation.
                    "basis": basis,
                },
                "contracts": sorted(kept, key=lambda c: (c["strike"], c["right"])),
                # Recorded rather than dropped: a chain with a hole in it and a
                # chain that never had those strikes look identical from the page.
                "failed": failed,
            }
        )
        checkpoint()
        print(
            f"    {name} {expiry}: {len(kept)}/{len(ordered)} traded "
            f"({fetched} fetched); step {step:g}; forward {fwd.symbol} ({basis})"
            + (f"; {len(failed)} could not be fetched" if failed else "")
        )
        if stopped:
            # The brake is about the session, not this expiry. Falling through
            # to the next one would append a block per remaining expiry, each
            # empty, each printed - fourteen lines saying nothing happened,
            # which reads as fourteen quiet expiries rather than as one stop.
            break

    refresh_futures()  # once more, so the cached forward spans the whole sweep
    manifest = checkpoint()
    traded = sum(len(b["contracts"]) for b in blocks)
    print(f"  {name}: {traded} of {total} contracts traded today; manifest written\n")
    return manifest


def count_or_all(value: str) -> int | None:
    """``all`` or a count. ``all`` is the default, and is what the page needs."""
    if value.lower() == "all":
        return None
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive count, or 'all'")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cache real option candles from Angel history")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--underlying", nargs="*", default=["NIFTY", "BANKNIFTY"])
    p.add_argument(
        "--strikes",
        type=count_or_all,
        default=None,
        help="each side of the money, or 'all' (default)",
    )
    p.add_argument(
        "--expiries", type=count_or_all, default=None, help="nearest N expiries, or 'all' (default)"
    )
    p.add_argument("--root", default=str(HISTORY_ROOT))
    args = p.parse_args(argv)

    dotenv.load()
    clock = RealClock()
    master = load_scrip_master(clock, refresh=False)
    creds = AngelCredentials.from_env()
    session = login(creds, clock)

    scope = "every listed strike" if args.strikes is None else f"{2 * args.strikes + 1} strikes"
    span = "every live expiry" if args.expiries is None else f"the nearest {args.expiries} expiries"
    print(f"\n  {args.date}: {scope} across {span}\n")
    for name in args.underlying:
        collect(
            session,
            creds.api_key,
            master,
            name,
            args.date,
            n_strikes=args.strikes,
            n_expiries=args.expiries,
            root=Path(args.root),
            clock=clock,
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
