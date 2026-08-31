"""What the recorder is holding right now, as JSON, for the live dashboard.

    python jobs/live_state.py                     print today's state
    python jobs/live_state.py --date 2026-08-31   a specific day

Read-only, and deliberately so: the recorder owns those files and this process
must never be a reason the recording stops. It seeks to the tail rather than
reading whole partitions, because by the close the tick file for a single index
is tens of megabytes and a dashboard polling every few seconds would otherwise
spend the afternoon re-reading the morning.

**Instrument ids here are the vendor's tokens, not the config's.** A live Angel
session stamps every record with Angel's own token - 26000 for the NIFTY index -
while `config/instruments.yaml` calls that instrument 1. `build_angel_chain`
says so in its docstring and it is the right call for a recorder: inventing an
id would mean translating every tick at the point where mistakes are most
expensive. But nothing downstream translates either, so a research job asked for
instrument 1 finds an empty day and reports it as one. This module resolves the
token off the filenames on disk for exactly that reason, and the seam is
recorded in `docs/model_audit.md` rather than papered over here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import RealClock, SessionCalendar
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.holidays import HolidayStore
from libs.ms_data.recorder import DATA_ROOT
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.registry import REGISTRY
from libs.ms_features.sources import OfflineSource

IST = ZoneInfo("Asia/Kolkata")

# How much of each tail to read. Ticks need enough history for the longest
# candle window in the catalogue (31 minutes) with room to spare; the chain
# needs one snapshot, but a snapshot of 124 quotes is about 40 KB, so a small
# read would return a fragment and no complete line at all.
TICK_TAIL_BYTES = 6_000_000
CHAIN_TAIL_BYTES = 2_000_000


def tail_lines(path: Path, max_bytes: int) -> list[str]:
    """The last complete lines of a file the recorder is still appending to.

    Public because `jobs/surface_report.py` reads the same partitions for M70,
    and a second copy of this would be a second place for the off-by-one at the
    seek boundary to be got wrong.

    The first line of a mid-file seek is almost always a fragment and is
    dropped. So is the last, whenever the file does not end in a newline -
    the recorder may be mid-write, and half a JSON object parses as nothing
    useful but raises in a way that reads like corruption.
    """
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            partial = True
        else:
            partial = False
        raw = fh.read()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    if partial and lines:
        lines = lines[1:]
    return [line for line in lines if line.strip()]


def _load(path: Path, cls: type[Any], max_bytes: int) -> list[Any]:
    out = []
    for line in tail_lines(path, max_bytes):
        try:
            out.append(cls.model_validate_json(line))
        except Exception:
            # A truncated final line is expected, not exceptional. Anything
            # else is a real decode problem and shows up as a gap in the count
            # the dashboard prints, which is where it belongs.
            continue
    return out


def discover_token(root: Path, day: date) -> int | None:
    """The underlying token, taken from the chain partition's own filenames."""
    partition = root / "chain" / f"date={day.isoformat()}"
    if not partition.exists():
        return None
    tokens = []
    for path in partition.glob("chain-*.ndjson"):
        try:
            tokens.append(int(path.stem.removeprefix("chain-")))
        except ValueError:
            continue
    return max(tokens) if tokens else None


def build_state(day: date, *, root: Path = DATA_ROOT) -> dict[str, Any]:
    now = datetime.now(IST)
    state: dict[str, Any] = {
        "day": day.isoformat(),
        "generated_at": now.isoformat(),
        "recording": False,
        "token": None,
        "spot": None,
        "features": {},
        "chain": {},
        "health": {},
        "notes": [],
    }

    token = discover_token(root, day)
    if token is None:
        state["notes"].append("no chain partition for this day - the recorder has not written one")
        return state
    state["token"] = token

    chain_path = root / "chain" / f"date={day.isoformat()}" / f"chain-{token}.ndjson"
    tick_path = root / "ticks" / f"date={day.isoformat()}" / f"ticks-{token}.ndjson"

    snaps: list[ChainSnapshot] = _load(chain_path, ChainSnapshot, CHAIN_TAIL_BYTES)
    ticks: list[Tick] = _load(tick_path, Tick, TICK_TAIL_BYTES)
    if not snaps:
        state["notes"].append("chain partition exists but holds no complete snapshot yet")
        return state

    latest = snaps[-1]
    state["spot"] = latest.spot
    state["recording"] = (now - latest.ts.astimezone(IST)) < timedelta(minutes=2)

    # Candles are rebuilt from ticks rather than read off disk. On a live
    # session the candle files are the thinnest stream the recorder has - a few
    # rows each across hundreds of files - so they are the last thing to reach
    # the disk, and a dashboard that waited for them would show nothing until
    # the first flush.
    cfg = cfgmod.load()
    # `with_holidays` is not optional. The bare calendar refuses to guess a year
    # it has no circular for, which is correct - and it means a calendar built
    # without the holiday store raises on the first tick of any real session.
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )
    builder = CandleBuilder(calendar)
    candles: list[Candle] = []
    for tick in sorted(ticks, key=lambda t: t.ts):
        candles.extend(builder.on_tick(tick))

    source = OfflineSource(candles=candles, chains=snaps)
    as_of = latest.ts + timedelta(seconds=1)
    values = REGISTRY.compute(source, token, as_of, underlying_id=token)
    state["features"] = {
        name: {"value": values.get(name), "group": REGISTRY.get(name).group}
        for name in REGISTRY.names()
    }

    quotes = latest.quotes
    strikes = sorted({q.strike for q in quotes})
    expiries = sorted({q.expiry.astimezone(IST).date().isoformat() for q in quotes})
    state["chain"] = {
        "ts": latest.ts.astimezone(IST).isoformat(),
        "completeness": latest.completeness,
        "quotes": len(quotes),
        "strikes": len(strikes),
        "strike_low": min(strikes) if strikes else None,
        "strike_high": max(strikes) if strikes else None,
        "expiries": expiries,
        "missing_iv": sum(1 for q in quotes if q.iv is None),
        "missing_book": sum(1 for q in quotes if q.bid is None or q.ask is None),
        "crossed": sum(1 for q in quotes if q.bid and q.ask and q.bid > q.ask),
        "ladder": _ladder(latest),
    }

    state["health"] = {
        "snapshots_in_tail": len(snaps),
        "ticks_in_tail": len(ticks),
        "candles_built": len(candles),
        "chain_age_seconds": (now - latest.ts.astimezone(IST)).total_seconds(),
        "chain_bytes": chain_path.stat().st_size if chain_path.exists() else 0,
        "tick_bytes": tick_path.stat().st_size if tick_path.exists() else 0,
        "candle_files": _candle_file_stats(root, day),
    }
    return state


def _ladder(snap: ChainSnapshot) -> list[dict[str, Any]]:
    """The front-expiry strikes around the money, calls and puts side by side."""
    if not snap.quotes:
        return []
    front = min(q.expiry for q in snap.quotes)
    legs = [q for q in snap.quotes if q.expiry == front]
    strikes = sorted({q.strike for q in legs})
    if not strikes:
        return []
    atm = min(strikes, key=lambda k: abs(k - snap.spot))
    idx = strikes.index(atm)
    window = strikes[max(0, idx - 7) : idx + 8]

    rows = []
    for strike in window:
        row: dict[str, Any] = {"strike": strike, "atm": strike == atm}
        for q in legs:
            if q.strike != strike:
                continue
            side = "call" if q.opt_type.value == "CE" else "put"
            row[side] = {
                "bid": q.bid,
                "ask": q.ask,
                "ltp": q.ltp,
                "oi": q.oi,
                "iv": q.iv,
                "volume": q.volume,
            }
        rows.append(row)
    return rows


def _candle_file_stats(root: Path, day: date) -> dict[str, int]:
    partition = root / "candles" / f"date={day.isoformat()}"
    if not partition.exists():
        return {"files": 0, "empty": 0}
    files = list(partition.glob("*.ndjson"))
    return {
        "files": len(files),
        "empty": sum(1 for f in files if f.stat().st_size == 0),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live recorder state as JSON (M91)")
    p.add_argument("--date", help="day to read (YYYY-MM-DD); default today IST")
    p.add_argument("--root", default=str(DATA_ROOT))
    args = p.parse_args(argv)

    day = date.fromisoformat(args.date) if args.date else RealClock().now().astimezone(IST).date()
    print(json.dumps(build_state(day, root=Path(args.root)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
