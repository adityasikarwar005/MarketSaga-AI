"""Per-strike option candles and the market's own implied outlook, as CSV.

    python jobs/option_sheet.py --date 2026-08-31
    python jobs/option_sheet.py --date 2026-08-31 --strike 24200
    python jobs/option_sheet.py --date 2026-08-31 --out build/sheets

Writes two files. They answer different questions and must not be read as one.

**`option_candles_<date>.csv`** is what was recorded: one row per contract per
minute, open/high/low/close/volume/open-interest, with the vendor token resolved
to a strike, a right and an expiry. This is the same series a chart draws for
24200 CE, and every number in it happened.

**`option_outlook_<date>.csv`** is what the market was *pricing* at the last
snapshot: per strike, the mid, the fitted implied volatility, and the
risk-neutral probability of the underlying finishing above that strike at the
front expiry.

**That probability is not a forecast, and the distinction is the whole reason
this file is separate.** It is the probability implied by option prices under
the risk-neutral measure, which is the real-world probability distorted by
whatever risk premium the market is charging - reliably so, and in a direction
nobody can measure from prices alone. It says what the market is charging, not
what will happen.

There is deliberately no third file of predicted candles. Nothing in this
repository has earned the right to produce one: five models have been measured
on real data and all five lost to a simpler baseline. A file of future
open/high/low/close values would be indistinguishable, in a spreadsheet, from
the recorded ones - and that is exactly why it is not written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from math import erf, log, sqrt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.schemas import Candle, ChainSnapshot
from libs.ms_data.recorder import DATA_ROOT
from libs.ms_options.surface import PriceBasis, fit_surface

IST = ZoneInfo("Asia/Kolkata")
OUT_DIR = Path("build") / "sheets"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def load_chain(day: date, root: Path) -> list[ChainSnapshot]:
    partition = root / "chain" / f"date={day.isoformat()}"
    out: list[ChainSnapshot] = []
    for path in sorted(partition.glob("chain-*.ndjson")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                snap = ChainSnapshot.model_validate_json(line)
            except Exception:
                continue  # a half-written final line while the recorder runs
            if snap.quotes:
                out.append(snap)
    return out


def contract_index(snapshots: list[ChainSnapshot]) -> dict[int, dict[str, Any]]:
    """Vendor token -> strike, right, expiry, taken from the day's own chain.

    The recording stamps Angel's token on every candle and nothing in the
    research layer translates it, so the chain is the only dictionary available -
    and the right one, since it came from the same session.
    """
    index: dict[int, dict[str, Any]] = {}
    for snap in snapshots:
        for q in snap.quotes:
            if q.instrument_id is not None and q.instrument_id not in index:
                index[q.instrument_id] = {
                    "strike": q.strike,
                    "right": q.opt_type.value,
                    "expiry": q.expiry.astimezone(IST).date().isoformat(),
                }
    return index


def load_candles(day: date, root: Path, tokens: set[int]) -> dict[int, list[Candle]]:
    partition = root / "candles" / f"date={day.isoformat()}"
    out: dict[int, list[Candle]] = defaultdict(list)
    for path in sorted(partition.glob("*.ndjson")):
        try:
            token = int(path.stem.removeprefix("candles-"))
        except ValueError:
            continue
        if token not in tokens:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out[token].append(Candle.model_validate_json(line))
            except Exception:
                continue
    for series in out.values():
        series.sort(key=lambda c: c.ts)
    return out


def write_candles(
    path: Path, candles: dict[int, list[Candle]], index: dict[int, dict[str, Any]]
) -> int:
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "time_ist",
                "strike",
                "right",
                "expiry",
                "token",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "trades",
                "vwap",
                "synthetic",
            ]
        )
        for token, series in sorted(
            candles.items(), key=lambda kv: (index[kv[0]]["strike"], kv[0])
        ):
            meta = index[token]
            for c in series:
                w.writerow(
                    [
                        c.ts.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
                        f"{meta['strike']:.0f}",
                        meta["right"],
                        meta["expiry"],
                        token,
                        f"{c.o:.2f}",
                        f"{c.h:.2f}",
                        f"{c.low:.2f}",
                        f"{c.c:.2f}",
                        c.v,
                        "" if c.oi is None else c.oi,
                        "" if c.trades is None else c.trades,
                        "" if c.vwap is None else f"{c.vwap:.4f}",
                        int(c.synthetic),
                    ]
                )
                rows += 1
    return rows


def write_outlook(path: Path, snapshot: ChainSnapshot) -> tuple[int, dict[str, Any]]:
    """Per strike: the market's mid, its fitted vol, and what it implies."""
    surface = fit_surface(snapshot, now=snapshot.ts, basis=PriceBasis.MID)
    if not surface.slices:
        raise SystemExit("  no expiry had enough usable quotes to fit - nothing to imply.")

    expiry, sl = min(surface.slices.items(), key=lambda kv: kv[1].t)
    f, t = sl.forward.forward, sl.t
    legs = {
        (q.strike, q.opt_type.value): q
        for q in snapshot.quotes
        if q.expiry.astimezone(IST).date() == expiry.astimezone(IST).date()
    }
    strikes = sorted({k for k, _ in legs})

    atm = min(strikes, key=lambda k: abs(k - f))
    call, put = legs.get((atm, "CE")), legs.get((atm, "PE"))
    straddle = (
        (call.mid + put.mid)
        if call and put and call.mid is not None and put.mid is not None
        else None
    )

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strike",
                "call_bid",
                "call_ask",
                "call_oi",
                "put_bid",
                "put_ask",
                "put_oi",
                "fitted_iv",
                "rn_prob_above",
                "rn_prob_below",
            ]
        )
        for k in strikes:
            vol = sl.params.vol(log(k / f), t)
            if vol <= 0.0 or t <= 0.0:
                continue
            d2 = (log(f / k) - 0.5 * vol * vol * t) / (vol * sqrt(t))
            above = _norm_cdf(d2)
            c, p = legs.get((k, "CE")), legs.get((k, "PE"))
            w.writerow(
                [
                    f"{k:.0f}",
                    "" if not c or c.bid is None else f"{c.bid:.2f}",
                    "" if not c or c.ask is None else f"{c.ask:.2f}",
                    "" if not c or c.oi is None else c.oi,
                    "" if not p or p.bid is None else f"{p.bid:.2f}",
                    "" if not p or p.ask is None else f"{p.ask:.2f}",
                    "" if not p or p.oi is None else p.oi,
                    f"{vol:.4f}",
                    f"{above:.4f}",
                    f"{1.0 - above:.4f}",
                ]
            )
            rows += 1

    return rows, {
        "expiry": expiry.astimezone(IST).date().isoformat(),
        "spot": snapshot.spot,
        "forward": f,
        "years_to_expiry": t,
        "atm_strike": atm,
        "atm_straddle": straddle,
        "implied_move_pct": None if straddle is None else straddle / snapshot.spot,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Option candles and implied outlook as CSV")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--strike", type=float, help="only this strike, both rights")
    args = p.parse_args(argv)

    root, out_dir = Path(args.root), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_chain(args.date, root)
    if not snapshots:
        raise SystemExit(f"  no recorded chain for {args.date} under {root}.")
    index = contract_index(snapshots)

    tokens = set(index)
    if args.strike is not None:
        tokens = {t for t, m in index.items() if m["strike"] == args.strike}
        if not tokens:
            listed = sorted({m["strike"] for m in index.values()})
            raise SystemExit(
                f"  strike {args.strike:.0f} is not in the recording. "
                f"Recorded: {listed[0]:.0f}..{listed[-1]:.0f}"
            )

    candles = load_candles(args.date, root, tokens)
    candles = {t: s for t, s in candles.items() if s}
    suffix = f"_{args.strike:.0f}" if args.strike is not None else ""
    candle_path = out_dir / f"option_candles_{args.date.isoformat()}{suffix}.csv"
    n_candles = write_candles(candle_path, candles, index)

    outlook_path = out_dir / f"option_outlook_{args.date.isoformat()}.csv"
    n_strikes, meta = write_outlook(outlook_path, snapshots[-1])

    print()
    print(f"  RECORDED - {candle_path}")
    print(f"    {n_candles:,} minute candles across {len(candles)} contract(s).")
    print("    Every number happened.")
    print()
    print(f"  IMPLIED  - {outlook_path}")
    print(f"    {n_strikes} strikes at the {meta['expiry']} expiry, from the last snapshot.")
    print(f"    spot {meta['spot']:,.2f}   forward {meta['forward']:,.2f}")
    if meta["implied_move_pct"] is not None:
        print(
            f"    the market is charging {meta['implied_move_pct']:.2%} "
            f"({meta['atm_straddle']:,.2f} points) for the move to expiry."
        )
    print()
    print("    rn_prob_above is RISK-NEUTRAL: what the prices imply, not a forecast.")
    print("    It is the real-world probability distorted by the risk premium the")
    print("    market charges - reliably, and by an amount prices cannot reveal.")
    print()
    print("  There is no third file of predicted candles, and there will not be one")
    print("  until something here has beaten a baseline on real data. Five have been")
    print("  measured and five have lost.")
    print(json.dumps({"candles": n_candles, "strikes": n_strikes}, indent=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
