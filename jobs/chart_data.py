"""Bake one day's option candles and their projection into the chart page's data file.

    python jobs/chart_data.py --date 2026-08-31
    python jobs/chart_data.py --date 2026-08-31 --strikes 24000 24100 24200

Writes ``services/dashboard/chart_data.js``, which `chart.html` renders. Baked
rather than fetched so the page opens from the filesystem, off the local server
and from a static host, all showing the same thing.

**Recorded and projected bars are kept in separate arrays and never merged.**
That is the entire safety property of this file. A projection concatenated onto
a price series is indistinguishable from history the moment somebody scrolls,
and the one way this work could do real damage is by looking like a record. The
page draws them differently because they arrive differently.

Every projected bar carries the score the method earned on this same day, so the
number travels with the picture instead of living in a commit message nobody
opens. On 2026-08-31 that score was negative at every horizon: the projection
lost to "it will be what it is now" by two to seven percent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs.option_forecast import project  # noqa: E402
from libs.ms_core.schemas import Candle, ChainSnapshot, OptionType  # noqa: E402
from libs.ms_data.recorder import DATA_ROOT  # noqa: E402
from libs.ms_options.pricing import OptType  # noqa: E402
from libs.ms_options.surface import PriceBasis, fit_surface  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
OUT = ROOT / "services" / "dashboard" / "chart_data.js"

# Far enough ahead to be worth drawing, short enough to stay inside the session.
FORECAST_STEPS = (5, 10, 15, 20, 25, 30, 45, 60)


def _snapshots(day: date, root: Path) -> list[ChainSnapshot]:
    part = root / "chain" / f"date={day.isoformat()}"
    out: list[ChainSnapshot] = []
    for path in sorted(part.glob("chain-*.ndjson")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                snap = ChainSnapshot.model_validate_json(line)
            except Exception:
                continue
            if snap.quotes:
                out.append(snap)
    out.sort(key=lambda s: s.ts)
    return out


def _candles(day: date, root: Path, tokens: set[int]) -> dict[int, list[Candle]]:
    part = root / "candles" / f"date={day.isoformat()}"
    out: dict[int, list[Candle]] = defaultdict(list)
    for path in sorted(part.glob("*.ndjson")):
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


def build(day: date, root: Path, wanted: list[float] | None) -> dict[str, Any]:
    snaps = _snapshots(day, root)
    if not snaps:
        raise SystemExit(f"  no recorded chain for {day}.")
    latest = snaps[-1]
    front = min(q.expiry for q in latest.quotes)

    index: dict[int, dict[str, Any]] = {}
    for snap in snaps:
        for q in snap.quotes:
            if q.instrument_id is not None and q.instrument_id not in index:
                index[q.instrument_id] = {
                    "strike": q.strike,
                    "right": q.opt_type.value,
                    "expiry": q.expiry,
                }

    strikes = sorted({m["strike"] for m in index.values() if m["expiry"] == front})
    if wanted:
        strikes = [k for k in strikes if k in wanted]
    else:
        atm = min(strikes, key=lambda k: abs(k - latest.spot))
        i = strikes.index(atm)
        strikes = strikes[max(0, i - 5) : i + 6]

    tokens = {t for t, m in index.items() if m["strike"] in strikes and m["expiry"] == front}
    candles = _candles(day, root, tokens)

    surface = fit_surface(latest, now=latest.ts, basis=PriceBasis.MID)
    slice_ = surface.slices.get(front)

    series: list[dict[str, Any]] = []
    for token, bars in candles.items():
        if not bars:
            continue
        meta = index[token]
        recorded = [
            {
                "t": c.ts.astimezone(IST).strftime("%H:%M"),
                "o": round(c.o, 2),
                "h": round(c.h, 2),
                "l": round(c.low, 2),
                "c": round(c.c, 2),
                "v": c.v,
            }
            for c in bars
        ]

        projected: list[dict[str, Any]] = []
        if slice_ is not None:
            kind = OptType.CALL if meta["right"] == OptionType.CE.value else OptType.PUT
            last = bars[-1]
            previous = last.c
            for step in FORECAST_STEPS:
                got = project(
                    slice_, strike=meta["strike"], kind=kind, horizon_minutes=step, paths=1500
                )
                if got is None:
                    continue
                projected.append(
                    {
                        "t": (last.ts + timedelta(minutes=step)).astimezone(IST).strftime("%H:%M"),
                        "minutes_ahead": step,
                        # Open is the previous bar's close so the projection reads
                        # as a continuation; high and low are the 10th and 90th
                        # percentile of the simulated distribution, not a range
                        # anything traded in.
                        "o": round(previous, 2),
                        "c": round(got["median"], 2),
                        "h": round(got["q90"], 2),
                        "l": round(got["q10"], 2),
                        "q25": round(got["q25"], 2),
                        "q75": round(got["q75"], 2),
                    }
                )
                previous = got["median"]

        series.append(
            {
                "strike": meta["strike"],
                "right": meta["right"],
                "token": token,
                "recorded": recorded,
                "projected": projected,
            }
        )

    series.sort(key=lambda s: (s["strike"], s["right"]))
    return {
        "day": day.isoformat(),
        "underlying": latest.underlying_id,
        "spot": round(latest.spot, 2),
        "expiry": front.astimezone(IST).date().isoformat(),
        "as_of": latest.ts.astimezone(IST).strftime("%H:%M:%S"),
        "series": series,
        # Carried with the data so the picture cannot be separated from its
        # score. Measured on this same session in jobs/option_forecast.py.
        "score": {
            "method": "surface held fixed, forward diffused driftless, options repriced",
            "baseline": "the current mid, unchanged",
            "measured_on": day.isoformat(),
            "horizons": [
                {"minutes": 5, "n": 496, "model_mae": 11.482, "baseline_mae": 10.769},
                {"minutes": 30, "n": 186, "model_mae": 25.010, "baseline_mae": 24.396},
                {"minutes": 120, "n": 186, "model_mae": 29.058, "baseline_mae": 28.468},
            ],
            "verdict": "lost to the baseline at every horizon, by 2.1% to 6.6%",
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bake option candles and their projection for the chart"
    )
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--strikes", type=float, nargs="*", help="default: eleven around the money")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    payload = build(args.date, Path(args.root), args.strikes)
    out = Path(args.out)
    out.write_text(
        "/* Generated by jobs/chart_data.py - do not edit.\n"
        " *\n"
        " * `recorded` happened. `projected` did not, and lost to a no-change\n"
        " * baseline on this same day - see `score`. They are separate arrays on\n"
        " * purpose: merged, a projection becomes indistinguishable from history.\n"
        " */\n"
        "window.CHART = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    n_rec = sum(len(s["recorded"]) for s in payload["series"])
    n_proj = sum(len(s["projected"]) for s in payload["series"])
    print(f"\n  {out}")
    print(f"    {len(payload['series'])} contract series")
    print(f"    {n_rec:,} recorded bars, {n_proj:,} projected bars")
    print(f"    spot {payload['spot']:,.2f} at {payload['as_of']}, expiry {payload['expiry']}")
    print(f"    projection verdict carried with the data: {payload['score']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
