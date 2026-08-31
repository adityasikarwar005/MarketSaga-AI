"""Forecast option candles forward, and measure whether the forecast is worth it.

    python jobs/option_forecast.py --date 2026-08-31 --backtest
    python jobs/option_forecast.py --date 2026-08-31 --at 12:30 --horizon 30

**The forecast is not a view on the market.** It says: hold the volatility
surface fixed, let the clock run, and let the underlying diffuse at the
volatility the market is charging. Two of those three are honest - the clock is
certain and the surface is observed - and the third is deliberately not a
prediction: the underlying is simulated as a driftless martingale around the
forward, because nothing in this repository has earned the right to give it a
drift. Five models have been measured on real data and five have lost.

So what is left when the market direction is assumed unknowable is **theta**,
and theta is not a guess. An option loses time value on a schedule, and on the
eve of expiry that schedule is steep. That is the only thing being forecast
here, and the backtest exists to say whether it is worth anything.

**The baseline is the current price.** In a market that is anywhere near
efficient, "it will be what it is now" is extremely hard to beat, and any
forecast that cannot beat it is noise wearing a model's clothes. The backtest
scores both on the same contracts and horizons and prints which won. If the
model loses, that is the result and it gets reported as the result.

The output columns are named `forecast_*` and never `open/high/low/close`, so a
forecast cannot be mistaken for a recording in a spreadsheet.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import exp, log, sqrt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.schemas import Candle, ChainSnapshot, OptionType
from libs.ms_data.recorder import DATA_ROOT
from libs.ms_models.cv import Experiment, ExperimentLog
from libs.ms_options.pricing import MINUTES_PER_YEAR, OptType, black76_price
from libs.ms_options.surface import ExpirySlice, PriceBasis, fit_surface

IST = ZoneInfo("Asia/Kolkata")
OUT_DIR = Path("build") / "sheets"
PATHS = 4000
SEED = 20260831


def _load_snapshots(day: date, root: Path) -> list[ChainSnapshot]:
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
                continue
            if snap.quotes:
                out.append(snap)
    out.sort(key=lambda s: s.ts)
    return out


def _mid(q: Any) -> float | None:
    return q.mid


def project(
    sl: ExpirySlice,
    *,
    strike: float,
    kind: OptType,
    horizon_minutes: float,
    paths: int = PATHS,
    seed: int = SEED,
    r: float = 0.065,
) -> dict[str, float] | None:
    """The distribution of one contract's price ``horizon_minutes`` from now.

    The surface is held fixed and the forward diffuses at the volatility the
    market is charging for it. No drift: under the forward measure the forward
    is a martingale, and adding a drift would be exactly the prediction this
    module declines to make.
    """
    t0 = sl.t
    t1 = t0 - horizon_minutes / MINUTES_PER_YEAR
    if t1 <= 0.0 or t0 <= 0.0:
        return None
    f0 = sl.forward.forward
    atm_vol = sl.params.vol(0.0, t0)
    if atm_vol <= 0.0:
        return None

    dt = t0 - t1
    rng = random.Random(seed)
    drift = -0.5 * atm_vol * atm_vol * dt
    vol_step = atm_vol * sqrt(dt)

    prices = []
    for _ in range(paths):
        f1 = f0 * exp(drift + vol_step * rng.gauss(0.0, 1.0))
        k = log(strike / f1)
        # The same slice, re-read at the new moneyness. Holding the surface
        # fixed while the forward moves is the assumption; it is wrong in a
        # sell-off and is stated rather than hidden.
        vol = sl.params.vol(k, t1)
        if vol <= 0.0:
            continue
        prices.append(black76_price(f1, strike, vol, t1, r, kind))
    if len(prices) < paths // 2:
        return None

    prices.sort()

    def q(p: float) -> float:
        return prices[min(len(prices) - 1, int(p * len(prices)))]

    return {
        "median": st.median(prices),
        "mean": st.fmean(prices),
        "q10": q(0.10),
        "q90": q(0.90),
        "q25": q(0.25),
        "q75": q(0.75),
    }


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _actual_prices(day: date, root: Path, tokens: set[int]) -> dict[int, dict[datetime, float]]:
    partition = root / "candles" / f"date={day.isoformat()}"
    out: dict[int, dict[datetime, float]] = defaultdict(dict)
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
                c = Candle.model_validate_json(line)
            except Exception:
                continue
            out[token][c.ts] = c.c
    return out


def backtest(
    snapshots: list[ChainSnapshot],
    day: date,
    root: Path,
    *,
    horizons: list[int],
    paths: int = 400,
    stride: int = 5,
) -> list[dict[str, Any]]:
    """Score the projection against 'it will be what it is now', on real data."""
    by_minute = {s.ts.replace(second=0, microsecond=0): s for s in snapshots}
    # Every minute is not an independent observation - consecutive snapshots of
    # one chain overlap almost completely - and scoring all of them would cost
    # hours to inflate n without adding evidence.
    minutes = sorted(by_minute)[::stride]
    tokens = {q.instrument_id for s in snapshots for q in s.quotes if q.instrument_id}
    actual = _actual_prices(day, root, tokens)

    results: list[dict[str, Any]] = []
    for horizon in horizons:
        model_err, base_err, n = [], [], 0
        for t0 in minutes:
            t1 = t0 + timedelta(minutes=horizon)
            snap = by_minute[t0]
            try:
                surface = fit_surface(snap, now=snap.ts, basis=PriceBasis.MID)
            except Exception:
                continue
            if not surface.slices:
                continue
            expiry, sl = min(surface.slices.items(), key=lambda kv: kv[1].t)
            for q in snap.quotes:
                if q.expiry != expiry or q.instrument_id is None:
                    continue
                now_px = _mid(q)
                later = actual.get(q.instrument_id, {}).get(t1)
                if now_px is None or later is None or now_px <= 0:
                    continue
                kind = OptType.CALL if q.opt_type is OptionType.CE else OptType.PUT
                proj = project(sl, strike=q.strike, kind=kind, horizon_minutes=horizon, paths=paths)
                if proj is None:
                    continue
                model_err.append(abs(proj["median"] - later))
                base_err.append(abs(now_px - later))
                n += 1
        if n:
            results.append(
                {
                    "horizon": horizon,
                    "n": n,
                    "model_mae": st.fmean(model_err),
                    "baseline_mae": st.fmean(base_err),
                    "improvement": 1.0 - st.fmean(model_err) / st.fmean(base_err),
                }
            )
    return results


def write_forecast(
    path: Path, snap: ChainSnapshot, sl: ExpirySlice, expiry: datetime, steps: list[int]
) -> int:
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "from_time_ist",
                "for_time_ist",
                "minutes_ahead",
                "strike",
                "right",
                "expiry",
                "price_now",
                "forecast_median",
                "forecast_q10",
                "forecast_q25",
                "forecast_q75",
                "forecast_q90",
                "forecast_change",
            ]
        )
        legs = sorted(
            (q for q in snap.quotes if q.expiry == expiry),
            key=lambda q: (q.strike, q.opt_type.value),
        )
        for q in legs:
            now_px = _mid(q)
            if now_px is None:
                continue
            kind = OptType.CALL if q.opt_type is OptionType.CE else OptType.PUT
            for h in steps:
                proj = project(sl, strike=q.strike, kind=kind, horizon_minutes=h)
                if proj is None:
                    continue
                w.writerow(
                    [
                        snap.ts.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
                        (snap.ts + timedelta(minutes=h)).astimezone(IST).strftime("%Y-%m-%d %H:%M"),
                        h,
                        f"{q.strike:.0f}",
                        q.opt_type.value,
                        expiry.astimezone(IST).date().isoformat(),
                        f"{now_px:.2f}",
                        f"{proj['median']:.2f}",
                        f"{proj['q10']:.2f}",
                        f"{proj['q25']:.2f}",
                        f"{proj['q75']:.2f}",
                        f"{proj['q90']:.2f}",
                        f"{proj['median'] - now_px:+.2f}",
                    ]
                )
                rows += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Project option prices forward, and score the projection"
    )
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--at", type=time.fromisoformat, help="project from this snapshot (HH:MM)")
    p.add_argument("--horizon", type=int, nargs="*", default=[5, 15, 30, 60])
    p.add_argument("--backtest", action="store_true", help="score against the no-change baseline")
    p.add_argument("--paths", type=int, default=400, help="Monte Carlo paths per contract")
    p.add_argument("--stride", type=int, default=5, help="score every Nth minute")
    args = p.parse_args(argv)

    root, out_dir = Path(args.root), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = _load_snapshots(args.date, root)
    if not snapshots:
        raise SystemExit(f"  no recorded chain for {args.date}.")

    if args.backtest:
        print()
        print("  BACKTEST - surface repricing against 'it will be what it is now'")
        print(f"  {len(snapshots):,} snapshots on {args.date}, front expiry only, mid prices.")
        print()
        rows = backtest(
            snapshots,
            args.date,
            root,
            horizons=args.horizon,
            paths=args.paths,
            stride=args.stride,
        )
        if not rows:
            raise SystemExit("  no (snapshot, outcome) pair lined up - nothing to score.")
        print(f"  {'horizon':>8}{'n':>9}{'model MAE':>12}{'baseline MAE':>14}{'better by':>11}")
        for r in rows:
            print(
                f"  {r['horizon']:>6}m{r['n']:>9,}{r['model_mae']:>12.3f}"
                f"{r['baseline_mae']:>14.3f}{r['improvement']:>10.1%}"
            )
        print()
        # A measured model goes in the trial log whether it won or lost. The log
        # exists to keep the trial count honest, and a result recorded only when
        # it is flattering makes every deflation computed from it a decoration.
        cfg = cfgmod.load(warn=False)
        log = ExperimentLog()
        for r in rows:
            log.record(
                Experiment(
                    name=f"option_forecast_surface_repricing_{r['horizon']}m",
                    ts=datetime.now(IST).isoformat(),
                    config_hash=cfg.config_hash,
                    dataset_version=f"live_chain_{args.date.isoformat()}",
                    feature_hash="none - repriced from the fitted surface, no feature vector",
                    metrics={
                        "n": float(r["n"]),
                        "model_mae": r["model_mae"],
                        "baseline_mae": r["baseline_mae"],
                        "improvement_vs_no_change": r["improvement"],
                    },
                    notes=(
                        "Surface held fixed, forward diffused driftless, options repriced. "
                        "Baseline is the current mid. One partial session of real NIFTY "
                        "option data."
                    ),
                )
            )
        print(f"  recorded {len(rows)} trial(s) in the experiment log.")
        print()

        won = [r for r in rows if r["improvement"] > 0]
        if len(won) == len(rows):
            print("  The projection beat the baseline at every horizon tested.")
        elif won:
            print(f"  The projection beat the baseline at {len(won)} of {len(rows)} horizons.")
        else:
            print("  The projection LOST to the baseline at every horizon. That is the result.")
        print("  MAE is in rupees per contract, on mid prices.")
        return 0

    snap = snapshots[-1]
    if args.at is not None:
        cutoff = datetime.combine(args.date, args.at, tzinfo=IST)
        eligible = [s for s in snapshots if s.ts.astimezone(IST) <= cutoff]
        if not eligible:
            raise SystemExit(f"  no snapshot at or before {args.at}.")
        snap = eligible[-1]

    surface = fit_surface(snap, now=snap.ts, basis=PriceBasis.MID)
    if not surface.slices:
        raise SystemExit("  no expiry had enough usable quotes to fit.")
    expiry, sl = min(surface.slices.items(), key=lambda kv: kv[1].t)

    path = out_dir / f"option_forecast_{args.date.isoformat()}.csv"
    n = write_forecast(path, snap, sl, expiry, args.horizon)
    print()
    print(f"  FORECAST - {path}")
    print(f"    {n:,} rows, from the {snap.ts.astimezone(IST):%H:%M} snapshot.")
    print("    Columns are forecast_*, never open/high/low/close: nothing here happened.")
    print()
    print("    Run --backtest before believing any of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
