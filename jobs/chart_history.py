"""Turn cached option history into the chart's data file, with a projection that was scored.

    python jobs/chart_history.py --date 2026-09-02
    python jobs/chart_history.py --date 2026-09-02 --underlying NIFTY --strikes 4

Reads what `jobs/option_history.py` cached and writes
``services/dashboard/chart_data.js``: real one-minute candles for every call and
put in a band around the money, for each underlying, plus a projection drawn
beside them.

**The projection is per contract, and that is the whole design.** The recorded
path in `jobs/chart_data.py` fits a volatility surface across the chain, which it
may do because a recorded chain is a set of quotes captured at one instant. This
data is not that: each bar is one contract's own prints inside one minute, so two
strikes carrying the same timestamp were never simultaneously available.
:class:`~libs.ms_options.surface.PriceBasis` measured what happens if you ignore
that - 36.9% of strike triples price a negative butterfly - so nothing here ever
sets two strikes against each other. Each contract's volatility is inverted from
its own close against the future's close **from the same minute**, and the
forward is diffused at that one number.

**No simulation, and none needed.** With volatility held flat, Black-76 is
monotone in the forward, so the q-th quantile of the price is exactly the price
at the q-th quantile of the forward - reflected for puts, which are decreasing in
it. `jobs/option_forecast.py` simulates because it re-reads a smile at the new
moneyness and loses that monotonicity; here the closed form is exact, instant and
deterministic, which also makes the backtest below cheap enough to run over every
minute of the day rather than a sample.

**The score is measured on the same day it is drawn on**, by walking every
minute, projecting forward, and comparing against what the contract actually did
- against the only baseline that matters, "it will be what it is now". Whatever
that comparison says is what the page prints under the chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from math import exp, sqrt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs.option_history import HISTORY_ROOT  # noqa: E402
from libs.ms_core.schemas import Candle  # noqa: E402
from libs.ms_options.pricing import (  # noqa: E402
    MINUTES_PER_YEAR,
    OptType,
    black76_price,
    implied_vol,
)

IST = ZoneInfo("Asia/Kolkata")
OUT = ROOT / "services" / "dashboard" / "chart_data.js"

RATE = 0.065

# Indian index options stop trading at the close on expiry day.
EXPIRY_TIME = (15, 30)

# Far enough ahead to be worth drawing, short enough to stay inside a session.
FORECAST_STEPS = (5, 10, 15, 20, 25, 30, 45, 60)

# Horizons the backtest scores at. Fewer than the drawn steps: the point is
# whether the method beats no-change at all, not a curve of near-identical rows.
SCORE_HORIZONS = (5, 15, 30, 60)

# Standard normal quantiles, written out rather than solved for. These five are
# the only ones the chart draws and they do not change.
Z = {"q10": -1.2816, "q25": -0.6745, "q50": 0.0, "q75": 0.6745, "q90": 1.2816}


def _expiry_dt(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=IST).replace(
        hour=EXPIRY_TIME[0], minute=EXPIRY_TIME[1]
    )


def _read(path: Path) -> list[Candle]:
    if not path.exists():
        return []
    out: list[Candle] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Candle.model_validate_json(line))
    out.sort(key=lambda c: c.ts)
    return out


def quantiles(
    f0: float, sigma: float, strike: float, t0: float, minutes: float, kind: OptType, r: float
) -> dict[str, float] | None:
    """The price distribution ``minutes`` ahead, in closed form.

    The forward is a driftless martingale - no view is being taken on direction,
    because nothing in this repository has earned one - so under the forward
    measure ``f`` is lognormal about ``f0`` with no drift beyond the Ito term.
    Volatility is this contract's own and is held flat, which is what makes the
    mapping from a forward quantile to a price quantile exact.

    A put is *decreasing* in the forward, so its upper price quantile comes from
    the forward's lower one. Mapping both the same way is a mistake that produces
    a perfectly plausible inverted band.
    """
    t1 = t0 - minutes / MINUTES_PER_YEAR
    if t1 <= 0.0 or t0 <= 0.0 or sigma <= 0.0:
        return None
    dt = t0 - t1
    drift = -0.5 * sigma * sigma * dt
    step = sigma * sqrt(dt)

    out: dict[str, float] = {}
    for name, z in Z.items():
        # Reflect for puts: the price's q-th quantile sits at the forward's
        # (1-q)-th, because the payoff runs the other way.
        zz = z if kind is OptType.CALL else -z
        f1 = f0 * exp(drift + step * zz)
        out[name] = black76_price(f1, strike, sigma, t1, r, kind)
    return out


def _anchor(bars: list[Candle], close: datetime) -> Candle:
    """The bar a projection starts from.

    On a session still running that is simply the last bar. On a finished day it
    is not: projecting forward from 15:30 puts every bar into a market that is
    shut, where nothing can trade and the forecast can never be checked. It also
    quietly set the chart's vertical scale, because an hour of diffusion into a
    closed market produces the widest band on the page.

    So on a completed day the anchor is moved back by the longest horizon drawn.
    The projection then lands **on top of the last hour that actually happened**,
    which is the only arrangement in which a reader can see for themselves
    whether it was any good. The real bars are still drawn, and still drawn
    filled; the projection is still hollow. Nothing is hidden by the overlap -
    the overlap is the point.
    """
    span = timedelta(minutes=max(FORECAST_STEPS))
    if bars[-1].ts + timedelta(minutes=min(FORECAST_STEPS)) <= close:
        return bars[-1]
    cutoff = close - span
    earlier = [c for c in bars if c.ts <= cutoff]
    return earlier[-1] if earlier else bars[0]


def _forward_at(future: list[Candle], ts: datetime) -> float | None:
    """The forward as of ``ts``, never after it.

    A bar-for-bar dict lookup would be simpler and would silently fall through on
    a minute the future did not trade, taking the option's price with it. Walking
    back to the last bar at or before ``ts`` is the point-in-time answer.
    """
    best: float | None = None
    for c in future:
        if c.ts > ts:
            break
        best = c.c
    return best


def score(
    contracts: list[dict[str, Any]],
    future: list[Candle],
    expiry: datetime,
    *,
    r: float,
) -> list[dict[str, Any]]:
    """Walk the day: project from every minute, compare against what happened.

    The baseline is the contract's price at the moment of the forecast. In a
    market anywhere near efficient that is very hard to beat, and a projection
    that cannot beat it is noise wearing a model's clothes.
    """
    rows: list[dict[str, Any]] = []
    for horizon in SCORE_HORIZONS:
        model_err: list[float] = []
        base_err: list[float] = []
        for entry in contracts:
            bars: list[Candle] = entry["bars"]
            kind = OptType.CALL if entry["right"] == "CE" else OptType.PUT
            strike = entry["strike"]
            by_ts = {c.ts: c for c in bars}
            for c in bars:
                later = by_ts.get(c.ts + timedelta(minutes=horizon))
                if later is None or c.c <= 0.0:
                    continue
                f0 = _forward_at(future, c.ts)
                if f0 is None:
                    continue
                t0 = (expiry - c.ts).total_seconds() / (365.0 * 24 * 3600)
                sigma = implied_vol(c.c, f0, strike, t0, r, kind)
                if sigma is None:
                    continue
                got = quantiles(f0, sigma, strike, t0, horizon, kind, r)
                if got is None:
                    continue
                model_err.append(abs(got["q50"] - later.c))
                base_err.append(abs(c.c - later.c))
        if not model_err:
            continue
        n = len(model_err)
        model_mae = sum(model_err) / n
        base_mae = sum(base_err) / n
        rows.append(
            {
                "minutes": horizon,
                "n": n,
                "model_mae": round(model_mae, 4),
                "baseline_mae": round(base_mae, 4),
                "delta_pct": round(100.0 * (model_mae - base_mae) / base_mae, 2),
            }
        )
    return rows


def _verdict(rows: list[dict[str, Any]]) -> str:
    """One sentence, and it has to survive being read on its own.

    Deliberately blunt about the margin as well as the direction: "beat the
    baseline" at 0.2% and at 20% are different findings, and a verdict that
    hides the number invites the first to be read as the second.
    """
    if not rows:
        return "not measurable on this day - too few paired observations"
    won = [r for r in rows if r["delta_pct"] <= 0]
    lost = [r for r in rows if r["delta_pct"] > 0]
    best = -min(r["delta_pct"] for r in rows)
    worst = -max(r["delta_pct"] for r in rows)
    if not lost:
        return f"beat the baseline at every horizon, by {worst:.1f}% to {best:.1f}%"
    if not won:
        return f"lost to the baseline at every horizon, by {-worst:.1f}% to {-best:.1f}%"
    where = ", ".join(f"{r['minutes']}m" for r in won)
    lost_where = ", ".join(f"{r['minutes']}m" for r in lost)
    return (
        f"beat the baseline at {where} by up to {best:.1f}%, lost at {lost_where} "
        f"by {-worst:.1f}% - margins this small on one day are not a result yet"
    )


def build(day: date, name: str, root: Path, *, r: float = RATE) -> dict[str, Any]:
    """One underlying's block: real candles, a projection, and the projection's score."""
    src = root / f"date={day.isoformat()}" / name
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"  no cached history for {name} on {day}. Run:\n"
            f"    python jobs/option_history.py --date {day.isoformat()} --underlying {name}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    future = _read(src / f"{manifest['future']['token']}.ndjson")
    if not future:
        raise SystemExit(f"  {name}: the cached future series is empty for {day}.")
    expiry_date = date.fromisoformat(manifest["expiry"])
    expiry = _expiry_dt(expiry_date)
    close = _expiry_dt(day)  # the session's own close, same clock time as expiry's

    loaded: list[dict[str, Any]] = []
    for c in manifest["contracts"]:
        bars = _read(src / f"{c['token']}.ndjson")
        if bars:
            loaded.append({**c, "bars": bars})
    if not loaded:
        raise SystemExit(f"  {name}: nothing cached has any bars for {day}.")

    measured = score(loaded, future, expiry, r=r)

    series: list[dict[str, Any]] = []
    for entry in loaded:
        bars = entry["bars"]
        kind = OptType.CALL if entry["right"] == "CE" else OptType.PUT
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

        anchor = _anchor(bars, close)
        f0 = _forward_at(future, anchor.ts)
        t0 = (expiry - anchor.ts).total_seconds() / (365.0 * 24 * 3600)
        sigma = implied_vol(anchor.c, f0, entry["strike"], t0, r, kind) if f0 else None

        projected: list[dict[str, Any]] = []
        if sigma is not None and f0 is not None:
            previous = anchor.c
            for step in FORECAST_STEPS:
                at = anchor.ts + timedelta(minutes=step)
                if at > close:
                    # The market is shut past here. A bar drawn at 16:00 is not a
                    # forecast of anything - nothing can trade into it - and it
                    # was setting the chart's whole vertical scale.
                    break
                got = quantiles(f0, sigma, entry["strike"], t0, step, kind, r)
                if got is None:
                    continue
                projected.append(
                    {
                        "t": at.astimezone(IST).strftime("%H:%M"),
                        "minutes_ahead": step,
                        # The open is the previous bar's centre so the projection
                        # reads as a continuation. High and low are the 10th and
                        # 90th percentile of a distribution - not a range anything
                        # traded in, which is why these bars are drawn hollow.
                        "o": round(previous, 2),
                        "c": round(got["q50"], 2),
                        "h": round(got["q90"], 2),
                        "l": round(got["q10"], 2),
                        "q25": round(got["q25"], 2),
                        "q75": round(got["q75"], 2),
                    }
                )
                previous = got["q50"]

        series.append(
            {
                "strike": entry["strike"],
                "right": entry["right"],
                "symbol": entry["symbol"],
                "token": entry["token"],
                "iv": round(sigma, 4) if sigma is not None else None,
                "anchor": anchor.ts.astimezone(IST).strftime("%H:%M"),
                "anchor_price": round(anchor.c, 2),
                "recorded": recorded,
                "projected": projected,
            }
        )

    series.sort(key=lambda s: (s["strike"], s["right"]))
    return {
        "name": name,
        "day": day.isoformat(),
        "source": "angel historical candles",
        "forward": {
            "symbol": manifest["future"]["symbol"],
            "close": manifest["future"]["close"],
            "bars": len(future),
        },
        "expiry": manifest["expiry"],
        "atm": manifest["atm"],
        "step": manifest["strike_step"],
        "as_of": future[-1].ts.astimezone(IST).strftime("%H:%M"),
        "series": series,
        "score": {
            "method": (
                "each contract's own implied volatility against the same-minute "
                "future, forward diffused driftless, repriced in closed form"
            ),
            "baseline": "the contract's price now, unchanged",
            "measured_on": day.isoformat(),
            "horizons": measured,
            "verdict": _verdict(measured),
            # Stated with the numbers because the numbers invite the wrong
            # reading. `n` counts every minute of every contract, and those
            # windows overlap almost completely - a 60-minute forecast made at
            # 10:00 and one made at 10:01 share 59 minutes of their outcome. So
            # the effective sample is nearer the number of contracts than to n,
            # and no confidence interval computed from n would mean anything.
            "caveat": (
                "one session, and the observations overlap: consecutive forecasts "
                "on the same contract share almost all of their outcome, so n is "
                "not an independent sample size and a small margin is not a finding"
            ),
            "point_estimate": (
                "the median, not the mean - MAE is minimised by the median, and "
                "under a driftless forward the two differ by the Ito term"
            ),
        },
    }


def write(blocks: list[dict[str, Any]], day: date, out: Path) -> Path:
    """Write the page's data file. Separate from `main` so `close_out` shares it.

    One writer, because the header below is the only place a reader is told what
    the two arrays mean, and a second writer is how a file acquires a version
    without one.
    """
    payload = {"day": day.isoformat(), "blocks": blocks}
    out.write_text(
        "/* Generated by jobs/chart_history.py - do not edit.\n"
        " *\n"
        " * `recorded` happened: real one-minute candles from Angel's historical\n"
        " * endpoint. `projected` did not happen. They are separate arrays on\n"
        " * purpose - merged, a projection becomes indistinguishable from history\n"
        " * the moment somebody scrolls. Each block carries the score its own\n"
        " * projection earned on its own day; see `score`.\n"
        " *\n"
        " * Written compactly, not prettily. Two underlyings of one-minute bars\n"
        " * indent to 3.2 MB, and this file is regenerated and committed every\n"
        " * session - the whitespace would be most of what the history stores.\n"
        " */\n"
        "window.CHART = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bake cached option history into the chart")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--underlying", nargs="*", default=["NIFTY", "BANKNIFTY"])
    p.add_argument("--root", default=str(HISTORY_ROOT))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    blocks = [build(args.date, name, Path(args.root)) for name in args.underlying]
    out = write(blocks, args.date, Path(args.out))

    print(f"\n  {out}")
    for b in blocks:
        n_rec = sum(len(s["recorded"]) for s in b["series"])
        n_proj = sum(len(s["projected"]) for s in b["series"])
        calls = sum(1 for s in b["series"] if s["right"] == "CE")
        puts = sum(1 for s in b["series"] if s["right"] == "PE")
        print(
            f"    {b['name']:10} {calls} calls + {puts} puts, {n_rec:,} real bars, "
            f"{n_proj} projected; forward {b['forward']['close']:,.2f}, expiry {b['expiry']}"
        )
        for row in b["score"]["horizons"]:
            print(
                f"      {row['minutes']:>3}m  n={row['n']:<6} model {row['model_mae']:>8.3f}  "
                f"baseline {row['baseline_mae']:>8.3f}  {row['delta_pct']:+.2f}%"
            )
        print(f"      verdict: {b['score']['verdict']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
