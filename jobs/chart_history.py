"""Turn cached option history into the chart's data files, with a projection that was scored.

    python jobs/chart_history.py --date 2026-09-03
    python jobs/chart_history.py --date 2026-09-03 --underlying NIFTY

Reads what `jobs/option_history.py` cached and writes one file per **expiry**:
real one-minute candles for every call and every put that traded, plus a
projection drawn beside them, plus the score that projection earned on that day.

**One file per expiry, not per underlying.** NIFTY lists eighteen live expiries
and 1568 contracts; a full session of those is far past what any page should
hand a reader in one download, and somebody looking at the 8 September weekly
should not be made to fetch June 2029 to do it. The index carries every rung of
every expiry - so the board is complete the moment the page opens - and the
candles for one expiry arrive when that expiry is opened.

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
that comparison says is what the page prints under the chart, per expiry, since
six days to expiry and three years to expiry are not the same question.

**A forward that was extrapolated does not get a projection.** NIFTY lists
options to 2031 and futures three months out, so most expiries have no future
behind them and `option_history.forward_for` marks their forward ``carried``.
Diffusing a forward that was itself invented from a rate assumption produces a
band that looks exactly like the ones drawn on real forwards. The candles are
real for those contracts either way, and the candles are what the ladder is for.
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

from jobs.option_history import HISTORY_ROOT, SESSION_OPEN  # noqa: E402
from libs.ms_core.schemas import Candle  # noqa: E402
from libs.ms_options.pricing import (  # noqa: E402
    MINUTES_PER_YEAR,
    OptType,
    black76_price,
    implied_vol,
)

IST = ZoneInfo("Asia/Kolkata")
OUT_DIR = ROOT / "services" / "dashboard"

# A contract that printed a handful of times has no chart in it - the gaps
# are most of the picture and a candle every twenty minutes reads as a
# quiet market rather than as an unlisted one. Thinner than this is listed
# in the ladder, with its last price, and has no chart to open.
MIN_BARS_FOR_CHART = 20

RATE = 0.065

# Indian index options stop trading at the close on expiry day.
EXPIRY_TIME = (15, 30)

# Every minute of the next hour. It used to be eight coarsening steps, which
# drew a fan that widened in visible jumps - and a step from 30 to 45 minutes is
# a bar covering fifteen minutes of diffusion sitting beside bars covering five,
# which reads as a jump in volatility rather than as a gap in the sampling.
# The pricing is closed form, so a bar a minute costs nothing to compute; it
# costs payload, which is why the projection is packed as arrays like the
# recorded bars are.
FORECAST_STEPS = tuple(range(1, 61))

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


def _mtime(path: Path) -> str | None:
    """When this series was last written, as exchange-local clock time."""
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=IST).strftime("%H:%M")


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

    The test is whether the session is over, not whether the whole fan fits
    before the close. Those were the same question while the shortest step was
    five minutes and they stopped being the same when the fan went to one-minute
    steps: a last bar at 15:29 is one minute short of 15:30, so "does the first
    step fit" started answering yes on a finished day and anchoring the whole
    projection at the close, which is the arrangement this function exists to
    avoid. It is also not "does the last step fit", which would drag the anchor
    back an hour at 15:00 on a day that is still running and still checkable.
    """
    if bars[-1].ts < close - timedelta(minutes=1):
        return bars[-1]
    cutoff = close - timedelta(minutes=max(FORECAST_STEPS))
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


YEAR_SECONDS = 365.0 * 24 * 3600


def _years(until: datetime, ts: datetime) -> float:
    return (until - ts).total_seconds() / YEAR_SECONDS


def forward_at_expiry(
    future_price: float,
    future_expiry: datetime,
    option_expiry: datetime,
    ts: datetime,
    *,
    r: float,
) -> float:
    """The traded future, discounted back to the option's own expiry.

    The two dates are rarely the same. NIFTY lists futures monthly and options
    weekly, so the 8 September weekly is priced against the 29 September future
    - twenty-one days of carry, which at 6.5% is about ninety index points on
    24000. Ninety points is a long way at-the-money: it moves the strike's
    moneyness by nearly two rungs, and the implied volatility inverted against
    it comes back wrong in a direction that looks like skew.

    Discounting removes the part of that which is carry. What it does not remove
    is the dividend the index pays between the two dates, which is why this is
    only used to pull a forward **backwards** onto a nearer expiry. Pushing one
    forwards past the last listed future - a 2029 option against a November 2026
    future - would be three years of assumed dividends dressed as a price, and
    `option_history.forward_for` marks those ``carried`` so nothing here prices
    them at all.
    """
    if future_expiry <= option_expiry:
        return future_price
    return future_price * exp(-r * _years(future_expiry, option_expiry))


def score(
    contracts: list[dict[str, Any]],
    forward: list[Candle],
    fut_expiry: datetime,
    expiry: datetime,
    *,
    r: float,
) -> list[dict[str, Any]]:
    """Walk the day: project from every minute, compare against what happened.

    The baseline is the contract's price at the moment of the forecast. In a
    market anywhere near efficient that is very hard to beat, and a projection
    that cannot beat it is noise wearing a model's clothes.

    The volatility inversion is done **once per bar** and reused across the
    horizons, which is not a micro-optimisation: it is the same number by
    construction - the horizon is what the volatility is diffused over, not
    something the inversion sees - and re-solving it per horizon made the pass
    over 2450 contracts four times longer than it needed to be for an answer
    that could not differ.
    """
    rows: list[dict[str, Any]] = []
    tally: dict[int, tuple[list[float], list[float]]] = {h: ([], []) for h in SCORE_HORIZONS}
    for entry in contracts:
        bars: list[Candle] = entry["bars"]
        kind = OptType.CALL if entry["right"] == "CE" else OptType.PUT
        strike = entry["strike"]
        by_ts = {c.ts: c for c in bars}
        for c in bars:
            if c.c <= 0.0:
                continue
            raw = _forward_at(forward, c.ts)
            if raw is None:
                continue
            f0 = forward_at_expiry(raw, fut_expiry, expiry, c.ts, r=r)
            t0 = _years(expiry, c.ts)
            sigma = implied_vol(c.c, f0, strike, t0, r, kind)
            if sigma is None:
                continue
            for horizon in SCORE_HORIZONS:
                later = by_ts.get(c.ts + timedelta(minutes=horizon))
                if later is None:
                    continue
                got = quantiles(f0, sigma, strike, t0, horizon, kind, r)
                if got is None:
                    continue
                model_err, base_err = tally[horizon]
                model_err.append(abs(got["q50"] - later.c))
                base_err.append(abs(c.c - later.c))

    for horizon in SCORE_HORIZONS:
        model_err, base_err = tally[horizon]
        if not model_err:
            continue
        n = len(model_err)
        model_mae = sum(model_err) / n
        base_mae = sum(base_err) / n
        if base_mae <= 0.0:
            continue
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


def _verdict(rows: list[dict[str, Any]], *, projected: bool = True) -> str:
    """One sentence, and it has to survive being read on its own.

    Deliberately blunt about the margin as well as the direction: "beat the
    baseline" at 0.2% and at 20% are different findings, and a verdict that
    hides the number invites the first to be read as the second.
    """
    if not projected:
        return (
            "not projected - no listed future expires on or after this expiry, so "
            "its forward would have to be extrapolated years past the last one"
        )
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


def build_board(
    day: date,
    name: str,
    src: Path,
    block: dict[str, Any],
    forward_bars: list[Candle],
    fut_expiry: date | None,
    *,
    r: float,
) -> dict[str, Any]:
    """One expiry of one underlying: its ladder, its candles, its projection, its score."""
    expiry_date = date.fromisoformat(block["expiry"])
    expiry = _expiry_dt(expiry_date)
    close = _expiry_dt(day)  # the session's own close, same clock time as expiry's
    opened = close.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1])
    fut_dt = _expiry_dt(fut_expiry) if fut_expiry else expiry
    # See forward_for / forward_at_expiry: a forward that had to be invented
    # gets a ladder and candles, and no projection drawn on top of them.
    projectable = block["forward"]["basis"] != "carried" and bool(forward_bars)

    loaded: list[dict[str, Any]] = []
    for c in block["contracts"]:
        path = src / f"{c['token']}.ndjson"
        bars = _read(path)
        if bars:
            # The manifest carries when the sweep reached this contract. A
            # manifest written before that field existed does not, and the
            # cached file's own mtime is the same fact - so fall back to it
            # rather than dropping the read window for a whole day's board.
            loaded.append({**c, "bars": bars, "read_at": c.get("read_at") or _mtime(path)})

    measured = score(loaded, forward_bars, fut_dt, expiry, r=r) if projectable and loaded else []

    series: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    for entry in loaded:
        bars = entry["bars"]
        kind = OptType.CALL if entry["right"] == "CE" else OptType.PUT
        # [minute since the open, o, h, l, c, v]. Four hundred contracts of
        # one-minute bars is a hundred and seventy thousand rows, and the key
        # names were more than half the bytes. The page rebuilds the clock time
        # from the index, which is also how it knows a missing minute is missing.
        recorded = [
            [
                int((c.ts - opened).total_seconds() // 60),
                round(c.o, 2),
                round(c.h, 2),
                round(c.low, 2),
                round(c.c, 2),
                c.v,
            ]
            for c in bars
        ]

        sigma: float | None = None
        projected: list[list[float]] = []
        anchor = _anchor(bars, close)
        if projectable:
            raw = _forward_at(forward_bars, anchor.ts)
            f0 = forward_at_expiry(raw, fut_dt, expiry, anchor.ts, r=r) if raw is not None else None
            t0 = _years(expiry, anchor.ts)
            sigma = implied_vol(anchor.c, f0, entry["strike"], t0, r, kind) if f0 else None
            if sigma is not None and f0 is not None:
                previous = anchor.c
                for step in FORECAST_STEPS:
                    at = anchor.ts + timedelta(minutes=step)
                    if at > close:
                        # The market is shut past here. A bar drawn at 16:00 is
                        # not a forecast of anything - nothing can trade into it
                        # - and it was setting the chart's whole vertical scale.
                        break
                    got = quantiles(f0, sigma, entry["strike"], t0, step, kind, r)
                    if got is None:
                        continue
                    # [minute since the open, o, h, l, c, q25, q75], packed for
                    # the same reason the recorded bars are: sixty steps a
                    # contract across 2450 contracts is where the key names
                    # would have gone. The index is minutes since the open
                    # rather than minutes ahead of the anchor, so a projected
                    # bar and a recorded bar are on one axis without the page
                    # having to convert between two clocks - which it used to
                    # do, from a "%H:%M" string, in a loop over every series.
                    # The open is the previous bar's centre so the projection
                    # reads as a continuation. High and low are the 10th and 90th
                    # percentile of a distribution - not a range anything traded
                    # in, which is why these bars are drawn hollow.
                    projected.append(
                        [
                            int((at - opened).total_seconds() // 60),
                            round(previous, 2),
                            round(got["q90"], 2),
                            round(got["q10"], 2),
                            round(got["q50"], 2),
                            round(got["q25"], 2),
                            round(got["q75"], 2),
                        ]
                    )
                    previous = got["q50"]

        chain.append(
            {
                "strike": entry["strike"],
                "right": entry["right"],
                "last": round(bars[-1].c, 2),
                "volume": sum(b.v for b in bars),
                "bars": len(bars),
                # Two different minutes, and conflating them is a mistake this
                # page made for one afternoon. `last_print` is when the contract
                # last traded - for a far wing that is often 09:16, and it is a
                # fact about the strike. `read_at` is when the sweep reached it -
                # a fact about the run. A sweep of 2450 contracts at nine a
                # minute is spread over hours, so both belong on the rung, and
                # the header's read window has to be built from the second.
                "last_print": bars[-1].ts.astimezone(IST).strftime("%H:%M"),
                "read_at": entry.get("read_at"),
                "chartable": len(bars) >= MIN_BARS_FOR_CHART,
            }
        )
        if len(bars) < MIN_BARS_FOR_CHART:
            continue

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
    chain.sort(key=lambda c: (c["strike"], c["right"]))
    # Two ends, not one. A sweep of 2450 contracts at the endpoint's own pace
    # runs for hours, so the board is genuinely ragged and the honest header is
    # a window rather than an instant. Built from when each rung was *read* -
    # the earliest last-print is 09:16 on almost any board, because some wing
    # traded once at the open, and it says nothing about when the sweep ran.
    reads = [c["read_at"] for c in chain if c["read_at"]]
    as_of = max(reads, default=None)
    read_from = min(reads, default=None)
    return {
        "name": name,
        "expiry": block["expiry"],
        "read_from": read_from,
        "open": opened.astimezone(IST).strftime("%H:%M"),
        "day": day.isoformat(),
        "source": "angel historical candles",
        "forward": {**block["forward"], "bars": len(forward_bars)},
        "atm": block["atm"],
        "step": block["strike_step"],
        "listed": block["listed"],
        "as_of": as_of,
        "chain": chain,
        "series": series,
        "failed": block.get("failed", []),
        "score": {
            "method": (
                "each contract's own implied volatility against the same-minute "
                "future discounted to this expiry, forward diffused driftless, "
                "repriced in closed form"
            ),
            "baseline": "the contract's price now, unchanged",
            "measured_on": day.isoformat(),
            "horizons": measured,
            "verdict": _verdict(measured, projected=projectable),
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


def build(day: date, name: str, root: Path, *, r: float = RATE) -> list[dict[str, Any]]:
    """Every live expiry of one underlying, nearest first."""
    src = root / f"date={day.isoformat()}" / name
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"  no cached history for {name} on {day}. Run:\n"
            f"    python jobs/option_history.py --date {day.isoformat()} --underlying {name}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "expiries" not in manifest:
        raise SystemExit(
            f"  {name}: the cached manifest for {day} is the single-expiry shape "
            "written before the chain covered every expiry. Re-run "
            f"jobs/option_history.py --date {day.isoformat()} to rebuild it; the "
            "candles themselves are still cached and will not be re-fetched."
        )

    futures = {int(f["token"]): f for f in manifest.get("futures", [])}
    bars_by_token: dict[int, list[Candle]] = {}
    for token in futures:
        bars_by_token[token] = _read(src / f"{token}.ndjson")

    boards: list[dict[str, Any]] = []
    for block in manifest["expiries"]:
        token = int(block["forward"]["token"])
        meta = futures.get(token)
        fut_expiry = date.fromisoformat(meta["expiry"]) if meta and meta.get("expiry") else None
        board = build_board(
            day,
            name,
            src,
            block,
            bars_by_token.get(token, []),
            fut_expiry,
            r=r,
        )
        if board["chain"]:
            boards.append(board)
    if not boards:
        raise SystemExit(f"  {name}: nothing cached has any bars for {day}.")
    return boards


HEADER = """/* Generated by jobs/chart_history.py - do not edit.
 *
 * `recorded` happened: real one-minute candles from Angel's historical
 * endpoint, packed as [minute since the open, o, h, l, c, v]. `projected` did
 * not happen: it is packed the same way, as [minute since the open, o, h, l,
 * c, q25, q75], where the high and low are the 90th and 10th percentile of a
 * distribution rather than a range anything traded in. They are separate arrays on purpose - merged, a
 * projection becomes indistinguishable from history the moment somebody
 * scrolls. Each expiry carries the score its own projection earned on its own
 * day, because six days to expiry and three years to expiry are not the same
 * question.
 */
"""


def _slug(expiry: str) -> str:
    return expiry.replace("-", "")


def write(boards: dict[str, list[dict[str, Any]]], day: date, out_dir: Path) -> list[Path]:
    """One file per expiry, plus a small index the page loads first.

    Split per expiry rather than per underlying because the universe grew by a
    factor of five. Every live NIFTY call and put is 1568 contracts over
    eighteen expiries; a full session of those in one file is tens of megabytes,
    and a reader looking at this Tuesday's weekly should not download June 2029
    to do it. The index carries every expiry's shape - its money, its grid, how
    many rungs traded, how its projection scored - so the frame and the expiry
    rail render before any payload arrives, and a payload is fetched only when
    its expiry is opened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clearing what this is about to supersede is part of writing, and it lives
    # here rather than in each caller for a reason that cost a working page.
    # `close_out` did it before calling `write`, and its tests stub `write` and
    # not the loop beside it - so running the test suite deleted the real
    # dashboard's payloads, silently, and nothing noticed until a deploy check
    # refused. A destructive step that a caller can stub out from under is a
    # step in the wrong place.
    #
    # Yesterday's expiries are not today's, and a leftover payload is a valid
    # file defining a valid global: the page would load it happily and date it
    # from its own contents, which is how a chart of a contract that has since
    # expired ends up on a page headed today.
    keep = {f"chart_{name}_{_slug(b['expiry'])}.js" for name, g in boards.items() for b in g}
    for stale in out_dir.glob("chart_*_*.js"):
        if stale.name not in keep:
            stale.unlink()

    written: list[Path] = []
    for name, group in boards.items():
        for b in group:
            slug = _slug(b["expiry"])
            path = out_dir / f"chart_{name}_{slug}.js"
            path.write_text(
                HEADER
                + f"window.CHART_{name}_{slug} = "
                + json.dumps(b, separators=(",", ":"))
                + ";\n",
                encoding="utf-8",
            )
            written.append(path)

    index = {
        "day": day.isoformat(),
        "underlyings": [
            {
                "name": name,
                "open": group[0]["open"],
                "as_of": max((b["as_of"] for b in group if b["as_of"]), default=None),
                "read_from": min((b["read_from"] for b in group if b["read_from"]), default=None),
                "contracts": sum(len(b["chain"]) for b in group),
                "charted": sum(len(b["series"]) for b in group),
                "listed": sum(b["listed"] for b in group),
                "expiries": [
                    {
                        "expiry": b["expiry"],
                        "file": f"chart_{name}_{_slug(b['expiry'])}.js",
                        "global": f"CHART_{name}_{_slug(b['expiry'])}",
                        "forward": b["forward"],
                        "atm": b["atm"],
                        "step": b["step"],
                        "as_of": b["as_of"],
                        "read_from": b["read_from"],
                        "listed": b["listed"],
                        "strikes": len({c["strike"] for c in b["chain"]}),
                        "contracts": len(b["chain"]),
                        "charted": len(b["series"]),
                        "score": b["score"],
                    }
                    for b in group
                ],
            }
            for name, group in boards.items()
        ],
    }
    idx = out_dir / "chart_index.js"
    idx.write_text(
        HEADER + "window.CHART_INDEX = " + json.dumps(index, indent=2) + ";\n", encoding="utf-8"
    )
    return [idx, *written]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bake cached option history into the chart")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--underlying", nargs="*", default=["NIFTY", "BANKNIFTY"])
    p.add_argument("--root", default=str(HISTORY_ROOT))
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args(argv)

    out = Path(args.out)
    boards = {name: build(args.date, name, Path(args.root)) for name in args.underlying}
    written = write(boards, args.date, out)

    print()
    print(f"  {len(written)} files, {sum(p.stat().st_size for p in written):,} bytes total")
    print(f"    {written[0].name:30} {written[0].stat().st_size:>11,} bytes")
    print()
    for name, group in boards.items():
        chained = sum(len(b["chain"]) for b in group)
        charted = sum(len(b["series"]) for b in group)
        print(f"    {name}: {chained} contracts traded, {charted} chartable, {len(group)} expiries")
        for b in group:
            calls = sum(1 for c in b["chain"] if c["right"] == "CE")
            puts = sum(1 for c in b["chain"] if c["right"] == "PE")
            fwd = b["forward"]
            print(
                f"      {b['expiry']}  {calls:>3}C {puts:>3}P  "
                f"{len(b['series']):>3} charted  forward {fwd['symbol']} ({fwd['basis']})"
            )
            print(f"        {b['score']['verdict']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
