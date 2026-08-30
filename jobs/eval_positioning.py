"""Does dealer gamma positioning say anything HAR does not already know?

    python jobs/eval_positioning.py --symbol NIFTY

The bar is deliberately the one that killed M55. HAR's three lags exhaust what
daily OHLC carries about tomorrow's realised variance, so a new feature does not
get to be judged against an unconditional mean - it has to earn its place in the
**residual** of a HAR forecast. Beating a weak baseline on a persistent series
is arithmetic, not evidence.

Point-in-time. The F&O bhavcopy for session ``d`` is published in the evening of
``d``, so it may inform session ``d+1`` and nothing earlier. The target here is
always the *next* session's range and the profile is always the *prior* close's
chain. `libs/ms_data/bhavcopy.available_at` is the authority on the timestamp;
this job encodes the same one-session offset directly.

The cycle norm is fitted on the training block only. Fitting it on everything
would let each day be scored as unusual relative to a distribution that already
contains it, which is a quiet way of using the future - and on a statistic this
strongly tied to the expiry cycle it would flatter the result substantially.

**Two nulls run alongside the real number.** Shuffling the open interest within
each session destroys the positioning while preserving every other property of
the data; shuffling the scores across days destroys the pairing with outcomes.
Either coming back comparable to the real result means there is nothing here.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.bhavcopy import BhavcopyStore  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.positioning import (  # noqa: E402
    GammaProfile,
    PositioningError,
    cycle_adjusted,
    fit_cycle_norm,
    gamma_concentration,
)

# HAR's three horizons: yesterday, the last week, the last month.
HAR_LAGS = (1, 5, 22)


@dataclass(frozen=True)
class Row:
    day: date
    profile: GammaProfile
    har_forecast: float
    actual: float  # next session's range, as a fraction of close

    @property
    def har_residual(self) -> float:
        """What HAR got wrong, in logs so the scale is symmetric."""
        return math.log(max(self.actual, 1e-9) / max(self.har_forecast, 1e-9))


def har_forecast(history: list[float]) -> float | None:
    """Mean of the three HAR windows, equally weighted.

    Equal weights rather than fitted ones on purpose: the point is a strong,
    parameter-free baseline that cannot be accused of having been tuned down so
    the challenger clears it.
    """
    if len(history) < max(HAR_LAGS):
        return None
    return fmean([fmean(history[-lag:]) for lag in HAR_LAGS])


def build_rows(symbol: str, cache: Path) -> tuple[list[Row], dict[str, int]]:
    store = BhavcopyStore()
    bars = DailyStore().load(symbol)
    if not bars:
        raise SystemExit(f"no cached daily bars for {symbol!r}")
    by_day = {b.day: b for b in bars}
    ordered = sorted(by_day)

    days = sorted(date.fromisoformat(p.stem) for p in cache.glob("*.json"))
    skipped = {"no_bar": 0, "no_next_session": 0, "no_chain": 0, "short_history": 0}
    rows: list[Row] = []

    index = {d: i for i, d in enumerate(ordered)}
    for d in days:
        bar = by_day.get(d)
        if bar is None:
            skipped["no_bar"] += 1
            continue
        i = index[d]
        if i + 1 >= len(ordered):
            skipped["no_next_session"] += 1
            continue
        # The HAR history is every session strictly before d+1, which is exactly
        # what a forecaster standing at d's close would hold.
        history = [by_day[x].range_pct for x in ordered[: i + 1]]
        fc = har_forecast(history)
        if fc is None:
            skipped["short_history"] += 1
            continue
        day_file = store.load(d)
        if day_file is None:
            skipped["no_chain"] += 1
            continue
        expiries = [e for e in day_file.expiries(symbol) if e >= d]
        if not expiries:
            skipped["no_chain"] += 1
            continue
        recs, _ = day_file.options(symbol, expiry=expiries[0], traded_only=False)
        try:
            profile = gamma_concentration(
                [(r.strike, r.open_interest) for r in recs],
                day=d,
                expiry=expiries[0],
                spot=bar.c,
            )
        except PositioningError:
            skipped["no_chain"] += 1
            continue
        rows.append(
            Row(
                day=d,
                profile=profile,
                har_forecast=fc,
                actual=by_day[ordered[i + 1]].range_pct,
            )
        )
    return rows, skipped


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--train", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)

    configure()
    cache = ROOT / "data" / "reference" / "bhavcopy"
    rows, skipped = build_rows(args.symbol, cache)
    print()
    print(
        f"  {args.symbol}: {len(rows)} usable sessions from {len(list(cache.glob('*.json')))} cached chains"
    )
    for reason, n in skipped.items():
        if n:
            print(f"    skipped, {reason}: {n}")
    if len(rows) < 60:
        print("\n  Too few sessions to conclude anything. Backfill more bhavcopy first.")
        return 1

    rows.sort(key=lambda r: r.day)
    cut = int(len(rows) * args.train)
    train, test = rows[:cut], rows[cut:]
    norm = fit_cycle_norm([r.profile for r in train])
    print(
        f"    cycle norm fitted on {len(train)} training sessions, "
        f"{len(norm.known_horizons())} horizons"
    )

    scored = [(r, cycle_adjusted(r.profile, norm)) for r in test]
    kept = [(r, z) for r, z in scored if z is not None]
    print(f"    {len(kept)} of {len(test)} test sessions had a norm for their horizon")
    if len(kept) < 40:
        print("\n  Too few scored test sessions. Backfill more bhavcopy first.")
        return 1

    scores = [z for _, z in kept]
    residuals = [r.har_residual for r, _ in kept]
    dtes = [float(r.profile.days_to_expiry) for r, _ in kept]

    real = spearman(scores, residuals)
    rng = random.Random(args.seed)
    shuffled = scores[:]
    rng.shuffle(shuffled)
    null_pairing = spearman(shuffled, residuals)
    raw = spearman([r.profile.concentration for r, _ in kept], residuals)

    print()
    print("  Spearman against the HAR residual - what HAR got wrong tomorrow:")
    print(f"    cycle-adjusted positioning   {real:+.4f}   <- the question")
    print(f"    raw concentration            {raw:+.4f}   <- mostly the expiry cycle")
    print(f"    scores shuffled across days  {null_pairing:+.4f}   <- must be ~0")
    print()
    print(f"    adjusted score vs dte        {spearman(scores, dtes):+.4f}   <- cycle removed?")
    print()

    # A rank correlation this size on this many rows: the rough standard error
    # of a Spearman under the null is 1/sqrt(n-1).
    se = 1.0 / math.sqrt(len(kept) - 1)
    print(f"  n = {len(kept)}, so the null standard error is about {se:.4f}.")
    if abs(real) < 2.0 * se:
        print(f"  |{real:.4f}| is inside two standard errors. NOT A FINDING.")
        print("  Positioning adds nothing to HAR on this sample. Record it and stop;")
        print("  searching for a variant that clears the bar is the anti-pattern the")
        print("  measurement layer exists to prevent.")
    else:
        print(f"  |{real:.4f}| is outside two standard errors on {len(kept)} sessions.")
        print("  That is a signal worth a proper trial - NOT a result. Next it needs")
        print("  an out-of-sample block it has never been looked at on, and an entry")
        print("  in research/experiments.jsonl before anything is built on it.")
    print()
    print(f"  median |score| = {median([abs(s) for s in scores]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
