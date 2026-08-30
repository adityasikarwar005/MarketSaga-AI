"""M54 - the quantile heads, measured on real daily bars.

    python jobs/eval_quantiles.py                       # NIFTY, horizons 1 and 5
    python jobs/eval_quantiles.py --symbol BANKNIFTY

Like M55 before it, this module had only ever been fitted to synthetic series
whose answer was known in advance. The heads forecast the distribution of the
forward return; what follows is the first evidence about whether they beat the
fallback they were built to replace.

**The baseline is the module's own fallback, and that is the point.** A head
that cannot fit falls back to Gaussian quantiles scaled by the volatility
forecast. If the fitted heads do not beat that, the machinery is elaborate
packaging around `sigma * z_alpha` - and the honest response is to say so, not
to find a baseline it does beat.

The Gaussian here is not a straw man. Its sigma is the trailing realised
volatility scaled to the horizon, so it sees exactly the same information the
heads do and adapts to regime the same way. What it cannot represent is skew:
it is symmetric by construction. Beating it therefore means capturing asymmetry
in the return distribution, which is the only thing the extra machinery is for.

**Two numbers, and neither alone is a result.** Mean pinball says whether the
quantiles are sharp. Coverage of the 5-95 band says whether they are honest. A
model can win on pinball by shrinking every interval and paying the loss on the
tails it now misses, which reads as an improvement and sizes positions off a
band that is too narrow on exactly the days that matter.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import UTC
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_models.quantile import DEFAULT_QUANTILES  # noqa: E402
from libs.ms_models.quantile_heads import (  # noqa: E402
    QuantileHeads,
    gaussian_forecast,
    mean_pinball,
    score_heads,
)
from libs.ms_models.volatility import realised_variance  # noqa: E402

WEEKLY, MONTHLY = 5, 22


def build_dataset(bars, *, horizon: int):
    """Rows, forward returns, and the trailing sigma the baseline forecasts with.

    Row ``i`` uses bars up to and including ``i`` and targets the log return
    from close ``i`` to close ``i + horizon``. Nothing in a row is knowable
    later than its own bar, which is the only property that makes the
    walk-forward split below mean anything.
    """
    rv = [realised_variance([b], estimator="parkinson") for b in bars]
    closes = [b.c for b in bars]

    rows: list[list[float]] = []
    targets: list[float] = []
    sigmas: list[float] = []
    for i in range(MONTHLY - 1, len(bars) - horizon):
        daily = rv[i]
        weekly = fmean(rv[i - WEEKLY + 1 : i + 1])
        monthly = fmean(rv[i - MONTHLY + 1 : i + 1])
        rows.append([daily, weekly, monthly])
        targets.append(math.log(closes[i + horizon] / closes[i]))
        # The baseline's sigma: trailing realised vol, scaled by root-time to
        # the horizon. Same information as the row, no skew.
        sigmas.append(math.sqrt(max(monthly, 0.0) * horizon))
    return rows, targets, sigmas


def walk_forward(rows, targets, sigmas, *, horizon: int, folds: int, min_train: int):
    """Train on the past, test on the future, repeatedly.

    At horizon > 1 neighbouring rows share outcome windows, so the test block
    is trimmed by ``horizon - 1`` rows at its front: without that the first
    test targets overlap returns the training side has already seen.
    """
    if len(rows) < min_train + folds * 10:
        return None
    fold_size = (len(rows) - min_train) // folds
    if fold_size < 10:
        return None

    head_fc, base_fc, actual = [], [], []
    fallbacks = 0
    for f in range(folds):
        train_end = min_train + f * fold_size
        embargo = horizon - 1
        test_lo = train_end + embargo
        test = range(test_lo, min(test_lo + fold_size, len(rows)))
        if not len(test):
            continue

        heads = QuantileHeads(quantiles=DEFAULT_QUANTILES, horizons=(horizon,))
        heads.fit(
            rows[:train_end],
            {horizon: targets[:train_end]},
            sigma_by_horizon={horizon: fmean(sigmas[:train_end])},
        )
        fallbacks += sum(1 for a in DEFAULT_QUANTILES if heads.used_fallback(a, horizon))
        for i in test:
            head_fc.append(heads.predict(rows[i], horizon=horizon))
            base_fc.append(gaussian_forecast(sigmas[i], quantiles=DEFAULT_QUANTILES))
            actual.append(targets[i])

    if not actual:
        return None
    return head_fc, base_fc, actual, fallbacks


def coverage(forecasts, actuals) -> float:
    lo, hi = min(DEFAULT_QUANTILES), max(DEFAULT_QUANTILES)
    inside = sum(
        1
        for f, a in zip(forecasts, actuals, strict=True)
        if f.quantiles[lo] <= a <= f.quantiles[hi]
    )
    return inside / len(actuals)


def dataset_version(bars) -> str:
    payload = ";".join(f"{b.day}:{b.o}:{b.h}:{b.low}:{b.c}" for b in bars)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--min-train", type=int, default=500)
    p.add_argument("--name", default="m54_quantile_heads_real_bars")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)

    configure()
    cfg = cfgmod.load(warn=False)
    clock = RealClock()

    bars = DailyStore().load(args.symbol)
    if not bars:
        print(f"  no cached daily bars for {args.symbol!r}.")
        print("  Fetch them first: python jobs/gap_backfill.py --fetch")
        return 1

    nominal = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES)
    print()
    print(f"  REAL daily bars: {args.symbol}, {len(bars):,} sessions")
    print(f"    {bars[0].day} .. {bars[-1].day}")
    print("    baseline: Gaussian quantiles from trailing realised vol (no skew)")
    print(f"    nominal band {nominal:.0%}")
    print()

    metrics: dict[str, float] = {"sessions": float(len(bars))}
    verdicts = []
    for horizon in args.horizons:
        rows, targets, sigmas = build_dataset(bars, horizon=horizon)
        out = walk_forward(
            rows, targets, sigmas, horizon=horizon, folds=args.folds, min_train=args.min_train
        )
        print(f"  horizon {horizon}d  ({len(rows):,} rows)")
        if out is None:
            print("    not enough rows to walk forward - no result, not a pass")
            continue
        head_fc, base_fc, actual, fallbacks = out

        head_pin = mean_pinball(head_fc, actual, quantiles=DEFAULT_QUANTILES)
        base_pin = mean_pinball(base_fc, actual, quantiles=DEFAULT_QUANTILES)
        gain = (base_pin - head_pin) / base_pin if base_pin else 0.0
        head_cov, base_cov = coverage(head_fc, actual), coverage(base_fc, actual)

        print(f"    tested on {len(actual):,} rows, {fallbacks} head-fallbacks")
        print(f"    Gaussian    pinball={base_pin:.6f}  cov{nominal:.0%}={base_cov:.1%}")
        print(f"    heads       pinball={head_pin:.6f}  cov{nominal:.0%}={head_cov:.1%}")
        print(f"    pinball improvement {gain:+.1%}")
        for s in score_heads(head_fc, actual, quantiles=DEFAULT_QUANTILES):
            print(s.render())

        sharper = gain > 0.0
        better_cov = abs(head_cov - nominal) < abs(base_cov - nominal)
        verdicts.append((horizon, sharper, better_cov, head_cov, base_cov))
        if sharper and better_cov:
            print("    -> beats the Gaussian fallback on both sharpness and honesty")
        elif sharper:
            print("    -> sharper but LESS calibrated: a narrower band that misses")
            print("       more often is not an improvement, it is a smaller number")
        elif better_cov:
            print("    -> loses on pinball, wins on coverage. The heads buy honesty")
            print("       with width. Pinball is the metric this is judged on, so")
            print("       this is not a pass - but it is not the same failure as")
            print("       being worse at both, and reporting it as one would hide")
            print("       the only thing the extra machinery demonstrably did.")
        else:
            print("    -> worse on both. Does not beat the Gaussian fallback.")

        # The number that matters more than either model beating the other.
        if head_cov < nominal - 0.02 and base_cov < nominal - 0.02:
            print(f"    !! BOTH under-cover: a {nominal:.0%} band that holds the outcome")
            print(f"       {head_cov:.1%} (heads) and {base_cov:.1%} (Gaussian) of the time.")
            print("       Real returns have fatter tails than either represents, so a")
            print("       position sized off either interval is sized too large - and")
            print("       most so on the days the tail actually arrives.")
        print()

        metrics[f"h{horizon}_head_pinball"] = head_pin
        metrics[f"h{horizon}_gauss_pinball"] = base_pin
        metrics[f"h{horizon}_improvement"] = gain
        metrics[f"h{horizon}_head_cov"] = head_cov
        metrics[f"h{horizon}_gauss_cov"] = base_cov
        metrics[f"h{horizon}_beats"] = float(sharper and better_cov)
        metrics[f"h{horizon}_nominal"] = nominal

    if not verdicts:
        print("  nothing evaluated.")
        return 1

    won = [h for h, sharp, cov, _hc, _bc in verdicts if sharp and cov]
    print("  ---")
    if len(won) == len(verdicts):
        print(f"  Beats its own Gaussian fallback at every horizon tested: {won}.")
    elif won:
        print(f"  Beats the fallback at {won} only, not at every horizon tested.")
    else:
        print("  Does NOT beat its own Gaussian fallback on pinball at any horizon.")
        print("  Recorded rather than retuned: a baseline swapped until the model")
        print("  wins measures the search, not the model.")
        if all(cov for _h, _s, cov, _hc, _bc in verdicts):
            print()
            print("  It is closer to nominal coverage at every horizon, though, which")
            print("  is the one thing the heads demonstrably bought. They are wider")
            print("  and more honest; the Gaussian is narrower and more confident")
            print("  than the data supports. Neither reaches nominal.")

    if args.no_log:
        print("\n  --no-log: nothing appended to the experiment log.")
        return 0

    log = ExperimentLog()
    entry = log.record(
        Experiment(
            name=args.name,
            ts=clock.now().astimezone(UTC).isoformat(),
            config_hash=cfg.config_hash,
            dataset_version=dataset_version(bars),
            feature_hash=hashlib.sha256(b"rv_daily,rv_weekly,rv_monthly").hexdigest()[:16],
            metrics=metrics,
            notes=args.note or f"{args.symbol} real daily bars, horizons {args.horizons}",
        )
    )
    print(f"\n  logged as trial {log.trial_count()} ({entry.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
