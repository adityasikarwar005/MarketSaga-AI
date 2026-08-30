"""M55 - the volatility model, measured on real daily bars.

    python jobs/eval_volatility.py                    # NIFTY, horizon 1
    python jobs/eval_volatility.py --symbol BANKNIFTY --horizon 5

Until this ran, M55 had only ever seen synthetic series whose answer was known
in advance. That is the condition this project treats as unproven regardless of
how many tests pass, so the number below is the first evidence about it.

**HAR-RV is the baseline and it is a strong one.** A three-term regression on
yesterday's, last week's and last month's realised variance is famously hard to
beat out of sample, and the whole gate for this module is beating it on QLIKE.
An improvement of a few tenths of a percent is not a result; it is the model
reproducing HAR with extra steps.

**QLIKE rather than MSE, and the two can disagree.** MSE on variance is
dominated by the handful of largest observations, so a model that is wrong most
days and right on three of them wins on MSE while being useless. QLIKE is scale
free and penalises under-prediction harder, which is the direction that costs
money when a position is sized off it.

**The realised variance here is an estimate, not a measurement.** Daily bars
give one high, one low, one open and one close, so Parkinson and Garman-Klass
are inferences from a range rather than a sum of intraday squared returns. Both
are reported, because they disagree about jumps - a bar that gaps and then
trades quietly reads as violent to Parkinson, and Garman-Klass is worse on that
axis, not better. Agreement between them is weak evidence; disagreement is a
signal that the day was gappy rather than volatile.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_models.volatility import (  # noqa: E402
    build_har_rows,
    realised_variance,
    walk_forward_evaluate,
)

ESTIMATORS = ("parkinson", "garman_klass")


def rv_series(bars, estimator: str) -> list[float]:
    """One realised-variance estimate per bar.

    Each estimate uses that bar and nothing else, so the series carries no
    lookahead of its own - the lookbacks are HAR's business, and building them
    here with a rolling helper is how a window quietly centres itself.
    """
    return [realised_variance([b], estimator=estimator) for b in bars]


def dataset_version(bars) -> str:
    """Hash of the exact bars measured, so a rerun is comparable or is not."""
    payload = ";".join(f"{b.day}:{b.o}:{b.h}:{b.low}:{b.c}" for b in bars)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def evaluate(bars, *, estimator: str, horizon: int, folds: int):
    series = rv_series(bars, estimator)
    rows = build_har_rows(series, horizon=horizon)
    if horizon > 1:
        # walk_forward_evaluate splits contiguously and does no purging. At
        # horizon > 1 neighbouring rows share outcome windows, so a fold
        # boundary leaks the future into the training side. Thinning to every
        # horizon-th row removes the overlap outright rather than trimming an
        # embargo and hoping the arithmetic is right.
        rows = rows[::horizon]
    return rows, walk_forward_evaluate(rows, n_folds=folds)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--horizon", type=int, default=1, help="periods ahead to forecast")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--name", default="m55_volatility_real_bars")
    p.add_argument("--note", default="", help="recorded with the trial")
    p.add_argument("--no-log", action="store_true", help="do not append to the experiment log")
    args = p.parse_args(argv)

    configure()
    cfg = cfgmod.load(warn=False)
    clock = RealClock()

    bars = DailyStore().load(args.symbol)
    if not bars:
        print(f"  no cached daily bars for {args.symbol!r}.")
        print("  Fetch them first: python jobs/gap_backfill.py --fetch")
        return 1

    print()
    print(f"  REAL daily bars: {args.symbol}, {len(bars):,} sessions")
    print(f"    {bars[0].day} .. {bars[-1].day}")
    print(f"    horizon {args.horizon}, {args.folds} folds, baseline HAR-RV")
    print()

    results = {}
    for estimator in ESTIMATORS:
        rows, ev = evaluate(bars, estimator=estimator, horizon=args.horizon, folds=args.folds)
        print(f"  {estimator}  ({len(rows):,} HAR rows)")
        if ev is None:
            print("    not enough rows to walk forward - no result, not a pass")
            continue
        print(ev.render())
        print()
        results[estimator] = ev

    if not results:
        print("  nothing evaluated.")
        return 1

    beat = [e for e, ev in results.items() if ev.passes]
    print("  ---")
    if len(beat) == len(results):
        print(f"  Beats HAR on QLIKE under both estimators: {', '.join(beat)}.")
    elif beat:
        print(f"  Beats HAR under {', '.join(beat)} only.")
        print("  One estimator is not a result. They disagree about jumps, so a")
        print("  model that wins under one is a model fitted to that estimator's")
        print("  blind spot until it wins under both.")
    else:
        print("  Does NOT beat HAR-RV under either estimator.")
        print("  That is the gate for this module, and it is not met. The number")
        print("  is recorded rather than tuned - a threshold moved until it passes")
        print("  measures nothing.")

    if args.no_log:
        print("\n  --no-log: nothing appended to the experiment log.")
        return 0

    # Recorded before the result is read, not after. An evaluation logged only
    # when the author likes the number is a highlight reel, not a trial count.
    log = ExperimentLog()
    metrics = {"sessions": float(len(bars)), "horizon": float(args.horizon)}
    for estimator, ev in results.items():
        short = "pk" if estimator == "parkinson" else "gk"
        metrics[f"{short}_har_qlike"] = ev.har_qlike
        metrics[f"{short}_model_qlike"] = ev.model_qlike
        metrics[f"{short}_improvement"] = ev.improvement
        metrics[f"{short}_beats_har"] = float(ev.passes)
    entry = log.record(
        Experiment(
            name=args.name,
            ts=clock.now().astimezone(UTC).isoformat(),
            config_hash=cfg.config_hash,
            dataset_version=dataset_version(bars),
            feature_hash=dataset_version(bars),
            metrics=metrics,
            notes=args.note or f"{args.symbol} real daily bars, horizon {args.horizon}",
        )
    )
    print(f"\n  logged as trial {log.trial_count()} ({entry.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
