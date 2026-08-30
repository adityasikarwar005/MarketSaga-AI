"""M53 - the direction model against the rules baseline, on real bars.

    python jobs/eval_direction.py

`direction.py` opens by saying this is "the first learned model that has to beat
the rules baseline on its own terms". Until this ran, it never had. The module
is marked verified in the tracker, has 23 tests, and carries **no Verify block
in the plan at all** - so R1, "a module is not done until its Verify block
passes", had nothing to check it against.

**Read this before the numbers. M53 is an intraday model and there is no
intraday history.** The features the rules baseline reads - `ret_5m_z`,
`efficiency_ratio_30m` - are five- and thirty-minute quantities, and what is
available is ten years of daily bars. So the analogues below are computed over
days instead of minutes, and what is measured is *this machinery against this
baseline on the data that exists*, which is a different question from the one
G5 asks.

A loss here is informative: the same code, the same baseline, and no edge on the
easier-to-sample horizon. A win here would **not** be a G5 pass, because G5 is
about an intraday model on intraday data. Nothing in this job may be reported as
one.

**Both models are handed the same feature dict**, which is the architecture
working as designed rather than a handicap: `Model.predict` takes a mapping and
each model reads what it knows. The baseline uses two of them; the learned model
uses all of them and is free to find more.

**The gate is the bootstrap interval, not the mean.** `compare()` requires the
confidence interval on the log-loss delta to exclude zero. A challenger that
wins on average with an interval straddling zero has not been shown to beat
anything, and shipping it on the point estimate is how a project accumulates
models that do nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import UTC
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.baseline import RulesBaseline, compare, score  # noqa: E402
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_models.direction import (  # noqa: E402
    DirectionModel,
    auc,
    brier_skill_score,
    reliability_curve,
    reliability_slope,
)

Z_WINDOW = 60
ER_WINDOW = 20

# G5's stated thresholds, so the report is against the plan rather than against
# whatever the run happened to produce.
G5_RELIABILITY = (0.85, 1.15)
G5_BRIER_SKILL = 0.02
G5_AUC = 0.55


def build_dataset(bars):
    """Feature rows and next-day direction labels, from daily bars only.

    Row ``i`` uses bars up to and including ``i`` and is labelled by the sign of
    the log return from close ``i`` to close ``i+1``. Nothing in a row is
    knowable later than its own bar.

    The names match what `RulesBaseline` reads. They are daily analogues of
    intraday quantities and the names are kept so the baseline can read them at
    all - which is worth stating plainly rather than hiding behind a rename.
    """
    closes = [b.c for b in bars]
    rets = [0.0] + [math.log(b / a) for a, b in pairwise(closes)]

    rows: list[dict[str, float | None]] = []
    labels: list[int] = []
    for i in range(Z_WINDOW, len(bars) - 1):
        window = rets[i - Z_WINDOW + 1 : i + 1]
        sigma = pstdev(window) or 1e-9
        er_window = rets[i - ER_WINDOW + 1 : i + 1]
        gross = sum(abs(r) for r in er_window) or 1e-9
        rows.append(
            {
                # what the rules baseline reads
                "ret_5m_z": rets[i] / sigma,
                "efficiency_ratio_30m": abs(sum(er_window)) / gross,
                # extra columns only the learned model can use
                "ret_5d": sum(rets[i - 4 : i + 1]),
                "ret_20d": sum(rets[i - 19 : i + 1]),
                "rv_20d": fmean(r * r for r in er_window),
                "range_pos": (
                    (bars[i].c - bars[i].low) / (bars[i].h - bars[i].low)
                    if bars[i].h > bars[i].low
                    else 0.5
                ),
            }
        )
        nxt = rets[i + 1]
        labels.append(1 if nxt > 0 else -1 if nxt < 0 else 0)
    return rows, labels


def walk_forward(rows, labels, *, folds: int, min_train: int):
    """Train on the past, predict the future, repeatedly. No purging needed:
    each label is the single day after its own row, so consecutive rows share no
    outcome window."""
    if len(rows) < min_train + folds * 10:
        return None
    fold = (len(rows) - min_train) // folds
    if fold < 10:
        return None

    base_p: list[float] = []
    model_p: list[float] = []
    outcomes: list[int] = []
    fitted_folds = 0
    calibrated_folds = 0
    no_disc = 0  # folds where the model's own gate said it cannot rank at all

    baseline = RulesBaseline()
    for f in range(folds):
        end = min_train + f * fold
        test = rows[end : end + fold]
        test_y = labels[end : end + fold]
        if not test:
            continue
        model = DirectionModel()
        try:
            model.fit(rows[:end], labels[:end])
        except Exception as exc:
            print(f"    fold {f}: did not fit ({type(exc).__name__}: {exc})")
            continue
        fitted_folds += 1
        if getattr(model, "calibrated", False):
            calibrated_folds += 1
        if getattr(model, "no_discrimination", False):
            no_disc += 1
        for feats, y in zip(test, test_y, strict=True):
            base_p.append(baseline.predict(feats).p_up)
            model_p.append(model.predict(feats).p_up)
            outcomes.append(y)

    if not outcomes:
        return None
    return base_p, model_p, outcomes, fitted_folds, calibrated_folds, no_disc


def dataset_version(bars) -> str:
    payload = ";".join(f"{b.day}:{b.o}:{b.h}:{b.low}:{b.c}" for b in bars)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--min-train", type=int, default=600)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--name", default="m53_direction_vs_rules_daily")
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

    rows, labels = build_dataset(bars)
    print()
    print(f"  REAL daily bars: {args.symbol}, {len(bars):,} sessions")
    print(f"    {bars[0].day} .. {bars[-1].day}")
    print()
    print("  NOT a G5 measurement. M53 is an intraday model and there is no")
    print("  intraday history; these are daily analogues of five- and thirty-")
    print("  minute features. A loss here is informative. A win here would not")
    print("  be a G5 pass and must not be reported as one.")
    print()

    out = walk_forward(rows, labels, folds=args.folds, min_train=args.min_train)
    if out is None:
        print("  not enough rows to walk forward - no result, not a pass")
        return 1
    base_p, model_p, outcomes, fitted, calibrated, no_disc = out

    up = sum(1 for y in outcomes if y > 0)
    print(
        f"  {len(outcomes):,} out-of-sample rows, {fitted}/{args.folds} folds fitted, "
        f"{calibrated} calibrated"
    )
    if no_disc:
        print(f"  {no_disc}/{fitted} folds: the model's OWN discrimination gate refused.")
        print("    has_discrimination() found the booster could not rank the held-out")
        print("    rows at all, so isotonic calibration was skipped rather than")
        print("    inventing a shape to correct. An independent verdict from inside")
        print("    the model, reached before any of the numbers below.")
        print("    It also means no interval reaches M75, which then halves the")
        print("    position for an unquantified estimate - correct behaviour, and")
        print("    worth knowing is happening.")
    print(
        f"  base rate: {up / len(outcomes):.1%} up  "
        f"(a model that always says 'up' is right that often)"
    )
    print()

    m0 = score("M0_rules", base_p, outcomes)
    m1 = score("M1_trees", model_p, outcomes)
    print(m0.render())
    print(m1.render())
    print()

    cmp = compare(base_p, model_p, outcomes, bootstrap=args.bootstrap)
    lo, hi = cmp.bootstrap_ci or (float("nan"), float("nan"))
    print(f"  delta log loss {cmp.delta_log_loss:+.5f}   " f"bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  beats baseline (CI excludes zero): {'YES' if cmp.beats_baseline else 'NO'}")
    print()

    binary = [(p, 1 if y > 0 else 0) for p, y in zip(model_p, outcomes, strict=True) if y != 0]
    probs = [p for p, _ in binary]
    ys = [y for _, y in binary]
    slope = reliability_slope(reliability_curve(probs, ys))
    skill = brier_skill_score(probs, ys)
    area = auc(probs, ys)

    print("  the other G5 criteria, on this data:")
    ok_slope = slope is not None and G5_RELIABILITY[0] <= slope <= G5_RELIABILITY[1]
    print(
        f"    reliability slope  {slope if slope is None else round(slope, 3)!s:<8} "
        f"want {G5_RELIABILITY[0]}-{G5_RELIABILITY[1]}   {'ok' if ok_slope else 'NO'}"
    )
    print(
        f"    Brier skill        {skill:<8.4f} want >{G5_BRIER_SKILL}          "
        f"{'ok' if skill > G5_BRIER_SKILL else 'NO'}"
    )
    print(
        f"    AUC                {area if area is None else round(area, 4)!s:<8} "
        f"want >={G5_AUC}         {'ok' if area is not None and area >= G5_AUC else 'NO'}"
    )
    print()

    if cmp.beats_baseline:
        print("  The learned model beats the rules baseline on daily bars with the")
        print("  interval excluding zero. That is a real result on this data and")
        print("  still says nothing about intraday, which is what G5 asks.")
    else:
        print("  The learned model does NOT beat the rules baseline on this data.")
        print("  Its own docstring names beating that baseline as the thing it")
        print("  exists to do. Recorded rather than retuned.")

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
            feature_hash=hashlib.sha256(",".join(sorted(rows[0])).encode()).hexdigest()[:16],
            metrics={
                "n": float(len(outcomes)),
                "m0_log_loss": m0.log_loss,
                "m1_log_loss": m1.log_loss,
                "delta_log_loss": cmp.delta_log_loss,
                "ci_lo": lo,
                "ci_hi": hi,
                "beats_baseline": float(cmp.beats_baseline),
                "reliability_slope": float("nan") if slope is None else slope,
                "brier_skill": skill,
                "auc": float("nan") if area is None else area,
                "folds_fitted": float(fitted),
                "folds_no_discrimination": float(no_disc),
            },
            notes=args.note
            or (
                f"{args.symbol} daily bars; DAILY ANALOGUES of intraday features, "
                "not a G5 measurement"
            ),
        )
    )
    print(f"\n  logged as trial {log.trial_count()} ({entry.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
