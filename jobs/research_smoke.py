"""End-to-end research smoke test on real recorded data.

    python jobs/research_smoke.py --date 2026-08-26

Runs the whole research stack against a recorded session: features, labels,
purged splits, split audit, and the M0 baseline scored against realised
outcomes. It proves the pieces compose, and it is the shape every future
training script will take.

The numbers it prints are not a result. One synthetic session is not evidence of
anything, and the baseline is meant to be beaten rather than to win. What this
job establishes is that the plumbing is honest: splits audit clean, effective
sample size is reported, and the scorecard is computed the same way the Phase 5
gate will compute it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import configure
from libs.ms_data.holidays import HolidayStore
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.leakage import detect_leakage, lint_all
from libs.ms_features.registry import REGISTRY
from libs.ms_features.sources import load_day
from libs.ms_models.baseline import RulesBaseline, climatology, compare, score
from libs.ms_models.cv import audit_all, purged_kfold, samples_from_labels, walk_forward
from libs.ms_models.labels import BarrierSpec, build_labels, summarise


def run(day: date, instrument_id: int, horizon: int) -> int:
    configure()
    cfg = cfgmod.load(warn=False)
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )

    print(f"\n  === {day} instrument {instrument_id} ===\n")

    loaded = load_day(day)
    if not loaded.candles:
        print("  nothing recorded for that day. Run: python tasks.py record")
        return 1
    print(f"  data      : {loaded.summary}")
    source = loaded.source()

    # ---- features -------------------------------------------------------
    print(f"\n  {lint_all().render().strip()}")
    bars = sorted(
        (c for c in loaded.candles if c.instrument_id == instrument_id), key=lambda c: c.ts
    )
    if len(bars) < 60:
        print(f"  only {len(bars)} candles for instrument {instrument_id} - nothing to do")
        return 1

    probe = bars[len(bars) // 2].ts + timedelta(minutes=1)
    leaks = detect_leakage(source, instrument_id, probe)
    print(f"  leakage   : {'none' if not leaks else str(len(leaks)) + ' FEATURES LEAK'}")
    for leak in leaks[:5]:
        print(f"    {leak}")

    # ---- labels ---------------------------------------------------------
    spec = BarrierSpec(horizon_minutes=horizon)
    labels = build_labels(loaded.candles, spec, calendar, instrument_id=instrument_id)
    if not labels:
        print("  no labels produced - session too short or volatility too low")
        return 1
    print("\n" + summarise(labels, calendar).render())

    # ---- splits ---------------------------------------------------------
    samples = samples_from_labels(labels)
    splits = purged_kfold(samples, n_splits=5, embargo=timedelta(minutes=30))
    audits = audit_all(samples, splits)
    clean = all(a.ok for a in audits)
    print(
        f"\n  splits    : {len(splits)} purged folds, "
        f"{sum(s.purged for s in splits):,} purged, {sum(s.embargoed for s in splits):,} embargoed"
    )
    print(f"  audit     : {'clean' if clean else 'LEAKS FOUND'}")
    for audit in audits:
        for leak in audit.leaks[:2]:
            print(f"    {audit.split}: {leak}")

    wf = walk_forward(samples, n_splits=4, anchored=True)
    print(f"  walkfwd   : {len(wf)} folds, " + ", ".join(s.sizes for s in wf[:2]) + " ...")

    # ---- baseline -------------------------------------------------------
    model = RulesBaseline()
    probs, outcomes = [], []
    for label in labels:
        features = REGISTRY.compute(source, instrument_id, label.t0 + timedelta(seconds=1))
        probs.append(model.predict(features).p_up)
        outcomes.append(int(label.outcome))

    base_rate = climatology(outcomes)
    always_flat = [0.5] * len(probs)
    result = compare(
        always_flat,
        probs,
        outcomes,
        baseline_name="coin_flip",
        challenger_name=model.name,
        bootstrap=1000,
    )

    print(f"\n  climatology (base rate of UP among barrier hits): {base_rate:.3f}")
    print(result.render())
    print(f"\n  {score(model.name, probs, outcomes).render().strip()}")

    print(
        "\n  Note: one session is not evidence. The baseline exists to be beaten,"
        "\n  and these numbers only show the pipeline is wired correctly."
    )
    return 0 if clean and not leaks else 1


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end research smoke test")
    p.add_argument("--date", required=True, help="recorded day (YYYY-MM-DD)")
    p.add_argument("--instrument", type=int, default=1)
    p.add_argument("--horizon", type=int, default=15, help="barrier horizon in minutes")
    args = p.parse_args()
    return run(date.fromisoformat(args.date), args.instrument, args.horizon)


if __name__ == "__main__":
    sys.exit(main())
