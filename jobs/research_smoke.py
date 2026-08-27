"""End-to-end research smoke test on real recorded data.

    python jobs/research_smoke.py --date 2026-08-26

Runs the whole research stack against a recorded session: features, labels,
purged splits, split audit, the time-of-day baseline, the impact library, and
the M0 baseline scored against realised outcomes. It proves the pieces compose,
and it is the shape every future training script will take.

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
from libs.ms_impact.baseline import (
    Regime,
    build_profile,
    classify_regimes,
    group_by_session,
)
from libs.ms_impact.study import build_library, observe_event, synthetic_events
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
    print()
    print("  " + lint_all().render().strip())
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
    print()
    print(summarise(labels, calendar).render())

    # ---- splits ---------------------------------------------------------
    samples = samples_from_labels(labels)
    splits = purged_kfold(samples, n_splits=5, embargo=timedelta(minutes=30))
    audits = audit_all(samples, splits)
    clean = all(a.ok for a in audits)
    print()
    print(
        f"  splits    : {len(splits)} purged folds, "
        f"{sum(s.purged for s in splits):,} purged, "
        f"{sum(s.embargoed for s in splits):,} embargoed"
    )
    print(f"  audit     : {'clean' if clean else 'LEAKS FOUND'}")
    for audit in audits:
        # Not `leak`: that name already holds a LeakReport from the feature
        # check above, and these are split-audit strings. Two different concepts
        # under one name is not a bug today and is one after the next edit.
        for overlap in audit.leaks[:2]:
            print(f"    {audit.split}: {overlap}")

    wf = walk_forward(samples, n_splits=4, anchored=True)
    print(f"  walkfwd   : {len(wf)} folds, " + ", ".join(s.sizes for s in wf[:2]) + " ...")

    # ---- impact library -------------------------------------------------
    per_session = group_by_session(loaded.candles, calendar, instrument_id=instrument_id)
    profile = build_profile(per_session, calendar, horizon_minutes=15)
    print()
    print(profile.render())

    regimes = classify_regimes(per_session)
    observations = []
    for event_id, cls, t0, severity in synthetic_events(per_session, calendar):
        event_day = calendar.local(t0).date()
        obs = observe_event(
            per_session[event_day],
            t0,
            event_id,
            cls,
            severity,
            profile,
            calendar,
            # .get() with an explicit default, not a bare .get(). A day that
            # classify_regimes could not label returns None, and passing that
            # through overrides observe_event's Regime.NORMAL default with None
            # - which then flows into abnormal_return and picks a baseline
            # keyed on nothing. Found by pyright; mypy never looked, because its
            # scope was libs/ only.
            regime=regimes.get(event_day, Regime.NORMAL),
        )
        if obs:
            observations.append(obs)

    if observations:
        library = build_library(observations, profile=profile, calendar=calendar)
        print()
        print(library.render(15))
        print(
            "\n  Those events are SYNTHETIC - random timestamps on a random walk."
            "\n  The library finding nothing is the only honest outcome when there"
            "\n  is nothing to find."
        )
    else:
        print("\n  no event observations - not enough history yet")

    # ---- baseline -------------------------------------------------------
    model = RulesBaseline()
    probs, outcomes = [], []
    for label in labels:
        features = REGISTRY.compute(source, instrument_id, label.t0 + timedelta(seconds=1))
        probs.append(model.predict(features).p_up)
        outcomes.append(int(label.outcome))

    base_rate = climatology(outcomes)
    result = compare(
        [0.5] * len(probs),
        probs,
        outcomes,
        baseline_name="coin_flip",
        challenger_name=model.name,
        bootstrap=1000,
    )

    print()
    print(f"  climatology (base rate of UP among barrier hits): {base_rate:.3f}")
    print(result.render())
    print()
    print(score(model.name, probs, outcomes).render())

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
