"""Weekly incremental and quarterly full retrains.

    python jobs/retrain.py --dry-run --report
    python jobs/retrain.py --kind incremental --apply

One rule does the work: **promote only if it beats the incumbent on the last
four weeks out-of-sample.** Every part of it is load-bearing, and this job's
whole purpose is to refuse when any part is not satisfied.

--dry-run is the default and --apply is the opposite. A pipeline that promotes
on every scheduled run changes what is trading without anyone deciding to, and
in the log that change looks like routine maintenance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_models.cv import ExperimentLog
from libs.ms_ops.registry import MIN_SHADOW_SESSIONS, ModelVersion, Registry, Stage
from libs.ms_ops.retrain import RetrainKind, Scored, oos_window, retrain, summarise


def demo_registry(today: date) -> tuple[Registry, ModelVersion, Scored, Scored]:
    """A registry with an incumbent and a challenger that has done its time.

    Synthetic, and it says so. There is no trained model to promote yet - the
    direction model has never seen real intraday data - so this exercises the
    gate rather than pretending to have passed it.
    """
    from datetime import timedelta

    r = Registry()
    r.register(
        ModelVersion(
            version="dir-v1",
            trained_on=today - timedelta(days=120),
            trained_through=today - timedelta(days=120),
            stage=Stage.PRODUCTION,
            promoted_on=today - timedelta(days=118),
        )
    )
    challenger = r.register(
        ModelVersion(
            version="dir-v2",
            trained_on=today - timedelta(days=35),
            trained_through=today - timedelta(days=35),
            stage=Stage.SHADOW,
            shadow_sessions=MIN_SHADOW_SESSIONS,
        )
    )
    start, end = oos_window(challenger)
    return (
        r,
        challenger,
        Scored("dir-v2", start, end, 480, 0.6412, 0.2210),
        Scored("dir-v1", start, end, 480, 0.6503, 0.2264),
    )


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Retraining pipeline (MA2)")
    p.add_argument("--kind", choices=[k.value for k in RetrainKind], default="incremental")
    p.add_argument("--dry-run", action="store_true", help="evaluate and report, change nothing")
    p.add_argument("--apply", action="store_true", help="promote if the check passes")
    p.add_argument("--report", action="store_true", help="print the evaluation")
    p.add_argument("--date", help="override the run date")
    args = p.parse_args()

    if args.apply and args.dry_run:
        print("  --dry-run and --apply mean opposite things. Pick one.")
        return 2

    today = date.fromisoformat(args.date) if args.date else datetime.now(UTC).date()
    kind = RetrainKind(args.kind)

    registry, challenger, challenger_score, incumbent_score = demo_registry(today)

    if args.report:
        print()
        print(registry.render())
        print()
        print(summarise([incumbent_score, challenger_score]))

    result = retrain(
        registry,
        kind=kind,
        today=today,
        challenger=challenger,
        challenger_score=challenger_score,
        incumbent_score=incumbent_score,
        apply=args.apply,
        experiment_log=ExperimentLog(),
    )

    print()
    print(result.render())

    if args.dry_run or not args.apply:
        print()
        print("  Nothing was changed. --apply promotes, and only if the check passes.")

    print()
    print("  Note: there is no trained model to promote yet - the direction model")
    print("  has never seen real intraday data. This exercises the gate rather than")
    print("  pretending to have passed it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
