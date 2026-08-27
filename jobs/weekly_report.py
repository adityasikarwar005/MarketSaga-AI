"""The weekly read, and the auto-disable decision that comes with it.

    python jobs/weekly_report.py --week-ending 2026-03-06 --report
    python jobs/weekly_report.py --week-ending 2026-03-06 --report --apply

Two things happen here and only one of them is reporting.

The report says what the week did. The **kill policy** decides what stops
trading, and the plan's note is that this decision does more for long-run
profitability than any model improvement. A model right on eight classes and
quietly bleeding on the ninth does not look broken - it looks like a mediocre
average, and the instinct is to improve the model rather than to stop trading
the ninth.

``--apply`` is required to actually disable anything. Without it the run is a
dry read, because a policy that silently changes what trades on every scheduled
run is a policy nobody remembers agreeing to.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_ops.attribution import (
    ROLLING_WINDOW,
    KillPolicy,
    brier_breakdown,
    build_weekly_report,
)
from libs.ms_risk.ledger import PaperLedger

REPORT_ROOT = Path(__file__).resolve().parents[1] / "research" / "weekly"
REGISTER = REPORT_ROOT / "kill_register.json"


def load_policy() -> KillPolicy:
    """The register is state, not a derived value.

    Recomputing it each week would re-enable a class the moment a quiet quarter
    pushed its rolling expectancy back above zero - the same class, the same
    model, and the same reason it lost money.
    """
    policy = KillPolicy()
    if not REGISTER.exists():
        return policy
    from libs.ms_ops.attribution import Disablement

    raw = json.loads(REGISTER.read_text(encoding="utf-8"))
    for name, d in raw.items():
        d["disabled_on"] = date.fromisoformat(d["disabled_on"])
        if d.get("reenabled_on"):
            d["reenabled_on"] = date.fromisoformat(d["reenabled_on"])
        policy.register[name] = Disablement(**d)
    return policy


def save_policy(policy: KillPolicy) -> None:
    REGISTER.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, d in policy.register.items():
        row = asdict(d)
        row["disabled_on"] = d.disabled_on.isoformat()
        row["reenabled_on"] = d.reenabled_on.isoformat() if d.reenabled_on else None
        payload[name] = row
    REGISTER.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def collect(ledger: PaperLedger, *, week_ending: date, window: timedelta) -> list:
    """Every resolved prediction inside the rolling window."""
    start = week_ending - window
    entries = []
    for stamp in ledger.days():
        day = date.fromisoformat(stamp)
        if start <= day <= week_ending:
            entries.extend(ledger.entries(datetime.combine(day, datetime.min.time(), tzinfo=UTC)))
    return entries


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Weekly attribution and kill policy (M94)")
    p.add_argument("--week-ending", required=True, help="YYYY-MM-DD")
    p.add_argument("--report", action="store_true", help="print the report")
    p.add_argument("--apply", action="store_true", help="actually disable what qualifies")
    p.add_argument("--model-version", default="unversioned")
    args = p.parse_args()

    week_ending = date.fromisoformat(args.week_ending)
    ledger = PaperLedger()
    entries = collect(ledger, week_ending=week_ending, window=ROLLING_WINDOW)

    if not entries:
        print()
        print(f"  nothing in the ledger for the quarter ending {week_ending}")
        print("  A weekly report over an empty ledger is not a good week, it is")
        print("  a week with no record - and those are not the same thing.")
        return 1

    policy = load_policy()
    before = set(policy.disabled_classes)
    report = build_weekly_report(
        entries, policy, week_ending=week_ending, model_version=args.model_version
    )

    if args.report:
        print()
        print(report.render())
        print()
        print(brier_breakdown(entries))

    if report.newly_disabled and not args.apply:
        print()
        print(f"  DRY RUN: {len(report.newly_disabled)} class(es) qualify to be disabled.")
        print("  Re-run with --apply to make it so. A policy that silently changes")
        print("  what trades on every scheduled run is one nobody remembers agreeing to.")
        # Undo the in-memory disablement so the saved register is untouched.
        for d in report.newly_disabled:
            policy.register.pop(d.event_class, None)
    elif args.apply:
        save_policy(policy)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        out = REPORT_ROOT / f"{week_ending.isoformat()}.txt"
        out.write_text(report.render(), encoding="utf-8")
        print()
        print(f"  archived to {out.name}; register now disables {policy.disabled_classes}")
        if set(policy.disabled_classes) != before:
            print("  Re-enabling any of these needs a changed model or a written")
            print("  justification. Waiting is not a change.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
