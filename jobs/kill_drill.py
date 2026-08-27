"""The monthly kill drill.

    python jobs/kill_drill.py --dry-run     rehearse, touch nothing
    python jobs/kill_drill.py --live        actually flatten (requires arming)

A kill switch that has never been pulled is a kill switch nobody knows the state
of. The blueprint says to test it monthly and to test what happens when the
broker returns an error during the flatten, because that is the scenario that
actually costs money.

The drill exercises the path that matters: escalate, flatten, discover what did
not close, escalate again. It runs against a broker that fails on purpose, so a
green drill means the failure handling works rather than that nothing went
wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.obs import configure
from libs.ms_risk.gate import (
    BookState,
    KillLevel,
    Position,
    RiskGate,
    flatten_all,
)

DRILL_LOG = Path(__file__).resolve().parents[1] / "research" / "kill_drills.jsonl"


class StubbornBroker:
    """Fails the first attempt on every position, then cooperates.

    Deliberately not a happy path. A drill against a broker that always works
    proves the drill runs, not that the flatten does.
    """

    def __init__(self, *, always_fail: set[str] | None = None) -> None:
        self.calls: dict[str, int] = {}
        self.always_fail = always_fail or set()

    def close(self, position: Position) -> None:
        self.calls[position.symbol] = self.calls.get(position.symbol, 0) + 1
        if position.symbol in self.always_fail:
            raise RuntimeError("broker rejected the order: instrument not tradeable")
        if self.calls[position.symbol] == 1:
            raise TimeoutError("broker did not respond within the timeout")


def drill(*, live: bool) -> int:
    cfg = cfgmod.load(warn=False)
    gate = RiskGate(cfg.risk_limits)

    positions = [
        Position(symbol="NIFTY24000CE", lots=2, is_long=True),
        Position(symbol="NIFTY24200CE", lots=1, is_long=False),
        Position(symbol="BANKNIFTY51000PE", lots=1, is_long=True),
    ]
    print()
    print(f"  {'LIVE' if live else 'DRY RUN'} kill drill, {len(positions)} open positions")
    print()

    print("  1. escalate on a dead feed")
    book = BookState(capital=500_000.0, open_positions=list(positions), feed_age_seconds=30.0)
    level = gate.assess_conditions(book)
    print(f"     kill level {level.name}, flatten required: {level.requires_flatten}")
    if not level.requires_flatten:
        print("     FAILED: a dead feed with open positions did not trigger a flatten")
        return 1

    print()
    print("  2. flatten, against a broker that fights back")
    broker = StubbornBroker(always_fail={"BANKNIFTY51000PE"})
    escalated: list[str] = []
    result = flatten_all(
        positions,
        broker.close if live else (lambda p: broker.close(p)),
        max_attempts=3,
        on_failure=lambda r: escalated.append(f"{len(r.failed)} positions still open"),
    )
    print(result.render())

    print()
    print("  3. the failure path")
    if result.complete:
        print("     FAILED: the drill's broker was set to refuse one position and it closed")
        return 1
    if not escalated:
        print("     FAILED: an incomplete flatten did not escalate")
        return 1
    print(f"     escalation hook fired: {escalated[0]}")

    print()
    print("  4. two L3 events in a week become a lockout")
    gate.escalate(KillLevel.L3_FLATTEN, "drill: second failure", today=datetime.now(UTC).date())
    print(f"     kill level {gate.kill_level.name}, needs a human: {gate.kill_level.needs_a_human}")
    if not gate.kill_level.needs_a_human:
        print("     FAILED: two L3 events did not lock out")
        return 1

    print()
    print("  5. the lockout refuses to clear without a written cause")
    try:
        gate.clear("ok")
    except ValueError as exc:
        print(f"     refused, correctly: {str(exc)[:70]}...")
    else:
        print("     FAILED: a lockout cleared without a root-cause note")
        return 1

    record = {
        "ts": datetime.now(UTC).isoformat(),
        "mode": "live" if live else "dry_run",
        "positions": len(positions),
        "closed": len(result.closed),
        "failed": sorted(result.failed),
        "attempts": result.attempts,
        "passed": True,
    }
    DRILL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DRILL_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + chr(10))

    print()
    print(f"  drill passed. Logged to {DRILL_LOG.name}")
    print("  A kill switch that has never been pulled is one nobody knows the state of.")
    return 0


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Monthly kill-switch drill (M76)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="rehearse, touch nothing")
    group.add_argument("--live", action="store_true", help="actually flatten")
    args = p.parse_args()

    if args.live:
        print()
        print("  --live flattens real positions. This drill ships with a stub broker,")
        print("  so --live is refused until a real adapter is wired in deliberately.")
        return 1
    return drill(live=False)


if __name__ == "__main__":
    sys.exit(main())
