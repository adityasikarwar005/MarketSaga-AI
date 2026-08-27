"""Read the trial log.

    python jobs/experiment_log.py --count      how many attempts have been made
    python jobs/experiment_log.py              the full history
    python jobs/experiment_log.py --verify     has the history been edited?

Every reported result is one of N attempts, and N is what separates a discovery
from the best of fifty coin flips. The deflated Sharpe ratio needs that N, so
this file is where it comes from - and the count is only meaningful if the runs
that did not work are still in it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_models.cv import EXPERIMENT_LOG, ExperimentLog, expected_max_sharpe


def main() -> int:
    p = argparse.ArgumentParser(description="Trial log (M51)")
    p.add_argument("--count", action="store_true", help="print the trial count only")
    p.add_argument("--verify", action="store_true", help="check the entries were not edited")
    p.add_argument("--prefix", default="", help="restrict to experiment names with this prefix")
    args = p.parse_args()

    log = ExperimentLog()
    entries = [e for e in log.all() if e.name.startswith(args.prefix)]

    if args.count:
        print(len(entries))
        return 0

    broken = log.verify()
    if args.verify:
        if broken:
            print(f"  CHAIN BROKEN at entries {broken}")
            print("  An entry was edited or removed. The trial count below is a lower bound")
            print("  at best, and every deflated number computed from it is void.")
            return 1
        print(f"  chain intact, {len(log.all())} entries")
        return 0

    if not entries:
        print(f"  no experiments recorded at {EXPERIMENT_LOG}")
        print("  A count of zero is not an achievement - it means nothing has been measured.")
        return 0

    print()
    print(f"  {'experiment':<28} {'when':<20} {'result'}")
    print("  " + "-" * 76)
    for e in entries:
        metrics = "  ".join(f"{k}={v:.4g}" for k, v in sorted(e.metrics.items()))
        print(f"  {e.name:<28} {e.ts[:19]:<20} {metrics}")
        if e.notes:
            print(f"  {'':<28} {e.notes}")

    n = len(entries)
    print()
    print(f"  {n} trials. Pure noise, tried {n} times, would produce a best Sharpe of about")
    print(
        f"  {expected_max_sharpe(n):.2f} - anything at or below that is not evidence of anything."
    )
    if broken:
        print(f"  WARNING: chain broken at {broken}; the history has been edited.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
