"""M23 - the duplicate rate, and the merges that were refused.

    python jobs/dedup_report.py --days 30
    python jobs/dedup_report.py --days 30 --show-refused 20

G2 asks for a duplicate rate below 3%. That number on its own is the least
useful thing this job prints, because **it improves when dedup gets more
aggressive**, and more aggressive is the direction that destroys events. A run
that merged every item into one cluster would report a superb rate.

So the refusals are printed next to it. Every one is a pair that SimHash
considered near-identical and the decisive-word or figure guard held apart -
"halts" against "resumes", 25 bps against 50. Those are the spot check the plan
asks for: read them, and if any is a genuine duplicate the guard is too blunt;
if any *should* have been refused and was not, the guard has a hole.

**Layer 3 does not exist**, so paraphrases are not being detected at all. The
report says so rather than letting a low duplicate rate read as a clean corpus.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_news.dedup import DEFAULT_WINDOW, Deduplicator, summarise  # noqa: E402
from libs.ms_news.pit_store import Partition, PitEventStore  # noqa: E402

G2_DUP_RATE = 0.03


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--window-minutes", type=float, default=DEFAULT_WINDOW.total_seconds() / 60)
    p.add_argument("--show-refused", type=int, default=10)
    args = p.parse_args(argv)

    configure()
    cfg = cfgmod.load(warn=False)
    clock = RealClock()
    now = clock.now()

    store = PitEventStore()
    result = store.as_of(now, partitions=(Partition.LIVE,), since=now - timedelta(days=args.days))
    items = list(result.items)

    print()
    print(f"  window {args.days} days, cluster window {args.window_minutes:g} minutes")
    print()

    if not items:
        counts = store.counts()
        enabled = [s for s in cfg.sources.sources if s.enabled]
        print(f"  no items. The store holds {counts['live']:,} live rows in total, and")
        print(f"  {len(enabled)} of {len(cfg.sources.sources)} sources are enabled.")
        print()
        print("  There is no duplicate rate to report and G2 cannot be assessed.")
        print("  A rate computed over zero items would be 0.0%, which reads as a")
        print("  clean corpus and is the most misleading number this job could")
        print("  print - so it does not print one.")
        return 1

    credibility = {s.id: s.credibility for s in cfg.sources.sources}
    dedup = Deduplicator(window=timedelta(minutes=args.window_minutes), credibility=credibility)
    report = summarise(dedup.add_all(items), dedup)

    for line in report.render():
        print(line)
    print()

    if report.rate < G2_DUP_RATE:
        print(f"  duplicate rate {report.rate:.1%} is under G2's {G2_DUP_RATE:.0%}.")
        print("  That is necessary and not sufficient: the rate falls when dedup")
        print("  merges more, and merging more is what destroys events. Read the")
        print("  refusals below before treating it as a pass.")
    else:
        print(f"  duplicate rate {report.rate:.1%} is at or above G2's {G2_DUP_RATE:.0%}.")

    if dedup.refused:
        print()
        print(f"  refused merges - the spot check ({len(dedup.refused):,} total):")
        for head, why in dedup.refused[: args.show_refused]:
            print(f"    {head[:12]}  {why}")
        if len(dedup.refused) > args.show_refused:
            print(f"    ... and {len(dedup.refused) - args.show_refused:,} more")
        print()
        print("  Each is a pair SimHash called near-identical and the guard held")
        print("  apart. If one of these is a genuine duplicate the guard is too")
        print("  blunt; if two items that should differ were merged instead, it")
        print("  has a hole. Both are findings; neither shows up in the rate.")

    print()
    print("  Layer 3 (embeddings) is not implemented, so paraphrases are not")
    print("  detected at all. The rate above is a rate for exact and near-exact")
    print("  text only, and a low one does not mean the corpus is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
