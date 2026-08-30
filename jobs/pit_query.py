"""M25 - query the point-in-time store, reproducibly.

    python jobs/pit_query.py --as-of "2026-07-14T11:03:00+05:30" --hash
    python jobs/pit_query.py --as-of ... --partitions live backfill --show 10

The ``--hash`` is the point. A point-in-time query that returns a different set
on two runs over the same store cannot support a reproducible backtest, and the
difference is invisible in a result you only look at once. Two runs of this
command on an unchanged store must print the same digest.

**The as-of must carry an offset.** A naive ``--as-of`` is refused rather than
assumed: reading ``11:03`` as UTC when it meant IST moves the boundary by five
and a half hours, and the query then returns a whole afternoon of items that
were not knowable. The same error class the log timestamps and the publish-time
resolver both guard against, arriving this time through a command line.

**Backfill is opt-in here too.** The default reads the live partition only,
because a live ``t_available`` was measured and a backfilled one was
reconstructed, and a digest computed over a mix of the two is not comparable
with one computed over either.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_news.pit_store import Partition, PitEventStore  # noqa: E402


def parse_as_of(raw: str) -> datetime:
    """ISO 8601 with an offset. Naive values are refused, not assumed."""
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(
            f"  --as-of {raw!r} is not ISO 8601. Expected something like\n"
            "  2026-07-14T11:03:00+05:30"
        ) from None
    if value.tzinfo is None:
        raise SystemExit(
            f"  --as-of {raw!r} has no timezone offset.\n"
            "  It is refused rather than assumed: read as UTC when it meant IST,\n"
            "  the point-in-time boundary moves five and a half hours and the\n"
            "  query returns an afternoon of items that were not knowable.\n"
            "  Write it as 2026-07-14T11:03:00+05:30."
        )
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--as-of", required=True, help="ISO 8601 with an offset")
    p.add_argument(
        "--partitions",
        nargs="+",
        default=["live"],
        choices=[m.value for m in Partition],
        help="default live only; backfill is opt-in",
    )
    p.add_argument("--source-id", type=int, default=None)
    p.add_argument("--lang", default=None)
    p.add_argument("--show", type=int, default=0, help="print this many headlines")
    p.add_argument("--hash", action="store_true", help="print only the digest")
    args = p.parse_args(argv)

    configure()
    as_of = parse_as_of(args.as_of)
    partitions = tuple(Partition(p_) for p_ in args.partitions)

    store = PitEventStore()
    result = store.as_of(as_of, partitions=partitions, source_id=args.source_id, lang=args.lang)

    if args.hash:
        print(result.digest)
        return 0

    print()
    print(result.render())

    counts = store.counts()
    print(f"  store holds {counts['live']:,} live and {counts['backfill']:,} backfill rows")
    if not any(counts.values()):
        print()
        print("  The store is empty. Every source in config/sources.yaml is still")
        print("  disabled pending its terms review, so nothing has been collected.")
        print("  The digest above is the digest of an empty query - stable, and")
        print("  not evidence that anything works end to end.")
        return 0

    if args.show:
        print()
        for item in result.items[: args.show]:
            flag = " ~" if item.ts_uncertain else "  "
            print(
                f"   {flag} {item.t_event.isoformat()}  [{item.publish_time_method}]  "
                f"{item.headline[:70]}"
            )
        if len(result) > args.show:
            print(f"      ... and {len(result) - args.show:,} more")
        print()
        print("  ~ marks ts_uncertain: the publish time is a guess or the ingest")
        print("    fallback, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
