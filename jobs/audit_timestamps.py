"""M22 - audit where the corpus's publish times actually came from.

    python jobs/audit_timestamps.py --sample 200 --manual-review

Gate G2 asks for publish times within five seconds of truth for at least 90% of
a sample of 200. That is a question about a corpus, and it has an answer only
once one exists.

**This job will not manufacture one.** Where there is no collected news it says
so and exits non-zero, because a verify command that passes on an empty corpus
is the most expensive kind of green: it certifies the gate rather than the
system, and the failure surfaces months later as research built on timestamps
nobody checked.

What it reports when there is a corpus is the **distribution of extraction
methods**, which is the thing the manual review needs in front of it before it
starts. Fifty items resolved from an RSS ``pubDate`` and fifty from a date
scraped out of prose are not one sample of a hundred; the second group is a
guess and its error is systematically larger. Reviewing a mixed sample and
quoting one accuracy number for it is how a corpus with a 30% guessed minority
comes to be described as 92% accurate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_news.normalise import PublishTimeMethod  # noqa: E402
from libs.ms_news.sources import Registry  # noqa: E402

NEWS_DIR = ROOT / "data" / "news"
G2_SAMPLE = 200
G2_WITHIN_SECONDS = 5.0
G2_ACCURACY = 0.90


def load_items(limit: int) -> list[dict]:
    """Whatever has been collected, newest files first. No corpus is not an error
    to swallow - the caller decides what to do with an empty list."""
    if not NEWS_DIR.exists():
        return []
    items: list[dict] = []
    for path in sorted(NEWS_DIR.glob("*.ndjson"), reverse=True):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(items) >= limit:
                return items
    return items


def report_no_corpus(registry: Registry) -> int:
    print()
    print(f"  no collected news under {NEWS_DIR.relative_to(ROOT)}.")
    print()
    enabled = registry.enabled()
    print(f"  {len(enabled)} of {len(registry.sources)} sources are enabled.")
    if not enabled:
        print("  Every source in config/sources.yaml is switched off, each with a")
        print("  tos_note reading TODO. That is the configured state, not a fault:")
        print("  reading a provider's terms is not a coding task, and a source")
        print("  enabled before somebody read them is a legal problem wearing a")
        print("  feature's clothes.")
        print()
        print("  The two highest-credibility slots need no API key at all - central")
        print("  bank releases and exchange notices are public. They are the ones")
        print("  worth clearing terms on first.")
    print()
    print(f"  G2 needs {G2_SAMPLE} items, {G2_ACCURACY:.0%} within {G2_WITHIN_SECONDS:.0f}s")
    print("  of truth. With no corpus that question has no answer, and reporting")
    print("  one would be certifying the gate rather than the system.")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample", type=int, default=G2_SAMPLE)
    p.add_argument(
        "--manual-review",
        action="store_true",
        help="write the sample out for a human to check against the source pages",
    )
    p.add_argument("--out", default=str(ROOT / "build" / "timestamp_review.ndjson"))
    args = p.parse_args(argv)

    configure()
    registry = Registry.from_config(cfgmod.load(warn=False).sources)

    items = load_items(args.sample)
    if not items:
        return report_no_corpus(registry)

    print()
    print(f"  sampled {len(items):,} items from {NEWS_DIR.relative_to(ROOT)}")
    print()

    methods = Counter(i.get("publish_time_method", "unknown") for i in items)
    uncertain = sum(1 for i in items if i.get("ts_uncertain"))
    print(f"    {'method':<20} {'items':>7} {'share':>7}  trusted")
    for name, n in methods.most_common():
        try:
            trusted = PublishTimeMethod(name).trusted
        except ValueError:
            trusted = False
        print(f"    {name:<20} {n:>7,} {n / len(items):>6.1%}  {'yes' if trusted else 'no'}")
    print()

    measured = sum(
        n
        for name, n in methods.items()
        if name in {m.value for m in PublishTimeMethod if m.trusted}
    )
    print(f"  {measured:,} of {len(items):,} ({measured / len(items):.1%}) come from a")
    print("  trusted extractor. The rest are guesses or the ingest fallback, and")
    print("  quoting one accuracy figure across both groups would hide that.")
    print(f"  {uncertain:,} are flagged ts_uncertain.")

    if len(items) < args.sample:
        print()
        print(f"  Only {len(items)} items available; G2 asks for {args.sample}.")
        print("  Not a failure - just not yet an answer.")

    if args.manual_review:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for i in items:
                fh.write(
                    json.dumps(
                        {
                            "id": i.get("id"),
                            "url": i.get("url"),
                            "headline": i.get("headline"),
                            "t_event": i.get("t_event"),
                            "publish_time_method": i.get("publish_time_method"),
                            "ts_uncertain": i.get("ts_uncertain"),
                            "truth": None,  # filled in by the reviewer
                        }
                    )
                    + "\n"
                )
        print()
        print(f"  wrote {len(items):,} rows to {out.relative_to(ROOT)} for review.")
        print("  Fill in `truth` from the source page. Review each method group")
        print("  separately: a mixed sample gives one number that describes none")
        print("  of them.")

    print()
    print("  G2 is not decided here. This job says where the timestamps came")
    print("  from; only the manual pass says whether they are right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
