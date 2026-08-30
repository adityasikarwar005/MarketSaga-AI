"""M23 - deduplication, and the merge that must never happen.

Wire services republish, aggregators copy, and one press release arrives from
five places in ninety seconds. Without dedup, M29 counts those five as five
independent confirmations of the same fact, and a position gets sized on
corroboration that does not exist.

**The dangerous direction is over-merging, not under-merging.** A missed
duplicate costs a little double-counting. A wrong merge destroys an event: two
different things that happened to the same subject become one, and the one that
survives is whichever arrived first. There is no later signal that anything went
wrong - the cluster looks exactly like a correct one.

So the layers are ordered by how much they are trusted, and each is allowed to
merge less than the last.

Layer 1 - exact content hash
------------------------------

Identical normalised text. Nothing to argue about, and it catches the bulk: the
same wire item fetched twice, or the same release on two of a publisher's feeds.

Layer 2 - SimHash, 64 bits, Hamming <= 3
------------------------------------------

SimHash over word shingles, at the threshold the plan specifies. What it
actually buys was measured rather than assumed, and the answer is less than the
plan implies.

**On a 47-word article body it does what it is meant to.** A byline appended
scores 2 and merges; one word changed scores 7 and does not; a sentence dropped
scores 10 and does not. So the layer catches boilerplate-level variation on full
text and nothing more.

**On headlines it is close to inert, and no threshold fixes that.** Measured
over a set of genuine duplicates and a set of opposite events, both on real
headline-length text:

    genuine duplicates   hamming 11, 16, 24, 26
    opposite events      hamming 12, 15, 17, 20, 21, 27, 28

The two distributions **overlap completely**. There is no cut that admits the
first set and excludes the second. A 3-shingle set over an eight-word headline
has five members, and changing one word moves three of them - the distance stops
carrying information at that length.

Hamming <= 3 is the number from web-scale document dedup, where the documents
are pages. It is kept because it is the conservative end of a scale that has no
safe end: it merges almost nothing, which is the direction of error this module
wants.

**Which makes the guard below the thing doing the work, not a supplement to
it.** :func:`decisive_tokens` and the figures in the text are checked *before*
distance is allowed to merge anything, so "RBI keeps the repo rate unchanged"
and "RBI cuts the repo rate to 6.25%" cannot become one event at any threshold
somebody later decides to loosen. It is deliberately blunt: it will refuse some
genuine duplicates, and that is the cheap direction.

The honest reading of all of this is that **real near-duplicate detection on
headlines needs layer 3**, and layer 3 does not exist.

Layer 3 - embeddings
----------------------

Not implemented, and not stubbed. The plan asks for sentence-embedding cosine
above 0.93 within a thirty-minute window; there are no embeddings until M27
exists, and a layer that always returns "no match" is indistinguishable in the
output from one that is working. :class:`DedupLayer` names it so the report can
state that paraphrase detection is *absent* rather than letting a silent zero
read as "no paraphrases found". Same reasoning as M25's missing vector index.

Clusters count sources, not members
-------------------------------------

M29 asks "how many independent sources said this". Five copies from one wire is
one source. :attr:`Cluster.source_count` is the size of the distinct source set,
and :attr:`Cluster.max_credibility` is the best source in it, because a cluster
containing an exchange notice is worth more than one containing five blogs.

Windows
---------

Similarity outside a time window is not duplication. "Sensex closes higher" is
the same text on many days, and a hash that matches a week later is a republish
rather than a second report of one event. Every layer respects the window.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import RawNewsItem

log = get_logger(__name__)

SIMHASH_BITS = 64
MAX_HAMMING = 3
SHINGLE = 3
DEFAULT_WINDOW = timedelta(minutes=30)

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Words that carry the *direction* of an event rather than its subject. Two
# items that disagree on any of these are reporting different things, however
# similar the rest of the sentence is. The list is short on purpose: every entry
# is a word whose presence flips what a position should do.
# Words that carry the *direction* of an event rather than its subject. Two
# items that disagree on any of these are reporting different things, however
# similar the rest of the sentence is.
#
# Stems, not inflections. The first version of this listed forms by hand and
# had "halt" and "halted" but not "halts", so "NSE halts trading" and "NSE
# resumes trading" were allowed to merge - and a neighbouring pair passed only
# because one of its two words happened to be in the list. A hand-maintained
# inflection table is precisely the kind of thing that rots silently.
DECISIVE_STEMS = frozenset(
    (
        # held vs moved - the distinction a rate decision turns on
        "hold",
        "unchanged",
        "steady",
        "maintain",
        # which way it moved
        "cut",
        "lower",
        "reduce",
        "raise",
        "hike",
        "increase",
        # which way a price or figure went
        "up",
        "down",
        "rise",
        "fall",
        "gain",
        "loss",
        "lose",
        "widen",
        "narrow",
        "surge",
        "plunge",
        "jump",
        "drop",
        # against an expectation
        "beat",
        "miss",
        "above",
        "below",
        # a decision that went one way or the other
        "approve",
        "reject",
        "grant",
        "deny",
        "halt",
        "resume",
        "suspend",
        "lift",
    )
)

# Irregular forms a suffix stripper cannot reach. Kept short and explicit; an
# entry here is a word whose past tense shares no prefix with its stem.
IRREGULAR = {
    "held": "hold",
    "rose": "rise",
    "fell": "fall",
    "lost": "lose",
    "beat": "beat",
    "cut": "cut",
    "missed": "miss",
}


def stem(word: str) -> str:
    """Crude suffix stripping, enough to make the decisive list stable.

    Not a linguistics exercise. It exists so that adding "halt" to the list also
    covers "halts", "halted" and "halting", because the version that did not do
    that shipped with a hole in it.
    """
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix in ("ing", "ed", "es", "s"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        base = word[: -len(suffix)]
        if base in DECISIVE_STEMS:
            return base
        # silent e: resumed -> resume, surged -> surge, plunged -> plunge
        if base + "e" in DECISIVE_STEMS:
            return base + "e"
        # doubled consonant: dropped -> drop, jumped -> jump
        if len(base) > 1 and base[-1] == base[-2] and base[:-1] in DECISIVE_STEMS:
            return base[:-1]
    return word


class DedupLayer(StrEnum):
    EXACT = "exact"
    SIMHASH = "simhash"
    EMBEDDING = "embedding"  # declared, not implemented - see the module docstring

    @property
    def implemented(self) -> bool:
        return self is not DedupLayer.EMBEDDING


class DedupError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = SHINGLE) -> list[str]:
    """Overlapping n-grams. Short texts fall back to their own words.

    A headline of four words has two 3-shingles, which is enough to be
    distinctive; one of two words has none, and hashing an empty set would make
    every short headline identical to every other.
    """
    words = tokens(text)
    if len(words) < n:
        return words
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def content_hash(text: str) -> str:
    """Layer 1. Over the token stream, so whitespace and case cannot split a pair."""
    return hashlib.sha256(" ".join(tokens(text)).encode("utf-8")).hexdigest()


def simhash(text: str, *, bits: int = SIMHASH_BITS) -> int:
    """Charikar SimHash over shingles.

    Each shingle votes on every bit with weight one; the sign of the column sum
    becomes the bit. Similar documents disagree on few bits, which is what makes
    Hamming distance meaningful here at all.
    """
    grams = shingles(text)
    if not grams:
        return 0
    columns = [0] * bits
    for gram in grams:
        h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            columns[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, total in enumerate(columns):
        if total > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def decisive_tokens(text: str) -> frozenset[str]:
    """The words that carry which way an event went, reduced to their stems."""
    return frozenset(st for t in tokens(text) if (st := stem(t)) in DECISIVE_STEMS)


def numbers(text: str) -> frozenset[str]:
    """Figures as written. 25 and 50 basis points are different decisions."""
    return frozenset(_NUMBER.findall(text))


def may_merge(a: str, b: str) -> tuple[bool, str]:
    """Whether two near-identical texts are allowed to be one event.

    Returns the reason as well as the verdict, because a refusal is the
    interesting outcome and a boolean alone cannot be audited.
    """
    da, db = decisive_tokens(a), decisive_tokens(b)
    if da != db:
        return False, f"decisive words differ: {sorted(da ^ db)}"
    na, nb = numbers(a), numbers(b)
    if na != nb:
        return False, f"figures differ: {sorted(na ^ nb)}"
    return True, "no decisive disagreement"


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #
@dataclass
class Cluster:
    """One event, and everything that reported it."""

    head_id: str
    head_text: str
    head_hash: str
    head_simhash: int
    first_seen: datetime
    member_ids: list[str] = field(default_factory=list)
    source_ids: set[int] = field(default_factory=set)
    max_credibility: float = 0.0
    joined_by: list[DedupLayer] = field(default_factory=list)

    @property
    def size(self) -> int:
        return 1 + len(self.member_ids)

    @property
    def source_count(self) -> int:
        """Distinct sources, not copies.

        M29 reads this as corroboration. Five items from one wire is one source
        saying something once, and counting it as five is how a single feed
        talks a position into existence.
        """
        return len(self.source_ids)


@dataclass(frozen=True)
class DedupResult:
    """What happened to one item."""

    item: RawNewsItem
    cluster_head: str
    layer: DedupLayer | None  # None means it started a cluster
    reason: str

    @property
    def is_duplicate(self) -> bool:
        return self.layer is not None


class Deduplicator:
    """Layered dedup over a rolling window.

    Not a store: it holds only what the window needs, and the point-in-time
    store keeps the items themselves. Feeding it in ingest order is the caller's
    job - the cluster head is the first item *seen*, and out-of-order feeding
    silently changes which item that is.
    """

    def __init__(
        self,
        *,
        window: timedelta = DEFAULT_WINDOW,
        max_hamming: int = MAX_HAMMING,
        credibility: dict[int, float] | None = None,
    ) -> None:
        self.window = window
        self.max_hamming = max_hamming
        self.credibility = credibility or {}
        self.clusters: dict[str, Cluster] = {}
        self.refused: list[tuple[str, str]] = []  # (head_id, reason) - audit trail

    def _live(self, now: datetime) -> list[Cluster]:
        return [c for c in self.clusters.values() if now - c.first_seen <= self.window]

    def add(self, item: RawNewsItem) -> DedupResult:
        text = item.headline if item.body is None else f"{item.headline}\n{item.body}"
        h = content_hash(text)
        sh = simhash(text)
        when = item.t_event

        for cluster in self._live(when):
            if cluster.head_hash == h:
                return self._attach(cluster, item, DedupLayer.EXACT, "identical text")

        for cluster in self._live(when):
            distance = hamming(cluster.head_simhash, sh)
            if distance > self.max_hamming:
                continue
            allowed, why = may_merge(cluster.head_text, text)
            if not allowed:
                # The whole point of the module. Counted and kept, because a
                # refusal that leaves no trace cannot be reviewed.
                self.refused.append((cluster.head_id, why))
                METRICS.inc("news_merge_refused")
                log.info(
                    "near-duplicate refused",
                    fields={"head": cluster.head_id, "hamming": distance, "reason": why},
                )
                continue
            return self._attach(cluster, item, DedupLayer.SIMHASH, f"hamming {distance}; {why}")

        cluster = Cluster(
            head_id=item.id,
            head_text=text,
            head_hash=h,
            head_simhash=sh,
            first_seen=when,
            source_ids={item.source_id},
            max_credibility=self.credibility.get(item.source_id, 0.0),
        )
        self.clusters[item.id] = cluster
        METRICS.inc("news_clusters_opened")
        return DedupResult(item=item, cluster_head=item.id, layer=None, reason="new event")

    def _attach(
        self, cluster: Cluster, item: RawNewsItem, layer: DedupLayer, reason: str
    ) -> DedupResult:
        cluster.member_ids.append(item.id)
        cluster.source_ids.add(item.source_id)
        cluster.max_credibility = max(
            cluster.max_credibility, self.credibility.get(item.source_id, 0.0)
        )
        cluster.joined_by.append(layer)
        METRICS.inc("news_duplicates_found", layer=layer.value)
        return DedupResult(
            item=item.model_copy(update={"duplicate_of": cluster.head_id}),
            cluster_head=cluster.head_id,
            layer=layer,
            reason=reason,
        )

    def add_all(self, items: Iterable[RawNewsItem]) -> list[DedupResult]:
        return [self.add(i) for i in items]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DedupReport:
    seen: int
    duplicates: int
    clusters: int
    by_layer: dict[str, int]
    refused: int
    corroborated: int  # clusters with more than one distinct source

    @property
    def rate(self) -> float:
        return self.duplicates / self.seen if self.seen else 0.0

    def render(self) -> list[str]:
        lines = [
            f"    {self.seen:,} items -> {self.clusters:,} events, "
            f"{self.duplicates:,} duplicates ({self.rate:.1%})",
        ]
        for layer in DedupLayer:
            if not layer.implemented:
                lines.append(
                    f"      {layer.value:<10} NOT IMPLEMENTED - no embeddings until M27. "
                    "Paraphrases are not being detected, rather than absent."
                )
                continue
            lines.append(f"      {layer.value:<10} {self.by_layer.get(layer.value, 0):,}")
        lines.append(
            f"      refused    {self.refused:,} near-matches held apart by the "
            "decisive-word and figure guards"
        )
        lines.append(
            f"      {self.corroborated:,} of {self.clusters:,} events have more than one "
            "distinct source"
        )
        return lines


def summarise(results: Sequence[DedupResult], dedup: Deduplicator) -> DedupReport:
    by_layer: dict[str, int] = {}
    for r in results:
        if r.layer is not None:
            by_layer[r.layer.value] = by_layer.get(r.layer.value, 0) + 1
    return DedupReport(
        seen=len(results),
        duplicates=sum(1 for r in results if r.is_duplicate),
        clusters=len(dedup.clusters),
        by_layer=by_layer,
        refused=len(dedup.refused),
        corroborated=sum(1 for c in dedup.clusters.values() if c.source_count > 1),
    )
