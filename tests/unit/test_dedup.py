"""M23 - deduplication, and the labelled set of merges that must never happen.

The spec names the trap: over-aggressive dedup merges two *different* events on
the same topic. That failure is silent. The cluster looks correct, M29 counts
one event where there were two, and the one that survives is whichever arrived
first - which on a rate decision means the model reads "unchanged" on a day the
rate was cut.

`FALSE_MERGE_SET` below is the labelled set the plan asks for. Every pair in it
is textually near-identical and semantically opposite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.dedup import (
    DECISIVE_STEMS,
    DedupLayer,
    Deduplicator,
    content_hash,
    decisive_tokens,
    hamming,
    may_merge,
    simhash,
    stem,
    summarise,
)

T0 = datetime(2026, 8, 30, 6, 0, tzinfo=UTC)

# 47 words. Layer 2 is only meaningful at this length - see the module docstring.
LONG_BODY = (
    "The Reserve Bank of India kept the policy repo rate unchanged at 6.50 percent "
    "following the Monetary Policy Committee meeting concluded on Friday morning, "
    "citing stable inflation and a resilient growth outlook for the coming quarters. "
    "The decision was unanimous among the six members of the committee."
)
CRED = {1: 1.00, 2: 1.00, 4: 0.75, 90: 0.25}


def _item(headline: str, *, source_id: int = 1, at: datetime = T0, body: str | None = None):
    return RawNewsItem(source_id=source_id, headline=headline, body=body, t_event=at, t_ingest=at)


# --------------------------------------------------------------------------- #
# The labelled false-merge set. Near-identical text, opposite meaning.
# --------------------------------------------------------------------------- #
FALSE_MERGE_SET = [
    (
        "RBI keeps the repo rate unchanged at 6.50%",
        "RBI cuts the repo rate to 6.25%",
    ),
    (
        "RBI raises the repo rate by 25 bps",
        "RBI cuts the repo rate by 25 bps",
    ),
    (
        "NSE halts trading in XYZ Ltd",
        "NSE resumes trading in XYZ Ltd",
    ),
    (
        "Board approves the merger with ABC Ltd",
        "Board rejects the merger with ABC Ltd",
    ),
    (
        "Quarterly profit rose 12% year on year",
        "Quarterly profit fell 12% year on year",
    ),
    (
        "Inflation print comes in above consensus",
        "Inflation print comes in below consensus",
    ),
    (
        "RBI cuts the repo rate by 25 bps",
        "RBI cuts the repo rate by 50 bps",
    ),
]


class TestFalseMergeGuard:
    @pytest.mark.parametrize(("a", "b"), FALSE_MERGE_SET)
    def test_false_merge_guard_keeps_opposite_events_apart(self, a: str, b: str) -> None:
        allowed, reason = may_merge(a, b)
        assert not allowed, (
            f"{a!r} and {b!r} would be merged into one event. They are near-identical "
            f"text and opposite meaning; the cluster that results looks correct and "
            f"the surviving item is whichever arrived first."
        )
        assert reason

    def test_false_merge_guard_is_load_bearing_because_no_threshold_separates_them(
        self,
    ) -> None:
        """The measurement that replaced an assumption I had written down.

        The docstring first claimed these pairs have a *small* Hamming distance
        and that the guard stops SimHash merging them. Measured, that is false:
        on headline-length text the genuine-duplicate and opposite-event
        distance distributions overlap completely, so there is no cut that
        admits one and excludes the other.

        The guard is not a supplement to the threshold. It is the only thing
        that makes any threshold safe to loosen, which is what this pins.
        """
        genuine = [
            (
                "RBI keeps the repo rate unchanged at 6.50%",
                "RBI keeps repo rate unchanged at 6.50%",
            ),
            ("NSE halts trading in XYZ Ltd", "NSE halts trading in XYZ Ltd, exchange says"),
            ("Board approves the merger with ABC Ltd", "Board approves merger with ABC Ltd"),
        ]
        dup_d = [hamming(simhash(a), simhash(b)) for a, b in genuine]
        opp_d = [hamming(simhash(a), simhash(b)) for a, b in FALSE_MERGE_SET]
        assert max(dup_d) >= min(opp_d), (
            "the two distance distributions have become separable, which would "
            "mean a threshold alone could do this job - re-derive the docstring's "
            "claim before relying on it"
        )
        for a, b in FALSE_MERGE_SET:
            assert not may_merge(a, b)[0]

    def test_false_merge_guard_refusals_are_recorded_with_a_reason(self) -> None:
        """A refusal that leaves no trace cannot be reviewed."""
        d = Deduplicator(max_hamming=64)  # force every pair through the guard
        d.add(_item("RBI keeps the repo rate unchanged at 6.50%"))
        d.add(_item("RBI cuts the repo rate to 6.25%", source_id=2))
        assert d.refused
        assert "decisive words differ" in d.refused[0][1]

    def test_the_guard_does_not_block_a_genuine_duplicate(self) -> None:
        allowed, _ = may_merge(
            "RBI keeps the repo rate unchanged at 6.50%",
            "RBI keeps the repo rate unchanged at 6.50% - full text",
        )
        assert allowed


class TestExact:
    def test_exact_duplicate_is_merged(self) -> None:
        d = Deduplicator()
        first = d.add(_item("RBI holds the policy rate"))
        second = d.add(_item("RBI holds the policy rate", source_id=4))
        assert first.layer is None
        assert second.layer is DedupLayer.EXACT
        assert second.item.duplicate_of == first.cluster_head

    def test_exact_match_ignores_case_and_spacing(self) -> None:
        assert content_hash("RBI  holds the RATE") == content_hash("rbi holds the rate")

    def test_exact_match_outside_the_window_is_a_republish_not_a_duplicate(self) -> None:
        """'Sensex closes higher' is the same text on many days."""
        d = Deduplicator(window=timedelta(minutes=30))
        d.add(_item("Sensex closes higher"))
        later = d.add(_item("Sensex closes higher", at=T0 + timedelta(days=1)))
        assert later.layer is None


class TestNearDuplicates:
    def test_near_dup_is_merged(self) -> None:
        """What layer 2 actually catches: boilerplate-level variation on a body.

        Measured: on a 47-word article, a byline appended scores 2 and merges;
        one word changed scores 7 and does not. The threshold is narrow and this
        test pins the width rather than the hope.
        """
        d = Deduplicator()
        head = d.add(_item("RBI policy decision", body=LONG_BODY))
        again = d.add(_item("RBI policy decision", body=LONG_BODY + " - Reuters", source_id=4))
        assert again.layer is DedupLayer.SIMHASH
        assert again.cluster_head == head.cluster_head

    def test_a_larger_edit_is_not_merged_at_this_threshold(self) -> None:
        """One changed word scores 7 against a limit of 3. Under-merging is the
        direction this module chooses when it has to choose."""
        d = Deduplicator()
        d.add(_item("RBI policy decision", body=LONG_BODY))
        other = d.add(
            _item("RBI policy decision", body=LONG_BODY.replace("resilient", "robust"), source_id=4)
        )
        assert other.layer is None

    def test_near_dup_respects_the_window(self) -> None:
        d = Deduplicator(window=timedelta(minutes=5))
        d.add(_item("Reserve Bank of India holds the policy repo rate steady"))
        later = d.add(
            _item(
                "Reserve Bank of India holds the policy repo rate steady",
                at=T0 + timedelta(hours=2),
            )
        )
        assert later.layer is None

    def test_simhash_is_deterministic(self) -> None:
        """Two runs on the same text must agree, or a cluster cannot be
        reproduced and neither can anything built from one."""
        text = "NSE issues a circular on margin requirements"
        assert simhash(text) == simhash(text)

    def test_a_short_headline_still_gets_a_distinctive_hash(self) -> None:
        """Below the shingle length the words themselves are used. Hashing an
        empty shingle set would make every short headline identical."""
        assert simhash("rate cut") != simhash("rate hike")


class TestParaphrase:
    def test_paraphrase_is_not_detected_and_that_is_declared(self) -> None:
        """Layer 3 does not exist and the report says so.

        A layer that always returns "no match" is indistinguishable in the
        output from one that is working, so the absence is named rather than
        left to be inferred from a zero.
        """
        d = Deduplicator()
        a = d.add(_item("RBI holds the repo rate"))
        b = d.add(_item("The central bank left borrowing costs where they were", source_id=4))
        assert b.layer is None, "if this ever merges, layer 3 arrived without a test"
        assert a.cluster_head != b.cluster_head

        report = summarise([a, b], d)
        rendered = "\n".join(report.render())
        assert "NOT IMPLEMENTED" in rendered
        assert "M27" in rendered

    def test_the_embedding_layer_reports_itself_as_unimplemented(self) -> None:
        assert not DedupLayer.EMBEDDING.implemented
        assert DedupLayer.EXACT.implemented
        assert DedupLayer.SIMHASH.implemented


class TestClusters:
    def test_a_cluster_counts_sources_not_copies(self) -> None:
        """M29 reads this as corroboration. Five items from one wire is one
        source saying something once."""
        d = Deduplicator(credibility=CRED)
        head = d.add(_item("NSE halts trading in XYZ Ltd", source_id=2))
        for _ in range(4):
            d.add(_item("NSE halts trading in XYZ Ltd", source_id=2))
        cluster = d.clusters[head.cluster_head]
        assert cluster.size == 5
        assert cluster.source_count == 1

    def test_two_sources_are_corroboration(self) -> None:
        d = Deduplicator(credibility=CRED)
        head = d.add(_item("NSE halts trading in XYZ Ltd", source_id=2))
        d.add(_item("NSE halts trading in XYZ Ltd", source_id=4))
        cluster = d.clusters[head.cluster_head]
        assert cluster.source_count == 2

    def test_max_credibility_takes_the_best_source_in_the_cluster(self) -> None:
        """A cluster containing an exchange notice is worth more than one
        containing five blogs."""
        d = Deduplicator(credibility=CRED)
        head = d.add(_item("NSE halts trading in XYZ Ltd", source_id=90))
        assert d.clusters[head.cluster_head].max_credibility == 0.25
        d.add(_item("NSE halts trading in XYZ Ltd", source_id=2))
        assert d.clusters[head.cluster_head].max_credibility == 1.00

    def test_the_head_is_the_first_item_seen(self) -> None:
        d = Deduplicator()
        first = d.add(_item("NSE halts trading in XYZ Ltd", source_id=4))
        second = d.add(
            _item("NSE halts trading in XYZ Ltd", source_id=2, at=T0 + timedelta(minutes=1))
        )
        assert second.cluster_head == first.item.id


class TestReport:
    def test_the_rate_is_duplicates_over_items(self) -> None:
        d = Deduplicator()
        results = d.add_all(
            [
                _item("NSE halts trading in XYZ Ltd"),
                _item("NSE halts trading in XYZ Ltd", source_id=4),
                _item("Board approves the merger with ABC Ltd", source_id=4),
            ]
        )
        report = summarise(results, d)
        assert report.seen == 3
        assert report.duplicates == 1
        assert report.clusters == 2
        assert report.rate == pytest.approx(1 / 3)

    def test_an_empty_run_reports_zero_rather_than_dividing_by_zero(self) -> None:
        d = Deduplicator()
        assert summarise([], d).rate == 0.0

    def test_refusals_are_surfaced_in_the_report(self) -> None:
        d = Deduplicator(max_hamming=64)
        results = d.add_all(
            [
                _item("RBI keeps the repo rate unchanged at 6.50%"),
                _item("RBI cuts the repo rate to 6.25%", source_id=4),
            ]
        )
        report = summarise(results, d)
        assert report.refused >= 1
        assert "refused" in "\n".join(report.render())


class TestDecisiveStemming:
    """The hole the first version shipped with.

    The decisive list was written as hand-typed inflections and had "halt" and
    "halted" but not "halts". So "NSE halts trading in XYZ Ltd" and "NSE resumes
    trading in XYZ Ltd" were allowed to merge into one event - a halt and its
    resumption becoming a single record, with whichever arrived first surviving.

    A neighbouring pair passed only because one of its two words happened to be
    in the table, which is worse than failing: it looked like the guard working.
    """

    @pytest.mark.parametrize(
        "word",
        ["halt", "halts", "halted", "halting", "resume", "resumes", "resumed"],
    )
    def test_every_inflection_reaches_the_same_stem(self, word: str) -> None:
        assert decisive_tokens(f"NSE {word} trading in XYZ Ltd")

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("NSE halts trading", "NSE resumes trading"),
            ("Board approves the deal", "Board rejects the deal"),
            ("Profit jumped 12%", "Profit dropped 12%"),
            ("Shares surged on the news", "Shares plunged on the news"),
        ],
    )
    def test_inflected_opposites_are_still_caught(self, a: str, b: str) -> None:
        assert not may_merge(a, b)[0]

    def test_doubled_consonants_are_handled(self) -> None:
        """dropped -> drop, not droppe."""
        assert stem("dropped") == "drop"
        assert stem("jumped") == "jump"

    def test_irregulars_that_a_suffix_stripper_cannot_reach(self) -> None:
        assert stem("held") == "hold"
        assert stem("rose") == "rise"
        assert stem("fell") == "fall"

    def test_an_ordinary_word_is_left_alone(self) -> None:
        """The stemmer only has to be right about the decisive list; mangling
        everything else would be a cost with no benefit."""
        assert stem("trading") == "trading"
        assert not decisive_tokens("NSE issues a circular on margin requirements")

    def test_every_stem_in_the_list_is_reachable_from_its_own_inflections(self) -> None:
        """The meta-test for the rot this class exists because of.

        Adding a word to DECISIVE_STEMS is only useful if the forms a publisher
        actually writes reduce to it. This walks the whole list rather than
        trusting that whoever added an entry also thought about its endings.

        `up` is excluded: it is an adverb here, and "ups"/"uping" are not forms
        any headline uses.
        """
        unreachable = [
            (base, base + suffix)
            for base in sorted(DECISIVE_STEMS)
            if base != "up"
            for suffix in ("s", "es", "ed", "ing")
            if stem(base + suffix) != base
        ]
        assert not unreachable, (
            "these inflections do not reduce to their stem, so a headline using "
            f"them slips past the guard: {unreachable}"
        )
