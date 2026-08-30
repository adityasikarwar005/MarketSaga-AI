"""M22 - the publish-time resolver, and the ways it can quietly lie.

The failures here do not raise. Each one produces a timestamp that looks like a
timestamp and is wrong by an amount that reads as latency, which is exactly the
quantity every news study in this repository is trying to measure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.normalise import (
    FUTURE_TOLERANCE,
    PublishTimeMethod,
    TimeCandidate,
    detect_language,
    normalise_text,
    resolve_publish_time,
    stamp_available,
)

IST = timezone(timedelta(hours=5, minutes=30))
INGEST = datetime(2026, 8, 30, 6, 0, tzinfo=UTC)


def _c(method: PublishTimeMethod, value: datetime | None) -> TimeCandidate:
    return TimeCandidate(method=method, value=value)


class TestCascadeOrder:
    def test_prefers_structured_over_meta_and_text(self) -> None:
        r = resolve_publish_time(
            [
                _c(PublishTimeMethod.IN_TEXT, INGEST - timedelta(hours=3)),
                _c(PublishTimeMethod.HTML_META, INGEST - timedelta(hours=2)),
                _c(PublishTimeMethod.STRUCTURED, INGEST - timedelta(hours=1)),
            ],
            t_ingest=INGEST,
        )
        assert r.method is PublishTimeMethod.STRUCTURED
        assert r.t_event == INGEST - timedelta(hours=1)
        assert r.measured

    def test_a_bad_structured_value_does_not_veto_a_good_meta_one(self) -> None:
        """The first *plausible* candidate wins, not the first present one.

        A publisher with a broken pubDate and a correct og:published_time is a
        real and common shape; falling all the way to ingest because the most
        trusted extractor was wrong throws away a good measurement.
        """
        r = resolve_publish_time(
            [
                _c(PublishTimeMethod.STRUCTURED, INGEST + timedelta(days=2)),
                _c(PublishTimeMethod.HTML_META, INGEST - timedelta(minutes=5)),
            ],
            t_ingest=INGEST,
        )
        assert r.method is PublishTimeMethod.HTML_META
        assert "in_the_future" in r.note

    def test_in_text_is_accepted_but_never_trusted(self) -> None:
        """Right often enough to be dangerous. Usable, flagged, not measured."""
        r = resolve_publish_time(
            [_c(PublishTimeMethod.IN_TEXT, INGEST - timedelta(hours=1))], t_ingest=INGEST
        )
        assert r.method is PublishTimeMethod.IN_TEXT
        assert r.uncertain
        assert not r.measured


class TestNaiveTimestamps:
    """The 5.5-hour error, in the direction that makes an item look knowable
    before it was."""

    def test_a_naive_value_with_no_source_tz_is_refused(self) -> None:
        naive = datetime(2026, 8, 30, 11, 0)  # 11:00 - but 11:00 where?
        r = resolve_publish_time([_c(PublishTimeMethod.STRUCTURED, naive)], t_ingest=INGEST)
        assert r.method is PublishTimeMethod.INGEST_FALLBACK
        assert r.uncertain
        assert "naive_no_source_tz" in r.note

    def test_a_naive_value_with_a_source_tz_is_converted_not_assumed(self) -> None:
        naive = datetime(2026, 8, 30, 11, 0)  # 11:00 IST
        r = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, naive)], t_ingest=INGEST, source_tz=IST
        )
        assert r.t_event == datetime(2026, 8, 30, 5, 30, tzinfo=UTC)
        assert r.measured

    def test_reading_a_naive_ist_value_as_utc_would_shift_it_by_five_and_a_half_hours(
        self,
    ) -> None:
        """The mistake this guard exists for, stated as an assertion.

        If the resolver ever defaults a naive value to UTC, this difference is
        what lands in t_event - and an item published at 11:00 IST would be
        recorded as knowable at 11:00 UTC, five and a half hours early.
        """
        naive = datetime(2026, 8, 30, 11, 0)
        correct = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, naive)], t_ingest=INGEST, source_tz=IST
        ).t_event
        wrong = naive.replace(tzinfo=UTC)
        assert wrong - correct == timedelta(hours=5, minutes=30)


class TestImplausibleTimestamps:
    def test_a_future_timestamp_is_rejected_rather_than_clamped(self) -> None:
        """Clamping produces a plausible timestamp and hides the fault."""
        r = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, INGEST + timedelta(hours=6))], t_ingest=INGEST
        )
        assert r.method is PublishTimeMethod.INGEST_FALLBACK
        assert r.t_event == INGEST, "the fallback is ingest, not the clamped future value"

    def test_small_clock_skew_is_tolerated(self) -> None:
        """Publisher clocks differ from ours by seconds. Rejecting on that
        would push good items onto the fallback for no gain."""
        r = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, INGEST + FUTURE_TOLERANCE - timedelta(seconds=1))],
            t_ingest=INGEST,
        )
        assert r.method is PublishTimeMethod.STRUCTURED

    def test_an_implausibly_old_value_is_rejected(self) -> None:
        """A two-digit year or a day/month swap, not a genuine archive item."""
        r = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, INGEST - timedelta(days=400))], t_ingest=INGEST
        )
        assert r.method is PublishTimeMethod.INGEST_FALLBACK
        assert "implausibly_old" in r.note

    def test_a_backfill_collector_can_widen_the_past_bound(self) -> None:
        old = INGEST - timedelta(days=400)
        r = resolve_publish_time(
            [_c(PublishTimeMethod.STRUCTURED, old)],
            t_ingest=INGEST,
            past_tolerance=timedelta(days=3650),
        )
        assert r.t_event == old


class TestFallback:
    def test_flags_uncertain_when_nothing_is_parseable(self) -> None:
        r = resolve_publish_time([], t_ingest=INGEST)
        assert r.method is PublishTimeMethod.INGEST_FALLBACK
        assert r.t_event == INGEST
        assert r.uncertain
        assert "upper bound" in r.note

    def test_the_fallback_is_never_reported_as_measured(self) -> None:
        assert not resolve_publish_time([], t_ingest=INGEST).measured


class TestTextNormalisation:
    def test_markup_is_stripped_before_entities_are_decoded(self) -> None:
        """The other order deletes text the publisher meant literally."""
        assert "<script>" in normalise_text("<p>use &lt;script&gt; carefully</p>")

    def test_nfkc_folds_lookalikes_so_two_copies_hash_the_same(self) -> None:
        """The same headline copied from two sites differs by a non-breaking
        space, hashes differently, and arrives in M23 as two events.

        The exotic spaces are written as escapes on purpose: a literal NBSP
        in source is invisible, and a reader cannot check a test they cannot
        see the input of.
        """
        plain = normalise_text("RBI holds rates")
        nbsp = normalise_text("RBI\u00a0holds\u00a0rates")
        narrow = normalise_text("RBI\u202fholds\u202frates")
        assert plain == nbsp == narrow == "RBI holds rates"

    def test_boilerplate_is_removed(self) -> None:
        text = normalise_text("Rates held steady.\nAlso read: our coverage\n")
        assert "Also read" not in text
        assert "Rates held steady." in text


class TestLanguage:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("RBI holds the repo rate", "en"),
            ("रिज़र्व बैंक ने दरें स्थिर रखीं", "hi"),
            ("12345 67890", "und"),
        ],
    )
    def test_script_detection(self, text: str, expected: str) -> None:
        assert detect_language(text) == expected


class TestAvailability:
    def _item(self) -> RawNewsItem:
        return RawNewsItem(
            source_id=2,
            headline="NSE circular",
            t_event=INGEST - timedelta(minutes=1),
            t_ingest=INGEST,
        )

    def test_available_is_stamped_after_the_pipeline(self) -> None:
        done = INGEST + timedelta(seconds=4)
        assert stamp_available(self._item(), now=done).t_available == done

    def test_available_is_never_before_ingest(self) -> None:
        """Availability before arrival is not a possible state; a clock that
        suggests it is a clock to distrust, not a fact to store."""
        earlier = INGEST - timedelta(seconds=30)
        assert stamp_available(self._item(), now=earlier).t_available == INGEST

    def test_known_at_uses_available_not_ingest(self) -> None:
        """The gap between the two is this system's own latency, and a strategy
        whose horizon is seconds cannot have it recorded as zero."""
        item = stamp_available(self._item(), now=INGEST + timedelta(seconds=10))
        assert not item.known_at(INGEST + timedelta(seconds=5))
        assert item.known_at(INGEST + timedelta(seconds=10))
