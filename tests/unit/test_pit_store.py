"""M25 - the point-in-time store, and the four-second lie.

Every failure here returns a plausible list of news items. None of them raise.
The difference between a correct store and a broken one is a strategy that is
either real or is astonishingly good at the open.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.pit_store import Partition, PitEventStore, PitStoreError

T0 = datetime(2026, 8, 30, 6, 0, tzinfo=UTC)


def _item(
    *,
    headline: str = "RBI holds",
    event: datetime = T0,
    ingest: datetime | None = None,
    available: datetime | None = None,
    source_id: int = 1,
    lang: str = "en",
) -> RawNewsItem:
    ingest = ingest or event
    return RawNewsItem(
        source_id=source_id,
        headline=headline,
        lang=lang,
        t_event=event,
        t_ingest=ingest,
        t_available=available,
    )


@pytest.fixture
def store(tmp_path: Path) -> PitEventStore:
    return PitEventStore(root=tmp_path / "news")


class TestAsOf:
    def test_as_of_excludes_future(self, store: PitEventStore) -> None:
        store.append(_item(headline="now", event=T0))
        store.append(_item(headline="later", event=T0 + timedelta(minutes=10)))
        got = store.as_of(T0 + timedelta(minutes=1))
        assert [i.headline for i in got.items] == ["now"]

    def test_the_gate_is_availability_not_publish_time(self, store: PitEventStore) -> None:
        """The bug this module exists to prevent.

        Published at 11:00:00, this system finished with it at 11:00:04. At
        11:00:01 it was not knowable, and a store filtering on t_event returns
        it anyway - which is four seconds of free information at an event
        horizon measured in seconds.
        """
        published = T0
        store.append(
            _item(
                headline="slow pipeline",
                event=published,
                ingest=published,
                available=published + timedelta(seconds=4),
            )
        )
        assert len(store.as_of(published + timedelta(seconds=1))) == 0
        assert len(store.as_of(published + timedelta(seconds=4))) == 1

    def test_ingest_is_the_fallback_when_available_is_unset(self, store: PitEventStore) -> None:
        store.append(_item(event=T0 - timedelta(hours=1), ingest=T0, available=None))
        assert len(store.as_of(T0 - timedelta(minutes=1))) == 0
        assert len(store.as_of(T0)) == 1

    def test_results_are_sorted_by_event_time(self, store: PitEventStore) -> None:
        store.append(_item(headline="second", event=T0 + timedelta(minutes=2)))
        store.append(_item(headline="first", event=T0 + timedelta(minutes=1)))
        got = store.as_of(T0 + timedelta(hours=1))
        assert [i.headline for i in got.items] == ["first", "second"]

    def test_filters_compose(self, store: PitEventStore) -> None:
        store.append(_item(headline="rbi", source_id=1))
        store.append(_item(headline="nse", source_id=2))
        store.append(_item(headline="hindi", source_id=1, lang="hi"))
        got = store.as_of(T0 + timedelta(hours=1), source_id=1, lang="en")
        assert [i.headline for i in got.items] == ["rbi"]


class TestPartitions:
    def test_backfill_partition_separate(self, store: PitEventStore) -> None:
        """A live t_available was measured; a backfilled one was reconstructed.

        Same field, two different meanings, and the difference is invisible at
        query time - so backfill is excluded unless it is asked for.
        """
        store.append(_item(headline="live"), partition=Partition.LIVE)
        store.append(_item(headline="archived"), partition=Partition.BACKFILL)

        default = store.as_of(T0 + timedelta(hours=1))
        assert [i.headline for i in default.items] == ["live"]
        assert default.partitions == (Partition.LIVE,)

        both = store.as_of(T0 + timedelta(hours=1), partitions=(Partition.LIVE, Partition.BACKFILL))
        assert sorted(i.headline for i in both.items) == ["archived", "live"]
        assert both.partitions == (Partition.LIVE, Partition.BACKFILL)

    def test_the_result_records_what_it_was_allowed_to_read(self, store: PitEventStore) -> None:
        """A number from live-only data and one with backfill mixed in are not
        comparable, and the moment it is produced is the only reliable moment to
        record which it was."""
        got = store.as_of(T0, partitions=(Partition.BACKFILL,))
        assert got.partitions == (Partition.BACKFILL,)
        assert "backfill" in got.render()

    def test_the_two_partitions_write_to_different_files(self, store: PitEventStore) -> None:
        a = store.append(_item(), partition=Partition.LIVE)
        b = store.append(_item(), partition=Partition.BACKFILL)
        assert a != b
        assert store.days(Partition.LIVE) == store.days(Partition.BACKFILL)


class TestDigest:
    def test_the_digest_is_reproducible(self, store: PitEventStore) -> None:
        for n in range(5):
            store.append(_item(headline=f"h{n}", event=T0 + timedelta(minutes=n)))
        t = T0 + timedelta(hours=1)
        assert store.as_of(t).digest == store.as_of(t).digest

    def test_the_digest_survives_a_different_file_read_order(self, tmp_path: Path) -> None:
        """Two days of files must hash the same however the directory lists
        them; otherwise a query is reproducible only on the machine that ran it."""
        s = PitEventStore(root=tmp_path / "news")
        s.append(_item(headline="mon", event=T0, ingest=T0))
        s.append(_item(headline="tue", event=T0 + timedelta(days=1), ingest=T0 + timedelta(days=1)))
        t = T0 + timedelta(days=2)
        first = s.as_of(t).digest
        assert PitEventStore(root=tmp_path / "news").as_of(t).digest == first

    def test_the_digest_changes_when_membership_changes(self, store: PitEventStore) -> None:
        store.append(_item(headline="one"))
        t = T0 + timedelta(hours=1)
        before = store.as_of(t).digest
        store.append(_item(headline="two"))
        assert store.as_of(t).digest != before

    def test_the_digest_separates_partition_choices(self, store: PitEventStore) -> None:
        store.append(_item(headline="live"))
        t = T0 + timedelta(hours=1)
        a = store.as_of(t, partitions=(Partition.LIVE,)).digest
        b = store.as_of(t, partitions=(Partition.LIVE, Partition.BACKFILL)).digest
        assert a != b, "two queries over different corpora must not share a digest"


class TestCorrections:
    def test_a_superseding_row_replaces_the_earlier_one(self, store: PitEventStore) -> None:
        original = _item(headline="Rate cut expected")
        store.append(original)
        correction = _item(headline="Correction: rate held", event=T0 + timedelta(minutes=5))
        store.append(correction, supersedes=original.id)
        got = store.as_of(T0 + timedelta(hours=1))
        assert [i.headline for i in got.items] == ["Correction: rate held"]

    def test_a_correction_that_has_not_arrived_yet_censors_nothing(
        self, store: PitEventStore
    ) -> None:
        """What was believed at the time is what as_of must return.

        A correction published at 11:05 does not make the 11:00 item unknowable
        at 11:02 - that would let the store rewrite history the moment history
        disagreed with it.
        """
        original = _item(headline="Rate cut expected")
        store.append(original)
        later = T0 + timedelta(minutes=5)
        store.append(
            _item(headline="Correction", event=later, ingest=later), supersedes=original.id
        )
        at_1102 = store.as_of(T0 + timedelta(minutes=2))
        assert [i.headline for i in at_1102.items] == ["Rate cut expected"]

    def test_the_original_row_is_never_edited(self, store: PitEventStore) -> None:
        original = _item(headline="Rate cut expected")
        path = store.append(original)
        before = path.read_text(encoding="utf-8")
        store.append(
            _item(headline="Correction", event=T0 + timedelta(minutes=5)), supersedes=original.id
        )
        assert before in path.read_text(encoding="utf-8")


class TestWriteGuards:
    def test_available_before_ingest_is_refused(self, store: PitEventStore) -> None:
        """Storing it would make as_of() return the item before it existed."""
        with pytest.raises(PitStoreError, match="precedes t_ingest"):
            store.append(_item(ingest=T0, available=T0 - timedelta(seconds=1)))

    def test_partitioned_by_ingest_not_event(self, store: PitEventStore) -> None:
        """A late item about last week must not rewrite last week's file."""
        p = store.append(_item(event=T0 - timedelta(days=7), ingest=T0))
        assert p.stem == T0.date().isoformat()

    def test_a_truncated_line_is_skipped_rather_than_fatal(self, store: PitEventStore) -> None:
        """What a killed writer leaves behind. 09:14 is the wrong time to crash
        on a half-written row."""
        path = store.append(_item(headline="good"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"partition": "live", "item": {"headl')
        got = store.as_of(T0 + timedelta(hours=1))
        assert [i.headline for i in got.items] == ["good"]


class TestEmpty:
    def test_an_empty_store_returns_an_empty_result_not_an_error(
        self, store: PitEventStore
    ) -> None:
        got = store.as_of(T0)
        assert len(got) == 0
        assert got.digest  # still a stable, reportable value
        assert store.counts() == {"live": 0, "backfill": 0}
