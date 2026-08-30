"""M21/M22 - the RBI and NSE collectors, exercised without touching either.

Every test here runs against bytes. A collector that can only be tested against
a live publisher is a collector first tested on the morning that publisher
changes something.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.clock import SimClock
from libs.ms_news.feeds import (
    IST,
    MAX_FEED_BYTES,
    FeedError,
    RssCollector,
    nse_collector,
    parse_feed,
    parse_feed_datetime,
    rbi_collector,
)
from libs.ms_news.normalise import PublishTimeMethod

NOW = datetime(2026, 8, 30, 6, 0, tzinfo=UTC)  # 11:30 IST

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>RBI Press Releases</title>
  <item>
    <title>RBI keeps the policy repo rate unchanged</title>
    <link>https://example.invalid/pr/1</link>
    <description>The Monetary Policy Committee decided to hold.</description>
    <pubDate>Sun, 30 Aug 2026 11:00:00 +0530</pubDate>
    <guid>pr-1</guid>
  </item>
  <item>
    <title>Governor's statement</title>
    <link>https://example.invalid/pr/2</link>
    <description>Text of the statement.</description>
    <pubDate>Sun, 30 Aug 2026 11:05:00 +0530</pubDate>
    <guid>pr-2</guid>
  </item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>NSE circulars</title>
  <entry>
    <title>Trading halt in XYZ</title>
    <link href="https://example.invalid/circ/9"/>
    <summary>Halted pending clarification.</summary>
    <published>2026-08-30T11:00:00+05:30</published>
    <id>circ-9</id>
  </entry>
</feed>"""


def _feed(payload: str):
    return lambda _url: payload.encode("utf-8")


class TestParsing:
    def test_rss_items_are_read(self) -> None:
        entries = parse_feed(RSS.encode())
        assert [e["guid"] for e in entries] == ["pr-1", "pr-2"]
        assert entries[0]["link"] == "https://example.invalid/pr/1"

    def test_atom_entries_are_read_including_the_href_link(self) -> None:
        """Atom puts the URL in an attribute, RSS in the element text. A parser
        that handles only one silently drops every url from the other."""
        entries = parse_feed(ATOM.encode())
        assert entries[0]["guid"] == "circ-9"
        assert entries[0]["link"] == "https://example.invalid/circ/9"
        assert entries[0]["published"] == "2026-08-30T11:00:00+05:30"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Sun, 30 Aug 2026 11:00:00 +0530", datetime(2026, 8, 30, 11, 0, tzinfo=IST)),
            ("2026-08-30T11:00:00+05:30", datetime(2026, 8, 30, 11, 0, tzinfo=IST)),
            ("", None),
            ("not a date", None),
        ],
    )
    def test_both_date_dialects(self, raw: str, expected: datetime | None) -> None:
        assert parse_feed_datetime(raw) == expected


class TestUntrustedInput:
    """A feed is somebody else's XML, fetched over the network."""

    def test_a_doctype_is_refused_without_parsing(self) -> None:
        """The billion-laughs shape: kilobytes of declarations, gigabytes in
        memory. A collector that dies of it goes quiet, which is the failure
        mode hardest to notice."""
        bomb = b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaa">]><rss/>'
        with pytest.raises(FeedError, match="DOCTYPE"):
            parse_feed(bomb)

    def test_an_oversized_body_is_refused(self) -> None:
        with pytest.raises(FeedError, match="larger than"):
            parse_feed(b"<rss/>" + b"x" * (MAX_FEED_BYTES + 1))

    def test_malformed_xml_raises_rather_than_returning_nothing(self) -> None:
        """Zero entries and a broken feed must not look the same: one is a quiet
        morning and the other is a source that needs attention."""
        with pytest.raises(FeedError, match="well-formed"):
            parse_feed(b"<rss><channel><item></rss>")


class TestPublishTime:
    def test_an_offset_pubdate_is_carried_through_as_measured(self) -> None:
        c = rbi_collector(fetcher=_feed(RSS))
        items = c.poll(clock=SimClock(NOW))
        assert items[0].publish_time_method == PublishTimeMethod.STRUCTURED.value
        assert items[0].t_event == datetime(2026, 8, 30, 5, 30, tzinfo=UTC)
        assert not items[0].ts_uncertain

    def test_a_naive_pubdate_uses_the_source_timezone_not_utc(self) -> None:
        """RBI and NSE publish in IST. Reading an unmarked 11:00 as UTC stamps
        the item five and a half hours early - lookahead that reads as speed."""
        naive = RSS.replace("Sun, 30 Aug 2026 11:00:00 +0530", "Sun, 30 Aug 2026 11:00:00")
        items = rbi_collector(fetcher=_feed(naive)).poll(clock=SimClock(NOW))
        assert items[0].t_event == datetime(2026, 8, 30, 5, 30, tzinfo=UTC)
        wrong = datetime(2026, 8, 30, 11, 0, tzinfo=UTC)
        assert wrong - items[0].t_event == timedelta(hours=5, minutes=30)

    def test_a_missing_pubdate_falls_back_and_is_flagged(self) -> None:
        without = RSS.replace("<pubDate>Sun, 30 Aug 2026 11:00:00 +0530</pubDate>", "")
        items = rbi_collector(fetcher=_feed(without)).poll(clock=SimClock(NOW))
        first = next(i for i in items if i.headline.startswith("RBI keeps"))
        assert first.publish_time_method == PublishTimeMethod.INGEST_FALLBACK.value
        assert first.ts_uncertain
        assert first.t_event == NOW


class TestPolling:
    def test_a_second_poll_emits_nothing_new(self) -> None:
        """Feeds repeat their whole window every poll. Without the seen-set a
        five-minute cadence turns one press release into a hundred and eighty
        items, each with a different ingest time and each looking like a
        separate event to M23."""
        c = rbi_collector(fetcher=_feed(RSS))
        assert len(c.poll(clock=SimClock(NOW))) == 2
        assert c.poll(clock=SimClock(NOW + timedelta(minutes=5))) == []

    def test_a_new_entry_still_comes_through(self) -> None:
        c = rbi_collector(fetcher=_feed(RSS))
        c.poll(clock=SimClock(NOW))
        extended = RSS.replace(
            "</channel>",
            "<item><title>Third</title><link>https://example.invalid/pr/3</link>"
            "<pubDate>Sun, 30 Aug 2026 11:10:00 +0530</pubDate><guid>pr-3</guid></item></channel>",
        )
        c._fetch = _feed(extended)
        assert [i.headline for i in c.poll(clock=SimClock(NOW))] == ["Third"]

    def test_an_entry_with_no_title_is_dropped(self) -> None:
        """Nothing downstream can classify or deduplicate it."""
        untitled = RSS.replace("<title>Governor's statement</title>", "<title></title>")
        items = rbi_collector(fetcher=_feed(untitled)).poll(clock=SimClock(NOW))
        assert len(items) == 1

    def test_the_seen_set_is_bounded(self) -> None:
        """An unbounded set on a long-lived collector is a slow memory leak that
        only shows up after the process has been up for weeks."""
        c = RssCollector(1, url="x", fetcher=_feed(RSS), max_seen=1)
        c.poll(clock=SimClock(NOW))
        assert len(c._seen) == 1


class TestItemShape:
    def test_fields_are_populated_from_the_entry(self) -> None:
        items = nse_collector(fetcher=_feed(ATOM)).poll(clock=SimClock(NOW))
        item = items[0]
        assert item.source_id == 2
        assert item.headline == "Trading halt in XYZ"
        assert item.url == "https://example.invalid/circ/9"
        assert item.lang == "en"
        assert item.content_hash and len(item.content_hash) == 64
        assert item.t_ingest == NOW

    def test_the_content_hash_covers_the_body_not_just_the_headline(self) -> None:
        """Two circulars with the same title and different text are two events."""
        other = ATOM.replace("Halted pending clarification.", "Resumed at 14:00.")
        a = nse_collector(fetcher=_feed(ATOM)).poll(clock=SimClock(NOW))[0]
        b = nse_collector(fetcher=_feed(other)).poll(clock=SimClock(NOW))[0]
        assert a.headline == b.headline
        assert a.content_hash != b.content_hash
