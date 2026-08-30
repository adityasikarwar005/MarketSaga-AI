"""M21/M22 - RSS and Atom collectors, and the two that need no account.

RBI publishes its press releases and NSE its circulars as ordinary syndication
feeds. They are the two highest-credibility slots in `config/sources.yaml`
(1.00 each, both `may_trigger_directional: true`), and neither needs a key, a
signup or a paid tier. The expensive sources in that file - a wire service, a
social firehose - are the ones that may never open a position on their own.

Parsing somebody else's XML is the risk here
----------------------------------------------

A feed is remote, untrusted input parsed by a library. ``xml.etree`` does not
resolve external entities, but it is still vulnerable to entity expansion - the
"billion laughs" shape, where a few kilobytes of declarations expand to
gigabytes in memory. A collector that dies of that takes its own process with
it, and under :class:`CollectorRunner` that is one source going quiet, which is
the failure mode hardest to notice.

So two guards before the parser sees anything: a byte cap, and an outright
refusal of any document carrying a ``DOCTYPE``. A syndication feed has no
legitimate reason to declare one, so this costs nothing real and removes the
whole class.

The timestamp is the point of the exercise
--------------------------------------------

An RSS ``pubDate`` is RFC 822 and usually carries an offset; an Atom
``published`` is ISO 8601 and effectively always does. When one arrives without
an offset it is handed to :func:`resolve_publish_time` as a naive value together
with the source's own timezone - never silently read as UTC. RBI and NSE are
both IST publishers, so an unmarked value read as UTC would be stamped five and
a half hours early, which is lookahead that looks exactly like being fast.

Nothing here is enabled
-------------------------

Every source in `config/sources.yaml` carries `enabled: false` and a `tos_note`
reading TODO, and this module does not change that. Writing a collector is an
engineering task; deciding that a publisher's terms permit storing what it
returns is not, and a source switched on before somebody read them is a legal
problem wearing a feature's clothes. :class:`RssCollector` will happily poll
whatever URL it is handed - the registry is what decides whether it should.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

from libs.ms_core.clock import Clock
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import RawNewsItem
from libs.ms_news.collectors import Collector, CollectorError
from libs.ms_news.normalise import (
    PublishTimeMethod,
    TimeCandidate,
    detect_language,
    normalise_text,
    resolve_publish_time,
)

log = get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Both are IST publishers. Recorded here rather than inferred, because the
# consequence of getting it wrong is a five-and-a-half-hour lookahead that reads
# as latency rather than as an error.
RBI_PRESS_RELEASES = "https://website.rbi.org.in/en/web/rbi/press-releases/rss"
NSE_CIRCULARS = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"

# A syndication feed is kilobytes. Anything past this is not a feed we
# understand, and reading it into memory to find that out is the failure.
MAX_FEED_BYTES = 8 * 1024 * 1024

_DOCTYPE = re.compile(rb"<!DOCTYPE", re.IGNORECASE)

# Atom and RSS put the same ideas under different names and namespaces.
_ATOM = "{http://www.w3.org/2005/Atom}"


class FeedError(CollectorError):
    """The feed could not be read as a feed."""


def _http_get(url: str, *, timeout: float = 20.0) -> bytes:  # pragma: no cover - network
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; MarketSage research)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read(MAX_FEED_BYTES + 1)
        return data


def parse_feed_datetime(raw: str | None) -> datetime | None:
    """RFC 822 or ISO 8601, whichever this feed speaks. ``None`` if neither.

    Returned as-is, aware or naive. Deciding what a naive value means is
    :func:`resolve_publish_time`'s job and needs the source's timezone, which
    this function does not have and must not guess.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _first(entry: ElementTree.Element, *paths: str) -> ElementTree.Element | None:
    for path in paths:
        found = entry.find(path)
        if found is not None:
            return found
    return None


def parse_feed(raw: bytes) -> list[dict[str, str]]:
    """Feed bytes to a list of ``{title, link, summary, published, guid}``.

    Handles RSS 2.0 and Atom, because RBI and NSE do not agree on which to
    publish and neither guarantees to keep publishing the one it does now.
    """
    if len(raw) > MAX_FEED_BYTES:
        raise FeedError(
            f"feed is larger than {MAX_FEED_BYTES:,} bytes. A syndication feed is "
            "kilobytes; this is either not a feed or is an attempt to exhaust "
            "memory, and neither is worth parsing to find out which."
        )
    if _DOCTYPE.search(raw):
        raise FeedError(
            "feed declares a DOCTYPE. A syndication feed has no legitimate need "
            "for one, and entity expansion inside it is how a few kilobytes "
            "become gigabytes in memory. Refused without parsing."
        )
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise FeedError(f"feed is not well-formed XML: {exc}") from None

    entries: list[dict[str, str]] = []
    for item in root.iter():
        tag = item.tag
        if tag not in ("item", f"{_ATOM}entry"):
            continue
        link_node = _first(item, "link", f"{_ATOM}link")
        link = _text(link_node)
        if not link and link_node is not None:
            link = link_node.attrib.get("href", "")
        entries.append(
            {
                "title": _text(_first(item, "title", f"{_ATOM}title")),
                "link": link,
                "summary": _text(_first(item, "description", f"{_ATOM}summary", f"{_ATOM}content")),
                "published": _text(_first(item, "pubDate", f"{_ATOM}published", f"{_ATOM}updated")),
                "guid": _text(_first(item, "guid", f"{_ATOM}id")),
            }
        )
    return entries


class RssCollector(Collector):
    """One syndication feed, polled.

    ``fetcher`` is injected so the parsing, the timestamp cascade and the
    deduplication can all be exercised without a network - a collector testable
    only against a live publisher is a collector tested on the morning it breaks.

    ``seen`` holds the guids already emitted. Feeds repeat their whole window on
    every poll, so without it a five-minute cadence turns one press release into
    a hundred and eighty items, all with different ingest times and all looking
    like separate events to M23.
    """

    def __init__(
        self,
        source_id: int,
        *,
        url: str,
        name: str = "",
        source_tz: timezone = IST,
        fetcher: Callable[[str], bytes] | None = None,
        max_seen: int = 5_000,
    ) -> None:
        super().__init__(source_id, name=name)
        self.url = url
        self.source_tz = source_tz
        self._fetch = fetcher or _http_get
        self._seen: dict[str, None] = {}
        self._max_seen = max_seen

    def _remember(self, key: str) -> None:
        self._seen[key] = None
        while len(self._seen) > self._max_seen:
            self._seen.pop(next(iter(self._seen)))

    def poll(self, *, clock: Clock) -> Sequence[RawNewsItem]:
        raw = self._fetch(self.url)
        entries = parse_feed(raw)
        t_ingest = clock.now()

        out: list[RawNewsItem] = []
        for entry in entries:
            headline = normalise_text(entry["title"])
            if not headline:
                # A feed entry with no title is not an event we can classify or
                # deduplicate. Counted so a feed that starts emitting them is
                # visible, rather than dropped into silence.
                METRICS.inc("news_entry_dropped", source=str(self.source_id), reason="no_title")
                continue

            key = entry["guid"] or entry["link"] or headline
            if key in self._seen:
                continue
            self._remember(key)

            body = normalise_text(entry["summary"]) or None
            resolved = resolve_publish_time(
                [
                    TimeCandidate(
                        method=PublishTimeMethod.STRUCTURED,
                        value=parse_feed_datetime(entry["published"]),
                        raw=entry["published"] or None,
                    )
                ],
                t_ingest=t_ingest,
                source_tz=self.source_tz,
            )
            payload = headline if body is None else f"{headline}\n{body}"
            out.append(
                RawNewsItem(
                    source_id=self.source_id,
                    url=entry["link"] or None,
                    lang=detect_language(payload),
                    headline=headline,
                    body=body,
                    content_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    publish_time_method=resolved.method.value,
                    t_event=resolved.t_event,
                    t_ingest=t_ingest,
                    ts_uncertain=resolved.uncertain,
                )
            )

        METRICS.inc("news_items_collected", source=str(self.source_id))
        log.info(
            "feed polled",
            fields={
                "source": self.source_id,
                "entries": len(entries),
                "new": len(out),
                "url": self.url,
            },
        )
        return out


def rbi_collector(
    source_id: int = 1,
    *,
    url: str = RBI_PRESS_RELEASES,
    fetcher: Callable[[str], bytes] | None = None,
) -> RssCollector:
    """Central bank releases: credibility 1.00, and may open a position alone.

    Source id 1 in `config/sources.yaml`, which is still `enabled: false`.
    Constructing this does not switch it on.
    """
    return RssCollector(source_id, url=url, name="rbi", source_tz=IST, fetcher=fetcher)


def nse_collector(
    source_id: int = 2,
    *,
    url: str = NSE_CIRCULARS,
    fetcher: Callable[[str], bytes] | None = None,
) -> RssCollector:
    """Exchange notices: halts and bans are public, and are the fastest signal
    in the file that something has gone wrong with an instrument.

    Source id 2, also still disabled.
    """
    return RssCollector(source_id, url=url, name="nse", source_tz=IST, fetcher=fetcher)
