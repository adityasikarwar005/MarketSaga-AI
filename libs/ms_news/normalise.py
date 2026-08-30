"""M22 - normalisation and the publish-time resolver.

This module owns the three timestamps, and the plan says of it: *this module
determines whether your research is real*. That is not rhetoric. Every backtest
in this repository gates on ``Timed.known_at``, which reads ``t_available``, and
every claim about news predicting anything is a claim about ``t_event`` being
when the world learned something rather than when we did.

Where the timestamp comes from is part of the timestamp
--------------------------------------------------------

A publish time lifted from an RSS ``<pubDate>`` and one guessed from "Posted 2
hours ago" in the body are not the same measurement, and averaging research over
both without being able to separate them is how a study concludes that news
moves markets when what it has measured is its own scraping latency.

So the cascade records **which extractor won** in
:attr:`RawNewsItem.publish_time_method`, and the plan's parenthetical - *you
will need this later* - is the whole reason the field exists. G2 asks for 90% of
a sample within five seconds of truth; that question cannot even be asked of a
corpus that does not know where its own timestamps came from.

A naive timestamp is not a UTC timestamp
------------------------------------------

Indian sources publish ``+0530``; some publish nothing at all. Reading a naive
value as UTC is a 5.5-hour error that lands in the direction that matters -
it makes an item look **older** than it is, so a backtest treats it as knowable
before it was. This project has already been bitten once by exactly that shape,
in its own log timestamps, and there the error was merely cosmetic.

So a naive value is never assumed. Either the caller passes ``source_tz`` -
which the collector knows, because it knows whose server it is talking to - and
it is applied explicitly, or the result is marked uncertain and the fallback is
used. There is no third path and no default of UTC.

A publish time after the ingest time is a parse error, not a scoop
--------------------------------------------------------------------

If the resolved ``t_event`` is later than ``t_ingest``, we have a record
claiming we read something before it was written. The realistic causes are a
timezone applied backwards, a day/month swap, or a two-digit year. None of them
are recoverable by clamping: clamping produces a plausible timestamp and hides
the fault, and the item then sits in the corpus being subtly wrong forever.
:func:`resolve_publish_time` rejects it to the fallback and says so.

``t_available`` is stamped after the pipeline, not at ingest
-------------------------------------------------------------

Ingest is when the bytes arrived. Available is when this system could actually
have acted on the item - after normalisation, deduplication and classification.
On a slow classifier those differ by seconds, and seconds are the entire
horizon of an event-driven strategy. :func:`stamp_available` exists so the
difference is recorded rather than assumed to be zero.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from libs.ms_core.clock import UTC
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import RawNewsItem

log = get_logger(__name__)

# A resolved publish time this far in the future is a parse error. Not zero:
# clock skew between a publisher's server and ours is real and is measured in
# seconds, and rejecting on a one-second skew would push good items onto the
# fallback path for no gain.
FUTURE_TOLERANCE = timedelta(seconds=90)

# And one this far in the past, from a live feed, is almost always a two-digit
# year or a day/month swap rather than a genuine archive item. Collectors that
# legitimately backfill history pass their own bound.
DEFAULT_PAST_TOLERANCE = timedelta(days=30)


class PublishTimeMethod(StrEnum):
    """Which extractor produced ``t_event``. Ordered most to least trusted."""

    STRUCTURED = "structured"  # RSS pubDate, Atom published, JSON-LD datePublished
    HTML_META = "html_meta"  # <meta property="article:published_time"> and kin
    IN_TEXT = "in_text"  # a date parsed out of the body
    INGEST_FALLBACK = "ingest_fallback"  # nothing usable; t_event = t_ingest

    @property
    def trusted(self) -> bool:
        """Whether a timestamp from this method may be treated as measured.

        ``IN_TEXT`` is deliberately excluded. A date in running prose is a
        guess about which of several dates in the article is the publication
        one, and it is right often enough to be dangerous.
        """
        return self in (PublishTimeMethod.STRUCTURED, PublishTimeMethod.HTML_META)


class RejectReason(StrEnum):
    NAIVE_NO_SOURCE_TZ = "naive_no_source_tz"
    IN_THE_FUTURE = "in_the_future"
    IMPLAUSIBLY_OLD = "implausibly_old"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class ResolvedTime:
    """A publish time, how it was obtained, and whether to believe it."""

    t_event: datetime
    method: PublishTimeMethod
    uncertain: bool
    note: str

    @property
    def measured(self) -> bool:
        return self.method.trusted and not self.uncertain


@dataclass(frozen=True)
class TimeCandidate:
    """One extractor's answer, before it has been judged."""

    method: PublishTimeMethod
    value: datetime | None
    raw: str | None = None


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
# Space and tab only: NFKC runs first in normalise_text and has already
# folded NO-BREAK, FIGURE and NARROW NO-BREAK spaces into plain spaces.
_WS = re.compile(r"[ 	]+")
_BLANKS = re.compile(r"\n{3,}")
_TAGS = re.compile(r"<[^>]+>")

# Boilerplate that survives tag stripping and then poisons both the content hash
# and the SimHash in M23: two genuinely different stories that share a footer
# look more alike than they are.
_BOILERPLATE = (
    re.compile(r"(?i)\b(all rights reserved|copyright\s*(©|\(c\))?\s*\d{4})\b.*$"),
    re.compile(r"(?i)^\s*(share this|follow us on|read more:?|also read:?)\b.*$", re.MULTILINE),
    re.compile(r"(?i)\bsubscribe to our newsletter\b.*$"),
)


def strip_markup(raw: str) -> str:
    """Tags out, entities decoded, in that order.

    Decoding first would turn ``&lt;script&gt;`` into something the tag pattern
    then removes, silently deleting text the publisher meant literally.
    """
    return html.unescape(_TAGS.sub(" ", raw))


def normalise_text(raw: str) -> str:
    """Canonical form for hashing and reading.

    NFKC first: without it the same headline copied from two sites differs by a
    non-breaking space or a full-width character, hashes differently, and
    arrives in M23 as two events instead of one.
    """
    text = unicodedata.normalize("NFKC", strip_markup(raw))
    for pattern in _BOILERPLATE:
        text = pattern.sub("", text)
    text = _WS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    return _BLANKS.sub("\n\n", "\n".join(line.strip() for line in text.split("\n"))).strip()


def detect_language(text: str) -> str:
    """``en``, ``hi``, or ``und`` - a script test, not a language model.

    Deliberately crude and deliberately honest about it. Devanagari means Hindi
    here; anything else with Latin letters is called English; everything else is
    ``und`` rather than guessed. A wrong language tag routes an item to a
    classifier that was never trained on it, and ``und`` at least surfaces that.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "und"
    devanagari = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
    if devanagari / len(letters) > 0.2:
        return "hi"
    latin = sum(1 for c in letters if c.isascii())
    return "en" if latin / len(letters) > 0.5 else "und"


# --------------------------------------------------------------------------- #
# Publish-time resolution
# --------------------------------------------------------------------------- #
def apply_source_tz(value: datetime, source_tz: timezone | None) -> datetime | None:
    """Attach a timezone to a naive value, or refuse.

    Refusing is the point. See the module docstring: a naive value read as UTC
    is a 5.5-hour error for an Indian source, and it errs towards making an item
    look knowable earlier than it was.
    """
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    if source_tz is None:
        return None
    return value.replace(tzinfo=source_tz).astimezone(UTC)


def resolve_publish_time(
    candidates: list[TimeCandidate],
    *,
    t_ingest: datetime,
    source_tz: timezone | None = None,
    past_tolerance: timedelta = DEFAULT_PAST_TOLERANCE,
) -> ResolvedTime:
    """First candidate that survives the checks, in trust order.

    Candidates are tried most-trusted first and the first *plausible* one wins -
    not the first present one. A structured timestamp that fails the future
    check does not veto a good HTML meta tag underneath it; it just loses.
    """
    rejected: list[str] = []
    for candidate in sorted(candidates, key=lambda c: _ORDER[c.method]):
        if candidate.value is None:
            continue
        aware = apply_source_tz(candidate.value, source_tz)
        if aware is None:
            rejected.append(f"{candidate.method.value}:{RejectReason.NAIVE_NO_SOURCE_TZ.value}")
            METRICS.inc("news_publish_time_rejected", reason=RejectReason.NAIVE_NO_SOURCE_TZ.value)
            continue
        if aware > t_ingest + FUTURE_TOLERANCE:
            rejected.append(f"{candidate.method.value}:{RejectReason.IN_THE_FUTURE.value}")
            METRICS.inc("news_publish_time_rejected", reason=RejectReason.IN_THE_FUTURE.value)
            continue
        if aware < t_ingest - past_tolerance:
            rejected.append(f"{candidate.method.value}:{RejectReason.IMPLAUSIBLY_OLD.value}")
            METRICS.inc("news_publish_time_rejected", reason=RejectReason.IMPLAUSIBLY_OLD.value)
            continue
        return ResolvedTime(
            t_event=aware,
            method=candidate.method,
            uncertain=not candidate.method.trusted,
            note=(
                f"accepted {candidate.method.value}"
                + (f"; rejected {', '.join(rejected)}" if rejected else "")
            ),
        )

    METRICS.inc("news_publish_time_fallback")
    return ResolvedTime(
        t_event=t_ingest,
        method=PublishTimeMethod.INGEST_FALLBACK,
        uncertain=True,
        note=(
            "no usable publish time; t_event is the ingest time and is an upper "
            "bound on the truth, not a measurement"
            + (f"; rejected {', '.join(rejected)}" if rejected else "")
        ),
    )


_ORDER = {
    PublishTimeMethod.STRUCTURED: 0,
    PublishTimeMethod.HTML_META: 1,
    PublishTimeMethod.IN_TEXT: 2,
    PublishTimeMethod.INGEST_FALLBACK: 3,
}


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def stamp_available(item: RawNewsItem, *, now: datetime) -> RawNewsItem:
    """Set ``t_available`` to when the pipeline actually finished with the item.

    Called at the end of the pipeline, never at ingest. The gap between the two
    is this system's own latency, and a strategy whose horizon is seconds cannot
    afford to have it silently recorded as zero.

    Never earlier than ``t_ingest``: availability before arrival is not a
    possible state, and a clock that suggests it is a clock to distrust rather
    than a fact to store.
    """
    available = max(now, item.t_ingest)
    if available > item.t_ingest:
        METRICS.observe(
            "news_pipeline_latency_seconds", (available - item.t_ingest).total_seconds()
        )
    return item.model_copy(update={"t_available": available})
