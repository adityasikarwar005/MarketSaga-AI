"""M10 - Holiday discovery.

A hand-typed holiday list is wrong within a year and nobody notices until a
Tuesday produces zero candles and someone spends a day debugging the feed.

So the calendar is *discovered*, from three independent sources, and every date
carries provenance - where it came from, when it was fetched, and whether the
other sources agree:

``NseHolidayProvider``
    The exchange's own published holiday master. Forward-looking and, crucially,
    it carries a **reason** ("Republic Day"), which is what lets the preflight
    check say *tomorrow is a holiday because X* rather than just refusing to run.

``InferredHolidayProvider``
    Derived from our own recorded data: a weekday on which no instrument printed
    a single tick was a closed day. Retrospective only, and it cannot tell you
    why - but it is ground truth about what actually happened, which makes it
    the right thing to check the published list against.

``ManualHolidayProvider``
    ``config/sessions.yaml``. Highest precedence, because a human who has read
    an exchange circular outranks a scraper.

The reconciler merges them and, on disagreement, **flags rather than picks**.
Two sources disagreeing about whether the market was open is not something to
resolve silently.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from libs.ms_core.clock import UTC, Clock
from libs.ms_core.obs import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLIDAY_DIR = REPO_ROOT / "data" / "reference" / "holidays"

NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_HOME_URL = "https://www.nseindia.com"


class Source(StrEnum):
    NSE = "nse"
    INFERRED = "inferred"
    MANUAL = "manual"


class Confidence(StrEnum):
    CONFIRMED = "confirmed"  # two or more sources agree
    SINGLE = "single"  # only one source knows about it
    DISPUTED = "disputed"  # sources disagree - needs a human


@dataclass(frozen=True)
class Coverage:
    """Which days a source is able to say anything about.

    The distinction that makes DISPUTED meaningful: a provider that returned
    nothing for a date because it has no data for that period is *silent*, not
    *disagreeing*. Only a source that could have seen the day and did not flag it
    contradicts one that did.
    """

    days: frozenset[date] | None = None  # None means "the whole period in scope"

    def covers(self, day: date) -> bool:
        return True if self.days is None else day in self.days

    @classmethod
    def everything(cls) -> Coverage:
        return cls(None)

    @classmethod
    def nothing(cls) -> Coverage:
        return cls(frozenset())


@dataclass(frozen=True)
class Holiday:
    day: date
    reason: str
    source: Source
    segment: str = "CM"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d


@dataclass
class ReconciledHoliday:
    day: date
    reason: str
    sources: list[Source]
    confidence: Confidence
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "reason": self.reason,
            "sources": [s.value for s in self.sources],
            "confidence": self.confidence.value,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReconciledHoliday:
        return cls(
            day=date.fromisoformat(d["day"]),
            reason=d["reason"],
            sources=[Source(s) for s in d["sources"]],
            confidence=Confidence(d["confidence"]),
            note=d.get("note", ""),
        )


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class HolidayProvider(ABC):
    @property
    @abstractmethod
    def source(self) -> Source: ...

    @abstractmethod
    def fetch(self, year: int) -> list[Holiday]: ...

    def coverage(self, year: int) -> Coverage:
        """Days this source could have an opinion on. Override when partial."""
        return Coverage.everything()


_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


def parse_nse_date(raw: str) -> date | None:
    """NSE publishes dates as ``26-Jan-2026``. Tolerate whitespace and case."""
    m = re.match(r"\s*(\d{1,2})-([A-Za-z]{3})-(\d{4})\s*$", raw)
    if not m:
        return None
    day, mon, year = m.groups()
    month = _MONTHS.get(mon.title())
    if month is None:
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def parse_nse_payload(
    payload: dict[str, Any], *, segments: Iterable[str] = ("CM", "FO")
) -> list[Holiday]:
    """Turn the holiday-master JSON into typed holidays.

    The payload is keyed by market segment. We take the cash and F&O segments;
    they normally agree, and where they do not, the segment field records which
    market the date applies to.
    """
    out: list[Holiday] = []
    seen: set[tuple[date, str]] = set()
    for segment in segments:
        for row in payload.get(segment) or []:
            if not isinstance(row, dict):
                continue
            raw = row.get("tradingDate") or row.get("trading_date") or ""
            day = parse_nse_date(str(raw))
            if day is None:
                log.warning("unparseable NSE holiday date", fields={"raw": raw})
                continue
            key = (day, segment)
            if key in seen:
                continue
            seen.add(key)
            reason = str(row.get("description") or row.get("Description") or "").strip()
            out.append(
                Holiday(
                    day=day,
                    reason=reason or "(no description published)",
                    source=Source.NSE,
                    segment=segment,
                )
            )
    return sorted(out, key=lambda h: (h.day, h.segment))


def _default_nse_fetcher() -> dict[str, Any]:
    """Fetch the holiday master over HTTP, stdlib only.

    The endpoint expects a browser-like session: a cookie from the home page and
    a normal user agent. This is a public page, not an authenticated API, but it
    is also not a documented contract - it can change or start refusing us at any
    time, which is exactly why the store keeps provenance and the inference
    provider exists to cross-check.
    """
    import http.cookiejar
    import urllib.request

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
        ("Accept", "application/json, text/plain, */*"),
        ("Accept-Language", "en-US,en;q=0.9"),
    ]
    opener.open(NSE_HOME_URL, timeout=15).read()  # establishes the cookie
    with opener.open(NSE_HOLIDAY_URL, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


class NseHolidayProvider(HolidayProvider):
    """The exchange's published list. The only source that knows *why*."""

    def __init__(self, fetcher: Callable[[], dict[str, Any]] | None = None) -> None:
        self._fetch = fetcher or _default_nse_fetcher

    @property
    def source(self) -> Source:
        return Source.NSE

    def fetch(self, year: int) -> list[Holiday]:
        payload = self._fetch()
        return [h for h in parse_nse_payload(payload) if h.day.year == year]


class InferredHolidayProvider(HolidayProvider):
    """Derived from recorded data: a weekday with no ticks at all was closed.

    Retrospective and reason-free, but it is a statement about what actually
    happened rather than what was announced. Its job is to catch the case where
    the published list is stale, or where an unscheduled closure occurred.
    """

    def __init__(self, observed_trading_days: Iterable[date]) -> None:
        self._observed = set(observed_trading_days)

    @property
    def source(self) -> Source:
        return Source.INFERRED

    def coverage(self, year: int) -> Coverage:
        """Weekdays inside the recorded span. Outside it, and on weekends, nothing.

        The weekday restriction is not a detail. ``fetch`` only ever infers
        holidays on weekdays, so claiming to cover a Saturday means this source
        counts as *silent* about every holiday that falls on one - and
        ``reconcile`` reads silence from a covering source as disagreement.

        The effect, before this was fixed: Independence Day 2026 falls on a
        Saturday, and it was reported DISPUTED with "verify against the exchange
        circular". Every weekend holiday would be, every year, forever. A
        warning that is always wrong is worse than no warning, because it
        teaches people to skip the ones that are right.
        """
        in_year = {d for d in self._observed if d.year == year}
        if not in_year:
            return Coverage.nothing()
        start, end = min(in_year), max(in_year)
        days, cursor = set(), start
        while cursor <= end:
            if cursor.weekday() < 5:
                days.add(cursor)
            cursor += timedelta(days=1)
        return Coverage(frozenset(days))

    def fetch(self, year: int) -> list[Holiday]:
        if not self._observed:
            return []
        # Only infer inside the span we actually recorded. Claiming a holiday for
        # a day we simply were not running is exactly the kind of confident
        # wrongness this module exists to prevent.
        in_year = {d for d in self._observed if d.year == year}
        if not in_year:
            return []
        start, end = min(in_year), max(in_year)
        out: list[Holiday] = []
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5 and cursor not in self._observed:
                out.append(
                    Holiday(
                        day=cursor,
                        reason="no market data recorded on a weekday",
                        source=Source.INFERRED,
                    )
                )
            cursor += timedelta(days=1)
        return out


class ManualHolidayProvider(HolidayProvider):
    """Whatever a human put in config. Outranks everything else."""

    def __init__(self, holidays_by_year: dict[int, Iterable[date]]) -> None:
        self._by_year = {y: list(days) for y, days in holidays_by_year.items()}

    @property
    def source(self) -> Source:
        return Source.MANUAL

    def coverage(self, year: int) -> Coverage:
        """An empty config is silence, not a claim that nothing is a holiday."""
        return Coverage.everything() if self._by_year.get(year) else Coverage.nothing()

    def fetch(self, year: int) -> list[Holiday]:
        return [
            Holiday(day=d, reason="pinned in config/sessions.yaml", source=Source.MANUAL)
            for d in self._by_year.get(year, [])
        ]


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
# Manual beats NSE beats inference, for the reason text. Precedence affects the
# label only - a disagreement about whether the day was open is always flagged.
_PRECEDENCE = {Source.MANUAL: 0, Source.NSE: 1, Source.INFERRED: 2}


def reconcile(
    per_source: dict[Source, list[Holiday]],
    *,
    coverage: dict[Source, Coverage] | None = None,
) -> list[ReconciledHoliday]:
    """Merge sources, keeping the best reason and recording who agreed.

    ``coverage`` says which days each source could speak about. A date claimed by
    the exchange but absent from inference over a span we fully recorded is
    DISPUTED - the market may have been open after all. A date absent from a
    source that has no data for that period is not a disagreement.
    """
    cov = coverage or {}
    by_day: dict[date, list[Holiday]] = {}
    for holidays in per_source.values():
        for h in holidays:
            by_day.setdefault(h.day, []).append(h)

    out: list[ReconciledHoliday] = []
    for day, entries in sorted(by_day.items()):
        entries.sort(key=lambda h: _PRECEDENCE[h.source])
        sources = sorted({h.source for h in entries}, key=lambda s: _PRECEDENCE[s])
        best = entries[0]

        silent = {s for s, c in cov.items() if s not in sources and c.covers(day)}
        if len(sources) >= 2:
            confidence, note = Confidence.CONFIRMED, ""
        elif silent:
            confidence = Confidence.DISPUTED
            note = (
                f"claimed by {sources[0].value} but not by "
                + ", ".join(sorted(s.value for s in silent))
                + " - verify against the exchange circular"
            )
        else:
            confidence, note = Confidence.SINGLE, ""

        out.append(
            ReconciledHoliday(
                day=day,
                reason=best.reason,
                sources=sources,
                confidence=confidence,
                note=note,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
@dataclass
class HolidayYear:
    year: int
    fetched_at: str
    holidays: list[ReconciledHoliday] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "fetched_at": self.fetched_at,
            "holidays": [h.to_dict() for h in self.holidays],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HolidayYear:
        return cls(
            year=int(d["year"]),
            fetched_at=str(d["fetched_at"]),
            holidays=[ReconciledHoliday.from_dict(x) for x in d.get("holidays", [])],
        )

    @property
    def disputed(self) -> list[ReconciledHoliday]:
        return [h for h in self.holidays if h.confidence is Confidence.DISPUTED]

    def dates(self) -> set[date]:
        return {h.day for h in self.holidays}


class HolidayStore:
    """Cached, provenance-carrying holiday calendar on disk.

    One JSON per year in ``data/reference/holidays/``. Cached rather than fetched
    at start-up on purpose: the service must start at 08:45 even when the
    exchange website is unreachable, and a stale-but-known calendar with a
    visible fetch date beats a network call in the start-up path.
    """

    def __init__(self, root: Path = HOLIDAY_DIR) -> None:
        self.root = root

    def path(self, year: int) -> Path:
        return self.root / f"{year}.json"

    def load(self, year: int) -> HolidayYear | None:
        p = self.path(year)
        if not p.exists():
            return None
        return HolidayYear.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, entry: HolidayYear) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(entry.year)
        p.write_text(json.dumps(entry.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    def years(self) -> list[int]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.glob("*.json"):
            try:
                out.append(int(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def as_calendar_dict(self) -> dict[int, frozenset[date]]:
        """The shape :class:`SessionCalendar` wants - minus the empty years.

        **A year with zero holidays is not a loaded year.** No exchange closes
        for nothing, so an empty list means the providers had nothing to say
        about that year, not that the market never shut.

        This distinction is load-bearing and was found the hard way. NSE's
        holiday endpoint serves only the current year, so syncing 2023-2025
        writes three files with zero holidays each. Handing those to the
        calendar replaces `CalendarNotLoadedError` - a refusal to guess, which
        is the entire design of `SessionCalendar` - with a confident
        `is_trading_day(2023-01-26) == True`. Republic Day, a Thursday, silently
        becomes a session. Running the sync would then leave the system *less*
        correct than never having run it, which is the worst possible shape for
        a maintenance command to have.

        Populate a past year by backfilling bhavcopy across it: the exchange
        publishes one for every session and none for a holiday, so
        `InferredHolidayProvider` can reconstruct a year the endpoint no longer
        serves. Reason-free, but right.
        """
        out: dict[int, frozenset[date]] = {}
        for y in self.years():
            entry = self.load(y)
            if entry is None:
                continue
            days = frozenset(entry.dates())
            if not days:
                log.warning(
                    "holiday year has no holidays, treating it as not loaded",
                    fields={"year": y, "path": str(self.path(y))},
                )
                continue
            out[y] = days
        return out

    # -- sync ---------------------------------------------------------------
    def sync(
        self,
        year: int,
        clock: Clock,
        *,
        providers: list[HolidayProvider],
        strict: bool = False,
    ) -> HolidayYear:
        """Fetch from every provider, reconcile, persist.

        A provider that fails is logged and skipped rather than aborting the
        sync - one unreachable website should not deprive the system of a
        calendar it could otherwise assemble. With ``strict`` the failure is
        raised instead, which is what CI wants.
        """
        per_source: dict[Source, list[Holiday]] = {}
        coverage: dict[Source, Coverage] = {}
        for provider in providers:
            try:
                found = provider.fetch(year)
            except Exception as exc:
                log.error(
                    "holiday provider failed",
                    fields={"source": provider.source.value, "year": year, "error": str(exc)},
                )
                if strict:
                    raise
                continue
            coverage[provider.source] = provider.coverage(year)
            per_source[provider.source] = found
            log.info(
                "holiday provider returned",
                fields={"source": provider.source.value, "year": year, "count": len(found)},
            )

        if not per_source:
            raise RuntimeError(
                f"no holiday source could be reached for {year}. "
                "The calendar will keep refusing to guess - that is deliberate. "
                "Either restore connectivity or pin the dates in config/sessions.yaml."
            )

        entry = HolidayYear(
            year=year,
            fetched_at=clock.now().astimezone(UTC).isoformat(),
            holidays=reconcile(per_source, coverage=coverage),
        )
        self.save(entry)
        if entry.disputed:
            log.warning(
                "disputed holidays need a human",
                fields={"year": year, "days": [h.day.isoformat() for h in entry.disputed]},
            )
        return entry


# --------------------------------------------------------------------------- #
# Queries the rest of the system asks
# --------------------------------------------------------------------------- #
def describe_day(store: HolidayStore, day: date) -> str:
    """Human-readable answer to 'is the market open, and if not, why?'."""
    if day.weekday() >= 5:
        return f"{day.isoformat()} is a {day.strftime('%A')} - weekend"
    entry = store.load(day.year)
    if entry is None:
        return f"{day.isoformat()}: no holiday calendar loaded for {day.year} - run: python tasks.py holidays --sync"
    for h in entry.holidays:
        if h.day == day:
            flag = " [DISPUTED]" if h.confidence is Confidence.DISPUTED else ""
            return f"{day.isoformat()} is a holiday: {h.reason}{flag}"
    return f"{day.isoformat()} is a trading day"


def next_holiday(
    store: HolidayStore, after: date, *, horizon_days: int = 120
) -> ReconciledHoliday | None:
    """The next closure, so the preflight check can announce it a day early."""
    for offset in range(1, horizon_days + 1):
        day = after + timedelta(days=offset)
        entry = store.load(day.year)
        if entry is None:
            continue
        for h in entry.holidays:
            if h.day == day:
                return h
    return None


def observed_trading_days_from_recordings(root: Path) -> set[date]:
    """Which days we actually captured data on, read from the recorder's layout.

    Feeds :class:`InferredHolidayProvider`. Reads directory names only - no data
    is parsed, so this stays fast as the archive grows.
    """
    out: set[date] = set()
    ticks_dir = root / "ticks"
    if not ticks_dir.exists():
        return out
    for part in ticks_dir.glob("date=*"):
        if not part.is_dir():
            continue
        try:
            day = date.fromisoformat(part.name.removeprefix("date="))
        except ValueError:
            continue
        if any(f.stat().st_size > 0 for f in part.glob("*.ndjson")):
            out.add(day)
    return out


def observed_trading_days_from_bhavcopy(root: Path | None = None) -> set[date]:
    """Which days the exchange itself published a bhavcopy for.

    A far stronger cross-check than the recorder's own layout, and available
    without having recorded anything: NSE publishes an F&O bhavcopy for every
    session and for no other day, going back years. So a cached bhavcopy is a
    statement by the exchange that it traded that day - which is exactly what
    :class:`InferredHolidayProvider` needs to turn a published holiday list from
    "announced" into "announced and independently corroborated".

    Directory listing only; no file is opened, so this stays fast as the cache
    fills. The span matters as much as the days: the provider infers holidays
    only *between* the earliest and latest day it can see, because a day nobody
    downloaded is not a day the market was shut.
    """
    from libs.ms_data.bhavcopy import BHAVCOPY_CACHE

    base = root if root is not None else BHAVCOPY_CACHE
    out: set[date] = set()
    # glob on a missing directory already yields nothing, so this is a statement
    # of intent rather than protection - kept to match the sibling function
    # above. The behaviour it describes is pinned by a test either way.
    if not base.exists():
        return out
    for f in base.glob("*.json"):
        try:
            out.add(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return out


def staleness_days(entry: HolidayYear, now: datetime) -> int:
    """How long since this year's calendar was refreshed."""
    fetched = datetime.fromisoformat(entry.fetched_at)
    return (now.astimezone(UTC) - fetched.astimezone(UTC)).days
