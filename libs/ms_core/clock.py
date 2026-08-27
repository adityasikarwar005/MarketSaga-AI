"""M02 - Clock and session calendar.

Two rules make everything downstream testable:

1. **The clock is injected.** Library code never calls ``datetime.now()``. It
   receives a :class:`Clock`. In production that is :class:`RealClock`; in the
   simulator it is :class:`SimClock`. Because the engine cannot tell the
   difference, a backtest exercises the same code path as live trading - which
   is the only way a backtest result means anything.

2. **Time is stored in UTC and reasoned about in exchange time.** Every stored
   timestamp is timezone-aware UTC. Session logic converts to exchange time at
   the boundary. Naive datetimes are rejected, loudly.

The calendar deliberately refuses to guess. If the holiday list for a year has
not been loaded, :meth:`SessionCalendar.is_trading_day` raises rather than
assuming the market was open - a wrong assumption there quietly corrupts every
statistic computed over that period.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from enum import StrEnum
from zoneinfo import ZoneInfo

from libs.ms_core.config import SessionsConfig

__all__ = [
    "UTC",
    "CalendarNotLoadedError",
    "Clock",
    "ClockDriftError",
    "DriftReading",
    "FrozenClock",
    "RealClock",
    "SessionCalendar",
    "SessionState",
    "SimClock",
    "check_drift",
]


class SessionState(StrEnum):
    CLOSED = "CLOSED"
    HOLIDAY = "HOLIDAY"
    PREOPEN = "PREOPEN"
    OPEN = "OPEN"
    SPECIAL = "SPECIAL"


class ClockDriftError(RuntimeError):
    """Raised when local time has drifted beyond the configured tolerance."""


class CalendarNotLoadedError(LookupError):
    """Raised when asked about a year whose holiday list has not been provided."""


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #
class Clock(ABC):
    """Source of 'what time is it'. Inject this; never import datetime.now()."""

    @abstractmethod
    def now(self) -> datetime:
        """Current instant, timezone-aware UTC."""

    def monotonic(self) -> float:
        """Monotonic seconds, for measuring durations rather than instants."""
        return time.monotonic()


class RealClock(Clock):
    """Wall-clock time. The only place ``datetime.now`` is allowed to appear."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock(Clock):
    """Replay clock. Advanced explicitly by the simulator, never on its own."""

    def __init__(self, start: datetime) -> None:
        self._now = _require_aware(start)

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        ts = _require_aware(ts)
        if ts < self._now:
            raise ValueError(
                f"SimClock cannot go backwards: {self._now.isoformat()} -> {ts.isoformat()}. "
                "The merged replay stream must be sorted by timestamp."
            )
        self._now = ts

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class FrozenClock(SimClock):
    """A SimClock that permits moving backwards. Tests only - never in a service."""

    def set(self, ts: datetime) -> None:
        self._now = _require_aware(ts)


def _require_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError(
            f"naive datetime {ts!r} rejected - every timestamp in MarketSage carries a timezone"
        )
    return ts.astimezone(UTC)


# --------------------------------------------------------------------------- #
# Drift monitor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriftReading:
    offset_ms: float
    ok: bool
    limit_ms: float


def check_drift(reference_utc: datetime, clock: Clock, limit_ms: float) -> DriftReading:
    """Compare our clock against a trusted reference (NTP or an exchange timestamp).

    Point-in-time correctness rests entirely on timestamps being right. A drifting
    clock does not produce an error, it produces plausible-looking data that is
    subtly wrong, which is far worse. Above the limit the system must refuse to
    emit signals rather than emit misdated ones.
    """
    offset_ms = (clock.now() - _require_aware(reference_utc)).total_seconds() * 1000.0
    return DriftReading(offset_ms=offset_ms, ok=abs(offset_ms) <= limit_ms, limit_ms=limit_ms)


# --------------------------------------------------------------------------- #
# Session calendar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionCalendar:
    tz: ZoneInfo
    preopen_start: dtime
    preopen_end: dtime
    continuous_start: dtime
    continuous_end: dtime
    expected_continuous_minutes: int
    holidays: dict[int, frozenset[date]]
    special_dates: frozenset[date] = frozenset()

    # Escape hatch for the synthetic feed ONLY. When true, every weekday counts
    # as a trading day. It cannot be set from config (see `from_config`), so it
    # can never leak into a path that touches real market data - which is the
    # whole point: real data must never be interpreted against a guessed calendar.
    assume_weekdays_are_trading: bool = False

    # ---- construction ----
    @classmethod
    def from_config(cls, sessions_cfg: SessionsConfig) -> SessionCalendar:
        """Build from the ``sessions`` section of the loaded config (M01)."""
        c = sessions_cfg
        return cls(
            tz=ZoneInfo(c.timezone),
            preopen_start=_parse_hhmm(c.preopen.start),
            preopen_end=_parse_hhmm(c.preopen.end),
            continuous_start=_parse_hhmm(c.continuous.start),
            continuous_end=_parse_hhmm(c.continuous.end),
            expected_continuous_minutes=c.expected_continuous_minutes,
            holidays={year: frozenset(days) for year, days in c.holidays.items()},
            special_dates=frozenset(
                s["date"] for s in c.special_sessions if isinstance(s, dict) and "date" in s
            ),
            # Never settable from config. Real data always needs a real calendar.
            assume_weekdays_are_trading=False,
        )

    def with_holidays(self, holidays: dict[int, frozenset[date]]) -> SessionCalendar:
        """Merge in a discovered calendar (from the holiday store, M10).

        Config-pinned dates are kept: a human who has read an exchange circular
        outranks a scraper. Discovered dates fill in the years config says
        nothing about.
        """
        merged = dict(self.holidays)
        for year, days in holidays.items():
            merged[year] = frozenset(merged.get(year, frozenset())) | frozenset(days)
        return replace(self, holidays=merged)

    def synthetic_variant(self) -> SessionCalendar:
        """A copy that treats every weekday as a trading day.

        For the synthetic feed and for demos only. Using it on real market data
        would turn an exchange holiday into 375 phantom missing candles, which
        looks exactly like a feed outage and wastes a day of debugging.
        """
        return replace(self, assume_weekdays_are_trading=True)

    # ---- queries ----
    def local(self, ts: datetime) -> datetime:
        """Convert a UTC instant to exchange local time."""
        return _require_aware(ts).astimezone(self.tz)

    def has_calendar_for(self, year: int) -> bool:
        return year in self.holidays

    def is_trading_day(self, day: date) -> bool:
        """True if the exchange was open on ``day``.

        Raises rather than guessing when the year's holiday list is missing.
        Silently treating an unknown day as a trading day produces 375 missing
        candles that look exactly like a feed outage.
        """
        if self.assume_weekdays_are_trading:
            return day.weekday() < 5
        if day.year not in self.holidays:
            raise CalendarNotLoadedError(
                f"no holiday list loaded for {day.year}. Populate config/sessions.yaml "
                f"from the exchange holiday circular before asking about {day}."
            )
        if day.weekday() >= 5:  # Sat/Sun
            return False
        return day not in self.holidays[day.year]

    def state(self, ts: datetime) -> SessionState:
        loc = self.local(ts)
        day = loc.date()
        if day in self.special_dates:
            return SessionState.SPECIAL
        try:
            trading = self.is_trading_day(day)
        except CalendarNotLoadedError:
            raise
        if not trading:
            return SessionState.HOLIDAY if day.weekday() < 5 else SessionState.CLOSED

        t = loc.time()
        if self.preopen_start <= t < self.preopen_end:
            return SessionState.PREOPEN
        # Left-closed, right-open: 15:30:00.000 is already CLOSED.
        if self.continuous_start <= t < self.continuous_end:
            return SessionState.OPEN
        return SessionState.CLOSED

    def is_open(self, ts: datetime) -> bool:
        return self.state(ts) is SessionState.OPEN

    def minutes_since_open(self, ts: datetime) -> int:
        """Whole minutes elapsed since the continuous session opened.

        This is the backbone of every time-of-day normalisation in the feature
        set. Intraday volatility is strongly U-shaped, so a raw return means
        nothing until it is expressed relative to its minute-of-day baseline.
        """
        loc = self.local(ts)
        open_at = loc.replace(
            hour=self.continuous_start.hour,
            minute=self.continuous_start.minute,
            second=0,
            microsecond=0,
        )
        return int((loc - open_at).total_seconds() // 60)

    def minutes_to_close(self, ts: datetime) -> int:
        loc = self.local(ts)
        close_at = loc.replace(
            hour=self.continuous_end.hour,
            minute=self.continuous_end.minute,
            second=0,
            microsecond=0,
        )
        return int((close_at - loc).total_seconds() // 60)

    def candle_bucket(self, ts: datetime, timeframe_minutes: int = 1) -> datetime:
        """The left edge of the bucket this instant belongs to, in UTC.

        Left-closed / right-open. A tick at exactly 09:20:00.000 belongs to the
        09:20 candle, not 09:19 - the single most common off-by-one in candle
        construction, and one that quietly shifts every sequence model's input.
        """
        loc = self.local(ts).replace(second=0, microsecond=0)
        if timeframe_minutes > 1:
            anchor = loc.replace(
                hour=self.continuous_start.hour, minute=self.continuous_start.minute
            )
            offset = int((loc - anchor).total_seconds() // 60)
            loc = anchor + timedelta(minutes=(offset // timeframe_minutes) * timeframe_minutes)
        return loc.astimezone(UTC)

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        """(open, close) of the continuous session on ``day``, in UTC."""
        start = datetime.combine(day, self.continuous_start, tzinfo=self.tz)
        end = datetime.combine(day, self.continuous_end, tzinfo=self.tz)
        return start.astimezone(UTC), end.astimezone(UTC)

    def expected_candle_count(self, timeframe_minutes: int = 1) -> int:
        return self.expected_continuous_minutes // timeframe_minutes


def _parse_hhmm(value: str) -> dtime:
    hh, _, mm = value.partition(":")
    return dtime(hour=int(hh), minute=int(mm))
