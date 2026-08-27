"""M02 verify - clock and session calendar."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import (
    UTC,
    CalendarNotLoadedError,
    FrozenClock,
    RealClock,
    SessionCalendar,
    SessionState,
    SimClock,
    check_drift,
)

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def cal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = cfgmod.load()
    base = SessionCalendar.from_config(cfg.sessions)
    # Give the calendar a known year so is_trading_day can answer at all.
    return SessionCalendar(
        tz=base.tz,
        preopen_start=base.preopen_start,
        preopen_end=base.preopen_end,
        continuous_start=base.continuous_start,
        continuous_end=base.continuous_end,
        expected_continuous_minutes=base.expected_continuous_minutes,
        holidays={2026: frozenset({date(2026, 8, 15)})},
    )


def ist(y, m, d, hh, mm, ss=0, us=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, us, tzinfo=IST).astimezone(UTC)


# --------------------------------------------------------------------- clocks
def test_real_clock_is_utc_aware():
    now = RealClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_sim_clock_cannot_go_backwards():
    c = SimClock(ist(2026, 8, 26, 10, 0))
    c.set(ist(2026, 8, 26, 10, 5))
    with pytest.raises(ValueError, match="backwards"):
        c.set(ist(2026, 8, 26, 9, 55))


def test_naive_datetime_rejected():
    with pytest.raises(ValueError, match="naive"):
        SimClock(datetime(2026, 8, 26, 10, 0))


def test_frozen_clock_may_rewind_for_tests():
    c = FrozenClock(ist(2026, 8, 26, 10, 0))
    c.set(ist(2026, 8, 26, 9, 0))
    assert c.now() == ist(2026, 8, 26, 9, 0)


# ---------------------------------------------------------------- boundaries
def test_candle_boundary_is_left_closed(cal):
    """A tick at exactly 09:20:00.000 belongs to the 09:20 candle, not 09:19."""
    exact = ist(2026, 8, 26, 9, 20, 0, 0)
    just_before = ist(2026, 8, 26, 9, 19, 59, 999_000)
    assert cal.candle_bucket(exact).astimezone(IST).minute == 20
    assert cal.candle_bucket(just_before).astimezone(IST).minute == 19


def test_close_instant_is_already_closed(cal):
    """15:30:00.000 is right-open: the session is over, not still running."""
    assert cal.state(ist(2026, 8, 26, 15, 29, 59)) is SessionState.OPEN
    assert cal.state(ist(2026, 8, 26, 15, 30, 0)) is SessionState.CLOSED


def test_preopen_is_distinct_from_open(cal):
    """Pre-open is a different price formation mechanism and must never be mixed in."""
    assert cal.state(ist(2026, 8, 26, 9, 5)) is SessionState.PREOPEN
    assert cal.state(ist(2026, 8, 26, 9, 15)) is SessionState.OPEN


def test_five_minute_bucket_anchors_to_open(cal):
    """5-minute buckets start at 09:15, not at :00 - otherwise the first bar is short."""
    b = cal.candle_bucket(ist(2026, 8, 26, 9, 22, 30), timeframe_minutes=5).astimezone(IST)
    assert (b.hour, b.minute) == (9, 20)


# ------------------------------------------------------------------ calendar
def test_refuses_to_guess_unknown_year(cal):
    """Assuming the market was open produces 375 phantom missing candles."""
    with pytest.raises(CalendarNotLoadedError, match="holiday list"):
        cal.is_trading_day(date(2031, 3, 4))


def test_weekend_and_holiday(cal):
    assert cal.is_trading_day(date(2026, 8, 26)) is True  # Wednesday
    assert cal.is_trading_day(date(2026, 8, 29)) is False  # Saturday
    assert cal.is_trading_day(date(2026, 8, 15)) is False  # loaded holiday


def test_minutes_since_open_and_to_close(cal):
    ts = ist(2026, 8, 26, 10, 15)
    assert cal.minutes_since_open(ts) == 60
    assert cal.minutes_to_close(ts) == 315


def test_expected_candle_count(cal):
    assert cal.expected_candle_count(1) == 375
    assert cal.expected_candle_count(5) == 75


def test_session_bounds_round_trip(cal):
    lo, hi = cal.session_bounds(date(2026, 8, 26))
    assert cal.state(lo) is SessionState.OPEN
    assert cal.state(hi) is SessionState.CLOSED
    assert (hi - lo).total_seconds() / 60 == 375


# --------------------------------------------------------------------- drift
def test_drift_within_and_beyond_limit():
    ref = datetime(2026, 8, 26, 4, 0, 0, tzinfo=UTC)
    ok = check_drift(ref, FrozenClock(ref + timedelta(milliseconds=20)), limit_ms=50)
    assert ok.ok and abs(ok.offset_ms - 20) < 1

    bad = check_drift(ref, FrozenClock(ref + timedelta(milliseconds=250)), limit_ms=50)
    assert not bad.ok


def test_no_wall_clock_calls_outside_clock_module():
    """Rule R6: library code must never read the wall clock directly."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    pattern = re.compile(r"datetime\.now\(|time\.time\(")
    for path in list((root / "libs").rglob("*.py")) + list((root / "services").rglob("*.py")):
        if path.name in {"clock.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{i}")
    assert (
        not offenders
    ), "wall-clock access outside clock.py - inject a Clock instead:\n  " + "\n  ".join(offenders)


# ------------------------------------------------- synthetic calendar guard
def test_config_calendar_never_assumes_trading_days(cfg):
    """The escape hatch must be unreachable from config.

    Interpreting real market data against a guessed calendar turns an exchange
    holiday into 375 phantom missing candles - indistinguishable from a feed
    outage, and a wasted day of debugging.
    """
    built = SessionCalendar.from_config(cfg.sessions)
    assert built.assume_weekdays_are_trading is False
    with pytest.raises(CalendarNotLoadedError):
        built.is_trading_day(date(2026, 8, 26))


def test_synthetic_variant_is_explicit_and_weekday_only(cfg):
    synth = SessionCalendar.from_config(cfg.sessions).synthetic_variant()
    assert synth.assume_weekdays_are_trading is True
    assert synth.is_trading_day(date(2026, 8, 26)) is True  # Wednesday
    assert synth.is_trading_day(date(2026, 8, 29)) is False  # Saturday
