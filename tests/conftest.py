"""Shared fixtures."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import UTC, SessionCalendar, SimClock

IST = ZoneInfo("Asia/Kolkata")
TEST_DAY = date(2026, 8, 26)  # a Wednesday


def ist(hh: int, mm: int, ss: int = 0, us: int = 0, day: date = TEST_DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, ss, us, tzinfo=IST).astimezone(UTC)


@pytest.fixture
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cfgmod.load()


@pytest.fixture
def calendar(cfg):
    """Calendar with a known holiday year, so is_trading_day can answer."""
    base = SessionCalendar.from_config(cfg.sessions)
    return SessionCalendar(
        tz=base.tz,
        preopen_start=base.preopen_start,
        preopen_end=base.preopen_end,
        continuous_start=base.continuous_start,
        continuous_end=base.continuous_end,
        expected_continuous_minutes=base.expected_continuous_minutes,
        holidays={2026: frozenset({date(2026, 8, 15)})},
    )


@pytest.fixture
def sim_clock():
    return SimClock(ist(9, 15))


@pytest.fixture(autouse=True)
def _reset_metrics():
    from libs.ms_core.obs import METRICS

    METRICS.reset()
    yield
    METRICS.reset()
