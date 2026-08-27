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


@pytest.fixture(autouse=True)
def _the_real_experiment_log_is_off_limits():
    """No test may append to `research/experiments.jsonl`.

    That file is the hash-chained record of every genuine attempt at finding an
    edge, and its only purpose is an honest trial count - the deflated Sharpe
    and every multiple-comparison correction downstream read it. A test that
    writes there adds a fabricated trial on every suite run, forever, and
    nothing about the result looks wrong: the chain still verifies, the entries
    are well-formed, and the count is simply too high.

    A bare `ExperimentLog()` defaults to that path, which is correct for
    `jobs/` and a mistake in a test. It was a mistake in one test for several
    sessions, and it took a `git status` to notice - hence a guard rather than
    a convention.
    """
    from libs.ms_models.cv import EXPERIMENT_LOG

    before = EXPERIMENT_LOG.stat().st_mtime_ns if EXPERIMENT_LOG.exists() else None
    yield
    after = EXPERIMENT_LOG.stat().st_mtime_ns if EXPERIMENT_LOG.exists() else None
    if before != after:
        raise AssertionError(
            f"this test wrote to {EXPERIMENT_LOG}, the real experiment log. Pass an "
            "explicit path - ExperimentLog(tmp_path / 'experiments.jsonl') - because "
            "a fabricated trial inflates the count every deflated statistic reads."
        )
