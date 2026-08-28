"""M95 - the morning briefing, and the two words in its gate that do the work.

The gate is *20 consecutive automated briefings*. Counting records gives twenty
for a system that ran on twenty scattered days across three months, which
satisfies the sentence and answers a different question from the one being
asked.

The other half is that a forecast published after the open is not a forecast,
and a briefing nobody scored is a claim nobody tested.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from libs.ms_core.clock import FrozenClock
from libs.ms_models.gap import GapForecast
from libs.ms_models.quantile import QuantileForecast
from libs.ms_ops.briefing import (
    GATE_STREAK,
    Briefing,
    BriefingError,
    BriefingLog,
    CalendarItem,
    Delivery,
    KeyLevel,
    consecutive_streak,
    evaluate_gate,
    publish,
    score_against_open,
)

SESSION = date(2026, 8, 27)  # a Thursday
OPEN = datetime(2026, 8, 27, 3, 45, tzinfo=UTC)  # 09:15 IST
BEFORE = datetime(2026, 8, 27, 1, 0, tzinfo=UTC)  # 06:30 IST
AFTER = datetime(2026, 8, 27, 4, 30, tzinfo=UTC)


def weekdays_only(day: date) -> bool:
    return day.weekday() < 5


def forecast(
    *,
    median: float = 0.0032,
    half_width: float = 0.0075,
    session: date = SESSION,
    calibrated: bool = True,
) -> GapForecast:
    """A band centred on ``median``.

    Built symmetrically rather than by scaling the median: quantiles at
    ``median/2`` and ``median*1.5`` invert as soon as the median is negative,
    which is a fixture that only works for forecasts pointing up.
    """
    return GapForecast(
        session=session,
        forecast=QuantileForecast(
            quantiles={
                0.05: median - half_width,
                0.25: median - half_width / 3,
                0.5: median,
                0.75: median + half_width / 3,
                0.95: median + half_width,
            }
        ),
        window_hours=17.5,
        spans_weekend=False,
        drivers=["SPX +0.8%", "USDINR flat"],
        calibrated=calibrated,
    )


def make(**kw) -> Briefing:
    base = {
        "session": SESSION,
        "forecast": forecast(),
        "clock": FrozenClock(BEFORE),
        "session_open": OPEN,
    }
    return publish(**{**base, **kw})


# --------------------------------------------------------------------------- #
# consecutive, not twenty
# --------------------------------------------------------------------------- #
def test_the_streak_counts_consecutive_trading_days():
    """Ten straight weekdays is ten, and the weekend in the middle is not a
    break - not running on a closed day is correct behaviour."""
    days = [date(2026, 8, 17) + timedelta(days=i) for i in range(12)]
    trading = [d for d in days if weekdays_only(d)]

    assert consecutive_streak(trading, through=trading[-1], is_trading_day=weekdays_only) == len(
        trading
    )


def test_a_scattered_count_does_not_satisfy_the_streak():
    """The failure the gate is written against. Twenty briefings on twenty
    scattered mornings answers "did it run twenty times", not "does it run every
    morning unattended"."""
    scattered = [date(2026, 6, 1) + timedelta(days=3 * i) for i in range(20)]
    scattered = [d for d in scattered if weekdays_only(d)]

    assert len(scattered) >= 12
    streak = consecutive_streak(scattered, through=scattered[-1], is_trading_day=weekdays_only)
    assert streak < 3, f"a scattered run scored a streak of {streak}"


def test_the_streak_breaks_on_a_missed_trading_day():
    days = [
        d for d in (date(2026, 8, 3) + timedelta(days=i) for i in range(20)) if weekdays_only(d)
    ]
    missed = days[-4]
    kept = [d for d in days if d != missed]

    assert consecutive_streak(kept, through=days[-1], is_trading_day=weekdays_only) == 3


def test_a_holiday_does_not_break_the_streak():
    """Skipping a closed day is correct. A streak that reset every Diwali would
    never reach twenty, which is the failure people notice - unlike the one
    where a scattered count reaches it while proving nothing."""
    holiday = date(2026, 8, 19)
    closed = lambda d: weekdays_only(d) and d != holiday  # noqa: E731
    days = [d for d in (date(2026, 8, 10) + timedelta(days=i) for i in range(16)) if closed(d)]

    assert consecutive_streak(days, through=days[-1], is_trading_day=closed) == len(days)


def test_the_streak_is_zero_when_the_latest_session_is_missing():
    """A run that stopped yesterday is not a live streak."""
    days = [
        d for d in (date(2026, 8, 3) + timedelta(days=i) for i in range(15)) if weekdays_only(d)
    ]
    assert consecutive_streak(days[:-1], through=days[-1], is_trading_day=weekdays_only) == 0


def test_the_gate_needs_twenty_consecutive_not_twenty_published(tmp_path: Path):
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    scattered = [
        d for d in (date(2026, 5, 4) + timedelta(days=3 * i) for i in range(30)) if weekdays_only(d)
    ][:22]

    for d in scattered:
        b = publish(
            session=d,
            forecast=forecast(session=d),
            clock=FrozenClock(datetime(d.year, d.month, d.day, 1, 0, tzinfo=UTC)),
            session_open=datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC),
        )
        log.record(b)
        log.record_score(
            score_against_open(
                b,
                actual_gap=0.001,
                clock=FrozenClock(datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC)),
                session_open=datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC),
            )
        )

    gate = evaluate_gate(log, through=scattered[-1], is_trading_day=weekdays_only)
    assert gate.published >= GATE_STREAK
    assert not gate.passes
    assert "only" in gate.render() and "consecutive" in gate.render()


def test_the_gate_passes_on_a_real_run(tmp_path: Path):
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    days = [
        d for d in (date(2026, 6, 1) + timedelta(days=i) for i in range(40)) if weekdays_only(d)
    ][:GATE_STREAK]

    for d in days:
        b = publish(
            session=d,
            forecast=forecast(session=d),
            clock=FrozenClock(datetime(d.year, d.month, d.day, 1, 0, tzinfo=UTC)),
            session_open=datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC),
        )
        log.record(b)
        log.record_score(
            score_against_open(
                b,
                actual_gap=0.002,
                clock=FrozenClock(datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC)),
                session_open=datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC),
            )
        )

    gate = evaluate_gate(log, through=days[-1], is_trading_day=weekdays_only)
    assert gate.streak == GATE_STREAK
    assert gate.passes
    assert "PASSES" in gate.render()


def test_the_gate_fails_while_briefings_are_unscored(tmp_path: Path):
    """A forecast nobody checked is a claim nobody tested, and a run of those is
    the shape of a system that looks operational and is not."""
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    days = [
        d for d in (date(2026, 6, 1) + timedelta(days=i) for i in range(40)) if weekdays_only(d)
    ][:GATE_STREAK]

    for d in days:
        log.record(
            publish(
                session=d,
                forecast=forecast(session=d),
                clock=FrozenClock(datetime(d.year, d.month, d.day, 1, 0, tzinfo=UTC)),
                session_open=datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC),
            )
        )

    gate = evaluate_gate(log, through=days[-1], is_trading_day=weekdays_only)
    assert gate.streak == GATE_STREAK
    assert not gate.passes
    assert "never scored" in gate.render()


# --------------------------------------------------------------------------- #
# published before the open
# --------------------------------------------------------------------------- #
def test_a_briefing_cannot_be_published_after_the_open():
    """A gap forecast written after the open is not a forecast, and its score
    means nothing."""
    with pytest.raises(BriefingError, match="is not a forecast"):
        make(clock=FrozenClock(AFTER))


def test_a_briefing_at_the_exact_open_is_refused():
    with pytest.raises(BriefingError, match="is not a forecast"):
        make(clock=FrozenClock(OPEN))


def test_the_clock_is_injected_not_read_from_the_wall():
    """A replayed morning has to produce the briefing it would have produced,
    and one stamped with today's time while forecasting March is not a record of
    anything."""
    import inspect

    assert "clock" in inspect.signature(publish).parameters
    src = inspect.getsource(publish)
    assert "datetime.now" not in src
    assert "clock.now()" in src


def test_the_published_timestamp_comes_from_the_clock():
    assert make(clock=FrozenClock(BEFORE)).published_at == BEFORE


def test_a_forecast_for_another_session_is_refused():
    with pytest.raises(BriefingError, match="forecast is for"):
        make(forecast=forecast(session=date(2026, 8, 26)))


# --------------------------------------------------------------------------- #
# scored after it, as a separate record
# --------------------------------------------------------------------------- #
def test_a_briefing_cannot_be_scored_before_the_open():
    with pytest.raises(BriefingError, match="no actual open to score against"):
        score_against_open(make(), actual_gap=0.004, clock=FrozenClock(BEFORE), session_open=OPEN)


def test_the_score_is_a_separate_record_pointing_back_by_id(tmp_path: Path):
    """Filling the score into the briefing would make that file a place where a
    published forecast can be edited once the answer is known."""
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    b = make()
    log.record(b)
    log.record_score(
        score_against_open(b, actual_gap=0.004, clock=FrozenClock(AFTER), session_open=OPEN)
    )

    assert log.scores_path != log.path
    recorded = json.loads(log.path.read_text(encoding="utf-8").strip())
    assert "actual_gap" not in recorded

    scored = json.loads(log.scores_path.read_text(encoding="utf-8").strip())
    assert scored["briefing_id"] == b.briefing_id
    assert scored["actual_gap"] == pytest.approx(0.004)


def test_the_score_reports_coverage_and_direction():
    b = make()
    inside = score_against_open(b, actual_gap=0.004, clock=FrozenClock(AFTER), session_open=OPEN)
    outside = score_against_open(b, actual_gap=0.05, clock=FrozenClock(AFTER), session_open=OPEN)

    assert inside.covered and "covered" in inside.render()
    assert not outside.covered and "OUTSIDE" in outside.render()
    assert inside.direction_right


def test_a_flat_call_is_not_scored_as_a_wrong_direction():
    """Counting an honest abstention as a miss makes not calling a direction
    look worse than calling one badly."""
    flat = make(forecast=forecast(median=0.0001))
    s = score_against_open(flat, actual_gap=-0.01, clock=FrozenClock(AFTER), session_open=OPEN)

    assert not s.committed
    assert not s.direction_right
    assert "no direction called" in s.render()


def test_the_error_is_actual_minus_predicted():
    s = score_against_open(make(), actual_gap=0.005, clock=FrozenClock(AFTER), session_open=OPEN)
    assert s.error == pytest.approx(0.005 - 0.0032)


# --------------------------------------------------------------------------- #
# the log is append-only
# --------------------------------------------------------------------------- #
def test_a_second_briefing_for_one_morning_is_refused(tmp_path: Path):
    """Publishing a replacement would let a forecast be swapped after the fact."""
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    log.record(make())

    with pytest.raises(BriefingError, match="already recorded"):
        log.record(make(forecast=forecast(median=-0.02)))


def test_unscored_lists_what_was_never_checked(tmp_path: Path):
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    b = make()
    log.record(b)
    assert log.unscored() == [SESSION]

    log.record_score(
        score_against_open(b, actual_gap=0.004, clock=FrozenClock(AFTER), session_open=OPEN)
    )
    assert log.unscored() == []


def test_an_empty_log_reads_as_empty(tmp_path: Path):
    log = BriefingLog(path=tmp_path / "briefings.ndjson")
    assert log.sessions() == []
    assert log.unscored() == []


# --------------------------------------------------------------------------- #
# what the reader sees
# --------------------------------------------------------------------------- #
def test_an_uncalibrated_band_is_labelled_in_the_text():
    """A 90% band that is not a 90% band is the most misleading thing a briefing
    can carry - it is wrong in the direction that makes the reader more
    confident."""
    b = make(forecast=forecast(calibrated=False))
    text = b.render()

    assert "UNCALIBRATED" in text
    assert "more confident than it is" in text
    assert any("do not size against it" in n for n in b.notes)


def test_a_calibrated_band_carries_no_such_warning():
    assert "UNCALIBRATED" not in make().render()


def test_the_briefing_has_all_four_sections():
    b = make(
        overnight_events=["SPX closed +0.8%"],
        key_levels=[KeyLevel("prior close", 24000.0, "yesterday's settlement")],
        calendar=[CalendarItem("11:00 IST", "RBI policy", "high")],
    )
    text = b.render()
    for section in ("Gap forecast", "Overnight", "Key levels", "Today"):
        assert f"## {section}" in text


def test_empty_sections_say_so_rather_than_vanishing():
    """A section that disappears when empty makes a quiet morning look like a
    broken template."""
    text = make().render()
    assert "nothing recorded" in text
    assert "none set" in text
    assert "nothing scheduled" in text


def test_a_key_level_must_say_why_it_matters():
    """A level with no reason is a number on a chart, and the briefing's job is
    to be a thing a person can argue with."""
    import inspect

    params = inspect.signature(KeyLevel).parameters
    assert "why" in params
    assert params["why"].default is inspect.Parameter.empty


def test_the_briefing_id_is_stable_for_the_same_inputs():
    """So a re-run cannot quietly create a second record for one morning."""
    assert make().briefing_id == make().briefing_id
    assert make().briefing_id != make(forecast=forecast(median=-0.02)).briefing_id


def test_the_delivery_channel_is_recorded():
    assert make(delivery=Delivery.PUSH).delivery is Delivery.PUSH
    assert make().delivery is Delivery.MARKDOWN
