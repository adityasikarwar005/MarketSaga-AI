"""M92 - alerting, where two rules pull in opposite directions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_risk.alerts import (
    RULES,
    Alerter,
    Channel,
    Severity,
    build,
    critical_kinds,
)

T0 = datetime(2026, 3, 4, 6, 30, tzinfo=UTC)


def alerter(**over) -> tuple[Alerter, list[tuple[str, Channel]]]:
    sent: list[tuple[str, Channel]] = []
    a = Alerter(send=lambda al, ch: sent.append((al.kind, ch)), **over)
    return a, sent


# ------------------------------------------------------------- the rate limit
def test_rate_limited():
    """An operator who learns that most notifications need no action stops
    reading them, and the one that did need action arrives into that habit."""
    a, sent = alerter(max_per_window=3, window=timedelta(minutes=10))
    for i in range(6):
        a.emit(build("position_exit", f"exit {i}", ts=T0 + timedelta(seconds=i)))

    assert len(sent) == 3
    assert len(a.suppressed) == 3
    assert all("rate limit" in s.reason for s in a.suppressed)


def test_the_limit_is_per_kind_not_global():
    """A global budget means a burst of one kind exhausts the allowance for
    every other, which is backwards - the burst is what should be throttled."""
    a, sent = alerter(max_per_window=2)
    for i in range(5):
        a.emit(build("position_exit", "noisy", ts=T0 + timedelta(seconds=i)))
    a.emit(build("tradeable_signal", "a real signal", ts=T0 + timedelta(seconds=6)))

    assert ("tradeable_signal", Channel.PUSH) in sent


def test_the_window_slides():
    a, sent = alerter(max_per_window=2, window=timedelta(minutes=10))
    for i in range(2):
        a.emit(build("position_exit", "x", ts=T0 + timedelta(seconds=i)))
    assert len(sent) == 2

    a.emit(build("position_exit", "blocked", ts=T0 + timedelta(minutes=1)))
    assert len(sent) == 2

    a.emit(build("position_exit", "allowed again", ts=T0 + timedelta(minutes=11)))
    assert len(sent) == 3


def test_a_dedupe_key_groups_a_burst():
    a, sent = alerter(max_per_window=1)
    for i in range(4):
        a.emit(
            build("position_exit", "same trade", ts=T0 + timedelta(seconds=i), dedupe_key="trade-1")
        )
    assert len(sent) == 1


# ----------------------------------------------------------- the exception
def test_risk_breach_not_mutable():
    """A risk breach is not a notification, it is an instruction. A rate limiter
    that can swallow it has converted a safety control into a convenience."""
    a, sent = alerter(max_per_window=1)
    a.mute(Channel.PUSH, until=T0 + timedelta(hours=8), reason="overnight")
    a.mute(Channel.SMS, until=T0 + timedelta(hours=8), reason="overnight")

    for i in range(5):
        a.emit(build("risk_breach", f"L3 flatten {i}", ts=T0 + timedelta(seconds=i)))

    assert len(sent) == 10, "a muted, rate-limited critical alert was suppressed"
    assert a.suppressed == []


def test_critical_severity_bypasses_by_severity_not_by_kind():
    """So a new critical kind is protected the moment it is added to the table,
    without anyone remembering to update a special case."""
    assert Severity.CRITICAL.bypasses_everything
    assert not Severity.ACTION.bypasses_everything
    assert set(critical_kinds()) == {"risk_breach", "system_health_l3"}


def test_there_is_no_flag_that_disables_the_bypass():
    """A configurable override would be configured on the day it mattered, by
    someone trying to get some sleep."""
    import inspect

    source = inspect.getsource(Alerter.emit)
    assert "bypasses_everything" in source
    params = inspect.signature(Alerter.emit).parameters
    assert set(params) == {"self", "alert"}, f"emit grew an override parameter: {set(params)}"


# ------------------------------------------------------------------- mutes
def test_a_mute_silences_one_channel_not_all():
    a, sent = alerter()
    a.mute(Channel.PUSH, until=T0 + timedelta(hours=1), reason="in a meeting")
    a.emit(build("risk_breach", "breach", ts=T0))

    channels = [c for _, c in sent]
    assert Channel.SMS in channels


def test_a_mute_expires():
    """The failure mode is a mute that lapses, not one that persists."""
    a, _sent = alerter()
    a.mute(Channel.PUSH, until=T0 + timedelta(hours=1), reason="meeting")

    assert a.muted_channels(T0) == {Channel.PUSH}
    assert a.muted_channels(T0 + timedelta(hours=2)) == set()


def test_a_mute_needs_a_reason():
    """An unexplained silence is indistinguishable from a broken alerter."""
    a, _ = alerter()
    with pytest.raises(ValueError, match="needs a reason"):
        a.mute(Channel.PUSH, until=T0 + timedelta(hours=1), reason="")


def test_an_alert_with_every_channel_muted_is_recorded_as_suppressed():
    a, sent = alerter()
    a.mute(Channel.PUSH, until=T0 + timedelta(hours=1), reason="meeting")
    a.emit(build("position_exit", "exit", ts=T0))

    assert sent == []
    assert len(a.suppressed) == 1
    assert "every channel muted" in a.suppressed[0].reason


# ------------------------------------------------------------------ the table
def test_every_kind_has_a_severity_and_a_channel():
    for kind, (severity, channels) in RULES.items():
        assert isinstance(severity, Severity), kind
        assert channels, f"{kind} would be built and delivered nowhere"


def test_an_unknown_kind_is_refused_rather_than_defaulted():
    """The default that is safe for a status message is the wrong one for a
    breach."""
    with pytest.raises(KeyError, match="unknown alert kind"):
        build("something_new", "?", ts=T0)


def test_a_breach_goes_to_two_channels():
    alert = build("risk_breach", "L3", ts=T0)
    assert Channel.PUSH in alert.channels and Channel.SMS in alert.channels


def test_a_high_severity_event_with_no_trade_is_still_worth_saying():
    """ "Something is happening, risk is elevated" is valuable even without a
    trade - and it is silent, so it does not compete with the actionable ones."""
    alert = build("high_severity_no_trade", "escalation", ts=T0)
    assert alert.severity is Severity.INFO
    assert alert.channels == (Channel.PUSH_SILENT,)


def test_calibration_decay_does_not_interrupt():
    assert build("calibration_decay", "slope drifting", ts=T0).severity is Severity.DIGEST


# ------------------------------------------------------------------ the digest
def test_the_digest_reports_what_was_not_sent():
    """Nothing happened and forty things happened and were suppressed look
    identical from the receiving end, and only one of them is a quiet day."""
    a, _ = alerter(max_per_window=1)
    for i in range(4):
        a.emit(build("position_exit", "x", ts=T0 + timedelta(seconds=i)))

    digest = a.digest()
    assert "1 delivered, 3 suppressed" in digest
    assert "rate limit" in digest


def test_the_digest_flags_suppressed_actionable_alerts():
    a, _ = alerter(max_per_window=1)
    for i in range(3):
        a.emit(build("tradeable_signal", "signal", ts=T0 + timedelta(seconds=i)))
    assert "actionable alerts were suppressed" in a.digest()


def test_a_quiet_digest_does_not_cry_wolf():
    a, _ = alerter()
    a.emit(build("calibration_decay", "slow drift", ts=T0))
    assert "NOTE" not in a.digest()
