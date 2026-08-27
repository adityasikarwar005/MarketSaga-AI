"""M92 - alerting.

Two rules do all the work here, and they pull in opposite directions.

**Rate-limit everything, so the alerts stay worth reading.** An operator who
learns that most notifications do not need action stops reading them, and the
one that did need action arrives into that habit. Alert fatigue is not a comfort
problem; it is the mechanism by which a working alarm fails.

**Except the ones that cannot be missed.** A risk breach is not a notification,
it is an instruction, and a rate limiter that can swallow it has converted a
safety control into a convenience. :class:`Severity.CRITICAL` bypasses both the
rate limit and the mute, and there is no parameter that changes that - a
configurable override would be configured, on the day it mattered, by someone
trying to get some sleep.

Muting is per-channel and never global
--------------------------------------

"Mute everything" is the button that gets pressed during the noisy hour and
forgotten by the quiet one. Mutes here attach to a channel and carry an expiry,
so the failure mode is a mute that lapses rather than one that persists.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum

from libs.ms_core.obs import get_logger

log = get_logger(__name__)


class Severity(IntEnum):
    """Ordered, so "at least this serious" is a comparison rather than a set."""

    DIGEST = 0  # slow-moving; does not need to interrupt
    INFO = 1  # something is happening
    ACTION = 2  # a trade, an exit
    CRITICAL = 3  # a risk breach. Never suppressed, never muted.

    @property
    def bypasses_everything(self) -> bool:
        return self is Severity.CRITICAL


class Channel(StrEnum):
    PUSH = "push"
    PUSH_SILENT = "push_silent"
    SMS = "sms"
    EMAIL = "email"
    DIGEST = "digest"


@dataclass(frozen=True)
class Alert:
    kind: str
    severity: Severity
    message: str
    channels: tuple[Channel, ...]
    ts: datetime
    dedupe_key: str = ""

    @property
    def key(self) -> str:
        """What the rate limiter counts. Defaults to the kind, so a burst of the
        same kind of thing is one alert rather than forty."""
        return self.dedupe_key or self.kind


# The rules from the blueprint, as data rather than as branches. A table can be
# read in one go and argued with; a chain of ifs cannot.
RULES: dict[str, tuple[Severity, tuple[Channel, ...]]] = {
    "tradeable_signal": (Severity.ACTION, (Channel.PUSH,)),
    "high_severity_no_trade": (Severity.INFO, (Channel.PUSH_SILENT,)),
    "position_exit": (Severity.ACTION, (Channel.PUSH,)),
    "risk_breach": (Severity.CRITICAL, (Channel.PUSH, Channel.SMS)),
    "system_health": (Severity.DIGEST, (Channel.EMAIL,)),
    "system_health_l3": (Severity.CRITICAL, (Channel.PUSH, Channel.SMS)),
    "calibration_decay": (Severity.DIGEST, (Channel.DIGEST,)),
}


def build(kind: str, message: str, *, ts: datetime, dedupe_key: str = "") -> Alert:
    """Construct an alert from the rule table, refusing unknown kinds.

    An unknown kind would otherwise get a default severity, and the default that
    is safe for a status message is the wrong one for a breach. Adding a kind
    should be a decision about which channel it belongs on.
    """
    if kind not in RULES:
        raise KeyError(
            f"unknown alert kind {kind!r}. Add it to RULES with a severity and a "
            "channel - defaulting it would give a breach the severity of a status "
            "message."
        )
    severity, channels = RULES[kind]
    return Alert(
        kind=kind,
        severity=severity,
        message=message,
        channels=channels,
        ts=ts,
        dedupe_key=dedupe_key,
    )


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
@dataclass
class Mute:
    """A channel silenced until a time. Never indefinitely."""

    channel: Channel
    until: datetime
    reason: str

    def active(self, at: datetime) -> bool:
        return at < self.until


@dataclass
class Suppressed:
    """An alert that was not delivered, and why.

    Kept rather than dropped. "Nothing happened" and "forty things happened and
    were suppressed" look identical from the receiving end, and only one of them
    is a quiet day.
    """

    alert: Alert
    reason: str


@dataclass
class Alerter:
    """Applies the rules, the rate limit and the mutes - in that order.

    ``window`` and ``max_per_window`` are per alert key, not global. A global
    budget means a burst of one kind of alert exhausts the allowance for every
    other kind, which is precisely backwards: the burst is the thing that should
    be throttled, not everything else.
    """

    send: Callable[[Alert, Channel], None]
    window: timedelta = timedelta(minutes=10)
    max_per_window: int = 3
    _recent: dict[str, deque[datetime]] = field(default_factory=dict, repr=False)
    _mutes: list[Mute] = field(default_factory=list, repr=False)
    suppressed: list[Suppressed] = field(default_factory=list)
    delivered: int = 0

    # ------------------------------------------------------------------ mutes
    def mute(self, channel: Channel, *, until: datetime, reason: str) -> None:
        """Silence one channel until a time.

        Per-channel and time-bounded on purpose. "Mute everything" is the button
        pressed during the noisy hour and forgotten by the quiet one, so the
        failure mode here is a mute that lapses rather than one that persists.
        """
        if not reason.strip():
            raise ValueError(
                "a mute needs a reason. An unexplained silence is indistinguishable "
                "from a broken alerter."
            )
        self._mutes.append(Mute(channel=channel, until=until, reason=reason.strip()))
        log.info("channel muted", fields={"channel": channel.value, "until": until.isoformat()})

    def muted_channels(self, at: datetime) -> set[Channel]:
        return {m.channel for m in self._mutes if m.active(at)}

    # ------------------------------------------------------------- rate limit
    def _within_limit(self, alert: Alert) -> bool:
        seen = self._recent.setdefault(alert.key, deque())
        cutoff = alert.ts - self.window
        while seen and seen[0] < cutoff:
            seen.popleft()
        return len(seen) < self.max_per_window

    def _record(self, alert: Alert) -> None:
        self._recent.setdefault(alert.key, deque()).append(alert.ts)

    # ---------------------------------------------------------------- deliver
    def emit(self, alert: Alert) -> list[Channel]:
        """Deliver on every channel that is not muted, subject to the rate limit.

        CRITICAL bypasses both, and there is no flag that changes that. A
        configurable override would be configured on the day it mattered, by
        someone trying to get some sleep.
        """
        if alert.severity.bypasses_everything:
            for channel in alert.channels:
                self.send(alert, channel)
            self.delivered += 1
            self._record(alert)
            log.info(
                "critical alert delivered",
                fields={"kind": alert.kind, "channels": [c.value for c in alert.channels]},
            )
            return list(alert.channels)

        if not self._within_limit(alert):
            self.suppressed.append(
                Suppressed(
                    alert,
                    f"rate limit: more than {self.max_per_window} of {alert.key!r} "
                    f"within {self.window}",
                )
            )
            return []

        muted = self.muted_channels(alert.ts)
        open_channels = [c for c in alert.channels if c not in muted]
        if not open_channels:
            self.suppressed.append(
                Suppressed(alert, f"every channel muted: {sorted(c.value for c in muted)}")
            )
            return []

        for channel in open_channels:
            self.send(alert, channel)
        self.delivered += 1
        self._record(alert)
        return open_channels

    # ----------------------------------------------------------------- report
    def digest(self) -> str:
        """What was sent, and what was not.

        The suppressed list is the interesting half. Nothing happened and forty
        things happened and were suppressed look identical from the receiving
        end, and only one of them is a quiet day.
        """
        lines = [f"  {self.delivered} delivered, {len(self.suppressed)} suppressed"]
        by_reason: dict[str, int] = {}
        for s in self.suppressed:
            head = s.reason.split(":")[0]
            by_reason[head] = by_reason.get(head, 0) + 1
        lines += [f"    {n} x {reason}" for reason, n in sorted(by_reason.items())]
        if any(s.alert.severity >= Severity.ACTION for s in self.suppressed):
            lines.append(
                "    NOTE: actionable alerts were suppressed. Check the rate limit "
                "is not hiding a genuine burst."
            )
        return "\n".join(lines)


def critical_kinds() -> Sequence[str]:
    """Which kinds can never be muted. Useful for asserting the table is sane."""
    return [k for k, (sev, _) in RULES.items() if sev is Severity.CRITICAL]
