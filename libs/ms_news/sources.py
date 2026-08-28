"""M20 - the source registry: rate limits, quota, and what a source is allowed to do.

The schema and the credibility rule already live in ``ms_core.config`` - a source
below 0.5 credibility may raise an alert and may never open a directional
position on its own, and the config refuses to load a file that says otherwise.

What is here is the operational half: how often a source may be polled, how much
of its quota is left, and whether it is currently trustworthy enough to act on.

A rate limit you did not enforce is a ban
-------------------------------------------

Every source in this file is somebody else's server. Exceeding a published limit
gets an IP blocked, and the block arrives on the morning the news actually
matters - because that is the morning the poller is busiest. So the limiter is
**pre-emptive**: :meth:`Budget.take` refuses before the request rather than
recording that it happened, and refusing costs one missed poll while being
banned costs the day.

Quota is tracked per source and per window, and the windows are wall-clock
aligned - a provider's "1000 requests per day" resets at their midnight, not
twenty-four hours after your first call.

Credibility is a weight, and one hard rule
--------------------------------------------

``credibility`` scales how much an item moves the impact score. It is not a
trust switch. The switch is ``may_trigger_directional``, and it is the line
between "this source can make us look at something" and "this source can make us
take a position" - the distinction the AP hack of 2013 is about.

:func:`can_open_directional` puts that rule behind a function rather than leaving
each caller to remember it, because the callers that forget are the ones written
at speed during an incident.

A disabled source is not a missing source
-------------------------------------------

Every source in ``config/sources.yaml`` currently has ``enabled: false`` and a
``tos_note`` beginning "TODO". That is the correct state: reading a provider's
terms is not a coding task, and a source enabled before somebody read them is a
legal problem wearing a feature's clothes. :func:`enabled_sources` returns an
empty list and :func:`readiness` says why, rather than either quietly returning
nothing or pretending the registry is broken.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from libs.ms_core.clock import Clock
from libs.ms_core.config import Source, SourcesConfig
from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Credibility tiers. The numbers are the boundaries the impact scorer and the
# confirmation state machine both read, so they live in one place.
TIER_OFFICIAL = 0.95  # central bank, exchange: an announcement, not a report
TIER_WIRE = 0.80  # a wire service with a correction policy
TIER_MEDIA = 0.50  # national business media
# Below TIER_MEDIA a source may raise an alert and never open a position.

# Severity 9+ requires two independent sources at or above this. M29 enforces
# it; the constant is here because it is a property of the registry.
CONFIRMATION_CREDIBILITY = 0.80


class Tier(StrEnum):
    OFFICIAL = "official"
    WIRE = "wire"
    MEDIA = "media"
    UNVERIFIED = "unverified"


class SourceError(RuntimeError):
    pass


def tier_of(source: Source) -> Tier:
    """Which band a source falls in, by credibility rather than by name.

    By the number, not by ``source_class``. A class string is a label somebody
    typed; the credibility is the weight everything downstream actually uses,
    and letting the two disagree means the tier shown in a report is not the
    tier the scorer applied.
    """
    if source.credibility >= TIER_OFFICIAL:
        return Tier.OFFICIAL
    if source.credibility >= TIER_WIRE:
        return Tier.WIRE
    if source.credibility >= TIER_MEDIA:
        return Tier.MEDIA
    return Tier.UNVERIFIED


def can_open_directional(source: Source) -> bool:
    """May an item from this source, alone, open a position?

    Behind a function rather than left to each caller, because the callers that
    forget the rule are the ones written at speed during an incident - which is
    exactly when a low-credibility source is loudest.
    """
    return source.may_trigger_directional and source.credibility >= TIER_MEDIA


def may_confirm(source: Source) -> bool:
    """May this source count toward the two needed for a severity-9 event?"""
    return source.credibility >= CONFIRMATION_CREDIBILITY


# --------------------------------------------------------------------------- #
# Rate limits and quota
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RateLimit:
    """What a provider says you may do. Both halves are usually published.

    ``per_window`` with ``window`` is the quota; ``min_interval`` is the
    burst rule. A limiter that honours only the quota will fire a hundred
    requests in the first second of a window and get blocked before it has used
    a tenth of what it was allowed.
    """

    per_window: int
    window: timedelta
    min_interval: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        if self.per_window <= 0:
            raise SourceError("per_window must be positive")
        if self.window <= timedelta(0):
            raise SourceError("window must be positive")


@dataclass
class Budget:
    """Remaining quota for one source, against a wall-clock aligned window.

    Aligned rather than rolling from first use: a provider's "1 000 per day"
    resets at their midnight. A rolling window silently allows nearly double
    across a reset boundary, and the overage looks like normal traffic right up
    until the block.
    """

    source_id: int
    limit: RateLimit
    used: int = 0
    window_start: datetime | None = None
    last_call: datetime | None = None

    def _align(self, now: datetime) -> datetime:
        """Start of the window containing ``now``, anchored at the epoch."""
        seconds = int(self.limit.window.total_seconds())
        stamp = int(now.timestamp())
        return datetime.fromtimestamp(stamp - (stamp % seconds), tz=now.tzinfo)

    def remaining(self, now: datetime) -> int:
        if self.window_start is None or self._align(now) != self.window_start:
            return self.limit.per_window
        return max(self.limit.per_window - self.used, 0)

    def ready_at(self, now: datetime) -> datetime:
        """The earliest instant a call is permitted."""
        if self.last_call is not None:
            earliest = self.last_call + self.limit.min_interval
            if earliest > now:
                return earliest
        if self.remaining(now) > 0:
            return now
        assert self.window_start is not None
        return self.window_start + self.limit.window

    def take(self, now: datetime) -> bool:
        """Consume one call if permitted. **Asks before, not after.**

        Recording a call that already happened is not a rate limiter, it is a
        log. Refusing costs one missed poll; being blocked costs the day, and
        the block arrives on the morning the news matters because that is the
        morning the poller is busiest.
        """
        current = self._align(now)
        if self.window_start != current:
            self.window_start = current
            self.used = 0

        if self.last_call is not None and now - self.last_call < self.limit.min_interval:
            return False
        if self.used >= self.limit.per_window:
            return False

        self.used += 1
        self.last_call = now
        return True

    def render(self, now: datetime) -> str:
        return (
            f"    source {self.source_id:<4} {self.remaining(now):>5}/{self.limit.per_window} "
            f"left in this {self.limit.window}"
        )


@dataclass
class Registry:
    """Every configured source, with its budget and its permissions."""

    sources: dict[int, Source] = field(default_factory=dict)
    budgets: dict[int, Budget] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, cfg: SourcesConfig, *, limits: dict[int, RateLimit] | None = None
    ) -> Registry:
        registry = cls()
        for source in cfg.sources:
            registry.sources[source.id] = source
            limit = (limits or {}).get(source.id, DEFAULT_LIMITS.get(tier_of(source)))
            if limit is not None:
                registry.budgets[source.id] = Budget(source_id=source.id, limit=limit)
        return registry

    def get(self, source_id: int) -> Source:
        if source_id not in self.sources:
            raise SourceError(f"source {source_id} is not in the registry")
        return self.sources[source_id]

    def enabled(self) -> list[Source]:
        return [s for s in self.sources.values() if s.enabled]

    def may_poll(self, source_id: int, *, clock: Clock) -> bool:
        """Enabled, and inside its budget. Both, in that order.

        A disabled source that is inside its budget is still not pollable, and
        checking the budget first would spend quota deciding not to use it.
        """
        source = self.get(source_id)
        if not source.enabled:
            return False
        budget = self.budgets.get(source_id)
        return True if budget is None else budget.take(clock.now())

    def render(self, *, clock: Clock) -> str:
        now = clock.now()
        lines = [
            f"  source registry: {len(self.sources)} configured, {len(self.enabled())} enabled"
        ]
        for source in sorted(self.sources.values(), key=lambda s: s.id):
            flags = []
            if not source.enabled:
                flags.append("disabled")
            if not can_open_directional(source):
                flags.append("alerts only")
            if may_confirm(source):
                flags.append("may confirm")
            lines.append(
                f"    {source.id:<4} {source.name[:34]:<34} {tier_of(source).value:<11} "
                f"c={source.credibility:.2f}  {', '.join(flags)}"
            )
            budget = self.budgets.get(source.id)
            if budget is not None:
                lines.append(budget.render(now))
        return "\n".join(lines)


# Conservative starting points by tier, used when a source has no explicit
# limit. Deliberately below what most providers allow: the cost of being too
# slow is a late item, and the cost of being too fast is a block.
DEFAULT_LIMITS: dict[Tier, RateLimit] = {
    Tier.OFFICIAL: RateLimit(
        per_window=720, window=timedelta(hours=1), min_interval=timedelta(seconds=4)
    ),
    Tier.WIRE: RateLimit(
        per_window=300, window=timedelta(hours=1), min_interval=timedelta(seconds=10)
    ),
    Tier.MEDIA: RateLimit(
        per_window=120, window=timedelta(hours=1), min_interval=timedelta(seconds=25)
    ),
    Tier.UNVERIFIED: RateLimit(
        per_window=60, window=timedelta(hours=1), min_interval=timedelta(seconds=55)
    ),
}


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
@dataclass
class Readiness:
    """Whether the news layer can collect anything, and what is stopping it."""

    configured: int
    enabled: int
    unreviewed_tos: list[str] = field(default_factory=list)

    @property
    def can_collect(self) -> bool:
        return self.enabled > 0

    def render(self) -> str:
        lines = [f"  {self.enabled}/{self.configured} sources enabled"]
        if self.unreviewed_tos:
            lines.append(
                f"    {len(self.unreviewed_tos)} source(s) still carry a TODO terms-of-service "
                "note. Reading a provider's terms is not a coding task, and a source "
                "enabled before somebody read them is a legal problem wearing a "
                "feature's clothes:"
            )
            lines += [f"      - {name}" for name in self.unreviewed_tos]
        if not self.can_collect:
            lines.append(
                "  -> the news layer cannot collect. That is the configured state, not "
                "a fault: every source is disabled pending its terms review."
            )
        return "\n".join(lines)


def readiness(cfg: SourcesConfig) -> Readiness:
    return Readiness(
        configured=len(cfg.sources),
        enabled=sum(1 for s in cfg.sources if s.enabled),
        unreviewed_tos=[
            s.name for s in cfg.sources if s.tos_note.strip().upper().startswith("TODO")
        ],
    )


def enabled_sources(cfg: SourcesConfig) -> list[Source]:
    """Sources that may actually be polled.

    Returns an empty list when nothing is enabled rather than raising: no
    enabled sources is the configured state today, and a collector that crashes
    on it would be crashing on a correct configuration.
    """
    return [s for s in cfg.sources if s.enabled]


def independent(sources: Sequence[Source]) -> bool:
    """Are these sources independent enough to corroborate each other?

    Two RSS feeds that both republish the same wire are one source wearing two
    names, and counting them as two is how a single-sourced rumour reaches the
    threshold that was meant to require corroboration. Independence is
    approximated by distinct ``source_class`` - a weak test, and named as one
    here so it is not mistaken for a strong one.
    """
    classes = {s.source_class for s in sources}
    return len(classes) == len(sources) and len(sources) > 1
