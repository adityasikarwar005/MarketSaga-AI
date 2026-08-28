"""M20 - the source registry.

    pytest tests/unit/test_sources.py -k "weights_in_range or all_have_tos_note"

Two rules matter more than the rest. A source below the media tier may raise an
alert and may never open a position on its own - that is the line the AP hack of
2013 is about. And a rate limit that records calls after making them is a log,
not a limiter; the block it fails to prevent arrives on the morning the news
matters, because that is the morning the poller is busiest.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import FrozenClock
from libs.ms_core.config import Source, SourcesConfig
from libs.ms_news.sources import (
    CONFIRMATION_CREDIBILITY,
    DEFAULT_LIMITS,
    TIER_MEDIA,
    Budget,
    RateLimit,
    Registry,
    SourceError,
    Tier,
    can_open_directional,
    enabled_sources,
    independent,
    may_confirm,
    readiness,
    tier_of,
)

T0 = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)


def source(
    sid: int = 1,
    *,
    credibility: float = 0.90,
    directional: bool = True,
    enabled: bool = True,
    source_class: str = "wire",
    tos: str = "reviewed 2026-08-01: RSS permitted, no archival",
) -> Source:
    return Source(
        id=sid,
        name=f"source-{sid}",
        source_class=source_class,
        credibility=credibility,
        typical_lag_seconds=5,
        may_trigger_directional=directional,
        tos_note=tos,
        enabled=enabled,
    )


@pytest.fixture
def shipped_sources() -> SourcesConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cfgmod.load().sources


# --------------------------------------------------------------------------- #
# weights_in_range
# --------------------------------------------------------------------------- #
def test_weights_in_range_for_every_shipped_source(shipped_sources: SourcesConfig):
    for s in shipped_sources.sources:
        assert 0.0 <= s.credibility <= 1.0, s.name


def test_weights_in_range_is_enforced_by_the_schema():
    with pytest.raises(ValueError):
        source(credibility=1.4)
    with pytest.raises(ValueError):
        source(credibility=-0.1)


def test_weights_in_range_and_the_tiers_partition_them():
    """Every credibility falls in exactly one tier, and the boundaries do not
    leave a gap a source can sit in unclassified."""
    seen = {tier_of(source(credibility=c)) for c in (1.0, 0.95, 0.94, 0.80, 0.79, 0.50, 0.49, 0.0)}
    assert seen == set(Tier)


def test_the_tier_comes_from_the_number_not_the_class_string():
    """A class string is a label somebody typed; the credibility is the weight
    everything downstream applies. Letting them disagree means the tier in a
    report is not the tier the scorer used."""
    mislabelled = source(credibility=0.25, source_class="primary_official", directional=False)
    assert tier_of(mislabelled) is Tier.UNVERIFIED


def test_a_low_credibility_source_may_never_open_a_position():
    """The line the AP hack is about. A source can make us look at something
    without being able to make us take a position."""
    assert not can_open_directional(source(credibility=0.25, directional=False))
    assert can_open_directional(source(credibility=0.90, directional=True))


def test_the_rule_holds_even_for_a_source_that_claims_otherwise():
    """Defence in depth, and the two layers guard different doors.

    The cross-field rule lives on `SourcesConfig`, so it fires when a *file* is
    loaded. A bare `Source` carries no such validator and can be constructed
    claiming `may_trigger_directional=True` at 0.25 credibility - by a test
    fixture, by a hand-built object, by some future loader that does not go
    through `SourcesConfig`.

    `can_open_directional` has to refuse that on its own, or the only thing
    standing between a 0.25-credibility source and a position is whichever path
    the object arrived by. A mutation test found this: dropping the credibility
    half of the check broke nothing, because every other test used a source that
    already said `False`.
    """
    contradictory = Source(
        id=99,
        name="loud",
        source_class="social_unverified",
        credibility=0.25,
        typical_lag_seconds=1,
        may_trigger_directional=True,  # the config would reject this; the object does not
        tos_note="reviewed",
        enabled=True,
    )
    assert contradictory.may_trigger_directional
    assert contradictory.credibility < TIER_MEDIA
    assert not can_open_directional(contradictory), (
        "a source below the media tier opened a directional position because it " "claimed it could"
    )


def test_the_config_refuses_a_low_credibility_directional_source():
    """Enforced at load, so the file itself cannot express the unsafe state."""
    with pytest.raises(ValueError, match="never open a directional position"):
        SourcesConfig(sources=[source(credibility=0.25, directional=True)])


def test_only_high_credibility_sources_may_confirm():
    """Severity 9 needs two independent sources at 0.8 or above. M29 enforces
    the count; this is the predicate it counts with."""
    assert may_confirm(source(credibility=CONFIRMATION_CREDIBILITY))
    assert not may_confirm(source(credibility=CONFIRMATION_CREDIBILITY - 0.01))


def test_two_feeds_of_the_same_class_are_not_independent():
    """Two RSS feeds republishing one wire are one source wearing two names, and
    counting them as two is how a single-sourced rumour clears a threshold that
    was meant to require corroboration."""
    same = [source(1, source_class="national_media"), source(2, source_class="national_media")]
    mixed = [source(1, source_class="wire"), source(2, source_class="primary_official")]

    assert not independent(same)
    assert independent(mixed)
    assert not independent([source(1)]), "one source is not corroboration"


# --------------------------------------------------------------------------- #
# all_have_tos_note
# --------------------------------------------------------------------------- #
def test_all_have_tos_note_in_the_shipped_config(shipped_sources: SourcesConfig):
    for s in shipped_sources.sources:
        assert s.tos_note.strip(), s.name


def test_all_have_tos_note_is_enforced_at_load():
    with pytest.raises(ValueError, match="tos_note is required"):
        SourcesConfig(sources=[source(tos="")])


def test_a_todo_tos_note_is_reported_as_unreviewed(shipped_sources: SourcesConfig):
    """Every shipped source still carries a TODO, and that is the correct state:
    reading a provider's terms is not a coding task."""
    state = readiness(shipped_sources)
    assert state.unreviewed_tos
    assert "legal problem wearing a feature's clothes" in state.render()


def test_nothing_is_enabled_yet_and_that_is_not_a_fault(shipped_sources: SourcesConfig):
    """A collector that crashed on an empty list would be crashing on a correct
    configuration."""
    state = readiness(shipped_sources)
    assert state.enabled == 0
    assert not state.can_collect
    assert enabled_sources(shipped_sources) == []
    assert "not a fault" in state.render()


def test_readiness_clears_once_a_source_is_enabled_and_reviewed():
    cfg = SourcesConfig(sources=[source(1, enabled=True), source(2, enabled=False)])
    state = readiness(cfg)

    assert state.can_collect
    assert state.unreviewed_tos == []
    assert len(enabled_sources(cfg)) == 1


# --------------------------------------------------------------------------- #
# rate limits: asked before, not recorded after
# --------------------------------------------------------------------------- #
def test_the_budget_refuses_before_the_call_not_after():
    """Recording a call that already happened is a log, not a limiter."""
    b = Budget(source_id=1, limit=RateLimit(per_window=2, window=timedelta(hours=1)))

    assert b.take(T0)
    assert b.take(T0)
    assert not b.take(T0), "a third call was permitted inside a limit of two"


def test_the_minimum_interval_is_enforced_separately_from_the_quota():
    """A limiter honouring only the quota fires a hundred requests in the first
    second of the window and gets blocked having used a tenth of its allowance."""
    b = Budget(
        source_id=1,
        limit=RateLimit(
            per_window=100, window=timedelta(hours=1), min_interval=timedelta(seconds=10)
        ),
    )
    assert b.take(T0)
    assert not b.take(T0 + timedelta(seconds=5))
    assert b.take(T0 + timedelta(seconds=10))


def test_the_window_is_wall_clock_aligned_not_rolling():
    """A provider's "1 000 per day" resets at their midnight. A rolling window
    allows nearly double across a reset boundary, and the overage looks like
    ordinary traffic right until the block."""
    hour = timedelta(hours=1)
    b = Budget(source_id=1, limit=RateLimit(per_window=2, window=hour))

    late = datetime(2026, 8, 27, 6, 59, tzinfo=UTC)
    assert b.take(late)
    assert b.take(late)
    assert not b.take(late)

    # One minute later is a new aligned window.
    assert b.take(datetime(2026, 8, 27, 7, 0, tzinfo=UTC))


def test_remaining_reports_a_full_allowance_in_a_fresh_window():
    b = Budget(source_id=1, limit=RateLimit(per_window=5, window=timedelta(hours=1)))
    b.take(T0)
    assert b.remaining(T0) == 4
    assert b.remaining(T0 + timedelta(hours=1)) == 5


def test_ready_at_says_when_the_next_call_is_permitted():
    b = Budget(
        source_id=1,
        limit=RateLimit(
            per_window=1, window=timedelta(hours=1), min_interval=timedelta(seconds=30)
        ),
    )
    b.take(T0)

    assert b.ready_at(T0) == T0 + timedelta(seconds=30)
    assert b.ready_at(T0 + timedelta(minutes=1)) == T0.replace(minute=0) + timedelta(hours=1)


def test_a_nonsense_limit_is_refused():
    with pytest.raises(SourceError, match="per_window must be positive"):
        RateLimit(per_window=0, window=timedelta(hours=1))
    with pytest.raises(SourceError, match="window must be positive"):
        RateLimit(per_window=10, window=timedelta(0))


def test_every_tier_has_a_default_limit():
    """A source with no explicit limit must not end up unlimited."""
    assert set(DEFAULT_LIMITS) == set(Tier)
    for tier, limit in DEFAULT_LIMITS.items():
        assert limit.per_window > 0, tier
        assert limit.min_interval > timedelta(0), tier


def test_the_defaults_are_stricter_for_less_trusted_tiers():
    assert (
        DEFAULT_LIMITS[Tier.OFFICIAL].per_window
        > DEFAULT_LIMITS[Tier.WIRE].per_window
        > DEFAULT_LIMITS[Tier.MEDIA].per_window
        > DEFAULT_LIMITS[Tier.UNVERIFIED].per_window
    )


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
def test_the_registry_gives_every_source_a_budget(shipped_sources: SourcesConfig):
    registry = Registry.from_config(shipped_sources)
    assert set(registry.budgets) == set(registry.sources)


def test_a_disabled_source_is_never_pollable_even_with_budget_left():
    cfg = SourcesConfig(sources=[source(1, enabled=False)])
    registry = Registry.from_config(cfg)

    assert not registry.may_poll(1, clock=FrozenClock(T0))
    assert registry.budgets[1].remaining(T0) == DEFAULT_LIMITS[Tier.WIRE].per_window, (
        "checking the budget before the enabled flag spent quota on a source we "
        "were never going to poll"
    )


def test_may_poll_consumes_budget_for_an_enabled_source():
    cfg = SourcesConfig(sources=[source(1, enabled=True)])
    registry = Registry.from_config(cfg)
    clock = FrozenClock(T0)

    before = registry.budgets[1].remaining(T0)
    assert registry.may_poll(1, clock=clock)
    assert registry.budgets[1].remaining(T0) == before - 1


def test_an_unknown_source_is_refused_rather_than_silently_allowed():
    with pytest.raises(SourceError, match="not in the registry"):
        Registry().get(404)


def test_an_explicit_limit_overrides_the_tier_default():
    cfg = SourcesConfig(sources=[source(1)])
    tight = RateLimit(per_window=3, window=timedelta(hours=1), min_interval=timedelta(seconds=1))
    registry = Registry.from_config(cfg, limits={1: tight})

    assert registry.budgets[1].limit is tight


def test_the_registry_render_shows_permissions_and_budget(shipped_sources: SourcesConfig):
    text = Registry.from_config(shipped_sources).render(clock=FrozenClock(T0))

    assert "disabled" in text
    assert "alerts only" in text, "the unverified social source should be marked"
    assert "may confirm" in text
    assert "left in this" in text
