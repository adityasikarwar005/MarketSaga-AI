"""M15 verify - cross-asset collector and the overnight window."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pytest

from libs.ms_core.clock import UTC, FrozenClock, SessionCalendar
from libs.ms_core.schemas import Candle
from libs.ms_data.crossasset import (
    CROSSASSET_ID_BASE,
    CrossAssetCollector,
    CrossAssetProvider,
    CrossAssetRegistry,
    CrossAssetSeries,
    SyntheticCrossAssetProvider,
    overnight_moves,
    overnight_window,
    previous_trading_day,
    registry_from_config,
)
from tests.conftest import ist

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")


@pytest.fixture
def registry(cfg):
    return registry_from_config(cfg.crossasset.series)


@pytest.fixture
def cal(cfg):
    """Calendar with a known 2026 holiday list so trading-day lookups can answer."""
    base = SessionCalendar.from_config(cfg.sessions)
    return base.with_holidays({2026: frozenset({date(2026, 8, 15), date(2026, 9, 14)})})


# ------------------------------------------------------------------ registry
def test_registry_loads_every_series(registry):
    assert len(registry) == 8
    assert registry.by_symbol("GIFT_NIFTY").category == "equity_futures"
    assert registry.by_symbol("BRENT").india_channel


def test_ids_cannot_collide_with_nse_instruments(registry, cfg):
    """Cross-asset bars share the Candle schema and the storage path, so a
    colliding id would silently merge two different assets into one series."""
    nse_ids = {i.id for i in cfg.instruments.instruments}
    for series in registry.all():
        assert series.instrument_id >= CROSSASSET_ID_BASE
        assert series.instrument_id not in nse_ids


def test_low_id_is_rejected():
    with pytest.raises(ValueError, match="must be >="):
        CrossAssetRegistry(
            [
                CrossAssetSeries(
                    instrument_id=5,
                    symbol="X",
                    name="X",
                    category="fx",
                    timezone="UTC",
                    open_time=dtime(0, 0),
                    close_time=dtime(23, 59),
                    spans_midnight=False,
                    weekdays_only=True,
                    typical_lag_seconds=1,
                    india_channel="test",
                )
            ]
        )


def test_unknown_symbol_is_a_clear_error(registry):
    with pytest.raises(KeyError, match="unknown cross-asset series"):
        registry.by_symbol("NOT_A_SERIES")


# -------------------------------------------------------------- trading hours
def test_normal_session_hours(registry):
    brent = registry.by_symbol("BRENT")  # 01:00-23:00 London, no midnight span
    london = ZoneInfo("Europe/London")
    assert brent.is_open(datetime(2026, 8, 26, 12, 0, tzinfo=london))
    assert not brent.is_open(datetime(2026, 8, 26, 23, 30, tzinfo=london))


def test_series_that_spans_midnight_is_open_at_night(registry):
    """The opposite test to a normal session. Inverting it marks a live feed as
    dead all night, every night."""
    us = registry.by_symbol("US_EQ_FUT")  # opens 18:00 NY, closes 17:00 next day
    assert us.is_open(datetime(2026, 8, 26, 22, 0, tzinfo=NY))  # after the open
    assert us.is_open(datetime(2026, 8, 27, 3, 0, tzinfo=NY))  # before the close
    assert not us.is_open(datetime(2026, 8, 26, 17, 30, tzinfo=NY))  # in the daily break


def test_weekend_is_closed_but_sunday_evening_opens(registry):
    """A Sunday-evening open is a Monday session for these venues."""
    us = registry.by_symbol("US_EQ_FUT")
    assert not us.is_open(datetime(2026, 8, 29, 12, 0, tzinfo=NY))  # Saturday midday
    assert us.is_open(datetime(2026, 8, 30, 19, 0, tzinfo=NY))  # Sunday 19:00


def test_open_now_lists_only_live_venues(registry):
    at_night_ist = datetime(2026, 8, 26, 23, 0, tzinfo=IST)
    open_symbols = {s.symbol for s in registry.open_now(at_night_ist)}
    assert "US_EQ_FUT" in open_symbols  # New York is mid-session
    assert "USDINR" not in open_symbols  # Indian FX hours are long over


def test_staleness_budget_is_generous_relative_to_lag(registry):
    for series in registry.all():
        assert series.staleness_budget() >= 120.0
        assert series.staleness_budget() >= series.typical_lag_seconds


# ---------------------------------------------------------- previous session
def test_previous_trading_day_skips_the_weekend(cal):
    monday = date(2026, 8, 31)
    assert monday.weekday() == 0
    assert previous_trading_day(cal, monday) == date(2026, 8, 28)  # Friday


def test_previous_trading_day_skips_a_holiday(cal):
    """2026-09-14 is Ganesh Chaturthi in the loaded calendar."""
    after = date(2026, 9, 15)
    assert previous_trading_day(cal, after) == date(2026, 9, 11)  # Friday before


def test_previous_trading_day_normal_case(cal):
    assert previous_trading_day(cal, date(2026, 8, 27)) == date(2026, 8, 26)


# ----------------------------------------------------------- overnight window
def test_window_runs_from_previous_close_to_the_cutoff(cal):
    w = overnight_window(cal, date(2026, 8, 27))
    assert w.previous_session == date(2026, 8, 26)
    assert w.start.astimezone(IST).strftime("%H:%M") == "15:30"
    assert w.end.astimezone(IST).strftime("%H:%M") == "08:45"
    assert 15 < w.hours < 20


def test_window_widens_after_a_weekend(cal):
    """Not `day - 1`. A model trained on a window that assumed 24 hours
    mis-weights every Monday it ever sees."""
    monday = overnight_window(cal, date(2026, 8, 31))
    tuesday = overnight_window(cal, date(2026, 8, 27))
    assert monday.spans_multiple_days
    assert not tuesday.spans_multiple_days
    assert monday.hours > tuesday.hours + 40


def test_window_ends_at_the_cutoff_not_the_open(cal):
    """Information published between 08:45 and 09:15 is real and deliberately
    unused: a model that consumes it cannot be run at 08:45 in production."""
    w = overnight_window(cal, date(2026, 8, 27))
    open_time = datetime(2026, 8, 27, 9, 15, tzinfo=IST).astimezone(UTC)
    assert w.end < open_time
    assert not w.contains(open_time)


def test_custom_cutoff_is_respected(cal):
    w = overnight_window(cal, date(2026, 8, 27), cutoff=dtime(8, 0))
    assert w.end.astimezone(IST).strftime("%H:%M") == "08:00"


def test_window_contains_is_right_open(cal):
    w = overnight_window(cal, date(2026, 8, 27))
    assert w.contains(w.start)
    assert not w.contains(w.end)


# --------------------------------------------------------------- provider
def test_synthetic_provider_is_deterministic(registry):
    p = SyntheticCrossAssetProvider(seed=11)
    series = registry.by_symbol("GIFT_NIFTY")
    start, end = ist(20, 0), ist(21, 0)
    a = p.fetch(series, start, end)
    b = p.fetch(series, start, end)
    assert a, "provider produced no bars"
    assert [c.model_dump_json(exclude={"trace_id"}) for c in a] == [
        c.model_dump_json(exclude={"trace_id"}) for c in b
    ]


def test_provider_respects_each_venue_hours(registry):
    p = SyntheticCrossAssetProvider()
    usdinr = registry.by_symbol("USDINR")  # 09:00-17:00 IST
    bars = p.fetch(usdinr, ist(20, 0), ist(23, 0))
    assert bars == [], "produced bars while the venue was closed"


def test_series_are_correlated_not_independent(registry, cal):
    """A generator of eight independent random walks would let a gap model pass
    its test while learning nothing, because there would be nothing to learn."""
    p = SyntheticCrossAssetProvider(seed=5)
    w = overnight_window(cal, date(2026, 8, 27))

    gift = p.fetch(registry.by_symbol("GIFT_NIFTY"), w.start, w.end)
    us = p.fetch(registry.by_symbol("US_EQ_FUT"), w.start, w.end)
    assert gift and us

    def returns(bars: list[Candle]) -> list[float]:
        return [b.c / b.o - 1.0 for b in bars]

    # Align on the overlapping minutes and check the sign agreement rate.
    us_by_ts = {b.ts: b for b in us}
    pairs = [(g, us_by_ts[g.ts]) for g in gift if g.ts in us_by_ts]
    assert len(pairs) > 60, "not enough overlapping minutes to test correlation"
    agree = sum(1 for g, u in pairs if (g.c - g.o) * (u.c - u.o) > 0)
    assert agree / len(pairs) > 0.6, "GIFT and US futures should move together"


def test_bars_carry_a_cross_asset_id(registry):
    p = SyntheticCrossAssetProvider()
    bars = p.fetch(registry.by_symbol("BRENT"), ist(15, 0), ist(16, 0))
    assert bars
    assert all(b.instrument_id >= CROSSASSET_ID_BASE for b in bars)
    assert all(b.tf == 1 for b in bars)


# --------------------------------------------------------- overnight moves
def test_moves_are_computed_per_series(registry, cal):
    p = SyntheticCrossAssetProvider(seed=7)
    w = overnight_window(cal, date(2026, 8, 27))
    bars = [b for s in registry.all() for b in p.fetch(s, w.start, w.end)]

    moves = overnight_moves(bars, w, registry)
    assert "GIFT_NIFTY" in moves and "US_EQ_FUT" in moves
    gift = moves["GIFT_NIFTY"]
    assert gift.bars > 100
    assert gift.high >= gift.low
    assert abs(gift.ret) < 0.2  # a sane overnight move


def test_bars_outside_the_window_are_excluded(registry, cal):
    """A single bar past the cutoff is enough to leak the open into the
    prediction."""
    p = SyntheticCrossAssetProvider(seed=7)
    w = overnight_window(cal, date(2026, 8, 27))
    inside = p.fetch(registry.by_symbol("GIFT_NIFTY"), w.start, w.end)
    after = p.fetch(registry.by_symbol("GIFT_NIFTY"), w.end, w.end + timedelta(minutes=30))
    assert inside and after

    only_inside = overnight_moves(inside, w, registry)["GIFT_NIFTY"]
    with_leak = overnight_moves([*inside, *after], w, registry)["GIFT_NIFTY"]
    assert only_inside.bars == with_leak.bars
    assert only_inside.last == with_leak.last


def test_unknown_instrument_ids_are_ignored(registry, cal):
    w = overnight_window(cal, date(2026, 8, 27))
    stray = Candle(
        ts=w.start + timedelta(minutes=1), instrument_id=99_999, tf=1, o=1.0, h=1.0, l=1.0, c=1.0
    )
    assert overnight_moves([stray], w, registry) == {}


# ------------------------------------------------------------- collector
def test_collector_deduplicates_repeated_bars(registry):
    """Providers routinely re-serve the most recent bar on every poll."""
    clock = FrozenClock(ist(22, 0))
    collector = CrossAssetCollector(registry, SyntheticCrossAssetProvider(seed=2), clock)
    start, end = ist(21, 0), ist(22, 0)

    first = collector.poll(start, end)
    second = collector.poll(start, end)
    assert first, "first poll returned nothing"
    assert second == [], "the same window was emitted twice"


def test_collector_only_polls_open_venues(registry):
    clock = FrozenClock(ist(23, 30))  # Indian FX closed, US futures open
    collector = CrossAssetCollector(registry, SyntheticCrossAssetProvider(), clock)
    bars = collector.poll(ist(23, 0), ist(23, 30))
    ids = {b.instrument_id for b in bars}
    assert registry.by_symbol("USDINR").instrument_id not in ids


def test_one_failing_venue_does_not_stop_the_others(registry):
    class Flaky(CrossAssetProvider):
        @property
        def name(self) -> str:
            return "flaky"

        def fetch(self, series, start, end):
            if series.symbol == "BRENT":
                raise ConnectionError("venue unreachable")
            return SyntheticCrossAssetProvider().fetch(series, start, end)

    clock = FrozenClock(ist(22, 0))
    collector = CrossAssetCollector(registry, Flaky(), clock)
    bars = collector.poll(ist(21, 0), ist(22, 0))
    assert bars, "one broken venue stopped the whole poll"
    assert registry.by_symbol("BRENT").instrument_id not in {b.instrument_id for b in bars}


def test_stale_series_are_reported(registry):
    clock = FrozenClock(ist(22, 0))
    collector = CrossAssetCollector(registry, SyntheticCrossAssetProvider(), clock)
    collector.poll(ist(21, 0), ist(22, 0))
    assert collector.stale_series(ist(22, 1)) == []

    much_later = ist(22, 0) + timedelta(hours=2)
    stale = collector.stale_series(much_later)
    assert stale, "a two-hour silence on an open venue should be reported"


def test_health_tracks_bar_counts(registry):
    clock = FrozenClock(ist(22, 0))
    collector = CrossAssetCollector(registry, SyntheticCrossAssetProvider(), clock)
    collector.poll(ist(21, 0), ist(22, 0))
    us = collector.health["US_EQ_FUT"]
    assert us.bars > 0 and us.last_bar_at is not None
