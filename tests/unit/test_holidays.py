"""M10 verify - holiday discovery, reconciliation and provenance.

No network. The NSE provider takes an injectable fetcher, so the parsing and
reconciliation logic is tested against fixed payloads.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from libs.ms_core.clock import CalendarNotLoadedError, FrozenClock, SessionCalendar
from libs.ms_data.holidays import (
    Confidence,
    Coverage,
    HolidayStore,
    InferredHolidayProvider,
    ManualHolidayProvider,
    NseHolidayProvider,
    Source,
    describe_day,
    next_holiday,
    observed_trading_days_from_recordings,
    parse_nse_date,
    parse_nse_payload,
    reconcile,
    staleness_days,
)
from tests.conftest import ist

NSE_PAYLOAD = {
    "CM": [
        {"tradingDate": "26-Jan-2026", "weekDay": "Monday", "description": "Republic Day"},
        {
            "tradingDate": "02-Oct-2026",
            "weekDay": "Friday",
            "description": "Mahatma Gandhi Jayanti",
        },
        {"tradingDate": "15-Aug-2026", "weekDay": "Saturday", "description": "Independence Day"},
    ],
    "FO": [
        {"tradingDate": "26-Jan-2026", "weekDay": "Monday", "description": "Republic Day"},
    ],
}


@pytest.fixture
def store(tmp_path):
    return HolidayStore(root=tmp_path / "holidays")


@pytest.fixture
def clock():
    return FrozenClock(ist(9, 0))


# --------------------------------------------------------------------- parsing
def test_parse_nse_date_formats():
    assert parse_nse_date("26-Jan-2026") == date(2026, 1, 26)
    assert parse_nse_date("  02-oct-2026 ") == date(2026, 10, 2)
    assert parse_nse_date("2026-01-26") is None
    assert parse_nse_date("31-Feb-2026") is None  # not a real date


def test_unparseable_date_is_skipped_not_guessed():
    """A date we cannot read must not become a date we invent."""
    payload = {"CM": [{"tradingDate": "garbage", "description": "X"}]}
    assert parse_nse_payload(payload) == []


def test_payload_parsing_keeps_the_reason():
    holidays = parse_nse_payload(NSE_PAYLOAD, segments=("CM",))
    assert len(holidays) == 3
    republic = next(h for h in holidays if h.day == date(2026, 1, 26))
    assert republic.reason == "Republic Day"
    assert republic.source is Source.NSE


def test_missing_description_is_labelled_not_blank():
    payload = {"CM": [{"tradingDate": "26-Jan-2026"}]}
    assert "no description" in parse_nse_payload(payload)[0].reason


def test_nse_provider_filters_by_year():
    provider = NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)
    assert len(provider.fetch(2026)) > 0
    assert provider.fetch(2025) == []


# ------------------------------------------------------------------- inference
def test_inference_finds_a_weekday_with_no_data():
    # Mon-Fri recorded except Wednesday
    observed = {date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 27), date(2026, 8, 28)}
    found = InferredHolidayProvider(observed).fetch(2026)
    assert [h.day for h in found] == [date(2026, 8, 26)]


def test_inference_never_claims_beyond_what_was_recorded():
    """Not running is not the same as the market being closed."""
    observed = {date(2026, 8, 24), date(2026, 8, 25)}
    found = InferredHolidayProvider(observed).fetch(2026)
    assert found == []  # nothing between the first and last recorded day


def test_inference_with_no_recordings_says_nothing():
    assert InferredHolidayProvider(set()).fetch(2026) == []


def test_inference_ignores_weekends():
    observed = {date(2026, 8, 24), date(2026, 8, 31)}  # two Mondays
    found = InferredHolidayProvider(observed).fetch(2026)
    assert all(h.day.weekday() < 5 for h in found)


# --------------------------------------------------------------- reconciliation
def test_two_sources_agreeing_is_confirmed():
    day = date(2026, 1, 26)
    inferred = InferredHolidayProvider({date(2026, 1, 23), date(2026, 1, 27), date(2026, 1, 28)})
    nse = NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)
    result = reconcile(
        {Source.NSE: nse.fetch(2026), Source.INFERRED: inferred.fetch(2026)},
        coverage={Source.NSE: nse.coverage(2026), Source.INFERRED: inferred.coverage(2026)},
    )
    republic = next(h for h in result if h.day == day)
    assert republic.confidence is Confidence.CONFIRMED
    assert set(republic.sources) == {Source.NSE, Source.INFERRED}


def test_disagreement_is_flagged_not_silently_resolved():
    """Two sources disagreeing about whether the market was open needs a human."""
    inferred = InferredHolidayProvider(
        # we recorded data on Republic Day, contradicting the published list
        {date(2026, 1, 23), date(2026, 1, 26), date(2026, 1, 27)}
    )
    nse = NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)
    result = reconcile(
        {Source.NSE: nse.fetch(2026), Source.INFERRED: inferred.fetch(2026)},
        coverage={Source.NSE: nse.coverage(2026), Source.INFERRED: inferred.coverage(2026)},
    )
    republic = next(h for h in result if h.day == date(2026, 1, 26))
    assert republic.confidence is Confidence.DISPUTED
    assert "verify" in republic.note


def test_manual_pin_wins_the_reason():
    day = date(2026, 1, 26)
    result = reconcile(
        {
            Source.MANUAL: ManualHolidayProvider({2026: [day]}).fetch(2026),
            Source.NSE: NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD).fetch(2026),
        }
    )
    republic = next(h for h in result if h.day == day)
    assert "config" in republic.reason
    assert republic.confidence is Confidence.CONFIRMED


def test_single_source_without_a_contradiction_is_not_disputed():
    nse = NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)
    result = reconcile({Source.NSE: nse.fetch(2026)}, coverage={Source.NSE: nse.coverage(2026)})
    assert all(h.confidence is Confidence.SINGLE for h in result)


def test_a_source_with_no_data_is_silent_not_disagreeing():
    """The bug this model exists to prevent: an empty config marking the entire
    published calendar as disputed."""
    nse = NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)
    empty_manual = ManualHolidayProvider({})
    no_recordings = InferredHolidayProvider(set())

    assert empty_manual.coverage(2026).covers(date(2026, 1, 26)) is False
    assert no_recordings.coverage(2026).covers(date(2026, 1, 26)) is False

    result = reconcile(
        {Source.NSE: nse.fetch(2026), Source.MANUAL: [], Source.INFERRED: []},
        coverage={
            Source.NSE: nse.coverage(2026),
            Source.MANUAL: empty_manual.coverage(2026),
            Source.INFERRED: no_recordings.coverage(2026),
        },
    )
    assert result
    assert not any(h.confidence is Confidence.DISPUTED for h in result)


def test_inference_coverage_stops_at_the_recorded_span():
    inferred = InferredHolidayProvider({date(2026, 8, 24), date(2026, 8, 28)})
    cov: Coverage = inferred.coverage(2026)
    assert cov.covers(date(2026, 8, 26)) is True  # inside the recorded span
    assert cov.covers(date(2026, 8, 31)) is False  # after we stopped recording
    assert cov.covers(date(2026, 1, 26)) is False  # long before we started


# ------------------------------------------------------------------- the store
def test_sync_persists_with_provenance(store, clock):
    entry = store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    assert store.path(2026).exists()
    reloaded = store.load(2026)
    assert reloaded is not None
    assert reloaded.dates() == entry.dates()
    assert reloaded.fetched_at == entry.fetched_at
    assert all(Source.NSE in h.sources for h in reloaded.holidays)


def test_sync_survives_one_failing_provider(store, clock):
    class Broken(NseHolidayProvider):
        def fetch(self, year):
            raise ConnectionError("nse unreachable")

    entry = store.sync(
        2026,
        clock,
        providers=[Broken(fetcher=lambda: {}), ManualHolidayProvider({2026: [date(2026, 1, 26)]})],
    )
    assert entry.dates() == {date(2026, 1, 26)}


def test_sync_raises_when_no_source_works(store, clock):
    class Broken(NseHolidayProvider):
        def fetch(self, year):
            raise ConnectionError("down")

    with pytest.raises(RuntimeError, match="no holiday source"):
        store.sync(2026, clock, providers=[Broken(fetcher=lambda: {})])


def test_strict_mode_propagates_the_failure(store, clock):
    class Broken(NseHolidayProvider):
        def fetch(self, year):
            raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        store.sync(2026, clock, providers=[Broken(fetcher=lambda: {})], strict=True)


def test_staleness_is_measurable(store, clock):
    entry = store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    assert staleness_days(entry, clock.now()) == 0
    assert staleness_days(entry, clock.now() + timedelta(days=100)) == 100


# --------------------------------------------------------------------- queries
def test_describe_day_gives_a_reason(store, clock):
    store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    assert "Republic Day" in describe_day(store, date(2026, 1, 26))
    assert "trading day" in describe_day(store, date(2026, 1, 27))
    assert "weekend" in describe_day(store, date(2026, 1, 25))  # Sunday


def test_describe_day_admits_when_it_does_not_know(store):
    msg = describe_day(store, date(2031, 3, 4))
    assert "no holiday calendar loaded" in msg
    assert "sync" in msg


def test_next_holiday_supports_a_day_ahead_warning(store, clock):
    store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    nxt = next_holiday(store, date(2026, 10, 1))
    assert nxt is not None
    assert nxt.day == date(2026, 10, 2)
    assert nxt.reason == "Mahatma Gandhi Jayanti"
    assert (nxt.day - date(2026, 10, 1)).days == 1  # "tomorrow is a holiday because X"


def test_observed_days_read_from_the_recorder_layout(tmp_path):
    root = tmp_path / "raw"
    for day in ("2026-08-24", "2026-08-25"):
        part = root / "ticks" / f"date={day}"
        part.mkdir(parents=True)
        (part / "ticks-1.ndjson").write_text('{"ts":"x"}\n', encoding="utf-8")
    empty = root / "ticks" / "date=2026-08-26"
    empty.mkdir(parents=True)
    (empty / "ticks-1.ndjson").write_text("", encoding="utf-8")  # started, captured nothing

    observed = observed_trading_days_from_recordings(root)
    assert observed == {date(2026, 8, 24), date(2026, 8, 25)}


# ------------------------------------------------------------------- calendar
def test_calendar_uses_the_discovered_dates(cfg, store, clock):
    store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    cal = SessionCalendar.from_config(cfg.sessions).with_holidays(store.as_calendar_dict())

    assert cal.is_trading_day(date(2026, 1, 26)) is False  # Republic Day
    assert cal.is_trading_day(date(2026, 1, 27)) is True
    with pytest.raises(CalendarNotLoadedError):
        cal.is_trading_day(date(2031, 3, 4))  # still refuses to guess


def test_config_pins_survive_a_discovered_calendar(cfg, store, clock):
    """A human who read the circular outranks a scraper - and is never dropped."""
    store.sync(2026, clock, providers=[NseHolidayProvider(fetcher=lambda: NSE_PAYLOAD)])
    base = SessionCalendar.from_config(cfg.sessions)
    pinned = base.with_holidays({2026: frozenset({date(2026, 3, 4)})})
    merged = pinned.with_holidays(store.as_calendar_dict())

    assert merged.is_trading_day(date(2026, 3, 4)) is False  # the pin
    assert merged.is_trading_day(date(2026, 1, 26)) is False  # the discovery
