"""Daily bar loader - parsing, caching, and the overnight leak guard."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from libs.ms_data.daily import (
    DailyBar,
    DailyBarProvider,
    DailyStore,
    approximate_window_hours,
    build_overnight_series,
    coverage_report,
    parse_yahoo,
    session_span,
)


def bars(n: int, *, start: date = date(2026, 1, 1), price: float = 100.0, step: float = 0.5):
    out, p = [], price
    day = start
    for _ in range(n):
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        o = p
        p = p + step
        out.append(
            DailyBar(
                day=day,
                o=round(o, 2),
                h=round(max(o, p) + 1, 2),
                low=round(min(o, p) - 1, 2),
                c=round(p, 2),
                v=1000,
            )
        )
        day += timedelta(days=1)
    return out


def yahoo_payload(days: int = 5, *, with_nulls: bool = False):
    base = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
    stamps = [base + i * 86400 for i in range(days)]
    opens = [100.0 + i for i in range(days)]
    closes = [101.0 + i for i in range(days)]
    if with_nulls:
        closes[2] = None
    return {
        "chart": {
            "result": [
                {
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": [o + 2 for o in opens],
                                "low": [o - 2 for o in opens],
                                "close": closes,
                                "volume": [1000] * days,
                            }
                        ]
                    },
                    "meta": {},
                }
            ]
        }
    }


# ------------------------------------------------------------------ parsing
def test_parses_a_payload():
    parsed = parse_yahoo(yahoo_payload(5))
    assert len(parsed) == 5
    assert parsed == sorted(parsed, key=lambda b: b.day)
    assert parsed[0].o == 100.0


def test_null_rows_are_dropped_not_filled():
    """Carrying a null close forward would invent a price. The caller sees a
    shorter series rather than a smooth one."""
    parsed = parse_yahoo(yahoo_payload(5, with_nulls=True))
    assert len(parsed) == 4


def test_empty_payload_is_not_an_error():
    assert parse_yahoo({}) == []
    assert parse_yahoo({"chart": {"result": None}}) == []


def test_bar_derived_values():
    b = DailyBar(day=date(2026, 1, 1), o=100.0, h=105.0, low=99.0, c=102.0)
    assert b.ret == pytest.approx(0.02)
    assert b.range_pct == pytest.approx(0.06)


# -------------------------------------------------------------------- cache
class FakeProvider(DailyBarProvider):
    def __init__(self, series):
        self.series = series
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def fetch(self, symbol, *, years=2):
        self.calls += 1
        return self.series


def test_cache_avoids_refetching(tmp_path):
    """Re-fetching a two-year history on every run is slow and rude to the source."""
    provider = FakeProvider(bars(30))
    store = DailyStore(root=tmp_path, provider=provider)

    first = store.get("X")
    second = store.get("X")
    assert first and len(first) == len(second)
    assert provider.calls == 1, "the second call went back to the source"


def test_refresh_forces_a_refetch(tmp_path):
    provider = FakeProvider(bars(30))
    store = DailyStore(root=tmp_path, provider=provider)
    store.get("X")
    store.get("X", refresh=True)
    assert provider.calls == 2


def test_cache_round_trips_exactly(tmp_path):
    original = bars(20)
    store = DailyStore(root=tmp_path, provider=FakeProvider(original))
    store.get("X")
    reloaded = store.load("X")
    assert [(b.day, b.o, b.c) for b in reloaded] == [(b.day, b.o, b.c) for b in original]


def test_store_lists_symbols(tmp_path):
    store = DailyStore(root=tmp_path, provider=FakeProvider(bars(10)))
    store.get("A")
    store.get("B")
    assert store.symbols() == ["A", "B"]


# ------------------------------------------------------- the overnight guard
def test_foreign_series_never_contributes_the_target_session():
    """The US session on day D closes long after Nifty has opened on D. Using it
    would be reading tomorrow's newspaper."""
    index = bars(20)
    foreign = bars(20, price=200.0, step=2.0)
    series = build_overnight_series(index, {"US": foreign})

    by_day = {b.day: b for b in foreign}
    assert series
    for entry in series:
        # Whatever was used must be dated strictly before the session predicted.
        used = [d for d in by_day if d < entry.session]
        assert used, "no prior foreign session available"
        latest_allowed = max(used)
        assert by_day[latest_allowed].ret == pytest.approx(entry.foreign_returns["US"])
        assert latest_allowed < entry.session


def test_gap_is_open_against_previous_close():
    index = [
        DailyBar(day=date(2026, 1, 5), o=100.0, h=101.0, low=99.0, c=100.0),
        DailyBar(day=date(2026, 1, 6), o=102.0, h=103.0, low=101.0, c=102.5),
    ]
    series = build_overnight_series(index, {})
    assert len(series) == 1
    assert series[0].gap == pytest.approx(0.02)
    assert series[0].session == date(2026, 1, 6)
    assert series[0].previous_session == date(2026, 1, 5)


def test_weekend_gap_is_flagged_and_wider():
    friday = DailyBar(day=date(2026, 1, 2), o=100.0, h=101.0, low=99.0, c=100.0)
    monday = DailyBar(day=date(2026, 1, 5), o=101.0, h=102.0, low=100.0, c=101.0)
    tuesday = DailyBar(day=date(2026, 1, 6), o=101.5, h=102.0, low=101.0, c=101.5)

    series = build_overnight_series([friday, monday, tuesday], {})
    weekend, normal = series[0], series[1]
    assert weekend.spans_multiple_days and weekend.calendar_days == 3
    assert not normal.spans_multiple_days
    assert approximate_window_hours(weekend) > approximate_window_hours(normal)


def test_window_hours_matches_the_session_clock():
    """Close at 15:30, cutoff at 08:45 the next session."""
    normal = build_overnight_series(
        [
            DailyBar(day=date(2026, 1, 5), o=100.0, h=101.0, low=99.0, c=100.0),
            DailyBar(day=date(2026, 1, 6), o=100.0, h=101.0, low=99.0, c=100.0),
        ],
        {},
    )[0]
    assert approximate_window_hours(normal) == pytest.approx(17.25)


def test_missing_foreign_series_is_absent_not_zero():
    """A series present on half the sessions is not a feature, it is a coin flip
    with extra steps."""
    index = bars(20)
    late = bars(5, start=date(2026, 1, 20), price=50.0)
    series = build_overnight_series(index, {"LATE": late})
    early = [e for e in series if e.session < date(2026, 1, 20)]
    assert early
    assert all("LATE" not in e.foreign_returns for e in early)


def test_coverage_report_surfaces_thin_series():
    index = bars(20)
    full = bars(20, price=200.0)
    partial = bars(5, start=date(2026, 1, 15), price=300.0)
    series = build_overnight_series(index, {"FULL": full, "THIN": partial})

    coverage = coverage_report(series, ["FULL", "THIN", "ABSENT"])
    assert coverage["FULL"] > 0.9
    assert coverage["THIN"] < 0.5
    assert coverage["ABSENT"] == 0.0


def test_coverage_of_nothing_is_empty():
    assert coverage_report([], ["A"]) == {}


def test_session_span():
    series = build_overnight_series(bars(10), {})
    span = session_span(series)
    assert span is not None and span[0] < span[1]
    assert session_span([]) is None


def test_single_session_produces_no_windows():
    """One bar has no previous close to gap from."""
    assert build_overnight_series(bars(1), {}) == []
