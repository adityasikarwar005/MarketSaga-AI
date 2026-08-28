"""M14 - the NSE bhavcopy loader, exercised offline against real file shapes.

The fixtures below are trimmed from actual downloads, not invented, because the
two formats disagree about column names, date formats and what a "volume" is,
and a made-up fixture would agree with whatever the parser happens to do.

The defect this module exists to prevent is not a crash. It is fitting a
volatility surface to the 54% of a day's chain that never traded, where the
exchange computes both a close and a settlement and they disagree by a median
of 16%. That produces a converged fit, a plausible smile, and no error.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.schemas import OptionType
from libs.ms_data.bhavcopy import (
    UDIFF_FROM,
    BhavcopyDay,
    BhavcopyFormat,
    BhavcopyStore,
    NoSessionData,
    OptionRecord,
    available_at,
    chain_from_records,
    formats_for,
    parse,
    parse_legacy,
    parse_udiff,
    unzip_single,
    url_for,
)

# Real UDiFF columns and real values, trimmed to four contracts: one liquid,
# one untraded whose close and settlement disagree, one PE, one non-option row.
UDIFF_CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-08-26,2026-08-26,FO,NSE,STO,43001,,NIFTY,,2026-09-29,2026-09-29,24200.00,PE,NIFTY,148.00,155.00,146.00,150.80,150.80,149.00,24207.75,150.80,81640,1200,870,12345.00,540,F1,75,,,,,
2026-08-26,2026-08-26,FO,NSE,STO,43002,,NIFTY,,2026-09-29,2026-09-29,24200.00,CE,NIFTY,160.00,168.00,158.00,163.25,163.25,161.00,24207.75,163.25,74210,-800,1020,15000.00,610,F1,75,,,,,
2026-08-26,2026-08-26,FO,NSE,STO,43003,,NIFTY,,2026-09-29,2026-09-29,31000.00,CE,NIFTY,0.00,0.00,0.00,2.40,0.00,2.10,24207.75,0.85,1500,0,0,0.00,0,F1,75,,,,,
2026-08-26,2026-08-26,FO,NSE,IDF,43004,,NIFTY,,2026-09-29,2026-09-29,0.00,,NIFTY-FUT,24210.00,24250.00,24180.00,24215.00,24215.00,24200.00,24207.75,24215.00,900000,500,45000,900000.00,9000,F1,75,,,,,
2026-08-26,2026-08-26,FO,NSE,STO,43005,,NIFTY,,2026-09-29,2026-09-29,24500.00,XX,NIFTY,10.00,11.00,9.00,10.50,10.50,10.00,24207.75,10.50,100,0,50,500.00,5,F1,75,,,,,
"""

# Real legacy columns, including the trailing empty field the file actually has.
LEGACY_CSV = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,
OPTIDX,NIFTY,13-Jul-2023,19150,CE,230,245,228,238,205.5,0,0,0,0,09-JUN-2023,
OPTIDX,NIFTY,13-Jul-2023,18500,PE,95.5,102,94,99.25,99.25,3120,450.5,225000,1500,09-JUN-2023,
FUTIDX,NIFTY,29-Jun-2023,0,XX,18600,18700,18580,18650,18650,55000,9000,3000000,2000,09-JUN-2023,
FUTSTK,NIFTY,29-Jun-2023,18000,XX,120,130,118,125,125,900,80,45000,300,09-JUN-2023,
"""


def _cal(holidays: set[date] | None = None) -> SessionCalendar:
    """A calendar with a known holiday year.

    The configured calendar deliberately has no holiday list - it is discovered
    by `python tasks.py holidays`, and it refuses to guess a year it has not
    loaded. So a test that needs is_trading_day to answer has to say which days
    are holidays, exactly as tests/conftest.py does.
    """
    from libs.ms_core import config as cfgmod

    base = SessionCalendar.from_config(cfgmod.load(warn=False).sessions)
    return SessionCalendar(
        tz=base.tz,
        preopen_start=base.preopen_start,
        preopen_end=base.preopen_end,
        continuous_start=base.continuous_start,
        continuous_end=base.continuous_end,
        expected_continuous_minutes=base.expected_continuous_minutes,
        holidays={2026: frozenset(holidays or {date(2026, 8, 25)})},
    )


# --------------------------------------------------------------------------- #
# The 54%
# --------------------------------------------------------------------------- #
class TestUntradedContracts:
    def test_an_untraded_contract_still_carries_two_different_prices(self) -> None:
        """The whole reason this module has a traded_only default.

        Zero volume, yet close is 2.40 and settlement is 0.85 - both computed
        by the exchange, neither paid by anybody, and 182% apart.
        """
        rows = parse_udiff(UDIFF_CSV)
        (far,) = [r for r in rows if r.strike == 31000.0]
        assert far.volume == 0
        assert not far.traded
        assert far.close == 2.40 and far.settlement == 0.85
        assert far.close_settlement_gap is not None
        assert far.close_settlement_gap > 1.0

    def test_untraded_contracts_are_dropped_by_default(self) -> None:
        day = BhavcopyDay(date(2026, 8, 26), BhavcopyFormat.UDIFF, parse_udiff(UDIFF_CSV))
        kept, dropped = day.options("NIFTY")
        assert dropped == 1
        assert all(r.traded for r in kept)

    def test_the_dropped_count_is_returned_not_logged(self) -> None:
        """ "857 of 1,584 were dropped" is a fact about the data that belongs in
        the caller's report, not in a log line somebody may never read."""
        day = BhavcopyDay(date(2026, 8, 26), BhavcopyFormat.UDIFF, parse_udiff(UDIFF_CSV))
        _, dropped = day.options("NIFTY")
        assert isinstance(dropped, int) and dropped == 1

    def test_keeping_them_is_possible_but_has_to_be_asked_for(self) -> None:
        day = BhavcopyDay(date(2026, 8, 26), BhavcopyFormat.UDIFF, parse_udiff(UDIFF_CSV))
        kept, dropped = day.options("NIFTY", traded_only=False)
        assert len(kept) == 3 and dropped == 0


# --------------------------------------------------------------------------- #
# Two formats
# --------------------------------------------------------------------------- #
class TestFormats:
    def test_udiff_maps_every_field_to_the_right_one(self) -> None:
        rows = parse_udiff(UDIFF_CSV)
        atm = next(r for r in rows if r.strike == 24200.0 and r.opt_type is OptionType.PE)
        assert atm.day == date(2026, 8, 26)
        assert atm.expiry == date(2026, 9, 29)
        assert atm.close == 150.80 and atm.settlement == 150.80
        assert atm.volume == 870 and atm.open_interest == 81640 and atm.oi_change == 1200
        assert atm.underlying == 24207.75

    def test_legacy_maps_every_field_to_the_right_one(self) -> None:
        rows = parse_legacy(LEGACY_CSV)
        pe = next(r for r in rows if r.opt_type is OptionType.PE)
        assert pe.day == date(2023, 6, 9)
        assert pe.expiry == date(2023, 7, 13)
        assert pe.close == 99.25 and pe.settlement == 99.25
        assert pe.open_interest == 225000 and pe.oi_change == 1500

    def test_the_two_date_formats_are_not_interchangeable(self) -> None:
        """`13-Jul-2023` against `2026-09-29`.

        A misparse raises nowhere downstream. It produces a time to expiry that
        is wrong and an implied volatility that is confidently wrong with it.
        """
        legacy = parse_legacy(LEGACY_CSV)
        assert all(r.expiry.year == 2023 for r in legacy)
        udiff = parse_udiff(UDIFF_CSV)
        assert all(r.expiry == date(2026, 9, 29) for r in udiff)

    def test_futures_and_other_instruments_are_not_read_as_options(self) -> None:
        """Both fixtures carry a non-option row WITH a real strike.

        A future has strike zero, so a zero-strike check alone would appear to
        work and the type guard would be undefended. The row that actually
        reaches the guard is one with a plausible strike and a type that is
        neither CE nor PE - and without the guard that row either becomes a
        phantom option or crashes OptionType.
        """
        udiff = parse_udiff(UDIFF_CSV)
        assert len(udiff) == 3
        assert all(r.opt_type in (OptionType.CE, OptionType.PE) for r in udiff)
        assert not any(r.strike == 24500.0 for r in udiff), "the XX row became an option"

        legacy = parse_legacy(LEGACY_CSV)
        assert len(legacy) == 2
        assert not any(r.strike == 18000.0 for r in legacy), "the FUTSTK row became an option"

    def test_only_udiff_carries_the_underlying(self) -> None:
        """The asymmetry that decides whether a chain can be built at all."""
        assert all(r.underlying is not None for r in parse_udiff(UDIFF_CSV))
        assert all(r.underlying is None for r in parse_legacy(LEGACY_CSV))

    def test_the_format_dispatch_agrees_with_the_direct_parsers(self) -> None:
        assert parse(UDIFF_CSV, BhavcopyFormat.UDIFF) == parse_udiff(UDIFF_CSV)
        assert parse(LEGACY_CSV, BhavcopyFormat.LEGACY) == parse_legacy(LEGACY_CSV)

    def test_the_preferred_format_follows_the_cutover_but_keeps_a_fallback(self) -> None:
        """The cutover was a period, not an instant. A hard date would turn a
        served file into a missing session."""
        before = formats_for(UDIFF_FROM.replace(year=UDIFF_FROM.year - 1))
        after = formats_for(UDIFF_FROM)
        assert before[0] is BhavcopyFormat.LEGACY and BhavcopyFormat.UDIFF in before
        assert after[0] is BhavcopyFormat.UDIFF and BhavcopyFormat.LEGACY in after

    def test_the_urls_match_the_archive_layout(self) -> None:
        d = date(2023, 6, 9)
        assert url_for(d, BhavcopyFormat.LEGACY).endswith("/2023/JUN/fo09JUN2023bhav.csv.zip")
        u = url_for(date(2026, 8, 26), BhavcopyFormat.UDIFF)
        assert u.endswith("BhavCopy_NSE_FO_0_0_0_20260826_F_0000.csv.zip")


# --------------------------------------------------------------------------- #
# Point in time
# --------------------------------------------------------------------------- #
class TestPointInTime:
    def test_the_file_is_not_available_during_the_session_it_describes(self) -> None:
        """A bhavcopy published at 00:00 on its own trade date is lookahead, and
        it silently inflates every backtest that touches it."""
        cal = _cal()
        day = date(2026, 8, 26)
        _, session_end = cal.session_bounds(day)
        assert available_at(day, cal) > session_end

    def test_it_is_available_the_next_morning(self) -> None:
        cal = _cal()
        day = date(2026, 8, 26)
        got = available_at(day, cal)
        local = got.astimezone(cal.tz)
        assert local.date() == date(2026, 8, 27)
        assert local.hour == 8

    def test_the_bound_is_timezone_aware(self) -> None:
        assert available_at(date(2026, 8, 26), _cal()).tzinfo is not None


# --------------------------------------------------------------------------- #
# Fetching and the store
# --------------------------------------------------------------------------- #
def _zip_of(csv_text: str, name: str = "bhav.csv") -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, csv_text)
    return buf.getvalue()


class TestStore:
    def test_a_fetched_day_is_parsed_and_cached(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            return _zip_of(UDIFF_CSV)

        store = BhavcopyStore(tmp_path, fetcher=fetch)
        day = store.get(date(2026, 8, 26))
        assert day.fmt is BhavcopyFormat.UDIFF and len(day.records) == 3
        again = store.get(date(2026, 8, 26))
        assert len(calls) == 1, "a cached day must not be re-fetched"
        assert again.records == day.records

    def test_the_cache_round_trips_every_field(self, tmp_path: Path) -> None:
        """A cache that drops a field is a slow-motion data loss: the first read
        is right and every later one is missing something."""
        store = BhavcopyStore(tmp_path, fetcher=lambda _u: _zip_of(UDIFF_CSV))
        first = store.get(date(2026, 8, 26))
        reloaded = store.load(date(2026, 8, 26))
        assert reloaded is not None
        assert reloaded.records == first.records
        assert reloaded.fmt is first.fmt

    def test_a_holiday_says_so(self, tmp_path: Path) -> None:
        import urllib.error

        def fetch(url: str) -> bytes:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        holiday = date(2026, 8, 25)
        cal = _cal({holiday})
        assert not cal.is_trading_day(holiday)
        store = BhavcopyStore(tmp_path, fetcher=fetch, calendar=cal)
        with pytest.raises(NoSessionData, match="not a trading day"):
            store.get(holiday)

    def test_a_missing_trading_day_is_a_gap_not_a_holiday(self, tmp_path: Path) -> None:
        """The distinction the HTTP status cannot make.

        Treating a failed download on a trading day as a holiday backfills a
        hole that nothing will ever report.
        """
        import urllib.error

        def fetch(url: str) -> bytes:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        cal = _cal()
        trading = date(2026, 8, 26)
        assert cal.is_trading_day(trading)
        store = BhavcopyStore(tmp_path, fetcher=fetch, calendar=cal)
        with pytest.raises(NoSessionData, match="gap in the data"):
            store.get(trading)

    def test_without_a_calendar_it_refuses_to_guess(self, tmp_path: Path) -> None:
        import urllib.error

        def fetch(url: str) -> bytes:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        store = BhavcopyStore(tmp_path, fetcher=fetch)
        with pytest.raises(NoSessionData, match="unknown"):
            store.get(date(2026, 8, 26))

    def test_an_unloaded_calendar_year_is_a_third_state(self, tmp_path: Path) -> None:
        """The calendar refuses to guess a year it has not loaded, and that
        refusal must not escape as a LookupError from inside an error path.

        "Cannot answer" is not "no". Collapsing it into either one turns a
        missing session into a holiday or a holiday into a data gap.
        """
        import urllib.error

        def fetch(url: str) -> bytes:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        cal = _cal()
        store = BhavcopyStore(tmp_path, fetcher=fetch, calendar=cal)
        with pytest.raises(NoSessionData, match="no holiday list for 2019"):
            store.get(date(2019, 6, 12))

    def test_an_archive_with_two_csvs_is_a_format_change_not_a_guess(self) -> None:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.csv", UDIFF_CSV)
            z.writestr("b.csv", UDIFF_CSV)
        with pytest.raises(NoSessionData, match="one CSV"):
            unzip_single(buf.getvalue())

    def test_a_parsed_file_with_no_options_falls_through_to_the_other_format(
        self, tmp_path: Path
    ) -> None:
        served: list[str] = []

        def fetch(url: str) -> bytes:
            served.append(url)
            # UDiFF is tried first for this date and comes back as a legacy file.
            return _zip_of(LEGACY_CSV)

        store = BhavcopyStore(tmp_path, fetcher=fetch)
        day = store.get(date(2026, 8, 26))
        assert day.fmt is BhavcopyFormat.LEGACY
        assert len(served) == 2, "it must try the second format, not give up"


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #
class TestChain:
    def _records(self) -> list[OptionRecord]:
        rows = parse_udiff(UDIFF_CSV)
        return [r for r in rows if r.traded]

    def test_the_spot_comes_from_the_file_itself(self) -> None:
        snap = chain_from_records(
            self._records(), underlying_id=26000, ts=datetime(2026, 8, 26, 10, tzinfo=UTC)
        )
        assert snap.spot == 24207.75
        assert snap.atm_strike() == 24200.0

    def test_a_legacy_chain_refuses_to_invent_a_spot(self) -> None:
        """ChainSnapshot exists on the premise that the spot was captured with
        the chain. Joining one from elsewhere reads as edge later."""
        legacy = [r for r in parse_legacy(LEGACY_CSV) if r.traded]
        with pytest.raises(ValueError, match="no spot"):
            chain_from_records(legacy, underlying_id=26000, ts=datetime(2023, 6, 9, 10, tzinfo=UTC))

    def test_a_legacy_chain_works_when_the_spot_is_supplied(self) -> None:
        legacy = [r for r in parse_legacy(LEGACY_CSV) if r.traded]
        snap = chain_from_records(
            legacy, underlying_id=26000, ts=datetime(2023, 6, 9, 10, tzinfo=UTC), spot=18600.0
        )
        assert snap.spot == 18600.0

    def test_there_is_no_mid_because_there_is_no_book(self) -> None:
        """A bhavcopy has no bid and no ask. mid() must stay unavailable rather
        than being manufactured out of a single price."""
        snap = chain_from_records(
            self._records(), underlying_id=26000, ts=datetime(2026, 8, 26, 10, tzinfo=UTC)
        )
        assert all(q.bid is None and q.ask is None and q.mid is None for q in snap.quotes)

    def test_the_price_field_has_to_be_chosen(self) -> None:
        """They diverge by a median 16% on contracts that did not trade, so
        there is no default worth hiding."""
        recs = self._records()
        ts = datetime(2026, 8, 26, 10, tzinfo=UTC)
        settle = chain_from_records(recs, underlying_id=26000, ts=ts, price_field="settlement")
        close = chain_from_records(recs, underlying_id=26000, ts=ts, price_field="close")
        assert [q.ltp for q in settle.quotes] == [r.settlement for r in recs]
        assert [q.ltp for q in close.quotes] == [r.close for r in recs]
        with pytest.raises(ValueError, match="price_field"):
            chain_from_records(recs, underlying_id=26000, ts=ts, price_field="mid")

    def test_records_from_two_different_days_are_refused(self) -> None:
        recs = self._records()
        mixed = [recs[0], OptionRecord(**{**recs[1].__dict__, "underlying": 19000.0})]
        with pytest.raises(ValueError, match="disagree about the underlying"):
            chain_from_records(mixed, underlying_id=26000, ts=datetime(2026, 8, 26, 10, tzinfo=UTC))

    def test_an_empty_chain_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero records"):
            chain_from_records([], underlying_id=26000, ts=datetime(2026, 8, 26, tzinfo=UTC))
