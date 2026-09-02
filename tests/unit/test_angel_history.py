"""M11 - Angel's historical candle endpoint, exercised without the network.

The endpoint that finally supplies intraday history. Two things about it are
different from the websocket the rest of the adapter is built around, and both
are the kind that produce plausible numbers rather than errors:

- prices come back in **rupees**, where the websocket sends paise
- the range cap is enforced by **returning fewer bars**, not by failing
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from libs.ms_core import env as envmod
from libs.ms_core.clock import SimClock
from libs.ms_data.angelone import (
    HISTORY_TZ,
    AngelAuthError,
    AngelCredentials,
    AngelSession,
    CandleInterval,
    historical_candles,
    login,
    parse_candle_row,
    strip_bearer,
)

START = datetime(2026, 8, 28, 3, 45, tzinfo=UTC)  # 09:15 IST
ROW = ["2026-08-28T09:15:00+05:30", 24180.5, 24215.0, 24170.25, 24200.0, 125000]

# Pacing is real behaviour and is asserted once, below. Everywhere else it is
# switched off: a test suite that sleeps is a test suite people stop running.
NO_PACING = timedelta(0)


@pytest.fixture(autouse=True)
def _paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(envmod.ENV_VAR, "paper")


def _session() -> AngelSession:
    return AngelSession(
        client_code="A123456", jwt_token="JWT", refresh_token="REF", feed_token="FEED"
    )


def _ok(rows: list[list[object]]) -> dict[str, object]:
    return {"status": True, "message": "SUCCESS", "data": rows}


class TestRowParsing:
    def test_a_row_becomes_a_candle_at_the_left_edge(self) -> None:
        """Angel sends the bucket's start. Reading it as the right edge shifts
        every bar by one interval, which on a one-minute series is a whole bar
        of lookahead."""
        c = parse_candle_row(ROW, instrument_id=26000, tf_minutes=1)
        assert c.ts == datetime(2026, 8, 28, 3, 45, tzinfo=UTC)
        assert c.o == 24180.5
        assert c.h == 24215.0
        assert c.low == 24170.25
        assert c.c == 24200.0
        assert c.v == 125000
        assert c.tf == 1

    def test_prices_are_rupees_not_paise(self) -> None:
        """The websocket sends 2418050 for this price. The plausibility band
        cannot tell the two apart - 24,180 and 241.80 are both plausible - so
        this is pinned here and cross-checked against an independent daily
        series in jobs/angel_history.py."""
        c = parse_candle_row(ROW, instrument_id=26000, tf_minutes=1)
        assert 10_000 < c.o < 100_000, "a NIFTY level in rupees, not paise"

    def test_a_naive_timestamp_is_refused(self) -> None:
        naive = ["2026-08-28T09:15:00", 1.0, 2.0, 0.5, 1.5, 10]
        with pytest.raises(ValueError, match="no offset"):
            parse_candle_row(naive, instrument_id=1, tf_minutes=1)

    def test_the_offset_is_applied_rather_than_assumed(self) -> None:
        c = parse_candle_row(ROW, instrument_id=1, tf_minutes=1)
        assert c.ts.astimezone(HISTORY_TZ).hour == 9
        assert c.ts.hour == 3, "09:15 IST is 03:45 UTC, not 09:15 UTC"

    def test_an_inconsistent_candle_is_refused(self) -> None:
        """Open outside the range is not a bar. Accepting it puts a price in the
        series that the instrument never traded at."""
        bad = ["2026-08-28T09:15:00+05:30", 99.0, 50.0, 40.0, 45.0, 10]
        with pytest.raises(ValueError, match="not internally consistent"):
            parse_candle_row(bad, instrument_id=1, tf_minutes=1)

    def test_a_short_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 6"):
            parse_candle_row(["2026-08-28T09:15:00+05:30", 1.0], instrument_id=1, tf_minutes=1)


class TestIntervals:
    @pytest.mark.parametrize(
        ("interval", "minutes"),
        [
            (CandleInterval.ONE_MINUTE, 1),
            (CandleInterval.FIFTEEN_MINUTE, 15),
            (CandleInterval.ONE_HOUR, 60),
        ],
    )
    def test_minutes(self, interval: CandleInterval, minutes: int) -> None:
        assert interval.minutes == minutes

    def test_a_day_is_one_session_not_twenty_four_hours(self) -> None:
        """375 minutes, 09:15 to 15:30. Using 1440 would make every daily bar
        claim to span time the exchange was shut."""
        assert CandleInterval.ONE_DAY.minutes == 375

    def test_every_interval_declares_a_range_cap(self) -> None:
        for interval in CandleInterval:
            assert interval.max_days > 0


class TestFetching:
    def test_a_single_window_is_one_request(self) -> None:
        calls: list[dict[str, str]] = []

        def fetch(params: dict[str, str]) -> dict[str, object]:
            calls.append(params)
            return _ok([ROW])

        out = historical_candles(
            _session(),
            "key",
            token=26000,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,
            start=START,
            end=START + timedelta(hours=6),
            fetch=fetch,
            min_interval=NO_PACING,
        )
        assert len(calls) == 1
        assert len(out) == 1
        assert calls[0]["interval"] == "ONE_MINUTE"
        assert calls[0]["symboltoken"] == "26000"

    def test_a_long_range_is_split_at_the_cap(self) -> None:
        """Exceeding the cap returns a short series rather than an error, which
        reads as a quiet period and silently truncates a backfill."""
        calls: list[dict[str, str]] = []

        def fetch(params: dict[str, str]) -> dict[str, object]:
            calls.append(params)
            return _ok([])

        historical_candles(
            _session(),
            "key",
            token=26000,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,  # 30-day cap
            start=START,
            end=START + timedelta(days=95),
            fetch=fetch,
            min_interval=NO_PACING,
        )
        assert len(calls) == 4, "95 days at a 30-day cap is four requests"

    def test_the_request_window_is_sent_in_exchange_local_time(self) -> None:
        calls: list[dict[str, str]] = []

        def fetch(params: dict[str, str]) -> dict[str, object]:
            calls.append(params)
            return _ok([])

        historical_candles(
            _session(),
            "key",
            token=26000,
            exchange="NSE",
            interval=CandleInterval.ONE_DAY,
            start=START,
            end=START,
            fetch=fetch,
            min_interval=NO_PACING,
        )
        assert calls[0]["fromdate"].endswith("09:15"), "03:45 UTC is 09:15 IST"

    def test_a_bar_repeated_across_a_boundary_appears_once(self) -> None:
        """Two identical bars look like a real repeated minute to everything
        downstream."""

        def fetch(_params: dict[str, str]) -> dict[str, object]:
            return _ok([ROW, ROW])

        out = historical_candles(
            _session(),
            "key",
            token=26000,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,
            start=START,
            end=START + timedelta(days=95),
            fetch=fetch,
            min_interval=NO_PACING,
        )
        assert len(out) == 1

    def test_candles_come_back_in_time_order(self) -> None:
        rows = [
            ["2026-08-28T09:17:00+05:30", 3.0, 3.0, 3.0, 3.0, 1],
            ["2026-08-28T09:15:00+05:30", 1.0, 1.0, 1.0, 1.0, 1],
            ["2026-08-28T09:16:00+05:30", 2.0, 2.0, 2.0, 2.0, 1],
        ]
        out = historical_candles(
            _session(),
            "key",
            token=1,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,
            start=START,
            end=START + timedelta(hours=1),
            fetch=lambda _p: _ok(rows),
            min_interval=NO_PACING,
        )
        assert [c.o for c in out] == [1.0, 2.0, 3.0]

    def test_an_empty_response_is_not_an_error(self) -> None:
        """A holiday, a halt, or a range before the contract listed."""
        out = historical_candles(
            _session(),
            "key",
            token=1,
            exchange="NSE",
            interval=CandleInterval.ONE_DAY,
            start=START,
            end=START,
            fetch=lambda _p: _ok([]),
            min_interval=NO_PACING,
        )
        assert out == []

    def test_a_failed_response_names_the_range_it_failed_on(self) -> None:
        with pytest.raises(AngelAuthError, match="historical request failed"):
            historical_candles(
                _session(),
                "key",
                token=1,
                exchange="NSE",
                interval=CandleInterval.ONE_DAY,
                start=START,
                end=START,
                fetch=lambda _p: {"status": False, "message": "Invalid token"},
                min_interval=NO_PACING,
            )


class TestGuards:
    def test_end_before_start_is_refused(self) -> None:
        with pytest.raises(ValueError, match="precedes start"):
            historical_candles(
                _session(),
                "key",
                token=1,
                exchange="NSE",
                interval=CandleInterval.ONE_DAY,
                start=START,
                end=START - timedelta(days=1),
                fetch=lambda _p: _ok([]),
                min_interval=NO_PACING,
            )

    def test_an_unknown_exchange_is_refused(self) -> None:
        """A completely unknown name."""
        with pytest.raises(KeyError, match="unknown exchange"):
            historical_candles(
                _session(),
                "key",
                token=1,
                exchange="NSE_XX",
                interval=CandleInterval.ONE_DAY,
                start=START,
                end=START,
                fetch=lambda _p: _ok([]),
                min_interval=NO_PACING,
            )

    def test_the_websocket_vocabulary_is_refused_and_translated(self) -> None:
        """The defect this guard exists for, rather than a typo.

        `NSE_CM` and `NSE_FO` are real names - of the *websocket's* exchange
        types - and `historical_candles` used to validate against that map and
        then forward the name to a REST endpoint that has never heard of it.
        The endpoint's answer is an empty body, which is byte for byte what its
        throttle returns, so every historical call this repository made read as
        "we are being rate limited" and none of them ever fetched a bar.

        A name that is wrong must therefore not merely be refused: it has to say
        which vocabulary it belongs to, or the next reader re-derives all of it.
        """
        with pytest.raises(KeyError, match="websocket's name"):
            historical_candles(
                _session(),
                "key",
                token=1,
                exchange="NSE_FO",
                interval=CandleInterval.ONE_DAY,
                start=START,
                end=START,
                fetch=lambda _p: _ok([]),
                min_interval=NO_PACING,
            )

    def test_research_cannot_fetch_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Research being credential-free is the guarantee that research code
        cannot reach a venue, and history is a venue call like any other."""
        monkeypatch.setenv(envmod.ENV_VAR, "research")
        with pytest.raises(envmod.EnvironmentError_):
            historical_candles(
                _session(),
                "key",
                token=1,
                exchange="NSE",
                interval=CandleInterval.ONE_DAY,
                start=START,
                end=START,
                fetch=lambda _p: _ok([]),
                min_interval=NO_PACING,
            )


class TestBearerPrefix:
    """The defect that made every authenticated REST call fail.

    `smartapi-python` stores the raw token on the client and then rewrites the
    dict it returns:

        self.setAccessToken(jwtToken)
        user["data"]["jwtToken"] = "Bearer " + jwtToken

    So the value read out of the response is not the value the client is using.
    Storing the prefixed one and feeding it back through setAccessToken builds
    an `Authorization: Bearer Bearer eyJ...` header, and the server answers
    `Invalid Token` - which reads as an expired session and sends the
    investigation to the credentials rather than to the prefix.
    """

    def test_the_prefix_is_stripped(self) -> None:
        assert strip_bearer("Bearer eyJhbGciOi") == "eyJhbGciOi"

    def test_a_bare_token_is_left_alone(self) -> None:
        assert strip_bearer("eyJhbGciOi") == "eyJhbGciOi"

    def test_only_one_prefix_is_removed(self) -> None:
        """Defensive: a doubled prefix is a bug upstream, and silently eating
        both would hide it."""
        assert strip_bearer("Bearer Bearer eyJ") == "Bearer eyJ"

    def test_surrounding_whitespace_does_not_defeat_it(self) -> None:
        assert strip_bearer("  Bearer   eyJhbGciOi  ") == "eyJhbGciOi"

    def test_login_stores_the_bare_token(self) -> None:
        """The session is what every later call reads, so the stripping has to
        happen once at the source rather than at each use."""

        class _Client:
            def generateSession(self, *_a: object) -> dict[str, object]:
                return {
                    "status": True,
                    "data": {
                        "jwtToken": "Bearer eyJhbGciOi",
                        "refreshToken": "REF",
                        "feedToken": "FEED",
                    },
                }

        session = login(
            AngelCredentials("k", "A1", "1111", "BASE32SECRET"),
            SimClock(datetime(2026, 8, 30, 4, 0, tzinfo=UTC)),
            connect=lambda _k: _Client(),
            totp=lambda _s: "123456",
            cache=None,
        )
        assert session.jwt_token == "eyJhbGciOi"
        assert not session.jwt_token.startswith("Bearer")


class TestPacing:
    """Chunked requests are paced, because the block they trip is silent.

    Measured on 2026-08-30: after roughly four logins and five candle requests
    inside ninety seconds, every later request came back with an **empty body**
    - no status, no message - and stayed that way for at least half an hour. The
    vendor SDK reports that as "Couldn't parse the JSON response", which reads
    as a broken response format rather than as throttling.
    """

    def test_consecutive_windows_are_spaced(self) -> None:
        stamps: list[float] = []

        def fetch(_params: dict[str, str]) -> dict[str, object]:
            stamps.append(time.monotonic())
            return _ok([])

        historical_candles(
            _session(),
            "key",
            token=1,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,  # 30-day cap
            start=START,
            end=START + timedelta(days=65),  # three windows
            fetch=fetch,
            min_interval=timedelta(seconds=0.05),
        )
        assert len(stamps) == 3
        gaps = [b - a for a, b in pairwise(stamps)]
        assert all(g >= 0.04 for g in gaps), f"windows were not spaced: {gaps}"

    def test_pacing_can_be_switched_off(self) -> None:
        """A caller with its own limiter, and every test in this file."""
        started = time.monotonic()
        historical_candles(
            _session(),
            "key",
            token=1,
            exchange="NSE",
            interval=CandleInterval.ONE_MINUTE,
            start=START,
            end=START + timedelta(days=95),
            fetch=lambda _p: _ok([]),
            min_interval=NO_PACING,
        )
        assert time.monotonic() - started < 0.5
