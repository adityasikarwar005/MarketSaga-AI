"""M11 - the Angel One adapter, exercised without a broker account.

The tests that matter here are not "does it connect". They are the four ways
this adapter can produce a working pipeline, plausible numbers and no error:

- prices decoded in paise and used as rupees
- the JWT passed where the feed token belongs
- a token resolved without a date, so a backfill gets this week's contract
- a decode failure swallowed on the socket thread, so the feed goes quiet

Each has a test that fails if the guard is removed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from itertools import islice
from pathlib import Path
from typing import ClassVar

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_core import env as envmod
from libs.ms_core.clock import SimClock
from libs.ms_core.schemas import OptionType
from libs.ms_data.angelone import (
    ANGEL_MAX_TOKENS_PER_REQUEST,
    EXCHANGE_TYPE,
    MODE_SNAP_QUOTE,
    AngelAuthError,
    AngelCredentials,
    AngelOneFeed,
    AngelSession,
    InstrumentMaster,
    PriceScaleError,
    ScripMasterError,
    fetch_scrip_master,
    load_scrip_master,
    login,
    paise_to_rupees,
    strike_to_rupees,
    tick_from_message,
)


@pytest.fixture(autouse=True)
def _paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Research may not hold broker credentials, by construction.

    Every constructor here calls require_broker_credentials, so the tests run in
    paper. That the guard exists is asserted separately below.
    """
    monkeypatch.setenv(envmod.ENV_VAR, "paper")


GOOD = {
    "ANGEL_API_KEY": "abc123",
    "ANGEL_CLIENT_CODE": "A123456",
    "ANGEL_PIN": "9087",
    "ANGEL_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
}


def _session(**over: object) -> AngelSession:
    base = {
        "client_code": "A123456",
        "jwt_token": "JWT",
        "refresh_token": "REFRESH",
        "feed_token": "FEED",
        "issued_on": date(2026, 8, 28),
    }
    base.update(over)
    return AngelSession(**base)  # type: ignore[arg-type]


def _snap(**over: object) -> dict[str, object]:
    msg: dict[str, object] = {
        "token": "26000",
        "exchange_timestamp": 1_756_000_000_000,
        "last_traded_price": 2_418_075,  # paise
        "last_traded_quantity": 75,
        "best_5_buy_data": [{"price": 2_418_050, "quantity": 150}],
        "best_5_sell_data": [{"price": 2_418_100, "quantity": 225}],
    }
    msg.update(over)
    return msg


# --------------------------------------------------------------------------- #
# The 100x
# --------------------------------------------------------------------------- #
class TestPaiseScaling:
    def test_the_wire_value_is_paise(self) -> None:
        assert paise_to_rupees(2_418_075) == pytest.approx(24180.75)

    def test_absent_stays_absent(self) -> None:
        """A missing bid is not a zero bid.

        Zero makes a one-sided book look two-sided and puts the mid halfway to
        nothing, which is a price the option was never offered at.
        """
        assert paise_to_rupees(None) is None

    def test_already_in_rupees_is_refused(self) -> None:
        """The mistake this whole function exists to catch.

        24180.75 divided a second time is 241.81 - a plausible option premium,
        internally consistent with every downstream check, and wrong by 100x.
        """
        with pytest.raises(PriceScaleError, match="plausible"):
            paise_to_rupees(0.5)  # 0.005 rupees: below any real tick

    def test_a_far_otm_option_at_five_paise_survives(self) -> None:
        """The sanity band must not reject real cheap options."""
        assert paise_to_rupees(5) == pytest.approx(0.05)

    def test_an_exact_zero_is_allowed_through(self) -> None:
        """Zero is a real value in a depth book, not a scaling error."""
        assert paise_to_rupees(0) == 0.0


class TestTickDecoding:
    def test_prices_land_in_rupees(self) -> None:
        t = tick_from_message(_snap())
        assert t.ltp == pytest.approx(24180.75)
        assert t.bid == pytest.approx(24180.50)
        assert t.ask == pytest.approx(24181.00)
        assert t.mid == pytest.approx(24180.75)

    def test_timestamp_is_the_exchange_one_not_arrival(self) -> None:
        t = tick_from_message(_snap(exchange_timestamp=1_756_000_000_000))
        assert t.ts == datetime.fromtimestamp(1_756_000_000, tz=UTC)
        assert t.ts.tzinfo is not None

    def test_a_payload_without_an_exchange_timestamp_is_refused(self) -> None:
        """Substituting arrival time folds network latency into t_event.

        Everything point-in-time downstream reads t_event literally, so a
        plausible-looking substitute is worse than a hard failure.
        """
        msg = _snap()
        del msg["exchange_timestamp"]
        with pytest.raises(KeyError, match="exchange_timestamp"):
            tick_from_message(msg)

    def test_sequence_number_is_none_because_the_vendor_sends_none(self) -> None:
        assert tick_from_message(_snap()).seq is None

    def test_a_one_sided_book_has_no_mid(self) -> None:
        t = tick_from_message(_snap(best_5_sell_data=[]))
        assert t.ask is None
        assert t.mid is None


# --------------------------------------------------------------------------- #
# Credentials and session
# --------------------------------------------------------------------------- #
class TestCredentials:
    def test_research_cannot_hold_them(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envmod.ENV_VAR, "research")
        with pytest.raises(envmod.EnvironmentError_):
            AngelCredentials.from_env(GOOD)

    def test_missing_fields_are_named(self) -> None:
        with pytest.raises(AngelAuthError, match="ANGEL_PIN"):
            AngelCredentials.from_env({**GOOD, "ANGEL_PIN": ""})

    def test_a_six_digit_totp_code_is_rejected(self) -> None:
        """The mistake that works once and then fails every morning after."""
        with pytest.raises(AngelAuthError, match="generated code"):
            AngelCredentials.from_env({**GOOD, "ANGEL_TOTP_SECRET": "483920"})

    def test_repr_does_not_leak(self) -> None:
        c = AngelCredentials.from_env(GOOD)
        text = repr(c)
        assert "JBSWY3DPEHPK3PXP" not in text
        assert "9087" not in text
        assert "A123456" in text


class _FakeClient:
    def __init__(self, payload: dict[str, object] | None = None, ok: bool = True) -> None:
        self.payload = payload
        self.ok = ok
        self.seen_totp: str | None = None

    def generateSession(self, client_code: str, pin: str, totp: str) -> dict[str, object]:
        self.seen_totp = totp
        if not self.ok:
            return {"status": False, "message": "Invalid totp"}
        return {"status": True, "data": self.payload}


class TestLogin:
    def _clock(self) -> SimClock:
        return SimClock(datetime(2026, 8, 28, 3, 45, tzinfo=UTC))

    def test_happy_path_keeps_the_three_tokens_apart(self, tmp_path: object) -> None:
        client = _FakeClient(
            {"jwtToken": "J", "refreshToken": "R", "feedToken": "F"},
        )
        s = login(
            AngelCredentials.from_env(GOOD),
            self._clock(),
            connect=lambda _k: client,
            totp=lambda _s: "111111",
            cache=None,
        )
        assert (s.jwt_token, s.refresh_token, s.feed_token) == ("J", "R", "F")
        assert s.issued_on == date(2026, 8, 28)

    def test_a_login_without_a_feed_token_is_fatal(self) -> None:
        """The socket would accept the JWT and then simply stay quiet.

        A quiet socket is indistinguishable from a quiet market, so the wrong
        credential has to be caught at login or it is not caught at all.
        """
        client = _FakeClient({"jwtToken": "J", "refreshToken": "R"})
        with pytest.raises(AngelAuthError, match="feedToken"):
            login(
                AngelCredentials.from_env(GOOD),
                self._clock(),
                connect=lambda _k: client,
                totp=lambda _s: "111111",
                cache=None,
            )

    def test_a_failed_login_names_the_usual_causes(self) -> None:
        with pytest.raises(AngelAuthError, match="TOTP secret"):
            login(
                AngelCredentials.from_env(GOOD),
                self._clock(),
                connect=lambda _k: _FakeClient(ok=False),
                totp=lambda _s: "111111",
                cache=None,
            )

    def test_todays_cache_is_reused_and_yesterdays_is_not(self, tmp_path: object) -> None:
        from pathlib import Path

        cache = Path(str(tmp_path)) / "sess.json"
        _session(issued_on=date(2026, 8, 28)).save(cache)
        calls: list[int] = []

        def connect(_k: str) -> _FakeClient:
            calls.append(1)
            return _FakeClient({"jwtToken": "J2", "refreshToken": "R2", "feedToken": "F2"})

        creds = AngelCredentials.from_env(GOOD)
        s = login(creds, self._clock(), connect=connect, totp=lambda _s: "1", cache=cache)
        assert s.feed_token == "FEED" and calls == []

        _session(issued_on=date(2026, 8, 27)).save(cache)
        s = login(creds, self._clock(), connect=connect, totp=lambda _s: "1", cache=cache)
        assert s.feed_token == "F2" and calls == [1], "a stale session must not be reused"


# --------------------------------------------------------------------------- #
# Instrument master
# --------------------------------------------------------------------------- #
MASTER = [
    {
        "token": "26000",
        "symbol": "NIFTY",
        "name": "NIFTY",
        "exch_seg": "NSE",
        "instrumenttype": "AMXIDX",
        "expiry": "",
        "strike": "0",
        "lotsize": "1",
    },
    {
        "token": "43001",
        "symbol": "NIFTY28AUG2524000CE",
        "name": "NIFTY",
        "exch_seg": "NFO",
        "instrumenttype": "OPTIDX",
        "expiry": "28AUG2025",
        "strike": "2400000",
        "lotsize": "75",
    },
    {
        "token": "51002",
        "symbol": "NIFTY28AUG2524000CE",
        "name": "NIFTY",
        "exch_seg": "NFO",
        "instrumenttype": "OPTIDX",
        "expiry": "28AUG2026",
        "strike": "2400000",
        "lotsize": "75",
    },
]


class TestInstrumentMaster:
    def _master(self) -> InstrumentMaster:
        return InstrumentMaster.from_payload(MASTER, fetched_on=date(2026, 8, 28))

    def test_strike_is_decoded_out_of_paise(self) -> None:
        row = self._master().resolve("NIFTY28AUG2524000CE", as_of=date(2026, 8, 1))
        assert row.strike == pytest.approx(24000.0)
        assert row.lot_size == 75

    def test_resolving_as_of_a_past_date_returns_that_dates_contract(self) -> None:
        """The M10 lookahead bug in vendor clothing.

        The same symbol names two contracts a year apart. Resolving without a
        date returns the live one, and a backfill of August 2025 would then be
        priced off a 2026 option.
        """
        m = self._master()
        assert m.resolve("NIFTY28AUG2524000CE", as_of=date(2025, 8, 1)).token == 43001
        assert m.resolve("NIFTY28AUG2524000CE", as_of=date(2026, 8, 1)).token == 51002

    def test_an_expired_only_symbol_raises_rather_than_guessing(self) -> None:
        m = InstrumentMaster.from_payload(MASTER[1:2], fetched_on=date(2026, 8, 28))
        with pytest.raises(KeyError, match="expired"):
            m.resolve("NIFTY28AUG2524000CE", as_of=date(2026, 8, 28))

    def test_an_instrument_with_no_expiry_resolves_at_any_date(self) -> None:
        m = self._master()
        for d in (date(2015, 1, 1), date(2030, 1, 1)):
            assert m.resolve("NIFTY", as_of=d).token == 26000

    def test_resolve_has_no_default_date(self) -> None:
        """If as_of ever grows a default it will be today, and today is the
        wrong answer for every historical query."""
        with pytest.raises(TypeError):
            self._master().resolve("NIFTY")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #
class _FakeSocket:
    def __init__(self, **kw: object) -> None:
        self.kw = kw
        self.subscriptions: list[tuple[int, list[dict[str, object]]]] = []
        self.connected = False
        self.closed = False
        self.on_data = None
        self.on_open = None
        self.on_error = None
        self.on_close = None

    def subscribe(self, _corr: str, mode: int, tokens: list[dict[str, object]]) -> None:
        self.subscriptions.append((mode, tokens))

    def connect(self) -> None:
        self.connected = True
        if self.on_open is not None:
            self.on_open(self)

    def close_connection(self) -> None:
        self.closed = True


def _feed(**kw: object) -> tuple[AngelOneFeed, list[_FakeSocket]]:
    made: list[_FakeSocket] = []

    def factory(**k: object) -> _FakeSocket:
        s = _FakeSocket(**k)
        made.append(s)
        return s

    f = AngelOneFeed(_session(), "abc123", socket_factory=factory, **kw)  # type: ignore[arg-type]
    return f, made


class TestFeed:
    def test_the_socket_gets_the_feed_token_not_the_jwt(self) -> None:
        """Angel returns three tokens and only one opens the stream.

        The wrong one connects successfully and then delivers nothing, so this
        cannot be caught by watching for an error.
        """
        f, made = _feed()
        f.subscribe([26000])
        f.connect()
        assert made[0].kw["feed_token"] == "FEED"
        assert made[0].kw["auth_token"] == "JWT"

    def test_subscription_carries_the_exchange_code_and_mode(self) -> None:
        f, made = _feed()
        f.subscribe([26000, 43001])
        f.connect()
        mode, payload = made[0].subscriptions[0]
        assert mode == MODE_SNAP_QUOTE
        assert payload[0]["exchangeType"] == EXCHANGE_TYPE["NSE_FO"]
        assert payload[0]["tokens"] == ["26000", "43001"]

    def test_over_budget_is_refused_rather_than_truncated(self) -> None:
        """A chain that stops updating at 09:40 looks like a quiet market."""
        f, _ = _feed()
        too_many = list(range(1, f.max_instruments + 2))
        with pytest.raises(ValueError, match="budget"):
            f.subscribe(too_many)

    def test_budget_is_per_connection_times_connections(self) -> None:
        f, _ = _feed()
        assert f.max_instruments == ANGEL_MAX_TOKENS_PER_REQUEST * 3

    def test_gap_detection_reports_that_it_cannot(self) -> None:
        """An empty gap list from this feed means 'cannot tell', not 'none'."""
        f, _ = _feed()
        assert f.gap_detection is False

    def test_ticks_arrive_in_rupees_through_the_queue(self) -> None:
        f, made = _feed()
        f.subscribe([26000])
        f.connect()
        made[0].on_data(made[0], [_snap()])  # type: ignore[misc]
        ticks = list(islice(f.stream(poll_seconds=0.01), 1))
        f.close()
        assert len(ticks) == 1
        assert ticks[0].ltp == pytest.approx(24180.75)
        assert ticks[0].instrument_id == 26000

    def test_an_undecodable_payload_is_counted_not_swallowed(self) -> None:
        """The vendor library eats exceptions raised in its own thread.

        A decode error that only manifests as silence is indistinguishable from
        a closed market, so the count is the only evidence there is.
        """
        f, made = _feed()
        f.subscribe([26000])
        f.connect()
        bad = _snap()
        del bad["last_traded_price"]
        made[0].on_data(made[0], [bad])  # type: ignore[misc]
        assert f.decode_errors == 1
        assert f._queue.qsize() == 0

    def test_a_full_queue_drops_and_counts_rather_than_blocking(self) -> None:
        """Blocking the socket thread stalls the heartbeat, and the vendor then
        drops the connection a few seconds later - a much worse failure than a
        counted drop."""
        f, made = _feed(queue_maxsize=1)
        f.subscribe([26000])
        f.connect()
        for _ in range(3):
            made[0].on_data(made[0], [_snap()])  # type: ignore[misc]
        assert f.dropped == 2

    def test_stream_before_subscribe_raises(self) -> None:
        f, _ = _feed()
        with pytest.raises(RuntimeError, match="subscribe"):
            next(f.stream(poll_seconds=0.01))

    def test_close_closes_the_socket(self) -> None:
        f, made = _feed()
        f.subscribe([26000])
        f.connect()
        f.close()
        assert made[0].closed

    def test_research_cannot_construct_a_feed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envmod.ENV_VAR, "research")
        with pytest.raises(envmod.EnvironmentError_):
            AngelOneFeed(_session(), "abc123", socket_factory=lambda **_k: _FakeSocket())


# --------------------------------------------------------------------------- #
# The scrip master download
# --------------------------------------------------------------------------- #
def _bulk_payload(n: int) -> list[dict[str, str]]:
    """A master of ``n`` plausible rows, so size checks can be tested by size."""
    return [
        {
            "token": str(100000 + i),
            "symbol": f"SCRIP{i}",
            "name": f"SCRIP{i}",
            "exch_seg": "NSE",
            "instrumenttype": "",
            "expiry": "",
            "strike": "0",
            "lotsize": "1",
        }
        for i in range(n)
    ]


class TestScripMasterDownload:
    """The master is the token map. Everything the feed subscribes to comes
    through it, and every failure below produces silence rather than an error."""

    CLOCK = SimClock(datetime(2026, 8, 28, 3, 45, tzinfo=UTC))

    def _getter(self, payload: object, calls: list[str]) -> Callable[[str], bytes]:
        def get(url: str) -> bytes:
            calls.append(url)
            return json.dumps(payload).encode("utf-8")

        return get

    def test_a_truncated_master_raises_rather_than_returning_few_rows(self) -> None:
        """The null test for this component.

        A short response decodes cleanly, resolves nothing, and subscribes the
        feed to no tokens. The feed then produces no ticks, which is exactly
        what a closed market produces. Nothing in the pipeline raises.
        """
        with pytest.raises(ScripMasterError, match="truncated"):
            fetch_scrip_master(self.CLOCK, getter=self._getter(_bulk_payload(50), []))

    def test_an_html_error_page_raises(self) -> None:
        """Angel serves HTML with a 200 on a bad day. json.loads is the only
        thing between that and a master of zero instruments."""
        with pytest.raises(ScripMasterError, match="did not parse as JSON"):
            fetch_scrip_master(self.CLOCK, getter=lambda _u: b"<html>maintenance</html>")

    def test_a_json_object_instead_of_a_list_raises(self) -> None:
        with pytest.raises(ScripMasterError, match="not a list"):
            fetch_scrip_master(self.CLOCK, getter=self._getter({"data": []}, []))

    def test_a_good_master_is_stamped_with_the_injected_clock(self) -> None:
        m = fetch_scrip_master(self.CLOCK, getter=self._getter(_bulk_payload(20_000), []))
        assert len(m.rows) == 20_000
        assert m.fetched_on == date(2026, 8, 28)

    def test_the_cache_round_trips(self, tmp_path: Path) -> None:
        """save/load must preserve expiry and strike exactly. Re-running the
        vendor expiry parser over already-normalised output would drop rows."""
        path = tmp_path / "master.json"
        InstrumentMaster.from_payload(MASTER, fetched_on=date(2026, 8, 28)).save(path)
        back = InstrumentMaster.load(path)
        assert back is not None
        assert back.fetched_on == date(2026, 8, 28)
        row = back.resolve("NIFTY28AUG2524000CE", as_of=date(2026, 8, 1))
        assert row.token == 51002
        assert row.expiry == date(2026, 8, 28)
        assert row.strike == pytest.approx(24000.0)

    def test_todays_cache_is_reused_without_a_download(self, tmp_path: Path) -> None:
        path = tmp_path / "master.json"
        calls: list[str] = []
        first = load_scrip_master(
            self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls)
        )
        second = load_scrip_master(
            self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls)
        )
        assert len(calls) == 1, "today's master was downloaded twice"
        assert len(second.rows) == len(first.rows)

    def test_yesterdays_cache_is_refetched(self, tmp_path: Path) -> None:
        """Not a cache policy - a correctness one.

        Angel republishes daily and an option's token belongs to one expiry, so
        a master from before a roll maps a live symbol to a token that now
        trades a different contract. resolve() cannot tell: the row it finds is
        perfectly well formed.
        """
        path = tmp_path / "master.json"
        stale = InstrumentMaster.from_payload(_bulk_payload(20_000), fetched_on=date(2026, 8, 27))
        stale.save(path)
        calls: list[str] = []
        load_scrip_master(self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls))
        assert calls, "a master fetched yesterday was accepted as today's"

    def test_refresh_downloads_even_when_the_cache_is_todays(self, tmp_path: Path) -> None:
        path = tmp_path / "master.json"
        calls: list[str] = []
        load_scrip_master(self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls))
        load_scrip_master(
            self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls), refresh=True
        )
        assert len(calls) == 2

    def test_a_corrupt_cache_is_ignored_rather_than_fatal(self, tmp_path: Path) -> None:
        """09:14 is the wrong time to crash on a half-written file."""
        path = tmp_path / "master.json"
        path.write_text("{not json", encoding="utf-8")
        assert InstrumentMaster.load(path) is None
        calls: list[str] = []
        m = load_scrip_master(
            self.CLOCK, path=path, getter=self._getter(_bulk_payload(20_000), calls)
        )
        assert len(m.rows) == 20_000

    def test_a_missing_cache_is_not_an_error(self, tmp_path: Path) -> None:
        assert InstrumentMaster.load(tmp_path / "absent.json") is None


class TestStrikeScale:
    """Strike is not in paise across the whole master.

    The first download of the real file raised on a USDINR strike, which is the
    good outcome: the guard band caught a scale the three-row synthetic fixture
    could never have shown. CDS publishes strike scaled by ten million.
    """

    # Real rows from the published master. The symbol carries its own strike,
    # which is what makes these assertions a closed form rather than a copy of
    # whatever the code happened to produce.
    CDS_ROWS: ClassVar[list[tuple[str, str, float]]] = [
        ("USDINR26O2397.25PE", "972500000.000000", 97.25),
        ("JPYINR26NOV57.25CE", "572500000.000000", 57.25),
        ("USDJPY26OCT167PE", "1670000000.000000", 167.0),
        ("694GS203626OCT103.25CE", "1032500000.000000", 103.25),
    ]

    @pytest.mark.parametrize(("symbol", "raw", "expected"), CDS_ROWS)
    def test_cds_strikes_decode_out_of_ten_million(
        self, symbol: str, raw: str, expected: float
    ) -> None:
        m = InstrumentMaster.from_payload(
            [
                {
                    "token": "10013",
                    "symbol": symbol,
                    "name": symbol[:6],
                    "exch_seg": "CDS",
                    "instrumenttype": "OPTCUR",
                    "expiry": "23OCT2026",
                    "strike": raw,
                    "lotsize": "1000",
                }
            ],
            fetched_on=date(2026, 8, 28),
        )
        assert m.rows[0].strike == pytest.approx(expected)

    def test_a_cds_strike_read_as_paise_is_refused(self) -> None:
        """The failure this guard exists for.

        97.25 read as paise is 9,725,000 - past the plausibility band, so it
        raises instead of quietly making every currency moneyness five orders
        of magnitude wrong.
        """
        with pytest.raises(PriceScaleError):
            strike_to_rupees(972_500_000.0, "NFO")

    def test_every_other_segment_stays_in_paise(self) -> None:
        for seg in ("NFO", "BFO", "MCX", "NCO", "NSE"):
            assert strike_to_rupees(2_400_000.0, seg) == pytest.approx(24_000.0)

    def test_the_segment_lookup_is_case_insensitive(self) -> None:
        assert strike_to_rupees(972_500_000.0, "cds") == pytest.approx(97.25)

    def test_an_unknown_segment_falls_back_to_paise(self) -> None:
        """A new segment must not silently inherit CDS's scale."""
        assert strike_to_rupees(2_400_000.0, "XYZ") == pytest.approx(24_000.0)

    def test_the_no_strike_sentinel_is_not_a_price(self) -> None:
        """Equities, futures and underlying rows carry -1, not an empty field.

        24,035 rows of the real master do, and exactly one negative value
        appears in the whole file - so it is a sentinel. Sent through the scale
        check it raises PriceScaleError on the most ordinary rows there are.
        """
        m = InstrumentMaster.from_payload(
            [
                {
                    "token": "2885",
                    "symbol": "RELIANCE-EQ",
                    "name": "RELIANCE",
                    "exch_seg": "NSE",
                    "instrumenttype": "",
                    "expiry": "",
                    "strike": "-1.000000",
                    "lotsize": "1",
                }
            ],
            fetched_on=date(2026, 8, 28),
        )
        assert m.rows[0].strike == 0.0


def _opt_row(
    symbol: str,
    expiry: str,
    strike: str,
    *,
    name: str = "NIFTY",
    seg: str = "NFO",
    itype: str = "OPTIDX",
    token: str = "1",
) -> dict[str, str]:
    return {
        "token": token,
        "symbol": symbol,
        "name": name,
        "exch_seg": seg,
        "instrumenttype": itype,
        "expiry": expiry,
        "strike": strike,
        "lotsize": "75",
    }


class TestOptionContracts:
    """Selecting the slice of the chain a live subscription will carry.

    The registry built from this decides what every tick is attributed to, so
    the failure modes are attribution failures, not lookup failures.
    """

    PAYLOAD: ClassVar[list[dict[str, str]]] = [
        _opt_row("NIFTY01SEP2624000CE", "01SEP2026", "2400000", token="101"),
        _opt_row("NIFTY01SEP2624000PE", "01SEP2026", "2400000", token="102"),
        _opt_row("NIFTY08SEP2624000CE", "08SEP2026", "2400000", token="103"),
        _opt_row("NIFTY15SEP2624000CE", "15SEP2026", "2400000", token="104"),
        _opt_row("NIFTY25AUG2624000CE", "25AUG2026", "2400000", token="105"),
        # same name, different segment - a currency option on a like-named index
        _opt_row(
            "NIFTY01SEP2624000CE", "01SEP2026", "9725000", seg="CDS", itype="OPTCUR", token="106"
        ),
        # the underlying itself: no expiry, not an option
        {
            "token": "26000",
            "symbol": "NIFTY",
            "name": "NIFTY",
            "exch_seg": "NSE",
            "instrumenttype": "",
            "expiry": "",
            "strike": "-1.000000",
            "lotsize": "1",
        },
    ]

    def _master(self) -> InstrumentMaster:
        return InstrumentMaster.from_payload(self.PAYLOAD, fetched_on=date(2026, 8, 30))

    def test_only_the_nearest_expiries_are_returned(self) -> None:
        rows = self._master().option_contracts("NIFTY", as_of=date(2026, 8, 30), n_expiries=2)
        assert sorted({r.expiry for r in rows}) == [date(2026, 9, 1), date(2026, 9, 8)]

    def test_expired_contracts_are_excluded(self) -> None:
        """Their tokens get reused. A tick attributed to a contract that stopped
        trading is not a missing tick - it is a wrong one."""
        rows = self._master().option_contracts("NIFTY", as_of=date(2026, 8, 30), n_expiries=9)
        assert all(r.expiry is not None and r.expiry >= date(2026, 8, 30) for r in rows)
        assert "NIFTY25AUG2624000CE" not in {r.symbol for r in rows}

    def test_a_different_segment_is_not_swept_in(self) -> None:
        """CDS carries its own OPTCUR contracts under names that collide."""
        rows = self._master().option_contracts("NIFTY", as_of=date(2026, 8, 30), n_expiries=9)
        assert all(r.exch_seg == "NFO" for r in rows)

    def test_the_underlying_row_is_not_an_option(self) -> None:
        rows = self._master().option_contracts("NIFTY", as_of=date(2026, 8, 30), n_expiries=9)
        assert 26000 not in {r.token for r in rows}

    def test_an_empty_selection_raises_rather_than_returning_nothing(self) -> None:
        """A silent feed and an empty subscription look identical downstream."""
        with pytest.raises(KeyError, match="no live"):
            self._master().option_contracts("BANKNIFTY", as_of=date(2026, 8, 30))

    def test_as_of_has_no_default(self) -> None:
        """A default would be today, and a registry built with today's expiries
        then pointed at a past session attributes ticks to contracts that were
        not listed yet."""
        with pytest.raises(TypeError):
            self._master().option_contracts("NIFTY")  # type: ignore[call-arg]


class TestOptType:
    def test_the_right_comes_from_the_symbol_suffix(self) -> None:
        """instrumenttype says OPTIDX for both sides, so it cannot answer this."""
        m = InstrumentMaster.from_payload(
            [
                _opt_row("NIFTY01SEP2624000CE", "01SEP2026", "2400000", token="1"),
                _opt_row("NIFTY01SEP2624000PE", "01SEP2026", "2400000", token="2"),
            ],
            fetched_on=date(2026, 8, 30),
        )
        by_token = {r.token: r.opt_type for r in m.rows}
        assert by_token[1] is OptionType.CE
        assert by_token[2] is OptionType.PE

    def test_a_non_option_row_has_no_right(self) -> None:
        m = InstrumentMaster.from_payload(
            [
                {
                    "token": "26000",
                    "symbol": "NIFTY",
                    "name": "NIFTY",
                    "exch_seg": "NSE",
                    "instrumenttype": "",
                    "expiry": "",
                    "strike": "-1.000000",
                    "lotsize": "1",
                }
            ],
            fetched_on=date(2026, 8, 30),
        )
        assert m.rows[0].opt_type is None


class TestTwoExchangeSubscription:
    """A chain snapshot needs the spot and the options from one stream.

    NIFTY quotes on NSE_CM; every option on it quotes on NSE_FO. Sending both
    under a single exchangeType is accepted at subscribe time and then delivers
    nothing for the tokens that do not belong there, which is a silence
    indistinguishable from a closed market.
    """

    def test_each_segment_gets_its_own_payload_entry(self) -> None:
        f, made = _feed()
        f.subscribe_on("NSE_FO", [43001, 51002])
        f.subscribe_on("NSE_CM", [26000])
        f.connect()
        sent = {
            payload[0]["exchangeType"]: payload[0]["tokens"]
            for _mode, payload in made[0].subscriptions
        }
        assert sent[EXCHANGE_TYPE["NSE_FO"]] == ["43001", "51002"]
        assert sent[EXCHANGE_TYPE["NSE_CM"]] == ["26000"]

    def test_the_budget_counts_across_segments(self) -> None:
        """Two segments are two halves of one connection budget, not two budgets."""
        f, _ = _feed()
        f.subscribe_on("NSE_FO", list(range(1, f.max_instruments)))
        with pytest.raises(ValueError, match="budget"):
            f.subscribe_on("NSE_CM", [900_001, 900_002])

    def test_a_token_already_held_elsewhere_is_not_resubscribed(self) -> None:
        f, made = _feed()
        f.subscribe_on("NSE_FO", [26000])
        f.subscribe_on("NSE_CM", [26000])
        f.connect()
        tokens = [t for _m, payload in made[0].subscriptions for t in payload[0]["tokens"]]
        assert tokens.count("26000") == 1

    def test_an_unknown_segment_is_refused(self) -> None:
        f, _ = _feed()
        with pytest.raises(KeyError, match="unknown exchange"):
            f.subscribe_on("NSE_XX", [26000])


class TestAngelChainWiring:
    """Scrip rows to the registry the ChainBuilder attributes ticks with.

    Every id here is Angel's own token, because that is what its ticks arrive
    stamped with. An invented id would attribute real quotes to a contract that
    does not exist, and the snapshot would look entirely normal.
    """

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            chain_name="NIFTY", n_expiries=2, strikes_per_side=15, chain_seconds=5.0
        )

    def _master(self) -> InstrumentMaster:
        return InstrumentMaster.from_payload(
            TestOptionContracts.PAYLOAD, fetched_on=date(2026, 8, 30)
        )

    def test_tokens_and_terms_survive_the_mapping(self) -> None:
        from libs.ms_core.clock import SessionCalendar
        from services.ingest_market.__main__ import build_angel_chain

        cal = SessionCalendar.from_config(cfgmod.load(warn=False).sessions)
        clock = SimClock(datetime(2026, 8, 30, 4, 0, tzinfo=UTC))
        _builder, index_token, option_tokens = build_angel_chain(
            self._args(), cal, clock, master=self._master()
        )
        assert index_token == 26000, "the underlying must be the index, not an option"
        assert sorted(option_tokens) == [101, 102, 103]
        assert 106 not in option_tokens, "the CDS contract is a different instrument"
        assert 105 not in option_tokens, "that contract had already expired"

    def test_the_registry_carries_the_right_and_the_strike(self) -> None:
        from libs.ms_core.clock import SessionCalendar
        from services.ingest_market.__main__ import build_angel_chain

        cal = SessionCalendar.from_config(cfgmod.load(warn=False).sessions)
        clock = SimClock(datetime(2026, 8, 30, 4, 0, tzinfo=UTC))
        builder, _idx, _toks = build_angel_chain(self._args(), cal, clock, master=self._master())
        reg = builder._registry
        ce = reg.get(101)
        pe = reg.get(102)
        assert ce is not None and pe is not None
        assert ce.opt_type is OptionType.CE and pe.opt_type is OptionType.PE
        assert ce.strike == pytest.approx(24_000.0), "strike must be out of paise"
        assert ce.expiry == date(2026, 9, 1)
        assert ce.underlying_id == 26000
