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

from datetime import UTC, date, datetime
from itertools import islice

import pytest

from libs.ms_core import env as envmod
from libs.ms_core.clock import SimClock
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
    login,
    paise_to_rupees,
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
