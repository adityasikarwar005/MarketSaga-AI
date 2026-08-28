"""M11 - the Upstox V3 adapter, exercised without a token or a network.

The traps here are different from Angel One's, and one of them is the opposite:
Upstox sends rupees where Angel sends paise, so the danger is somebody who has
just fixed the Angel adapter adding a `/ 100` that makes every number a hundred
times too small while leaving every internal consistency check happy.

The others are the snapshot that is not a tick, the trade time that is not now,
the int64 that arrives as a string, and the instrument key that is not an int.
Each has a test that fails when its guard is removed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import islice
from typing import Any

import pytest

from libs.ms_core import env as envmod
from libs.ms_data.upstox import (
    MAX_INSTRUMENTS_FULL,
    MAX_INSTRUMENTS_LTPC,
    STALE_TRADE_SECONDS,
    FeedType,
    InstrumentKeyMap,
    SegmentStatus,
    UnmappedInstrument,
    UpstoxAuthError,
    UpstoxFeed,
    authorize,
    decode_feed_response,
    token_from_env,
)

NIFTY = "NSE_INDEX|Nifty 50"
CE = "NSE_FO|52000"

KEYMAP = InstrumentKeyMap({NIFTY: 26000, CE: 52000})

# 2025-08-24T04:26:40Z, as protobuf renders int64: a string.
TRADE_MS = "1756009600000"
FEED_MS = "1756009601000"


@pytest.fixture(autouse=True)
def _paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(envmod.ENV_VAR, "paper")


def _full_feed(**over: Any) -> dict[str, Any]:
    """A `full` mode message, shaped the way MessageToDict renders the proto."""
    ltpc = {"ltp": 24180.75, "ltt": TRADE_MS, "ltq": "75", "cp": 24100.0}
    ltpc.update(over.pop("ltpc", {}))
    payload: dict[str, Any] = {
        "type": "live_feed",
        "currentTs": FEED_MS,
        "feeds": {
            CE: {
                "fullFeed": {
                    "marketFF": {
                        "ltpc": ltpc,
                        "marketLevel": {
                            "bidAskQuote": [
                                {"bidQ": "150", "bidP": 24180.50, "askQ": "225", "askP": 24181.00},
                                {"bidQ": "300", "bidP": 24180.25, "askQ": "75", "askP": 24181.25},
                            ]
                        },
                        "oi": 1234500.0,
                        "vtt": "9876543",
                    }
                },
                "requestMode": "full_d5",
            }
        },
    }
    payload.update(over)
    return payload


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
class TestPrices:
    def test_prices_are_rupees_and_are_not_divided(self) -> None:
        """The proto declares `double ltp`. There is no paise here.

        This is the inverse of the Angel One adapter, and the failure mode is
        somebody carrying that fix across: a `/ 100` leaves the spread, the
        moneyness and every ratio internally consistent while putting NIFTY at
        241.81.
        """
        d = decode_feed_response(_full_feed(), KEYMAP)
        (tick,) = d.ticks
        assert tick.ltp == pytest.approx(24180.75)
        assert tick.bid == pytest.approx(24180.50)
        assert tick.ask == pytest.approx(24181.00)
        assert tick.mid == pytest.approx(24180.75)

    def test_only_the_best_quote_is_used(self) -> None:
        d = decode_feed_response(_full_feed(), KEYMAP)
        (tick,) = d.ticks
        assert tick.bid_qty == 150 and tick.ask_qty == 225

    def test_a_missing_book_is_absent_not_zero(self) -> None:
        """proto3 omits defaults, so absent must never become a zero price.

        A zero bid makes a one-sided book look two-sided and puts the mid
        halfway to nothing.
        """
        p = _full_feed()
        del p["feeds"][CE]["fullFeed"]["marketFF"]["marketLevel"]
        (tick,) = decode_feed_response(p, KEYMAP).ticks
        assert tick.bid is None and tick.ask is None and tick.mid is None


# --------------------------------------------------------------------------- #
# int64 as string
# --------------------------------------------------------------------------- #
class TestInt64AsString:
    def test_quantities_arrive_as_strings_and_come_out_as_ints(self) -> None:
        """Protobuf's JSON mapping renders int64 as a string.

        Left as a string, `qty` is a TypeError in arithmetic on a good day and
        a string concatenation on a bad one.
        """
        (tick,) = decode_feed_response(_full_feed(), KEYMAP).ticks
        assert tick.qty == 75 and isinstance(tick.qty, int)
        assert isinstance(tick.bid_qty, int) and isinstance(tick.ask_qty, int)

    def test_the_timestamp_string_is_parsed_not_passed_through(self) -> None:
        (tick,) = decode_feed_response(_full_feed(), KEYMAP).ticks
        assert tick.ts == datetime.fromtimestamp(int(TRADE_MS) / 1000, tz=UTC)
        assert tick.ts.tzinfo is not None

    def test_open_interest_is_a_count_despite_being_declared_double(self) -> None:
        (tick,) = decode_feed_response(_full_feed(), KEYMAP).ticks
        assert tick.oi == 1234500 and isinstance(tick.oi, int)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
class TestTradeTime:
    def test_t_event_is_the_trade_time_not_the_feed_clock(self) -> None:
        """currentTs advances every second; ltt only when a trade happens.

        Using currentTs would claim a trade occurred now. On an illiquid strike
        the honest answer is hours ago, and Tick.mid's own docstring already
        warns that a stale LTP manufactures signal that does not exist.
        """
        (tick,) = decode_feed_response(_full_feed(), KEYMAP).ticks
        assert tick.ts != datetime.fromtimestamp(int(FEED_MS) / 1000, tz=UTC)
        assert tick.ts == datetime.fromtimestamp(int(TRADE_MS) / 1000, tz=UTC)

    def test_a_stale_trade_is_counted_rather_than_hidden(self) -> None:
        old = str(int(FEED_MS) - int(STALE_TRADE_SECONDS * 1000) - 5_000)
        d = decode_feed_response(_full_feed(ltpc={"ltt": old}), KEYMAP)
        assert d.stale_trades == 1

    def test_a_fresh_trade_is_not_counted_stale(self) -> None:
        d = decode_feed_response(_full_feed(), KEYMAP)
        assert d.stale_trades == 0

    def test_with_no_ltt_the_feed_clock_is_the_fallback(self) -> None:
        p = _full_feed()
        del p["feeds"][CE]["fullFeed"]["marketFF"]["ltpc"]["ltt"]
        (tick,) = decode_feed_response(p, KEYMAP).ticks
        assert tick.ts == datetime.fromtimestamp(int(FEED_MS) / 1000, tz=UTC)

    def test_with_neither_the_tick_is_refused(self) -> None:
        """t_event has to mean something. A guessed one is worse than none."""
        p = _full_feed()
        del p["feeds"][CE]["fullFeed"]["marketFF"]["ltpc"]["ltt"]
        del p["currentTs"]
        d = decode_feed_response(p, KEYMAP)
        assert d.ticks == []
        assert any("t_event" in s for s in d.skipped)


# --------------------------------------------------------------------------- #
# The snapshot that is not a tick
# --------------------------------------------------------------------------- #
class TestInitialFeed:
    def test_the_initial_snapshot_is_dropped_by_default(self) -> None:
        """Before the open it holds yesterday's close.

        Recorded as a tick it becomes today's first trade at a price nobody
        traded today - and it is indistinguishable from a real one afterwards.
        """
        d = decode_feed_response(_full_feed(type="initial_feed"), KEYMAP)
        assert d.ticks == []
        assert d.feed_type is FeedType.INITIAL
        assert any("initial_feed" in s for s in d.skipped)

    def test_it_can_be_asked_for_explicitly(self) -> None:
        d = decode_feed_response(_full_feed(type="initial_feed"), KEYMAP, accept_initial=True)
        assert len(d.ticks) == 1

    def test_a_live_feed_is_never_dropped(self) -> None:
        assert len(decode_feed_response(_full_feed(), KEYMAP).ticks) == 1


# --------------------------------------------------------------------------- #
# Instrument keys
# --------------------------------------------------------------------------- #
class TestInstrumentKeys:
    def test_an_unmapped_key_raises_rather_than_being_hashed(self) -> None:
        """A fabricated id produces ticks for an instrument that does not exist,
        and they look exactly like real ticks."""
        p = _full_feed()
        p["feeds"]["NSE_FO|99999"] = p["feeds"].pop(CE)
        with pytest.raises(UnmappedInstrument, match="fabricated"):
            decode_feed_response(p, KEYMAP)

    def test_the_mapped_id_is_used(self) -> None:
        (tick,) = decode_feed_response(_full_feed(), KEYMAP).ticks
        assert tick.instrument_id == 52000

    def test_two_keys_for_one_id_is_refused_at_subscribe(self) -> None:
        """Ticks would arrive twice under one id and read as volume."""
        feed, _ = _feed(keymap=InstrumentKeyMap({NIFTY: 26000, CE: 26000}))
        with pytest.raises(ValueError, match="both"):
            feed.subscribe([26000])


# --------------------------------------------------------------------------- #
# Market status
# --------------------------------------------------------------------------- #
class TestSegmentStatus:
    def test_the_vendor_answers_the_quiet_versus_dead_question(self) -> None:
        p = {
            "type": "market_info",
            "currentTs": FEED_MS,
            "marketInfo": {"segmentStatus": {"NSE_FO": "NORMAL_OPEN", "NSE_EQ": "PRE_OPEN_START"}},
        }
        d = decode_feed_response(p, KEYMAP)
        assert d.segment_status["NSE_FO"] is SegmentStatus.NORMAL_OPEN
        assert d.ticks == []

    def test_market_open_is_none_until_the_vendor_has_said(self) -> None:
        feed, _ = _feed()
        assert feed.market_open is None, "an unasked question must not answer itself"

    def test_market_open_follows_the_reported_status(self) -> None:
        feed, made = _feed()
        feed.subscribe_keys([CE])
        feed.connect()
        made[0].deliver(
            {"type": "market_info", "marketInfo": {"segmentStatus": {"NSE_FO": "NORMAL_OPEN"}}}
        )
        assert feed.market_open is True
        made[0].deliver(
            {"type": "market_info", "marketInfo": {"segmentStatus": {"NSE_FO": "NORMAL_CLOSE"}}}
        )
        assert feed.market_open is False


# --------------------------------------------------------------------------- #
# Token and authorisation
# --------------------------------------------------------------------------- #
class TestToken:
    def test_research_cannot_hold_a_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envmod.ENV_VAR, "research")
        with pytest.raises(envmod.EnvironmentError_):
            token_from_env({"UPSTOX_ACCESS_TOKEN": "abc"})

    def test_a_missing_token_explains_which_kind_to_get(self) -> None:
        with pytest.raises(UpstoxAuthError, match="Analytics Token"):
            token_from_env({})

    def test_the_token_is_read_from_the_environment(self) -> None:
        assert token_from_env({"UPSTOX_ACCESS_TOKEN": " abc "}) == "abc"


class TestAuthorize:
    def test_the_wss_url_is_returned(self) -> None:
        def fetch(_u: str, _h: dict[str, str]) -> dict[str, Any]:
            return {"data": {"authorized_redirect_uri": "wss://x.upstox.com/feeds?code=1"}}

        assert authorize("tok", fetch=fetch).startswith("wss://")

    def test_the_bearer_header_is_sent(self) -> None:
        seen: dict[str, str] = {}

        def fetch(_u: str, headers: dict[str, str]) -> dict[str, Any]:
            seen.update(headers)
            return {"data": {"authorized_redirect_uri": "wss://x"}}

        authorize("tok", fetch=fetch)
        assert seen["Authorization"] == "Bearer tok"

    def test_a_403_names_the_analytics_token_case_first(self) -> None:
        """The single most reported problem with this endpoint.

        A bare 403 sends people to regenerate a token that was never the
        problem, so the message says what to test instead.
        """

        def fetch(_u: str, _h: dict[str, str]) -> dict[str, Any]:
            raise PermissionError("HTTP 403")

        with pytest.raises(UpstoxAuthError, match="Analytics Token"):
            authorize("tok", fetch=fetch)

    def test_an_empty_url_is_fatal_here_not_later(self) -> None:
        def fetch(_u: str, _h: dict[str, str]) -> dict[str, Any]:
            return {"data": {}}

        with pytest.raises(UpstoxAuthError, match="no authorized_redirect_uri"):
            authorize("tok", fetch=fetch)


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #
class _FakeSocket:
    def __init__(self, url: str, on_message: Any) -> None:
        self.url = url
        self._on_message = on_message
        self.sent: list[dict[str, Any]] = []
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def send(self, payload: bytes) -> None:
        import json as _json

        self.sent.append(_json.loads(payload.decode("utf-8")))

    def close(self) -> None:
        self.closed = True

    def deliver(self, payload: dict[str, Any]) -> None:
        """Hand the feed a message, as bytes it will pass to the decoder."""
        import json as _json

        self._on_message(_json.dumps(payload).encode("utf-8"))


def _feed(**kw: Any) -> tuple[UpstoxFeed, list[_FakeSocket]]:
    import json as _json

    made: list[_FakeSocket] = []

    def factory(*, url: str, on_message: Any) -> _FakeSocket:
        s = _FakeSocket(url, on_message)
        made.append(s)
        return s

    keymap = kw.pop("keymap", KEYMAP)
    feed = UpstoxFeed(
        "tok",
        keymap,
        socket_factory=factory,
        authorizer=lambda _t: "wss://x.upstox.com/feeds?code=1",
        decoder=lambda raw: _json.loads(raw.decode("utf-8")),
        **kw,
    )
    return feed, made


class TestFeed:
    def test_the_subscription_message_matches_the_v3_shape(self) -> None:
        feed, made = _feed()
        feed.subscribe_keys([CE, NIFTY])
        feed.connect()
        (msg,) = made[0].sent
        assert msg["method"] == "sub"
        assert msg["data"] == {"mode": "full", "instrumentKeys": [CE, NIFTY]}
        assert msg["guid"]

    def test_subscribe_translates_internal_ids_to_vendor_keys(self) -> None:
        feed, made = _feed()
        feed.subscribe([52000])
        feed.connect()
        assert made[0].sent[0]["data"]["instrumentKeys"] == [CE]

    def test_an_unknown_internal_id_is_refused(self) -> None:
        feed, _ = _feed()
        with pytest.raises(UnmappedInstrument, match="no vendor key"):
            feed.subscribe([999])

    def test_over_budget_is_refused_rather_than_truncated(self) -> None:
        feed, _ = _feed()
        with pytest.raises(ValueError, match="per connection"):
            feed.subscribe_keys([f"NSE_FO|{i}" for i in range(MAX_INSTRUMENTS_FULL + 1)])

    def test_ticks_reach_the_consumer_in_rupees(self) -> None:
        feed, made = _feed()
        feed.subscribe_keys([CE])
        feed.connect()
        made[0].deliver(_full_feed())
        ticks = list(islice(feed.stream(poll_seconds=0.01), 1))
        feed.close()
        assert len(ticks) == 1
        assert ticks[0].ltp == pytest.approx(24180.75)
        assert ticks[0].instrument_id == 52000

    def test_an_undecodable_message_is_counted_not_swallowed(self) -> None:
        """The websocket library eats exceptions raised in its own thread, and
        the only symptom is silence - which looks like a closed market."""
        feed, made = _feed()
        feed.subscribe_keys([CE])
        feed.connect()
        made[0]._on_message(b"not protobuf, not json")
        assert feed.decode_errors == 1

    def test_an_unmapped_key_arriving_live_is_counted_not_fatal(self) -> None:
        """It must stop the tick, not the recorder. A day's data is worth more
        than the tidiness of crashing on one bad key."""
        feed, made = _feed()
        feed.subscribe_keys([CE])
        feed.connect()
        p = _full_feed()
        p["feeds"]["NSE_FO|99999"] = p["feeds"].pop(CE)
        made[0].deliver(p)
        assert feed.decode_errors == 1
        assert feed._queue.qsize() == 0

    def test_a_full_queue_drops_and_counts_rather_than_blocking(self) -> None:
        feed, made = _feed(queue_maxsize=1)
        feed.subscribe_keys([CE])
        feed.connect()
        for _ in range(3):
            made[0].deliver(_full_feed())
        assert feed.dropped == 2

    def test_gap_detection_reports_that_it_cannot(self) -> None:
        feed, _ = _feed()
        assert feed.gap_detection is False

    def test_stream_before_subscribe_raises(self) -> None:
        feed, _ = _feed()
        with pytest.raises(RuntimeError, match="subscribe"):
            next(feed.stream(poll_seconds=0.01))

    def test_close_closes_the_socket(self) -> None:
        feed, made = _feed()
        feed.subscribe_keys([CE])
        feed.connect()
        feed.close()
        assert made[0].closed

    def test_research_cannot_construct_a_feed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envmod.ENV_VAR, "research")
        with pytest.raises(envmod.EnvironmentError_):
            UpstoxFeed("tok", KEYMAP)

    def test_the_budget_follows_the_mode(self) -> None:
        """ltpc carries 5000 per connection, full carries 2000.

        A consumer budgeting its chain against the wrong number either wastes
        half the capacity or plans a subscription the vendor will refuse.
        """
        full, _ = _feed()
        ltpc, _ = _feed(mode="ltpc")
        assert full.max_instruments == MAX_INSTRUMENTS_FULL
        assert ltpc.max_instruments == MAX_INSTRUMENTS_LTPC
        ltpc.subscribe_keys([f"NSE_FO|{i}" for i in range(MAX_INSTRUMENTS_FULL + 1)])
