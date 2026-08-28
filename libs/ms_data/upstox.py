"""M11 - Upstox Market Data Feed V3 adapter.

Implements :class:`MarketFeed` over Upstox's V3 websocket. Nothing downstream
imports this module: consumers depend on the protocol, so the vendor stays a
one-file decision.

Chosen over the alternatives for one reason that matters to an unattended
recorder: the **Analytics Token is valid for a year and is read-only**. No daily
OAuth, and no code path in this adapter can place an order because the token
itself cannot. That is a stronger guarantee than the environment guards in
``ms_core.env``, which are ours to get wrong; this one is the venue's.

The decoding is split from the transport on purpose. ``decode_feed_response``
takes a plain dict and is exercised offline against fixtures, so every trap
below has a test that does not need a broker account, a network, or protobuf.

Six things this adapter is built around
---------------------------------------

**Prices are ``double`` and already in rupees.** The proto declares ``double
ltp``, not an integer. This is the opposite of Angel One, where the wire carries
paise - and the reason it is stated here and pinned by a test is that somebody
who has just debugged the Angel adapter will reach for a ``/ 100`` and every
number will still look plausible afterwards.

**``ltt`` is the last *traded* time, not now.** On an illiquid strike it can be
hours old while ``currentTs`` advances every second. Stamping ``t_event`` with
``currentTs`` claims a trade happened that did not, and ``Tick.mid``'s own
docstring already warns that a stale LTP manufactures signal that does not
exist. So ``ltt`` is used when present and the staleness is reported rather than
smoothed away.

**The first message after subscribing is a snapshot, not a tick.** ``Type`` is
``initial_feed``; its contents are the last known state, which on a pre-open
subscription is yesterday. Feeding it into the recorder writes yesterday's
close as today's first trade. It is dropped unless explicitly asked for.

**``feeds`` is keyed by an instrument *key*, a string like ``NSE_FO|52000``.**
``Tick.instrument_id`` is an int, and the mapping between them belongs to the
instrument master (M10), not to a hash function. An unmapped key raises: a
fabricated id silently attributes ticks to an instrument that does not exist.

**int64 fields arrive as strings.** Protobuf's JSON mapping renders 64-bit
integers as strings to survive Javascript, so ``ltt``, ``vtt`` and ``currentTs``
come back as ``"1756000000000"``. Arithmetic on that is a ``TypeError`` on a
good day and a string concatenation on a bad one.

**proto3 omits defaults.** A field equal to zero is simply absent from the
decoded dict. Absent therefore means "zero or unset", never "missing data", and
nothing here treats a missing bid as a reason to raise.

One thing the vendor gets right that Angel One does not: ``MarketInfo``
carries a per-segment market status. A quiet market and a dead socket are
otherwise indistinguishable from the outside, and this turns that guess into an
answer - see :attr:`UpstoxFeed.segment_status`.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from queue import Empty, Queue
from typing import Any

from libs.ms_core import env as envmod
from libs.ms_core.clock import UTC
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Tick
from libs.ms_data.feed import MarketFeed

log = get_logger(__name__)

AUTHORIZE_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
PROTO_URL = "https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto"

SRC_UPSTOX = 2

# Documented V3 limits for a standard (non-Plus) account. Configuration, not
# truth - re-verify before Gate G1.
MAX_CONNECTIONS = 2
MAX_INSTRUMENTS_LTPC = 5000
MAX_INSTRUMENTS_FULL = 2000

# A trade older than this, relative to the feed's own clock, is not a current
# price. Not a rejection - an annotation, because on a far OTM strike a
# genuinely old last trade is the honest state of the world.
STALE_TRADE_SECONDS = 60.0


class UpstoxAuthError(RuntimeError):
    """Raised when the feed cannot be authorised."""


class UnmappedInstrument(KeyError):
    """Raised when a vendor instrument key has no internal id.

    Deliberately fatal. The alternative is hashing the key into an int, which
    produces ticks attributed to an instrument that does not exist anywhere in
    the instrument master - and they look exactly like real ticks.
    """


class FeedType(StrEnum):
    INITIAL = "initial_feed"
    LIVE = "live_feed"
    MARKET_INFO = "market_info"


class SegmentStatus(StrEnum):
    PRE_OPEN_START = "PRE_OPEN_START"
    PRE_OPEN_END = "PRE_OPEN_END"
    NORMAL_OPEN = "NORMAL_OPEN"
    NORMAL_CLOSE = "NORMAL_CLOSE"
    CLOSING_START = "CLOSING_START"
    CLOSING_END = "CLOSING_END"


# --------------------------------------------------------------------------- #
# Token
# --------------------------------------------------------------------------- #
def token_from_env(env: dict[str, str] | None = None) -> str:
    """The access token, from the environment and nowhere else.

    Either an Analytics Token (a year, read-only) or a daily OAuth token - the
    adapter cannot tell them apart and does not need to. Both are bearer tokens
    and both are read-only for this feed's purposes.
    """
    envmod.require_broker_credentials()
    src = env if env is not None else dict(os.environ)
    token = (src.get("UPSTOX_ACCESS_TOKEN") or "").strip().strip("\"'")
    if not token:
        raise UpstoxAuthError(
            "UPSTOX_ACCESS_TOKEN is not set. Put it in .env (gitignored) and run "
            "through `python tasks.py`, which loads it.\n"
            "  Prefer an Analytics Token: one year, read-only, no daily login. "
            "Generate it from the Upstox developer console.\n"
            "  A daily OAuth access token also works and expires the next "
            "trading day, which for an unattended recorder means it fails on a "
            "morning nobody is watching."
        )
    return token


# --------------------------------------------------------------------------- #
# Instrument keys
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstrumentKeyMap:
    """Vendor instrument key -> the instrument id used everywhere else.

    Built from the instrument master, not derived. Upstox keys look like
    ``NSE_INDEX|Nifty 50`` and ``NSE_FO|52000``; there is no arithmetic that
    turns one into an int without inventing something.
    """

    to_id: dict[str, int] = field(default_factory=dict)

    def resolve(self, key: str) -> int:
        try:
            return self.to_id[key]
        except KeyError:
            raise UnmappedInstrument(
                f"instrument key {key!r} has no id in the map. Ticks for it would "
                "have to be attributed to a fabricated id, and they would look "
                "exactly like real ticks. Add it to the instrument master, or "
                "stop subscribing to it."
            ) from None

    @property
    def keys(self) -> list[str]:
        return list(self.to_id)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _as_int(raw: Any) -> int | None:
    """Protobuf renders int64 as a JSON string. Undo that, once, here."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):  # bool is an int subclass; never a quantity
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _as_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _ts_from_millis(raw: Any) -> datetime | None:
    ms = _as_int(raw)
    if ms is None or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def _first_quote(feed: dict[str, Any]) -> dict[str, Any]:
    """Best bid/ask, from whichever shape the requested mode produced."""
    full = feed.get("fullFeed") or {}
    market = full.get("marketFF") or {}
    level = market.get("marketLevel") or {}
    quotes = level.get("bidAskQuote") or []
    if quotes:
        return dict(quotes[0])
    first_level = feed.get("firstLevelWithGreeks") or {}
    return dict(first_level.get("firstDepth") or {})


def _ltpc(feed: dict[str, Any]) -> dict[str, Any]:
    """LTPC sits at one of three depths depending on the mode."""
    if "ltpc" in feed and isinstance(feed["ltpc"], dict):
        return dict(feed["ltpc"])
    full = feed.get("fullFeed") or {}
    for branch in ("marketFF", "indexFF"):
        node = full.get(branch) or {}
        if isinstance(node.get("ltpc"), dict):
            return dict(node["ltpc"])
    first_level = feed.get("firstLevelWithGreeks") or {}
    if isinstance(first_level.get("ltpc"), dict):
        return dict(first_level["ltpc"])
    return {}


def _open_interest(feed: dict[str, Any]) -> int | None:
    """``oi`` is declared ``double`` but counts contracts."""
    full = feed.get("fullFeed") or {}
    market = full.get("marketFF") or {}
    raw = market.get("oi")
    if raw is None:
        raw = (feed.get("firstLevelWithGreeks") or {}).get("oi")
    value = _as_float(raw)
    return int(value) if value is not None else None


@dataclass(frozen=True)
class DecodedFeed:
    """One websocket message, decoded.

    ``skipped`` is part of the result rather than a log line. A message that
    produced no ticks is the normal case for ``market_info``, and it is a
    reportable problem for ``live_feed`` - the caller cannot tell those apart
    from an empty list.
    """

    ticks: list[Tick]
    feed_type: FeedType | None
    segment_status: dict[str, SegmentStatus]
    skipped: list[str]
    current_ts: datetime | None

    @property
    def stale_trades(self) -> int:
        if self.current_ts is None:
            return 0
        return sum(
            1 for t in self.ticks if (self.current_ts - t.ts).total_seconds() > STALE_TRADE_SECONDS
        )


def decode_feed_response(
    payload: dict[str, Any],
    keymap: InstrumentKeyMap,
    *,
    accept_initial: bool = False,
) -> DecodedFeed:
    """One ``FeedResponse`` dict into ticks.

    ``accept_initial`` defaults to False and should almost always stay there.
    The initial snapshot is the last known state, which before the open is
    yesterday's close; recorded as a tick it becomes today's first trade at a
    price nobody traded today.
    """
    raw_type = payload.get("type")
    try:
        feed_type = FeedType(raw_type) if raw_type else None
    except ValueError:
        feed_type = None

    status: dict[str, SegmentStatus] = {}
    for segment, value in ((payload.get("marketInfo") or {}).get("segmentStatus") or {}).items():
        try:
            status[str(segment)] = SegmentStatus(str(value))
        except ValueError:
            log.warning("unknown segment status", fields={"segment": segment, "status": value})

    current_ts = _ts_from_millis(payload.get("currentTs"))

    ticks: list[Tick] = []
    skipped: list[str] = []

    if feed_type is FeedType.INITIAL and not accept_initial:
        n = len(payload.get("feeds") or {})
        if n:
            skipped.append(
                f"initial_feed snapshot of {n} instruments dropped; it is the last "
                "known state, not a trade that happened now"
            )
        return DecodedFeed([], feed_type, status, skipped, current_ts)

    for key, feed in (payload.get("feeds") or {}).items():
        if not isinstance(feed, dict):
            continue
        instrument_id = keymap.resolve(str(key))

        ltpc = _ltpc(feed)
        ltp = _as_float(ltpc.get("ltp"))
        if ltp is None:
            skipped.append(f"{key}: no ltp in the message")
            continue

        # The trade time, not the feed's clock. currentTs would claim a trade
        # happened now; on an illiquid strike the real answer is hours ago.
        ts = _ts_from_millis(ltpc.get("ltt")) or current_ts
        if ts is None:
            skipped.append(f"{key}: neither ltt nor currentTs, so t_event is unknowable")
            continue

        quote = _first_quote(feed)
        ticks.append(
            Tick(
                ts=ts,
                instrument_id=instrument_id,
                ltp=ltp,
                qty=_as_int(ltpc.get("ltq")),
                bid=_as_float(quote.get("bidP")),
                ask=_as_float(quote.get("askP")),
                bid_qty=_as_int(quote.get("bidQ")),
                ask_qty=_as_int(quote.get("askQ")),
                oi=_open_interest(feed),
                seq=None,  # V3 carries no per-instrument sequence number
                src=SRC_UPSTOX,
            )
        )

    return DecodedFeed(ticks, feed_type, status, skipped, current_ts)


# --------------------------------------------------------------------------- #
# Authorisation
# --------------------------------------------------------------------------- #
def authorize(
    token: str, *, fetch: Callable[[str, dict[str, str]], dict[str, Any]] | None = None
) -> str:
    """Exchange the token for the one-shot wss URL.

    ``fetch`` is injected so the failure branches are testable without a
    network. The 403 message is long on purpose: it is the single most reported
    problem with this endpoint and the bare status code sends people looking at
    their token when the answer is usually elsewhere.
    """
    getter = fetch if fetch is not None else _https_get_json
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        body = getter(AUTHORIZE_URL, headers)
    except PermissionError as exc:
        raise UpstoxAuthError(
            f"{exc}\n"
            "  403 from the feed authorize endpoint. In order of how often it is\n"
            "  the answer:\n"
            "    - the token is an Analytics Token and this account has not been\n"
            "      granted websocket access. The documentation lists the feed as\n"
            "      supported without a static IP, and there are open reports of\n"
            "      exactly this 403 anyway. A daily OAuth access token is the\n"
            "      workaround, at the cost of a login every morning.\n"
            "    - the app has no market-data permission enabled.\n"
            "    - the token is valid but expired; a daily token dies the next\n"
            "      trading day.\n"
            "  The token itself is fine if REST market-data calls work - test one\n"
            "  before assuming this is a credential problem."
        ) from None
    except Exception as exc:
        raise UpstoxAuthError(f"feed authorize failed: {exc}") from None

    url = ((body or {}).get("data") or {}).get("authorized_redirect_uri") or ""
    if not url:
        raise UpstoxAuthError(
            f"authorize returned no authorized_redirect_uri. Body: {body!r}. "
            "Connecting without one is not possible, and an empty string would "
            "fail later somewhere with no context."
        )
    return str(url)


def _https_get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:  # pragma: no cover
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise PermissionError(f"HTTP 403 from {url}") from None
        raise


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #
class UpstoxFeed(MarketFeed):
    """Market Data Feed V3 behind the :class:`MarketFeed` protocol.

    Protobuf decoding happens on the socket thread and so does the conversion
    to :class:`Tick`, for the same reason as in the Angel One adapter: a message
    that cannot be decoded is a problem to see at the message that caused it,
    not a queue of dictionaries that fails later with no context.
    """

    def __init__(
        self,
        token: str,
        keymap: InstrumentKeyMap,
        *,
        mode: str = "full",
        socket_factory: Callable[..., Any] | None = None,
        authorizer: Callable[[str], str] = authorize,
        decoder: Callable[[bytes], dict[str, Any]] | None = None,
        queue_maxsize: int = 20_000,
        accept_initial: bool = False,
    ) -> None:
        envmod.require_broker_credentials()
        if mode not in ("ltpc", "full", "option_greeks"):
            raise ValueError(f"unknown mode {mode!r}")
        self._token = token
        self._keymap = keymap
        self._mode = mode
        self._socket_factory = socket_factory
        self._authorizer = authorizer
        self._decoder = decoder
        self._accept_initial = accept_initial
        self._queue: Queue[Tick] = Queue(maxsize=queue_maxsize)
        self._subscribed: list[str] = []
        self._socket: Any | None = None
        self._closed = False

        self.decode_errors = 0
        self.dropped = 0
        self.skipped = 0
        self.stale_trades = 0
        self.segment_status: dict[str, SegmentStatus] = {}

    @property
    def name(self) -> str:
        return "upstox"

    @property
    def max_instruments(self) -> int:  # type: ignore[override]
        """Mode-dependent, so it cannot be a fixed class attribute.

        `ltpc` carries 5000 per connection and `full` carries 2000. A consumer
        that budgets its option chain against the wrong one either wastes half
        the capacity or plans a subscription the vendor will refuse.
        """
        return MAX_INSTRUMENTS_LTPC if self._mode == "ltpc" else MAX_INSTRUMENTS_FULL

    @property
    def gap_detection(self) -> bool:
        """False. V3 carries no per-instrument sequence number.

        An empty gap list from this feed means "cannot tell", not "none". What
        it does carry instead is :attr:`segment_status`, which answers the
        question gap detection is usually reached for: is the silence a closed
        market or a dead socket.
        """
        return False

    @property
    def market_open(self) -> bool | None:
        """None until the vendor has said. Never assumed."""
        if not self.segment_status:
            return None
        return any(s is SegmentStatus.NORMAL_OPEN for s in self.segment_status.values())

    # -- subscription -------------------------------------------------------
    def subscribe(self, instrument_ids: Sequence[int]) -> None:
        """Protocol entry point, which speaks internal ids.

        The reverse lookup is built here rather than stored, because a keymap
        with two keys for one id is a configuration error worth failing on.
        """
        reverse: dict[int, str] = {}
        for key, iid in self._keymap.to_id.items():
            if iid in reverse:
                raise ValueError(
                    f"instrument id {iid} maps to both {reverse[iid]!r} and {key!r}. "
                    "Ticks would arrive twice under one id and look like volume."
                )
            reverse[iid] = key
        missing = [i for i in instrument_ids if i not in reverse]
        if missing:
            raise UnmappedInstrument(f"no vendor key for instrument ids {missing}")
        self.subscribe_keys([reverse[i] for i in instrument_ids])

    def subscribe_keys(self, keys: Sequence[str]) -> None:
        new = [k for k in dict.fromkeys(keys) if k not in self._subscribed]
        if not new:
            return
        cap = self.max_instruments
        total = len(self._subscribed) + len(new)
        if total > cap:
            raise ValueError(
                f"{total} instruments requested but {self._mode!r} mode allows "
                f"{cap} per connection. Narrow the strike range or drop an "
                "expiry - do not silently truncate, a chain that stops updating "
                "halfway through the morning looks like a quiet market."
            )
        self._subscribed.extend(new)
        if self._socket is not None:
            self._send_subscription(new)

    def _send_subscription(self, keys: Sequence[str]) -> None:
        assert self._socket is not None
        message = {
            "guid": str(uuid.uuid4()),
            "method": "sub",
            "data": {"mode": self._mode, "instrumentKeys": list(keys)},
        }
        self._socket.send(json.dumps(message).encode("utf-8"))

    # -- transport ----------------------------------------------------------
    def _decode(self, raw: bytes) -> dict[str, Any]:
        if self._decoder is not None:
            return self._decoder(raw)
        return _protobuf_decode(raw)  # pragma: no cover - needs the vendor proto

    def _on_message(self, raw: bytes) -> None:
        """Socket-thread callback. Never raises into the transport's thread.

        An exception here is swallowed by the websocket library and the stream
        goes quiet, which is indistinguishable from a closed market. Counting is
        the only evidence there is.
        """
        try:
            payload = self._decode(raw)
            decoded = decode_feed_response(
                payload, self._keymap, accept_initial=self._accept_initial
            )
        except Exception as exc:
            self.decode_errors += 1
            METRICS.inc("feed_decode_error_total", source=self.name)
            log.warning("undecodable message", fields={"error": str(exc)})
            return

        if decoded.segment_status:
            self.segment_status.update(decoded.segment_status)
        self.stale_trades += decoded.stale_trades
        for reason in decoded.skipped:
            self.skipped += 1
            log.warning("feed entry skipped", fields={"reason": reason})

        for tick in decoded.ticks:
            try:
                self._queue.put_nowait(tick)
            except Exception:
                # Blocking here stalls the socket thread, the heartbeat stops,
                # and the vendor closes the connection seconds later. Counting a
                # drop is the smaller failure.
                self.dropped += 1
                METRICS.inc("feed_dropped_total", source=self.name)
            else:
                METRICS.inc("feed_tick_total", source=self.name)

    def connect(self) -> None:
        if self._socket is not None:
            return
        if not self._subscribed:
            raise RuntimeError("subscribe() before connect()")
        url = self._authorizer(self._token)
        if self._socket_factory is None:  # pragma: no cover - needs websocket-client
            self._socket = _default_socket(url, self._on_message)
        else:
            self._socket = self._socket_factory(url=url, on_message=self._on_message)
        self._socket.connect()
        self._send_subscription(self._subscribed)

    def stream(self, *, poll_seconds: float = 0.5) -> Iterator[Tick]:
        if not self._subscribed:
            raise RuntimeError("subscribe() before stream()")
        if self._socket is None:
            self.connect()
        while not self._closed:
            try:
                yield self._queue.get(timeout=poll_seconds)
            except Empty:
                continue

    def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._socket.close()
            self._socket = None


def _protobuf_decode(raw: bytes) -> dict[str, Any]:  # pragma: no cover
    """Decode a V3 frame using the generated protobuf module.

    The schema is published at :data:`PROTO_URL` and ships generated inside the
    official SDK. Imported lazily so that the repository, the tests and CI all
    run without protobuf installed.
    """
    try:
        from google.protobuf.json_format import MessageToDict
        from upstox_client.feeder.proto import MarketDataFeedV3_pb2 as pb
    except ImportError:
        raise UpstoxAuthError(
            'upstox-python-sdk and protobuf are missing. pip install -e ".[broker]"'
        ) from None
    message = pb.FeedResponse()
    message.ParseFromString(raw)
    decoded: dict[str, Any] = MessageToDict(message)
    return decoded


def _default_socket(url: str, on_message: Callable[[bytes], None]) -> Any:  # pragma: no cover
    try:
        import websocket
    except ImportError:
        raise UpstoxAuthError('websocket-client is missing. pip install -e ".[broker]"') from None

    class _Socket:
        def __init__(self) -> None:
            self._ws = websocket.WebSocketApp(
                url,
                on_message=lambda _ws, msg: on_message(
                    msg if isinstance(msg, bytes) else msg.encode("utf-8")
                ),
                on_error=lambda _ws, err: log.warning("socket error", fields={"error": str(err)}),
                on_close=lambda *_a: log.info("socket closed"),
            )
            self._thread: Any | None = None

        def connect(self) -> None:
            import threading

            self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            self._thread.start()

        def send(self, payload: bytes) -> None:
            self._ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)

        def close(self) -> None:
            self._ws.close()

    return _Socket()
