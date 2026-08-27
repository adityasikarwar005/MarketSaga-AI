"""M11 - Zerodha Kite adapter.

Implements :class:`MarketFeed` over Kite Connect. Nothing downstream imports
this module: consumers depend on the protocol, so replacing the broker later is
a one-file change.

``kiteconnect`` is imported lazily, so the repository installs, tests and runs
without it - the synthetic feed covers the pipeline until credentials exist.

Three Kite realities this adapter is built around
-------------------------------------------------

**The access token expires daily.** Kite issues a token that dies each morning.
This is not a key you set once; it is a login you perform every trading day. The
adapter treats a missing or expired token as an ordinary, expected condition
with a clear remedy, rather than a crash at 09:14.

**There is no option chain endpoint.** A chain is assembled by subscribing to
the individual option instrument tokens and grouping them. That consumes
subscription budget quickly, so :class:`SubscriptionPlan` is used rather than
hoping the limit is generous.

**Ticks are throttled snapshots, not tick-by-tick.** Roughly one update per
second per instrument. Entirely adequate for 1-30 minute horizons; measure the
actual rate yourself with ``jobs/measure_feed_rate.py`` rather than trusting any
documented figure, including this docstring.

Verify the current limits and endpoints against the official documentation
before Gate G1 - vendor constraints change without notice.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import webbrowser
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from libs.ms_core import env as envmod
from libs.ms_core.clock import UTC, Clock
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Tick
from libs.ms_data.feed import GapDetector, MarketFeed, SubscriptionPlan

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = REPO_ROOT / "data" / ".kite_session.json"
INSTRUMENT_CACHE = REPO_ROOT / "data" / "reference" / "kite_instruments.csv"

# Documented Kite limits at the time of writing. Treated as configuration, not
# as truth: `python jobs/check_feed_limits.py` re-verifies them against the API.
KITE_MAX_INSTRUMENTS_PER_WS = 3000
KITE_MAX_WS_CONNECTIONS = 3


class KiteAuthError(RuntimeError):
    """Raised when there is no usable access token."""


# --------------------------------------------------------------------------- #
# Session / daily login
# --------------------------------------------------------------------------- #
@dataclass
class KiteSession:
    api_key: str
    access_token: str
    public_token: str | None = None
    user_id: str | None = None
    issued_on: date | None = None

    def is_fresh_for(self, today: date) -> bool:
        """Kite tokens do not survive the night. Same-day means usable."""
        return self.issued_on == today

    def to_json(self) -> str:
        return json.dumps(
            {
                "api_key": self.api_key,
                "access_token": self.access_token,
                "public_token": self.public_token,
                "user_id": self.user_id,
                "issued_on": self.issued_on.isoformat() if self.issued_on else None,
            },
            indent=2,
        )

    @classmethod
    def load(cls, path: Path = TOKEN_FILE) -> KiteSession | None:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        issued = raw.get("issued_on")
        return cls(
            api_key=raw["api_key"],
            access_token=raw["access_token"],
            public_token=raw.get("public_token"),
            user_id=raw.get("user_id"),
            issued_on=date.fromisoformat(issued) if issued else None,
        )

    def save(self, path: Path = TOKEN_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        # The token is a live credential. Keep it out of git and off the terminal.
        log.info("kite session saved", fields={"path": str(path), "user_id": self.user_id})


def build_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Kite's login checksum: sha256(api_key + request_token + api_secret)."""
    return hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()


def login_url(api_key: str) -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def interactive_login(
    api_key: str,
    api_secret: str,
    clock: Clock,
    *,
    open_browser: bool = True,
    request_token: str | None = None,
) -> KiteSession:
    """Complete the daily login and persist the session.

    Kite's flow: open the login URL, authenticate, and Kite redirects to the app's
    configured redirect URL carrying a ``request_token``. That token is exchanged,
    with the api_secret, for the day's access token.

    Pass ``request_token`` to skip the browser step (useful when the redirect is
    captured by a local handler or pasted manually).
    """
    envmod.require_broker_credentials()

    if request_token is None:
        url = login_url(api_key)
        print(
            f"\n  Open this URL, log in, then paste the request_token from the redirect:\n\n  {url}\n"
        )
        if open_browser:
            with contextlib.suppress(Exception):  # headless environments
                webbrowser.open(url)
        request_token = input("  request_token: ").strip()

    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise ImportError(
            "kiteconnect is not installed. pip install kiteconnect\n"
            "(kept out of the base dependencies so the repo runs without a broker)"
        ) from exc

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    session = KiteSession(
        api_key=api_key,
        access_token=data["access_token"],
        public_token=data.get("public_token"),
        user_id=data.get("user_id"),
        issued_on=clock.now().astimezone(UTC).date(),
    )
    session.save()
    return session


def require_session(clock: Clock, *, path: Path = TOKEN_FILE) -> KiteSession:
    """Load today's session or explain exactly how to get one.

    Called at the top of every service that touches the broker, so a stale token
    produces a one-line instruction at start-up instead of an authentication
    error in the middle of the opening auction.
    """
    session = KiteSession.load(path)
    if session is None:
        raise KiteAuthError(
            "no Kite session found. Run:  python tasks.py login\n"
            "(Kite issues a fresh access token every day - this is expected, not a bug.)"
        )
    today = clock.now().astimezone(UTC).date()
    if not session.is_fresh_for(today):
        raise KiteAuthError(
            f"Kite session was issued on {session.issued_on} and expires overnight. "
            "Run:  python tasks.py login"
        )
    return session


# --------------------------------------------------------------------------- #
# Instrument master sync
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KiteInstrument:
    instrument_token: int
    exchange_token: int
    tradingsymbol: str
    name: str
    expiry: date | None
    strike: float
    tick_size: float
    lot_size: int
    instrument_type: str  # EQ / FUT / CE / PE
    segment: str
    exchange: str


def parse_instrument_dump(rows: Sequence[dict[str, Any]]) -> list[KiteInstrument]:
    out: list[KiteInstrument] = []
    for r in rows:
        expiry_raw = str(r.get("expiry") or "").strip()
        expiry: date | None = None
        if expiry_raw:
            try:
                expiry = date.fromisoformat(expiry_raw[:10])
            except ValueError:
                expiry = None
        out.append(
            KiteInstrument(
                instrument_token=int(r["instrument_token"]),
                exchange_token=int(r.get("exchange_token") or 0),
                tradingsymbol=str(r["tradingsymbol"]),
                name=str(r.get("name") or ""),
                expiry=expiry,
                strike=float(r.get("strike") or 0.0),
                tick_size=float(r.get("tick_size") or 0.0),
                lot_size=int(r.get("lot_size") or 0),
                instrument_type=str(r.get("instrument_type") or ""),
                segment=str(r.get("segment") or ""),
                exchange=str(r.get("exchange") or ""),
            )
        )
    return out


def load_cached_instruments(path: Path = INSTRUMENT_CACHE) -> list[KiteInstrument]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python tasks.py sync-instruments\n"
            "The dump carries real lot sizes and tick sizes - it is what clears the "
            "'unverified instrument' warning in config."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return parse_instrument_dump(list(csv.DictReader(fh)))


def select_option_chain(
    instruments: Sequence[KiteInstrument],
    underlying: str,
    spot: float,
    *,
    n_expiries: int = 2,
    strikes_per_side: int = 15,
    on_or_after: date | None = None,
) -> list[KiteInstrument]:
    """Pick the option tokens worth subscribing to.

    The full chain is far larger than the subscription budget, and the far wings
    are illiquid enough that they contribute noise rather than signal. Selecting
    a window around the money is a deliberate trade-off, recorded here rather
    than buried in a service.
    """
    opts = [
        i
        for i in instruments
        if i.name == underlying
        and i.instrument_type in ("CE", "PE")
        and i.expiry is not None
        and (on_or_after is None or i.expiry >= on_or_after)
    ]
    if not opts:
        return []
    expiries = sorted({i.expiry for i in opts if i.expiry})[:n_expiries]
    strikes = sorted({i.strike for i in opts if i.expiry in expiries})
    if not strikes:
        return []
    atm = min(strikes, key=lambda s: abs(s - spot))
    atm_idx = strikes.index(atm)
    lo = max(0, atm_idx - strikes_per_side)
    hi = min(len(strikes), atm_idx + strikes_per_side + 1)
    window = set(strikes[lo:hi])
    return [i for i in opts if i.expiry in expiries and i.strike in window]


# --------------------------------------------------------------------------- #
# Feed adapter
# --------------------------------------------------------------------------- #
class KiteFeed(MarketFeed):
    """Live ticks from KiteTicker, normalised into :class:`Tick`.

    The websocket delivers on its own thread; ticks are handed to the consumer
    through a bounded queue. Bounded, not unbounded: if a consumer stalls, the
    correct behaviour is to drop ticks and say so loudly, not to accumulate a
    multi-gigabyte backlog and fail at an unrelated place an hour later.
    """

    max_instruments = KITE_MAX_INSTRUMENTS_PER_WS

    def __init__(
        self,
        session: KiteSession,
        clock: Clock,
        *,
        queue_size: int = 50_000,
        mode: str = "full",
        ticker: Any | None = None,
    ) -> None:
        envmod.require_broker_credentials()
        self._session = session
        self._clock = clock
        self._queue: Queue[Tick] = Queue(maxsize=queue_size)
        self._tokens: list[int] = []
        self._mode = mode
        self._closed = False
        self._gaps = GapDetector()
        self._plan = SubscriptionPlan(KITE_MAX_INSTRUMENTS_PER_WS, KITE_MAX_WS_CONNECTIONS)
        self._ticker = ticker  # injectable for tests
        self.dropped = 0

    @property
    def name(self) -> str:
        return "kite"

    # -- lifecycle ----------------------------------------------------------
    def _build_ticker(self) -> Any:
        try:
            from kiteconnect import KiteTicker
        except ImportError as exc:
            raise ImportError("kiteconnect is not installed. pip install kiteconnect") from exc
        return KiteTicker(self._session.api_key, self._session.access_token)

    def subscribe(self, instrument_ids: Sequence[int]) -> None:
        """``instrument_ids`` are Kite instrument tokens."""
        self._plan.plan(list(self._tokens) + list(instrument_ids))
        for t in instrument_ids:
            if t not in self._tokens:
                self._tokens.append(t)
        METRICS.gauge("feed_subscribed_instruments", len(self._tokens), source=self.name)
        log.info(
            "kite subscription planned",
            fields={
                "instruments": len(self._tokens),
                "connections": len(self._plan.assignments),
                "utilisation": round(self._plan.utilisation(), 3),
            },
        )

    def _on_ticks(self, _ws: Any, raw_ticks: list[dict[str, Any]]) -> None:
        for raw in raw_ticks:
            tick = self.normalise(raw, self._clock.now())
            if tick is None:
                continue
            self._gaps.check(tick)
            try:
                self._queue.put_nowait(tick)
            except Exception:
                self.dropped += 1
                METRICS.inc("feed_dropped_total", source=self.name)

    def normalise(self, raw: dict[str, Any], received_at: datetime) -> Tick | None:
        """Map a Kite tick dict onto our schema.

        ``exchange_timestamp`` is preferred over arrival time whenever Kite
        supplies it: the exchange's own clock is the closest thing to ground
        truth we have, and point-in-time correctness depends on it.
        """
        token = raw.get("instrument_token")
        ltp = raw.get("last_price")
        if token is None or ltp is None:
            return None

        ts = raw.get("exchange_timestamp") or received_at
        if isinstance(ts, datetime) and ts.tzinfo is None:
            # Kite hands back naive datetimes in exchange local time.
            ts = ts.replace(tzinfo=self._exchange_tz())

        depth = raw.get("depth") or {}
        buy = (depth.get("buy") or [{}])[0]
        sell = (depth.get("sell") or [{}])[0]

        return Tick(
            ts=ts.astimezone(UTC) if isinstance(ts, datetime) else received_at,
            instrument_id=int(token),
            ltp=float(ltp),
            qty=raw.get("last_traded_quantity"),
            bid=float(buy["price"]) if buy.get("price") else None,
            ask=float(sell["price"]) if sell.get("price") else None,
            bid_qty=buy.get("quantity"),
            ask_qty=sell.get("quantity"),
            oi=raw.get("oi"),
            seq=None,  # Kite does not expose a sequence number
            src=1,
        )

    @staticmethod
    def _exchange_tz() -> Any:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Kolkata")

    def start(self) -> None:
        if self._ticker is None:
            self._ticker = self._build_ticker()
        tk = self._ticker

        def on_connect(ws: Any, _response: Any) -> None:
            log.info("kite websocket connected", fields={"instruments": len(self._tokens)})
            ws.subscribe(self._tokens)
            ws.set_mode(self._mode, self._tokens)
            # Sequence numbers, if any, restart on reconnect.
            self._gaps.reset()

        def on_close(_ws: Any, code: Any, reason: Any) -> None:
            log.warning("kite websocket closed", fields={"code": code, "reason": str(reason)})
            METRICS.inc("feed_disconnect_total", source=self.name)

        def on_error(_ws: Any, code: Any, reason: Any) -> None:
            log.error("kite websocket error", fields={"code": code, "reason": str(reason)})

        tk.on_ticks = self._on_ticks
        tk.on_connect = on_connect
        tk.on_close = on_close
        tk.on_error = on_error
        tk.connect(threaded=True)

    def stream(self) -> Iterator[Tick]:
        while not self._closed:
            try:
                yield self._queue.get(timeout=1.0)
            except Empty:
                continue

    def close(self) -> None:
        self._closed = True
        if self._ticker is not None:
            with contextlib.suppress(Exception):
                self._ticker.close()

    @property
    def gaps(self) -> GapDetector:
        return self._gaps
