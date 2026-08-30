"""M11 - Angel One SmartAPI market data adapter.

Implements :class:`MarketFeed` over Angel One's SmartAPI. Nothing downstream
imports this module: consumers depend on the protocol, so the vendor is a
one-file decision.

``smartapi-python`` and ``pyotp`` are imported lazily, so the repository
installs, tests and runs without them. Every class here is exercised offline -
the login flow, the price decoding and the tick assembly all take injected
callables, because an adapter that can only be tested against a live broker is
an adapter that is tested on a trading morning.

Five Angel One realities this adapter is built around
-----------------------------------------------------

**Prices arrive in paise, as integers.** A NIFTY tick at 24,180.75 comes over
the wire as ``2418075``. Feeding that number into anything downstream produces a
"working pipeline, plausible-looking numbers and no error" - the exact failure
shape this project keeps finding - except every volatility, every band and every
strike distance is off by two orders of magnitude. :func:`paise_to_rupees` is the
only conversion, and it refuses prices outside a sanity band rather than scaling
whatever it is handed.

**The websocket wants the feed token, not the JWT.** ``generateSession`` returns
three tokens and only ``feedToken`` opens the stream. Passing the JWT is
accepted at connect time and then yields no ticks, which reads as a quiet market
rather than a wrong credential. :class:`AngelSession` therefore keeps them as
separate fields with separate names and no generic ``token`` accessor to grab by
mistake.

**Login is a TOTP, so it happens daily and it can fail at 09:14.** The secret is
the base32 string behind the authenticator QR code, not the six digits it
displays. A six-digit code stored in ``.env`` works once and then fails every
subsequent morning, which reads as a broken adapter.

**There are no sequence numbers.** :class:`GapDetector` is therefore inert
against this feed - not broken, but genuinely unable to see a dropped message.
That is stated here and reported by :attr:`AngelOneFeed.gap_detection` rather
than left for someone to infer from an empty gap list. Data quality falls back
to the candle-count checks in M19, which are weaker and honest.

**Instrument tokens are not stable.** The scrip master is republished daily and
an option's token belongs to one expiry. Resolving a token without a date is the
lookahead bug from M10 wearing a vendor's clothes, so :meth:`InstrumentMaster.
resolve` requires the date it is resolving as of.

The numeric limits below are the documented ones at the time of writing and are
treated as configuration rather than as truth. Verify them against the live API
before Gate G1; vendor limits change without notice.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from time import monotonic, sleep
from typing import Any

from libs.ms_core import env as envmod
from libs.ms_core.clock import UTC, Clock
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, OptionType, Tick
from libs.ms_data.feed import MarketFeed

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_FILE = REPO_ROOT / "data" / ".angel_session.json"
SCRIP_MASTER_CACHE = REPO_ROOT / "data" / "reference" / "angel_scrip_master.json"
SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# Documented SmartAPI limits at the time of writing. Configuration, not truth.
ANGEL_MAX_TOKENS_PER_REQUEST = 1000
ANGEL_MAX_WS_CONNECTIONS = 3

# Websocket subscription modes. 1=LTP, 2=Quote, 3=SnapQuote (adds depth).
MODE_LTP = 1
MODE_QUOTE = 2
MODE_SNAP_QUOTE = 3

# SmartAPI exchange type codes, as sent on the websocket.
EXCHANGE_TYPE = {"NSE_CM": 1, "NSE_FO": 2, "BSE_CM": 3, "BSE_FO": 4, "MCX_FO": 5, "CDE_FO": 7}

# A price outside this band is not a price. NIFTY is five figures, a far OTM
# option is a few paise, and nothing legitimate on these exchanges sits outside
# it - so a value that does is a scaling error, not a market.
MIN_PLAUSIBLE_RUPEES = 0.01
MAX_PLAUSIBLE_RUPEES = 1_000_000.0

# NSE cash and F&O together run to six figures of rows. A master with fewer than
# this is a truncated download, not a quiet day.
MIN_PLAUSIBLE_SCRIP_ROWS = 10_000

# Strike is NOT in paise across the whole scrip master. The currency and
# interest-rate segment (CDS) publishes it scaled by ten million: USDINR's 97.25
# strike arrives as 972500000. Every other segment uses paise.
#
# This was measured, not assumed. Angel's option symbols carry their own strike
# (USDINR26O23_97.25_PE), which makes the file self-checking: splitting each
# symbol on the expiry from its own `expiry` field and comparing the tail
# against the `strike` field gives 9,290 CDS rows at 1e7 with no exceptions,
# and NFO, BFO, MCX and NCO uniformly at 1e2. The reason for 1e7 is not
# documented anywhere the author could find, so it is recorded as an observed
# fact rather than explained.
#
# Left as paise, a currency strike decodes to 9,725,000 instead of 97.25 -
# every moneyness, every strike distance and every surface fit on the CDS
# segment wrong by five orders of magnitude, with nothing raising.
STRIKE_SCALE_BY_SEGMENT = {"CDS": 10_000_000.0}
DEFAULT_STRIKE_SCALE = 100.0

# Anything without a strike - equities, futures, the underlying rows - carries
# -1, not an empty field. Exactly one negative value appears in the whole master
# across 24,035 rows, so this is a sentinel rather than a data error, and it
# must not reach the scale check: -1 paise is a "not a plausible strike" raise
# on the most ordinary rows in the file.
NO_STRIKE_SENTINEL = -1.0

USER_AGENT = "Mozilla/5.0 (compatible; MarketSage research)"


class AngelAuthError(RuntimeError):
    """Raised when there is no usable session."""


class ScripMasterError(RuntimeError):
    """Raised when a downloaded scrip master cannot be trusted.

    A truncated response, or the HTML error page Angel serves on a bad day,
    parses to zero rows without raising anything. Nothing downstream would
    complain either: the feed would subscribe to no tokens and stay silent,
    which is indistinguishable from a closed market. That is the same failure
    shape as the paise bug and it gets the same treatment - refuse at the
    boundary rather than pass on a plausible-looking emptiness.
    """


class PriceScaleError(ValueError):
    """Raised when a decoded price cannot plausibly be a price.

    Deliberately not a warning. The paise-vs-rupees mistake produces numbers
    that look like numbers, and every downstream check - spread, volatility,
    moneyness - stays internally consistent while being 100x wrong.
    """


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AngelCredentials:
    """The four values SmartAPI needs, and nothing else.

    Read from the environment, never from an argument: a credential passed on a
    command line lands in shell history and in the process list of everyone on
    the machine.
    """

    api_key: str
    client_code: str
    pin: str
    totp_secret: str

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> AngelCredentials:
        envmod.require_broker_credentials()
        src = env if env is not None else dict(os.environ)
        names = ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET")
        values = [src.get(n, "").strip() for n in names]
        missing = [n for n, v in zip(names, values, strict=True) if not v]
        if missing:
            raise AngelAuthError(
                f"missing from the environment: {', '.join(missing)}. "
                "Fill them in .env (which is gitignored) and re-run through "
                "`python tasks.py`, which loads it. ANGEL_TOTP_SECRET is the "
                "base32 secret behind the authenticator QR code, not the six "
                "digits it shows - a six-digit code logs in once and then fails "
                "every morning after, which reads as a broken adapter."
            )
        api_key, client_code, pin, totp_secret = values
        if totp_secret.isdigit():
            raise AngelAuthError(
                f"ANGEL_TOTP_SECRET is {len(totp_secret)} digits, so it is a "
                "generated code rather than the secret. The secret is the base32 "
                "string Angel shows once when TOTP is enabled; re-enable TOTP to "
                "see it again. A code expires in thirty seconds."
            )
        return AngelCredentials(api_key, client_code, pin, totp_secret)

    def __repr__(self) -> str:  # pragma: no cover - trivial, but load-bearing
        """Never render the secrets. Repr lands in tracebacks and log lines."""
        return f"AngelCredentials(client_code={self.client_code!r}, ...)"


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class AngelSession:
    """A day's login.

    The three tokens are separate fields on purpose. ``feed_token`` is the only
    one the websocket accepts, and a generic ``token`` attribute is exactly the
    thing that gets passed to the wrong call.
    """

    client_code: str
    jwt_token: str
    refresh_token: str
    feed_token: str
    issued_on: date | None = None

    def is_fresh_for(self, today: date) -> bool:
        """SmartAPI sessions do not survive the night. Same-day means usable."""
        return self.issued_on == today

    def to_json(self) -> str:
        return json.dumps(
            {
                "client_code": self.client_code,
                "jwt_token": self.jwt_token,
                "refresh_token": self.refresh_token,
                "feed_token": self.feed_token,
                "issued_on": self.issued_on.isoformat() if self.issued_on else None,
            },
            indent=2,
        )

    @staticmethod
    def from_json(raw: str) -> AngelSession:
        d = json.loads(raw)
        issued = d.get("issued_on")
        return AngelSession(
            client_code=d["client_code"],
            jwt_token=d["jwt_token"],
            refresh_token=d["refresh_token"],
            feed_token=d["feed_token"],
            issued_on=date.fromisoformat(issued) if issued else None,
        )

    def save(self, path: Path = SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        # The file holds a live trading credential for the rest of the day.
        with contextlib.suppress(OSError, NotImplementedError):
            path.chmod(0o600)

    @staticmethod
    def load(path: Path = SESSION_FILE) -> AngelSession | None:
        if not path.exists():
            return None
        try:
            return AngelSession.from_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            # A corrupt cache is a reason to log in again, not to crash at 09:14.
            log.warning("angel session cache unreadable, ignoring", fields={"path": str(path)})
            return None


def strip_bearer(token: str) -> str:
    """Return the bare JWT, with any ``Bearer `` prefix removed.

    ``smartapi-python`` stores the raw token on the client and then rewrites the
    *returned dict* with a prefix:

        self.setAccessToken(jwtToken)
        user["data"]["jwtToken"] = "Bearer " + jwtToken

    So the value a caller reads out of the response is not the value the client
    is using. Feeding the prefixed one back through ``setAccessToken`` produces
    an ``Authorization: Bearer Bearer eyJ...`` header, and the server answers
    ``Invalid Token`` - which reads as an expired session rather than as a
    doubled prefix, and sends the investigation to the credentials.

    Stored bare here so every later use adds exactly one prefix.
    """
    token = token.strip()
    prefix = "Bearer "
    return token[len(prefix) :].strip() if token.startswith(prefix) else token


def _totp_now(secret: str) -> str:
    try:
        import pyotp
    except ImportError:  # pragma: no cover - depends on the extra
        raise AngelAuthError('pyotp is missing. pip install -e ".[broker]"') from None
    code: str = pyotp.TOTP(secret).now()
    return code


def _vendor_import_error(exc: ImportError) -> AngelAuthError:
    """Report what actually failed, not what is usually wrong.

    ``smartapi-python`` imports ``logzero`` without declaring it, so on a clean
    install the package is present and importing it still raises. Reporting that
    as "smartapi-python is missing" sends the reader to reinstall a package that
    is already there - the fix is named in the exception and nowhere else.
    """
    missing = getattr(exc, "name", "") or "a dependency"
    if missing in ("SmartApi", "smartapi"):
        return AngelAuthError('smartapi-python is missing. pip install -e ".[broker]"')
    return AngelAuthError(
        f"smartapi-python is installed but cannot import: {exc}. The vendor "
        f"package imports {missing!r} without declaring it, so pip does not "
        f"bring it in. Install it directly: pip install {missing}"
    )


def _smartconnect(api_key: str) -> Any:  # pragma: no cover - needs the vendor package
    try:
        from SmartApi import SmartConnect
    except ImportError as exc:
        raise _vendor_import_error(exc) from None
    return SmartConnect(api_key=api_key)


def login(
    creds: AngelCredentials,
    clock: Clock,
    *,
    connect: Callable[[str], Any] | None = None,
    totp: Callable[[str], str] = _totp_now,
    cache: Path | None = SESSION_FILE,
) -> AngelSession:
    """Log in, or reuse today's cached session.

    ``connect`` and ``totp`` are injected so the whole flow is testable without
    a broker account and without a working authenticator.
    """
    envmod.require_broker_credentials()
    today = clock.now().astimezone(UTC).date()

    if cache is not None:
        cached = AngelSession.load(cache)
        if cached is not None and cached.is_fresh_for(today):
            log.info("reusing today's angel session", fields={"client": cached.client_code})
            return cached

    make_client = connect if connect is not None else _smartconnect
    client = make_client(creds.api_key)
    data = client.generateSession(creds.client_code, creds.pin, totp(creds.totp_secret))

    if not isinstance(data, dict) or not data.get("status", False):
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message") or data.get("errorcode") or "")
        raise AngelAuthError(
            f"SmartAPI login failed: {message or data!r}. The usual causes, in "
            "order of how often they happen: the TOTP secret is a generated "
            "code rather than the base32 secret; the PIN is the login PIN and "
            "not the app password; the API key belongs to a different app type "
            "than the one being used."
        )

    payload = data.get("data") or {}
    feed_token = payload.get("feedToken") or ""
    if not feed_token:
        # Better to stop here than to open a socket that will simply stay quiet.
        raise AngelAuthError(
            "login succeeded but returned no feedToken. The websocket accepts "
            "the JWT at connect time and then yields no ticks, which reads as a "
            "quiet market rather than a wrong credential - so this is fatal here."
        )

    session = AngelSession(
        client_code=creds.client_code,
        jwt_token=strip_bearer(payload.get("jwtToken", "")),
        refresh_token=payload.get("refreshToken", ""),
        feed_token=feed_token,
        issued_on=today,
    )
    if cache is not None:
        session.save(cache)
    log.info("angel session established", fields={"client": creds.client_code})
    return session


# --------------------------------------------------------------------------- #
# Price decoding
# --------------------------------------------------------------------------- #
def paise_to_rupees(raw: int | float | None) -> float | None:
    """The one place the 100x lives.

    ``None`` in, ``None`` out - an absent bid is absent, not zero. A zero bid
    makes a one-sided book look two-sided and puts a mid halfway to nothing.
    """
    if raw is None:
        return None
    rupees = float(raw) / 100.0
    if rupees == 0.0:
        return 0.0
    if not (MIN_PLAUSIBLE_RUPEES <= abs(rupees) <= MAX_PLAUSIBLE_RUPEES):
        raise PriceScaleError(
            f"{raw!r} decodes to {rupees:,.2f}, which is not a plausible price on "
            f"these exchanges ({MIN_PLAUSIBLE_RUPEES} to {MAX_PLAUSIBLE_RUPEES:,.0f}). "
            "Either the field is not a price, or it is already in rupees and is "
            "being divided a second time. Do not scale past this."
        )
    return rupees


def _decode_strike(raw: float | None, exch_seg: str) -> float:
    """Strike in rupees, or 0.0 for the instruments that do not have one."""
    if raw is None or raw == NO_STRIKE_SENTINEL:
        return 0.0
    return strike_to_rupees(raw, exch_seg) or 0.0


def strike_to_rupees(raw: int | float | None, exch_seg: str) -> float | None:
    """Decode a scrip-master strike, which is not always in paise.

    Separate from :func:`paise_to_rupees` because the scale is a property of the
    segment rather than of the field, and a single shared constant is exactly
    how the CDS rows came to be read as millions.
    """
    if raw is None:
        return None
    scale = STRIKE_SCALE_BY_SEGMENT.get(exch_seg.upper(), DEFAULT_STRIKE_SCALE)
    rupees = float(raw) / scale
    if rupees == 0.0:
        return 0.0
    if not (MIN_PLAUSIBLE_RUPEES <= abs(rupees) <= MAX_PLAUSIBLE_RUPEES):
        raise PriceScaleError(
            f"{raw!r} on segment {exch_seg!r} decodes to {rupees:,.2f}, which is "
            f"not a plausible strike ({MIN_PLAUSIBLE_RUPEES} to "
            f"{MAX_PLAUSIBLE_RUPEES:,.0f}). Either the segment's scale is not "
            f"{scale:,.0f} or the field is not a strike; do not scale past this."
        )
    return rupees


def tick_from_message(msg: dict[str, Any], *, src: int = 1) -> Tick:
    """Decode one SmartWebSocketV2 payload into a :class:`Tick`.

    ``exchange_timestamp`` is epoch milliseconds and is used in preference to
    arrival time. Stamping a tick with the moment it was parsed folds network
    and GIL latency into the timestamp, and every point-in-time guarantee in
    this system rests on ``t_event`` meaning what it says.
    """
    token = msg.get("token")
    if token is None:
        raise KeyError("websocket payload has no token; cannot attribute the tick")

    epoch_ms = msg.get("exchange_timestamp")
    if epoch_ms is None:
        raise KeyError(
            "websocket payload has no exchange_timestamp. Substituting arrival "
            "time would fold network latency into t_event, and every "
            "point-in-time guarantee downstream reads t_event literally."
        )
    ts = datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=UTC)

    ltp = paise_to_rupees(msg.get("last_traded_price"))
    if ltp is None:
        raise KeyError("websocket payload has no last_traded_price")

    best = msg.get("best_5_buy_data") or []
    ask_book = msg.get("best_5_sell_data") or []
    bid = paise_to_rupees(best[0].get("price")) if best else None
    ask = paise_to_rupees(ask_book[0].get("price")) if ask_book else None

    return Tick(
        ts=ts,
        instrument_id=int(token),
        ltp=ltp,
        qty=msg.get("last_traded_quantity"),
        bid=bid,
        ask=ask,
        bid_qty=best[0].get("quantity") if best else None,
        ask_qty=ask_book[0].get("quantity") if ask_book else None,
        oi=msg.get("open_interest"),
        seq=None,  # SmartAPI carries no sequence number. See the module docstring.
        src=src,
    )


# --------------------------------------------------------------------------- #
# Instrument master
# --------------------------------------------------------------------------- #
@dataclass
class ScripRow:
    token: int
    symbol: str
    name: str
    exch_seg: str
    instrument_type: str
    expiry: date | None
    strike: float
    lot_size: int

    @property
    def opt_type(self) -> OptionType | None:
        """CE or PE, or None when this row is not an option.

        Angel does not carry the right in a field of its own - it is the last
        two characters of the contract symbol. ``instrument_type`` says OPTIDX
        for both sides, so it cannot answer this.
        """
        if not self.instrument_type.startswith("OPT"):
            return None
        tail = self.symbol.strip().upper()[-2:]
        if tail == "CE":
            return OptionType.CE
        if tail == "PE":
            return OptionType.PE
        return None


@dataclass
class InstrumentMaster:
    """The scrip master, resolved as of a date.

    Angel republishes this file daily and an option's token belongs to one
    expiry. Looking a token up without saying *when* is the M10 lookahead bug in
    vendor clothing: a backfill run today would resolve every past session's
    option to this week's token.
    """

    rows: list[ScripRow] = field(default_factory=list)
    fetched_on: date | None = None

    @staticmethod
    def from_payload(payload: Sequence[dict[str, Any]], *, fetched_on: date) -> InstrumentMaster:
        rows: list[ScripRow] = []
        for r in payload:
            try:
                token = int(r["token"])
            except (KeyError, TypeError, ValueError):
                continue
            exch_seg = str(r.get("exch_seg", ""))
            rows.append(
                ScripRow(
                    token=token,
                    symbol=str(r.get("symbol", "")),
                    name=str(r.get("name", "")),
                    exch_seg=exch_seg,
                    instrument_type=str(r.get("instrumenttype", "")),
                    expiry=_parse_expiry(r.get("expiry")),
                    strike=_decode_strike(_as_float(r.get("strike")), exch_seg),
                    lot_size=int(_as_float(r.get("lotsize")) or 0),
                )
            )
        return InstrumentMaster(rows=rows, fetched_on=fetched_on)

    def option_contracts(
        self,
        name: str,
        *,
        as_of: date,
        n_expiries: int = 2,
        exch_seg: str = "NFO",
    ) -> list[ScripRow]:
        """Live option contracts for one underlying, nearest expiries first.

        Selects on ``name`` rather than ``symbol`` because an option's symbol is
        the whole contract - ``NIFTY29DEC2623000CE`` - while ``name`` is the
        underlying it settles against.

        ``as_of`` has no default for the same reason :meth:`resolve` has none. A
        default would be today, and a registry built with today's expiries then
        used to interpret a past session attributes ticks to contracts that had
        not been listed yet. Contracts expiring before ``as_of`` are excluded:
        they are not tradable and their tokens get reused.
        """
        rows = [
            r
            for r in self.rows
            if r.name == name
            and r.exch_seg == exch_seg
            and r.instrument_type.startswith("OPT")
            and r.expiry is not None
            and r.expiry >= as_of
        ]
        if not rows:
            raise KeyError(
                f"no live {exch_seg} option contracts named {name!r} on or after "
                f"{as_of} in a master fetched {self.fetched_on}. Subscribing to "
                "an empty set produces a feed that is silent rather than one "
                "that errors."
            )
        expiries = sorted({r.expiry for r in rows if r.expiry})[:n_expiries]
        keep = {e for e in expiries}
        return sorted(
            (r for r in rows if r.expiry in keep),
            key=lambda r: (r.expiry or date.max, r.strike, r.symbol),
        )

    def resolve(self, symbol: str, *, as_of: date) -> ScripRow:
        """Token for ``symbol``, valid on ``as_of``.

        ``as_of`` has no default. A default would be today, and today is exactly
        the wrong answer for every historical query.
        """
        matches = [r for r in self.rows if r.symbol == symbol]
        if not matches:
            raise KeyError(f"no scrip named {symbol!r} in the master")
        live = [r for r in matches if r.expiry is None or r.expiry >= as_of]
        if not live:
            latest = max(r.expiry for r in matches if r.expiry)
            raise KeyError(
                f"{symbol!r} exists in the master but every contract expired by "
                f"{latest}, and {as_of} is after that. Resolving it anyway would "
                "return a token that traded a different contract."
            )
        return min(live, key=lambda r: (r.expiry or date.max, r.token))

    def save(self, path: Path = SCRIP_MASTER_CACHE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fetched_on": self.fetched_on.isoformat() if self.fetched_on else None,
                    "rows": [
                        {
                            "token": r.token,
                            "symbol": r.symbol,
                            "name": r.name,
                            "exch_seg": r.exch_seg,
                            "instrument_type": r.instrument_type,
                            "expiry": r.expiry.isoformat() if r.expiry else None,
                            "strike": r.strike,
                            "lot_size": r.lot_size,
                        }
                        for r in self.rows
                    ],
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path = SCRIP_MASTER_CACHE) -> InstrumentMaster | None:
        """Read a cached master, or ``None`` when there is no usable one.

        Expiries are read back with :func:`date.fromisoformat`, not with
        :func:`_parse_expiry`: :meth:`save` already normalised them, and running
        the vendor parser over its own output would silently drop every row on
        the day the two formats disagree.
        """
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            rows = [
                ScripRow(
                    token=int(r["token"]),
                    symbol=str(r["symbol"]),
                    name=str(r["name"]),
                    exch_seg=str(r["exch_seg"]),
                    instrument_type=str(r["instrument_type"]),
                    expiry=date.fromisoformat(r["expiry"]) if r.get("expiry") else None,
                    strike=float(r["strike"]),
                    lot_size=int(r["lot_size"]),
                )
                for r in d["rows"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A corrupt cache is a reason to download again, not to crash at 09:14.
            log.warning("angel scrip master cache unreadable, ignoring", fields={"path": str(path)})
            return None
        fetched = d.get("fetched_on")
        return InstrumentMaster(
            rows=rows, fetched_on=date.fromisoformat(fetched) if fetched else None
        )


def _http_get(url: str, *, timeout: float = 60.0) -> bytes:  # pragma: no cover - network
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
        return data


def fetch_scrip_master(
    clock: Clock,
    *,
    getter: Callable[[str], bytes] | None = None,
    url: str = SCRIP_MASTER_URL,
) -> InstrumentMaster:
    """Download the scrip master as published today.

    Deliberately credential-free. The file is public, so the token map can be
    built - and this entire path exercised - before an Angel account exists,
    which is the difference between finding a problem tonight and finding it at
    09:14. ``getter`` is injected for the same reason every other network call
    in this adapter is.
    """
    fetch = getter if getter is not None else _http_get
    raw = fetch(url)
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ScripMasterError(
            f"the scrip master at {url} did not parse as JSON ({exc}). The usual "
            "cause is an HTML error or maintenance page served with a 200, which "
            "would otherwise decode to zero instruments."
        ) from None
    if not isinstance(payload, list):
        raise ScripMasterError(
            f"the scrip master at {url} is a {type(payload).__name__}, not a list "
            "of instruments. The published layout has changed and reading it "
            "anyway would be a guess."
        )
    master = InstrumentMaster.from_payload(payload, fetched_on=clock.now().astimezone(UTC).date())
    if len(master.rows) < MIN_PLAUSIBLE_SCRIP_ROWS:
        raise ScripMasterError(
            f"the scrip master decoded to {len(master.rows):,} rows, below the "
            f"{MIN_PLAUSIBLE_SCRIP_ROWS:,} floor. NSE cash and F&O together are "
            "six figures, so this is a truncated download. Accepting it would "
            "subscribe the feed to a handful of tokens - or none - and a silent "
            "feed reads as a closed market."
        )
    return master


def load_scrip_master(
    clock: Clock,
    *,
    path: Path = SCRIP_MASTER_CACHE,
    getter: Callable[[str], bytes] | None = None,
    url: str = SCRIP_MASTER_URL,
    refresh: bool = False,
) -> InstrumentMaster:
    """Today's scrip master: the cache when it is today's, otherwise downloaded.

    The freshness rule is a correctness rule, not a cache policy. Angel
    republishes this file daily and an option's token belongs to one expiry, so
    a master from before a roll maps a live symbol to a token that now trades a
    different contract - and :meth:`InstrumentMaster.resolve` cannot tell,
    because the row it finds looks entirely well formed.

    The date is taken in UTC, matching :func:`login`. Between midnight and 05:30
    IST that lags the exchange day by one, which costs an extra download and
    never returns a stale master; the opposite bias would.
    """
    today = clock.now().astimezone(UTC).date()
    if not refresh:
        cached = InstrumentMaster.load(path)
        if cached is not None and cached.fetched_on == today:
            log.info("reusing today's angel scrip master", fields={"rows": len(cached.rows)})
            return cached
    master = fetch_scrip_master(clock, getter=getter, url=url)
    master.save(path)
    log.info("angel scrip master downloaded", fields={"rows": len(master.rows)})
    return master


def _as_float(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_expiry(raw: Any) -> date | None:
    """Angel writes expiries as ``26DEC2024``. Equities have none at all."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip().upper(), "%d%b%Y").date()
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #
class AngelOneFeed(MarketFeed):
    """SmartWebSocketV2 behind the :class:`MarketFeed` protocol.

    The socket runs on its own thread and pushes decoded ticks onto a queue,
    which :meth:`stream` drains. Decoding happens on the socket thread on
    purpose: a payload that cannot be decoded is a problem to see once, at the
    tick that caused it, rather than a queue of raw dictionaries that fails
    later somewhere with no context.
    """

    max_instruments = ANGEL_MAX_TOKENS_PER_REQUEST * ANGEL_MAX_WS_CONNECTIONS

    def __init__(
        self,
        session: AngelSession,
        api_key: str,
        *,
        exchange: str = "NSE_FO",
        mode: int = MODE_SNAP_QUOTE,
        socket_factory: Callable[..., Any] | None = None,
        queue_maxsize: int = 20_000,
    ) -> None:
        envmod.require_broker_credentials()
        if exchange not in EXCHANGE_TYPE:
            raise KeyError(f"unknown exchange {exchange!r}; known: {sorted(EXCHANGE_TYPE)}")
        self._session = session
        self._api_key = api_key
        self._exchange = exchange
        self._mode = mode
        self._socket_factory = socket_factory
        self._queue: Queue[Tick] = Queue(maxsize=queue_maxsize)
        # Keyed by exchange segment. The spot and its options do not live on
        # the same one - NIFTY is NSE_CM and its contracts are NSE_FO - and a
        # chain snapshot must carry both from one stream, because a spot
        # joined afterwards drags in a lag that reads as edge later.
        self._subscribed: dict[str, list[int]] = {}
        self._socket: Any | None = None
        self._closed = False
        self.decode_errors = 0
        self.dropped = 0

    @property
    def name(self) -> str:
        return "angelone"

    @property
    def gap_detection(self) -> bool:
        """False, and stated rather than implied.

        SmartAPI sends no sequence number, so a dropped message is invisible
        here. An empty gap list from this feed means "cannot tell", not "none".
        """
        return False

    # -- lifecycle ----------------------------------------------------------
    def _build_socket(self) -> Any:
        if self._socket_factory is not None:
            return self._socket_factory(
                auth_token=self._session.jwt_token,
                api_key=self._api_key,
                client_code=self._session.client_code,
                feed_token=self._session.feed_token,
            )
        try:  # pragma: no cover - requires the vendor package
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError as exc:  # pragma: no cover
            raise _vendor_import_error(exc) from None
        return SmartWebSocketV2(  # pragma: no cover
            self._session.jwt_token,
            self._api_key,
            self._session.client_code,
            self._session.feed_token,
        )

    def _on_data(self, _ws: Any, message: Any) -> None:
        """Socket-thread callback. Never raises into the vendor's thread.

        An exception here is swallowed by the vendor library and the stream goes
        quiet, which is indistinguishable from a closed market. Counting the
        failures and logging them is the only way the condition is visible.
        """
        payloads = message if isinstance(message, list) else [message]
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            try:
                tick = tick_from_message(payload, src=1)
            except (KeyError, ValueError, TypeError) as exc:
                self.decode_errors += 1
                METRICS.inc("feed_decode_error_total", source=self.name)
                log.warning("undecodable tick", fields={"error": str(exc)})
                continue
            try:
                self._queue.put_nowait(tick)
            except Exception:
                # A full queue means the consumer is slower than the market.
                # Dropping the newest tick and counting it beats blocking the
                # socket thread, which would stall the heartbeat and get the
                # connection dropped by the vendor a few seconds later.
                self.dropped += 1
                METRICS.inc("feed_dropped_total", source=self.name)
            else:
                METRICS.inc("feed_tick_total", source=self.name)

    def subscribe(self, instrument_ids: Sequence[int]) -> None:
        """Subscribe on this feed's default exchange."""
        self.subscribe_on(self._exchange, instrument_ids)

    def subscribe_on(self, exchange: str, instrument_ids: Sequence[int]) -> None:
        """Subscribe tokens on a named exchange segment.

        Separate from :meth:`subscribe` because one feed legitimately spans two
        segments: an index quotes on NSE_CM while every option on it quotes on
        NSE_FO. Sending them under one exchangeType is accepted by the API and
        then yields nothing for the tokens that do not belong there - a silence
        indistinguishable from a closed market.
        """
        if exchange not in EXCHANGE_TYPE:
            raise KeyError(f"unknown exchange {exchange!r}; known: {sorted(EXCHANGE_TYPE)}")
        held = self._subscribed.setdefault(exchange, [])
        already = {i for ids in self._subscribed.values() for i in ids}
        new = [i for i in dict.fromkeys(instrument_ids) if i not in already]
        if not new:
            return
        total = len(already) + len(new)
        if total > self.max_instruments:
            raise ValueError(
                f"{total} instruments requested but the budget is "
                f"{self.max_instruments} ({ANGEL_MAX_WS_CONNECTIONS} connections x "
                f"{ANGEL_MAX_TOKENS_PER_REQUEST}). Narrow the strike range or drop "
                "an expiry - do not silently truncate, a chain that stops updating "
                "halfway through the morning looks like a quiet market."
            )
        held.extend(new)
        if self._socket is not None:
            self._push_subscription({exchange: new})

    def _push_subscription(self, by_exchange: dict[str, list[int]]) -> None:
        assert self._socket is not None
        n = 0
        for exchange, tokens in by_exchange.items():
            for i in range(0, len(tokens), ANGEL_MAX_TOKENS_PER_REQUEST):
                chunk = tokens[i : i + ANGEL_MAX_TOKENS_PER_REQUEST]
                self._socket.subscribe(
                    f"{self.name}-{n}",
                    self._mode,
                    [
                        {
                            "exchangeType": EXCHANGE_TYPE[exchange],
                            "tokens": [str(t) for t in chunk],
                        }
                    ],
                )
                n += 1

    def connect(self) -> None:
        if self._socket is not None:
            return
        if not self._subscribed:
            raise RuntimeError("subscribe() before connect()")
        sock = self._build_socket()
        sock.on_data = self._on_data
        sock.on_open = lambda _ws: self._push_subscription(self._subscribed)
        sock.on_error = lambda _ws, err: log.warning("socket error", fields={"error": str(err)})
        sock.on_close = lambda _ws: log.info("socket closed")
        self._socket = sock
        sock.connect()

    def stream(self, *, poll_seconds: float = 0.5) -> Iterator[Tick]:
        if not self._subscribed:
            raise RuntimeError("subscribe() before stream()")
        if self._socket is None:
            self.connect()
        while not self._closed:
            try:
                yield self._queue.get(timeout=poll_seconds)
            except Empty:
                # Silence is ambiguous - a quiet market and a dead socket look
                # identical here. FeedHealth is what resolves it; this loop just
                # keeps going so the heartbeat has something to observe.
                continue

    def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close_connection()
            self._socket = None


# --------------------------------------------------------------------------- #
# Historical candles
# --------------------------------------------------------------------------- #
# The REST historical endpoint returns prices in RUPEES. The websocket returns
# them in PAISE. Both are Angel, both describe the same instrument, and the
# module docstring above is about the second one.
#
# The plausibility band cannot catch a mix-up between them: NIFTY at 24,180 and
# the same figure read as paise, 241.80, are both inside it. So there is no
# guard here that would fire - what there is instead is
# `jobs/angel_history.py --cross-check`, which compares a fetched daily series
# against the ten years of NIFTY daily bars this project already holds from an
# unrelated provider. Two sources agreeing on the level is the only evidence
# available that the scale is right, and it is worth having once rather than
# assuming forever.
HISTORICAL_PRICES_ARE_RUPEES = True

# Angel returns "2026-08-28T09:15:00+05:30" - offset included. Parsed, never
# assumed: an unmarked value read as UTC would place every intraday bar 5.5
# hours early, which is the same error shape as everywhere else in this file.
HISTORY_TZ = timezone(timedelta(hours=5, minutes=30))

# Angel documents 3 requests/second on the historical endpoint. The default here
# is slower, because what was observed is worse than a 429.
#
# Measured on 2026-08-30: roughly four logins and five candle requests inside
# ninety seconds, and every subsequent request returned an **empty body** - not
# a rate-limit status, not an error message, just b"". The vendor SDK then
# raises "Couldn't parse the JSON response", which reads as a broken response
# format rather than as throttling, and it persisted for at least half an hour.
#
# A backfill loop is exactly the shape that trips this: chunked requests issued
# as fast as the network allows. Pacing them is cheaper than discovering the
# block halfway through a ten-year fetch and not knowing why.
HISTORY_MIN_INTERVAL = timedelta(seconds=0.5)


class CandleInterval(StrEnum):
    """Angel's interval names, and how much history each request may span.

    ``max_days`` are the vendor's documented per-request caps at the time of
    writing. They are configuration rather than truth, like the subscription
    limits above: verify them against the live API before relying on a backfill
    plan built from them. Exceeding one is not an error the API explains well -
    it returns a short series, which reads as a quiet period.
    """

    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    TEN_MINUTE = "TEN_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"

    @property
    def minutes(self) -> int:
        return {
            "ONE_MINUTE": 1,
            "THREE_MINUTE": 3,
            "FIVE_MINUTE": 5,
            "TEN_MINUTE": 10,
            "FIFTEEN_MINUTE": 15,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 60,
            "ONE_DAY": 375,  # one Indian equity session, not 1440
        }[self.value]

    @property
    def max_days(self) -> int:
        return {
            "ONE_MINUTE": 30,
            "THREE_MINUTE": 60,
            "FIVE_MINUTE": 100,
            "TEN_MINUTE": 100,
            "FIFTEEN_MINUTE": 200,
            "THIRTY_MINUTE": 200,
            "ONE_HOUR": 400,
            "ONE_DAY": 2000,
        }[self.value]


def _history_fetcher(session: AngelSession, api_key: str) -> Callable[[dict[str, str]], Any]:
    """Bind a SmartConnect to this session for historical calls."""

    def fetch(params: dict[str, str]) -> Any:  # pragma: no cover - needs the vendor package
        client = _smartconnect(api_key)
        client.setAccessToken(session.jwt_token)
        return client.getCandleData(params)

    return fetch


def parse_candle_row(row: Sequence[Any], *, instrument_id: int, tf_minutes: int) -> Candle:
    """One ``[ts, o, h, l, c, v]`` row to a :class:`Candle`.

    ``ts`` is the **left edge** of the bucket, which is what Angel sends and
    what ``Candle`` documents. Reading it as the right edge shifts every bar by
    one interval, and on a one-minute series that is a whole bar of lookahead.
    """
    if len(row) < 6:
        raise ValueError(f"candle row has {len(row)} fields, expected 6: {row!r}")
    ts = datetime.fromisoformat(str(row[0]))
    if ts.tzinfo is None:
        raise ValueError(
            f"candle timestamp {row[0]!r} carries no offset. Reading it as UTC "
            "would place every bar 5.5 hours early; refusing rather than guessing."
        )
    o, h, low, c = (float(row[i]) for i in (1, 2, 3, 4))
    if not (low <= o <= h and low <= c <= h):
        raise ValueError(
            f"candle at {ts.isoformat()} is not internally consistent: "
            f"o={o} h={h} l={low} c={c}. Open and close must lie inside the range."
        )
    return Candle(
        ts=ts.astimezone(UTC),
        instrument_id=instrument_id,
        tf=tf_minutes,
        o=o,
        h=h,
        l=low,
        c=c,
        v=int(row[5]),
    )


def historical_candles(
    session: AngelSession,
    api_key: str,
    *,
    token: int,
    exchange: str,
    interval: CandleInterval,
    start: datetime,
    end: datetime,
    fetch: Callable[[dict[str, str]], Any] | None = None,
    min_interval: timedelta = HISTORY_MIN_INTERVAL,
) -> list[Candle]:
    """Candles for one instrument over a date range, chunked to the API's cap.

    Requests are split at ``interval.max_days`` because exceeding the cap
    returns a **short series rather than an error** - which reads as a quiet
    period and silently truncates a backfill.

    Chunk boundaries can repeat a bar, so the result is deduplicated on
    timestamp and returned in order. Two identical bars in a candle series look
    like a real repeated minute to everything downstream.

    An empty response is not an error: a holiday, a halt, or a range before the
    contract listed all legitimately have no bars.
    """
    envmod.require_broker_credentials()
    if exchange not in EXCHANGE_TYPE:
        raise KeyError(f"unknown exchange {exchange!r}; known: {sorted(EXCHANGE_TYPE)}")
    if end < start:
        raise ValueError(f"end {end.isoformat()} precedes start {start.isoformat()}")

    call = fetch if fetch is not None else _history_fetcher(session, api_key)
    span = timedelta(days=interval.max_days)
    by_ts: dict[datetime, Candle] = {}
    gap = min_interval.total_seconds()
    previous = 0.0

    window_start = start
    while window_start <= end:
        # Paced, not retried. See HISTORY_MIN_INTERVAL: the block this avoids
        # answers with an empty body rather than a status, so a retry loop would
        # keep firing at something that never says stop.
        if previous and gap:
            waited = gap - (monotonic() - previous)
            if waited > 0:
                sleep(waited)
        previous = monotonic()
        window_end = min(window_start + span, end)
        params = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": interval.value,
            "fromdate": window_start.astimezone(HISTORY_TZ).strftime("%Y-%m-%d %H:%M"),
            "todate": window_end.astimezone(HISTORY_TZ).strftime("%Y-%m-%d %H:%M"),
        }
        payload = call(params)
        if not isinstance(payload, dict) or not payload.get("status", False):
            message = ""
            if isinstance(payload, dict):
                message = str(payload.get("message") or payload.get("errorcode") or "")
            raise AngelAuthError(
                f"historical request failed for {token} {interval.value} "
                f"{params['fromdate']}..{params['todate']}: {message or payload!r}"
            )
        for row in payload.get("data") or []:
            candle = parse_candle_row(row, instrument_id=token, tf_minutes=interval.minutes)
            by_ts[candle.ts] = candle
        window_start = window_end + timedelta(minutes=1)

    out = [by_ts[k] for k in sorted(by_ts)]
    METRICS.observe("angel_history_candles", float(len(out)))
    log.info(
        "historical candles fetched",
        fields={"token": token, "interval": interval.value, "candles": len(out)},
    )
    return out
