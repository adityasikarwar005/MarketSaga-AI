"""M14 - NSE F&O bhavcopy: real option prices, free, with no account.

Eight years of daily option data for every listed contract - strike, expiry,
close, settlement, open interest, volume - published by the exchange itself and
downloadable without a broker account, a token or a subscription. Until this
module existed, exactly one component in this repository had ever seen real
market data.

What it is not: it is daily, so there is no intraday here and no live feed. And
it carries no bid or ask, which matters more than it sounds - `OptionQuote.mid`
is what everything downstream is supposed to value against, precisely because
"a stale LTP on an illiquid strike manufactures signal that does not exist".
A bhavcopy row cannot produce a mid. Read the next section before using one as
though it could.

More than half of a day's chain never traded
--------------------------------------------

Measured on the NIFTY chain for 2026-08-26:

    1,584 option contracts listed
      727 actually traded            (46%)
      857 zero volume, and every one of them still carries a close AND a
          settlement price           (54%)

On those untraded contracts the two numbers disagree by a **median of 15.9%**,
78% at the ninetieth percentile, and 2273% at the worst. Both are computed;
neither is a price anybody paid. NSE derives them for contracts that did not
trade, which is the correct thing for an exchange to do and the wrong thing to
fit a volatility surface to.

Fitting SVI to all 1,584 would fit the exchange's own model on more than half
the points, converge beautifully, and report a smile that nothing traded at.
That is this project's recurring failure shape exactly: a working pipeline,
plausible numbers, and no error. So ``traded_only`` defaults to True, the number
dropped is returned rather than logged, and there is no ``price`` property that
quietly picks close or settlement for you - the caller says which and why.

Two formats, and one of them is missing the spot
------------------------------------------------

NSE replaced the old bhavcopy with UDiFF in mid-2024. Both are still served for
their own eras, with different column names for the same quantities and
different date formats (``13-Jul-2023`` against ``2026-09-29``). A column
mapped to the wrong field does not fail; it produces an expiry that is off,
a time-to-expiry that is off, and an implied volatility that is confidently
wrong.

The asymmetry that matters: **UDiFF carries ``UndrlygPric`` and legacy carries
nothing like it.** ``ChainSnapshot`` requires the spot to be captured *with* the
chain, because joining a separately-timestamped spot "introduces a lag you
cannot undo, and it shows up later as a phantom edge in every options
backtest". So a chain built from a legacy file requires the spot to be passed in
explicitly, and says so, rather than reaching for a close from somewhere else.

Point in time
-------------

A bhavcopy is published after the session it describes. ``available_at`` returns
a deliberately conservative bound - the following morning - because the exact
historical publication time is not recoverable, and a bound that is too early is
lookahead while a bound that is too late merely costs a day.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from libs.ms_core.clock import UTC, CalendarNotLoadedError, SessionCalendar
from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
BHAVCOPY_CACHE = REPO_ROOT / "data" / "reference" / "bhavcopy"

UDIFF_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{mon}/fo{day:02d}{mon}{year}bhav.csv.zip"
)
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# The archive serves UDiFF from mid-2024 and legacy before it, with an overlap.
# Verified by probe: 2024-06-11 answers on both, 2023-06-09 only on legacy,
# 2025-06-11 only on UDiFF.
UDIFF_FROM = date(2024, 7, 1)

USER_AGENT = "Mozilla/5.0 (compatible; MarketSage research)"

# A bhavcopy is published in the evening of the session it describes. The exact
# historical time is not recoverable, so the bound is the next morning: too late
# costs a day of latency, too early is lookahead.
PUBLICATION_BOUND_HOUR_IST = 8


class BhavcopyFormat(StrEnum):
    UDIFF = "udiff"
    LEGACY = "legacy"


class NoSessionData(RuntimeError):
    """The exchange published nothing for that date.

    Expected on a holiday and a real problem on a trading day, which is why the
    distinction is drawn by the calendar rather than by the HTTP status.
    """


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptionRecord:
    """One contract on one day, as the exchange published it.

    ``close`` and ``settlement`` are kept apart and neither is promoted to "the
    price". On a contract that traded they usually agree; on one that did not
    they are two different computed numbers, and choosing silently between them
    is how a surface gets fitted to an arithmetic artefact.
    """

    day: date
    symbol: str
    expiry: date
    strike: float
    opt_type: OptionType
    close: float
    settlement: float
    volume: int
    open_interest: int
    oi_change: int
    open: float | None = None
    high: float | None = None
    low: float | None = None
    underlying: float | None = None  # UDiFF only; legacy does not carry it

    @property
    def traded(self) -> bool:
        return self.volume > 0

    @property
    def close_settlement_gap(self) -> float | None:
        """Relative disagreement between the two published numbers.

        On an untraded contract this is the size of the modelling assumption
        being taken on trust. Median 15.9% on a real day.
        """
        if self.settlement <= 0:
            return None
        return abs(self.close - self.settlement) / self.settlement


@dataclass(frozen=True)
class BhavcopyDay:
    day: date
    fmt: BhavcopyFormat
    records: list[OptionRecord]

    def options(
        self,
        symbol: str,
        *,
        expiry: date | None = None,
        traded_only: bool = True,
    ) -> tuple[list[OptionRecord], int]:
        """Contracts for a symbol, and how many were dropped for never trading.

        The count comes back rather than going to a log, because "857 of 1,584
        were dropped" is a fact about the data that belongs in whatever report
        the caller is writing - not a line somebody may or may not read.
        """
        rows = [r for r in self.records if r.symbol == symbol]
        if expiry is not None:
            rows = [r for r in rows if r.expiry == expiry]
        if not traded_only:
            return rows, 0
        kept = [r for r in rows if r.traded]
        return kept, len(rows) - len(kept)

    def expiries(self, symbol: str) -> list[date]:
        return sorted({r.expiry for r in self.records if r.symbol == symbol})


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _num(raw: Any, *, default: float | None = None) -> float | None:
    if raw is None or str(raw).strip() in ("", "-"):
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _parse_udiff_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_legacy_date(raw: str) -> date:
    """``13-Jul-2023``. Not interchangeable with the UDiFF form.

    A misparse here does not raise anywhere downstream - it produces a
    time-to-expiry that is wrong and an implied volatility that is confidently
    wrong with it.
    """
    return datetime.strptime(str(raw).strip(), "%d-%b-%Y").date()


def parse_udiff(text: str, *, day: date | None = None) -> list[OptionRecord]:
    out: list[OptionRecord] = []
    for row in csv.DictReader(text.splitlines()):
        opt = (row.get("OptnTp") or "").strip()
        if opt not in ("CE", "PE"):
            continue  # futures and the rest of the segment
        strike = _num(row.get("StrkPric"))
        settlement = _num(row.get("SttlmPric"))
        close = _num(row.get("ClsPric"))
        if strike is None or strike <= 0 or settlement is None or close is None:
            continue
        trad_dt = row.get("TradDt")
        out.append(
            OptionRecord(
                day=_parse_udiff_date(trad_dt) if trad_dt else (day or date.min),
                symbol=(row.get("TckrSymb") or "").strip(),
                expiry=_parse_udiff_date(row["XpryDt"]),
                strike=strike,
                opt_type=OptionType(opt),
                close=close,
                settlement=settlement,
                volume=int(_num(row.get("TtlTradgVol"), default=0.0) or 0.0),
                open_interest=int(_num(row.get("OpnIntrst"), default=0.0) or 0.0),
                oi_change=int(_num(row.get("ChngInOpnIntrst"), default=0.0) or 0.0),
                open=_num(row.get("OpnPric")),
                high=_num(row.get("HghPric")),
                low=_num(row.get("LwPric")),
                underlying=_num(row.get("UndrlygPric")),
            )
        )
    return out


def parse_legacy(text: str, *, day: date | None = None) -> list[OptionRecord]:
    out: list[OptionRecord] = []
    for row in csv.DictReader(text.splitlines()):
        # OPTION_TYP alone is the discriminator: it is CE or PE on options and
        # blank or XX on everything else, including futures rows that carry a
        # real strike. An INSTRUMENT check on top of it was redundant - mutation
        # testing removed it and no test noticed, which is the definition of a
        # branch that is not doing work.
        opt = (row.get("OPTION_TYP") or "").strip()
        if opt not in ("CE", "PE"):
            continue
        strike = _num(row.get("STRIKE_PR"))
        settlement = _num(row.get("SETTLE_PR"))
        close = _num(row.get("CLOSE"))
        if strike is None or strike <= 0 or settlement is None or close is None:
            continue
        stamp = row.get("TIMESTAMP")
        out.append(
            OptionRecord(
                day=(
                    datetime.strptime(str(stamp).strip(), "%d-%b-%Y").date()
                    if stamp
                    else (day or date.min)
                ),
                symbol=(row.get("SYMBOL") or "").strip(),
                expiry=_parse_legacy_date(row["EXPIRY_DT"]),
                strike=strike,
                opt_type=OptionType(opt),
                close=close,
                settlement=settlement,
                # CONTRACTS is the traded lot count; the legacy file has no
                # share-count volume field, and calling it volume anyway would
                # make the two formats silently incomparable.
                volume=int(_num(row.get("CONTRACTS"), default=0.0) or 0.0),
                open_interest=int(_num(row.get("OPEN_INT"), default=0.0) or 0.0),
                oi_change=int(_num(row.get("CHG_IN_OI"), default=0.0) or 0.0),
                open=_num(row.get("OPEN")),
                high=_num(row.get("HIGH")),
                low=_num(row.get("LOW")),
                underlying=None,  # the legacy file does not publish it
            )
        )
    return out


def parse(text: str, fmt: BhavcopyFormat, *, day: date | None = None) -> list[OptionRecord]:
    return (parse_udiff if fmt is BhavcopyFormat.UDIFF else parse_legacy)(text, day=day)


# --------------------------------------------------------------------------- #
# Point in time
# --------------------------------------------------------------------------- #
def available_at(day: date, calendar: SessionCalendar) -> datetime:
    """When a replay may first release this file. Conservative on purpose.

    The exchange publishes in the evening; the exact historical minute is not
    recoverable. A bound that is too early is lookahead and silently inflates
    every backtest. A bound that is too late costs one day of latency and
    nothing else, so it is the bound that gets chosen.
    """
    _, session_end = calendar.session_bounds(day)
    morning_after = (session_end + timedelta(days=1)).astimezone(calendar.tz)
    return morning_after.replace(
        hour=PUBLICATION_BOUND_HOUR_IST, minute=0, second=0, microsecond=0
    ).astimezone(UTC)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def url_for(day: date, fmt: BhavcopyFormat) -> str:
    if fmt is BhavcopyFormat.UDIFF:
        return UDIFF_URL.format(d=day.strftime("%Y%m%d"))
    mon = _MONTHS[day.month - 1]
    return LEGACY_URL.format(year=day.year, mon=mon, day=day.day)


def formats_for(day: date) -> tuple[BhavcopyFormat, ...]:
    """Preferred format first, the other as a fallback.

    Both are tried because the cutover was a period rather than an instant, and
    a hard date would turn a served file into a missing session.
    """
    if day >= UDIFF_FROM:
        return (BhavcopyFormat.UDIFF, BhavcopyFormat.LEGACY)
    return (BhavcopyFormat.LEGACY, BhavcopyFormat.UDIFF)


def _http_get(url: str, *, timeout: float = 60.0) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
        return data


def unzip_single(raw: bytes) -> str:
    """The archive holds exactly one CSV. More than one is a format change."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise NoSessionData(
            f"expected one CSV in the archive, found {names!r}. The file layout "
            "has changed and parsing the first one would be a guess."
        )
    return z.read(names[0]).decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class BhavcopyStore:
    """Disk cache, one JSON per session.

    Cached rather than re-fetched because the exchange's archive is not a
    high-availability service and a backfill of eight years is two thousand
    requests. The cache is the parsed form, not the zip: a stored zip means
    re-parsing on every read, and re-parsing is where the format bugs are.
    """

    def __init__(
        self,
        root: Path = BHAVCOPY_CACHE,
        *,
        fetcher: Any = None,
        calendar: SessionCalendar | None = None,
    ) -> None:
        self.root = root
        self._fetch = fetcher if fetcher is not None else _http_get
        self.calendar = calendar

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.json"

    def load(self, day: date) -> BhavcopyDay | None:
        p = self.path(day)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return BhavcopyDay(
            day=date.fromisoformat(raw["day"]),
            fmt=BhavcopyFormat(raw["format"]),
            records=[
                OptionRecord(
                    day=date.fromisoformat(r["day"]),
                    symbol=r["symbol"],
                    expiry=date.fromisoformat(r["expiry"]),
                    strike=r["strike"],
                    opt_type=OptionType(r["opt_type"]),
                    close=r["close"],
                    settlement=r["settlement"],
                    volume=r["volume"],
                    open_interest=r["oi"],
                    oi_change=r["oi_change"],
                    open=r.get("open"),
                    high=r.get("high"),
                    low=r.get("low"),
                    underlying=r.get("underlying"),
                )
                for r in raw["records"]
            ],
        )

    def save(self, dayfile: BhavcopyDay) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(dayfile.day)
        p.write_text(
            json.dumps(
                {
                    "day": dayfile.day.isoformat(),
                    "format": dayfile.fmt.value,
                    "records": [
                        {
                            "day": r.day.isoformat(),
                            "symbol": r.symbol,
                            "expiry": r.expiry.isoformat(),
                            "strike": r.strike,
                            "opt_type": r.opt_type.value,
                            "close": r.close,
                            "settlement": r.settlement,
                            "volume": r.volume,
                            "oi": r.open_interest,
                            "oi_change": r.oi_change,
                            "open": r.open,
                            "high": r.high,
                            "low": r.low,
                            "underlying": r.underlying,
                        }
                        for r in dayfile.records
                    ],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return p

    def get(self, day: date, *, refresh: bool = False) -> BhavcopyDay:
        if not refresh:
            cached = self.load(day)
            if cached is not None:
                return cached

        errors: list[str] = []
        for fmt in formats_for(day):
            url = url_for(day, fmt)
            try:
                raw = self._fetch(url)
            except urllib.error.HTTPError as exc:
                errors.append(f"{fmt.value}: HTTP {exc.code}")
                continue
            except Exception as exc:  # reported below, never swallowed
                errors.append(f"{fmt.value}: {type(exc).__name__}")
                continue
            records = parse(unzip_single(raw), fmt, day=day)
            if not records:
                errors.append(f"{fmt.value}: parsed zero option rows")
                continue
            dayfile = BhavcopyDay(day=day, fmt=fmt, records=records)
            self.save(dayfile)
            log.info(
                "bhavcopy stored",
                fields={"day": day.isoformat(), "format": fmt.value, "records": len(records)},
            )
            return dayfile

        # Three states, not two. "Cannot answer" is different from "no" and the
        # difference decides whether a missing day is a holiday or a hole. The
        # calendar refuses to guess an unloaded year by design, and that refusal
        # must not surface as a LookupError from inside an error path.
        trading: bool | None = None
        unloaded = False
        if self.calendar is not None:
            try:
                trading = self.calendar.is_trading_day(day)
            except CalendarNotLoadedError:
                unloaded = True
        detail = "; ".join(errors)
        if trading is False:
            raise NoSessionData(f"{day} is not a trading day, so there is no bhavcopy ({detail})")
        raise NoSessionData(
            f"no bhavcopy for {day} ({detail}). "
            + (
                "The calendar says this WAS a trading day, so this is a gap in the "
                "data and not a holiday - do not backfill around it silently."
                if trading is True
                else (
                    f"The calendar has no holiday list for {day.year}, so whether "
                    "this is a holiday or a missing session is unknown. Run "
                    "`python tasks.py holidays` to discover it."
                    if unloaded
                    else "No calendar was supplied, so whether this is a holiday or "
                    "a missing session is unknown. Pass one to find out."
                )
            )
        )


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvexityCheck:
    """How much of a day's marks price a negative butterfly.

    A call (or put) price must be convex in strike: for three equally spaced
    strikes, ``C(K1) - 2*C(K2) + C(K3) >= 0``. Where it is not, the butterfly
    spread built from those three contracts has negative cost and unbounded
    payoff - so the three prices cannot all have been available at once.

    This is a measurement of the *data*, taken before any model touches it, and
    it is the number that says whether a surface fitted on these marks can mean
    anything. Fitting first and reading the fit's arbitrage report instead
    conflates two questions: whether the fitter works, and whether its inputs
    were mutually consistent to begin with.
    """

    triples: int
    violations: int
    worst: float
    worst_detail: str

    @property
    def rate(self) -> float:
        return self.violations / self.triples if self.triples else 0.0

    def render(self) -> list[str]:
        if not self.triples:
            return ["    no equally-spaced strike triples to check"]
        return [
            f"    {self.violations:,} of {self.triples:,} equally-spaced strike "
            f"triples price a NEGATIVE butterfly ({self.rate:.1%}).",
            f"    worst {self.worst:.2f}: {self.worst_detail}",
        ]


def negative_butterflies(
    records: Sequence[OptionRecord], *, price_field: str = "settlement"
) -> ConvexityCheck:
    """Convexity in strike, per expiry and option type, on the raw marks.

    Only equally spaced triples are tested. NIFTY lists 50-point strikes near
    the money and 100-point strikes in the wings, and a triple that straddles
    the change is not required to be convex by this inequality - counting it
    would manufacture violations that are an artefact of the strike ladder.
    """
    if price_field not in ("settlement", "close"):
        raise ValueError(f"price_field must be 'settlement' or 'close', not {price_field!r}")

    groups: dict[tuple[date, OptionType], list[tuple[float, float]]] = {}
    for r in records:
        px = r.settlement if price_field == "settlement" else r.close
        groups.setdefault((r.expiry, r.opt_type), []).append((r.strike, px))

    triples = violations = 0
    worst = 0.0
    detail = "none"
    for (expiry, opt_type), rows in groups.items():
        rows.sort()
        for (k1, c1), (k2, c2), (k3, c3) in zip(rows, rows[1:], rows[2:], strict=False):
            if abs((k2 - k1) - (k3 - k2)) > 1e-6:
                continue
            triples += 1
            butterfly = c1 - 2.0 * c2 + c3
            if butterfly < -1e-9:
                violations += 1
                if butterfly < worst:
                    worst = butterfly
                    detail = (
                        f"{expiry} {opt_type} {k1:.0f}/{k2:.0f}/{k3:.0f} "
                        f"settling {c1:.2f}/{c2:.2f}/{c3:.2f}"
                    )
    return ConvexityCheck(triples=triples, violations=violations, worst=worst, worst_detail=detail)


def chain_from_records(
    records: Sequence[OptionRecord],
    *,
    underlying_id: int,
    ts: datetime,
    spot: float | None = None,
    price_field: str = "settlement",
) -> ChainSnapshot:
    """A :class:`ChainSnapshot` from one day's contracts.

    ``spot`` may be omitted only when the records carry ``underlying`` - which
    UDiFF does and the legacy format does not. There is deliberately no fallback
    to a close from elsewhere: ``ChainSnapshot`` exists on the premise that the
    spot was captured with the chain, and a separately-timestamped spot "shows
    up later as a phantom edge in every options backtest".

    ``price_field`` has no safe default worth hiding. ``settlement`` is what the
    exchange marks positions at; ``close`` is the last traded price where one
    exists. They diverge by a median 15.9% on contracts that did not trade.
    """
    if price_field not in ("settlement", "close"):
        raise ValueError(f"price_field must be 'settlement' or 'close', not {price_field!r}")
    if not records:
        raise ValueError("cannot build a chain from zero records")

    if spot is None:
        spots = {r.underlying for r in records if r.underlying is not None and r.underlying > 0}
        if not spots:
            raise ValueError(
                "no spot: these records carry no underlying price, which means "
                "they came from the legacy bhavcopy format. Pass spot= from a "
                "source timestamped with this session - do not join one from "
                "elsewhere, the lag cannot be undone and reads as edge later."
            )
        if len(spots) > 1:
            raise ValueError(
                f"records disagree about the underlying: {sorted(spots)}. They are "
                "not all the same instrument, or not all the same day."
            )
        spot = spots.pop()

    quotes = [
        OptionQuote(
            expiry=datetime.combine(r.expiry, datetime.min.time(), tzinfo=UTC),
            strike=r.strike,
            opt_type=r.opt_type,
            ltp=(r.settlement if price_field == "settlement" else r.close),
            bid=None,  # a bhavcopy has no book. mid() is correctly unavailable.
            ask=None,
            volume=r.volume,
            oi=r.open_interest,
            oi_change=r.oi_change,
        )
        for r in records
    ]
    return ChainSnapshot(ts=ts, underlying_id=underlying_id, spot=spot, quotes=quotes)
