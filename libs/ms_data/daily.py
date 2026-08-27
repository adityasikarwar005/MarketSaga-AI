"""Daily bars from a free public source.

This exists so the overnight gap forecaster can be trained on real sessions
before any broker account exists. It is the smallest possible bridge to real
data: daily OHLC for the index and for the cross-asset series that drive the
overnight window.

Two things to be clear about.

**The source is unofficial and replaceable.** The provider is behind an
interface for the same reason the broker is: a free public endpoint can change,
rate-limit or disappear, and terms of use restrict redistribution. It is
adequate for personal research on daily bars and is not the foundation for
anything that trades. Swap in a licensed feed when there is one.

**Everything is cached.** Re-fetching a two-year history on every run is slow and
rude to the source. The cache is keyed by symbol, written to
``data/reference/daily/``, and refreshed only when asked.

The leak this module is built to prevent
----------------------------------------

With daily data, "overnight" has to be inferred rather than measured, and the
obvious inference is wrong. To predict Nifty's open on day D:

* The US session on day D closes at about 02:30 IST on day **D+1** - long after
  Nifty has opened. Using the US close on D is reading tomorrow's newspaper.
* Tokyo opens around 05:30 IST and closes around 11:30 IST on D. Its open is
  knowable at the 08:45 cutoff; its close is not.

So every foreign series contributes its **D-1** return and nothing later. That
is conservative - it discards a few genuinely-knowable hours of Asian trading -
and it is enforced by construction below rather than by comment, because a
convention this easy to violate will eventually be violated.
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DAILY_CACHE = REPO_ROOT / "data" / "reference" / "daily"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval=1d"

# Our cross-asset symbols mapped to the public source's tickers. Kept here
# rather than in config/crossasset.yaml because it is a property of this
# particular provider, not of the series.
#
# GIFT_NIFTY is deliberately absent. There is no free daily proxy for it, and
# the tempting substitute - the Nifty index itself - is a different variable
# wearing the same name: it would make the "overnight driver" nothing more than
# Nifty's own previous session, and the naive baseline built on it would look
# strong for entirely the wrong reason. A series we cannot source is reported as
# missing, not quietly replaced.
YAHOO_SYMBOLS: dict[str, str] = {
    "NIFTY": "%5ENSEI",
    "BANKNIFTY": "%5ENSEBANK",
    "US_EQ_FUT": "ES%3DF",
    "BRENT": "BZ%3DF",
    "GOLD": "GC%3DF",
    "DXY": "DX-Y.NYB",
    "USDINR": "INR%3DX",
    "US10Y": "%5ETNX",
    "ASIA_IDX": "%5EN225",
}


@dataclass(frozen=True)
class DailyBar:
    day: date
    o: float
    h: float
    low: float
    c: float
    v: int = 0

    @property
    def ret(self) -> float:
        return (self.c / self.o - 1.0) if self.o > 0 else 0.0

    @property
    def range_pct(self) -> float:
        return ((self.h - self.low) / self.o) if self.o > 0 else 0.0


class DailyBarProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch(self, symbol: str, *, years: int = 2) -> list[DailyBar]: ...


class YahooDailyProvider(DailyBarProvider):
    """Public chart endpoint. Unofficial, uncontracted, adequate for research."""

    def __init__(self, symbols: Mapping[str, str] | None = None, timeout: float = 20.0) -> None:
        self.symbols = dict(symbols or YAHOO_SYMBOLS)
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "yahoo_daily"

    def fetch(self, symbol: str, *, years: int = 2) -> list[DailyBar]:
        ticker = self.symbols.get(symbol)
        if ticker is None:
            raise KeyError(f"no public ticker mapped for {symbol!r}")
        url = YAHOO_CHART.format(symbol=ticker, range=f"{years}y")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return parse_yahoo(payload)


def parse_yahoo(payload: dict[str, Any]) -> list[DailyBar]:
    """Turn the chart payload into daily bars, dropping incomplete rows.

    Rows with a null close are common - holidays, halts, and the current
    unfinished session. Carrying them forward would invent prices; dropping them
    is the only honest option, and the caller sees a shorter series rather than
    a smooth one.
    """
    result = (payload.get("chart") or {}).get("result")
    if not result:
        return []
    block = result[0]
    stamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]

    opens, highs, lows, closes = (
        quote.get("open") or [],
        quote.get("high") or [],
        quote.get("low") or [],
        quote.get("close") or [],
    )
    volumes = quote.get("volume") or []

    out: list[DailyBar] = []
    for i, stamp in enumerate(stamps):
        try:
            o, h, low, c = opens[i], highs[i], lows[i], closes[i]
        except IndexError:
            continue
        if None in (o, h, low, c) or o <= 0 or c <= 0:
            continue
        out.append(
            DailyBar(
                day=datetime.fromtimestamp(stamp, tz=UTC).date(),
                o=float(o),
                h=float(h),
                low=float(low),
                c=float(c),
                v=int(volumes[i]) if i < len(volumes) and volumes[i] else 0,
            )
        )
    out.sort(key=lambda b: b.day)
    return out


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class DailyStore:
    """Disk cache for daily series, one JSON per symbol."""

    def __init__(self, root: Path = DAILY_CACHE, provider: DailyBarProvider | None = None) -> None:
        self.root = root
        self.provider = provider or YahooDailyProvider()

    def path(self, symbol: str) -> Path:
        return self.root / f"{symbol}.json"

    def load(self, symbol: str) -> list[DailyBar]:
        p = self.path(symbol)
        if not p.exists():
            return []
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [
            DailyBar(
                day=date.fromisoformat(r["day"]),
                o=r["o"],
                h=r["h"],
                low=r["l"],
                c=r["c"],
                v=r.get("v", 0),
            )
            for r in raw["bars"]
        ]

    def save(self, symbol: str, bars: Sequence[DailyBar], *, source: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(symbol)
        p.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "source": source,
                    "bars": [
                        {
                            "day": b.day.isoformat(),
                            "o": b.o,
                            "h": b.h,
                            "l": b.low,
                            "c": b.c,
                            "v": b.v,
                        }
                        for b in bars
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return p

    def get(self, symbol: str, *, years: int = 2, refresh: bool = False) -> list[DailyBar]:
        if not refresh:
            cached = self.load(symbol)
            if cached:
                return cached
        bars = self.provider.fetch(symbol, years=years)
        if bars:
            self.save(symbol, bars, source=self.provider.name)
        log.info(
            "daily series fetched",
            fields={"symbol": symbol, "bars": len(bars), "source": self.provider.name},
        )
        return bars

    def symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))


# --------------------------------------------------------------------------- #
# Overnight state from daily bars
# --------------------------------------------------------------------------- #
@dataclass
class DailyOvernight:
    """What the world did before an Indian session, from daily bars only."""

    session: date
    previous_session: date
    gap: float  # (open - previous close) / previous close  <- the target
    prior_session_return: float
    calendar_days: int
    foreign_returns: dict[str, float] = field(default_factory=dict)
    foreign_ranges: dict[str, float] = field(default_factory=dict)

    @property
    def spans_multiple_days(self) -> bool:
        return self.calendar_days > 1


def build_overnight_series(
    index_bars: Sequence[DailyBar],
    foreign: Mapping[str, Sequence[DailyBar]],
) -> list[DailyOvernight]:
    """Pair each Indian session's gap with the foreign moves that preceded it.

    Every foreign series contributes its most recent session **strictly before**
    the Indian session being predicted. The lookup below is the enforcement: it
    cannot return a bar dated on or after the target session, so no amount of
    later carelessness can pull tomorrow's US close into today's forecast.
    """
    index_sorted = sorted(index_bars, key=lambda b: b.day)
    foreign_sorted = {k: sorted(v, key=lambda b: b.day) for k, v in foreign.items()}

    def most_recent_before(bars: Sequence[DailyBar], day: date) -> DailyBar | None:
        prior = [b for b in bars if b.day < day]
        return prior[-1] if prior else None

    out: list[DailyOvernight] = []
    for prev, current in pairwise(index_sorted):
        if prev.c <= 0:
            continue
        entry = DailyOvernight(
            session=current.day,
            previous_session=prev.day,
            gap=current.o / prev.c - 1.0,
            prior_session_return=prev.ret,
            calendar_days=(current.day - prev.day).days,
        )
        for symbol, bars in foreign_sorted.items():
            bar = most_recent_before(bars, current.day)
            if bar is None:
                continue
            entry.foreign_returns[symbol] = bar.ret
            entry.foreign_ranges[symbol] = bar.range_pct
        out.append(entry)
    return out


def coverage_report(series: Sequence[DailyOvernight], symbols: Sequence[str]) -> dict[str, float]:
    """Fraction of sessions for which each foreign series actually reported.

    A series present on half the sessions is not a feature, it is a coin flip
    with extra steps - and without this number nobody would notice.
    """
    if not series:
        return {}
    return {s: sum(1 for e in series if s in e.foreign_returns) / len(series) for s in symbols}


def session_span(series: Sequence[DailyOvernight]) -> tuple[date, date] | None:
    if not series:
        return None
    days = sorted(e.session for e in series)
    return days[0], days[-1]


def approximate_window_hours(entry: DailyOvernight) -> float:
    """Hours between the Indian close and the next session's cutoff.

    Approximated from calendar days because daily bars carry no clock. Close at
    15:30, cutoff at 08:45 the next session: 17.25 hours, plus 24 for each
    additional calendar day in between.
    """
    return 17.25 + 24.0 * max(0, entry.calendar_days - 1)
