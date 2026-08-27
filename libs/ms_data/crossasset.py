"""M15 - Cross-asset collector and the overnight window.

Roughly half of an Indian index's total price change arrives as an overnight
gap. Most retail systems never see it because they only look at intraday
candles. This module is where that half of the signal enters the system.

Two things here are harder than they look:

**24x7 collection.** Every series keeps its own hours in its own timezone, and
several span midnight. "Is this series trading right now?" is a per-series
question, and answering it with Indian market hours - or worse, not asking -
produces staleness alarms all night and silence when a feed actually dies.

**The overnight window.** For an Indian session on day D, the window runs from
the *previous trading day's* close to a hard cutoff before D's open. Previous
trading day, not previous calendar day: after a long weekend the window is four
days wide, and a model trained on a window that quietly assumed 24 hours will
mis-weight every Monday and every post-holiday session it ever sees.

Cross-asset bars reuse the ``Candle`` schema with instrument ids in a reserved
range. They are semantically different from NSE instruments and structurally
identical, so they share the recorder, the archive and the research layer rather
than duplicating all three.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

from libs.ms_core.clock import UTC, Clock, SessionCalendar
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle

log = get_logger(__name__)

# Cross-asset instrument ids live at or above this. Anything below is an NSE
# instrument from config/instruments.yaml.
CROSSASSET_ID_BASE = 1000


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrossAssetSeries:
    instrument_id: int
    symbol: str
    name: str
    category: str
    timezone: str
    open_time: dtime
    close_time: dtime
    spans_midnight: bool
    weekdays_only: bool
    typical_lag_seconds: float
    india_channel: str
    provider: str = ""
    enabled: bool = False

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_open(self, ts: datetime) -> bool:
        """Whether this series is trading at ``ts``, in its own timezone.

        A series that spans midnight (most futures and FX) is open when the local
        time is after the open OR before the close - the opposite test to a
        normal session, and inverting it silently marks a live feed as dead all
        night.
        """
        local = ts.astimezone(self.tz)
        t = local.time()
        if self.weekdays_only and local.weekday() >= 5:
            # Saturday is always closed. Sunday evening is not: for these venues
            # the week opens on Sunday night local time, and treating it as a
            # weekend loses the first hours of every Monday session.
            sunday_open = self.spans_midnight and local.weekday() == 6 and t >= self.open_time
            if not sunday_open:
                return False
        if self.spans_midnight:
            return t >= self.open_time or t < self.close_time
        return self.open_time <= t < self.close_time

    def staleness_budget(self) -> float:
        """How long silence is tolerable before something is wrong.

        Generous relative to the quoted lag: these are minute bars from vendors
        with their own retry behaviour, and a tight bound here produces alerts
        nobody reads.
        """
        return max(120.0, self.typical_lag_seconds * 20)


class CrossAssetRegistry:
    def __init__(self, series: Iterable[CrossAssetSeries]) -> None:
        self._by_id = {s.instrument_id: s for s in series}
        self._by_symbol = {s.symbol: s for s in self._by_id.values()}
        for s in self._by_id.values():
            if s.instrument_id < CROSSASSET_ID_BASE:
                raise ValueError(
                    f"{s.symbol}: cross-asset ids must be >= {CROSSASSET_ID_BASE} so they "
                    "cannot collide with NSE instruments"
                )

    def get(self, instrument_id: int) -> CrossAssetSeries | None:
        return self._by_id.get(instrument_id)

    def by_symbol(self, symbol: str) -> CrossAssetSeries:
        try:
            return self._by_symbol[symbol]
        except KeyError as exc:
            raise KeyError(f"unknown cross-asset series {symbol!r}") from exc

    def all(self) -> list[CrossAssetSeries]:
        return sorted(self._by_id.values(), key=lambda s: s.instrument_id)

    def enabled(self) -> list[CrossAssetSeries]:
        return [s for s in self.all() if s.enabled]

    def open_now(self, ts: datetime) -> list[CrossAssetSeries]:
        return [s for s in self.all() if s.is_open(ts)]

    def __len__(self) -> int:
        return len(self._by_id)


def registry_from_config(cfg_series: Sequence[object]) -> CrossAssetRegistry:
    """Build from the ``crossasset`` section of the loaded config (M01)."""
    out = []
    for s in cfg_series:
        out.append(
            CrossAssetSeries(
                instrument_id=s.instrument_id,  # type: ignore[attr-defined]
                symbol=s.symbol,  # type: ignore[attr-defined]
                name=s.name,  # type: ignore[attr-defined]
                category=s.category,  # type: ignore[attr-defined]
                timezone=s.timezone,  # type: ignore[attr-defined]
                open_time=_hhmm(s.open_time),  # type: ignore[attr-defined]
                close_time=_hhmm(s.close_time),  # type: ignore[attr-defined]
                spans_midnight=s.spans_midnight,  # type: ignore[attr-defined]
                weekdays_only=s.weekdays_only,  # type: ignore[attr-defined]
                typical_lag_seconds=s.typical_lag_seconds,  # type: ignore[attr-defined]
                india_channel=s.india_channel,  # type: ignore[attr-defined]
                provider=s.provider,  # type: ignore[attr-defined]
                enabled=s.enabled,  # type: ignore[attr-defined]
            )
        )
    return CrossAssetRegistry(out)


def _hhmm(value: str) -> dtime:
    hh, _, mm = value.partition(":")
    return dtime(hour=int(hh), minute=int(mm))


# --------------------------------------------------------------------------- #
# The overnight window
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OvernightWindow:
    """What was knowable between two Indian sessions."""

    session: date
    previous_session: date
    start: datetime  # previous Indian close, UTC
    end: datetime  # the point-in-time cutoff before this session's open, UTC

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    @property
    def spans_multiple_days(self) -> bool:
        """True after a weekend or holiday. Roughly a third of all sessions."""
        return (self.session - self.previous_session).days > 1

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end


def previous_trading_day(calendar: SessionCalendar, day: date, *, max_lookback: int = 10) -> date:
    """The last session before ``day`` on which the exchange was open.

    Not ``day - 1``: after a long weekend the previous session is four days back,
    and a window that assumes 24 hours mis-weights every Monday it ever sees.
    """
    cursor = day - timedelta(days=1)
    for _ in range(max_lookback):
        try:
            if calendar.is_trading_day(cursor):
                return cursor
        except LookupError:
            # No calendar for that year. Fall back to weekdays rather than
            # crashing, and let the quality gate flag the missing calendar.
            if cursor.weekday() < 5:
                return cursor
        cursor -= timedelta(days=1)
    raise LookupError(f"no trading day found in the {max_lookback} days before {day}")


def overnight_window(
    calendar: SessionCalendar, session: date, *, cutoff: dtime | None = None
) -> OvernightWindow:
    """The window a gap forecast for ``session`` is allowed to see.

    Ends at the cutoff, not at the open. Anything published between the cutoff
    and 09:15 is real information that we are choosing not to use, because a
    model that consumes it cannot be run in production at 08:45.
    """
    prev = previous_trading_day(calendar, session)
    _open_prev, close_prev = calendar.session_bounds(prev)
    cut = cutoff or dtime(8, 45)
    end_local = datetime.combine(session, cut, tzinfo=calendar.tz)
    return OvernightWindow(
        session=session,
        previous_session=prev,
        start=close_prev.astimezone(UTC),
        end=end_local.astimezone(UTC),
    )


@dataclass
class SeriesMove:
    symbol: str
    instrument_id: int
    first: float
    last: float
    bars: int
    high: float
    low: float

    @property
    def ret(self) -> float:
        return (self.last / self.first - 1.0) if self.first else 0.0

    @property
    def range_pct(self) -> float:
        return ((self.high - self.low) / self.first) if self.first else 0.0


def overnight_moves(
    bars: Iterable[Candle], window: OvernightWindow, registry: CrossAssetRegistry
) -> dict[str, SeriesMove]:
    """Per-series move across the window. The gap forecaster's feature vector.

    Only bars strictly inside the window are used. A bar stamped at exactly the
    cutoff is excluded - right-open, the same convention as everywhere else, and
    a single included bar there is enough to leak the open into the prediction.
    """
    grouped: dict[int, list[Candle]] = {}
    for bar in bars:
        if not window.contains(bar.ts):
            continue
        grouped.setdefault(bar.instrument_id, []).append(bar)

    out: dict[str, SeriesMove] = {}
    for instrument_id, series_bars in grouped.items():
        meta = registry.get(instrument_id)
        if meta is None:
            continue
        series_bars.sort(key=lambda c: c.ts)
        out[meta.symbol] = SeriesMove(
            symbol=meta.symbol,
            instrument_id=instrument_id,
            first=series_bars[0].o,
            last=series_bars[-1].c,
            bars=len(series_bars),
            high=max(c.h for c in series_bars),
            low=min(c.low for c in series_bars),
        )
    return out


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class CrossAssetProvider(ABC):
    """A source of minute bars for one or more series."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch(self, series: CrossAssetSeries, start: datetime, end: datetime) -> list[Candle]: ...


@dataclass(frozen=True)
class SyntheticSpec:
    start_price: float
    annual_vol: float
    # Correlation to the Indian equity risk factor. Negative for crude and the
    # dollar; positive for global equity futures. Without these the gap model
    # would train on noise and every test of it would pass vacuously.
    beta_to_risk: float


DEFAULT_SYNTHETIC: dict[str, SyntheticSpec] = {
    "GIFT_NIFTY": SyntheticSpec(24_800.0, 0.14, 1.00),
    "US_EQ_FUT": SyntheticSpec(5_600.0, 0.13, 0.85),
    "BRENT": SyntheticSpec(78.0, 0.32, -0.35),
    "GOLD": SyntheticSpec(2_450.0, 0.14, -0.20),
    "DXY": SyntheticSpec(104.5, 0.07, -0.45),
    "USDINR": SyntheticSpec(83.6, 0.05, -0.55),
    "US10Y": SyntheticSpec(4.25, 0.20, -0.30),
    "ASIA_IDX": SyntheticSpec(38_500.0, 0.15, 0.60),
}


class SyntheticCrossAssetProvider(CrossAssetProvider):
    """Deterministic, correlated minute bars.

    Correlation is the point. A generator producing eight independent random
    walks would let a gap model "pass" its test while learning nothing, because
    there would be nothing to learn. Here every series shares a common risk
    factor with its own beta, so a real relationship exists to be recovered.
    """

    def __init__(
        self,
        specs: dict[str, SyntheticSpec] | None = None,
        *,
        seed: int = 3,
    ) -> None:
        self._specs = specs or DEFAULT_SYNTHETIC
        self._seed = seed

    @property
    def name(self) -> str:
        return "synthetic"

    def fetch(self, series: CrossAssetSeries, start: datetime, end: datetime) -> list[Candle]:
        spec = self._specs.get(series.symbol)
        if spec is None:
            return []

        # Seeded per (series, window) so a re-fetch of the same window returns
        # the same bars - the archive must not change under a retry.
        rng = random.Random((self._seed, series.symbol, int(start.timestamp())).__hash__())
        risk_rng = random.Random((self._seed, int(start.timestamp())).__hash__())

        per_minute_vol = spec.annual_vol / math.sqrt(252 * 375)
        price = spec.start_price
        out: list[Candle] = []
        cursor = start.replace(second=0, microsecond=0)

        while cursor < end:
            if not series.is_open(cursor):
                cursor += timedelta(minutes=1)
                continue
            common = risk_rng.gauss(0.0, 1.0)
            idio = rng.gauss(0.0, 1.0)
            shock = per_minute_vol * (
                spec.beta_to_risk * common + math.sqrt(max(0.0, 1 - spec.beta_to_risk**2)) * idio
            )
            o = price
            price = max(1e-6, price * math.exp(shock))
            hi = max(o, price) * (1 + abs(rng.gauss(0, 0.0002)))
            lo = min(o, price) * (1 - abs(rng.gauss(0, 0.0002)))
            out.append(
                Candle(
                    ts=cursor.astimezone(UTC),
                    instrument_id=series.instrument_id,
                    tf=1,
                    o=round(o, 4),
                    h=round(hi, 4),
                    l=round(lo, 4),
                    c=round(price, 4),
                    v=0,
                )
            )
            cursor += timedelta(minutes=1)
        return out


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
@dataclass
class SeriesHealth:
    symbol: str
    last_bar_at: datetime | None = None
    bars: int = 0

    def silence_seconds(self, now: datetime) -> float | None:
        if self.last_bar_at is None:
            return None
        return (now - self.last_bar_at).total_seconds()


class CrossAssetCollector:
    """Polls providers and emits bars, 24x7.

    Deduplicates on ``(instrument_id, ts)``: providers routinely re-serve the
    most recent bar on every poll, and writing it repeatedly would corrupt every
    aggregate computed downstream.
    """

    def __init__(
        self,
        registry: CrossAssetRegistry,
        provider: CrossAssetProvider,
        clock: Clock,
    ) -> None:
        self._registry = registry
        self._provider = provider
        self._clock = clock
        self._seen: set[tuple[int, datetime]] = set()
        self.health: dict[str, SeriesHealth] = {
            s.symbol: SeriesHealth(s.symbol) for s in registry.all()
        }

    def poll(self, start: datetime, end: datetime, *, only_open: bool = True) -> list[Candle]:
        now = self._clock.now()
        fresh: list[Candle] = []
        for series in self._registry.all():
            if only_open and not series.is_open(now):
                continue
            try:
                bars = self._provider.fetch(series, start, end)
            except Exception as exc:
                # One unreachable venue must not stop the other seven.
                log.error(
                    "cross-asset fetch failed",
                    fields={"symbol": series.symbol, "error": str(exc)},
                )
                METRICS.inc("crossasset_fetch_error_total", symbol=series.symbol)
                continue
            for bar in bars:
                key = (bar.instrument_id, bar.ts)
                if key in self._seen:
                    continue
                self._seen.add(key)
                fresh.append(bar)
                h = self.health[series.symbol]
                h.bars += 1
                h.last_bar_at = bar.ts
            METRICS.inc("crossasset_bars_total", value=float(len(bars)), symbol=series.symbol)
        return sorted(fresh, key=lambda c: (c.ts, c.instrument_id))

    def stale_series(self, now: datetime) -> list[str]:
        """Series that are open but have gone quiet for longer than their budget."""
        out = []
        for series in self._registry.all():
            if not series.is_open(now):
                continue
            silence = self.health[series.symbol].silence_seconds(now)
            if silence is not None and silence > series.staleness_budget():
                out.append(series.symbol)
        return out
