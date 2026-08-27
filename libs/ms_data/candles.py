"""M13 - Candle builder.

Turning ticks into OHLCV looks like a warm-up exercise and is in fact the most
common source of quiet data corruption in a trading system. Six rules, each of
which exists because breaking it produces data that looks fine and is not:

1. **Left-closed, right-open buckets.** A tick at exactly 09:20:00.000 belongs to
   the 09:20 candle. Getting this wrong shifts every sequence model's input by
   one bar and nobody notices for months.

2. **Synthetic candles for silent minutes.** A minute with no trades must emit a
   zero-volume candle carried forward from the previous close, flagged
   ``synthetic``. Omitting it leaves a hole that breaks any model expecting a
   regular grid - and worse, makes a data outage indistinguishable from a quiet
   market.

3. **Session tagging.** Pre-open is a different price formation mechanism.
   Continuous-session statistics that include it are wrong.

4. **Late ticks never mutate a sealed candle.** They produce a new version plus
   a correction record. A backtest must be able to ask what the candle looked
   like at the time, not what it looks like now.

5. **Roll-ups derive from 1-minute candles**, not from ticks again, so a 5-minute
   candle can never disagree with the five 1-minute candles inside it.

6. **Determinism.** The same ticks in the same order produce byte-identical
   candles. ``tests/replay/test_candle_determinism.py`` enforces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

from libs.ms_core.clock import SessionCalendar, SessionState
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, Tick

log = get_logger(__name__)


@dataclass
class LateTick:
    """A tick that arrived after its candle was sealed."""

    tick: Tick
    bucket: datetime
    sealed_version: int


@dataclass
class _Accumulator:
    instrument_id: int
    bucket: datetime
    o: float
    h: float
    low: float
    c: float
    v: int = 0
    trades: int = 0
    turnover: float = 0.0
    oi: int | None = None

    def update(self, tick: Tick) -> None:
        self.h = max(self.h, tick.ltp)
        self.low = min(self.low, tick.ltp)
        self.c = tick.ltp
        if tick.qty:
            self.v += tick.qty
            self.turnover += tick.qty * tick.ltp
        self.trades += 1
        if tick.oi is not None:
            self.oi = tick.oi

    def seal(self, tf: int, version: int = 1) -> Candle:
        return Candle(
            ts=self.bucket,
            instrument_id=self.instrument_id,
            tf=tf,
            o=self.o,
            h=self.h,
            l=self.low,
            c=self.c,
            v=self.v,
            oi=self.oi,
            trades=self.trades,
            vwap=round(self.turnover / self.v, 4) if self.v else None,
            synthetic=False,
            version=version,
        )


class CandleBuilder:
    """Streaming 1-minute candle construction.

    Usage::

        builder = CandleBuilder(calendar)
        for tick in ticks:
            for candle in builder.on_tick(tick):
                bus.publish(Topic.MARKET_CANDLE, candle)
        closing = builder.flush()

    A candle is emitted only once the bucket after it has begun, because until
    then it can still change. That one-bar delay is real and the decision layer
    must live with it - pretending otherwise is look-ahead bias.
    """

    def __init__(
        self,
        calendar: SessionCalendar,
        *,
        tf: int = 1,
        emit_synthetic: bool = True,
        sessions: tuple[SessionState, ...] = (SessionState.OPEN,),
    ) -> None:
        self._cal = calendar
        self._tf = tf
        self._emit_synthetic = emit_synthetic
        self._sessions = sessions
        self._open: dict[int, _Accumulator] = {}
        self._last_close: dict[int, float] = {}
        self._sealed_version: dict[tuple[int, datetime], int] = {}
        self.late_ticks: list[LateTick] = []

    # -- main path ----------------------------------------------------------
    def on_tick(self, tick: Tick) -> list[Candle]:
        state = self._cal.state(tick.ts)
        if state not in self._sessions:
            # Rule 3: pre-open and closed-session ticks never enter continuous stats.
            METRICS.inc("candle_tick_out_of_session_total", state=state.value)
            return []

        bucket = self._cal.candle_bucket(tick.ts, self._tf)  # Rule 1
        acc = self._open.get(tick.instrument_id)

        if acc is not None and bucket < acc.bucket:
            return self._handle_late(tick, bucket)

        emitted: list[Candle] = []
        if acc is None or bucket > acc.bucket:
            if acc is not None:
                emitted.extend(self._close_through(acc, bucket))
            self._open[tick.instrument_id] = _Accumulator(
                instrument_id=tick.instrument_id,
                bucket=bucket,
                o=tick.ltp,
                h=tick.ltp,
                low=tick.ltp,
                c=tick.ltp,
            )
            acc = self._open[tick.instrument_id]

        acc.update(tick)
        return emitted

    def _close_through(self, acc: _Accumulator, new_bucket: datetime) -> list[Candle]:
        """Seal the open candle, then fill every silent minute up to ``new_bucket``."""
        out = [self._seal(acc)]
        if not self._emit_synthetic:
            return out
        step = timedelta(minutes=self._tf)
        cursor = acc.bucket + step
        while cursor < new_bucket:  # Rule 2
            if self._cal.state(cursor) in self._sessions:
                out.append(self._synthetic(acc.instrument_id, cursor, acc.c))
            cursor += step
        return out

    def _seal(self, acc: _Accumulator) -> Candle:
        key = (acc.instrument_id, acc.bucket)
        version = self._sealed_version.get(key, 0) + 1
        self._sealed_version[key] = version
        candle = acc.seal(self._tf, version)
        self._last_close[acc.instrument_id] = candle.c
        METRICS.inc("candle_sealed_total", tf=str(self._tf))
        return candle

    def _synthetic(self, instrument_id: int, bucket: datetime, close: float) -> Candle:
        self._sealed_version[(instrument_id, bucket)] = 1
        METRICS.inc("candle_synthetic_total", tf=str(self._tf))
        return Candle(
            ts=bucket,
            instrument_id=instrument_id,
            tf=self._tf,
            o=close,
            h=close,
            l=close,
            c=close,
            v=0,
            trades=0,
            vwap=None,
            synthetic=True,
            version=1,
        )

    def _handle_late(self, tick: Tick, bucket: datetime) -> list[Candle]:
        """Rule 4: a sealed candle is never edited in place."""
        version = self._sealed_version.get((tick.instrument_id, bucket), 0)
        self.late_ticks.append(LateTick(tick=tick, bucket=bucket, sealed_version=version))
        METRICS.inc("candle_late_tick_total", instrument=str(tick.instrument_id))
        log.warning(
            "late tick after candle sealed",
            fields={
                "instrument_id": tick.instrument_id,
                "bucket": bucket.isoformat(),
                "tick_ts": tick.ts.isoformat(),
                "sealed_version": version,
            },
        )
        return []

    def flush(self, up_to: datetime | None = None) -> list[Candle]:
        """Seal everything still open. Call at session end."""
        out: list[Candle] = []
        for iid in sorted(self._open):
            acc = self._open[iid]
            out.append(self._seal(acc))
            if up_to is not None and self._emit_synthetic:
                step = timedelta(minutes=self._tf)
                cursor = acc.bucket + step
                last_close = acc.c
                while cursor < up_to:
                    if self._cal.state(cursor) in self._sessions:
                        out.append(self._synthetic(iid, cursor, last_close))
                    cursor += step
        self._open.clear()
        return out

    def build(self, ticks: Iterable[Tick], *, up_to: datetime | None = None) -> list[Candle]:
        """Batch convenience: ticks in, candles out, deterministic."""
        out: list[Candle] = []
        for t in ticks:
            out.extend(self.on_tick(t))
        out.extend(self.flush(up_to))
        return out


# --------------------------------------------------------------------------- #
# Roll-ups
# --------------------------------------------------------------------------- #
def rollup(candles: Iterable[Candle], calendar: SessionCalendar, target_tf: int) -> list[Candle]:
    """Aggregate 1-minute candles into a higher timeframe.

    Rule 5: derived from 1-minute candles rather than re-scanned from ticks, so a
    5-minute candle can never disagree with the five inside it.
    """
    groups: dict[tuple[int, datetime], list[Candle]] = {}
    for c in candles:
        if c.tf != 1:
            raise ValueError(f"rollup expects 1-minute candles, got tf={c.tf}")
        bucket = calendar.candle_bucket(c.ts, target_tf)
        groups.setdefault((c.instrument_id, bucket), []).append(c)

    out: list[Candle] = []
    for (iid, bucket), members in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        members.sort(key=lambda c: c.ts)
        volume = sum(m.v for m in members)
        turnover = sum((m.vwap or m.c) * m.v for m in members)
        out.append(
            Candle(
                ts=bucket,
                instrument_id=iid,
                tf=target_tf,
                o=members[0].o,
                h=max(m.h for m in members),
                l=min(m.low for m in members),
                c=members[-1].c,
                v=volume,
                oi=members[-1].oi,
                trades=sum(m.trades or 0 for m in members),
                vwap=round(turnover / volume, 4) if volume else None,
                synthetic=all(m.synthetic for m in members),
                version=1,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class SessionCandleReport:
    instrument_id: int
    expected: int
    actual: int
    synthetic: int
    inconsistent: list[datetime] = field(default_factory=list)
    duplicates: list[datetime] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.actual == self.expected and not self.inconsistent and not self.duplicates

    @property
    def synthetic_pct(self) -> float:
        return self.synthetic / self.actual if self.actual else 0.0


def verify_session(
    candles: Iterable[Candle], calendar: SessionCalendar
) -> list[SessionCandleReport]:
    """The check that runs nightly (M19).

    A session must contain exactly the expected number of candles - 375 for a
    1-minute continuous session. Anything else is either a feed problem or a
    calendar problem, and both need a human before research runs on that day.
    """
    by_instrument: dict[int, list[Candle]] = {}
    for c in candles:
        by_instrument.setdefault(c.instrument_id, []).append(c)

    reports: list[SessionCandleReport] = []
    for iid, items in sorted(by_instrument.items()):
        items.sort(key=lambda c: (c.ts, c.version))
        tf = items[0].tf if items else 1
        latest: dict[datetime, Candle] = {}
        duplicates: list[datetime] = []
        for c in items:
            if c.ts in latest and c.version == latest[c.ts].version:
                duplicates.append(c.ts)
            latest[c.ts] = c
        reports.append(
            SessionCandleReport(
                instrument_id=iid,
                expected=calendar.expected_candle_count(tf),
                actual=len(latest),
                synthetic=sum(1 for c in latest.values() if c.synthetic),
                inconsistent=[c.ts for c in latest.values() if not c.is_consistent()],
                duplicates=duplicates,
            )
        )
    return reports


def iter_gaps(candles: Iterable[Candle], tf: int = 1) -> Iterator[tuple[datetime, datetime]]:
    """Missing spans in an otherwise regular grid, for the QC report."""
    items = sorted(candles, key=lambda c: c.ts)
    step = timedelta(minutes=tf)
    for a, b in pairwise(items):
        if b.ts - a.ts > step:
            yield a.ts + step, b.ts
