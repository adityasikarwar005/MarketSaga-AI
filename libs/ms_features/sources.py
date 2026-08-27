"""M41 + M42 - the two materialisations.

``OfflineSource`` reads history; ``OnlineSource`` maintains rolling buffers fed
by the live bus. They implement one interface, so the feature function cannot
tell which side it is running on - and the parity test (M43) proves it every
commit.

The asymmetry that has to be handled carefully: offline knows everything and
must be made to forget the future; online knows only the past and must be made
to remember enough of it. Both failure modes are silent. Offline leaks; online
returns a shorter window than the feature asked for and quietly produces a
different number.

So :class:`OnlineSource` records when it was asked for more history than it
holds, and the parity test fails on that rather than on the resulting value
mismatch, which would be far harder to diagnose.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick
from libs.ms_features.registry import ContextSource

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #
class OfflineSource(ContextSource):
    """History in memory, sorted once, sliced by bisect.

    Deliberately not backed by a live query: a research run computes millions of
    vectors and a per-call database round trip would make the materialiser
    unusable. Loading a day and slicing it is both faster and easier to reason
    about.
    """

    def __init__(
        self,
        candles: Iterable[Candle] = (),
        ticks: Iterable[Tick] = (),
        chains: Iterable[ChainSnapshot] = (),
    ) -> None:
        self._candles: dict[tuple[int, int], list[Candle]] = {}
        for c in candles:
            self._candles.setdefault((c.instrument_id, c.tf), []).append(c)
        for candle_series in self._candles.values():
            candle_series.sort(key=lambda c: (c.ts, c.version))

        self._ticks: dict[int, list[Tick]] = {}
        for t in ticks:
            self._ticks.setdefault(t.instrument_id, []).append(t)
        for tick_series in self._ticks.values():
            tick_series.sort(key=lambda t: t.ts)

        self._chains: dict[int, list[ChainSnapshot]] = {}
        for snap in chains:
            self._chains.setdefault(snap.underlying_id, []).append(snap)
        for chain_series in self._chains.values():
            chain_series.sort(key=lambda c: c.ts)

        self._candle_keys: dict[tuple[int, int], list[datetime]] = {
            k: [c.ts for c in v] for k, v in self._candles.items()
        }
        self._tick_keys: dict[int, list[datetime]] = {
            k: [t.ts for t in v] for k, v in self._ticks.items()
        }
        self._chain_keys: dict[int, list[datetime]] = {
            k: [s.ts for s in v] for k, v in self._chains.items()
        }

    def candles(
        self, instrument_id: int, start: datetime, end: datetime, tf: int
    ) -> Sequence[Candle]:
        series = self._candles.get((instrument_id, tf))
        if not series:
            return []
        keys = self._candle_keys[(instrument_id, tf)]
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_left(keys, end)  # right-open
        return series[lo:hi]

    def ticks(self, instrument_id: int, start: datetime, end: datetime) -> Sequence[Tick]:
        series = self._ticks.get(instrument_id)
        if not series:
            return []
        keys = self._tick_keys[instrument_id]
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_left(keys, end)
        return series[lo:hi]

    def chain(self, underlying_id: int, as_of: datetime) -> ChainSnapshot | None:
        series = self._chains.get(underlying_id)
        if not series:
            return None
        keys = self._chain_keys[underlying_id]
        # STRICTLY before as_of, matching the right-open convention used for
        # candles, tick windows and the overnight window.
        #
        # A snapshot stamped exactly at the decision instant has not been
        # processed yet in any real system, and - decisively - the online path
        # cannot guarantee it has arrived: at that instant the merge may deliver
        # the candle first. A convention the online path cannot honour is a
        # convention that produces train/serve skew, so both sides use `<`.
        idx = bisect.bisect_left(keys, as_of) - 1
        return series[idx] if idx >= 0 else None


# --------------------------------------------------------------------------- #
# Online
# --------------------------------------------------------------------------- #
@dataclass
class Coverage:
    """How far back this buffer actually reaches."""

    oldest: datetime | None = None
    newest: datetime | None = None
    count: int = 0


@dataclass
class ShortWindow:
    """A feature asked for more history than the buffer holds."""

    instrument_id: int
    kind: str
    requested_from: datetime
    available_from: datetime | None

    def __str__(self) -> str:
        have = self.available_from.isoformat() if self.available_from else "nothing"
        return (
            f"{self.kind} for instrument {self.instrument_id}: asked from "
            f"{self.requested_from.isoformat()}, buffer starts at {have}"
        )


class OnlineSource(ContextSource):
    """Rolling buffers fed by the live bus.

    ``max_minutes`` must exceed the longest window any registered feature asks
    for. When it does not, the feature silently computes over a shorter history
    and produces a different number from its offline twin - so every truncated
    read is recorded and the parity test fails on *that*, which points straight
    at the cause instead of at a mysterious value mismatch.

    A buffer that has not filled yet is a different thing entirely and is not
    recorded: at 09:20 there genuinely is no 30-minute history, offline has none
    either, and both sides correctly return None. Only a request older than the
    buffer could ever hold - a real capacity overflow - is a fault.
    """

    def __init__(self, *, max_minutes: int = 120, max_ticks: int = 20_000) -> None:
        self._max = timedelta(minutes=max_minutes)
        self._max_ticks = max_ticks
        self._candles: dict[tuple[int, int], deque[Candle]] = {}
        self._ticks: dict[int, deque[Tick]] = {}
        self._chains: dict[int, ChainSnapshot] = {}
        self._chain_history: dict[int, deque[ChainSnapshot]] = {}
        self.short_windows: list[ShortWindow] = []

    # -- ingestion ----------------------------------------------------------
    def on_candle(self, candle: Candle) -> None:
        key = (candle.instrument_id, candle.tf)
        buf = self._candles.setdefault(key, deque())
        buf.append(candle)
        self._trim_by_time(buf, candle.ts)

    def on_tick(self, tick: Tick) -> None:
        buf = self._ticks.setdefault(tick.instrument_id, deque(maxlen=self._max_ticks))
        buf.append(tick)

    def on_chain(self, snap: ChainSnapshot) -> None:
        self._chains[snap.underlying_id] = snap
        hist = self._chain_history.setdefault(snap.underlying_id, deque(maxlen=512))
        hist.append(snap)

    def _trim_by_time(self, buf: deque[Candle], newest: datetime) -> None:
        cutoff = newest - self._max
        while buf and buf[0].ts < cutoff:
            buf.popleft()

    # -- reads --------------------------------------------------------------
    def candles(
        self, instrument_id: int, start: datetime, end: datetime, tf: int
    ) -> Sequence[Candle]:
        buf = self._candles.get((instrument_id, tf))
        if not buf:
            return []
        if start < buf[-1].ts - self._max:
            self._note_short(instrument_id, "candles", start, buf[0].ts)
        return [c for c in buf if start <= c.ts < end]

    def ticks(self, instrument_id: int, start: datetime, end: datetime) -> Sequence[Tick]:
        buf = self._ticks.get(instrument_id)
        if not buf:
            return []
        if len(buf) == self._max_ticks and buf[0].ts > start:
            # The ring buffer is full and has already discarded older ticks.
            self._note_short(instrument_id, "ticks", start, buf[0].ts)
        return [t for t in buf if start <= t.ts < end]

    def chain(self, underlying_id: int, as_of: datetime) -> ChainSnapshot | None:
        hist = self._chain_history.get(underlying_id)
        if not hist:
            return None
        best: ChainSnapshot | None = None
        for snap in hist:
            if snap.ts < as_of and (best is None or snap.ts > best.ts):
                best = snap
        return best

    def _note_short(
        self, instrument_id: int, kind: str, requested: datetime, available: datetime | None
    ) -> None:
        self.short_windows.append(ShortWindow(instrument_id, kind, requested, available))
        METRICS.inc("feature_short_window_total", kind=kind)

    # -- introspection ------------------------------------------------------
    def coverage(self, instrument_id: int, tf: int = 1) -> Coverage:
        buf = self._candles.get((instrument_id, tf))
        if not buf:
            return Coverage()
        return Coverage(oldest=buf[0].ts, newest=buf[-1].ts, count=len(buf))

    def warm(self, instrument_id: int, minutes: int, tf: int = 1) -> bool:
        """Whether the buffer holds enough history for a window of ``minutes``.

        A service must not emit features before this is true, or its first
        predictions of the session are computed on a partial history and are
        quietly different from anything the model was trained on.
        """
        cov = self.coverage(instrument_id, tf)
        if cov.oldest is None or cov.newest is None:
            return False
        return (cov.newest - cov.oldest) >= timedelta(minutes=minutes)

    def reset_short_windows(self) -> None:
        self.short_windows.clear()


# --------------------------------------------------------------------------- #
# Loading history into an offline source
# --------------------------------------------------------------------------- #
@dataclass
class LoadedDay:
    candles: list[Candle] = field(default_factory=list)
    ticks: list[Tick] = field(default_factory=list)
    chains: list[ChainSnapshot] = field(default_factory=list)

    def source(self) -> OfflineSource:
        return OfflineSource(self.candles, self.ticks, self.chains)

    @property
    def summary(self) -> str:
        return (
            f"{len(self.candles):,} candles, {len(self.ticks):,} ticks, "
            f"{len(self.chains):,} chain snapshots"
        )


def load_day(day: date, *, root: Path | None = None, with_ticks: bool = False) -> LoadedDay:
    """Read one recorded day off disk into memory for research.

    Ticks are optional and off by default: a session is tens of thousands of
    them, most features never look at one, and loading them by default makes
    every research run slower for no benefit.
    """
    from libs.ms_data.recorder import DATA_ROOT, read_stream

    base = root or DATA_ROOT
    out = LoadedDay()

    def read(stream: str, cls: type[Any]) -> list[Any]:
        partition = base / stream / f"date={day.isoformat()}"
        if not partition.exists():
            return []
        rows: list[Any] = []
        for path in sorted(partition.glob("*.ndjson")):
            rows.extend(read_stream(path, cls))
        return rows

    out.candles = read("candles", Candle)
    out.chains = read("chain", ChainSnapshot)
    if with_ticks:
        out.ticks = read("ticks", Tick)
    log.info("loaded day", fields={"day": str(day), "summary": out.summary})
    return out
