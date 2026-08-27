"""M40 - Feature registry.

The most common reason a backtest works and a live system does not is
*train/serve skew*: the offline feature was computed differently from the online
one. So a feature is defined exactly once, and the two materialisations are two
callers of the same function.

The second rule is stronger and is what this module is really for:

    A feature function cannot read the future, because it is never handed
    anything that contains the future.

Features do not receive a DataFrame. They receive a :class:`FeatureContext`
bound to one instrument and one ``as_of`` instant, and every accessor on that
context filters internally. There is no argument a feature can pass, and no
attribute it can reach, that returns a row from after ``as_of``. Leakage becomes
a thing you have to work at rather than a thing you fall into.

The subtlest case, and the one that catches everyone: **the candle containing
``as_of`` is excluded.** At 09:20:30 the 09:20 candle is half-formed - its close
has not happened yet. Including it leaks thirty seconds of the future into every
feature built on it, which is more than enough to make a model look brilliant
and be useless.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick

log = get_logger(__name__)

FeatureFn = Callable[["FeatureContext"], float | None]


class FeatureError(RuntimeError):
    """Raised when a feature is defined or used incorrectly."""


# --------------------------------------------------------------------------- #
# The data view a feature is allowed to see
# --------------------------------------------------------------------------- #
class ContextSource(ABC):
    """Where a context gets its data. Offline and online both implement this.

    That one interface, implemented twice, is the entire basis of the parity
    guarantee: the feature function cannot tell which side it is running on.
    """

    @abstractmethod
    def candles(
        self, instrument_id: int, start: datetime, end: datetime, tf: int
    ) -> Sequence[Candle]: ...

    @abstractmethod
    def ticks(self, instrument_id: int, start: datetime, end: datetime) -> Sequence[Tick]: ...

    @abstractmethod
    def chain(self, underlying_id: int, as_of: datetime) -> ChainSnapshot | None: ...


@dataclass
class AccessLog:
    """What a feature actually touched. Used by the leakage detector."""

    max_ts: datetime | None = None
    calls: int = 0

    def record(self, ts: datetime | None) -> None:
        self.calls += 1
        if ts is not None and (self.max_ts is None or ts > self.max_ts):
            self.max_ts = ts


class FeatureContext:
    """Bound to one instrument and one instant. The only way a feature reads data.

    Every window is right-open at ``as_of`` and, for candles, right-open at the
    *bucket* containing ``as_of`` - see the module docstring for why that second
    part matters more than it looks.
    """

    def __init__(
        self,
        source: ContextSource,
        instrument_id: int,
        as_of: datetime,
        *,
        underlying_id: int | None = None,
        access_log: AccessLog | None = None,
        values: dict[str, float | None] | None = None,
    ) -> None:
        self.instrument_id = instrument_id
        self.as_of = as_of
        self.underlying_id = underlying_id if underlying_id is not None else instrument_id
        self._source = source
        self._log = access_log
        # Same trap as above: an empty dict is falsy, and `values or {}` would
        # hand the first feature a *copy* of the accumulator rather than the
        # accumulator itself.
        self._values = {} if values is None else values

    # -- data accessors -----------------------------------------------------
    def candles(
        self, minutes: int, *, tf: int = 1, instrument_id: int | None = None
    ) -> Sequence[Candle]:
        """The last ``minutes`` of completed candles before ``as_of``.

        The bucket containing ``as_of`` is excluded because it is not finished.
        """
        iid = instrument_id if instrument_id is not None else self.instrument_id
        cutoff = self._completed_bucket_start(tf)
        start = cutoff - timedelta(minutes=minutes)
        bars = self._source.candles(iid, start, cutoff, tf)
        self._note(bars[-1].ts if bars else None)
        return bars

    def ticks(self, seconds: float, *, instrument_id: int | None = None) -> Sequence[Tick]:
        """Ticks in ``[as_of - seconds, as_of)``. Right-open, always."""
        iid = instrument_id if instrument_id is not None else self.instrument_id
        start = self.as_of - timedelta(seconds=seconds)
        rows = self._source.ticks(iid, start, self.as_of)
        self._note(rows[-1].ts if rows else None)
        return rows

    def chain(self, *, underlying_id: int | None = None) -> ChainSnapshot | None:
        """The most recent chain snapshot strictly before ``as_of``.

        Right-open like every other window here. A snapshot stamped exactly at
        the decision instant is excluded because the online path cannot promise
        it has arrived by then, and a rule only one side can keep is skew.
        """
        uid = underlying_id if underlying_id is not None else self.underlying_id
        snap = self._source.chain(uid, self.as_of)
        self._note(snap.ts if snap else None)
        return snap

    def value(self, name: str) -> float | None:
        """Another feature's value in this same vector, for composition.

        Only features already computed in this pass are visible, which the
        registry guarantees by evaluating in dependency order.
        """
        if name not in self._values:
            raise FeatureError(
                f"feature {name!r} is not available here - declare it in `deps` so the "
                "registry computes it first"
            )
        return self._values[name]

    # -- internals ----------------------------------------------------------
    def _completed_bucket_start(self, tf: int) -> datetime:
        """Start of the bucket containing ``as_of`` - i.e. the first excluded edge."""
        minute = self.as_of.replace(second=0, microsecond=0)
        if tf > 1:
            minute -= timedelta(minutes=minute.minute % tf)
        return minute

    def _note(self, ts: datetime | None) -> None:
        if self._log is not None:
            self._log.record(ts)


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureDef:
    name: str
    version: int
    fn: FeatureFn
    deps: tuple[str, ...] = ()
    window_minutes: int = 0
    ttl_seconds: float = 5.0
    description: str = ""
    group: str = "misc"

    @property
    def key(self) -> str:
        """Name and version together. Changing logic requires a version bump."""
        return f"{self.name}@v{self.version}"

    def compute(self, ctx: FeatureContext) -> float | None:
        value = self.fn(ctx)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or value != value:  # NaN check
            raise FeatureError(
                f"{self.key} returned {value!r}. A feature returns a finite number or None; "
                "NaN propagates silently into a model and is never noticed."
            )
        return float(value)


class FeatureRegistry:
    def __init__(self) -> None:
        self._defs: dict[str, FeatureDef] = {}

    def register(self, definition: FeatureDef) -> FeatureDef:
        existing = self._defs.get(definition.name)
        if existing is not None and existing.version == definition.version:
            raise FeatureError(
                f"{definition.name} v{definition.version} is already registered. "
                "Changing a feature's logic requires a version bump, never an in-place edit - "
                "otherwise a model trained last month cannot be reproduced."
            )
        self._defs[definition.name] = definition
        return definition

    def get(self, name: str) -> FeatureDef:
        try:
            return self._defs[name]
        except KeyError as exc:
            raise FeatureError(f"unknown feature {name!r}") from exc

    def names(self) -> list[str]:
        return sorted(self._defs)

    def all(self) -> list[FeatureDef]:
        return [self._defs[n] for n in self.names()]

    def by_group(self, group: str) -> list[FeatureDef]:
        return [d for d in self.all() if d.group == group]

    def __len__(self) -> int:
        return len(self._defs)

    def __contains__(self, name: object) -> bool:
        return name in self._defs

    # -- ordering -----------------------------------------------------------
    def resolution_order(self, names: Sequence[str] | None = None) -> list[str]:
        """Topological order, so a feature's dependencies are computed first.

        A cycle is an error rather than something to break arbitrarily: a
        feature that depends on itself has no defined value, and picking one
        would hide the mistake.
        """
        wanted = list(names) if names is not None else self.names()
        order: list[str] = []
        state: dict[str, int] = {}  # 0 = visiting, 1 = done

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if state.get(name) == 1:
                return
            if state.get(name) == 0:
                cycle = " -> ".join([*trail, name])
                raise FeatureError(f"dependency cycle: {cycle}")
            state[name] = 0
            for dep in self.get(name).deps:
                visit(dep, (*trail, name))
            state[name] = 1
            order.append(name)

        for name in wanted:
            visit(name, ())
        return order

    # -- evaluation ---------------------------------------------------------
    def compute(
        self,
        source: ContextSource,
        instrument_id: int,
        as_of: datetime,
        *,
        names: Sequence[str] | None = None,
        underlying_id: int | None = None,
        access_log: AccessLog | None = None,
    ) -> dict[str, float | None]:
        """Compute a feature vector. The one entry point both paths use."""
        values: dict[str, float | None] = {}
        for name in self.resolution_order(names):
            definition = self.get(name)
            ctx = FeatureContext(
                source,
                instrument_id,
                as_of,
                underlying_id=underlying_id,
                access_log=access_log,
                values=values,
            )
            try:
                values[name] = definition.compute(ctx)
            except FeatureError:
                raise
            except Exception as exc:
                # A single broken feature must not take down the vector, but it
                # must be loud: a silently-None feature looks like missing data.
                log.error(
                    "feature raised",
                    fields={"feature": definition.key, "error": str(exc)},
                )
                METRICS.inc("feature_error_total", feature=definition.name)
                values[name] = None
        return values

    def vector_hash(self, names: Sequence[str] | None = None) -> str:
        """Identifies the exact feature set and versions used.

        Stored with every prediction, so a model's inputs can be reconstructed
        months later rather than approximated.
        """
        keys = [self.get(n).key for n in (names if names is not None else self.names())]
        blob = json.dumps(sorted(keys)).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


REGISTRY = FeatureRegistry()


def feature(
    *,
    name: str,
    version: int,
    deps: Sequence[str] = (),
    window_minutes: int = 0,
    ttl_seconds: float = 5.0,
    group: str = "misc",
    registry: FeatureRegistry | None = None,
) -> Callable[[FeatureFn], FeatureFn]:
    """Declare a feature.

    ``version`` is not decoration. A model records the versions it trained on,
    and an in-place edit to a feature's logic silently invalidates every model
    that used it - with no way to detect that afterwards.
    """

    def wrap(fn: FeatureFn) -> FeatureFn:
        # `registry or REGISTRY` would be wrong: an empty FeatureRegistry is
        # falsy (it defines __len__), so a fresh registry passed by a caller
        # would silently be replaced by the global one. Identity, not truthiness.
        target = REGISTRY if registry is None else registry
        params = list(inspect.signature(fn).parameters)
        if len(params) != 1:
            raise FeatureError(
                f"{name}: a feature takes exactly one argument, the FeatureContext. "
                f"Got {params}. Anything else is a way to smuggle in data the context "
                "would have filtered."
            )
        target.register(
            FeatureDef(
                name=name,
                version=version,
                fn=fn,
                deps=tuple(deps),
                window_minutes=window_minutes,
                ttl_seconds=ttl_seconds,
                description=(fn.__doc__ or "").strip().split("\n")[0],
                group=group,
            )
        )
        return fn

    return wrap


@dataclass
class FeatureVector:
    instrument_id: int
    as_of: datetime
    values: dict[str, float | None] = field(default_factory=dict)
    vector_hash: str = ""

    @property
    def null_count(self) -> int:
        return sum(1 for v in self.values.values() if v is None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "as_of": self.as_of.isoformat(),
            "vector_hash": self.vector_hash,
            "null_count": self.null_count,
            **{k: v for k, v in sorted(self.values.items())},
        }
