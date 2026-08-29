"""M05 - Observability: structured logs, trace propagation, metrics.

The system has nine planes and a dozen models. Six weeks after a bad trade, the
only question that matters is *why did it do that?* - and the only way to answer
it is if one identifier followed the news item all the way to the fill.

That identifier is the ``trace_id``. It is created once, at ingest, and carried
in a context variable so every log line, metric and message inherits it without
anyone having to pass it around by hand.

Deliberately dependency-free: no prometheus_client, no structlog. Metrics render
to the Prometheus text format on demand, which is all a scrape endpoint needs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_stage: ContextVar[str | None] = ContextVar("stage", default=None)


# --------------------------------------------------------------------------- #
# Trace context
# --------------------------------------------------------------------------- #
def current_trace_id() -> str | None:
    return _trace_id.get()


def current_stage() -> str | None:
    return _stage.get()


@contextmanager
def trace(trace_id: str, stage: str | None = None) -> Iterator[str]:
    """Bind a trace id (and optionally a stage) for the duration of the block."""
    t_token = _trace_id.set(trace_id)
    s_token = _stage.set(stage) if stage is not None else None
    try:
        yield trace_id
    finally:
        _trace_id.reset(t_token)
        if s_token is not None:
            _stage.reset(s_token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark a pipeline stage and record its latency automatically.

    Every stage in the latency budget (blueprint Ch.3) is measured this way, so
    the p50/p99 table in the dashboard is real measurement rather than a design
    document nobody re-checked.
    """
    token = _stage.set(name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        METRICS.observe("stage_latency_seconds", time.perf_counter() - t0, stage=name)
        _stage.reset(token)


# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    """One JSON object per line, with trace context attached automatically.

    ``converter`` is set explicitly. :meth:`logging.Formatter.formatTime`
    defaults to ``time.localtime``, and the ``ts`` below appends a ``Z`` - so
    the default renders local time and asserts it is UTC. On an IST machine
    every line in this repository was stamped 5.5 hours ahead of the instant it
    described, crossing the date boundary for anything logged after 18:30 UTC.

    It survived because it is invisible where it is usually looked for: CI runs
    with TZ=UTC, where localtime and gmtime agree and the bug cannot reproduce.
    R6 keeps `datetime.now` out of library code and had nothing to catch here -
    this module never calls a clock, it inherits a default from `logging`.
    """

    # staticmethod so mypy resolves the overload against the base class's
    # Callable[[float | None], struct_time]; `JsonFormatter.converter` still
    # reads back as time.gmtime itself, which is what the test pins.
    converter = staticmethod(time.gmtime)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (tid := current_trace_id()) is not None:
            payload["trace_id"] = tid
        if (st := current_stage()) is not None:
            payload["stage"] = st
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class _ContextLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        fields = kwargs.pop("fields", None)
        if fields:
            kwargs.setdefault("extra", {})["extra_fields"] = fields
        return msg, kwargs


_configured = False
_lock = threading.Lock()


def configure(level: str | None = None, *, json_output: bool | None = None) -> None:
    """Idempotent logging setup. Safe to call from every entry point."""
    global _configured
    with _lock:
        if _configured:
            return
        lvl = (level or os.environ.get("MARKETSAGE_LOG_LEVEL", "INFO")).upper()
        as_json = (
            json_output
            if json_output is not None
            else os.environ.get("MARKETSAGE_LOG_JSON", "1") == "1"
        )
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            JsonFormatter()
            if as_json
            else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(lvl)
        _configured = True


def get_logger(name: str) -> _ContextLogger:
    configure()
    return _ContextLogger(logging.getLogger(name), {})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
# Latency buckets tuned to the budget in blueprint Ch.3: most stages are single
# -digit milliseconds, and what matters is the p99 tail, not the mean.
DEFAULT_BUCKETS = (
    0.0001,
    0.00025,
    0.0005,  # sub-millisecond: most stages live here
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)


@dataclass
class _Histogram:
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.total += value
        self.n += 1
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def quantile(self, q: float) -> float:
        """Bucket-boundary estimate. Good enough for alerting, not for a paper."""
        if self.n == 0:
            return 0.0
        target = q * self.n
        seen = 0
        for i, c in enumerate(self.counts):
            seen += c
            if seen >= target:
                return self.buckets[i] if i < len(self.buckets) else float("inf")
        return float("inf")


class Metrics:
    """Tiny in-process metric registry with a Prometheus text renderer."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hists: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted(labels.items()))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            k = self._key(name, labels)
            if k not in self._hists:
                self._hists[k] = _Histogram()
            self._hists[k].observe(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": {self._fmt(k): v for k, v in self._counters.items()},
                "gauges": {self._fmt(k): v for k, v in self._gauges.items()},
                "histograms": {
                    self._fmt(k): {
                        "n": h.n,
                        "mean": h.total / h.n if h.n else 0.0,
                        "p50": h.quantile(0.50),
                        "p95": h.quantile(0.95),
                        "p99": h.quantile(0.99),
                    }
                    for k, h in self._hists.items()
                },
            }

    @staticmethod
    def _fmt(key: tuple[str, tuple[tuple[str, str], ...]]) -> str:
        name, labels = key
        if not labels:
            return name
        return name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"

    def render_prometheus(self, prefix: str = "marketsage_") -> str:
        lines: list[str] = []
        with self._lock:
            for k, v in self._counters.items():
                lines.append(f"{prefix}{self._fmt(k)} {v}")
            for k, v in self._gauges.items():
                lines.append(f"{prefix}{self._fmt(k)} {v}")
            for (name, labels), h in self._hists.items():
                lab = dict(labels)
                cumulative = 0
                for i, edge in enumerate(h.buckets):
                    cumulative += h.counts[i]
                    lines.append(
                        f"{prefix}{name}_bucket"
                        + "{"
                        + ",".join(f'{a}="{b}"' for a, b in lab.items())
                        + (("," if lab else "") + f'le="{edge}"')
                        + "}"
                        + f" {cumulative}"
                    )
                lines.append(f"{prefix}{self._fmt((name + '_sum', labels))} {h.total}")
                lines.append(f"{prefix}{self._fmt((name + '_count', labels))} {h.n}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Tests only. A production process never resets its counters."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._hists.clear()


METRICS = Metrics()
