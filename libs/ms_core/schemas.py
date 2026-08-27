"""M03 - Schemas and contracts.

The most important file in the repository. Every module talks to every other
module through these types and nothing else (megaplan rule R3). Get them right
and any module can be rewritten in a weekend; get them wrong and the codebase
fuses into one lump.

Three conventions, enforced by the base classes below:

**The three timestamps.** Every record carries ``t_event`` (when it happened in
the world), ``t_ingest`` (when we first saw it) and ``t_available`` (when it
became usable for a decision, after processing). Backtests may only read records
whose ``t_available`` is at or before the decision time. A feature computed from
``t_event`` alone is look-ahead bias wearing a disguise.

**Trace id.** One identifier flows from the raw news item all the way to the
fill. Six weeks from now, "why did it do that?" must be answerable with one query.

**Schema version.** Adding a field is a minor change. Changing what a field
means is a major change and needs a new topic. Consumers validate on receipt.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

Price = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
Score01 = Annotated[float, Field(ge=0, le=1)]
Severity = Annotated[float, Field(ge=0, le=10)]


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
# Random ids are right in production and fatal in replay: a byte-identical
# re-run is impossible if every message carries a fresh uuid. So the id source is
# pluggable. Production uses uuid4; the simulator and the determinism tests swap
# in a counter, which is what makes "replay the same day twice, get the same
# hash" an achievable guarantee rather than an aspiration.
def _random_id() -> str:
    return uuid.uuid4().hex


_id_source: Callable[[], str] = _random_id


def new_id() -> str:
    return _id_source()


def new_trace_id() -> str:
    return _id_source()


@contextmanager
def deterministic_ids(prefix: str = "det") -> Iterator[None]:
    """Make every generated id reproducible for the duration of the block.

    Used by the replay engine and by ``tests/replay/test_determinism.py``.
    Never used in a live process - two runs would then collide on ids.
    """
    global _id_source
    previous = _id_source
    counter = itertools.count(1)
    _id_source = lambda: f"{prefix}{next(counter):016d}"  # noqa: E731
    try:
        yield
    finally:
        _id_source = previous


def is_deterministic_id_mode() -> bool:
    return _id_source is not _random_id


class Msg(BaseModel):
    """Base for every message on the bus and every row in the stores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    trace_id: str = Field(default_factory=new_trace_id)

    @field_validator("*", mode="before")
    @classmethod
    def _reject_naive_datetimes(cls, v: Any) -> Any:
        if isinstance(v, datetime) and v.tzinfo is None:
            raise ValueError("naive datetime rejected - all timestamps must be timezone-aware")
        return v


class Timed(Msg):
    """A record whose availability for decisions is explicitly tracked."""

    t_event: datetime
    t_ingest: datetime
    t_available: datetime | None = None
    ts_uncertain: bool = False

    def known_at(self, decision_time: datetime) -> bool:
        """Whether this record could legitimately inform a decision at ``decision_time``.

        The backtest and the feature store both gate on this. It is the single
        rule that separates a real result from a fantasy one.
        """
        ref = self.t_available or self.t_ingest
        return ref <= decision_time


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class Tick(Msg):
    ts: datetime
    instrument_id: int
    ltp: Price
    qty: int | None = None
    bid: Price | None = None
    ask: Price | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    oi: int | None = None
    seq: int | None = None  # vendor sequence number, for gap detection
    src: int = 0

    @property
    def mid(self) -> float | None:
        """Mid price, or None when the book is one-sided.

        Everything downstream that needs a price for valuation uses mid, never
        last-traded price: a stale LTP on an illiquid strike manufactures signal
        that does not exist.
        """
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class Candle(Msg):
    ts: datetime  # left edge of the bucket, left-closed / right-open
    instrument_id: int
    tf: int = Field(gt=0, description="timeframe in minutes")
    o: Price
    h: Price
    low: Price = Field(alias="l")
    c: Price
    v: int = 0
    oi: int | None = None
    trades: int | None = None
    vwap: float | None = None
    synthetic: bool = False  # no ticks arrived; carried forward from previous close
    version: int = 1  # bumped by a late-tick correction, never edited in place

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @field_validator("h")
    @classmethod
    def _high_is_highest(cls, v: float, info: Any) -> float:
        o = info.data.get("o")
        if o is not None and v < o:
            raise ValueError(f"high {v} below open {o}")
        return v

    def is_consistent(self) -> bool:
        return self.low <= min(self.o, self.c) and self.h >= max(self.o, self.c)


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class OptionQuote(Msg):
    instrument_id: int | None = None
    expiry: datetime
    strike: float = Field(gt=0)
    opt_type: OptionType
    ltp: Price | None = None
    bid: Price | None = None
    ask: Price | None = None
    volume: int | None = None
    oi: int | None = None
    oi_change: int | None = None
    iv: float | None = Field(default=None, ge=0, le=5)

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct_of_mid(self) -> float | None:
        m = self.mid
        if m is None or m <= 0 or self.ask is None or self.bid is None:
            return None
        return (self.ask - self.bid) / m


class ChainSnapshot(Msg):
    """A full option chain captured at one instant.

    ``spot`` is captured *with* the chain, never joined afterwards. Joining a
    separately-timestamped spot introduces a lag you cannot undo, and it shows up
    later as a phantom edge in every options backtest.
    """

    ts: datetime
    underlying_id: int
    spot: Price
    quotes: list[OptionQuote]
    completeness: Score01 = 1.0  # fraction of expected strikes actually present

    def atm_strike(self) -> float:
        strikes = sorted({q.strike for q in self.quotes})
        if not strikes:
            raise ValueError("empty chain")
        return min(strikes, key=lambda k: abs(k - self.spot))


# --------------------------------------------------------------------------- #
# News and events
# --------------------------------------------------------------------------- #
class RawNewsItem(Timed):
    id: str = Field(default_factory=new_id)
    source_id: int
    url: str | None = None
    lang: str = "en"
    headline: str
    body: str | None = None
    content_hash: str | None = None
    simhash: int | None = None
    publish_time_method: str = "unknown"  # which extractor produced t_event
    duplicate_of: str | None = None


class Confirmation(StrEnum):
    SINGLE_SOURCE = "SINGLE_SOURCE"
    CORROBORATED = "CORROBORATED"
    OFFICIAL = "OFFICIAL"
    CONFLICTING = "CONFLICTING"
    RETRACTED = "RETRACTED"


class ClassifiedEvent(Timed):
    id: str = Field(default_factory=new_id)
    news_id: str
    event_class: str  # "family.class", must exist in taxonomy.yaml
    subclass: str | None = None
    polarity: float = Field(
        ge=-1, le=1, description="risk-off(-1) .. risk-on(+1) for INDIAN equities"
    )
    severity_raw: Severity
    severity_cal: Severity | None = None  # after empirical recalibration (AI-4)
    confidence: Probability
    novelty: Score01 = 1.0
    credibility: Score01 = 0.5
    confirmation: Confirmation = Confirmation.SINGLE_SOURCE
    source_count: int = 1
    entities: list[str] = Field(default_factory=list)
    geo: list[str] = Field(default_factory=list)
    surprise: float | None = None  # (actual - consensus) / sigma, scheduled events only
    expected: bool = False  # was it on the calendar?
    model_version: str = "none"

    def may_trade_directionally(self) -> bool:
        """Hard gate. Deliberately duplicated in the risk gate - defence in depth.

        The 2013 wire-service hack moved global markets on a single false
        headline. One unconfirmed source is an alert, never a position.
        """
        if self.confirmation in (Confirmation.RETRACTED, Confirmation.CONFLICTING):
            return False
        if self.severity_raw >= 9 and self.source_count < 2:
            return False
        return self.confirmation is not Confirmation.SINGLE_SOURCE or self.credibility >= 0.9


# --------------------------------------------------------------------------- #
# Impact, features, predictions
# --------------------------------------------------------------------------- #
class Quantiles(Msg):
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    @field_validator("q95")
    @classmethod
    def _monotone(cls, v: float, info: Any) -> float:
        d = info.data
        prev = [d.get(k) for k in ("q05", "q25", "q50", "q75")]
        seq = [x for x in prev if x is not None] + [v]
        if seq != sorted(seq):
            raise ValueError(f"quantiles must be non-decreasing, got {seq}")
        return v


class ImpactEstimate(Msg):
    """Output of M34. A distribution with a sample size, never a bare number."""

    event_id: str
    horizon_min: int
    r_hat: float  # expected abnormal return, fraction
    quantiles: Quantiles | None = None
    n_eff: int = 0  # effective historical sample; 0 means "we do not know"
    vol_ratio_hat: float | None = None  # sigma_post / sigma_pre
    d_iv_hat: float | None = None  # expected IV change, volatility points
    is_unknown: bool = False  # true when the class has too little history

    def trustworthy(self, min_sample: int) -> bool:
        return not self.is_unknown and self.n_eff >= min_sample


class FeatureSnapshot(Msg):
    ts: datetime
    instrument_id: int
    values: dict[str, float]
    feature_hash: str  # identifies the exact feature set and versions used
    null_count: int = 0


class Prediction(Msg):
    ts: datetime
    instrument_id: int
    horizon_min: int
    p_up: Probability
    p_up_ci: tuple[float, float] | None = None
    quantiles: Quantiles | None = None
    sigma_hat: float | None = None
    d_iv_hat: float | None = None
    meta_take: bool = False  # meta-labeller's verdict: act on this or skip
    model_versions: dict[str, str] = Field(default_factory=dict)
    feature_hash: str = ""
    degraded: bool = False  # produced by a fallback model


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Leg(Msg):
    side: Side
    strike: float
    opt_type: OptionType
    qty_lots: int = Field(gt=0)
    expiry: datetime
    price: Price | None = None


class StructureType(StrEnum):
    NONE = "NONE"
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    BULL_CALL_DEBIT = "BULL_CALL_DEBIT"
    BEAR_PUT_DEBIT = "BEAR_PUT_DEBIT"
    LONG_STRADDLE = "LONG_STRADDLE"
    LONG_STRANGLE = "LONG_STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"


class TradeMode(StrEnum):
    PAPER = "PAPER"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"


class Recommendation(Msg):
    """The system's public output. Immutable, fully explained, always auditable."""

    reco_id: str = Field(default_factory=new_id)
    ts_emitted: datetime
    mode: TradeMode = TradeMode.PAPER

    event_id: str | None = None
    prediction: Prediction
    structure: StructureType
    legs: list[Leg] = Field(default_factory=list)

    net_debit: float | None = None
    max_profit: float | None = None
    max_loss: float | None = None
    greeks_net: dict[str, float] = Field(default_factory=dict)

    e_pnl: float | None = None
    p_profit: Probability | None = None
    cvar5: float | None = None
    score: float | None = None

    lots: int = 0
    capital_at_risk: float = 0.0

    target: float | None = None
    stop: float | None = None
    time_stop: datetime | None = None

    explanation: list[str] = Field(default_factory=list)
    gates_passed: list[str] = Field(default_factory=list)
    config_hash: str = ""

    @field_validator("explanation")
    @classmethod
    def _must_explain(cls, v: list[str], info: Any) -> list[str]:
        # A recommendation without a reason cannot be debugged, cannot be
        # attributed, and cannot be improved. This is a hard requirement.
        if info.data.get("structure") not in (None, StructureType.NONE) and not v:
            raise ValueError("a non-NONE recommendation must carry at least one explanation line")
        return v

    @property
    def is_actionable(self) -> bool:
        return self.structure is not StructureType.NONE and self.lots > 0


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class OrderState(StrEnum):
    INTENT = "INTENT"
    SUBMITTED = "SUBMITTED"
    ACKED = "ACKED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class Order(Msg):
    order_id: str = Field(default_factory=new_id)
    reco_id: str
    ts: datetime
    instrument_id: int
    side: Side
    qty_lots: int = Field(gt=0)
    limit_price: Price | None = None
    state: OrderState = OrderState.INTENT
    mode: TradeMode = TradeMode.PAPER
    broker_ref: str | None = None
    reject_reason: str | None = None


class Fill(Msg):
    fill_id: str = Field(default_factory=new_id)
    order_id: str
    ts: datetime
    qty_lots: int = Field(gt=0)
    price: Price
    slippage_vs_mid: float | None = None
    costs: float = 0.0


class Outcome(Msg):
    """Resolved result of a prediction. Written back by M93 after the horizon.

    Predictions and outcomes live in one table. A prediction whose outcome is
    never recorded teaches the system nothing, and the class-level kill policy
    (M94) depends entirely on this being complete.
    """

    reco_id: str
    resolved_at: datetime
    realised_return: float
    predicted_p_up: Probability
    direction_correct: bool | None = None
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    exit_reason: Literal["target", "stop", "time_stop", "vol_stop", "flatten", "expired"]
    event_class: str | None = None
