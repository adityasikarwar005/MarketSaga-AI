"""M01 - Config registry.

Every tunable number in MarketSage lives in ``config/*.yaml`` and arrives here.
Nothing in ``libs/`` or ``services/`` may hard-code a threshold (megaplan rule R5).

Three guarantees this module provides:

1. **Frozen.** A loaded config cannot be mutated at runtime.
2. **Hashed.** ``Config.config_hash`` is logged with every decision so any past
   recommendation can be reproduced with the exact settings that produced it.
3. **Point-in-time.** Cost tables carry ``effective_from``; asking for costs
   resolves to the row that was in force on that date, never the latest one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

CONFIG_FILES = (
    "instruments",
    "sessions",
    "crossasset",
    "costs",
    "risk_limits",
    "sources",
    "taxonomy",
)


class Frozen(BaseModel):
    """Base for every config model: immutable and strict about unknown keys.

    ``extra="forbid"`` is deliberate. A typo in a YAML key must be a loud error,
    not a silently ignored setting that makes you debug a model for two days.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #
class LotSizeEntry(Frozen):
    effective_from: date
    lot_size: int = Field(gt=0)
    source: str  # circular reference or "TODO: verify"


class Instrument(Frozen):
    id: int
    symbol: str
    name: str
    kind: Literal["index", "index_future", "index_option", "equity"]
    exchange: Literal["NSE", "BSE", "MCX"]
    segment: Literal["CASH", "FUT", "OPT"]
    tick_size: float = Field(gt=0)
    lot_sizes: list[LotSizeEntry] = Field(default_factory=list)
    expiry_weekday: int | None = None  # 0=Mon .. 6=Sun, for derivative underlyings
    verified: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _lot_sizes_sorted(self) -> Instrument:
        dates = [e.effective_from for e in self.lot_sizes]
        if dates != sorted(dates):
            raise ValueError(f"{self.symbol}: lot_sizes must be sorted by effective_from")
        return self

    def lot_size_on(self, on: date) -> int:
        """Lot size in force on ``on``.

        Lot sizes change by exchange circular. Using today's lot size to value a
        trade from eight months ago silently corrupts every historical P&L number,
        which is why this is a lookup and not an attribute.
        """
        applicable = [e for e in self.lot_sizes if e.effective_from <= on]
        if not applicable:
            raise LookupError(
                f"{self.symbol}: no lot size known on {on} "
                f"(earliest entry is {self.lot_sizes[0].effective_from if self.lot_sizes else 'none'})"
            )
        return applicable[-1].lot_size


class InstrumentsConfig(Frozen):
    instruments: list[Instrument]

    @model_validator(mode="after")
    def _unique_ids(self) -> InstrumentsConfig:
        ids = [i.id for i in self.instruments]
        symbols = [i.symbol for i in self.instruments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate instrument id")
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate instrument symbol")
        return self


# --------------------------------------------------------------------------- #
# sessions / calendar
# --------------------------------------------------------------------------- #
class SessionWindow(Frozen):
    start: str  # "HH:MM"
    end: str


class SessionsConfig(Frozen):
    timezone: str
    preopen: SessionWindow
    continuous: SessionWindow
    expected_continuous_minutes: int = Field(gt=0)
    holidays: dict[int, list[date]] = Field(default_factory=dict)
    special_sessions: list[dict[str, Any]] = Field(default_factory=list)

    def holidays_for(self, year: int) -> list[date]:
        return self.holidays.get(year, [])


# --------------------------------------------------------------------------- #
# cross-asset series
# --------------------------------------------------------------------------- #
class CrossAssetSeriesConfig(Frozen):
    instrument_id: int = Field(ge=1000)  # reserved range; cannot collide with NSE ids
    symbol: str
    name: str
    category: Literal["equity_futures", "commodity", "fx", "rates", "index"]
    timezone: str
    open_time: str
    close_time: str
    spans_midnight: bool = False
    weekdays_only: bool = True
    typical_lag_seconds: float = Field(ge=0)
    india_channel: str
    provider: str = ""
    enabled: bool = False


class CrossAssetConfig(Frozen):
    series: list[CrossAssetSeriesConfig]
    gap_cutoff_ist: str

    @model_validator(mode="after")
    def _unique(self) -> CrossAssetConfig:
        ids = [s.instrument_id for s in self.series]
        symbols = [s.symbol for s in self.series]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate cross-asset instrument_id")
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate cross-asset symbol")
        for s in self.series:
            if not s.india_channel:
                raise ValueError(
                    f"{s.symbol}: india_channel is required - a series nobody can explain "
                    "the transmission mechanism for is a coincidence waiting to be traded"
                )
        return self


# --------------------------------------------------------------------------- #
# costs (point-in-time)
# --------------------------------------------------------------------------- #
class CostSchedule(Frozen):
    effective_from: date
    brokerage_per_order: float = Field(ge=0)
    exchange_txn_pct_of_premium: float = Field(ge=0)
    stt_pct_sell_premium: float = Field(ge=0)
    sebi_fee_pct_of_turnover: float = Field(ge=0)
    stamp_duty_pct_buy: float = Field(ge=0)
    gst_pct_on_charges: float = Field(ge=0)
    source: str


class CostsConfig(Frozen):
    schedules: list[CostSchedule]

    @model_validator(mode="after")
    def _sorted(self) -> CostsConfig:
        dates = [s.effective_from for s in self.schedules]
        if dates != sorted(dates):
            raise ValueError("cost schedules must be sorted by effective_from")
        if not dates:
            raise ValueError("at least one cost schedule required")
        return self

    def on(self, day: date) -> CostSchedule:
        """The cost schedule in force on ``day``.

        Statutory rates change. A backtest that applies today's rates to a trade
        from two years ago is not a backtest, it is a guess.
        """
        applicable = [s for s in self.schedules if s.effective_from <= day]
        if not applicable:
            raise LookupError(f"no cost schedule known on {day}")
        return applicable[-1]


# --------------------------------------------------------------------------- #
# risk limits
# --------------------------------------------------------------------------- #
class RiskLimits(Frozen):
    per_trade_capital_at_risk_pct: float = Field(gt=0, le=10)
    daily_loss_limit_pct: float = Field(gt=0, le=20)
    weekly_loss_limit_pct: float = Field(gt=0, le=40)
    max_concurrent_positions: int = Field(gt=0)
    max_trades_per_day: int = Field(gt=0)
    max_trades_per_event: int = Field(gt=0)
    max_portfolio_delta: float
    max_portfolio_vega: float
    slippage_guard_pct_of_premium: float = Field(gt=0)
    no_entry_first_minutes: int = Field(ge=0)
    no_entry_last_minutes: int = Field(ge=0)
    forced_flatten_time: str
    expiry_day_size_multiplier: float = Field(gt=0, le=1)
    kelly_fraction: float = Field(gt=0, le=1)
    feed_staleness_halt_seconds: float = Field(gt=0)
    clock_drift_halt_ms: float = Field(gt=0)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
class Source(Frozen):
    id: int
    name: str
    source_class: str
    credibility: float = Field(ge=0, le=1)
    typical_lag_seconds: float = Field(ge=0)
    may_trigger_directional: bool
    tos_note: str
    enabled: bool = True


class SourcesConfig(Frozen):
    sources: list[Source]

    @model_validator(mode="after")
    def _rules(self) -> SourcesConfig:
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate source id")
        for s in self.sources:
            if s.credibility < 0.5 and s.may_trigger_directional:
                raise ValueError(
                    f"source {s.name!r}: credibility {s.credibility} is below 0.5 but "
                    "may_trigger_directional is true - low-credibility sources may only "
                    "raise alerts, never open a directional position"
                )
            if not s.tos_note:
                raise ValueError(f"source {s.name!r}: tos_note is required")
        return self


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #
class EventClass(Frozen):
    sev_range: tuple[float, float]
    scheduled: bool = False
    needs_consensus: bool = False
    require_sources: int = 1
    tradeable: bool = True
    intraday_only: bool = False

    @model_validator(mode="after")
    def _range_ok(self) -> EventClass:
        lo, hi = self.sev_range
        if not (0 <= lo <= hi <= 10):
            raise ValueError(f"bad sev_range {self.sev_range}")
        return self


class TaxonomyGates(Frozen):
    min_severity: float
    min_credibility: float
    min_novelty: float
    max_already_priced: float
    min_effective_sample: int


class TaxonomyConfig(Frozen):
    version: int
    families: dict[str, dict[str, dict[str, EventClass]]]
    gates: TaxonomyGates

    def class_names(self) -> list[str]:
        out: list[str] = []
        for family, body in self.families.items():
            for cls in body.get("classes", {}):
                out.append(f"{family}.{cls}")
        return sorted(out)

    def get(self, qualified: str) -> EventClass:
        family, _, cls = qualified.partition(".")
        try:
            return self.families[family]["classes"][cls]
        except KeyError as exc:
            raise KeyError(f"unknown event class {qualified!r}") from exc


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
class Config(Frozen):
    instruments: InstrumentsConfig
    sessions: SessionsConfig
    crossasset: CrossAssetConfig
    costs: CostsConfig
    risk_limits: RiskLimits
    sources: SourcesConfig
    taxonomy: TaxonomyConfig
    config_hash: str

    def instrument(self, symbol: str) -> Instrument:
        for i in self.instruments.instruments:
            if i.symbol == symbol:
                return i
        raise KeyError(f"unknown instrument {symbol!r}")

    def instrument_by_id(self, instrument_id: int) -> Instrument:
        for i in self.instruments.instruments:
            if i.id == instrument_id:
                return i
        raise KeyError(f"unknown instrument id {instrument_id}")

    def unverified_instruments(self) -> list[str]:
        """Instruments whose reference data has not been checked against the exchange.

        Trading on unverified lot sizes is how you discover your position was
        three times the size you thought. :func:`load` warns about these; the
        risk gate (M76) will refuse them outright.
        """
        return [i.symbol for i in self.instruments.instruments if not i.verified]


def _read_yaml(name: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load(config_dir: Path | str | None = None, *, warn: bool = True) -> Config:
    """Load, validate, freeze and hash the whole configuration."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    raw = {name: _read_yaml(name, directory) for name in CONFIG_FILES}

    cfg = Config(
        instruments=InstrumentsConfig(**raw["instruments"]),
        sessions=SessionsConfig(**raw["sessions"]),
        crossasset=CrossAssetConfig(**raw["crossasset"]),
        costs=CostsConfig(**raw["costs"]),
        risk_limits=RiskLimits(**raw["risk_limits"]),
        sources=SourcesConfig(**raw["sources"]),
        taxonomy=TaxonomyConfig(**raw["taxonomy"]),
        config_hash=_hash_payload(raw),
    )

    if warn:
        unverified = cfg.unverified_instruments()
        if unverified:
            import warnings

            warnings.warn(
                "Unverified instrument reference data: "
                + ", ".join(unverified)
                + ". Check lot sizes and tick sizes against the exchange circular and set "
                "verified: true before any sizing decision.",
                UserWarning,
                stacklevel=2,
            )
    return cfg


@lru_cache(maxsize=1)
def get() -> Config:
    """Process-wide cached config. Use :func:`load` in tests to get a fresh one."""
    return load()
