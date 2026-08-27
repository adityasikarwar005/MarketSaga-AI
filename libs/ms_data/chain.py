"""M14 - Option chain snapshotter.

The scarcest data this project will ever own. Historical option chains are not
freely available at any useful granularity, which means every chain snapshot not
captured today is one that has to be bought later, or done without.

Two decisions that are easy to get wrong and expensive to fix:

**Spot is captured with the chain, never joined afterwards.** A separately
timestamped spot introduces a lag that cannot be undone, and it shows up later
as a phantom edge in every options backtest - the model appears to predict moves
it has in fact already seen.

**Incomplete snapshots are stored, flagged, not discarded.** A chain that is 60%
complete is evidence about a moment; throwing it away leaves a hole that looks
identical to an outage. ``completeness`` records the truth and the quality gates
decide what to do with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise

from libs.ms_core.clock import UTC, SessionCalendar
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType, Tick
from libs.ms_options.pricing import (
    MINUTES_PER_YEAR,
    OptType,
    forward_from_spot,
    implied_vol,
    put_call_parity_gap,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Instrument registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptionInstrument:
    """Maps a feed instrument id onto what it actually is."""

    instrument_id: int
    underlying_id: int
    expiry: date
    strike: float
    opt_type: OptionType

    @property
    def key(self) -> tuple[date, float, OptionType]:
        return (self.expiry, self.strike, self.opt_type)


class OptionRegistry:
    """Lookup from feed instrument id to contract terms.

    Populated from the broker's instrument dump (M10). Nothing in the chain
    builder guesses what a token means - an unknown id is dropped and counted,
    because silently mis-assigning a strike corrupts a whole expiry.
    """

    def __init__(self, instruments: Iterable[OptionInstrument]) -> None:
        self._by_id: dict[int, OptionInstrument] = {i.instrument_id: i for i in instruments}
        self._by_underlying: dict[int, list[OptionInstrument]] = {}
        for i in self._by_id.values():
            self._by_underlying.setdefault(i.underlying_id, []).append(i)
        self.unknown_ids: set[int] = set()

    def get(self, instrument_id: int) -> OptionInstrument | None:
        found = self._by_id.get(instrument_id)
        if found is None:
            self.unknown_ids.add(instrument_id)
        return found

    def for_underlying(self, underlying_id: int) -> list[OptionInstrument]:
        return self._by_underlying.get(underlying_id, [])

    def expiries(self, underlying_id: int) -> list[date]:
        return sorted({i.expiry for i in self.for_underlying(underlying_id)})

    def strikes(self, underlying_id: int, expiry: date) -> list[float]:
        return sorted({i.strike for i in self.for_underlying(underlying_id) if i.expiry == expiry})

    def __len__(self) -> int:
        return len(self._by_id)


# --------------------------------------------------------------------------- #
# Spec: what we intend to capture
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChainSpec:
    """Which slice of the chain to capture, and how fresh it must be.

    The full chain is far larger than any subscription budget, and the far wings
    are illiquid enough to contribute noise rather than signal. Recording the
    intended window here is what makes ``completeness`` meaningful: a fraction of
    a stated target, not of a vague hope.
    """

    underlying_id: int
    n_expiries: int = 2
    strikes_per_side: int = 15
    snapshot_seconds: float = 5.0
    max_quote_age_seconds: float = 15.0
    risk_free_rate: float = 0.065

    def expected_quote_count(self) -> int:
        return self.n_expiries * (2 * self.strikes_per_side + 1) * 2

    def select_expiries(self, registry: OptionRegistry, on: date) -> list[date]:
        return [e for e in registry.expiries(self.underlying_id) if e >= on][: self.n_expiries]

    def select_strikes(self, registry: OptionRegistry, expiry: date, spot: float) -> list[float]:
        strikes = registry.strikes(self.underlying_id, expiry)
        if not strikes:
            return []
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        lo = max(0, atm_idx - self.strikes_per_side)
        hi = min(len(strikes), atm_idx + self.strikes_per_side + 1)
        return strikes[lo:hi]


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
@dataclass
class _LiveQuote:
    quote: OptionQuote
    ts: datetime


class ChainBuilder:
    """Assembles chain snapshots from a stream of ticks.

    Feed every tick in; call :meth:`maybe_snapshot` on each one. A snapshot is
    emitted when the cadence elapses and a spot price is available - never
    without spot, because a chain with no underlying is not a chain.
    """

    def __init__(
        self,
        spec: ChainSpec,
        registry: OptionRegistry,
        calendar: SessionCalendar,
        *,
        compute_iv: bool = True,
    ) -> None:
        self._spec = spec
        self._registry = registry
        self._cal = calendar
        self._compute_iv = compute_iv
        self._quotes: dict[int, _LiveQuote] = {}
        self._spot: float | None = None
        self._spot_ts: datetime | None = None
        self._last_snapshot: datetime | None = None
        self.dropped_unknown = 0

    # -- ingestion ----------------------------------------------------------
    def on_tick(self, tick: Tick) -> None:
        if tick.instrument_id == self._spec.underlying_id:
            self._spot = tick.ltp
            self._spot_ts = tick.ts
            return

        contract = self._registry.get(tick.instrument_id)
        if contract is None or contract.underlying_id != self._spec.underlying_id:
            self.dropped_unknown += 1
            return

        self._quotes[tick.instrument_id] = _LiveQuote(
            quote=OptionQuote(
                instrument_id=tick.instrument_id,
                expiry=datetime.combine(contract.expiry, datetime.min.time(), tzinfo=UTC),
                strike=contract.strike,
                opt_type=contract.opt_type,
                ltp=tick.ltp,
                bid=tick.bid,
                ask=tick.ask,
                volume=tick.qty,
                oi=tick.oi,
            ),
            ts=tick.ts,
        )

    # -- emission -----------------------------------------------------------
    def due(self, now: datetime) -> bool:
        if self._last_snapshot is None:
            return True
        return (now - self._last_snapshot).total_seconds() >= self._spec.snapshot_seconds

    def maybe_snapshot(self, now: datetime) -> ChainSnapshot | None:
        if not self.due(now) or self._spot is None:
            return None
        return self.snapshot(now)

    def snapshot(self, now: datetime) -> ChainSnapshot | None:
        """Build a snapshot from whatever is currently known.

        Stale quotes are excluded rather than carried forward. A price from
        forty seconds ago is not what the market is showing now, and pretending
        otherwise is how a backtest fills at a price nobody could have got.
        """
        if self._spot is None:
            return None

        today = self._cal.local(now).date()
        expiries = set(self._spec.select_expiries(self._registry, today))
        wanted: set[tuple[date, float, OptionType]] = set()
        for expiry in expiries:
            for strike in self._spec.select_strikes(self._registry, expiry, self._spot):
                wanted.add((expiry, strike, OptionType.CE))
                wanted.add((expiry, strike, OptionType.PE))

        fresh: list[OptionQuote] = []
        stale = 0
        for instrument_id, live in self._quotes.items():
            contract = self._registry.get(instrument_id)
            if contract is None or contract.key not in wanted:
                continue
            age = (now - live.ts).total_seconds()
            if age > self._spec.max_quote_age_seconds:
                stale += 1
                continue
            fresh.append(self._with_iv(live.quote, contract, now, today))

        completeness = len(fresh) / len(wanted) if wanted else 0.0
        self._last_snapshot = now

        METRICS.gauge("chain_completeness", completeness, underlying=str(self._spec.underlying_id))
        METRICS.inc("chain_snapshot_total", underlying=str(self._spec.underlying_id))
        if stale:
            METRICS.inc("chain_stale_quote_total", value=float(stale))

        if completeness < 0.95:
            log.warning(
                "incomplete chain snapshot - stored and flagged, not discarded",
                fields={
                    "completeness": round(completeness, 3),
                    "have": len(fresh),
                    "want": len(wanted),
                    "stale": stale,
                },
            )

        return ChainSnapshot(
            ts=now,
            underlying_id=self._spec.underlying_id,
            spot=self._spot,  # captured with the chain, never joined later
            quotes=sorted(fresh, key=lambda q: (q.expiry, q.strike, q.opt_type.value)),
            completeness=min(1.0, completeness),
        )

    def _with_iv(
        self, quote: OptionQuote, contract: OptionInstrument, now: datetime, today: date
    ) -> OptionQuote:
        """Attach implied volatility, computed from mid and only when solvable."""
        if not self._compute_iv or self._spot is None:
            return quote
        mid = quote.mid
        if mid is None or mid <= 0:
            return quote  # one-sided book: no opinion is the honest answer

        minutes = minutes_to_expiry(self._cal, now, contract.expiry)
        t = minutes / MINUTES_PER_YEAR
        if t <= 0:
            return quote
        r = self._spec.risk_free_rate
        fwd = forward_from_spot(self._spot, r, t)
        iv = implied_vol(
            mid,
            fwd,
            contract.strike,
            t,
            r,
            OptType.CALL if contract.opt_type is OptionType.CE else OptType.PUT,
        )
        return quote.model_copy(update={"iv": iv}) if iv is not None else quote

    @property
    def spot(self) -> float | None:
        return self._spot


def minutes_to_expiry(calendar: SessionCalendar, now: datetime, expiry: date) -> float:
    """Trading minutes remaining, not calendar minutes.

    An option expiring in three calendar days has far fewer trading minutes than
    the naive difference suggests once a weekend is in the way, and pricing it
    with calendar time systematically overstates its value.
    """
    local_now = calendar.local(now)
    today = local_now.date()
    if expiry < today:
        return 0.0

    if expiry == today:
        close = local_now.replace(
            hour=calendar.continuous_end.hour,
            minute=calendar.continuous_end.minute,
            second=0,
            microsecond=0,
        )
        return max(0.0, (close - local_now).total_seconds() / 60.0)

    minutes = float(max(0, calendar.minutes_to_close(now)))
    cursor = today + timedelta(days=1)
    while cursor <= expiry:
        try:
            trading = calendar.is_trading_day(cursor)
        except LookupError:
            # No calendar for that year. Fall back to weekdays and say so, rather
            # than either crashing the pricer or silently counting weekends.
            trading = cursor.weekday() < 5
        if trading:
            minutes += calendar.expected_continuous_minutes
        cursor += timedelta(days=1)
    return minutes


# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #
@dataclass
class ChainIssue:
    kind: str
    detail: str
    strike: float | None = None


@dataclass
class ChainQualityReport:
    completeness: float
    quotes: int
    issues: list[ChainIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.completeness >= 0.95 and not self.issues

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.issues:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out


def check_chain(
    snap: ChainSnapshot,
    *,
    rate: float = 0.065,
    minutes_to_expiry_hint: float | None = None,
    max_parity_gap_pct: float = 0.02,
) -> ChainQualityReport:
    """Validate a snapshot before anything downstream trusts it.

    Runs the checks from the blueprint's data quality table: crossed books,
    implied volatility sanity, monotonicity of calls in strike, and put-call
    parity. Parity is the most useful of the four - it catches a single stale
    leg that would otherwise distort a whole smile fit.
    """
    report = ChainQualityReport(completeness=snap.completeness, quotes=len(snap.quotes))

    by_key: dict[tuple[float, float, str], OptionQuote] = {}
    for q in snap.quotes:
        by_key[(q.expiry.timestamp(), q.strike, q.opt_type.value)] = q

        if q.bid is not None and q.ask is not None and q.bid > q.ask:
            report.issues.append(ChainIssue("crossed_book", f"bid {q.bid} > ask {q.ask}", q.strike))
        if q.iv is not None and not (0.03 < q.iv < 3.0):
            report.issues.append(ChainIssue("iv_out_of_range", f"iv={q.iv:.4f}", q.strike))

    # Calls must be non-increasing in strike; puts non-decreasing.
    for opt_type, expect_decreasing in ((OptionType.CE, True), (OptionType.PE, False)):
        rows = sorted(
            (q for q in snap.quotes if q.opt_type is opt_type and q.mid is not None),
            key=lambda q: (q.expiry, q.strike),
        )
        for a, b in pairwise(rows):
            if a.expiry != b.expiry:
                continue
            assert a.mid is not None and b.mid is not None
            violated = (b.mid > a.mid + 1e-9) if expect_decreasing else (b.mid < a.mid - 1e-9)
            if violated:
                report.issues.append(
                    ChainIssue(
                        "non_monotone",
                        f"{opt_type.value} {a.strike}->{b.strike}: {a.mid:.2f}->{b.mid:.2f}",
                        b.strike,
                    )
                )

    if minutes_to_expiry_hint is not None and minutes_to_expiry_hint > 0:
        t = minutes_to_expiry_hint / MINUTES_PER_YEAR
        fwd = forward_from_spot(snap.spot, rate, t)
        for (_exp, strike, kind), q in by_key.items():
            if kind != OptionType.CE.value:
                continue
            put = by_key.get((_exp, strike, OptionType.PE.value))
            if put is None or q.mid is None or put.mid is None:
                continue
            gap = put_call_parity_gap(q.mid, put.mid, fwd, strike, t, rate)
            if abs(gap) > max_parity_gap_pct * snap.spot / 100.0 * 5:
                report.issues.append(
                    ChainIssue("parity_gap", f"gap={gap:.2f} at strike {strike}", strike)
                )

    return report


def atm_straddle(snap: ChainSnapshot) -> tuple[OptionQuote, OptionQuote] | None:
    """The at-the-money call and put, which is where the implied move comes from."""
    if not snap.quotes:
        return None
    strike = snap.atm_strike()
    near_expiry = min(q.expiry for q in snap.quotes)
    call = next(
        (
            q
            for q in snap.quotes
            if q.strike == strike and q.opt_type is OptionType.CE and q.expiry == near_expiry
        ),
        None,
    )
    put = next(
        (
            q
            for q in snap.quotes
            if q.strike == strike and q.opt_type is OptionType.PE and q.expiry == near_expiry
        ),
        None,
    )
    return (call, put) if call and put else None


def liquid_quotes(
    snap: ChainSnapshot,
    *,
    min_oi: int = 0,
    max_spread_pct: float = 0.10,
) -> list[OptionQuote]:
    """Filter to strikes that could actually be traded.

    A theoretically optimal strike that cannot be exited is a trap, so liquidity
    is applied as a hard eligibility constraint rather than a soft preference.
    """
    out = []
    for q in snap.quotes:
        spread = q.spread_pct_of_mid
        if spread is None or spread > max_spread_pct:
            continue
        if min_oi and (q.oi or 0) < min_oi:
            continue
        out.append(q)
    return out


def build_registry_from_specs(
    underlying_id: int,
    expiries: Sequence[date],
    strikes: Sequence[float],
    *,
    first_id: int = 100_000,
) -> OptionRegistry:
    """Deterministic synthetic registry, for tests and the synthetic feed."""
    instruments: list[OptionInstrument] = []
    next_id = first_id
    for expiry in expiries:
        for strike in strikes:
            for opt_type in (OptionType.CE, OptionType.PE):
                instruments.append(
                    OptionInstrument(
                        instrument_id=next_id,
                        underlying_id=underlying_id,
                        expiry=expiry,
                        strike=strike,
                        opt_type=opt_type,
                    )
                )
                next_id += 1
    return OptionRegistry(instruments)
