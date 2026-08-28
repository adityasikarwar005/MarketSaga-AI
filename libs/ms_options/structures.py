"""M72 - what could be traded, and what could actually be got out of.

The enumerator's job is to produce candidates. The filter's job is to make sure
none of them is a structure you can enter and not leave.

No-trade is a candidate, not a fallback
-----------------------------------------

:func:`enumerate_structures` always returns a :data:`NO_TRADE` entry, on every
chain, including an empty one. That is not a convenience.

An enumerator that returns only the tradeable structures hands the ranker a set
in which *something* always wins, and "the best of a bad set" is indistinguishable
in the output from "a good opportunity". Both arrive as a structure with a score.
Making doing nothing a candidate that competes on merit is what lets a quiet
Tuesday produce a recommendation of no position, on the same axis, for the same
reason - rather than as an exception path somebody has to remember to write.

Eligibility is per leg, and that is still not enough
-----------------------------------------------------

Every leg of every structure is checked. Checking the leg you chose first and
assuming the wings are fine is how an iron condor gets built with two
untradeable legs: it prices beautifully, its Greeks are sensible, and one side
of it cannot be closed.

But per-leg liquidity passing does not make a structure tradeable, and this is
the part that gets missed. **Crossing cost compounds with leg count.** Four legs
each inside a 20%-of-mid spread cap is a round trip that crosses four spreads on
the way in and four on the way out. On a condor whose maximum profit is the net
credit, that can exceed everything the structure could ever make - while every
individual leg passes every individual filter.

So :class:`Candidate` carries ``crossing_cost`` and
:func:`viable_after_crossing` compares it against what the structure can win.
A structure that cannot clear its own spread is not a candidate that ranks
poorly; it is not a candidate.

The grid is filtered before the structures are built
------------------------------------------------------

Not after. Building every combination from the full strike ladder and then
discarding the ones with an ineligible leg produces the same answer far more
slowly, and it invites the real bug: a "nearest strike" helper that searches the
full ladder and returns a strike no structure is allowed to use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import OptionQuote
from libs.ms_options.greeks import Leg
from libs.ms_options.pricing import OptType

log = get_logger(__name__)

# Liquidity floors. Deliberately not "reasonable defaults" - each one is the
# level below which the quote stops being a price and starts being a placeholder.
MIN_OI = 500
MIN_VOLUME = 100
MAX_SPREAD_PCT = 0.10  # of mid

# A structure must be able to make more than it costs to cross, by this margin.
# At 1.0 a structure that exactly pays its own spread would qualify, which is a
# trade whose best case is breaking even.
MIN_EDGE_TO_CROSSING = 2.0


class StructureKind(StrEnum):
    """The shapes this enumerator can build.

    CALENDAR is deferred rather than forgotten: it spans two expiries, which
    makes its risk a function of the term structure rather than of one slice,
    and M70 fits slices independently. Adding it before that is modelled would
    give it a P&L distribution that is confidently wrong.
    """

    NO_TRADE = "no_trade"
    NAKED = "naked"
    VERTICAL_DEBIT = "vertical_debit"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"


class RejectReason(StrEnum):
    OI_TOO_LOW = "oi_too_low"
    VOLUME_TOO_LOW = "volume_too_low"
    SPREAD_TOO_WIDE = "spread_too_wide"
    NO_TWO_SIDED_MARKET = "no_two_sided_market"


class StructureError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Eligible:
    """One tradeable contract, with the numbers a structure needs to be costed."""

    strike: float
    kind: OptType
    expiry: datetime
    bid: float
    ask: float
    oi: int
    volume: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else math.inf


@dataclass
class GridStats:
    eligible: int = 0
    rejected: dict[RejectReason, int] = field(default_factory=dict)

    @property
    def seen(self) -> int:
        return self.eligible + sum(self.rejected.values())

    def reject(self, reason: RejectReason) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def render(self) -> str:
        lines = [f"    {self.eligible}/{self.seen} contracts eligible"]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {reason.value:<22} {n}")
        return "\n".join(lines)


@dataclass
class Grid:
    """The strikes a structure may be built from. Nothing else is reachable."""

    by_key: dict[tuple[float, OptType], Eligible] = field(default_factory=dict)
    stats: GridStats = field(default_factory=GridStats)

    def get(self, strike: float, kind: OptType) -> Eligible | None:
        return self.by_key.get((strike, kind))

    def strikes(self, kind: OptType) -> list[float]:
        return sorted(s for (s, k) in self.by_key if k is kind)

    @property
    def both_sides(self) -> list[float]:
        """Strikes with an eligible call *and* an eligible put.

        Straddles and condors need both. A strike where only the call trades is
        eligible for a naked call and for nothing that pairs.
        """
        calls = set(self.strikes(OptType.CALL))
        return sorted(calls & set(self.strikes(OptType.PUT)))

    def nearest(self, target: float, kind: OptType) -> float | None:
        """The closest **eligible** strike, or None.

        Searching the full ladder here is the bug this signature exists to
        prevent: it returns a strike that looks right, is not in the grid, and
        every structure built on it fails at the broker instead of at the filter.
        """
        candidates = self.strikes(kind)
        return min(candidates, key=lambda s: abs(s - target)) if candidates else None

    def __len__(self) -> int:
        return len(self.by_key)

    def render(self) -> str:
        return self.stats.render()


def build_grid(
    quotes: Sequence[OptionQuote],
    *,
    expiry: datetime,
    min_oi: int = MIN_OI,
    min_volume: int = MIN_VOLUME,
    max_spread_pct: float = MAX_SPREAD_PCT,
) -> Grid:
    """Filter first, enumerate second.

    ``volume`` and ``oi`` missing are treated as failures rather than as passes.
    An absent field is not evidence of liquidity, and defaulting it to "fine" is
    how a chain with a broken feed produces a full grid.
    """
    grid = Grid()
    for q in quotes:
        if q.expiry != expiry:
            continue
        grid.stats.eligible += 0  # counted below

        mid = q.mid  # None unless bid > 0 and ask present
        if mid is None or mid <= 0.0:
            grid.stats.reject(RejectReason.NO_TWO_SIDED_MARKET)
            continue
        if q.oi is None or q.oi < min_oi:
            grid.stats.reject(RejectReason.OI_TOO_LOW)
            continue
        if q.volume is None or q.volume < min_volume:
            grid.stats.reject(RejectReason.VOLUME_TOO_LOW)
            continue
        spread_pct = q.spread_pct_of_mid
        if spread_pct is None or spread_pct > max_spread_pct:
            grid.stats.reject(RejectReason.SPREAD_TOO_WIDE)
            continue

        kind = OptType.CALL if str(q.opt_type).upper().startswith("C") else OptType.PUT
        assert q.bid is not None and q.ask is not None  # mid is not None implies both
        grid.by_key[(q.strike, kind)] = Eligible(
            strike=q.strike,
            kind=kind,
            expiry=q.expiry,
            bid=q.bid,
            ask=q.ask,
            oi=q.oi,
            volume=q.volume,
        )
        grid.stats.eligible += 1
    return grid


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One structure that could be traded, and what it costs to get in and out."""

    kind: StructureKind
    legs: tuple[Leg, ...]
    contracts: tuple[Eligible, ...]
    expiry: datetime | None = None

    @property
    def is_no_trade(self) -> bool:
        return self.kind is StructureKind.NO_TRADE

    @property
    def net_debit(self) -> float:
        """Positive is paid, negative is received. Priced at mid, per unit.

        Mid rather than the crossing price on purpose: this is the structure's
        economic value, and the cost of crossing is a separate number so that
        neither one can quietly absorb the other.
        """
        total = 0.0
        for leg, c in zip(self.legs, self.contracts, strict=True):
            total += c.mid * leg.lots * (1 if leg.is_long else -1)
        return total

    @property
    def crossing_cost(self) -> float:
        """Round-trip spread across every leg, per unit.

        The full spread per leg, not the half: entering crosses from mid to the
        touch, and exiting crosses back. Four legs is four spreads, which is the
        number that makes a condor untradeable while each of its legs passes.
        """
        return sum(c.spread * leg.lots for leg, c in zip(self.legs, self.contracts, strict=True))

    @property
    def width(self) -> float:
        """Widest strike separation within one option kind. 0 for single legs."""
        by_kind: dict[OptType, list[float]] = {}
        for leg in self.legs:
            by_kind.setdefault(leg.kind, []).append(leg.strike)
        widths = [max(v) - min(v) for v in by_kind.values() if len(v) > 1]
        return max(widths) if widths else 0.0

    @property
    def max_profit(self) -> float | None:
        """``None`` means unbounded, which is a different thing from large."""
        if self.kind in (StructureKind.NO_TRADE,):
            return 0.0
        if self.kind in (StructureKind.NAKED, StructureKind.STRADDLE, StructureKind.STRANGLE):
            return None  # long premium, unbounded upside
        if self.kind is StructureKind.VERTICAL_DEBIT:
            return self.width - self.net_debit
        if self.kind is StructureKind.IRON_CONDOR:
            return -self.net_debit  # a credit received
        raise StructureError(f"no max profit defined for {self.kind}")

    @property
    def max_loss(self) -> float:
        if self.kind is StructureKind.NO_TRADE:
            return 0.0
        if self.kind is StructureKind.IRON_CONDOR:
            return self.width + self.net_debit  # width less the credit
        return self.net_debit  # every other structure here is a debit

    def render(self) -> str:
        if self.is_no_trade:
            return "    no_trade         -                        (always a candidate)"
        legs = " ".join(
            f"{'+' if leg.is_long else '-'}{leg.strike:.0f}{leg.kind.value[0]}" for leg in self.legs
        )
        mp = "unbounded" if self.max_profit is None else f"{self.max_profit:.2f}"
        return (
            f"    {self.kind.value:<16} {legs:<24} debit={self.net_debit:+8.2f} "
            f"cross={self.crossing_cost:6.2f}  max+={mp:<10} max-={self.max_loss:.2f}"
        )


NO_TRADE = Candidate(kind=StructureKind.NO_TRADE, legs=(), contracts=())


def viable_after_crossing(candidate: Candidate, *, min_ratio: float = MIN_EDGE_TO_CROSSING) -> bool:
    """Can this structure win more than it costs to cross?

    No-trade is always viable - it crosses nothing. An unbounded structure is
    judged against its own debit rather than against infinity, because "can make
    unlimited money" is not an answer to "can it clear its spread".

    This is the check that per-leg liquidity cannot make: four legs each inside
    the spread cap is a round trip crossing four spreads, and on a condor whose
    maximum profit is the credit received, that routinely exceeds everything the
    structure could ever make.
    """
    if candidate.is_no_trade:
        return True
    if candidate.crossing_cost <= 0.0:
        return True
    ceiling = candidate.max_profit
    if ceiling is None:
        ceiling = abs(candidate.net_debit)
    return ceiling >= min_ratio * candidate.crossing_cost


# --------------------------------------------------------------------------- #
# The enumerator
# --------------------------------------------------------------------------- #
def _leg(c: Eligible, *, is_long: bool, lots: int, lot_size: int) -> Leg:
    return Leg(
        strike=c.strike,
        kind=c.kind,
        expiry=c.expiry,
        lots=lots,
        lot_size=lot_size,
        is_long=is_long,
    )


def _build(
    kind: StructureKind, spec: Sequence[tuple[Eligible, bool]], *, lots: int, lot_size: int
) -> Candidate:
    legs = tuple(_leg(c, is_long=long, lots=lots, lot_size=lot_size) for c, long in spec)
    return Candidate(
        kind=kind,
        legs=legs,
        contracts=tuple(c for c, _ in spec),
        expiry=spec[0][0].expiry if spec else None,
    )


def enumerate_structures(
    grid: Grid,
    *,
    forward: float,
    lot_size: int,
    lots: int = 1,
    max_width_steps: int = 4,
    min_ratio: float = MIN_EDGE_TO_CROSSING,
) -> list[Candidate]:
    """Every structure buildable from the eligible grid, plus no-trade.

    Reads nothing but ``grid``. Every strike in every returned structure came
    from it, which is what makes "the grid is limited to eligible strikes" a
    property of the enumerator rather than a comment.
    """
    out: list[Candidate] = [NO_TRADE]

    calls = grid.strikes(OptType.CALL)
    puts = grid.strikes(OptType.PUT)
    paired = grid.both_sides

    # --- naked long, one leg -------------------------------------------------
    for kind, strikes in ((OptType.CALL, calls), (OptType.PUT, puts)):
        for s in strikes:
            c = grid.get(s, kind)
            if c is not None:
                out.append(_build(StructureKind.NAKED, [(c, True)], lots=lots, lot_size=lot_size))

    # --- vertical debit spreads ---------------------------------------------
    # Calls: buy the lower strike, sell the higher. Reversing it makes a credit
    # spread with inverted risk, priced and displayed identically.
    for i, low in enumerate(calls):
        for high in calls[i + 1 : i + 1 + max_width_steps]:
            a, b = grid.get(low, OptType.CALL), grid.get(high, OptType.CALL)
            if a and b:
                out.append(
                    _build(
                        StructureKind.VERTICAL_DEBIT,
                        [(a, True), (b, False)],
                        lots=lots,
                        lot_size=lot_size,
                    )
                )
    # Puts: buy the higher strike, sell the lower.
    for i, high in enumerate(reversed(puts)):
        lower = list(reversed(puts))[i + 1 : i + 1 + max_width_steps]
        for low in lower:
            a, b = grid.get(high, OptType.PUT), grid.get(low, OptType.PUT)
            if a and b:
                out.append(
                    _build(
                        StructureKind.VERTICAL_DEBIT,
                        [(a, True), (b, False)],
                        lots=lots,
                        lot_size=lot_size,
                    )
                )

    # --- straddles: same strike, both kinds ----------------------------------
    for s in paired:
        c, p = grid.get(s, OptType.CALL), grid.get(s, OptType.PUT)
        if c and p:
            out.append(
                _build(StructureKind.STRADDLE, [(c, True), (p, True)], lots=lots, lot_size=lot_size)
            )

    # --- strangles: OTM call above the forward, OTM put below ----------------
    for cs in [s for s in calls if s > forward]:
        for ps in [s for s in puts if s < forward]:
            c, p = grid.get(cs, OptType.CALL), grid.get(ps, OptType.PUT)
            if c and p:
                out.append(
                    _build(
                        StructureKind.STRANGLE,
                        [(c, True), (p, True)],
                        lots=lots,
                        lot_size=lot_size,
                    )
                )

    # --- iron condors: short strangle, long wings further out ----------------
    otm_calls = [s for s in calls if s > forward]
    otm_puts = [s for s in puts if s < forward]
    for i, short_call in enumerate(otm_calls):
        for long_call in otm_calls[i + 1 : i + 1 + max_width_steps]:
            for j, short_put in enumerate(reversed(otm_puts)):
                for long_put in list(reversed(otm_puts))[j + 1 : j + 1 + max_width_steps]:
                    parts = [
                        (grid.get(short_call, OptType.CALL), False),
                        (grid.get(long_call, OptType.CALL), True),
                        (grid.get(short_put, OptType.PUT), False),
                        (grid.get(long_put, OptType.PUT), True),
                    ]
                    if any(c is None for c, _ in parts):
                        continue
                    out.append(
                        _build(
                            StructureKind.IRON_CONDOR,
                            [(c, long) for c, long in parts if c is not None],
                            lots=lots,
                            lot_size=lot_size,
                        )
                    )

    viable = [c for c in out if viable_after_crossing(c, min_ratio=min_ratio)]
    dropped = len(out) - len(viable)
    if dropped:
        log.info(
            "structures dropped on crossing cost",
            fields={"dropped": dropped, "kept": len(viable)},
        )
    return viable


def summarise(candidates: Sequence[Candidate], grid: Grid) -> str:
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.kind.value] = counts.get(c.kind.value, 0) + 1
    lines = ["  eligible grid:", grid.render(), "  candidates:"]
    for kind, n in sorted(counts.items()):
        lines.append(f"    {kind:<16} {n}")
    lines.append(
        "    no_trade is present on every chain, including an empty one - so that "
        "doing nothing competes on the same axis rather than being an exception "
        "path somebody has to remember to write."
    )
    return "\n".join(lines)
