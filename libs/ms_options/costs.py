"""M74 - the cost model.

Where most retail quant projects quietly die. An index-option strategy that
looks profitable gross is usually not profitable net, and the gap is not a
rounding error: on a two-leg spread held for an hour, round-trip costs routinely
eat a double-digit percentage of the premium. Every edge this project ever
reports has to survive this file.

What is charged, and on what
----------------------------

The confusing part of Indian option costs is that the components are levied on
three different bases, and using the wrong one is a large, silent error:

* **Brokerage** - flat per executed *order*. Not per lot, not per leg-value. A
  four-order spread pays it four times regardless of size.
* **Exchange transaction charge** - on **premium** turnover, not notional. Since
  premium is a small fraction of notional, charging it on notional overstates
  costs by an order of magnitude and is the more common mistake.
* **STT** - on the **sell** side only, on premium, for a squared-off option. On
  *exercise* it is charged on **intrinsic value**, which is a completely
  different and much larger number (see below).
* **SEBI fee** - on premium turnover.
* **Stamp duty** - buy side only.
* **GST** - on the *charges*, not on the turnover. It compounds on brokerage
  plus exchange plus SEBI fee, so it is roughly an 18% surcharge on the fee
  stack rather than a tax on the trade.

The STT trap, and what changed about it
--------------------------------------

Letting an in-the-money option expire has cost people money for reasons that had
nothing to do with being wrong about the market. How large that risk is depends
entirely on **what base the exercise STT is levied on**, and that base has not
always been the same one:

* On the **full settlement value** - roughly spot times quantity - the charge is
  enormous relative to the position. Seventy-five units of an option on a 24,000
  index pays on 1.8 million of notional, which can exceed everything the option
  is worth. This is the version the folklore describes, and it did wipe out
  accounts that had called the direction correctly.
* On **intrinsic value** - settlement minus strike - the charge is far smaller,
  and squaring off stops being automatically cheaper. Closing pays brokerage,
  slippage and its own sell-side STT, and on realistic inputs the arithmetic
  comes out the other way.

``ExerciseBase`` makes that choice explicit rather than baking one in, because
the right answer depends on which rule was in force on the day, and a backtest
has to price an old trade under the old rule.

So this module does **not** assert that squaring off is always right. It costs
both sides and reports which is cheaper, naming the base it assumed. An earlier
draft of this file asserted the folklore in its docstring, and its own
arithmetic contradicted it on the first realistic example - which is the whole
argument for computing rather than asserting.

Slippage
--------

The statutory part above is exact arithmetic. Slippage is not: it is an estimate
that has to be calibrated from your own fills, and until it is, it is a guess
with a placeholder attached. ``SlippageModel`` is deliberately explicit about
whether it has been calibrated, because an uncalibrated slippage estimate
presented as a cost is how a backtest acquires false precision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from statistics import fmean, median

from libs.ms_core.config import CostSchedule
from libs.ms_core.obs import get_logger

log = get_logger(__name__)


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class ExerciseBase(StrEnum):
    """What the exercise STT is charged on. The whole size of the trap.

    Explicit and never defaulted silently: the two bases differ by roughly the
    inverse of moneyness, which for a near-the-money option is a factor of fifty
    or more.
    """

    INTRINSIC = "intrinsic"  # settlement minus strike
    SETTLEMENT_VALUE = "settlement_value"  # the full contract value


@dataclass(frozen=True)
class Leg:
    """One order in one instrument. Quantity is in **lots**, never fractional."""

    side: Side
    price: float  # premium per unit
    lots: int
    lot_size: int
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.lots <= 0:
            raise ValueError("lots must be positive; express direction with side")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.price < 0:
            raise ValueError("premium cannot be negative")

    @property
    def units(self) -> int:
        return self.lots * self.lot_size

    @property
    def premium_turnover(self) -> float:
        return self.price * self.units


@dataclass(frozen=True)
class CostBreakdown:
    """Every component, kept separate.

    Separate because a single total hides which assumption is wrong. When the
    net edge disappears, the question is always *which* component ate it, and a
    lump sum cannot answer that.
    """

    brokerage: float = 0.0
    exchange: float = 0.0
    stt: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    slippage: float = 0.0
    orders: int = 0
    premium_turnover: float = 0.0
    slippage_calibrated: bool = False

    @property
    def statutory(self) -> float:
        """Everything that is exact arithmetic rather than an estimate."""
        return self.brokerage + self.exchange + self.stt + self.sebi + self.stamp_duty + self.gst

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            exchange=self.exchange + other.exchange,
            stt=self.stt + other.stt,
            sebi=self.sebi + other.sebi,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            gst=self.gst + other.gst,
            slippage=self.slippage + other.slippage,
            orders=self.orders + other.orders,
            premium_turnover=self.premium_turnover + other.premium_turnover,
            slippage_calibrated=self.slippage_calibrated and other.slippage_calibrated,
        )

    def as_pct_of_premium(self) -> float:
        return self.total / self.premium_turnover if self.premium_turnover > 0 else 0.0

    def render(self) -> str:
        warn = "" if self.slippage_calibrated else "  (slippage UNCALIBRATED)"
        return (
            f"  {self.orders} orders on {self.premium_turnover:,.0f} premium turnover\n"
            f"    brokerage {self.brokerage:>8.2f}  exchange {self.exchange:>8.2f}  "
            f"STT {self.stt:>8.2f}\n"
            f"    SEBI {self.sebi:>8.4f}  stamp {self.stamp_duty:>8.2f}  "
            f"GST {self.gst:>8.2f}\n"
            f"    slippage {self.slippage:>8.2f}{warn}\n"
            f"    total {self.total:,.2f} = {self.as_pct_of_premium():.2%} of premium"
        )


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #
@dataclass
class SlippageModel:
    """Half-spread plus impact, in premium points per unit.

    Uncalibrated by default and loudly so. A backtest that silently applies a
    made-up slippage number produces a net P&L with false precision - the
    statutory costs are exact to the paisa and the slippage beside them is a
    guess, and the sum reads as though it were all exact.

    Calibrate from your own fills with :func:`calibrate_slippage` and refresh it
    weekly, because spreads widen with volatility and the estimate goes stale in
    exactly the conditions where it matters most.
    """

    half_spread_points: float = 0.5
    impact_per_lot: float = 0.0
    calibrated: bool = False
    samples: int = 0
    calibrated_on: date | None = None

    def cost(self, leg: Leg) -> float:
        """Crossing the spread costs the same whichever way you are going."""
        per_unit = self.half_spread_points + self.impact_per_lot * max(0, leg.lots - 1)
        return per_unit * leg.units


@dataclass(frozen=True)
class Fill:
    """One observed fill, for calibrating slippage against intent."""

    side: Side
    intended_price: float
    filled_price: float
    lots: int
    day: date

    @property
    def adverse_points(self) -> float:
        """Positive means the fill was worse than intended.

        Signed by side: paying more when buying and receiving less when selling
        are the same event, and averaging them unsigned would report zero
        slippage for a book that is losing on both.
        """
        if self.side is Side.BUY:
            return self.filled_price - self.intended_price
        return self.intended_price - self.filled_price


def calibrate_slippage(fills: Sequence[Fill], *, min_fills: int = 30) -> SlippageModel:
    """Fit half-spread and per-lot impact from observed fills.

    The median, not the mean. One fill during a circuit halt can be worth
    hundreds of normal fills, and a mean slippage dominated by a single
    unrepeatable event is not a forecast of anything.

    Impact is estimated as the slope of adverse points against size, floored at
    zero: a negative estimated impact means larger orders filled *better*, which
    is a small-sample artefact rather than a discovery about the market.
    """
    if len(fills) < min_fills:
        log.info(
            "slippage not calibrated: too few fills",
            fields={"fills": len(fills), "needed": min_fills},
        )
        return SlippageModel(samples=len(fills))

    adverse = [f.adverse_points for f in fills]
    half_spread = median(adverse)

    sizes = [float(f.lots) for f in fills]
    mean_size = fmean(sizes)
    sxx = sum((s - mean_size) ** 2 for s in sizes)
    impact = 0.0
    if sxx > 0:
        mean_adverse = fmean(adverse)
        sxy = sum((s - mean_size) * (a - mean_adverse) for s, a in zip(sizes, adverse, strict=True))
        impact = max(0.0, sxy / sxx)

    model = SlippageModel(
        half_spread_points=max(0.0, half_spread),
        impact_per_lot=impact,
        calibrated=True,
        samples=len(fills),
        calibrated_on=max(f.day for f in fills),
    )
    log.info(
        "slippage calibrated",
        fields={
            "fills": len(fills),
            "half_spread": round(model.half_spread_points, 4),
            "impact_per_lot": round(model.impact_per_lot, 4),
        },
    )
    return model


# --------------------------------------------------------------------------- #
# The cost engine
# --------------------------------------------------------------------------- #
def leg_cost(
    leg: Leg,
    schedule: CostSchedule,
    *,
    slippage: SlippageModel | None = None,
) -> CostBreakdown:
    """Statutory charges plus slippage for one order.

    Each component is levied on its own base, and the bases are not the same
    number - see the module docstring. Charging the exchange fee on notional
    rather than premium is the single most common way this arithmetic goes
    wrong, and it overstates costs by roughly the inverse of moneyness.
    """
    premium = leg.premium_turnover

    brokerage = schedule.brokerage_per_order
    exchange = premium * schedule.exchange_txn_pct_of_premium
    sebi = premium * schedule.sebi_fee_pct_of_turnover
    stt = premium * schedule.stt_pct_sell_premium if leg.side is Side.SELL else 0.0
    stamp = premium * schedule.stamp_duty_pct_buy if leg.side is Side.BUY else 0.0

    # GST is on the charges, not the turnover. STT and stamp duty are taxes and
    # are not themselves taxed.
    gst = (brokerage + exchange + sebi) * schedule.gst_pct_on_charges

    model = slippage or SlippageModel()
    return CostBreakdown(
        brokerage=brokerage,
        exchange=exchange,
        stt=stt,
        sebi=sebi,
        stamp_duty=stamp,
        gst=gst,
        slippage=model.cost(leg),
        orders=1,
        premium_turnover=premium,
        slippage_calibrated=model.calibrated,
    )


def round_trip_cost(
    legs: Sequence[Leg],
    schedule: CostSchedule,
    *,
    slippage: SlippageModel | None = None,
) -> CostBreakdown:
    """Cost of opening **and** closing every leg.

    A two-leg spread is four orders, not two. Costing only the entry is the
    error that makes a strategy look viable at exactly twice its real edge, and
    it is easy to make because the exit legs do not exist yet when the trade is
    being evaluated.

    The exit is priced at the same premium as the entry. That is not a forecast
    - it is a deliberate convention that keeps this function about *charges*.
    A caller with real exit prices should sum ``leg_cost`` over actual fills.
    """
    total = CostBreakdown(slippage_calibrated=True)
    for leg in legs:
        exit_leg = Leg(
            side=Side.SELL if leg.side is Side.BUY else Side.BUY,
            price=leg.price,
            lots=leg.lots,
            lot_size=leg.lot_size,
            symbol=leg.symbol,
        )
        total = total + leg_cost(leg, schedule, slippage=slippage)
        total = total + leg_cost(exit_leg, schedule, slippage=slippage)
    return total


def breakeven_pct_of_premium(
    legs: Sequence[Leg],
    schedule: CostSchedule,
    *,
    slippage: SlippageModel | None = None,
) -> float:
    """How far the position must move, as a fraction of premium paid, to break even.

    The number that belongs in front of a human before they act. "This trade
    needs a 14% move in the premium just to get back to flat" is a sentence that
    stops bad trades; a rupee cost figure is not.
    """
    costs = round_trip_cost(legs, schedule, slippage=slippage)
    net_premium = sum(leg.premium_turnover * (1 if leg.side is Side.BUY else -1) for leg in legs)
    base = abs(net_premium)
    if base <= 0:
        return math.inf
    return costs.total / base


# --------------------------------------------------------------------------- #
# The STT trap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpiryDecision:
    """Square off, or let it expire. Almost always the former."""

    intrinsic: float
    expiry_stt: float
    square_off_cost: float
    square_off: bool
    reason: str
    base: ExerciseBase = ExerciseBase.INTRINSIC

    @property
    def saving(self) -> float:
        return self.expiry_stt - self.square_off_cost

    def render(self) -> str:
        verdict = "SQUARE OFF" if self.square_off else "let it expire"
        return (
            f"  intrinsic {self.intrinsic:,.2f}  expiry STT {self.expiry_stt:,.2f}  "
            f"square-off {self.square_off_cost:,.2f}\n"
            f"    -> {verdict}: {self.reason}"
        )


# Placeholder, like every other rate in this file. Verify against the current
# Finance Act before any profitability claim - audit_cost_config() refuses until
# then.
EXERCISE_STT_PCT = 0.00125


def expiry_stt(
    intrinsic_per_unit: float,
    units: int,
    *,
    rate: float = EXERCISE_STT_PCT,
    base: ExerciseBase = ExerciseBase.INTRINSIC,
    settlement_price: float | None = None,
) -> float:
    """STT charged when an in-the-money option is exercised at expiry.

    Under ``INTRINSIC`` the charge is on settlement minus strike. Under
    ``SETTLEMENT_VALUE`` it is on the full contract value - the regime the
    horror stories come from - which requires ``settlement_price``.

    Raises rather than guessing when the settlement base is asked for without a
    price to apply it to. Quietly falling back to intrinsic would report a
    charge fifty times too small for the exact case the caller was worried about.
    """
    if intrinsic_per_unit <= 0:
        return 0.0
    if base is ExerciseBase.SETTLEMENT_VALUE:
        if settlement_price is None:
            raise ValueError(
                "settlement_price is required for the settlement-value base - "
                "it is not derivable from intrinsic alone"
            )
        return settlement_price * units * rate
    return intrinsic_per_unit * units * rate


def should_square_off(
    leg: Leg,
    intrinsic_per_unit: float,
    schedule: CostSchedule,
    *,
    slippage: SlippageModel | None = None,
    base: ExerciseBase = ExerciseBase.INTRINSIC,
    settlement_price: float | None = None,
) -> ExpiryDecision:
    """Compare the exercise charge against the cost of closing the position.

    Only long options face this: a short position that expires in the money is
    assigned, and the exercise STT falls on the exercising counterparty, not on
    the writer. Getting that backwards would make the model refuse to hold any
    short option to expiry, which is a normal and often intended thing to do.
    """
    if leg.side is Side.SELL:
        return ExpiryDecision(
            intrinsic=intrinsic_per_unit * leg.units,
            expiry_stt=0.0,
            square_off_cost=0.0,
            square_off=False,
            reason="short position: exercise STT falls on the buyer, not the writer",
        )

    if intrinsic_per_unit <= 0:
        return ExpiryDecision(
            intrinsic=0.0,
            expiry_stt=0.0,
            square_off_cost=0.0,
            square_off=False,
            reason="out of the money at expiry: it lapses worthless and costs nothing",
        )

    charge = expiry_stt(intrinsic_per_unit, leg.units, base=base, settlement_price=settlement_price)
    close = leg_cost(
        Leg(
            side=Side.SELL,
            price=intrinsic_per_unit,
            lots=leg.lots,
            lot_size=leg.lot_size,
            symbol=leg.symbol,
        ),
        schedule,
        slippage=slippage,
    )
    square = charge > close.total
    ratio = charge / close.total if close.total > 0 else float("inf")
    return ExpiryDecision(
        intrinsic=intrinsic_per_unit * leg.units,
        expiry_stt=charge,
        square_off_cost=close.total,
        square_off=square,
        base=base,
        reason=(
            f"exercise STT on {base.value} is {ratio:.1f}x the cost of closing"
            if square
            else f"closing costs {1 / ratio:.1f}x the exercise charge on {base.value}"
        ),
    )


# --------------------------------------------------------------------------- #
# Does the edge survive?
# --------------------------------------------------------------------------- #
@dataclass
class EdgeAfterCosts:
    """The only version of an edge that means anything."""

    gross_edge: float  # expected P&L before costs, in rupees
    costs: CostBreakdown
    lots: int

    @property
    def net_edge(self) -> float:
        return self.gross_edge - self.costs.total

    @property
    def cost_to_edge(self) -> float:
        """Costs as a fraction of the gross edge. Above 1 means there is nothing left."""
        return self.costs.total / self.gross_edge if self.gross_edge > 0 else math.inf

    @property
    def survives(self) -> bool:
        return self.net_edge > 0

    def render(self) -> str:
        verdict = "survives" if self.survives else "DOES NOT SURVIVE COSTS"
        ratio = "inf" if math.isinf(self.cost_to_edge) else f"{self.cost_to_edge:.1%}"
        warn = "" if self.costs.slippage_calibrated else "  (slippage uncalibrated)"
        return (
            f"  gross {self.gross_edge:,.2f} - costs {self.costs.total:,.2f} "
            f"= net {self.net_edge:,.2f}{warn}\n"
            f"    costs are {ratio} of gross edge  ->  {verdict}"
        )


def edge_after_costs(
    gross_edge: float,
    legs: Sequence[Leg],
    schedule: CostSchedule,
    *,
    slippage: SlippageModel | None = None,
) -> EdgeAfterCosts:
    return EdgeAfterCosts(
        gross_edge=gross_edge,
        costs=round_trip_cost(legs, schedule, slippage=slippage),
        lots=sum(leg.lots for leg in legs),
    )


@dataclass
class CostConfigAudit:
    """Whether the rates in force are real or still placeholders."""

    effective_from: date
    source: str
    verified: bool
    slippage_calibrated: bool
    notes: list[str] = field(default_factory=list)

    @property
    def safe_for_profitability_claims(self) -> bool:
        return self.verified and self.slippage_calibrated

    def render(self) -> str:
        lines = [f"  cost schedule effective {self.effective_from}: {self.source}"]
        lines += [f"    - {n}" for n in self.notes]
        lines.append(
            "    -> safe for profitability claims"
            if self.safe_for_profitability_claims
            else "    -> NOT safe for profitability claims"
        )
        return "\n".join(lines)


def audit_cost_config(
    schedule: CostSchedule, slippage: SlippageModel | None = None
) -> CostConfigAudit:
    """Refuse to let placeholder rates masquerade as measured ones.

    ``config/costs.yaml`` ships with plausible magnitudes so the pipeline runs
    end to end, and every one of them is marked TODO. A backtest built on those
    is a rehearsal, not a result - and the difference has to be visible in the
    output rather than remembered from a comment in a YAML file.
    """
    notes = []
    verified = "TODO" not in schedule.source and "PLACEHOLDER" not in schedule.source.upper()
    if not verified:
        notes.append("statutory rates are placeholders - verify against the source circulars")

    model = slippage or SlippageModel()
    if not model.calibrated:
        notes.append(
            f"slippage is a guess ({model.half_spread_points} points) - "
            "calibrate from your own fills"
        )
    elif model.samples < 100:
        notes.append(f"slippage calibrated on only {model.samples} fills")

    return CostConfigAudit(
        effective_from=schedule.effective_from,
        source=schedule.source,
        verified=verified,
        slippage_calibrated=model.calibrated,
        notes=notes,
    )
