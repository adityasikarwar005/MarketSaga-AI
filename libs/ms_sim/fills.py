"""M81 - the fill and slippage model.

Every optimistic backtest in history is optimistic here. The model has an edge,
the costs are subtracted, and then the fills are assumed to happen at the price
that was wanted — which is the one assumption that is never true and the one
that never shows up as an error.

This module exists to be pessimistic on purpose, and to say why each pessimism
is there.

The tie rule
------------

A bar has a high and a low. If a position's target sits below the high and its
stop sits above the low, both were touched, and a bar does not record which came
first. There are three ways to resolve it:

* assume the **target** filled — flatters every result, and is what a naive
  implementation does by accident because targets are usually checked first;
* assume the **stop** filled — pessimistic, and cannot be beaten by a real fill;
* model the path — needs tick data the backtest does not have at bar resolution.

This takes the second. Not because stops always fill first, but because the
error is bounded and one-directional: a real day can only ever be better than
what is reported, and a backtest that can only understate is one whose number
means something. The opposite convention produces a strategy that looks
profitable in exactly the sessions that were most volatile.

Marketable versus resting
-------------------------

A marketable order crosses the spread and fills at the **far touch** — the price
someone else is willing to trade at, not the mid. Backtests that fill at mid are
claiming half the spread as free profit on every leg, which on a four-order
option spread is most of the edge.

A resting limit order is different: it only fills if the market **trades
through** it, and even then only if the queue in front of it cleared. Assuming a
limit order fills the moment price touches it assumes you were first in the
queue every time, which nobody is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# How much of the resting size at a price level is assumed to be ahead of ours.
# Being last in the queue is the assumption that cannot be beaten by reality.
DEFAULT_QUEUE_AHEAD = 1.0

# Impact grows with size relative to what is showing. Square-root is the
# standard shape and is used here as an admitted approximation - the constant is
# a placeholder until real fills calibrate it.
DEFAULT_IMPACT_COEFF = 0.5


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class Quote:
    """One side of the book, at the moment an order is considered."""

    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("quotes must be positive")
        if self.ask < self.bid:
            raise ValueError(
                f"crossed quote: ask {self.ask} below bid {self.bid}. A crossed "
                "book is a data problem, and filling against one invents money."
            )

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0.0

    def far_touch(self, side: Side) -> float:
        """What a marketable order actually pays. Never the mid."""
        return self.ask if side is Side.BUY else self.bid

    def near_touch(self, side: Side) -> float:
        return self.bid if side is Side.BUY else self.ask

    def size_at(self, side: Side) -> int:
        """Displayed size on the side a marketable order would hit."""
        return self.ask_size if side is Side.BUY else self.bid_size


@dataclass(frozen=True)
class Order:
    side: Side
    qty: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive; direction is the side")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("a limit order needs a limit price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError(
                "a market order with a limit price is two different orders. " "Say which one."
            )


@dataclass(frozen=True)
class Fill:
    qty: int
    price: float
    ts: datetime | None = None
    reason: str = ""

    @property
    def filled(self) -> bool:
        return self.qty > 0


@dataclass
class FillResult:
    """What happened, and what did not.

    ``unfilled`` is a first-class part of the answer. A model that silently
    returns a smaller fill than requested lets a backtest quietly trade a
    different size than the one it sized, and the difference never appears.
    """

    fills: list[Fill] = field(default_factory=list)
    unfilled: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)

    @property
    def average_price(self) -> float | None:
        if not self.filled_qty:
            return None
        return sum(f.qty * f.price for f in self.fills) / self.filled_qty

    @property
    def complete(self) -> bool:
        return self.unfilled == 0 and bool(self.fills)

    @property
    def partial(self) -> bool:
        return bool(self.fills) and self.unfilled > 0

    def render(self) -> str:
        avg = "-" if self.average_price is None else f"{self.average_price:.2f}"
        head = f"  filled {self.filled_qty} at {avg}"
        if self.unfilled:
            head += f", {self.unfilled} UNFILLED"
        return "\n".join([head] + [f"    {n}" for n in self.notes])


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class FillModel:
    """Marketable orders cross; resting orders wait their turn.

    ``calibrated`` is False until real fills have been fed through
    ``ms_options.costs.calibrate_slippage``. An uncalibrated fill model is a set
    of plausible guesses, and a P&L computed from it should say so rather than
    presenting the same number of decimal places as the statutory costs.
    """

    impact_coeff: float = DEFAULT_IMPACT_COEFF
    queue_ahead: float = DEFAULT_QUEUE_AHEAD
    allow_partial: bool = True
    calibrated: bool = False

    # ---------------------------------------------------------------- impact
    def impact(self, order: Order, quote: Quote) -> float:
        """Extra cost, in price units, for wanting more than is showing.

        Zero when the order fits inside the displayed size. Beyond that it grows
        with the square root of the excess, scaled by the spread - a wide market
        punishes size harder than a tight one, which is the whole reason
        liquidity is a feature in the sizing layer.
        """
        showing = quote.size_at(order.side)
        if showing <= 0:
            # Nothing displayed. Not free - unknown, and unknown is expensive.
            return quote.spread * self.impact_coeff
        if order.qty <= showing:
            return 0.0
        excess = (order.qty - showing) / showing
        return quote.spread * self.impact_coeff * (excess**0.5)

    # ----------------------------------------------------------------- market
    def fill_market(self, order: Order, quote: Quote, *, ts: datetime | None = None) -> FillResult:
        """Fill at the far touch, plus impact. Never at the mid.

        Filling at the mid claims half the spread as free profit on every leg.
        On a four-order option spread that is most of the edge, and it is the
        single most common way a backtest lies.
        """
        result = FillResult()
        touch = quote.far_touch(order.side)
        slip = self.impact(order, quote)
        price = touch + slip if order.side is Side.BUY else touch - slip

        if price <= 0:
            result.unfilled = order.qty
            result.notes.append("impact drove the price non-positive; treated as no fill")
            return result

        showing = quote.size_at(order.side)
        if self.allow_partial and 0 < showing < order.qty:
            # Fill what is displayed at the touch, and the rest deeper. Two
            # fills rather than one average, because a caller reconciling
            # against a broker sees two.
            result.fills.append(Fill(showing, touch, ts, "displayed size at the touch"))
            rest = order.qty - showing
            deeper = touch + slip if order.side is Side.BUY else touch - slip
            result.fills.append(Fill(rest, deeper, ts, "remainder, walked the book"))
            result.notes.append(
                f"{showing} showing against {order.qty} wanted - the remainder paid impact"
            )
        else:
            result.fills.append(Fill(order.qty, price, ts, "marketable, far touch plus impact"))

        if not self.calibrated:
            result.notes.append(
                "fill model UNCALIBRATED - these prices are plausible, not measured"
            )
        return result

    # ------------------------------------------------------------------ limit
    def fill_limit(
        self,
        order: Order,
        quote: Quote,
        *,
        traded_through: bool,
        traded_volume: int = 0,
        ts: datetime | None = None,
    ) -> FillResult:
        """A resting order fills only if the market traded through it.

        ``traded_through`` means price went past the limit, not merely touched
        it. Touching means the order *may* have been at the front of the queue;
        trading through means everything at that level cleared, which is the only
        case where being last in the queue still fills.

        The queue discount then applies: of the volume that traded, some fraction
        was ahead. ``queue_ahead = 1.0`` assumes all of it was, which is the
        assumption reality cannot beat.
        """
        assert order.limit_price is not None
        result = FillResult()
        limit = order.limit_price

        marketable = limit >= quote.ask if order.side is Side.BUY else limit <= quote.bid
        if marketable:
            # It was never resting - it crossed on arrival. Fill it as such, but
            # never worse than the limit the caller set.
            crossed = self.fill_market(order, quote, ts=ts)
            capped = []
            for f in crossed.fills:
                price = min(f.price, limit) if order.side is Side.BUY else max(f.price, limit)
                capped.append(Fill(f.qty, price, ts, "marketable on arrival, capped at the limit"))
            crossed.fills = capped
            crossed.notes.append("limit was marketable on arrival - it crossed rather than rested")
            return crossed

        if not traded_through:
            result.unfilled = order.qty
            result.notes.append(
                "price never traded through the limit. Touching is not filling: "
                "a touch only fills an order that was at the front of the queue."
            )
            return result

        available = int(traded_volume * (1.0 - min(1.0, self.queue_ahead)))
        if available <= 0:
            result.unfilled = order.qty
            result.notes.append(
                f"traded through on {traded_volume} lots, but the queue ahead "
                "absorbed all of it - being last in the queue is the assumption "
                "reality cannot beat"
            )
            return result

        qty = (
            min(order.qty, available)
            if self.allow_partial
            else (order.qty if available >= order.qty else 0)
        )
        if qty <= 0:
            result.unfilled = order.qty
            result.notes.append("not enough traded through to fill the whole order")
            return result

        result.fills.append(Fill(qty, limit, ts, "resting limit, traded through"))
        result.unfilled = order.qty - qty
        if result.unfilled:
            result.notes.append(f"{result.unfilled} left resting after the queue discount")
        return result


# --------------------------------------------------------------------------- #
# The tie rule
# --------------------------------------------------------------------------- #
class BarOutcome(StrEnum):
    NONE = "NONE"
    TARGET = "TARGET"
    STOP = "STOP"
    BOTH_ASSUMED_STOP = "BOTH_ASSUMED_STOP"


@dataclass(frozen=True)
class ExitCheck:
    outcome: BarOutcome
    price: float | None
    reason: str

    @property
    def exited(self) -> bool:
        return self.outcome is not BarOutcome.NONE


def check_exit(
    *,
    is_long: bool,
    high: float,
    low: float,
    target: float,
    stop: float,
) -> ExitCheck:
    """Which exit a bar triggered, resolving ties against the position.

    A bar records a high and a low and not the order they happened in. When both
    the target and the stop lie inside it, the honest answer is "unknown", and
    the useful one is the pessimistic one.

    Assuming the target filled is what a naive implementation does by accident,
    because targets tend to get checked first - and it turns the most volatile
    sessions, the ones where both levels are hit, into the most profitable ones.
    That is backwards, and it is invisible: the equity curve simply looks good.

    Assuming the stop filled bounds the error in one direction. A real day can
    only be better than reported, which is the only kind of backtest error worth
    having.
    """
    if is_long:
        hit_target = high >= target
        hit_stop = low <= stop
    else:
        hit_target = low <= target
        hit_stop = high >= stop

    if hit_target and hit_stop:
        return ExitCheck(
            BarOutcome.BOTH_ASSUMED_STOP,
            stop,
            "both levels inside one bar - the bar does not say which came first, "
            "so the stop is assumed. A real day can only be better than this.",
        )
    if hit_stop:
        return ExitCheck(BarOutcome.STOP, stop, "stop touched")
    if hit_target:
        return ExitCheck(BarOutcome.TARGET, target, "target touched")
    return ExitCheck(BarOutcome.NONE, None, "neither level touched")


def worst_case_sequence(checks: Sequence[ExitCheck]) -> int:
    """How many exits were resolved by the tie rule rather than by the data.

    Reported rather than buried. If a large share of a backtest's exits came from
    ties, the result is largely a statement about the tie convention and not
    about the strategy - and that is worth knowing before quoting the number.
    """
    return sum(1 for c in checks if c.outcome is BarOutcome.BOTH_ASSUMED_STOP)
