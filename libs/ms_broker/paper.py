"""M80 - the paper broker.

Runs every session from Phase 5 onward, forever. Not as a rehearsal that stops
once live trading starts: it keeps running alongside, and the divergence between
what paper thought would happen and what actually filled is the cheapest
available measurement of how wrong the fill model is.

It implements the same :class:`~libs.ms_broker.protocol.Broker` interface the
live adapter will, and a test asserts the two expose exactly the same surface.
That test is the point of the module. A paper broker with a convenient extra
method is a paper broker the strategy has learned to depend on.

Costs are charged, not assumed away
-----------------------------------

Every fill pays through ``ms_options.costs``. A paper account that ignores
brokerage, STT and GST reports a gross P&L wearing a net P&L's clothes, and on a
four-order option spread the difference is most of the edge. The cost schedule
is looked up **by trade date**, so a paper run over old sessions is charged the
rates that applied then.

Position arithmetic
-------------------

The part that looks trivial and is not. Adding to a position averages the price;
reducing one realises P&L against the average and leaves it unchanged; crossing
through zero does both, in that order, and the new average is the price of the
crossing fill alone rather than a blend with the side that just closed.

Getting the last case wrong produces an average price that is a mixture of a
long and a short, which is not a price of anything. It shows up as P&L that
drifts slowly away from the truth, and it is invisible until a reconciliation.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import datetime

from libs.ms_broker.protocol import Account, Broker, OrderAck, OrderStatus, Position
from libs.ms_core.config import CostSchedule
from libs.ms_core.obs import get_logger
from libs.ms_options.costs import Leg, SlippageModel, leg_cost
from libs.ms_options.costs import Side as CostSide
from libs.ms_sim.fills import FillModel, FillResult, Order, OrderType, Quote, Side

log = get_logger(__name__)


class PaperBroker(Broker):
    """Fills through the M81 model, charges real costs, tracks real positions."""

    def __init__(
        self,
        *,
        starting_cash: float,
        schedule: CostSchedule,
        fill_model: FillModel | None = None,
        slippage: SlippageModel | None = None,
        lot_size: int = 1,
    ) -> None:
        self._account = Account(cash=starting_cash)
        self.schedule = schedule
        self.fills = fill_model or FillModel()
        self.slippage = slippage
        self.lot_size = lot_size
        self._quotes: dict[str, Quote] = {}
        self._ids = itertools.count(1)
        self._working: dict[str, tuple[str, Order]] = {}
        self.fill_log: list[tuple[str, str, int, float]] = []

    # ------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return "paper"

    @property
    def is_live(self) -> bool:
        """False, always, and asked rather than inferred.

        A mode that can be derived from configuration is a mode that can be
        derived wrongly. This one is a property of the class.
        """
        return False

    # ---------------------------------------------------------------- quotes
    def set_quote(self, symbol: str, quote: Quote) -> None:
        """Feed the current book. Orders placed without one are rejected."""
        self._quotes[symbol] = quote

    # ---------------------------------------------------------------- orders
    def place(self, symbol: str, order: Order, *, ts: datetime) -> OrderAck:
        order_id = f"paper-{next(self._ids):06d}"
        quote = self._quotes.get(symbol)
        if quote is None:
            return OrderAck(
                order_id,
                OrderStatus.REJECTED,
                ts,
                f"no quote for {symbol!r} - a paper fill invented without a book "
                "is the exact fiction this module exists to avoid",
            )

        if order.order_type is OrderType.MARKET:
            result = self.fills.fill_market(order, quote, ts=ts)
        else:
            # A resting order does not fill on arrival unless it was marketable.
            # Whether it later fills is decided by :meth:`work_order` when the
            # market trades through it.
            result = self.fills.fill_limit(order, quote, traded_through=False, ts=ts)
            if not result.fills:
                self._working[order_id] = (symbol, order)
                return OrderAck(order_id, OrderStatus.ACCEPTED, ts, "resting")

        return self._book(order_id, symbol, order, result, ts)

    def work_order(
        self, order_id: str, *, traded_through: bool, traded_volume: int, ts: datetime
    ) -> OrderAck | None:
        """Give a resting order a chance to fill against what has traded."""
        if order_id not in self._working:
            return None
        symbol, order = self._working[order_id]
        quote = self._quotes.get(symbol)
        if quote is None:
            return None
        result = self.fills.fill_limit(
            order, quote, traded_through=traded_through, traded_volume=traded_volume, ts=ts
        )
        if not result.fills:
            return None
        if result.unfilled == 0:
            del self._working[order_id]
        return self._book(order_id, symbol, order, result, ts)

    def cancel(self, order_id: str, *, ts: datetime) -> bool:
        """Whether the cancel was accepted - not whether the order was still live.

        An order can fill in the microsecond between deciding to cancel and the
        cancel arriving. Returning True for "it is definitely gone" would be a
        promise this cannot keep and the live adapter certainly cannot.
        """
        return self._working.pop(order_id, None) is not None

    # ------------------------------------------------------------- accounting
    def _book(
        self, order_id: str, symbol: str, order: Order, result: FillResult, ts: datetime
    ) -> OrderAck:
        if not result.fills:
            return OrderAck(order_id, OrderStatus.REJECTED, ts, "; ".join(result.notes))

        pos = self._account.positions.setdefault(symbol, Position(symbol=symbol))
        for fill in result.fills:
            signed = fill.qty if order.side is Side.BUY else -fill.qty
            self._apply(pos, signed, fill.price)
            self._account.cash -= signed * fill.price * self.lot_size

            cost = leg_cost(
                Leg(
                    side=CostSide.BUY if order.side is Side.BUY else CostSide.SELL,
                    price=fill.price,
                    lots=fill.qty,
                    lot_size=self.lot_size,
                    symbol=symbol,
                ),
                self.schedule,
                slippage=self.slippage,
            )
            self._account.cash -= cost.total
            self._account.costs_paid += cost.total
            pos.fills += 1
            self.fill_log.append((order_id, symbol, signed, fill.price))

        status = OrderStatus.PARTIAL if result.unfilled else OrderStatus.FILLED
        log.info(
            "paper fill",
            fields={
                "order_id": order_id,
                "symbol": symbol,
                "qty": result.filled_qty,
                "avg": round(result.average_price or 0.0, 4),
                "unfilled": result.unfilled,
            },
        )
        return OrderAck(order_id, status, ts, "; ".join(result.notes))

    def _apply(self, pos: Position, signed_qty: int, price: float) -> None:
        """Add to, reduce, or cross through a position.

        The three cases are genuinely different and the third is the one that
        gets written wrong. Crossing zero closes the old side completely -
        realising against the old average - and opens the new side at the
        crossing price alone. Blending the two produces an average that is a
        mixture of a long and a short, which is not the price of anything.
        """
        old_qty, old_avg = pos.qty, pos.avg_price
        new_qty = old_qty + signed_qty

        if old_qty == 0 or (old_qty > 0) == (signed_qty > 0):
            # Opening, or adding to the same side.
            total = abs(old_qty) + abs(signed_qty)
            pos.avg_price = (abs(old_qty) * old_avg + abs(signed_qty) * price) / total
            pos.qty = new_qty
            return

        closing = min(abs(signed_qty), abs(old_qty))
        direction = 1 if old_qty > 0 else -1
        pos.realised_pnl += direction * closing * (price - old_avg) * self.lot_size
        self._account.realised_pnl += direction * closing * (price - old_avg) * self.lot_size

        if abs(signed_qty) <= abs(old_qty):
            # Reduced, possibly to flat. The average of what remains is unchanged.
            pos.qty = new_qty
            if pos.qty == 0:
                pos.avg_price = 0.0
            return

        # Crossed through zero. The new side starts at this fill's price alone.
        pos.qty = new_qty
        pos.avg_price = price

    # --------------------------------------------------------------- readback
    def positions(self) -> Sequence[Position]:
        return [p for p in self._account.positions.values() if not p.is_flat]

    def account(self) -> Account:
        return self._account

    def open_orders(self) -> Sequence[str]:
        return sorted(self._working)

    def mark_to_market(self) -> dict[str, float]:
        """Marks from the last quote seen, mid-price.

        Mid rather than the touch, because marking a position at the price it
        could be liquidated at conflates valuation with exit cost - the exit
        cost is charged when the exit happens, and charging it twice understates
        every open position.
        """
        return {s: q.mid for s, q in self._quotes.items()}
