"""The broker interface, which paper and live both implement.

"Same interface as the live adapter — this is the point." A paper broker that
diverges from the live one is a rehearsal for a different performance: the
strategy is validated against behaviour the real broker does not have, and the
divergence is discovered with money on the table.

So the interface is defined here, once, and neither implementation is allowed to
add a method the other lacks. The test that enforces that is not decoration
either — it is the only thing standing between "we tested it" and "we tested
something else".

What the interface deliberately does not promise
------------------------------------------------

**That an order fills.** ``place`` returns an acknowledgement, not a fill. Real
brokers accept an order and tell you about the fill later, sometimes never, and
an interface that returns a fill synchronously teaches every caller a habit that
breaks in production.

**That the position you think you have is the position you have.** ``positions``
is what the *broker* believes. Reconciling that against what the system believes
is a separate problem (M83) and pretending they are the same is how a phantom
position gets doubled.

**That cancel succeeds.** It returns whether the broker accepted the request. An
order can fill in the microsecond between deciding to cancel and the cancel
arriving, and code that assumes otherwise has an off-by-one-position bug waiting
for a volatile day.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from libs.ms_sim.fills import Order


class OrderStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class OrderAck:
    """What comes back from placing an order. Not a fill."""

    order_id: str
    status: OrderStatus
    ts: datetime
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status in (OrderStatus.ACCEPTED, OrderStatus.FILLED, OrderStatus.PARTIAL)


@dataclass
class Position:
    """A holding, as the broker sees it.

    ``qty`` is signed: positive is long, negative is short. Not a side field,
    because a position that flips through zero is one position changing sign and
    two fields make that two positions with a moment of inconsistency between
    them.
    """

    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    realised_pnl: float = 0.0
    fills: int = 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    def unrealised(self, mark: float) -> float:
        """Marked against a price the caller supplies.

        Deliberately not stored. An unrealised P&L cached on the position is a
        number that is wrong the moment the market moves, and it will be read
        long after it went stale.
        """
        return self.qty * (mark - self.avg_price)


@dataclass
class Account:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0
    costs_paid: float = 0.0

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus the marked value of what is open.

        A symbol with no mark is not treated as worthless - it raises. Marking a
        position at zero because the quote is missing turns a data gap into a
        loss, and the loss looks real.
        """
        total = self.cash
        for sym, pos in self.positions.items():
            if pos.is_flat:
                continue
            if sym not in marks:
                raise KeyError(
                    f"no mark for open position {sym!r}. Marking it at zero would "
                    "turn a missing quote into a realised-looking loss."
                )
            total += pos.unrealised(marks[sym])
        return total


class Broker(ABC):
    """Everything the execution layer is allowed to depend on.

    Kept small on purpose. Every method added here is a method the live adapter
    must implement faithfully, and each one is a place the two can drift.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Whether real money moves. Never inferred, always asked."""

    @abstractmethod
    def place(self, symbol: str, order: Order, *, ts: datetime) -> OrderAck:
        """Submit an order. Returns an acknowledgement, not a fill."""

    @abstractmethod
    def cancel(self, order_id: str, *, ts: datetime) -> bool:
        """Request a cancel. Returns whether the broker accepted the request -
        not whether the order was still live when it arrived."""

    @abstractmethod
    def positions(self) -> Sequence[Position]:
        """What the broker believes is open. Not what the system believes."""

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def open_orders(self) -> Sequence[str]:
        """Order ids the broker still considers working."""
