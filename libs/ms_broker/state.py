"""M83 - the order state machine, and reconciliation against the broker.

Every other module in this system can be wrong and cost a bad trade. This one
can be wrong and cost a position nobody knows exists.

The state that makes it work is UNKNOWN
---------------------------------------

The textbook lifecycle is ``intent -> submitted -> acked -> partial -> filled``
with ``rejected`` and ``cancelled`` as exits, and it is missing the only state
that matters in production. When an order is submitted and nothing comes back -
the socket dropped, the request timed out, the broker is having a morning - the
order is not failed. It is **unknown**, and the difference is the difference
between retrying safely and holding twice the position you sized for.

A system without UNKNOWN has to guess, and both guesses are wrong:

* Treat it as rejected and retry: if it did reach the broker, you are now long
  two lots and the second one was never risk-checked.
* Treat it as filled: if it did not reach the broker, you are flat while every
  downstream module believes you are hedged.

So a timeout moves the order to UNKNOWN, and **only reconciliation may move it
out**. Not a retry, not a guess, not a timer.

Idempotency is a key, not a hope
--------------------------------

Every intent gets a ``client_order_id`` before it is submitted, and a retry
carries the same one. Fills carry ids too, so a duplicate fill event adds
nothing. Without this a duplicated acknowledgement - which is ordinary, brokers
resend - increments the filled quantity, and the position record silently
doubles while every individual message was valid.

Reconciliation reports; it does not trade
------------------------------------------

:func:`reconcile` compares what this system believes against what the broker
says and returns the differences. It never fixes them by placing an order. A
broker snapshot can be stale by seconds, and an auto-correcting reconciler
turns a momentary data gap into a real position, which then needs correcting.
The loop that follows is worse than the mismatch.

The rejection nobody plans for
------------------------------

Flattening three positions and having the second rejected is the case that
breaks naive code, because the obvious implementation - a loop that raises -
leaves the third position open while the operator reads a stack trace about the
second. :func:`flatten_sequence` attempts every one, records each outcome, and
escalates loudly with the remainder named.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from libs.ms_broker.protocol import OrderAck, OrderStatus, Position
from libs.ms_core.obs import get_logger

log = get_logger(__name__)


class OrderState(StrEnum):
    INTENT = "intent"  # decided, not yet sent
    SUBMITTED = "submitted"  # sent, nothing back yet
    ACKED = "acked"  # broker has it
    PARTIAL = "partial"  # some quantity filled
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"  # never sent; a decision, not a failure
    UNKNOWN = "unknown"  # sent, fate unestablished - see the module docstring


TERMINAL = frozenset(
    {OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.ABANDONED}
)

# What may follow what. UNKNOWN is reachable from anything that has been sent,
# because that is exactly when the connection can drop; it is escapable only
# through reconciliation, which is enforced separately rather than here.
ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.INTENT: frozenset({OrderState.SUBMITTED, OrderState.ABANDONED}),
    OrderState.SUBMITTED: frozenset(
        {OrderState.ACKED, OrderState.REJECTED, OrderState.FILLED, OrderState.UNKNOWN}
    ),
    OrderState.ACKED: frozenset(
        {
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIAL: frozenset(
        {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.UNKNOWN}
    ),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.ACKED,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.ABANDONED: frozenset(),
}


class OrderStateError(RuntimeError):
    pass


class ReconciliationHalt(RuntimeError):
    """Raised when the mismatch is one that must not be traded through."""


@dataclass(frozen=True)
class Fill:
    """One execution. Carries an id because brokers resend."""

    fill_id: str
    qty: int
    price: float
    ts: datetime


@dataclass
class ManagedOrder:
    """One order, its state, and every transition that got it there.

    ``client_order_id`` is assigned before submission and survives retries. It
    is the only thing that lets a resend be recognised as the same order rather
    than a second one.
    """

    client_order_id: str
    symbol: str
    qty: int  # signed: positive long, negative short
    state: OrderState = OrderState.INTENT
    broker_order_id: str | None = None
    fills: list[Fill] = field(default_factory=list)
    _seen_fills: set[str] = field(default_factory=set, repr=False)
    history: list[tuple[datetime, OrderState, str]] = field(default_factory=list)
    unknown_since: datetime | None = None

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def avg_fill_price(self) -> float:
        total = self.filled_qty
        if total == 0:
            return 0.0
        return sum(f.qty * f.price for f in self.fills) / total

    @property
    def needs_reconciliation(self) -> bool:
        return self.state is OrderState.UNKNOWN

    # ------------------------------------------------------------- transitions
    def transition(
        self, to: OrderState, *, ts: datetime, reason: str = "", _reconciled: bool = False
    ) -> ManagedOrder:
        """Move to a new state, or refuse.

        Refusing is the point. An order that can go from FILLED back to ACKED
        because a late message arrived out of order is an order whose state is a
        function of network timing.
        """
        if to is self.state and to is not OrderState.PARTIAL:
            # Idempotent: a resent acknowledgement is not a state change. PARTIAL
            # is excluded because a second partial fill is real news.
            return self

        if to not in ALLOWED[self.state]:
            raise OrderStateError(
                f"{self.client_order_id}: {self.state.value} -> {to.value} is not a "
                f"legal transition. {_why_not(self.state, to)}"
            )

        if self.state is OrderState.UNKNOWN and not _reconciled:
            raise OrderStateError(
                f"{self.client_order_id} is UNKNOWN and may only be resolved by "
                "reconciliation against the broker. Guessing here is how a position "
                "gets doubled or a hedge silently disappears."
            )

        self.state = to
        self.unknown_since = ts if to is OrderState.UNKNOWN else None
        self.history.append((ts, to, reason))
        return self

    def apply_ack(self, ack: OrderAck, *, ts: datetime | None = None) -> ManagedOrder:
        """Fold a broker acknowledgement in. Safe to call twice with the same one.

        Brokers resend acknowledgements. If the second one moved the state or
        added quantity, an ordinary network condition would corrupt the record -
        and every individual message would have been valid.
        """
        at = ts or ack.ts
        if self.broker_order_id is None:
            self.broker_order_id = ack.order_id
        elif self.broker_order_id != ack.order_id:
            raise OrderStateError(
                f"{self.client_order_id} was acknowledged as {self.broker_order_id} "
                f"and now as {ack.order_id}. Two broker ids for one client order id "
                "means the order was placed twice."
            )

        target = {
            OrderStatus.ACCEPTED: OrderState.ACKED,
            OrderStatus.REJECTED: OrderState.REJECTED,
            OrderStatus.FILLED: OrderState.FILLED,
            OrderStatus.PARTIAL: OrderState.PARTIAL,
            OrderStatus.CANCELLED: OrderState.CANCELLED,
        }[ack.status]

        reconciled = self.state is OrderState.UNKNOWN
        return self.transition(
            target, ts=at, reason=ack.reason or f"ack {ack.status}", _reconciled=reconciled
        )

    def apply_fill(self, fill: Fill) -> ManagedOrder:
        """Record an execution. Duplicates by ``fill_id`` are dropped.

        Without the id check a resent fill event increases ``filled_qty`` and the
        position record doubles, quietly, in a way that looks like a busy market.
        """
        if fill.fill_id in self._seen_fills:
            log.info(
                "duplicate fill ignored",
                fields={"client_order_id": self.client_order_id, "fill_id": fill.fill_id},
            )
            return self

        # Compared on magnitude, so a short order is checked the same way. Doing
        # this with a bare `>` works for longs and silently permits any overfill
        # on shorts, where the quantities are negative.
        if abs(self.filled_qty + fill.qty) > abs(self.qty):
            raise OrderStateError(
                f"{self.client_order_id}: filling {fill.qty} on top of "
                f"{self.filled_qty} exceeds the ordered {self.qty}. An overfill is "
                "either a broker error or a duplicate that got past the id check, "
                "and both need a human."
            )

        self._seen_fills.add(fill.fill_id)
        self.fills.append(fill)
        reconciled = self.state is OrderState.UNKNOWN
        done = self.filled_qty == self.qty
        return self.transition(
            OrderState.FILLED if done else OrderState.PARTIAL,
            ts=fill.ts,
            reason=f"fill {fill.qty} @ {fill.price}",
            _reconciled=reconciled,
        )

    def mark_unknown(self, *, ts: datetime, reason: str) -> ManagedOrder:
        """The submit did not come back. Not a rejection."""
        return self.transition(OrderState.UNKNOWN, ts=ts, reason=reason)

    def render(self) -> str:
        bits = [f"  {self.client_order_id:<18} {self.symbol:<16} {self.state.value:<10}"]
        if self.fills:
            bits.append(f"{self.filled_qty}/{self.qty} @ {self.avg_fill_price:.2f}")
        else:
            bits.append(f"0/{self.qty}")
        if self.state is OrderState.UNKNOWN:
            bits.append("<- needs reconciliation")
        return "  ".join(bits)


def _why_not(frm: OrderState, to: OrderState) -> str:
    if frm in TERMINAL:
        return (
            f"{frm.value} is terminal. A late message arriving out of order must "
            "not reopen a finished order."
        )
    if to is OrderState.INTENT:
        return "nothing returns to intent; the decision was already made."
    return "the lifecycle does not admit it."


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
class MismatchKind(StrEnum):
    PHANTOM = "phantom"  # we believe we hold it; the broker does not
    UNTRACKED = "untracked"  # the broker holds it; we do not
    QUANTITY = "quantity"  # both hold it, differently
    STUCK_UNKNOWN = "stuck_unknown"  # an order whose fate is still unestablished


@dataclass(frozen=True)
class Mismatch:
    kind: MismatchKind
    symbol: str
    ours: int
    theirs: int
    detail: str = ""

    @property
    def must_halt(self) -> bool:
        """Whether trading may continue while this stands.

        UNTRACKED always halts: a position the system does not know about is one
        no risk limit is counting, no kill switch will flatten, and no stop is
        watching. It is the one mismatch that gets worse on its own.
        """
        return self.kind in (MismatchKind.UNTRACKED, MismatchKind.STUCK_UNKNOWN)

    def render(self) -> str:
        mark = "HALT" if self.must_halt else "diff"
        return (
            f"    {mark}  {self.kind.value:<14} {self.symbol:<16} "
            f"ours={self.ours:<6} broker={self.theirs:<6} {self.detail}"
        )


@dataclass
class Reconciliation:
    ts: datetime
    mismatches: list[Mismatch] = field(default_factory=list)
    checked: int = 0

    @property
    def clean(self) -> bool:
        return not self.mismatches

    @property
    def halting(self) -> list[Mismatch]:
        return [m for m in self.mismatches if m.must_halt]

    def raise_if_halting(self) -> None:
        if self.halting:
            raise ReconciliationHalt(
                f"{len(self.halting)} mismatch(es) that must not be traded through:\n"
                + "\n".join(m.render() for m in self.halting)
            )

    def render(self) -> str:
        head = f"  reconciliation at {self.ts.isoformat()} ({self.checked} symbols)"
        if self.clean:
            return f"{head}\n    clean"
        lines = [head] + [m.render() for m in self.mismatches]
        if self.halting:
            lines.append(
                f"  -> {len(self.halting)} halting. Reported, not corrected: a broker "
                "snapshot can be stale by seconds, and trading to fix one turns a "
                "data gap into a real position."
            )
        return "\n".join(lines)


def reconcile(
    ours: Sequence[Position],
    theirs: Sequence[Position],
    *,
    ts: datetime,
    open_orders: Iterable[ManagedOrder] = (),
) -> Reconciliation:
    """Compare beliefs. Report differences. Place nothing.

    Symbols are compared as a union, not by iterating ours - iterating only over
    what we believe we hold makes an untracked broker position structurally
    invisible, which is the single mismatch that matters most.
    """
    ours_by = {p.symbol: p for p in ours}
    theirs_by = {p.symbol: p for p in theirs}
    result = Reconciliation(ts=ts)

    for symbol in sorted(set(ours_by) | set(theirs_by)):
        a = ours_by.get(symbol)
        b = theirs_by.get(symbol)
        our_qty = a.qty if a else 0
        their_qty = b.qty if b else 0
        result.checked += 1

        if our_qty == their_qty:
            continue
        if their_qty == 0:
            result.mismatches.append(
                Mismatch(
                    MismatchKind.PHANTOM,
                    symbol,
                    our_qty,
                    0,
                    "we think we hold it and the broker does not - a fill we recorded "
                    "that never happened, or an exit we missed",
                )
            )
        elif our_qty == 0:
            result.mismatches.append(
                Mismatch(
                    MismatchKind.UNTRACKED,
                    symbol,
                    0,
                    their_qty,
                    "the broker holds it and we do not - no limit counts it, no kill "
                    "switch flattens it, no stop is watching it",
                )
            )
        else:
            result.mismatches.append(
                Mismatch(
                    MismatchKind.QUANTITY,
                    symbol,
                    our_qty,
                    their_qty,
                    f"off by {their_qty - our_qty}",
                )
            )

    for order in open_orders:
        if order.needs_reconciliation:
            result.mismatches.append(
                Mismatch(
                    MismatchKind.STUCK_UNKNOWN,
                    order.symbol,
                    order.qty,
                    theirs_by[order.symbol].qty if order.symbol in theirs_by else 0,
                    f"{order.client_order_id} has been UNKNOWN since "
                    f"{order.unknown_since} and the broker's view does not settle it",
                )
            )
    return result


def resolve_unknown(
    order: ManagedOrder,
    *,
    broker_orders: Sequence[str],
    broker_position: Position | None,
    ts: datetime,
) -> ManagedOrder:
    """Establish the fate of an UNKNOWN order from the broker's own view.

    This is the only sanctioned way out of UNKNOWN. It resolves in the safe
    direction on ambiguity: if the broker still has the order working, the order
    is ACKED, not filled - assuming a fill would let the caller believe it is
    done and stop watching.
    """
    if order.state is not OrderState.UNKNOWN:
        return order

    if order.broker_order_id and order.broker_order_id in broker_orders:
        return order.transition(
            OrderState.ACKED, ts=ts, reason="broker still has it working", _reconciled=True
        )

    if broker_position is not None and broker_position.qty != 0:
        return order.transition(
            OrderState.FILLED,
            ts=ts,
            reason=f"broker shows {broker_position.qty} in {order.symbol}",
            _reconciled=True,
        )

    # The broker has no working order and no position. It never landed - which is
    # the one case where a retry is safe, and it is safe only because this was
    # established rather than assumed.
    return order.transition(
        OrderState.REJECTED,
        ts=ts,
        reason="broker has neither the order nor a position - it never landed",
        _reconciled=True,
    )


# --------------------------------------------------------------------------- #
# Flattening, including the part that goes wrong
# --------------------------------------------------------------------------- #
@dataclass
class FlattenOutcome:
    symbol: str
    ok: bool
    detail: str


@dataclass
class FlattenReport:
    outcomes: list[FlattenOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[FlattenOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def complete(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = ["  flatten:"] + [
            f"    {'ok  ' if o.ok else 'FAIL'} {o.symbol:<16} {o.detail}" for o in self.outcomes
        ]
        if self.failed:
            lines.append(
                f"  -> {len(self.failed)} still open: "
                f"{', '.join(o.symbol for o in self.failed)}. Escalate; do not retry "
                "in a loop - whatever rejected the first attempt will reject the next."
            )
        return "\n".join(lines)


def flatten_sequence(
    symbols: Sequence[str], close: Callable[[str], OrderAck], *, ts: datetime
) -> FlattenReport:
    """Close every position, including after one refuses.

    The obvious implementation raises on the first rejection and leaves the rest
    open while an operator reads a stack trace about the one that failed. During
    a kill event that is the difference between flat and holding two thirds of
    the book.
    """
    report = FlattenReport()
    for symbol in symbols:
        try:
            ack = close(symbol)
        # Broad on purpose: a close that raises must not stop the remaining ones.
        except Exception as exc:
            report.outcomes.append(FlattenOutcome(symbol, False, f"{type(exc).__name__}: {exc}"))
            continue
        if ack.accepted:
            report.outcomes.append(FlattenOutcome(symbol, True, f"accepted as {ack.order_id}"))
        else:
            report.outcomes.append(
                FlattenOutcome(symbol, False, f"{ack.status}: {ack.reason or 'no reason given'}")
            )

    if report.failed:
        log.info(
            "flatten incomplete",
            fields={"still_open": [o.symbol for o in report.failed]},
        )
    return report


# --------------------------------------------------------------------------- #
# The audit log
# --------------------------------------------------------------------------- #
@dataclass
class AuditLog:
    """Append-only, hash-chained record of every state change.

    Chained for the same reason the experiment log is: a record that can be
    edited after the fact is a record of what someone wanted to have happened.
    Each line carries the hash of the one before it, so a deletion or an edit
    breaks the chain at a detectable point.

    ``sink`` is the off-host shipper. Shipping is a deployment concern and the
    adapter is deliberately trivial - what matters here is that the log is
    written before the shipping is attempted, so a network failure loses the
    copy and never the original.
    """

    path: Path
    sink: Callable[[str], None] | None = None
    _last_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                self._last_hash = json.loads(lines[-1])["hash"]

    def record(self, order: ManagedOrder, *, event: str, ts: datetime) -> str:
        payload = {
            "ts": ts.isoformat(),
            "event": event,
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "state": order.state.value,
            "qty": order.qty,
            "filled_qty": order.filled_qty,
            "prev": self._last_hash,
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        line = json.dumps({**payload, "hash": digest}, sort_keys=True, separators=(",", ":"))

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._last_hash = digest

        if self.sink is not None:
            try:
                self.sink(line)
            # Broad on purpose; see the comment below.
            except Exception as exc:
                # The local record already exists. A shipper that can take the
                # process down with it turns a monitoring outage into a trading
                # outage.
                log.info("audit shipping failed", fields={"error": str(exc)})
        return digest

    def verify(self) -> tuple[bool, str]:
        """Walk the chain. Returns where it breaks, if it does."""
        if not self.path.exists():
            return True, "no log yet"
        prev = ""
        for i, raw in enumerate(self.path.read_text(encoding="utf-8").strip().splitlines(), 1):
            entry = json.loads(raw)
            claimed = entry.pop("hash")
            if entry["prev"] != prev:
                return False, f"line {i}: chain broken - an entry was removed or reordered"
            body = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != claimed:
                return False, f"line {i}: contents were edited after the fact"
            prev = claimed
        return True, "chain intact"
