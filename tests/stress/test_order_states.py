"""M83 - the four ways an order record goes wrong while every message is valid.

The verify command names three of them:

    pytest tests/stress/test_order_states.py -k "duplicate_ack or lost_ack or reconcile_mismatch"

None of these are exotic. Brokers resend acknowledgements as a matter of course,
connections drop mid-submit on any morning with volume, and a broker's position
snapshot disagreeing with ours is the normal state of affairs for a second or
two after every fill. What makes them dangerous is that the wrong handling of
each one produces a system that looks fine: no exception, no alert, and a
position record that is quietly incorrect.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from libs.ms_broker.protocol import OrderAck, OrderStatus, Position
from libs.ms_broker.state import (
    ALLOWED,
    TERMINAL,
    AuditLog,
    Fill,
    ManagedOrder,
    MismatchKind,
    OrderState,
    OrderStateError,
    ReconciliationHalt,
    flatten_sequence,
    reconcile,
    resolve_unknown,
)

T0 = datetime(2026, 8, 28, 9, 20, tzinfo=UTC)


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def order(qty: int = 3, symbol: str = "NIFTY24500CE") -> ManagedOrder:
    return ManagedOrder(client_order_id="c-001", symbol=symbol, qty=qty)


def ack(
    status: OrderStatus = OrderStatus.ACCEPTED, *, oid: str = "b-77", minutes: int = 0
) -> OrderAck:
    return OrderAck(order_id=oid, status=status, ts=at(minutes))


def submitted(qty: int = 3) -> ManagedOrder:
    return order(qty).transition(OrderState.SUBMITTED, ts=at(0), reason="sent")


# --------------------------------------------------------------------------- #
# duplicate_ack
# --------------------------------------------------------------------------- #
def test_duplicate_ack_is_idempotent():
    """Brokers resend. The second copy must change nothing."""
    o = submitted()
    o.apply_ack(ack(minutes=1))
    before = (o.state, o.filled_qty, len(o.history))

    o.apply_ack(ack(minutes=1))
    o.apply_ack(ack(minutes=2))

    assert (o.state, o.filled_qty, len(o.history)) == before


def test_duplicate_ack_does_not_double_the_fill():
    """The failure that costs money. A resent FILLED acknowledgement that adds
    quantity leaves the record showing six lots against an order for three, and
    every message involved was well-formed."""
    o = submitted(qty=3)
    o.apply_ack(ack(minutes=1))
    o.apply_fill(Fill("f-1", 3, 120.0, at(2)))
    assert o.filled_qty == 3

    o.apply_fill(Fill("f-1", 3, 120.0, at(2)))  # the same fill, resent
    assert o.filled_qty == 3, "a duplicate fill doubled the position"
    assert len(o.fills) == 1


def test_duplicate_ack_with_a_different_broker_id_is_an_error():
    """Two broker ids for one client order id means it was placed twice, and
    that is the one thing the idempotency key exists to make visible."""
    o = submitted()
    o.apply_ack(ack(oid="b-77", minutes=1))

    with pytest.raises(OrderStateError, match="placed twice"):
        o.apply_ack(ack(oid="b-99", minutes=2))


def test_duplicate_partial_fills_with_distinct_ids_both_count():
    """The mirror image: two genuinely different partials must not be collapsed
    by an over-eager deduplication."""
    o = submitted(qty=5)
    o.apply_ack(ack(minutes=1))
    o.apply_fill(Fill("f-1", 2, 120.0, at(2)))
    o.apply_fill(Fill("f-2", 3, 121.0, at(3)))

    assert o.filled_qty == 5
    assert o.state is OrderState.FILLED
    assert o.avg_fill_price == pytest.approx((2 * 120.0 + 3 * 121.0) / 5)


def test_an_overfill_raises_rather_than_being_absorbed():
    o = submitted(qty=3)
    o.apply_ack(ack(minutes=1))
    o.apply_fill(Fill("f-1", 2, 120.0, at(2)))

    with pytest.raises(OrderStateError, match="exceeds the ordered"):
        o.apply_fill(Fill("f-2", 2, 120.0, at(3)))


def test_an_overfill_on_a_short_is_caught_too():
    """Signed quantities. A magnitude comparison catches both; a bare `>` catches
    longs and silently permits any overfill on shorts."""
    o = ManagedOrder(client_order_id="c-s", symbol="NIFTY24500PE", qty=-3)
    o.transition(OrderState.SUBMITTED, ts=at(0))
    o.apply_ack(ack(minutes=1))
    o.apply_fill(Fill("f-1", -2, 90.0, at(2)))

    with pytest.raises(OrderStateError, match="exceeds the ordered"):
        o.apply_fill(Fill("f-2", -2, 90.0, at(3)))


# --------------------------------------------------------------------------- #
# lost_ack
# --------------------------------------------------------------------------- #
def test_lost_ack_becomes_unknown_not_rejected():
    """The whole reason UNKNOWN exists. A submit that times out has not failed -
    it has an unestablished outcome, and calling it a rejection invites the
    retry that doubles the position."""
    o = submitted()
    o.mark_unknown(ts=at(1), reason="submit timed out after 5s")

    assert o.state is OrderState.UNKNOWN
    assert o.needs_reconciliation
    assert o.state is not OrderState.REJECTED


def test_lost_ack_cannot_be_resolved_by_guessing():
    """Not by a retry, not by a timer, not by an optimistic caller."""
    o = submitted()
    o.mark_unknown(ts=at(1), reason="socket dropped")

    with pytest.raises(OrderStateError, match="only be resolved by reconciliation"):
        o.transition(OrderState.REJECTED, ts=at(2), reason="assuming it failed")

    assert o.state is OrderState.UNKNOWN


def test_lost_ack_resolves_to_filled_when_the_broker_shows_a_position():
    """It did land. Retrying here would have doubled the position."""
    o = submitted()
    o.mark_unknown(ts=at(1), reason="socket dropped")

    resolve_unknown(
        o,
        broker_orders=[],
        broker_position=Position(symbol=o.symbol, qty=3, avg_price=120.0),
        ts=at(5),
    )
    assert o.state is OrderState.FILLED


def test_lost_ack_resolves_to_rejected_only_when_the_broker_has_neither():
    """No working order and no position. This is the one case where a retry is
    safe, and it is safe only because it was established rather than assumed."""
    o = submitted()
    o.mark_unknown(ts=at(1), reason="socket dropped")

    resolve_unknown(o, broker_orders=[], broker_position=None, ts=at(5))
    assert o.state is OrderState.REJECTED
    assert "never landed" in o.history[-1][2]


def test_lost_ack_resolves_to_acked_when_the_order_is_still_working():
    """The ambiguous case resolves in the safe direction. Calling it filled would
    let the caller stop watching an order that is still live."""
    o = submitted()
    o.apply_ack(ack(oid="b-77", minutes=1))
    o.mark_unknown(ts=at(2), reason="connection lost while waiting for the fill")

    resolve_unknown(o, broker_orders=["b-77"], broker_position=None, ts=at(5))
    assert o.state is OrderState.ACKED
    assert not o.needs_reconciliation


def test_a_resolved_unknown_can_then_proceed_normally():
    o = submitted()
    o.apply_ack(ack(oid="b-77", minutes=1))
    o.mark_unknown(ts=at(2), reason="connection lost")
    resolve_unknown(o, broker_orders=["b-77"], broker_position=None, ts=at(5))

    o.apply_fill(Fill("f-1", 3, 120.0, at(6)))
    assert o.state is OrderState.FILLED


# --------------------------------------------------------------------------- #
# reconcile_mismatch
# --------------------------------------------------------------------------- #
def test_reconcile_mismatch_finds_an_untracked_broker_position():
    """The most dangerous one, and the reason the comparison iterates the union
    rather than our own book: a position we do not know about is one no limit
    counts, no kill switch flattens and no stop is watching."""
    result = reconcile(
        ours=[Position("NIFTY24500CE", qty=2)],
        theirs=[Position("NIFTY24500CE", qty=2), Position("NIFTY24000PE", qty=-4)],
        ts=at(10),
    )

    kinds = {m.kind for m in result.mismatches}
    assert MismatchKind.UNTRACKED in kinds
    assert not result.clean
    assert result.halting


def test_reconcile_mismatch_halts_rather_than_trading():
    """It reports. It never places an order. A broker snapshot can be stale by
    seconds, and correcting one by trading turns a data gap into a real
    position - which then needs correcting."""
    result = reconcile(ours=[], theirs=[Position("BANKNIFTY52000CE", qty=5)], ts=at(10))
    with pytest.raises(ReconciliationHalt, match="must not be traded through"):
        result.raise_if_halting()
    assert "Reported, not corrected" in result.render()


def test_reconcile_mismatch_finds_a_phantom():
    """We believe we hold it and the broker does not: a fill recorded that never
    happened, or an exit we missed."""
    result = reconcile(ours=[Position("NIFTY24500CE", qty=3)], theirs=[], ts=at(10))
    assert [m.kind for m in result.mismatches] == [MismatchKind.PHANTOM]


def test_reconcile_mismatch_on_quantity_does_not_halt():
    """A partial fill we have not processed yet looks exactly like this, and
    halting on it would halt after every busy minute."""
    result = reconcile(
        ours=[Position("NIFTY24500CE", qty=3)],
        theirs=[Position("NIFTY24500CE", qty=2)],
        ts=at(10),
    )
    assert [m.kind for m in result.mismatches] == [MismatchKind.QUANTITY]
    assert not result.halting
    result.raise_if_halting()


def test_reconcile_mismatch_is_clean_when_the_books_agree():
    """The null test. A reconciler that finds something every run is one whose
    output gets ignored within a week."""
    positions = [Position("NIFTY24500CE", qty=3), Position("NIFTY24000PE", qty=-2)]
    result = reconcile(ours=positions, theirs=list(positions), ts=at(10))

    assert result.clean
    assert result.checked == 2
    assert "clean" in result.render()


def test_reconcile_mismatch_ignores_matching_flat_positions():
    """Zero on both sides is agreement, not a mismatch."""
    result = reconcile(
        ours=[Position("NIFTY24500CE", qty=0)],
        theirs=[Position("NIFTY24500CE", qty=0)],
        ts=at(10),
    )
    assert result.clean


def test_reconcile_mismatch_flags_an_order_stuck_unknown():
    """An UNKNOWN order that the broker's view does not settle. Trading onward
    means sizing the next position without knowing the current one."""
    o = submitted()
    o.mark_unknown(ts=at(1), reason="timeout")

    result = reconcile(ours=[], theirs=[], ts=at(10), open_orders=[o])
    assert [m.kind for m in result.mismatches] == [MismatchKind.STUCK_UNKNOWN]
    assert result.halting


# --------------------------------------------------------------------------- #
# success-but-not-placed
# --------------------------------------------------------------------------- #
def test_success_but_not_placed_is_caught_by_reconciliation():
    """The broker returned ACCEPTED and then has neither the order nor the
    position. Nothing in the acknowledgement path can detect this - the ack was
    a valid, successful response - so reconciliation is the only thing that
    will."""
    o = submitted()
    o.apply_ack(ack(oid="b-77", minutes=1))
    assert o.state is OrderState.ACKED

    result = reconcile(
        ours=[Position(o.symbol, qty=3)],  # what the ack led us to believe
        theirs=[],  # the broker has nothing
        ts=at(10),
    )
    assert [m.kind for m in result.mismatches] == [MismatchKind.PHANTOM]


# --------------------------------------------------------------------------- #
# rejection mid-flatten
# --------------------------------------------------------------------------- #
def test_rejection_mid_flatten_does_not_abandon_the_rest():
    """The case that breaks naive code. A loop that raises on the second symbol
    leaves the third open while an operator reads a stack trace about the
    second - during a kill event, the difference between flat and holding a
    third of the book."""
    attempted: list[str] = []

    def close(symbol: str) -> OrderAck:
        attempted.append(symbol)
        if symbol == "B":
            return OrderAck("", OrderStatus.REJECTED, at(1), reason="insufficient margin")
        return OrderAck(f"b-{symbol}", OrderStatus.ACCEPTED, at(1))

    report = flatten_sequence(["A", "B", "C"], close, ts=at(1))

    assert attempted == ["A", "B", "C"], "the flatten stopped at the rejection"
    assert [o.symbol for o in report.failed] == ["B"]
    assert not report.complete


def test_a_raising_close_does_not_stop_the_flatten_either():
    """Rejections arrive as exceptions too - a dropped connection on symbol two
    must not leave symbol three open."""
    attempted: list[str] = []

    def close(symbol: str) -> OrderAck:
        attempted.append(symbol)
        if symbol == "B":
            raise ConnectionError("socket closed")
        return OrderAck(f"b-{symbol}", OrderStatus.ACCEPTED, at(1))

    report = flatten_sequence(["A", "B", "C"], close, ts=at(1))

    assert attempted == ["A", "B", "C"]
    assert [o.symbol for o in report.failed] == ["B"]
    assert "ConnectionError" in report.failed[0].detail


def test_the_flatten_report_names_what_is_still_open():
    """An incomplete flatten that does not say which symbols survived is an
    alert an operator cannot act on."""

    def close(symbol: str) -> OrderAck:
        return OrderAck("", OrderStatus.REJECTED, at(1), reason="frozen")

    text = flatten_sequence(["A", "B"], close, ts=at(1)).render()
    assert "A, B" in text
    assert "do not retry in a loop" in text


def test_a_complete_flatten_says_so():
    def close(symbol: str) -> OrderAck:
        return OrderAck(f"b-{symbol}", OrderStatus.ACCEPTED, at(1))

    report = flatten_sequence(["A", "B"], close, ts=at(1))
    assert report.complete
    assert not report.failed


# --------------------------------------------------------------------------- #
# the state machine itself
# --------------------------------------------------------------------------- #
def test_terminal_states_are_terminal():
    """A late message arriving out of order must not reopen a finished order."""
    for state in TERMINAL:
        o = order()
        o.state = state
        with pytest.raises(OrderStateError, match="terminal"):
            o.transition(OrderState.ACKED, ts=at(1))


def test_nothing_returns_to_intent():
    o = submitted()
    with pytest.raises(OrderStateError, match="nothing returns to intent"):
        o.transition(OrderState.INTENT, ts=at(1))


def test_an_order_cannot_skip_submission():
    o = order()
    with pytest.raises(OrderStateError, match="not a legal transition"):
        o.transition(OrderState.FILLED, ts=at(1))


def test_every_state_has_a_transition_entry():
    """A state missing from the table would raise KeyError rather than refusing,
    and the error would name the wrong problem."""
    assert set(ALLOWED) == set(OrderState)


def test_abandoning_an_intent_is_legal_and_terminal():
    """A decision not to send is not a failure, and it must not look like one in
    the audit log."""
    o = order().transition(OrderState.ABANDONED, ts=at(1), reason="gate refused it")
    assert o.is_terminal


def test_remaining_tracks_the_unfilled_part():
    o = submitted(qty=5)
    o.apply_ack(ack(minutes=1))
    o.apply_fill(Fill("f-1", 2, 120.0, at(2)))
    assert o.remaining == 3
    assert o.state is OrderState.PARTIAL


# --------------------------------------------------------------------------- #
# the audit log
# --------------------------------------------------------------------------- #
def test_the_audit_chain_verifies(tmp_path: Path):
    log = AuditLog(path=tmp_path / "orders.jsonl")
    o = submitted()
    log.record(o, event="submitted", ts=at(0))
    o.apply_ack(ack(minutes=1))
    log.record(o, event="acked", ts=at(1))

    ok, msg = log.verify()
    assert ok, msg


def test_an_edited_audit_line_breaks_the_chain(tmp_path: Path):
    """A record that can be edited after the fact is a record of what someone
    wanted to have happened."""
    path = tmp_path / "orders.jsonl"
    log = AuditLog(path=path)
    o = submitted()
    log.record(o, event="submitted", ts=at(0))
    o.apply_ack(ack(minutes=1))
    log.record(o, event="acked", ts=at(1))

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["qty"] = 99
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, msg = log.verify()
    assert not ok
    assert "edited after the fact" in msg


def test_a_removed_audit_line_breaks_the_chain(tmp_path: Path):
    path = tmp_path / "orders.jsonl"
    log = AuditLog(path=path)
    o = submitted()
    for i, event in enumerate(("submitted", "acked", "filled")):
        log.record(o, event=event, ts=at(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    ok, msg = log.verify()
    assert not ok
    assert "removed or reordered" in msg


def test_a_failing_shipper_does_not_take_the_process_down(tmp_path: Path):
    """The local record already exists. A shipper that can raise turns a
    monitoring outage into a trading outage."""

    def sink(_line: str) -> None:
        raise ConnectionError("off-host collector unreachable")

    log = AuditLog(path=tmp_path / "orders.jsonl", sink=sink)
    log.record(submitted(), event="submitted", ts=at(0))

    ok, _ = log.verify()
    assert ok, "the local write was lost when shipping failed"


def test_the_chain_continues_across_a_restart(tmp_path: Path):
    """A new process must pick up the previous hash, or every restart silently
    starts a fresh chain and the gap is invisible."""
    path = tmp_path / "orders.jsonl"
    AuditLog(path=path).record(submitted(), event="submitted", ts=at(0))

    reopened = AuditLog(path=path)
    reopened.record(submitted(), event="acked", ts=at(1))

    ok, msg = reopened.verify()
    assert ok, msg
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
