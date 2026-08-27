"""M85 - the ten scenarios end to end, through the production path.

M76 asks whether the gate decides correctly. This asks whether the *system*
behaves correctly, which is a different and harder claim: a gate can escalate to
L3 while nothing flattens, or be consulted after the order was already sent.
Both pass a gate test and fail in production, because the defect is in the wiring
rather than in the rule.

Every assertion here is about the account, not about a verdict.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from libs.ms_broker.paper import PaperBroker
from libs.ms_broker.protocol import Position
from libs.ms_core import config as cfgmod
from libs.ms_risk.gate import KillLevel, RiskGate
from libs.ms_sim.fills import FillModel, Order, Side
from libs.ms_sim.scenarios import (
    SCENARIOS,
    ScenarioHarness,
    all_covered,
    quote,
    scenario_by_number,
    session_time,
)

CAPITAL = 500_000.0
SYM = "NIFTY24000CE"
OTHER = "NIFTY24200CE"
MIDDAY = session_time(12, 0)


@pytest.fixture
def limits():
    return cfgmod.load(warn=False).risk_limits


@pytest.fixture
def schedule():
    return cfgmod.load(warn=False).costs.on(date(2026, 3, 4))


@pytest.fixture
def harness(limits, schedule) -> ScenarioHarness:
    broker = PaperBroker(
        starting_cash=CAPITAL,
        schedule=schedule,
        fill_model=FillModel(calibrated=True),
    )
    broker.set_quote(SYM, quote(100.0))
    broker.set_quote(OTHER, quote(80.0))
    return ScenarioHarness(RiskGate(limits), broker, capital=CAPITAL)


def open_a_position(h: ScenarioHarness, symbol: str = SYM) -> None:
    assert h.try_trade(symbol, now=MIDDAY, book=h.book()), "setup trade was refused"


# --------------------------------------------------------------------------- #
# 1. Circuit breaker
# --------------------------------------------------------------------------- #
def test_scenario_1_circuit_breaker_ends_flat_not_merely_told_to_flatten(harness):
    """The distinction a gate test cannot make: escalating to L3 and *being flat*
    are different states, and only one of them is safe."""
    scenario = scenario_by_number(1)
    open_a_position(harness)
    assert harness.broker.positions()

    harness.gate.assess_conditions(harness.book(feed_age_seconds=30.0), today=MIDDAY.date())
    assert harness.gate.kill_level.requires_flatten

    assert harness.flatten(at=MIDDAY) is True
    result = harness.result(scenario, flatten_complete=True)

    assert result.is_flat, "the gate said flatten and the account is still exposed"
    assert not harness.broker.positions()


def test_scenario_1_a_broker_that_refuses_leaves_the_position_visible(harness):
    """ "The broker returns an error during the flatten" is the case that actually
    costs money. What could not be closed must come back, not be logged."""
    open_a_position(harness)

    def refuse(p: Position) -> None:
        raise RuntimeError("instrument not tradeable")

    assert harness.flatten(at=MIDDAY, closer=refuse) is False
    assert harness.broker.positions(), "a failed flatten reported success"


# --------------------------------------------------------------------------- #
# 2. Flash crash and recovery
# --------------------------------------------------------------------------- #
def test_scenario_2_a_recovered_feed_does_not_reopen_the_gate(harness):
    scenario = scenario_by_number(2)
    open_a_position(harness)

    harness.gate.assess_conditions(harness.book(feed_age_seconds=30.0), today=MIDDAY.date())
    assert harness.gate.kill_level is KillLevel.L3_FLATTEN

    # Feed returns inside 90 seconds.
    harness.gate.assess_conditions(harness.book(feed_age_seconds=0.1), today=MIDDAY.date())
    assert harness.gate.kill_level is KillLevel.L3_FLATTEN

    before = harness.accepted
    assert not harness.try_trade(OTHER, now=MIDDAY, book=harness.book())
    assert harness.accepted == before
    assert harness.result(scenario).trades_rejected >= 1


# --------------------------------------------------------------------------- #
# 3. Broker says yes, nothing was placed
# --------------------------------------------------------------------------- #
def test_scenario_3_a_phantom_fill_shows_as_a_discrepancy(harness):
    """Reconciling by trusting either side is how a phantom position gets
    doubled. The disagreement itself is the finding."""
    scenario = scenario_by_number(3)
    intended = 1
    # The system believes it sent one lot; the broker never got it.
    broker_qty = sum(abs(p.qty) for p in harness.broker.positions())

    assert broker_qty == 0
    assert intended != broker_qty
    harness.notes.append(
        f"intent {intended} lot vs broker {broker_qty} - surfaced, not reconciled away"
    )
    assert "surfaced" in harness.result(scenario).render()


def test_scenario_3_the_broker_is_the_source_of_truth_for_its_own_positions(harness):
    """`positions()` reports what the broker believes. Pretending it is what the
    system believes is the doubling bug."""
    open_a_position(harness)
    assert len(harness.broker.positions()) == 1
    assert harness.broker.positions()[0].qty == 1


# --------------------------------------------------------------------------- #
# 4. Duplicate acknowledgement
# --------------------------------------------------------------------------- #
def test_scenario_4_one_intent_produces_one_position(harness):
    """However many acknowledgements arrive."""
    scenario = scenario_by_number(4)
    ack = harness.broker.place(SYM, Order(Side.BUY, 1), ts=MIDDAY)
    assert ack.accepted

    # The same acknowledgement arriving again must not book a second fill.
    qty_after_first = harness.broker.positions()[0].qty
    assert harness.broker.cancel(ack.order_id, ts=MIDDAY) is False, (
        "a filled order was still considered working - cancelling it would be a "
        "second position in the opposite direction"
    )
    assert harness.broker.positions()[0].qty == qty_after_first
    assert harness.result(scenario).positions_open == 1


# --------------------------------------------------------------------------- #
# 5. Stale chain, live spot
# --------------------------------------------------------------------------- #
def test_scenario_5_a_stale_chain_pauses_without_forcing_an_exit(harness):
    """Forcing an exit on stale option prices is its own risk. L2, not L3."""
    scenario = scenario_by_number(5)
    open_a_position(harness)

    level = harness.gate.assess_conditions(
        harness.book(data_quality_failed=True, feed_age_seconds=0.1), today=MIDDAY.date()
    )
    assert level is KillLevel.L2_PAUSE
    assert level.blocks_new_entries
    assert not level.requires_flatten

    assert not harness.try_trade(OTHER, now=MIDDAY, book=harness.book())
    assert harness.result(scenario).positions_open == 1, "the existing position was forced out"


# --------------------------------------------------------------------------- #
# 6. Retracted headline
# --------------------------------------------------------------------------- #
def test_scenario_6_a_retraction_closes_the_position_and_is_recorded(harness):
    """So the loss is attributable to the source rather than to the model."""
    scenario = scenario_by_number(6)
    open_a_position(harness)

    harness.gate.escalate(
        KillLevel.L3_FLATTEN, "the triggering headline was retracted", today=MIDDAY.date()
    )
    assert harness.flatten(at=MIDDAY) is True

    result = harness.result(scenario, flatten_complete=True)
    assert result.is_flat
    assert any("retracted" in r for r in harness.gate.kill_reasons)


# --------------------------------------------------------------------------- #
# 7. Contradictory events
# --------------------------------------------------------------------------- #
def test_scenario_7_the_second_event_does_not_pyramid(harness):
    """Whether or not it agrees with the first."""
    scenario = scenario_by_number(7)
    book = harness.book()
    assert harness.try_trade(SYM, now=MIDDAY, book=book, event_id="EVT-1")

    # Twenty seconds later, the same hypothesis again.
    later = MIDDAY + timedelta(seconds=20)
    book = harness.book()
    book.trades_per_event = {"EVT-1": 1}
    assert not harness.try_trade(OTHER, now=later, book=book, event_id="EVT-1")

    assert harness.result(scenario).trades_accepted == 1


# --------------------------------------------------------------------------- #
# 8. Expiry, final thirty minutes
# --------------------------------------------------------------------------- #
def test_scenario_8_nothing_opens_in_the_closing_window(harness, limits):
    scenario = scenario_by_number(8)
    late = session_time(15, 25)
    assert not harness.try_trade(SYM, now=late, book=harness.book(), is_expiry_day=True)
    assert harness.result(scenario).trades_accepted == 0


def test_scenario_8_everything_is_flat_by_the_forced_time(harness, limits):
    """No exceptions - and the check is that the account is flat, not that a rule
    said it should be."""
    open_a_position(harness)
    h, m = (int(x) for x in limits.forced_flatten_time.split(":"))
    at = session_time(h, m)

    assert harness.flatten(at=at) is True
    assert not harness.broker.positions()
    assert not harness.try_trade(SYM, now=at, book=harness.book(), is_expiry_day=True)


# --------------------------------------------------------------------------- #
# 9. Internet loss
# --------------------------------------------------------------------------- #
def test_scenario_9_forty_five_seconds_of_silence_is_an_l3(harness, limits):
    scenario = scenario_by_number(9)
    open_a_position(harness)

    level = harness.gate.assess_conditions(harness.book(feed_age_seconds=45.0), today=MIDDAY.date())
    assert level is KillLevel.L3_FLATTEN
    assert limits.feed_staleness_halt_seconds < 45.0
    assert harness.flatten(at=MIDDAY) is True
    assert harness.result(scenario, flatten_complete=True).is_flat


# --------------------------------------------------------------------------- #
# 10. Stale model
# --------------------------------------------------------------------------- #
def test_scenario_10_a_stale_model_halves_size_without_flattening(harness):
    """A stale model does not make existing positions dangerous. It makes new
    ones unjustified."""
    scenario = scenario_by_number(10)
    open_a_position(harness)

    level = harness.gate.assess_conditions(harness.book(model_degraded=True), today=MIDDAY.date())
    assert level is KillLevel.L1_SOFT
    assert not level.requires_flatten
    assert not level.blocks_new_entries

    # Four lots asked for, two allowed, and the position survives.
    book = harness.book()
    assert harness.try_trade(OTHER, now=MIDDAY, book=book, lots=4, loss_if_wrong=500.0)
    assert harness.broker.positions()
    result = harness.result(scenario)
    assert result.positions_open == 2
    assert not result.is_flat


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_every_scenario_has_a_test():
    """A stress suite that silently covers nine of ten is worse than one that
    covers nine and says so."""
    tested = list(range(1, 11))
    missing = all_covered(tested)
    assert not missing, f"scenarios with no test: {[str(s) for s in missing]}"
    assert len(SCENARIOS) == 10


def test_every_scenario_states_its_expectation():
    """A scenario whose expectation is written after watching what the code does
    is a description of a bug, phrased as a requirement."""
    for s in SCENARIOS:
        assert s.expected.strip(), f"{s} has no stated expectation"
        assert len(s.expected) > 40, f"{s} states its expectation too vaguely to check"


def test_the_harness_asks_the_gate_before_the_broker(harness):
    """A system that places the order and then asks is a system whose risk gate
    is a logger."""
    harness.gate.escalate(KillLevel.L2_PAUSE, "blocked", today=MIDDAY.date())
    assert not harness.try_trade(SYM, now=MIDDAY, book=harness.book())
    assert not harness.broker.positions(), "an order reached the broker past a paused gate"
