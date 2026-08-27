"""M76 - the ten scenarios from the blueprint, plus the gate's own invariants.

"A system that has not been tested against these has not been tested." Each of
the ten below names the expected behaviour first, because a scenario test whose
expectation is written after watching what the code does is a description, not a
test.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_risk.gate import (
    BookState,
    FlattenResult,
    KillLevel,
    Position,
    ProposedTrade,
    RiskGate,
    Verdict,
    flatten_all,
    limit_change_requires_adr,
)

MIDDAY = datetime(2026, 3, 4, 12, 0)
CAPITAL = 500_000.0


@pytest.fixture
def limits():
    return cfgmod.load(warn=False).risk_limits


@pytest.fixture
def gate(limits):
    return RiskGate(limits)


def book(**kwargs) -> BookState:
    return BookState(capital=CAPITAL, **kwargs)


def trade(**kwargs) -> ProposedTrade:
    defaults = {
        "symbol": "NIFTY24000CE",
        "lots": 1,
        "is_long": True,
        "loss_if_wrong": 5000.0,
    }
    return ProposedTrade(**{**defaults, **kwargs})


def pos(symbol: str = "NIFTY24000CE", **kwargs) -> Position:
    defaults = {"lots": 1, "is_long": True}
    return Position(symbol=symbol, **{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# The ten scenarios
# --------------------------------------------------------------------------- #
def test_1_circuit_breaker_halts_trading_mid_position(gate):
    """Expected: the index halts, the feed goes silent, and an open position is
    stranded. That is an L3 - flatten - not a pause, because nobody is watching
    the price the position is marked against."""
    state = book(open_positions=[pos()], feed_age_seconds=30.0)
    level = gate.assess_conditions(state, today=MIDDAY.date())

    assert level is KillLevel.L3_FLATTEN
    assert level.requires_flatten
    assert not gate.evaluate(trade(), state, now=MIDDAY).allowed


def test_2_flash_crash_and_recovery_does_not_un_escalate(gate):
    """Expected: the feed comes back within 90 seconds and the kill level does
    NOT fall on its own. A recovered feed does not undo the fact that it went
    away with a position open."""
    stranded = book(open_positions=[pos()], feed_age_seconds=30.0)
    gate.assess_conditions(stranded, today=MIDDAY.date())
    assert gate.kill_level is KillLevel.L3_FLATTEN

    recovered = book(open_positions=[pos()], feed_age_seconds=0.1)
    assert gate.assess_conditions(recovered, today=MIDDAY.date()) is KillLevel.L3_FLATTEN


def test_3_broker_reports_success_but_the_order_was_not_placed(gate):
    """Expected: the flatten reports which symbols it could not confirm rather
    than reporting overall success. A flatten that claims success while a
    position is open is worse than one that fails loudly - the operator stops
    looking."""
    positions = [pos("A"), pos("B")]
    phantom = {"A"}

    def close(p: Position) -> None:
        if p.symbol in phantom:
            raise RuntimeError("broker acknowledged but no fill was reported")

    result = flatten_all(positions, close, max_attempts=2)
    assert not result.complete
    assert "A" in result.failed
    assert "B" in result.closed
    assert "STILL OPEN A" in result.render()


def test_4_duplicate_acknowledgement_does_not_double_close(gate):
    """Expected: a position that closed on the first attempt is not retried, so
    a duplicate acknowledgement cannot become a second order in the opposite
    direction."""
    calls: list[str] = []
    positions = [pos("A"), pos("B")]
    failing = {"B"}

    def close(p: Position) -> None:
        calls.append(p.symbol)
        if p.symbol in failing and calls.count("B") == 1:
            raise RuntimeError("timeout")

    result = flatten_all(positions, close, max_attempts=3)
    assert calls.count("A") == 1, f"A was closed {calls.count('A')} times"
    assert result.complete


def test_5_stale_option_chain_with_a_live_spot_feed(gate):
    """Expected: this is a data-quality failure, which is an L2 pause and not an
    L3. The spot feed being live means positions can still be managed, and
    forcing an exit is itself a risk."""
    state = book(open_positions=[pos()], data_quality_failed=True, feed_age_seconds=0.1)
    level = gate.assess_conditions(state, today=MIDDAY.date())

    assert level is KillLevel.L2_PAUSE
    assert level.blocks_new_entries
    assert not level.requires_flatten


def test_6_a_false_headline_retracted_four_minutes_later(gate):
    """Expected: retraction of the triggering news is an L3. The position was
    opened on information that turned out not to exist, so there is no thesis
    left to manage - only exposure."""
    gate.escalate(
        KillLevel.L3_FLATTEN,
        "the triggering headline was retracted",
        today=MIDDAY.date(),
    )
    assert gate.kill_level.requires_flatten
    assert not gate.evaluate(trade(), book(), now=MIDDAY).allowed


def test_7_two_contradictory_high_severity_events_20_seconds_apart(gate):
    """Expected: contradictory events are an L2 pause. And separately, the
    per-event limit must already prevent a second trade on the same hypothesis
    even without the pause."""
    gate.escalate(KillLevel.L2_PAUSE, "contradictory events", today=MIDDAY.date())
    assert gate.kill_level.blocks_new_entries

    fresh = RiskGate(gate.limits)
    state = book(trades_per_event={"EVT-1": 1})
    decision = fresh.evaluate(trade(event_id="EVT-1"), state, now=MIDDAY)
    assert decision.verdict is Verdict.REJECT
    assert any("pyramiding" in b for b in decision.breaches)


def test_8_expiry_day_final_thirty_minutes(gate, limits):
    """Expected: the last ten minutes block entries outright, and everything
    after the forced flatten time is refused regardless."""
    late = datetime(2026, 3, 4, 15, 25)
    decision = gate.evaluate(trade(), book(), now=late, is_expiry_day=True)

    assert decision.verdict is Verdict.REJECT
    assert any("last" in b or "forced flatten" in b for b in decision.breaches)


def test_9_internet_loss_for_45_seconds_with_an_open_position(gate, limits):
    """Expected: L3. Forty-five seconds is far past the staleness threshold, and
    the position is open the whole time."""
    state = book(open_positions=[pos()], feed_age_seconds=45.0)
    assert gate.assess_conditions(state, today=MIDDAY.date()) is KillLevel.L3_FLATTEN
    assert limits.feed_staleness_halt_seconds < 45.0


def test_10_model_server_returns_a_stale_model_version(gate):
    """Expected: a degraded model is an L1 - signals become advisory and size is
    halved. It is not a flatten: the model being stale does not make existing
    positions dangerous, it makes new ones unjustified."""
    state = book(model_degraded=True)
    level = gate.assess_conditions(state, today=MIDDAY.date())

    assert level is KillLevel.L1_SOFT
    assert not level.blocks_new_entries

    decision = gate.evaluate(trade(lots=4, loss_if_wrong=500.0), state, now=MIDDAY)
    assert decision.verdict is Verdict.RESIZE
    assert decision.lots == 2
    assert any("halved" in w for w in decision.warnings)


def test_a_soft_kill_that_cannot_shrink_says_so_rather_than_rejecting(gate):
    """L1 is "reduce size by half; notify" - the system keeps working. At one
    lot there is no half, and rejecting there would silently turn a soft kill
    into a hard one for every account already at the minimum tradeable size."""
    state = book(model_degraded=True)
    decision = gate.evaluate(trade(lots=1), state, now=MIDDAY)

    assert decision.verdict is Verdict.ALLOW
    assert decision.lots == 1
    assert any("could NOT reduce size" in w for w in decision.warnings)


# --------------------------------------------------------------------------- #
# The hierarchy
# --------------------------------------------------------------------------- #
def test_the_kill_level_never_falls_on_its_own(gate):
    gate.escalate(KillLevel.L3_FLATTEN, "feed loss", today=date(2026, 3, 4))
    gate.escalate(KillLevel.L1_SOFT, "drift alert", today=date(2026, 3, 4))
    assert gate.kill_level is KillLevel.L3_FLATTEN


def test_two_l3_events_in_a_week_become_a_lockout(gate):
    """The second one is evidence the first was not understood."""
    gate.escalate(KillLevel.L3_FLATTEN, "feed loss", today=date(2026, 3, 2))
    assert gate.kill_level is KillLevel.L3_FLATTEN
    gate.escalate(KillLevel.L3_FLATTEN, "clock drift", today=date(2026, 3, 5))
    assert gate.kill_level is KillLevel.L4_LOCKOUT


def test_two_l3_events_far_apart_do_not_lock_out(gate):
    gate.escalate(KillLevel.L3_FLATTEN, "feed loss", today=date(2026, 1, 2))
    gate.clear("Root cause: upstream feed maintenance window, now scheduled outside session hours.")
    gate.escalate(KillLevel.L3_FLATTEN, "clock drift", today=date(2026, 3, 5))
    assert gate.kill_level is KillLevel.L3_FLATTEN


def test_an_unexplained_loss_locks_out_immediately(gate):
    """A loss nobody can account for is a loss that can happen again at any size."""
    gate.assess_conditions(book(unexplained_loss=True), today=MIDDAY.date())
    assert gate.kill_level is KillLevel.L4_LOCKOUT
    assert gate.kill_level.needs_a_human


def test_clearing_a_lockout_requires_a_written_note(gate):
    """The failure being prevented is a system that recovers by itself and takes
    the same loss again on the same undiagnosed cause."""
    gate.assess_conditions(book(unexplained_loss=True), today=MIDDAY.date())
    with pytest.raises(ValueError, match="root-cause note"):
        gate.clear("fixed")
    with pytest.raises(ValueError, match="root-cause note"):
        gate.clear("   ")

    gate.clear(
        "Reconciliation gap: a fill arrived after the daily close and was booked to the wrong day."
    )
    assert gate.kill_level is KillLevel.NONE


def test_a_lesser_kill_can_be_cleared_without_ceremony(gate):
    gate.escalate(KillLevel.L1_SOFT, "drift", today=MIDDAY.date())
    gate.clear("drift resolved")
    assert gate.kill_level is KillLevel.NONE


# --------------------------------------------------------------------------- #
# The limits
# --------------------------------------------------------------------------- #
def test_the_gate_denies_by_default(gate):
    """A bug in a check must leave a trade blocked, not permitted. Nothing here
    reaches ALLOW without passing every check."""
    decision = gate.evaluate(trade(lots=0), book(), now=MIDDAY)
    assert decision.verdict is Verdict.REJECT
    assert not decision.allowed


def test_an_oversized_trade_is_resized_not_rejected(gate, limits):
    """Resize down where possible: the limit is on exposure, and a smaller
    version of the same trade does not breach it."""
    state = book()
    big = trade(lots=10, loss_if_wrong=1000.0)
    decision = gate.evaluate(big, state, now=MIDDAY)

    assert decision.verdict is Verdict.RESIZE
    risked = decision.lots * 1000.0 / CAPITAL * 100
    assert risked <= limits.per_trade_capital_at_risk_pct + 1e-9


def test_a_single_lot_too_large_to_resize_is_rejected(gate):
    decision = gate.evaluate(trade(lots=1, loss_if_wrong=50_000.0), book(), now=MIDDAY)
    assert decision.verdict is Verdict.REJECT
    assert any("cannot be resized down" in b for b in decision.breaches)


def test_the_concurrent_position_limit_rejects(gate, limits):
    state = book(open_positions=[pos(f"S{i}") for i in range(limits.max_concurrent_positions)])
    assert gate.evaluate(trade(), state, now=MIDDAY).verdict is Verdict.REJECT


def test_the_daily_trade_limit_rejects(gate, limits):
    state = book(trades_today=limits.max_trades_per_day)
    decision = gate.evaluate(trade(), state, now=MIDDAY)
    assert decision.verdict is Verdict.REJECT
    assert any("trades today" in b for b in decision.breaches)


def test_delta_is_checked_on_the_resulting_book_not_the_increment(gate, limits):
    """The limit is on the exposure that would exist afterwards. Checking only
    the increment lets a book drift past the band one small trade at a time."""
    nearly = limits.max_portfolio_delta - 0.05
    state = book(open_positions=[pos(delta=nearly)])
    decision = gate.evaluate(trade(delta=0.30), state, now=MIDDAY)

    assert decision.verdict is Verdict.REJECT
    assert any("portfolio delta" in b for b in decision.breaches)


def test_vega_is_checked_the_same_way(gate, limits):
    state = book(open_positions=[pos(vega=limits.max_portfolio_vega - 0.5)])
    decision = gate.evaluate(trade(vega=2.0), state, now=MIDDAY)
    assert decision.verdict is Verdict.REJECT
    assert any("volatility bet" in b for b in decision.breaches)


def test_the_daily_loss_limit_has_no_override(gate, limits):
    """ "Stop for the day. Flatten everything. No overrides." - and there is no
    parameter on evaluate() that could grant one."""
    lost = -(limits.daily_loss_limit_pct / 100) * CAPITAL
    state = book(realised_pnl_today=lost)
    decision = gate.evaluate(trade(), state, now=MIDDAY)

    assert gate.kill_level.blocks_new_entries
    assert decision.verdict is Verdict.REJECT


def test_a_profitable_day_is_not_a_loss(gate):
    state = book(realised_pnl_today=50_000.0)
    assert state.daily_loss_pct == 0.0
    assert gate.assess_conditions(state, today=MIDDAY.date()) is KillLevel.NONE


def test_no_entries_in_the_opening_minutes(gate, limits):
    early = datetime(2026, 3, 4, 9, 15 + limits.no_entry_first_minutes - 1)
    decision = gate.evaluate(trade(), book(), now=early)
    assert decision.verdict is Verdict.REJECT
    assert any("into the session" in b for b in decision.breaches)


def test_nothing_is_allowed_after_the_forced_flatten_time(gate, limits):
    h, m = (int(x) for x in limits.forced_flatten_time.split(":"))
    after = datetime(2026, 3, 4, h, m + 1)
    decision = gate.evaluate(trade(), book(), now=after)
    assert decision.verdict is Verdict.REJECT
    assert any("no exceptions" in b for b in decision.breaches)


def test_a_clean_trade_at_a_sensible_time_is_allowed(gate):
    """The gate has to be able to say yes, or the tests above prove nothing."""
    decision = gate.evaluate(trade(), book(), now=MIDDAY)
    assert decision.verdict is Verdict.ALLOW
    assert decision.allowed
    assert not decision.breaches


def test_every_rejection_carries_a_reason(gate, limits):
    """A rejection with no reason is not auditable, and this is the one place
    where every decision has to be explicable months later."""
    cases = [
        (trade(), book(trades_today=limits.max_trades_per_day), MIDDAY),
        (trade(), book(), datetime(2026, 3, 4, 15, 25)),
        (trade(lots=1, loss_if_wrong=99_999.0), book(), MIDDAY),
    ]
    for proposed, state, when in cases:
        decision = RiskGate(limits).evaluate(proposed, state, now=when)
        assert decision.verdict is Verdict.REJECT
        assert decision.breaches, "rejected with no reason recorded"


# --------------------------------------------------------------------------- #
# Flatten
# --------------------------------------------------------------------------- #
def test_flatten_closes_everything_when_the_broker_cooperates():
    closed: list[str] = []
    positions = [pos("A"), pos("B"), pos("C")]
    result = flatten_all(positions, lambda p: closed.append(p.symbol))

    assert result.complete
    assert sorted(result.closed) == ["A", "B", "C"]
    assert result.attempts == 1


def test_one_failure_does_not_strand_the_others():
    """A single try/except around the whole loop would leave every position
    after the first error open, and the log would show one failure rather than
    all of them."""
    positions = [pos("A"), pos("B"), pos("C")]

    def close(p: Position) -> None:
        if p.symbol == "A":
            raise RuntimeError("rejected")

    result = flatten_all(positions, close, max_attempts=1)
    assert set(result.closed) == {"B", "C"}
    assert set(result.failed) == {"A"}


def test_a_transient_failure_is_retried():
    attempts: dict[str, int] = {}

    def close(p: Position) -> None:
        attempts[p.symbol] = attempts.get(p.symbol, 0) + 1
        if attempts[p.symbol] < 3:
            raise TimeoutError("broker busy")

    result = flatten_all([pos("A")], close, max_attempts=3)
    assert result.complete
    assert attempts["A"] == 3


def test_an_incomplete_flatten_calls_the_escalation_hook():
    seen: list[FlattenResult] = []

    def close(p: Position) -> None:
        raise RuntimeError("down")

    result = flatten_all([pos("A")], close, max_attempts=2, on_failure=seen.append)
    assert not result.complete
    assert seen and seen[0] is result
    assert "Escalate to a human now" in result.render()


def test_flatten_needs_nothing_but_a_broker_and_a_list():
    """The reason to flatten is usually that something broke. A flatten path
    that imports the model layer, the feature store or the feed is one that does
    not work on the day it is needed - while passing every test written against
    a healthy system."""
    import ast
    from pathlib import Path

    source = Path("libs/ms_risk/gate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"libs.ms_models", "libs.ms_features", "libs.ms_data", "libs.ms_broker"}
    assert not any(
        mod.startswith(f) for mod in imported for f in forbidden
    ), f"the risk gate imports something that can be down when it is needed: {imported}"


def test_flatten_of_an_empty_book_is_not_a_failure():
    result = flatten_all([], lambda p: None)
    assert result.complete
    assert result.closed == []


# --------------------------------------------------------------------------- #
# Limit-change discipline
# --------------------------------------------------------------------------- #
def test_changing_a_limit_without_an_adr_is_flagged(limits):
    """A risk limit that can be edited in a commit gets edited at the worst
    moment - after a losing week, by the person who lost it."""
    loosened = limits.model_copy(update={"daily_loss_limit_pct": 8.0})
    missing = limit_change_requires_adr(limits, loosened, [])
    assert "daily_loss_limit_pct" in missing


def test_a_documented_change_passes(limits):
    loosened = limits.model_copy(update={"daily_loss_limit_pct": 8.0})
    missing = limit_change_requires_adr(
        limits, loosened, ["0003-raise-daily_loss_limit_pct-after-review.md"]
    )
    assert missing == []


def test_an_unchanged_config_needs_no_adr(limits):
    assert limit_change_requires_adr(limits, limits, []) == []


def test_several_changes_each_need_their_own_record(limits):
    loosened = limits.model_copy(update={"daily_loss_limit_pct": 8.0, "max_trades_per_day": 20})
    missing = limit_change_requires_adr(limits, loosened, ["0003-raise-daily_loss_limit_pct.md"])
    assert missing == ["max_trades_per_day"]


def test_the_session_window_is_configurable_not_hardcoded(limits):
    """A gate that hardcodes 09:15 is a gate that is wrong on every muhurat
    session and every schedule change."""
    gate = RiskGate(limits, session_open=time(10, 0), session_close=time(14, 0))
    early = datetime(2026, 3, 4, 10, 1)
    assert gate.evaluate(trade(), book(), now=early).verdict is Verdict.REJECT
