"""M85 - the stress scenarios, driven through the production path.

M76 already tests these ten against the risk gate. This is a different question
and the more important one.

The gate deciding correctly and the *system* behaving correctly are not the same
claim. A gate can escalate to L3 while nothing actually flattens. It can refuse a
trade that was already sent. It can be consulted after the position was opened
rather than before. Every one of those passes a gate test and fails in
production, because the defect is in the wiring rather than in the rule.

So these run the whole chain — event, discovery delay, decision, gate, broker,
record — and assert what the *account* looks like afterwards. The gate's verdict
is an input to that, not the answer.

Expected behaviour is written before the implementation
-------------------------------------------------------

Each scenario carries its ``expected`` string as data. Not documentation: a
scenario whose expectation is written after watching what the code does is a
description of a bug, phrased as a requirement. Reading them together is also
the fastest way to find the one that says something nobody agrees with.

The harness is deliberately small
---------------------------------

It wires real components together and does nothing clever. Anything clever here
would be a second implementation of the system, and a stress suite that tests a
parallel implementation tests the parallel implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from libs.ms_broker.paper import PaperBroker
from libs.ms_broker.protocol import Position
from libs.ms_core.obs import get_logger
from libs.ms_risk.gate import (
    BookState,
    KillLevel,
    ProposedTrade,
    RiskGate,
    Verdict,
    flatten_all,
)
from libs.ms_risk.gate import Position as GatePosition
from libs.ms_sim.fills import Order, Quote, Side

log = get_logger(__name__)


@dataclass(frozen=True)
class Scenario:
    """One thing that goes wrong, and what the system is supposed to do about it."""

    number: int
    name: str
    expected: str

    def __str__(self) -> str:
        return f"{self.number}. {self.name}"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        1,
        "circuit breaker halts trading mid-position",
        "feed goes silent with a position open -> L3 -> every position is closed, "
        "and the account ends flat rather than merely 'told to flatten'",
    ),
    Scenario(
        2,
        "flash crash and recovery within 90 seconds",
        "the kill level does not fall when the feed returns, and no new entry is "
        "accepted afterwards - a recovered feed does not undo what happened while "
        "it was gone",
    ),
    Scenario(
        3,
        "broker reports success but the order was not placed",
        "the broker's position and the system's intent disagree, and the "
        "discrepancy is surfaced rather than reconciled away by trusting either "
        "side",
    ),
    Scenario(
        4,
        "duplicate order acknowledgement",
        "the position is not doubled - one intent produces one position however "
        "many acknowledgements arrive",
    ),
    Scenario(
        5,
        "option chain goes stale while spot stays live",
        "L2 pause: no new entries, existing positions still manageable. Forcing an "
        "exit on stale option prices is its own risk",
    ),
    Scenario(
        6,
        "false catastrophic headline, retracted after four minutes",
        "L3 on the retraction, the position is closed, and the record says the "
        "trigger was withdrawn - so the loss is attributable to the source rather "
        "than to the model",
    ),
    Scenario(
        7,
        "two contradictory high-severity events 20 seconds apart",
        "at most one trade on the hypothesis. The second event does not pyramid, "
        "whether or not it agrees with the first",
    ),
    Scenario(
        8,
        "expiry day, final thirty minutes, IV collapsing",
        "no new entries in the closing window, and everything is flat by the "
        "forced-flatten time with no exceptions",
    ),
    Scenario(
        9,
        "internet loss for 45 seconds with an open position",
        "L3 - 45 seconds is far past the staleness threshold and the position was "
        "open the whole time",
    ),
    Scenario(
        10,
        "model server returns predictions from a stale model version",
        "L1: signals become advisory and size is halved. Not a flatten - a stale "
        "model does not make existing positions dangerous, it makes new ones "
        "unjustified",
    ),
)


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    """What actually happened, in the terms the expectation is written in."""

    scenario: Scenario
    kill_level: KillLevel
    positions_open: int
    trades_accepted: int
    trades_rejected: int
    flatten_complete: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.positions_open == 0

    def render(self) -> str:
        lines = [
            f"  {self.scenario}",
            f"    expected: {self.scenario.expected}",
            f"    kill level {self.kill_level.name}, {self.positions_open} open, "
            f"{self.trades_accepted} accepted, {self.trades_rejected} rejected",
        ]
        if self.flatten_complete is not None:
            lines.append(f"    flatten complete: {self.flatten_complete}")
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


class ScenarioHarness:
    """Wires a gate and a broker together and runs a scenario against both.

    Small on purpose. Anything clever here would be a second implementation of
    the system, and a stress suite that tests a parallel implementation tests
    the parallel implementation.
    """

    def __init__(self, gate: RiskGate, broker: PaperBroker, *, capital: float) -> None:
        self.gate = gate
        self.broker = broker
        self.capital = capital
        self.accepted = 0
        self.rejected = 0
        self.notes: list[str] = []

    def book(self, **over: object) -> BookState:
        state = BookState(capital=self.capital)
        for k, v in over.items():
            setattr(state, k, v)
        # The broker's position and the gate's are different types on purpose:
        # the broker tracks average price and realised P&L, the gate tracks delta
        # and vega. Converting drops the greeks, which the broker does not know
        # and cannot invent - a scenario that needs them supplies them itself
        # rather than having zeros quietly stand in for real exposure.
        state.open_positions = [
            GatePosition(symbol=p.symbol, lots=abs(p.qty), is_long=p.is_long)
            for p in self.broker.positions()
        ]
        return state

    def try_trade(
        self,
        symbol: str,
        *,
        now: datetime,
        book: BookState,
        lots: int = 1,
        loss_if_wrong: float = 5000.0,
        event_id: str | None = None,
        is_expiry_day: bool = False,
    ) -> bool:
        """The full path: gate first, broker only if the gate allows.

        Gate *first*. A system that places the order and then asks is a system
        whose risk gate is a logger.
        """
        decision = self.gate.evaluate(
            ProposedTrade(
                symbol=symbol,
                lots=lots,
                is_long=True,
                loss_if_wrong=loss_if_wrong,
                event_id=event_id,
            ),
            book,
            now=now,
            is_expiry_day=is_expiry_day,
        )
        if decision.verdict is Verdict.REJECT:
            self.rejected += 1
            self.notes.extend(f"rejected: {b}" for b in decision.breaches[:2])
            return False

        ack = self.broker.place(symbol, Order(Side.BUY, decision.lots), ts=now)
        if not ack.accepted:
            self.rejected += 1
            self.notes.append(f"broker refused: {ack.reason}")
            return False
        self.accepted += 1
        return True

    def flatten(self, *, at: datetime, closer: Callable[[Position], None] | None = None) -> bool:
        """Actually close everything, and report whether it worked.

        The distinction M76 could not test: escalating to L3 and *being flat*
        are different states, and only one of them is safe. A gate that raises
        the level while positions stay open has done its job and left the
        account exposed.

        ``closer`` is injectable so a scenario can make the broker fail on
        purpose - "the broker returns an error during the flatten" is the case
        that actually costs money.
        """
        positions = list(self.broker.positions())

        def close_through_broker(p: Position) -> None:
            side = Side.SELL if p.is_long else Side.BUY
            ack = self.broker.place(p.symbol, Order(side, abs(p.qty)), ts=at)
            if not ack.accepted:
                raise RuntimeError(f"broker refused the close: {ack.reason}")

        return flatten_all(positions, closer or close_through_broker).complete

    def result(self, scenario: Scenario, **extra: object) -> ScenarioResult:
        return ScenarioResult(
            scenario=scenario,
            kill_level=self.gate.kill_level,
            positions_open=len(self.broker.positions()),
            trades_accepted=self.accepted,
            trades_rejected=self.rejected,
            notes=list(self.notes),
            **extra,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Helpers the scenarios share
# --------------------------------------------------------------------------- #
def quote(price: float, *, spread: float = 2.0, size: int = 1000) -> Quote:
    return Quote(bid=price - spread / 2, ask=price + spread / 2, bid_size=size, ask_size=size)


def session_time(hour: int, minute: int, *, day: date | None = None) -> datetime:
    d = day or date(2026, 3, 4)
    return datetime(d.year, d.month, d.day, hour, minute)


def scenario_by_number(n: int) -> Scenario:
    for s in SCENARIOS:
        if s.number == n:
            return s
    raise KeyError(f"no scenario {n}")


def all_covered(tested: Sequence[int]) -> list[Scenario]:
    """Which scenarios have no test.

    Exists so the suite can fail when a scenario is added to the list and
    forgotten - a stress suite that silently covers nine of ten is worse than
    one that covers nine and says so.
    """
    seen = set(tested)
    return [s for s in SCENARIOS if s.number not in seen]
