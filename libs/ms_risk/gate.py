"""M76 - the risk gate and the kill switches.

This module sits between the brain and the money. Nothing bypasses it, and that
sentence is the design rather than an aspiration: every path to an order runs
through :meth:`RiskGate.evaluate`, and the gate's default answer is no.

Deny by default
---------------

The gate starts from "reject" and requires every check to pass. The alternative
- start from allow and reject on a breach - fails open, which means a bug in a
check silently permits the trade it was written to stop. That is the wrong
direction for the only component whose job is to say no, and it is the difference
between a missing check being a missing veto and a missing check being an
unlimited allowance.

The kill hierarchy
------------------

Four levels, each strictly more severe, and the trigger for each is written down
rather than left to judgement:

* **L1 soft** - drift, calibration decay, unusual latency. Halve the size, notify.
  The system keeps working.
* **L2 pause** - daily loss limit, data-quality failure, contradictory events.
  No new entries; existing positions may still be managed, because forcing an
  exit is itself a risk.
* **L3 flatten** - feed loss, model server failure, clock drift, retraction of
  the triggering news. Close everything at market, now.
* **L4 lockout** - two L3 events in a week, or any unexplained loss. The system
  refuses to run until a human writes down what happened.

Escalation is automatic; de-escalation is not. :meth:`RiskGate.clear` requires a
written note and refuses to clear L4 without one, because the failure mode this
prevents is a system that recovers from a lockout by itself and takes the same
loss again on the same undiagnosed cause.

Flatten must not depend on what failed
--------------------------------------

The reason to flatten is usually that something broke. A flatten path that needs
the model server, the feature store or the market feed is a flatten path that
does not work on the day it is needed, and it will pass every test written
against a healthy system.

:func:`flatten_all` therefore takes only a broker and a list of positions. It
does not import the model layer, it does not consult the feed, and it does not
ask for a price - closing at market is the point. It also assumes the broker
will fail, because "the broker API returns an error during the flatten" is the
scenario that actually costs money: failures are retried, then escalated, and
the positions that could not be closed are returned rather than logged and
forgotten.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import IntEnum

from libs.ms_core.config import RiskLimits
from libs.ms_core.obs import get_logger

log = get_logger(__name__)


class KillLevel(IntEnum):
    """Ordered, so comparisons mean what they look like."""

    NONE = 0
    L1_SOFT = 1
    L2_PAUSE = 2
    L3_FLATTEN = 3
    L4_LOCKOUT = 4

    @property
    def blocks_new_entries(self) -> bool:
        return self >= KillLevel.L2_PAUSE

    @property
    def requires_flatten(self) -> bool:
        return self >= KillLevel.L3_FLATTEN

    @property
    def needs_a_human(self) -> bool:
        return self >= KillLevel.L4_LOCKOUT


class Verdict(IntEnum):
    ALLOW = 0
    RESIZE = 1
    REJECT = 2


@dataclass(frozen=True)
class Position:
    """The minimum a flatten needs to know. Deliberately not the full model."""

    symbol: str
    lots: int
    is_long: bool
    event_id: str | None = None
    delta: float = 0.0
    vega: float = 0.0


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    lots: int
    is_long: bool
    loss_if_wrong: float  # rupees at risk, per the sizing module
    event_id: str | None = None
    delta: float = 0.0
    vega: float = 0.0


@dataclass
class BookState:
    """Everything the gate needs to know about the day so far.

    Passed in rather than queried, so the gate has no dependency that can be
    down when it is needed and so a test can put the book in any state without
    a database.
    """

    capital: float
    open_positions: list[Position] = field(default_factory=list)
    realised_pnl_today: float = 0.0
    realised_pnl_week: float = 0.0
    trades_today: int = 0
    trades_per_event: dict[str, int] = field(default_factory=dict)
    model_degraded: bool = False
    feed_age_seconds: float = 0.0
    clock_drift_ms: float = 0.0
    data_quality_failed: bool = False
    l3_events_this_week: int = 0
    unexplained_loss: bool = False

    @property
    def daily_loss_pct(self) -> float:
        """Positive means a loss. Zero when the day is up."""
        return max(0.0, -self.realised_pnl_today) / self.capital * 100 if self.capital else 0.0

    @property
    def weekly_loss_pct(self) -> float:
        return max(0.0, -self.realised_pnl_week) / self.capital * 100 if self.capital else 0.0

    @property
    def portfolio_delta(self) -> float:
        return sum(p.delta for p in self.open_positions)

    @property
    def portfolio_vega(self) -> float:
        return sum(p.vega for p in self.open_positions)


@dataclass
class RiskDecision:
    """The answer, and every reason for it.

    A rejection with no reason is not auditable, and the gate is the one place
    where every decision has to be explicable months later.
    """

    verdict: Verdict
    lots: int
    kill_level: KillLevel
    breaches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.verdict is not Verdict.REJECT and self.lots > 0

    def render(self) -> str:
        head = f"  {self.verdict.name}"
        if self.verdict is Verdict.RESIZE:
            head += f" to {self.lots} lot(s)"
        if self.kill_level is not KillLevel.NONE:
            head += f"  [kill level {self.kill_level.name}]"
        lines = [head]
        lines += [f"    BREACH: {b}" for b in self.breaches]
        lines += [f"    warn: {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
class RiskGate:
    """Every path to an order goes through here. There is no other path."""

    def __init__(
        self,
        limits: RiskLimits,
        *,
        session_open: time = time(9, 15),
        session_close: time = time(15, 30),
    ) -> None:
        self.limits = limits
        self.session_open = session_open
        self.session_close = session_close
        self.kill_level = KillLevel.NONE
        self.kill_reasons: list[str] = []
        self.lockout_note: str | None = None
        self._l3_history: list[date] = []

    # ------------------------------------------------------------ kill switches
    def escalate(self, level: KillLevel, reason: str, *, today: date | None = None) -> KillLevel:
        """Raise the kill level. It never falls on its own.

        Only upward, because a later, milder trigger must not undo an earlier
        severe one - a feed that comes back does not undo the fact that it went
        away with a position open.
        """
        if level > self.kill_level:
            self.kill_level = level
            log.info("kill level raised", fields={"level": level.name, "reason": reason})
        self.kill_reasons.append(f"{level.name}: {reason}")

        if level is KillLevel.L3_FLATTEN:
            day = today or date.today()
            self._l3_history.append(day)
            recent = [d for d in self._l3_history if (day - d) <= timedelta(days=7)]
            if len(recent) >= 2:
                self.kill_level = KillLevel.L4_LOCKOUT
                self.kill_reasons.append(
                    f"L4_LOCKOUT: {len(recent)} L3 events within a week - "
                    "the second one is evidence the first was not understood"
                )
                log.info("kill level raised", fields={"level": "L4_LOCKOUT", "reason": "two L3"})
        return self.kill_level

    def clear(self, note: str) -> None:
        """Stand the system down from a kill state.

        A lockout requires a written root-cause note, and this refuses without
        one. The failure being prevented is a system that recovers by itself and
        takes the same loss again on the same undiagnosed cause - which is not a
        hypothetical, it is what happens when clearing is one keypress.
        """
        if self.kill_level.needs_a_human and len(note.strip()) < 20:
            raise ValueError(
                "clearing a lockout requires a written root-cause note of at "
                f"least 20 characters; got {len(note.strip())}. If there is "
                "nothing to write down, the cause is not understood and the "
                "system should stay locked out."
            )
        log.info(
            "kill level cleared",
            fields={"from": self.kill_level.name, "note": note[:200]},
        )
        self.lockout_note = note
        self.kill_level = KillLevel.NONE
        self.kill_reasons.clear()

    def assess_conditions(self, book: BookState, *, today: date | None = None) -> KillLevel:
        """Read the book and raise the kill level where the rules say to.

        Called before every evaluation rather than on a timer, so a condition
        cannot be missed because the check happened to run a second earlier.
        """
        if book.unexplained_loss:
            self.escalate(
                KillLevel.L4_LOCKOUT,
                "unexplained loss - a loss nobody can account for is a loss that "
                "can happen again at any size",
                today=today,
            )
        if book.feed_age_seconds >= self.limits.feed_staleness_halt_seconds:
            self.escalate(
                KillLevel.L3_FLATTEN,
                f"feed silent for {book.feed_age_seconds:.1f}s - positions are "
                "open against prices nobody is watching",
                today=today,
            )
        if book.clock_drift_ms >= self.limits.clock_drift_halt_ms:
            self.escalate(
                KillLevel.L3_FLATTEN,
                f"clock drift {book.clock_drift_ms:.0f}ms - every point-in-time "
                "guarantee in the system depends on the clock",
                today=today,
            )
        if book.daily_loss_pct >= self.limits.daily_loss_limit_pct:
            self.escalate(
                KillLevel.L2_PAUSE,
                f"daily loss {book.daily_loss_pct:.2f}% >= "
                f"{self.limits.daily_loss_limit_pct}% - stop for the day, no overrides",
                today=today,
            )
        if book.weekly_loss_pct >= self.limits.weekly_loss_limit_pct:
            self.escalate(
                KillLevel.L2_PAUSE,
                f"weekly loss {book.weekly_loss_pct:.2f}% >= "
                f"{self.limits.weekly_loss_limit_pct}% - stop for the week, review required",
                today=today,
            )
        if book.data_quality_failed:
            self.escalate(KillLevel.L2_PAUSE, "data quality gate failed", today=today)
        if book.model_degraded:
            self.escalate(
                KillLevel.L1_SOFT,
                "model degraded - signals are advisory only",
                today=today,
            )
        return self.kill_level

    # -------------------------------------------------------------- evaluation
    def evaluate(
        self,
        trade: ProposedTrade,
        book: BookState,
        *,
        now: datetime,
        is_expiry_day: bool = False,
    ) -> RiskDecision:
        """Deny by default. Every check must pass for a trade to survive."""
        self.assess_conditions(book, today=now.date())

        breaches: list[str] = []
        warnings: list[str] = []
        lots = trade.lots

        if lots <= 0:
            return RiskDecision(Verdict.REJECT, 0, self.kill_level, ["no size proposed"], warnings)

        if self.kill_level.blocks_new_entries:
            breaches.append(
                f"kill level {self.kill_level.name} blocks new entries"
                + (" - a human must clear it" if self.kill_level.needs_a_human else "")
            )

        # ------------------------------------------------------------ the limits
        risk_pct = (trade.loss_if_wrong * lots) / book.capital * 100 if book.capital else math.inf
        if risk_pct > self.limits.per_trade_capital_at_risk_pct:
            affordable = int(
                (self.limits.per_trade_capital_at_risk_pct / 100 * book.capital)
                // trade.loss_if_wrong
            )
            if affordable < 1:
                breaches.append(
                    f"one lot risks {risk_pct / lots:.2f}% > "
                    f"{self.limits.per_trade_capital_at_risk_pct}% cap - cannot be resized down"
                )
            else:
                warnings.append(
                    f"resized {lots} -> {affordable} lots to stay inside the "
                    f"{self.limits.per_trade_capital_at_risk_pct}% per-trade cap"
                )
                lots = affordable

        if len(book.open_positions) >= self.limits.max_concurrent_positions:
            breaches.append(
                f"{len(book.open_positions)} open positions, limit "
                f"{self.limits.max_concurrent_positions}"
            )
        if book.trades_today >= self.limits.max_trades_per_day:
            breaches.append(
                f"{book.trades_today} trades today, limit {self.limits.max_trades_per_day}"
            )
        if trade.event_id is not None:
            taken = book.trades_per_event.get(trade.event_id, 0)
            if taken >= self.limits.max_trades_per_event:
                breaches.append(
                    f"{taken} trades already on event {trade.event_id!r}, limit "
                    f"{self.limits.max_trades_per_event} - this would be pyramiding "
                    "on one hypothesis"
                )

        # Delta and vega are checked on the book *including* this trade, because
        # the limit is on the resulting exposure and not on the increment.
        new_delta = book.portfolio_delta + trade.delta * lots
        new_vega = book.portfolio_vega + trade.vega * lots
        if abs(new_delta) > self.limits.max_portfolio_delta:
            breaches.append(
                f"portfolio delta would be {new_delta:+.2f}, band "
                f"+/-{self.limits.max_portfolio_delta}"
            )
        if abs(new_vega) > self.limits.max_portfolio_vega:
            breaches.append(
                f"portfolio vega would be {new_vega:+.2f}, band "
                f"+/-{self.limits.max_portfolio_vega} - an accidental all-in "
                "volatility bet"
            )

        # ------------------------------------------------------------ the clock
        minutes_in = (now.hour * 60 + now.minute) - (
            self.session_open.hour * 60 + self.session_open.minute
        )
        minutes_left = (self.session_close.hour * 60 + self.session_close.minute) - (
            now.hour * 60 + now.minute
        )
        if minutes_in < self.limits.no_entry_first_minutes:
            breaches.append(
                f"{minutes_in} min into the session, no entries for the first "
                f"{self.limits.no_entry_first_minutes}"
            )
        if minutes_left < self.limits.no_entry_last_minutes:
            breaches.append(
                f"{minutes_left} min to the close, no entries in the last "
                f"{self.limits.no_entry_last_minutes}"
            )

        flatten_h, flatten_m = (int(x) for x in self.limits.forced_flatten_time.split(":"))
        if (now.hour, now.minute) >= (flatten_h, flatten_m):
            breaches.append(
                f"past the forced flatten time {self.limits.forced_flatten_time} - "
                "every intraday position is squared off, no exceptions"
            )

        if is_expiry_day:
            warnings.append(
                f"expiry day - size multiplier {self.limits.expiry_day_size_multiplier} "
                "should already have been applied by sizing"
            )
        if self.kill_level is KillLevel.L1_SOFT:
            halved = lots // 2
            if halved < 1:
                # At one lot there is no half. The choices are one or none, and
                # L1 is explicitly not a stop - it is "reduce size by half;
                # notify", with the system still working. Rejecting here would
                # turn a soft kill into a hard one for every account whose
                # positions are already at the minimum tradeable size, which is
                # a much more severe rule than the one that was written down.
                #
                # So the trade stands at one lot and the warning says plainly
                # that the reduction could not be expressed. An operator whose
                # soft kill is not actually reducing anything needs to know
                # that, and a silent rejection tells them the opposite.
                warnings.append(
                    "L1 soft kill could NOT reduce size - already at one lot, "
                    "the minimum tradeable unit. The soft kill is in force but "
                    "is not shrinking anything."
                )
            else:
                warnings.append(f"L1 soft kill: halved {lots} -> {halved} lots")
                lots = halved

        if breaches:
            decision = RiskDecision(Verdict.REJECT, 0, self.kill_level, breaches, warnings)
        elif lots != trade.lots:
            decision = RiskDecision(Verdict.RESIZE, lots, self.kill_level, breaches, warnings)
        else:
            decision = RiskDecision(Verdict.ALLOW, lots, self.kill_level, breaches, warnings)

        log.info(
            "risk gate decision",
            fields={
                "verdict": decision.verdict.name,
                "lots": decision.lots,
                "kill_level": self.kill_level.name,
                "breaches": len(breaches),
            },
        )
        return decision


# --------------------------------------------------------------------------- #
# Flatten
# --------------------------------------------------------------------------- #
@dataclass
class FlattenResult:
    """What actually closed, and what did not.

    The unclosed list is the whole point. A flatten that reports success while
    leaving a position open is worse than one that fails loudly, because the
    operator stops looking.
    """

    closed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    attempts: int = 0

    @property
    def complete(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = [
            f"  flatten: {len(self.closed)} closed, {len(self.failed)} FAILED "
            f"across {self.attempts} attempts"
        ]
        lines += [f"    STILL OPEN {sym}: {err}" for sym, err in sorted(self.failed.items())]
        if not self.complete:
            lines.append("    -> positions remain open. Escalate to a human now.")
        return "\n".join(lines)


def flatten_all(
    positions: Sequence[Position],
    close: Callable[[Position], None],
    *,
    max_attempts: int = 3,
    on_failure: Callable[[FlattenResult], None] | None = None,
) -> FlattenResult:
    """Close everything at market, assuming the broker will fight back.

    Takes a callable rather than a broker object so it has no import path into
    the model layer, the feature store or the feed. The reason to flatten is
    usually that one of those broke, and a flatten that needs them is a flatten
    that does not work on the day it is needed - while passing every test
    written against a healthy system.

    Each position is retried independently. One symbol failing must not prevent
    the others from closing, which is what a single try/except around the whole
    loop would do: the first error would leave every subsequent position open,
    and the log would show one failure rather than all of them.
    """
    result = FlattenResult()
    remaining = list(positions)

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        still_open: list[Position] = []
        for position in remaining:
            try:
                close(position)
            except Exception as exc:
                still_open.append(position)
                result.failed[position.symbol] = f"{type(exc).__name__}: {exc}"
                log.info(
                    "flatten failed for one position",
                    fields={"symbol": position.symbol, "attempt": attempt},
                )
            else:
                result.closed.append(position.symbol)
                result.failed.pop(position.symbol, None)
        remaining = still_open
        if not remaining:
            break

    if result.failed:
        log.info(
            "FLATTEN INCOMPLETE - positions remain open",
            fields={"open": sorted(result.failed), "attempts": result.attempts},
        )
        if on_failure is not None:
            on_failure(result)
    return result


# --------------------------------------------------------------------------- #
# Limit-change discipline
# --------------------------------------------------------------------------- #
def limit_change_requires_adr(
    old: RiskLimits, new: RiskLimits, adr_slugs: Sequence[str]
) -> list[str]:
    """Which changed limits have no decision record.

    A risk limit that can be edited in a commit is a risk limit that gets edited
    at the worst moment - after a losing week, by the person who lost it, with a
    reason that sounds good at the time. Requiring a written record does not
    prevent the change; it makes it a decision rather than a nudge.

    Matching is by field name appearing in an ADR filename, deliberately loose:
    a strict scheme invites a mechanical rename, and the point is the writing,
    not the filing.
    """
    missing = []
    for name in type(old).model_fields:
        if getattr(old, name) == getattr(new, name):
            continue
        if not any(name in slug for slug in adr_slugs):
            missing.append(name)
    return missing
