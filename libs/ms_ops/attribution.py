"""M94 - attribution, calibration, and the class-level kill policy.

The plan's note on this module says the auto-disable policy does more for
long-run profitability than any model improvement, and that is worth taking
literally rather than as encouragement. A model that is right on eight event
classes and quietly bleeds on the ninth does not look broken - it looks like a
model with a mediocre average, and the instinct is to improve the model. The
cheaper and larger win is to stop trading the ninth.

Two conditions, and both are necessary
--------------------------------------

A class is disabled when it has **at least 25 trades** *and* **negative
expectancy** over a rolling quarter.

The trade count is not a formality. With five trades, a losing streak is the
overwhelmingly likely outcome of a coin flip with an edge, and a policy that
disables on it will disable half the working classes in the book - then look
effective, because the remaining ones are the ones that got lucky. The threshold
is what separates "this does not work" from "this has not had a chance".

The window is rolling rather than since-inception, because a class that lost
money for six months and has worked for the last three is a class that changed,
and an all-time average hides exactly that.

Re-enabling requires a change, not patience
-------------------------------------------

This is the part that makes the policy real. A disabled class that comes back
after a cooling-off period comes back to the same model, the same features and
the same market, and loses money again - the only thing that elapsed was the
operator's memory.

:meth:`KillPolicy.reenable` therefore demands a **changed model version** or a
written justification. Time alone does nothing, and there is no parameter that
makes it do something.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from libs.ms_core.obs import get_logger
from libs.ms_models.direction import reliability_curve, reliability_slope
from libs.ms_models.ensemble import brier_decomposition, brier_skill
from libs.ms_risk.ledger import LedgerEntry, Resolution

log = get_logger(__name__)

# Below this, a losing run is what a working edge looks like some of the time.
MIN_TRADES_TO_JUDGE = 25

# The rolling window. A class that lost for six months and has worked for three
# is a class that changed, and an all-time average hides that.
ROLLING_WINDOW = timedelta(days=92)


class Verdict(str):
    """Rendered verbatim in the report, so the words are the interface."""


WORKING = Verdict("working")
LOSING = Verdict("losing")
TOO_FEW = Verdict("too few trades to judge")
DISABLED = Verdict("DISABLED")


# --------------------------------------------------------------------------- #
# Per-class attribution
# --------------------------------------------------------------------------- #
@dataclass
class ClassReport:
    """One event class, and whether it deserves to keep trading."""

    event_class: str
    trades: int = 0
    wins: int = 0
    gross: float = 0.0
    costs: float = 0.0
    risked: float = 0.0
    predicted_edge: float = 0.0
    realised_edge: float = 0.0
    disabled: bool = False

    @property
    def net(self) -> float:
        return self.gross - self.costs

    @property
    def hit_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy(self) -> float:
        return self.net / self.trades if self.trades else 0.0

    @property
    def avg_r(self) -> float:
        """Net per unit risked. Comparable across classes that size differently.

        Expectancy in rupees says a class that trades big is important; avg R
        says whether it is any good. Reporting only the first is how a large,
        mediocre class hides behind a small, excellent one.
        """
        return self.net / self.risked if self.risked else 0.0

    @property
    def cost_drag(self) -> float:
        return self.costs / abs(self.gross) if self.gross else math.inf

    @property
    def verdict(self) -> Verdict:
        if self.disabled:
            return DISABLED
        if self.trades < MIN_TRADES_TO_JUDGE:
            return TOO_FEW
        return LOSING if self.expectancy <= 0 else WORKING

    def render(self) -> str:
        drag = "inf" if math.isinf(self.cost_drag) else f"{self.cost_drag:.0%}"
        return (
            f"    {self.event_class:<30} n={self.trades:<4} hit={self.hit_rate:.0%}  "
            f"avgR={self.avg_r:+.3f}  net={self.net:>10,.0f}  drag={drag:>6}  "
            f"{self.verdict}"
        )


def attribute(
    entries: Sequence[LedgerEntry], *, disabled: Sequence[str] = ()
) -> dict[str, ClassReport]:
    """Per-class performance from the ledger, worst first.

    Only resolved entries count. Including unresolvable ones as zeros would
    dilute every average toward nothing and make a class with missing outcomes
    look safer than one that was measured.
    """
    off = set(disabled)
    out: dict[str, ClassReport] = {}
    for e in entries:
        if e.resolution is not Resolution.RESOLVED or e.net_pnl is None:
            continue
        r = out.setdefault(e.event_class, ClassReport(e.event_class, disabled=e.event_class in off))
        r.trades += 1
        r.wins += 1 if e.net_pnl > 0 else 0
        r.gross += e.gross_pnl or 0.0
        r.costs += e.costs or 0.0
        r.risked += abs(e.entry_price) * e.lots
        # Edge realisation: how much of the predicted move actually arrived.
        r.predicted_edge += abs(e.p_up - 0.5) * 2
        if e.exit_price is not None and e.entry_price:
            r.realised_edge += abs(e.exit_price - e.entry_price) / e.entry_price
    return dict(sorted(out.items(), key=lambda kv: kv[1].net))


def edge_realisation_slope(reports: Sequence[ClassReport]) -> float | None:
    """Realised move against predicted edge, across classes.

    A slope near zero means the model's confidence carries no information about
    how far things actually move - which is compatible with a good hit rate and
    is why hit rate alone is not enough. A slope well below one means the edge
    is real but consistently smaller than claimed, which is a sizing problem
    rather than a model problem.
    """
    points = [
        (r.predicted_edge / r.trades, r.realised_edge / r.trades) for r in reports if r.trades
    ]
    if len(points) < 3:
        return None
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / sxx


# --------------------------------------------------------------------------- #
# The kill policy
# --------------------------------------------------------------------------- #
class KillPolicyError(RuntimeError):
    pass


@dataclass
class Disablement:
    """A class taken out of service, and what it would take to bring it back."""

    event_class: str
    disabled_on: date
    trades: int
    expectancy: float
    model_version: str
    reason: str
    reenabled_on: date | None = None
    reenabled_because: str = ""

    @property
    def active(self) -> bool:
        return self.reenabled_on is None


@dataclass
class KillPolicy:
    """Disables what does not work, and refuses to un-disable it on a hunch.

    The register is the state. A policy that recomputes from scratch each week
    would silently re-enable a class the moment a quiet quarter pushed its
    rolling expectancy back above zero - which is the same class, the same
    model, and the same reason it lost money.
    """

    register: dict[str, Disablement] = field(default_factory=dict)

    @property
    def disabled_classes(self) -> list[str]:
        return sorted(k for k, d in self.register.items() if d.active)

    def is_disabled(self, event_class: str) -> bool:
        d = self.register.get(event_class)
        return d is not None and d.active

    def evaluate(
        self, reports: dict[str, ClassReport], *, today: date, model_version: str
    ) -> list[Disablement]:
        """Disable every class that has earned it. Returns what changed.

        Both conditions, always. With five trades a losing streak is what a
        working edge looks like some of the time, and a policy that acts on it
        disables half the book and then looks effective because the survivors
        got lucky.
        """
        newly: list[Disablement] = []
        for name, r in reports.items():
            if self.is_disabled(name):
                continue
            if r.trades < MIN_TRADES_TO_JUDGE or r.expectancy > 0:
                continue
            d = Disablement(
                event_class=name,
                disabled_on=today,
                trades=r.trades,
                expectancy=r.expectancy,
                model_version=model_version,
                reason=(
                    f"{r.trades} trades over the rolling quarter with expectancy "
                    f"{r.expectancy:+,.2f} per trade"
                ),
            )
            self.register[name] = d
            r.disabled = True
            newly.append(d)
            log.info(
                "class auto-disabled",
                fields={"class": name, "trades": r.trades, "expectancy": r.expectancy},
            )
        return newly

    def reenable(
        self,
        event_class: str,
        *,
        today: date,
        model_version: str,
        justification: str = "",
    ) -> Disablement:
        """Bring a class back - but only if something actually changed.

        A disabled class that returns after a cooling-off period returns to the
        same model, the same features and the same market, and loses money
        again. The only thing that elapsed was the operator's memory.

        So either the model version differs from the one that was losing, or a
        human writes down why this time is different. Time alone does nothing,
        and there is no parameter that makes it do something.
        """
        d = self.register.get(event_class)
        if d is None or not d.active:
            raise KillPolicyError(f"{event_class} is not disabled")

        changed_model = model_version != d.model_version
        written = len(justification.strip()) >= 30

        if not changed_model and not written:
            raise KillPolicyError(
                f"{event_class} was disabled under model {d.model_version!r} and "
                f"the model is still {model_version!r}. Re-enabling needs a changed "
                "model or a written justification of at least 30 characters - "
                "waiting is not a change, and the class will lose money again for "
                "the same reason it did the first time."
            )

        d.reenabled_on = today
        d.reenabled_because = (
            f"model {d.model_version} -> {model_version}"
            if changed_model
            else justification.strip()
        )
        log.info(
            "class re-enabled",
            fields={"class": event_class, "because": d.reenabled_because},
        )
        return d

    def render(self) -> str:
        if not self.register:
            return "  no class has been disabled"
        lines = []
        for d in sorted(self.register.values(), key=lambda x: x.event_class):
            state = "disabled" if d.active else f"re-enabled {d.reenabled_on}"
            lines.append(f"    {d.event_class:<30} {state}  ({d.reason})")
            if not d.active:
                lines.append(f"      because: {d.reenabled_because}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The weekly report
# --------------------------------------------------------------------------- #
@dataclass
class WeeklyReport:
    week_ending: date
    classes: dict[str, ClassReport]
    policy: KillPolicy
    newly_disabled: list[Disablement] = field(default_factory=list)
    reliability_slope: float | None = None
    brier_skill: float | None = None
    edge_slope: float | None = None
    coverage: float | None = None

    @property
    def net(self) -> float:
        return sum(r.net for r in self.classes.values())

    @property
    def trades(self) -> int:
        return sum(r.trades for r in self.classes.values())

    def render(self) -> str:
        lines = [
            f"  week ending {self.week_ending}: {self.trades} resolved trades, "
            f"net {self.net:,.0f}",
            "",
            "  by class, worst first:",
        ]
        lines += [r.render() for r in self.classes.values()]

        lines += ["", "  calibration:"]
        slope = "n/a" if self.reliability_slope is None else f"{self.reliability_slope:.3f}"
        skill = "n/a" if self.brier_skill is None else f"{self.brier_skill:+.4f}"
        edge = "n/a" if self.edge_slope is None else f"{self.edge_slope:.3f}"
        lines.append(f"    reliability slope {slope}   (1.0 is honest; below 1 is overconfident)")
        lines.append(f"    Brier skill       {skill}   (above 0 beats the base rate)")
        lines.append(f"    edge realisation  {edge}   (how much of the claimed move arrived)")
        if self.coverage is not None:
            lines.append(f"    interval coverage {self.coverage:.1%}")

        if self.newly_disabled:
            lines += ["", "  DISABLED THIS WEEK:"]
            lines += [f"    {d.event_class}: {d.reason}" for d in self.newly_disabled]
            lines.append(
                "    Re-enabling needs a changed model or a written justification. "
                "Waiting is not a change."
            )
        lines += ["", "  kill register:", self.policy.render()]
        return "\n".join(lines)


def build_weekly_report(
    entries: Sequence[LedgerEntry],
    policy: KillPolicy,
    *,
    week_ending: date,
    model_version: str,
) -> WeeklyReport:
    """Everything the week is allowed to claim, and what it cost to claim it."""
    reports = attribute(entries, disabled=policy.disabled_classes)
    newly = policy.evaluate(reports, today=week_ending, model_version=model_version)

    resolved = [
        e for e in entries if e.resolution is Resolution.RESOLVED and e.was_right is not None
    ]
    slope = skill = None
    if len(resolved) >= 20:
        probs = [e.p_up for e in resolved]
        outcomes = [1 if e.was_right else 0 for e in resolved]
        slope = reliability_slope(reliability_curve(probs, outcomes))
        skill = brier_skill(probs, outcomes)

    covered = [e for e in resolved if e.p_lower is not None]
    coverage = None
    if len(covered) >= 20:
        inside = sum(1 for e in covered if e.p_lower is not None and e.p_lower <= e.p_up)
        coverage = inside / len(covered)

    return WeeklyReport(
        week_ending=week_ending,
        classes=reports,
        policy=policy,
        newly_disabled=newly,
        reliability_slope=slope,
        brier_skill=skill,
        edge_slope=edge_realisation_slope(list(reports.values())),
        coverage=coverage,
    )


def brier_breakdown(entries: Sequence[LedgerEntry]) -> str:
    """Reliability against resolution, for the weekly read.

    Two failures needing different responses: high reliability error means the
    probabilities are mislabelled and calibration fixes it; low resolution means
    the model is not separating outcomes at all and no calibration will help.
    """
    resolved = [
        e for e in entries if e.resolution is Resolution.RESOLVED and e.was_right is not None
    ]
    if len(resolved) < 20:
        return "  too few resolved predictions to decompose"
    d = brier_decomposition([e.p_up for e in resolved], [1 if e.was_right else 0 for e in resolved])
    return d.render()
