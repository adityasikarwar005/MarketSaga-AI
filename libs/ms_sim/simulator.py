"""M82 - the event-driven simulator.

Owns the only number anyone is allowed to quote about profitability.

Every other module in this project can be wrong and produce a worse model. This
one can be wrong and produce a *better-looking* one, which is the failure that
ends projects. So the design is arranged around a single principle:

**A decision made at time T cannot be filled at time T's price.**

That sentence is the whole module. Violating it is the most profitable bug in
quantitative finance and the hardest to see, because nothing errors, nothing
looks out of place, and the equity curve simply goes up.

The three delays, and why each is separate
------------------------------------------

An order reaches the market strictly later than the event that motivated it, and
the gap has three distinct parts that are easy to collapse into one and wrong to:

* **Discovery delay** - the event happened at T, and you learned about it at
  T + d. Three seconds is forced in every run and cannot be configured to zero.
  A headline is not on your screen the instant it is filed, and a backtest that
  acts at the event timestamp is trading on a wire nobody has.
* **Decision latency** - feature computation, model inference, risk checks. This
  is sampled from a measured distribution rather than assumed constant, because
  the tail matters more than the mean: the slow decisions are the ones made
  during bursts, which is exactly when prices are moving.
* **Round trip to the exchange** - submission, acknowledgement, fill.

They are separate because they behave differently under load. Discovery does not
change when your machine is busy; decision latency doubles. Collapsing them into
one constant hides that the worst latency arrives at the worst moment.

Prices are read from the future of the decision, never its past
---------------------------------------------------------------

:class:`Simulator` computes a decision against the book **as of the decision
time**, and then fills it against the book **as of the arrival time**. The two
are different objects, fetched separately, and there is no code path that can
use one for the other. This is enforced by construction rather than by care,
because care does not survive a refactor.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from libs.ms_core.obs import get_logger
from libs.ms_sim.fills import BarOutcome, ExitCheck, Quote, Side, check_exit

log = get_logger(__name__)

# Forced in every run. Not a default - a floor. An event is not knowable at its
# own timestamp, and the three seconds is the cheapest honest approximation of
# the gap between a thing happening and a system knowing it happened.
MIN_DISCOVERY_DELAY = timedelta(seconds=3)


class LookaheadError(RuntimeError):
    """Raised when something tries to read a price from before its own decision."""


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
@dataclass
class LatencyModel:
    """Decision latency, sampled rather than assumed.

    A constant latency is wrong in the direction that flatters: the slow
    decisions happen during bursts, and bursts are when prices move. Sampling
    from a right-skewed distribution keeps the tail, and the tail is the part
    that costs money.

    ``measured`` is False until real timings have been fed in. An unmeasured
    latency model is a guess, and a backtest built on one should say so rather
    than presenting the same precision as its cost arithmetic.
    """

    median_ms: float = 120.0
    p99_ms: float = 800.0
    measured: bool = False
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.median_ms <= 0 or self.p99_ms < self.median_ms:
            raise ValueError("p99 latency cannot be below the median")
        self._rng = random.Random(self.seed)

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float], *, seed: int = 0) -> LatencyModel:
        """Fit from measured decision timings.

        Median and p99 rather than mean and standard deviation: the distribution
        is right-skewed and a mean is dragged around by the tail it is supposed
        to summarise.
        """
        if len(samples_ms) < 30:
            raise ValueError(
                f"{len(samples_ms)} timings is not enough to characterise a tail. "
                "An unmeasured latency model is honest; a badly measured one is not."
            )
        ordered = sorted(samples_ms)
        return cls(
            median_ms=statistics.median(ordered),
            p99_ms=ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
            measured=True,
            seed=seed,
        )

    def sample(self) -> timedelta:
        """Log-normal, matched to the median and the p99.

        Right-skewed on purpose. Symmetric noise around a median produces fast
        decisions as often as slow ones, and there is no such thing as a
        negative-latency decision.
        """
        # sigma from the p99: p99 = median * exp(2.326 * sigma)
        sigma = (
            math.log(self.p99_ms / self.median_ms) / 2.326 if self.p99_ms > self.median_ms else 0.0
        )
        ms = self.median_ms * math.exp(self._rng.gauss(0.0, sigma)) if sigma else self.median_ms
        return timedelta(milliseconds=max(1.0, ms))


# --------------------------------------------------------------------------- #
# The blotter
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    """One completed round trip, with every timestamp that mattered."""

    symbol: str
    event_class: str
    t_event: datetime
    t_decided: datetime
    t_filled: datetime
    entry: float
    exit: float
    qty: int
    is_long: bool
    costs: float
    exit_reason: str
    tie_broken: bool = False

    @property
    def gross(self) -> float:
        direction = 1 if self.is_long else -1
        return direction * (self.exit - self.entry) * self.qty

    @property
    def net(self) -> float:
        return self.gross - self.costs

    @property
    def won(self) -> bool:
        return self.net > 0

    @property
    def decision_lag(self) -> timedelta:
        return self.t_decided - self.t_event

    @property
    def fill_lag(self) -> timedelta:
        return self.t_filled - self.t_decided


@dataclass
class ClassAttribution:
    """Per-class performance. The number that decides what gets disabled."""

    event_class: str
    trades: int = 0
    wins: int = 0
    gross: float = 0.0
    costs: float = 0.0

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
    def cost_drag(self) -> float:
        """Costs as a share of gross. Above 1.0 means the edge never existed."""
        return self.costs / abs(self.gross) if self.gross else math.inf

    def render(self) -> str:
        drag = "inf" if math.isinf(self.cost_drag) else f"{self.cost_drag:.0%}"
        return (
            f"    {self.event_class:<34} n={self.trades:<4} hit={self.hit_rate:.0%}  "
            f"gross={self.gross:>10,.0f}  net={self.net:>10,.0f}  cost drag {drag}"
        )


@dataclass
class Blotter:
    """Every trade, and what can be said about them collectively."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    starting_capital: float = 0.0

    def record(self, trade: Trade) -> None:
        self.trades.append(trade)

    @property
    def net(self) -> float:
        return sum(t.net for t in self.trades)

    @property
    def gross(self) -> float:
        return sum(t.gross for t in self.trades)

    @property
    def costs(self) -> float:
        return sum(t.costs for t in self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross wins over gross losses, net of costs. The G7 gate asks for 1.2."""
        wins = sum(t.net for t in self.trades if t.net > 0)
        losses = -sum(t.net for t in self.trades if t.net < 0)
        return wins / losses if losses > 0 else math.inf if wins > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        return sum(1 for t in self.trades if t.won) / len(self.trades) if self.trades else 0.0

    @property
    def tie_broken_share(self) -> float:
        """What fraction of exits came from the tie rule rather than the data.

        Reported prominently. A backtest whose exits are mostly ties is largely
        a statement about the tie convention, and its profit factor is a
        property of that choice rather than of the strategy.
        """
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.tie_broken) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        """Peak-to-trough on the equity curve, as a fraction. G7 asks for <= 12%."""
        if not self.equity_curve:
            return 0.0
        peak = -math.inf
        worst = 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return worst

    def by_class(self) -> dict[str, ClassAttribution]:
        out: dict[str, ClassAttribution] = {}
        for t in self.trades:
            a = out.setdefault(t.event_class, ClassAttribution(t.event_class))
            a.trades += 1
            a.wins += t.won
            a.gross += t.gross
            a.costs += t.costs
        return dict(sorted(out.items(), key=lambda kv: kv[1].net))

    def render(self) -> str:
        if not self.trades:
            return "  no trades"
        pf = "inf" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        # Net shown against capital as well as in rupees. A net of -474 beside a
        # max drawdown of "0.0%" reads as a contradiction until you notice the
        # capital is a million - the percentages and the absolutes are on
        # different scales and printing only one invites the wrong reading.
        rel = (
            f" ({self.net / self.starting_capital:+.3%} of capital)"
            if self.starting_capital
            else ""
        )
        lines = [
            f"  {len(self.trades)} trades, hit rate {self.hit_rate:.0%}",
            f"  gross {self.gross:,.0f}  costs {self.costs:,.0f}  net {self.net:,.0f}{rel}",
            f"  profit factor {pf}   max drawdown {self.max_drawdown:.3%}",
        ]
        if self.tie_broken_share > 0.2:
            lines.append(
                f"  WARNING: {self.tie_broken_share:.0%} of exits were resolved by the "
                "tie rule, not by the data. This result is substantially a "
                "statement about that convention."
            )
        else:
            lines.append(f"  exits resolved by the tie rule: {self.tie_broken_share:.0%}")
        lines.append("  by event class, worst first:")
        lines += [a.render() for a in self.by_class().values()]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimEvent:
    """Something that happened, and when it actually happened."""

    t_event: datetime
    event_class: str
    symbol: str
    payload: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Intent:
    """What the strategy wants to do. Not what it gets."""

    symbol: str
    is_long: bool
    qty: int
    target: float
    stop: float


@dataclass
class SimConfig:
    discovery_delay: timedelta = MIN_DISCOVERY_DELAY
    latency: LatencyModel = field(default_factory=LatencyModel)
    starting_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.discovery_delay < MIN_DISCOVERY_DELAY:
            raise ValueError(
                f"discovery_delay {self.discovery_delay} is below the "
                f"{MIN_DISCOVERY_DELAY.total_seconds():.0f}s floor. An event is not "
                "knowable at its own timestamp, and a backtest that acts there is "
                "trading on a wire nobody has. This floor is not configurable."
            )


class Simulator:
    """Drives events through a strategy and fills against a later book.

    The strategy is a callable so the *production* decision code runs here rather
    than a parallel implementation. A simulator with its own logic tests the
    simulator.
    """

    def __init__(
        self,
        config: SimConfig,
        *,
        book_at: Callable[[str, datetime], Quote | None],
        bars_after: Callable[[str, datetime], Iterator[tuple[datetime, float, float]]],
        cost_of: Callable[[Intent, float], float],
    ) -> None:
        self.config = config
        self._book_at = book_at
        self._bars_after = bars_after
        self._cost_of = cost_of
        self.blotter = Blotter(starting_capital=config.starting_capital)
        self.skipped_no_book = 0

    def decision_time(self, event: SimEvent) -> datetime:
        """When the strategy is allowed to look at anything."""
        return event.t_event + self.config.discovery_delay

    def arrival_time(self, decided_at: datetime) -> datetime:
        """When the order reaches the market. Strictly after the decision."""
        return decided_at + self.config.latency.sample()

    def book_for_decision(self, symbol: str, at: datetime) -> Quote | None:
        """The book the strategy may look at. Never later than ``at``."""
        return self._book_at(symbol, at)

    def book_for_fill(self, symbol: str, at: datetime, *, decided_at: datetime) -> Quote | None:
        """The book the order actually meets.

        Refuses to return a book from before the decision. That is the lookahead
        this whole module exists to prevent, and a guard that raises is worth
        more than a comment saying not to.
        """
        if at < decided_at:
            raise LookaheadError(
                f"fill requested at {at.isoformat()} for a decision made at "
                f"{decided_at.isoformat()} - an order cannot reach the market "
                "before the decision to send it"
            )
        return self._book_at(symbol, at)

    def run(
        self,
        events: Sequence[SimEvent],
        strategy: Callable[[SimEvent, Quote, datetime], Intent | None],
    ) -> Blotter:
        equity = self.config.starting_capital
        self.blotter.equity_curve.append((events[0].t_event if events else datetime.min, equity))

        for event in sorted(events, key=lambda e: e.t_event):
            decided_at = self.decision_time(event)
            seen = self.book_for_decision(event.symbol, decided_at)
            if seen is None:
                self.skipped_no_book += 1
                continue

            intent = strategy(event, seen, decided_at)
            if intent is None or intent.qty <= 0:
                continue

            arrives = self.arrival_time(decided_at)
            filled_book = self.book_for_fill(intent.symbol, arrives, decided_at=decided_at)
            if filled_book is None:
                self.skipped_no_book += 1
                continue

            # The far touch, on the side the order actually crosses. Not the
            # mid: the mid is a number nobody trades at.
            entry = filled_book.far_touch(Side.BUY if intent.is_long else Side.SELL)
            exit_price, exit_reason, tie = self._walk_to_exit(intent, arrives)
            if exit_price is None:
                continue

            costs = self._cost_of(intent, entry)
            trade = Trade(
                symbol=intent.symbol,
                event_class=event.event_class,
                t_event=event.t_event,
                t_decided=decided_at,
                t_filled=arrives,
                entry=entry,
                exit=exit_price,
                qty=intent.qty,
                is_long=intent.is_long,
                costs=costs,
                exit_reason=exit_reason,
                tie_broken=tie,
            )
            self.blotter.record(trade)
            equity += trade.net
            self.blotter.equity_curve.append((arrives, equity))

        log.info(
            "simulation complete",
            fields={
                "events": len(events),
                "trades": len(self.blotter.trades),
                "skipped_no_book": self.skipped_no_book,
                "net": round(self.blotter.net, 2),
                "tie_share": round(self.blotter.tie_broken_share, 4),
            },
        )
        return self.blotter

    def _walk_to_exit(self, intent: Intent, from_ts: datetime) -> tuple[float | None, str, bool]:
        """Step forward bar by bar until a level is touched.

        Bars strictly after the fill. A bar that was already in progress when the
        order arrived has a high and a low that partly predate the position, and
        crediting those is the same lookahead in a smaller costume.
        """
        for _ts, high, low in self._bars_after(intent.symbol, from_ts):
            check: ExitCheck = check_exit(
                is_long=intent.is_long,
                high=high,
                low=low,
                target=intent.target,
                stop=intent.stop,
            )
            if check.exited and check.price is not None:
                return (
                    check.price,
                    check.reason,
                    check.outcome is BarOutcome.BOTH_ASSUMED_STOP,
                )
        return None, "never exited within the available data", False
