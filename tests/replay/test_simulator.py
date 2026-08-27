"""M82 - the simulator that owns the only profitability number.

The tests that matter here are the ones that fail when the simulator becomes
optimistic, because an optimistic simulator does not error - it produces a
better-looking result, and that is the failure that ends projects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_sim.fills import Quote
from libs.ms_sim.simulator import (
    MIN_DISCOVERY_DELAY,
    Blotter,
    Intent,
    LatencyModel,
    LookaheadError,
    SimConfig,
    SimEvent,
    Simulator,
    Trade,
)

T0 = datetime(2026, 3, 4, 6, 30, tzinfo=UTC)
SYM = "NIFTY24000CE"


def quote(price: float) -> Quote:
    return Quote(bid=price - 1, ask=price + 1, bid_size=1000, ask_size=1000)


def rising_book(base: float = 100.0, per_second: float = 0.5):
    """A market that moves away from you while you think.

    Deliberately adverse: any delay costs money, so a simulator that ignores
    delay reports a better number and the difference is visible.
    """

    def book_at(symbol: str, at: datetime) -> Quote | None:
        seconds = (at - T0).total_seconds()
        return quote(base + per_second * seconds)

    return book_at


def bars(highs_lows: list[tuple[float, float]]):
    def after(symbol: str, ts: datetime):
        for i, (high, low) in enumerate(highs_lows):
            yield ts + timedelta(minutes=i + 1), high, low

    return after


def free() -> float:
    return 0.0


def sim(*, config: SimConfig | None = None, book=None, bar_data=None, cost=None) -> Simulator:
    return Simulator(
        config or SimConfig(),
        book_at=book or rising_book(),
        bars_after=bar_data or bars([(200.0, 50.0)]),
        cost_of=cost or (lambda intent, price: 0.0),
    )


# ------------------------------------------------------------- the lookahead
def test_a_fill_cannot_be_read_from_before_its_decision():
    """The most profitable bug in quantitative finance, and the hardest to see:
    nothing errors, nothing looks wrong, the equity curve simply goes up."""
    s = sim()
    decided = T0 + timedelta(seconds=10)
    with pytest.raises(LookaheadError, match="cannot reach the market before"):
        s.book_for_fill(SYM, T0, decided_at=decided)


def test_the_fill_book_is_a_different_read_from_the_decision_book():
    """Two separate fetches at two separate times. If one could serve for both,
    a refactor would eventually make it."""
    s = sim(book=rising_book(base=100.0, per_second=1.0))
    decided = T0 + timedelta(seconds=5)
    arrives = decided + timedelta(seconds=2)

    seen = s.book_for_decision(SYM, decided)
    filled = s.book_for_fill(SYM, arrives, decided_at=decided)
    assert seen is not None and filled is not None
    assert filled.mid > seen.mid, "the market did not move between decision and fill"


def test_the_discovery_delay_has_a_floor_that_cannot_be_configured_away():
    """An event is not knowable at its own timestamp. A backtest that acts there
    is trading on a wire nobody has."""
    with pytest.raises(ValueError, match="not configurable"):
        SimConfig(discovery_delay=timedelta(0))
    with pytest.raises(ValueError, match="below the 3s floor"):
        SimConfig(discovery_delay=timedelta(seconds=1))

    assert SimConfig().discovery_delay >= MIN_DISCOVERY_DELAY


def test_the_decision_happens_strictly_after_the_event():
    s = sim()
    event = SimEvent(t_event=T0, event_class="geo.military", symbol=SYM)
    assert s.decision_time(event) == T0 + MIN_DISCOVERY_DELAY
    assert s.decision_time(event) > event.t_event


def test_the_order_arrives_strictly_after_the_decision():
    s = sim()
    decided = T0 + timedelta(seconds=3)
    for _ in range(50):
        assert s.arrival_time(decided) > decided


def test_delay_costs_money_in_an_adverse_market():
    """The check that a delay is actually being applied. If the simulator
    silently filled at the decision price this would be equal, not worse."""
    fast = SimConfig(latency=LatencyModel(median_ms=1.0, p99_ms=1.0, seed=1))
    slow = SimConfig(latency=LatencyModel(median_ms=5000.0, p99_ms=5000.0, seed=1))

    entries = []
    for cfg in (fast, slow):
        s = sim(config=cfg, book=rising_book(base=100.0, per_second=1.0))
        blotter = s.run(
            [SimEvent(T0, "geo.military", SYM)],
            lambda e, q, t: Intent(SYM, True, 10, target=1e9, stop=-1e9),
        )
        # target/stop unreachable, so no trade is booked - read the fill book directly
        entries.append(
            s.book_for_fill(
                SYM,
                s.arrival_time(s.decision_time(SimEvent(T0, "c", SYM))),
                decided_at=s.decision_time(SimEvent(T0, "c", SYM)),
            )
        )
        assert blotter is not None
    assert entries[1] is not None and entries[0] is not None
    assert entries[1].mid > entries[0].mid, "a slower decision did not pay a worse price"


# ------------------------------------------------------------------- latency
def test_latency_is_right_skewed_not_symmetric():
    """There is no such thing as a negative-latency decision, and symmetric
    noise produces fast decisions as often as slow ones."""
    model = LatencyModel(median_ms=100.0, p99_ms=1000.0, seed=7)
    samples = [model.sample().total_seconds() * 1000 for _ in range(4000)]

    assert min(samples) > 0
    mean = sum(samples) / len(samples)
    median = sorted(samples)[len(samples) // 2]
    assert mean > median, "a right-skewed distribution has its mean above its median"


def test_latency_respects_the_median_and_the_tail():
    model = LatencyModel(median_ms=120.0, p99_ms=800.0, seed=3)
    samples = sorted(model.sample().total_seconds() * 1000 for _ in range(5000))
    assert samples[len(samples) // 2] == pytest.approx(120.0, rel=0.1)
    assert samples[int(0.99 * len(samples))] == pytest.approx(800.0, rel=0.25)


def test_an_unmeasured_latency_model_says_so():
    assert LatencyModel().measured is False


def test_fitting_from_too_few_timings_is_refused():
    """An unmeasured model is honest. A badly measured one is not."""
    with pytest.raises(ValueError, match="not enough to characterise a tail"):
        LatencyModel.from_samples([100.0] * 10)


def test_fitting_uses_the_median_not_the_mean():
    """The distribution is right-skewed, and a mean is dragged around by the
    tail it is supposed to summarise."""
    samples = [100.0] * 99 + [100_000.0]
    model = LatencyModel.from_samples(samples)
    assert model.median_ms == pytest.approx(100.0)
    assert model.measured


def test_an_impossible_latency_shape_is_refused():
    with pytest.raises(ValueError, match="p99 latency cannot be below the median"):
        LatencyModel(median_ms=500.0, p99_ms=100.0)


# ------------------------------------------------------------------ the exit
def test_a_position_exits_only_on_bars_after_the_fill():
    """A bar already in progress when the order arrived has a high and a low
    that partly predate the position. Crediting those is the same lookahead in a
    smaller costume."""
    seen: list[datetime] = []

    def after(symbol: str, ts: datetime):
        seen.append(ts)
        yield ts + timedelta(minutes=1), 200.0, 50.0

    s = sim(bar_data=after)
    s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=150.0, stop=50.0),
    )
    assert seen and seen[0] > T0 + MIN_DISCOVERY_DELAY


def test_a_tie_is_recorded_on_the_trade():
    """So the blotter can report what share of its exits came from the
    convention rather than the data."""
    s = sim(bar_data=bars([(200.0, 50.0)]))
    blotter = s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=150.0, stop=90.0),
    )
    assert blotter.trades
    assert blotter.trades[0].tie_broken
    assert blotter.tie_broken_share == 1.0


def test_a_position_that_never_exits_is_not_booked():
    """Marking an open position as a closed trade at the last available price
    invents an exit that did not happen."""
    s = sim(bar_data=bars([(101.0, 99.0)] * 5))
    blotter = s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=1e6, stop=-1e6),
    )
    assert blotter.trades == []


# --------------------------------------------------------------- the blotter
def make_trade(net_target: float, *, klass: str = "geo.military", tie: bool = False) -> Trade:
    return Trade(
        symbol=SYM,
        event_class=klass,
        t_event=T0,
        t_decided=T0 + timedelta(seconds=3),
        t_filled=T0 + timedelta(seconds=4),
        entry=100.0,
        exit=100.0 + net_target,
        qty=1,
        is_long=True,
        costs=0.0,
        exit_reason="test",
        tie_broken=tie,
    )


def test_profit_factor_is_net_of_costs():
    b = Blotter()
    for t in (make_trade(10), make_trade(10), make_trade(-5)):
        b.record(t)
    assert b.profit_factor == pytest.approx(20 / 5)


def test_max_drawdown_is_peak_to_trough():
    b = Blotter()
    b.equity_curve = [(T0, 100.0), (T0, 120.0), (T0, 90.0), (T0, 130.0)]
    assert b.max_drawdown == pytest.approx((120 - 90) / 120)


def test_attribution_orders_the_worst_class_first():
    """The number that decides what gets disabled belongs at the top."""
    b = Blotter()
    b.record(make_trade(50, klass="good"))
    b.record(make_trade(-80, klass="bad"))
    classes = list(b.by_class())
    assert classes[0] == "bad"


def test_cost_drag_above_one_means_the_edge_never_existed():
    b = Blotter()
    t = make_trade(10)
    t = Trade(**{**t.__dict__, "costs": 25.0})
    b.record(t)
    attribution = b.by_class()["geo.military"]
    assert attribution.cost_drag > 1.0
    assert attribution.net < 0


def test_a_blotter_dominated_by_ties_warns_loudly():
    """A result whose exits are mostly ties is a statement about the convention,
    and its profit factor is a property of that choice."""
    b = Blotter()
    for _ in range(8):
        b.record(make_trade(10, tie=True))
    for _ in range(2):
        b.record(make_trade(10, tie=False))
    assert b.tie_broken_share == pytest.approx(0.8)
    assert "WARNING" in b.render()


def test_a_blotter_with_few_ties_reports_without_alarm():
    b = Blotter()
    b.record(make_trade(10, tie=True))
    for _ in range(19):
        b.record(make_trade(10, tie=False))
    assert "WARNING" not in b.render()


def test_an_empty_blotter_says_nothing_rather_than_zero():
    b = Blotter()
    assert b.render() == "  no trades"
    assert b.hit_rate == 0.0
    assert b.max_drawdown == 0.0


# ----------------------------------------------------------------- the run
def test_a_missing_book_skips_rather_than_inventing_a_price():
    s = Simulator(
        SimConfig(),
        book_at=lambda sym, at: None,
        bars_after=bars([(200.0, 50.0)]),
        cost_of=lambda i, p: 0.0,
    )
    blotter = s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=150.0, stop=50.0),
    )
    assert blotter.trades == []
    assert s.skipped_no_book == 1


def test_a_strategy_returning_nothing_books_nothing():
    s = sim()
    assert s.run([SimEvent(T0, "geo.military", SYM)], lambda e, q, t: None).trades == []


def test_costs_are_charged_through_the_supplied_model():
    s = sim(cost=lambda intent, price: 500.0)
    blotter = s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=150.0, stop=50.0),
    )
    assert blotter.trades
    assert blotter.costs == 500.0
    assert blotter.trades[0].net == blotter.trades[0].gross - 500.0


def test_every_trade_keeps_all_three_timestamps():
    """Without them a losing backtest cannot be diagnosed - was it the model, the
    discovery delay, or the latency?"""
    s = sim()
    blotter = s.run(
        [SimEvent(T0, "geo.military", SYM)],
        lambda e, q, t: Intent(SYM, True, 10, target=150.0, stop=50.0),
    )
    t = blotter.trades[0]
    assert t.t_event < t.t_decided < t.t_filled
    assert t.decision_lag >= MIN_DISCOVERY_DELAY
    assert t.fill_lag > timedelta(0)
