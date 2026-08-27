"""M11 verify - feed protocol, gap detection, health, synthetic source."""

from __future__ import annotations

from datetime import timedelta
from itertools import pairwise

import pytest

from libs.ms_core.clock import FrozenClock, SimClock
from libs.ms_core.schemas import Tick, deterministic_ids
from libs.ms_data.feed import (
    FeedHealth,
    FeedState,
    GapDetector,
    SubscriptionPlan,
    SyntheticFeed,
    SyntheticInstrument,
    option_chain_token_estimate,
)
from tests.conftest import ist

INSTRUMENTS = [
    SyntheticInstrument(instrument_id=1, start_price=24_800, annual_vol=0.14),
    SyntheticInstrument(instrument_id=2, start_price=54_200, annual_vol=0.18),
]


def feed(clock: SimClock, calendar, **kw) -> SyntheticFeed:
    f = SyntheticFeed(INSTRUMENTS, calendar, clock, **kw)
    f.subscribe([1, 2])
    return f


# ------------------------------------------------------------ gap detection
def test_gap_detected_and_sized():
    g = GapDetector()
    assert g.check(Tick(ts=ist(9, 20), instrument_id=1, ltp=1.0, seq=10)) is None
    assert g.check(Tick(ts=ist(9, 20, 1), instrument_id=1, ltp=1.0, seq=11)) is None
    gap = g.check(Tick(ts=ist(9, 20, 2), instrument_id=1, ltp=1.0, seq=15))
    assert gap is not None and gap.size == 3 and gap.expected_seq == 12


def test_out_of_order_is_not_a_gap():
    g = GapDetector()
    g.check(Tick(ts=ist(9, 20), instrument_id=1, ltp=1.0, seq=10))
    assert g.check(Tick(ts=ist(9, 20), instrument_id=1, ltp=1.0, seq=9)) is None
    assert g.gaps == []


def test_no_sequence_number_means_no_false_guarantee():
    """Kite supplies no sequence number. Better to admit that than fabricate it."""
    g = GapDetector()
    assert g.check(Tick(ts=ist(9, 20), instrument_id=1, ltp=1.0, seq=None)) is None
    assert g.gaps == []


def test_reconnect_resets_sequence_tracking():
    g = GapDetector()
    g.check(Tick(ts=ist(9, 20), instrument_id=1, ltp=1.0, seq=500))
    g.reset()
    assert g.check(Tick(ts=ist(9, 21), instrument_id=1, ltp=1.0, seq=1)) is None
    assert g.gaps == []


# -------------------------------------------------------------------- health
def test_silence_during_market_hours_halts(calendar):
    clock = FrozenClock(ist(10, 0))
    h = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=3.0)
    h.on_message()
    assert h.evaluate() is FeedState.LIVE

    clock.set(ist(10, 0, 5))
    assert h.evaluate() is FeedState.HALTED


def test_silence_outside_market_hours_is_not_a_fault(calendar):
    """A quiet socket at 20:00 is a closed exchange, not a broken feed."""
    clock = FrozenClock(ist(20, 0))
    h = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=3.0)
    h.on_message()
    clock.set(ist(20, 5))
    assert h.evaluate() is not FeedState.HALTED


def test_recovery_after_reconnect(calendar):
    clock = FrozenClock(ist(10, 0))
    h = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=3.0)
    h.on_message()
    h.on_reconnect()
    assert h.state is FeedState.CONNECTING
    h.on_message()
    assert h.state is FeedState.LIVE
    assert h.reconnects == 1


# ----------------------------------------------------------------- synthetic
def test_deterministic_given_a_seed(calendar):
    with deterministic_ids():
        a = feed(SimClock(ist(9, 15)), calendar, seed=42).generate_session(ist(9, 15), ist(9, 20))
    with deterministic_ids():
        b = feed(SimClock(ist(9, 15)), calendar, seed=42).generate_session(ist(9, 15), ist(9, 20))
    assert [t.model_dump_json() for t in a] == [t.model_dump_json() for t in b]


def test_ids_are_random_by_default(calendar):
    """Deterministic ids are a replay tool. A live process must never use them -
    two runs would collide on every id."""
    from libs.ms_core.schemas import is_deterministic_id_mode

    assert not is_deterministic_id_mode()
    with deterministic_ids():
        assert is_deterministic_id_mode()
    assert not is_deterministic_id_mode()

    a = feed(SimClock(ist(9, 15)), calendar, seed=42).generate_session(ist(9, 15), ist(9, 16))
    b = feed(SimClock(ist(9, 15)), calendar, seed=42).generate_session(ist(9, 15), ist(9, 16))
    assert [t.ltp for t in a] == [t.ltp for t in b]
    assert [t.trace_id for t in a] != [t.trace_id for t in b]


def test_different_seeds_differ(calendar):
    a = feed(SimClock(ist(9, 15)), calendar, seed=1).generate_session(ist(9, 15), ist(9, 20))
    b = feed(SimClock(ist(9, 15)), calendar, seed=2).generate_session(ist(9, 15), ist(9, 20))
    assert [t.ltp for t in a] != [t.ltp for t in b]


def test_only_emits_during_continuous_session(calendar):
    ticks = feed(SimClock(ist(9, 0)), calendar).generate_session(ist(9, 0), ist(9, 20))
    assert ticks, "expected some ticks once the session opened"
    assert all(t.ts >= ist(9, 15) for t in ticks), "pre-open ticks leaked into the session"


def test_book_is_two_sided_and_ordered(calendar):
    for t in feed(SimClock(ist(9, 15)), calendar).generate_session(ist(9, 15), ist(9, 17)):
        assert t.bid is not None and t.ask is not None
        assert t.bid < t.ask
        assert t.mid is not None


def test_intraday_volatility_is_u_shaped(calendar):
    """Time-of-day normalisation is a core feature - flat test data would make
    every test of it pass vacuously."""
    f = feed(SimClock(ist(9, 15)), calendar, seed=11)
    open_ticks = f.generate_session(ist(9, 15), ist(9, 45))
    f2 = feed(SimClock(ist(12, 0)), calendar, seed=11)
    midday_ticks = f2.generate_session(ist(12, 0), ist(12, 30))

    def spread_of_returns(ticks):
        prices = [t.ltp for t in ticks if t.instrument_id == 1]
        rets = [abs(b / a - 1) for a, b in pairwise(prices)]
        return sum(rets) / len(rets)

    assert spread_of_returns(open_ticks) > spread_of_returns(midday_ticks)


def test_injected_gaps_are_detectable(calendar):
    f = feed(SimClock(ist(9, 15)), calendar, seed=5, gap_probability=0.2)
    g = GapDetector()
    for t in f.generate_session(ist(9, 15), ist(9, 25)):
        g.check(t)
    assert g.gaps, "gap injection produced no detectable gaps"


def test_stream_requires_subscription(calendar):
    f = SyntheticFeed(INSTRUMENTS, calendar, SimClock(ist(9, 15)))
    with pytest.raises(RuntimeError, match="subscribe"):
        next(iter(f.stream()))


def test_unknown_instrument_rejected(calendar):
    f = SyntheticFeed(INSTRUMENTS, calendar, SimClock(ist(9, 15)))
    with pytest.raises(KeyError):
        f.subscribe([999])


# --------------------------------------------------------- subscription plan
def test_plan_splits_across_connections():
    plan = SubscriptionPlan(max_per_connection=100, max_connections=3)
    assignments = plan.plan(list(range(250)))
    assert len(assignments) == 3
    assert sum(len(v) for v in assignments.values()) == 250
    assert all(len(v) <= 100 for v in assignments.values())


def test_plan_refuses_to_silently_truncate():
    """A chain that quietly stops updating at 09:40 is worse than a start-up error."""
    plan = SubscriptionPlan(max_per_connection=10, max_connections=2)
    with pytest.raises(ValueError, match="capacity"):
        plan.plan(list(range(50)))


def test_chain_token_budget_fits_kite_limits():
    """Two indices, two expiries, 15 strikes either side of the money."""
    need = option_chain_token_estimate(n_underlyings=2, n_expiries=2, strikes_per_side=15)
    assert need == 240
    plan = SubscriptionPlan(max_per_connection=3000, max_connections=3)
    plan.plan(list(range(need + 10)))  # + a few index/future/VIX tokens
    assert plan.utilisation() < 0.05, "chain should be a small fraction of the budget"


def test_health_reports_silence_seconds(calendar):
    clock = FrozenClock(ist(10, 0))
    h = FeedHealth(clock=clock, calendar=calendar, staleness_halt_seconds=3.0)
    assert h.silence_seconds() is None
    h.on_message()
    clock.set(ist(10, 0) + timedelta(seconds=1.5))
    assert 1.4 < h.silence_seconds() < 1.6
