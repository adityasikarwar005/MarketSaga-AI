"""M50 verify - triple-barrier labelling."""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.schemas import Candle
from libs.ms_models.labels import (
    BarrierSpec,
    Outcome,
    apply_uniqueness,
    build_labels,
    build_meta_labels,
    effective_sample_size,
    summarise,
    trailing_sigma,
)
from tests.conftest import ist


@pytest.fixture
def cal(cfg):
    return SessionCalendar.from_config(cfg.sessions).synthetic_variant()


def flat(n: int = 120, price: float = 100.0, wobble: float = 0.0004) -> list[Candle]:
    """A gently oscillating session - no barrier is hit without help."""
    out, p = [], price
    for i in range(n):
        o = p
        p = p * (1 + wobble * ((i % 5) - 2))
        out.append(
            Candle(
                ts=ist(9, 15) + timedelta(minutes=i),
                instrument_id=1,
                tf=1,
                o=round(o, 2),
                h=round(max(o, p) * 1.0002, 2),
                l=round(min(o, p) * 0.9998, 2),
                c=round(p, 2),
                v=1000,
            )
        )
    return out


def with_move(n: int, at: int, size: float, bars: list[Candle] | None = None) -> list[Candle]:
    """Inject a sharp move from bar ``at`` onwards."""
    base = bars or flat(n)
    out = []
    for i, c in enumerate(base):
        if i < at:
            out.append(c)
            continue
        mult = 1 + size
        out.append(
            c.model_copy(
                update={
                    "o": round(c.o * mult, 2),
                    "h": round(c.h * mult, 2),
                    "low": round(c.low * mult, 2),
                    "c": round(c.c * mult, 2),
                }
            )
        )
    return out


# ----------------------------------------------------------------- volatility
def test_sigma_needs_enough_bars():
    spec = BarrierSpec()
    assert trailing_sigma(flat(3), spec) is None
    assert trailing_sigma(flat(30), spec) is not None


def test_sigma_is_zero_on_a_flat_series():
    """A truly flat series has no volatility, and a barrier at zero distance is
    meaningless - better to skip the label than to place one at the entry price."""
    bars = [
        Candle(
            ts=ist(9, 15) + timedelta(minutes=i),
            instrument_id=1,
            tf=1,
            o=100.0,
            h=100.0,
            l=100.0,
            c=100.0,
        )
        for i in range(30)
    ]
    assert trailing_sigma(bars, BarrierSpec()) is None


# --------------------------------------------------------------------- basics
def test_labels_are_produced(cal):
    labels = build_labels(flat(200), BarrierSpec(), cal)
    assert labels
    assert all(x.upper > x.entry > x.lower for x in labels)


def test_barriers_scale_with_volatility(cal):
    """A fixed-point barrier is a coin flip in a calm hour and unreachable in a
    quiet one."""
    calm = build_labels(flat(200, wobble=0.0002), BarrierSpec(), cal)
    wild = build_labels(flat(200, wobble=0.0020), BarrierSpec(), cal)
    assert calm and wild
    calm_width = sum((x.upper - x.lower) / x.entry for x in calm) / len(calm)
    wild_width = sum((x.upper - x.lower) / x.entry for x in wild) / len(wild)
    assert wild_width > calm_width * 3


def test_volatility_uses_only_prior_bars(cal):
    """The bar we enter on must not inform the barrier placed on it."""
    bars = flat(200)
    spec = BarrierSpec(vol_lookback_minutes=30)
    labels = build_labels(bars, spec, cal)
    by_ts = {b.ts: i for i, b in enumerate(bars)}
    for label in labels[:5]:
        idx = by_ts[label.t0]
        expected = trailing_sigma(bars[idx - spec.vol_lookback_minutes : idx], spec)
        assert label.sigma == pytest.approx(expected)


def test_an_upward_move_produces_an_up_label(cal):
    bars = with_move(200, at=60, size=0.01)
    labels = build_labels(bars, BarrierSpec(k_up=1.0, k_down=1.0), cal)
    around = [
        x
        for x in labels
        if ist(9, 15) + timedelta(minutes=55) <= x.t0 < ist(9, 15) + timedelta(minutes=60)
    ]
    assert around
    assert any(x.outcome is Outcome.UP for x in around)


def test_a_downward_move_produces_a_down_label(cal):
    bars = with_move(200, at=60, size=-0.01)
    labels = build_labels(bars, BarrierSpec(k_up=1.0, k_down=1.0), cal)
    around = [
        x
        for x in labels
        if ist(9, 15) + timedelta(minutes=55) <= x.t0 < ist(9, 15) + timedelta(minutes=60)
    ]
    assert around
    assert any(x.outcome is Outcome.DOWN for x in around)


def test_quiet_market_times_out(cal):
    # Small enough that wide barriers are never reached, but large enough to
    # survive two-decimal rounding - below that, sigma is genuinely zero and the
    # labeller correctly refuses to place a barrier at all.
    labels = build_labels(flat(200, wobble=0.0001), BarrierSpec(k_up=5.0, k_down=5.0), cal)
    assert labels
    assert all(x.outcome is Outcome.TIMEOUT for x in labels)


# ------------------------------------------------------------------ the traps
def test_intrabar_ambiguity_resolves_against_the_trade(cal):
    """OHLC cannot say which barrier came first. Guessing favourably is how a
    backtest gains several points of accuracy that do not exist."""
    bars = flat(120)
    spec = BarrierSpec(k_up=0.5, k_down=0.5, vol_lookback_minutes=30)
    labels = build_labels(bars, spec, cal)
    assert labels

    # Widen one bar so it engulfs both barriers.
    target_idx = 40
    wide = list(bars)
    c = wide[target_idx]
    wide[target_idx] = c.model_copy(update={"h": c.h * 1.05, "low": c.low * 0.95})

    relabelled = build_labels(wide, spec, cal)
    engulfing = [x for x in relabelled if x.ambiguous]
    assert engulfing, "no ambiguous label produced"
    assert all(x.outcome is Outcome.DOWN for x in engulfing)


def test_no_barrier_crosses_the_session_close(cal):
    labels = build_labels(flat(375), BarrierSpec(horizon_minutes=30), cal)
    _, close = cal.session_bounds(cal.local(ist(9, 15)).date())
    assert labels
    assert all(x.deadline <= close for x in labels)
    assert all(x.exit_ts <= close for x in labels)


def test_no_entries_too_close_to_the_close(cal):
    """A position opened at 15:28 cannot reach a fifteen-minute barrier."""
    spec = BarrierSpec(horizon_minutes=15, no_entry_before_close=20)
    labels = build_labels(flat(375), spec, cal)
    _, close = cal.session_bounds(cal.local(ist(9, 15)).date())
    assert labels
    assert all(x.t0 < close - timedelta(minutes=20) for x in labels)


# ---------------------------------------------------------------- uniqueness
def test_overlapping_labels_get_low_uniqueness(cal):
    """Two labels whose windows overlap observe mostly the same price path."""
    labels = build_labels(flat(200), BarrierSpec(horizon_minutes=15), cal, entry_every=1)
    assert labels
    mean_uniqueness = sum(x.uniqueness for x in labels) / len(labels)
    assert mean_uniqueness < 0.4


def test_thinning_entries_raises_uniqueness(cal):
    dense = build_labels(flat(300), BarrierSpec(horizon_minutes=15), cal, entry_every=1)
    sparse = build_labels(flat(300), BarrierSpec(horizon_minutes=15), cal, entry_every=20)
    assert dense and sparse
    dense_u = sum(x.uniqueness for x in dense) / len(dense)
    sparse_u = sum(x.uniqueness for x in sparse) / len(sparse)
    assert sparse_u > dense_u


def test_effective_sample_size_is_far_below_the_row_count(cal):
    """A thousand rows at 0.05 uniqueness is fifty observations, and a
    significance test that thinks it has a thousand will find an edge in noise."""
    labels = build_labels(flat(300), BarrierSpec(horizon_minutes=15), cal)
    n_eff = effective_sample_size(labels)
    assert 0 < n_eff < len(labels) * 0.5


def test_uniqueness_of_a_single_label_is_one():
    from libs.ms_models.labels import Label

    only = Label(
        t0=ist(9, 20),
        instrument_id=1,
        entry=100.0,
        upper=101.0,
        lower=99.0,
        deadline=ist(9, 35),
        outcome=Outcome.TIMEOUT,
        exit_ts=ist(9, 35),
        exit_price=100.0,
        sigma=0.001,
    )
    apply_uniqueness([only])
    assert only.uniqueness == pytest.approx(1.0)


# --------------------------------------------------------------- meta-labels
def test_meta_labels_reflect_whether_the_call_paid(cal):
    labels = build_labels(with_move(200, at=60, size=0.01), BarrierSpec(k_up=1.0, k_down=1.0), cal)
    long_everything = dict.fromkeys((x.t0 for x in labels), 1)
    meta = build_meta_labels(labels, long_everything)
    assert meta
    winners = [m for m in meta if m.take]
    assert winners
    assert all(m.realised > 0 for m in winners)


def test_costs_turn_marginal_wins_into_skips(cal):
    """A call that is right by less than the spread should have been skipped."""
    labels = build_labels(flat(200), BarrierSpec(), cal)
    sides = dict.fromkeys((x.t0 for x in labels), 1)
    free = build_meta_labels(labels, sides, cost=0.0)
    costly = build_meta_labels(labels, sides, cost=0.01)
    assert sum(m.take for m in costly) < sum(m.take for m in free)


def test_zero_side_is_not_labelled(cal):
    labels = build_labels(flat(200), BarrierSpec(), cal)
    assert build_meta_labels(labels, dict.fromkeys((x.t0 for x in labels), 0)) == []


# ------------------------------------------------------------------- report
def test_summary_reports_balance_and_effective_n(cal):
    labels = build_labels(flat(300), BarrierSpec(), cal)
    report = summarise(labels, cal)
    assert report.total == len(labels)
    assert report.up + report.down + report.timeout == report.total
    assert 0.0 <= report.balance <= 1.0
    assert report.effective_n <= report.total
    assert "effective n" in report.render()


def test_empty_input_is_not_an_error(cal):
    assert build_labels([], BarrierSpec(), cal) == []
    assert summarise([]).total == 0


def test_bad_spec_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        BarrierSpec(k_up=0)
    with pytest.raises(ValueError, match="positive"):
        BarrierSpec(horizon_minutes=0)
