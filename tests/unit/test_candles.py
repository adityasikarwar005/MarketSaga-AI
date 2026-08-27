"""M13 verify - candle builder. One test per correctness rule."""

from __future__ import annotations

import pytest

from libs.ms_core.clock import SessionState
from libs.ms_core.schemas import Tick
from libs.ms_data.candles import CandleBuilder, rollup, verify_session
from tests.conftest import ist


def t(hh, mm, ss=0, us=0, price=100.0, iid=1, qty=25, oi=None) -> Tick:
    return Tick(ts=ist(hh, mm, ss, us), instrument_id=iid, ltp=price, qty=qty, oi=oi)


# --------------------------------------------- Rule 1: left-closed buckets
def test_tick_at_exact_boundary_opens_the_new_candle(calendar):
    """09:20:00.000 belongs to the 09:20 candle. Off by one here shifts every
    sequence model's input by a bar and nobody notices for months."""
    b = CandleBuilder(calendar)
    b.on_tick(t(9, 19, 59, 999_000, price=100))
    emitted = b.on_tick(t(9, 20, 0, 0, price=101))
    assert len(emitted) == 1
    sealed = emitted[0]
    assert sealed.ts == ist(9, 19)
    assert sealed.c == 100  # the boundary tick did NOT land in 09:19


# ------------------------------------- Rule 2: synthetic candles for silence
def test_silent_minutes_produce_flagged_synthetic_candles(calendar):
    b = CandleBuilder(calendar)
    b.on_tick(t(9, 20, price=100))
    emitted = b.on_tick(t(9, 24, price=105))

    assert [c.ts for c in emitted] == [ist(9, 20), ist(9, 21), ist(9, 22), ist(9, 23)]
    real, synth = emitted[0], emitted[1:]
    assert not real.synthetic
    assert all(c.synthetic for c in synth)
    assert all(c.o == c.h == c.low == c.c == 100 for c in synth)
    assert all(c.v == 0 for c in synth)


def test_a_hole_is_never_left_in_the_grid(calendar):
    """Omitting silent minutes makes an outage indistinguishable from a quiet market."""
    b = CandleBuilder(calendar)
    candles = b.build([t(9, 20, price=100), t(9, 30, price=110)], up_to=ist(9, 31))
    stamps = [c.ts for c in candles]
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps))
    assert len(stamps) == 11  # 09:20 .. 09:30 inclusive, no gaps


# -------------------------------------------------- Rule 3: session tagging
def test_preopen_ticks_are_excluded(calendar):
    b = CandleBuilder(calendar)
    assert b.on_tick(t(9, 5, price=100)) == []
    assert b.build([t(9, 5, price=100)]) == []


def test_preopen_can_be_captured_separately(calendar):
    b = CandleBuilder(calendar, sessions=(SessionState.PREOPEN,))
    out = b.build([t(9, 2, price=100), t(9, 6, price=101)])
    assert out and all(c.ts < ist(9, 15) for c in out)


# --------------------------------------------------- Rule 4: late-tick rule
def test_late_tick_never_mutates_a_sealed_candle(calendar):
    b = CandleBuilder(calendar)
    b.on_tick(t(9, 20, price=100))
    sealed = b.on_tick(t(9, 21, price=101))[0]
    assert sealed.c == 100

    assert b.on_tick(t(9, 20, 30, price=999)) == []
    assert len(b.late_ticks) == 1
    late = b.late_ticks[0]
    assert late.bucket == ist(9, 20) and late.sealed_version == 1
    assert sealed.c == 100, "the sealed candle was mutated in place"


# --------------------------------------------------------- Rule 5: roll-ups
def test_rollup_cannot_disagree_with_its_minutes(calendar):
    b = CandleBuilder(calendar)
    ticks = [t(9, 15 + i // 4, i % 4 * 15, price=100 + i, qty=25) for i in range(20)]
    minutes = b.build(ticks, up_to=ist(9, 20))
    five = rollup(minutes, calendar, target_tf=5)

    assert len(five) == 1
    c5 = five[0]
    assert c5.ts == ist(9, 15)
    assert c5.o == minutes[0].o
    assert c5.c == minutes[-1].c
    assert c5.h == max(m.h for m in minutes)
    assert c5.low == min(m.low for m in minutes)
    assert c5.v == sum(m.v for m in minutes)


def test_rollup_refuses_non_minute_input(calendar):
    b = CandleBuilder(calendar)
    minutes = b.build([t(9, 15, price=100), t(9, 20, price=101)], up_to=ist(9, 21))
    five = rollup(minutes, calendar, target_tf=5)
    with pytest.raises(ValueError, match="1-minute"):
        rollup(five, calendar, target_tf=15)


# ------------------------------------------------------- Rule 6: OHLC logic
def test_ohlc_values_are_correct(calendar):
    b = CandleBuilder(calendar)
    b.build([t(9, 20, 0, price=100), t(9, 20, 20, price=105), t(9, 20, 40, price=98)])
    c = b.build([t(9, 20, 0, price=100), t(9, 20, 20, price=105), t(9, 20, 40, price=98)])[0]
    assert (c.o, c.h, c.low, c.c) == (100, 105, 98, 98)
    assert c.is_consistent()


def test_vwap_and_volume(calendar):
    b = CandleBuilder(calendar)
    c = b.build([t(9, 20, 0, price=100, qty=100), t(9, 20, 30, price=110, qty=100)])[0]
    assert c.v == 200
    assert c.vwap == pytest.approx(105.0)


def test_open_interest_carries_the_latest_value(calendar):
    b = CandleBuilder(calendar)
    c = b.build([t(9, 20, 0, price=100, oi=1000), t(9, 20, 30, price=101, oi=1200)])[0]
    assert c.oi == 1200


def test_multiple_instruments_are_independent(calendar):
    b = CandleBuilder(calendar)
    out = b.build(
        [t(9, 20, price=100, iid=1), t(9, 20, price=500, iid=2), t(9, 21, price=101, iid=1)]
    )
    by_id = {}
    for c in out:
        by_id.setdefault(c.instrument_id, []).append(c)
    assert set(by_id) == {1, 2}
    assert by_id[1][0].o == 100
    assert by_id[2][0].o == 500


# ---------------------------------------------------------- session verify
def test_full_session_has_exactly_375_candles(calendar):
    b = CandleBuilder(calendar)
    candles = b.build([t(9, 15, price=100), t(15, 29, price=110)], up_to=ist(15, 30))
    reports = verify_session(candles, calendar)
    assert len(reports) == 1
    r = reports[0]
    assert r.expected == 375
    assert r.actual == 375, f"expected a complete grid, got {r.actual}"
    assert r.ok


def test_report_flags_an_incomplete_session(calendar):
    b = CandleBuilder(calendar)
    candles = b.build([t(9, 15, price=100), t(10, 0, price=101)], up_to=ist(10, 1))
    r = verify_session(candles, calendar)[0]
    assert not r.ok
    assert r.actual < r.expected


def test_synthetic_share_is_reported(calendar):
    """A session that is 90% synthetic is a dead feed wearing a complete grid."""
    b = CandleBuilder(calendar)
    candles = b.build([t(9, 15, price=100), t(9, 25, price=100)], up_to=ist(9, 26))
    r = verify_session(candles, calendar)[0]
    assert r.synthetic_pct > 0.5
