"""M72 - what could be traded, and what could be got out of.

    pytest tests/unit/test_structures.py -k "illiquid_excluded or no_trade_always_candidate"

The two names in the verify command are the two things that go wrong.

An illiquid leg does not announce itself: the structure prices, its Greeks are
sensible, and one side of it cannot be closed. And an enumerator without a
no-trade candidate hands the ranker a set in which something always wins, so
"the best of a bad set" and "a good opportunity" arrive in exactly the same
shape.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.schemas import OptionQuote, OptionType
from libs.ms_options.pricing import OptType
from libs.ms_options.structures import (
    MAX_SPREAD_PCT,
    MIN_OI,
    MIN_VOLUME,
    NO_TRADE,
    Candidate,
    Eligible,
    Grid,
    RejectReason,
    StructureKind,
    build_grid,
    enumerate_structures,
    summarise,
    viable_after_crossing,
)

TS = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
EXPIRY = TS + timedelta(days=28)
OTHER_EXPIRY = TS + timedelta(days=56)
FORWARD = 24120.0
LOT = 75


def q(
    strike: float,
    opt: OptionType,
    *,
    mid: float = 100.0,
    spread_pct: float = 0.02,
    oi: int = 5000,
    volume: int = 900,
    expiry: datetime = EXPIRY,
    bid: float | None = None,
    ask: float | None = None,
) -> OptionQuote:
    if bid is None or ask is None:
        half = mid * spread_pct / 2.0
        bid, ask = mid - half, mid + half
    return OptionQuote(
        expiry=expiry, strike=strike, opt_type=opt, bid=bid, ask=ask, oi=oi, volume=volume
    )


def realistic_mid(strike: float, opt: OptionType) -> float:
    """Intrinsic plus a time value that decays away from the forward.

    A flat ladder where every strike costs the same is not a chain: a condor
    built on it has exactly zero credit, so the crossing filter correctly
    refuses every one and a test asserting condors exist fails for a reason
    that has nothing to do with the code.
    """
    intrinsic = max(FORWARD - strike, 0.0) if opt is OptionType.CE else max(strike - FORWARD, 0.0)
    time_value = 90.0 * math.exp(-(((strike - FORWARD) / 420.0) ** 2))
    return intrinsic + time_value


def ladder(
    *,
    strikes: tuple[float, ...] = (23600.0, 23800.0, 24000.0, 24200.0, 24400.0, 24600.0),
    mid: float | None = None,
    **kw,
) -> list[OptionQuote]:
    out = []
    for s in strikes:
        for opt in (OptionType.CE, OptionType.PE):
            out.append(q(s, opt, mid=mid if mid is not None else realistic_mid(s, opt), **kw))
    return out


def elig(strike: float, kind: OptType, *, bid: float, ask: float) -> Eligible:
    return Eligible(strike=strike, kind=kind, expiry=EXPIRY, bid=bid, ask=ask, oi=5000, volume=900)


# --------------------------------------------------------------------------- #
# no_trade_always_candidate
# --------------------------------------------------------------------------- #
def test_no_trade_always_candidate_on_a_normal_chain():
    grid = build_grid(ladder(), expiry=EXPIRY)
    candidates = enumerate_structures(grid, forward=FORWARD, lot_size=LOT)

    assert any(c.is_no_trade for c in candidates)


def test_no_trade_always_candidate_on_an_empty_chain():
    """The case that matters. An enumerator that returns nothing forces every
    caller to handle the empty list, and one of them will not."""
    grid = build_grid([], expiry=EXPIRY)
    candidates = enumerate_structures(grid, forward=FORWARD, lot_size=LOT)

    assert candidates == [NO_TRADE]


def test_no_trade_always_candidate_when_every_quote_is_illiquid():
    """Nothing tradeable is a normal Tuesday, not an error."""
    grid = build_grid(ladder(oi=1, volume=0), expiry=EXPIRY)
    candidates = enumerate_structures(grid, forward=FORWARD, lot_size=LOT)

    assert len(grid) == 0
    assert candidates == [NO_TRADE]


def test_no_trade_always_candidate_is_first_so_it_is_never_the_leftover():
    """Position is not cosmetic: a ranker that breaks ties by list order should
    tie toward doing nothing, not toward whatever happened to be enumerated
    last."""
    grid = build_grid(ladder(), expiry=EXPIRY)
    assert enumerate_structures(grid, forward=FORWARD, lot_size=LOT)[0].is_no_trade


def test_no_trade_always_candidate_costs_and_risks_nothing():
    assert NO_TRADE.net_debit == 0.0
    assert NO_TRADE.crossing_cost == 0.0
    assert NO_TRADE.max_loss == 0.0
    assert NO_TRADE.max_profit == 0.0
    assert viable_after_crossing(NO_TRADE)


def test_no_trade_always_candidate_survives_the_crossing_filter():
    """It crosses nothing, so no threshold may exclude it."""
    assert viable_after_crossing(NO_TRADE, min_ratio=1e9)


def test_the_summary_says_why_no_trade_is_there():
    grid = build_grid(ladder(), expiry=EXPIRY)
    text = summarise(enumerate_structures(grid, forward=FORWARD, lot_size=LOT), grid)
    assert "competes on the same axis" in text


# --------------------------------------------------------------------------- #
# illiquid_excluded
# --------------------------------------------------------------------------- #
def test_illiquid_excluded_on_open_interest():
    quotes = [*ladder(), q(25000.0, OptionType.CE, oi=MIN_OI - 1)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert grid.get(25000.0, OptType.CALL) is None
    assert grid.stats.rejected[RejectReason.OI_TOO_LOW] == 1


def test_illiquid_excluded_on_volume():
    quotes = [*ladder(), q(25000.0, OptionType.CE, volume=MIN_VOLUME - 1)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert grid.get(25000.0, OptType.CALL) is None
    assert grid.stats.rejected[RejectReason.VOLUME_TOO_LOW] == 1


def test_illiquid_excluded_on_spread():
    quotes = [*ladder(), q(25000.0, OptionType.CE, spread_pct=MAX_SPREAD_PCT + 0.05)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert grid.get(25000.0, OptType.CALL) is None
    assert grid.stats.rejected[RejectReason.SPREAD_TOO_WIDE] == 1


def test_illiquid_excluded_when_a_field_is_missing_rather_than_low():
    """An absent field is not evidence of liquidity. Defaulting it to "fine" is
    how a chain with a broken feed produces a full grid."""
    grid = build_grid([q(24000.0, OptionType.CE, oi=None), q(24200.0, OptionType.CE, volume=None)], expiry=EXPIRY)  # type: ignore[arg-type]
    assert len(grid) == 0


def test_illiquid_excluded_from_every_leg_not_just_the_first():
    """The iron-condor case. Checking the leg you chose first and assuming the
    wings are fine builds a structure that prices beautifully and cannot be
    closed on one side."""
    quotes = ladder(strikes=(23600.0, 23800.0, 24000.0, 24200.0, 24400.0, 24600.0))
    # Make only the far call wing untradeable.
    quotes = [x for x in quotes if not (x.strike == 24600.0 and x.opt_type is OptionType.CE)]
    quotes.append(q(24600.0, OptionType.CE, oi=1))

    grid = build_grid(quotes, expiry=EXPIRY)
    candidates = enumerate_structures(grid, forward=FORWARD, lot_size=LOT)

    touching = [
        c
        for c in candidates
        if any(leg.strike == 24600.0 and leg.kind is OptType.CALL for leg in c.legs)
    ]
    assert touching == [], f"{len(touching)} structures used the illiquid 24600 call"


def test_illiquid_excluded_means_the_grid_is_the_only_source_of_strikes():
    """Every strike in every candidate came from the eligible grid. This is what
    makes "the grid is limited to eligible strikes" a property rather than a
    comment."""
    grid = build_grid(ladder(), expiry=EXPIRY)
    allowed = set(grid.by_key)

    for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT):
        for leg in c.legs:
            assert (leg.strike, leg.kind) in allowed


def test_illiquid_excluded_from_nearest_which_searches_only_the_grid():
    """A nearest-strike helper that searches the full ladder returns a strike
    that looks right, is not in the grid, and fails at the broker rather than at
    the filter."""
    quotes = [*ladder(), q(24100.0, OptionType.CE, oi=1)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert grid.nearest(24100.0, OptType.CALL) != 24100.0
    assert grid.nearest(24100.0, OptType.CALL) in (24000.0, 24200.0)


def test_nearest_returns_none_rather_than_inventing_a_strike():
    assert Grid().nearest(24000.0, OptType.CALL) is None


def test_a_different_expiry_is_not_in_this_grid():
    """Calendars are deferred, so a quote from another expiry has no business
    in a single-expiry grid - and silently including it would build a spread
    whose two legs expire on different days."""
    quotes = [*ladder(), q(24000.0, OptionType.CE, expiry=OTHER_EXPIRY)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert all(c.expiry == EXPIRY for c in grid.by_key.values())


def test_only_strikes_with_both_sides_can_pair():
    """A strike where only the call trades is eligible for a naked call and for
    nothing that needs a put."""
    quotes = [x for x in ladder() if not (x.strike == 24000.0 and x.opt_type is OptionType.PE)]
    grid = build_grid(quotes, expiry=EXPIRY)

    assert 24000.0 in grid.strikes(OptType.CALL)
    assert 24000.0 not in grid.both_sides

    straddles = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.STRADDLE
    ]
    assert all(leg.strike != 24000.0 for c in straddles for leg in c.legs)


# --------------------------------------------------------------------------- #
# crossing cost - what per-leg liquidity cannot catch
# --------------------------------------------------------------------------- #
def test_four_legs_each_inside_the_cap_can_still_be_untradeable():
    """The composition failure. Every leg passes every filter; the round trip
    crosses four spreads; and on a condor whose maximum profit is the credit
    received, that exceeds everything the structure could ever make."""
    # Every leg inside the spread cap, but the credit is thin.
    quotes = ladder(strikes=(23800.0, 24000.0, 24200.0, 24400.0), mid=20.0, spread_pct=0.09)
    grid = build_grid(quotes, expiry=EXPIRY)
    assert len(grid) == 8, "every leg was supposed to pass the per-leg filters"

    condors = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT, min_ratio=0.0)
        if c.kind is StructureKind.IRON_CONDOR
    ]
    assert condors, "the fixture should produce at least one condor"
    thin = condors[0]
    assert thin.crossing_cost > 0.0

    # With the real threshold it is gone.
    kept = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.IRON_CONDOR
    ]
    assert len(kept) < len(condors)


def test_crossing_cost_counts_every_leg():
    a = elig(24000.0, OptType.CALL, bid=99.0, ask=101.0)
    b = elig(24200.0, OptType.CALL, bid=49.0, ask=51.0)
    from libs.ms_options.greeks import Leg

    spread = Candidate(
        kind=StructureKind.VERTICAL_DEBIT,
        legs=(
            Leg(24000.0, OptType.CALL, EXPIRY, 1, LOT, True),
            Leg(24200.0, OptType.CALL, EXPIRY, 1, LOT, False),
        ),
        contracts=(a, b),
    )
    assert spread.crossing_cost == pytest.approx(2.0 + 2.0)


def test_an_unbounded_structure_is_judged_against_its_own_debit():
    """ "Can make unlimited money" is not an answer to "can it clear its
    spread"."""
    from libs.ms_options.greeks import Leg

    cheap = Candidate(
        kind=StructureKind.NAKED,
        legs=(Leg(24000.0, OptType.CALL, EXPIRY, 1, LOT, True),),
        contracts=(elig(24000.0, OptType.CALL, bid=1.0, ask=3.0),),
    )
    assert cheap.max_profit is None
    assert not viable_after_crossing(cheap), "a 2.00 spread on a 2.00 option cannot clear itself"

    rich = Candidate(
        kind=StructureKind.NAKED,
        legs=(Leg(24000.0, OptType.CALL, EXPIRY, 1, LOT, True),),
        contracts=(elig(24000.0, OptType.CALL, bid=99.0, ask=101.0),),
    )
    assert viable_after_crossing(rich)


def test_unbounded_is_none_not_a_large_number():
    """`None` and 1e9 behave the same in a max() and differently in a ratio."""
    from libs.ms_options.greeks import Leg

    naked = Candidate(
        kind=StructureKind.NAKED,
        legs=(Leg(24000.0, OptType.CALL, EXPIRY, 1, LOT, True),),
        contracts=(elig(24000.0, OptType.CALL, bid=99.0, ask=101.0),),
    )
    assert naked.max_profit is None


# --------------------------------------------------------------------------- #
# the structures themselves
# --------------------------------------------------------------------------- #
def test_every_named_structure_kind_is_produced():
    grid = build_grid(ladder(), expiry=EXPIRY)
    kinds = {c.kind for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)}

    assert kinds >= {
        StructureKind.NO_TRADE,
        StructureKind.NAKED,
        StructureKind.VERTICAL_DEBIT,
        StructureKind.STRADDLE,
        StructureKind.STRANGLE,
        StructureKind.IRON_CONDOR,
    }


def test_a_call_vertical_buys_the_lower_strike():
    """Reversing it makes a credit spread with inverted risk, priced and
    displayed identically."""
    grid = build_grid(ladder(), expiry=EXPIRY)
    verticals = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.VERTICAL_DEBIT
        and all(leg.kind is OptType.CALL for leg in c.legs)
    ]
    assert verticals
    for v in verticals:
        long_leg = next(leg for leg in v.legs if leg.is_long)
        short_leg = next(leg for leg in v.legs if not leg.is_long)
        assert long_leg.strike < short_leg.strike


def test_a_put_vertical_buys_the_higher_strike():
    grid = build_grid(ladder(), expiry=EXPIRY)
    verticals = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.VERTICAL_DEBIT and all(leg.kind is OptType.PUT for leg in c.legs)
    ]
    assert verticals
    for v in verticals:
        long_leg = next(leg for leg in v.legs if leg.is_long)
        short_leg = next(leg for leg in v.legs if not leg.is_long)
        assert long_leg.strike > short_leg.strike


def test_a_condor_has_its_wings_outside_its_shorts():
    grid = build_grid(ladder(), expiry=EXPIRY)
    condors = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT, min_ratio=0.0)
        if c.kind is StructureKind.IRON_CONDOR
    ]
    assert condors
    for c in condors:
        calls = [leg for leg in c.legs if leg.kind is OptType.CALL]
        puts = [leg for leg in c.legs if leg.kind is OptType.PUT]
        short_call = next(leg for leg in calls if not leg.is_long)
        long_call = next(leg for leg in calls if leg.is_long)
        short_put = next(leg for leg in puts if not leg.is_long)
        long_put = next(leg for leg in puts if leg.is_long)

        assert long_call.strike > short_call.strike
        assert long_put.strike < short_put.strike


def test_a_strangle_straddles_the_forward():
    grid = build_grid(ladder(), expiry=EXPIRY)
    strangles = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.STRANGLE
    ]
    assert strangles
    for s in strangles:
        call = next(leg for leg in s.legs if leg.kind is OptType.CALL)
        put = next(leg for leg in s.legs if leg.kind is OptType.PUT)
        assert put.strike < FORWARD < call.strike


def test_a_straddle_uses_one_strike_for_both_kinds():
    grid = build_grid(ladder(), expiry=EXPIRY)
    straddles = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.STRADDLE
    ]
    assert straddles
    for s in straddles:
        assert len({leg.strike for leg in s.legs}) == 1
        assert {leg.kind for leg in s.legs} == {OptType.CALL, OptType.PUT}
        assert all(leg.is_long for leg in s.legs)


def test_a_debit_structure_reports_a_positive_debit():
    grid = build_grid(ladder(), expiry=EXPIRY)
    for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT):
        if c.kind in (StructureKind.NAKED, StructureKind.STRADDLE, StructureKind.STRANGLE):
            assert c.net_debit > 0.0


def test_the_calendar_kind_is_deliberately_absent():
    """Deferred rather than forgotten: it spans two expiries, which makes its
    risk a function of the term structure, and M70 fits slices independently."""
    assert not hasattr(StructureKind, "CALENDAR")
    assert StructureKind.__doc__ is not None
    assert "deferred" in StructureKind.__doc__


def test_legs_carry_a_positive_lot_count_and_direction_separately():
    grid = build_grid(ladder(), expiry=EXPIRY)
    for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT):
        for leg in c.legs:
            assert leg.lots > 0
            assert leg.lot_size == LOT


def test_the_render_marks_direction_per_leg():
    grid = build_grid(ladder(), expiry=EXPIRY)
    verticals = [
        c
        for c in enumerate_structures(grid, forward=FORWARD, lot_size=LOT)
        if c.kind is StructureKind.VERTICAL_DEBIT
    ]
    text = verticals[0].render()
    assert "+" in text and "-" in text


def test_the_grid_stats_break_down_the_rejections():
    quotes = [
        *ladder(),
        q(25000.0, OptionType.CE, oi=1),
        q(25200.0, OptionType.CE, volume=0),
        q(25400.0, OptionType.CE, spread_pct=0.5),
    ]
    grid = build_grid(quotes, expiry=EXPIRY)
    text = grid.render()

    assert "oi_too_low" in text
    assert "volume_too_low" in text
    assert "spread_too_wide" in text
