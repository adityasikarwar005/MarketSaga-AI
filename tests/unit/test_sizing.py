"""M75 position sizing - Kelly under an estimated probability, and every way
the answer is allowed to get smaller."""

from __future__ import annotations

import pytest

from libs.ms_core import config as cfgmod
from libs.ms_options.sizing import (
    MIN_MEANINGFUL_FRACTION,
    BookRisk,
    KellyInput,
    Modifier,
    correlation_modifier,
    expiry_modifier,
    kelly_fraction,
    loss_streak_modifier,
    size_position,
    standard_modifiers,
    time_of_day_modifier,
    uncertainty_modifier,
)

CAPITAL = 500_000.0
LOT = 75


@pytest.fixture(scope="module")
def limits():
    return cfgmod.load(warn=False).risk_limits


def edge(p: float = 0.58, lower: float | None = 0.52) -> KellyInput:
    return KellyInput(p_win=p, win_amount=8500.0, loss_amount=6400.0, p_lower=lower)


# ------------------------------------------------------------------ the formula
def test_kelly_uses_the_odds_not_only_the_probability():
    """A debit spread risking 85 to make 115 and a naked long risking 100 to make
    400 are different bets at the same hit rate. Code that assumes even money
    sizes both as though they were the first."""
    tight = KellyInput(p_win=0.6, win_amount=115.0, loss_amount=85.0)
    wide = KellyInput(p_win=0.6, win_amount=400.0, loss_amount=100.0)
    assert kelly_fraction(wide) != kelly_fraction(tight)
    assert kelly_fraction(wide) > kelly_fraction(tight)


def test_kelly_is_zero_not_negative_without_an_edge():
    """A negative Kelly says "take the other side". Acting on it here would
    silently invert a signal the direction model never gave - the caller asked
    how much, not which way."""
    assert kelly_fraction(KellyInput(p_win=0.3, win_amount=100.0, loss_amount=100.0)) == 0.0
    assert kelly_fraction(KellyInput(p_win=0.5, win_amount=100.0, loss_amount=100.0)) == 0.0


def test_kelly_matches_the_even_money_closed_form():
    """At even odds f* = 2p - 1. A wrong formula still returns plausible
    fractions, so it is worth pinning against something checkable by hand."""
    for p in (0.55, 0.6, 0.75):
        k = KellyInput(p_win=p, win_amount=100.0, loss_amount=100.0)
        assert kelly_fraction(k) == pytest.approx(2 * p - 1)


def test_an_unbounded_loss_is_refused_rather_than_sized():
    with pytest.raises(ValueError, match="loss_amount must be positive"):
        KellyInput(p_win=0.6, win_amount=100.0, loss_amount=0.0)


def test_a_lower_bound_above_the_estimate_is_refused():
    with pytest.raises(ValueError, match="no greater than p_win"):
        KellyInput(p_win=0.55, win_amount=100.0, loss_amount=100.0, p_lower=0.60)


# ------------------------------------------------- the estimate is not the truth
def test_shrinks_on_high_variance():
    """The defence that matters most and is usually skipped. Kelly's derivation
    assumes you know p; sizing on the point estimate bets a quantity nobody
    has."""
    confident = kelly_fraction(edge(0.58, 0.56))
    unsure = kelly_fraction(edge(0.58, 0.44))
    assert unsure < confident
    assert unsure < confident / 2


def test_a_point_estimate_is_flagged_as_the_least_reliable_case(limits):
    k = edge(lower=None)
    decision = size_position(k, capital=CAPITAL, lot_size=LOT, limits=limits)
    assert not k.uncertain
    assert any("point estimate" in n for n in decision.notes)


def test_the_uncertainty_modifier_is_separate_from_the_lower_bound():
    """Both are wanted: the bound changes which bet is sized, the modifier
    changes how much of it to take."""
    tight = uncertainty_modifier(0.60, 0.58)
    wide = uncertainty_modifier(0.60, 0.45)
    assert tight.value > wide.value
    assert uncertainty_modifier(0.60, None).value == 0.5


def test_the_uncertainty_modifier_has_a_floor():
    """However wide the interval, this alone must not zero the position - the
    Kelly substitution has already priced most of that in."""
    assert uncertainty_modifier(0.9, 0.0).value >= 1 / 3


# --------------------------------------------------------------------- the cap
def test_never_exceeds_cap(limits):
    """Whatever Kelly says, the per-trade cap is the number that must hold."""
    huge = KellyInput(p_win=0.95, win_amount=50_000.0, loss_amount=1000.0, p_lower=0.94)
    decision = size_position(huge, capital=CAPITAL, lot_size=LOT, limits=limits)
    assert decision.capital_at_risk_pct <= limits.per_trade_capital_at_risk_pct / 100 + 1e-9
    assert "per-trade cap" in decision.cap_reason


def test_the_cap_holds_across_a_range_of_edges(limits):
    for p, lower in ((0.6, 0.55), (0.75, 0.7), (0.9, 0.85), (0.99, 0.98)):
        k = KellyInput(p_win=p, win_amount=10_000.0, loss_amount=2000.0, p_lower=lower)
        decision = size_position(k, capital=CAPITAL, lot_size=LOT, limits=limits)
        assert decision.capital_at_risk_pct <= limits.per_trade_capital_at_risk_pct / 100 + 1e-9


def test_the_cap_holds_with_modifiers_too(limits):
    k = KellyInput(p_win=0.9, win_amount=50_000.0, loss_amount=1000.0, p_lower=0.88)
    mods = standard_modifiers(k, limits=limits)
    decision = size_position(k, capital=CAPITAL, lot_size=LOT, limits=limits, modifiers=mods)
    assert decision.capital_at_risk_pct <= limits.per_trade_capital_at_risk_pct / 100 + 1e-9


# ------------------------------------------------------------------ whole lots
def test_rounds_down(limits):
    """A fractional lot cannot be traded, and rounding to nearest would turn a
    0.6-lot answer into a position 67% larger than intended - through the cap,
    not up against it."""
    k = edge()
    decision = size_position(k, capital=CAPITAL, lot_size=LOT, limits=limits)
    budget = decision.capped_fraction * CAPITAL
    assert decision.lots == int(budget // decision.loss_per_lot)
    assert decision.capital_at_risk <= budget + 1e-9


def test_rounding_down_never_breaches_the_cap(limits):
    for capital in (100_000.0, 250_000.0, 333_333.0, 1_000_000.0, 4_567_890.0):
        decision = size_position(edge(), capital=capital, lot_size=LOT, limits=limits)
        assert decision.capital_at_risk_pct <= limits.per_trade_capital_at_risk_pct / 100 + 1e-9


def test_a_size_of_zero_is_reported_not_swallowed(limits):
    """A silent skip is a bug that looks like a quiet day."""
    decision = size_position(edge(), capital=40_000.0, lot_size=LOT, limits=limits)
    assert decision.rejected
    assert decision.notes
    assert any("too small" in n for n in decision.notes)


def test_a_small_account_and_a_vetoing_context_give_different_reasons(limits):
    """Telling a user to fund the account when the real cause was a risk control
    would be advice to override that control."""
    k = edge()
    small = size_position(k, capital=40_000.0, lot_size=LOT, limits=limits)
    assert any("account is too small" in n for n in small.notes)

    vetoed = size_position(
        k,
        capital=CAPITAL,
        lot_size=LOT,
        limits=limits,
        modifiers=standard_modifiers(
            k,
            limits=limits,
            is_expiry_day=True,
            consecutive_losses=3,
            open_symbols=["NIFTY24200CE"],
            proposed_symbol="NIFTY24000CE",
        ),
    )
    assert vetoed.rejected
    assert any("the context is not" in n for n in vetoed.notes)


def test_a_token_position_is_refused(limits):
    """Below the meaningful floor a position pays four orders of costs to
    express an opinion the sizing just declined to hold."""
    barely = KellyInput(p_win=0.5001, win_amount=100.0, loss_amount=100.0, p_lower=0.5001)
    decision = size_position(barely, capital=CAPITAL, lot_size=LOT, limits=limits)
    assert decision.rejected
    assert decision.capped_fraction == 0.0 or decision.capped_fraction >= MIN_MEANINGFUL_FRACTION


# ------------------------------------------------------------------ modifiers
def test_a_modifier_may_only_reduce(limits):
    """A multiplier above one turns a risk control into a leverage knob, and it
    would be an easy thing to add by accident."""
    with pytest.raises(ValueError, match="leverage knob"):
        Modifier("boost", 1.5)
    with pytest.raises(ValueError, match="outside"):
        Modifier("negative", -0.1)


def test_expiry_day_halves_the_size(limits):
    assert expiry_modifier(False, limits).value == 1.0
    assert expiry_modifier(True, limits).value == limits.expiry_day_size_multiplier
    assert expiry_modifier(True, limits).value < 1.0


def test_a_loss_streak_halves_and_halves_again():
    assert loss_streak_modifier(0).value == 1.0
    assert loss_streak_modifier(2).value == 1.0
    assert loss_streak_modifier(3).value == 0.5
    assert loss_streak_modifier(6).value == 0.25
    assert loss_streak_modifier(60).value >= 0.125


def test_the_open_and_close_are_a_different_market():
    assert time_of_day_modifier(5, 300).value < 1.0
    assert time_of_day_modifier(120, 10).value < 1.0
    assert time_of_day_modifier(120, 120).value == 1.0


def test_correlation_assumes_dependence_not_independence():
    """The default correlation between two index-derivative positions is 0.7,
    not 0. Independence is the wrong prior for a book that is by construction
    all on one underlying."""
    alone = correlation_modifier([], "NIFTY24000CE")
    crowded = correlation_modifier(["A", "B", "C"], "NIFTY24000CE")
    assert alone.value == 1.0
    assert crowded.value < 0.4


def test_explicit_correlations_override_the_default():
    independent = correlation_modifier(["GOLD"], "NIFTY24000CE", {("NIFTY24000CE", "GOLD"): 0.0})
    assert independent.value == 1.0


def test_every_modifier_is_named_and_explained(limits):
    """A size that came out small has to say why. A single combined number is
    unarguable, which in a sizing path is a defect."""
    k = edge()
    decision = size_position(
        k,
        capital=CAPITAL,
        lot_size=LOT,
        limits=limits,
        modifiers=standard_modifiers(k, limits=limits),
    )
    assert len(decision.modifiers) == 5
    assert all(m.name and m.reason for m in decision.modifiers)
    text = decision.render()
    for m in decision.modifiers:
        assert m.name in text


def test_modifiers_compound_multiplicatively(limits):
    k = edge()
    mods = [Modifier("a", 0.5), Modifier("b", 0.5)]
    decision = size_position(k, capital=CAPITAL, lot_size=LOT, limits=limits, modifiers=mods)
    assert decision.combined_modifier == pytest.approx(0.25)


# --------------------------------------------------------------------- vetoes
def test_the_position_limit_is_a_veto_not_a_modifier(limits):
    """Some limits reduce a size. This one refuses it, and the difference has to
    survive a modifier stack that would otherwise have allowed a lot."""
    decision = size_position(
        edge(),
        capital=CAPITAL,
        lot_size=LOT,
        limits=limits,
        open_positions=limits.max_concurrent_positions,
    )
    assert decision.rejected
    assert "concurrent positions" in decision.cap_reason


def test_no_edge_means_no_position(limits):
    flat = KellyInput(p_win=0.5, win_amount=100.0, loss_amount=100.0, p_lower=0.5)
    decision = size_position(flat, capital=CAPITAL, lot_size=LOT, limits=limits)
    assert decision.rejected
    assert decision.raw_kelly == 0.0
    assert any("no Kelly edge" in n for n in decision.notes)


def test_zero_capital_is_refused(limits):
    with pytest.raises(ValueError, match="capital must be positive"):
        size_position(edge(), capital=0.0, lot_size=LOT, limits=limits)


# ----------------------------------------------------------------- book risk
def test_a_book_on_one_underlying_is_barely_diversified():
    """Five positions each risking 1.5%, all long the same index through
    different strikes, is not five 1.5% bets."""
    book = BookRisk(positions={f"NIFTY{s}CE": 7500.0 for s in (23800, 24000, 24200, 24400)})
    assert book.naive_total == 30_000.0
    assert book.diversification_benefit > 0.8
    assert book.correlated_total < book.naive_total


def test_perfect_correlation_gives_no_benefit_at_all():
    book = BookRisk(positions={"A": 1000.0, "B": 1000.0}, default_correlation=1.0)
    assert book.correlated_total == pytest.approx(book.naive_total)
    assert book.diversification_benefit == pytest.approx(1.0)


def test_genuine_independence_shows_a_real_benefit():
    book = BookRisk(positions={"A": 1000.0, "B": 1000.0}, default_correlation=0.0)
    assert book.correlated_total < book.naive_total * 0.75


def test_an_empty_book_risks_nothing():
    book = BookRisk(positions={})
    assert book.naive_total == 0.0
    assert book.correlated_total == 0.0
    assert book.diversification_benefit == 0.0


def test_a_single_position_is_its_own_risk():
    book = BookRisk(positions={"A": 5000.0})
    assert book.correlated_total == pytest.approx(5000.0)
