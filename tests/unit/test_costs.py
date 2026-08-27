"""M74 cost model - what is charged, on what base, and the expiry decision.

Every test here corresponds to a way of getting option costs wrong that leaves
the pipeline running and the numbers plausible.
"""

from __future__ import annotations

from datetime import date

import pytest

from libs.ms_core.config import CostSchedule
from libs.ms_options.costs import (
    EXERCISE_STT_PCT,
    CostBreakdown,
    ExerciseBase,
    Fill,
    Leg,
    Side,
    SlippageModel,
    audit_cost_config,
    breakeven_pct_of_premium,
    calibrate_slippage,
    edge_after_costs,
    expiry_stt,
    leg_cost,
    round_trip_cost,
    should_square_off,
)

LOT = 75


def schedule(effective: date = date(2020, 1, 1), source: str = "verified 2026 circular"):
    return CostSchedule(
        effective_from=effective,
        brokerage_per_order=20.0,
        exchange_txn_pct_of_premium=0.00053,
        stt_pct_sell_premium=0.000625,
        sebi_fee_pct_of_turnover=0.000001,
        stamp_duty_pct_buy=0.00003,
        gst_pct_on_charges=0.18,
        source=source,
    )


def no_slippage() -> SlippageModel:
    return SlippageModel(half_spread_points=0.0, calibrated=True, samples=999)


# ------------------------------------------------------- what is charged on what
def test_exchange_fee_is_on_premium_not_notional():
    """The most common and largest error in this arithmetic. Premium is a small
    fraction of notional, so charging on notional overstates costs by roughly
    the inverse of moneyness - an order of magnitude, not a rounding error."""
    sched = schedule()
    leg = Leg(Side.BUY, 180.0, 1, LOT)
    cost = leg_cost(leg, sched, slippage=no_slippage())

    premium_turnover = 180.0 * LOT
    assert cost.exchange == pytest.approx(premium_turnover * sched.exchange_txn_pct_of_premium)
    # Not on a notional of, say, 24000 * 75.
    assert cost.exchange < 24000 * LOT * sched.exchange_txn_pct_of_premium / 100


def test_stt_is_charged_on_the_sell_side_only():
    sched = schedule()
    buy = leg_cost(Leg(Side.BUY, 180.0, 1, LOT), sched, slippage=no_slippage())
    sell = leg_cost(Leg(Side.SELL, 180.0, 1, LOT), sched, slippage=no_slippage())
    assert buy.stt == 0.0
    assert sell.stt == pytest.approx(180.0 * LOT * sched.stt_pct_sell_premium)


def test_stamp_duty_is_charged_on_the_buy_side_only():
    sched = schedule()
    assert leg_cost(Leg(Side.SELL, 180.0, 1, LOT), sched, slippage=no_slippage()).stamp_duty == 0.0
    assert leg_cost(Leg(Side.BUY, 180.0, 1, LOT), sched, slippage=no_slippage()).stamp_duty > 0


def test_gst_is_on_the_charges_not_on_the_turnover():
    """GST is an ~18% surcharge on the fee stack, not a tax on the trade. And
    STT and stamp duty are themselves taxes - they are not taxed again."""
    sched = schedule()
    cost = leg_cost(Leg(Side.SELL, 180.0, 1, LOT), sched, slippage=no_slippage())
    expected = (cost.brokerage + cost.exchange + cost.sebi) * sched.gst_pct_on_charges
    assert cost.gst == pytest.approx(expected)
    assert cost.gst < cost.premium_turnover * 0.01


def test_brokerage_is_per_order_not_per_lot():
    """A flat per-order charge means size does not change it. Scaling it by lots
    would make large trades look disproportionately expensive and small ones
    look free."""
    sched = schedule()
    one = leg_cost(Leg(Side.BUY, 180.0, 1, LOT), sched, slippage=no_slippage())
    ten = leg_cost(Leg(Side.BUY, 180.0, 10, LOT), sched, slippage=no_slippage())
    assert one.brokerage == ten.brokerage == sched.brokerage_per_order


# --------------------------------------------------------------- point in time
def test_pit_rates_price_an_old_trade_with_old_rates():
    """A backtest that applies today's statutory rates to a two-year-old trade
    is not a backtest. The config lookup is the mechanism; this is the check."""
    from libs.ms_core.config import CostsConfig

    old = schedule(date(2020, 1, 1))
    new = CostSchedule(
        effective_from=date(2025, 1, 1),
        brokerage_per_order=20.0,
        exchange_txn_pct_of_premium=0.00053,
        stt_pct_sell_premium=0.001,  # the rate went up
        sebi_fee_pct_of_turnover=0.000001,
        stamp_duty_pct_buy=0.00003,
        gst_pct_on_charges=0.18,
        source="verified",
    )
    costs = CostsConfig(schedules=[old, new])

    assert costs.on(date(2024, 6, 1)).stt_pct_sell_premium == 0.000625
    assert costs.on(date(2026, 6, 1)).stt_pct_sell_premium == 0.001

    leg = Leg(Side.SELL, 180.0, 1, LOT)
    then = leg_cost(leg, costs.on(date(2024, 6, 1)), slippage=no_slippage())
    now = leg_cost(leg, costs.on(date(2026, 6, 1)), slippage=no_slippage())
    assert now.stt > then.stt


def test_rates_before_the_first_schedule_are_refused():
    from libs.ms_core.config import CostsConfig

    costs = CostsConfig(schedules=[schedule(date(2020, 1, 1))])
    with pytest.raises(LookupError):
        costs.on(date(2019, 12, 31))


# ------------------------------------------------------------------ round trip
def test_four_orders_for_spread():
    """A two-leg spread is four orders, not two. Costing only the entry makes a
    strategy look viable at twice its real edge - and it is an easy mistake,
    because the exit legs do not exist yet when the trade is evaluated."""
    sched = schedule()
    legs = [Leg(Side.BUY, 180.0, 1, LOT), Leg(Side.SELL, 95.0, 1, LOT)]
    cost = round_trip_cost(legs, sched, slippage=no_slippage())

    assert cost.orders == 4
    assert cost.brokerage == pytest.approx(4 * sched.brokerage_per_order)


def test_a_round_trip_pays_stt_on_both_closing_sells():
    """Every leg is sold once - the long on exit, the short on entry."""
    sched = schedule()
    legs = [Leg(Side.BUY, 180.0, 1, LOT), Leg(Side.SELL, 95.0, 1, LOT)]
    cost = round_trip_cost(legs, sched, slippage=no_slippage())
    expected = (180.0 + 95.0) * LOT * sched.stt_pct_sell_premium
    assert cost.stt == pytest.approx(expected)


def test_breakeven_is_expressed_against_net_premium():
    """The number that belongs in front of a human. "This needs a 4% move in the
    premium just to get back to flat" stops bad trades; a rupee figure does not."""
    sched = schedule()
    legs = [Leg(Side.BUY, 180.0, 1, LOT), Leg(Side.SELL, 95.0, 1, LOT)]
    pct = breakeven_pct_of_premium(legs, sched, slippage=no_slippage())
    net_premium = (180.0 - 95.0) * LOT
    cost = round_trip_cost(legs, sched, slippage=no_slippage())
    assert pct == pytest.approx(cost.total / net_premium)
    assert 0 < pct < 1


def test_breakeven_of_a_zero_cost_structure_is_infinite_not_zero():
    """A structure with no net premium cannot recover its costs as a percentage
    of nothing. Reporting 0% would read as "free"."""
    sched = schedule()
    legs = [Leg(Side.BUY, 100.0, 1, LOT), Leg(Side.SELL, 100.0, 1, LOT)]
    assert breakeven_pct_of_premium(legs, sched, slippage=no_slippage()) == float("inf")


def test_breakdowns_add_component_by_component():
    a = CostBreakdown(brokerage=20.0, stt=5.0, orders=1, premium_turnover=100.0)
    b = CostBreakdown(brokerage=20.0, stt=3.0, orders=1, premium_turnover=50.0)
    total = a + b
    assert total.brokerage == 40.0 and total.stt == 8.0
    assert total.orders == 2 and total.premium_turnover == 150.0


def test_an_uncalibrated_component_makes_the_whole_total_uncalibrated():
    """One guessed component contaminates the sum. A total that is exact to the
    paisa in five terms and a guess in the sixth is a guess."""
    exact = CostBreakdown(slippage_calibrated=True)
    guess = CostBreakdown(slippage_calibrated=False)
    assert not (exact + guess).slippage_calibrated


# ------------------------------------------------------------------- the legs
def test_lots_are_whole_and_positive():
    with pytest.raises(ValueError, match="lots must be positive"):
        Leg(Side.BUY, 100.0, 0, LOT)
    with pytest.raises(ValueError, match="lots must be positive"):
        Leg(Side.BUY, 100.0, -1, LOT)


def test_direction_is_the_side_not_a_negative_quantity():
    """Expressing a short as negative lots would make every turnover calculation
    below silently negative."""
    short = Leg(Side.SELL, 100.0, 2, LOT)
    assert short.units == 150 and short.premium_turnover > 0


def test_a_negative_premium_is_refused():
    with pytest.raises(ValueError, match="premium"):
        Leg(Side.BUY, -1.0, 1, LOT)


# --------------------------------------------------------------- the STT trap
def test_itm_expiry_flagged_under_the_settlement_value_base():
    """The regime the folklore describes: the charge is on the whole contract
    value, which can exceed what the option is worth."""
    sched = schedule()
    long_call = Leg(Side.BUY, 5.0, 1, LOT, "23000CE")
    decision = should_square_off(
        long_call,
        intrinsic_per_unit=520.0,
        schedule=sched,
        slippage=no_slippage(),
        base=ExerciseBase.SETTLEMENT_VALUE,
        settlement_price=23520.0,
    )
    assert decision.square_off
    assert decision.expiry_stt > 10 * decision.square_off_cost
    assert "settlement_value" in decision.reason


def test_on_the_intrinsic_base_the_answer_can_go_the_other_way():
    """And the module must say so rather than assert the folklore. Squaring off
    pays brokerage, slippage and its own sell-side STT."""
    sched = schedule()
    long_call = Leg(Side.BUY, 5.0, 1, LOT, "23000CE")
    decision = should_square_off(
        long_call, intrinsic_per_unit=520.0, schedule=sched, slippage=no_slippage()
    )
    assert decision.base is ExerciseBase.INTRINSIC
    assert decision.expiry_stt == pytest.approx(520.0 * LOT * EXERCISE_STT_PCT)


def test_the_two_bases_differ_by_the_inverse_of_moneyness():
    """Which is why the base can never be implicit."""
    intrinsic = expiry_stt(520.0, LOT)
    settlement = expiry_stt(
        520.0, LOT, base=ExerciseBase.SETTLEMENT_VALUE, settlement_price=23520.0
    )
    assert settlement / intrinsic == pytest.approx(23520.0 / 520.0)


def test_the_settlement_base_refuses_to_guess_a_price():
    """Falling back to intrinsic would report a charge fifty times too small for
    the exact case the caller was worried about."""
    with pytest.raises(ValueError, match="settlement_price is required"):
        expiry_stt(520.0, LOT, base=ExerciseBase.SETTLEMENT_VALUE)


def test_an_out_of_the_money_option_costs_nothing_to_let_lapse():
    sched = schedule()
    decision = should_square_off(Leg(Side.BUY, 5.0, 1, LOT), -50.0, sched)
    assert not decision.square_off
    assert decision.expiry_stt == 0.0
    assert "lapses worthless" in decision.reason


def test_a_short_position_does_not_pay_the_exercise_charge():
    """Assignment puts the exercise STT on the buyer. Getting this backwards
    would make the model refuse to hold any short option to expiry, which is a
    normal and often intended thing to do."""
    sched = schedule()
    decision = should_square_off(Leg(Side.SELL, 5.0, 1, LOT), 520.0, sched)
    assert not decision.square_off
    assert decision.expiry_stt == 0.0
    assert "writer" in decision.reason


# ------------------------------------------------------------------- slippage
def test_slippage_is_uncalibrated_until_it_is_calibrated():
    """A backtest that silently applies a made-up slippage number produces net
    P&L with false precision - exact statutory costs summed with a guess."""
    model = SlippageModel()
    assert not model.calibrated
    cost = leg_cost(Leg(Side.BUY, 180.0, 1, LOT), schedule())
    assert not cost.slippage_calibrated
    assert "UNCALIBRATED" in cost.render()


def test_calibration_needs_enough_fills():
    fills = [Fill(Side.BUY, 100.0, 100.5, 1, date(2026, 8, 1)) for _ in range(5)]
    model = calibrate_slippage(fills)
    assert not model.calibrated
    assert model.samples == 5


def test_calibration_uses_the_median_not_the_mean():
    """One fill during a halt can be worth hundreds of normal fills. A mean
    dominated by a single unrepeatable event forecasts nothing."""
    fills = [Fill(Side.BUY, 100.0, 100.5, 1, date(2026, 8, 1)) for _ in range(40)]
    fills.append(Fill(Side.BUY, 100.0, 400.0, 1, date(2026, 8, 2)))  # a halt
    model = calibrate_slippage(fills)
    assert model.calibrated
    assert model.half_spread_points == pytest.approx(0.5, abs=0.05)


def test_adverse_slippage_is_signed_by_side():
    """Paying more when buying and receiving less when selling are the same
    event. Averaging them unsigned reports zero slippage for a book losing on
    both."""
    buy = Fill(Side.BUY, 100.0, 100.5, 1, date(2026, 8, 1))
    sell = Fill(Side.SELL, 100.0, 99.5, 1, date(2026, 8, 1))
    assert buy.adverse_points == pytest.approx(0.5)
    assert sell.adverse_points == pytest.approx(0.5)


def test_impact_is_never_negative():
    """A negative estimated impact means larger orders filled better, which is a
    small-sample artefact rather than a discovery about the market."""
    fills = [
        Fill(Side.BUY, 100.0, 100.0 + (0.6 - 0.05 * lots), lots, date(2026, 8, 1))
        for lots in range(1, 41)
    ]
    model = calibrate_slippage(fills)
    assert model.impact_per_lot >= 0.0


def test_impact_scales_the_cost_with_size():
    model = SlippageModel(half_spread_points=0.5, impact_per_lot=0.1, calibrated=True)
    one = model.cost(Leg(Side.BUY, 180.0, 1, LOT))
    five = model.cost(Leg(Side.BUY, 180.0, 5, LOT))
    assert five > 5 * one


# ---------------------------------------------------------- does the edge live
def test_an_edge_smaller_than_costs_does_not_survive():
    sched = schedule()
    legs = [Leg(Side.BUY, 180.0, 1, LOT), Leg(Side.SELL, 95.0, 1, LOT)]
    thin = edge_after_costs(50.0, legs, sched, slippage=no_slippage())
    fat = edge_after_costs(5000.0, legs, sched, slippage=no_slippage())
    assert not thin.survives and thin.cost_to_edge > 1
    assert fat.survives and fat.cost_to_edge < 1
    assert "DOES NOT SURVIVE" in thin.render()


def test_a_zero_gross_edge_has_infinite_cost_ratio_not_zero():
    sched = schedule()
    legs = [Leg(Side.BUY, 180.0, 1, LOT)]
    result = edge_after_costs(0.0, legs, sched, slippage=no_slippage())
    assert result.cost_to_edge == float("inf")
    assert not result.survives


# ------------------------------------------------------------------ the audit
def test_placeholder_rates_are_refused_for_profitability_claims():
    """config/costs.yaml ships with plausible magnitudes marked TODO so the
    pipeline runs end to end. A backtest built on those is a rehearsal, and the
    difference has to be visible in the output rather than remembered from a
    comment in a YAML file."""
    audit = audit_cost_config(schedule(source="TODO: PLACEHOLDER - verify"))
    assert not audit.verified
    assert not audit.safe_for_profitability_claims
    assert "NOT safe" in audit.render()


def test_the_shipped_config_is_still_marked_unverified():
    """If this ever fails, someone filled in the real rates - update the gate
    ledger rather than the test."""
    from libs.ms_core import config as cfgmod

    cfg = cfgmod.load(warn=False)
    audit = audit_cost_config(cfg.costs.on(date(2026, 8, 27)))
    assert not audit.safe_for_profitability_claims


def test_verified_rates_still_need_calibrated_slippage():
    """Both halves, or the total is a guess wearing exact arithmetic."""
    audit = audit_cost_config(schedule(), SlippageModel())
    assert audit.verified
    assert not audit.safe_for_profitability_claims

    ok = audit_cost_config(schedule(), SlippageModel(calibrated=True, samples=500))
    assert ok.safe_for_profitability_claims


def test_thin_calibration_is_noted_even_when_it_passes():
    audit = audit_cost_config(schedule(), SlippageModel(calibrated=True, samples=40))
    assert audit.safe_for_profitability_claims
    assert any("only 40 fills" in n for n in audit.notes)
