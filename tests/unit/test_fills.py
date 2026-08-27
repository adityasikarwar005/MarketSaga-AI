"""M81 - the fill model, which exists to be pessimistic on purpose."""

from __future__ import annotations

import pytest

from libs.ms_sim.fills import (
    BarOutcome,
    FillModel,
    Order,
    OrderType,
    Quote,
    Side,
    check_exit,
    worst_case_sequence,
)


def quote(bid: float = 99.0, ask: float = 101.0, bid_size: int = 100, ask_size: int = 100) -> Quote:
    return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


# --------------------------------------------------------------- the tie rule
def test_conservative_tie():
    """Both levels inside one bar, and the bar does not say which came first.
    Assuming the target flatters every result and turns the most volatile
    sessions - the ones where both are hit - into the most profitable ones."""
    check = check_exit(is_long=True, high=110.0, low=90.0, target=105.0, stop=95.0)

    assert check.outcome is BarOutcome.BOTH_ASSUMED_STOP
    assert check.price == 95.0
    assert "can only be better" in check.reason


def test_conservative_tie_for_a_short_too():
    check = check_exit(is_long=False, high=110.0, low=90.0, target=95.0, stop=105.0)
    assert check.outcome is BarOutcome.BOTH_ASSUMED_STOP
    assert check.price == 105.0


def test_an_unambiguous_target_is_a_target():
    """The rule must not be so pessimistic that it never reports a win."""
    check = check_exit(is_long=True, high=110.0, low=99.0, target=105.0, stop=95.0)
    assert check.outcome is BarOutcome.TARGET
    assert check.price == 105.0


def test_an_unambiguous_stop_is_a_stop():
    check = check_exit(is_long=True, high=101.0, low=90.0, target=105.0, stop=95.0)
    assert check.outcome is BarOutcome.STOP


def test_a_bar_touching_neither_exits_nothing():
    check = check_exit(is_long=True, high=101.0, low=99.0, target=105.0, stop=95.0)
    assert check.outcome is BarOutcome.NONE
    assert not check.exited
    assert check.price is None


def test_ties_are_counted_and_reportable():
    """If most of a backtest's exits came from ties, the result is largely a
    statement about the tie convention rather than about the strategy."""
    checks = [
        check_exit(is_long=True, high=110.0, low=90.0, target=105.0, stop=95.0),
        check_exit(is_long=True, high=110.0, low=99.0, target=105.0, stop=95.0),
        check_exit(is_long=True, high=110.0, low=90.0, target=105.0, stop=95.0),
    ]
    assert worst_case_sequence(checks) == 2


def test_a_touch_exactly_on_the_level_counts():
    """>= and <=, not > and <. A stop that only triggers when price goes past it
    is a stop that does not trigger on the bar that gapped straight to it."""
    assert check_exit(is_long=True, high=105.0, low=99.0, target=105.0, stop=95.0).exited
    assert check_exit(is_long=True, high=101.0, low=95.0, target=105.0, stop=95.0).exited


# ------------------------------------------------------------------- the book
def test_a_marketable_order_pays_the_far_touch_never_the_mid():
    """Filling at the mid claims half the spread as free profit on every leg.
    On a four-order option spread that is most of the edge."""
    q = quote(bid=99.0, ask=101.0)
    model = FillModel()

    buy = model.fill_market(Order(Side.BUY, 10), q)
    sell = model.fill_market(Order(Side.SELL, 10), q)

    assert buy.average_price == 101.0
    assert sell.average_price == 99.0
    assert buy.average_price != q.mid and sell.average_price != q.mid


def test_slippage_scales_with_spread():
    """A wide market punishes size harder than a tight one - which is why
    liquidity is a feature in the sizing layer."""
    model = FillModel()
    tight = model.impact(Order(Side.BUY, 500), quote(bid=99.5, ask=100.5, ask_size=100))
    wide = model.impact(Order(Side.BUY, 500), quote(bid=95.0, ask=105.0, ask_size=100))
    assert wide > tight * 4


def test_impact_is_zero_inside_the_displayed_size():
    """Wanting less than is showing costs nothing extra. An impact model that
    charges for every order overstates the cost of small trades, which is most
    of them."""
    model = FillModel()
    assert model.impact(Order(Side.BUY, 50), quote(ask_size=100)) == 0.0
    assert model.impact(Order(Side.BUY, 100), quote(ask_size=100)) == 0.0
    assert model.impact(Order(Side.BUY, 400), quote(ask_size=100)) > 0.0


def test_impact_grows_with_size():
    model = FillModel()
    sizes = [model.impact(Order(Side.BUY, n), quote(ask_size=100)) for n in (200, 400, 800, 1600)]
    assert sizes == sorted(sizes)


def test_an_empty_book_is_expensive_not_free():
    """Nothing displayed is not the same as infinite liquidity, and a model that
    treats it as free fills every order at the touch on the quietest strikes."""
    model = FillModel()
    assert model.impact(Order(Side.BUY, 10), quote(ask_size=0)) > 0.0


def test_partial_fill():
    """A model that silently returns a smaller fill lets a backtest trade a
    different size than the one it sized, and the difference never appears."""
    model = FillModel(allow_partial=True)
    result = model.fill_market(Order(Side.BUY, 250), quote(ask_size=100))

    assert result.filled_qty == 250
    assert len(result.fills) == 2
    assert result.fills[0].qty == 100 and result.fills[0].price == 101.0
    assert result.fills[1].qty == 150 and result.fills[1].price > 101.0
    assert result.average_price is not None and result.average_price > 101.0
    assert any("remainder paid impact" in n for n in result.notes)


def test_an_uncalibrated_model_says_so():
    result = FillModel().fill_market(Order(Side.BUY, 10), quote())
    assert any("UNCALIBRATED" in n for n in result.notes)

    calibrated = FillModel(calibrated=True).fill_market(Order(Side.BUY, 10), quote())
    assert not any("UNCALIBRATED" in n for n in calibrated.notes)


# ------------------------------------------------------------------- resting
def test_a_resting_limit_does_not_fill_on_a_touch():
    """Touching only fills an order that was at the front of the queue.
    Assuming otherwise assumes you were first every time, which nobody is."""
    model = FillModel()
    order = Order(Side.BUY, 10, OrderType.LIMIT, limit_price=98.0)
    result = model.fill_limit(order, quote(bid=99.0, ask=101.0), traded_through=False)

    assert result.unfilled == 10
    assert not result.fills
    assert any("Touching is not filling" in n for n in result.notes)


def test_a_resting_limit_fills_when_the_market_trades_through_it():
    model = FillModel(queue_ahead=0.5)
    order = Order(Side.BUY, 10, OrderType.LIMIT, limit_price=98.0)
    result = model.fill_limit(
        order, quote(bid=99.0, ask=101.0), traded_through=True, traded_volume=100
    )
    assert result.filled_qty == 10
    assert result.fills[0].price == 98.0


def test_the_queue_discount_can_absorb_the_whole_trade():
    """queue_ahead=1.0 assumes everything that traded was ahead of us, which is
    the assumption reality cannot beat."""
    model = FillModel(queue_ahead=1.0)
    order = Order(Side.BUY, 10, OrderType.LIMIT, limit_price=98.0)
    result = model.fill_limit(
        order, quote(bid=99.0, ask=101.0), traded_through=True, traded_volume=100
    )
    assert result.unfilled == 10
    assert any("queue ahead absorbed" in n for n in result.notes)


def test_a_limit_that_is_marketable_on_arrival_crosses():
    """It never rested, so it never queued - but it must not fill worse than the
    limit the caller set."""
    model = FillModel()
    order = Order(Side.BUY, 10, OrderType.LIMIT, limit_price=105.0)
    result = model.fill_limit(order, quote(bid=99.0, ask=101.0), traded_through=False)

    assert result.filled_qty == 10
    assert result.average_price is not None and result.average_price <= 105.0
    assert any("marketable on arrival" in n for n in result.notes)


def test_a_marketable_limit_is_never_filled_worse_than_its_limit():
    model = FillModel()
    order = Order(Side.BUY, 400, OrderType.LIMIT, limit_price=101.2)
    result = model.fill_limit(order, quote(bid=99.0, ask=101.0, ask_size=50), traded_through=False)
    assert all(f.price <= 101.2 for f in result.fills)


# ------------------------------------------------------------------- refusals
def test_a_crossed_book_is_refused():
    """Filling against a crossed quote invents money, and a crossed book is a
    data problem rather than an opportunity."""
    with pytest.raises(ValueError, match="crossed quote"):
        Quote(bid=101.0, ask=99.0)


def test_direction_is_the_side_not_a_negative_quantity():
    with pytest.raises(ValueError, match="qty must be positive"):
        Order(Side.BUY, -5)


def test_a_limit_order_needs_a_limit():
    with pytest.raises(ValueError, match="needs a limit price"):
        Order(Side.BUY, 10, OrderType.LIMIT)


def test_a_market_order_with_a_limit_is_two_orders():
    with pytest.raises(ValueError, match="two different orders"):
        Order(Side.BUY, 10, OrderType.MARKET, limit_price=100.0)
