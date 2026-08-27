"""M80 - the paper broker, and the interface it shares with the live one."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from libs.ms_broker.paper import PaperBroker
from libs.ms_broker.protocol import Account, Broker, OrderStatus, Position
from libs.ms_core.config import CostSchedule
from libs.ms_sim.fills import FillModel, Order, OrderType, Quote, Side

TS = datetime(2026, 3, 4, 6, 30, tzinfo=UTC)
SYM = "NIFTY24000CE"


@pytest.fixture
def schedule() -> CostSchedule:
    return CostSchedule(
        effective_from=datetime(2020, 1, 1).date(),
        brokerage_per_order=20.0,
        exchange_txn_pct_of_premium=0.00053,
        stt_pct_sell_premium=0.000625,
        sebi_fee_pct_of_turnover=0.000001,
        stamp_duty_pct_buy=0.00003,
        gst_pct_on_charges=0.18,
        source="test",
    )


@pytest.fixture
def broker(schedule) -> PaperBroker:
    b = PaperBroker(
        starting_cash=1_000_000.0,
        schedule=schedule,
        fill_model=FillModel(calibrated=True),
        lot_size=1,
    )
    b.set_quote(SYM, Quote(bid=99.0, ask=101.0, bid_size=1000, ask_size=1000))
    return b


# --------------------------------------------------------------- the interface
def test_same_interface_as_live():
    """A paper broker that diverges from the live one is a rehearsal for a
    different performance. This is the test that keeps them honest - a
    convenient extra method on paper is one the strategy learns to depend on and
    that production does not have."""
    required = {
        name
        for name, member in inspect.getmembers(Broker)
        if getattr(member, "__isabstractmethod__", False)
    }
    assert required, "the Broker protocol declares nothing - it cannot constrain anything"

    for name in required:
        assert hasattr(PaperBroker, name), f"PaperBroker is missing {name}"

    # And it must not be abstract any more - every method really implemented.
    assert not getattr(PaperBroker, "__abstractmethods__", set())


def test_the_paper_broker_is_a_broker(broker):
    assert isinstance(broker, Broker)


def test_is_live_is_a_property_of_the_class_not_configuration(broker):
    """A mode that can be derived from configuration is one that can be derived
    wrongly. PAPER and LIVE differ by whether real money moves."""
    assert broker.is_live is False
    assert broker.name == "paper"


def test_placing_returns_an_acknowledgement_not_a_fill(broker):
    """Real brokers accept an order and tell you about the fill later. An
    interface that returns a fill synchronously teaches a habit that breaks in
    production."""
    ack = broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    assert ack.order_id
    assert ack.status in set(OrderStatus)
    assert not hasattr(ack, "fill_price")


# ------------------------------------------------------------- position maths
def test_position_math(broker):
    """Adding averages, reducing realises against the average, and crossing zero
    does both - with the new side starting at the crossing price alone."""
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    pos = broker.account().positions[SYM]
    assert pos.qty == 10
    assert pos.avg_price == 101.0

    # Add at a different price: the average moves, nothing is realised.
    broker.set_quote(SYM, Quote(bid=109.0, ask=111.0, bid_size=1000, ask_size=1000))
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    pos = broker.account().positions[SYM]
    assert pos.qty == 20
    assert pos.avg_price == pytest.approx((10 * 101.0 + 10 * 111.0) / 20)
    assert pos.realised_pnl == 0.0

    # Reduce: realises against the average, leaves the average alone.
    avg_before = pos.avg_price
    broker.place(SYM, Order(Side.SELL, 5), ts=TS)
    pos = broker.account().positions[SYM]
    assert pos.qty == 15
    assert pos.avg_price == pytest.approx(avg_before)
    assert pos.realised_pnl == pytest.approx(5 * (109.0 - avg_before))


def test_crossing_zero_starts_the_new_side_at_the_crossing_price(broker):
    """Blending the two produces an average that is a mixture of a long and a
    short, which is not the price of anything. It shows up as P&L drifting away
    from the truth, invisible until a reconciliation."""
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    broker.set_quote(SYM, Quote(bid=119.0, ask=121.0, bid_size=1000, ask_size=1000))
    broker.place(SYM, Order(Side.SELL, 25), ts=TS)

    pos = broker.account().positions[SYM]
    assert pos.qty == -15
    assert pos.avg_price == 119.0, "the new short must not be blended with the closed long"
    assert pos.realised_pnl == pytest.approx(10 * (119.0 - 101.0))


def test_closing_to_flat_clears_the_average(broker):
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    broker.place(SYM, Order(Side.SELL, 10), ts=TS)
    pos = broker.account().positions[SYM]
    assert pos.is_flat
    assert pos.avg_price == 0.0
    assert broker.positions() == []


def test_a_short_makes_money_when_price_falls(broker):
    broker.place(SYM, Order(Side.SELL, 10), ts=TS)
    broker.set_quote(SYM, Quote(bid=89.0, ask=91.0, bid_size=1000, ask_size=1000))
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)

    pos = broker.account().positions[SYM]
    assert pos.is_flat
    assert pos.realised_pnl == pytest.approx(10 * (99.0 - 91.0))


def test_unrealised_is_computed_not_stored(broker):
    """An unrealised P&L cached on the position is wrong the moment the market
    moves, and it will be read long after it went stale."""
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    pos = broker.account().positions[SYM]

    assert pos.unrealised(mark=111.0) == pytest.approx(10 * (111.0 - 101.0))
    assert pos.unrealised(mark=91.0) == pytest.approx(10 * (91.0 - 101.0))
    assert not hasattr(pos, "unrealised_pnl")


# --------------------------------------------------------------------- costs
def test_every_fill_pays_costs(broker):
    """A paper account that ignores brokerage, STT and GST reports a gross P&L
    wearing a net P&L's clothes."""
    before = broker.account().cash
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    account = broker.account()

    assert account.costs_paid > 0
    spent = before - account.cash
    assert spent > 10 * 101.0, "cash moved by the premium alone - costs were not charged"
    assert spent == pytest.approx(10 * 101.0 + account.costs_paid)


def test_a_round_trip_at_the_same_price_loses_money(broker):
    """The most important sanity check in any paper broker: buying and selling
    at an unchanged price must lose, because crossing the spread twice and
    paying four sets of charges is what trading costs."""
    start = broker.account().cash
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    broker.place(SYM, Order(Side.SELL, 10), ts=TS)

    assert broker.account().cash < start
    assert broker.account().positions[SYM].is_flat


# ------------------------------------------------------------------ equity
def test_equity_refuses_to_mark_a_position_it_has_no_price_for(broker):
    """Marking at zero because the quote is missing turns a data gap into a loss
    that looks real."""
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    with pytest.raises(KeyError, match="no mark"):
        broker.account().equity({})


def test_equity_counts_open_positions(broker):
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    account = broker.account()
    flat = account.equity({SYM: 101.0})
    up = account.equity({SYM: 111.0})
    assert up == pytest.approx(flat + 10 * 10.0)


def test_marks_come_from_the_mid_not_the_touch(broker):
    """Marking at the liquidation price conflates valuation with exit cost, and
    the exit cost is charged when the exit happens - charging it twice
    understates every open position."""
    marks = broker.mark_to_market()
    assert marks[SYM] == pytest.approx(100.0)


# ------------------------------------------------------------------- orders
def test_an_order_without_a_quote_is_rejected(schedule):
    """A paper fill invented without a book is the exact fiction this module
    exists to avoid."""
    b = PaperBroker(starting_cash=100_000.0, schedule=schedule)
    ack = b.place("UNKNOWN", Order(Side.BUY, 1), ts=TS)
    assert ack.status is OrderStatus.REJECTED
    assert not ack.accepted
    assert "no quote" in ack.reason


def test_a_resting_limit_stays_open_until_the_market_trades_through(broker):
    ack = broker.place(SYM, Order(Side.BUY, 10, OrderType.LIMIT, limit_price=95.0), ts=TS)
    assert ack.status is OrderStatus.ACCEPTED
    assert ack.order_id in broker.open_orders()
    assert not broker.positions()

    filled = broker.work_order(ack.order_id, traded_through=False, traded_volume=100, ts=TS)
    assert filled is None
    assert ack.order_id in broker.open_orders()


def test_a_resting_limit_fills_on_a_trade_through(schedule):
    b = PaperBroker(
        starting_cash=1_000_000.0,
        schedule=schedule,
        fill_model=FillModel(queue_ahead=0.0, calibrated=True),
    )
    b.set_quote(SYM, Quote(bid=99.0, ask=101.0, bid_size=1000, ask_size=1000))
    ack = b.place(SYM, Order(Side.BUY, 10, OrderType.LIMIT, limit_price=95.0), ts=TS)

    done = b.work_order(ack.order_id, traded_through=True, traded_volume=500, ts=TS)
    assert done is not None and done.status is OrderStatus.FILLED
    assert b.account().positions[SYM].qty == 10
    assert b.account().positions[SYM].avg_price == 95.0
    assert ack.order_id not in b.open_orders()


def test_cancel_reports_acceptance_not_certainty(broker):
    """An order can fill in the microsecond between deciding to cancel and the
    cancel arriving. True would be a promise this cannot keep."""
    ack = broker.place(SYM, Order(Side.BUY, 10, OrderType.LIMIT, limit_price=95.0), ts=TS)
    assert broker.cancel(ack.order_id, ts=TS) is True
    assert broker.cancel(ack.order_id, ts=TS) is False, "cancelling twice is not two cancels"
    assert ack.order_id not in broker.open_orders()


def test_a_partial_fill_is_reported_as_partial(schedule):
    b = PaperBroker(
        starting_cash=1_000_000.0,
        schedule=schedule,
        fill_model=FillModel(queue_ahead=0.5, calibrated=True),
    )
    b.set_quote(SYM, Quote(bid=99.0, ask=101.0, bid_size=1000, ask_size=1000))
    ack = b.place(SYM, Order(Side.BUY, 100, OrderType.LIMIT, limit_price=95.0), ts=TS)

    done = b.work_order(ack.order_id, traded_through=True, traded_volume=100, ts=TS)
    assert done is not None and done.status is OrderStatus.PARTIAL
    assert b.account().positions[SYM].qty == 50
    assert ack.order_id in b.open_orders(), "the remainder is still working"


def test_positions_reports_what_the_broker_believes(broker):
    """Not what the system believes. Reconciling the two is a separate problem,
    and pretending they are the same is how a phantom position gets doubled."""
    broker.place(SYM, Order(Side.BUY, 10), ts=TS)
    positions = broker.positions()
    assert len(positions) == 1
    assert isinstance(positions[0], Position)
    assert isinstance(broker.account(), Account)
