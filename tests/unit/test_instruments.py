"""M10 - what a symbol meant on a given day.

The plan's trap for this module is one line: *lot sizes change over time. Store
them with validity intervals or every historical P&L is wrong.* The same
sentence applies to the symbols themselves, and more sharply — a stale lot size
scales a number, a reused symbol attaches it to the wrong contract entirely.

None of these failures raise. That is the whole difficulty: the wrong contract
is a real contract, its price is a real price, and the P&L that comes out is a
number somebody will act on.
"""

from __future__ import annotations

from datetime import date

import pytest

from libs.ms_core.config import Instrument, LotSizeEntry
from libs.ms_data.instruments import (
    AmbiguousSymbol,
    Contract,
    ContractKind,
    InstrumentError,
    InstrumentMaster,
    Right,
    days_to_expiry,
    expiries_between,
    is_monthly,
    monthly_expiry,
    weekly_expiries,
)

THU = 3  # Nifty weekly/monthly expiry weekday


def weekdays_only(day: date) -> bool:
    return day.weekday() < 5


def with_holidays(*holidays: date):
    """A trading-day test that also knows about a few closed weekdays."""
    closed = set(holidays)

    def test(day: date) -> bool:
        return day.weekday() < 5 and day not in closed

    return test


def option(
    iid: int,
    symbol: str,
    *,
    strike: float,
    right: Right = Right.CALL,
    expiry: date = date(2026, 8, 27),
    valid_from: date = date(2026, 7, 30),
    valid_to: date | None = None,
    token: int = 1000,
    lot: int = 75,
) -> Contract:
    return Contract(
        instrument_id=iid,
        tradingsymbol=symbol,
        exchange_token=token,
        underlying="NIFTY",
        kind=ContractKind.OPTION,
        expiry=expiry,
        valid_from=valid_from,
        valid_to=valid_to or expiry,
        lot_size=lot,
        tick_size=0.05,
        strike=strike,
        right=right,
    )


# --------------------------------------------------------------------------- #
# resolve_at_date - symbols get reused
# --------------------------------------------------------------------------- #
def test_resolve_at_date_distinguishes_two_contracts_with_one_symbol():
    """The trap the module exists for. The same tradingsymbol, two contracts,
    two different expiries — and nothing about resolving to the wrong one
    fails. It is a real contract with a real price, and the P&L it produces is
    a number somebody will act on."""
    august = option(1, "NIFTY24500CE", strike=24500, expiry=date(2026, 8, 27))
    december = option(
        2,
        "NIFTY24500CE",
        strike=24500,
        expiry=date(2026, 12, 31),
        valid_from=date(2026, 12, 1),
        token=2000,
    )
    master = InstrumentMaster().extend([august, december])

    assert master.resolve_at_date("NIFTY24500CE", date(2026, 8, 10)).instrument_id == 1
    assert master.resolve_at_date("NIFTY24500CE", date(2026, 12, 15)).instrument_id == 2


def test_resolve_at_date_refuses_a_date_the_symbol_did_not_cover():
    """Resolving to the nearest window would book the trade against a contract
    that had not been listed yet, and the trade would look ordinary."""
    master = InstrumentMaster().extend([option(1, "NIFTY24500CE", strike=24500)])

    with pytest.raises(InstrumentError, match="existed, but not on"):
        master.resolve_at_date("NIFTY24500CE", date(2026, 6, 1))


def test_resolve_at_date_names_the_windows_it_does_know():
    """An error that says only "not found" sends someone to the wrong place."""
    master = InstrumentMaster().extend([option(1, "NIFTY24500CE", strike=24500)])

    with pytest.raises(InstrumentError, match=r"2026-07-30\.\.2026-08-27"):
        master.resolve_at_date("NIFTY24500CE", date(2027, 1, 1))


def test_resolve_at_date_on_an_unknown_symbol_says_so_differently():
    master = InstrumentMaster()
    with pytest.raises(InstrumentError, match="not in the instrument master"):
        master.resolve_at_date("NIFTY99999CE", date(2026, 8, 10))


def test_two_contracts_cannot_share_a_symbol_on_the_same_day():
    """Overlapping windows make resolution a coin toss, so the master refuses
    the second one at insert time rather than at lookup time — by which point
    it is a bad instrument dump that has already been trusted."""
    master = InstrumentMaster().extend([option(1, "NIFTY24500CE", strike=24500)])

    with pytest.raises(InstrumentError, match="coin toss"):
        master.add(
            option(2, "NIFTY24500CE", strike=24500, valid_from=date(2026, 8, 20), token=2000)
        )


def test_a_dict_would_have_hidden_this():
    """The structural point. Keyed by symbol, the second contract overwrites the
    first: the master reports one entry, every earlier trade silently resolves
    to the later contract, and nothing anywhere fails."""
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500, expiry=date(2026, 8, 27)),
            option(
                2,
                "NIFTY24500CE",
                strike=24500,
                expiry=date(2026, 12, 31),
                valid_from=date(2026, 12, 1),
                token=2000,
            ),
        ]
    )
    assert len(master.history("NIFTY24500CE")) == 2
    assert master.reused_symbols() == {"NIFTY24500CE": 2}


def test_exchange_tokens_are_recycled_too():
    """Brokers reuse numeric tokens on a schedule nobody publishes, so a token
    resolves on one date and not another — which is why stored records must
    point at instrument_id."""
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500, token=5555),
            option(
                2,
                "NIFTY25000PE",
                strike=25000,
                right=Right.PUT,
                token=5555,
                expiry=date(2026, 12, 31),
                valid_from=date(2026, 12, 1),
            ),
        ]
    )
    assert master.resolve_token_at_date(5555, date(2026, 8, 10)).instrument_id == 1
    assert master.resolve_token_at_date(5555, date(2026, 12, 15)).instrument_id == 2


def test_two_contracts_cannot_hold_one_token_at_once():
    master = InstrumentMaster().extend([option(1, "NIFTY24500CE", strike=24500, token=5555)])
    with pytest.raises(InstrumentError, match="recycle tokens"):
        master.add(option(2, "NIFTY25000CE", strike=25000, token=5555))


def test_instrument_id_is_never_reused():
    """The one identifier a stored trade may point at. If it were reusable,
    nothing in the ledger would mean anything a year later."""
    master = InstrumentMaster().extend([option(1, "NIFTY24500CE", strike=24500)])
    with pytest.raises(InstrumentError, match="must never be reused"):
        master.add(option(1, "NIFTY25000CE", strike=25000, token=2000))


def test_by_id_is_stable_across_reuse():
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500),
            option(
                2,
                "NIFTY24500CE",
                strike=24500,
                expiry=date(2026, 12, 31),
                valid_from=date(2026, 12, 1),
                token=2000,
            ),
        ]
    )
    assert master.by_id(1).expiry == date(2026, 8, 27)
    assert master.by_id(2).expiry == date(2026, 12, 31)


def test_ambiguous_symbol_is_its_own_error_type():
    """So a caller can distinguish "reused and I gave no date" from "never
    existed" — they need different fixes."""
    assert issubclass(AmbiguousSymbol, InstrumentError)


# --------------------------------------------------------------------------- #
# lot_size_history
# --------------------------------------------------------------------------- #
def test_lot_size_history_values_a_trade_at_the_size_then_in_force():
    """The plan's stated trap. Using today's lot size to value a trade from
    eight months ago scales every historical P&L by the ratio, and the result
    is a plausible number in the right units."""
    inst = Instrument(
        id=1,
        symbol="NIFTY",
        name="Nifty 50",
        kind="index_option",
        exchange="NSE",
        segment="OPT",
        tick_size=0.05,
        lot_sizes=[
            LotSizeEntry(effective_from=date(2024, 4, 1), lot_size=50, source="circular-2024-11"),
            LotSizeEntry(effective_from=date(2025, 11, 1), lot_size=75, source="circular-2025-42"),
        ],
    )
    assert inst.lot_size_on(date(2025, 6, 1)) == 50
    assert inst.lot_size_on(date(2026, 8, 26)) == 75


def test_lot_size_history_uses_the_entry_in_force_on_the_boundary():
    inst = Instrument(
        id=1,
        symbol="NIFTY",
        name="Nifty 50",
        kind="index_option",
        exchange="NSE",
        segment="OPT",
        tick_size=0.05,
        lot_sizes=[
            LotSizeEntry(effective_from=date(2024, 4, 1), lot_size=50, source="c1"),
            LotSizeEntry(effective_from=date(2025, 11, 1), lot_size=75, source="c2"),
        ],
    )
    assert inst.lot_size_on(date(2025, 10, 31)) == 50
    assert inst.lot_size_on(date(2025, 11, 1)) == 75


def test_lot_size_history_refuses_a_date_before_the_first_entry():
    """Falling back to the earliest known size would answer a question the data
    cannot answer, and the answer would look like a measurement."""
    inst = Instrument(
        id=1,
        symbol="NIFTY",
        name="Nifty 50",
        kind="index_option",
        exchange="NSE",
        segment="OPT",
        tick_size=0.05,
        lot_sizes=[LotSizeEntry(effective_from=date(2024, 4, 1), lot_size=50, source="c1")],
    )
    with pytest.raises(LookupError, match="no lot size known"):
        inst.lot_size_on(date(2023, 1, 1))


def test_lot_size_history_must_be_sorted():
    with pytest.raises(ValueError, match="sorted by effective_from"):
        Instrument(
            id=1,
            symbol="NIFTY",
            name="Nifty 50",
            kind="index_option",
            exchange="NSE",
            segment="OPT",
            tick_size=0.05,
            lot_sizes=[
                LotSizeEntry(effective_from=date(2025, 11, 1), lot_size=75, source="c2"),
                LotSizeEntry(effective_from=date(2024, 4, 1), lot_size=50, source="c1"),
            ],
        )


def test_a_contract_carries_the_lot_size_it_was_listed_with():
    """Belt and braces: the contract records its own lot size, so a historical
    fill does not depend on the underlying's history being loaded."""
    c = option(1, "NIFTY24500CE", strike=24500, lot=50)
    assert c.lot_size == 50


# --------------------------------------------------------------------------- #
# expiry_calendar
# --------------------------------------------------------------------------- #
def test_expiry_calendar_finds_the_last_thursday():
    assert monthly_expiry(2026, 8, weekday=THU, is_trading_day=weekdays_only) == date(2026, 8, 27)


def test_expiry_calendar_rolls_back_off_a_holiday_never_forward():
    """A contract cannot settle after the date printed on it. Rolling forward
    makes a position appear to survive a day longer than it could have, and a
    daily-close backtest will never show it."""
    closed = with_holidays(date(2026, 8, 27))
    assert monthly_expiry(2026, 8, weekday=THU, is_trading_day=closed) == date(2026, 8, 26)


def test_expiry_calendar_rolls_back_over_consecutive_holidays():
    closed = with_holidays(date(2026, 8, 27), date(2026, 8, 26))
    assert monthly_expiry(2026, 8, weekday=THU, is_trading_day=closed) == date(2026, 8, 25)


def test_expiry_calendar_rolls_back_across_a_weekend():
    """A Thursday and Friday shut means the previous Wednesday, not Saturday."""
    closed = with_holidays(date(2026, 8, 6))
    weeklies = weekly_expiries(2026, 8, weekday=THU, is_trading_day=closed)
    assert date(2026, 8, 5) in weeklies


def test_expiry_calendar_gives_up_rather_than_looping_forever():
    with pytest.raises(InstrumentError, match="need a human"):
        monthly_expiry(2026, 8, weekday=THU, is_trading_day=lambda _d: False)


def test_expiry_calendar_lists_every_weekly_in_the_month():
    weeklies = weekly_expiries(2026, 8, weekday=THU, is_trading_day=weekdays_only)
    assert weeklies == [
        date(2026, 8, 6),
        date(2026, 8, 13),
        date(2026, 8, 20),
        date(2026, 8, 27),
    ]


def test_the_last_weekly_is_the_monthly_not_an_extra_contract():
    """Listing them separately double-counts the open interest and invents a
    roll that never happened."""
    weeklies = weekly_expiries(2026, 8, weekday=THU, is_trading_day=weekdays_only)
    monthly = monthly_expiry(2026, 8, weekday=THU, is_trading_day=weekdays_only)

    assert weeklies[-1] == monthly
    assert weeklies.count(monthly) == 1
    assert is_monthly(monthly, weekday=THU, is_trading_day=weekdays_only)
    assert not is_monthly(weeklies[0], weekday=THU, is_trading_day=weekdays_only)


def test_expiry_calendar_does_not_emit_a_date_twice_after_rolling_back():
    """Two adjacent weeklies rolling onto the same day is possible with a long
    shutdown, and emitting it twice would create a phantom contract."""
    closed = with_holidays(
        date(2026, 8, 13), date(2026, 8, 12), date(2026, 8, 11), date(2026, 8, 10)
    )
    weeklies = weekly_expiries(2026, 8, weekday=THU, is_trading_day=closed)
    assert len(weeklies) == len(set(weeklies))


def test_expiry_calendar_spans_a_year_boundary():
    got = list(
        expiries_between(
            date(2026, 12, 20), date(2027, 1, 15), weekday=THU, is_trading_day=weekdays_only
        )
    )
    assert got == [date(2026, 12, 24), date(2026, 12, 31), date(2027, 1, 7), date(2027, 1, 14)]
    assert got == sorted(got)


def test_days_to_expiry_counts_trading_days_not_calendar_days():
    """Calendar days price a Thursday-to-Monday gap as three days of decay when
    the market is open for one of them."""
    assert days_to_expiry(date(2026, 8, 27), date(2026, 8, 31), is_trading_day=weekdays_only) == 3


def test_days_to_expiry_counts_the_expiry_day_itself():
    assert days_to_expiry(date(2026, 8, 27), date(2026, 8, 27), is_trading_day=weekdays_only) == 1


def test_days_to_expiry_refuses_a_past_expiry():
    with pytest.raises(InstrumentError, match="cannot expire in the past"):
        days_to_expiry(date(2026, 8, 27), date(2026, 8, 20), is_trading_day=weekdays_only)


# --------------------------------------------------------------------------- #
# the contract record itself
# --------------------------------------------------------------------------- #
def test_an_option_needs_a_strike_and_a_right():
    with pytest.raises(ValueError, match="strike and a right"):
        Contract(
            instrument_id=1,
            tradingsymbol="NIFTY24500CE",
            exchange_token=1,
            underlying="NIFTY",
            kind=ContractKind.OPTION,
            expiry=date(2026, 8, 27),
            valid_from=date(2026, 7, 30),
            valid_to=date(2026, 8, 27),
            lot_size=75,
            tick_size=0.05,
        )


def test_a_future_has_neither():
    with pytest.raises(ValueError, match="neither strike nor right"):
        Contract(
            instrument_id=1,
            tradingsymbol="NIFTYAUGFUT",
            exchange_token=1,
            underlying="NIFTY",
            kind=ContractKind.FUTURE,
            expiry=date(2026, 8, 27),
            valid_from=date(2026, 7, 30),
            valid_to=date(2026, 8, 27),
            lot_size=75,
            tick_size=0.05,
            strike=24500.0,
            right=Right.CALL,
        )


def test_a_backwards_validity_window_is_refused():
    with pytest.raises(ValueError, match="precedes valid_from"):
        option(
            1, "NIFTY24500CE", strike=24500, valid_from=date(2026, 9, 1), valid_to=date(2026, 8, 1)
        )


def test_the_chain_is_the_chain_as_it_stood_that_day():
    """An expired strike must not reappear in a chain rebuilt for a later date."""
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500, token=1),
            option(2, "NIFTY24600CE", strike=24600, token=2),
            option(
                3,
                "NIFTY24700CE",
                strike=24700,
                token=3,
                valid_from=date(2026, 8, 20),
            ),
        ]
    )
    early = master.chain("NIFTY", date(2026, 8, 27), on=date(2026, 8, 10))
    late = master.chain("NIFTY", date(2026, 8, 27), on=date(2026, 8, 25))

    assert [c.strike for c in early] == [24500.0, 24600.0]
    assert [c.strike for c in late] == [24500.0, 24600.0, 24700.0]


def test_live_on_excludes_contracts_that_had_not_listed():
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500, token=1),
            option(2, "NIFTY24600CE", strike=24600, token=2, valid_from=date(2026, 8, 20)),
        ]
    )
    assert len(master.live_on(date(2026, 8, 1))) == 1
    assert len(master.live_on(date(2026, 8, 25))) == 2


def test_the_render_flags_reuse_rather_than_hiding_it():
    master = InstrumentMaster().extend(
        [
            option(1, "NIFTY24500CE", strike=24500),
            option(
                2,
                "NIFTY24500CE",
                strike=24500,
                expiry=date(2026, 12, 31),
                valid_from=date(2026, 12, 1),
                token=2000,
            ),
        ]
    )
    assert "reused across contracts" in master.render()


def test_an_empty_master_renders_without_crashing():
    assert "empty" in InstrumentMaster().render()
