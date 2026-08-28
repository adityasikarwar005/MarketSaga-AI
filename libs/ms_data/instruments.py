"""M10 - the instrument master: what a symbol meant on a given day.

The holiday calendar and the lot-size history already live elsewhere
(``ms_data/holidays.py`` and ``Instrument.lot_size_on`` in ``ms_core/config``).
What is here is the part those two do not cover: **derivative contracts, and the
fact that their identifiers are not unique over time.**

Symbols get reused, and so do tokens
-------------------------------------

``NIFTY24500CE`` is not an instrument. It is a string that referred to one
contract in August and may refer to a different one in December, because the
exchange reuses tradingsymbols once a contract has expired and brokers recycle
numeric tokens on a schedule nobody publishes.

That makes ``resolve(symbol)`` an unanswerable question, and the damage from
answering it anyway is quiet: the trade books against the wrong contract, the
P&L attaches to a strike that was never traded, and every number involved is a
plausible number. So resolution takes a date, always, and refuses without one.

Expiry moves backwards, never forwards
---------------------------------------

When an expiry day falls on a holiday the contract expires on the **previous**
trading day. Moving it forward would settle a contract after the date printed on
it, and the off-by-one is invisible in backtests that only ever look at daily
closes - the position simply appears to survive a day longer than it could have.

The weekly/monthly distinction matters for the same reason: the last weekly
expiry of a month *is* the monthly expiry, and treating them as two contracts
double-counts the open interest and produces a phantom roll.
"""

from __future__ import annotations

import calendar as _calendar
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from libs.ms_core.obs import get_logger

log = get_logger(__name__)


class Right(StrEnum):
    CALL = "CE"
    PUT = "PE"


class ContractKind(StrEnum):
    FUTURE = "FUT"
    OPTION = "OPT"


class InstrumentError(LookupError):
    pass


class AmbiguousSymbol(InstrumentError):
    """One string, more than one contract, and no date to choose between them."""


@dataclass(frozen=True)
class Contract:
    """One tradable derivative, and the window in which its identifiers meant it.

    ``valid_from`` / ``valid_to`` are the listing and delisting dates - the
    interval during which this tradingsymbol and token referred to *this*
    contract. Both are inclusive. A resolution outside the window is not this
    contract, however well the string matches.
    """

    instrument_id: int
    tradingsymbol: str
    exchange_token: int
    underlying: str
    kind: ContractKind
    expiry: date
    valid_from: date
    valid_to: date
    lot_size: int
    tick_size: float
    strike: float | None = None
    right: Right | None = None

    def __post_init__(self) -> None:
        if self.valid_to < self.valid_from:
            raise ValueError(
                f"{self.tradingsymbol}: valid_to {self.valid_to} precedes valid_from "
                f"{self.valid_from}"
            )
        if self.kind is ContractKind.OPTION and (self.strike is None or self.right is None):
            raise ValueError(f"{self.tradingsymbol}: an option needs both a strike and a right")
        if self.kind is ContractKind.FUTURE and (self.strike is not None or self.right is not None):
            raise ValueError(f"{self.tradingsymbol}: a future has neither strike nor right")
        if self.lot_size <= 0 or self.tick_size <= 0:
            raise ValueError(f"{self.tradingsymbol}: lot and tick sizes must be positive")

    def covers(self, on: date) -> bool:
        return self.valid_from <= on <= self.valid_to

    @property
    def is_option(self) -> bool:
        return self.kind is ContractKind.OPTION

    def render(self) -> str:
        what = (
            f"{self.strike:.0f}{self.right.value}" if self.is_option else self.kind.value  # type: ignore[union-attr]
        )
        return (
            f"    {self.tradingsymbol:<22} token={self.exchange_token:<10} "
            f"{self.underlying} {what} exp={self.expiry}  "
            f"valid {self.valid_from}..{self.valid_to}  lot={self.lot_size}"
        )


# --------------------------------------------------------------------------- #
# The master
# --------------------------------------------------------------------------- #
@dataclass
class InstrumentMaster:
    """Every contract that has existed, indexed so a lookup needs a date.

    The indexes are lists rather than dicts on purpose. A dict keyed by
    tradingsymbol is exactly the structure that makes reuse invisible: the second
    contract overwrites the first, the master reports one entry, and the earlier
    trades resolve to the later contract without anything failing.
    """

    contracts: list[Contract] = field(default_factory=list)
    _by_symbol: dict[str, list[Contract]] = field(default_factory=dict, repr=False)
    _by_token: dict[int, list[Contract]] = field(default_factory=dict, repr=False)
    _by_id: dict[int, Contract] = field(default_factory=dict, repr=False)

    def add(self, contract: Contract) -> Contract:
        if contract.instrument_id in self._by_id:
            raise InstrumentError(
                f"instrument_id {contract.instrument_id} is already registered to "
                f"{self._by_id[contract.instrument_id].tradingsymbol}. The id is the "
                "one identifier that must never be reused - it is what every stored "
                "trade points at."
            )

        for existing in self._by_symbol.get(contract.tradingsymbol, []):
            if _overlaps(existing, contract):
                raise InstrumentError(
                    f"{contract.tradingsymbol} is valid {contract.valid_from}.."
                    f"{contract.valid_to} and also {existing.valid_from}.."
                    f"{existing.valid_to}. Two contracts sharing a symbol on the same "
                    "day makes resolution a coin toss."
                )
        for existing in self._by_token.get(contract.exchange_token, []):
            if _overlaps(existing, contract):
                raise InstrumentError(
                    f"token {contract.exchange_token} is claimed by both "
                    f"{existing.tradingsymbol} and {contract.tradingsymbol} on "
                    "overlapping dates. Brokers recycle tokens; two live at once is "
                    "a bad instrument dump, not a recycle."
                )

        self.contracts.append(contract)
        self._by_symbol.setdefault(contract.tradingsymbol, []).append(contract)
        self._by_token.setdefault(contract.exchange_token, []).append(contract)
        self._by_id[contract.instrument_id] = contract
        return contract

    def extend(self, contracts: Sequence[Contract]) -> InstrumentMaster:
        for c in contracts:
            self.add(c)
        return self

    # ------------------------------------------------------------- resolution
    def resolve_at_date(self, symbol: str, on: date) -> Contract:
        """The contract this symbol referred to on ``on``.

        The date is required and there is no overload without it. A default of
        "today" would silently re-point every historical lookup at whatever
        currently holds the string, and the result would still be a valid
        contract - just not the one that was traded.
        """
        candidates = [c for c in self._by_symbol.get(symbol, []) if c.covers(on)]
        if not candidates:
            ever = self._by_symbol.get(symbol, [])
            if ever:
                windows = ", ".join(f"{c.valid_from}..{c.valid_to}" for c in ever)
                raise InstrumentError(
                    f"{symbol} existed, but not on {on}. Known windows: {windows}. "
                    "Resolving to the nearest one would book the trade against a "
                    "contract that had not been listed yet."
                )
            raise InstrumentError(f"{symbol} is not in the instrument master")
        if len(candidates) > 1:  # pragma: no cover - add() rejects overlaps
            raise AmbiguousSymbol(f"{symbol} resolves to {len(candidates)} contracts on {on}")
        return candidates[0]

    def resolve_token_at_date(self, token: int, on: date) -> Contract:
        """Same rule for exchange tokens, which are recycled just as freely."""
        candidates = [c for c in self._by_token.get(token, []) if c.covers(on)]
        if not candidates:
            raise InstrumentError(
                f"token {token} maps to nothing on {on}. A token that resolves on one "
                "date and not another is normal - they are recycled - which is why "
                "stored records must reference instrument_id, never the token."
            )
        return candidates[0]

    def by_id(self, instrument_id: int) -> Contract:
        """The stable identifier. This is the one a stored trade may point at."""
        if instrument_id not in self._by_id:
            raise InstrumentError(f"instrument_id {instrument_id} is not in the master")
        return self._by_id[instrument_id]

    def history(self, symbol: str) -> list[Contract]:
        """Every contract that has ever worn this symbol, oldest first.

        Exists so that "was this symbol reused?" is a question with an answer,
        rather than something discovered from a P&L that does not reconcile.
        """
        return sorted(self._by_symbol.get(symbol, []), key=lambda c: c.valid_from)

    def reused_symbols(self) -> dict[str, int]:
        """Symbols worn by more than one contract. A report, not an error."""
        return {s: len(cs) for s, cs in self._by_symbol.items() if len(cs) > 1}

    def live_on(self, on: date) -> list[Contract]:
        return [c for c in self.contracts if c.covers(on)]

    def chain(self, underlying: str, expiry: date, *, on: date) -> list[Contract]:
        """The option chain for one expiry, as it stood on ``on``."""
        return sorted(
            (
                c
                for c in self.contracts
                if c.underlying == underlying
                and c.expiry == expiry
                and c.is_option
                and c.covers(on)
            ),
            key=lambda c: (c.strike or 0.0, c.right or ""),
        )

    def render(self) -> str:
        if not self.contracts:
            return "  instrument master is empty"
        lines = [f"  instrument master: {len(self.contracts)} contracts"]
        lines += [c.render() for c in sorted(self.contracts, key=lambda c: c.valid_from)]
        reused = self.reused_symbols()
        if reused:
            lines.append(
                f"  {len(reused)} symbol(s) reused across contracts: "
                + ", ".join(f"{s} x{n}" for s, n in sorted(reused.items()))
            )
        return "\n".join(lines)


def _overlaps(a: Contract, b: Contract) -> bool:
    return a.valid_from <= b.valid_to and b.valid_from <= a.valid_to


# --------------------------------------------------------------------------- #
# The expiry calendar
# --------------------------------------------------------------------------- #
TradingDayTest = Callable[[date], bool]


def _roll_back(day: date, is_trading_day: TradingDayTest, *, limit: int = 10) -> date:
    """Step backwards to the previous trading day.

    Backwards, never forwards. A contract cannot settle after the date printed on
    it, and rolling forward produces a position that appears to survive a day
    longer than it could have - which a daily-close backtest will never show.
    """
    cursor = day
    for _ in range(limit):
        if is_trading_day(cursor):
            return cursor
        cursor = date.fromordinal(cursor.toordinal() - 1)
    raise InstrumentError(
        f"no trading day within {limit} days before {day}. Either the holiday "
        "calendar is wrong or the exchange was shut for a fortnight; both need a human."
    )


def monthly_expiry(year: int, month: int, *, weekday: int, is_trading_day: TradingDayTest) -> date:
    """The last ``weekday`` of the month, rolled back off holidays."""
    last = _calendar.monthrange(year, month)[1]
    cursor = date(year, month, last)
    while cursor.weekday() != weekday:
        cursor = date.fromordinal(cursor.toordinal() - 1)
    return _roll_back(cursor, is_trading_day)


def weekly_expiries(
    year: int, month: int, *, weekday: int, is_trading_day: TradingDayTest
) -> list[date]:
    """Every weekly expiry in the month, holiday-adjusted, in order.

    The last one is the monthly expiry - the same contract, not an extra. A
    calendar that lists them separately double-counts the open interest and
    invents a roll that never happened, so this returns one entry for that date
    and :func:`is_monthly` is how you ask which it is.
    """
    out: list[date] = []
    seen: set[date] = set()
    cursor = date(year, month, 1)
    while cursor.weekday() != weekday:
        cursor = date.fromordinal(cursor.toordinal() + 1)
    while cursor.month == month:
        adjusted = _roll_back(cursor, is_trading_day)
        if adjusted not in seen:
            seen.add(adjusted)
            out.append(adjusted)
        cursor = date.fromordinal(cursor.toordinal() + 7)
    return out


def is_monthly(expiry: date, *, weekday: int, is_trading_day: TradingDayTest) -> bool:
    return expiry == monthly_expiry(
        expiry.year, expiry.month, weekday=weekday, is_trading_day=is_trading_day
    )


def expiries_between(
    start: date, end: date, *, weekday: int, is_trading_day: TradingDayTest
) -> Iterator[date]:
    """Weekly expiries across a range, inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        for d in weekly_expiries(y, m, weekday=weekday, is_trading_day=is_trading_day):
            if start <= d <= end:
                yield d
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def days_to_expiry(on: date, expiry: date, *, is_trading_day: TradingDayTest) -> int:
    """Trading days remaining, counting the expiry day itself.

    Calendar days would price a Thursday-to-Monday gap as three days of decay
    when the market is open for one of them.
    """
    if expiry < on:
        raise InstrumentError(f"{expiry} is before {on}; a contract cannot expire in the past")
    return sum(
        1
        for n in range(on.toordinal(), expiry.toordinal() + 1)
        if is_trading_day(date.fromordinal(n))
    )
