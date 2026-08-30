"""`verified: true` has to mean something a test can check.

The flag gates sizing: M76 refuses to size an unverified instrument, and the
header of `config/instruments.yaml` describes a four-step procedure ending in
"flip `verified: true`". Nothing enforced the first three steps. A flag set by
hand, on the honour system, next to a `source:` field still reading
"TODO: verify" is not a verification - it is a claim, and it would pass every
test in the suite while a wrong lot size silently multiplied a position.

So these tests bind the flag to its evidence. Flipping `verified` without
citing a circular fails here, which is the only moment anyone is looking.

The lot sizes themselves are pinned too. NSE/FAOP/70616 cut NIFTY 75 -> 65 and
BANKNIFTY 35 -> 30 from the January 2026 series; those numbers were read back
out of Angel's live scrip master on 2026-08-30 and agreed across all live
contracts. Pinning them means a careless edit fails rather than propagates.
"""

from __future__ import annotations

from datetime import date

import pytest

from libs.ms_core.config import Instrument, load

TODO = "todo"
# Read off NSE/FAOP/70616 and confirmed against every live NFO contract in
# Angel's scrip master on 2026-08-30.
CURRENT_LOTS = {"NIFTY-FUT": 65, "BANKNIFTY-FUT": 30}
TUESDAY = 1


@pytest.fixture(scope="module")
def instruments() -> list[Instrument]:
    return list(load(warn=False).instruments.instruments)


class TestVerifiedMeansSourced:
    def test_a_verified_instrument_has_no_todo_left_in_it(
        self, instruments: list[Instrument]
    ) -> None:
        """The flag and the evidence must move together.

        This is the guard the config header assumed and never had.
        """
        for inst in instruments:
            if not inst.verified:
                continue
            unsourced = [e for e in inst.lot_sizes if TODO in e.source.lower()]
            assert not unsourced, (
                f"{inst.symbol} is marked verified: true, but "
                f"{len(unsourced)} of its {len(inst.lot_sizes)} lot-size entries "
                f"still say TODO (from {unsourced[0].effective_from}). Either "
                "cite the circular for every entry or set verified: false - a "
                "flag that outruns its evidence is worse than no flag, because "
                "M76 will size on it."
            )

    def test_a_verified_instrument_cites_something(self, instruments: list[Instrument]) -> None:
        """A `source` of "" or "n/a" passes the TODO check and says nothing."""
        for inst in instruments:
            if not inst.verified:
                continue
            for entry in inst.lot_sizes:
                assert len(entry.source) >= 20, (
                    f"{inst.symbol} {entry.effective_from} cites {entry.source!r}, "
                    "which is too short to identify a circular."
                )


class TestPinnedContractSpec:
    @pytest.mark.parametrize(("symbol", "lot"), sorted(CURRENT_LOTS.items()))
    def test_current_lot_size_matches_the_circular(
        self, instruments: list[Instrument], symbol: str, lot: int
    ) -> None:
        inst = next(i for i in instruments if i.symbol == symbol)
        got = inst.lot_size_on(date(2026, 8, 30))
        assert got == lot, (
            f"{symbol} lot is {got}; NSE/FAOP/70616 and Angel's live scrip "
            f"master both say {lot}. If NSE has revised it again, add a new "
            "dated entry and update this pin in the same commit."
        )

    def test_the_pre_2026_nifty_lot_is_still_reachable(self, instruments: list[Instrument]) -> None:
        """Adding the new entry must not have overwritten the old one.

        The backtest runs from 2016. A lot size table that only knows today
        prices every historical trade wrong, in one direction, invisibly.
        """
        nifty = next(i for i in instruments if i.symbol == "NIFTY-FUT")
        assert nifty.lot_size_on(date(2024, 6, 1)) == 75

    def test_index_expiry_day_is_tuesday(self, instruments: list[Instrument]) -> None:
        """NSE moved index expiry off Thursday. The config said 3 for two years."""
        for symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
            inst = next(i for i in instruments if i.symbol == symbol)
            assert inst.expiry_weekday == TUESDAY, (
                f"{symbol} expiry_weekday is {inst.expiry_weekday}; every live "
                "NFO expiry that is not a holiday roll falls on a Tuesday."
            )
