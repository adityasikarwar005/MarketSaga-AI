"""Where the strike ladder gets centred, and the three ways that goes wrong.

The centre used to be a constant, 24,800, baked into an argparse default. NIFTY
closed at 24,175.65 on 2026-08-28, so a session started without the flag built
its grid 625 points off centre and nothing raised: the recorder ran, the file
filled, and the strikes at the bottom of the ladder were wings nobody can price.
That is the worst shape a bug can take here, because the session cannot be
recorded again the next morning.

These tests pin the replacement. The guard is expressed in strikes rather than
in percent for a reason worth keeping: the first version used 5% of the index,
and the 625-point miss it was written to catch is 2.58% of NIFTY, so it passed.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import date, timedelta

import pytest

from libs.ms_data.daily import DailyBar, DailyStore
from services.ingest_market.__main__ import (
    MAX_CLOSE_STALENESS_DAYS,
    REGISTRY_STRIKES_PER_SIDE,
    STRIKE_STEP,
    _ladder_centre,
)

TODAY = date(2026, 8, 31)
LAST_CLOSE = 24_175.65


@pytest.fixture
def store(tmp_path):
    s = DailyStore(root=tmp_path / "daily")
    s.save(
        "NIFTY",
        [
            DailyBar(
                day=date(2026, 8, 28) - timedelta(days=i),
                o=LAST_CLOSE,
                h=LAST_CLOSE,
                low=LAST_CLOSE,
                c=LAST_CLOSE,
                v=0,
            )
            for i in reversed(range(5))
        ],
        source="test",
    )
    return s


def args(**kw) -> Namespace:
    return Namespace(
        chain_spot=kw.get("chain_spot"),
        chain_name=kw.get("chain_name", "NIFTY"),
        strikes_per_side=kw.get("strikes_per_side", 15),
    )


def test_the_default_is_the_last_close_not_a_constant(store):
    assert _ladder_centre(args(), TODAY, store=store) == pytest.approx(LAST_CLOSE)


def test_an_explicit_centre_near_the_close_is_taken_as_given(store):
    assert _ladder_centre(args(chain_spot=24_200.0), TODAY, store=store) == pytest.approx(24_200.0)


def test_the_case_that_motivated_the_guard_is_warned_about_not_refused(store, capsys):
    """24,800 against a 24,175.65 close is twelve strikes out.

    It is not fatal - the money is still inside a grid that reaches 25 strikes
    either way - and refusing it would be wrong. What it costs is the bottom of
    the ladder, and the warning says how much.
    """
    assert _ladder_centre(args(chain_spot=24_800.0), TODAY, store=store) == pytest.approx(24_800.0)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "12 strikes" in out
    assert "clipped" in out


def test_a_centre_that_puts_the_money_outside_the_grid_is_refused(store, capsys):
    off = LAST_CLOSE + STRIKE_STEP * (REGISTRY_STRIKES_PER_SIDE + 1)
    assert _ladder_centre(args(chain_spot=off), TODAY, store=store) is None
    assert "refusing" in capsys.readouterr().out


def test_a_centre_on_the_close_is_neither_warned_about_nor_refused(store, capsys):
    assert _ladder_centre(args(chain_spot=LAST_CLOSE), TODAY, store=store) == pytest.approx(
        LAST_CLOSE
    )
    out = capsys.readouterr().out
    assert "WARNING" not in out and "refusing" not in out


def test_the_clip_warning_tracks_the_ladder_width(store, capsys):
    """A narrower ladder tolerates more drift, and the message has to reflect it.

    Ten strikes of drift clips a 15-wide ladder and leaves a 10-wide one intact,
    because the grid reaches 25 either way. A warning keyed to a fixed distance
    rather than to the geometry would be wrong for one of the two.
    """
    off = LAST_CLOSE + STRIKE_STEP * 12
    _ladder_centre(args(chain_spot=off, strikes_per_side=15), TODAY, store=store)
    assert "WARNING" in capsys.readouterr().out
    _ladder_centre(args(chain_spot=off, strikes_per_side=10), TODAY, store=store)
    assert "WARNING" not in capsys.readouterr().out


def test_no_daily_history_is_refused_rather_than_guessed(tmp_path, capsys):
    empty = DailyStore(root=tmp_path / "nothing")
    assert _ladder_centre(args(), TODAY, store=empty) is None
    out = capsys.readouterr().out
    assert "refusing" in out and "gap_backfill" in out


def test_a_stale_close_is_refused_rather_than_centred_on(tmp_path, capsys):
    """Centring on a close from months ago is no better than the constant it
    replaced, and it would look like the fix was working."""
    s = DailyStore(root=tmp_path / "stale")
    old = TODAY - timedelta(days=MAX_CLOSE_STALENESS_DAYS + 30)
    s.save(
        "NIFTY",
        [DailyBar(day=old, o=20_000.0, h=20_000.0, low=20_000.0, c=20_000.0, v=0)],
        source="test",
    )
    assert _ladder_centre(args(), TODAY, store=s) is None
    assert "refusing" in capsys.readouterr().out


def test_an_explicit_centre_still_works_without_any_history(tmp_path):
    """The override is the escape hatch for a machine with no daily cache, so it
    must not be gated on the very thing it exists to substitute for."""
    empty = DailyStore(root=tmp_path / "nothing")
    assert _ladder_centre(args(chain_spot=24_200.0), TODAY, store=empty) == pytest.approx(24_200.0)


def test_a_different_underlying_reads_its_own_close(tmp_path):
    s = DailyStore(root=tmp_path / "multi")
    for symbol, close in (("NIFTY", 24_175.65), ("BANKNIFTY", 57_496.30)):
        s.save(
            "" + symbol,
            [DailyBar(day=date(2026, 8, 28), o=close, h=close, low=close, c=close, v=0)],
            source="test",
        )
    assert _ladder_centre(args(chain_name="BANKNIFTY"), TODAY, store=s) == pytest.approx(57_496.30)
