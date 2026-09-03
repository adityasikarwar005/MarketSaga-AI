"""The page itself, run against payloads the packer wrote.

Every other test here checks the Python half: that `write` splits the files the
way it says, that a bar index means minutes since the open, that a thin contract
keeps its rung. None of them can fail if the *page* stops being able to read any
of it, and that is the half with the history. The chain was eleven strikes
wearing the word "every" for a day, and before that the renderer hung completely
on a layout loop. Both were found by opening it in a browser, which nothing in
CI does.

The failure mode is what makes it worth a test rather than a habit: a page that
cannot read its payload does not error. It draws the ladder from the index,
requests the candles, and leaves the pane empty. There is one failed line in a
console nobody is looking at.

So this stubs the smallest browser the page will accept, runs the page's own
`init()`, and asserts it drew what the index promised. It is not a rendering
test - the canvas is a no-op. It is the only test that spans the two languages
this feature is written in, and the contract between them is a set of field
names that nothing else checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from jobs.chart_history import write

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests" / "js" / "render_check.mjs"
PAGE = ROOT / "services" / "dashboard" / "chart.html"
DAY = date(2026, 9, 3)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="needs node to run the page's own script; the Python half is tested either way",
)


def _series(strike: float, right: str, *, projected: bool) -> dict:
    recorded = [[i, 100.0, 101.0, 99.0, 100.5, 10 + i] for i in range(60)]
    fan = (
        [[59 + k, 100.5, 102.0, 99.0, 100.4, 100.0, 101.0] for k in range(1, 61)]
        if projected
        else []
    )
    return {
        "strike": strike,
        "right": right,
        "symbol": f"NIFTY08SEP26{int(strike)}{right}",
        "token": int(strike) + (1 if right == "CE" else 2),
        "iv": 0.123,
        "anchor": "10:14",
        "anchor_price": 100.5,
        "recorded": recorded,
        "projected": fan,
    }


def _board(name: str, expiry: str, atm: float, *, basis: str) -> dict:
    projected = basis != "carried"
    strikes = [atm - 50.0, atm, atm + 50.0]
    chain = [
        {
            "strike": k,
            "right": r,
            "last": 100.0 + k / 1000.0,
            "volume": 500,
            "bars": 60 if k != strikes[-1] else 3,
            "last_print": "09:16" if k == strikes[-1] else "11:14",
            "read_at": "11:32",
            "chartable": k != strikes[-1],
        }
        for k in strikes
        for r in ("CE", "PE")
    ]
    return {
        "name": name,
        "expiry": expiry,
        "open": "09:15",
        "day": DAY.isoformat(),
        "source": "angel historical candles",
        "as_of": "11:43",
        "read_from": "11:32",
        "listed": 8,
        "atm": atm,
        "step": 50.0,
        "forward": {
            "symbol": "NIFTY29SEP26FUT",
            "close": atm + 8.0,
            "bars": 148,
            "basis": basis,
            "token": 68407,
            "expiry": "2026-09-29",
        },
        "chain": chain,
        "series": [_series(k, r, projected=projected) for k in strikes[:-1] for r in ("CE", "PE")],
        "failed": [],
        "score": {
            "method": "closed form",
            "baseline": "the contract's price now, unchanged",
            "measured_on": DAY.isoformat(),
            "horizons": (
                [{"minutes": 5, "n": 100, "model_mae": 1.0, "baseline_mae": 1.1, "delta_pct": -9.1}]
                if projected
                else []
            ),
            "verdict": (
                "beat the baseline at every horizon, by 9.1% to 9.1%"
                if projected
                else "not projected - no listed future expires on or after this expiry"
            ),
            "caveat": "one session, and the observations overlap",
            "point_estimate": "the median, not the mean",
        },
    }


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("dashboard")
    write(
        {
            "NIFTY": [
                _board("NIFTY", "2026-09-08", 24000.0, basis="bracketing"),
                _board("NIFTY", "2029-06-26", 24000.0, basis="carried"),
            ],
            "BANKNIFTY": [_board("BANKNIFTY", "2026-09-29", 57500.0, basis="same-cycle")],
        },
        DAY,
        out,
    )
    shutil.copy(PAGE, out / "chart.html")
    proc = subprocess.run(
        ["node", str(HARNESS), str(out)],
        capture_output=True,
        text=True,
        # Explicit, because `text=True` alone decodes with the *locale* encoding
        # and this repository runs on Windows, where that is cp1252. Node writes
        # UTF-8. The en dash in the page's own header came back as three
        # characters and the assertion failed on a string that was correct at
        # both ends - a decoding fault in the test harness reported as a defect
        # in the page.
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        "the page's own script could not be run at all:\n" + proc.stderr[-2000:]
    )
    return json.loads(proc.stdout)


def test_the_page_runs_against_what_the_packer_wrote(rendered: dict) -> None:
    assert rendered["threw"] is None, (
        f"chart.html raised {rendered['threw']} on a payload this repository "
        "generated. In a browser this is an empty pane and one console line."
    )


def test_the_rail_has_a_rung_for_every_expiry(rendered: dict) -> None:
    assert rendered["railButtons"] == 2, "the expiry rail did not list both expiries"
    assert rendered["railSelected"] == 1, "no expiry, or more than one, reads as selected"


def test_the_ladder_is_every_strike_not_a_band(rendered: dict) -> None:
    """The correction that started this. Eleven strikes is not every call and put."""
    assert rendered["ladderRows"] == 3, "the ladder dropped strikes the chain carries"
    assert rendered["ladderCurrent"] == 1


def test_a_strike_opens_its_call_and_its_put_together(rendered: dict) -> None:
    """A strike is one position seen from two directions; a toggle hides that."""
    assert rendered["callTitle"] == "24,000 CE"
    assert rendered["putTitle"] == "24,000 PE"


def test_the_header_reads_the_expiry_not_the_underlying(rendered: dict) -> None:
    assert rendered["expiry"] == "8 Sept"
    assert rendered["forward"] == "24,008.00"
    assert "8" in rendered["counts"] and "charted" in rendered["counts"]


def test_the_read_window_is_when_the_sweep_ran(rendered: dict) -> None:
    """Not when contracts last printed, which is a different fact entirely.

    Some far wing prints once at 09:16 and stays quiet, so the earliest last-bar
    on almost any board is just after the open - and a header built from it
    announced that a sweep which began at 11:00 had started at 09:15.
    """
    # Built rather than written. The page separates the range with an en dash,
    # which is right on the page and is a character ruff refuses in source -
    # reasonably, since it is a hyphen to every reader and not to the parser.
    en_dash = chr(0x2013)
    assert rendered["asOf"] == f"swept 11:32{en_dash}11:43"


def test_an_expiry_with_no_future_behind_it_says_so(rendered: dict) -> None:
    """Rather than drawing a band that looks exactly like the honest ones."""
    assert "No forecast" in rendered["carriedVerdict"], rendered["carriedVerdict"]
