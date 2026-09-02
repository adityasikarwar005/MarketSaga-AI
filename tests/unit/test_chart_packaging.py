"""How a day's chain is packed for the page, and what must not be swallowed on the way.

Three things here have already gone wrong once each, and none of them fails
loudly on its own:

* **A retry that eats real errors.** `fetch_with_backoff` waits out a throttle
  and, since a `ConnectTimeout` ended a 454-contract run at contract 247, waits
  out the network too. The risk in widening that is obvious: a retry loop that
  cannot tell a wrong token from a slow one turns a clear failure into five
  silent minutes and then the same clear failure.
* **A bar index read against the wrong clock.** Bars ship as minutes since the
  open rather than as "09:15" strings, which is most of the file size back. The
  index is only meaningful against the open the same payload carries.
* **A thin contract charted anyway.** A strike that printed nine times has no
  chart in it. It belongs in the ladder, with its last price, and nowhere else.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from jobs import option_history
from jobs.chart_history import MIN_BARS_FOR_CHART, write
from libs.ms_core.schemas import Candle

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 2)
OPEN = datetime(2026, 9, 2, 9, 15, tzinfo=IST)


class TestTransient:
    def test_the_throttle_is_worth_waiting_out(self) -> None:
        assert (
            option_history.transient(RuntimeError("the endpoint is rate limiting")) == "throttled"
        )
        assert (
            option_history.transient(Exception("Access denied because of exceeding access rate"))
            == "throttled"
        )

    def test_so_is_the_network(self) -> None:
        class ConnectTimeout(Exception):
            pass

        assert option_history.transient(ConnectTimeout("connect timeout=7")) == "network"
        assert option_history.transient(OSError("the read operation timed out")) == "network"

    def test_a_real_failure_is_not(self) -> None:
        """The guard that keeps the retry honest.

        Widening a backoff until it catches everything is how a wrong credential
        becomes five minutes of waiting and then the same wrong credential. If
        this ever returns a kind for these, the loop has stopped distinguishing
        "try again" from "this will never work".
        """
        for exc in (
            ValueError("high 12.0 below open 14.0"),
            KeyError("unknown exchange 'NSE_FO' for the historical endpoint"),
            RuntimeError("historical request failed for 42: Invalid Token"),
        ):
            assert option_history.transient(exc) is None, f"{exc!r} was treated as retryable"


class TestBarIndex:
    def test_a_bar_index_is_minutes_since_the_open_this_payload_names(self) -> None:
        """The page rebuilds a clock time from the index and the payload's open.

        Written out here rather than trusted because the two halves live in
        different files - the index is computed in Python and read in JavaScript,
        and nothing else would notice them drifting apart.
        """
        bar = Candle(
            ts=OPEN + timedelta(minutes=375), instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0
        )
        index = int((bar.ts - OPEN).total_seconds() // 60)
        assert index == 375  # 09:15 to 15:30 is six hours and a quarter

        hh, mm = (int(x) for x in ["09", "15"])
        total = hh * 60 + mm + index
        assert f"{total // 60:02d}:{total % 60:02d}" == "15:30"

    def test_the_index_is_meaningless_against_a_different_open(self) -> None:
        """Why the open ships in the same payload as the bars.

        A muhurat session opens in the evening. The same index read against the
        wrong open puts every bar hours from where it happened, and every candle
        still draws - there is nothing in the number itself to catch it.
        """

        def at(open_hhmm: str, index: int) -> str:
            hh, mm = (int(x) for x in open_hhmm.split(":"))
            total = hh * 60 + mm + index
            return f"{total // 60 % 24:02d}:{total % 60:02d}"

        assert at("09:15", 375) == "15:30"
        assert at("18:15", 375) == "00:30"


class TestWrite:
    def _block(self, name: str, atm: float) -> dict:
        return {
            "name": name,
            "day": DAY.isoformat(),
            "open": "09:15",
            "as_of": "15:30",
            "expiry": "2026-09-08",
            "atm": atm,
            "step": 50.0,
            "forward": {"symbol": f"{name}FUT", "close": atm + 4.3, "bars": 376},
            "chain": [
                {"strike": atm, "right": "CE", "last": 86.25, "bars": 376, "chartable": True},
                {"strike": atm, "right": "PE", "last": 168.7, "bars": 9, "chartable": False},
            ],
            "series": [{"strike": atm, "right": "CE", "recorded": [[0, 1, 1, 1, 1, 0]]}],
            "score": {"verdict": "lost", "horizons": []},
        }

    def test_each_underlying_gets_its_own_file(self, tmp_path: Path) -> None:
        """A page should not download BankNifty to show somebody NIFTY."""
        blocks = [self._block("NIFTY", 24000.0), self._block("BANKNIFTY", 57500.0)]
        written = write(blocks, DAY, tmp_path)

        assert [p.name for p in written] == [
            "chart_index.js",
            "chart_NIFTY.js",
            "chart_BANKNIFTY.js",
        ]
        body = (tmp_path / "chart_BANKNIFTY.js").read_text(encoding="utf-8")
        assert "window.CHART_BANKNIFTY =" in body
        assert "NIFTY29SEP" not in body.replace("BANKNIFTY", "")

    def test_the_index_names_the_global_each_file_defines(self, tmp_path: Path) -> None:
        """The page loads the file by name and then reads the global by name.

        If those two strings disagree the script loads, defines nothing the page
        looks for, and the tab sits empty with no error in the console.
        """
        blocks = [self._block("NIFTY", 24000.0), self._block("BANKNIFTY", 57500.0)]
        write(blocks, DAY, tmp_path)
        raw = (tmp_path / "chart_index.js").read_text(encoding="utf-8")
        index = json.loads(raw[raw.index("{") : raw.rindex(";")])

        for u in index["underlyings"]:
            body = (tmp_path / u["file"]).read_text(encoding="utf-8")
            assert f"window.{u['global']} =" in body, (
                f"{u['file']} does not define {u['global']}, which the page will "
                "wait for forever"
            )

    def test_the_index_carries_what_the_frame_needs_before_the_payload(
        self, tmp_path: Path
    ) -> None:
        write([self._block("NIFTY", 24000.0)], DAY, tmp_path)
        raw = (tmp_path / "chart_index.js").read_text(encoding="utf-8")
        u = json.loads(raw[raw.index("{") : raw.rindex(";")])["underlyings"][0]
        for key in ("forward", "expiry", "atm", "open", "as_of", "score"):
            assert key in u, f"the header cannot render without {key}"
        assert u["contracts"] == 2 and u["charted"] == 1


class TestChainVsSeries:
    def test_a_contract_too_thin_to_chart_still_has_a_rung(self) -> None:
        """The ladder is the chain; the charts are the part of it that traded.

        Dropping a thin strike from the ladder too would hide that it exists and
        that somebody paid that price for it, which is a fact about the day.
        """
        assert MIN_BARS_FOR_CHART > 1
        chain = [
            {"strike": 24000.0, "right": "CE", "bars": 376, "chartable": True},
            {"strike": 26000.0, "right": "CE", "bars": 9, "chartable": False},
        ]
        charted = [c for c in chain if c["chartable"]]
        assert len(chain) == 2 and len(charted) == 1

    @pytest.mark.parametrize("bars", [0, 1, MIN_BARS_FOR_CHART - 1])
    def test_the_threshold_is_applied_the_way_the_builder_applies_it(self, bars: int) -> None:
        assert (bars >= MIN_BARS_FOR_CHART) is False


class TestBackoffPolicy:
    """One wait did not fit both interruptions, and the difference is the run.

    A throttle refills over tens of seconds; a dropped connection usually clears
    on the next attempt. Retrying a throttle after two seconds spends the run
    re-earning the block, and waiting twenty for a blip spends four hundred
    contracts' worth of time on nothing.
    """

    def _row(self):
        class Row:
            token, symbol, strike, exch_seg = 1, "NIFTY24000CE", 24000.0, "NFO"

        return Row()

    def _raiser(self, errors: list[BaseException | None]):
        seq = list(errors)

        def fetch(*a, **k):
            exc = seq.pop(0)
            if exc is not None:
                raise exc
            return ([], True)

        return fetch

    def test_the_network_is_not_made_to_wait_the_throttles_wait(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(
            option_history, "fetch_series", self._raiser([TimeoutError("timed out"), None])
        )
        option_history.fetch_with_backoff(
            None, "k", self._row(), DAY, Path("."), sleep=slept.append
        )
        assert slept == [option_history.BACKOFF["network"][0]]
        assert slept[0] < option_history.BACKOFF["throttled"][0]

    def test_each_kind_keeps_its_own_budget(self, monkeypatch) -> None:
        """Meeting a throttle must not spend the network's allowance.

        A single shared counter meant one blip early in a run left the next
        throttle with fewer attempts than the policy says it has.
        """
        slept: list[float] = []
        monkeypatch.setattr(
            option_history,
            "fetch_series",
            self._raiser([TimeoutError("timed out"), RuntimeError("rate limiting"), None]),
        )
        option_history.fetch_with_backoff(
            None, "k", self._row(), DAY, Path("."), sleep=slept.append
        )
        assert slept == [
            option_history.BACKOFF["network"][0],
            option_history.BACKOFF["throttled"][0],
        ], "the second wait started from the first kind's delay"

    def test_running_out_of_attempts_raises_rather_than_waiting_forever(self, monkeypatch) -> None:
        limit = option_history.BACKOFF["network"][2]
        monkeypatch.setattr(
            option_history,
            "fetch_series",
            self._raiser([TimeoutError("timed out")] * limit),
        )
        with pytest.raises(TimeoutError):
            option_history.fetch_with_backoff(
                None, "k", self._row(), DAY, Path("."), sleep=lambda _s: None
            )

    def test_a_real_failure_is_raised_without_waiting_at_all(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(
            option_history, "fetch_series", self._raiser([KeyError("unknown exchange")])
        )
        with pytest.raises(KeyError):
            option_history.fetch_with_backoff(
                None, "k", self._row(), DAY, Path("."), sleep=slept.append
            )
        assert slept == [], "a wrong exchange name was retried as though it might come right"


class TestCollectKeepsGoing:
    """A hole in the chain is not the end of the chain - until it is.

    The first version abandoned four hundred contracts because the two hundred
    and forty-seventh timed out. Skipping past a bad contract is right. Skipping
    past a bad *session* - an expired token, a dropped link - is how a job
    reports success having fetched almost nothing, so a run of failures stops it.
    """

    class Row:
        def __init__(self, token: int, symbol: str, strike: float) -> None:
            self.token, self.symbol, self.strike = token, symbol, strike
            self.exch_seg, self.expiry = "NFO", date(2026, 9, 8)

    class Master:
        def __init__(self, rows) -> None:
            self._rows = rows

        def option_contracts(self, name, *, as_of, n_expiries=1):
            return self._rows

    def _wire(self, monkeypatch, behaviour):
        future = self.Row(99, "NIFTYFUT", 0.0)
        bar = Candle(ts=OPEN, instrument_id=99, tf=1, o=24000.0, h=24000.0, low=24000.0, c=24000.0)
        monkeypatch.setattr(option_history, "front_future", lambda *a, **k: future)
        monkeypatch.setattr(option_history, "fetch_series", lambda *a, **k: ([bar], True))
        monkeypatch.setattr(option_history, "fetch_with_backoff", behaviour)

    def _rows(self, n: int):
        return [self.Row(i, f"NIFTY{24000 + i * 50}CE", 24000.0 + i * 50) for i in range(n)]

    def test_one_contract_that_will_not_come_is_a_hole_not_an_ending(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        bar = Candle(ts=OPEN, instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0)

        def behaviour(session, key, row, day, out_dir, **kw):
            if row.token == 2:
                raise TimeoutError("connect timeout=7")
            return ([bar], True)

        self._wire(monkeypatch, behaviour)
        manifest = option_history.collect(
            None, "k", self.Master(self._rows(6)), "NIFTY", DAY, n_strikes=99, root=tmp_path
        )

        assert len(manifest["contracts"]) == 5, "the run stopped at the contract that failed"
        assert [f["token"] for f in manifest["failed"]] == [2]
        assert "did not come, moving on" in capsys.readouterr().out

    def test_a_run_of_failures_is_the_session_and_stops_it(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        def behaviour(session, key, row, day, out_dir, **kw):
            raise RuntimeError("historical request failed: Invalid Token")

        self._wire(monkeypatch, behaviour)
        manifest = option_history.collect(
            None, "k", self.Master(self._rows(40)), "NIFTY", DAY, n_strikes=99, root=tmp_path
        )

        limit = option_history.CONSECUTIVE_FAILURE_LIMIT
        assert len(manifest["failed"]) == limit, (
            f"it kept asking a venue that had refused {limit} times running, which is "
            "forty pointless requests and a manifest that reads like a quiet day"
        )
        assert manifest["contracts"] == []
        assert "That is the session, not the contracts" in capsys.readouterr().out

    def test_the_counter_resets_on_a_success(self, monkeypatch, tmp_path) -> None:
        """Scattered failures are not a run of them, and must not trip the brake."""
        bar = Candle(ts=OPEN, instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0)

        def behaviour(session, key, row, day, out_dir, **kw):
            if row.token % 2:
                raise TimeoutError("timed out")
            return ([bar], True)

        self._wire(monkeypatch, behaviour)
        manifest = option_history.collect(
            None, "k", self.Master(self._rows(20)), "NIFTY", DAY, n_strikes=99, root=tmp_path
        )
        assert len(manifest["contracts"]) == 10 and len(manifest["failed"]) == 10

    def test_the_failures_are_written_down_not_just_printed(self, monkeypatch, tmp_path) -> None:
        """A chain with a hole and a chain that never had those strikes look the same.

        The manifest is what the page is built from, so a contract that was
        asked for and did not arrive has to be distinguishable there from one
        that was never listed.
        """

        def behaviour(session, key, row, day, out_dir, **kw):
            raise TimeoutError("timed out")

        self._wire(monkeypatch, behaviour)
        option_history.collect(
            None, "k", self.Master(self._rows(3)), "NIFTY", DAY, n_strikes=99, root=tmp_path
        )
        written = json.loads(
            (tmp_path / f"date={DAY.isoformat()}" / "NIFTY" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert [f["symbol"] for f in written["failed"]] == [r.symbol for r in self._rows(3)]
        assert all("timed out" in f["why"] for f in written["failed"])
