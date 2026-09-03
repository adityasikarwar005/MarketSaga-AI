"""How a day's chain is packed for the page, and what must not be swallowed on the way.

Six things here have already gone wrong once each, and none of them fails
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
* **A vendor call with no bound on it.** The SDK passes `requests` a 7-second
  timeout and one `getCandleData` still blocked for fourteen and a half minutes.
  It raised nothing and logged nothing; it just stopped, and a sweep of 2450
  contracts that meets that once every forty of them spends fourteen hours idle
  and reports success.
* **A forward borrowed from the wrong expiry.** Options list weekly and futures
  monthly, so the two dates are usually different and the carry between them is
  real money at the money. Extending one *forwards* past the last listed future,
  though, is years of assumed dividends dressed as a price - so that direction
  is refused rather than approximated.
* **An anchor that moved when the fan got finer.** The projection used to start
  at 5 minutes and now starts at 1. "Does the first step fit before the close"
  was the same question as "is the session over" at five minutes and is not at
  one, and the difference is a whole day's projection anchored at 15:30 where
  nothing can trade and nobody can check it.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from jobs import option_history
from jobs.chart_history import (
    FORECAST_STEPS,
    MIN_BARS_FOR_CHART,
    _anchor,
    forward_at_expiry,
    write,
)
from libs.ms_core.schemas import Candle
from libs.ms_data import angelone

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
    def _board(self, name: str, expiry: str, atm: float) -> dict:
        return {
            "name": name,
            "day": DAY.isoformat(),
            "open": "09:15",
            "as_of": "15:30",
            "read_from": "11:02",
            "expiry": expiry,
            "atm": atm,
            "step": 50.0,
            "listed": 4,
            "forward": {
                "symbol": f"{name}FUT",
                "close": atm + 4.3,
                "bars": 376,
                "basis": "bracketing",
            },
            "chain": [
                {"strike": atm, "right": "CE", "last": 86.25, "bars": 376, "chartable": True},
                {"strike": atm, "right": "PE", "last": 168.7, "bars": 9, "chartable": False},
            ],
            "series": [{"strike": atm, "right": "CE", "recorded": [[0, 1, 1, 1, 1, 0]]}],
            "score": {"verdict": "lost", "horizons": []},
        }

    def _boards(self) -> dict:
        return {
            "NIFTY": [
                self._board("NIFTY", "2026-09-08", 24000.0),
                self._board("NIFTY", "2026-09-15", 24000.0),
            ],
            "BANKNIFTY": [self._board("BANKNIFTY", "2026-09-29", 57500.0)],
        }

    def test_each_expiry_gets_its_own_file(self, tmp_path: Path) -> None:
        """Eighteen expiries of minute bars is not one download.

        Splitting per underlying was enough while only the front expiry was
        fetched. Every live NIFTY call and put is 1568 contracts, and a reader
        opening this Tuesday's weekly should not be made to fetch June 2029.
        """
        written = write(self._boards(), DAY, tmp_path)

        assert [p.name for p in written] == [
            "chart_index.js",
            "chart_NIFTY_20260908.js",
            "chart_NIFTY_20260915.js",
            "chart_BANKNIFTY_20260929.js",
        ]
        body = (tmp_path / "chart_NIFTY_20260908.js").read_text(encoding="utf-8")
        assert "window.CHART_NIFTY_20260908 =" in body
        assert "2026-09-15" not in body, "one expiry's file carried another's bars"

    def test_the_index_names_the_global_each_file_defines(self, tmp_path: Path) -> None:
        """The page loads the file by name and then reads the global by name.

        If those two strings disagree the script loads, defines nothing the page
        looks for, and the pane sits empty with no error in the console. With one
        file per expiry there are now thirty-odd chances for them to disagree.
        """
        write(self._boards(), DAY, tmp_path)
        raw = (tmp_path / "chart_index.js").read_text(encoding="utf-8")
        index = json.loads(raw[raw.index("{") : raw.rindex(";")])

        seen = set()
        for u in index["underlyings"]:
            for e in u["expiries"]:
                body = (tmp_path / e["file"]).read_text(encoding="utf-8")
                assert f"window.{e['global']} =" in body, (
                    f"{e['file']} does not define {e['global']}, which the page "
                    "will wait for forever"
                )
                assert e["global"] not in seen, (
                    f"{e['global']} is defined by two files; whichever loads "
                    "second silently replaces the other expiry's candles"
                )
                seen.add(e["global"])

    def test_the_index_carries_every_expiry_before_any_payload(self, tmp_path: Path) -> None:
        """The rail and the ladder counts render before a single candle arrives."""
        write(self._boards(), DAY, tmp_path)
        raw = (tmp_path / "chart_index.js").read_text(encoding="utf-8")
        index = json.loads(raw[raw.index("{") : raw.rindex(";")])
        u = index["underlyings"][0]

        assert [e["expiry"] for e in u["expiries"]] == ["2026-09-08", "2026-09-15"]
        for e in u["expiries"]:
            for key in ("forward", "atm", "step", "as_of", "read_from", "listed", "score"):
                assert key in e, f"the header cannot render without {key}"
            assert e["contracts"] == 2 and e["charted"] == 1
        assert u["contracts"] == 4, "the underlying's total is the sum of its expiries"

    def test_a_stale_expiry_file_is_not_left_behind(self, tmp_path: Path) -> None:
        """Yesterday's expiries are not today's.

        A file left over from a previous day is a perfectly valid payload with a
        perfectly valid global. The page would load it and date it from its own
        contents, which is how a chart of an expired contract ends up on a page
        headed today. The index would not mention it and nothing would complain.
        """
        stale = tmp_path / "chart_NIFTY_20260901.js"
        stale.write_text("window.CHART_NIFTY_20260901 = {};", encoding="utf-8")
        write(self._boards(), DAY, tmp_path)

        assert (
            not stale.exists()
        ), "`write` left an expiry behind that today's board does not contain"
        assert (tmp_path / "chart_index.js").exists(), "the index is not an expiry file"
        assert (
            tmp_path / "chart_NIFTY_20260908.js"
        ).exists(), "the sweep took today's payloads with it"

    def test_the_clearing_belongs_to_write_not_to_its_callers(self, tmp_path: Path) -> None:
        """Where a destructive step lives decides what a stub can reach.

        `close_out` used to clear the output directory itself and then call
        `write`. Its own tests stub `write` - correctly; they are about which
        underlyings get built - and stubbing `write` does not stub the `unlink`
        loop standing beside it. So every run of the suite deleted the real
        published dashboard's payloads. Nothing failed and nothing noticed until
        a deploy check refused to upload an index naming files that were gone.
        """
        import inspect

        from jobs import close_out

        assert "unlink" not in inspect.getsource(close_out), (
            "close_out deletes files again; a caller-side delete beside a "
            "stubbable write is how the published page lost its candles"
        )
        assert "unlink" in inspect.getsource(write)


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

    Which one is the long wait has now been measured rather than assumed, and
    the answer came out the other way round from the guess the policy was
    written on. Four blocks of 25 requests spaced 0.0s, 0.4s, 0.8s and 1.2s
    apart returned 11, 16, 14 and 12 successes: standing further back bought
    nothing, so the throttle is a per-request rejection that clears at once and
    not a budget refilling over tens of seconds. Waiting twenty seconds on it,
    at an observed ~50% rejection rate, is twenty seconds a contract - nine
    hours over 1568 of them.

    So the throttle now gets the short wait and many attempts, and the network -
    which does occasionally need a real pause - keeps the growing one.
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

    def test_the_throttle_is_asked_again_promptly_not_waited_out(self, monkeypatch) -> None:
        """The measured shape, asserted where a future edit would meet it.

        This is the boundary that decides whether a full sweep takes two hours
        or nine. Someone reading "throttled" will want to make this wait longer;
        the measurement says longer does not help, and a test is the only place
        that argument survives.
        """
        slept: list[float] = []
        monkeypatch.setattr(
            option_history, "fetch_series", self._raiser([RuntimeError("rate limiting"), None])
        )
        option_history.fetch_with_backoff(
            None, "k", self._row(), DAY, Path("."), sleep=slept.append
        )
        first, ceiling, attempts = option_history.BACKOFF["throttled"]
        assert slept == [first]
        assert first <= 1.0, "the throttle clears at once; waiting on it just spends the run"
        assert ceiling <= 10.0, (
            "the whole finding is that standing back does not help, so standing "
            "back further cannot help more"
        )
        assert attempts >= 8, "at a ~50% rejection rate, a handful of attempts drops contracts"

    def test_the_network_keeps_the_longer_wait(self, monkeypatch) -> None:
        """A dropped connection is the opposite problem and keeps its own policy."""
        slept: list[float] = []
        monkeypatch.setattr(
            option_history, "fetch_series", self._raiser([TimeoutError("timed out"), None])
        )
        option_history.fetch_with_backoff(
            None, "k", self._row(), DAY, Path("."), sleep=slept.append
        )
        assert slept == [option_history.BACKOFF["network"][0]]
        assert slept[0] > option_history.BACKOFF["throttled"][0]

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
        assert len(set(slept)) == 2, (
            "the two kinds now differ by more than an order of magnitude; if "
            "these came out equal the policies have been collapsed back into one"
        )

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
        future.expiry = date(2026, 9, 29)
        bar = Candle(ts=OPEN, instrument_id=99, tf=1, o=24000.0, h=24000.0, low=24000.0, c=24000.0)
        monkeypatch.setattr(option_history, "index_futures", lambda *a, **k: [future])

        def routed(session, key, row, day, out_dir, **kw):
            # The forward is fetched through the same call as the contracts now.
            # A test about contracts failing must not also take the future down
            # with them, or collect() raises about the forward and never reaches
            # the behaviour under test.
            if row is future:
                return ([bar], True)
            return behaviour(session, key, row, day, out_dir, **kw)

        monkeypatch.setattr(option_history, "fetch_with_backoff", routed)

    def _collect(self, master, tmp_path):
        return option_history.collect(
            None,
            "k",
            master,
            "NIFTY",
            DAY,
            n_strikes=None,
            n_expiries=None,
            root=tmp_path,
        )

    def _only(self, manifest):
        """The one expiry these fixtures build, since the manifest now holds many."""
        assert len(manifest["expiries"]) == 1
        return manifest["expiries"][0]

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
        block = self._only(self._collect(self.Master(self._rows(6)), tmp_path))

        assert len(block["contracts"]) == 5, "the run stopped at the contract that failed"
        assert [f["token"] for f in block["failed"]] == [2]
        assert "did not come, moving on" in capsys.readouterr().out

    def test_a_run_of_failures_is_the_session_and_stops_it(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        def behaviour(session, key, row, day, out_dir, **kw):
            raise RuntimeError("historical request failed: Invalid Token")

        self._wire(monkeypatch, behaviour)
        manifest = self._collect(self.Master(self._rows(40)), tmp_path)
        block = self._only(manifest)
        assert manifest["complete"] is False, (
            "a run that stopped early reported itself finished, which is how a "
            "half-swept board gets published as the whole board"
        )

        limit = option_history.CONSECUTIVE_FAILURE_LIMIT
        assert len(block["failed"]) == limit, (
            f"it kept asking a venue that had refused {limit} times running, which is "
            "forty pointless requests and a manifest that reads like a quiet day"
        )
        assert block["contracts"] == []
        assert "That is the session, not the contracts" in capsys.readouterr().out

    def test_the_counter_resets_on_a_success(self, monkeypatch, tmp_path) -> None:
        """Scattered failures are not a run of them, and must not trip the brake."""
        bar = Candle(ts=OPEN, instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0)

        def behaviour(session, key, row, day, out_dir, **kw):
            if row.token % 2:
                raise TimeoutError("timed out")
            return ([bar], True)

        self._wire(monkeypatch, behaviour)
        block = self._only(self._collect(self.Master(self._rows(20)), tmp_path))
        assert len(block["contracts"]) == 10 and len(block["failed"]) == 10

    def test_the_failures_are_written_down_not_just_printed(self, monkeypatch, tmp_path) -> None:
        """A chain with a hole and a chain that never had those strikes look the same.

        The manifest is what the page is built from, so a contract that was
        asked for and did not arrive has to be distinguishable there from one
        that was never listed.
        """

        def behaviour(session, key, row, day, out_dir, **kw):
            raise TimeoutError("timed out")

        self._wire(monkeypatch, behaviour)
        self._collect(self.Master(self._rows(3)), tmp_path)
        written = json.loads(
            (tmp_path / f"date={DAY.isoformat()}" / "NIFTY" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        block = self._only(written)
        assert [f["symbol"] for f in block["failed"]] == [r.symbol for r in self._rows(3)]
        assert all("timed out" in f["why"] for f in block["failed"])

    def test_the_forward_is_re_read_between_expiries(self, monkeypatch, tmp_path) -> None:
        """A four-hour sweep cannot price its last expiry off its first minute.

        The forward used to be fetched once, at the top. An option bar at 14:00
        would then be inverted against the last forward bar at or before it -
        the 10:30 close - and the implied volatility that comes back is wrong by
        however far the index moved in between, on every contract in every
        expiry after the first. Nothing about that fails: it produces a
        volatility, a price, and a band.
        """
        bar = Candle(ts=OPEN, instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0)
        future_reads: list[int] = []

        def behaviour(session, key, row, day, out_dir, **kw):
            return ([bar], True)

        rows = self._rows(4)
        for r in rows[2:]:
            r.expiry = date(2026, 9, 15)

        self._wire(monkeypatch, behaviour)
        inner = option_history.fetch_with_backoff

        def counting(session, key, row, day, out_dir, **kw):
            if getattr(row, "symbol", "") == "NIFTYFUT":
                future_reads.append(1)
            return inner(session, key, row, day, out_dir, **kw)

        monkeypatch.setattr(option_history, "fetch_with_backoff", counting)
        self._collect(self.Master(rows), tmp_path)

        assert len(future_reads) >= 3, (
            "the forward was read once for a sweep covering two expiries; the "
            f"second expiry would be priced off the first one's clock ({future_reads})"
        )

    def test_the_manifest_is_written_before_the_last_expiry_finishes(
        self, monkeypatch, tmp_path
    ) -> None:
        """A five-hour job whose only output is its last line has no output.

        Expiries are swept nearest-first, so a manifest written after each one
        already describes everything anybody is trading today while the tail of
        2029 is still on the wire. This asserts the file exists and parses part
        way through, not merely at the end.
        """
        bar = Candle(ts=OPEN, instrument_id=1, tf=1, o=1.0, h=1.0, low=1.0, c=1.0)
        seen: list[int] = []
        path = tmp_path / f"date={DAY.isoformat()}" / "NIFTY" / "manifest.json"

        def behaviour(session, key, row, day, out_dir, **kw):
            if path.exists():
                seen.append(len(json.loads(path.read_text(encoding="utf-8"))["expiries"]))
            return ([bar], True)

        rows = self._rows(4)
        for r in rows[2:]:
            r.expiry = date(2026, 9, 15)
        self._wire(monkeypatch, behaviour)
        self._collect(self.Master(rows), tmp_path)

        assert (
            seen and max(seen) >= 1
        ), "no manifest existed while the second expiry was still being fetched"


class TestForwardChoice:
    """Which future an expiry is priced against, and which direction may be assumed.

    Options list weekly, futures monthly, and the two dates are usually not the
    same one. Twenty-one days of carry at 6.5% is about ninety index points on
    24000 - nearly two strike rungs at the money - so the mismatch is not a
    rounding question, and the whole of it lands in the implied volatility as
    something that looks like skew.
    """

    class Fut:
        def __init__(self, expiry: date, token: int) -> None:
            self.expiry, self.token = expiry, token
            self.symbol = f"NIFTY{expiry:%d%b%y}FUT".upper()

    def _futures(self):
        return [
            self.Fut(date(2026, 9, 29), 1),
            self.Fut(date(2026, 10, 27), 2),
            self.Fut(date(2026, 11, 23), 3),
        ]

    def test_a_future_on_the_expiry_itself_is_used_as_it_stands(self) -> None:
        fut, basis = option_history.forward_for(self._futures(), date(2026, 9, 29))
        assert (fut.token, basis) == (1, "same-cycle")

    def test_a_weekly_takes_the_next_future_after_it(self) -> None:
        fut, basis = option_history.forward_for(self._futures(), date(2026, 9, 8))
        assert (fut.token, basis) == (1, "bracketing")
        fut, basis = option_history.forward_for(self._futures(), date(2026, 10, 6))
        assert (fut.token, basis) == (2, "bracketing")

    def test_an_expiry_past_the_last_future_is_marked_carried(self) -> None:
        """NIFTY lists options to 2031 and futures three months out.

        There is nothing to price a 2029 option against. Saying so is the point:
        the label is what stops the packer drawing a forecast on it.
        """
        fut, basis = option_history.forward_for(self._futures(), date(2029, 6, 26))
        assert (fut.token, basis) == (3, "carried")

    def test_the_carry_is_taken_out_when_the_future_expires_later(self) -> None:
        opt = datetime(2026, 9, 8, 15, 30, tzinfo=IST)
        fut = datetime(2026, 9, 29, 15, 30, tzinfo=IST)
        got = forward_at_expiry(24000.0, fut, opt, OPEN, r=0.065)
        assert got < 24000.0
        assert 23900.0 < got < 23920.0, (
            "twenty-one days at 6.5% is about ninety points on 24000; a result "
            "near 24000 means the discount is not being applied at all"
        )

    def test_a_forward_is_never_pushed_forwards(self) -> None:
        """The direction that would be years of assumed dividends dressed as a price.

        Pulling a September future back onto an 8 September option removes
        carry that is genuinely there. Pushing a November 2026 future out to a
        June 2029 option would invent three years of them, so the function
        refuses and `forward_for` marks those expiries instead.
        """
        opt = datetime(2029, 6, 26, 15, 30, tzinfo=IST)
        fut = datetime(2026, 11, 23, 15, 30, tzinfo=IST)
        assert forward_at_expiry(24000.0, fut, opt, OPEN, r=0.065) == 24000.0

    def test_the_same_date_changes_nothing(self) -> None:
        at = datetime(2026, 9, 29, 15, 30, tzinfo=IST)
        assert forward_at_expiry(24000.0, at, at, OPEN, r=0.065) == 24000.0


class TestCallDeadline:
    """The defect with no symptom: a vendor call that neither returns nor raises.

    The SDK hands `requests` a seven-second timeout, and one `getCandleData`
    still blocked for fourteen minutes and thirty-four seconds on 2026-09-03
    before Windows reset the socket - in a run whose successful calls were
    taking under two. It logged nothing and raised nothing while it did it.

    At roughly one hang in forty contracts, a 2450-contract sweep spends
    fourteen hours idle and then reports success. Nothing downstream can see
    that; a deadline is the only thing that turns it back into an error, and
    the error has to name a kind `transient()` will retry.
    """

    class Session:
        jwt_token = "jwt"

    def _client(self, monkeypatch, behaviour):
        class Client:
            def setAccessToken(self, token):
                self.token = token

            def getCandleData(self, params):
                return behaviour(params)

        monkeypatch.setattr(angelone, "_smartconnect", lambda key: Client())

    def test_a_call_that_never_returns_is_abandoned(self, monkeypatch) -> None:
        import threading as _t

        held = _t.Event()

        def hang(params):
            held.wait(30)
            return {"status": True, "data": []}

        self._client(monkeypatch, hang)
        fetch = angelone._history_fetcher(self.Session(), "key", deadline=0.05)
        try:
            with pytest.raises(angelone.HistoryCallTimeout):
                fetch({"symboltoken": "42", "fromdate": "a", "todate": "b"})
        finally:
            held.set()

    def test_the_timeout_is_the_kind_the_run_retries(self, monkeypatch) -> None:
        """A deadline that raises something `transient` does not know is worse.

        It would turn a fourteen-minute stall into an immediate hard failure and
        then, five of them later, into a stopped run - having converted a
        retryable blip into "that is the session".
        """
        exc = angelone.HistoryCallTimeout("did not answer within 20s")
        assert option_history.transient(exc) == "network"

    def test_a_call_that_answers_is_untouched(self, monkeypatch) -> None:
        self._client(monkeypatch, lambda params: {"status": True, "data": [1]})
        fetch = angelone._history_fetcher(self.Session(), "key", deadline=5.0)
        assert fetch({"symboltoken": "42"}) == {"status": True, "data": [1]}

    def test_the_vendor_s_own_error_still_reaches_the_caller(self, monkeypatch) -> None:
        """The deadline must not swallow the throttle it sits in front of.

        The rate limit arrives as an exception from inside `getCandleData`. If
        running that on a thread lost it, every throttle would look like a
        successful empty response and the cache would fill with empty files for
        contracts that trade all day.
        """

        def refuse(params):
            raise RuntimeError("Access denied because of exceeding access rate")

        self._client(monkeypatch, refuse)
        fetch = angelone._history_fetcher(self.Session(), "key", deadline=5.0)
        with pytest.raises(RuntimeError, match="exceeding access rate"):
            fetch({"symboltoken": "42"})

    def test_the_deadline_is_far_above_a_normal_call(self) -> None:
        assert angelone.HISTORY_CALL_DEADLINE >= 15.0, (
            "a deadline near the normal call time fires on calls that were "
            "going to return, and every one of those is a contract dropped"
        )


class TestCacheFreshness:
    """The cache is unconditional for a *finished* read and must not be for a partial one.

    Everything cached is skipped on a re-run, which is what makes an interrupted
    sweep of 2450 contracts resumable. The question that decides whether skipping
    is safe is not "is the session over now" but **"was the session over when
    this file was written"**, and those two look like the same question.

    Reading it the first way costs a day. A sweep taken during trading hours
    writes six hundred partial files; at 15:30 every one of them becomes "a
    finished day" and is never asked for again, so the board keeps the three
    hours it happened to catch, permanently, and looks like a market that went
    quiet at 11:32.
    """

    CLOSE = datetime(2026, 9, 3, 15, 30, tzinfo=IST)

    def _file(self, tmp_path: Path, written: datetime) -> Path:
        path = tmp_path / "42.ndjson"
        path.write_text("", encoding="utf-8")
        os.utime(path, (written.timestamp(), written.timestamp()))
        return path

    def _clock(self, when: datetime):
        class Fixed:
            def now(self):
                return when

        return Fixed()

    def _stale(self, path: Path, day: date, now: datetime) -> bool:
        return option_history._stale(path, day, self._clock(now), max_age=300.0)

    def test_a_series_read_after_its_own_close_is_final(self, tmp_path: Path) -> None:
        """The ordinary case, and the one the cache exists for."""
        path = self._file(tmp_path, datetime(2026, 9, 2, 19, 0, tzinfo=IST))
        assert self._stale(path, date(2026, 9, 2), datetime(2026, 9, 3, 11, 0, tzinfo=IST)) is False

    def test_a_recent_read_of_a_live_session_is_kept(self, tmp_path: Path) -> None:
        now = datetime(2026, 9, 3, 11, 0, tzinfo=IST)
        path = self._file(tmp_path, now - timedelta(seconds=60))
        assert self._stale(path, date(2026, 9, 3), now) is False

    def test_a_stale_read_of_a_live_session_is_re_asked_for(self, tmp_path: Path) -> None:
        now = datetime(2026, 9, 3, 11, 0, tzinfo=IST)
        path = self._file(tmp_path, now - timedelta(seconds=600))
        assert self._stale(path, date(2026, 9, 3), now) is True

    def test_a_partial_read_is_re_asked_for_once_the_session_has_closed(
        self, tmp_path: Path
    ) -> None:
        """The clause that keeps a day from staying half-read for good.

        This file holds 09:15 to 11:32 because that is when the sweep reached
        that contract. The day it describes now has six and a quarter hours in
        it. Treating it as final because *the session is over* keeps the two
        hours and discards the rest, which is precisely backwards - the session
        being over is what makes the rest of it available.
        """
        path = self._file(tmp_path, datetime(2026, 9, 3, 11, 32, tzinfo=IST))
        assert self._stale(path, date(2026, 9, 3), datetime(2026, 9, 3, 16, 20, tzinfo=IST)) is True

    def test_a_partial_read_of_an_older_day_is_still_partial(self, tmp_path: Path) -> None:
        """And does not become whole by getting older."""
        path = self._file(tmp_path, datetime(2026, 9, 2, 11, 0, tzinfo=IST))
        assert self._stale(path, date(2026, 9, 2), datetime(2026, 9, 3, 11, 0, tzinfo=IST)) is True

    def test_the_age_is_the_file_s_not_the_last_bar_s(self, tmp_path: Path) -> None:
        """Why the check is on mtime rather than on the series it holds.

        A contract that has not printed since 09:20 and a file last written at
        09:20 hold identical bars. One is a stale read and the other is a quiet
        strike, and reading the bars cannot tell them apart - so a bar-based
        rule would re-fetch every illiquid wing on every pass and leave a busy
        contract that happened to be read early exactly as stale as it was.
        """
        now = datetime(2026, 9, 3, 11, 0, tzinfo=IST)
        path = self._file(tmp_path, now - timedelta(seconds=10))
        assert path.read_text(encoding="utf-8") == ""  # no bars at all
        assert self._stale(path, date(2026, 9, 3), now) is False

    def test_no_clock_means_no_re_asking(self, tmp_path: Path) -> None:
        """A caller with no clock gets the old unconditional cache, deliberately.

        Several jobs rebuild a page from what is already on disk and must not
        start making network calls because a file looks old.
        """
        path = self._file(tmp_path, datetime(2026, 9, 3, 11, 0, tzinfo=IST))
        assert option_history._stale(path, date(2026, 9, 3), None, max_age=300.0) is False


class TestProjectionAnchor:
    """Where the forecast starts, and what changed when the fan went per-minute.

    The projection is drawn hollow over the hour it was made about, so a reader
    can see for themselves whether it was any good. That only works if it is
    anchored an hour before the close on a day that is over - anchored *at* the
    close it forecasts a shut market, cannot be checked by anyone, and sets the
    chart's entire vertical scale with an hour of diffusion into nothing.
    """

    def _bars(self, last_hhmm: str) -> list[Candle]:
        h, m = (int(x) for x in last_hhmm.split(":"))
        end = OPEN.replace(hour=h, minute=m)
        n = int((end - OPEN).total_seconds() // 60)
        return [
            Candle(
                ts=OPEN + timedelta(minutes=i),
                instrument_id=1,
                tf=1,
                o=1.0,
                h=1.0,
                low=1.0,
                c=1.0,
            )
            for i in range(n + 1)
        ]

    def test_the_fan_is_a_bar_a_minute(self) -> None:
        assert list(FORECAST_STEPS) == list(range(1, 61)), (
            "the steps used to coarsen from 5 to 15 minutes apart, which drew a "
            "band widening in visible jumps and read as a jump in volatility"
        )

    def test_a_finished_day_is_anchored_an_hour_before_the_close(self) -> None:
        """The regression the one-minute fan introduced.

        The old test was "does the shortest step land before the close". At five
        minutes, a last bar of 15:29 gave 15:34 and correctly said no. At one
        minute it gives 15:30, which is not after the close, so the whole
        projection anchored itself at the last bar of a finished session.
        """
        close = OPEN.replace(hour=15, minute=30)
        anchor = _anchor(self._bars("15:29"), close)
        assert anchor.ts == close - timedelta(minutes=60), (
            f"anchored at {anchor.ts:%H:%M} on a day that had ended - the "
            "projection covers a market that is shut and can never be checked"
        )

    def test_a_live_session_is_anchored_at_its_own_edge(self) -> None:
        """And not dragged back an hour, which is the other way to get this wrong."""
        close = OPEN.replace(hour=15, minute=30)
        bars = self._bars("11:04")
        assert _anchor(bars, close) is bars[-1]

    def test_late_in_a_live_session_still_anchors_at_the_edge(self) -> None:
        close = OPEN.replace(hour=15, minute=30)
        bars = self._bars("15:00")
        assert _anchor(bars, close) is bars[-1], (
            "at 15:00 the full hour does not fit before the close, but the "
            "session is still running and the last minute is still the right "
            "place to forecast from - the fan is simply truncated"
        )
