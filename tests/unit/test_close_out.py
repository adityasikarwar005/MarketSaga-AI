"""Closing out a day from history - the guards, not the arithmetic.

The arithmetic is `jobs/chart_history.py`'s and is tested there. What this job
owns is the set of conditions under which it must *not* proceed, and each one is
a way a scheduled job quietly does damage while reporting success:

* running on a day the exchange was shut,
* running with an environment that cannot hold a credential,
* replacing a good page with an empty one because a fetch failed,
* filing half a session as the session.

It is scheduled, so nobody watches it. Everything it refuses to do has to be
refused in the log rather than discovered in the page.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from jobs import close_out
from libs.ms_core import dotenv

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 9, 2)


@pytest.fixture(autouse=True)
def pristine_environ():
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


class _Calendar:
    def __init__(self, *, trading: bool = True, closed_hour: int = 15) -> None:
        self._trading = trading
        self._closed_hour = closed_hour

    def is_trading_day(self, day: date) -> bool:
        return self._trading

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        opened = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        return opened, opened.replace(hour=self._closed_hour, minute=30)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A job with its calendar, credentials and network stubbed out."""
    env_file = tmp_path / ".env"
    env_file.write_text("MARKETSAGE_ENV=paper\nANGEL_API_KEY=k\n", encoding="utf-8")
    os.environ.pop("MARKETSAGE_ENV", None)
    monkeypatch.setattr(dotenv, "ENV_FILE", env_file)
    monkeypatch.setattr(close_out, "LOG_DIR", tmp_path / "sessions")
    monkeypatch.setattr(close_out, "_calendar", lambda: _Calendar())
    monkeypatch.setattr(close_out, "load_scrip_master", lambda *a, **k: _Master())
    monkeypatch.setattr(close_out, "login", lambda *a, **k: object())
    monkeypatch.setattr(
        close_out.AngelCredentials, "from_env", staticmethod(lambda *a, **k: _Creds())
    )
    return tmp_path


class _Creds:
    api_key = "k"


class _Master:
    fetched_on = DAY


def _log(tmp_path, day: date = DAY) -> str:
    path = tmp_path / "sessions" / f"closeout_{day.isoformat()}.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_a_closed_exchange_is_not_a_day_to_close_out(monkeypatch, wired):
    """The holiday list is the project's, not the scheduler's.

    A task set to fire every weekday must do the right thing on Ganesh
    Chaturthi without anybody remembering to disable it.
    """
    monkeypatch.setattr(close_out, "_calendar", lambda: _Calendar(trading=False))
    called: list[Any] = []
    monkeypatch.setattr(close_out, "login", lambda *a, **k: called.append(1))

    assert close_out.main(["--date", DAY.isoformat()]) == 0
    assert not called, "it logged in on a day the exchange was shut"
    assert "not a trading day" in _log(wired)


def test_an_environment_that_cannot_hold_a_credential_stops_it(monkeypatch, wired):
    os.environ["MARKETSAGE_ENV"] = "not-an-environment"
    assert close_out.main(["--date", DAY.isoformat()]) == 2
    assert "REFUSING" in _log(wired)


class _At:
    """A clock stopped at one instant. See close_out.main's docstring.

    This used to be the real one, and the test therefore asserted something
    about the day it was written on rather than about the job. It passed on
    2026-09-02 and failed on 2026-09-03, having caught nothing in between.
    """

    def __init__(self, when: datetime) -> None:
        self._when = when

    def now(self) -> datetime:
        return self._when


def test_a_session_still_running_is_called_out(monkeypatch, wired):
    """A partial day charts fine. Filed silently as the day, it is a lie."""
    monkeypatch.setattr(close_out, "_calendar", lambda: _Calendar(closed_hour=23))
    midday = datetime(DAY.year, DAY.month, DAY.day, 12, 0, tzinfo=IST)
    assert close_out.main(["--date", DAY.isoformat(), "--dry-run"], clock=_At(midday)) == 0
    log = _log(wired)
    assert "still open" in log
    assert "690 more minutes" in log, "the warning did not say how much of the day was missing"
    assert "the day so far, not the day" in log


def test_a_finished_session_is_not_called_out(monkeypatch, wired):
    """The other half of the same guard, which nothing asserted before.

    A warning that fires on every run is a warning nobody reads, and this one is
    the difference between "this is the day" and "this is the day so far".
    """
    monkeypatch.setattr(close_out, "_calendar", lambda: _Calendar(closed_hour=15))
    evening = datetime(DAY.year, DAY.month, DAY.day, 20, 0, tzinfo=IST)
    assert close_out.main(["--date", DAY.isoformat(), "--dry-run"], clock=_At(evening)) == 0
    log = _log(wired)
    assert "still open" not in log
    assert "the day so far, not the day" not in log


def test_a_failed_fetch_does_not_overwrite_the_page(monkeypatch, wired):
    """The page on disk is the last good day. An empty one is strictly worse.

    A scheduled job that rewrites its output before checking it had anything to
    write is how a working dashboard becomes a blank one overnight, with a
    success exit code.
    """
    monkeypatch.setattr(
        close_out.option_history,
        "collect",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("no cached history")),
    )
    wrote: list[Any] = []
    monkeypatch.setattr(close_out.chart_history, "write", lambda *a, **k: wrote.append(1))

    assert close_out.main(["--date", DAY.isoformat()]) == 1
    assert not wrote, "the chart was rewritten after every underlying failed to fetch"
    assert "Not rewriting" in _log(wired)


def test_one_underlying_failing_does_not_take_the_other_down(monkeypatch, wired):
    """Half a page beats none, as long as the log says which half."""

    def collect(session, key, master, name, day, **kw):
        if name == "BANKNIFTY":
            raise SystemExit("the front future returned no bars")
        return {"underlying": name}

    monkeypatch.setattr(close_out.option_history, "collect", collect)
    # `build` returns one board per expiry now, and `write` takes them keyed by
    # underlying - the page is split per expiry rather than per underlying.
    monkeypatch.setattr(
        close_out.chart_history,
        "build",
        lambda day, name, root: [
            {
                "name": name,
                "expiry": "2026-09-08",
                "chain": [{"strike": 1.0, "right": "CE"}],
                "series": [{"recorded": [1, 2, 3]}],
                "forward": {"close": 24000.0},
                "score": {"verdict": "did not beat anything"},
            }
        ],
    )
    built: dict[str, Any] = {}

    def write(boards, *a):
        built.update(boards)
        return [
            Path(f"chart_{name}_{b['expiry'].replace('-', '')}.js")
            for name, group in boards.items()
            for b in group
        ]

    monkeypatch.setattr(close_out.chart_history, "write", write)

    assert close_out.main(["--date", DAY.isoformat()]) == 0
    assert list(built) == ["NIFTY"]
    assert "BANKNIFTY" in _log(wired)


def test_it_loads_dotenv_before_reaching_for_a_credential(wired):
    """The defect that cost 2026-09-01 and 2026-09-02, in a second scheduled job."""
    close_out.main(["--date", DAY.isoformat(), "--dry-run"])
    assert os.environ.get("MARKETSAGE_ENV") == "paper"
    assert os.environ.get("ANGEL_API_KEY") == "k"
    assert "environment is 'paper'" in _log(wired)


def test_being_late_is_not_a_failure(monkeypatch, wired):
    """The whole point: a run at 20:00 gets the same day as a run at 15:45."""
    monkeypatch.setattr(close_out, "_calendar", lambda: _Calendar(closed_hour=15))
    assert close_out.main(["--date", DAY.isoformat(), "--dry-run"]) == 0
    log = _log(wired)
    assert "still open" not in log
    assert (datetime.now(IST) - timedelta(days=1)).date() <= DAY
