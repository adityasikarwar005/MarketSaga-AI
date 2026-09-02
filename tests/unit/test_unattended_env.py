"""The unattended runner recorded nothing for two days, with correct credentials.

`jobs/trading_day.py` was written on 2026-08-31 and has never once recorded a
day. Both mornings it has run, it ended the same way about fifty seconds after
the Windows task fired, and nobody looked until the afternoon of the second::

    ok    scrip master downloaded, 140,452 instruments
    FAIL  environment is 'research', which may not hold broker credentials.
    preflight FAILED - see the log. Not starting the recorder.

`.env` said ``MARKETSAGE_ENV=paper`` throughout. The credentials were right; the
process could not see them. ``tasks.py`` read `.env` on the way into each
command that needs one, and `jobs/trading_day.py` spawns its steps directly -
``jobs/angel_preflight.py``, not ``tasks.py angel-check``. A child inherits its
parent's *environment*, not its parent's imports, and the scheduler starts that
parent with neither the environment name nor the Angel keys. So the preflight
resolved the safe default, refused exactly as designed, and the runner stopped.

Every part of that behaved correctly in isolation, which is why the suite had
nothing to say about it. The module was tested by reading it and by running
`--dry-run` from a shell that already had the environment - which is exactly the
one condition under which the defect cannot appear. It shipped unobserved, and
"the scheduled task exists" got recorded as the goal being met.

These tests assert the contract that was missing: **whatever the runner is going
to spawn, it must be able to see what `.env` holds.** The assertions are on
``os.environ`` rather than on a real child process, because ``os.environ`` is
precisely what ``subprocess.run`` copies into one - checking it there tests the
handover instead of testing Python's process model.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

import tasks
from jobs import trading_day
from libs.ms_core import dotenv
from libs.ms_core import env as envmod

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(autouse=True)
def pristine_environ():
    """`dotenv.load` writes to the real `os.environ`, so put it back after."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


class _AlwaysOpen:
    """A calendar that needs no config and no holiday file.

    `main` is being driven here for its environment handling, and a real
    `SessionCalendar` would tie these tests to whichever years
    `config/sessions.yaml` happens to cover - a failure about the wrong thing.
    """

    def is_trading_day(self, day: date) -> bool:
        return True

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        opened = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        return opened, opened.replace(hour=15, minute=30)


def _dry_run(tmp_path, monkeypatch, env_file) -> str:
    """Run the runner's plan-only path and give back what it logged."""
    monkeypatch.setattr(dotenv, "ENV_FILE", env_file)
    monkeypatch.setattr(trading_day, "LOG_DIR", tmp_path / "sessions")
    monkeypatch.setattr(trading_day, "_calendar", _AlwaysOpen)

    rc = trading_day.main(["--dry-run", "--date", "2026-09-02"])
    assert rc == 0, f"the runner exited {rc} on a day it was told was open"
    return (tmp_path / "sessions" / "session_2026-09-02.log").read_text(encoding="utf-8")


def test_the_runner_puts_the_credentials_where_its_children_will_look(tmp_path, monkeypatch):
    """The regression. Without this the preflight sees an empty environment."""
    env_file = tmp_path / ".env"
    env_file.write_text("MARKETSAGE_ENV=paper\nANGEL_API_KEY=k\n", encoding="utf-8")
    for name in ("MARKETSAGE_ENV", "ANGEL_API_KEY"):
        os.environ.pop(name, None)

    log = _dry_run(tmp_path, monkeypatch, env_file)

    assert os.environ.get("MARKETSAGE_ENV") == "paper", (
        "the runner did not read .env into its own environment, so every step "
        "it spawns will start without one. This is the defect that cost "
        "2026-09-01 and 2026-09-02: the preflight refuses in 'research' and "
        "the recorder never starts."
    )
    assert os.environ.get("ANGEL_API_KEY") == "k", (
        "the environment name came through but the credentials did not. The "
        "preflight would then pass and the login would fail instead, which "
        "reads as a bad key rather than a missing file."
    )
    envmod.require_broker_credentials()  # the question the preflight is about to ask
    assert "environment is 'paper'" in log


def test_an_environment_without_a_dotenv_is_named_in_the_log(tmp_path, monkeypatch):
    """The null case: what does it report when there is nothing to load?

    Not "clean". A runner that cannot find `.env` still runs, and still fails
    at the preflight - but the reason now appears in the line above the plan
    rather than fifty seconds later inside a child's output, which is where it
    sat unread for two days.
    """
    os.environ.pop("MARKETSAGE_ENV", None)

    log = _dry_run(tmp_path, monkeypatch, tmp_path / "there-is-no-env-file")

    assert "environment is 'research'" in log, (
        "the runner did not say which environment it resolved. That line is "
        "the entire diagnosis on a morning that recorded nothing."
    )
    assert "0 name(s) read from .env" in log


def test_an_exported_value_outranks_the_file(tmp_path):
    """A person at a terminal beats a file on disk, unless asked otherwise."""
    env_file = tmp_path / ".env"
    env_file.write_text("MARKETSAGE_ENV=paper\n", encoding="utf-8")
    os.environ["MARKETSAGE_ENV"] = "replay"

    assert dotenv.load(env_file) == ()
    assert os.environ["MARKETSAGE_ENV"] == "replay"

    assert dotenv.load(env_file, override=True) == ("MARKETSAGE_ENV",)
    assert os.environ["MARKETSAGE_ENV"] == "paper"


def test_the_loader_skips_what_is_not_a_setting(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# a comment\n\nMARKETSAGE_ENV=paper\nANGEL_PIN="1234"\n  SPACED = a value \nbare\n',
        encoding="utf-8",
    )
    for name in ("MARKETSAGE_ENV", "ANGEL_PIN", "SPACED"):
        os.environ.pop(name, None)

    assert dotenv.load(env_file) == ("MARKETSAGE_ENV", "ANGEL_PIN", "SPACED")
    assert os.environ["ANGEL_PIN"] == "1234"
    assert os.environ["SPACED"] == "a value"


def test_the_interactive_path_resolves_the_same_file(tmp_path, monkeypatch):
    """`tasks.py` and the runner must not hold two opinions about `.env`.

    Two loaders is how the pair drifts apart again - one gains `override`, or a
    search path, or a different quoting rule, and the difference only shows up
    on the path nobody runs by hand.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("MARKETSAGE_ENV=paper\n", encoding="utf-8")
    os.environ.pop("MARKETSAGE_ENV", None)
    monkeypatch.setattr(dotenv, "ENV_FILE", env_file)

    tasks._load_dotenv()

    assert os.environ.get("MARKETSAGE_ENV") == "paper", (
        "tasks.py no longer goes through libs.ms_core.dotenv, so the two "
        "entry points can disagree about where the secrets live."
    )
