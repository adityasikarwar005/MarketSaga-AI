"""`python tasks.py record` - the entry point the live morning goes through.

Two defects lived here together, and each hid the other. The task did not load
`.env`, so `services.ingest_market` looked for the broker credentials in an
environment nobody had populated; and it dropped everything after the source
argument, so `--chain` could not be passed at all. Since the recorder refuses
`--source angel` without `--chain`, the second failure fired first and the
first was never reached.

Neither is visible from reading the recorder: both are in the launcher.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tasks():
    spec = importlib.util.spec_from_file_location("marketsage_tasks", ROOT / "tasks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launched(tasks, monkeypatch):
    """Capture the command `record` would have run, without running it."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(tasks, "run", lambda *a, **kw: calls.append(a) or 0)
    monkeypatch.setattr(tasks, "_load_dotenv", lambda: calls.append(("<dotenv>",)))
    return calls


def _argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["tasks.py", "record", *args])


def test_the_bare_form_still_runs_a_full_synthetic_session(tasks, launched, monkeypatch):
    _argv(monkeypatch)
    tasks.record()
    command = launched[-1]
    assert "--source" in command and command[command.index("--source") + 1] == "synthetic"
    assert "--minutes" in command and "375" in command


def test_flags_reach_the_recorder(tasks, launched, monkeypatch):
    """`--chain` is the flag the live session cannot run without, and the task
    used to drop it on the floor."""
    _argv(monkeypatch, "angel", "--chain", "--chain-name", "NIFTY")
    tasks.record()
    command = launched[-1]
    assert command[command.index("--source") + 1] == "angel"
    assert "--chain" in command
    assert command[command.index("--chain-name") + 1] == "NIFTY"


def test_the_synthetic_defaults_do_not_override_what_was_asked_for(tasks, launched, monkeypatch):
    """A caller who names a session length must get it, not 375 minutes as well."""
    _argv(monkeypatch, "synthetic", "--minutes", "5")
    tasks.record()
    command = launched[-1]
    assert command.count("--minutes") == 1
    assert command[command.index("--minutes") + 1] == "5"


def test_flags_without_a_source_still_mean_synthetic(tasks, launched, monkeypatch):
    _argv(monkeypatch, "--chain")
    tasks.record()
    command = launched[-1]
    assert command[command.index("--source") + 1] == "synthetic"
    assert "--chain" in command
    assert "--minutes" not in command  # the caller asked for something specific


def test_the_env_file_is_loaded_before_the_recorder_starts(tasks, launched, monkeypatch):
    """`AngelCredentials.from_env` reads `os.environ` and nothing inside
    `services.ingest_market` populates it. Without this the live command fails
    before it reaches the network, with an error about missing credentials that
    are in fact sitting in `.env`."""
    _argv(monkeypatch, "angel", "--chain")
    tasks.record()
    assert launched[0] == ("<dotenv>",), "dotenv was not loaded, or not loaded first"


def test_the_loader_does_not_overwrite_a_real_environment_variable(tasks, monkeypatch, tmp_path):
    """An explicit `MARKETSAGE_ENV=research` on the command line must win over
    the file, or the safety switch can be flipped by a stale checkout.

    Patched at `dotenv.ENV_FILE` rather than at `tasks.ROOT`: the parsing
    moved to `libs.ms_core.dotenv` so `jobs/trading_day.py` could share it,
    and one canonical answer to "where is `.env`" is the point of the move.
    """
    from libs.ms_core import dotenv

    env_file = tmp_path / ".env"
    monkeypatch.setattr(dotenv, "ENV_FILE", env_file)
    env_file.write_text("MARKETSAGE_ENV=paper\nSOMETHING_NEW=x\n", encoding="utf-8")
    monkeypatch.setenv("MARKETSAGE_ENV", "research")
    monkeypatch.delenv("SOMETHING_NEW", raising=False)
    tasks._load_dotenv()
    import os

    assert os.environ["MARKETSAGE_ENV"] == "research"
    assert os.environ["SOMETHING_NEW"] == "x"
