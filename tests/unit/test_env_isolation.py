"""M06 verify - environment isolation and the live arming gate."""

from __future__ import annotations

import pytest

from libs.ms_core import env as envmod
from libs.ms_core.env import Env, EnvironmentError_


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(envmod.ENV_VAR, raising=False)
    monkeypatch.delenv(envmod.ARM_VAR, raising=False)


def test_default_is_the_safest_environment():
    assert envmod.current() is Env.RESEARCH


def test_invalid_environment_is_loud(monkeypatch):
    monkeypatch.setenv(envmod.ENV_VAR, "prod")
    with pytest.raises(EnvironmentError_, match="not a valid environment"):
        envmod.current()


def test_research_cannot_reach_broker(monkeypatch):
    """A notebook must not be able to obtain a credential to trade with."""
    monkeypatch.setenv(envmod.ENV_VAR, "research")
    with pytest.raises(EnvironmentError_, match="may not hold broker credentials"):
        envmod.require_broker_credentials()


def test_paper_may_hold_credentials(monkeypatch):
    monkeypatch.setenv(envmod.ENV_VAR, "paper")
    envmod.require_broker_credentials()  # must not raise


def test_live_requires_double_arm(monkeypatch, tmp_path):
    monkeypatch.setenv(envmod.ENV_VAR, "live")
    monkeypatch.setattr(envmod, "ARM_FILE", tmp_path / ".live-armed")

    # neither signal
    with pytest.raises(EnvironmentError_, match="not armed"):
        envmod.require_live("place an order")

    # env var only - the signal most likely to be left set by accident
    monkeypatch.setenv(envmod.ARM_VAR, "1")
    with pytest.raises(EnvironmentError_, match="not armed"):
        envmod.require_live("place an order")

    # file only
    monkeypatch.delenv(envmod.ARM_VAR)
    (tmp_path / ".live-armed").write_text("armed", encoding="utf-8")
    with pytest.raises(EnvironmentError_, match="not armed"):
        envmod.require_live("place an order")

    # both
    monkeypatch.setenv(envmod.ARM_VAR, "1")
    envmod.require_live("place an order")  # must not raise


def test_paper_env_cannot_go_live(monkeypatch, tmp_path):
    monkeypatch.setenv(envmod.ENV_VAR, "paper")
    monkeypatch.setenv(envmod.ARM_VAR, "1")
    monkeypatch.setattr(envmod, "ARM_FILE", tmp_path / ".live-armed")
    (tmp_path / ".live-armed").write_text("armed", encoding="utf-8")
    with pytest.raises(EnvironmentError_, match="not 'live'"):
        envmod.require_live("place an order")


def test_describe_reports_status():
    d = envmod.describe()
    assert d["env"] == "research"
    assert d["live_armed"] is False


def test_arm_file_is_gitignored():
    """The arming file must never be committed - that would defeat the whole gate."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".live-armed" in ignored
