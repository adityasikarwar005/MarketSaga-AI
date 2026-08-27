"""M00 verify - smoke test.

Infrastructure checks skip cleanly when Docker is not running, so the suite is
useful from the first minute. They become hard requirements at Gate G1 - see
`test_infra_is_required_by_gate_g1` at the bottom.
"""

from __future__ import annotations

import socket
import warnings

import pytest


def _reachable(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


POSTGRES_UP = _reachable("127.0.0.1", 5433)
REDIS_UP = _reachable("127.0.0.1", 6379)


def test_imports():
    from libs.ms_core import clock, config, env, schemas  # noqa: F401


def test_config_loads():
    from libs.ms_core import config as cfgmod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = cfgmod.load()
    assert cfg.config_hash


def test_repo_layout():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for expected in ("config", "libs/ms_core", "services", "jobs", "tests", "docs"):
        assert (root / expected).is_dir(), f"missing {expected}"


@pytest.mark.infra
@pytest.mark.skipif(not POSTGRES_UP, reason="Postgres not reachable - run: docker compose up -d")
def test_postgres_reachable():
    assert POSTGRES_UP


@pytest.mark.infra
@pytest.mark.skipif(not REDIS_UP, reason="Redis not reachable - run: docker compose up -d")
def test_redis_reachable():
    assert REDIS_UP


def test_infra_is_required_by_gate_g1():
    """Documents the promotion: infra may be optional now, not at G1.

    Once the data spine exists, a green suite without Postgres and Redis would be
    a false green - the determinism and loader tests cannot run without them.
    """
    if POSTGRES_UP and REDIS_UP:
        return
    pytest.skip(
        "Infrastructure is down. Fine during Sprint 1. Before Gate G1, install "
        "Docker Desktop, run `docker compose up -d`, and delete this skip."
    )
