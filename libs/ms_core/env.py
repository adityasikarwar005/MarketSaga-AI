"""M06 - Environment isolation and the live-trading arming gate.

Four environments, deliberately unequal in what they are allowed to touch:

===========  =====================  ==============================================
Environment  Data                   Allowed to
===========  =====================  ==============================================
research     historical Parquet     read history. **No broker credentials, ever.**
replay       recorded live days     drive the simulator
paper        live feeds             produce simulated fills
live         live feeds             place real orders, only when double-armed
===========  =====================  ==============================================

The research environment is credential-free by construction, not by convention.
A notebook cannot accidentally place an order because it cannot obtain a
credential to place one with.

Live trading requires **two independent signals**: an environment variable and a
file on disk. One of them can be set by accident; both together cannot.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_FILE = REPO_ROOT / ".live-armed"
ENV_VAR = "MARKETSAGE_ENV"
ARM_VAR = "MARKETSAGE_LIVE_ARMED"


class Env(StrEnum):
    RESEARCH = "research"
    REPLAY = "replay"
    PAPER = "paper"
    LIVE = "live"


class EnvironmentError_(RuntimeError):
    """Raised on an attempt to do something the current environment forbids."""


def current() -> Env:
    """Current environment. Defaults to the safest one."""
    raw = os.environ.get(ENV_VAR, Env.RESEARCH.value).strip().lower()
    try:
        return Env(raw)
    except ValueError as exc:
        raise EnvironmentError_(
            f"{ENV_VAR}={raw!r} is not a valid environment. "
            f"Choose one of: {', '.join(e.value for e in Env)}"
        ) from exc


def may_hold_broker_credentials(env: Env | None = None) -> bool:
    return (env or current()) is not Env.RESEARCH


def require_broker_credentials(env: Env | None = None) -> None:
    """Guard placed in every broker adapter constructor."""
    e = env or current()
    if not may_hold_broker_credentials(e):
        raise EnvironmentError_(
            f"environment {e.value!r} may not hold broker credentials. "
            "Research code must never be able to reach a trading venue - "
            f"set {ENV_VAR}=paper if you meant to run the paper engine."
        )


def is_live_armed() -> bool:
    """True only when BOTH arming signals are present.

    Two independent signals, because a single one gets left set by accident
    exactly once - and once is enough.
    """
    return os.environ.get(ARM_VAR) == "1" and ARM_FILE.exists()


def require_live(action: str) -> None:
    """Guard placed immediately before any code path that can move real money."""
    e = current()
    if e is not Env.LIVE:
        raise EnvironmentError_(f"refusing to {action}: environment is {e.value!r}, not 'live'")
    if not is_live_armed():
        raise EnvironmentError_(
            f"refusing to {action}: live trading is not armed. "
            f"Both are required: {ARM_VAR}=1 and the file {ARM_FILE.name} present at the repo root. "
            "This is intentional friction."
        )


def describe() -> dict[str, object]:
    """Status line for the dashboard and the preflight check."""
    e = current()
    return {
        "env": e.value,
        "may_hold_credentials": may_hold_broker_credentials(e),
        "live_armed": is_live_armed(),
        "arm_file_present": ARM_FILE.exists(),
        "arm_var_set": os.environ.get(ARM_VAR) == "1",
    }
