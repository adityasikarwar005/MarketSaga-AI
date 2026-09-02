"""The local secret store - `.env` at the repo root, read into the process.

Six lines of parsing that used to live inside ``tasks.py``. It is a module
because *who calls it* turned out to matter more than what it does.

``tasks.py`` loaded ``.env`` on the way into the commands that need a
credential, so every interactive path had one. ``jobs/trading_day.py`` runs
unattended and spawns its steps directly - ``jobs/angel_preflight.py`` rather
than ``tasks.py angel-check`` - and a child inherits its parent's environment,
not the parent's imports. Under the Windows scheduler that environment contains
no ``MARKETSAGE_ENV`` and no ``ANGEL_*``, so the preflight resolved the default
environment, ``research``, and correctly refused to hold broker credentials.
The runner stopped before the open.

That cost 2026-09-01 and 2026-09-02: two trading days recorded nothing. The
`.env` on disk was right the whole time, which is why it read as a credential
problem. Nothing was wrong with the credentials; they were never in the room.

A shared loader does not by itself prevent the next version of this - a caller
that forgets to call it still gets an empty environment. What it prevents is
the two paths *disagreeing* about where the file is and what wins, which is the
part that stays wrong quietly.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def load(path: Path | None = None, *, override: bool = False) -> tuple[str, ...]:
    """Read ``KEY=value`` lines into ``os.environ``. Returns the names it set.

    Blank lines and ``#`` comments are skipped, and surrounding quotes are
    stripped from the value.

    An existing variable wins unless ``override`` is passed: someone who
    exported a name before starting the process meant it, and a file on disk
    should not silently outrank a person at a terminal. That also keeps CI -
    which has no ``.env`` and sets what it needs directly - unaffected.
    """
    env_file = ENV_FILE if path is None else path
    if not env_file.exists():
        return ()

    loaded: list[str] = []
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = value.strip().strip("\"'")
        loaded.append(key)
    return tuple(loaded)
