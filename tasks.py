#!/usr/bin/env python
"""M00 - task runner.

Cross-platform, no `make` required.

    python tasks.py setup     create .venv and install everything
    python tasks.py login     daily Kite login (token expires every night)
    python tasks.py sync-instruments   pull Kite's instrument dump
    python tasks.py record [synthetic|kite]   start the recorder
    python tasks.py up        docker compose up -d
    python tasks.py down      docker compose down
    python tasks.py test      pytest
    python tasks.py lint      ruff + black --check
    python tasks.py fmt       black + ruff --fix
    python tasks.py typecheck mypy
    python tasks.py verify    the full gate check: lint + typecheck + test
    python tasks.py holidays  discover the exchange calendar (and why)
    python tasks.py qc        data quality gates for every recorded day
    python tasks.py research  end-to-end research smoke test on a recorded day
    python tasks.py replay    replay a recorded day through the live code path
    python tasks.py archive   NDJSON -> Parquet, and the research coverage report
    python tasks.py migrate   apply database migrations
    python tasks.py status    where the project stands right now
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _py() -> str:
    return str(PY) if PY.exists() else sys.executable


def run(*args: str, check: bool = True) -> int:
    print(f"  $ {' '.join(args)}")
    return subprocess.run(args, cwd=ROOT, check=check).returncode


def setup() -> None:
    if not VENV.exists():
        print("creating .venv ...")
        run(sys.executable, "-m", "venv", str(VENV))
    run(_py(), "-m", "pip", "install", "--upgrade", "pip", "-q")
    run(_py(), "-m", "pip", "install", "-e", ".[dev]", "-q")
    print("\nready. next:  python tasks.py test")


def up() -> None:
    if shutil.which("docker") is None:
        sys.exit(
            "docker not found.\n"
            "Install Docker Desktop, then re-run. Until then the suite still runs -\n"
            "infrastructure tests skip themselves (see tests/unit/test_smoke.py)."
        )
    run("docker", "compose", "up", "-d")


def down() -> None:
    run("docker", "compose", "down")


def test() -> None:
    run(_py(), "-m", "pytest", "-q", check=False)


def lint() -> None:
    run(_py(), "-m", "ruff", "check", ".", check=False)
    run(_py(), "-m", "black", "--check", ".", check=False)


def fmt() -> None:
    run(_py(), "-m", "black", ".", check=False)
    run(_py(), "-m", "ruff", "check", "--fix", ".", check=False)


def typecheck() -> None:
    run(_py(), "-m", "mypy", check=False)


def verify() -> None:
    """What CI runs. A module is not done until this is green."""
    print("\n=== lint ===")
    lint()
    print("\n=== typecheck ===")
    typecheck()
    print("\n=== test ===")
    test()


def status() -> None:
    import warnings

    sys.path.insert(0, str(ROOT))
    from libs.ms_core import config as cfgmod
    from libs.ms_core import env as envmod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = cfgmod.load()

    print(f"env            : {envmod.current().value}")
    print(f"live armed     : {envmod.is_live_armed()}")
    print(f"config hash    : {cfg.config_hash}")
    print(f"instruments    : {len(cfg.instruments.instruments)}")
    print(f"event classes  : {len(cfg.taxonomy.class_names())}")
    print(
        f"sources enabled: {sum(1 for s in cfg.sources.sources if s.enabled)}"
        f" / {len(cfg.sources.sources)}"
    )
    from libs.ms_data.holidays import HolidayStore, next_holiday

    store = HolidayStore()
    known_years = store.years()
    pinned = cfg.sessions.holidays
    print(
        f"holiday years  : {known_years or 'NONE DISCOVERED'}"
        f"{f' (+{sorted(pinned)} pinned)' if pinned else ''}"
    )
    today = _today()
    nxt = next_holiday(store, today) if known_years else None
    if nxt:
        d = (nxt.day - today).days
        when = "tomorrow" if d == 1 else f"in {d} days"
        print(f"next closure   : {nxt.day} ({when}) - {nxt.reason}")
    for w in caught:
        print(f"\n  WARNING: {w.message}")

    print("\nopen blockers before Gate G0:")
    blockers = []
    if cfg.unverified_instruments():
        blockers.append("verify instrument reference data against the exchange circular")
    if not known_years:
        blockers.append("discover the exchange holiday calendar: python tasks.py holidays")
    if not any(s.enabled for s in cfg.sources.sources):
        blockers.append("enable at least one news source (and record its ToS)")
    if any(s.source.startswith("TODO") for s in cfg.costs.schedules):
        blockers.append("replace placeholder cost rates with real statutory rates")
    if shutil.which("docker") is None:
        blockers.append("install Docker Desktop")
    captured = [p for p in (ROOT / "data").rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not captured:
        blockers.append(
            "M12: start the recorder - no market data captured yet. "
            "This is the blocker that compounds: every day it is not running is a day "
            "of event-conditioned samples you can never get back."
        )
    for b in blockers:
        print(f"  - {b}")
    if not blockers:
        print("  none. sign G0 in docs/gates.md.")


def research() -> None:
    """Features, labels, purged splits and the M0 baseline on one recorded day."""
    run(_py(), "jobs/research_smoke.py", *sys.argv[2:], check=False)


def gap() -> None:
    """Fit and score the overnight gap forecaster on cached daily bars."""
    run(_py(), "jobs/gap_backfill.py", *sys.argv[2:], check=False)


def trials() -> None:
    """Every attempt ever scored. The count is what makes a result a result."""
    run(_py(), "jobs/experiment_log.py", *sys.argv[2:], check=False)


def kill_drill() -> None:
    """The monthly drill. A kill switch never pulled is one nobody knows."""
    run(_py(), "jobs/kill_drill.py", *(sys.argv[2:] or ["--dry-run"]), check=False)


def qc() -> None:
    """Data quality gates. A FAIL blocks research on that day, by design."""
    run(_py(), "jobs/qc.py", *(sys.argv[2:] or ["--all"]), check=False)


def replay() -> None:
    """Replay a recorded day through the same code path production uses."""
    run(_py(), "jobs/replay.py", *sys.argv[2:], check=False)


def archive() -> None:
    """Nightly conversion for research. The recorder keeps its NDJSON untouched."""
    args = sys.argv[2:] or ["--all"]
    run(_py(), "jobs/archive.py", *args, check=False)


def migrate() -> None:
    """Apply pending database migrations."""
    sys.path.insert(0, str(ROOT))
    from libs.ms_data.store import Store, StoreUnavailable

    try:
        with Store() as s:
            applied = s.migrate()
            if applied:
                for label in applied:
                    print(f"  applied {label}")
            else:
                print("  already up to date")
            for k, v in s.health().items():
                print(f"  {k:<14}: {v}")
    except StoreUnavailable as exc:
        sys.exit(f"  {exc}")


def holidays() -> None:
    """Discover the exchange holiday calendar. No manual list to keep stale."""
    args = sys.argv[2:]
    if not args:
        from datetime import date

        year = date.today().year
        args = ["--years", str(year), str(year + 1)]
    run(_py(), "jobs/sync_holidays.py", *args, check=False)


def login() -> None:
    """Daily Kite login. Kite's access token expires overnight, every night."""
    import os

    sys.path.insert(0, str(ROOT))
    _load_dotenv()
    from libs.ms_broker.kite import interactive_login
    from libs.ms_core import env as envmod
    from libs.ms_core.clock import RealClock

    if envmod.current() is envmod.Env.RESEARCH:
        sys.exit(
            "MARKETSAGE_ENV is 'research', which may not hold broker credentials.\n"
            "Set MARKETSAGE_ENV=paper in your .env first."
        )
    key, secret = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_API_SECRET")
    if not key or not secret:
        sys.exit(
            "KITE_API_KEY / KITE_API_SECRET not set.\n"
            "Copy .env.example to .env and fill them in (never commit .env)."
        )
    s = interactive_login(key, secret, RealClock())
    print(f"\n  logged in as {s.user_id}. Token is valid for today only.")


def sync_instruments() -> None:
    """Pull Kite's instrument dump - real lot sizes, tick sizes, expiries.

    This is what clears the 'unverified instrument reference data' warning:
    the broker's own dump is the authority, not a hand-typed YAML guess.
    """
    import csv

    sys.path.insert(0, str(ROOT))
    _load_dotenv()
    from libs.ms_broker.kite import INSTRUMENT_CACHE, require_session
    from libs.ms_core.clock import RealClock

    session = require_session(RealClock())
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("pip install kiteconnect")

    kite = KiteConnect(api_key=session.api_key)
    kite.set_access_token(session.access_token)
    rows = kite.instruments()
    INSTRUMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with INSTRUMENT_CACHE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} instruments -> {INSTRUMENT_CACHE}")
    print("  next: reconcile config/instruments.yaml against this dump, then set verified: true")


def record() -> None:
    """Start the recorder. The single most important thing in this repository."""
    src = sys.argv[2] if len(sys.argv) > 2 else "synthetic"
    args = ["-m", "services.ingest_market", "--source", src]
    if src == "synthetic":
        args += ["--minutes", "375", "--verify", "--metrics"]
    run(_py(), *args, check=False)


def _today():
    from datetime import date

    return date.today()


def _load_dotenv() -> None:
    """Minimal .env loader - avoids a dependency for six lines of work."""
    import os

    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


COMMANDS = {
    "setup": setup,
    "login": login,
    "sync-instruments": sync_instruments,
    "holidays": holidays,
    "archive": archive,
    "qc": qc,
    "research": research,
    "gap": gap,
    "trials": trials,
    "kill-drill": kill_drill,
    "replay": replay,
    "migrate": migrate,
    "record": record,
    "up": up,
    "down": down,
    "test": test,
    "lint": lint,
    "fmt": fmt,
    "typecheck": typecheck,
    "verify": verify,
    "status": status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
