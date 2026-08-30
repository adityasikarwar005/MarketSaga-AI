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
    python tasks.py ci        the six CI stages, each reporting separately
    python tasks.py restore-drill   the monthly restore drill's self-test
    python tasks.py dashboard  serve the workspace at http://127.0.0.1:8787
    python tasks.py holidays  discover the exchange calendar (and why)
    python tasks.py qc        data quality gates for every recorded day
    python tasks.py research  end-to-end research smoke test on a recorded day
    python tasks.py replay    replay a recorded day through the live code path
    python tasks.py archive   NDJSON -> Parquet, and the research coverage report
    python tasks.py migrate   apply database migrations
    python tasks.py status    where the project stands right now
    python tasks.py gates     every gate, derived rather than remembered
"""

from __future__ import annotations

import os
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


def _docker() -> str | None:
    r"""Path to docker.exe, or None.

    `shutil.which` alone is not enough on Windows. Docker Desktop installs to
    %LOCALAPPDATA%\Programs\DockerDesktop on current versions - not to
    Program Files, which older docs say - and it adds its bin directory to the
    *user* PATH, which a shell opened before the install never sees.

    The failure that produced this: `tasks.py status` reported "install Docker
    Desktop" as an open blocker for a day while both containers were up and
    healthy and all 17 infrastructure tests passed against them. A blocker list
    that names something already done is a blocker list people stop reading.
    """
    found = shutil.which("docker")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "DockerDesktop"
        / "resources"
        / "bin"
        / "docker.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Docker"
        / "Docker"
        / "resources"
        / "bin"
        / "docker.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def up() -> None:
    if shutil.which("docker") is None:
        sys.exit(
            "docker not found.\n"
            "Install Docker Desktop, then re-run. Until then the suite still runs -\n"
            "infrastructure tests skip themselves (see tests/unit/test_smoke.py)."
        )
    run("docker", "compose", "up", "-d")


def down() -> None:
    docker = _docker()
    if docker is None:
        sys.exit("docker not found on PATH or at either default install location")
    run(docker, "compose", "down")


# `check=False` throughout is deliberate: a lint failure must not stop the type
# checker from running, or you fix one thing per round trip. What was NOT
# deliberate is that the return codes were then dropped on the floor - see
# `verify` below.


def test() -> int:
    return run(_py(), "-m", "pytest", "-q", check=False)


def lint() -> int:
    ruff = run(_py(), "-m", "ruff", "check", ".", check=False)
    black = run(_py(), "-m", "black", "--check", ".", check=False)
    return ruff or black


def fmt() -> int:
    black = run(_py(), "-m", "black", ".", check=False)
    ruff = run(_py(), "-m", "ruff", "check", "--fix", ".", check=False)
    return black or ruff


def typecheck() -> int:
    return run(_py(), "-m", "mypy", check=False)


def verify() -> int:
    """What CI runs. A module is not done until this is green.

    It could not be red. Every stage ran with `check=False`, the return codes
    were discarded, and the dispatcher at the bottom of this file called the
    command without looking at what it returned - so `tasks.py verify` exited 0
    whether the suite passed, failed, or never ran at all.

    Nothing was caught locally as a result. GitHub Actions runs the same tools
    as separate steps and was the only thing able to fail, which is the shape
    of the pipeline-red-for-four-pushes story in CLAUDE.md and of the ruff
    0.16.4/0.16.5 incident in tests/unit/test_toolchain_pins.py, where a "green
    verify" went red nineteen seconds after the push. The version drift there
    was real but it was never sufficient: verify would have been green with the
    matching ruff too, because verify was green unconditionally.

    Every stage still runs even after one fails - that part was right, and it
    is why this reports all the failing stages rather than only the first.
    """
    failed = []
    for name, stage in (("lint", lint), ("typecheck", typecheck), ("test", test)):
        print(f"\n=== {name} ===")
        if stage() != 0:
            failed.append(name)
    if failed:
        print(f"\nverify: FAILED at {', '.join(failed)}")
        return 1
    print("\nverify: green")
    return 0


def ci() -> None:
    """MA4 - the six stages, each reporting which guarantee it establishes.

    `verify` answers "is this commit green". This answers "which guarantee
    broke", which is the question you have once the answer is no.
    """
    run(_py(), str(ROOT / "jobs" / "ci_stages.py"), *sys.argv[2:], check=False)


def restore_drill() -> None:
    """MA5 - prove the drill detects damage, without needing a clean machine."""
    args = sys.argv[2:] or ["--self-test"]
    run(_py(), str(ROOT / "jobs" / "restore_drill.py"), *args, check=False)


def bhavcopy() -> None:
    """M14 - backfill NSE F&O bhavcopy into the local cache.

    `python tasks.py bhavcopy --days 30` or `--from YYYY-MM-DD --to YYYY-MM-DD`.
    One request per session, cached, never re-fetched. Eight years is about two
    thousand requests to a free public archive, so the default is thirty days.
    """
    run(_py(), "jobs/bhavcopy_backfill.py", *sys.argv[2:], check=False)


def angel_history() -> None:
    """Angel historical candles: cross-check the price scale, or fetch intraday.

    `python tasks.py angel-history --cross-check` compares Angel's own daily
    series against the ten years of NIFTY bars already on disk from an unrelated
    provider. That is the only available evidence that this endpoint returns
    rupees rather than paise - the plausibility band cannot tell 24,180 from
    241.80, and a consistent 100x survives every downstream check.
    """
    _load_dotenv()
    run(_py(), "jobs/angel_history.py", *sys.argv[2:], check=False)


def gates() -> None:
    """Every gate's real state, derived from the repository.

    `docs/gates.md` is the signed ledger; this reads the experiment log, the
    filesystem and the config and says what is true right now, so the ledger can
    be checked against something rather than trusted. Add --strict to exit
    non-zero unless every gate passes.
    """
    run(_py(), "jobs/gate_report.py", *sys.argv[2:], check=False)


def angel_check() -> None:
    """Verify the Angel One credentials before a trading morning needs them.

    `python tasks.py angel-check` stops before the network; add `--connect` to
    log in for real. Run it the evening before - every failure it catches
    otherwise shows up as a quiet feed during the opening auction.
    """
    _load_dotenv()
    args = sys.argv[2:]
    run(_py(), "jobs/angel_preflight.py", *args, check=False)


def dashboard() -> None:
    """M91 - serve the dashboard.

    A plain static server. The workspace reads its session from `data.js`
    rather than fetching JSON, so it also opens straight from the filesystem -
    but served is how it will actually run, and a thing only ever opened one
    way acquires assumptions about that way.
    """
    import http.server

    root = ROOT / "services" / "dashboard"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8787

    class Handler(http.server.SimpleHTTPRequestHandler):
        # HTTP/1.1 so the browser may keep connections alive. On a
        # single-threaded server that would make things worse, not better -
        # it is safe only because of the threading server below.
        protocol_version = "HTTP/1.1"

        def __init__(self, *a: object, **kw: object) -> None:
            super().__init__(*a, directory=str(root), **kw)  # type: ignore[arg-type]

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    # ThreadingHTTPServer, not TCPServer. A browser opens several connections
    # in parallel and holds them open; a single-threaded server serialises
    # them, and every request after the first waits on a socket timeout.
    #
    # The first version of this used TCPServer and the page took 173 seconds
    # to load. That reads as "the dashboard is slow", which sends the
    # investigation into the front end - the four assets themselves each
    # returned in milliseconds under curl, one at a time.
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as srv:
        srv.daemon_threads = True
        print(f"  MarketSage Desk on http://127.0.0.1:{port}  (ctrl-c to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped")


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
    if unverified := cfg.unverified_instruments():
        # Name them. "verify instrument reference data" read as all six for
        # months after NIFTY-FUT had been verified against NSE/FAOP/70616, and
        # a blocker nobody can act on is one nobody looks at.
        blockers.append(
            "verify instrument reference data against the exchange circular: "
            + ", ".join(sorted(unverified))
        )
    if not known_years:
        blockers.append("discover the exchange holiday calendar: python tasks.py holidays")
    if not any(s.enabled for s in cfg.sources.sources):
        blockers.append("enable at least one news source (and record its ToS)")
    if todo_costs := [s for s in cfg.costs.schedules if s.source.startswith("TODO")]:
        # The current schedule is real (Angel's own breakup, 2026-08-30). What
        # is still a placeholder is history, and the backtest runs from 2016 -
        # so this stays a blocker, but it should say which years are guessed
        # rather than imply today's rates are made up.
        dates = ", ".join(str(s.effective_from) for s in todo_costs)
        blockers.append(f"cost rates are still placeholders for: {dates}")
    if _docker() is None:
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
    "ci": ci,
    "restore-drill": restore_drill,
    "bhavcopy": bhavcopy,
    "gates": gates,
    "angel-history": angel_history,
    "angel-check": angel_check,
    "dashboard": dashboard,
    "status": status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    # `or 0` because most commands return None. Discarding this return
    # value is what let `verify` report success for the life of the file.
    sys.exit(COMMANDS[sys.argv[1]]() or 0)
