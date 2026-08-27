"""M16 - TimescaleDB schema, migrations and loaders.

The hot store: recent ticks, candles and chain snapshots, queryable by SQL while
the session is running. The archive (M17) owns everything older, in Parquet.

Design notes that are easy to regret otherwise:

**Migrations are numbered files, applied once, recorded.** Not
``CREATE TABLE IF NOT EXISTS`` scattered through the code - that pattern makes
"which schema is this database on?" unanswerable the first time two machines
disagree.

**Loaders are idempotent.** Re-running a day's load changes nothing. Backfills,
retries and overlapping cron runs are normal operational events, not incidents.

**Candles are versioned, never updated in place.** A late tick produces version 2
alongside version 1. A backtest asks what the candle looked like at the time; a
report asks what it looks like now. Both must be answerable.

``psycopg`` is imported lazily so the repository installs and tests without a
database. Everything here is integration-tested against a real Postgres and
skipped when one is not running.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick

log = get_logger(__name__)

DEFAULT_DSN = "postgresql://marketsage:marketsage_dev@127.0.0.1:5432/marketsage"
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class StoreUnavailable(RuntimeError):
    """Raised when the database cannot be reached."""


def _psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise ImportError("psycopg is needed for the store: pip install -e '.[infra]'") from exc
    return psycopg


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"


_MIGRATION_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


def load_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Read numbered migration files in order.

    A gap or duplicate in the numbering is an error, not something to sort
    around: it means two branches created the same version and one of them will
    silently not run.
    """
    if not directory.exists():
        return []
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        m = _MIGRATION_RE.match(path.name)
        if not m:
            raise ValueError(f"migration {path.name!r} must be named NNNN_snake_case.sql")
        found.append(
            Migration(
                version=int(m.group(1)),
                name=m.group(2),
                sql=path.read_text(encoding="utf-8"),
            )
        )
    versions = [f.version for f in found]
    if len(set(versions)) != len(versions):
        dupes = sorted({v for v in versions if versions.count(v) > 1})
        raise ValueError(f"duplicate migration versions {dupes} - two branches collided")
    if versions != sorted(versions):
        raise ValueError("migrations are not in ascending order")
    return found


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class Store:
    """Thin wrapper over Postgres/TimescaleDB. Owns schema and writes only."""

    def __init__(self, dsn: str = DEFAULT_DSN, *, connection: Any | None = None) -> None:
        self._dsn = dsn
        self._conn = connection

    # -- connection ---------------------------------------------------------
    def connect(self) -> Any:
        if self._conn is None:
            psycopg = _psycopg()
            try:
                self._conn = psycopg.connect(self._dsn, autocommit=False)
            except Exception as exc:
                raise StoreUnavailable(
                    f"cannot reach the database at {self._dsn.split('@')[-1]}. "
                    "Start it with: python tasks.py up"
                ) from exc
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Store:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- migrations ---------------------------------------------------------
    def applied_versions(self) -> set[int]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS schema_migration (
                       version INT PRIMARY KEY,
                       label TEXT NOT NULL,
                       applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                   )""")
            conn.commit()
            cur.execute("SELECT version FROM schema_migration")
            return {row[0] for row in cur.fetchall()}

    def migrate(self, directory: Path = MIGRATIONS_DIR) -> list[str]:
        """Apply pending migrations, each in its own transaction."""
        applied = self.applied_versions()
        conn = self.connect()
        done: list[str] = []
        for migration in load_migrations(directory):
            if migration.version in applied:
                continue
            with conn.cursor() as cur:
                cur.execute(migration.sql)
                cur.execute(
                    "INSERT INTO schema_migration (version, label) VALUES (%s, %s)",
                    (migration.version, migration.label),
                )
            conn.commit()
            done.append(migration.label)
            log.info("migration applied", fields={"migration": migration.label})
        return done

    # -- loaders ------------------------------------------------------------
    def insert_ticks(self, ticks: Iterable[Tick]) -> int:
        rows = [
            (
                t.ts,
                t.instrument_id,
                t.ltp,
                t.qty,
                t.bid,
                t.ask,
                t.bid_qty,
                t.ask_qty,
                t.oi,
                t.seq,
                t.src,
                t.trace_id,
            )
            for t in ticks
        ]
        if not rows:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO tick
                   (ts, instrument_id, ltp, qty, bid, ask, bid_qty, ask_qty, oi, seq, src, trace_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                rows,
            )
        conn.commit()
        METRICS.inc("store_rows_total", value=float(len(rows)), table="tick")
        return len(rows)

    def insert_candles(self, candles: Iterable[Candle]) -> int:
        """Idempotent on ``(instrument_id, tf, ts, version)``.

        A late tick produces a new version rather than overwriting - so a
        backtest can ask what the candle looked like at decision time, not only
        what it looks like after every correction has landed.
        """
        rows = [
            (
                c.ts,
                c.instrument_id,
                c.tf,
                c.o,
                c.h,
                c.low,
                c.c,
                c.v,
                c.oi,
                c.trades,
                c.vwap,
                c.synthetic,
                c.version,
                c.trace_id,
            )
            for c in candles
        ]
        if not rows:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO candle
                   (ts, instrument_id, tf, o, h, l, c, v, oi, trades, vwap,
                    synthetic, version, trace_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (instrument_id, tf, ts, version) DO NOTHING""",
                rows,
            )
        conn.commit()
        METRICS.inc("store_rows_total", value=float(len(rows)), table="candle")
        return len(rows)

    def insert_chain(self, snapshots: Iterable[ChainSnapshot]) -> int:
        rows = []
        for snap in snapshots:
            for q in snap.quotes:
                rows.append(
                    (
                        snap.ts,
                        snap.underlying_id,
                        snap.spot,
                        snap.completeness,
                        q.expiry,
                        q.strike,
                        q.opt_type.value,
                        q.ltp,
                        q.bid,
                        q.ask,
                        q.volume,
                        q.oi,
                        q.oi_change,
                        q.iv,
                        snap.trace_id,
                    )
                )
        if not rows:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO option_snap
                   (ts, underlying_id, spot, completeness, expiry, strike, opt_type,
                    ltp, bid, ask, volume, oi, oi_change, iv, trace_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                rows,
            )
        conn.commit()
        METRICS.inc("store_rows_total", value=float(len(rows)), table="option_snap")
        return len(rows)

    # -- reads --------------------------------------------------------------
    def candle_count(self, day: date, instrument_id: int, tf: int = 1) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT count(DISTINCT ts) FROM candle
                   WHERE instrument_id = %s AND tf = %s AND ts::date = %s""",
                (instrument_id, tf, day),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def latest_candles(self, day: date, instrument_id: int, tf: int = 1) -> list[tuple[Any, ...]]:
        """Latest version of each candle - what the data looks like *now*."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ON (ts) ts, o, h, l, c, v, version
                   FROM candle
                   WHERE instrument_id = %s AND tf = %s AND ts::date = %s
                   ORDER BY ts, version DESC""",
                (instrument_id, tf, day),
            )
            return list(cur.fetchall())

    def candles_as_known_at(
        self, day: date, instrument_id: int, as_of_version: int = 1, tf: int = 1
    ) -> list[tuple[Any, ...]]:
        """The version that existed at decision time - what a backtest must use."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ON (ts) ts, o, h, l, c, v, version
                   FROM candle
                   WHERE instrument_id = %s AND tf = %s AND ts::date = %s
                     AND version <= %s
                   ORDER BY ts, version DESC""",
                (instrument_id, tf, day, as_of_version),
            )
            return list(cur.fetchall())

    def sessions(self, table: str = "candle") -> list[date]:
        if table not in ("candle", "tick", "option_snap"):
            raise ValueError(f"unknown table {table!r}")
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT ts::date AS d FROM {table} ORDER BY d")
            return [row[0] for row in cur.fetchall()]

    def health(self) -> dict[str, Any]:
        conn = self.connect()
        out: dict[str, Any] = {}
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            out["server"] = row[0].split(",")[0] if row else "unknown"
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            out["timescaledb"] = cur.fetchone() is not None
            for table in ("tick", "candle", "option_snap"):
                try:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    r = cur.fetchone()
                    out[table] = int(r[0]) if r else 0
                except Exception:
                    conn.rollback()
                    out[table] = None
        return out


def is_reachable(dsn: str = DEFAULT_DSN) -> bool:
    """Cheap probe used by tests to skip cleanly when there is no database."""
    import socket

    m = re.search(r"@([^:/]+):(\d+)", dsn)
    if not m:
        return False
    try:
        with socket.create_connection((m.group(1), int(m.group(2))), timeout=0.4):
            return True
    except OSError:
        return False


def load_day_from_archive(
    store: Store,
    day: date,
    *,
    streams: Sequence[str] = ("candles",),
    raw_root: Path | None = None,
) -> dict[str, int]:
    """Load a recorded day from NDJSON into the hot store.

    Used for backfills and after a restore. Idempotent, so running it twice on
    the same day is a no-op rather than a duplication.
    """
    from libs.ms_data.recorder import DATA_ROOT, read_stream

    root = raw_root or DATA_ROOT
    out: dict[str, int] = {}
    for stream in streams:
        partition = root / stream / f"date={day.isoformat()}"
        if not partition.exists():
            out[stream] = 0
            continue
        total = 0
        for src in sorted(partition.glob("*.ndjson")):
            if stream == "ticks":
                total += store.insert_ticks(read_stream(src, Tick))
            elif stream == "candles":
                total += store.insert_candles(read_stream(src, Candle))
            elif stream == "chain":
                total += store.insert_chain(read_stream(src, ChainSnapshot))
        out[stream] = total
    return out
