"""M16 verify - TimescaleDB schema, migrations and loaders.

Skips cleanly when no database is running, so the suite stays useful before
Docker is installed. These become hard requirements at Gate G1: the determinism
and loader guarantees cannot be checked without a real database.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.schemas import Candle, Tick
from libs.ms_data.store import (
    DEFAULT_DSN,
    Migration,
    Store,
    is_reachable,
    load_migrations,
)
from tests.conftest import TEST_DAY, ist

pytestmark = pytest.mark.infra

# A socket probe cannot tell OUR database from someone else's Postgres on 5432,
# so the fixture decides by actually connecting. The probe stays as a fast path
# for the common "nothing is running" case.
DB_LISTENING = is_reachable(DEFAULT_DSN)
requires_db = pytest.mark.skipif(
    not DB_LISTENING, reason="nothing listening on 5432 - run: python tasks.py up"
)


# ---------------------------------------------------- migrations (no db needed)
def test_migration_files_are_well_formed():
    migrations = load_migrations()
    assert migrations, "expected at least the core schema migration"
    assert migrations[0].version == 1
    assert all(m.sql.strip() for m in migrations)


def test_migrations_are_ordered_and_unique():
    versions = [m.version for m in load_migrations()]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_badly_named_migration_is_rejected(tmp_path):
    """A file the runner cannot parse must fail loudly, not be skipped."""
    (tmp_path / "add_stuff.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="NNNN_snake_case"):
        load_migrations(tmp_path)


def test_duplicate_versions_are_rejected(tmp_path):
    """Two branches creating version 0002 means one of them silently never runs."""
    (tmp_path / "0002_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0002_b.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate migration versions"):
        load_migrations(tmp_path)


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert load_migrations(tmp_path / "nope") == []


def test_migration_label():
    assert Migration(3, "add_features", "SELECT 1;").label == "0003_add_features"


def test_unreachable_dsn_is_detected():
    assert is_reachable("postgresql://u:p@127.0.0.1:1/none") is False


# ------------------------------------------------------------- with a database
@pytest.fixture
def store():
    s = Store()
    try:
        s.connect()
        s.migrate()
    except Exception as exc:  # unreachable, wrong credentials, or a foreign server
        s.close()
        pytest.skip(f"MarketSage database not available: {exc}")
    with s.connect().cursor() as cur:
        cur.execute("TRUNCATE tick, candle, option_snap")
    s.connect().commit()
    yield s
    s.close()


def some_ticks(n: int = 50) -> list[Tick]:
    return [
        Tick(ts=ist(9, 20, i % 60), instrument_id=1, ltp=24_800.0 + i, qty=25, seq=i)
        for i in range(n)
    ]


def some_candles(n: int = 10, version: int = 1) -> list[Candle]:
    return [
        Candle(
            ts=ist(9, 15) + timedelta(minutes=i),
            instrument_id=1,
            tf=1,
            o=100.0 + i,
            h=101.0 + i,
            l=99.0 + i,
            c=100.5 + i,
            v=1000,
            version=version,
        )
        for i in range(n)
    ]


@requires_db
def test_migrate_is_idempotent(store):
    assert store.migrate() == []  # already applied by the fixture
    assert 1 in store.applied_versions()


@requires_db
def test_health_reports_the_server(store):
    health = store.health()
    assert "PostgreSQL" in health["server"]
    assert health["tick"] == 0


@requires_db
def test_insert_and_count_candles(store):
    assert store.insert_candles(some_candles(10)) == 10
    assert store.candle_count(TEST_DAY, instrument_id=1) == 10


@requires_db
def test_candle_load_is_idempotent(store):
    """Backfills, retries and overlapping cron runs are normal, not incidents."""
    candles = some_candles(10)
    store.insert_candles(candles)
    store.insert_candles(candles)  # again
    assert store.candle_count(TEST_DAY, instrument_id=1) == 10


@requires_db
def test_tick_load_and_sessions(store):
    store.insert_ticks(some_ticks(50))
    assert store.sessions("tick") == [TEST_DAY]


@requires_db
def test_a_late_tick_creates_a_new_version_beside_the_old_one(store):
    """The reason the primary key includes version.

    A backtest asks what the candle looked like at decision time; a report asks
    what it looks like now. An UPDATE would destroy the first question forever.
    """
    store.insert_candles(some_candles(3, version=1))
    corrected = [c.model_copy(update={"c": 999.0, "version": 2}) for c in some_candles(3)]
    store.insert_candles(corrected)

    # Still three distinct timestamps, not six.
    assert store.candle_count(TEST_DAY, instrument_id=1) == 3

    latest = store.latest_candles(TEST_DAY, instrument_id=1)
    assert all(row[-1] == 2 for row in latest)
    assert all(float(row[4]) == 999.0 for row in latest)

    as_known = store.candles_as_known_at(TEST_DAY, instrument_id=1, as_of_version=1)
    assert all(row[-1] == 1 for row in as_known)
    assert all(float(row[4]) != 999.0 for row in as_known)


@requires_db
def test_empty_insert_is_a_no_op(store):
    assert store.insert_ticks([]) == 0
    assert store.insert_candles([]) == 0
    assert store.insert_chain([]) == 0


@requires_db
def test_unknown_table_is_rejected(store):
    with pytest.raises(ValueError, match="unknown table"):
        store.sessions("robert'); DROP TABLE candle;--")
