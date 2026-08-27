"""M17 verify - Parquet archive, dataset versions, DuckDB research layer."""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.clock import FrozenClock, SimClock
from libs.ms_core.schemas import Tick
from libs.ms_data.archive import (
    Research,
    archive_all,
    archive_day,
    available_days,
    build_dataset_version,
    verify_dataset_version,
)
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.chain import ChainBuilder, ChainSpec, build_registry_from_specs
from libs.ms_data.feed import SyntheticFeed, SyntheticInstrument
from libs.ms_data.recorder import Recorder
from libs.ms_data.synthetic_chain import SyntheticChainFeed, strike_grid
from tests.conftest import TEST_DAY, ist

UNDERLYING = 1
SPOT = 24_800.0
INSTRUMENTS = [
    SyntheticInstrument(instrument_id=UNDERLYING, start_price=SPOT, annual_vol=0.14),
    SyntheticInstrument(instrument_id=2, start_price=54_200.0, annual_vol=0.18),
]


# Module-scoped: generating a session costs a few seconds, and regenerating it
# for each of twenty tests turns a fast suite into one people start skipping.
# The fixtures are read-only, so sharing them is safe.
@pytest.fixture(scope="module")
def calendar_mod():
    import warnings

    from libs.ms_core import config as cfgmod
    from libs.ms_core.clock import SessionCalendar

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = cfgmod.load()
    return SessionCalendar.from_config(cfg.sessions)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, calendar_mod):
    """A short recorded session on disk: ticks, candles and chain."""
    cal = calendar_mod.synthetic_variant()
    raw = tmp_path_factory.mktemp("raw_session") / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(INSTRUMENTS, cal, clock, seed=11)
    feed.subscribe([UNDERLYING, 2])

    expiries = [TEST_DAY + timedelta(days=2)]
    registry = build_registry_from_specs(
        UNDERLYING, expiries, strike_grid(SPOT, step=50.0, n_per_side=10)
    )
    chain_feed = SyntheticChainFeed(registry, cal, underlying_id=UNDERLYING, seed=5)
    chain_builder = ChainBuilder(
        ChainSpec(underlying_id=UNDERLYING, n_expiries=1, strikes_per_side=5), registry, cal
    )

    builder = CandleBuilder(cal)
    rec = Recorder(clock, cal, root=raw, fsync_every=10_000)
    end = ist(9, 30)

    for tick in feed.stream():
        if tick.ts >= end:
            break
        rec.write_tick(tick)
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
        chain_builder.on_tick(tick)
        if tick.instrument_id == UNDERLYING:
            for opt in chain_feed.quotes_for(tick.ltp, tick.ts):
                chain_builder.on_tick(opt)
            snap = chain_builder.maybe_snapshot(tick.ts)
            if snap is not None:
                rec.write_chain(snap)
    for candle in builder.flush(end):
        rec.write("candles", candle, key=candle.instrument_id)
    rec.close()
    return raw


@pytest.fixture(scope="module")
def archived(recorded, tmp_path_factory):
    parquet = tmp_path_factory.mktemp("archive") / "parquet"
    archive_all(TEST_DAY, raw_root=recorded, parquet_root=parquet)
    return parquet


# --------------------------------------------------------------- archiving
def test_archives_every_stream(recorded, tmp_path):
    parquet = tmp_path / "parquet"
    results = {r.stream: r for r in archive_all(TEST_DAY, raw_root=recorded, parquet_root=parquet)}
    assert results["ticks"].rows > 0
    assert results["candles"].rows > 0
    assert results["chain"].rows > 0
    assert all(f.exists() for r in results.values() for f in r.files)


def test_archiving_is_idempotent(recorded, tmp_path):
    """A repeated or retried run must be harmless, not a duplication."""
    a, b = tmp_path / "a", tmp_path / "b"
    archive_day("candles", TEST_DAY, raw_root=recorded, parquet_root=a)
    archive_day("candles", TEST_DAY, raw_root=recorded, parquet_root=a)  # again
    archive_day("candles", TEST_DAY, raw_root=recorded, parquet_root=b)

    files_a = sorted(a.rglob("*.parquet"))
    files_b = sorted(b.rglob("*.parquet"))
    assert files_a and [f.name for f in files_a] == [f.name for f in files_b]
    for fa, fb in zip(files_a, files_b, strict=True):
        assert fa.read_bytes() == fb.read_bytes(), "archive is not byte-reproducible"


def test_missing_day_is_not_an_error(tmp_path):
    result = archive_day("ticks", TEST_DAY, raw_root=tmp_path / "nope", parquet_root=tmp_path / "p")
    assert result.rows == 0 and not result.ok


def test_unknown_stream_is_rejected(recorded, tmp_path):
    with pytest.raises(ValueError, match="unknown stream"):
        archive_day("nonsense", TEST_DAY, raw_root=recorded, parquet_root=tmp_path / "p")


def test_available_days_lists_recorded_partitions(recorded):
    assert available_days("ticks", raw_root=recorded) == [TEST_DAY]
    assert available_days("chain", raw_root=recorded) == [TEST_DAY]


# ------------------------------------------------------------- round trip
def test_parquet_preserves_every_tick(recorded, archived):
    """Nothing is lost or reshaped in the conversion."""
    from libs.ms_data.recorder import read_stream

    source = []
    for f in sorted((recorded / "ticks" / f"date={TEST_DAY.isoformat()}").glob("*.ndjson")):
        source.extend(read_stream(f, Tick))

    with Research(archived) as r:
        df = r.ticks(ist(9, 0), ist(16, 0))

    assert len(df) == len(source)
    assert set(df["instrument_id"].unique()) == {t.instrument_id for t in source}
    assert float(df["ltp"].sum()) == pytest.approx(sum(t.ltp for t in source), rel=1e-9)


def test_candles_survive_the_round_trip(recorded, archived):
    with Research(archived) as r:
        df = r.candles(ist(9, 15), ist(9, 30))
    assert len(df) > 0
    assert set(df.columns) >= {"ts", "instrument_id", "o", "h", "l", "c", "v", "synthetic"}
    assert (df["h"] >= df["l"]).all()
    assert (df["h"] >= df["o"]).all()


# ------------------------------------------------------------- research
def test_candle_window_is_right_open(recorded, archived):
    """Must match the candle bucket convention exactly, or every label built on
    top of it is misaligned by a bar."""
    with Research(archived) as r:
        df = r.candles(ist(9, 15), ist(9, 20))
    stamps = sorted(df["ts"].unique())
    assert len(stamps) == 5  # 09:15..09:19, not 09:20


def test_instrument_filter(recorded, archived):
    with Research(archived) as r:
        df = r.candles(ist(9, 15), ist(9, 30), instruments=[2])
    assert set(df["instrument_id"].unique()) == {2}


def test_synthetic_candles_can_be_excluded(recorded, archived):
    with Research(archived) as r:
        allc = r.candles(ist(9, 15), ist(9, 30))
        real = r.candles(ist(9, 15), ist(9, 30), include_synthetic=False)
    assert len(real) <= len(allc)
    assert not real["synthetic"].any()


def test_chain_as_of_never_reaches_forward(recorded, archived):
    """The point-in-time primitive. Anything that reaches ahead, even by one
    snapshot, is look-ahead bias that survives right up until real money."""
    import pandas as pd

    with Research(archived) as r:
        mid = r.chain_as_of(ist(9, 22), UNDERLYING, max_age_seconds=600)
        assert len(mid) > 0
        # Compare as UTC instants; DuckDB and pandas each have their own idea of
        # how to carry a timezone, and a naive comparison here would pass by luck.
        latest = pd.Timestamp(mid["ts"].max()).tz_convert("UTC")
        assert latest <= pd.Timestamp(ist(9, 22)).tz_convert("UTC")

        early = r.chain_as_of(ist(9, 16), UNDERLYING, max_age_seconds=600)
        assert len(early) > 0, "expected a snapshot before 09:16"
        assert pd.Timestamp(early["ts"].max()).tz_convert("UTC") < latest


def test_chain_as_of_returns_one_snapshot_not_a_mixture(recorded, archived):
    with Research(archived) as r:
        df = r.chain_as_of(ist(9, 25), UNDERLYING, max_age_seconds=600)
    assert len(df) > 0
    assert df["ts"].nunique() == 1, "rows came from more than one snapshot"
    assert df["spot"].nunique() == 1


def test_chain_as_of_respects_max_age(recorded, archived):
    """A stale chain is no chain. Better to return nothing than a price nobody
    could have traded on."""
    with Research(archived) as r:
        df = r.chain_as_of(ist(14, 0), UNDERLYING, max_age_seconds=30)
    assert len(df) == 0


def test_empty_archive_returns_empty_frames_not_errors(tmp_path):
    with Research(tmp_path / "empty") as r:
        assert len(r.candles(ist(9, 15), ist(9, 30))) == 0
        assert len(r.ticks(ist(9, 15), ist(9, 30))) == 0
        assert len(r.chain_as_of(ist(9, 20), UNDERLYING)) == 0
        assert r.sessions() == []


def test_coverage_report(recorded, archived):
    with Research(archived) as r:
        cov = r.coverage()
    assert set(cov["stream"]) == {"ticks", "candles", "chain"}
    assert cov.set_index("stream").loc["candles", "rows"] > 0


def test_sessions_lists_archived_days(recorded, archived):
    with Research(archived) as r:
        assert r.sessions("candles") == [TEST_DAY]


# ----------------------------------------------------------- dataset version
def test_dataset_version_is_reproducible(recorded, archived, tmp_path):
    clock = FrozenClock(ist(18, 0))
    d = tmp_path / "datasets"
    a = build_dataset_version(
        "v1", TEST_DAY, TEST_DAY, clock.now(), parquet_root=archived, dataset_dir=d
    )
    b = build_dataset_version(
        "v2", TEST_DAY, TEST_DAY, clock.now(), parquet_root=archived, dataset_dir=d
    )
    assert a.files
    assert a.digest == b.digest, "same data must produce the same digest"


def test_dataset_version_verifies(recorded, archived, tmp_path):
    d = tmp_path / "datasets"
    build_dataset_version(
        "v1",
        TEST_DAY,
        TEST_DAY,
        FrozenClock(ist(18, 0)).now(),
        parquet_root=archived,
        dataset_dir=d,
    )
    ok, problems = verify_dataset_version("v1", parquet_root=archived, dataset_dir=d)
    assert ok and not problems


def test_dataset_version_detects_modified_data(recorded, tmp_path):
    """'Which data produced this model' must have an exact answer months later."""
    archived = tmp_path / "parquet"  # own copy: this test corrupts a file
    archive_all(TEST_DAY, raw_root=recorded, parquet_root=archived)
    d = tmp_path / "datasets"
    build_dataset_version(
        "v1",
        TEST_DAY,
        TEST_DAY,
        FrozenClock(ist(18, 0)).now(),
        parquet_root=archived,
        dataset_dir=d,
    )
    target = next(iter(sorted(archived.rglob("*.parquet"))))
    target.write_bytes(target.read_bytes() + b"\x00")

    ok, problems = verify_dataset_version("v1", parquet_root=archived, dataset_dir=d)
    assert not ok
    assert any("content changed" in p for p in problems)


def test_dataset_version_detects_deleted_data(recorded, tmp_path):
    archived = tmp_path / "parquet"  # own copy: this test deletes a file
    archive_all(TEST_DAY, raw_root=recorded, parquet_root=archived)
    d = tmp_path / "datasets"
    build_dataset_version(
        "v1",
        TEST_DAY,
        TEST_DAY,
        FrozenClock(ist(18, 0)).now(),
        parquet_root=archived,
        dataset_dir=d,
    )
    next(iter(sorted(archived.rglob("*.parquet")))).unlink()
    ok, problems = verify_dataset_version("v1", parquet_root=archived, dataset_dir=d)
    assert not ok and any("missing" in p for p in problems)


def test_unknown_dataset_version_is_reported(tmp_path):
    ok, problems = verify_dataset_version("nope", parquet_root=tmp_path, dataset_dir=tmp_path)
    assert not ok and problems
