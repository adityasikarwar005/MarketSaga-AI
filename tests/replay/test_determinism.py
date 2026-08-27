"""Gate G1, first half: determinism.

The requirement is blunt - replay the same input twice and get a byte-identical
result. Without it, nothing downstream can be debugged: a backtest that differs
run to run cannot distinguish a code change from noise, and a discrepancy
between simulated and live behaviour can never be attributed.

These tests run in CI on every commit and are never marked skip.
"""

from __future__ import annotations

import hashlib
import json

from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import SimClock
from libs.ms_core.schemas import Tick, deterministic_ids
from libs.ms_data.candles import CandleBuilder, verify_session
from libs.ms_data.feed import SyntheticFeed, SyntheticInstrument
from libs.ms_data.recorder import Recorder, read_stream
from tests.conftest import ist

INSTRUMENTS = [
    SyntheticInstrument(instrument_id=1, start_price=24_800, annual_vol=0.14),
    SyntheticInstrument(instrument_id=2, start_price=54_200, annual_vol=0.18),
]
SEED = 20260826


def _run_pipeline(calendar, root, *, seed: int = SEED, end=None):
    """Feed -> bus -> recorder -> candles. The production path, on synthetic input.

    Wrapped in ``deterministic_ids`` exactly as the replay engine (M18) will be:
    message ids must be reproducible or a byte-identical re-run is impossible.
    """
    with deterministic_ids():
        return _pipeline_body(calendar, root, seed, end)


def _pipeline_body(calendar, root, seed, end):
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(INSTRUMENTS, calendar, clock, seed=seed)
    feed.subscribe([1, 2])
    bus = InMemoryBus()
    builder = CandleBuilder(calendar)
    recorder = Recorder(clock, calendar, root=root, fsync_every=10_000)

    end = end or ist(10, 15)
    candles = []
    for tick in feed.stream():
        if tick.ts >= end:
            break
        bus.publish(Topic.MARKET_TICK, tick)
        recorder.write_tick(tick)
        candles.extend(builder.on_tick(tick))
    candles.extend(builder.flush(end))
    recorder.close()
    return bus, candles


def _hash(objs) -> str:
    """Full hash, trace ids included. For comparing two runs of the SAME path."""
    h = hashlib.sha256()
    for o in objs:
        h.update(o.model_dump_json(by_alias=True).encode())
    return h.hexdigest()


def _content_hash(objs) -> str:
    """Hash of the data only, trace ids excluded.

    For comparing two DIFFERENT paths to the same answer - live construction
    versus rebuilding from disk. A trace id records how a message travelled, not
    what it says, and the two journeys are legitimately different. The prices,
    volumes and timestamps must be identical; the provenance stamps must not be.
    """
    h = hashlib.sha256()
    for o in objs:
        h.update(o.model_dump_json(by_alias=True, exclude={"trace_id"}).encode())
    return h.hexdigest()


def test_candles_are_byte_identical_across_runs(calendar, tmp_path):
    _, a = _run_pipeline(calendar, tmp_path / "a")
    _, b = _run_pipeline(calendar, tmp_path / "b")
    assert a, "pipeline produced no candles"
    assert _hash(a) == _hash(b)


def test_recorded_files_are_byte_identical_across_runs(calendar, tmp_path):
    _run_pipeline(calendar, tmp_path / "a")
    _run_pipeline(calendar, tmp_path / "b")

    files_a = sorted((tmp_path / "a").rglob("*.ndjson"))
    files_b = sorted((tmp_path / "b").rglob("*.ndjson"))
    assert files_a, "recorder wrote nothing"
    assert [f.name for f in files_a] == [f.name for f in files_b]

    for fa, fb in zip(files_a, files_b, strict=True):
        assert fa.read_bytes() == fb.read_bytes(), f"{fa.name} differed between runs"


def test_manifest_hashes_match_across_runs(calendar, tmp_path):
    _run_pipeline(calendar, tmp_path / "a")
    _run_pipeline(calendar, tmp_path / "b")

    def manifests(root):
        return {
            p.parent.name + "/" + p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(root.rglob("_manifest.json"))
        }

    ma, mb = manifests(tmp_path / "a"), manifests(tmp_path / "b")
    assert ma and ma.keys() == mb.keys()
    for key in ma:
        assert [e["sha256"] for e in ma[key]] == [e["sha256"] for e in mb[key]]


def test_round_trip_through_disk_preserves_every_tick(calendar, tmp_path):
    """What was written is exactly what was received - no lossy serialisation."""
    root = tmp_path / "r"
    bus, _ = _run_pipeline(calendar, root)
    published = [d.typed(Tick) for d in bus.subscribe(Topic.MARKET_TICK, "verify", "c")]

    recovered: list[Tick] = []
    for path in sorted(root.rglob("ticks-*.ndjson")):
        recovered.extend(read_stream(path, Tick))  # type: ignore[arg-type]

    assert len(recovered) == len(published)
    assert {t.model_dump_json() for t in recovered} == {t.model_dump_json() for t in published}


def test_candles_rebuilt_from_disk_match_the_live_ones(calendar, tmp_path):
    """The offline path and the online path must agree. Same idea as the M43
    feature parity test, applied one layer down."""
    root = tmp_path / "r"
    _, live = _run_pipeline(calendar, root)

    ticks: list[Tick] = []
    for path in sorted(root.rglob("ticks-*.ndjson")):
        ticks.extend(read_stream(path, Tick))  # type: ignore[arg-type]
    ticks.sort(key=lambda t: (t.ts, t.instrument_id, t.seq or 0))

    rebuilt = CandleBuilder(calendar).build(ticks, up_to=ist(10, 15))
    key = lambda c: (c.ts, c.instrument_id)  # noqa: E731
    assert len(rebuilt) == len(live)
    assert _content_hash(sorted(live, key=key)) == _content_hash(sorted(rebuilt, key=key))


def test_full_session_replay_yields_a_complete_grid(calendar, tmp_path):
    """375 one-minute candles per instrument, no more, no less."""
    _, candles = _run_pipeline(calendar, tmp_path / "full", end=ist(15, 30))
    reports = verify_session(candles, calendar)
    assert len(reports) == 2
    for r in reports:
        assert r.expected == 375
        assert r.actual == 375, f"instrument {r.instrument_id}: {r.actual} candles"
        assert not r.inconsistent
        assert not r.duplicates


def test_different_seed_produces_a_different_hash(calendar, tmp_path):
    """Guards against a determinism test that passes because nothing varies."""
    _, a = _run_pipeline(calendar, tmp_path / "a", seed=1)
    _, b = _run_pipeline(calendar, tmp_path / "b", seed=2)
    assert _hash(a) != _hash(b)
