"""M43 verify - offline/online parity and leakage detection.

The single most important test in the repository.

Train/serve skew is the reason a backtest works and a live system does not, and
it is invisible: both sides run, both produce numbers, and the numbers differ by
a little. This test replays a recorded day through the online path, materialises
the same day offline, and asserts every feature matches at every instant.

If it passes, a model trained offline sees the same inputs live. If it fails,
nothing downstream can be trusted, which is why it runs on every commit and is
never marked skip.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.bus import InMemoryBus, Topic
from libs.ms_core.clock import SessionCalendar, SimClock
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.chain import ChainBuilder, ChainSpec, build_registry_from_specs
from libs.ms_data.feed import SyntheticFeed, SyntheticInstrument
from libs.ms_data.recorder import Recorder
from libs.ms_data.replay import build_engine
from libs.ms_data.synthetic_chain import SyntheticChainFeed, strike_grid
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.leakage import (
    RecordingSource,
    compare_vectors,
    detect_leakage,
    lint_all,
    lint_feature,
)
from libs.ms_features.registry import REGISTRY, FeatureContext, FeatureRegistry, feature
from libs.ms_features.sources import OfflineSource, OnlineSource
from tests.conftest import TEST_DAY, ist

UNDERLYING = 1
SPOT = 24_800.0
# Group D reads these through the ordinary candle accessor - cross-asset series
# share the Candle schema and sit in the reserved 1000+ range - so parity for
# them is parity for everything else, provided the fixture actually records them.
USDINR_ID = 1006
ASIA_IDX_ID = 1008
INSTRUMENTS = [
    SyntheticInstrument(instrument_id=UNDERLYING, start_price=SPOT, annual_vol=0.14),
    SyntheticInstrument(instrument_id=USDINR_ID, start_price=88.20, annual_vol=0.05),
    SyntheticInstrument(instrument_id=ASIA_IDX_ID, start_price=3_180.0, annual_vol=0.12),
]


@pytest.fixture(scope="module")
def calendar_mod():
    import warnings

    from libs.ms_core import config as cfgmod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = cfgmod.load()
    return SessionCalendar.from_config(cfg.sessions).synthetic_variant()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, calendar_mod):
    """A recorded session with candles and a chain - the parity input."""
    cal = calendar_mod
    raw = tmp_path_factory.mktemp("parity_src") / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(INSTRUMENTS, cal, clock, seed=23)
    feed.subscribe([UNDERLYING, USDINR_ID, ASIA_IDX_ID])

    # Two expiries, not one. `iv_term_slope` is the difference between them and
    # a single-expiry fixture would leave it None at every instant - which the
    # parity test would happily pass, because None equals None on both sides.
    # Nine strikes a side clears the eight `gamma_concentration_shape` needs
    # before it will call a handful of contracts a profile.
    registry = build_registry_from_specs(
        UNDERLYING,
        [TEST_DAY + timedelta(days=2), TEST_DAY + timedelta(days=9)],
        strike_grid(SPOT, step=50.0, n_per_side=12),
    )
    chain_feed = SyntheticChainFeed(registry, cal, underlying_id=UNDERLYING, seed=8)
    chain_builder = ChainBuilder(
        ChainSpec(underlying_id=UNDERLYING, n_expiries=2, strikes_per_side=9), registry, cal
    )
    builder = CandleBuilder(cal)
    rec = Recorder(clock, cal, root=raw, fsync_every=10_000)
    end = ist(11, 0)  # long enough for every window in the catalogue to fill

    for tick in feed.stream():
        if tick.ts >= end:
            break
        rec.write_tick(tick)
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
        # Only the underlying prices the chain. A USD/INR tick carries an `ltp`
        # of about 88, and feeding it to `quotes_for` would build a strike
        # ladder around 88 rupees without complaining about it.
        if tick.instrument_id != UNDERLYING:
            continue
        chain_builder.on_tick(tick)
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
def parity_run(recorded):
    """Replay the day online while collecting the offline history alongside.

    Vectors are computed at the same instants on both sides, from the same
    definitions, through the same registry - the only difference is which
    ContextSource is behind the context.
    """
    online = OnlineSource(max_minutes=400)
    bus = InMemoryBus()
    engine, clock = build_engine(bus, TEST_DAY, root=recorded)

    all_candles, all_chains = [], []
    online_vectors: dict = {}

    def consume(topic, msg):
        if topic is Topic.MARKET_CANDLE:
            online.on_candle(msg)
            all_candles.append(msg)
            # Compute at the replay's CURRENT instant, never ahead of it. Asking
            # for a future as_of would mean the online source has legitimately
            # not received data the offline source already holds, and the
            # resulting mismatch would be the harness's fault, not the code's.
            as_of = clock.now()
            online_vectors[as_of] = REGISTRY.compute(online, UNDERLYING, as_of)
        elif topic is Topic.MARKET_CHAIN:
            online.on_chain(msg)
            all_chains.append(msg)

    engine.run([TEST_DAY], on_message=consume)

    offline = OfflineSource(candles=all_candles, chains=all_chains)
    offline_vectors = {
        as_of: REGISTRY.compute(offline, UNDERLYING, as_of) for as_of in online_vectors
    }
    return offline_vectors, online_vectors, offline, online, all_candles, all_chains


# --------------------------------------------------------------------- parity
def test_offline_and_online_agree_on_every_value(parity_run):
    offline_vectors, online_vectors, _offline, online, _c, _ch = parity_run
    assert len(online_vectors) > 60, "not enough instants to be a meaningful test"

    mismatches = []
    compared = 0
    for as_of, online_values in sorted(online_vectors.items()):
        offline_values = offline_vectors[as_of]
        compared += len(offline_values)
        mismatches.extend(compare_vectors(offline_values, online_values, as_of))

    assert compared > 1000
    assert not mismatches, "train/serve skew:\n  " + "\n  ".join(str(m) for m in mismatches[:10])
    assert not online.short_windows, (
        "the online buffer was shorter than a feature asked for - the values happen to "
        "match here, but they will not once the buffer rolls:\n  "
        + "\n  ".join(str(s) for s in online.short_windows[:5])
    )


def test_parity_covers_every_registered_feature(parity_run):
    """A feature absent from the comparison is a feature nobody is checking."""
    offline_vectors, _online, _os, _on, _c, _ch = parity_run
    seen: set[str] = set()
    for values in offline_vectors.values():
        seen.update(values)
    assert seen == set(REGISTRY.names())


def test_features_actually_produce_values(parity_run):
    """Guards against a parity test that passes because both sides are all None."""
    offline_vectors, _online, _os, _on, _c, _ch = parity_run
    late = [v for as_of, v in sorted(offline_vectors.items())][-1]
    computed = {k: v for k, v in late.items() if v is not None}
    assert len(computed) >= 18, f"only {len(computed)} features produced a value"


def test_every_registered_feature_produces_a_value_somewhere(parity_run):
    """None on both sides is parity, and it is worthless.

    `test_parity_covers_every_registered_feature` proves each feature was
    *compared*; it does not prove either side ever computed anything. A feature
    whose chain lookup is broken, or whose expiry filter excludes everything, is
    None offline and None online and sails through every assertion above. The
    whole of group C was added against a fixture that had one expiry, where
    `iv_term_slope` was structurally None at every instant - exactly this hole.
    """
    offline_vectors, _online, _os, _on, _c, _ch = parity_run
    ever: set[str] = set()
    for values in offline_vectors.values():
        ever.update(k for k, v in values.items() if v is not None)
    missing = sorted(set(REGISTRY.names()) - ever)
    assert not missing, (
        "these features were never computed at any instant of the replay - they are "
        f"None on both sides, so parity is meaningless for them: {missing}"
    )


def test_comparison_detects_a_real_difference():
    """The comparator must not pass everything."""
    a = {"x": 1.0, "y": None}
    assert compare_vectors(a, dict(a), ist(9, 20)) == []
    assert compare_vectors(a, {"x": 1.5, "y": None}, ist(9, 20))
    assert compare_vectors(a, {"x": 1.0, "y": 2.0}, ist(9, 20))


def test_comparison_tolerates_floating_point_noise():
    a = {"x": 1.0}
    b = {"x": 1.0 + 1e-15}
    assert compare_vectors(a, b, ist(9, 20)) == []


# ------------------------------------------------------------------ leakage
def test_no_registered_feature_reads_the_future(parity_run):
    offline_vectors, _online, offline, _on, _c, _ch = parity_run
    as_of = sorted(offline_vectors)[len(offline_vectors) // 2]
    leaks = detect_leakage(offline, UNDERLYING, as_of)
    assert not leaks, "features read data from after their as_of:\n  " + "\n  ".join(
        str(x) for x in leaks
    )


def test_the_detector_catches_a_leaky_source(parity_run):
    """A detector nobody has watched fire is not a detector.

    The realistic leak is not a rogue feature - the context makes that hard -
    but a buggy ContextSource. A chain lookup that rounds the wrong way returns
    the NEXT snapshot instead of the previous one, and every feature built on it
    silently sees a few seconds into the future. This is a bug I nearly wrote:
    `bisect_right(keys, as_of)` without stepping back.
    """
    _ov, _on, _offline, _online, all_candles, all_chains = parity_run

    class OffByOneSource(OfflineSource):
        """A bisect that never steps back - returns the NEXT snapshot."""

        def chain(self, underlying_id, as_of):
            series = self._chains.get(underlying_id) or []
            future = [s for s in series if s.ts > as_of]
            return future[0] if future else super().chain(underlying_id, as_of)

    leaky = OffByOneSource(candles=all_candles, chains=all_chains)

    leaks = detect_leakage(leaky, UNDERLYING, ist(10, 0))
    assert leaks, "an off-by-one chain lookup went undetected"
    assert all(x.seconds_ahead > 0 for x in leaks)
    assert {x.feature for x in leaks} & {"atm_iv", "chain_age_seconds"}


def test_recording_source_tracks_the_latest_timestamp(parity_run):
    _ov, _on2, offline, _o, _c, _ch = parity_run
    rec = RecordingSource(offline)
    ctx = FeatureContext(rec, UNDERLYING, ist(10, 0))
    ctx.candles(minutes=30)
    assert rec.latest is not None and rec.latest < ist(10, 0)
    rec.reset()
    assert rec.latest is None


# -------------------------------------------------------------- static lint
def test_the_catalogue_passes_the_linter():
    report = lint_all()
    assert report.checked >= 20
    assert report.ok, report.render()


def test_linter_rejects_wall_clock_access():
    """The dynamic detector cannot see this: a feature reading the clock never
    asks the context for anything."""
    reg = FeatureRegistry()

    @feature(name="clock_reader", version=1, registry=reg)
    def clock_reader(ctx: FeatureContext) -> float | None:
        from datetime import datetime

        return float(datetime.now().hour)

    findings = lint_feature(reg.get("clock_reader"))
    assert any("now" in f.problem for f in findings)


def test_linter_rejects_direct_storage_access():
    reg = FeatureRegistry()

    @feature(name="archive_reader", version=1, registry=reg)
    def archive_reader(ctx: FeatureContext) -> float | None:
        import duckdb

        return float(duckdb.__version__[0])

    findings = lint_feature(reg.get("archive_reader"))
    assert any("duckdb" in f.problem for f in findings)


def test_linter_rejects_a_feature_that_ignores_its_context():
    """Reading from a module global produces a number with no point-in-time
    filter anywhere near it."""
    reg = FeatureRegistry()

    @feature(name="constant", version=1, registry=reg)
    def constant(ctx: FeatureContext) -> float | None:
        return 42.0

    findings = lint_feature(reg.get("constant"))
    assert any("never uses its context" in f.problem for f in findings)


def test_lint_report_renders_findings():
    reg = FeatureRegistry()

    @feature(name="bad", version=1, registry=reg)
    def bad(ctx: FeatureContext) -> float | None:
        return 1.0

    report = lint_all(reg)
    assert not report.ok
    assert "bad" in report.render()
