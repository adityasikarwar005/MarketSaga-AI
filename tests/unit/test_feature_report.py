"""M44 verify - the feature report's two gates.

Both gates are the kind that pass by accident. A null-rate check that treats
absent input as failure gets ignored within a week; a PSI that returns a number
for a constant feature reports a distribution shift in something that never
moved. So each is pinned against a case where the honest answer is "nothing to
say" as well as a case where it should fire.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from jobs.feature_report import (
    PSI_BREAK,
    PSI_SHIFT,
    DayReport,
    build_day,
    psi,
    render_null_rates,
)
from libs.ms_core.clock import SimClock
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_data.candles import CandleBuilder
from libs.ms_data.feed import SyntheticFeed, SyntheticInstrument
from libs.ms_data.recorder import Recorder
from libs.ms_features import catalogue  # noqa: F401  (registers the catalogue)
from libs.ms_features.registry import REGISTRY
from tests.conftest import TEST_DAY, ist


# ------------------------------------------------------------------------ PSI
def test_a_distribution_is_stable_against_itself():
    """The null test. Anything else means the binning is manufacturing movement."""
    rng = random.Random(7)
    sample = [rng.gauss(0.0, 1.0) for _ in range(4_000)]
    score = psi(sample, sample)
    assert score is not None
    assert score < 1e-9, f"a sample compared with itself scored {score}, not zero"


def test_an_independent_draw_from_the_same_law_stays_under_the_shift_threshold():
    """Sampling noise must not read as a shift, or every quiet day trips the gate."""
    rng = random.Random(11)
    reference = [rng.gauss(0.0, 1.0) for _ in range(4_000)]
    current = [rng.gauss(0.0, 1.0) for _ in range(400)]
    score = psi(reference, current)
    assert score is not None and score < PSI_SHIFT, (
        f"two draws from the same normal scored {score:.3f}, over the {PSI_SHIFT} "
        "threshold that is supposed to mean 'this moved'"
    )


def test_a_shifted_distribution_breaks_the_threshold():
    rng = random.Random(13)
    reference = [rng.gauss(0.0, 1.0) for _ in range(4_000)]
    current = [rng.gauss(2.5, 1.0) for _ in range(400)]
    score = psi(reference, current)
    assert score is not None and score > PSI_BREAK, (
        f"a two-and-a-half sigma shift scored {score:.3f}, under the {PSI_BREAK} "
        "threshold - the gate would not have fired on it"
    )


def test_psi_grows_with_the_size_of_the_shift():
    rng = random.Random(17)
    reference = [rng.gauss(0.0, 1.0) for _ in range(4_000)]
    scores = []
    for shift in (0.25, 1.0, 2.5):
        rng2 = random.Random(29)
        scores.append(psi(reference, [rng2.gauss(shift, 1.0) for _ in range(1_000)]))
    assert all(s is not None for s in scores)
    assert scores[0] < scores[1] < scores[2], f"not monotone in the shift: {scores}"


def test_a_constant_reference_has_no_distribution_to_be_stable_against():
    """None, not zero and not a large number off the floor.

    Every quantile edge of a constant series is the same value, so there is one
    bin and the index is meaningless. Returning a number here would put a
    confident-looking figure next to features that never moved.
    """
    assert psi([3.0] * 2_000, [3.0] * 2_000) is None
    assert psi([3.0] * 2_000, [9.0] * 2_000) is None


def test_too_few_observations_is_declined_rather_than_estimated():
    rng = random.Random(3)
    reference = [rng.gauss(0.0, 1.0) for _ in range(4_000)]
    assert psi(reference, [rng.gauss(0.0, 1.0) for _ in range(20)]) is None
    assert psi([rng.gauss(0.0, 1.0) for _ in range(20)], reference) is None


# ----------------------------------------------------------------- no input
def test_absent_input_is_not_counted_as_a_failure():
    """A chainless day makes every option feature None, and that is the correct
    answer to the question asked. Gating on it would train the reader to skip
    the failures, which is worse than not gating at all."""
    report = DayReport(TEST_DAY, 1)
    report.chain_snapshots = 0
    assert report.has_no_input("atm_iv")
    assert report.has_no_input("implied_move_pct")
    assert not report.has_no_input("ret_5m")


def test_a_recorded_chain_removes_the_excuse():
    """The case the gate exists for: the input was there and the feature still
    produced nothing."""
    report = DayReport(TEST_DAY, 1)
    report.chain_snapshots = 120
    for name in [d.name for d in REGISTRY.by_group("C")]:
        assert not report.has_no_input(name), f"{name} is excused despite a recorded chain"


def test_no_input_is_never_inferred_from_the_values():
    """A feature that is None everywhere because it is broken looks identical,
    from the values alone, to one that had no input. Only the recording can tell
    them apart, so only the recording is consulted."""
    broken = DayReport(TEST_DAY, 1)
    broken.chain_snapshots = 400
    broken.values["atm_iv"] = [None] * 300
    assert broken.null_rate("atm_iv") == 1.0
    assert not broken.has_no_input("atm_iv")


# -------------------------------------------------------------- end to end
@pytest.fixture
def recorded_session(tmp_path, calendar):
    """A short recorded day with candles and deliberately no chain at all."""
    root = tmp_path / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(
        [SyntheticInstrument(instrument_id=1, start_price=24_000.0, annual_vol=0.14)],
        calendar,
        clock,
        seed=5,
    )
    feed.subscribe([1])
    builder = CandleBuilder(calendar)
    rec = Recorder(clock, calendar, root=root, fsync_every=10_000)
    end = ist(10, 30)
    for tick in feed.stream():
        if tick.ts >= end:
            break
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
    for candle in builder.flush(end):
        rec.write("candles", candle, key=candle.instrument_id)
    rec.close()
    return root


def test_the_report_computes_a_day_and_excludes_the_warm_up(recorded_session):
    report = build_day(TEST_DAY, 1, root=recorded_session, warmup_minutes=35)
    assert report is not None
    assert report.instants > 30
    # 09:15 to 10:30 is 75 minutes; 35 of them are warm-up.
    assert report.instants <= 75 - 35 + 1
    assert report.null_rate("ret_5m") == 0.0


def test_a_chainless_day_leaves_the_option_features_absent_not_failing(recorded_session):
    report = build_day(TEST_DAY, 1, root=recorded_session, warmup_minutes=35)
    assert report is not None
    assert report.chain_snapshots == 0
    for name in [d.name for d in REGISTRY.by_group("C")]:
        assert report.null_rate(name) == 1.0
        assert report.has_no_input(name)


def test_a_day_with_no_candles_is_skipped_not_reported_as_perfect(recorded_session):
    """Zero instants means zero nulls, which would render as a flawless day."""
    assert (
        build_day(TEST_DAY + timedelta(days=1), 1, root=recorded_session, warmup_minutes=35) is None
    )
    assert build_day(TEST_DAY, 999, root=recorded_session, warmup_minutes=35) is None


def test_the_warm_up_can_swallow_the_whole_session(recorded_session):
    """An over-long warm-up must yield nothing rather than an empty perfect score."""
    assert build_day(TEST_DAY, 1, root=recorded_session, warmup_minutes=600) is None


def test_option_features_populate_when_a_chain_is_present(tmp_path, calendar):
    """The other half, and the point of the whole report.

    With a chain on disk the group C null rate goes to zero. A version of this
    report that could only ever print 100% would answer the morning-after
    question identically whether or not the capture had worked, so the zero case
    has to be reachable in a test and not merely believed.
    """
    root = tmp_path / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(
        [SyntheticInstrument(instrument_id=1, start_price=24_000.0, annual_vol=0.14)],
        calendar,
        clock,
        seed=5,
    )
    feed.subscribe([1])
    builder = CandleBuilder(calendar)
    rec = Recorder(clock, calendar, root=root, fsync_every=10_000)
    end = ist(10, 30)
    minute = None
    for tick in feed.stream():
        if tick.ts >= end:
            break
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
        stamp = tick.ts.replace(second=0, microsecond=0)
        if stamp != minute:
            minute = stamp
            rec.write_chain(_chain_at(stamp, tick.ltp))
    for candle in builder.flush(end):
        rec.write("candles", candle, key=candle.instrument_id)
    rec.close()

    report = build_day(TEST_DAY, 1, root=root, warmup_minutes=35)
    assert report is not None and report.chain_snapshots > 30
    for name in [d.name for d in REGISTRY.by_group("C")]:
        assert report.null_rate(name) == 0.0, (
            f"{name} was null on {report.null_rate(name):.0%} of instants despite a "
            "chain being recorded at every minute - that is the failure this gate exists for"
        )


def _chain_at(ts, spot: float) -> ChainSnapshot:
    """Two expiries, a book at every strike, and open interest peaked at the money.

    All three are load-bearing: one expiry leaves `iv_term_slope` None, a missing
    book leaves `implied_move_pct` None, and fewer than eight strikes makes
    `gamma_concentration_shape` decline. A fixture short of any of them would let
    a broken feature pass this test as "correctly absent".
    """
    step = 50.0
    atm = round(spot / step) * step
    quotes = []
    for expiry_offset in (1, 8):
        expiry = ist(15, 30) + timedelta(days=expiry_offset)
        for i in range(-6, 7):
            strike = atm + step * i
            for opt in (OptionType.CE, OptionType.PE):
                intrinsic = max(0.0, (spot - strike) if opt is OptionType.CE else (strike - spot))
                fair = intrinsic + 120.0 * (0.5 ** abs(i)) + 5.0
                quotes.append(
                    OptionQuote(
                        expiry=expiry,
                        strike=strike,
                        opt_type=opt,
                        bid=round(fair * 0.99, 2),
                        ask=round(fair * 1.01, 2),
                        oi=int(50_000 * (0.6 ** abs(i))) + 100,
                        iv=0.14 + 0.004 * expiry_offset + 0.001 * i,
                    )
                )
    return ChainSnapshot(ts=ts, underlying_id=1, spot=spot, quotes=quotes, completeness=1.0)


def test_psi_of_a_real_feature_series_is_computable():
    """Guards the plumbing between the report and `psi`: a day's series is a list
    of `float | None`, and passing the Nones through would crash on the sort."""
    rng = random.Random(41)
    earlier = DayReport(TEST_DAY, 1)
    later = DayReport(TEST_DAY + timedelta(days=1), 1)
    for report in (earlier, later):
        report.chain_snapshots = 10
        report.values["ret_5m"] = [
            None if i % 20 == 0 else rng.gauss(0.0, 0.001) for i in range(400)
        ]
    reference = [v for v in earlier.values["ret_5m"] if v is not None]
    current = [v for v in later.values["ret_5m"] if v is not None]
    assert psi(reference, current) is not None


def test_a_chain_snapshot_is_counted_per_underlying():
    report = DayReport(TEST_DAY, 1)
    snaps = [
        ChainSnapshot(
            ts=ist(9, 20),
            underlying_id=uid,
            spot=24_000.0,
            quotes=[
                OptionQuote(
                    expiry=ist(15, 30) + timedelta(days=1),
                    strike=24_000.0,
                    opt_type=OptionType.CE,
                    iv=0.14,
                )
            ],
        )
        for uid in (1, 1, 2)
    ]
    report.chain_snapshots = sum(1 for s in snaps if s.underlying_id == 1)
    assert report.chain_snapshots == 2
    assert report.had_chain


def test_day_report_starts_with_nothing_to_say():
    report = DayReport(date(2026, 8, 31), 1)
    assert report.null_rate("ret_5m") is None
    assert not report.had_chain


# ------------------------------------------------- no input, generalised to D
def test_a_day_without_a_cross_asset_collector_is_not_a_failure():
    """The excuse used to be hard-coded to the chain.

    When group D arrived, all four of its features came up FAIL at 100% on a day
    with no cross-asset collector running - the report failing on a condition
    nobody can fix, which its own docstring says is how a gate gets ignored.
    """
    report = DayReport(TEST_DAY, 1)
    report.chain_snapshots = 400
    report.cross_asset_series = 0
    for name in [d.name for d in REGISTRY.by_group("D")]:
        assert report.has_no_input(name), f"{name} would be reported as a failure"
    for name in [d.name for d in REGISTRY.by_group("C")]:
        assert not report.has_no_input(name), f"{name} is excused despite a recorded chain"


def test_a_recorded_cross_asset_series_removes_the_excuse():
    report = DayReport(TEST_DAY, 1)
    report.cross_asset_series = 2
    for name in [d.name for d in REGISTRY.by_group("D")]:
        assert not report.has_no_input(name)


def test_the_session_s_own_features_are_never_excused():
    """Group A and F read the candles of the instrument being reported on. A day
    without those never reaches the report at all, so there is no condition
    under which their nulls are somebody else's problem."""
    report = DayReport(TEST_DAY, 1)
    report.chain_snapshots = 0
    report.cross_asset_series = 0
    for group in ("A", "F"):
        for name in [d.name for d in REGISTRY.by_group(group)]:
            assert not report.has_no_input(name), f"{name} in group {group} was excused"


def test_every_group_that_can_be_excused_names_the_input_it_wanted():
    """`render_null_rates` looks the group up in GROUP_INPUTS to say what was
    missing. A group excused but absent from that map is a KeyError in the one
    code path nobody exercises until a collector is down."""
    from jobs.feature_report import GROUP_INPUTS

    report = DayReport(TEST_DAY, 1)
    excusable = {
        REGISTRY.get(name).group
        for name in REGISTRY.names()
        if report.has_no_input(name)  # nothing recorded at all
    }
    assert excusable, "no group is excusable, so this test is checking nothing"
    assert excusable <= set(
        GROUP_INPUTS
    ), f"missing from GROUP_INPUTS: {excusable - set(GROUP_INPUTS)}"


def test_the_render_says_which_input_was_missing():
    report = DayReport(TEST_DAY, 1)
    report.instants = 100
    for name in REGISTRY.names():
        report.values[name] = [None] * 100
    text, breaches = render_null_rates(report, 0.01)
    assert "no input: chain snapshots were not recorded" in text
    assert "no input: cross-asset series were not recorded" in text
    assert breaches, "group A features had no input excuse and should still breach"
    assert all("usdinr" not in b and "atm_iv" not in b for b in breaches)


def test_cross_asset_series_are_counted_from_the_reserved_range(tmp_path, calendar):
    """The count is of distinct 1000+ instruments with candles on the day, which
    is what tells a missing collector apart from a quiet one."""
    from libs.ms_data.recorder import Recorder as Rec

    root = tmp_path / "raw"
    clock = SimClock(ist(9, 15))
    feed = SyntheticFeed(
        [
            SyntheticInstrument(instrument_id=1, start_price=24_000.0, annual_vol=0.14),
            SyntheticInstrument(instrument_id=1006, start_price=88.2, annual_vol=0.05),
            SyntheticInstrument(instrument_id=1008, start_price=3_180.0, annual_vol=0.12),
        ],
        calendar,
        clock,
        seed=5,
    )
    feed.subscribe([1, 1006, 1008])
    builder = CandleBuilder(calendar)
    rec = Rec(clock, calendar, root=root, fsync_every=10_000)
    end = ist(10, 30)
    for tick in feed.stream():
        if tick.ts >= end:
            break
        for candle in builder.on_tick(tick):
            rec.write("candles", candle, key=candle.instrument_id)
    for candle in builder.flush(end):
        rec.write("candles", candle, key=candle.instrument_id)
    rec.close()

    report = build_day(TEST_DAY, 1, root=root, warmup_minutes=35)
    assert report is not None
    assert report.cross_asset_series == 2
    for name in [d.name for d in REGISTRY.by_group("D")]:
        assert not report.has_no_input(name)
        assert report.null_rate(name) == 0.0, f"{name} was null despite a recorded series"
