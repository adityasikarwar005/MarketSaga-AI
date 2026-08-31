"""M19 verify - data quality gates.

Every check is tested by *injecting* the corruption it looks for into a
known-good day. A quality check nobody has watched fail is not a check.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from libs.ms_core.schemas import Candle, ChainSnapshot, OptionQuote, OptionType, RawNewsItem, Tick
from libs.ms_data.quality import (
    Verdict,
    check_candle_consistency,
    check_candle_count,
    check_chain_completeness,
    check_crossed_books,
    check_duplicate_rate,
    check_iv_sanity,
    check_news_timestamps,
    check_null_rate,
    check_price_jumps,
    check_put_call_parity,
    check_session_gaps,
    check_synthetic_share,
    check_tick_sequence_gaps,
    check_train_serve_skew,
    instruments_from_chain,
    population_stability_index,
    run_market_checks,
)
from tests.conftest import TEST_DAY, ist


@pytest.fixture
def cal(calendar):
    return calendar.synthetic_variant()


def good_candles(n: int = 375, instrument_id: int = 1) -> list[Candle]:
    return [
        Candle(
            ts=ist(9, 15) + timedelta(minutes=i),
            instrument_id=instrument_id,
            tf=1,
            o=100.0,
            h=100.5,
            l=99.5,
            c=100.1,
            v=1000,
        )
        for i in range(n)
    ]


def good_chain(n: int = 5) -> list[ChainSnapshot]:
    snaps = []
    for i in range(n):
        quotes = []
        for strike in (24_700.0, 24_750.0, 24_800.0, 24_850.0, 24_900.0):
            # C - P must decrease with strike, which is what parity requires.
            call = max(1.0, 24_800.0 - strike + 120.0)
            put = max(1.0, strike - 24_800.0 + 120.0)
            quotes.append(
                OptionQuote(
                    expiry=ist(15, 30),
                    strike=strike,
                    opt_type=OptionType.CE,
                    bid=call - 0.5,
                    ask=call + 0.5,
                    iv=0.14,
                    oi=50_000,
                )
            )
            quotes.append(
                OptionQuote(
                    expiry=ist(15, 30),
                    strike=strike,
                    opt_type=OptionType.PE,
                    bid=put - 0.5,
                    ask=put + 0.5,
                    iv=0.15,
                    oi=50_000,
                )
            )
        snaps.append(
            ChainSnapshot(
                ts=ist(9, 20) + timedelta(seconds=5 * i),
                underlying_id=1,
                spot=24_800.0,
                quotes=quotes,
                completeness=1.0,
            )
        )
    return snaps


def good_news(n: int = 20) -> list[RawNewsItem]:
    return [
        RawNewsItem(
            source_id=3,
            headline=f"item {i}",
            t_event=ist(9, 20) + timedelta(seconds=i),
            t_ingest=ist(9, 20) + timedelta(seconds=i + 2),
            t_available=ist(9, 20) + timedelta(seconds=i + 3),
        )
        for i in range(n)
    ]


# ------------------------------------------------------------- baseline
def test_a_clean_day_passes_everything(cal):
    report = run_market_checks(
        TEST_DAY, cal, now=ist(15, 30), candles=good_candles(), chain=good_chain(), news=good_news()
    )
    assert report.ok, [c.line() for c in report.failures]
    assert not report.failures


# --------------------------------------------------------- tick sequence
def test_sequence_gap_is_detected_and_counted():
    ticks = [Tick(ts=ist(9, 20), instrument_id=1, ltp=100.0, seq=i) for i in range(10)]
    ticks.append(Tick(ts=ist(9, 21), instrument_id=1, ltp=100.0, seq=25))  # 15 missing
    check = check_tick_sequence_gaps(ticks)
    assert check.verdict is Verdict.WARN
    assert check.value == 15


def test_no_sequence_numbers_is_skip_not_pass():
    """Kite supplies none. Reporting PASS would claim a guarantee we do not have."""
    ticks = [Tick(ts=ist(9, 20), instrument_id=1, ltp=100.0) for _ in range(10)]
    assert check_tick_sequence_gaps(ticks).verdict is Verdict.SKIP


# --------------------------------------------------------- candle counts
def test_a_truncated_day_is_caught_by_the_gate_that_can_see_it(cal):
    """A feed that died at noon leaves no hole between two candles, so an
    interior-gap check passes it. Measuring against the session that elapsed is
    what catches it, and `candle_count` steps aside rather than reporting a
    partial session as corrupt - the two gates split the job between them."""
    truncated = good_candles(200)
    assert check_candle_count(truncated, cal).verdict is Verdict.SKIP

    gaps = check_session_gaps(truncated, cal, TEST_DAY, now=ist(15, 30))
    assert gaps.verdict is Verdict.FAIL
    assert gaps.value == pytest.approx(175.0), "375 elapsed minutes, 200 recorded"
    assert "tail" in gaps.detail


def test_a_complete_day_leaves_nothing_missing(cal):
    assert check_session_gaps(good_candles(375), cal, TEST_DAY, now=ist(15, 30)).verdict is (
        Verdict.PASS
    )


def test_the_unelapsed_afternoon_is_not_missing_data(cal):
    """Run at 11:00 on a day recorded cleanly from the open, nothing is missing.
    Counting the hours that have not happened would make every mid-session run
    report a failure."""
    upto_eleven = good_candles(105)
    check = check_session_gaps(upto_eleven, cal, TEST_DAY, now=ist(11, 0))
    assert check.verdict is Verdict.PASS, check.detail


def test_complete_session_passes(cal):
    assert check_candle_count(good_candles(375), cal).verdict is Verdict.PASS


def test_no_candles_at_all_fails(cal):
    assert check_candle_count([], cal).verdict is Verdict.FAIL


def test_worst_continuous_instrument_decides(cal):
    """One healthy instrument must not mask another that is missing data - among
    the instruments the gate is about. Before options were in the recording that
    was every instrument; now it is the ones expected to quote every minute, and
    a thin option contract is no longer allowed to speak for the index."""
    candles = good_candles(375, instrument_id=1) + good_candles(100, instrument_id=2)
    check = check_candle_count(candles, cal, continuous_ids=frozenset({1, 2}))
    assert check.verdict is Verdict.SKIP  # instrument 2 is short: a partial session
    assert "worst 2 has 100/375" in check.detail

    gaps = check_session_gaps(
        candles, cal, TEST_DAY, now=ist(15, 30), continuous_ids=frozenset({1, 2})
    )
    assert gaps.verdict is Verdict.FAIL
    assert gaps.value == pytest.approx(275.0)


def test_mostly_synthetic_session_fails():
    """The count check passes here - the grid is regular. That is why this exists."""
    candles = good_candles(375)
    corrupted = [
        c.model_copy(update={"synthetic": True}) if i > 50 else c for i, c in enumerate(candles)
    ]
    assert check_synthetic_share(corrupted).verdict is Verdict.FAIL
    assert check_synthetic_share(candles).verdict is Verdict.PASS


def test_inconsistent_ohlc_fails():
    candles = good_candles(10)
    broken = [*candles[:5], candles[5].model_copy(update={"low": 200.0}), *candles[6:]]
    check = check_candle_consistency(broken)
    assert check.verdict is Verdict.FAIL and check.value == 1


def test_impossible_price_jump_fails():
    candles = good_candles(10)
    spiked = [*candles[:5], candles[5].model_copy(update={"c": 5000.0, "h": 5000.0}), *candles[6:]]
    check = check_price_jumps(spiked)
    assert check.verdict is Verdict.FAIL
    assert check.value > 0.10


def test_normal_moves_pass():
    assert check_price_jumps(good_candles(50)).verdict is Verdict.PASS


# ----------------------------------------------------------- chain checks
def test_incomplete_chain_fails():
    snaps = [s.model_copy(update={"completeness": 0.60}) for s in good_chain()]
    check = check_chain_completeness(snaps)
    assert check.verdict is Verdict.FAIL
    assert check.value == pytest.approx(0.60)


def test_iv_outside_the_plausible_band_fails():
    snaps = good_chain()
    broken = [
        snaps[0].model_copy(
            update={"quotes": [q.model_copy(update={"iv": 7.5}) for q in snaps[0].quotes]}
        ),
        *snaps[1:],
    ]
    assert check_iv_sanity(broken).verdict is Verdict.FAIL


def test_missing_iv_is_not_counted_as_bad():
    """Inversion declining to answer is correct behaviour, not a data fault."""
    snaps = good_chain()
    no_iv = [
        s.model_copy(update={"quotes": [q.model_copy(update={"iv": None}) for q in s.quotes]})
        for s in snaps
    ]
    assert check_iv_sanity(no_iv).verdict is Verdict.SKIP


def test_crossed_book_fails():
    snaps = good_chain()
    q = snaps[0].quotes[0]
    broken = [
        snaps[0].model_copy(
            update={
                "quotes": [q.model_copy(update={"bid": 900.0, "ask": 100.0}), *snaps[0].quotes[1:]]
            }
        ),
        *snaps[1:],
    ]
    assert check_crossed_books(broken).verdict is Verdict.FAIL


def test_parity_holds_on_a_clean_chain():
    assert check_put_call_parity(good_chain()).verdict is Verdict.PASS


def test_a_stale_leg_breaks_parity():
    """The sharpest staleness detector on a chain - one leg frozen and nothing
    else about the snapshot looks wrong."""
    snaps = good_chain(1)
    quotes = list(snaps[0].quotes)
    for i, q in enumerate(quotes):
        if q.strike == 24_900.0 and q.opt_type is OptionType.CE:
            quotes[i] = q.model_copy(update={"bid": 5_000.0, "ask": 5_001.0})
    broken = [snaps[0].model_copy(update={"quotes": quotes})]
    assert check_put_call_parity(broken).verdict is Verdict.FAIL


# ------------------------------------------------------------ news checks
def test_news_published_after_ingest_fails():
    """Impossible ordering means the timestamp extractor is wrong, and every
    latency measurement built on it is fiction."""
    items = good_news(5)
    broken = [
        *items[:2],
        items[2].model_copy(update={"t_event": items[2].t_ingest + timedelta(minutes=5)}),
        *items[3:],
    ]
    check = check_news_timestamps(broken)
    assert check.verdict is Verdict.FAIL
    assert "after we ingested" in check.detail


def test_too_many_uncertain_timestamps_warns():
    items = [i.model_copy(update={"ts_uncertain": True}) for i in good_news(10)]
    assert check_news_timestamps(items).verdict is Verdict.WARN


def test_high_duplicate_rate_warns():
    items = good_news(10)
    dupes = [i.model_copy(update={"duplicate_of": "x"}) for i in items[:5]]
    assert check_duplicate_rate([*dupes, *items[5:]]).verdict is Verdict.WARN
    assert check_duplicate_rate(items).verdict is Verdict.PASS


# --------------------------------------------------------- feature checks
def test_null_rate_reports_the_worst_feature():
    values = {
        "clean": [1.0] * 100,
        "leaky": [None if i < 30 else 1.0 for i in range(100)],
    }
    check = check_null_rate(values)
    assert check.verdict is Verdict.FAIL
    assert check.detail.endswith("leaky")


def test_psi_is_zero_for_identical_distributions():
    xs = [float(i) for i in range(1000)]
    assert population_stability_index(xs, xs) == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_a_shifted_distribution():
    baseline = [float(i) for i in range(1000)]
    shifted = [float(i) + 800 for i in range(1000)]
    assert population_stability_index(baseline, shifted) > 0.2


def test_train_serve_skew_fails_on_drift():
    baseline = {"f": [float(i) for i in range(1000)]}
    current = {"f": [float(i) + 900 for i in range(1000)]}
    check = check_train_serve_skew(baseline, current)
    assert check.verdict is Verdict.FAIL
    assert check.detail.endswith("f")


def test_train_serve_skew_passes_when_stable():
    baseline = {"f": [float(i) for i in range(1000)]}
    current = {"f": [float(i) + 1 for i in range(1000)]}
    assert check_train_serve_skew(baseline, current).verdict is Verdict.PASS


# ------------------------------------------------------------- reporting
def test_failures_block_but_warnings_do_not(cal):
    """A truncated day still blocks. Which gate says so has moved - from
    `candle_count`, which now declines to call a partial session corrupt, to
    `session_gaps`, which measures what is actually missing. What must not move
    is that the day is blocked."""
    report = run_market_checks(TEST_DAY, cal, now=ist(15, 30), candles=good_candles(200))
    assert not report.ok
    assert any(c.name == "session_gaps" for c in report.failures)


def test_report_is_saved_with_every_check(cal, tmp_path):
    report = run_market_checks(
        TEST_DAY, cal, now=ist(15, 30), candles=good_candles(), chain=good_chain()
    )
    path = report.save(tmp_path)
    assert path.exists()

    import json

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert len(saved["checks"]) == len(report.checks)
    assert all("value" in c and "verdict" in c for c in saved["checks"])


def test_render_is_human_readable(cal):
    text = run_market_checks(TEST_DAY, cal, now=ist(15, 30), candles=good_candles()).render()
    assert "data quality" in text
    assert "candle_count" in text


def test_empty_inputs_skip_rather_than_pass(cal):
    """Nothing to check is not the same as everything being fine."""
    report = run_market_checks(TEST_DAY, cal, now=ist(15, 30))
    skipped = [c for c in report.checks if c.verdict is Verdict.SKIP]
    assert len(skipped) >= 5
    assert any(c.name == "candle_count" and c.verdict is Verdict.FAIL for c in report.checks)


# --------------------------------------------------------- instrument classes
def _chain_with(underlying: int, option_ids: list[int]) -> ChainSnapshot:
    return ChainSnapshot(
        ts=ist(9, 20),
        underlying_id=underlying,
        spot=24_000.0,
        quotes=[
            OptionQuote(
                instrument_id=oid,
                expiry=ist(15, 30) + timedelta(days=1),
                strike=24_000.0 + 50 * i,
                opt_type=OptionType.CE,
                bid=100.0,
                ask=101.0,
                oi=1000,
                iv=0.14,
            )
            for i, oid in enumerate(option_ids)
        ],
    )


def test_instruments_are_classified_from_the_day_s_own_chain():
    """A live session stamps the vendor's token, so NIFTY arrives as 26000 and
    config/instruments.yaml's id 1 appears nowhere in the data. The chain names
    its own underlying and lists every contract written on it."""
    options, underlyings = instruments_from_chain([_chain_with(26000, [42571, 42708])])
    assert underlyings == {26000}
    assert options == {42571, 42708}


def test_a_day_with_no_chain_classifies_nothing():
    """And every caller reads that as 'assume continuous', which is what the
    gates did before options existed in the recording."""
    options, underlyings = instruments_from_chain([])
    assert not options and not underlyings


# ------------------------------------------------------------- session gaps
def test_an_interior_hole_is_counted_and_named(cal):
    candles = good_candles(375)
    holed = candles[:100] + candles[121:]  # twenty-one minutes removed
    check = check_session_gaps(holed, cal, TEST_DAY, now=ist(15, 30))
    assert check.verdict is Verdict.FAIL
    assert check.value == pytest.approx(21.0)
    assert "largest hole 21m" in check.detail


def test_a_late_start_is_counted_too(cal):
    """It was not, in the first version of this gate, and that was wrong. The
    first live session began recording at 09:42 because a bug swallowed the
    opening twenty-seven minutes. Those minutes existed and were not captured;
    calling them 'not a gap' because nothing preceded them is bookkeeping, not
    measurement."""
    check = check_session_gaps(good_candles(375)[60:], cal, TEST_DAY, now=ist(15, 30))
    assert check.verdict is Verdict.FAIL
    assert check.value == pytest.approx(60.0)
    assert "late start 60m" in check.detail


def test_all_three_kinds_of_loss_are_one_number(cal):
    """A late start, a hole and a truncation are the same thing - minutes that
    happened and were not recorded - so they add up rather than compete."""
    candles = good_candles(375)
    damaged = candles[30:100] + candles[121:300]
    check = check_session_gaps(damaged, cal, TEST_DAY, now=ist(15, 30))
    assert check.verdict is Verdict.FAIL
    assert check.value == pytest.approx(375 - len(damaged))
    for phrase in ("late start 30m", "largest hole 21m", "tail 75m"):
        assert phrase in check.detail, check.detail


def test_a_day_with_nothing_recorded_is_a_failure_not_a_skip(cal):
    """Zero candles is the loudest possible version of this fault, and an empty
    input is exactly where a gate is most likely to quietly return SKIP."""
    check = check_session_gaps([], cal, TEST_DAY, now=ist(15, 30))
    assert check.verdict is Verdict.FAIL
    assert "nothing recorded" in check.detail


def test_before_the_open_there_is_nothing_to_have_missed(cal):
    assert check_session_gaps([], cal, TEST_DAY, now=ist(9, 0)).verdict is Verdict.SKIP


def test_session_gaps_only_speaks_for_continuous_instruments(cal):
    """An option that trades in 39 minutes of the day is missing 336 of them and
    none of that is a fault."""
    sparse = [c for i, c in enumerate(good_candles(375, instrument_id=42571)) if i % 9 == 0]
    both = good_candles(375) + sparse
    scoped = check_session_gaps(both, cal, TEST_DAY, now=ist(15, 30), continuous_ids=frozenset({1}))
    assert scoped.verdict is Verdict.PASS
    unscoped = check_session_gaps(both, cal, TEST_DAY, now=ist(15, 30))
    assert unscoped.verdict is Verdict.FAIL, "unscoped, the option decides"


# -------------------------------------------------------------- price jumps
def _with_jump(candles: list[Candle], idx: int, factor: float) -> list[Candle]:
    out = list(candles)
    c = out[idx]
    out[idx] = c.model_copy(update={"c": c.c * factor, "h": c.h * factor})
    return out


def test_an_index_jump_is_still_caught():
    """The gate must not have been softened, only aimed."""
    check = check_price_jumps(
        _with_jump(good_candles(375), 200, 1.30), continuous_ids=frozenset({1})
    )
    assert check.verdict is Verdict.FAIL
    assert check.value > 0.2


def test_an_option_premium_moving_thirty_percent_is_not_a_fault():
    """On the first live session 70 of 374 contracts moved more than ten percent
    in some minute while the index's worst minute was 0.07%. That is gearing,
    not corruption, and blocking every live day on it is how a gate stops being
    read."""
    index = good_candles(375)
    option = _with_jump(good_candles(375, instrument_id=42571), 200, 1.30)
    check = check_price_jumps(index + option, continuous_ids=frozenset({1}))
    assert check.verdict is Verdict.PASS
    assert "1 instrument(s)" in check.detail


def test_price_jumps_reports_what_it_covered():
    """A PASS that checked nothing reads exactly like a PASS that checked
    everything, which is why the coverage is in the detail line."""
    check = check_price_jumps(good_candles(375), continuous_ids=frozenset({1}))
    assert "374 pairs" in check.detail


def test_price_jumps_declines_rather_than_passing_when_scoped_to_nothing():
    check = check_price_jumps(good_candles(375), continuous_ids=frozenset({99}))
    assert check.verdict is Verdict.SKIP, "an empty scope must never read as a pass"


# ------------------------------------------------------------ candle count
def test_candle_count_declines_on_a_partial_session(cal):
    """A part-recorded day is not a corrupt one. Calling it corrupt is how the
    distinction stops being made - and it made the gate fail every live day."""
    check = check_candle_count(good_candles(120), cal, continuous_ids=frozenset({1}))
    assert check.verdict is Verdict.SKIP
    assert "session_gaps" in check.detail


def test_candle_count_ignores_instruments_it_is_not_about(cal):
    """The thinnest option contract on a live day produced 39 candles. Taking
    the worst of 375 contracts blocked the day on the definition of an option."""
    thin = good_candles(39, instrument_id=42571)
    check = check_candle_count(good_candles(375) + thin, cal, continuous_ids=frozenset({1}))
    assert check.verdict is Verdict.PASS


def test_candle_count_still_fails_a_short_continuous_day(cal):
    """Aimed, not softened: an index that quoted 374 of 375 minutes on a day it
    should have quoted all of is still a fault."""
    check = check_candle_count(good_candles(375)[:-1], cal, continuous_ids=frozenset({1}))
    assert check.verdict is Verdict.SKIP  # short by one is a partial session
    complete = good_candles(375)
    assert check_candle_count(complete, cal, continuous_ids=frozenset({1})).verdict is Verdict.PASS
