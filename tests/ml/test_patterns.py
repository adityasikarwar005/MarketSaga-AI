"""Candlestick patterns, and whether detecting them is worth anything.

The detector half is ordinary: precise predicates over numbers, checked against
hand-built candles whose shape is known. The half that matters is the other one.

Testing nineteen patterns is nineteen tests. At an uncorrected p < 0.05 the
chance of at least one false positive is 62%, so a module that reports "shooting
star, p = 0.048" without saying how many patterns it looked at is reporting the
search rather than the market. The tests below fix the correction in place and
check that the pipeline finds nothing when there is nothing to find.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from libs.ms_models.patterns import (
    ALPHA,
    CATALOGUE,
    MIN_OCCURRENCES,
    Candle,
    Direction,
    Evaluation,
    PatternError,
    detect,
    evaluate,
    forward_returns,
    null_check,
    sidak_alpha,
)

D0 = date(2020, 1, 1)


def c(o: float, h: float, low: float, cl: float, i: int = 0) -> Candle:
    return Candle(day=D0 + timedelta(days=i), o=o, h=h, low=low, c=cl)


def falling(n: int = 6, start: float = 110.0) -> list[Candle]:
    """A clear downtrend, so reversal patterns have the context they need."""
    out = []
    for i in range(n):
        p = start - i * 2.0
        out.append(c(p, p + 0.4, p - 2.4, p - 2.0, i))
    return out


def rising(n: int = 6, start: float = 100.0) -> list[Candle]:
    out = []
    for i in range(n):
        p = start + i * 2.0
        out.append(c(p, p + 2.4, p - 0.4, p + 2.0, i))
    return out


def fires(name: str, series: list[Candle]) -> bool:
    return len(series) - 1 in detect(series)[name]


# --------------------------------------------------------------------------- #
# the shapes
# --------------------------------------------------------------------------- #
def test_a_doji_is_a_tiny_body_in_a_real_range():
    assert fires("doji", [c(100, 102, 98, 100.05)])
    assert not fires("doji", [c(100, 102, 98, 101.5)])


def test_a_zero_range_candle_is_not_a_doji():
    """A halted or untraded bar has body 0 and range 0. Calling it a doji reads
    a halt as a signal, and the ratio is 0/0."""
    assert not fires("doji", [c(100, 100, 100, 100)])


def test_a_hammer_needs_a_long_lower_shadow_and_a_downtrend():
    body = c(100, 100.3, 94, 100.2, 6)
    assert fires("hammer", [*falling(), body])


def test_the_same_shape_after_a_rise_is_a_hanging_man_not_a_hammer():
    """A hammer in an uptrend is a hanging man, and the two are opposite
    signals. Merging them averages a bullish and a bearish claim into noise."""
    body = c(110, 110.3, 104, 110.2, 6)
    up = [*rising(), body]
    assert fires("hanging_man", up)
    assert not fires("hammer", up)


def test_the_same_holds_for_the_inverted_pair():
    star = c(100, 106, 99.8, 100.2, 6)
    assert fires("inverted_hammer", [*falling(), star])
    assert fires("shooting_star", [*rising(start=100.0), c(110, 116, 109.8, 110.2, 6)])


def test_bullish_engulfing_needs_the_second_body_to_cover_the_first():
    prev = c(100, 100.5, 97, 97.5, 5)
    now = c(97, 101.5, 96.8, 101, 6)
    assert fires("bullish_engulfing", [*falling(5), prev, now])

    # A body that only partly covers is not an engulfing.
    narrow = c(98, 100, 97.5, 99, 6)
    assert not fires("bullish_engulfing", [*falling(5), prev, narrow])


def test_bearish_engulfing_is_the_mirror():
    # rising(5) closes at 110, so the setup candle has to continue from there.
    # An earlier version of this put it at 100, which broke the uptrend the
    # pattern requires - the fixture failed the definition, not the detector.
    prev = c(110, 113, 109.5, 112.5, 5)
    now = c(113, 113.2, 109, 109.5, 6)
    assert fires("bearish_engulfing", [*rising(5), prev, now])


def test_a_harami_is_contained_by_the_previous_body():
    prev = c(105, 105.5, 99, 99.5, 5)
    inside = c(101, 102, 100.5, 102, 6)
    assert fires("bullish_harami", [*falling(5), prev, inside])


def test_morning_star_needs_a_small_middle_candle():
    long_down = c(110, 110.2, 103.8, 104, 4)
    small = c(103.5, 104.2, 103, 103.6, 5)
    long_up = c(104, 108.5, 103.9, 108, 6)
    assert fires("morning_star", [*falling(4), long_down, small, long_up])

    big_middle = c(103.5, 110, 100, 109, 5)
    assert not fires("morning_star", [*falling(4), long_down, big_middle, long_up])


def test_three_white_soldiers_must_each_open_and_close_higher():
    ladder = [
        c(100, 104.2, 99.8, 104, 0),
        c(101, 106.2, 100.8, 106, 1),
        c(103, 108.2, 102.8, 108, 2),
    ]
    assert fires("three_white_soldiers", ladder)

    ladder[2] = c(103, 105.2, 102.8, 105, 2)  # closes below the previous close
    assert not fires("three_white_soldiers", ladder)


def test_abandoned_baby_needs_both_gaps():
    """TradingView's own example. Two gaps make it rare by construction, and a
    version that skips the gap check fires constantly."""
    a = c(110, 110.2, 104, 104.5, 4)
    baby = c(102, 102.4, 101.6, 102.05, 5)  # gaps below, and is a doji
    up = c(104, 108, 103.5, 107.5, 6)  # gaps back above
    assert fires("abandoned_baby_bull", [*falling(4), a, baby, up])

    no_gap = c(104, 104.4, 103.6, 104.05, 5)
    assert not fires("abandoned_baby_bull", [*falling(4), a, no_gap, up])


def test_every_catalogue_entry_declares_its_lookback():
    """A predicate reading index i-2 with a lookback of 0 reads the end of the
    list instead, and the pattern fires on the wrong bars."""
    for p in CATALOGUE:
        assert p.lookback >= 0
        assert p.direction in Direction
        assert p.predicate is not None


def test_detection_never_reads_before_the_start():
    tiny = [c(100, 101, 99, 100.5, i) for i in range(3)]
    found = detect(tiny)
    assert all(all(i >= 0 for i in ix) for ix in found.values())


# --------------------------------------------------------------------------- #
# the correction, which is the point
# --------------------------------------------------------------------------- #
def test_sidak_tightens_with_the_number_of_tests():
    assert sidak_alpha(0.05, 1) == pytest.approx(0.05)
    assert sidak_alpha(0.05, 19) == pytest.approx(0.0027, abs=1e-4)
    assert sidak_alpha(0.05, 19) < sidak_alpha(0.05, 5) < sidak_alpha(0.05, 1)


def test_nineteen_uncorrected_tests_would_be_wrong_most_of_the_time():
    """The number that makes the correction necessary rather than tidy."""
    assert 1 - (1 - 0.05) ** 19 > 0.60


def test_the_evaluation_corrects_for_every_pattern_it_looked_at():
    """Not for the ones that fired. Looking at nineteen and reporting the best
    is nineteen tests whether or not eighteen came back empty."""
    candles = random_walk(600)
    ev = evaluate(candles, horizon=1)
    assert ev.tests == len(CATALOGUE)
    assert ev.threshold == pytest.approx(sidak_alpha(ALPHA, len(CATALOGUE)))
    assert "19 patterns tested" in ev.render()
    assert "Sidak" in ev.render()


def test_the_report_states_the_count_it_corrected_for():
    """A corrected p-value whose n is not stated is an uncorrected p-value."""
    ev = evaluate(random_walk(600), horizon=1)
    text = ev.render()
    assert str(len(CATALOGUE)) in text
    assert "uncorrected" in text


# --------------------------------------------------------------------------- #
# the null test
# --------------------------------------------------------------------------- #
def random_walk(n: int, seed: int = 3, sigma: float = 0.01) -> list[Candle]:
    """Candles with no structure whatever. Every pattern here is an accident."""
    rng = random.Random(seed)
    px = 100.0
    out = []
    for i in range(n):
        o = px
        cl = o * (1 + rng.gauss(0, sigma))
        hi = max(o, cl) * (1 + abs(rng.gauss(0, sigma / 2)))
        lo = min(o, cl) * (1 - abs(rng.gauss(0, sigma / 2)))
        out.append(c(o, hi, lo, cl, i))
        px = cl
    return out


def test_no_pattern_is_significant_on_a_random_walk():
    """The null test. Patterns still fire - shapes appear in noise - but none of
    them may survive the correction, or the pipeline manufactures findings."""
    ev = evaluate(random_walk(1500), horizon=1)
    assert sum(r.n for r in ev.results) > 100, "nothing fired; the test proves nothing"
    assert ev.significant == [], [r.name for r in ev.significant]


def test_the_null_check_finds_almost_nothing_on_shuffled_outcomes():
    """Real shapes, randomised outcomes. Every genuine relationship is destroyed
    and every artefact of the method survives, so what comes back is what this
    pipeline reports when there is nothing to report."""
    mean_fp, worst = null_check(random_walk(1200), horizon=1, trials=10, seed=1)
    assert mean_fp < 1.0, f"{mean_fp:.2f} false positives per trial"
    assert worst <= 2


def test_a_planted_relationship_is_found():
    """The other direction. A null test that always passes proves the detector
    is blind, not that the market is efficient."""
    candles = random_walk(1500, seed=11)
    fwd = forward_returns(candles, 1)
    hits = set(detect(candles)["doji"])
    # Add a large, unmistakable drift after every doji.
    planted = [
        (r + 0.05 if i in hits else r) if not math.isnan(r) else r for i, r in enumerate(fwd)
    ]

    ev = evaluate(candles, horizon=1, returns=planted)
    doji = next(r for r in ev.results if r.name == "doji")
    assert doji.significant(ev.threshold), f"p={doji.p_value:.6f} on a planted 5% drift"


# --------------------------------------------------------------------------- #
# the edge measure
# --------------------------------------------------------------------------- #
def test_edge_is_measured_against_the_unconditional_drift():
    """An index that drifts up makes every bullish pattern look right and every
    bearish one look wrong, and the drift is available without detecting
    anything."""
    from libs.ms_models.patterns import PatternResult

    bull = PatternResult("x", Direction.BULLISH, 50, 0.004, 0.004, 0.0, 1.0, 0.5)
    assert bull.edge == pytest.approx(0.0), "a pattern that matches the drift has no edge"

    better = PatternResult("x", Direction.BULLISH, 50, 0.006, 0.004, 0.0, 1.0, 0.5)
    assert better.edge == pytest.approx(0.002)


def test_a_bearish_pattern_scores_positively_when_price_falls():
    from libs.ms_models.patterns import PatternResult

    bear = PatternResult("x", Direction.BEARISH, 50, -0.003, 0.001, 0.0, 1.0, 0.5)
    assert bear.edge > 0


def test_a_thin_pattern_is_never_significant_however_small_its_p():
    from libs.ms_models.patterns import PatternResult

    thin = PatternResult("rare", Direction.BULLISH, MIN_OCCURRENCES - 1, 0.05, 0.0, 9.0, 1e-9, 1.0)
    assert not thin.significant(0.05), "a pattern seen twice cannot be evidence"


def test_evaluating_too_little_data_is_refused():
    with pytest.raises(PatternError, match="not enough to evaluate"):
        evaluate([c(100, 101, 99, 100, i) for i in range(10)], horizon=1)


def test_forward_returns_run_off_the_end_as_nan_not_zero():
    """Padding with zero would add a fake flat outcome to whichever pattern
    happened to fire last."""
    fwd = forward_returns([c(100, 101, 99, 100, i) for i in range(5)], 2)
    assert math.isnan(fwd[-1]) and math.isnan(fwd[-2])
    assert not math.isnan(fwd[0])


def test_an_empty_evaluation_renders_without_claiming_anything():
    ev = Evaluation(tests=19)
    assert "0 of 19 significant" in ev.render()
