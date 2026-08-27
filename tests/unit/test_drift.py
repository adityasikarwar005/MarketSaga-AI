"""MA3 - do the drift monitors fire, and more importantly do they stay quiet?

A drift monitor has two failure modes and only one of them is visible. It can
cry wolf, which an operator notices within a week and starts routing around. Or
it can sit at zero forever while the model rots underneath it, which nobody
notices at all - the dashboard is green, the metric is a number, and the number
is meaningless.

The second is the one these tests are mostly about.
"""

from __future__ import annotations

import math
import random

import pytest

from libs.ms_ops.drift import (
    BRIER_DEGRADATION_BREACH,
    KL_BREACH,
    MIN_SAMPLES,
    PSI_BREACH,
    PSI_WATCH,
    DriftGate,
    DriftReading,
    Severity,
    build_reference,
    check_feature_psi,
    check_prediction_kl,
    check_rolling_brier,
    kl_divergence,
    psi,
    rolling_brier,
)


def normal(rng: random.Random, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
    return [rng.gauss(mu, sigma) for _ in range(n)]


@pytest.fixture
def rng() -> random.Random:
    return random.Random(20260826)


# --------------------------------------------------------------------------- #
# The null tests - what does it say when nothing changed?
# --------------------------------------------------------------------------- #
def test_psi_is_near_zero_when_nothing_changed(rng: random.Random):
    """The same distribution twice. A monitor that reports drift here reports it
    every morning, and an operator who sees it every morning stops reading it."""
    reference = build_reference("gap", normal(rng, 5000))
    fresh = normal(rng, 2000)

    value = psi(reference, fresh)
    assert value < PSI_WATCH, f"PSI {value:.4f} on an unchanged distribution"


def test_psi_stays_quiet_across_many_unchanged_draws(rng: random.Random):
    """Once is luck. Forty draws from the reference distribution, and none of
    them may breach - a monitor that fires on one morning in forty fires roughly
    every other week for no reason."""
    reference = build_reference("gap", normal(rng, 5000))

    values = [psi(reference, normal(rng, 500)) for _ in range(40)]
    breaches = [v for v in values if v >= PSI_BREACH]

    assert not breaches, f"{len(breaches)}/40 false breaches, worst {max(values):.4f}"
    assert max(values) < PSI_WATCH, f"worst unchanged draw reached {max(values):.4f}"


def test_kl_is_near_zero_on_the_same_predictions(rng: random.Random):
    reference = [rng.betavariate(2, 2) for _ in range(4000)]
    fresh = [rng.betavariate(2, 2) for _ in range(1000)]
    assert kl_divergence(reference, fresh) < KL_BREACH


def test_brier_does_not_breach_on_an_unchanged_model(rng: random.Random):
    """Same model, same world, a new week of outcomes. The rolling Brier moves
    around by sampling noise and must not read that as decay."""
    probs = [rng.uniform(0.3, 0.7) for _ in range(1000)]
    outcomes = [1 if rng.random() < p else 0 for p in probs]
    reference_brier = rolling_brier(probs, outcomes)

    fresh_p = [rng.uniform(0.3, 0.7) for _ in range(500)]
    fresh_y = [1 if rng.random() < p else 0 for p in fresh_p]

    reading = check_rolling_brier(reference_brier, fresh_p, fresh_y)
    assert reading.severity is Severity.OK, reading.detail


# --------------------------------------------------------------------------- #
# The bins come from the reference. This is the whole thing.
# --------------------------------------------------------------------------- #
def test_the_reference_bins_are_fixed_and_that_is_what_detects_the_shift(rng: random.Random):
    """The bug this exists to prevent.

    Recomputing bin edges on the new data makes PSI report ~0 forever, because
    quantile bins fitted to a distribution always contain a tenth of it each -
    whatever that distribution is. The metric follows the drift it was supposed
    to detect, and every dashboard stays green.

    Both numbers below come from the same shifted data. The first uses the
    reference's edges and finds the shift. The second rebuilds the edges and
    finds nothing. The gap between them is the bug.
    """
    reference = build_reference("gap", normal(rng, 5000))
    shifted = normal(rng, 2000, mu=0.9)

    against_fixed_bins = psi(reference, shifted)
    refitted = build_reference("gap", shifted)
    against_refitted_bins = psi(refitted, shifted)

    assert against_fixed_bins >= PSI_BREACH, (
        f"a 0.9-sigma shift scored {against_fixed_bins:.4f} against fixed bins"
    )
    assert against_refitted_bins < PSI_WATCH, (
        "the refitted-bin comparison was supposed to be blind to this shift; if it "
        "is not, this test no longer demonstrates anything"
    )


def test_values_entirely_outside_the_reference_range_breach(rng: random.Random):
    """Every observation past the last reference edge. This is a market the
    model has never been fitted on, and it must not land in a bin quietly."""
    reference = build_reference("iv", normal(rng, 3000, mu=0.0, sigma=1.0))
    elsewhere = normal(rng, 1000, mu=8.0, sigma=0.5)

    reading = check_feature_psi(reference, elsewhere)
    assert reading.severity is Severity.BREACH
    assert reading.blocks


def test_an_empty_bin_does_not_produce_infinity(rng: random.Random):
    """A bin the new data never reaches makes the naive ratio divide by zero,
    and an infinite metric is indistinguishable from a broken one - it renders
    as `inf`, breaches everything, and tells nobody what happened."""
    reference = build_reference("x", normal(rng, 3000))
    narrow = [0.01] * 1000  # every observation in one bin

    value = psi(reference, narrow)
    assert not math.isnan(value), "PSI was NaN"
    assert not math.isinf(value)
    assert value > PSI_BREACH


# --------------------------------------------------------------------------- #
# ...and it does fire when something did change
# --------------------------------------------------------------------------- #
def test_psi_grows_with_the_size_of_the_shift(rng: random.Random):
    """Monotone in the shift, or the number is not a measurement of anything."""
    reference = build_reference("gap", normal(rng, 6000))
    values = [psi(reference, normal(rng, 2000, mu=m)) for m in (0.0, 0.3, 0.6, 1.2)]

    assert values == sorted(values), f"PSI not monotone in the shift: {values}"
    assert values[0] < PSI_WATCH < values[-1]


def test_a_variance_change_with_the_same_mean_is_still_drift(rng: random.Random):
    """Volatility regimes move the spread, not the centre. A monitor watching
    only the mean would call this stable, and it is the more dangerous change -
    the model's inputs are the same size on average and nothing like the same."""
    reference = build_reference("ret", normal(rng, 5000, sigma=1.0))
    wider = normal(rng, 2000, mu=0.0, sigma=2.5)

    assert psi(reference, wider) >= PSI_BREACH


def test_kl_catches_a_prediction_distribution_that_collapsed(rng: random.Random):
    """The model went from spread-out probabilities to always saying 0.5. The
    inputs can look perfectly stable while this happens - it is what a feature
    going flat does to the output."""
    reference = [rng.betavariate(2, 2) for _ in range(4000)]
    collapsed = [rng.gauss(0.5, 0.01) for _ in range(1000)]

    reading = check_prediction_kl(reference, collapsed)
    assert reading.severity is Severity.BREACH
    assert "calibration" in reading.detail


def test_brier_breaches_when_the_model_actually_got_worse(rng: random.Random):
    probs = [rng.uniform(0.2, 0.8) for _ in range(800)]
    good = [1 if rng.random() < p else 0 for p in probs]
    reference_brier = rolling_brier(probs, good)

    # The same probabilities, outcomes now unrelated to them.
    unrelated = [rng.randint(0, 1) for _ in probs]
    reading = check_rolling_brier(reference_brier, probs, unrelated)

    assert reading.severity is Severity.BREACH
    assert "%" in reading.detail


def test_the_brier_threshold_is_a_relative_degradation():
    """Stated as a fraction, so it means the same thing on an easy horizon and a
    hard one. An absolute threshold would fire constantly on the hard one and
    never on the easy one."""
    probs = [0.5] * 400
    outcomes = [1] * 200 + [0] * 200  # brier exactly 0.25
    ref = 0.25 / (1.0 + BRIER_DEGRADATION_BREACH + 0.01)

    assert check_rolling_brier(ref, probs, outcomes).severity is Severity.BREACH
    assert check_rolling_brier(0.25, probs, outcomes).severity is Severity.OK


# --------------------------------------------------------------------------- #
# Not enough data is not the same as no drift
# --------------------------------------------------------------------------- #
def test_too_few_samples_reports_unknown_not_ok(rng: random.Random):
    """The distinction the dashboard has to preserve. "I measured and found
    nothing" and "I could not measure" are different states, and collapsing them
    into a green tick is how a monitor goes quiet without anyone noticing."""
    reference = build_reference("gap", normal(rng, 3000))
    reading = check_feature_psi(reference, normal(rng, 20, mu=5.0))

    assert reading.severity is Severity.UNKNOWN
    assert not reading.blocks
    assert "not the same as no drift" in reading.detail


def test_a_reference_cannot_be_built_from_too_little_data(rng: random.Random):
    with pytest.raises(ValueError, match="not enough"):
        build_reference("thin", normal(rng, MIN_SAMPLES - 1))


def test_unknown_does_not_block_execution(rng: random.Random):
    """A monitor that cannot measure would otherwise halt trading every quiet
    morning, and an operator who is halted every quiet morning learns to
    override the gate - which costs more than the monitor ever saved."""
    gate = DriftGate().evaluate(
        [
            DriftReading("psi[gap]", float("nan"), Severity.UNKNOWN, 12),
            DriftReading("prediction_kl", 0.01, Severity.OK, 900),
        ]
    )
    assert gate.allows_auto_execution
    assert len(gate.unknown) == 1
    assert "not evidence of its absence" in gate.render()


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_one_breach_blocks_auto_execution():
    gate = DriftGate().evaluate(
        [
            DriftReading("psi[gap]", 0.02, Severity.OK, 900),
            DriftReading("psi[iv]", 0.41, Severity.BREACH, 900),
            DriftReading("rolling_brier", 0.19, Severity.OK, 900),
        ]
    )
    assert not gate.allows_auto_execution
    assert [r.monitor for r in gate.breaches] == ["psi[iv]"]
    assert "AUTO-EXECUTION BLOCKED" in gate.render()


def test_watch_does_not_block():
    """The middle band exists so that a shift can be visible without being an
    outage. If WATCH blocked, there would be no middle band."""
    gate = DriftGate().evaluate([DriftReading("psi[gap]", 0.15, Severity.WATCH, 900)])
    assert gate.allows_auto_execution
    assert "WATCH" in gate.render()


def test_the_breach_clears_when_the_metric_recovers(rng: random.Random):
    """And only then. A gate that can be dismissed is a notification."""
    reference = build_reference("gap", normal(rng, 5000))
    gate = DriftGate()

    gate.evaluate([check_feature_psi(reference, normal(rng, 1000, mu=1.5))])
    assert not gate.allows_auto_execution

    gate.evaluate([check_feature_psi(reference, normal(rng, 1000))])
    assert gate.allows_auto_execution, "the metric recovered and the gate stayed shut"


def test_an_empty_gate_allows_execution():
    """Nothing configured is not a breach. It is also not a monitor, which the
    render makes visible rather than showing a green tick."""
    gate = DriftGate()
    assert gate.allows_auto_execution
    assert "auto-execution allowed" in gate.render()


def test_psi_breach_blocks_auto_execution_end_to_end(rng: random.Random):
    """The whole chain, because each half can be right while the join is not.

    A monitor that measures correctly and a gate that blocks correctly still
    leave a system that trades through drift, if nothing carries the reading
    from one to the other. That join is the part with no natural test, so this
    is it: real feature data, a real shift, and the decision the execution path
    actually consults.
    """
    reference = build_reference("gap", normal(rng, 5000))
    drifted = normal(rng, 1000, mu=1.4)

    gate = DriftGate().evaluate([check_feature_psi(reference, drifted)])

    assert not gate.allows_auto_execution
    assert gate.breaches[0].monitor == "psi[gap]"
    assert gate.breaches[0].value >= PSI_BREACH
