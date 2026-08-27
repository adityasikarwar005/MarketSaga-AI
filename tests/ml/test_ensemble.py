"""M59 - stacking, calibration, conformal coverage, Brier decomposition."""

from __future__ import annotations

import math
import random

import pytest

from libs.ms_models.ensemble import (
    OOFRow,
    OutOfFoldMatrix,
    SplitConformal,
    StackedEnsemble,
    brier_decomposition,
    brier_skill,
    calibrate,
    coverage_by_group,
    empirical_coverage,
    merge_oof,
    oof_predictions,
)


def oof_rows(n: int, seed: int, *, models: dict[str, float], base_rate: float = 0.5):
    """Base predictions of varying quality against a shared truth.

    ``models`` maps a name to how informative it is: 0 is a coin flip, 1 is
    perfect knowledge of the label.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        y = 1 if rng.random() < base_rate else 0
        preds = {}
        for name, quality in models.items():
            signal = (y - 0.5) * 2 * quality
            preds[name] = min(0.99, max(0.01, 0.5 + signal * 0.4 + rng.gauss(0, 0.12)))
        rows.append(OOFRow(index=i, fold=i * 4 // n, predictions=preds, label=y))
    return rows


# ------------------------------------------------------- out-of-fold discipline
def test_a_row_without_provenance_is_refused():
    """ "These predictions are out-of-sample" has to be checkable months later,
    not remembered. An in-sample base prediction turns the stack into a machine
    for amplifying whichever model memorised hardest."""
    with pytest.raises(ValueError, match="no fold recorded"):
        OutOfFoldMatrix([OOFRow(index=0, fold=-1, predictions={"a": 0.5}, label=1)])


def test_models_must_cover_the_same_rows():
    rows = [
        OOFRow(index=0, fold=0, predictions={"a": 0.5, "b": 0.5}, label=1),
        OOFRow(index=1, fold=0, predictions={"a": 0.5}, label=0),
    ]
    with pytest.raises(ValueError, match="expected"):
        OutOfFoldMatrix(rows)


def test_an_empty_matrix_is_refused():
    with pytest.raises(ValueError, match="no out-of-fold rows"):
        OutOfFoldMatrix([])


def test_oof_predictions_never_score_a_row_its_model_saw():
    """The whole guarantee, checked directly: every fold's predictions must come
    from a fit whose training slice ended before the row."""
    seen: list[tuple[int, int]] = []

    def fit_predict(train_rows, train_labels, test_rows):
        seen.append((len(train_rows), len(test_rows)))
        return [0.5] * len(test_rows)

    rows = list(range(300))
    labels = [i % 2 for i in rows]
    out = oof_predictions(rows, labels, fit_predict, name="m", n_folds=4, min_train=100)

    assert out
    for train_size, _ in seen:
        # Every prediction in this fold is for an index >= train_size.
        for r in out:
            if r.fold == [s[0] for s in seen].index(train_size):
                assert r.index >= train_size


def test_oof_predictions_returns_nothing_when_history_is_too_short():
    assert oof_predictions([1, 2, 3], [0, 1, 0], lambda a, b, c: [], name="m") == []


def test_merge_drops_rows_a_model_never_saw():
    """Filling a missing base prediction with 0.5 says that model was undecided,
    when in fact it was never asked - and the weight learned is a weight on an
    absence."""
    a = [OOFRow(index=i, fold=0, predictions={"a": 0.6}, label=1) for i in range(5)]
    b = [OOFRow(index=i, fold=0, predictions={"b": 0.7}, label=1) for i in range(3)]
    merged = merge_oof(a, b)
    assert len(merged) == 3
    assert all(set(r.predictions) == {"a", "b"} for r in merged)


def test_merge_of_nothing_is_empty():
    assert merge_oof() == []


# ------------------------------------------------------------------- stacking
def test_the_stack_is_rejected_when_it_cannot_beat_its_best_member():
    """A stacker that does not earn its place must not ship. The fallback is the
    best single model, and that is documented behaviour rather than an error."""
    rows = oof_rows(400, 1, models={"good": 0.9, "copy": 0.9})
    stack = StackedEnsemble().fit(OutOfFoldMatrix(rows))
    assert stack.report is not None
    if not stack.report.used:
        assert stack.fallback == stack.report.best_single
        row = rows[0].predictions
        assert stack.predict(row) == row[stack.report.best_single]


def test_the_stack_learns_to_downweight_a_useless_member():
    """One informative model and one coin flip. Whatever the stack decides, it
    must not end up worse than the informative model alone."""
    rows = oof_rows(500, 2, models={"good": 0.9, "noise": 0.0})
    stack = StackedEnsemble().fit(OutOfFoldMatrix(rows))
    assert stack.report is not None
    assert stack.report.stack_log_loss <= stack.report.best_single_log_loss + 0.01
    assert abs(stack.weights["good"]) > abs(stack.weights["noise"])


def test_the_prior_pulls_toward_equal_weighting_not_toward_zero():
    """With little to go on the stack should degrade to the average of its
    members, not to an opinionless constant."""
    rows = oof_rows(150, 3, models={"a": 0.0, "b": 0.0})
    stack = StackedEnsemble(l2=50.0).fit(OutOfFoldMatrix(rows))
    for w in stack.weights.values():
        assert w == pytest.approx(0.5, abs=0.15), f"weights collapsed to {stack.weights}"


def test_stacking_refuses_too_few_rows():
    rows = oof_rows(40, 4, models={"a": 0.5})
    with pytest.raises(ValueError, match="not enough to stack"):
        StackedEnsemble().fit(OutOfFoldMatrix(rows))


def test_predicting_without_every_base_model_is_an_error():
    rows = oof_rows(300, 5, models={"a": 0.7, "b": 0.4})
    stack = StackedEnsemble().fit(OutOfFoldMatrix(rows))
    with pytest.raises(KeyError, match="missing base predictions"):
        stack.predict({"a": 0.6})


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="fit\\(\\) before predict"):
        StackedEnsemble().predict({"a": 0.5})


# ---------------------------------------------------------------- calibration
def test_calibration_is_skipped_on_a_ranking_that_is_noise():
    rng = random.Random(6)
    probs = [rng.random() for _ in range(300)]
    labels = [rng.randint(0, 1) for _ in range(300)]
    cal, report = calibrate(probs, labels)
    assert cal is None
    assert not report.applied
    assert "discrimination" in report.reason


def test_calibration_is_skipped_when_there_are_too_few_rows():
    cal, report = calibrate([0.6] * 20, [1] * 10 + [0] * 10)
    assert cal is None and not report.applied
    assert "rows" in report.reason


def test_calibration_improves_a_genuinely_overconfident_model():
    rng = random.Random(7)
    probs, labels = [], []
    for _ in range(800):
        p_true = rng.random()
        probs.append(min(0.99, max(0.01, 0.5 + (p_true - 0.5) * 2.2)))
        labels.append(1 if rng.random() < p_true else 0)
    cal, report = calibrate(probs, labels)
    assert cal is not None and report.applied
    assert report.ece_after < report.ece_before


def test_g5_calibration_needs_both_slope_and_skill():
    """A model can be perfectly calibrated and useless - always predicting the
    base rate has a fine reliability slope and no resolution at all."""
    rng = random.Random(8)
    labels = [1 if rng.random() < 0.5 else 0 for _ in range(400)]
    flat = [0.5] * 400
    _, report = calibrate(flat, labels)
    assert not report.passes_g5_calibration
    assert report.brier_skill <= 0.02


# ------------------------------------------------------------------ conformal
def test_conformal_coverage_is_within_tolerance():
    rng = random.Random(9)
    predicted = [rng.gauss(0, 1) for _ in range(600)]
    actual = [p + rng.gauss(0, 0.5) for p in predicted]

    conformal = SplitConformal(alpha=0.1).calibrate(predicted[:300], actual[:300])
    intervals = [conformal.interval(p) for p in predicted[300:]]
    coverage = empirical_coverage(intervals, actual[300:])
    assert 0.85 <= coverage <= 0.95, f"coverage {coverage:.1%}"


def test_the_n_plus_one_correction_is_present():
    """Without it the intervals are quietly a few percent narrow, and the
    shortfall looks like bad luck rather than a missing term."""
    residuals = [float(i) for i in range(1, 21)]
    conformal = SplitConformal(alpha=0.1).calibrate([0.0] * 20, residuals)
    # ceil((20 + 1) * 0.9) = 19  ->  the 19th smallest residual, which is 19.0.
    # Without the +1 it would be ceil(20 * 0.9) = 18  ->  18.0.
    assert conformal.half_width == 19.0


def test_conformal_refuses_when_the_quantile_would_be_the_maximum():
    """With too few residuals a "99% interval" is just the largest error seen,
    which is a statement about the sample rather than about coverage."""
    with pytest.raises(ValueError, match="not enough"):
        SplitConformal(alpha=0.1).calibrate([0.0] * 10, [1.0] * 10)


def test_an_uncalibrated_conformal_says_so():
    interval = SplitConformal().interval(1.5)
    assert not interval.calibrated
    assert interval.width == 0.0


def test_alpha_must_be_a_probability():
    with pytest.raises(ValueError, match="alpha"):
        SplitConformal(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        SplitConformal(alpha=1.0)


def test_coverage_by_group_exposes_where_the_marginal_guarantee_hides_a_failure():
    """90% overall can be 98% on quiet days and 60% on volatile ones - and the
    volatile ones are exactly when the interval was supposed to help."""
    rng = random.Random(10)
    predicted = [0.0] * 400
    actual = [rng.gauss(0, 0.2) for _ in range(200)] + [rng.gauss(0, 3.0) for _ in range(200)]
    groups = ["quiet"] * 200 + ["volatile"] * 200

    conformal = SplitConformal(alpha=0.1)
    conformal.half_width = 1.0
    intervals = [conformal.interval(p) for p in predicted]

    by_group = coverage_by_group(intervals, actual, groups)
    assert by_group["quiet"][0] > by_group["volatile"][0]
    overall = empirical_coverage(intervals, actual)
    assert by_group["volatile"][0] < overall


# ------------------------------------------------------- Brier decomposition
def test_the_decomposition_adds_up():
    rng = random.Random(11)
    probs, labels = [], []
    for _ in range(1000):
        p = rng.random()
        probs.append(p)
        labels.append(1 if rng.random() < p else 0)
    d = brier_decomposition(probs, labels)
    assert abs(d.residual) < 1e-9, f"decomposition does not reconcile: {d.residual}"


def test_a_calibrated_but_useless_model_has_no_resolution():
    """The distinction the decomposition exists to make: reliability is what
    calibration fixes, resolution is what only a better model can raise."""
    rng = random.Random(12)
    labels = [1 if rng.random() < 0.4 else 0 for _ in range(1000)]
    always_base = [0.4] * 1000
    d = brier_decomposition(always_base, labels)
    assert d.resolution < 0.001
    assert d.reliability < 0.01


def test_a_sharp_model_has_resolution():
    rng = random.Random(13)
    probs, labels = [], []
    for _ in range(1000):
        y = rng.randint(0, 1)
        probs.append(0.85 if y else 0.15)
        labels.append(y)
    d = brier_decomposition(probs, labels)
    assert d.resolution > 0.1


def test_decomposition_of_nothing_is_not_a_crash():
    d = brier_decomposition([], [])
    assert math.isnan(d.brier)


def test_brier_skill_is_against_climatology_not_a_coin_flip():
    """A base rate of 0.9 makes 0.5 a terrible reference. Scoring against it
    would hand every model a large free improvement."""
    labels = [1] * 900 + [0] * 100
    assert brier_skill([0.9] * 1000, labels) == pytest.approx(0.0, abs=0.01)
    assert brier_skill([0.5] * 1000, labels) < 0


def test_binning_cost_is_reported_not_hidden():
    """The gap between the raw and binned Brier is what the decomposition cannot
    see. Reporting it is the difference between a residual that means something
    and one that gets waved off as rounding."""
    rng = random.Random(14)
    probs = [rng.random() for _ in range(600)]
    labels = [1 if rng.random() < p else 0 for p in probs]

    coarse = brier_decomposition(probs, labels, bins=3)
    fine = brier_decomposition(probs, labels, bins=25)
    assert abs(coarse.residual) < 1e-9 and abs(fine.residual) < 1e-9
    assert abs(coarse.discretisation) > abs(fine.discretisation)


def test_the_stack_does_not_diverge_on_confident_inputs():
    """Log-odds reach +/-4.6 once base probabilities are clamped. A fixed step
    size on those diverges - the first version reported a stacked log loss of
    14.9 against 0.72 for its best single member."""
    rows = oof_rows(400, 15, models={"a": 0.95, "b": 0.9, "c": 0.2})
    stack = StackedEnsemble().fit(OutOfFoldMatrix(rows))
    assert stack.report is not None
    assert stack.report.stack_log_loss < 1.0, f"diverged to {stack.report.stack_log_loss}"
    assert all(abs(w) < 20 for w in stack.weights.values()), stack.weights


def test_a_strong_penalty_shrinks_toward_the_prior_without_oscillating():
    """With gradient descent, lr * l2 above one made this oscillate instead of
    shrink. Newton has no step size to get wrong."""
    rows = oof_rows(300, 16, models={"a": 0.6, "b": 0.6})
    weak = StackedEnsemble(l2=0.01).fit(OutOfFoldMatrix(rows))
    strong = StackedEnsemble(l2=1e5).fit(OutOfFoldMatrix(rows))
    for name in weak.names:
        assert abs(strong.weights[name] - 0.5) < abs(weak.weights[name] - 0.5) + 1e-9
        assert abs(strong.weights[name] - 0.5) < 0.01
