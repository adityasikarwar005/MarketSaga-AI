"""M53 direction model - schema discipline, calibration, and the null case."""

from __future__ import annotations

import math
import random

import pytest

from libs.ms_models.direction import (
    MISSING,
    DirectionModel,
    IsotonicCalibrator,
    auc,
    brier_skill_score,
    expected_calibration_error,
    has_discrimination,
    reliability_curve,
    reliability_slope,
)
from libs.ms_models.trees import log_loss

NAMES = [f"f{i}" for i in range(12)]


def sample(n: int, seed: int, *, signal: bool = True, missing: float = 0.0):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        raw: dict[str, float | None] = {k: rng.gauss(0, 1) for k in NAMES}
        z = 1.1 * raw["f0"] - 0.8 * raw["f1"] + 0.6 * raw["f0"] * raw["f2"] if signal else 0.0
        y = 1 if rng.random() < 1 / (1 + math.exp(-z)) else -1
        if rng.random() < 0.08:
            y = 0  # timeout, as the triple-barrier labeller emits
        for k in NAMES:
            if rng.random() < missing:
                raw[k] = None
        rows.append(raw)
        labels.append(y)
    return rows, labels


def directional(probs, labels):
    kept = [(p, 1 if y > 0 else 0) for p, y in zip(probs, labels, strict=True) if y != 0]
    return [p for p, _ in kept], [y for _, y in kept]


# --------------------------------------------------------------- the null case
def test_a_model_with_no_signal_holds_no_opinion():
    """The whole point. Coin-flip labels must produce something indistinguishable
    from the base rate - not a calibrated version of an accident."""
    rows, labels = sample(1600, 3, signal=False)
    test_rows, test_labels = sample(700, 4, signal=False)
    model = DirectionModel().fit(rows, labels)

    probs, ys = directional([model.predict(r).p_up for r in test_rows], test_labels)
    assert max(abs(p - 0.5) for p in probs) < 0.10, "opined on noise"

    base = log_loss([sum(ys) / len(ys)] * len(ys), ys)
    skill = (base - log_loss(probs, ys)) / base
    assert skill < 0.02, f"found {skill:.1%} skill in coin flips"


def test_calibration_is_skipped_when_there_is_nothing_to_calibrate():
    """Isotonic corrects the levels of a ranking and trusts the ranking itself.
    Fitted to a model with no ranking ability it manufactures an opinion: a
    booster whose raw output spanned [0.475, 0.496] came out spanning
    [0.046, 0.659]."""
    rows, labels = sample(1600, 7, signal=False)
    model = DirectionModel().fit(rows, labels)
    assert model.no_discrimination
    assert model.calibrator is None
    assert "no discrimination" in " ".join(model.predict(rows[0]).reasons)


def test_discrimination_gate_is_a_significance_level():
    """One standard error passes on noise about a third of the time. The bar has
    to be a significance level, not a gesture at one."""
    rng = random.Random(5)
    passes = 0
    trials = 60
    for _ in range(trials):
        probs = [rng.random() for _ in range(300)]
        labels = [rng.randint(0, 1) for _ in range(300)]
        passes += has_discrimination(probs, labels)
    assert passes / trials <= 0.12, f"passed on {passes}/{trials} noise samples"


def test_discrimination_gate_lets_a_real_ranking_through():
    rng = random.Random(6)
    probs, labels = [], []
    for _ in range(300):
        y = rng.randint(0, 1)
        probs.append(rng.gauss(0.6 if y else 0.4, 0.15))
        labels.append(y)
    assert has_discrimination(probs, labels)


# ------------------------------------------------------------- the signal case
def test_it_learns_a_real_signal_and_stays_calibrated():
    rows, labels = sample(1600, 1)
    test_rows, test_labels = sample(700, 2)
    model = DirectionModel().fit(rows, labels)
    assert model.report is not None and model.report.calibrated

    probs, ys = directional([model.predict(r).p_up for r in test_rows], test_labels)
    assert log_loss(probs, ys) < 0.65, "no better than a coin flip"
    area = auc(probs, ys)
    assert area is not None and area > 0.65
    assert brier_skill_score(probs, ys) > 0.05

    slope = reliability_slope(reliability_curve(probs, ys))
    assert slope is not None and 0.7 < slope < 1.5, f"reliability slope {slope}"


def test_calibration_improves_the_calibration_error():
    rows, labels = sample(1600, 11)
    model = DirectionModel().fit(rows, labels)
    assert model.report is not None
    assert model.report.ece_after < model.report.ece_before


def test_the_edge_is_capped():
    """A direction call on an index over a short horizon that claims 90% is
    describing a bug, not a conviction."""
    rows, labels = sample(1600, 12)
    model = DirectionModel(max_edge=0.2).fit(rows, labels)
    for r in rows[:200]:
        assert abs(model.predict(r).p_up - 0.5) <= 0.2 + 1e-9


# -------------------------------------------------------------- schema and gaps
def test_column_order_is_fixed_not_inherited_from_a_dict():
    """A model that iterates a dict is one ordering change away from scoring the
    wrong columns, with the training run still green."""
    rows, labels = sample(1600, 21)
    model = DirectionModel().fit(rows, labels)
    assert model.names == sorted(model.names)

    row = rows[0]
    shuffled = dict(reversed(list(row.items())))
    assert model.predict(row).p_up == model.predict(shuffled).p_up


def test_a_missing_feature_is_not_a_zero():
    """Filling an absent feature with 0.0 claims it was average. It is not the
    same input and must not produce the same answer."""
    rows, labels = sample(1600, 22, missing=0.15)
    model = DirectionModel().fit(rows, labels)

    present: dict[str, float | None] = dict.fromkeys(NAMES, 0.0)
    absent: dict[str, float | None] = dict(present)
    absent["f0"] = None
    assert model._vector(present)[model.names.index("f0")] == 0.0
    assert model._vector(absent)[model.names.index("f0")] == MISSING


def test_confidence_falls_as_inputs_go_missing():
    rows, labels = sample(1600, 23, missing=0.15)
    model = DirectionModel().fit(rows, labels)
    full: dict[str, float | None] = dict.fromkeys(NAMES, 0.5)
    half: dict[str, float | None] = dict(full)
    for k in NAMES[:6]:
        half[k] = None
    assert model.predict(full).confidence == pytest.approx(1.0)
    assert model.predict(half).confidence == pytest.approx(0.5)


def test_timeouts_are_dropped_from_fitting_not_coded_as_a_class():
    """A barrier that was never touched is not a direction, and coding it as
    either class teaches the model something that did not happen."""
    rows, labels = sample(1600, 24)
    model = DirectionModel().fit(rows, labels)
    assert model.report is not None
    assert model.report.rows == sum(1 for y in labels if y != 0)


def test_refuses_to_fit_on_too_few_directional_rows():
    rows, labels = sample(300, 25)
    labels = [0] * 250 + labels[250:]
    with pytest.raises(ValueError, match="directional rows"):
        DirectionModel().fit(rows, labels)


# ------------------------------------------------------------------- isotonic
def test_isotonic_cannot_claim_certainty_from_a_handful_of_rows():
    """Plain PAV on 400 rows of noise produced nine blocks, six valued at exactly
    0.0 or 1.0 - and improved log loss while doing it."""
    rng = random.Random(0)
    raw = [0.5 + rng.gauss(0, 0.02) for _ in range(400)]
    labels = [rng.randint(0, 1) for _ in range(400)]
    cal = IsotonicCalibrator().fit(raw, labels)

    out = [cal.apply(p) for p in raw]
    assert min(out) > 0.05 and max(out) < 0.95, f"range [{min(out):.3f}, {max(out):.3f}]"
    assert all(0.0 < y < 1.0 for y in cal.ys)


def test_isotonic_is_monotone():
    rng = random.Random(1)
    raw = [rng.random() for _ in range(500)]
    labels = [1 if rng.random() < p else 0 for p in raw]
    cal = IsotonicCalibrator().fit(raw, labels)
    grid = [i / 50 for i in range(51)]
    mapped = [cal.apply(x) for x in grid]
    assert mapped == sorted(mapped)


def test_isotonic_still_corrects_real_overconfidence():
    rng = random.Random(2)
    raw, labels = [], []
    for _ in range(1500):
        p_true = rng.random()
        raw.append(min(0.999, max(0.001, 0.5 + (p_true - 0.5) * 2.2)))
        labels.append(1 if rng.random() < p_true else 0)
    cal = IsotonicCalibrator().fit(raw, labels)
    before = expected_calibration_error(raw, labels)
    after = expected_calibration_error([cal.apply(p) for p in raw], labels)
    assert after < before, f"{before:.4f} -> {after:.4f}"
