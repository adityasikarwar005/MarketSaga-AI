"""M55 volatility model - estimators, QLIKE, the log-retransformation bias,
and what the model does when there is nothing to add to HAR."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_core.schemas import Candle
from libs.ms_models.volatility import (
    HARModel,
    VolatilityModel,
    build_har_rows,
    close_to_close,
    garman_klass,
    mse,
    parkinson,
    qlike,
    realised_variance,
    smearing_factor,
    walk_forward_evaluate,
)


def bar(o: float, h: float, low: float, c: float, minute: int = 0) -> Candle:
    return Candle(
        instrument_id=1,
        ts=datetime(2026, 3, 2, 4, 0, tzinfo=UTC) + timedelta(minutes=minute),
        tf=1,
        o=o,
        h=h,
        low=low,
        c=c,
        v=100,
    )


def walk(n: int, seed: int, sigma: float = 0.001) -> list[Candle]:
    """A random walk sampled into bars whose ranges reflect ``sigma``."""
    rng = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        o = px
        ticks = [px * math.exp(rng.gauss(0, sigma)) for _ in range(12)]
        px = ticks[-1]
        out.append(bar(o, max(o, *ticks), min(o, *ticks), px, i))
    return out


# ------------------------------------------------------------------ estimators
def test_a_swinging_bar_that_closes_flat_is_not_calm():
    """The reason range estimators exist. Close-to-close reads this as zero risk."""
    swung = [bar(100.0, 102.0, 98.0, 100.0)]
    assert close_to_close(swung) == 0.0
    assert parkinson(swung) > 0.0
    assert garman_klass(swung) > 0.0


def test_range_estimators_are_less_noisy_than_squared_returns():
    """Same underlying volatility, many samples: the better estimator should have
    the smaller spread. This is the whole justification for using them."""
    import statistics as st

    park, ctc = [], []
    for seed in range(60):
        bars = walk(40, seed)
        park.append(parkinson(bars))
        ctc.append(close_to_close(bars))
    cv_park = st.pstdev(park) / st.fmean(park)
    cv_ctc = st.pstdev(ctc) / st.fmean(ctc)
    assert cv_park < cv_ctc, f"parkinson cv {cv_park:.3f} vs close-to-close {cv_ctc:.3f}"


def test_estimators_scale_with_volatility():
    quiet = parkinson(walk(200, 1, sigma=0.0005))
    loud = parkinson(walk(200, 1, sigma=0.002))
    assert loud > quiet * 4, f"quiet {quiet:.3e} loud {loud:.3e}"


def test_estimators_are_zero_not_negative_on_degenerate_input():
    flat = [bar(100.0, 100.0, 100.0, 100.0)]
    assert parkinson(flat) == 0.0
    assert garman_klass(flat) >= 0.0
    assert parkinson([]) == 0.0
    assert garman_klass([]) == 0.0


def test_unknown_estimator_is_refused():
    with pytest.raises(ValueError, match="unknown estimator"):
        realised_variance(walk(10, 0), estimator="magic")


# ---------------------------------------------------------------------- QLIKE
def test_qlike_is_minimised_at_the_truth():
    truth = [0.0004] * 200
    assert qlike(truth, truth) == pytest.approx(0.0, abs=1e-12)
    assert qlike([0.0002] * 200, truth) > 0
    assert qlike([0.0008] * 200, truth) > 0


def test_qlike_punishes_under_forecasting_harder():
    """Half the true variance must cost more than twice it. Under-forecasting
    volatility is what leaves a position larger than intended."""
    truth = [0.0004] * 100
    too_small = qlike([0.0002] * 100, truth)
    too_large = qlike([0.0008] * 100, truth)
    assert too_small > too_large


def test_qlike_is_scale_free_where_mse_is_not():
    """MSE on variance is dominated by the loudest observations, so a model can
    look good by getting a few crisis days right and nothing else."""
    calm = [1e-6] * 100
    crisis = [1e-3] * 100
    truth = calm + crisis
    # Right on the crisis, 50% off everywhere else.
    predicted = [1.5e-6] * 100 + [1e-3] * 100
    # Right everywhere else, 50% off on the crisis.
    other = [1e-6] * 100 + [1.5e-3] * 100

    assert mse(predicted, truth) < mse(other, truth)
    assert qlike(predicted, truth) == pytest.approx(qlike(other, truth), rel=0.01)


# -------------------------------------------------- the retransformation trap
def test_exponentiating_a_log_fit_under_predicts_without_smearing():
    """exp(mean of logs) is the median, not the mean. For a right-skewed target
    that is a permanent downward bias which no other diagnostic surfaces."""
    rng = random.Random(0)
    residuals = [rng.gauss(0, 0.4) for _ in range(5000)]
    factor = smearing_factor(residuals)
    assert factor > 1.05, f"smearing factor {factor:.4f} - expected roughly 8%"
    # For normal residuals it should be close to exp(sigma^2 / 2).
    assert factor == pytest.approx(math.exp(0.4**2 / 2), rel=0.06)


def test_smearing_of_nothing_is_one():
    assert smearing_factor([]) == 1.0
    assert smearing_factor([0.0, 0.0]) == pytest.approx(1.0)


def test_har_predictions_are_not_systematically_low():
    """The bias this catches is invisible in any relative metric - the model can
    rank perfectly and still be wrong by a constant factor every single time."""
    rng = random.Random(3)
    series = []
    level = 4e-4
    for _ in range(600):
        level = 0.85 * level + 0.15 * 4e-4
        series.append(level * math.exp(rng.gauss(0, 0.5)))

    rows = build_har_rows(series)
    model = HARModel().fit(rows[:400])
    test = rows[400:]
    predicted = model.predict_all(test)
    actual = [r.target for r in test]

    ratio = sum(actual) / sum(predicted)
    assert 0.85 < ratio < 1.15, f"predictions are off by a factor of {ratio:.3f}"


# ------------------------------------------------------------------ HAR rows
def test_har_rows_never_look_forward():
    """A row at i uses periods up to i and targets what came after. If the
    lookbacks could reach past i the whole model would score beautifully and
    mean nothing."""
    series = list(range(100))
    rows = build_har_rows([float(s) for s in series], horizon=1, weekly=5, monthly=22)
    assert rows
    for offset, row in enumerate(rows):
        i = 21 + offset
        assert row.daily == float(i)
        assert row.weekly == pytest.approx(sum(range(i - 4, i + 1)) / 5)
        assert row.monthly == pytest.approx(sum(range(i - 21, i + 1)) / 22)
        assert row.target == float(i + 1)


def test_har_rows_stop_before_the_target_runs_out():
    rows = build_har_rows([1.0] * 30, horizon=5, weekly=5, monthly=22)
    assert all(r.target > 0 for r in rows)
    assert len(rows) == 30 - 22 - 5 + 1


def test_har_rows_carry_extra_features_aligned_to_the_row():
    series = [1.0 + i for i in range(60)]
    extra = [{"hour": float(i % 6)} for i in range(60)]
    rows = build_har_rows(series, extra=extra)
    for offset, row in enumerate(rows):
        assert row.extra["hour"] == float((21 + offset) % 6)


# ------------------------------------------------------------------ the model
def clustered_series(n: int, seed: int, *, driver: bool = False):
    """Volatility that clusters, optionally with a feature that helps predict it."""
    rng = random.Random(seed)
    series, extra = [], []
    level = 4e-4
    for _ in range(n):
        shock = rng.gauss(0, 1)
        level = math.exp(0.92 * math.log(level) + 0.08 * math.log(4e-4) + 0.15 * shock)
        series.append(level * math.exp(rng.gauss(0, 0.3)))
        # A feature that leads the next period's level when driver is on.
        extra.append({"lead": shock if driver else rng.gauss(0, 1)})
    return series, extra


def test_the_booster_is_discarded_when_it_adds_nothing():
    """The failure mode this design exists to prevent: an augmented model that
    is worse than the baseline it augments. If the extra features carry nothing,
    the model must *be* HAR."""
    series, extra = clustered_series(500, 11, driver=False)
    rows = build_har_rows(series, extra=extra)
    model = VolatilityModel().fit(rows)

    assert model.report is not None
    assert (
        model.report.model_qlike <= model.report.har_qlike + 1e-12
    ), "the augmented model scored worse than HAR and was kept anyway"


def test_a_discarded_booster_leaves_har_predictions_untouched():
    series, extra = clustered_series(500, 12, driver=False)
    rows = build_har_rows(series, extra=extra)
    model = VolatilityModel().fit(rows)
    if model.report and not model.report.augmented:
        assert model.booster is None
        # And HAR is refitted on every row. The held-out slice answered one
        # keep-or-drop question; withholding a quarter of the data from a
        # three-parameter regression afterwards is a handicap with nothing to
        # show for it - it cost 0.8% against plain HAR on a series with no
        # driver at all, purely from the smaller fit.
        har = HARModel().fit(rows)
        assert model.predict(rows[-1]) == pytest.approx(har.predict(rows[-1]), rel=1e-9)


def test_a_model_with_no_driver_is_never_worse_than_har():
    """The guarantee the residual formulation is supposed to buy. Measured
    across a walk-forward on a series whose extra feature is pure noise."""
    series, extra = clustered_series(700, 31, driver=False)
    rows = build_har_rows(series, extra=extra)
    result = walk_forward_evaluate(rows, n_folds=4)
    assert result is not None
    assert (
        result.improvement >= -1e-9
    ), f"augmentation cost {result.improvement:.1%} against its own baseline"


def test_har_alone_beats_a_constant_forecast():
    """If HAR cannot beat 'tomorrow looks like the long-run average' then the
    volatility clustering this whole module assumes is not in the data, and the
    test bench is wrong rather than the model."""
    series, _ = clustered_series(600, 13)
    rows = build_har_rows(series)
    split = int(len(rows) * 0.7)
    model = HARModel().fit(rows[:split])
    test = rows[split:]
    actual = [r.target for r in test]

    constant = sum(r.target for r in rows[:split]) / split
    assert qlike(model.predict_all(test), actual) < qlike([constant] * len(test), actual)


def test_walk_forward_reports_both_metrics():
    series, extra = clustered_series(600, 14)
    rows = build_har_rows(series, extra=extra)
    result = walk_forward_evaluate(rows, n_folds=3)
    assert result is not None
    assert result.n_test > 0
    text = result.render()
    assert "QLIKE" in text and "MSE" in text


def test_walk_forward_refuses_when_there_is_not_enough_history():
    series, _ = clustered_series(130, 15)
    rows = build_har_rows(series)
    assert walk_forward_evaluate(rows, n_folds=4) is None


def test_refuses_to_fit_on_too_few_rows():
    series, _ = clustered_series(150, 16)
    rows = build_har_rows(series)[:50]
    with pytest.raises(ValueError, match="not enough"):
        VolatilityModel().fit(rows)


def test_volatility_is_the_square_root_of_variance():
    series, extra = clustered_series(500, 17)
    rows = build_har_rows(series, extra=extra)
    model = VolatilityModel().fit(rows)
    row = rows[-1]
    assert model.predict_volatility(row) == pytest.approx(math.sqrt(model.predict(row)))


def test_the_booster_is_kept_when_a_real_driver_exists():
    """The other half of the guarantee. A model that never accepts an
    augmentation is not disciplined, it is broken - and this test is what tells
    the two apart."""
    rng = random.Random(41)
    series, extra = [], []
    level = 4e-4
    pending = 0.0
    for _ in range(900):
        level = math.exp(
            0.90 * math.log(level) + 0.10 * math.log(4e-4) + 0.12 * rng.gauss(0, 1) + 0.8 * pending
        )
        series.append(level * math.exp(rng.gauss(0, 0.25)))
        pending = rng.gauss(0, 1)
        extra.append({"lead": pending})

    rows = build_har_rows(series, extra=extra)
    result = walk_forward_evaluate(rows, n_folds=4)
    assert result is not None
    assert result.augmented_folds >= 2, "never accepted a genuine driver"
    assert result.improvement > 0.05, f"only {result.improvement:.1%} from a real driver"


def test_keeping_the_booster_needs_a_margin_not_just_a_win():
    """A booster that merely wins on the validation slice has shown nothing: the
    slice is finite. Beating HAR by less than the noise in the comparison is a
    coin toss, and it cost 1.1% on test rows before this was enforced."""
    from libs.ms_models.volatility import beats_with_margin

    rng = random.Random(7)
    realised = [4e-4 * math.exp(rng.gauss(0, 0.5)) for _ in range(200)]
    baseline = [4e-4] * 200

    # A challenger that is noisily different and only barely better on average -
    # which is exactly what a booster fitted to nothing looks like on a finite
    # validation slice.
    challenger = [4e-4 * math.exp(rng.gauss(0.002, 0.35)) for _ in range(200)]
    assert not beats_with_margin(baseline, challenger, realised)

    # Genuinely better: predicts the realisation itself.
    assert beats_with_margin(baseline, list(realised), realised)


def test_margin_test_refuses_a_worse_challenger():
    from libs.ms_models.volatility import beats_with_margin

    realised = [4e-4] * 100
    assert not beats_with_margin([4e-4] * 100, [8e-4] * 100, realised)


def test_margin_test_needs_enough_observations():
    from libs.ms_models.volatility import beats_with_margin

    assert not beats_with_margin([4e-4] * 4, [4e-4] * 4, [4e-4] * 4)
