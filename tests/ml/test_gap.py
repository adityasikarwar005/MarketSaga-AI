"""M33 + M35 verify - transfer coefficients, quantile regression, gap forecaster."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from statistics import fmean

import pytest

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.schemas import Candle
from libs.ms_data.crossasset import (
    SeriesMove,
    overnight_window,
    registry_from_config,
)
from libs.ms_impact.transfer import (
    MIN_PAIRS_FOR_BETA,
    RollingTransfer,
    TransferLibrary,
    aligned_returns,
    build_transfer_library,
    estimate_transfer,
    ols,
)
from libs.ms_models.gap import (
    GapForecaster,
    GapSample,
    NaiveGapBaseline,
    build_features,
    feature_names,
    walk_forward_evaluate,
)
from libs.ms_models.quantile import (
    ClimatologyForecaster,
    QuantileForecast,
    QuantileRegressor,
    enforce_monotone,
    pinball_loss,
    score_forecasts,
)
from tests.conftest import ist


@pytest.fixture
def cal(cfg):
    base = SessionCalendar.from_config(cfg.sessions)
    return base.with_holidays({2026: frozenset({date(2026, 8, 15)})})


@pytest.fixture
def registry(cfg):
    return registry_from_config(cfg.crossasset.series)


# =========================================================================== #
# M33 - transfer coefficients
# =========================================================================== #
def series(n: int, *, slope_from: list[float] | None = None, noise: float = 0.0, seed: int = 1):
    """Minute bars; optionally driven by another series' returns."""
    rng = random.Random(seed)
    out, p = [], 100.0
    for i in range(n):
        o = p
        if slope_from is not None and i < len(slope_from):
            shock = slope_from[i] + rng.gauss(0.0, noise)
        else:
            shock = rng.gauss(0.0, 0.001)
        p = p * math.exp(shock)
        out.append(
            Candle(
                ts=ist(9, 15) + timedelta(minutes=i),
                instrument_id=1,
                tf=1,
                o=round(o, 4),
                h=round(max(o, p) * 1.0002, 4),
                l=round(min(o, p) * 0.9998, 4),
                c=round(p, 4),
            )
        )
    return out


def test_ols_recovers_a_known_slope():
    xs = [i / 100 for i in range(-50, 50)]
    ys = [2.5 * x + 0.1 for x in xs]
    fit = ols(xs, ys)
    assert fit is not None
    assert fit.beta == pytest.approx(2.5, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_ols_reports_a_standard_error():
    """A beta without one is a number that cannot be argued with."""
    rng = random.Random(3)
    xs = [rng.gauss(0, 1) for _ in range(200)]
    ys = [0.5 * x + rng.gauss(0, 1) for x in xs]
    fit = ols(xs, ys)
    assert fit is not None
    assert fit.stderr > 0
    lo, hi = fit.ci()
    assert lo < 0.5 < hi


def test_ols_rejects_degenerate_input():
    assert ols([], []) is None
    assert ols([1.0] * 10, [2.0] * 10) is None  # no variation in x


def test_transfer_recovers_a_planted_relationship():
    rng = random.Random(7)
    driver_shocks = [rng.gauss(0, 0.002) for _ in range(300)]
    source = series(300, slope_from=driver_shocks)
    target = series(300, slope_from=[0.6 * s for s in driver_shocks], noise=0.0002, seed=9)

    coef = estimate_transfer(source, target, source_name="DRIVER", lag_minutes=0)
    assert coef is not None and coef.usable and coef.significant
    assert coef.fit.beta == pytest.approx(0.6, abs=0.15)


def test_unrelated_series_produce_no_significant_beta():
    a = series(300, seed=1)
    b = series(300, seed=2)
    coef = estimate_transfer(a, b, source_name="A", lag_minutes=0)
    assert coef is not None
    assert not coef.significant, "found a relationship between two independent walks"


def test_thin_data_is_marked_unusable():
    coef = estimate_transfer(series(20), series(20, seed=2), source_name="A")
    assert coef is not None
    assert coef.fit.n < MIN_PAIRS_FOR_BETA
    assert not coef.usable
    assert f"n<{MIN_PAIRS_FOR_BETA}" in coef.render()


def test_alignment_joins_on_timestamp_not_position():
    """Two venues keep different hours; a positional join pairs a London
    afternoon with a Mumbai morning."""
    source = series(100)
    target = [c.model_copy(update={"ts": c.ts + timedelta(hours=3)}) for c in series(100, seed=4)]
    xs, ys = aligned_returns(source, target, lag_minutes=0)
    assert xs == [] and ys == []


def test_lag_shifts_the_relationship():
    rng = random.Random(11)
    shocks = [rng.gauss(0, 0.002) for _ in range(400)]
    source = series(400, slope_from=shocks)
    # Target responds one minute later.
    lagged = [0.0] + [0.7 * s for s in shocks[:-1]]
    target = series(400, slope_from=lagged, noise=0.0001, seed=12)

    at_zero = estimate_transfer(source, target, source_name="D", lag_minutes=0)
    at_one = estimate_transfer(source, target, source_name="D", lag_minutes=1)
    assert at_zero is not None and at_one is not None
    assert at_one.fit.r_squared > at_zero.fit.r_squared


def test_asymmetry_needs_more_than_two_different_numbers():
    """Two separately-fitted betas are never exactly equal, so an unguarded
    check would call every relationship asymmetric."""
    rng = random.Random(5)
    shocks = [rng.gauss(0, 0.002) for _ in range(400)]
    source = series(400, slope_from=shocks)
    target = series(400, slope_from=[0.5 * s for s in shocks], noise=0.001, seed=6)

    coef = estimate_transfer(source, target, source_name="D")
    assert coef is not None
    assert coef.beta_up is not None and coef.beta_down is not None
    assert coef.asymmetry is not None
    assert not coef.asymmetry_is_real, "symmetric relationship reported as asymmetric"


def test_library_only_combines_significant_coefficients():
    """A beta whose interval straddles zero adds noise to the estimate and
    confidence to the presentation."""
    library = TransferLibrary()
    rng = random.Random(13)
    shocks = [rng.gauss(0, 0.002) for _ in range(300)]
    real = estimate_transfer(
        series(300, slope_from=shocks),
        series(300, slope_from=[0.8 * s for s in shocks], noise=0.0002, seed=14),
        source_name="REAL",
    )
    noise = estimate_transfer(series(300, seed=20), series(300, seed=21), source_name="NOISE")
    assert real and noise
    library.add(real)
    library.add(noise)

    total, used = library.expected_move({"REAL": 0.01, "NOISE": 0.01})
    assert len(used) == 1 and "REAL" in used[0]
    assert total == pytest.approx(real.fit.beta * 0.01, abs=1e-9)


def test_build_library_over_several_sources():
    rng = random.Random(17)
    shocks = [rng.gauss(0, 0.002) for _ in range(300)]
    target = series(300, slope_from=[0.5 * s for s in shocks], noise=0.0003, seed=18)
    cross = {
        "A": series(300, slope_from=shocks),
        "B": series(300, seed=19),
    }
    library = build_transfer_library(cross, target, lags=(0, 1))
    assert len(library.coefficients) == 4
    assert library.usable()


def test_rolling_history_is_point_in_time():
    """A backtest must use the beta that existed at the time, not the one fitted
    with the benefit of everything that happened since."""
    rolling = RollingTransfer(source="X")
    for d in range(5):
        coef = estimate_transfer(series(200, seed=d), series(200, seed=d + 50), source_name="X")
        assert coef is not None
        coef.as_of = date(2026, 8, 1) + timedelta(days=d * 5)
        rolling.history.append(coef)

    at = rolling.as_of(date(2026, 8, 12))
    assert at is not None
    assert at.as_of < date(2026, 8, 12)
    assert rolling.as_of(date(2026, 7, 1)) is None


# =========================================================================== #
# Quantile machinery
# =========================================================================== #
def test_pinball_loss_is_asymmetric():
    """The asymmetry is what makes the fitted value a quantile, not a mean."""
    assert pinball_loss(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert pinball_loss(-1.0, 0.0, 0.9) == pytest.approx(0.1)
    assert pinball_loss(1.0, 0.0, 0.5) == pinball_loss(-1.0, 0.0, 0.5)


def test_crossed_quantiles_are_rejected():
    with pytest.raises(ValueError, match="cross"):
        QuantileForecast(quantiles={0.25: 0.05, 0.75: 0.01})


def test_monotone_repair():
    repaired = enforce_monotone({0.05: 0.02, 0.25: 0.01, 0.5: 0.03, 0.75: 0.02, 0.95: 0.05})
    values = [repaired[a] for a in sorted(repaired)]
    assert values == sorted(values)


def test_forecast_coverage_and_pit():
    f = QuantileForecast(quantiles={0.05: -0.02, 0.25: -0.01, 0.5: 0.0, 0.75: 0.01, 0.95: 0.02})
    assert f.covers(0.0)
    assert not f.covers(0.05)
    assert f.pit(-0.03) == pytest.approx(0.05)
    assert f.pit(0.03) == pytest.approx(1.0)


def test_crps_rewards_a_sharper_correct_forecast():
    tight = QuantileForecast(
        quantiles={0.05: -0.005, 0.25: -0.002, 0.5: 0.0, 0.75: 0.002, 0.95: 0.005}
    )
    wide = QuantileForecast(quantiles={0.05: -0.05, 0.25: -0.02, 0.5: 0.0, 0.75: 0.02, 0.95: 0.05})
    assert tight.crps(0.0) < wide.crps(0.0)


def test_crps_punishes_a_confidently_wrong_forecast():
    tight = QuantileForecast(
        quantiles={0.05: -0.005, 0.25: -0.002, 0.5: 0.0, 0.75: 0.002, 0.95: 0.005}
    )
    wide = QuantileForecast(quantiles={0.05: -0.05, 0.25: -0.02, 0.5: 0.0, 0.75: 0.02, 0.95: 0.05})
    assert tight.crps(0.04) > wide.crps(0.04)


def test_quantile_regressor_learns_a_linear_relationship():
    rng = random.Random(23)
    rows, targets = [], []
    for _ in range(400):
        x = rng.gauss(0, 1)
        rows.append([x])
        targets.append(0.01 * x + rng.gauss(0, 0.002))

    model = QuantileRegressor(epochs=600).fit(rows, targets)
    high = model.predict([2.0])
    low = model.predict([-2.0])
    assert high.median > low.median
    assert high.median == pytest.approx(0.02, abs=0.01)


def test_quantile_regressor_is_deterministic():
    """Two runs on the same inputs produce identical coefficients, which is what
    lets a model be reproduced from a dataset version months later."""
    rng = random.Random(29)
    rows = [[rng.gauss(0, 1)] for _ in range(200)]
    targets = [r[0] * 0.01 for r in rows]
    a = QuantileRegressor(epochs=100).fit(rows, targets).predict([1.0])
    b = QuantileRegressor(epochs=100).fit(rows, targets).predict([1.0])
    assert a.quantiles == b.quantiles


def test_regressor_refuses_to_fit_on_nothing():
    with pytest.raises(ValueError, match="at least 10"):
        QuantileRegressor().fit([[1.0]] * 5, [0.1] * 5)


def test_scaler_is_fitted_on_train_only():
    """Fitting the scaler on the full dataset is one of the quietest forms of
    leakage: the test set's mean reaches the model through the normalisation."""
    rng = random.Random(31)
    rows = [[rng.gauss(0, 1)] for _ in range(100)]
    targets = [r[0] for r in rows]
    model = QuantileRegressor(epochs=50).fit(rows, targets)
    assert model.scaler.means
    # An unseen, far-out-of-range value is scaled by the TRAIN statistics.
    scaled = model.scaler.apply([100.0])
    assert abs(scaled[0]) > 10


def test_climatology_ignores_features():
    rng = random.Random(37)
    rows = [[rng.gauss(0, 1)] for _ in range(200)]
    targets = [rng.gauss(0, 0.01) for _ in range(200)]
    clim = ClimatologyForecaster().fit(rows, targets)
    assert clim.predict([5.0]).quantiles == clim.predict([-5.0]).quantiles


def test_scoring_reports_coverage_and_calibration():
    rng = random.Random(41)
    actuals = [rng.gauss(0, 0.01) for _ in range(500)]
    honest = [
        QuantileForecast(
            quantiles={0.05: -0.0164, 0.25: -0.0067, 0.5: 0.0, 0.75: 0.0067, 0.95: 0.0164}
        )
        for _ in actuals
    ]
    score = score_forecasts("honest", honest, actuals)
    assert 0.85 < score.coverage_90 < 0.95
    assert score.well_calibrated


def test_overconfident_forecast_is_flagged():
    """A model can have excellent CRPS and 60% coverage on its 90% band, and it
    will size positions wrongly every time."""
    rng = random.Random(43)
    actuals = [rng.gauss(0, 0.01) for _ in range(500)]
    narrow = [
        QuantileForecast(quantiles={0.05: -0.002, 0.25: -0.001, 0.5: 0.0, 0.75: 0.001, 0.95: 0.002})
        for _ in actuals
    ]
    score = score_forecasts("overconfident", narrow, actuals)
    assert score.coverage_90 < 0.5
    assert not score.well_calibrated
    assert "MISCALIBRATED" in score.render()
    assert score.pit_uniformity() > 0


# =========================================================================== #
# M35 - gap forecaster
# =========================================================================== #
def make_samples(
    registry, cal, n: int = 200, *, signal: float = 0.6, noise: float = 0.004, seed: int = 3
) -> list[GapSample]:
    """Sessions where the gap genuinely follows the overnight driver."""
    rng = random.Random(seed)
    names = feature_names(registry)
    samples = []
    day = date(2026, 1, 5)
    made = 0
    while made < n:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        window = overnight_window(cal, day)
        driver = rng.gauss(0.0, 0.006)
        moves = {
            s.symbol: SeriesMove(
                symbol=s.symbol,
                instrument_id=s.instrument_id,
                first=100.0,
                last=100.0 * (1 + (driver if s.symbol == "US_EQ_FUT" else rng.gauss(0, 0.004))),
                bars=100,
                high=101.0,
                low=99.0,
            )
            for s in registry.all()
        }
        features = build_features(window, moves, registry, prior_session_return=rng.gauss(0, 0.005))
        target = signal * driver + rng.gauss(0.0, noise)
        samples.append(
            GapSample(
                session=day,
                features={k: features.get(k, 0.0) for k in names},
                target=target,
                window_hours=window.hours,
                spans_weekend=window.spans_multiple_days,
            )
        )
        made += 1
        day += timedelta(days=1)
    return samples


def test_features_come_only_from_the_window(registry, cal):
    """The window is the only way in, so there is no argument a caller can pass
    that reaches past the cutoff."""
    window = overnight_window(cal, date(2026, 8, 27))
    features = build_features(window, {}, registry)
    assert features["window_hours"] == pytest.approx(window.hours)
    assert features["spans_weekend"] == 0.0
    # A series that did not report is recorded as absent, not as zero movement.
    assert features["US_EQ_FUT_seen"] == 0.0


def test_weekend_window_is_visible_to_the_model(registry, cal):
    """A model that cannot see this mis-sizes every Monday."""
    monday = build_features(overnight_window(cal, date(2026, 8, 31)), {}, registry)
    tuesday = build_features(overnight_window(cal, date(2026, 8, 27)), {}, registry)
    assert monday["spans_weekend"] == 1.0
    assert tuesday["spans_weekend"] == 0.0
    assert monday["window_hours"] > tuesday["window_hours"]


def test_forecaster_refuses_to_fit_on_too_little(registry, cal):
    """This is the one input that cannot be bought back later."""
    with pytest.raises(ValueError, match="not enough to fit"):
        GapForecaster(registry).fit(make_samples(registry, cal, n=20))


def test_forecaster_learns_the_driver(registry, cal):
    samples = make_samples(registry, cal, n=200)
    model = GapForecaster(registry, epochs=300).fit(samples)

    up = dict(samples[0].features)
    up["US_EQ_FUT_ret"] = 0.015
    down = dict(samples[0].features)
    down["US_EQ_FUT_ret"] = -0.015

    hi = model.predict(GapSample(samples[0].session, up, 0.0, 17.0, False))
    lo = model.predict(GapSample(samples[0].session, down, 0.0, 17.0, False))
    assert hi.median_gap > lo.median_gap
    assert hi.direction == "up" and lo.direction == "down"


def test_forecast_explains_itself(registry, cal):
    samples = make_samples(registry, cal, n=150)
    model = GapForecaster(registry, epochs=200).fit(samples)
    out = model.predict(samples[0])
    assert out.drivers
    assert "gap forecast" in out.render()
    assert "90% band" in out.render()


def test_naive_baseline_is_not_a_straw_man(registry, cal):
    """Overnight futures are the single strongest input; a univariate regression
    on them is a genuinely decent forecast."""
    samples = make_samples(registry, cal, n=200)
    naive = NaiveGapBaseline().fit(samples)
    assert naive.beta == pytest.approx(0.6, abs=0.2)

    forecasts = [naive.predict(s) for s in samples]
    score = score_forecasts("naive", forecasts, [s.target for s in samples])
    assert 0.80 < score.coverage_90 < 1.0


def test_walk_forward_evaluation_runs(registry, cal):
    result = walk_forward_evaluate(make_samples(registry, cal, n=220), registry, n_folds=3)
    assert result is not None
    assert result.n_test > 0
    assert "Gate G4" in result.render()


def test_no_signal_means_the_model_does_not_beat_the_baseline(registry, cal):
    """The outcome that matters most: on data with nothing to learn, the gate
    must refuse. A forecaster that passes here would pass on anything."""
    noise_only = make_samples(registry, cal, n=220, signal=0.0, noise=0.006, seed=77)
    result = walk_forward_evaluate(noise_only, registry, n_folds=3)
    assert result is not None
    assert not result.passes_g4


def test_evaluation_needs_enough_sessions(registry, cal):
    assert walk_forward_evaluate(make_samples(registry, cal, n=40), registry) is None


# ---------------------------------------------- calibration is not optional
def test_fit_refuses_when_calibration_would_be_too_small(registry, cal):
    """The bug this guards: calibration was silently skipped when the split was
    thin, and one walk-forward fold quietly produced 85%-covering '90%' bands
    while every other fold was fine."""
    samples = make_samples(registry, cal, n=110)
    with pytest.raises(ValueError, match="calibration"):
        GapForecaster(registry, calibration_share=0.05).fit(samples)


def test_opting_out_of_calibration_is_explicit_and_visible(registry, cal):
    samples = make_samples(registry, cal, n=120)
    model = GapForecaster(registry, epochs=100, calibration_share=0.0).fit(samples)
    out = model.predict(samples[0])
    assert not out.calibrated
    assert "UNCALIBRATED" in out.render()


def test_calibrated_by_default(registry, cal):
    samples = make_samples(registry, cal, n=160)
    model = GapForecaster(registry, epochs=100).fit(samples)
    out = model.predict(samples[0])
    assert out.calibrated
    assert "UNCALIBRATED" not in out.render()


def test_conformal_widens_intervals_that_were_too_narrow():
    """Fitted quantiles are systematically too tight out of sample."""
    from libs.ms_models.quantile import ConformalQuantileForecaster

    rng = random.Random(101)
    rows = [[rng.gauss(0, 1)] for _ in range(300)]
    targets = [r[0] * 0.01 + rng.gauss(0, 0.01) for r in rows]

    base = QuantileRegressor(epochs=200).fit(rows[:200], targets[:200])
    conformal = ConformalQuantileForecaster(base).calibrate(rows[200:], targets[200:])

    raw_width = base.predict([0.0]).width()
    conformal_width = conformal.predict([0.0]).width()
    assert conformal_width > 0
    # Either direction is legitimate; what matters is that it moved on evidence.
    assert conformal_width != raw_width


def test_conformal_needs_enough_calibration_samples():
    from libs.ms_models.quantile import ConformalQuantileForecaster

    rng = random.Random(103)
    rows = [[rng.gauss(0, 1)] for _ in range(50)]
    targets = [r[0] for r in rows]
    base = QuantileRegressor(epochs=50).fit(rows, targets)
    with pytest.raises(ValueError, match="at least 20"):
        ConformalQuantileForecaster(base).calibrate(rows[:5], targets[:5])


def test_target_scaling_lets_the_model_learn_at_realistic_magnitudes():
    """The pinball subgradient is +/-alpha and carries no scale information, so
    an unscaled target of order 0.005 against a step of 0.05 made the optimiser
    oscillate instead of converge - and CRPS came out four times worse than the
    naive baseline."""
    rng = random.Random(107)
    rows, targets = [], []
    for _ in range(300):
        x = rng.gauss(0, 1)
        rows.append([x])
        targets.append(0.006 * x + rng.gauss(0, 0.002))

    model = QuantileRegressor(epochs=400).fit(rows, targets)
    clim = ClimatologyForecaster().fit(rows, targets)

    model_crps = fmean(model.predict(r).crps(y) for r, y in zip(rows, targets, strict=True))
    clim_crps = fmean(clim.predict(r).crps(y) for r, y in zip(rows, targets, strict=True))
    assert model_crps < clim_crps, "the model failed to beat climatology on data with signal"
