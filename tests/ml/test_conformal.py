"""Split-conformal calibration, and the ways a width correction can cheat.

The measured problem: a nominal 90% band from M54 holds the outcome 83-85% of
the time, and M75 sizes 1.09x to 11.38x too large as a result. The obvious fix -
widen until coverage reaches 90% - reports a pass on the data it was tuned on.

So the tests here are mostly about what the correction must *refuse* to do:
invent a correction where none is needed, only ever widen, move the centre, or
claim a marginal number while a regime underneath it is wrong.
"""

from __future__ import annotations

import math
import random

import pytest

from libs.ms_models.conformal import (
    MIN_CALIBRATION_ROWS,
    ConformalError,
    apply_conformal,
    apply_mondrian,
    band_edges,
    conformity_score,
    coverage,
    coverage_by_regime,
    fit_conformal,
    fit_mondrian,
    spread,
)
from libs.ms_models.quantile import QuantileForecast

NOMINAL = 0.9


def _band(lo: float, mid: float, hi: float) -> QuantileForecast:
    return QuantileForecast(quantiles={0.05: lo, 0.5: mid, 0.95: hi})


def _series(n: int, *, half_width: float, sigma: float, seed: int = 11):
    """Bands of a fixed half-width against normal outcomes.

    With half_width == 1.645 * sigma the band is exactly a 90% interval, so the
    fitted scale must come out at 1. Anything narrower needs widening by a
    factor this test can compute in advance.
    """
    rng = random.Random(seed)
    forecasts = [_band(-half_width, 0.0, half_width) for _ in range(n)]
    actuals = [rng.gauss(0.0, sigma) for _ in range(n)]
    return forecasts, actuals


class TestNullCase:
    """The null test for this component: do not correct what is not broken."""

    def test_a_correctly_calibrated_band_is_left_alone(self) -> None:
        forecasts, actuals = _series(4000, half_width=1.645, sigma=1.0)
        band = fit_conformal(forecasts, actuals, nominal=NOMINAL)
        assert band.scale == pytest.approx(1.0, abs=0.06), (
            "a band that already covers 90% was rescaled - the method is "
            "manufacturing a correction rather than measuring one"
        )

    def test_an_over_covering_band_is_narrowed_not_left_alone(self) -> None:
        """A floor at 1.0 would make this a one-way ratchet that can only ever
        agree with the diagnosis it was built for."""
        forecasts, actuals = _series(4000, half_width=3.0, sigma=1.0)
        band = fit_conformal(forecasts, actuals, nominal=NOMINAL)
        assert band.scale < 1.0


class TestScaling:
    def test_an_under_covering_band_is_widened_by_about_the_right_factor(self) -> None:
        """Half-width 1.0 against sigma 1.0 covers ~68%. Reaching 90% needs
        1.645, so the scale should land near that."""
        forecasts, actuals = _series(4000, half_width=1.0, sigma=1.0)
        band = fit_conformal(forecasts, actuals, nominal=NOMINAL)
        assert band.scale == pytest.approx(1.645, rel=0.08)

    def test_the_median_does_not_move(self) -> None:
        """Only the width was measured wrong. Moving the centre would be a
        second, unmeasured correction riding along with the first."""
        f = _band(-1.0, 0.25, 2.0)
        band = fit_conformal(*_series(500, half_width=1.0, sigma=1.0), nominal=NOMINAL)
        assert apply_conformal(f, band).quantiles[0.5] == 0.25

    def test_an_asymmetric_band_stays_asymmetric(self) -> None:
        """A skewed forecast that is right about the skew should keep it."""
        f = _band(-1.0, 0.0, 3.0)
        band = fit_conformal(*_series(500, half_width=1.0, sigma=1.0), nominal=NOMINAL)
        out = apply_conformal(f, band).quantiles
        assert (out[0.95] - out[0.5]) / (out[0.5] - out[0.05]) == pytest.approx(3.0)

    def test_every_quantile_is_scaled_not_only_the_outer_pair(self) -> None:
        f = QuantileForecast(quantiles={0.05: -2.0, 0.25: -1.0, 0.5: 0.0, 0.75: 1.0, 0.95: 2.0})
        band = fit_conformal(*_series(500, half_width=1.0, sigma=1.0), nominal=NOMINAL)
        out = apply_conformal(f, band).quantiles
        assert out[0.25] == pytest.approx(-band.scale)
        assert out[0.75] == pytest.approx(band.scale)


class TestOutOfSample:
    def test_calibrating_on_one_block_gives_nominal_coverage_on_another(self) -> None:
        """The whole point. The scale is chosen on data the measurement does
        not include, so the coverage below is a result rather than a restated
        objective."""
        cal_f, cal_a = _series(2000, half_width=1.0, sigma=1.0, seed=1)
        test_f, test_a = _series(2000, half_width=1.0, sigma=1.0, seed=2)

        before = coverage(test_f, test_a, nominal=NOMINAL)
        band = fit_conformal(cal_f, cal_a, nominal=NOMINAL)
        after = coverage([apply_conformal(f, band) for f in test_f], test_a, nominal=NOMINAL)

        assert before < 0.75, "the uncalibrated band should be visibly short"
        assert after == pytest.approx(NOMINAL, abs=0.03)

    def test_coverage_is_bought_with_width(self) -> None:
        """The two numbers only mean something together: a wider band covering
        more is not news."""
        cal_f, cal_a = _series(1000, half_width=1.0, sigma=1.0, seed=3)
        band = fit_conformal(cal_f, cal_a, nominal=NOMINAL)
        widened = [apply_conformal(f, band) for f in cal_f]
        assert spread(widened, nominal=NOMINAL) > spread(cal_f, nominal=NOMINAL)


class TestRegimes:
    def test_the_marginal_number_can_hide_a_broken_regime(self) -> None:
        """M54's own audit already recorded this shape: 97.1% quiet and 84.6%
        busy averaging to something that reads as calibrated."""
        rng = random.Random(5)
        forecasts, actuals, vol = [], [], []
        for i in range(1200):
            busy = i % 2 == 0
            sigma = 3.0 if busy else 0.3
            forecasts.append(_band(-1.645, 0.0, 1.645))
            actuals.append(rng.gauss(0.0, sigma))
            vol.append(sigma)

        marginal = coverage(forecasts, actuals, nominal=NOMINAL)
        by_regime = dict(
            (label, cov)
            for label, _n, cov in coverage_by_regime(forecasts, actuals, vol, nominal=NOMINAL)
        )
        assert by_regime["quiet"] > marginal > by_regime["busy"]
        assert by_regime["quiet"] - by_regime["busy"] > 0.5, (
            "the regimes should be far apart here - this test exists to show "
            "the marginal number averaging over them"
        )

    def test_regime_buckets_partition_the_sample(self) -> None:
        forecasts, actuals = _series(300, half_width=1.0, sigma=1.0)
        vol = [float(i) for i in range(300)]
        buckets = coverage_by_regime(forecasts, actuals, vol, nominal=NOMINAL)
        assert sum(n for _label, n, _cov in buckets) == 300


class TestGuards:
    def test_too_few_calibration_rows_is_refused(self) -> None:
        """At n=20 the 90th percentile is the second-largest value and one
        outlier moves it entirely."""
        forecasts, actuals = _series(MIN_CALIBRATION_ROWS - 1, half_width=1.0, sigma=1.0)
        with pytest.raises(ConformalError, match="too few"):
            fit_conformal(forecasts, actuals, nominal=NOMINAL)

    def test_mismatched_lengths_are_refused(self) -> None:
        forecasts, actuals = _series(200, half_width=1.0, sigma=1.0)
        with pytest.raises(ConformalError, match="same length"):
            fit_conformal(forecasts, actuals[:-1], nominal=NOMINAL)

    def test_a_forecast_without_the_needed_quantiles_is_refused(self) -> None:
        f = QuantileForecast(quantiles={0.25: -1.0, 0.5: 0.0, 0.75: 1.0})
        with pytest.raises(ConformalError, match="no quantile at"):
            conformity_score(f, 0.5, nominal=NOMINAL)

    def test_levels_are_matched_by_proximity_not_by_arithmetic(self) -> None:
        """(1.0 - 0.9) / 2 is 0.04999999999999999, and a dict keyed on 0.05 does
        not contain it. Deriving lookup keys by arithmetic raised on every
        forecast until this was matched by closeness instead."""
        assert (1.0 - NOMINAL) / 2.0 != 0.05
        assert band_edges(_band(-1.0, 0.0, 1.0), NOMINAL) == (0.05, 0.95)

    def test_a_level_that_is_genuinely_absent_still_raises(self) -> None:
        """Proximity matching must not become "close enough to anything"."""
        f = QuantileForecast(quantiles={0.2: -1.0, 0.5: 0.0, 0.8: 1.0})
        with pytest.raises(ConformalError, match="no quantile at"):
            band_edges(f, NOMINAL)

    def test_a_degenerate_band_scores_infinite_rather_than_being_skipped(self) -> None:
        """Dropping it would quietly improve the calibration set by removing its
        worst members."""
        assert conformity_score(_band(0.0, 0.0, 0.0), 1.0, nominal=NOMINAL) == math.inf

    def test_an_all_degenerate_calibration_set_is_refused(self) -> None:
        forecasts = [_band(0.0, 0.0, 0.0) for _ in range(200)]
        actuals = [1.0] * 200
        with pytest.raises(ConformalError, match="infinite"):
            fit_conformal(forecasts, actuals, nominal=NOMINAL)

    def test_the_finite_sample_rank_is_used(self) -> None:
        """ceil((n+1) * nominal), not the plain percentile. On 100 rows at 90%
        that is the 91st value, and the correction is why the method has a
        guarantee rather than an asymptotic hope."""
        forecasts = [_band(-1.0, 0.0, 1.0) for _ in range(100)]
        actuals = [i / 100.0 for i in range(100)]  # scores 0.00 .. 0.99
        band = fit_conformal(forecasts, actuals, nominal=0.9)
        assert band.scale == pytest.approx(0.90), "the 91st of 100 sorted scores"

    def test_a_scale_at_the_top_of_the_set_is_flagged(self) -> None:
        """An extrapolation dressed as a quantile.

        At n=120 and nominal 0.99 the finite-sample rank is ceil(121 * 0.99) =
        120, which is the whole set - so the scale is the largest score seen and
        there is nothing above it to have chosen from.
        """
        wide = QuantileForecast(quantiles={0.005: -1.0, 0.5: 0.0, 0.995: 1.0})
        forecasts = [wide for _ in range(120)]
        actuals = [0.0] * 119 + [50.0]
        band = fit_conformal(forecasts, actuals, nominal=0.99)
        assert band.extrapolating
        assert "top of the calibration set" in band.render()


def _heteroscedastic(n: int, *, half_width: float, quiet: float, busy: float, seed: int = 7):
    """Bands of fixed width against outcomes whose scale switches by regime.

    A fixed-width band over-covers the quiet half and under-covers the busy one -
    the shape measured on real bars, where the raw heads reached 98.0% quiet and
    88.1% busy. One global scale cannot fix that; it moves both together.
    """
    rng = random.Random(seed)
    forecasts, actuals, difficulty = [], [], []
    for i in range(n):
        sigma = busy if i % 2 else quiet
        forecasts.append(_band(-half_width, 0.0, half_width))
        actuals.append(rng.gauss(0.0, sigma))
        difficulty.append(sigma)
    return forecasts, actuals, difficulty


class TestMondrianNullCase:
    """The null test: buckets must not invent a difference that is not there."""

    def test_homoscedastic_data_gives_near_identical_scales(self) -> None:
        forecasts, actuals = _series(1800, half_width=1.0, sigma=1.0)
        difficulty = [1.0 + i * 1e-9 for i in range(1800)]  # meaningless ordering
        bands = fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=3)
        assert bands.spread_of_scales < 0.25, (
            "the buckets found a regime difference in data that has none - "
            f"scales {bands.scales}"
        )

    def test_and_agrees_with_the_global_scale_there(self) -> None:
        forecasts, actuals = _series(1800, half_width=1.0, sigma=1.0)
        difficulty = [1.0] * 1800
        glob = fit_conformal(forecasts, actuals, nominal=NOMINAL)
        bands = fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=3)
        assert all(abs(s - glob.scale) < 0.25 for s in bands.scales)


class TestMondrianOnRegimes:
    def test_the_busy_bucket_gets_the_wider_scale(self) -> None:
        forecasts, actuals, difficulty = _heteroscedastic(
            1800, half_width=1.645, quiet=0.4, busy=2.5
        )
        bands = fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=2)
        quiet_scale, busy_scale = bands.scales
        assert busy_scale > quiet_scale
        assert quiet_scale < 1.0 < busy_scale, (
            "the quiet band should narrow and the busy one widen - a single "
            "global scale can do neither"
        )

    def test_a_global_scale_gets_the_marginal_number_right_and_both_regimes_wrong(
        self,
    ) -> None:
        """The failure this whole module exists for, measured.

        On heteroscedastic data a single scale lands marginal coverage at 90.8%
        against a nominal 90% - a number that reads as calibrated - while the
        quiet tercile sits at 100% and the busy one at 81.6%. Both are wrong,
        in opposite directions, and the average hides it.
        """
        cal_f, cal_a, _cal_d = _heteroscedastic(1800, half_width=1.645, quiet=0.4, busy=2.5, seed=1)
        test_f, test_a, test_d = _heteroscedastic(
            1800, half_width=1.645, quiet=0.4, busy=2.5, seed=2
        )
        glob = fit_conformal(cal_f, cal_a, nominal=NOMINAL)
        scaled = [apply_conformal(f, glob) for f in test_f]

        marginal = coverage(scaled, test_a, nominal=NOMINAL)
        covs = [
            c
            for _l, _n, c in coverage_by_regime(scaled, test_a, test_d, nominal=NOMINAL, buckets=2)
        ]

        assert abs(marginal - NOMINAL) < 0.02, "the marginal number should look fine"
        assert max(covs) - min(covs) > 0.15, (
            "and the regimes underneath it should not be - if they agree here, "
            "this test has stopped demonstrating anything"
        )

    def test_per_bucket_scales_close_the_gap_the_global_one_leaves(self) -> None:
        cal_f, cal_a, cal_d = _heteroscedastic(1800, half_width=1.645, quiet=0.4, busy=2.5, seed=1)
        test_f, test_a, test_d = _heteroscedastic(
            1800, half_width=1.645, quiet=0.4, busy=2.5, seed=2
        )

        def gap(forecasts: list[QuantileForecast]) -> float:
            covs = [
                c
                for _l, _n, c in coverage_by_regime(
                    forecasts, test_a, test_d, nominal=NOMINAL, buckets=2
                )
            ]
            return max(covs) - min(covs)

        glob = fit_conformal(cal_f, cal_a, nominal=NOMINAL)
        global_gap = gap([apply_conformal(f, glob) for f in test_f])
        bands = fit_mondrian(cal_f, cal_a, cal_d, nominal=NOMINAL, buckets=2)
        mondrian_gap = gap(
            [apply_mondrian(f, d, bands) for f, d in zip(test_f, test_d, strict=True)]
        )

        assert global_gap > 0.15
        assert mondrian_gap < global_gap / 4, (
            f"per-bucket scales did not close the gap: global {global_gap:.1%}, "
            f"mondrian {mondrian_gap:.1%}"
        )

    def test_it_does_not_buy_that_by_wrecking_the_marginal_number(self) -> None:
        """Closing a gap by making both halves equally wrong would satisfy the
        test above and be worse than doing nothing."""
        cal_f, cal_a, cal_d = _heteroscedastic(1800, half_width=1.645, quiet=0.4, busy=2.5, seed=1)
        test_f, test_a, test_d = _heteroscedastic(
            1800, half_width=1.645, quiet=0.4, busy=2.5, seed=2
        )
        bands = fit_mondrian(cal_f, cal_a, cal_d, nominal=NOMINAL, buckets=2)
        out = [apply_mondrian(f, d, bands) for f, d in zip(test_f, test_d, strict=True)]
        assert coverage(out, test_a, nominal=NOMINAL) == pytest.approx(NOMINAL, abs=0.03)


class TestMondrianGuards:
    def test_bucket_edges_come_from_the_calibration_block(self) -> None:
        """Recomputing them on the test block lets the test data choose its own
        boundaries, which is a quiet way of fitting to what is being measured."""
        forecasts, actuals, difficulty = _heteroscedastic(1200, half_width=1.0, quiet=0.5, busy=2.0)
        bands = fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=2)
        assert len(bands.edges) == 1
        assert min(difficulty) <= bands.edges[0] <= max(difficulty)

    def test_a_value_outside_the_calibration_range_lands_in_an_end_bucket(self) -> None:
        """Refusing to predict on a quieter-than-ever day would be worse, and a
        silent global fallback would hide that it happened."""
        forecasts, actuals, difficulty = _heteroscedastic(1200, half_width=1.0, quiet=0.5, busy=2.0)
        bands = fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=2)
        assert bands.bucket_of(-99.0) == 0
        assert bands.bucket_of(99.0) == len(bands.bands) - 1

    def test_too_few_rows_per_bucket_is_refused(self) -> None:
        """Splitting a calibration set does not create information."""
        forecasts, actuals = _series(250, half_width=1.0, sigma=1.0)
        with pytest.raises(ConformalError, match="does not create information"):
            fit_mondrian(forecasts, actuals, [1.0] * 250, nominal=NOMINAL, buckets=3)

    def test_mismatched_lengths_are_refused(self) -> None:
        forecasts, actuals = _series(600, half_width=1.0, sigma=1.0)
        with pytest.raises(ConformalError, match="same length"):
            fit_mondrian(forecasts, actuals, [1.0] * 599, nominal=NOMINAL, buckets=2)

    def test_the_render_names_every_bucket_and_its_range(self) -> None:
        forecasts, actuals, difficulty = _heteroscedastic(1200, half_width=1.0, quiet=0.5, busy=2.0)
        lines = "\n".join(
            fit_mondrian(forecasts, actuals, difficulty, nominal=NOMINAL, buckets=2).render()
        )
        assert "-inf" in lines and "+inf" in lines
        assert "mondrian" in lines
