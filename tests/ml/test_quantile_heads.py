"""M54 - the quantile heads, and the failure that looks calibrated.

The interesting one here is not a crash. Newton boosting on the pinball loss
divides by a second derivative that is identically zero, and the resulting model
trains, orders correctly, and reports **marginal coverage that looks right** -
while its interval barely moves with conditional volatility. Too wide when it is
quiet, too narrow when it is not, and the two errors cancel in the headline
number.

So the tests here measure conditional behaviour, not just coverage.
"""

from __future__ import annotations

import random
from statistics import NormalDist, fmean

import pytest

from libs.ms_models.quantile import DEFAULT_QUANTILES, QuantileForecast
from libs.ms_models.quantile_heads import (
    MIN_ROWS_PER_HEAD,
    QuantileBooster,
    QuantileHeadError,
    QuantileHeads,
    empirical_quantile,
    gaussian_forecast,
    gaussian_quantile,
    mean_pinball,
    pinball_gradient,
    score_heads,
)

SEED = 20260828


def heteroscedastic(n: int = 1200, seed: int = SEED):
    """y = 2*x1 + noise whose sd triples with x2. x3 is pure noise.

    The conditional quantile is known in closed form, which is what lets these
    tests ask whether the model found it rather than whether it looks tidy.
    """
    rng = random.Random(seed)
    rows, ys = [], []
    for _ in range(n):
        x1, x2 = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0)
        rows.append([x1, x2, rng.uniform(-1.0, 1.0)])
        ys.append(2.0 * x1 + rng.gauss(0.0, sd(x2)))
    return rows, ys


def sd(x2: float) -> float:
    return 0.5 + 1.5 * x2


def true_quantile(row, alpha: float) -> float:
    return 2.0 * row[0] + sd(row[1]) * NormalDist().inv_cdf(alpha)


def split(rows, ys, cut: int = 900):
    return rows[:cut], ys[:cut], rows[cut:], ys[cut:]


# --------------------------------------------------------------------------- #
# leaf renewal - the thing that makes it a quantile model
# --------------------------------------------------------------------------- #
def test_renewal_tracks_conditional_volatility_and_without_it_the_band_is_flat():
    """The finding. Both models report plausible marginal coverage; only one
    widens its interval when the noise widens.

    The true 5-95 half-width more than triples between the quiet and busy ends
    of this data. A band that does not move is too wide on the quiet days -
    which costs trades - and exactly as narrow as before on the busy ones,
    which costs money.
    """
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, _ = split(rows, ys)

    def half_widths(renew: bool) -> tuple[float, float]:
        hi = QuantileBooster(0.95, renew_leaves=renew).fit(tr_r, tr_y)
        lo = QuantileBooster(0.05, renew_leaves=renew).fit(tr_r, tr_y)
        quiet = fmean((hi.predict(r) - lo.predict(r)) / 2 for r in te_r if r[1] < 0.25)
        busy = fmean((hi.predict(r) - lo.predict(r)) / 2 for r in te_r if r[1] > 0.75)
        return quiet, busy

    q_on, b_on = half_widths(True)
    q_off, b_off = half_widths(False)
    ratio_on, ratio_off = b_on / q_on, b_off / q_off

    assert ratio_on > 1.6, f"with renewal the band only widened {ratio_on:.2f}x"
    assert ratio_off < 1.4, (
        f"without renewal the band widened {ratio_off:.2f}x; this test no longer "
        "demonstrates anything"
    )
    assert ratio_on > ratio_off * 1.3


def test_renewal_gets_closer_to_the_true_conditional_quantile():
    """Measured against a closed form rather than against itself."""
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, _ = split(rows, ys)

    def mae(renew: bool, alpha: float) -> float:
        m = QuantileBooster(alpha, renew_leaves=renew).fit(tr_r, tr_y)
        return fmean(abs(m.predict(r) - true_quantile(r, alpha)) for r in te_r)

    for alpha in (0.05, 0.95):
        on, off = mae(True, alpha), mae(False, alpha)
        assert on < off * 0.75, f"a={alpha}: renewal {on:.3f} vs without {off:.3f}"


def test_without_renewal_it_never_stops_early():
    """The steps never converge on anything, so the validation loss never
    settles and the fit runs to the tree cap. That is a visible symptom of an
    invisible problem."""
    rows, ys = heteroscedastic()
    tr_r, tr_y, _, _ = split(rows, ys)

    off = QuantileBooster(0.95, renew_leaves=False, n_estimators=200).fit(tr_r, tr_y)
    on = QuantileBooster(0.95, renew_leaves=True, n_estimators=200).fit(tr_r, tr_y)

    assert len(off.trees) == 200
    assert len(on.trees) < 100
    assert on.report is not None and on.report.stopped_early


def test_marginal_coverage_alone_would_not_have_caught_it():
    """Why the tests above measure conditional behaviour. Both models land near
    their nominal rate marginally - the over- and under-coverage cancel."""
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, te_y = split(rows, ys)

    for renew in (True, False):
        hi = QuantileBooster(0.95, renew_leaves=renew).fit(tr_r, tr_y)
        lo = QuantileBooster(0.05, renew_leaves=renew).fit(tr_r, tr_y)
        inside = sum(
            1 for r, a in zip(te_r, te_y, strict=True) if lo.predict(r) <= a <= hi.predict(r)
        ) / len(te_y)
        assert 0.82 < inside < 0.97, f"renew={renew} marginal coverage {inside:.1%}"


def test_the_hessian_is_documented_as_a_device_not_a_derivative():
    """The pinball loss has no curvature. A constant hessian is there to make
    the split search well-defined and every leaf value it produces is thrown
    away."""
    import libs.ms_models.quantile_heads as mod

    assert mod._UNIT_HESSIAN == 1.0
    assert mod.__doc__ is not None and "second derivative is zero" in mod.__doc__


# --------------------------------------------------------------------------- #
# does it actually fit a quantile?
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_each_head_recovers_its_own_nominal_rate(alpha: float):
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, te_y = split(rows, ys)
    m = QuantileBooster(alpha).fit(tr_r, tr_y)

    below = sum(1 for r, a in zip(te_r, te_y, strict=True) if a <= m.predict(r)) / len(te_y)
    assert abs(below - alpha) < 0.06, f"a={alpha} put {below:.1%} of outcomes below its line"


def test_a_head_beats_the_unconditional_quantile_it_starts_from():
    """The base score is the marginal quantile. A model that does not improve on
    it has found nothing, and its pinball loss would say so."""
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, te_y = split(rows, ys)
    m = QuantileBooster(0.5).fit(tr_r, tr_y)

    from libs.ms_models.quantile import pinball_loss

    fitted = fmean(pinball_loss(a, m.predict(r), 0.5) for r, a in zip(te_r, te_y, strict=True))
    flat = fmean(pinball_loss(a, m.base_score, 0.5) for a in te_y)
    assert fitted < flat * 0.85


def test_a_head_on_pure_noise_stays_near_the_unconditional_quantile():
    """The null test. No conditional signal, so the model must not invent one -
    it should stop early and sit close to the marginal quantile everywhere."""
    rng = random.Random(SEED)
    rows = [[rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(1200)]
    ys = [rng.gauss(0.0, 1.0) for _ in rows]
    tr_r, tr_y, te_r, _ = split(rows, ys)

    m = QuantileBooster(0.9).fit(tr_r, tr_y)
    preds = m.predict_all(te_r)
    spread = max(preds) - min(preds)

    assert spread < 1.2, f"invented a conditional spread of {spread:.2f} out of noise"
    assert len(m.trees) < 60, f"grew {len(m.trees)} trees on data with no signal"


def test_the_fit_is_deterministic():
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, _ = split(rows, ys)
    a = QuantileBooster(0.75).fit(tr_r, tr_y).predict_all(te_r)
    b = QuantileBooster(0.75).fit(tr_r, tr_y).predict_all(te_r)
    assert a == b


def test_a_thin_slice_is_refused_rather_than_fitted():
    rows, ys = heteroscedastic(n=MIN_ROWS_PER_HEAD - 1)
    with pytest.raises(QuantileHeadError, match="tail of a sample"):
        QuantileBooster(0.05).fit(rows, ys)


def test_alpha_outside_the_open_interval_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(QuantileHeadError, match="strictly between"):
            QuantileBooster(bad)


def test_the_gradient_is_the_pinball_subgradient():
    assert pinball_gradient(1.0, 0.0, 0.9) == pytest.approx(-0.9)
    assert pinball_gradient(-1.0, 0.0, 0.9) == pytest.approx(0.1)


def test_the_empirical_quantile_interpolates():
    """Nearest-rank would let a leaf of thirty residuals step to only thirty
    values, and the quantisation shows up as a staircase in interval width."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert empirical_quantile(values, 0.5) == pytest.approx(2.0)
    assert empirical_quantile(values, 0.25) == pytest.approx(1.0)
    assert empirical_quantile(values, 0.125) == pytest.approx(0.5)
    assert empirical_quantile([], 0.5) == 0.0
    assert empirical_quantile([7.0], 0.9) == 7.0


# --------------------------------------------------------------------------- #
# the grid of heads
# --------------------------------------------------------------------------- #
def make_heads(horizons=(15, 30), **kw) -> tuple[QuantileHeads, list, list]:
    rows, ys = heteroscedastic()
    tr_r, tr_y, te_r, te_y = split(rows, ys)
    targets = {h: [y * (1.0 + h / 100.0) for y in tr_y] for h in horizons}
    heads = QuantileHeads(horizons=horizons, **kw).fit(tr_r, targets)
    return heads, te_r, te_y


def test_one_head_per_alpha_and_horizon():
    heads, _, _ = make_heads(horizons=(15, 30))
    assert set(heads.heads) == {(a, h) for a in DEFAULT_QUANTILES for h in (15, 30)}


def test_predictions_are_monotone_across_alphas():
    """Five independently fitted models have no reason to agree, and a 25th
    percentile above a 75th is a distribution with negative mass."""
    heads, te_r, _ = make_heads()
    for row in te_r[:50]:
        f = heads.predict(row, horizon=15)
        values = [f.quantiles[a] for a in sorted(DEFAULT_QUANTILES)]
        assert values == sorted(values), values


def test_the_number_of_monotonicity_repairs_is_reported_not_swallowed():
    """A head needing its order repaired on a third of its rows is a head that
    is not fitted, and the sort makes the output usable without making it
    right."""
    heads, te_r, _ = make_heads()
    for row in te_r[:100]:
        heads.predict(row, horizon=15)

    assert heads.report.predictions == 100
    assert 0.0 <= heads.report.repair_rate <= 1.0
    assert "monotonicity repaired" in heads.report.render()


def test_an_unfitted_horizon_is_refused():
    heads, te_r, _ = make_heads(horizons=(15,))
    with pytest.raises(QuantileHeadError, match="was not fitted"):
        heads.predict(te_r[0], horizon=60)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(QuantileHeadError, match="fit\\(\\) before predict"):
        QuantileHeads().predict([0.0, 0.0, 0.0], horizon=15)


def test_a_missing_horizon_in_the_targets_is_an_error_not_a_silent_skip():
    rows, ys = heteroscedastic()
    with pytest.raises(QuantileHeadError, match="no targets for horizon"):
        QuantileHeads(horizons=(15, 30)).fit(rows, {15: ys})


# --------------------------------------------------------------------------- #
# the Gaussian fallback
# --------------------------------------------------------------------------- #
def test_the_fallback_is_per_head_not_per_model():
    """A 5% head can fail on a quiet quarter while the median head is fine.
    Dropping the whole distribution for that loses information the trade could
    have used."""
    rows, ys = heteroscedastic(n=MIN_ROWS_PER_HEAD - 1)
    heads = QuantileHeads(horizons=(15,)).fit(rows, {15: ys}, sigma_by_horizon={15: 1.0})

    assert heads.heads == {}
    assert len(heads.report.fallbacks) == len(DEFAULT_QUANTILES)
    assert all(heads.used_fallback(a, 15) for a in DEFAULT_QUANTILES)

    f = heads.predict(rows[0], horizon=15)
    assert f.quantiles[0.5] == pytest.approx(0.0, abs=1e-9)
    assert f.quantiles[0.95] > f.quantiles[0.05]


def test_a_failed_head_with_no_sigma_raises_rather_than_predicting_nothing():
    """Finding out at predict time is the wrong end of the day."""
    rows, ys = heteroscedastic(n=MIN_ROWS_PER_HEAD - 1)
    with pytest.raises(QuantileHeadError, match="no sigma was supplied"):
        QuantileHeads(horizons=(15,)).fit(rows, {15: ys})


def test_the_fallback_matches_the_normal_distribution():
    nd = NormalDist(0.0, 2.0)
    for alpha in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        assert gaussian_quantile(2.0, alpha) == pytest.approx(nd.inv_cdf(alpha), abs=1e-6)


def test_the_fallback_is_symmetric_and_says_so():
    """It cannot represent skew, which matters when a fallback head is carrying
    a tail."""
    f = gaussian_forecast(1.5)
    assert f.quantiles[0.05] == pytest.approx(-f.quantiles[0.95], abs=1e-9)
    assert f.quantiles[0.25] == pytest.approx(-f.quantiles[0.75], abs=1e-9)
    assert gaussian_quantile.__doc__ is not None
    assert "skew" in gaussian_quantile.__doc__


def test_a_negative_sigma_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        gaussian_quantile(-1.0, 0.5)


def test_a_zero_sigma_collapses_to_a_point_rather_than_erroring():
    f = gaussian_forecast(0.0)
    assert all(v == pytest.approx(0.0) for v in f.quantiles.values())


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_heads_reports_pinball_per_alpha_and_coverage_on_the_outer_band():
    forecasts = [
        QuantileForecast(quantiles={a: NormalDist().inv_cdf(a) for a in DEFAULT_QUANTILES})
        for _ in range(500)
    ]
    rng = random.Random(SEED)
    actuals = [rng.gauss(0.0, 1.0) for _ in forecasts]

    scores = score_heads(forecasts, actuals)
    assert [s.alpha for s in scores] == sorted(DEFAULT_QUANTILES)
    outer = [s for s in scores if s.alpha in (0.05, 0.95)]
    assert all(s.coverage is not None for s in outer)
    assert abs(outer[0].coverage - 0.90) < 0.05


def test_mean_pinball_prefers_the_better_forecaster():
    rng = random.Random(SEED)
    actuals = [rng.gauss(0.0, 1.0) for _ in range(500)]
    good = [
        QuantileForecast(quantiles={a: NormalDist().inv_cdf(a) for a in DEFAULT_QUANTILES})
        for _ in actuals
    ]
    wide = [
        QuantileForecast(quantiles={a: 4.0 * NormalDist().inv_cdf(a) for a in DEFAULT_QUANTILES})
        for _ in actuals
    ]
    assert mean_pinball(good, actuals) < mean_pinball(wide, actuals)


def test_the_report_names_which_heads_fell_back():
    rows, ys = heteroscedastic(n=MIN_ROWS_PER_HEAD - 1)
    heads = QuantileHeads(horizons=(15,)).fit(rows, {15: ys}, sigma_by_horizon={15: 1.0})
    text = heads.report.render()
    assert "FALLBACK to Gaussian" in text


def test_the_report_shows_whether_each_head_stopped_early_or_hit_the_cap():
    heads, _, _ = make_heads(horizons=(15,))
    text = heads.report.render()
    assert "trees=" in text
    assert "early" in text or "cap" in text


# --------------------------------------------------------------------------- #
# gaps a mutation test found: these three defences were unprotected
# --------------------------------------------------------------------------- #
def test_a_crossing_head_is_actually_repaired_and_counted():
    """The monotonicity test above never fired: on well-behaved data the raw
    heads already come out ordered, so deleting `enforce_monotone` changed
    nothing and no test noticed.

    This forces the crossing instead of hoping for one.
    """

    class Crossing:
        """Stands in for a head that has fitted itself above its neighbours."""

        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, _row) -> float:
            return self.value

    heads, te_r, _ = make_heads(horizons=(15,))
    # 0.25 above 0.75: a distribution with negative mass between them.
    heads.heads[(0.25, 15)] = Crossing(9.0)  # type: ignore[assignment]
    heads.heads[(0.75, 15)] = Crossing(1.0)  # type: ignore[assignment]

    before = heads.report.monotonicity_repairs
    f = heads.predict(te_r[0], horizon=15)
    values = [f.quantiles[a] for a in sorted(DEFAULT_QUANTILES)]

    assert values == sorted(values), f"a crossing survived the repair: {values}"
    assert heads.report.monotonicity_repairs == before + 1


def test_the_base_score_is_the_marginal_quantile_it_has_to_beat():
    """Pinned directly rather than inferred from downstream behaviour. Boosting
    recovers from a zero start given enough trees, so nothing else here notices
    if the starting point is wrong - it just costs trees to walk back to it."""
    rows, ys = heteroscedastic()
    tr_r, tr_y, _, _ = split(rows, ys)

    for alpha in (0.05, 0.5, 0.95):
        m = QuantileBooster(alpha).fit(tr_r, tr_y)
        expected_from = tr_y[: len(tr_y) - int(len(tr_y) * m.validation_share)]
        assert m.base_score == pytest.approx(empirical_quantile(expected_from, alpha))


def test_the_reported_validation_loss_belongs_to_the_model_that_is_returned():
    """Early stopping finds the best tree count; the trees grown after it must
    be dropped. Keeping them leaves a model whose reported validation loss is a
    number about a different model - which is the kind of report that survives
    every review, because it is internally consistent and simply not about the
    thing in front of you.
    """
    from libs.ms_models.quantile import pinball_loss

    rows, ys = heteroscedastic()
    tr_r, tr_y, _, _ = split(rows, ys)

    alpha = 0.9
    m = QuantileBooster(alpha).fit(tr_r, tr_y)
    assert m.report is not None and m.report.valid_pinball is not None

    n_valid = int(len(tr_r) * m.validation_share)
    valid_rows, valid_y = tr_r[len(tr_r) - n_valid :], tr_y[len(tr_y) - n_valid :]
    actual = fmean(
        pinball_loss(a, m.predict(r), alpha) for r, a in zip(valid_rows, valid_y, strict=True)
    )

    assert actual == pytest.approx(m.report.valid_pinball, rel=1e-9), (
        f"the report says {m.report.valid_pinball:.6f} and the returned model scores "
        f"{actual:.6f}; the trees after the best validation point were not dropped"
    )
