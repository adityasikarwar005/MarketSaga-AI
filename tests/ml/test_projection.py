"""M62 - scoring the projection, and the baselines that make the score mean anything.

Two things here are easy to get wrong in ways that flatter the model.

A baseline nobody fitted is a baseline the model beats by construction, and the
margin over it reads as skill. And the option-implied distribution is
risk-neutral, so beating it on CRPS can be the variance risk premium rather
than forecast skill - a volatility-selling edge with a volatility seller's tail.

The PIT test had a third: it compared observed bucket counts against equal
expected counts, when a five-quantile grid partitions the outcome space into
buckets of 5%, 20%, 25%, 25%, 20% and 5%.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from libs.ms_models.projection import (
    COVERAGE_TOLERANCE,
    MIN_CRPS_IMPROVEMENT,
    MIN_EVAL_SAMPLES,
    BaselineComparison,
    GarchBaseline,
    ImpliedBaseline,
    ProjectionError,
    RandomWalkBaseline,
    chi_square_critical,
    evaluate_gate,
    gaussian_forecast,
    pit_test,
    walk_forward,
)
from libs.ms_models.quantile import (
    DEFAULT_QUANTILES,
    QuantileForecast,
    pit_bucket_probabilities,
    score_forecasts,
)

ND = NormalDist()
SEED = 9


def garch_series(n: int = 1200, seed: int = SEED, *, a: float = 0.10, b: float = 0.85):
    """A series with real volatility clustering, plus its true conditional variance."""
    rng = random.Random(seed)
    h = 1e-4
    series: list[float] = []
    variances: list[float] = []
    for _ in range(n):
        variances.append(h)
        r = math.sqrt(h) * rng.gauss(0.0, 1.0)
        series.append(r)
        h = 1e-5 + a * r * r + b * h
    return series, variances


def oracle_forecasts(variances, warmup: int):
    return [
        gaussian_forecast(math.sqrt(variances[warmup + i])) for i in range(len(variances) - warmup)
    ]


def constant_scores(scale: float, n: int = 600, seed: int = 3):
    rng = random.Random(seed)
    actuals = [rng.gauss(0.0, 1.0) for _ in range(n)]
    f = QuantileForecast(quantiles={a: ND.inv_cdf(a) * scale for a in DEFAULT_QUANTILES})
    return score_forecasts(f"scale-{scale}", [f] * n, actuals)


# --------------------------------------------------------------------------- #
# the PIT test that was wrong
# --------------------------------------------------------------------------- #
def test_the_pit_buckets_are_not_equal_in_width():
    """Five cut points make six buckets, and their nominal probabilities are
    5, 20, 25, 25, 20 and 5 percent. Assuming a sixth each is the defect this
    replaced."""
    probs = pit_bucket_probabilities(DEFAULT_QUANTILES)
    assert probs == pytest.approx([0.05, 0.20, 0.25, 0.25, 0.20, 0.05])
    assert sum(probs) == pytest.approx(1.0)


def test_a_perfectly_calibrated_model_passes_the_pit_test():
    """It did not before. Against equal expected counts a perfect model scored
    0.150 rather than 0, and a model whose 90% band covered 63% scored 0.152 -
    1.4% worse while being catastrophically miscalibrated."""
    score = constant_scores(1.0, n=20000)
    assert score.coverage_90 == pytest.approx(0.90, abs=0.01)

    result = pit_test(score)
    assert result.uniform, f"chi2 {result.statistic:.1f} vs critical {result.critical:.1f}"


def test_the_pit_test_discriminates_where_it_used_to_be_flat():
    """The failure that mattered: near the old floor there was no signal."""
    perfect = pit_test(constant_scores(1.0, n=20000)).statistic
    slightly_narrow = pit_test(constant_scores(0.90, n=20000)).statistic
    badly_narrow = pit_test(constant_scores(0.55, n=20000)).statistic

    assert perfect < slightly_narrow < badly_narrow
    assert slightly_narrow > perfect * 10, "a 4-point coverage miss should be visible"
    assert not pit_test(constant_scores(0.90, n=20000)).uniform


def test_a_too_wide_forecast_fails_the_pit_test_too():
    """Over-covering is miscalibration as well - it sizes every position too
    small and skips trades that were there."""
    wide = constant_scores(2.0, n=20000)
    assert wide.coverage_90 > 0.97
    assert not pit_test(wide).uniform


def test_the_pit_test_reports_a_critical_value_not_just_a_statistic():
    """A statistic without one is a number nobody can act on, and the gate says
    "PIT uniform" rather than "PIT statistic small"."""
    result = pit_test(constant_scores(1.0))
    assert result.dof == len(pit_bucket_probabilities(DEFAULT_QUANTILES)) - 1 == 5
    assert result.critical > 0
    assert "critical" in result.render()


def test_the_chi_square_approximation_matches_the_published_table():
    """Wilson-Hilferty rather than scipy. Pinned against the table so the
    approximation cannot drift into being wrong where it is used."""
    table = {2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592, 8: 15.507, 10: 18.307}
    for dof, want in table.items():
        assert chi_square_critical(dof) == pytest.approx(want, rel=0.01), f"dof={dof}"


def test_the_chi_square_approximation_is_weakest_at_one_dof():
    """2.4% low there, under 1% everywhere this module uses it. Stated rather
    than hidden, because the one place it would matter is a two-bucket PIT."""
    assert chi_square_critical(1) == pytest.approx(3.841, rel=0.03)
    assert chi_square_critical(1) < 3.841


def test_pit_test_on_nothing_is_refused():
    with pytest.raises(ProjectionError, match="no PIT values"):
        pit_test(score_forecasts("empty", [], []))


# --------------------------------------------------------------------------- #
# the baselines
# --------------------------------------------------------------------------- #
def test_the_garch_baseline_is_fitted_not_assumed():
    """A baseline given textbook constants is one the model beats by
    construction, and the margin says more about the constants than the model."""
    series, _ = garch_series()
    g = GarchBaseline().fit(series[:600])

    assert 0.0 < g.alpha < 0.999
    assert 0.0 < g.beta < 0.999
    assert g.alpha + g.beta < 1.0
    assert g.log_likelihood > -math.inf


def test_the_garch_fit_finds_the_generating_parameters_roughly():
    series, _ = garch_series(n=2000, a=0.10, b=0.85)
    g = GarchBaseline().fit(series)
    assert abs(g.alpha - 0.10) <= 0.08
    assert abs(g.beta - 0.85) <= 0.15


def test_the_garch_fit_beats_its_own_untuned_defaults():
    """The point of fitting: the chosen parameters must actually score better
    on the training likelihood than a plausible fixed pair."""
    series, _ = garch_series(n=1500)
    g = GarchBaseline().fit(series)
    unconditional = sum(r * r for r in series) / len(series)

    fixed = GarchBaseline._log_likelihood(series, unconditional * (1 - 0.05 - 0.90), 0.05, 0.90)
    assert g.log_likelihood >= fixed


def test_the_garch_baseline_refuses_a_short_history():
    with pytest.raises(ProjectionError, match="beats by construction"):
        GarchBaseline().fit([0.01] * 20)


def test_the_random_walk_baseline_widens_with_realised_volatility():
    calm = RandomWalkBaseline().fit([0.001 * (-1) ** i for i in range(300)]).predict()
    wild = RandomWalkBaseline().fit([0.02 * (-1) ** i for i in range(300)]).predict()
    assert wild.width() > calm.width() * 5


def test_the_random_walk_baseline_uses_only_its_window():
    long_ago = [0.05 * (-1) ** i for i in range(500)]
    recent = [0.001 * (-1) ** i for i in range(300)]
    windowed = RandomWalkBaseline(window=250).fit(long_ago + recent).predict()
    assert windowed.width() < 0.02


def test_the_implied_baseline_is_marked_risk_neutral():
    """The caveat has to live on the object, not in a comment: this is the
    baseline whose result is most likely to look like the strongest one."""
    assert ImpliedBaseline.risk_neutral is True
    assert ImpliedBaseline.__doc__ is not None
    assert "risk-neutral" in ImpliedBaseline.__doc__.lower()


def test_the_implied_baseline_scales_by_root_time():
    a = ImpliedBaseline().fit_from_iv(0.13, horizon_years=0.01).predict()
    b = ImpliedBaseline().fit_from_iv(0.13, horizon_years=0.04).predict()
    assert b.width() == pytest.approx(a.width() * 2.0, rel=1e-9)


def test_beating_the_risk_neutral_baseline_carries_its_caveat_into_the_output():
    """Not into a comment. A result that reads as skill and is a premium has to
    say so where the result is printed."""
    challenger = constant_scores(1.0)
    implied = constant_scores(2.0)  # much wider, as a risk-neutral density is

    c = BaselineComparison(baseline=implied, challenger=challenger, risk_neutral=True)
    assert c.beaten
    assert "variance risk premium" in c.body()
    assert "volatility seller" in c.body()


def test_a_beaten_physical_baseline_carries_no_such_caveat():
    c = BaselineComparison(
        baseline=constant_scores(2.0), challenger=constant_scores(1.0), risk_neutral=False
    )
    assert c.beaten
    assert "variance risk premium" not in c.body()


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_the_gate_needs_every_baseline_beaten_not_their_average():
    """Beating a mean lets a model lose badly to one baseline and pass, and the
    one it loses to is the informative one."""
    challenger = constant_scores(1.0)
    easy = constant_scores(3.0)  # trivially beaten
    hard = constant_scores(1.0)  # a tie

    gate = evaluate_gate(challenger, [(easy, False), (hard, False)])
    assert not gate.passes
    assert gate.unbeaten == [hard.name]
    assert "not their average" in gate.render()


def test_the_gate_fails_on_coverage_even_with_a_good_crps():
    """A model with better CRPS and broken coverage produces confident-looking
    intervals that are wrong, and a position sized off them is sized off
    nothing."""
    narrow = constant_scores(0.55, n=2000)
    assert abs(narrow.coverage_90 - 0.90) > COVERAGE_TOLERANCE

    gate = evaluate_gate(narrow, [(constant_scores(4.0, n=2000), False)])
    assert not gate.coverage_ok
    assert not gate.passes


def test_the_gate_fails_when_the_pit_is_not_uniform():
    slightly_off = constant_scores(0.88, n=20000)
    gate = evaluate_gate(slightly_off, [(constant_scores(4.0, n=20000), False)])
    assert not gate.pit.uniform
    assert not gate.passes


def test_a_gate_with_no_baselines_does_not_pass():
    """A gate with nothing to beat is a gate that passes."""
    gate = evaluate_gate(constant_scores(1.0), [])
    assert not gate.passes
    assert "nothing to beat" in gate.render()


def test_the_gate_refuses_a_thin_evaluation():
    """Below a couple of hundred outcomes the coverage estimate is wider than
    the tolerance it is being checked against."""
    thin = constant_scores(1.0, n=MIN_EVAL_SAMPLES - 1)
    gate = evaluate_gate(thin, [(constant_scores(4.0, n=MIN_EVAL_SAMPLES - 1), False)])
    assert not gate.passes
    assert "coverage estimate is wider" in gate.render()


def test_comparing_scores_from_different_samples_is_refused():
    """CRPS across two different sets of outcomes compares two models on two
    different questions."""
    with pytest.raises(ProjectionError, match="two different questions"):
        evaluate_gate(constant_scores(1.0, n=600), [(constant_scores(2.0, n=500), False)])


def test_the_gate_passes_when_everything_is_met():
    challenger = constant_scores(1.0, n=2000)
    weak_a = constant_scores(2.5, n=2000)
    weak_b = constant_scores(3.0, n=2000)

    gate = evaluate_gate(challenger, [(weak_a, False), (weak_b, False)])
    assert gate.coverage_ok and gate.pit.uniform
    assert all(c.crps_improvement >= MIN_CRPS_IMPROVEMENT for c in gate.comparisons)
    assert gate.passes
    assert "PASSES" in gate.render()


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def test_walk_forward_refits_each_baseline_on_data_up_to_the_point_only():
    """Fitting once on everything lets the baselines see the outcomes they are
    scored on. That makes them look better and the challenger's margin smaller -
    the error that flatters the wrong side and so never gets questioned."""
    import inspect

    src = inspect.getsource(walk_forward)
    assert "history[: warmup + i]" in src
    assert "everything up to, not including" in src


def test_walk_forward_scores_everything_on_the_same_outcomes():
    series, variances = garch_series()
    warmup = 600
    result = walk_forward(series, oracle_forecasts(variances, warmup), warmup=warmup)

    assert result.challenger.n == result.random_walk.n == result.garch.n
    assert result.challenger.n == len(series) - warmup


def test_walk_forward_says_when_the_implied_baseline_was_not_scored():
    """Gate G6 names three baselines. Passing with two is passing two thirds of
    the gate, and the report has to say which."""
    series, variances = garch_series()
    result = walk_forward(series, oracle_forecasts(variances, 600), warmup=600)

    assert result.implied is None
    assert any("two thirds of the gate" in n for n in result.gate.notes)


def test_walk_forward_scores_the_implied_baseline_when_given_vols():
    series, variances = garch_series()
    warmup = 600
    n_eval = len(series) - warmup
    result = walk_forward(
        series,
        oracle_forecasts(variances, warmup),
        warmup=warmup,
        implied_vols=[0.16] * n_eval,
        horizon_years=1 / 252,
    )
    assert result.implied is not None
    assert len(result.gate.comparisons) == 3
    assert any(c.risk_neutral for c in result.gate.comparisons)


def test_walk_forward_refuses_a_mismatched_number_of_forecasts():
    series, variances = garch_series()
    with pytest.raises(ProjectionError, match="one for one"):
        walk_forward(series, oracle_forecasts(variances, 600)[:-5], warmup=600)


def test_walk_forward_refuses_a_warmup_too_short_to_fit_a_baseline():
    series, _ = garch_series()
    with pytest.raises(ProjectionError, match="at least 30"):
        walk_forward(series, [], warmup=10)


# --------------------------------------------------------------------------- #
# how much headroom the gate's 8% actually is
# --------------------------------------------------------------------------- #
def test_an_oracle_barely_beats_a_fitted_garch_on_garch_data():
    """Worth knowing before anyone tunes toward the threshold.

    A challenger handed the *true* conditional variance beats a fitted GARCH by
    about a fifth of a percent on data a GARCH generated - because there, the
    fitted GARCH is the oracle. The gate's 8% is only reachable if the real
    series carries structure GARCH misses, which is what the gate is really
    asking about.

    The corollary is the useful half: on a series without such structure, a
    comfortable 8% margin over a fitted GARCH is more likely a baseline that was
    not fitted than a model that is good.
    """
    series, variances = garch_series(seed=17)
    warmup = 600
    result = walk_forward(series, oracle_forecasts(variances, warmup), warmup=warmup)

    against_garch = (result.garch.crps - result.challenger.crps) / result.garch.crps
    assert abs(against_garch) < 0.02, f"oracle beat fitted GARCH by {against_garch:+.2%}"
    assert against_garch < MIN_CRPS_IMPROVEMENT
    assert not result.gate.passes


def test_an_oracle_does_beat_the_random_walk():
    """The conditioning is worth something - just not eight percent against a
    model that shares its structure."""
    series, variances = garch_series(seed=9)
    warmup = 600
    result = walk_forward(series, oracle_forecasts(variances, warmup), warmup=warmup)

    against_rw = (result.random_walk.crps - result.challenger.crps) / result.random_walk.crps
    assert against_rw > 0.0


def test_the_report_renders_every_baseline_and_the_verdict():
    series, variances = garch_series()
    text = walk_forward(series, oracle_forecasts(variances, 600), warmup=600).render()

    assert "random-walk" in text
    assert "garch(1,1)" in text
    assert "GATE G6" in text
