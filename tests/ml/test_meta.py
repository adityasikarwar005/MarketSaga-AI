"""M57 meta-labeller - the model whose job is to say no."""

from __future__ import annotations

import math
import random

import pytest

from libs.ms_models.meta import (
    SUSPICIOUS_APPROVAL_RATE,
    MetaLabeller,
    MetaSample,
    build_meta_labels,
    evaluate_at,
    evaluate_meta,
    threshold_for_recall,
    threshold_from_loss_ratio,
)


def world(n: int, seed: int, *, signal: bool = True):
    """Primary calls in a world where costs eat the small moves.

    The point of the setup: liquidity determines cost, cost determines whether a
    correct call actually pays, and the primary model never sees liquidity.
    """
    rng = random.Random(seed)
    probs, gross, costs, feats = [], [], [], []
    for _ in range(n):
        liquidity = rng.random()
        vol = math.exp(rng.gauss(0, 0.5))
        edge = rng.uniform(0.0, 0.25) if signal else 0.0
        p = 0.5 + edge * rng.choice([1, -1])
        correct = rng.random() < (0.5 + edge)
        move = vol * rng.expovariate(1 / 300.0) * (1 if correct else -1)
        cost = 120.0 + 400.0 * (1 - liquidity)
        probs.append(p)
        gross.append(move)
        costs.append(cost)
        feats.append({"liquidity": liquidity, "vol": vol, "hour": float(rng.randint(9, 15))})
    return build_meta_labels(probs, gross, costs, feats)


# ------------------------------------------------------------ the post-cost target
def test_the_label_is_profit_after_costs_not_direction():
    """A directionally correct call that does not clear its costs is a 0. This
    is the entire reason the module exists, and it is invisible to any metric
    that stops at direction."""
    correct_but_pointless = MetaSample(features={}, primary_prob=0.7, gross_pnl=80.0, cost=120.0)
    correct_and_paid = MetaSample(features={}, primary_prob=0.7, gross_pnl=800.0, cost=120.0)
    assert correct_but_pointless.gross_pnl > 0
    assert correct_but_pointless.paid == 0
    assert correct_and_paid.paid == 1


def test_min_edge_raises_the_bar_explicitly():
    """Zero is "did it make money". Anything else is a different question, and
    it has to be stated rather than implied."""
    breakeven = build_meta_labels([0.7], [200.0], [150.0], [{}])[0]
    demanding = build_meta_labels([0.7], [200.0], [150.0], [{}], min_edge=100.0)[0]
    assert breakeven.paid == 1
    assert demanding.paid == 0


def test_cost_to_edge_is_available_as_a_feature():
    """The most informative thing this model gets and the primary never sees."""
    samples = build_meta_labels([0.7], [400.0], [100.0], [{"liquidity": 0.5}])
    assert samples[0].features["cost_to_edge"] == pytest.approx(0.25)
    assert samples[0].features["primary_edge"] == pytest.approx(0.2)


def test_a_zero_move_has_infinite_cost_ratio_not_zero():
    sample = MetaSample(features={}, primary_prob=0.6, gross_pnl=0.0, cost=150.0)
    assert sample.cost_to_edge == math.inf
    assert sample.paid == 0


def test_caller_supplied_features_are_not_overwritten():
    samples = build_meta_labels([0.7], [400.0], [100.0], [{"primary_prob": 0.99}])
    assert samples[0].features["primary_prob"] == 0.99


# -------------------------------------------------------------- it must decline
def test_it_declines_most_of_what_it_sees():
    """A meta-labeller that approves nearly everything is a rubber stamp with a
    log loss."""
    model = MetaLabeller().fit(world(2000, 1))
    assert model.report is not None
    assert model.report.is_filtering
    assert model.report.approval_rate < 0.5, f"approved {model.report.approval_rate:.0%}"


def test_a_high_approval_rate_is_reported_rather_than_corrected():
    """Forcing the rate down by moving the threshold would hide the finding
    instead of surfacing it."""
    model = MetaLabeller(threshold=0.01).fit(world(1500, 2))
    assert model.report is not None
    if model.report.approval_rate >= SUSPICIOUS_APPROVAL_RATE:
        assert not model.report.is_filtering
        assert "NOT FILTERING" in model.report.render()
        assert any("not filtering anything" in n for n in model.report.notes)


def test_it_approves_nothing_when_nothing_predicts_anything():
    """The null case. Outcomes are coin flips and no feature carries signal, so
    the honest filter approves none of them."""
    rng = random.Random(9)
    samples = build_meta_labels(
        [0.5] * 2000,
        [rng.gauss(0, 300) for _ in range(2000)],
        [150.0] * 2000,
        [{"a": rng.random(), "b": rng.random()} for _ in range(2000)],
    )
    model = MetaLabeller().fit(samples)
    assert model.report is not None
    assert model.report.approval_rate < 0.1, f"approved {model.report.approval_rate:.0%} of noise"


# ------------------------------------------------------------------- the gate
def test_it_raises_precision_against_the_fallback_at_fixed_recall():
    """Precision alone is trivially maximised by approving one trade. Fixing
    recall first is what makes the comparison mean anything."""
    model = MetaLabeller().fit(world(2000, 3))
    result = evaluate_meta(model, world(1200, 4), target_recall=0.5)
    assert result.precision_gain > 0
    assert result.model.precision > result.take_everything.precision


def test_it_improves_post_cost_expectancy():
    model = MetaLabeller().fit(world(2000, 5))
    result = evaluate_meta(model, world(1200, 6), target_recall=0.5)
    assert result.expectancy_gain > 0
    assert result.passes


def test_earning_its_place_is_not_the_same_as_being_profitable():
    """The distinction that keeps a filter from being shipped on top of a
    strategy with no edge. On this world the model cuts the loss per trade
    roughly in half and the filtered book still loses - because the underlying
    calls had no edge to filter."""
    model = MetaLabeller().fit(world(2000, 7))
    result = evaluate_meta(model, world(1200, 8), target_recall=0.5)
    assert result.passes
    assert not result.net_positive
    assert "cannot manufacture an edge" in result.render()


def test_the_comparison_includes_the_fallback_not_only_take_everything():
    """A model that beats "take everything" but not "take the confident ones"
    has shown that primary confidence is informative, which was already known."""
    model = MetaLabeller().fit(world(2000, 10))
    result = evaluate_meta(model, world(1200, 11), target_recall=0.5)
    assert result.fallback.approved > 0
    assert result.model.precision > result.fallback.precision


# -------------------------------------------------------------- the arithmetic
def test_expectancy_is_per_approved_trade_not_per_row():
    """A filter approving three trades a year with a huge average is a different
    animal from one approving three hundred."""
    samples = [
        MetaSample(features={}, primary_prob=0.7, gross_pnl=1000.0, cost=100.0),
        MetaSample(features={}, primary_prob=0.7, gross_pnl=-500.0, cost=100.0),
        MetaSample(features={}, primary_prob=0.7, gross_pnl=-500.0, cost=100.0),
    ]
    result = evaluate_at(samples, [1.0, 0.0, 0.0], 0.5)
    assert result.approved == 1
    assert result.expectancy == pytest.approx(900.0)


def test_a_threshold_nothing_clears_reports_zero_not_a_division_error():
    samples = [MetaSample(features={}, primary_prob=0.7, gross_pnl=100.0, cost=10.0)]
    result = evaluate_at(samples, [0.1], 0.9)
    assert result.approved == 0
    assert result.precision == 0.0 and result.expectancy == 0.0


def test_threshold_for_recall_picks_the_highest_that_still_reaches_it():
    samples = [
        MetaSample(features={}, primary_prob=0.6, gross_pnl=500.0, cost=100.0) for _ in range(5)
    ] + [MetaSample(features={}, primary_prob=0.6, gross_pnl=-500.0, cost=100.0) for _ in range(5)]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    t = threshold_for_recall(samples, scores, 0.6)
    assert evaluate_at(samples, scores, t).recall >= 0.6


# ---------------------------------------------------------------- the fallback
def test_the_fallback_is_a_real_filter_not_an_error_path():
    """ "Take the confident ones" is a legitimate rule, and it is the thing the
    fitted model has to beat."""
    model = MetaLabeller()
    confident = MetaSample(features={}, primary_prob=0.75, gross_pnl=0.0, cost=0.0)
    unsure = MetaSample(features={}, primary_prob=0.51, gross_pnl=0.0, cost=0.0)
    assert model._fallback_score(confident) > model._fallback_score(unsure)


def test_the_fallback_uses_the_same_scale_as_a_fitted_score():
    """Otherwise switching to the fallback silently changes what the threshold
    means."""
    model = MetaLabeller()
    for prob in (0.0, 0.3, 0.5, 0.7, 1.0):
        score = model._fallback_score(
            MetaSample(features={}, primary_prob=prob, gross_pnl=0.0, cost=0.0)
        )
        assert 0.0 <= score <= 1.0


def test_a_single_outcome_class_falls_back_rather_than_pretending():
    """Every call paid, or none did. There is nothing to learn, and a model that
    fitted anyway would approve or refuse everything for no reason."""
    samples = [
        MetaSample(features={"a": 1.0}, primary_prob=0.6, gross_pnl=1000.0, cost=10.0)
        for _ in range(300)
    ]
    model = MetaLabeller().fit(samples)
    assert model.using_fallback
    assert model.booster is None
    assert model.report is not None
    assert any("nothing to learn" in n for n in model.report.notes)


def test_explain_says_which_path_produced_the_score():
    model = MetaLabeller()
    text = model.explain(MetaSample(features={}, primary_prob=0.8, gross_pnl=0.0, cost=0.0))
    assert "fallback" in text
    assert "TAKE" in text or "SKIP" in text


def test_refuses_to_fit_on_too_few_rows():
    with pytest.raises(ValueError, match="not enough to fit"):
        MetaLabeller().fit(world(50, 12))


# ---------------------------------------------------------------- the asymmetry
def test_a_costlier_mistake_means_fewer_trades():
    """The direction that matters, and the one that was silently backwards when
    the asymmetry lived in class weighting instead of the threshold: raising the
    weight from 1 to 4 moved the approval rate from 11% to 30%."""
    samples = world(2000, 13)
    lenient = MetaLabeller(loss_ratio=1.0).fit(samples)
    strict = MetaLabeller(loss_ratio=6.0).fit(samples)
    assert lenient.report is not None and strict.report is not None
    assert strict.report.approval_rate < lenient.report.approval_rate, (
        f"loss_ratio 6 approved {strict.report.approval_rate:.0%} against "
        f"{lenient.report.approval_rate:.0%} at 1"
    )


def test_the_threshold_is_derived_from_the_loss_ratio():
    """A threshold picked by hand encodes an asymmetry nobody wrote down, and it
    is then impossible to tell whether a later change reflects new information
    or someone wanting more trades."""
    assert threshold_from_loss_ratio(1.0) == pytest.approx(0.5)
    assert threshold_from_loss_ratio(2.0) == pytest.approx(2 / 3)
    assert threshold_from_loss_ratio(9.0) == pytest.approx(0.9)


def test_an_impossible_loss_ratio_is_refused():
    with pytest.raises(ValueError, match="loss_ratio must be positive"):
        threshold_from_loss_ratio(0.0)


def test_the_default_threshold_is_above_a_coin_flip():
    from libs.ms_models.meta import DEFAULT_APPROVAL_THRESHOLD

    assert DEFAULT_APPROVAL_THRESHOLD > 0.5


def test_an_explicit_threshold_still_wins():
    model = MetaLabeller(loss_ratio=2.0, threshold=0.9)
    assert model.threshold == 0.9


def test_calibration_is_not_fighting_the_training_distribution():
    """The scores a calibrated model produces should track the observed base
    rate. If class weighting were still distorting the fit, this would drift."""
    samples = world(2000, 14)
    model = MetaLabeller().fit(samples)
    assert model.report is not None and model.report.calibrated
    mean_score = sum(model.score(s) for s in samples) / len(samples)
    assert mean_score == pytest.approx(
        model.report.base_rate, abs=0.08
    ), f"mean score {mean_score:.3f} against a base rate of {model.report.base_rate:.3f}"
