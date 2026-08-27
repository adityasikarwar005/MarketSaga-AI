"""M52 verify - the rules baseline and model comparison."""

from __future__ import annotations

import pytest

from libs.ms_models.baseline import (
    BaselineConfig,
    Opinion,
    RulesBaseline,
    climatology,
    compare,
    score,
)

STRONG_UP_EVENT = {
    "severity": 8.0,
    "credibility": 0.9,
    "novelty": 0.8,
    "prior_return": 0.004,
    "n_eff": 20,
}
STRONG_DOWN_EVENT = {**STRONG_UP_EVENT, "prior_return": -0.004}


@pytest.fixture
def model():
    return RulesBaseline()


# --------------------------------------------------------------------- basics
def test_no_input_means_a_coin_flip(model):
    opinion = model.predict({})
    assert opinion.p_up == pytest.approx(0.5)
    assert opinion.direction == 0
    assert "coin flip" in opinion.reasons[0]


def test_it_is_deterministic(model):
    features = {"ret_5m_z": 1.2, "efficiency_ratio_30m": 0.6}
    a = model.predict(features, event=STRONG_UP_EVENT)
    b = model.predict(features, event=STRONG_UP_EVENT)
    assert a.p_up == b.p_up
    assert a.reasons == b.reasons


def test_it_is_never_confident(model):
    """A rules baseline that outputs 0.85 is not confident, it is broken."""
    extreme = {
        "severity": 10.0,
        "credibility": 1.0,
        "novelty": 1.0,
        "prior_return": 0.05,
        "n_eff": 200,
    }
    opinion = model.predict({"ret_5m_z": 2.0, "efficiency_ratio_30m": 1.0}, event=extreme)
    assert 0.5 < opinion.p_up <= 0.5 + model.config.max_edge + 1e-9
    assert opinion.edge <= model.config.max_edge


# ---------------------------------------------------------------- event gates
def test_direction_follows_the_prior(model):
    up = model.predict({}, event=STRONG_UP_EVENT)
    down = model.predict({}, event=STRONG_DOWN_EVENT)
    assert up.direction == 1
    assert down.direction == -1
    assert up.p_up + down.p_up == pytest.approx(1.0)


def test_low_severity_produces_no_direction(model):
    quiet = {**STRONG_UP_EVENT, "severity": 3.0}
    opinion = model.predict({}, event=quiet)
    assert opinion.direction == 0
    assert any("below gate" in r for r in opinion.reasons)


def test_low_credibility_produces_an_alert_not_a_direction(model):
    """The AP-hack rule. One unconfirmed source is never a position."""
    rumour = {**STRONG_UP_EVENT, "credibility": 0.3}
    opinion = model.predict({}, event=rumour)
    assert opinion.direction == 0
    assert any("alert only" in r for r in opinion.reasons)


def test_stale_news_produces_no_direction(model):
    old = {**STRONG_UP_EVENT, "novelty": 0.1}
    opinion = model.predict({}, event=old)
    assert opinion.direction == 0
    assert any("already priced" in r for r in opinion.reasons)


def test_thin_history_produces_no_direction(model):
    """Refusing to guess is what the impact library was built to make possible."""
    rare = {**STRONG_UP_EVENT, "n_eff": 3}
    opinion = model.predict({}, event=rare)
    assert opinion.direction == 0
    assert any("no usable prior" in r for r in opinion.reasons)


def test_missing_prior_produces_no_direction(model):
    unknown = {**STRONG_UP_EVENT, "prior_return": None}
    assert model.predict({}, event=unknown).direction == 0


# ------------------------------------------------------------------ momentum
def test_momentum_alone_moves_the_needle(model):
    up = model.predict({"ret_5m_z": 1.0})
    down = model.predict({"ret_5m_z": -1.0})
    assert up.p_up > 0.5 > down.p_up


def test_momentum_confirms_an_aligned_event(model):
    plain = model.predict({}, event=STRONG_UP_EVENT)
    confirmed = model.predict({"ret_5m_z": 1.0}, event=STRONG_UP_EVENT)
    assert confirmed.p_up > plain.p_up


def test_momentum_contradicting_an_event_reduces_conviction(model):
    plain = model.predict({}, event=STRONG_UP_EVENT)
    contradicted = model.predict({"ret_5m_z": -1.5}, event=STRONG_UP_EVENT)
    assert contradicted.p_up < plain.p_up


# ------------------------------------------------------------------- dampers
def test_a_move_already_made_is_not_chased(model):
    """The market has already priced it. Following now is chasing."""
    fresh = model.predict({"ret_5m_z": 1.5}, event=STRONG_UP_EVENT)
    stale = model.predict({"ret_5m_z": 3.5}, event=STRONG_UP_EVENT)
    assert stale.edge < fresh.edge
    assert any("chasing" in r for r in stale.reasons)


def test_thin_data_damps_conviction(model):
    clean = model.predict({"ret_5m_z": 1.0, "synthetic_share_15m": 0.0}, event=STRONG_UP_EVENT)
    thin = model.predict({"ret_5m_z": 1.0, "synthetic_share_15m": 0.8}, event=STRONG_UP_EVENT)
    assert thin.edge < clean.edge
    assert any("carried forward" in r for r in thin.reasons)


def test_opening_and_closing_ranges_are_damped(model):
    mid = model.predict({"ret_5m_z": 1.0}, event=STRONG_UP_EVENT)
    opening = model.predict({"ret_5m_z": 1.0, "is_opening_range": 1.0}, event=STRONG_UP_EVENT)
    closing = model.predict({"ret_5m_z": 1.0, "is_closing_range": 1.0}, event=STRONG_UP_EVENT)
    assert opening.edge < mid.edge
    assert closing.edge < mid.edge


def test_choppy_market_reduces_trust_in_momentum(model):
    trending = model.predict({"ret_5m_z": 1.0, "efficiency_ratio_30m": 0.8})
    choppy = model.predict({"ret_5m_z": 1.0, "efficiency_ratio_30m": 0.1})
    assert choppy.p_up < trending.p_up
    assert any("choppy" in r for r in choppy.reasons)


def test_stale_chain_is_reported_not_silently_ignored(model):
    opinion = model.predict({"ret_5m_z": 1.0, "chain_age_seconds": 300.0})
    assert any("stale" in r for r in opinion.reasons)


def test_every_prediction_explains_itself(model):
    for features, event in (
        ({}, None),
        ({"ret_5m_z": 1.0}, None),
        ({"ret_5m_z": 1.0}, STRONG_UP_EVENT),
        ({}, {**STRONG_UP_EVENT, "credibility": 0.1}),
    ):
        assert model.predict(features, event=event).reasons


def test_config_is_adjustable(model):
    timid = RulesBaseline(BaselineConfig(max_edge=0.02))
    bold = RulesBaseline(BaselineConfig(max_edge=0.20))
    features, event = {"ret_5m_z": 1.0}, STRONG_UP_EVENT
    assert timid.predict(features, event=event).edge < bold.predict(features, event=event).edge


# -------------------------------------------------------------------- scoring
def test_scorecard_computes_log_loss_and_accuracy():
    probs = [0.6, 0.6, 0.4, 0.4]
    outcomes = [1, 1, -1, -1]
    card = score("m", probs, outcomes)
    assert card.n == 4
    assert card.accuracy == 1.0
    assert card.log_loss < 0.7


def test_timeouts_count_for_log_loss_but_not_accuracy():
    """A model that was confident when nothing happened should still pay."""
    confident = score("m", [0.9, 0.9], [0, 0])
    humble = score("m", [0.5, 0.5], [0, 0])
    assert confident.log_loss > humble.log_loss
    assert confident.directional_n == 0
    assert confident.accuracy == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        score("m", [0.5], [1, -1])


def test_climatology_is_the_base_rate():
    assert climatology([1, 1, 1, -1]) == pytest.approx(0.75)
    assert climatology([0, 0]) == pytest.approx(0.5)
    assert climatology([]) == pytest.approx(0.5)


# ----------------------------------------------------------------- comparison
def test_a_better_model_beats_the_baseline():
    outcomes = [1, -1] * 100
    baseline = [0.5] * 200
    challenger = [0.7 if y > 0 else 0.3 for y in outcomes]
    result = compare(baseline, challenger, outcomes, bootstrap=500)
    assert result.delta_log_loss > 0
    assert result.beats_baseline
    assert "BEATS" in result.render()


def test_an_equal_model_does_not_beat_the_baseline():
    """A challenger whose interval straddles zero has not been shown to beat
    anything, and shipping it on the point estimate is how a project
    accumulates models that do nothing."""
    outcomes = [1, -1] * 100
    same = [0.5] * 200
    result = compare(same, list(same), outcomes, bootstrap=500)
    assert not result.beats_baseline


def test_a_worse_model_does_not_beat_the_baseline():
    outcomes = [1, -1] * 100
    baseline = [0.6 if y > 0 else 0.4 for y in outcomes]
    challenger = [0.4 if y > 0 else 0.6 for y in outcomes]
    result = compare(baseline, challenger, outcomes, bootstrap=500)
    assert result.delta_log_loss < 0
    assert not result.beats_baseline


def test_comparison_without_an_interval_never_claims_a_win():
    outcomes = [1, -1] * 50
    challenger = [0.9 if y > 0 else 0.1 for y in outcomes]
    result = compare([0.5] * 100, challenger, outcomes, bootstrap=0)
    assert result.bootstrap_ci is None
    assert not result.beats_baseline


def test_opinion_edge_and_direction():
    assert Opinion(p_up=0.62).direction == 1
    assert Opinion(p_up=0.38).direction == -1
    assert Opinion(p_up=0.5).direction == 0
    assert Opinion(p_up=0.62).edge == pytest.approx(0.12)
