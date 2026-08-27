"""M77 - the record that makes a decision reconstructable six months later."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from libs.ms_risk.gate import KillLevel, RiskDecision, Verdict
from libs.ms_risk.recommendation import (
    Action,
    Exits,
    Forecast,
    IncompleteRecommendation,
    Leg,
    Mode,
    Recommendation,
    RecommendationStore,
    Sizing,
    Structure,
    Trigger,
    assemble,
    store_then_publish,
)

TS = datetime(2026, 8, 26, 3, 50, 41, tzinfo=UTC)


def trigger() -> Trigger:
    return Trigger(
        event_id="evt-1",
        event_class="geopolitical.military_action",
        severity=8.2,
        novelty=0.91,
        credibility=0.88,
        sources=("src_a", "src_b"),
        seconds_since_event=4.1,
    )


def forecast(*, interval: bool = True) -> Forecast:
    return Forecast(
        horizon_min=15,
        p_up=0.31,
        p_up_ci=(0.24, 0.39) if interval else None,
        quantiles={"q05": -0.0142, "q50": -0.0033, "q95": 0.0061},
        sigma_forecast=0.0089,
        implied_move=0.0052,
    )


def structure() -> Structure:
    return Structure(
        type="bear_put_debit_spread",
        legs=(Leg("BUY", 19500, "PE", 1), Leg("SELL", 19300, "PE", 1)),
        net_debit=78.5,
        max_profit=121.5,
        max_loss=78.5,
        breakeven=19421.5,
    )


def allowed() -> RiskDecision:
    return RiskDecision(Verdict.ALLOW, 2, KillLevel.NONE)


def rejected(*reasons: str) -> RiskDecision:
    return RiskDecision(Verdict.REJECT, 0, KillLevel.L2_PAUSE, list(reasons))


def build(**over):
    base = {
        "reco_id": "reco-0001",
        "trace_id": "trace-0001",
        "ts_emitted": TS,
        "mode": Mode.PAPER,
        "trigger": trigger(),
        "forecast": forecast(),
        "explanation": ["Military escalation, severity 8.2, first mention 4.1s ago"],
        "gates_passed": ["credibility>=0.8", "two_sources", "liquidity_ok"],
        "model_versions": {"dir": "ens-v12", "vol": "har-gb-v4"},
        "risk": allowed(),
        "structure": structure(),
        "sizing": Sizing(kelly_frac=0.061, used_frac=0.015, lots=2, capital_at_risk=11775),
        "exits": Exits(target=121.5, stop=-39.0, time_stop="09:36:00"),
    }
    return assemble(**{**base, **over})


# --------------------------------------------------------------------- schema
def test_schema_complete():
    """Every field the blueprint's record specifies, present and populated."""
    rec = build()
    payload = json.loads(rec.to_json())

    for key in (
        "reco_id",
        "trace_id",
        "ts_emitted",
        "mode",
        "action",
        "trigger",
        "forecast",
        "structure",
        "sizing",
        "exits",
        "explanation",
        "gates_passed",
        "model_versions",
    ):
        assert key in payload, f"missing {key}"
        assert payload[key] not in (None, [], {}, ""), f"{key} is present but empty"

    assert payload["trigger"]["event_class"] == "geopolitical.military_action"
    assert payload["forecast"]["p_up_ci"] == [0.24, 0.39]
    assert len(payload["structure"]["legs"]) == 2


def test_mode_has_no_default():
    """PAPER and LIVE differ by whether real money moves. A field that defaults
    to one of them eventually defaults to the wrong one on a path nobody
    tested."""
    fields = {f.name: f for f in dataclasses.fields(Recommendation)}
    assert fields["mode"].default is dataclasses.MISSING
    assert fields["mode"].default_factory is dataclasses.MISSING

    # And assemble() cannot be called without one either - a required field on
    # the record is no protection if the builder in front of it fills one in.
    kwargs = {
        "reco_id": "r",
        "trace_id": "t",
        "ts_emitted": TS,
        "trigger": trigger(),
        "forecast": forecast(),
        "explanation": ["why"],
        "gates_passed": [],
        "model_versions": {"dir": "v1"},
        "risk": rejected("blocked"),
    }
    with pytest.raises(TypeError, match="mode"):
        assemble(**kwargs)  # type: ignore[call-arg]


def test_mode_is_always_in_the_rendered_output():
    for mode in (Mode.PAPER, Mode.LIVE, Mode.DEGRADED):
        assert f"[{mode.value}]" in build(mode=mode).render()


def test_the_json_round_trips():
    rec = build()
    payload = json.loads(rec.to_json())
    assert payload["ts_emitted"] == TS.isoformat()
    assert payload["mode"] == "PAPER"
    assert payload["action"] == "TAKE"


# ---------------------------------------------------------------- explanation
def test_explanation_non_empty():
    """Six months from now this is the only evidence of what was believed."""
    with pytest.raises(IncompleteRecommendation, match="explanation is empty"):
        build(explanation=[])
    with pytest.raises(IncompleteRecommendation, match="explanation is empty"):
        build(explanation=["", "   "])


def test_explanation_keeps_its_order():
    """Ordered, because the reasons build on each other - the class prior only
    means something after the class, and the structure choice only after the
    IV."""
    reasons = [
        "Military escalation, severity 8.2",
        "Historical prior for this class: median -0.31% over 15m (n=23)",
        "Forecast |move| 1.71x implied -> premium buying justified",
        "IV rank 62 -> spread chosen over naked put to cut vega",
    ]
    assert list(build(explanation=reasons).explanation) == reasons


def test_a_missing_interval_is_recorded_in_the_explanation():
    """Not fatal - a model that cannot produce an interval is a real state - but
    the reader has to be able to see which happened, because sizing treats a
    point estimate very differently."""
    rec = build(forecast=forecast(interval=False))
    assert not rec.forecast.has_interval
    assert any("no interval" in e for e in rec.explanation)
    assert "NO INTERVAL" in rec.render()


def test_model_versions_are_required():
    with pytest.raises(IncompleteRecommendation, match="model_versions is empty"):
        build(model_versions={})


# ----------------------------------------------------------------- immutable
def test_immutable():
    """A mutable recommendation is one that can be edited after the fact to
    match what happened, and the edit will be well-intentioned."""
    rec = build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.mode = Mode.LIVE  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.action = Action.SKIP  # type: ignore[misc]


def test_nested_records_are_frozen_too():
    """A frozen shell around mutable parts is not immutable."""
    rec = build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.trigger.severity = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.forecast.p_up = 0.99  # type: ignore[misc]
    assert isinstance(rec.explanation, tuple)
    assert isinstance(rec.gates_passed, tuple)


def test_storing_twice_is_refused(tmp_path):
    """A corrected decision is a new record superseding the old one, not an edit
    to it - otherwise the evidence of what was actually believed is gone."""
    store = RecommendationStore(tmp_path)
    rec = build()
    store.save(rec)
    with pytest.raises(FileExistsError, match="immutable"):
        store.save(rec)


# ------------------------------------------------------------ store-then-emit
def test_stored_before_published(tmp_path):
    order: list[str] = []

    class Watched(RecommendationStore):
        def save(self, rec):
            order.append("stored")
            return super().save(rec)

    store_then_publish(build(), Watched(tmp_path), lambda r: order.append("published"))
    assert order == ["stored", "published"]


def test_nothing_is_published_when_the_store_fails(tmp_path):
    """The other ordering produces recommendations that were acted on and never
    recorded - precisely the population that most needs studying."""
    published: list[str] = []

    class Broken(RecommendationStore):
        def save(self, rec):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store_then_publish(build(), Broken(tmp_path), lambda r: published.append(r.reco_id))
    assert published == []


def test_a_publish_failure_leaves_the_record_behind(tmp_path):
    store = RecommendationStore(tmp_path)
    rec = build()

    def explode(_):
        raise RuntimeError("bus down")

    with pytest.raises(RuntimeError, match="bus down"):
        store_then_publish(rec, store, explode)
    assert store.load(rec.reco_id)["reco_id"] == rec.reco_id


def test_a_stored_record_reconstructs(tmp_path):
    store = RecommendationStore(tmp_path)
    rec = build()
    store.save(rec)

    back = store.load(rec.reco_id)
    assert back["explanation"] == list(rec.explanation)
    assert back["model_versions"] == dict(rec.model_versions)
    assert back["sizing"]["lots"] == 2
    assert back["kill_level"] == "NONE"


def test_loading_something_that_was_never_stored_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        RecommendationStore(tmp_path).load("nope")


# ------------------------------------------------------------- the risk gate
def test_a_rejected_trade_becomes_a_skip_record_not_no_record():
    """A refusal is a decision, and the reasons for it are exactly what you want
    when asking why the system did nothing on a day it should have."""
    rec = build(
        risk=rejected("daily loss limit breached", "2 open positions"),
        sizing=None,
        exits=None,
        structure=None,
    )

    assert rec.action is Action.SKIP
    assert not rec.is_actionable
    assert "daily loss limit breached" in rec.rejected_because
    assert rec.kill_level == "L2_PAUSE"
    assert "BLOCKED" in rec.render()


def test_a_skip_carries_no_structure_or_size():
    """Recording a structure that was never traded invites it being read as one
    that was."""
    rec = build(risk=rejected("blocked"))
    assert rec.structure is None
    assert rec.sizing is None
    assert rec.exits is None


def test_an_allowed_trade_without_sizing_is_refused():
    with pytest.raises(IncompleteRecommendation, match="no sizing"):
        build(sizing=None)


def test_an_allowed_trade_without_exits_is_refused():
    """A position with a target and no stop and no time stop has no plan for
    being wrong, which is the case it spends most of its life in."""
    with pytest.raises(IncompleteRecommendation, match="no exits"):
        build(exits=None)


def test_an_allowed_trade_without_a_structure_is_refused():
    with pytest.raises(IncompleteRecommendation, match="no structure"):
        build(structure=None)


# ----------------------------------------------------------------- the edges
def test_edge_ratio_compares_forecast_against_what_is_already_priced():
    """Below 1.0 there is nothing to buy: the predicted move is smaller than the
    one already in the premium."""
    f = forecast()
    assert f.edge_ratio == pytest.approx(0.0089 / 0.0052)
    assert f.edge_ratio is not None and f.edge_ratio > 1.0


def test_edge_ratio_is_none_when_either_side_is_missing():
    assert Forecast(horizon_min=15, p_up=0.5, implied_move=0.005).edge_ratio is None
    assert Forecast(horizon_min=15, p_up=0.5, sigma_forecast=0.008).edge_ratio is None
    assert (
        Forecast(horizon_min=15, p_up=0.5, sigma_forecast=0.008, implied_move=0.0).edge_ratio
        is None
    )


def test_is_actionable_needs_a_real_size():
    rec = build(sizing=Sizing(kelly_frac=0.0, used_frac=0.0, lots=0, capital_at_risk=0.0))
    assert rec.action is Action.TAKE
    assert not rec.is_actionable, "zero lots is not an actionable recommendation"
