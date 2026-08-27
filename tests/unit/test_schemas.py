"""M03 verify - schemas and contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from libs.ms_core.schemas import (
    Candle,
    ChainSnapshot,
    ClassifiedEvent,
    Confirmation,
    Leg,
    OptionQuote,
    OptionType,
    Prediction,
    Quantiles,
    RawNewsItem,
    Recommendation,
    Side,
    StructureType,
    Tick,
    Timed,
)

UTC = UTC
T0 = datetime(2026, 8, 26, 4, 0, 0, tzinfo=UTC)


def test_naive_datetime_rejected_everywhere():
    with pytest.raises(ValueError, match="naive"):
        Tick(ts=datetime(2026, 8, 26, 4, 0), instrument_id=1, ltp=100.0)


def test_messages_are_frozen():
    t = Tick(ts=T0, instrument_id=1, ltp=100.0)
    with pytest.raises(ValidationError):
        t.ltp = 200.0  # type: ignore[misc]


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Tick(ts=T0, instrument_id=1, ltp=100.0, whoops=1)  # type: ignore[call-arg]


def test_roundtrip_json():
    c = Candle(ts=T0, instrument_id=1, tf=1, o=100, h=102, l=99, c=101, v=500)
    again = Candle.model_validate(json.loads(c.model_dump_json(by_alias=True)))
    assert again == c


def test_trace_id_present_and_unique():
    a = Tick(ts=T0, instrument_id=1, ltp=1.0)
    b = Tick(ts=T0, instrument_id=1, ltp=1.0)
    assert a.trace_id and a.trace_id != b.trace_id


# ------------------------------------------------------- point-in-time rule
def test_known_at_uses_availability_not_event_time():
    """A feature computed from t_event alone is look-ahead bias in disguise."""
    item = RawNewsItem(
        source_id=1,
        headline="x",
        t_event=T0,
        t_ingest=T0 + timedelta(seconds=30),
        t_available=T0 + timedelta(seconds=45),
    )
    decision = T0 + timedelta(seconds=40)
    assert item.t_event < decision  # it had already happened in the world
    assert not item.known_at(decision)  # but we could not yet act on it
    assert item.known_at(T0 + timedelta(seconds=50))


def test_timed_falls_back_to_ingest_when_available_missing():
    item = RawNewsItem(source_id=1, headline="x", t_event=T0, t_ingest=T0 + timedelta(seconds=5))
    assert not item.known_at(T0 + timedelta(seconds=1))
    assert item.known_at(T0 + timedelta(seconds=6))


def test_all_timed_subclasses_carry_three_timestamps():
    for cls in Timed.__subclasses__():
        fields = set(cls.model_fields)
        assert {"t_event", "t_ingest", "t_available"} <= fields, cls.__name__


# -------------------------------------------------------------- market data
def test_mid_is_none_on_one_sided_book():
    """Valuation uses mid, never LTP - a stale last price manufactures fake signal."""
    assert Tick(ts=T0, instrument_id=1, ltp=100.0).mid is None
    assert Tick(ts=T0, instrument_id=1, ltp=100.0, bid=99.5, ask=100.5).mid == 100.0


def test_candle_consistency():
    good = Candle(ts=T0, instrument_id=1, tf=1, o=100, h=102, l=99, c=101)
    assert good.is_consistent()
    bad = Candle(ts=T0, instrument_id=1, tf=1, o=100, h=102, l=101, c=99)
    assert not bad.is_consistent()


def test_high_below_open_rejected():
    with pytest.raises(ValueError, match="below open"):
        Candle(ts=T0, instrument_id=1, tf=1, o=100, h=99, l=98, c=99)


def test_chain_carries_its_own_spot():
    """Joining a separately-timestamped spot introduces a lag you cannot undo."""
    assert "spot" in ChainSnapshot.model_fields
    snap = ChainSnapshot(
        ts=T0,
        underlying_id=1,
        spot=19_512.0,
        quotes=[
            OptionQuote(expiry=T0, strike=19_500, opt_type=OptionType.PE, bid=78, ask=79),
            OptionQuote(expiry=T0, strike=19_600, opt_type=OptionType.CE, bid=61, ask=63),
        ],
    )
    assert snap.atm_strike() == 19_500


def test_option_spread_pct():
    q = OptionQuote(expiry=T0, strike=19_500, opt_type=OptionType.PE, bid=78, ask=82)
    assert abs(q.spread_pct_of_mid - 4 / 80) < 1e-9


# ------------------------------------------------------------------- events
def test_quantiles_must_be_monotone():
    Quantiles(q05=-0.01, q25=-0.005, q50=0.0, q75=0.004, q95=0.009)
    with pytest.raises(ValueError, match="non-decreasing"):
        Quantiles(q05=0.01, q25=-0.005, q50=0.0, q75=0.004, q95=0.009)


def _event(**kw) -> ClassifiedEvent:
    base = dict(
        news_id="n1",
        event_class="geopolitical.military_action",
        polarity=-0.8,
        severity_raw=8.2,
        confidence=0.9,
        t_event=T0,
        t_ingest=T0,
    )
    base.update(kw)
    return ClassifiedEvent(**base)


def test_single_low_credibility_source_cannot_trade_directionally():
    e = _event(credibility=0.3, confirmation=Confirmation.SINGLE_SOURCE, source_count=1)
    assert not e.may_trade_directionally()


def test_extreme_severity_needs_two_sources():
    """The most extreme headlines are the most likely to be fake."""
    one = _event(
        severity_raw=9.5, credibility=1.0, confirmation=Confirmation.OFFICIAL, source_count=1
    )
    two = _event(
        severity_raw=9.5, credibility=1.0, confirmation=Confirmation.OFFICIAL, source_count=2
    )
    assert not one.may_trade_directionally()
    assert two.may_trade_directionally()


def test_retracted_event_can_never_trade():
    assert not _event(
        confirmation=Confirmation.RETRACTED, credibility=1.0
    ).may_trade_directionally()


def test_conflicting_event_can_never_trade():
    assert not _event(
        confirmation=Confirmation.CONFLICTING, credibility=1.0
    ).may_trade_directionally()


# ----------------------------------------------------------- recommendation
def _prediction() -> Prediction:
    return Prediction(ts=T0, instrument_id=1, horizon_min=15, p_up=0.31)


def test_recommendation_requires_an_explanation():
    """A recommendation without a reason cannot be debugged or attributed."""
    with pytest.raises(ValueError, match="explanation"):
        Recommendation(
            ts_emitted=T0,
            prediction=_prediction(),
            structure=StructureType.BEAR_PUT_DEBIT,
            legs=[Leg(side=Side.BUY, strike=19_500, opt_type=OptionType.PE, qty_lots=1, expiry=T0)],
            explanation=[],
        )


def test_no_trade_recommendation_needs_no_explanation():
    """Silence is the system's most common and most correct output."""
    r = Recommendation(ts_emitted=T0, prediction=_prediction(), structure=StructureType.NONE)
    assert not r.is_actionable


def test_actionable_requires_lots():
    r = Recommendation(
        ts_emitted=T0,
        prediction=_prediction(),
        structure=StructureType.BEAR_PUT_DEBIT,
        legs=[Leg(side=Side.BUY, strike=19_500, opt_type=OptionType.PE, qty_lots=1, expiry=T0)],
        explanation=["escalation, severity 8.2"],
        lots=0,
    )
    assert not r.is_actionable
