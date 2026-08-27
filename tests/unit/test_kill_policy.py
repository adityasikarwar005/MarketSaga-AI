"""M94 - the policy the plan says does more for profitability than any model.

A model that is right on eight event classes and quietly bleeds on the ninth
does not look broken. It looks like a model with a mediocre average, and the
instinct is to improve the model. The cheaper and larger win is to stop trading
the ninth.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from libs.ms_ops.attribution import (
    DISABLED,
    LOSING,
    MIN_TRADES_TO_JUDGE,
    TOO_FEW,
    WORKING,
    ClassReport,
    KillPolicy,
    KillPolicyError,
    attribute,
    brier_breakdown,
    build_weekly_report,
    edge_realisation_slope,
)
from libs.ms_risk.ledger import LedgerEntry, Resolution

TODAY = date(2026, 3, 4)
T0 = datetime(2026, 3, 4, 6, 30, tzinfo=UTC)


def resolved(
    event_class: str,
    net: float,
    *,
    n: int = 1,
    p_up: float = 0.62,
    entry: float = 100.0,
) -> list[LedgerEntry]:
    """`n` resolved predictions in one class, each netting `net`."""
    out = []
    for i in range(n):
        out.append(
            LedgerEntry(
                entry_id=f"{event_class}-{i}",
                reco_id=f"r-{i}",
                ts_predicted=T0 + timedelta(minutes=i),
                symbol="NIFTY24000CE",
                event_class=event_class,
                horizon_minutes=15,
                p_up=p_up,
                p_lower=p_up - 0.06,
                entry_price=entry,
                lots=1,
                is_long=True,
                mode="PAPER",
                resolution=Resolution.RESOLVED,
                ts_resolved=T0 + timedelta(minutes=i + 15),
                exit_price=entry + net,
                gross_pnl=net,
                costs=0.0,
            )
        )
    return out


def report(name: str, *, trades: int, expectancy: float) -> ClassReport:
    return ClassReport(
        event_class=name,
        trades=trades,
        wins=trades // 2,
        gross=expectancy * trades,
        costs=0.0,
        risked=100.0 * trades,
    )


# ---------------------------------------------------------------- auto-disable
def test_auto_disables():
    """Both conditions: enough trades to judge, and negative expectancy."""
    policy = KillPolicy()
    reports = {
        "losing": report("losing", trades=40, expectancy=-12.0),
        "working": report("working", trades=40, expectancy=+8.0),
    }
    newly = policy.evaluate(reports, today=TODAY, model_version="v1")

    assert [d.event_class for d in newly] == ["losing"]
    assert policy.is_disabled("losing")
    assert not policy.is_disabled("working")
    assert reports["losing"].verdict == DISABLED


def test_auto_disables_needs_enough_trades():
    """With five trades a losing streak is what a working edge looks like some
    of the time. A policy that acts on it disables half the book and then looks
    effective, because the survivors got lucky."""
    policy = KillPolicy()
    reports = {"unlucky": report("unlucky", trades=5, expectancy=-50.0)}

    assert policy.evaluate(reports, today=TODAY, model_version="v1") == []
    assert not policy.is_disabled("unlucky")
    assert reports["unlucky"].verdict == TOO_FEW


def test_the_threshold_is_exactly_at_the_boundary():
    policy = KillPolicy()
    just_under = {"a": report("a", trades=MIN_TRADES_TO_JUDGE - 1, expectancy=-5.0)}
    assert policy.evaluate(just_under, today=TODAY, model_version="v1") == []

    at_it = {"b": report("b", trades=MIN_TRADES_TO_JUDGE, expectancy=-5.0)}
    assert len(policy.evaluate(at_it, today=TODAY, model_version="v1")) == 1


def test_break_even_counts_as_losing():
    """Zero expectancy is a class that pays costs to achieve nothing, and the
    costs are already inside the number."""
    policy = KillPolicy()
    reports = {"flat": report("flat", trades=40, expectancy=0.0)}
    assert len(policy.evaluate(reports, today=TODAY, model_version="v1")) == 1


def test_a_class_is_not_disabled_twice():
    policy = KillPolicy()
    reports = {"losing": report("losing", trades=40, expectancy=-12.0)}
    assert len(policy.evaluate(reports, today=TODAY, model_version="v1")) == 1
    assert policy.evaluate(reports, today=TODAY, model_version="v1") == []


# --------------------------------------------------------------- the re-enable
def test_requires_change_to_reenable():
    """A disabled class that comes back after a cooling-off period comes back to
    the same model, the same features and the same market, and loses money
    again. The only thing that elapsed was the operator's memory."""
    policy = KillPolicy()
    policy.evaluate(
        {"losing": report("losing", trades=40, expectancy=-12.0)},
        today=TODAY,
        model_version="v1",
    )

    with pytest.raises(KillPolicyError, match="waiting is not a change"):
        policy.reenable("losing", today=TODAY + timedelta(days=180), model_version="v1")

    assert policy.is_disabled("losing"), "time alone re-enabled a losing class"


def test_a_changed_model_is_a_change():
    policy = KillPolicy()
    policy.evaluate(
        {"losing": report("losing", trades=40, expectancy=-12.0)},
        today=TODAY,
        model_version="v1",
    )
    d = policy.reenable("losing", today=TODAY + timedelta(days=7), model_version="v2")

    assert not policy.is_disabled("losing")
    assert "v1 -> v2" in d.reenabled_because


def test_a_written_justification_is_a_change():
    policy = KillPolicy()
    policy.evaluate(
        {"losing": report("losing", trades=40, expectancy=-12.0)},
        today=TODAY,
        model_version="v1",
    )
    d = policy.reenable(
        "losing",
        today=TODAY + timedelta(days=7),
        model_version="v1",
        justification="Liquidity provider returned to this strike; spreads back to 0.6 from 4.0",
    )
    assert not policy.is_disabled("losing")
    assert "Liquidity provider" in d.reenabled_because


def test_a_token_justification_is_not_a_justification():
    policy = KillPolicy()
    policy.evaluate(
        {"losing": report("losing", trades=40, expectancy=-12.0)},
        today=TODAY,
        model_version="v1",
    )
    with pytest.raises(KillPolicyError, match="at least 30 characters"):
        policy.reenable("losing", today=TODAY, model_version="v1", justification="fixed it")


def test_reenabling_something_that_was_never_disabled_is_an_error():
    with pytest.raises(KillPolicyError, match="not disabled"):
        KillPolicy().reenable("never", today=TODAY, model_version="v1")


def test_the_register_survives_a_quiet_quarter():
    """A policy that recomputes from scratch would re-enable a class the moment
    a quiet quarter pushed its rolling expectancy above zero - the same class,
    the same model, the same reason it lost money."""
    policy = KillPolicy()
    policy.evaluate(
        {"losing": report("losing", trades=40, expectancy=-12.0)},
        today=TODAY,
        model_version="v1",
    )
    # A later week where the class happens to look fine, because it is not trading.
    policy.evaluate(
        {"losing": report("losing", trades=0, expectancy=0.0)},
        today=TODAY + timedelta(days=90),
        model_version="v1",
    )
    assert policy.is_disabled("losing")


# ---------------------------------------------------------------- attribution
def test_only_resolved_entries_are_attributed():
    """Counting unresolvable ones as zeros dilutes every average toward nothing
    and makes a class with missing outcomes look safer than one measured."""
    entries = resolved("geo", 10.0, n=3)
    entries.append(
        LedgerEntry(
            entry_id="pending",
            reco_id="r",
            ts_predicted=T0,
            symbol="X",
            event_class="geo",
            horizon_minutes=15,
            p_up=0.6,
            p_lower=0.55,
            entry_price=100.0,
            lots=1,
            is_long=True,
            mode="PAPER",
        )
    )
    reports = attribute(entries)
    assert reports["geo"].trades == 3


def test_attribution_puts_the_worst_class_first():
    """The number that decides what gets disabled belongs at the top."""
    entries = resolved("good", 20.0, n=5) + resolved("bad", -30.0, n=5)
    assert list(attribute(entries)) == ["bad", "good"]


def test_avg_r_compares_classes_that_size_differently():
    """Expectancy in rupees says a class that trades big is important. avg R
    says whether it is any good."""
    big = attribute(resolved("big", 10.0, n=10, entry=1000.0))["big"]
    small = attribute(resolved("small", 5.0, n=10, entry=50.0))["small"]

    assert big.net > small.net
    assert small.avg_r > big.avg_r, "the smaller, better class was hidden by the bigger one"


def test_a_disabled_class_is_marked_in_the_table():
    entries = resolved("off", -5.0, n=30)
    reports = attribute(entries, disabled=["off"])
    assert reports["off"].verdict == DISABLED


def test_verdicts_read_as_words_not_codes():
    assert str(WORKING) == "working"
    assert str(LOSING) == "losing"
    assert "too few" in str(TOO_FEW)


# ------------------------------------------------------------ edge realisation
def test_edge_realisation_needs_three_classes_to_fit_a_line():
    assert edge_realisation_slope([report("a", trades=10, expectancy=1.0)]) is None


def test_edge_realisation_is_none_when_every_class_predicts_the_same():
    """No spread in the predictor means no slope, and returning zero would read
    as "confidence carries no information" rather than "there was nothing to
    measure"."""
    same = [
        ClassReport(f"c{i}", trades=10, predicted_edge=2.0, realised_edge=float(i))
        for i in range(4)
    ]
    assert edge_realisation_slope(same) is None


def test_edge_realisation_recovers_a_planted_relationship():
    reports = [
        ClassReport(f"c{i}", trades=10, predicted_edge=i * 10.0, realised_edge=i * 5.0)
        for i in range(1, 5)
    ]
    slope = edge_realisation_slope(reports)
    assert slope is not None and slope == pytest.approx(0.5, rel=0.01)


# --------------------------------------------------------------- weekly report
def test_the_weekly_report_names_what_it_disabled():
    policy = KillPolicy()
    entries = resolved("losing", -8.0, n=30) + resolved("working", 12.0, n=30)
    rpt = build_weekly_report(entries, policy, week_ending=TODAY, model_version="v1")

    assert [d.event_class for d in rpt.newly_disabled] == ["losing"]
    text = rpt.render()
    assert "DISABLED THIS WEEK" in text
    assert "Waiting is not a change" in text


def test_the_report_carries_calibration_not_only_pnl():
    """A class can be profitable and badly calibrated, and sizing reads the
    calibration."""
    entries = resolved("a", 5.0, n=30) + resolved("b", -3.0, n=30)
    rpt = build_weekly_report(entries, KillPolicy(), week_ending=TODAY, model_version="v1")
    text = rpt.render()
    assert "reliability slope" in text
    assert "Brier skill" in text
    assert "edge realisation" in text


def test_a_thin_week_reports_n_a_rather_than_a_number():
    """Twelve trades cannot support a reliability slope, and printing one would
    invite acting on it."""
    rpt = build_weekly_report(
        resolved("a", 5.0, n=12), KillPolicy(), week_ending=TODAY, model_version="v1"
    )
    assert rpt.reliability_slope is None
    assert "n/a" in rpt.render()


def test_an_empty_week_does_not_crash():
    rpt = build_weekly_report([], KillPolicy(), week_ending=TODAY, model_version="v1")
    assert rpt.trades == 0
    assert "no class has been disabled" in rpt.render()


def test_the_brier_breakdown_separates_what_calibration_can_fix():
    """High reliability error means the probabilities are mislabelled and
    calibration fixes it. Low resolution means no calibration will help."""
    text = brier_breakdown(resolved("a", 5.0, n=40))
    assert "reliability" in text and "resolution" in text


def test_the_brier_breakdown_refuses_a_thin_sample():
    assert "too few" in brier_breakdown(resolved("a", 5.0, n=5))
