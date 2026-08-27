"""M93 - the audit spine.

Every test here is about one question: can this day's record be believed?
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.ms_risk.ledger import (
    LedgerEntry,
    LedgerError,
    PaperLedger,
    Resolution,
    resolve,
    to_rows,
    verify,
)

T0 = datetime(2026, 3, 4, 6, 30, tzinfo=UTC)


def entry(**over) -> LedgerEntry:
    base = {
        "entry_id": "e-0001",
        "reco_id": "reco-0001",
        "ts_predicted": T0,
        "symbol": "NIFTY24000CE",
        "event_class": "geo.military",
        "horizon_minutes": 15,
        "p_up": 0.62,
        "p_lower": 0.55,
        "entry_price": 100.0,
        "lots": 2,
        "is_long": True,
        "mode": "PAPER",
    }
    return LedgerEntry(**{**base, **over})


@pytest.fixture
def ledger(tmp_path) -> PaperLedger:
    return PaperLedger(tmp_path)


# --------------------------------------------------------- one row, not a join
def test_prediction_and_outcome_live_in_the_same_row(ledger):
    """Split across two tables joined by an id, a prediction whose outcome was
    never written looks identical to one whose outcome row was lost, and both
    look identical to one that was never meant to have an outcome."""
    e = entry()
    ledger.append(e)
    resolve(ledger, e, now=e.horizon_ends, exit_price=105.0)

    [row] = ledger.entries(T0)
    assert row.p_up == 0.62 and row.entry_price == 100.0  # the prediction
    assert row.exit_price == 105.0 and row.gross_pnl == 10.0  # the outcome
    assert row.resolution is Resolution.RESOLVED


def test_unresolved_is_a_state_the_row_carries_with_a_reason(ledger):
    e = entry()
    ledger.append(e)
    resolve(
        ledger, e, now=e.horizon_ends, exit_price=None, reason="instrument delisted mid-session"
    )

    [row] = ledger.entries(T0)
    assert row.resolution is Resolution.UNRESOLVABLE
    assert "delisted" in row.unresolved_reason
    assert row.exit_price is None


def test_an_unresolvable_prediction_must_say_why(ledger):
    """A row with no outcome and no explanation cannot be told apart from one
    that was lost, and that ambiguity makes every later report unauditable."""
    e = entry()
    ledger.append(e)
    with pytest.raises(LedgerError, match="documented reason"):
        resolve(ledger, e, now=e.horizon_ends, exit_price=None)
    with pytest.raises(LedgerError, match="documented reason"):
        resolve(ledger, e, now=e.horizon_ends, exit_price=None, reason="   ")


# ------------------------------------------------------------ never deleted
def test_resolution_appends_rather_than_editing(ledger):
    """Editing in place means the file no longer shows what was believed at
    prediction time, which is the one thing it exists to preserve."""
    e = entry()
    ledger.append(e)
    resolve(ledger, e, now=e.horizon_ends, exit_price=105.0)

    history = ledger.history(T0, "e-0001")
    assert len(history) == 2
    assert history[0].resolution is Resolution.PENDING
    assert history[1].resolution is Resolution.RESOLVED
    assert history[0].exit_price is None


def test_the_reader_takes_the_latest_and_the_earlier_stays_on_disk(ledger):
    e = entry()
    ledger.append(e)
    resolve(ledger, e, now=e.horizon_ends, exit_price=105.0)

    assert len(ledger.entries(T0)) == 1, "the superseded record leaked into the current view"
    assert len(ledger.history(T0, "e-0001")) == 2, "the audit trail was lost"


def test_resolving_twice_is_refused(ledger):
    """A corrected outcome is a new entry superseding this one, not an edit."""
    e = entry()
    ledger.append(e)
    resolved = resolve(ledger, e, now=e.horizon_ends, exit_price=105.0)
    with pytest.raises(LedgerError, match="already RESOLVED"):
        resolve(ledger, resolved, now=e.horizon_ends, exit_price=110.0)


# --------------------------------------------------------------- the horizon
def test_resolving_early_is_refused(ledger):
    """An outcome resolved before its horizon measures a different quantity than
    the one predicted - and the number would be perfectly plausible."""
    e = entry(horizon_minutes=15)
    ledger.append(e)
    with pytest.raises(LedgerError, match="measures a different quantity"):
        resolve(ledger, e, now=T0 + timedelta(minutes=14), exit_price=105.0)


def test_resolving_exactly_at_the_horizon_is_allowed(ledger):
    e = entry(horizon_minutes=15)
    ledger.append(e)
    assert (
        resolve(ledger, e, now=e.horizon_ends, exit_price=105.0).resolution is Resolution.RESOLVED
    )


def test_a_late_resolution_is_visible_rather_than_invisible(ledger):
    """A systematically late resolution is a systematically different
    measurement, and nobody notices unless it is recorded."""
    e = entry()
    ledger.append(e)
    late = e.horizon_ends + timedelta(minutes=20)
    resolved = resolve(ledger, e, now=late, exit_price=105.0)

    assert resolved.resolution_lag == timedelta(minutes=20)
    audit = verify(ledger, T0, now=late)
    assert audit.late_resolutions
    assert "systematically different" in audit.render()


# ----------------------------------------------------------- pending is not wrong
def test_a_pending_prediction_is_not_a_wrong_one(ledger):
    """Defaulting it to False makes every hit rate drift down as the day goes
    on, which looks like a model decaying."""
    e = entry()
    assert e.was_right is None
    assert e.net_pnl is None


def test_direction_is_scored_against_the_move_not_the_pnl(ledger):
    e = entry(p_up=0.62)
    ledger.append(e)
    right = resolve(ledger, e, now=e.horizon_ends, exit_price=105.0)
    assert right.was_right is True

    e2 = entry(entry_id="e-0002", p_up=0.62)
    ledger.append(e2)
    wrong = resolve(ledger, e2, now=e2.horizon_ends, exit_price=95.0)
    assert wrong.was_right is False


def test_a_short_makes_money_when_price_falls(ledger):
    e = entry(is_long=False, p_up=0.3)
    ledger.append(e)
    resolved = resolve(ledger, e, now=e.horizon_ends, exit_price=90.0)
    assert resolved.gross_pnl == pytest.approx(20.0)
    assert resolved.was_right is True


def test_costs_are_subtracted_from_gross(ledger):
    e = entry()
    ledger.append(e)
    resolved = resolve(ledger, e, now=e.horizon_ends, exit_price=105.0, costs=4.0)
    assert resolved.gross_pnl == 10.0
    assert resolved.net_pnl == 6.0


# ------------------------------------------------------------------- the audit
def test_verify_separates_pending_from_overdue(ledger):
    """A count of "resolved" alone cannot tell a healthy day from one where half
    the outcomes were quietly never written."""
    still_open = entry(entry_id="open", ts_predicted=T0, horizon_minutes=60)
    overdue = entry(entry_id="overdue", ts_predicted=T0, horizon_minutes=5)
    ledger.append(still_open)
    ledger.append(overdue)

    audit = verify(ledger, T0, now=T0 + timedelta(minutes=30))
    assert audit.pending == 2
    assert audit.overdue == [
        f"overdue (geo.military) horizon ended {overdue.horizon_ends.isoformat()}"
    ]
    assert not audit.complete
    assert "NOT auditable" in audit.render()


def test_a_day_where_everything_is_accounted_for_is_auditable(ledger):
    resolved = entry(entry_id="a")
    unresolvable = entry(entry_id="b")
    ledger.append(resolved)
    ledger.append(unresolvable)
    resolve(ledger, resolved, now=resolved.horizon_ends, exit_price=105.0)
    resolve(ledger, unresolvable, now=unresolvable.horizon_ends, exit_price=None, reason="feed gap")

    audit = verify(ledger, T0, now=T0 + timedelta(hours=2))
    assert audit.complete
    assert audit.accounted_for == 2
    assert "auditable" in audit.render()


def test_a_prediction_still_inside_its_horizon_is_not_a_gap(ledger):
    ledger.append(entry(horizon_minutes=60))
    audit = verify(ledger, T0, now=T0 + timedelta(minutes=10))
    assert audit.pending == 1
    assert audit.complete, "an open horizon was counted as a missing outcome"


# ---------------------------------------------------------------- projection
def test_rows_carry_the_reason_a_prediction_had_no_outcome(ledger):
    e = entry()
    ledger.append(e)
    resolve(ledger, e, now=e.horizon_ends, exit_price=None, reason="chain went stale")

    [row] = to_rows(ledger.entries(T0))
    assert row["resolution"] == "UNRESOLVABLE"
    assert row["unresolved_reason"] == "chain went stale"
    assert row["net_pnl"] is None
    assert row["was_right"] is None


def test_rows_compute_nothing_the_file_does_not_contain():
    """The file is the source of truth and this is a projection of it. Anything
    derived here is a second place for the two to disagree."""
    e = entry()
    [row] = to_rows([e])
    assert row["net_pnl"] is None
    assert row["p_lower"] == 0.55
    assert set(row) >= {"entry_id", "reco_id", "p_up", "p_lower", "resolution"}


def test_an_interval_free_prediction_records_that_honestly():
    [row] = to_rows([entry(p_lower=None)])
    assert row["p_lower"] is None, "a missing interval must not become a number"


# ------------------------------------------------------------------ round trip
def test_an_entry_survives_a_write_and_a_read(ledger):
    e = entry()
    ledger.append(e)
    [back] = ledger.entries(T0)
    assert back == e


def test_days_lists_what_is_recorded(ledger):
    ledger.append(entry())
    ledger.append(entry(entry_id="x", ts_predicted=T0 + timedelta(days=1)))
    assert ledger.days() == ["2026-03-04", "2026-03-05"]


def test_reading_a_day_with_nothing_recorded_is_empty_not_an_error(ledger):
    assert ledger.entries(T0) == []
    assert verify(ledger, T0, now=T0).total == 0
