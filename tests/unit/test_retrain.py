"""MA2 + M5A - the registry, and the one rule retraining lives by.

promote only if it beats the incumbent on the last four weeks out-of-sample
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from libs.ms_models.cv import ExperimentLog
from libs.ms_ops.registry import (
    MIN_SHADOW_SESSIONS,
    ModelVersion,
    Registry,
    RegistryError,
    Stage,
    production_model,
    stamp,
    versions_in,
)
from libs.ms_ops.retrain import (
    MIN_IMPROVEMENT,
    MIN_OOS_PREDICTIONS,
    RetrainKind,
    Scored,
    due,
    evaluate_promotion,
    oos_window,
    retrain,
)

TODAY = date(2026, 3, 4)


def model(version: str, *, through: date, stage: Stage = Stage.STAGING, sessions: int = 0):
    return ModelVersion(
        version=version,
        trained_on=through,
        trained_through=through,
        stage=stage,
        shadow_sessions=sessions,
    )


def ready_challenger(version: str = "v2", *, through: date | None = None) -> ModelVersion:
    return model(
        version,
        through=through or (TODAY - timedelta(days=35)),
        stage=Stage.SHADOW,
        sessions=MIN_SHADOW_SESSIONS,
    )


def score(m: ModelVersion, *, log_loss: float, n: int = 500) -> Scored:
    start, end = oos_window(m)
    return Scored(
        version=m.version, window_start=start, window_end=end, n=n, log_loss=log_loss, brier=0.2
    )


# --------------------------------------------------------------- the registry
def test_only_one_model_can_be_live():
    """Two live models means predictions come from whichever one the caller
    happened to load, and attribution is meaningless."""
    r = Registry()
    a = r.register(model("v1", through=TODAY))
    b = r.register(model("v2", through=TODAY))
    a.stage = Stage.PRODUCTION
    b.stage = Stage.PRODUCTION

    with pytest.raises(RegistryError, match="2 models are in production"):
        r.production()


def test_nothing_in_production_is_a_state_not_an_error():
    """Day one and after an emergency archive are both real, and a caller forced
    to catch an exception will eventually catch it and continue."""
    assert Registry().production() is None


def test_asking_for_the_production_model_refuses_a_fallback():
    """A system trading on a default model is one whose operator does not know
    what it is trading on."""
    with pytest.raises(RegistryError, match="Refusing rather than loading a default"):
        production_model(Registry())


def test_a_duplicate_version_is_refused():
    r = Registry()
    r.register(model("v1", through=TODAY))
    with pytest.raises(RegistryError, match="already registered"):
        r.register(model("v1", through=TODAY))


def test_shadow_cannot_be_restarted():
    """Restarting it would reset a session count that is supposed to be
    evidence."""
    r = Registry()
    r.register(model("v1", through=TODAY))
    r.start_shadow("v1")
    r.record_shadow_session("v1")
    with pytest.raises(RegistryError, match="not STAGING"):
        r.start_shadow("v1")


def test_promotion_archives_the_incumbent():
    r = Registry()
    old = r.register(model("v1", through=TODAY, stage=Stage.PRODUCTION))
    new = r.register(ready_challenger())
    check = evaluate_promotion(
        new,
        old,
        challenger_score=score(new, log_loss=0.60),
        incumbent_score=Scored("v1", *oos_window(new), 500, 0.65, 0.2),
        today=TODAY,
    )
    r.promote("v2", check, today=TODAY)

    assert old.stage is Stage.ARCHIVED and old.archived_on == TODAY
    assert r.production() is not None and r.production().version == "v2"


def test_promotion_needs_a_check_for_that_model():
    """Promoting on someone else's evidence is how the wrong model goes live."""
    r = Registry()
    a = r.register(ready_challenger("v2"))
    r.register(ready_challenger("v3"))
    check = evaluate_promotion(
        a, None, challenger_score=score(a, log_loss=0.6), incumbent_score=None, today=TODAY
    )
    with pytest.raises(RegistryError, match="check is for"):
        r.promote("v3", check, today=TODAY)


def test_archiving_needs_a_reason():
    """The reason a model was pulled is the most useful thing about it later."""
    r = Registry()
    r.register(model("v1", through=TODAY))
    with pytest.raises(RegistryError, match="needs a reason"):
        r.archive("v1", today=TODAY, reason="  ")


def test_every_prediction_carries_its_version():
    """A week that mixes two versions and reports one hit rate is reporting a
    number about nothing, and the mixing is invisible."""
    p = stamp("v2", {"p_up": 0.6})
    assert p["model_version"] == "v2"
    with pytest.raises(ValueError, match="cannot be attributed"):
        stamp("", {"p_up": 0.6})


def test_more_than_one_version_in_a_batch_is_visible():
    batch = [stamp("v1", {}), stamp("v1", {}), stamp("v2", {})]
    assert versions_in(batch) == {"v1", "v2"}


# ----------------------------------------------------------- the shadow gate
def test_a_model_cannot_go_live_without_shadowing():
    """Shadow is the only stage producing evidence nobody chose: the challenger
    predicts on live data and trades nothing."""
    c = model("v2", through=TODAY - timedelta(days=35), stage=Stage.STAGING)
    check = evaluate_promotion(
        c, None, challenger_score=score(c, log_loss=0.5), incumbent_score=None, today=TODAY
    )
    assert not check.approved
    assert any("not SHADOW" in b for b in check.blockers)


def test_a_short_shadow_is_not_a_shadow():
    c = model("v2", through=TODAY - timedelta(days=35), stage=Stage.SHADOW, sessions=5)
    check = evaluate_promotion(
        c, None, challenger_score=score(c, log_loss=0.5), incumbent_score=None, today=TODAY
    )
    assert any("shadow sessions" in b for b in check.blockers)


# ------------------------------------------------------------ the OOS window
def test_the_window_starts_after_the_challengers_training_boundary():
    """Defining it from today would let a model trained through last Friday be
    scored on last Thursday, and the win would be large, clean and entirely
    fake."""
    c = model("v2", through=date(2026, 2, 1))
    start, end = oos_window(c)
    assert start == date(2026, 2, 2)
    assert (end - start).days == 28


def test_a_window_overlapping_training_data_is_refused():
    c = ready_challenger(through=date(2026, 2, 1))
    overlapping = Scored("v2", date(2026, 1, 20), date(2026, 2, 17), 500, 0.4, 0.15)
    check = evaluate_promotion(
        c, None, challenger_score=overlapping, incumbent_score=None, today=TODAY
    )
    assert any("not out-of-sample" in b for b in check.blockers)


def test_too_few_predictions_is_noise_whatever_it_says():
    c = ready_challenger()
    thin = score(c, log_loss=0.3, n=MIN_OOS_PREDICTIONS - 1)
    check = evaluate_promotion(c, None, challenger_score=thin, incumbent_score=None, today=TODAY)
    assert any("is noise" in b for b in check.blockers)


# ---------------------------------------------------- against the incumbent
def test_a_challenger_compared_against_nothing_wins_every_time():
    r = Registry()
    incumbent = r.register(model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION))
    c = ready_challenger()
    check = evaluate_promotion(
        c, incumbent, challenger_score=score(c, log_loss=0.4), incumbent_score=None, today=TODAY
    )
    assert any("compared against nothing" in b for b in check.blockers)


def test_both_must_be_scored_on_the_same_window():
    """Otherwise the comparison is between a model and a different question."""
    c = ready_challenger()
    incumbent = model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION)
    elsewhere = Scored("v1", date(2025, 1, 1), date(2025, 1, 29), 500, 0.7, 0.2)
    check = evaluate_promotion(
        c,
        incumbent,
        challenger_score=score(c, log_loss=0.4),
        incumbent_score=elsewhere,
        today=TODAY,
    )
    assert any("different windows" in b for b in check.blockers)


def test_a_marginal_win_is_a_coin_toss():
    """A challenger ahead by a thousandth has won a coin toss, and promoting on
    it means promoting every week for no reason."""
    c = ready_challenger()
    incumbent = model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION)
    start, end = oos_window(c)
    check = evaluate_promotion(
        c,
        incumbent,
        challenger_score=score(c, log_loss=0.6000),
        incumbent_score=Scored("v1", start, end, 500, 0.6001, 0.2),
        today=TODAY,
    )
    assert not check.approved
    assert any("coin toss" in b for b in check.blockers)


def test_a_real_win_is_approved():
    c = ready_challenger()
    incumbent = model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION)
    start, end = oos_window(c)
    check = evaluate_promotion(
        c,
        incumbent,
        challenger_score=score(c, log_loss=0.58),
        incumbent_score=Scored("v1", start, end, 500, 0.65, 0.2),
        today=TODAY,
    )
    assert check.approved, check.blockers
    assert any("better than the incumbent" in e for e in check.evidence)
    assert MIN_IMPROVEMENT < 0.65 - 0.58


def test_every_blocker_is_reported_not_just_the_first():
    """A challenger that fails on three counts and is fixed for one comes back
    and fails on the other two, and each round trip is a week."""
    c = model("v2", through=TODAY - timedelta(days=35), stage=Stage.STAGING, sessions=0)
    thin = score(c, log_loss=0.9, n=10)
    check = evaluate_promotion(c, None, challenger_score=thin, incumbent_score=None, today=TODAY)
    assert len(check.blockers) >= 2


# ------------------------------------------------------------------- the run
def test_a_retrain_is_logged_whether_or_not_it_promotes(tmp_path):
    """A trial count that only includes the winners is not a trial count, and
    every deflated statistic downstream reads it."""
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    r = Registry()
    incumbent = r.register(model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION))
    c = r.register(ready_challenger())
    start, end = oos_window(c)

    result = retrain(
        r,
        kind=RetrainKind.INCREMENTAL,
        today=TODAY,
        challenger=c,
        challenger_score=score(c, log_loss=0.66),  # worse than the incumbent
        incumbent_score=Scored("v1", start, end, 500, 0.65, 0.2),
        experiment_log=log,
    )
    assert not result.promoted
    assert log.trial_count("retrain_") == 1
    assert incumbent.is_live, "a losing challenger replaced the incumbent"


def test_promotion_needs_apply():
    """A pipeline that promotes on every scheduled run changes what is trading
    without anyone deciding to."""
    r = Registry()
    r.register(model("v1", through=TODAY - timedelta(days=90), stage=Stage.PRODUCTION))
    c = r.register(ready_challenger())
    start, end = oos_window(c)

    dry = retrain(
        r,
        kind=RetrainKind.INCREMENTAL,
        today=TODAY,
        challenger=c,
        challenger_score=score(c, log_loss=0.58),
        incumbent_score=Scored("v1", start, end, 500, 0.65, 0.2),
    )
    assert dry.check is not None and dry.check.approved
    assert not dry.promoted
    assert r.production().version == "v1"

    applied = retrain(
        r,
        kind=RetrainKind.INCREMENTAL,
        today=TODAY,
        challenger=c,
        challenger_score=score(c, log_loss=0.58),
        incumbent_score=Scored("v1", start, end, 500, 0.65, 0.2),
        apply=True,
    )
    assert applied.promoted
    assert r.production().version == "v2"


def test_a_run_with_no_challenger_is_not_a_failure():
    r = Registry()
    result = retrain(
        r,
        kind=RetrainKind.FULL,
        today=TODAY,
        challenger=None,
        challenger_score=None,
        incumbent_score=None,
    )
    assert not result.promoted
    assert "nothing to evaluate" in result.notes


def test_the_report_names_the_trial_count(tmp_path):
    """Fifty-two retrains a year is fifty-two attempts at beating the
    incumbent.

    The log goes to tmp_path, and that is not incidental. A bare
    ``ExperimentLog()`` defaults to the repository's own
    ``research/experiments.jsonl`` - the hash-chained record whose only job is
    honest trial counting - so a test written that way appends a fabricated
    trial on every run, forever, and every deflated statistic downstream reads
    the inflated count. This test did exactly that until it was caught by a
    ``git status`` showing the log modified after a suite run.
    """
    r = Registry()
    c = r.register(ready_challenger())
    result = retrain(
        r,
        kind=RetrainKind.INCREMENTAL,
        today=TODAY,
        challenger=c,
        challenger_score=score(c, log_loss=0.58),
        incumbent_score=None,
        experiment_log=ExperimentLog(tmp_path / "experiments.jsonl"),
    )
    assert "has not necessarily found anything" in result.render()


# --------------------------------------------------------------------- timing
def test_incremental_is_weekly_and_full_is_quarterly():
    assert due(RetrainKind.INCREMENTAL, today=TODAY, last_run=TODAY - timedelta(days=7))
    assert not due(RetrainKind.INCREMENTAL, today=TODAY, last_run=TODAY - timedelta(days=3))
    assert due(RetrainKind.FULL, today=TODAY, last_run=TODAY - timedelta(days=90))
    assert not due(RetrainKind.FULL, today=TODAY, last_run=TODAY - timedelta(days=30))


def test_a_first_run_is_always_due():
    assert due(RetrainKind.FULL, today=TODAY, last_run=None)


def test_the_retrain_does_not_read_the_wall_clock(tmp_path):
    """R6, and the second time this guard has caught me in one session. A
    retrain replayed over a past quarter would otherwise stamp its trial with
    today's date, and the experiment log would show fifty-two attempts that all
    happened on one afternoon."""
    from datetime import UTC, datetime

    from libs.ms_core.clock import SimClock

    frozen = SimClock(datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    r = Registry()
    c = r.register(ready_challenger())

    retrain(
        r,
        kind=RetrainKind.INCREMENTAL,
        today=TODAY,
        challenger=c,
        challenger_score=score(c, log_loss=0.58),
        incumbent_score=None,
        experiment_log=log,
        clock=frozen,
    )
    assert log.all()[-1].ts.startswith("2026-01-15")


def test_no_module_in_ms_ops_reads_the_wall_clock():
    import pathlib
    import re

    offenders = []
    for path in pathlib.Path("libs/ms_ops").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"datetime\.now\(|time\.time\(", line) and not line.strip().startswith(
                "#"
            ):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, "wall-clock access in ms_ops:\n  " + "\n  ".join(offenders)
