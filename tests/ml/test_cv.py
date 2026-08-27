"""M51 verify - cross-validation that does not lie.

Every test here exists because the failure it guards against produces a working
pipeline, plausible numbers, and no error message.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from libs.ms_models.cv import (
    Experiment,
    ExperimentLog,
    HoldoutViolation,
    LockedHoldout,
    Sample,
    audit_all,
    audit_split,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    purge_and_embargo,
    purged_kfold,
    samples_from_labels,
    sharpe_ratio,
    walk_forward,
)
from tests.conftest import ist


def samples(n: int = 100, horizon_minutes: int = 15, step_minutes: int = 5) -> list[Sample]:
    return [
        Sample(
            index=i,
            t0=ist(9, 15) + timedelta(minutes=i * step_minutes),
            t_end=ist(9, 15) + timedelta(minutes=i * step_minutes + horizon_minutes),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------- purging
def test_overlapping_train_samples_are_purged():
    """A sample entered at 10:00 with a fifteen-minute horizon is still resolving
    at 10:10. Leaving it in train hands the model the test sample's price path."""
    rows = samples(100)
    test_idx = list(range(40, 50))
    train_idx = [i for i in range(100) if i not in test_idx]
    kept, purged, _ = purge_and_embargo(rows, train_idx, test_idx, embargo=timedelta(0))
    assert purged > 0
    for i in kept:
        assert not any(rows[i].overlaps(rows[t]) for t in test_idx)


def test_embargo_drops_the_buffer_after_the_test_fold():
    """Even non-overlapping samples minutes apart are nearly the same observation."""
    rows = samples(100)
    test_idx = list(range(40, 50))
    train_idx = [i for i in range(100) if i not in test_idx]
    no_embargo, _, e0 = purge_and_embargo(rows, train_idx, test_idx, embargo=timedelta(0))
    with_embargo, _, e1 = purge_and_embargo(rows, train_idx, test_idx, embargo=timedelta(hours=1))
    assert e0 == 0 and e1 > 0
    assert len(with_embargo) < len(no_embargo)


def test_samples_from_the_same_event_never_split():
    """Two rows from one event are the same observation wearing different
    timestamps, however far apart they are."""
    rows = [
        Sample(
            index=i,
            t0=ist(9, 15) + timedelta(hours=i),
            t_end=ist(9, 20) + timedelta(hours=i),
            group="event-A" if i % 2 == 0 else f"event-{i}",
        )
        for i in range(20)
    ]
    test_idx = [0]  # an event-A sample
    train_idx = list(range(1, 20))
    kept, purged, _ = purge_and_embargo(rows, train_idx, test_idx, embargo=timedelta(0))
    assert purged > 0
    assert all(rows[i].group != "event-A" for i in kept)


# ----------------------------------------------------------------- k-fold
def test_purged_kfold_produces_clean_splits():
    rows = samples(200)
    splits = purged_kfold(rows, n_splits=5, embargo=timedelta(minutes=30))
    assert len(splits) == 5
    for audit in audit_all(rows, splits):
        assert audit.ok, audit.leaks[:3]


def test_purged_kfold_covers_every_sample_exactly_once_in_test():
    rows = samples(200)
    splits = purged_kfold(rows, n_splits=4)
    seen: list[int] = []
    for s in splits:
        seen.extend(s.test)
    assert sorted(seen) == list(range(200))


def test_purging_actually_costs_training_data():
    """A purge that removes nothing is a purge that is not working."""
    rows = samples(200)
    splits = purged_kfold(rows, n_splits=5, embargo=timedelta(minutes=30))
    assert sum(s.purged for s in splits) > 0
    assert sum(s.embargoed for s in splits) > 0


def test_too_few_folds_is_rejected():
    with pytest.raises(ValueError, match="at least two folds"):
        purged_kfold(samples(50), n_splits=1)


# ----------------------------------------------------------- walk forward
def test_walk_forward_never_trains_on_the_future():
    """The only shape production has."""
    rows = samples(200)
    for split in walk_forward(rows, n_splits=4, anchored=True):
        latest_train = max(rows[i].t_end for i in split.train)
        earliest_test = min(rows[i].t0 for i in split.test)
        assert latest_train <= earliest_test


def test_anchored_training_set_grows():
    rows = samples(300)
    splits = walk_forward(rows, n_splits=4, anchored=True)
    sizes = [len(s.train) for s in splits]
    assert sizes == sorted(sizes)


def test_rolling_training_set_does_not_grow():
    rows = samples(300)
    splits = walk_forward(rows, n_splits=4, anchored=False)
    sizes = [len(s.train) for s in splits]
    assert max(sizes) - min(sizes) < max(sizes) * 0.5


def test_walk_forward_splits_are_clean():
    rows = samples(200)
    for audit in audit_all(rows, walk_forward(rows, n_splits=4)):
        assert audit.ok, audit.leaks[:3]


# --------------------------------------------------- combinatorial purged CV
def test_cpcv_produces_many_paths():
    """One walk-forward gives one Sharpe, and one number cannot say whether it
    was luck."""
    rows = samples(240)
    splits = combinatorial_purged_cv(rows, n_groups=6, n_test_groups=2)
    assert len(splits) == 15  # C(6,2)
    for audit in audit_all(rows, splits):
        assert audit.ok, audit.leaks[:3]


def test_cpcv_rejects_impossible_configuration():
    with pytest.raises(ValueError, match="smaller than"):
        combinatorial_purged_cv(samples(100), n_groups=3, n_test_groups=3)


# ------------------------------------------------------------- the auditor
def test_auditor_catches_a_broken_purge():
    """A purge implemented with a subtly wrong comparison still produces splits,
    still trains models, and still reports numbers."""
    rows = samples(100)
    from libs.ms_models.cv import Split

    leaky = Split(name="no_purge", train=list(range(0, 60)), test=list(range(50, 60)))
    audit = audit_split(rows, leaky)
    assert not audit.ok
    assert any("overlaps" in x or "share indices" in x for x in audit.leaks)


def test_auditor_passes_a_clean_split():
    rows = samples(100)
    splits = purged_kfold(rows, n_splits=5, embargo=timedelta(hours=1))
    assert all(audit_split(rows, s).ok for s in splits)


# ------------------------------------------------------------ deflated Sharpe
def test_expected_max_sharpe_grows_with_trials():
    """The number nobody computes: try two hundred configurations on random data
    and the best of them looks impressive."""
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) < expected_max_sharpe(200)
    assert expected_max_sharpe(200) > 2.0


def test_deflated_sharpe_falls_as_trials_rise():
    kwargs = {"n_obs": 500, "skew": 0.0, "kurtosis": 3.0}
    honest = deflated_sharpe_ratio(2.0, n_trials=1, **kwargs)
    fished = deflated_sharpe_ratio(2.0, n_trials=500, **kwargs)
    assert honest > 0.9
    assert fished < honest
    assert fished < 0.6, "a Sharpe of 2 after 500 trials should not look convincing"


def test_deflated_sharpe_needs_observations():
    assert deflated_sharpe_ratio(2.0, n_trials=1, n_obs=2) == 0.0


def test_sharpe_of_a_constant_series_is_zero():
    assert sharpe_ratio([0.01] * 50) == 0.0
    assert sharpe_ratio([0.01]) == 0.0


def test_sharpe_has_the_right_sign():
    import random

    rng = random.Random(3)
    good = [rng.gauss(0.002, 0.01) for _ in range(500)]
    bad = [-x for x in good]
    assert sharpe_ratio(good) > 0
    assert sharpe_ratio(bad) < 0


# ------------------------------------------------------------ experiment log
def test_experiment_log_counts_trials(tmp_path):
    """A deflated Sharpe computed with n_trials=1 after fifty attempts is not a
    correction, it is a decoration."""
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    assert log.trial_count() == 0
    for i in range(7):
        log.record(
            Experiment(
                name=f"direction_v{i}",
                ts=ist(18, 0).isoformat(),
                config_hash="abc",
                dataset_version="v1",
                feature_hash="def",
                metrics={"log_loss": 0.69 - i * 0.001},
            )
        )
    assert log.trial_count() == 7
    assert log.trial_count("direction") == 7
    assert log.trial_count("volatility") == 0


def test_experiment_log_is_append_only(tmp_path):
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    e = Experiment(name="a", ts="t", config_hash="c", dataset_version="d", feature_hash="f")
    log.record(e)
    log.record(e)
    assert len(log.all()) == 2


# ------------------------------------------------------------ locked holdout
def test_holdout_is_excluded_from_training_data():
    rows = samples(200, step_minutes=60)
    holdout = LockedHoldout(start=date(2026, 8, 26), end=date(2026, 12, 31))
    trainable = holdout.filter_out(rows)
    assert len(trainable) < len(rows)
    assert all(not holdout.contains(s.t0.date()) for s in trainable)


def test_reading_the_holdout_requires_unlocking(tmp_path):
    rows = samples(50)
    holdout = LockedHoldout(
        start=date(2026, 8, 26), end=date(2026, 12, 31), access_log=tmp_path / "access.jsonl"
    )
    with pytest.raises(HoldoutViolation, match="locked"):
        holdout.unlock(rows, reason="just curious")


def test_unlocking_requires_a_reason(tmp_path):
    holdout = LockedHoldout(
        start=date(2026, 8, 26), end=date(2026, 12, 31), access_log=tmp_path / "access.jsonl"
    )
    with pytest.raises(HoldoutViolation, match="requires a reason"):
        holdout.unlock(samples(10), reason="   ", unlocked=True)


def test_every_access_leaves_a_trace(tmp_path):
    """The log is what tells you months later that the 'out-of-sample' number was
    in fact the fourth look."""
    rows = samples(50)
    holdout = LockedHoldout(
        start=date(2026, 8, 26), end=date(2026, 12, 31), access_log=tmp_path / "access.jsonl"
    )
    assert holdout.access_count() == 0
    got = holdout.unlock(rows, reason="Phase 7 gate evaluation", unlocked=True)
    assert got
    assert holdout.access_count() == 1

    holdout.unlock(rows, reason="second look", unlocked=True)
    assert holdout.access_count() == 2


# ---------------------------------------------------------------- adapters
def test_labels_convert_to_samples(cfg):
    from libs.ms_core.clock import SessionCalendar
    from libs.ms_models.labels import BarrierSpec, build_labels
    from tests.ml.test_labels import flat

    cal = SessionCalendar.from_config(cfg.sessions).synthetic_variant()
    labels = build_labels(flat(200), BarrierSpec(), cal)
    rows = samples_from_labels(labels)
    assert len(rows) == len(labels)
    assert all(r.t_end >= r.t0 for r in rows)

    splits = purged_kfold(rows, n_splits=4, embargo=timedelta(minutes=30))
    for audit in audit_all(rows, splits):
        assert audit.ok, audit.leaks[:3]
