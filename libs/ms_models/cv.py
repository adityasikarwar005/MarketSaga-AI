"""M51 - Cross-validation, without lying to yourself.

Ordinary K-fold on time-series data does not merely overstate performance - it
invents it. Two mechanisms, both invisible in the output:

**Overlap.** A sample entered at 10:00 with a fifteen-minute horizon is still
resolving at 10:10. If that one is in train and the 10:10 sample is in test,
the model has already seen most of the test sample's price path. Purging removes
those.

**Serial correlation.** Even non-overlapping samples minutes apart are nearly
the same observation. An embargo drops a buffer after each test fold.

Then there is the problem no split fixes: **testing many strategies and keeping
the best one.** With enough attempts a Sharpe of 2 appears in pure noise. The
deflated Sharpe ratio corrects for how many attempts there were, which only
works if the attempts were counted - hence the experiment log, which is
append-only and which every training run is expected to write to.

And the locked holdout, which is not a technique but a rule: the most recent
three months, evaluated **once**, at the Phase 7 gate. Every additional look
burns the only honest estimate the project will ever have, so the guard makes
looking difficult and always leaves a trace.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_LOG = REPO_ROOT / "build" / "experiments.jsonl"
HOLDOUT_ACCESS_LOG = REPO_ROOT / "build" / "holdout_access.jsonl"


@dataclass(frozen=True)
class Sample:
    """One labelled observation, and crucially when its outcome finished."""

    index: int
    t0: datetime
    t_end: datetime
    group: str | None = None  # event id: samples from one event never split apart

    def overlaps(self, other: Sample) -> bool:
        return self.t0 < other.t_end and other.t0 < self.t_end


@dataclass
class Split:
    name: str
    train: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    purged: int = 0
    embargoed: int = 0

    @property
    def sizes(self) -> str:
        return f"train={len(self.train):,} test={len(self.test):,}"


def _contiguous_folds(n: int, k: int) -> list[list[int]]:
    """Contiguous folds in time order. Shuffling would destroy the ordering the
    purge depends on."""
    if k < 2:
        raise ValueError("need at least two folds")
    if n < k:
        raise ValueError(f"cannot make {k} folds from {n} samples")
    size, rest = divmod(n, k)
    folds, start = [], 0
    for i in range(k):
        stop = start + size + (1 if i < rest else 0)
        folds.append(list(range(start, stop)))
        start = stop
    return folds


def purge_and_embargo(
    samples: Sequence[Sample],
    train_idx: Iterable[int],
    test_idx: Sequence[int],
    *,
    embargo: timedelta,
) -> tuple[list[int], int, int]:
    """Drop train samples that overlap the test window, then the embargo buffer."""
    if not test_idx:
        return list(train_idx), 0, 0
    test_samples = [samples[i] for i in test_idx]
    test_start = min(s.t0 for s in test_samples)
    test_end = max(s.t_end for s in test_samples)
    test_groups = {s.group for s in test_samples if s.group is not None}

    kept: list[int] = []
    purged = embargoed = 0
    for i in train_idx:
        s = samples[i]
        # A sample from an event that also appears in test is the same
        # observation wearing a different timestamp.
        if s.group is not None and s.group in test_groups:
            purged += 1
            continue
        if s.t0 < test_end and test_start < s.t_end:
            purged += 1
            continue
        if test_end <= s.t0 < test_end + embargo:
            embargoed += 1
            continue
        kept.append(i)
    return kept, purged, embargoed


def purged_kfold(
    samples: Sequence[Sample], *, n_splits: int = 5, embargo: timedelta = timedelta(days=1)
) -> list[Split]:
    ordered = sorted(range(len(samples)), key=lambda i: samples[i].t0)
    folds = _contiguous_folds(len(ordered), n_splits)
    out: list[Split] = []
    for f, fold in enumerate(folds):
        test = [ordered[i] for i in fold]
        raw_train = [ordered[i] for i in range(len(ordered)) if i not in set(fold)]
        train, purged, embargoed = purge_and_embargo(samples, raw_train, test, embargo=embargo)
        out.append(
            Split(
                name=f"purged_kfold_{f}", train=train, test=test, purged=purged, embargoed=embargoed
            )
        )
    return out


def walk_forward(
    samples: Sequence[Sample],
    *,
    n_splits: int = 5,
    anchored: bool = True,
    embargo: timedelta = timedelta(days=1),
) -> list[Split]:
    """Train on the past, test on the future. The only shape production has.

    Report both anchored (expanding window) and rolling: an edge that survives
    one and not the other is telling you something about how fast it decays.
    """
    ordered = sorted(range(len(samples)), key=lambda i: samples[i].t0)
    folds = _contiguous_folds(len(ordered), n_splits + 1)
    out: list[Split] = []
    for f in range(1, len(folds)):
        test = [ordered[i] for i in folds[f]]
        train_pos = folds[0] if not anchored else []
        if anchored:
            for g in range(f):
                train_pos = [*train_pos, *folds[g]]
        else:
            train_pos = folds[f - 1]
        raw_train = [ordered[i] for i in train_pos]
        train, purged, embargoed = purge_and_embargo(samples, raw_train, test, embargo=embargo)
        out.append(
            Split(
                name=f"walk_forward_{'anchored' if anchored else 'rolling'}_{f}",
                train=train,
                test=test,
                purged=purged,
                embargoed=embargoed,
            )
        )
    return out


def combinatorial_purged_cv(
    samples: Sequence[Sample],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: timedelta = timedelta(days=1),
) -> list[Split]:
    """Many backtest paths instead of one.

    A single walk-forward gives one Sharpe, and one number cannot say whether it
    was luck. Testing every combination of held-out groups gives a
    *distribution*, and the spread of that distribution is usually more
    informative than its centre.
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be smaller than n_groups")
    ordered = sorted(range(len(samples)), key=lambda i: samples[i].t0)
    groups = _contiguous_folds(len(ordered), n_groups)

    out: list[Split] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_pos = [i for g in combo for i in groups[g]]
        train_pos = [i for g in range(n_groups) if g not in combo for i in groups[g]]
        test = [ordered[i] for i in test_pos]
        raw_train = [ordered[i] for i in train_pos]
        train, purged, embargoed = purge_and_embargo(samples, raw_train, test, embargo=embargo)
        out.append(
            Split(
                name="cpcv_" + "_".join(str(g) for g in combo),
                train=train,
                test=test,
                purged=purged,
                embargoed=embargoed,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Deflated Sharpe
# --------------------------------------------------------------------------- #
EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF by bisection. Precision here is irrelevant next to the
    uncertainty in the inputs, and it keeps this module dependency-free."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_max_sharpe(n_trials: int, *, variance: float = 1.0) -> float:
    """The best Sharpe you would expect from ``n_trials`` of pure noise.

    This is the number nobody computes. Try two hundred configurations on random
    data and the best of them looks impressive - and that is before any real
    signal is involved.
    """
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(variance)
    a = _norm_ppf(1 - 1 / n_trials)
    b = _norm_ppf(1 - 1 / (n_trials * math.e))
    return sd * ((1 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float | None = None,
) -> float:
    """Probability that the observed Sharpe beats what noise would have produced.

    Below roughly 0.95 the result is not evidence of anything, however good the
    raw number looked. ``n_trials`` must be the honest count of everything tried,
    which is what the experiment log is for.
    """
    if n_obs < 3:
        return 0.0
    bar = expected_max_sharpe(n_trials) if benchmark is None else benchmark
    denom = 1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe * sharpe
    if denom <= 0:
        return 0.0
    z = (sharpe - bar) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def sharpe_ratio(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    return (mean / sd) * math.sqrt(periods_per_year) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# Experiment log
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    name: str
    ts: str
    config_hash: str
    dataset_version: str
    feature_hash: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


class ExperimentLog:
    """Append-only record of everything tried.

    Its only job is to make the trial count honest. A deflated Sharpe computed
    with ``n_trials=1`` after fifty attempts is not a correction, it is a
    decoration - and the temptation to forget the forty-nine that did not work
    is exactly what this file exists to resist.
    """

    def __init__(self, path: Path = EXPERIMENT_LOG) -> None:
        self.path = path

    def record(self, experiment: Experiment) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(experiment), sort_keys=True) + "\n")

    def all(self) -> list[Experiment]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Experiment(**json.loads(line)))
        return out

    def trial_count(self, name_prefix: str = "") -> int:
        return sum(1 for e in self.all() if e.name.startswith(name_prefix))


# --------------------------------------------------------------------------- #
# Locked holdout
# --------------------------------------------------------------------------- #
class HoldoutViolation(RuntimeError):
    """Raised on an attempt to read the locked holdout without unlocking it."""


@dataclass
class LockedHoldout:
    """The most recent slice, reserved for a single evaluation.

    Not a technique - a rule, enforced with friction. Unlocking requires an
    explicit reason, every access is logged, and the log is what tells you months
    later that the "out-of-sample" number was in fact the fourth look.
    """

    start: date
    end: date
    access_log: Path = HOLDOUT_ACCESS_LOG

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def filter_out(self, samples: Sequence[Sample]) -> list[Sample]:
        """Everything a model is allowed to train on."""
        return [s for s in samples if not self.contains(s.t0.date())]

    def access_count(self) -> int:
        if not self.access_log.exists():
            return 0
        return sum(
            1 for line in self.access_log.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    def unlock(
        self, samples: Sequence[Sample], *, reason: str, unlocked: bool = False
    ) -> list[Sample]:
        if not unlocked:
            raise HoldoutViolation(
                f"the holdout ({self.start} .. {self.end}) is locked. It may be evaluated "
                "once, at the Phase 7 gate. Pass unlocked=True with a written reason, and "
                "understand that this is the only honest estimate the project gets."
            )
        if not reason.strip():
            raise HoldoutViolation("unlocking the holdout requires a reason")

        previous = self.access_count()
        self.access_log.parent.mkdir(parents=True, exist_ok=True)
        with self.access_log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "start": self.start.isoformat(),
                        "end": self.end.isoformat(),
                        "reason": reason,
                        "prior_accesses": previous,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if previous:
            log.error(
                "the locked holdout has been read before",
                fields={"prior_accesses": previous, "reason": reason},
            )
        return [s for s in samples if self.contains(s.t0.date())]


# --------------------------------------------------------------------------- #
# Split validation
# --------------------------------------------------------------------------- #
@dataclass
class SplitAudit:
    split: str
    leaks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaks


def audit_split(samples: Sequence[Sample], split: Split) -> SplitAudit:
    """Prove a split actually purged what it claims to have purged.

    A purge implemented with a subtly wrong comparison still produces splits,
    still trains models, and still reports numbers. This is the check that says
    whether those numbers mean anything.
    """
    audit = SplitAudit(split=split.name)
    test_samples = [samples[i] for i in split.test]
    test_groups = {s.group for s in test_samples if s.group is not None}

    for i in split.train:
        s = samples[i]
        if s.group is not None and s.group in test_groups:
            audit.leaks.append(f"train sample {i} shares group {s.group!r} with test")
            continue
        for t in test_samples:
            if s.overlaps(t):
                audit.leaks.append(
                    f"train sample {i} ({s.t0}..{s.t_end}) overlaps test ({t.t0}..{t.t_end})"
                )
                break
    if set(split.train) & set(split.test):
        audit.leaks.append("train and test share indices")
    return audit


def audit_all(samples: Sequence[Sample], splits: Sequence[Split]) -> list[SplitAudit]:
    return [audit_split(samples, s) for s in splits]


def samples_from_labels(labels: Sequence[Any]) -> list[Sample]:
    """Adapt triple-barrier labels (M50) into CV samples."""
    return [
        Sample(index=i, t0=x.t0, t_end=x.exit_ts, group=x.event_id) for i, x in enumerate(labels)
    ]
