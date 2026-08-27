"""M59 - ensemble, calibration and conformal intervals.

The layer that turns several models into one number a risk gate can act on, and
then says how much to trust it. With M53 and M55 this closes Gate G5.

The design rule here is the opposite of the usual instinct: **keep the combiner
smaller than the things it combines.**

A stacker sees out-of-fold predictions - a few hundred rows, a handful of
columns, every column already the output of a model that was itself fitted. It
is the single easiest place in a system like this to overfit, and the damage is
invisible, because the stacked model is validated on the same folds that
produced its inputs. So the meta-learner is a logistic regression on a small
number of weights, with a prior that pulls it toward equal weighting, and it has
to beat the best single model by a margin before it is used at all.

Out-of-fold means out-of-fold
-----------------------------

The one thing that must be right. A base model's prediction for row *i* has to
come from a fit that never saw row *i*. If any base prediction is in-sample, the
meta-learner learns to trust whichever model memorised hardest, and the whole
ensemble is then a machine for amplifying the most overfitted component. That
failure produces excellent backtests.

``oof_predictions`` builds them by walking forward, and ``OutOfFoldMatrix``
refuses to be constructed with a row whose provenance is not recorded.

Conformal intervals
-------------------

Split conformal on a chronologically later slice: sort the absolute residuals,
take the ``ceil((n+1)(1-alpha))``-th, and use it as a symmetric half-width. The
``n+1`` is not a rounding detail - it is what makes coverage valid at finite
sample size rather than asymptotically, and dropping it produces intervals that
are quietly a few percent narrow.

The guarantee is marginal, not conditional: 90% of intervals cover, but not
necessarily 90% within any particular regime. That distinction matters when
sizing, so ``coverage_by_group`` exists to show where it fails.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from libs.ms_core.obs import get_logger
from libs.ms_models.direction import (
    IsotonicCalibrator,
    expected_calibration_error,
    has_discrimination,
    reliability_curve,
    reliability_slope,
)
from libs.ms_models.trees import log_loss

log = get_logger(__name__)

MIN_OOF_ROWS = 100
MIN_CALIBRATION_ROWS = 60


# --------------------------------------------------------------------------- #
# Out-of-fold predictions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OOFRow:
    """One row's out-of-fold predictions, and which fit produced them."""

    index: int
    fold: int
    predictions: dict[str, float]
    label: int

    def vector(self, names: Sequence[str]) -> list[float]:
        return [self.predictions[n] for n in names]


class OutOfFoldMatrix:
    """Out-of-fold predictions from every base model, with provenance.

    Refuses rows whose fold is not recorded, because "these predictions are
    out-of-sample" is a claim that has to be checkable months later rather than
    remembered.
    """

    def __init__(self, rows: Sequence[OOFRow]) -> None:
        if not rows:
            raise ValueError("no out-of-fold rows")
        names = set(rows[0].predictions)
        for r in rows:
            if set(r.predictions) != names:
                raise ValueError(
                    f"row {r.index} has models {sorted(r.predictions)}, "
                    f"expected {sorted(names)} - a base model that is missing for "
                    "some rows cannot be stacked without deciding what its "
                    "absence means"
                )
            if r.fold < 0:
                raise ValueError(f"row {r.index} has no fold recorded")
        self.rows = list(rows)
        self.names = sorted(names)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> list[int]:
        return [r.label for r in self.rows]

    def column(self, name: str) -> list[float]:
        return [r.predictions[name] for r in self.rows]

    def folds(self) -> set[int]:
        return {r.fold for r in self.rows}


def oof_predictions(
    rows: Sequence[object],
    labels: Sequence[int],
    fit_predict: Callable[[Sequence[object], Sequence[int], Sequence[object]], list[float]],
    *,
    name: str,
    n_folds: int = 4,
    min_train: int = MIN_OOF_ROWS,
) -> list[OOFRow]:
    """Walk-forward out-of-fold predictions from one base model.

    Walk-forward rather than K-fold: a K-fold prediction for an early row is
    made by a model fitted partly on later rows, which is not a thing that can
    happen in production and makes the stacker's training data unlike anything
    it will ever see.
    """
    out: list[OOFRow] = []
    if len(rows) < min_train + n_folds:
        return out
    fold_size = (len(rows) - min_train) // n_folds
    if fold_size < 1:
        return out

    for f in range(n_folds):
        train_end = min_train + f * fold_size
        test_slice = rows[train_end : train_end + fold_size]
        if not test_slice:
            continue
        preds = fit_predict(rows[:train_end], labels[:train_end], test_slice)
        for offset, p in enumerate(preds):
            i = train_end + offset
            out.append(OOFRow(index=i, fold=f, predictions={name: p}, label=labels[i]))
    return out


def merge_oof(*sets: Sequence[OOFRow]) -> list[OOFRow]:
    """Combine per-model out-of-fold rows, keeping only rows every model covers.

    Dropping the partial rows is deliberate. Filling a missing base prediction
    with 0.5 tells the meta-learner that model was undecided, when in fact it
    was never asked - and the weight it then learns is a weight on an absence.
    """
    if not sets:
        return []
    by_index: dict[int, dict[str, float]] = {}
    folds: dict[int, int] = {}
    labels: dict[int, int] = {}
    for rows in sets:
        for r in rows:
            by_index.setdefault(r.index, {}).update(r.predictions)
            folds[r.index] = r.fold
            labels[r.index] = r.label

    expected = len({k for rows in sets for r in rows for k in r.predictions})
    return [
        OOFRow(index=i, fold=folds[i], predictions=preds, label=labels[i])
        for i, preds in sorted(by_index.items())
        if len(preds) == expected
    ]


# --------------------------------------------------------------------------- #
# The meta-learner
# --------------------------------------------------------------------------- #
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    a = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (a[i][n] - sum(a[i][c] * out[c] for c in range(i + 1, n))) / a[i][i]
    return out


def _irls(
    design: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float,
    prior: float,
    iterations: int,
) -> list[float]:
    """Penalised logistic regression by iteratively reweighted least squares.

    Newton's method rather than gradient descent, and not for speed. A fixed
    step size on inputs that are log-odds - which reach +/-4.6 once the base
    probabilities are clamped - diverges: the first version of this reported a
    stacked log loss of 14.9 against 0.72 for its best single member, and with a
    strong penalty it oscillated instead of shrinking, because ``lr * l2``
    exceeded one. Newton has no step size to get wrong.

    The penalty pulls the weights toward ``prior`` - equal weighting - rather
    than toward zero, so a stack with little to go on degrades to the average of
    its members instead of to a constant. The intercept is left unpenalised: it
    carries the base rate, which is not a thing to shrink.
    """
    n_features = len(design[0]) + 1
    rows = [[1.0, *r] for r in design]
    target = [prior] * n_features
    target[0] = 0.0  # the intercept has no prior to shrink toward
    beta = list(target)

    for _ in range(iterations):
        hessian = [[0.0] * n_features for _ in range(n_features)]
        gradient = [0.0] * n_features
        for row, y in zip(rows, labels, strict=True):
            eta = sum(b * x for b, x in zip(beta, row, strict=True))
            p = _sigmoid(eta)
            weight = max(p * (1 - p), 1e-6)
            resid = p - y
            for a in range(n_features):
                gradient[a] += resid * row[a]
                for c in range(n_features):
                    hessian[a][c] += weight * row[a] * row[c]
        for a in range(n_features):
            # The likelihood terms above are summed, not averaged, so the
            # penalty is on the same scale as l2 pseudo-observations. Scaling it
            # by len(rows) instead - as this first did - makes l2=1 worth five
            # hundred observations on a five-hundred-row fit, which pins every
            # weight to the prior and produced a stack measurably worse than its
            # own best member.
            penalty = 0.0 if a == 0 else l2
            gradient[a] += penalty * (beta[a] - target[a])
            hessian[a][a] += penalty + 1e-8

        step = _solve(hessian, gradient)
        if step is None:
            break
        beta = [b - s for b, s in zip(beta, step, strict=True)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return beta


@dataclass
class StackReport:
    rows: int
    models: list[str]
    weights: dict[str, float]
    intercept: float
    best_single: str
    best_single_log_loss: float
    stack_log_loss: float
    used: bool
    reason: str

    def render(self) -> str:
        ws = "  ".join(f"{k}={v:+.3f}" for k, v in sorted(self.weights.items()))
        verdict = "stack used" if self.used else f"stack rejected: {self.reason}"
        return (
            f"  ensemble: {self.rows:,} out-of-fold rows over {len(self.models)} models\n"
            f"    best single {self.best_single} log loss {self.best_single_log_loss:.5f}, "
            f"stack {self.stack_log_loss:.5f}\n"
            f"    weights {ws}  intercept {self.intercept:+.3f}\n"
            f"    {verdict}"
        )


class StackedEnsemble:
    """Logistic regression on the base models' log-odds.

    On the log-odds rather than the probabilities, because averaging
    probabilities pulls everything toward the middle and a model that is
    confidently right gets no more say than one that is barely leaning.

    Three deliberate limits, all aimed at the same thing:

    * one weight per model and one intercept, and nothing else;
    * an L2 prior centred on **equal weights**, not on zero, so that with little
      data the stack degrades to the average of its members rather than to an
      opinionless constant;
    * it has to beat the best single member on the out-of-fold rows by a margin,
      or ``used`` is False and the best single model is what ships.

    A stacker with more parameters than this overfits instantly, and its
    backtest looks wonderful while it does.
    """

    def __init__(self, *, l2: float = 1.0, iterations: int = 25) -> None:
        self.l2 = l2
        self.iterations = iterations
        self.names: list[str] = []
        self.weights: dict[str, float] = {}
        self.intercept = 0.0
        self.report: StackReport | None = None
        self.fallback: str | None = None

    def fit(self, matrix: OutOfFoldMatrix, *, margin: float = 0.002) -> StackedEnsemble:
        if len(matrix) < MIN_OOF_ROWS:
            raise ValueError(
                f"{len(matrix)} out-of-fold rows is not enough to stack " f"(need {MIN_OOF_ROWS})"
            )
        self.names = matrix.names
        labels = matrix.labels
        design = [[_logit(p) for p in r.vector(self.names)] for r in matrix.rows]

        prior = 1.0 / len(self.names)
        beta = _irls(design, labels, l2=self.l2, prior=prior, iterations=self.iterations)
        b, w = beta[0], beta[1:]

        self.weights = dict(zip(self.names, w, strict=True))
        self.intercept = b

        stack_loss = log_loss([self._predict_row(r) for r in design], labels)
        singles = {name: log_loss(matrix.column(name), labels) for name in self.names}
        best_single = min(singles, key=lambda k: singles[k])

        used = stack_loss < singles[best_single] - margin
        self.fallback = None if used else best_single
        self.report = StackReport(
            rows=len(matrix),
            models=list(self.names),
            weights=dict(self.weights),
            intercept=b,
            best_single=best_single,
            best_single_log_loss=singles[best_single],
            stack_log_loss=stack_loss,
            used=used,
            reason=(
                "" if used else f"did not beat {best_single} by {margin:.3f} on out-of-fold rows"
            ),
        )
        log.info(
            "stack fitted",
            fields={
                "rows": len(matrix),
                "models": self.names,
                "used": used,
                "stack_log_loss": round(stack_loss, 6),
                "best_single_log_loss": round(singles[best_single], 6),
            },
        )
        return self

    def _predict_row(self, logits: Sequence[float]) -> float:
        return _sigmoid(
            self.intercept
            + sum(self.weights[n] * x for n, x in zip(self.names, logits, strict=True))
        )

    def predict(self, predictions: dict[str, float]) -> float:
        """Combine one row of base predictions.

        Falls back to the best single member whenever the stack did not earn its
        place, which is the documented behaviour rather than a failure path.
        """
        if self.report is None:
            raise RuntimeError("fit() before predict()")
        missing = [n for n in self.names if n not in predictions]
        if missing:
            raise KeyError(f"missing base predictions for {missing}")
        if self.fallback is not None:
            return predictions[self.fallback]
        return self._predict_row([_logit(predictions[n]) for n in self.names])


# --------------------------------------------------------------------------- #
# Calibration of the combined output
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationReport:
    rows: int
    applied: bool
    reason: str
    log_loss_before: float
    log_loss_after: float
    ece_before: float
    ece_after: float
    slope_before: float | None
    slope_after: float | None
    brier_skill: float

    @property
    def passes_g5_calibration(self) -> bool:
        """Reliability slope in [0.85, 1.15] and Brier skill above 0.02."""
        return (
            self.slope_after is not None
            and 0.85 <= self.slope_after <= 1.15
            and self.brier_skill > 0.02
        )

    def render(self) -> str:
        head = f"  calibration on {self.rows:,} later rows: " + (
            "applied" if self.applied else f"skipped ({self.reason})"
        )
        slope = "n/a" if self.slope_after is None else f"{self.slope_after:.3f}"
        before = "n/a" if self.slope_before is None else f"{self.slope_before:.3f}"
        return (
            f"{head}\n"
            f"    log loss {self.log_loss_before:.5f} -> {self.log_loss_after:.5f}\n"
            f"    ECE {self.ece_before:.4f} -> {self.ece_after:.4f}\n"
            f"    reliability slope {before} -> {slope} (want 0.85-1.15)\n"
            f"    Brier skill {self.brier_skill:+.4f} (want > 0.02)  ->  "
            f"{'PASSES' if self.passes_g5_calibration else 'does not pass'} G5 calibration"
        )


def brier_skill(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Against climatology - the base rate - not against 0.5."""
    if not probs:
        return float("nan")
    base = sum(labels) / len(labels)
    ref = sum((base - y) ** 2 for y in labels) / len(labels)
    got = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)
    return (ref - got) / ref if ref > 0 else 0.0


def calibrate(
    probs: Sequence[float], labels: Sequence[int]
) -> tuple[IsotonicCalibrator | None, CalibrationReport]:
    """Fit isotonic on a chronologically later slice, if there is anything to fit.

    The discrimination gate is the same one the direction model uses, and for
    the same reason: isotonic corrects the levels of a ranking and takes the
    ranking on trust, so applied to a ranking that is noise it manufactures
    confidence out of nothing.
    """
    before_loss = log_loss(probs, labels)
    before_ece = expected_calibration_error(probs, labels)
    before_slope = reliability_slope(reliability_curve(probs, labels))

    def report(applied: bool, reason: str, after: Sequence[float]) -> CalibrationReport:
        return CalibrationReport(
            rows=len(probs),
            applied=applied,
            reason=reason,
            log_loss_before=before_loss,
            log_loss_after=log_loss(after, labels),
            ece_before=before_ece,
            ece_after=expected_calibration_error(after, labels),
            slope_before=before_slope,
            slope_after=reliability_slope(reliability_curve(after, labels)),
            brier_skill=brier_skill(after, labels),
        )

    if len(probs) < MIN_CALIBRATION_ROWS:
        return None, report(False, f"only {len(probs)} rows", probs)
    if not has_discrimination(probs, labels):
        return None, report(False, "no discrimination to calibrate", probs)

    cal = IsotonicCalibrator().fit(probs, labels)
    return cal, report(True, "", [cal.apply(p) for p in probs])


# --------------------------------------------------------------------------- #
# Split conformal for magnitude
# --------------------------------------------------------------------------- #
@dataclass
class ConformalInterval:
    point: float
    low: float
    high: float
    alpha: float
    calibrated: bool

    @property
    def width(self) -> float:
        return self.high - self.low

    def covers(self, actual: float) -> bool:
        return self.low <= actual <= self.high


class SplitConformal:
    """Symmetric prediction intervals from held-out absolute residuals.

    The quantile index is ``ceil((n + 1) * (1 - alpha))``, not
    ``ceil(n * (1 - alpha))``. That plus-one is the entire finite-sample
    guarantee; without it the intervals are systematically a little narrow and
    the coverage shortfall is small enough to look like bad luck.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.half_width: float | None = None
        self.n_calibration = 0

    def calibrate(self, predicted: Sequence[float], actual: Sequence[float]) -> SplitConformal:
        residuals = sorted(abs(p - a) for p, a in zip(predicted, actual, strict=True))
        n = len(residuals)
        if n < 20:
            raise ValueError(
                f"{n} calibration residuals is not enough for a "
                f"{1 - self.alpha:.0%} interval - the quantile would be the maximum"
            )
        k = math.ceil((n + 1) * (1 - self.alpha))
        if k > n:
            raise ValueError(
                f"{n} residuals cannot support {1 - self.alpha:.0%} coverage; "
                f"need at least {math.ceil(1 / self.alpha) - 1}"
            )
        self.half_width = residuals[k - 1]
        self.n_calibration = n
        return self

    def interval(self, point: float) -> ConformalInterval:
        if self.half_width is None:
            return ConformalInterval(point, point, point, self.alpha, calibrated=False)
        return ConformalInterval(
            point=point,
            low=point - self.half_width,
            high=point + self.half_width,
            alpha=self.alpha,
            calibrated=True,
        )


def empirical_coverage(intervals: Sequence[ConformalInterval], actual: Sequence[float]) -> float:
    if not intervals:
        return float("nan")
    return sum(i.covers(a) for i, a in zip(intervals, actual, strict=True)) / len(intervals)


def coverage_by_group(
    intervals: Sequence[ConformalInterval],
    actual: Sequence[float],
    groups: Sequence[str],
) -> dict[str, tuple[float, int]]:
    """Coverage per group, because the conformal guarantee is marginal.

    Ninety percent of intervals cover overall, and that says nothing about any
    particular regime. A model whose intervals cover 98% of quiet days and 60%
    of volatile ones reports 90% and is dangerous in exactly the situation where
    the interval was supposed to help.
    """
    buckets: dict[str, list[bool]] = {}
    for interval, a, g in zip(intervals, actual, groups, strict=True):
        buckets.setdefault(g, []).append(interval.covers(a))
    return {g: (sum(hits) / len(hits), len(hits)) for g, hits in sorted(buckets.items())}


# --------------------------------------------------------------------------- #
# Brier decomposition
# --------------------------------------------------------------------------- #
@dataclass
class BrierDecomposition:
    """Murphy's decomposition: ``Brier = reliability - resolution + uncertainty``."""

    brier: float  # of the raw forecasts
    binned_brier: float  # of the forecasts replaced by their bin means
    reliability: float  # lower is better: how far claims are from frequencies
    resolution: float  # higher is better: how much the forecasts separate
    uncertainty: float  # the target's own variance; not the model's to change
    bins: int

    @property
    def residual(self) -> float:
        """Exactly zero, up to floating point. If it is not, the arithmetic is wrong.

        Reconciles the *binned* Brier score, which is the only thing the
        identity actually holds for. Murphy's decomposition assumes the forecast
        is constant within each bin; with continuous forecasts there is a
        within-bin variance term that does not vanish, and asserting the
        three-way identity on raw probabilities fails by a few thousandths
        forever - small enough to be dismissed as rounding, which is how a real
        arithmetic error would hide here too.
        """
        return self.binned_brier - (self.reliability - self.resolution + self.uncertainty)

    @property
    def discretisation(self) -> float:
        """What the binning itself costs. Large means too few bins to describe
        the forecast distribution, and every term above is coarser than it looks."""
        return self.brier - self.binned_brier

    def render(self) -> str:
        return (
            f"  Brier {self.brier:.5f} = reliability {self.reliability:.5f} "
            f"- resolution {self.resolution:.5f} + uncertainty {self.uncertainty:.5f}\n"
            f"    reliability is the part calibration fixes; resolution is the part "
            f"only a better model can raise"
        )


def brier_decomposition(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> BrierDecomposition:
    """Split the Brier score into what calibration can fix and what it cannot.

    Worth doing because the two failures need different responses. High
    reliability error means the probabilities are mislabelled and isotonic will
    help. Low resolution means the model is not separating outcomes at all, and
    no amount of calibration will change that - it needs different features.
    """
    n = len(probs)
    if n == 0:
        nan = float("nan")
        return BrierDecomposition(nan, nan, nan, nan, nan, bins)

    base = sum(labels) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / n

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        buckets[min(int(p * bins), bins - 1)].append((p, y))

    reliability = resolution = binned = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        k = len(bucket)
        claimed = sum(p for p, _ in bucket) / k
        observed = sum(y for _, y in bucket) / k
        reliability += k * (claimed - observed) ** 2
        resolution += k * (observed - base) ** 2
        binned += sum((claimed - y) ** 2 for _, y in bucket)

    return BrierDecomposition(
        brier=brier,
        binned_brier=binned / n,
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base * (1 - base),
        bins=bins,
    )


# --------------------------------------------------------------------------- #
# The G5 report
# --------------------------------------------------------------------------- #
@dataclass
class G5Report:
    """Everything Gate G5 asks, in one place."""

    stack: StackReport | None
    calibration: CalibrationReport
    decomposition: BrierDecomposition
    beats_baseline: bool
    baseline_name: str
    baseline_log_loss: float
    model_log_loss: float
    bootstrap_ci: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.beats_baseline and self.calibration.passes_g5_calibration

    def render(self) -> str:
        lines = []
        if self.stack:
            lines.append(self.stack.render())
        lines.append(self.calibration.render())
        lines.append(self.decomposition.render())
        ci = (
            f"  [{self.bootstrap_ci[0]:+.5f}, {self.bootstrap_ci[1]:+.5f}]"
            if self.bootstrap_ci
            else "  (no interval computed)"
        )
        lines.append(
            f"  vs {self.baseline_name}: {self.baseline_log_loss:.5f} -> "
            f"{self.model_log_loss:.5f}{ci}"
        )
        lines += [f"  note: {n}" for n in self.notes]
        lines.append(f"  Gate G5: {'PASSES' if self.passes else 'does not pass'}")
        return "\n".join(lines)
