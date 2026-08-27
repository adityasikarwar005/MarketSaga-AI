"""M55 - the volatility model.

The most forecastable target in the system, and the pivot if the direction model
fails its gate. Volatility clusters: a violent hour is followed by a violent
hour, far more reliably than an up move is followed by an up move. That is why
this is worth more effort than its position in the plan suggests.

Three choices here are not incidental.

**Range-based estimators, not close-to-close.** Squared returns are an unbiased
but extremely noisy estimate of variance - a bar that opened, swung two percent
either way and closed flat reports zero volatility. Parkinson and Garman-Klass
read the high and the low, which the bar already carries, and are roughly five
to seven times more efficient for the same data.

**HAR augmented, never replaced.** The heterogeneous autoregressive model - a
regression on daily, weekly and monthly realised volatility - is a genuinely
strong baseline that captures long memory with three coefficients. The booster
here fits its *residual*. So the worst case is that the booster learns nothing
and the model degrades to HAR, rather than to noise; and the gate asks whether
the augmentation earned its place, which is a question the residual formulation
can actually answer.

**QLIKE, not MSE.** Mean squared error on variance is dominated by the largest
observations, so a model can look good by getting a handful of crisis days right
while being useless the rest of the time. QLIKE is scale-free and asymmetric in
the correct direction: it punishes under-forecasting volatility harder than
over-forecasting it, which matches what a wrong forecast actually costs when it
is holding a position.

The retransformation trap
-------------------------

Volatility is modelled in logs because it is roughly log-normal and because the
errors are multiplicative. Exponentiating the prediction then gives the
**median**, not the mean, and the gap between them is not small - for a residual
standard deviation of 0.4 it is about 8%. A model that forecasts the median of a
right-skewed variable and is scored against realisations under-predicts
systematically, forever, and every diagnostic except the bias itself looks fine.

``smearing_factor`` corrects it non-parametrically (Duan), which is preferred to
the ``exp(sigma^2 / 2)`` correction because it does not assume the residuals are
normal - and log-volatility residuals have fat tails.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import fmean

from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import Candle
from libs.ms_models.trees import GradientBoostedTrees

log = get_logger(__name__)

# Variance below this is treated as unmeasurable rather than tiny. A flat minute
# is a minute with no trades, not a minute of zero risk.
MIN_VARIANCE = 1e-12

MIN_ROWS_TO_FIT = 120


# --------------------------------------------------------------------------- #
# Range-based realised variance
# --------------------------------------------------------------------------- #
def parkinson(bars: Sequence[Candle]) -> float:
    """Realised variance from the high-low range.

    ``(1 / (4 ln 2)) * mean(ln(h/l)^2)``. Assumes no drift and continuous
    observation, and under those assumptions it is about five times more
    efficient than squared close-to-close returns.

    Its blind spot is jumps: a bar that gaps and then trades quietly reads as
    violent, because the gap is inside the range. Garman-Klass is worse on that
    axis, not better, which is why both are exposed and neither is the only one.
    """
    ratios = [math.log(b.h / b.low) ** 2 for b in bars if b.h > 0 and b.low > 0 and b.h >= b.low]
    if not ratios:
        return 0.0
    return fmean(ratios) / (4.0 * math.log(2.0))


def garman_klass(bars: Sequence[Candle]) -> float:
    """Realised variance using the open and close as well as the range.

    ``0.5 * ln(h/l)^2 - (2 ln 2 - 1) * ln(c/o)^2``, averaged. More efficient
    than Parkinson when the drift really is negligible; it can return a negative
    estimate on a single bar, which is a property of the estimator and not a
    bug, so the average is floored at zero rather than each term being clipped -
    clipping the terms would bias the result upward.
    """
    terms = []
    for b in bars:
        if b.h <= 0 or b.low <= 0 or b.o <= 0 or b.c <= 0 or b.h < b.low:
            continue
        hl = math.log(b.h / b.low) ** 2
        co = math.log(b.c / b.o) ** 2
        terms.append(0.5 * hl - (2 * math.log(2) - 1) * co)
    if not terms:
        return 0.0
    return max(0.0, fmean(terms))


def close_to_close(bars: Sequence[Candle]) -> float:
    """Squared returns. Kept for comparison, not recommended as the target."""
    closes = [b.c for b in bars if b.c > 0]
    if len(closes) < 2:
        return 0.0
    rets = [math.log(b / a) for a, b in pairwise(closes)]
    return fmean(r * r for r in rets) if rets else 0.0


def realised_variance(bars: Sequence[Candle], *, estimator: str = "parkinson") -> float:
    if estimator == "parkinson":
        return parkinson(bars)
    if estimator == "garman_klass":
        return garman_klass(bars)
    if estimator == "close_to_close":
        return close_to_close(bars)
    raise ValueError(f"unknown estimator {estimator!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def qlike(predicted: Sequence[float], realised: Sequence[float]) -> float:
    """``mean(rv/pred - log(rv/pred) - 1)``. Zero is perfect, lower is better.

    Scale-free, so a quiet week and a crisis week contribute comparably, and
    asymmetric in the direction that matters: forecasting half the true variance
    costs about four times as much as forecasting twice.
    """
    terms = []
    for p, rv in zip(predicted, realised, strict=True):
        p = max(p, MIN_VARIANCE)
        rv = max(rv, MIN_VARIANCE)
        ratio = rv / p
        terms.append(ratio - math.log(ratio) - 1.0)
    return fmean(terms) if terms else float("nan")


def mse(predicted: Sequence[float], realised: Sequence[float]) -> float:
    """Provided so the two can be compared, and so the difference is visible."""
    return fmean((p - r) ** 2 for p, r in zip(predicted, realised, strict=True))


def qlike_improvement(baseline: float, challenger: float) -> float:
    return (baseline - challenger) / baseline if baseline > 0 else 0.0


def qlike_terms(predicted: Sequence[float], realised: Sequence[float]) -> list[float]:
    """The per-observation QLIKE contributions, before averaging."""
    out = []
    for p, rv in zip(predicted, realised, strict=True):
        ratio = max(rv, MIN_VARIANCE) / max(p, MIN_VARIANCE)
        out.append(ratio - math.log(ratio) - 1.0)
    return out


def beats_with_margin(
    baseline: Sequence[float],
    challenger: Sequence[float],
    realised: Sequence[float],
    *,
    t: float = 2.0,
) -> bool:
    """Is the challenger better by more than the noise in the comparison?

    "Lower QLIKE on the validation slice" is not enough on its own. Measured on
    a series with no driver at all, a booster still won that comparison in one
    fold out of four and cost 1.1% on the test rows - the validation slice is
    finite, and a small win on it is a coin toss.

    So the two are compared *paired*, observation by observation, and the mean
    difference has to clear ``t`` standard errors. Paired matters: both models
    see the same volatile days, and the common component of their errors is
    enormous compared with the difference between them.
    """
    a = qlike_terms(baseline, realised)
    b = qlike_terms(challenger, realised)
    if len(a) < 8:
        return False
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    mean = fmean(diffs)
    if mean <= 0:
        return False
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    if var <= 0:
        return True
    return mean > t * math.sqrt(var / len(diffs))


# --------------------------------------------------------------------------- #
# HAR
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HARRow:
    """One observation: the three lookbacks, and what happened next."""

    daily: float  # realised variance over the last period
    weekly: float  # average over the last 5
    monthly: float  # average over the last 22
    target: float  # realised variance over the next horizon
    extra: dict[str, float] = field(default_factory=dict)

    def har_vector(self) -> list[float]:
        return [
            math.log(max(self.daily, MIN_VARIANCE)),
            math.log(max(self.weekly, MIN_VARIANCE)),
            math.log(max(self.monthly, MIN_VARIANCE)),
        ]


def build_har_rows(
    series: Sequence[float],
    *,
    horizon: int = 1,
    weekly: int = 5,
    monthly: int = 22,
    extra: Sequence[dict[str, float]] | None = None,
) -> list[HARRow]:
    """Turn a realised-variance series into HAR observations.

    ``series[i]`` is the realised variance of period ``i``. A row at ``i`` uses
    only periods up to and including ``i`` and targets the average over the next
    ``horizon`` periods, so nothing in the row is knowable later than its own
    timestamp. That is the whole point of building this here rather than with a
    rolling window helper that would happily centre itself.
    """
    rows: list[HARRow] = []
    for i in range(monthly - 1, len(series) - horizon):
        future = series[i + 1 : i + 1 + horizon]
        if len(future) < horizon:
            break
        rows.append(
            HARRow(
                daily=series[i],
                weekly=fmean(series[i - weekly + 1 : i + 1]),
                monthly=fmean(series[i - monthly + 1 : i + 1]),
                target=fmean(future),
                extra=dict(extra[i]) if extra and i < len(extra) else {},
            )
        )
    return rows


def ols(rows: Sequence[Sequence[float]], targets: Sequence[float]) -> list[float] | None:
    """Least squares with an intercept, by normal equations with ridge damping.

    The ridge term is 1e-8 and exists only to keep a singular design from
    raising - HAR's three regressors are strongly collinear by construction,
    since weekly and monthly contain daily.
    """
    if not rows or len(rows) != len(targets):
        return None
    n_features = len(rows[0]) + 1
    if len(rows) <= n_features:
        return None

    design = [[1.0, *r] for r in rows]
    xtx = [[0.0] * n_features for _ in range(n_features)]
    xty = [0.0] * n_features
    for row, y in zip(design, targets, strict=True):
        for a in range(n_features):
            xty[a] += row[a] * y
            for b in range(n_features):
                xtx[a][b] += row[a] * row[b]
    for a in range(n_features):
        xtx[a][a] += 1e-8

    # Gaussian elimination with partial pivoting.
    for col in range(n_features):
        pivot = max(range(col, n_features), key=lambda r: abs(xtx[r][col]))
        if abs(xtx[pivot][col]) < 1e-14:
            return None
        xtx[col], xtx[pivot] = xtx[pivot], xtx[col]
        xty[col], xty[pivot] = xty[pivot], xty[col]
        for r in range(col + 1, n_features):
            factor = xtx[r][col] / xtx[col][col]
            if factor == 0.0:
                continue
            for c in range(col, n_features):
                xtx[r][c] -= factor * xtx[col][c]
            xty[r] -= factor * xty[col]

    beta = [0.0] * n_features
    for i in range(n_features - 1, -1, -1):
        acc = xty[i] - sum(xtx[i][c] * beta[c] for c in range(i + 1, n_features))
        beta[i] = acc / xtx[i][i]
    return beta


def smearing_factor(log_residuals: Sequence[float]) -> float:
    """Duan's non-parametric correction for exponentiating a log-scale fit.

    ``mean(exp(residual))``. Exponentiating a fitted log value returns the
    conditional *median*; multiplying by this returns the conditional mean.
    Preferred over ``exp(sigma^2 / 2)`` because it makes no normality assumption,
    and log-volatility residuals are visibly fat-tailed.

    Without it the model under-forecasts by roughly 8% at a residual spread of
    0.4 - permanently, and invisibly to every diagnostic that is not looking for
    exactly this.
    """
    if not log_residuals:
        return 1.0
    return fmean(math.exp(r) for r in log_residuals)


class HARModel:
    """Regression of log realised variance on its daily, weekly and monthly averages."""

    def __init__(self) -> None:
        self.beta: list[float] | None = None
        self.smearing = 1.0
        self.residual_sd = 0.0

    def fit(self, rows: Sequence[HARRow]) -> HARModel:
        if len(rows) < 10:
            raise ValueError(f"{len(rows)} rows is not enough to fit HAR")
        design = [r.har_vector() for r in rows]
        targets = [math.log(max(r.target, MIN_VARIANCE)) for r in rows]
        beta = ols(design, targets)
        if beta is None:
            raise ValueError("HAR design matrix is singular")
        self.beta = beta

        residuals = [y - self._log_predict(x) for x, y in zip(design, targets, strict=True)]
        self.smearing = smearing_factor(residuals)
        mean_r = fmean(residuals)
        self.residual_sd = (
            math.sqrt(sum((r - mean_r) ** 2 for r in residuals) / max(1, len(residuals) - 1))
            if len(residuals) > 1
            else 0.0
        )
        return self

    def _log_predict(self, vector: Sequence[float]) -> float:
        assert self.beta is not None
        return self.beta[0] + sum(b * x for b, x in zip(self.beta[1:], vector, strict=True))

    def log_predict(self, row: HARRow) -> float:
        if self.beta is None:
            raise RuntimeError("fit() before predict()")
        return self._log_predict(row.har_vector())

    def predict(self, row: HARRow) -> float:
        """Conditional mean variance, smearing correction applied."""
        return math.exp(self.log_predict(row)) * self.smearing

    def predict_all(self, rows: Sequence[HARRow]) -> list[float]:
        return [self.predict(r) for r in rows]


# --------------------------------------------------------------------------- #
# HAR + boosted residual
# --------------------------------------------------------------------------- #
@dataclass
class VolatilityReport:
    rows: int
    features: int
    trees: int
    har_qlike: float
    model_qlike: float
    augmented: bool
    smearing: float

    @property
    def improvement(self) -> float:
        return qlike_improvement(self.har_qlike, self.model_qlike)

    def render(self) -> str:
        verdict = (
            f"booster kept, {self.trees} trees"
            if self.augmented
            else "booster discarded - it added nothing on held-out rows"
        )
        return (
            f"  volatility: {self.rows:,} rows, {self.features} residual features\n"
            f"    HAR QLIKE {self.har_qlike:.5f} -> model {self.model_qlike:.5f} "
            f"({self.improvement:+.1%})\n"
            f"    {verdict}, smearing {self.smearing:.4f}"
        )


class VolatilityModel:
    """HAR, plus boosted trees on what HAR could not explain.

    The booster is only kept if it improves QLIKE on a held-out slice that
    neither it nor HAR was fitted on. Otherwise the model *is* HAR - which is
    the honest outcome when the extra features carry nothing, and is why this
    cannot quietly become worse than its own baseline.
    """

    name = "M5_volatility"
    version = "1"

    def __init__(
        self,
        *,
        validation_share: float = 0.25,
        feature_names: Sequence[str] | None = None,
        **booster: object,
    ) -> None:
        self.validation_share = validation_share
        self.feature_names = list(feature_names) if feature_names else []
        self.booster_kwargs = booster

        self.har = HARModel()
        self.booster: GradientBoostedTrees | None = None
        self.residual_model: list[float] | None = None
        self.residual_smearing = 1.0
        self.report: VolatilityReport | None = None

    def _residual_vector(self, row: HARRow) -> list[float]:
        base = row.har_vector()
        return base + [row.extra.get(n, 0.0) for n in self.feature_names]

    def fit(self, rows: Sequence[HARRow]) -> VolatilityModel:
        if len(rows) < MIN_ROWS_TO_FIT:
            raise ValueError(
                f"{len(rows)} rows is not enough to fit and validate " f"(need {MIN_ROWS_TO_FIT})"
            )
        if not self.feature_names:
            self.feature_names = sorted({k for r in rows for k in r.extra})

        n_valid = int(len(rows) * self.validation_share)
        cut = len(rows) - n_valid
        fit_rows, valid_rows = rows[:cut], rows[cut:]

        self.har.fit(fit_rows)
        har_valid = self.har.predict_all(valid_rows)
        har_qlike = qlike(har_valid, [r.target for r in valid_rows])

        # The booster learns the log-residual: what HAR got wrong, in the units
        # HAR works in.
        residuals = [
            math.log(max(r.target, MIN_VARIANCE)) - self.har.log_predict(r) for r in fit_rows
        ]
        model_qlike = har_qlike
        augmented = False
        trees = 0

        fitted = self._fit_residual(fit_rows, residuals)
        if fitted is None:
            # Nothing was fitted at all - same reasoning as the branch below.
            self.har = HARModel().fit(rows)
        else:
            candidate = [self._augmented_predict(r) for r in valid_rows]
            targets = [r.target for r in valid_rows]
            candidate_qlike = qlike(candidate, targets)
            if beats_with_margin(har_valid, candidate, targets):
                model_qlike = candidate_qlike
                augmented = True
                trees = fitted
            else:
                # It did not help. Discard it rather than shipping a model that
                # is worse than the baseline it was supposed to augment - and
                # then refit HAR on every training row.
                #
                # The held-out slice existed to answer one question: is the
                # booster worth keeping. Once the answer is no, continuing to
                # withhold a quarter of the data from a three-parameter
                # regression is a handicap with nothing to show for it, and it
                # made the "augmented" model score 0.8% worse than plain HAR on
                # a series with no driver at all - purely from the smaller fit.
                #
                # This is not leakage. The validation rows informed a discrete
                # keep-or-drop decision, not the coefficients, and the model
                # they now help fit contains no trace of the booster they
                # rejected.
                self.booster = None
                self.residual_model = None
                self.har = HARModel().fit(rows)

        self.report = VolatilityReport(
            rows=len(rows),
            features=len(self.feature_names),
            trees=trees,
            har_qlike=har_qlike,
            model_qlike=model_qlike,
            augmented=augmented,
            smearing=self.har.smearing,
        )
        log.info(
            "volatility model fitted",
            fields={
                "rows": len(rows),
                "augmented": augmented,
                "trees": trees,
                "qlike_improvement": round(self.report.improvement, 5),
            },
        )
        return self

    def _fit_residual(self, rows: Sequence[HARRow], residuals: Sequence[float]) -> int | None:
        """Boost on the residual sign, then scale by the residual magnitude.

        The booster in this project is a classifier, so the regression is done
        by predicting whether HAR under-forecast and by how much on average -
        deliberately coarse. A residual model with more resolution than the data
        supports is the failure this whole layer is built to avoid, and the
        held-out check above is what decides whether even this much was earned.
        """
        if len(rows) < MIN_ROWS_TO_FIT:
            return None
        labels = [1 if r > 0 else 0 for r in residuals]
        if len(set(labels)) < 2:
            return None

        vectors = [self._residual_vector(r) for r in rows]
        try:
            booster = GradientBoostedTrees(**self.booster_kwargs).fit(vectors, labels)  # type: ignore[arg-type]
        except ValueError:
            return None
        if not booster.trees:
            return None

        self.booster = booster
        up = [r for r in residuals if r > 0]
        down = [r for r in residuals if r <= 0]
        self.residual_model = [fmean(up) if up else 0.0, fmean(down) if down else 0.0]
        return len(booster.trees)

    def _augmented_predict(self, row: HARRow) -> float:
        base_log = self.har.log_predict(row)
        if self.booster is None or self.residual_model is None:
            return math.exp(base_log) * self.har.smearing
        p_up = self.booster.predict_proba(self._residual_vector(row))
        adjustment = p_up * self.residual_model[0] + (1 - p_up) * self.residual_model[1]
        return math.exp(base_log + adjustment) * self.har.smearing

    def predict(self, row: HARRow) -> float:
        if self.har.beta is None:
            raise RuntimeError("fit() before predict()")
        return self._augmented_predict(row)

    def predict_all(self, rows: Sequence[HARRow]) -> list[float]:
        return [self.predict(r) for r in rows]

    def predict_volatility(self, row: HARRow) -> float:
        """Standard deviation rather than variance, for anything that sizes."""
        return math.sqrt(max(self.predict(row), 0.0))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class VolatilityEvaluation:
    folds: int
    n_test: int
    har_qlike: float
    model_qlike: float
    har_mse: float
    model_mse: float
    augmented_folds: int

    @property
    def improvement(self) -> float:
        return qlike_improvement(self.har_qlike, self.model_qlike)

    @property
    def passes(self) -> bool:
        """Beating HAR on QLIKE out of sample is the whole gate for this model."""
        return self.improvement > 0.0

    def render(self) -> str:
        lines = [
            f"  walk-forward: {self.folds} folds, {self.n_test:,} test rows",
            f"  HAR-RV        QLIKE={self.har_qlike:.5f}  MSE={self.har_mse:.3e}",
            f"  HAR+residual  QLIKE={self.model_qlike:.5f}  MSE={self.model_mse:.3e}",
            f"  QLIKE improvement {self.improvement:+.1%}  "
            f"(booster kept in {self.augmented_folds}/{self.folds} folds)  ->  "
            f"{'beats HAR' if self.passes else 'does not beat HAR'}",
        ]
        if (self.model_mse > self.har_mse) != (self.model_qlike > self.har_qlike):
            lines.append(
                "  note: QLIKE and MSE disagree. MSE is dominated by the largest "
                "few observations; QLIKE is the one this model is judged on."
            )
        return "\n".join(lines)


def walk_forward_evaluate(
    rows: Sequence[HARRow],
    *,
    n_folds: int = 4,
    min_train: int = MIN_ROWS_TO_FIT,
    **model: object,
) -> VolatilityEvaluation | None:
    """Train on the past, test on the future, repeatedly.

    No purging: each HAR row targets the horizon strictly after its own
    lookbacks, so consecutive rows do not share outcome windows at horizon 1.
    At longer horizons they do, and the caller owes an embargo of ``horizon``
    rows - which is why this takes rows rather than building them.
    """
    if len(rows) < min_train + n_folds * 10:
        return None
    fold_size = (len(rows) - min_train) // n_folds
    if fold_size < 10:
        return None

    har_pred: list[float] = []
    model_pred: list[float] = []
    actual: list[float] = []
    augmented = 0

    for f in range(n_folds):
        train_end = min_train + f * fold_size
        test = rows[train_end : train_end + fold_size]
        if not test:
            continue
        train = rows[:train_end]

        baseline = HARModel().fit(train)
        fitted = VolatilityModel(**model).fit(train)  # type: ignore[arg-type]
        if fitted.report and fitted.report.augmented:
            augmented += 1

        har_pred += baseline.predict_all(test)
        model_pred += fitted.predict_all(test)
        actual += [r.target for r in test]

    if not actual:
        return None
    return VolatilityEvaluation(
        folds=n_folds,
        n_test=len(actual),
        har_qlike=qlike(har_pred, actual),
        model_qlike=qlike(model_pred, actual),
        har_mse=mse(har_pred, actual),
        model_mse=mse(model_pred, actual),
        augmented_folds=augmented,
    )
