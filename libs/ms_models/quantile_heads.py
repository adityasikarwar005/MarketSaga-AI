"""M54 - the quantile heads: `q_a(r_h)` for a in {.05,.25,.5,.75,.95}, per horizon.

One boosted model per (alpha, horizon), monotonicity enforced afterwards, and a
Gaussian fallback from the volatility model's sigma when a head cannot be fitted.

The thing that makes this work is not the boosting
----------------------------------------------------

Pinball loss is piecewise linear. Its **second derivative is zero everywhere**,
and Newton boosting divides by it:

    leaf value = -sum(g) / (sum(h) + l2)

With ``h = 0`` that reduces to ``-sum(g)/l2``. The gradient of the pinball loss
is ``-alpha`` above the prediction and ``1-alpha`` below it, so ``sum(g)`` counts
how many points sit on each side and knows nothing about *how far*. The leaf step
becomes a function of the sign imbalance and the regularisation constant, and of
nothing else in the data.

The model still trains, and this is where it gets interesting: it does not fail
in the way you would expect. **Marginal coverage still looks fine.** What breaks
is the conditional part - the interval stops tracking volatility.

Measured on heteroscedastic data whose true 5-95 half-width triples between the
quiet and busy regimes:

===================  ==========================  ==============  =============
                     half-width ratio (true 2.6)  coverage quiet  coverage busy
===================  ==========================  ==============  =============
leaf renewal on      1.89                        91.2%           84.6%
leaf renewal off     1.28                        97.1%           84.6%
===================  ==========================  ==============  =============

Without renewal the band is nearly flat across regimes: far too wide when it is
quiet, and exactly as narrow as before when it is not. The two errors cancel in
the marginal number, so the headline coverage reads as calibrated while every
individual row is wrong - too small a position on the quiet days and too large a
one on precisely the days that matter. It also never early-stops, running to the
tree cap because the steps never converge on anything.

Renewal is a large improvement and not a complete one: 1.89 against a true 2.6
is still under-tracking. That is a limit of depth-3 trees on this much data, not
a defect, and it is stated here so nobody reads the fix as exact.

The fix is **leaf-value renewal**, which is what LightGBM and sklearn do and what
the card's "LightGBM with pinball objective" quietly implies: use a constant
hessian to *choose the splits*, then set each leaf's value to the empirical
alpha-quantile of the residuals that landed in it. The tree structure comes from
the gradient; the step size comes from the data.

:class:`QuantileBooster` has a ``renew_leaves`` flag, defaulting to on, purely so
a test can turn it off and demonstrate what it buys. A defence nobody can
switch off is a defence nobody has measured.

Monotonicity is enforced after the fact, not during
-----------------------------------------------------

Five independently fitted models have no reason to produce ordered outputs, and
on a thin slice they will not: a 25th percentile above a 75th is a distribution
with negative mass, which sizes a position off a width that does not exist.
:func:`enforce_monotone` sorts them. That is a repair, and the number of times it
had to repair anything is reported rather than swallowed - a head that needs
sorting on a third of its rows is a head that is not fitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from libs.ms_core.obs import get_logger
from libs.ms_models.quantile import (
    DEFAULT_QUANTILES,
    QuantileForecast,
    enforce_monotone,
    pinball_loss,
)
from libs.ms_models.trees import Binner, Tree, build_tree

log = get_logger(__name__)

MIN_ROWS_PER_HEAD = 120
DEFAULT_HORIZONS = (15, 30, 60)

# Constant hessian. It exists to make the split search well-defined, not to
# describe the loss - the pinball loss has no curvature. Every leaf value it
# produces is discarded and replaced; see the module docstring.
_UNIT_HESSIAN = 1.0


class QuantileHeadError(RuntimeError):
    pass


def pinball_gradient(actual: float, predicted: float, alpha: float) -> float:
    """d(pinball)/d(prediction). Piecewise constant, which is the whole problem."""
    return -alpha if actual > predicted else (1.0 - alpha)


def empirical_quantile(values: Sequence[float], alpha: float) -> float:
    """Type-7 linear interpolation, matching numpy's default.

    Interpolated rather than nearest-rank because a leaf holding thirty
    residuals would otherwise only ever step to one of thirty values, and the
    quantisation shows up as a staircase in the fitted interval width.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = alpha * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _leaf_index(tree: Tree, binned: Sequence[int]) -> int:
    i = 0
    while not tree.nodes[i].is_leaf:
        node = tree.nodes[i]
        i = node.left if binned[node.feature] <= node.bin_threshold else node.right
    return i


# --------------------------------------------------------------------------- #
# One head
# --------------------------------------------------------------------------- #
@dataclass
class HeadReport:
    alpha: float
    horizon: int
    trees: int
    trees_cap: int
    train_pinball: float
    valid_pinball: float | None
    stopped_early: bool

    def render(self) -> str:
        v = "n/a" if self.valid_pinball is None else f"{self.valid_pinball:.6f}"
        stop = "early" if self.stopped_early else "cap"
        return (
            f"    a={self.alpha:.2f} h={self.horizon:<4} trees={self.trees}/{self.trees_cap} "
            f"({stop})  train={self.train_pinball:.6f}  valid={v}"
        )


class QuantileBooster:
    """Boosted quantile regression for one alpha.

    ``renew_leaves`` is not a tuning knob. With it off this is Newton boosting
    on a loss with no curvature, so the leaf step is a function of the sign
    imbalance and the regularisation constant and of nothing else in the data.
    It is exposed only so a test can measure what that costs.
    """

    def __init__(
        self,
        alpha: float,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        l2: float = 1.0,
        min_gain: float = 1e-6,
        validation_share: float = 0.2,
        patience: int = 15,
        max_bins: int = 64,
        renew_leaves: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise QuantileHeadError(f"alpha must be strictly between 0 and 1, got {alpha}")
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.min_gain = min_gain
        self.validation_share = validation_share
        self.patience = patience
        self.max_bins = max_bins
        self.renew_leaves = renew_leaves

        self.binner = Binner()
        self.trees: list[Tree] = []
        self.base_score = 0.0
        self.fitted = False
        self.report: HeadReport | None = None

    def fit(
        self, rows: Sequence[Sequence[float]], targets: Sequence[float], *, horizon: int = 0
    ) -> QuantileBooster:
        if len(rows) != len(targets):
            raise QuantileHeadError("rows and targets must be the same length")
        if len(rows) < MIN_ROWS_PER_HEAD:
            raise QuantileHeadError(
                f"{len(rows)} rows for the a={self.alpha} head, need {MIN_ROWS_PER_HEAD}. "
                "A quantile fitted on less than that is describing the tail of a "
                "sample, not of a distribution."
            )

        # Chronological split, same reasoning as the classifier: a randomly
        # drawn validation set is interleaved with training data and its
        # early-stopping signal is optimistic for the same reason an unpurged
        # fold is.
        n_valid = int(len(rows) * self.validation_share)
        if n_valid < self.min_samples_leaf:
            n_valid = 0
        cut = len(rows) - n_valid
        train_rows, valid_rows = list(rows[:cut]), list(rows[cut:])
        train_y, valid_y = list(targets[:cut]), list(targets[cut:])

        self.binner.fit(train_rows, max_bins=self.max_bins)
        binned = self.binner.apply_all(train_rows)
        valid_binned = self.binner.apply_all(valid_rows) if valid_rows else []

        # The unconditional quantile is the right place to start: it is what the
        # model must beat, and starting at zero makes the first few trees spend
        # themselves walking to it.
        self.base_score = empirical_quantile(train_y, self.alpha)
        self.trees = []

        f = [self.base_score] * len(train_rows)
        f_valid = [self.base_score] * len(valid_rows)

        best_valid = math.inf
        best_n = 0
        since_best = 0
        stopped_early = False

        for _ in range(self.n_estimators):
            grad = [pinball_gradient(train_y[i], f[i], self.alpha) for i in range(len(f))]
            hess = [_UNIT_HESSIAN] * len(f)

            tree = build_tree(
                binned,
                grad,
                hess,
                list(range(len(train_rows))),
                self.binner,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2=self.l2,
                min_gain=self.min_gain,
            )

            if self.renew_leaves:
                self._renew(tree, binned, train_y, f)

            for i, b in enumerate(binned):
                f[i] += self.lr * tree.predict_binned(b)
            for i, b in enumerate(valid_binned):
                f_valid[i] += self.lr * tree.predict_binned(b)
            self.trees.append(tree)

            if valid_rows:
                loss = self._mean_pinball(valid_y, f_valid)
                if loss < best_valid - 1e-12:
                    best_valid, best_n, since_best = loss, len(self.trees), 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        stopped_early = True
                        break

        if valid_rows and best_n:
            # Drop the trees that came after the best validation score. Keeping
            # them would mean the reported valid loss belongs to a model that is
            # not the one being returned.
            self.trees = self.trees[:best_n]
            f = [self.base_score] * len(train_rows)
            for tree in self.trees:
                for i, b in enumerate(binned):
                    f[i] += self.lr * tree.predict_binned(b)

        self.fitted = True
        self.report = HeadReport(
            alpha=self.alpha,
            horizon=horizon,
            trees=len(self.trees),
            trees_cap=self.n_estimators,
            train_pinball=self._mean_pinball(train_y, f),
            valid_pinball=best_valid if (valid_rows and best_n) else None,
            stopped_early=stopped_early,
        )
        return self

    def _renew(
        self,
        tree: Tree,
        binned: Sequence[Sequence[int]],
        targets: Sequence[float],
        current: Sequence[float],
    ) -> None:
        """Replace every leaf's Newton step with the quantile of its residuals.

        The tree structure came from the gradient, which is where the gradient's
        information actually is. The step size comes from the data, which is the
        only place it can come from when the loss has no curvature.
        """
        buckets: dict[int, list[float]] = {}
        for i, b in enumerate(binned):
            buckets.setdefault(_leaf_index(tree, b), []).append(targets[i] - current[i])
        for leaf, residuals in buckets.items():
            tree.nodes[leaf].value = empirical_quantile(residuals, self.alpha)

    def _mean_pinball(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        if not actual:
            return math.nan
        return sum(
            pinball_loss(a, p, self.alpha) for a, p in zip(actual, predicted, strict=True)
        ) / len(actual)

    def predict(self, row: Sequence[float]) -> float:
        if not self.fitted:
            raise QuantileHeadError("fit() before predict()")
        b = self.binner.apply(row)
        return self.base_score + self.lr * sum(t.predict_binned(b) for t in self.trees)

    def predict_all(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict(r) for r in rows]


# --------------------------------------------------------------------------- #
# The grid of heads
# --------------------------------------------------------------------------- #
@dataclass
class HeadsReport:
    heads: list[HeadReport] = field(default_factory=list)
    fallbacks: list[tuple[float, int, str]] = field(default_factory=list)
    monotonicity_repairs: int = 0
    predictions: int = 0

    @property
    def repair_rate(self) -> float:
        return self.monotonicity_repairs / self.predictions if self.predictions else 0.0

    def render(self) -> str:
        lines = [f"  quantile heads ({len(self.heads)} fitted)"]
        lines += [h.render() for h in self.heads]
        for alpha, horizon, why in self.fallbacks:
            lines.append(f"    a={alpha:.2f} h={horizon:<4} FALLBACK to Gaussian: {why}")
        if self.predictions:
            lines.append(
                f"    monotonicity repaired on {self.monotonicity_repairs}/{self.predictions} "
                f"predictions ({self.repair_rate:.1%})"
            )
            if self.repair_rate > 0.10:
                lines.append(
                    "    A head needing its order repaired this often is not fitted. "
                    "The sort makes the output usable and does not make it right."
                )
        return "\n".join(lines)


class QuantileHeads:
    """One model per (alpha, horizon), with a Gaussian fallback per head.

    The fallback is per head rather than per model on purpose. A 5% head can
    fail to fit on a quiet quarter while the median head is perfectly good, and
    dropping the whole distribution for that loses information the trade could
    have used.
    """

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        **booster_kwargs: object,
    ) -> None:
        self.quantiles = tuple(sorted(quantiles))
        self.horizons = tuple(horizons)
        self._kwargs = booster_kwargs
        self.heads: dict[tuple[float, int], QuantileBooster] = {}
        self.gaussian_sigma: dict[int, float] = {}
        self.report = HeadsReport()
        self.fitted = False

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        targets_by_horizon: dict[int, Sequence[float]],
        *,
        sigma_by_horizon: dict[int, float] | None = None,
    ) -> QuantileHeads:
        """Fit every head. A head that cannot fit falls back rather than raising.

        ``sigma_by_horizon`` is the volatility model's forecast, and it is the
        card's named fallback: Gaussian quantiles from AI-10's sigma. Without it
        a failed head has nothing to fall back to and the caller finds out at
        predict time, which is the wrong end of the day.
        """
        sigmas = sigma_by_horizon or {}
        for horizon in self.horizons:
            if horizon not in targets_by_horizon:
                raise QuantileHeadError(f"no targets for horizon {horizon}")
            targets = targets_by_horizon[horizon]
            if horizon in sigmas:
                self.gaussian_sigma[horizon] = sigmas[horizon]

            for alpha in self.quantiles:
                booster = QuantileBooster(alpha, **self._kwargs)  # type: ignore[arg-type]
                try:
                    booster.fit(rows, targets, horizon=horizon)
                except QuantileHeadError as exc:
                    if horizon not in self.gaussian_sigma:
                        raise QuantileHeadError(
                            f"the a={alpha} h={horizon} head could not be fitted "
                            f"({exc}) and no sigma was supplied to fall back to. "
                            "Pass sigma_by_horizon from the volatility model."
                        ) from exc
                    self.report.fallbacks.append((alpha, horizon, str(exc)[:80]))
                    continue
                self.heads[(alpha, horizon)] = booster
                if booster.report is not None:
                    self.report.heads.append(booster.report)

        self.fitted = True
        return self

    def predict(self, row: Sequence[float], *, horizon: int) -> QuantileForecast:
        """The distribution at one horizon, ordered.

        Ordering is repaired after the fact. Five independently fitted models
        have no reason to agree, and a 25th percentile above a 75th is a
        distribution with negative mass - which sizes a position off a width
        that does not exist.
        """
        if not self.fitted:
            raise QuantileHeadError("fit() before predict()")
        if horizon not in self.horizons:
            raise QuantileHeadError(f"horizon {horizon} was not fitted")

        raw: dict[float, float] = {}
        for alpha in self.quantiles:
            head = self.heads.get((alpha, horizon))
            if head is not None:
                raw[alpha] = head.predict(row)
            else:
                raw[alpha] = gaussian_quantile(self.gaussian_sigma[horizon], alpha)

        ordered = enforce_monotone(raw)
        self.report.predictions += 1
        if any(ordered[a] != raw[a] for a in self.quantiles):
            self.report.monotonicity_repairs += 1
        return QuantileForecast(quantiles=ordered)

    def used_fallback(self, alpha: float, horizon: int) -> bool:
        return (alpha, horizon) not in self.heads


# --------------------------------------------------------------------------- #
# The fallback
# --------------------------------------------------------------------------- #
def _inv_norm_cdf(p: float) -> float:
    """Acklam's inverse normal CDF. Accurate to about 1.15e-9 across (0,1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def gaussian_quantile(sigma: float, alpha: float, *, mu: float = 0.0) -> float:
    """The card's named fallback: Gaussian quantiles from the sigma forecast.

    A worse forecast than a fitted head and a much better one than nothing. It
    is symmetric by construction, so it cannot represent skew - which is worth
    knowing when a fallback head is carrying a tail.
    """
    if sigma < 0.0:
        raise ValueError("sigma cannot be negative")
    return mu + sigma * _inv_norm_cdf(alpha)


def gaussian_forecast(
    sigma: float, *, quantiles: Sequence[float] = DEFAULT_QUANTILES, mu: float = 0.0
) -> QuantileForecast:
    return QuantileForecast(
        quantiles={a: gaussian_quantile(sigma, a, mu=mu) for a in sorted(quantiles)}
    )


# --------------------------------------------------------------------------- #
# Scoring against the gate
# --------------------------------------------------------------------------- #
@dataclass
class HeadScore:
    alpha: float
    pinball: float
    coverage: float | None = None  # only meaningful for a paired interval

    def render(self) -> str:
        cov = "" if self.coverage is None else f"  coverage={self.coverage:.1%}"
        return f"    a={self.alpha:.2f}  pinball={self.pinball:.6f}{cov}"


def score_heads(
    forecasts: Sequence[QuantileForecast],
    actuals: Sequence[float],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> list[HeadScore]:
    """Pinball per alpha, plus coverage for the outer band."""
    out = []
    for alpha in sorted(quantiles):
        losses = [
            pinball_loss(a, f.quantiles[alpha], alpha)
            for f, a in zip(forecasts, actuals, strict=True)
        ]
        out.append(
            HeadScore(alpha=alpha, pinball=sum(losses) / len(losses) if losses else math.nan)
        )

    lo, hi = min(quantiles), max(quantiles)
    inside = sum(
        1
        for f, a in zip(forecasts, actuals, strict=True)
        if f.quantiles[lo] <= a <= f.quantiles[hi]
    )
    nominal = hi - lo
    for score in out:
        if score.alpha in (lo, hi):
            score.coverage = inside / len(actuals) if actuals else None
    log.info(
        "quantile heads scored",
        fields={"n": len(actuals), "nominal": nominal, "empirical": inside / max(len(actuals), 1)},
    )
    return out


def mean_pinball(
    forecasts: Sequence[QuantileForecast],
    actuals: Sequence[float],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> float:
    """One number across every alpha, for comparing whole models."""
    scores = score_heads(forecasts, actuals, quantiles=quantiles)
    return sum(s.pinball for s in scores) / len(scores)
