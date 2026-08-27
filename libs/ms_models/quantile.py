"""Quantile regression and distributional scoring.

The blueprint's central claim is that a point forecast is false precision: what
a trade needs is a distribution. This module is the machinery for producing one
and, more importantly, for scoring it honestly.

Linear quantile regression fitted by gradient descent on the pinball loss. Not
because gradient descent is the best solver - linear programming is exact - but
because this is dependency-free, deterministic, and small enough to read. When a
model's numbers decide whether real money moves, being able to read the fitting
code matters more than the last few percent of optimiser efficiency.

Three scoring rules, and the reason there are three:

**Pinball loss** scores each quantile separately. It says which parts of the
distribution are wrong.

**CRPS** scores the whole distribution as one number, so two forecasters can be
compared. It is the headline metric for the gap model's gate.

**Coverage and PIT** ask whether the intervals mean what they claim. A forecast
can have excellent CRPS and 60% coverage on its 90% band, and that forecast will
size positions wrongly every time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean, pstdev

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def pinball_loss(actual: float, predicted: float, alpha: float) -> float:
    """Asymmetric loss: under-predicting the 95th percentile costs little,
    over-predicting it costs a lot. That asymmetry is what makes the fitted
    value a quantile rather than a mean."""
    diff = actual - predicted
    return alpha * diff if diff >= 0 else (alpha - 1.0) * diff


@dataclass(frozen=True)
class QuantileForecast:
    """A predictive distribution expressed as quantiles."""

    quantiles: dict[float, float]

    def __post_init__(self) -> None:
        levels = sorted(self.quantiles)
        values = [self.quantiles[a] for a in levels]
        if values != sorted(values):
            raise ValueError(
                f"quantiles cross: {dict(zip(levels, values, strict=True))}. "
                "A 25th percentile above the 75th is not a distribution."
            )

    @property
    def median(self) -> float:
        return self.quantiles.get(0.5, fmean(self.quantiles.values()))

    def interval(self, lo: float = 0.05, hi: float = 0.95) -> tuple[float, float]:
        return self.quantiles[lo], self.quantiles[hi]

    def covers(self, actual: float, lo: float = 0.05, hi: float = 0.95) -> bool:
        a, b = self.interval(lo, hi)
        return a <= actual <= b

    def width(self, lo: float = 0.05, hi: float = 0.95) -> float:
        a, b = self.interval(lo, hi)
        return b - a

    def pit(self, actual: float) -> float:
        """Where the outcome fell in the predicted distribution, in [0, 1].

        Over many forecasts these should be uniform. A hump in the middle means
        the intervals are too wide; humps at the edges mean they are too narrow
        and the model is more confident than it has earned.
        """
        levels = sorted(self.quantiles)
        for a in levels:
            if actual <= self.quantiles[a]:
                return a
        return 1.0

    def crps(self, actual: float) -> float:
        """Continuous ranked probability score, approximated from the quantiles.

        The integral of the squared difference between the predicted CDF and the
        step function at the outcome. Lower is better; it collapses to absolute
        error for a point forecast, which is what makes it a fair comparison
        against a baseline that only predicts a number.
        """
        levels = sorted(self.quantiles)
        # Average pinball loss over the quantile grid approximates CRPS up to a
        # factor of two - a standard identity, and exact in the limit of a dense
        # grid.
        return 2.0 * fmean(pinball_loss(actual, self.quantiles[a], a) for a in levels)

    def render(self) -> str:
        return "  ".join(
            f"q{int(a * 100):02d}={self.quantiles[a]:+.3%}" for a in sorted(self.quantiles)
        )


def enforce_monotone(values: dict[float, float]) -> dict[float, float]:
    """Repair crossed quantiles by taking a running maximum.

    Independently fitted quantiles cross occasionally, especially in the tails
    where data is thin. Rearranging is the standard remedy and never makes the
    forecast worse by any proper scoring rule.
    """
    levels = sorted(values)
    out: dict[float, float] = {}
    running = -math.inf
    for a in levels:
        running = max(running, values[a])
        out[a] = running
    return out


# --------------------------------------------------------------------------- #
# Linear quantile regression
# --------------------------------------------------------------------------- #
@dataclass
class Standardiser:
    """Feature scaling, fitted on train only.

    Fitting the scaler on the full dataset is one of the quietest forms of
    leakage: the test set's mean and spread reach the model through the
    normalisation, and nothing in the output looks wrong.
    """

    means: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]]) -> Standardiser:
        if not rows:
            return self
        n_features = len(rows[0])
        self.means = [fmean([r[j] for r in rows]) for j in range(n_features)]
        self.sigmas = [
            (pstdev([r[j] for r in rows]) or 1.0) if len(rows) > 1 else 1.0
            for j in range(n_features)
        ]
        return self

    def apply(self, row: Sequence[float]) -> list[float]:
        if not self.means:
            return list(row)
        return [
            (v - m) / s if s > 0 else 0.0
            for v, m, s in zip(row, self.means, self.sigmas, strict=True)
        ]


class QuantileRegressor:
    """One linear model per quantile level, fitted on the pinball loss.

    Deterministic given the data: no random initialisation, no shuffling. Two
    runs on the same inputs produce identical coefficients, which is what lets
    a model be reproduced from a dataset version months later.
    """

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        *,
        learning_rate: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-4,
    ) -> None:
        self.quantiles = tuple(quantiles)
        self.lr = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: dict[float, list[float]] = {}
        self.intercepts: dict[float, float] = {}
        self.scaler = Standardiser()
        # The target is standardised too, and that is not cosmetic. The pinball
        # subgradient is +/-alpha regardless of how large the residual is - it
        # carries no scale information at all, unlike squared loss. So a fixed
        # learning rate is only meaningful relative to the target's own spread:
        # with gaps of order 0.5% and a step of 0.05, one update moves the
        # intercept further than the entire range of the data and the optimiser
        # oscillates instead of converging.
        self.y_mean = 0.0
        self.y_sigma = 1.0
        self.n_features = 0
        self.fitted = False

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> QuantileRegressor:
        if len(rows) != len(targets):
            raise ValueError("rows and targets must be the same length")
        if len(rows) < 10:
            raise ValueError(f"need at least 10 samples to fit, got {len(rows)}")

        self.n_features = len(rows[0])
        self.scaler = Standardiser().fit(rows)
        scaled = [self.scaler.apply(r) for r in rows]

        self.y_mean = fmean(targets)
        self.y_sigma = (pstdev(targets) if len(targets) > 1 else 0.0) or 1.0
        z_targets = [(y - self.y_mean) / self.y_sigma for y in targets]

        for alpha in self.quantiles:
            w = [0.0] * self.n_features
            # Start the intercept at the empirical quantile: the fit then only
            # has to learn the feature response, which converges far faster and
            # degrades gracefully when the features carry nothing.
            b = _empirical_quantile(z_targets, alpha)
            n = len(scaled)
            for _ in range(self.epochs):
                grad_w = [0.0] * self.n_features
                grad_b = 0.0
                for row, y in zip(scaled, z_targets, strict=True):
                    pred = b + sum(wi * xi for wi, xi in zip(w, row, strict=True))
                    # Subgradient of the pinball loss.
                    g = -alpha if y >= pred else (1.0 - alpha)
                    grad_b += g
                    for j, xj in enumerate(row):
                        grad_w[j] += g * xj
                b -= self.lr * grad_b / n
                for j in range(self.n_features):
                    w[j] -= self.lr * (grad_w[j] / n + self.l2 * w[j])
            self.weights[alpha] = w
            self.intercepts[alpha] = b

        self.fitted = True
        return self

    def predict(self, row: Sequence[float]) -> QuantileForecast:
        if not self.fitted:
            raise RuntimeError("fit() before predict()")
        scaled = self.scaler.apply(row)
        raw = {
            alpha: self.y_mean
            + self.y_sigma
            * (
                self.intercepts[alpha]
                + sum(w * x for w, x in zip(self.weights[alpha], scaled, strict=True))
            )
            for alpha in self.quantiles
        }
        return QuantileForecast(quantiles=enforce_monotone(raw))


def _empirical_quantile(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = alpha * (len(ordered) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


class ClimatologyForecaster:
    """Predicts the historical distribution, ignoring every feature.

    The floor under any distributional model. A forecaster that cannot beat this
    has learned nothing about conditions - it has only learned what the target
    looks like on average, which the empirical quantiles already knew.
    """

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> None:
        self.quantiles = tuple(quantiles)
        self._forecast: QuantileForecast | None = None

    def fit(
        self, _rows: Sequence[Sequence[float]], targets: Sequence[float]
    ) -> ClimatologyForecaster:
        self._forecast = QuantileForecast(
            quantiles=enforce_monotone({a: _empirical_quantile(targets, a) for a in self.quantiles})
        )
        return self

    def predict(self, _row: Sequence[float]) -> QuantileForecast:
        if self._forecast is None:
            raise RuntimeError("fit() before predict()")
        return self._forecast


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class DistributionScore:
    name: str
    n: int = 0
    crps: float = 0.0
    pinball: float = 0.0
    coverage_90: float = 0.0
    coverage_50: float = 0.0
    mean_width_90: float = 0.0
    pit_values: list[float] = field(default_factory=list)

    @property
    def coverage_error_90(self) -> float:
        return abs(self.coverage_90 - 0.90)

    @property
    def well_calibrated(self) -> bool:
        """90% band covers 90% of outcomes, within three points."""
        return self.coverage_error_90 <= 0.03

    def pit_uniformity(self, bins: int = 5) -> float:
        """Chi-square-like departure from uniform. Zero is perfect.

        A hump in the middle means intervals too wide; humps at the edges mean
        too narrow, which is the direction that loses money.
        """
        if not self.pit_values:
            return 0.0
        counts = [0] * bins
        for p in self.pit_values:
            counts[min(bins - 1, int(p * bins))] += 1
        expected = len(self.pit_values) / bins
        return sum((c - expected) ** 2 / expected for c in counts) / len(self.pit_values)

    def render(self) -> str:
        cal = "calibrated" if self.well_calibrated else "MISCALIBRATED"
        return (
            f"  {self.name:<18} n={self.n:<5} CRPS={self.crps:.6f}  "
            f"pinball={self.pinball:.6f}  cov90={self.coverage_90:.1%} ({cal})  "
            f"cov50={self.coverage_50:.1%}  width90={self.mean_width_90:.3%}"
        )


def score_forecasts(
    name: str, forecasts: Sequence[QuantileForecast], actuals: Sequence[float]
) -> DistributionScore:
    if len(forecasts) != len(actuals):
        raise ValueError("forecasts and actuals must be the same length")
    out = DistributionScore(name=name, n=len(actuals))
    if not actuals:
        return out

    crps_total = pinball_total = width_total = 0.0
    covered_90 = covered_50 = 0
    for f, y in zip(forecasts, actuals, strict=True):
        crps_total += f.crps(y)
        pinball_total += fmean(pinball_loss(y, f.quantiles[a], a) for a in sorted(f.quantiles))
        width_total += f.width()
        covered_90 += int(f.covers(y, 0.05, 0.95))
        covered_50 += int(f.covers(y, 0.25, 0.75))
        out.pit_values.append(f.pit(y))

    n = len(actuals)
    out.crps = crps_total / n
    out.pinball = pinball_total / n
    out.mean_width_90 = width_total / n
    out.coverage_90 = covered_90 / n
    out.coverage_50 = covered_50 / n
    return out


@dataclass
class DistributionComparison:
    baseline: DistributionScore
    challenger: DistributionScore

    @property
    def crps_improvement(self) -> float:
        """Relative reduction in CRPS. The gap model's gate asks for 10%."""
        if self.baseline.crps <= 0:
            return 0.0
        return (self.baseline.crps - self.challenger.crps) / self.baseline.crps

    def passes_gate(self, *, min_improvement: float = 0.10) -> bool:
        """Beating the baseline is not enough on its own.

        A model with better CRPS and broken coverage produces confident-looking
        intervals that are wrong, and a position sized off those intervals is
        sized off nothing.
        """
        return self.crps_improvement >= min_improvement and self.challenger.well_calibrated

    def render(self) -> str:
        verdict = "PASSES gate" if self.passes_gate() else "does not pass gate"
        return (
            self.baseline.render()
            + "\n"
            + self.challenger.render()
            + f"\n  CRPS improvement {self.crps_improvement:+.1%} "
            f"(need +10% and calibrated)  ->  {verdict}"
        )


# --------------------------------------------------------------------------- #
# Conformal calibration
# --------------------------------------------------------------------------- #
class ConformalQuantileForecaster:
    """Wraps a quantile model so its intervals mean what they claim.

    Quantiles fitted on ``n`` samples are systematically too tight out of
    sample - the model has seen the training tails and not the real ones - so a
    nominally 90% band routinely covers 83%. Tuning the model until coverage
    happens to land is fitting the metric rather than fixing the cause.

    Conformalized quantile regression fixes it properly. On a held-out
    calibration set, measure how far outside each predicted interval the outcome
    actually fell, then widen every interval by the appropriate empirical
    quantile of those misses. The result carries a finite-sample marginal
    coverage guarantee that does not depend on the base model being any good -
    a badly fitted model conformalises to wide, honest intervals rather than
    narrow, dishonest ones.

    The cost is real and worth stating: part of the training data is spent on
    calibration rather than fitting. With few sessions that trade is painful,
    and it is still the right one - an interval nobody can trust cannot size a
    position.
    """

    def __init__(self, base: QuantileRegressor) -> None:
        self.base = base
        self.adjustments: dict[tuple[float, float], float] = {}
        self.calibrated_on = 0

    @staticmethod
    def _pairs(levels: Sequence[float]) -> list[tuple[float, float]]:
        """Symmetric quantile pairs, widest first. The median is left alone."""
        ordered = sorted(levels)
        out: list[tuple[float, float]] = []
        i, j = 0, len(ordered) - 1
        while i < j:
            out.append((ordered[i], ordered[j]))
            i += 1
            j -= 1
        return out

    def calibrate(
        self, rows: Sequence[Sequence[float]], targets: Sequence[float]
    ) -> ConformalQuantileForecaster:
        if len(rows) != len(targets):
            raise ValueError("rows and targets must be the same length")
        if len(rows) < 20:
            raise ValueError(
                f"conformal calibration needs at least 20 held-out samples, got {len(rows)}. "
                "Fewer than that and the adjustment is itself noise."
            )

        forecasts = [self.base.predict(r) for r in rows]
        for lo_a, hi_a in self._pairs(self.base.quantiles):
            nominal = hi_a - lo_a
            # Conformity score: how far outside the interval the outcome fell.
            # Negative when it fell comfortably inside, which correctly NARROWS
            # an interval that was too wide.
            scores = [
                max(f.quantiles[lo_a] - y, y - f.quantiles[hi_a])
                for f, y in zip(forecasts, targets, strict=True)
            ]
            scores.sort()
            n = len(scores)
            # The (n+1) correction is what turns an empirical quantile into a
            # finite-sample guarantee rather than an asymptotic hope.
            rank = min(n - 1, max(0, math.ceil((n + 1) * nominal) - 1))
            self.adjustments[(lo_a, hi_a)] = scores[rank]

        self.calibrated_on = len(rows)
        log.info(
            "conformal calibration applied",
            fields={
                "samples": len(rows),
                "adjustments": {
                    f"{a:.2f}-{b:.2f}": round(v, 6) for (a, b), v in self.adjustments.items()
                },
            },
        )
        return self

    def predict(self, row: Sequence[float]) -> QuantileForecast:
        raw = self.base.predict(row)
        if not self.adjustments:
            return raw
        widened = dict(raw.quantiles)
        for (lo_a, hi_a), q in self.adjustments.items():
            widened[lo_a] = raw.quantiles[lo_a] - q
            widened[hi_a] = raw.quantiles[hi_a] + q
        return QuantileForecast(quantiles=enforce_monotone(widened))
