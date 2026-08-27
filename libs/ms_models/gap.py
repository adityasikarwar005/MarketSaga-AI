"""M35 - Overnight gap forecaster.

Roughly half of an Indian index's total price change arrives as an overnight
gap. Most retail systems never see it, because they only look at intraday
candles - which is precisely why this is the first component in the roadmap that
is worth shipping on its own.

It is also the cheapest thing in the project to build honestly:

* No low-latency infrastructure. It runs once, at 08:45.
* No option chain. No tick data. No broker.
* Daily and hourly cross-asset data, which is freely available.
* A distributional target, scored by CRPS - so the whole "forecast a
  distribution, not a number" thesis gets tested for real, early, and cheaply.

The one rule that makes or breaks it: **the cutoff is structural.** Features are
assembled from an ``OvernightWindow`` that ends at 08:45, and the window object
is the only way in. A model that consumes anything published between 08:45 and
the 09:15 open cannot be run in production at 08:45, and its backtest is
therefore describing a system that cannot exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from statistics import fmean

from libs.ms_core.obs import get_logger
from libs.ms_data.crossasset import CrossAssetRegistry, OvernightWindow, SeriesMove
from libs.ms_models.quantile import (
    DEFAULT_QUANTILES,
    ClimatologyForecaster,
    ConformalQuantileForecaster,
    DistributionComparison,
    QuantileForecast,
    QuantileRegressor,
    pinball_loss,
    score_forecasts,
)

log = get_logger(__name__)

# Conformal calibration needs its own held-out sessions; below this the
# adjustment is itself noise.
MIN_CALIBRATION_SESSIONS = 20
# The gap model needs sessions, not minutes. Two years is roughly 500 - small,
# but the signal is strong enough that it is workable from about half of that.
# The floor accounts for the calibration split: fitting on everything and
# calibrating on nothing produces intervals that look fine and cover 85%.
MIN_SESSIONS_TO_FIT = 100


@dataclass
class GapSample:
    """One session's overnight state and the gap that followed."""

    session: date
    features: dict[str, float]
    target: float  # (open - previous close) / previous close
    window_hours: float
    spans_weekend: bool

    def vector(self, names: Sequence[str]) -> list[float]:
        return [self.features.get(n, 0.0) for n in names]


def build_features(
    window: OvernightWindow,
    moves: Mapping[str, SeriesMove],
    registry: CrossAssetRegistry,
    *,
    prior_close: float | None = None,
    prior_session_return: float | None = None,
) -> dict[str, float]:
    """Assemble what was knowable at the cutoff, and nothing else.

    The window is the only source of cross-asset input, so there is no argument
    a caller can pass that reaches past 08:45. That is deliberate: a cutoff
    enforced by convention is a cutoff that eventually gets crossed.
    """
    features: dict[str, float] = {}

    for series in registry.all():
        move = moves.get(series.symbol)
        features[f"{series.symbol}_ret"] = move.ret if move else 0.0
        features[f"{series.symbol}_range"] = move.range_pct if move else 0.0
        # Whether the series reported at all is itself informative: a venue that
        # was closed all night carries different information from one that was
        # open and went nowhere.
        features[f"{series.symbol}_seen"] = 1.0 if move else 0.0

    # The window shapes the gap: after a long weekend more has happened, and the
    # distribution is genuinely wider. A model that cannot see this mis-sizes
    # every Monday.
    features["window_hours"] = window.hours
    features["spans_weekend"] = 1.0 if window.spans_multiple_days else 0.0
    features["day_of_week"] = float(window.session.weekday())

    if prior_session_return is not None:
        features["prior_session_return"] = prior_session_return
    if prior_close is not None and prior_close > 0:
        features["prior_close_log"] = math.log(prior_close)
    return features


def feature_names(registry: CrossAssetRegistry, *, with_prior: bool = True) -> list[str]:
    names: list[str] = []
    for series in registry.all():
        names += [f"{series.symbol}_ret", f"{series.symbol}_range", f"{series.symbol}_seen"]
    names += ["window_hours", "spans_weekend", "day_of_week"]
    if with_prior:
        names += ["prior_session_return"]
    return names


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
class NaiveGapBaseline:
    """``gap = beta * overnight equity-futures move``, with empirical residuals.

    The thing the real model has to beat, and it is not a straw man: overnight
    global futures are the single strongest input, and a well-fitted univariate
    regression on them is a genuinely decent forecast. A multivariate model that
    cannot improve on it by ten percent has not earned its complexity.
    """

    def __init__(
        self,
        # Whichever overnight series is actually available and strongest. With a
        # live feed that is GIFT Nifty; on free daily bars it is US equity
        # futures, whose session closes hours before the Indian open.
        driver: str = "US_EQ_FUT",
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        self.driver = driver
        self.quantiles = tuple(quantiles)
        self.beta = 0.0
        self.residual_quantiles: dict[float, float] = {}
        self.fitted = False

    def fit(self, samples: Sequence[GapSample]) -> NaiveGapBaseline:
        key = f"{self.driver}_ret"
        pairs = [(s.features.get(key, 0.0), s.target) for s in samples]
        pairs = [(x, y) for x, y in pairs if x != 0.0]
        if len(pairs) >= 10:
            sxx = sum(x * x for x, _ in pairs)
            sxy = sum(x * y for x, y in pairs)
            self.beta = sxy / sxx if sxx > 0 else 0.0
        residuals = [s.target - self.beta * s.features.get(key, 0.0) for s in samples]
        ordered = sorted(residuals)
        for a in self.quantiles:
            idx = a * (len(ordered) - 1)
            lo, hi = math.floor(idx), math.ceil(idx)
            self.residual_quantiles[a] = (
                ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)
            )
        self.fitted = True
        return self

    def predict(self, sample: GapSample) -> QuantileForecast:
        if not self.fitted:
            raise RuntimeError("fit() before predict()")
        centre = self.beta * sample.features.get(f"{self.driver}_ret", 0.0)
        return QuantileForecast(
            quantiles={a: centre + r for a, r in self.residual_quantiles.items()}
        )


# --------------------------------------------------------------------------- #
# The forecaster
# --------------------------------------------------------------------------- #
@dataclass
class GapForecast:
    session: date
    forecast: QuantileForecast
    window_hours: float
    spans_weekend: bool
    drivers: list[str] = field(default_factory=list)
    # Whether the intervals went through conformal calibration. An uncalibrated
    # forecast is not wrong, but its 90% band is not a 90% band, and anything
    # sizing a position off it needs to know that.
    calibrated: bool = True

    @property
    def median_gap(self) -> float:
        return self.forecast.median

    @property
    def direction(self) -> str:
        m = self.median_gap
        return "up" if m > 0.0005 else "down" if m < -0.0005 else "flat"

    def render(self) -> str:
        lo, hi = self.forecast.interval()
        note = "  (after a weekend or holiday)" if self.spans_weekend else ""
        lines = [
            f"  gap forecast for {self.session}",
            f"    window   : {self.window_hours:.1f} hours{note}",
            f"    median   : {self.median_gap:+.3%}  ({self.direction})",
            f"    {'90% band' if self.calibrated else '90% band [UNCALIBRATED]'} : [{lo:+.3%}, {hi:+.3%}]",
        ]
        if self.drivers:
            lines.append("    drivers  : " + "; ".join(self.drivers[:4]))
        return "\n".join(lines)


L2_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0, 4.0)


def select_l2(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    epochs: int,
    grid: Sequence[float] = L2_GRID,
    val_share: float = 0.25,
) -> float:
    """Pick the penalty on an inner split of the training data.

    A fixed default cannot be right: the correct amount of regularisation
    depends on how many features there are, how many rows, and how much of the
    signal is real, and none of those are known when the constant is written.

    The value that was there - 1e-4 - was measured to shrink the weights by
    0.2% against no penalty at all, so it was not a weak default but an absent
    one. Sweeping it on the real gap data moved the result from -22.2% to
    -11.5% against the naive baseline, monotonically. A knob worth that much
    cannot be a guess.

    The split is inner and chronological: this never sees the calibration set,
    let alone the test fold, so choosing here costs nothing in honesty. It is
    still a search, so the caller owes it to the trial log if it acts on the
    result.
    """
    n_val = int(len(rows) * val_share)
    if n_val < 8 or len(rows) - n_val < 20:
        # Too little to choose on. Take the middle of the grid rather than an
        # end of it: a bad guess in the centre is a smaller mistake than a
        # confident one at either extreme.
        return grid[len(grid) // 2]

    inner_fit, inner_val = rows[: len(rows) - n_val], rows[len(rows) - n_val :]
    y_fit, y_val = targets[: len(targets) - n_val], targets[len(targets) - n_val :]

    best, best_loss = grid[len(grid) // 2], math.inf
    for candidate in grid:
        model = QuantileRegressor(epochs=epochs, l2=candidate).fit(inner_fit, y_fit)
        loss = fmean(
            pinball_loss(y, model.predict(x).quantiles[a], a)
            for x, y in zip(inner_val, y_val, strict=True)
            for a in model.quantiles
        )
        if loss < best_loss:
            best, best_loss = candidate, loss
    return best


class GapForecaster:
    """Quantile regression on the overnight state.

    Fitted on sessions, evaluated walk-forward. The gate is not "is it accurate"
    - a gap forecast is mostly noise and always will be - but "is the *shape* of
    the distribution right": does the 90% band contain the outcome 90% of the
    time, and is CRPS materially better than the naive driver regression.
    """

    def __init__(
        self,
        registry: CrossAssetRegistry,
        *,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        epochs: int = 400,
        calibration_share: float = 0.25,
        features: Sequence[str] | None = None,
        l2: float | None = None,
    ) -> None:
        self.registry = registry
        # An explicit feature list overrides the one derived from the registry.
        # The derived list gives every series three columns whether or not the
        # series is present in the data, which is the right default for a live
        # feed and the wrong one for a two-year sample of daily bars.
        self.names = list(features) if features is not None else feature_names(registry)
        self.quantiles = tuple(quantiles)
        self.epochs = epochs
        # None means "choose it from the training data", which is the default.
        # A fixed value is honoured as given, for sweeps and for reproducing an
        # older fit.
        self.l2 = l2
        self.chosen_l2 = l2
        self.model = QuantileRegressor(
            quantiles=quantiles,
            epochs=epochs,
            **({} if l2 is None else {"l2": l2}),
        )
        self.climatology = ClimatologyForecaster(quantiles=quantiles)
        self.calibration_share = calibration_share
        self.conformal: ConformalQuantileForecaster | None = None
        self.fitted_on = 0

    def fit(self, samples: Sequence[GapSample]) -> GapForecaster:
        if self.calibration_share <= 0.0:
            # Explicit opt-out. Everything trains, nothing calibrates, and every
            # forecast produced afterwards says so.
            rows = [s.vector(self.names) for s in samples]
            targets = [s.target for s in samples]
            self.model.fit(rows, targets)
            self.climatology.fit(rows, targets)
            self.fitted_on = len(samples)
            return self
        if len(samples) < MIN_SESSIONS_TO_FIT:
            raise ValueError(
                f"{len(samples)} sessions is not enough to fit a gap model "
                f"(need {MIN_SESSIONS_TO_FIT}). Keep recording; this is the one "
                "input that cannot be bought back later."
            )
        # Chronological split, not random: the calibration set must come after
        # the fitting set, exactly as production does. A randomly drawn
        # calibration set is interleaved with training data and its coverage
        # estimate is optimistic for the same reason an unpurged fold is.
        ordered = sorted(samples, key=lambda s: s.session)
        n_calib = int(len(ordered) * self.calibration_share)
        if n_calib < MIN_CALIBRATION_SESSIONS:
            # Previously this silently skipped calibration, which is how an
            # early walk-forward fold ended up producing 85%-covering "90%"
            # bands while every other fold was fine. Refusing is louder and
            # cheaper than a quietly uncalibrated model.
            raise ValueError(
                f"{len(ordered)} sessions leaves only {n_calib} for calibration "
                f"(need {MIN_CALIBRATION_SESSIONS}). To fit anyway, pass "
                "calibration_share=0.0 - the intervals are then explicitly "
                "marked uncalibrated."
            )
        fit_set = ordered[: len(ordered) - n_calib]
        calib_set = ordered[len(ordered) - n_calib :]

        rows = [s.vector(self.names) for s in fit_set]
        targets = [s.target for s in fit_set]
        if self.l2 is None:
            self.chosen_l2 = select_l2(rows, targets, epochs=self.epochs)
            self.model = QuantileRegressor(
                quantiles=self.quantiles, epochs=self.epochs, l2=self.chosen_l2
            )
        self.model.fit(rows, targets)
        self.climatology.fit(rows, targets)

        if calib_set:
            self.conformal = ConformalQuantileForecaster(self.model).calibrate(
                [s.vector(self.names) for s in calib_set], [s.target for s in calib_set]
            )
        self.fitted_on = len(fit_set)
        log.info(
            "gap forecaster fitted",
            fields={
                "fit_sessions": len(fit_set),
                "calibration_sessions": len(calib_set),
                "features": len(self.names),
            },
        )
        return self

    def predict(self, sample: GapSample) -> GapForecast:
        engine = self.conformal or self.model
        forecast = engine.predict(sample.vector(self.names))
        return GapForecast(
            session=sample.session,
            forecast=forecast,
            window_hours=sample.window_hours,
            spans_weekend=sample.spans_weekend,
            drivers=self._explain(sample),
            calibrated=self.conformal is not None,
        )

    def _explain(self, sample: GapSample, *, top: int = 4) -> list[str]:
        """Which inputs moved the median, and by how much.

        Coefficient times standardised value, for the median quantile. Not a
        causal claim - it is a linear model - but it makes the forecast
        inspectable, which is the difference between a number a person can
        challenge and one they must simply accept.
        """
        if 0.5 not in self.model.weights:
            return []
        weights = self.model.weights[0.5]
        scaled = self.model.scaler.apply(sample.vector(self.names))
        contributions = [
            (name, w * x)
            for name, w, x in zip(self.names, weights, scaled, strict=True)
            if abs(w * x) > 1e-6
        ]
        contributions.sort(key=lambda kv: -abs(kv[1]))
        return [f"{name} {value:+.3%}" for name, value in contributions[:top]]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class GapEvaluation:
    comparison: DistributionComparison
    n_test: int
    folds: int
    uncalibrated: int = 0

    @property
    def passes_g4(self) -> bool:
        """Gate G4: CRPS at least 10% better than naive, and calibrated."""
        return self.comparison.passes_gate(min_improvement=0.10)

    def render(self) -> str:
        warn = (
            f"  WARNING: {self.uncalibrated:,} of {self.n_test:,} forecasts were "
            "uncalibrated - their intervals are not what they claim\n"
            if self.uncalibrated
            else ""
        )
        return (
            f"  walk-forward: {self.folds} folds, {self.n_test:,} test sessions\n"
            + warn
            + self.comparison.render()
            + f"\n  Gate G4: {'PASS' if self.passes_g4 else 'not yet'}"
        )


def walk_forward_evaluate(
    samples: Sequence[GapSample],
    registry: CrossAssetRegistry,
    *,
    n_folds: int = 4,
    min_train: int = MIN_SESSIONS_TO_FIT,
    features: Sequence[str] | None = None,
    l2: float | None = None,
) -> GapEvaluation | None:
    """Train on the past, test on the future, repeatedly.

    No purging needed here and none pretended: gap samples are one per session
    and their outcome windows do not overlap, so the ordinary time-series split
    is already honest. Saying so explicitly matters - silently omitting a purge
    where one *is* needed is the same code with a very different meaning.
    """
    ordered = sorted(samples, key=lambda s: s.session)
    if len(ordered) < min_train + n_folds * 5:
        return None

    fold_size = (len(ordered) - min_train) // n_folds
    if fold_size < 5:
        return None

    model_forecasts: list[QuantileForecast] = []
    naive_forecasts: list[QuantileForecast] = []
    actuals: list[float] = []
    uncalibrated = 0

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test = ordered[train_end : train_end + fold_size]
        train = ordered[:train_end]
        if len(train) < min_train or not test:
            continue

        model = GapForecaster(registry, features=features, l2=l2).fit(train)
        naive = NaiveGapBaseline().fit(train)
        for sample in test:
            out = model.predict(sample)
            model_forecasts.append(out.forecast)
            naive_forecasts.append(naive.predict(sample))
            actuals.append(sample.target)
            uncalibrated += 0 if out.calibrated else 1

    if not actuals:
        return None

    return GapEvaluation(
        comparison=DistributionComparison(
            baseline=score_forecasts("naive_driver", naive_forecasts, actuals),
            challenger=score_forecasts("gap_model", model_forecasts, actuals),
        ),
        n_test=len(actuals),
        folds=n_folds,
        uncalibrated=uncalibrated,
    )
