"""M62 - scoring the projection, against baselines that were given a fair chance.

Pinball, CRPS, PIT, coverage and interval width already live in ``quantile.py``.
What is here is the part that decides whether any of it is worth trading:
**three baselines, a walk-forward harness, and Gate G6.**

A baseline you did not try to make good
-----------------------------------------

The plan names random-walk and GARCH. It is easy to write a GARCH baseline with
textbook parameters, beat it comfortably, and report an 8% CRPS improvement that
is entirely an artefact of not having fitted the thing you were measuring
against. :class:`GarchBaseline` therefore fits ``alpha`` and ``beta`` by
maximising the training log-likelihood over a grid. It is still not a
state-of-the-art volatility model, and it is not supposed to be - but it is a
baseline that was given a chance to win.

The option-implied baseline is not measuring what the other two measure
-------------------------------------------------------------------------

This is the one to be careful with. The distribution implied by option prices is
**risk-neutral**, not physical. It carries a variance risk premium: on index
options it is systematically wider than the distribution outcomes actually come
from, because sellers charge for bearing the risk.

So a model that beats the implied distribution on CRPS has not necessarily
forecast better. It may only have noticed that implied volatility runs above
realised, which is a well-documented feature of the market and not a discovery.
The premium is worth harvesting, but harvesting it is a *volatility selling*
strategy with the risk profile of one, and it must not be reported as
distributional forecast skill.

:class:`ImpliedBaseline` carries that caveat in its own comparison output rather
than in a comment somewhere, because the number it produces is the one most
likely to look like the strongest result.

Both, not the average
-----------------------

Gate G6 asks for CRPS to beat both baselines by 8%. That is each of them
separately. Beating the mean of three baselines lets a model lose badly to one
and still pass, and the one it loses to is the informative one.

How much headroom 8% actually is
----------------------------------

Worth measuring before anyone tunes toward it. An **oracle** challenger - one
handed the true conditional variance of the process - was scored against these
baselines on GARCH-generated series:

======  ===============  ==========
seed    vs random-walk   vs GARCH
======  ===============  ==========
9       +2.81%           +0.22%
17      +0.61%           -0.01%
23      +0.64%           +0.10%
======  ===============  ==========

A perfect forecast beats a *fitted* GARCH by essentially nothing on data a
GARCH generated, which is not surprising once stated: there, the fitted GARCH
is the oracle. The 8% is only reachable if the real series carries structure
GARCH misses - event-conditioned bursts, intraday seasonality, jumps - which is
exactly what the gate is asking about.

The number to be suspicious of is therefore a comfortable pass. On a series
with no such structure, an 8% margin over a fitted GARCH is more likely a
baseline that was not fitted than a model that is good.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

from libs.ms_core.obs import get_logger
from libs.ms_models.quantile import (
    DEFAULT_QUANTILES,
    DistributionScore,
    QuantileForecast,
    enforce_monotone,
    pit_bucket_probabilities,
    score_forecasts,
)

log = get_logger(__name__)

# Gate G6, from the plan.
COVERAGE_TOLERANCE = 0.03  # 90% band must cover 90% +/- 3 points
MIN_CRPS_IMPROVEMENT = 0.08  # against every baseline, separately
PIT_ALPHA = 0.05  # significance for the uniformity test

MIN_EVAL_SAMPLES = 200

_ND = NormalDist()


class ProjectionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# The PIT test
# --------------------------------------------------------------------------- #
def chi_square_critical(dof: int, p: float = 0.95) -> float:
    """Upper-tail chi-square critical value, Wilson-Hilferty.

    Accurate to about 0.3% in the range that matters here (dof 3-10), which is
    checked against the published table in the tests. Written out rather than
    imported because scipy is not a dependency and a five-line approximation is
    easier to audit than a table nobody re-derives.
    """
    if dof < 1:
        raise ProjectionError("degrees of freedom must be at least 1")
    z = _ND.inv_cdf(p)
    return dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3


@dataclass(frozen=True)
class PitTest:
    """Is the probability integral transform uniform, as a decision?

    A statistic without a critical value is a number nobody can act on, and the
    gate says "PIT uniform" rather than "PIT statistic small".
    """

    statistic: float
    dof: int
    critical: float
    n: int

    @property
    def uniform(self) -> bool:
        return self.statistic <= self.critical

    def render(self) -> str:
        verdict = "uniform" if self.uniform else "NOT uniform"
        return (
            f"    PIT chi2 {self.statistic:9.2f} on {self.dof} dof "
            f"(critical {self.critical:.2f} at p={PIT_ALPHA})  {verdict}"
        )


def pit_test(score: DistributionScore, *, alpha: float = PIT_ALPHA) -> PitTest:
    if not score.pit_values or not score.alphas:
        raise ProjectionError("no PIT values to test")
    dof = len(pit_bucket_probabilities(score.alphas)) - 1
    return PitTest(
        statistic=score.pit_uniformity(),
        dof=dof,
        critical=chi_square_critical(dof, 1.0 - alpha),
        n=score.n,
    )


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def gaussian_forecast(
    sigma: float, *, mu: float = 0.0, quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> QuantileForecast:
    return QuantileForecast(
        quantiles=enforce_monotone({a: mu + sigma * _ND.inv_cdf(a) for a in sorted(quantiles)})
    )


class RandomWalkBaseline:
    """Tomorrow looks like the recent past. Constant variance, no conditioning.

    The floor. A distributional model that cannot beat an unconditional Gaussian
    fitted on a rolling window has not learned anything about conditions.
    """

    name = "random-walk"

    def __init__(self, *, window: int = 250, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.window = window
        self.quantiles = tuple(quantiles)
        self._sigma = 0.0

    def fit(self, history: Sequence[float]) -> RandomWalkBaseline:
        recent = list(history[-self.window :])
        if len(recent) < 2:
            raise ProjectionError("need at least two observations")
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / (len(recent) - 1)
        self._sigma = math.sqrt(max(var, 0.0))
        return self

    def predict(self) -> QuantileForecast:
        return gaussian_forecast(self._sigma, quantiles=self.quantiles)


class GarchBaseline:
    """GARCH(1,1) with parameters fitted, not assumed.

    Fitting matters. With textbook constants this is a baseline that was never
    given a chance, and an 8% improvement over it says more about the constants
    than about the model being tested.

    The fit is a grid search on the Gaussian log-likelihood - coarse, but it is
    a real search over the parameters that drive the variance recursion, and it
    is auditable in a way a black-box optimiser is not.
    """

    name = "garch(1,1)"

    def __init__(self, *, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.quantiles = tuple(quantiles)
        self.alpha = 0.0
        self.beta = 0.0
        self.omega = 0.0
        self._h = 0.0
        self.log_likelihood = -math.inf

    @staticmethod
    def _log_likelihood(returns: Sequence[float], omega: float, a: float, b: float) -> float:
        h = sum(r * r for r in returns) / len(returns)
        total = 0.0
        for r in returns:
            if h <= 0.0:
                return -math.inf
            total += -0.5 * (math.log(2.0 * math.pi * h) + r * r / h)
            h = omega + a * r * r + b * h
        return total

    def fit(self, history: Sequence[float]) -> GarchBaseline:
        if len(history) < 30:
            raise ProjectionError(
                f"{len(history)} observations is not enough to fit a GARCH baseline. "
                "An unfitted baseline is one the model beats by construction."
            )
        unconditional = sum(r * r for r in history) / len(history)
        best = (-math.inf, 0.05, 0.90)
        for a in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25):
            for b in (0.60, 0.70, 0.80, 0.85, 0.90, 0.94):
                if a + b >= 0.999:
                    continue
                omega = unconditional * (1.0 - a - b)
                ll = self._log_likelihood(history, omega, a, b)
                if ll > best[0]:
                    best = (ll, a, b)
        self.log_likelihood, self.alpha, self.beta = best
        self.omega = unconditional * (1.0 - self.alpha - self.beta)

        h = unconditional
        for r in history:
            h = self.omega + self.alpha * r * r + self.beta * h
        self._h = h
        return self

    def predict(self) -> QuantileForecast:
        return gaussian_forecast(math.sqrt(max(self._h, 0.0)), quantiles=self.quantiles)


class ImpliedBaseline:
    """The option market's own distribution. **Risk-neutral, not physical.**

    Beating this on CRPS is not evidence of forecast skill on its own. Index
    options carry a variance risk premium, so the implied distribution is
    systematically wider than the one outcomes come from - and a model that
    merely notices implied vol runs above realised will beat it every time
    without having forecast anything.

    That is a volatility-selling edge with a volatility seller's risk profile:
    small, steady gains and an occasional very large loss. It is worth having
    and it must not be reported as distributional skill, so
    :class:`BaselineComparison` says so wherever this baseline is the one being
    beaten.
    """

    name = "option-implied"
    risk_neutral = True

    def __init__(self, *, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.quantiles = tuple(quantiles)
        self._sigma = 0.0

    def fit_from_iv(self, iv: float, *, horizon_years: float) -> ImpliedBaseline:
        if iv < 0.0 or horizon_years <= 0.0:
            raise ProjectionError("iv must be non-negative and the horizon positive")
        self._sigma = iv * math.sqrt(horizon_years)
        return self

    def predict(self) -> QuantileForecast:
        return gaussian_forecast(self._sigma, quantiles=self.quantiles)


# --------------------------------------------------------------------------- #
# Comparison and the gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineComparison:
    baseline: DistributionScore
    challenger: DistributionScore
    risk_neutral: bool = False

    @property
    def crps_improvement(self) -> float:
        if self.baseline.crps <= 0.0:
            return 0.0
        return (self.baseline.crps - self.challenger.crps) / self.baseline.crps

    @property
    def beaten(self) -> bool:
        return self.crps_improvement >= MIN_CRPS_IMPROVEMENT

    def body(self) -> str:
        """The comparison without a leading indent, so a caller can prefix it."""
        mark = "beaten" if self.beaten else "NOT beaten"
        line = (
            f"vs {self.baseline.name:<16} CRPS {self.baseline.crps:.6f} -> "
            f"{self.challenger.crps:.6f}  {self.crps_improvement:+.1%}  {mark}"
        )
        if self.risk_neutral and self.beaten:
            line += (
                "\n      This baseline is risk-neutral. Beating it may be the variance "
                "risk premium rather than forecast skill - a volatility-selling edge "
                "with a volatility seller's tail, not a better distribution."
            )
        return line

    def render(self) -> str:
        return f"    {self.body()}"


@dataclass
class GateG6:
    """90% coverage within 3 points, PIT uniform, CRPS beats every baseline by 8%."""

    challenger: DistributionScore
    pit: PitTest
    comparisons: list[BaselineComparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def coverage_ok(self) -> bool:
        return abs(self.challenger.coverage_90 - 0.90) <= COVERAGE_TOLERANCE

    @property
    def unbeaten(self) -> list[str]:
        return [c.baseline.name for c in self.comparisons if not c.beaten]

    @property
    def passes(self) -> bool:
        return (
            self.coverage_ok
            and self.pit.uniform
            and bool(self.comparisons)
            and not self.unbeaten
            and self.challenger.n >= MIN_EVAL_SAMPLES
        )

    def render(self) -> str:
        lines = [f"  GATE G6 - {self.challenger.name}, n={self.challenger.n}"]
        lines.append(
            f"    [{'PASS' if self.coverage_ok else 'FAIL'}]  90% coverage "
            f"{self.challenger.coverage_90:.1%} within {COVERAGE_TOLERANCE:.0%} of 90%"
        )
        lines.append(f"    [{'PASS' if self.pit.uniform else 'FAIL'}]  {self.pit.render().strip()}")
        for c in self.comparisons:
            lines.append(f"    [{'PASS' if c.beaten else 'FAIL'}]  {c.body()}")
        if self.challenger.n < MIN_EVAL_SAMPLES:
            lines.append(
                f"    [FAIL]  {self.challenger.n} evaluated, needs {MIN_EVAL_SAMPLES} - "
                "below that the coverage estimate is wider than the tolerance it is "
                "being checked against"
            )
        if not self.comparisons:
            lines.append(
                "    [FAIL]  no baselines were scored. A gate with nothing to beat "
                "is a gate that passes."
            )
        lines.append(f"  -> Gate G6 {'PASSES' if self.passes else 'does not pass'}")
        if self.unbeaten:
            lines.append(
                f"     unbeaten: {', '.join(self.unbeaten)}. Every baseline separately, "
                "not their average - beating a mean lets a model lose badly to one, "
                "and the one it loses to is the informative one."
            )
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def evaluate_gate(
    challenger: DistributionScore,
    baselines: Sequence[tuple[DistributionScore, bool]],
) -> GateG6:
    """``baselines`` pairs each score with whether it is risk-neutral."""
    gate = GateG6(challenger=challenger, pit=pit_test(challenger))
    for score, risk_neutral in baselines:
        if score.n != challenger.n:
            raise ProjectionError(
                f"{score.name} was scored on {score.n} outcomes and the challenger on "
                f"{challenger.n}. Comparing CRPS across different samples compares two "
                "models on two different questions."
            )
        gate.comparisons.append(
            BaselineComparison(baseline=score, challenger=challenger, risk_neutral=risk_neutral)
        )
    log.info(
        "gate G6 evaluated",
        fields={"passes": gate.passes, "unbeaten": gate.unbeaten, "n": challenger.n},
    )
    return gate


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    challenger: DistributionScore
    random_walk: DistributionScore
    garch: DistributionScore
    implied: DistributionScore | None
    gate: GateG6

    def render(self) -> str:
        lines = ["  walk-forward evaluation", self.challenger.render(), self.random_walk.render()]
        lines.append(self.garch.render())
        if self.implied is not None:
            lines.append(self.implied.render())
        lines.append("")
        lines.append(self.gate.render())
        return "\n".join(lines)


def walk_forward(
    history: Sequence[float],
    challenger_forecasts: Sequence[QuantileForecast],
    *,
    warmup: int,
    implied_vols: Sequence[float] | None = None,
    horizon_years: float = 1.0,
    name: str = "challenger",
) -> WalkForwardResult:
    """Score the challenger and every baseline on the same out-of-sample points.

    Each baseline is refitted on the history **up to** each evaluation point and
    never past it. Fitting once on everything would let the baselines see the
    outcomes they are being scored on, which makes them look better and the
    challenger's margin smaller - the error that flatters the wrong side and so
    never gets questioned.
    """
    if warmup < 30:
        raise ProjectionError("warmup must be at least 30 observations to fit a GARCH baseline")
    n_eval = len(history) - warmup
    if n_eval <= 0:
        raise ProjectionError(f"{len(history)} observations with warmup {warmup} leaves nothing")
    if len(challenger_forecasts) != n_eval:
        raise ProjectionError(
            f"{len(challenger_forecasts)} challenger forecasts for {n_eval} evaluation "
            "points. They must line up one for one or the scores are of different sets."
        )
    if implied_vols is not None and len(implied_vols) != n_eval:
        raise ProjectionError(f"{len(implied_vols)} implied vols for {n_eval} evaluation points")

    actuals: list[float] = []
    rw_forecasts: list[QuantileForecast] = []
    garch_forecasts: list[QuantileForecast] = []
    implied_forecasts: list[QuantileForecast] = []

    for i in range(n_eval):
        seen = history[: warmup + i]  # everything up to, not including, the outcome
        actuals.append(history[warmup + i])
        rw_forecasts.append(RandomWalkBaseline().fit(seen).predict())
        garch_forecasts.append(GarchBaseline().fit(seen).predict())
        if implied_vols is not None:
            implied_forecasts.append(
                ImpliedBaseline()
                .fit_from_iv(implied_vols[i], horizon_years=horizon_years)
                .predict()
            )

    challenger = score_forecasts(name, challenger_forecasts, actuals)
    rw = score_forecasts(RandomWalkBaseline.name, rw_forecasts, actuals)
    garch = score_forecasts(GarchBaseline.name, garch_forecasts, actuals)
    implied = (
        score_forecasts(ImpliedBaseline.name, implied_forecasts, actuals)
        if implied_vols is not None
        else None
    )

    pairs: list[tuple[DistributionScore, bool]] = [(rw, False), (garch, False)]
    if implied is not None:
        pairs.append((implied, True))

    gate = evaluate_gate(challenger, pairs)
    if implied is None:
        gate.notes.append(
            "no implied vols supplied, so the option-implied baseline was not scored. "
            "Gate G6 names it, and passing without it is passing two thirds of the gate."
        )
    return WalkForwardResult(
        challenger=challenger, random_walk=rw, garch=garch, implied=implied, gate=gate
    )
