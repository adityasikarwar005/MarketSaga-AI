"""M60 - the path sampler.

Volatility-clustering paths with student-t innovations, conditioned on the
volatility model's forecast, then bent so their terminal distribution agrees
with the quantile heads.

Why this one module uses numpy
-------------------------------

Every other model here is deliberately dependency-free - "being able to read the
fitting code matters more than the last few percent of optimiser efficiency", as
``quantile.py`` puts it. This one is not, and the reason is measured rather than
assumed: 5 000 paths of 60 steps takes **443 ms** in pure Python against a 60 ms
budget. That is not a gap you close by tidying loops.

Vectorised it runs at a **49 ms median** on the development machine (min 36,
max 58 over five runs), of which about 22 ms is drawing the student-t
innovations - already the fastest of the three ways numpy offers to make them.
The headroom against the budget is real but thin, so the latency test guards at
twice the budget rather than at the budget itself: a timing assertion that
flakes on a loaded runner is a timing assertion somebody deletes.

Two distributions, and they are not the same distribution
-----------------------------------------------------------

A GARCH process produces paths whose *shape* is right - quiet stretches, bursts,
fat tails - and whose *terminal spread* is whatever the recursion happens to
give. The quantile heads produce a terminal distribution that has been fitted to
data and scored on pinball loss, and say nothing about the route.

Using the raw simulated terminal distribution throws away the fitted model.
Using the heads without paths gives no route to reprice along. So the paths are
simulated for their shape and then **bent to land where the heads say**: each
path keeps its increments and takes a linear drift adjustment sized to move its
terminal value onto the mapped one.

The tails are where this goes wrong quietly
---------------------------------------------

The heads cover 0.05 to 0.95. Everything outside that has to be extrapolated,
and the obvious thing - clamp to the outermost head - is the one choice that
cannot be right: it flattens both tails onto a single value, so every path in
the worst 5% ends at exactly the 5th percentile.

CVaR is the *mean of the tail*. Clamped, it becomes the 5th percentile itself,
which is the boundary of the tail rather than anything inside it. The number
still computes. It is still negative. It is uniformly too optimistic, by more
the fatter the real tail is, and nothing about the output says so.

So the tails keep the simulated distribution's own shape - which is where the
student-t fat tails live - scaled to attach continuously at the outermost heads.
``extrapolate_tails=False`` exists so a test can measure what clamping costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from libs.ms_core.obs import get_logger
from libs.ms_models.quantile import QuantileForecast

log = get_logger(__name__)

# GARCH(1,1). alpha + beta below 1 or the variance process has no stationary
# distribution and every long path drifts to infinity.
DEFAULT_ALPHA = 0.08
DEFAULT_BETA = 0.90

# Degrees of freedom for the innovations. Below about 3 the kurtosis is infinite
# and the sample moments stop meaning anything run to run; above about 10 it is
# a normal wearing a hat.
DEFAULT_NU = 5.0

MIN_PATHS_FOR_TAILS = 200


class PathError(RuntimeError):
    pass


@dataclass
class PathSet:
    """Simulated paths and the variance process that produced them.

    ``levels`` is ``(n_paths, steps + 1)`` cumulative log-return, starting at 0.
    ``variance`` is ``(n_paths, steps)`` - kept because vol clustering is a
    property of it, and checking clustering on returns alone cannot separate
    "the model has it" from "this draw happened to look like it".
    """

    levels: np.ndarray
    variance: np.ndarray
    nu: float
    calibrated: bool = False
    tail_method: str = "simulated"

    @property
    def n_paths(self) -> int:
        return int(self.levels.shape[0])

    @property
    def steps(self) -> int:
        return int(self.levels.shape[1] - 1)

    @property
    def terminal(self) -> np.ndarray:
        return self.levels[:, -1]

    def returns(self) -> np.ndarray:
        return np.diff(self.levels, axis=1)

    def quantile(self, alpha: float) -> float:
        return float(np.quantile(self.terminal, alpha))

    def render(self) -> str:
        q = [self.quantile(a) for a in (0.05, 0.25, 0.5, 0.75, 0.95)]
        return (
            f"  {self.n_paths} paths x {self.steps} steps, student-t nu={self.nu:g}"
            f"{' (calibrated)' if self.calibrated else ''}\n"
            f"    terminal q05..q95: " + "  ".join(f"{v:+.5f}" for v in q) + "\n"
            f"    tails: {self.tail_method}"
        )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(
    *,
    n_paths: int,
    steps: int,
    variance0: float,
    omega: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    nu: float = DEFAULT_NU,
    seed: int = 0,
) -> PathSet:
    """GARCH(1,1) with student-t innovations, started from the vol forecast.

    ``variance0`` is per step and comes from the volatility model - that is what
    "conditioned on M55" means here. Starting every simulation from the
    long-run variance instead would produce the same distribution on a calm
    morning and on the morning after a gap, which is the one thing a
    conditional model exists to avoid.

    ``omega`` defaults to the value that makes ``variance0`` the stationary
    variance, so a path with no shocks stays where the forecast put it.
    """
    if n_paths <= 0 or steps <= 0:
        raise PathError("n_paths and steps must both be positive")
    if variance0 <= 0.0:
        raise PathError(f"variance0 must be positive, got {variance0}")
    if alpha < 0.0 or beta < 0.0:
        raise PathError("alpha and beta must be non-negative")
    if alpha + beta >= 1.0:
        raise PathError(
            f"alpha + beta = {alpha + beta:.4f} >= 1: the variance process is not "
            "stationary and long paths diverge. The clustering would look "
            "spectacular and mean nothing."
        )
    if nu <= 2.0:
        raise PathError(
            f"nu = {nu} has infinite variance; the innovations cannot be scaled to "
            "unit variance and every moment computed from these paths is noise"
        )

    if omega is None:
        omega = variance0 * (1.0 - alpha - beta)

    rng = np.random.default_rng(seed)
    # Scaled to unit variance so `variance` means variance. A raw student-t has
    # variance nu/(nu-2), and skipping this scales every path by 1.29 at nu=5 -
    # a 29% error in every interval that looks like a modelling choice.
    innovations = rng.standard_t(nu, size=(steps, n_paths)) / math.sqrt(nu / (nu - 2.0))

    # Accumulated step-major, then transposed once. Writing `levels[:, s] = x`
    # into a (n_paths, steps) array is a strided write down a column of a
    # row-major array, and it costs more than the entire GARCH recursion: 79 ms
    # against a 60 ms budget, versus 26 ms this way. The transpose at the end is
    # one contiguous copy.
    levels_t = np.zeros((steps + 1, n_paths))
    variance_t = np.empty((steps, n_paths))
    h = np.full(n_paths, variance0)
    x = np.zeros(n_paths)
    sigma = np.empty(n_paths)
    scratch = np.empty(n_paths)

    for s in range(steps):
        z = innovations[s]
        variance_t[s] = h
        np.sqrt(h, out=sigma)
        np.multiply(sigma, z, out=sigma)
        x += sigma
        levels_t[s + 1] = x

        # h_{t+1} = omega + alpha * eps_t^2 + beta * h_t, with eps = sigma * z.
        np.multiply(z, z, out=scratch)
        np.multiply(scratch, h, out=scratch)
        np.multiply(h, beta, out=h)
        h += omega
        h += alpha * scratch

    return PathSet(
        levels=np.ascontiguousarray(levels_t.T),
        variance=np.ascontiguousarray(variance_t.T),
        nu=nu,
    )


# --------------------------------------------------------------------------- #
# Calibration to the quantile heads
# --------------------------------------------------------------------------- #
def quantile_map(
    simulated: np.ndarray,
    target: QuantileForecast,
    *,
    extrapolate_tails: bool = True,
) -> np.ndarray:
    """Map each simulated terminal onto the target distribution, rank for rank.

    Inside the head range this is piecewise-linear interpolation between the
    fitted quantiles. Outside it the simulated distribution's own tail shape is
    kept and rescaled to attach continuously at the outermost head - so the
    student-t tail survives instead of being flattened onto a single value.

    ``extrapolate_tails=False`` clamps instead, which is what makes every
    tail-risk number optimistic. It is here to be measured, not to be used.
    """
    alphas = sorted(target.quantiles)
    if len(alphas) < 2:
        raise PathError("need at least two target quantiles to build a map")
    targets = [target.quantiles[a] for a in alphas]
    if targets != sorted(targets):
        raise PathError(
            f"target quantiles are not monotone: {targets}. A non-monotone map "
            "reorders paths, and the terminal distribution stops being one."
        )

    n = simulated.size
    if n < MIN_PATHS_FOR_TAILS:
        raise PathError(
            f"{n} paths is too few to place a tail; need {MIN_PATHS_FOR_TAILS}. "
            "Below that the outer quantiles are single observations."
        )

    order = np.argsort(simulated)
    # Mid-rank plotting positions: rank 0 maps to 1/(2n), not to 0, so the most
    # extreme path is not asked for the 0th percentile of a distribution that
    # does not define one.
    u = np.empty(n)
    u[order] = (np.arange(n) + 0.5) / n

    lo_a, hi_a = alphas[0], alphas[-1]
    sim_lo = float(np.quantile(simulated, lo_a))
    sim_hi = float(np.quantile(simulated, hi_a))

    mapped = np.interp(u, alphas, targets)

    if extrapolate_tails:
        # Keep the simulated tail's shape, scaled so it meets the outermost head
        # exactly. The scale is the ratio of target to simulated distance from
        # the inner reference point, which makes the join continuous.
        lower = u < lo_a
        if lower.any():
            ref_a = alphas[1]
            sim_ref = float(np.quantile(simulated, ref_a))
            sim_span = sim_lo - sim_ref
            tgt_span = targets[0] - targets[1]
            scale = (tgt_span / sim_span) if abs(sim_span) > 1e-12 else 1.0
            mapped[lower] = targets[0] + (simulated[lower] - sim_lo) * scale

        upper = u > hi_a
        if upper.any():
            ref_a = alphas[-2]
            sim_ref = float(np.quantile(simulated, ref_a))
            sim_span = sim_hi - sim_ref
            tgt_span = targets[-1] - targets[-2]
            scale = (tgt_span / sim_span) if abs(sim_span) > 1e-12 else 1.0
            mapped[upper] = targets[-1] + (simulated[upper] - sim_hi) * scale

    return mapped


def calibrate(
    paths: PathSet, target: QuantileForecast, *, extrapolate_tails: bool = True
) -> PathSet:
    """Bend the paths so their terminal distribution matches the heads.

    Each path keeps its increments and receives a linear drift adjustment sized
    so its terminal lands on the mapped value. Rescaling multiplicatively would
    be the other option and it explodes for any path that happens to end near
    zero, which on an intraday horizon is most of them.
    """
    mapped = quantile_map(paths.terminal, target, extrapolate_tails=extrapolate_tails)
    shift = mapped - paths.terminal

    steps = paths.steps
    ramp = np.arange(steps + 1) / steps  # 0 at entry, 1 at the horizon
    levels = paths.levels + shift[:, None] * ramp[None, :]

    return PathSet(
        levels=levels,
        variance=paths.variance,
        nu=paths.nu,
        calibrated=True,
        tail_method=(
            "simulated shape, scaled to the outer heads"
            if extrapolate_tails
            else "CLAMPED - tail risk is understated"
        ),
    )


# --------------------------------------------------------------------------- #
# Measuring what the paths have
# --------------------------------------------------------------------------- #
def squared_return_autocorrelation(paths: PathSet, *, lag: int = 1) -> float:
    """ACF of squared returns. Positive and decaying is volatility clustering.

    Squared returns rather than returns: a GARCH process has uncorrelated
    returns by construction, so an ACF on the returns themselves would read
    zero and say nothing about whether the clustering is there.
    """
    r = paths.returns()
    if r.shape[1] <= lag + 1:
        raise PathError(f"need more than {lag + 1} steps to measure lag-{lag} autocorrelation")
    sq = r * r
    a = sq[:, :-lag].ravel()
    b = sq[:, lag:].ravel()
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a_c * a_c).sum()) * float((b_c * b_c).sum()))
    return float((a_c * b_c).sum() / denom) if denom > 0 else 0.0


def excess_kurtosis(paths: PathSet) -> float:
    """Of the pooled returns. A normal gives 0; fat tails give more."""
    r = paths.returns().ravel()
    c = r - r.mean()
    m2 = float((c * c).mean())
    m4 = float((c**4).mean())
    return m4 / (m2 * m2) - 3.0 if m2 > 0 else 0.0


@dataclass
class PathDiagnostics:
    n_paths: int
    steps: int
    acf1: float
    kurtosis: float
    terminal_sd: float
    calibrated: bool
    tail_method: str
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"  {self.n_paths} paths x {self.steps} steps"
            f"{' (calibrated)' if self.calibrated else ''}",
            f"    lag-1 ACF of squared returns  {self.acf1:+.4f}   "
            "(positive means volatility clusters; iid gives ~0)",
            f"    excess kurtosis               {self.kurtosis:+.3f}   " "(normal is 0)",
            f"    terminal sd                   {self.terminal_sd:.6f}",
            f"    tails                         {self.tail_method}",
        ]
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def diagnose(paths: PathSet) -> PathDiagnostics:
    acf1 = squared_return_autocorrelation(paths)
    diag = PathDiagnostics(
        n_paths=paths.n_paths,
        steps=paths.steps,
        acf1=acf1,
        kurtosis=excess_kurtosis(paths),
        terminal_sd=float(paths.terminal.std()),
        calibrated=paths.calibrated,
        tail_method=paths.tail_method,
    )
    if acf1 < 0.01:
        diag.notes.append(
            "no volatility clustering detected. These paths are effectively iid, "
            "and every path-dependent exit rule priced on them will look better "
            "than it is."
        )
    if not paths.calibrated:
        diag.notes.append(
            "not calibrated to the quantile heads - the terminal spread is "
            "whatever the GARCH recursion produced, not what the fitted model says"
        )
    return diag
