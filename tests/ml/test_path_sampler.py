"""M60 - the path sampler.

    pytest tests/ml/test_path_sampler.py -k "terminal_matches_quantiles or vol_clustering_present or latency"

Three things have to be true at once and each can be false while the other two
look fine: the paths have to cluster (or every path-dependent exit rule prices
better than it will trade), they have to land where the fitted heads say (or the
whole quantile model is thrown away at the last step), and it has to happen fast
enough to run per decision.

The fourth thing, which the verify command does not name and which is the one
that fails silently, is the tails.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from libs.ms_models.paths import (
    DEFAULT_NU,
    MIN_PATHS_FOR_TAILS,
    PathError,
    PathSet,
    calibrate,
    diagnose,
    excess_kurtosis,
    quantile_map,
    simulate,
    squared_return_autocorrelation,
)
from libs.ms_models.quantile import QuantileForecast

VAR0 = 1e-6  # ~0.1% per step
SEED = 7

TARGET = QuantileForecast(
    quantiles={0.05: -0.020, 0.25: -0.006, 0.50: 0.0008, 0.75: 0.008, 0.95: 0.024}
)


def paths(**kw) -> PathSet:
    base = {"n_paths": 4000, "steps": 60, "variance0": VAR0, "seed": SEED}
    return simulate(**{**base, **kw})


def cvar(p: PathSet, alpha: float = 0.05) -> float:
    t = p.terminal
    return float(t[t <= np.quantile(t, alpha)].mean())


# --------------------------------------------------------------------------- #
# vol_clustering_present
# --------------------------------------------------------------------------- #
def test_vol_clustering_present_in_the_squared_returns():
    """Measured on squared returns, not returns. A GARCH process has
    uncorrelated returns by construction, so an ACF on the returns themselves
    reads zero and says nothing about whether the clustering is there."""
    acf = squared_return_autocorrelation(paths())
    assert acf > 0.05, f"lag-1 ACF of squared returns is {acf:.4f}; these paths are iid"


def test_vol_clustering_present_only_because_the_model_puts_it_there():
    """The control. Same generator, same seed, alpha=beta=0 - which is iid by
    construction - must show no clustering. Without this the test above passes
    on any process with fat tails."""
    clustered = squared_return_autocorrelation(paths())
    flat = squared_return_autocorrelation(paths(alpha=0.0, beta=0.0))

    assert abs(flat) < 0.02, f"the iid control clustered at {flat:.4f}"
    assert clustered > flat * 5


def test_vol_clustering_present_and_reported_when_it_is_not():
    """A diagnostic that only ever says "fine" is not a diagnostic."""
    diag = diagnose(paths(alpha=0.0, beta=0.0))
    assert any("no volatility clustering" in n for n in diag.notes)

    assert not any("no volatility clustering" in n for n in diagnose(paths()).notes)


def test_the_innovations_are_fat_tailed():
    """Student-t is the point: a normal-innovation GARCH understates how often
    the market does something abrupt, which is exactly when a stop gets run."""
    assert excess_kurtosis(paths()) > 1.0
    assert excess_kurtosis(paths(nu=200.0)) < 1.5, "nu=200 should be close to normal"


def test_the_innovations_are_scaled_to_unit_variance():
    """A raw student-t has variance nu/(nu-2). Skipping the scaling multiplies
    every path by 1.29 at nu=5 - a 29% error in every interval, and one that
    looks like a modelling choice rather than a bug."""
    p = paths(alpha=0.0, beta=0.0, n_paths=20000, steps=30)
    # With alpha=beta=0 the variance is constant at variance0, so the terminal
    # variance is exactly steps * variance0.
    expected_sd = float(np.sqrt(30 * VAR0))
    assert float(p.terminal.std()) == pytest.approx(expected_sd, rel=0.05)


def test_the_variance_process_is_kept_not_just_the_returns():
    p = paths()
    assert p.variance.shape == (p.n_paths, p.steps)
    assert (p.variance > 0).all()


# --------------------------------------------------------------------------- #
# terminal_matches_quantiles
# --------------------------------------------------------------------------- #
def test_terminal_matches_quantiles_after_calibration():
    """The whole reason calibration exists: without it the terminal spread is
    whatever the GARCH recursion happened to produce, and the fitted quantile
    model is thrown away at the last step."""
    c = calibrate(paths(), TARGET)
    for alpha, want in TARGET.quantiles.items():
        got = c.quantile(alpha)
        assert got == pytest.approx(want, abs=2e-5), f"a={alpha}: {got:+.6f} vs {want:+.6f}"


def test_terminal_matches_quantiles_was_not_already_true():
    """If the raw simulation happened to match, the test above would prove
    nothing about the calibration."""
    raw = paths()
    off = max(abs(raw.quantile(a) - v) for a, v in TARGET.quantiles.items())
    assert off > 1e-3, f"the raw simulation was already within {off:.2e} of the target"


def test_terminal_matches_quantiles_without_disturbing_the_clustering():
    """Each path keeps its increments and takes a linear drift adjustment, so
    the shape survives the bend. A calibration that flattened the clustering
    would trade one correct number for a wrong path."""
    raw = paths()
    c = calibrate(raw, TARGET)

    before = squared_return_autocorrelation(raw)
    after = squared_return_autocorrelation(c)
    assert after > 0.05
    assert after == pytest.approx(before, rel=0.35)


def test_terminal_matches_quantiles_preserves_the_ordering_of_paths():
    """Rank for rank. A map that reorders paths makes the terminal distribution
    stop being a distribution of anything."""
    raw = paths()
    c = calibrate(raw, TARGET)
    assert (np.argsort(raw.terminal) == np.argsort(c.terminal)).all()


def test_terminal_matches_quantiles_refuses_a_non_monotone_target():
    """Refused twice, and the outer refusal fires first.

    `QuantileForecast` will not construct a crossing distribution at all, which
    is where this should be caught. `quantile_map` checks again anyway, because
    it takes the mapping and not the type, and a caller assembling one by hand
    would otherwise reorder every path silently.
    """
    with pytest.raises(ValueError, match="quantiles cross"):
        QuantileForecast(quantiles={0.05: 0.01, 0.5: 0.0, 0.95: 0.02})

    crossing = QuantileForecast(quantiles={0.05: -0.01, 0.5: 0.0, 0.95: 0.02})
    object.__setattr__(crossing, "quantiles", {0.05: 0.01, 0.5: 0.0, 0.95: 0.02})
    with pytest.raises(PathError, match="not monotone"):
        quantile_map(paths().terminal, crossing)


def test_terminal_matches_quantiles_refuses_too_few_paths():
    with pytest.raises(PathError, match="too few to place a tail"):
        quantile_map(paths(n_paths=MIN_PATHS_FOR_TAILS - 1).terminal, TARGET)


def test_calibration_is_marked_on_the_result():
    assert not paths().calibrated
    assert calibrate(paths(), TARGET).calibrated
    assert "not calibrated" in diagnose(paths()).render()


# --------------------------------------------------------------------------- #
# the tails - the failure the verify command does not name
# --------------------------------------------------------------------------- #
def test_clamping_the_tails_makes_cvar_optimistic():
    """The quiet one. Clamping every path beyond the outermost head onto that
    head flattens the tail onto a single value, so CVaR - the *mean of the
    tail* - becomes the 5th percentile itself, which is the tail's boundary
    rather than anything inside it.

    The number still computes, is still negative, and is uniformly too kind.
    """
    raw = paths()
    good = calibrate(raw, TARGET, extrapolate_tails=True)
    clamped = calibrate(raw, TARGET, extrapolate_tails=False)

    assert cvar(clamped) > cvar(good), "clamping did not understate the tail"
    understatement = 1.0 - cvar(clamped) / cvar(good)
    assert understatement > 0.15, f"only {understatement:.1%} - check the fixture has a tail"


def test_clamping_collapses_the_worst_path_onto_the_fifth_percentile():
    """Which is the visible symptom, if anyone looks: the worst outcome the
    simulation can produce becomes exactly the 5% quantile."""
    clamped = calibrate(paths(), TARGET, extrapolate_tails=False)
    assert float(clamped.terminal.min()) == pytest.approx(TARGET.quantiles[0.05], abs=1e-9)

    good = calibrate(paths(), TARGET, extrapolate_tails=True)
    assert float(good.terminal.min()) < TARGET.quantiles[0.05] * 2


def test_the_extrapolated_tail_joins_continuously_at_the_outer_head():
    """A discontinuity at the join would put a gap in the distribution, which
    shows up downstream as a strike nothing ever prices through."""
    c = calibrate(paths(), TARGET, extrapolate_tails=True)
    t = np.sort(c.terminal)
    n = t.size
    # Around the 5% point the spacing should not jump.
    i = int(0.05 * n)
    inner = float(np.diff(t[i : i + 40]).mean())
    outer = float(np.diff(t[max(i - 40, 1) : i]).mean())
    assert 0.2 < outer / inner < 12.0, f"spacing jumps {outer / inner:.1f}x at the join"


def test_the_tail_method_is_recorded_on_the_result():
    """So a report cannot show a CVaR without showing how its tail was made."""
    clamped = calibrate(paths(), TARGET, extrapolate_tails=False)
    assert "CLAMPED" in clamped.tail_method
    assert "understated" in clamped.render()
    assert "CLAMPED" not in calibrate(paths(), TARGET).tail_method


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def test_latency_of_five_thousand_paths():
    """The plan's budget is 60 ms for 5 000 paths.

    Asserted at twice that, on the median of five runs. The measured median on
    the development machine is 49 ms (min 36, max 58), so an assertion at the
    budget itself would flake on any loaded runner - and a timing test that
    flakes is a timing test somebody deletes. At 120 ms this still catches the
    regression that matters, which is the strided-column-write shape that cost
    79 ms before the arrays were made step-major.
    """
    simulate(n_paths=100, steps=5, variance0=VAR0)  # warm numpy up

    timings = []
    for i in range(5):
        started = time.perf_counter()
        simulate(n_paths=5000, steps=60, variance0=VAR0, seed=i)
        timings.append((time.perf_counter() - started) * 1000.0)

    median = sorted(timings)[2]
    assert median < 120.0, f"median {median:.1f} ms over five runs (budget 60, guard 120)"


def test_latency_scales_with_the_work_not_worse():
    """Doubling the paths should roughly double the time. Anything much worse
    means an accidental copy inside the step loop."""
    simulate(n_paths=100, steps=5, variance0=VAR0)

    def timed(n: int) -> float:
        times = []
        for _ in range(3):
            started = time.perf_counter()
            simulate(n_paths=n, steps=60, variance0=VAR0, seed=1)
            times.append(time.perf_counter() - started)
        return min(times)

    small, large = timed(2000), timed(4000)
    assert large < small * 4.0, f"2x the paths cost {large / small:.1f}x the time"


# --------------------------------------------------------------------------- #
# guards on the process itself
# --------------------------------------------------------------------------- #
def test_a_non_stationary_variance_process_is_refused():
    """alpha + beta >= 1 diverges, and the clustering would look spectacular."""
    with pytest.raises(PathError, match="not stationary"):
        simulate(n_paths=100, steps=10, variance0=VAR0, alpha=0.2, beta=0.85)


def test_infinite_variance_innovations_are_refused():
    with pytest.raises(PathError, match="infinite variance"):
        simulate(n_paths=100, steps=10, variance0=VAR0, nu=2.0)


def test_a_non_positive_starting_variance_is_refused():
    with pytest.raises(PathError, match="variance0 must be positive"):
        simulate(n_paths=100, steps=10, variance0=0.0)


def test_the_starting_variance_is_the_conditioning():
    """ "Conditioned on M55" means this. Two forecasts must produce two different
    distributions, or the volatility model is decorative."""
    quiet = simulate(n_paths=4000, steps=60, variance0=VAR0, seed=SEED)
    busy = simulate(n_paths=4000, steps=60, variance0=VAR0 * 9.0, seed=SEED)
    assert float(busy.terminal.std()) > float(quiet.terminal.std()) * 2.0


def test_omega_defaults_to_keeping_the_forecast_as_the_stationary_level():
    """A path with no shocks should stay where the forecast put it, rather than
    drifting toward a long-run variance nobody asked for."""
    p = simulate(n_paths=4000, steps=200, variance0=VAR0, seed=SEED)
    assert float(p.variance[:, -1].mean()) == pytest.approx(VAR0, rel=0.35)


def test_the_simulation_is_deterministic_for_a_seed():
    a = simulate(n_paths=500, steps=20, variance0=VAR0, seed=3)
    b = simulate(n_paths=500, steps=20, variance0=VAR0, seed=3)
    assert (a.levels == b.levels).all()

    c = simulate(n_paths=500, steps=20, variance0=VAR0, seed=4)
    assert not (a.levels == c.levels).all()


def test_paths_start_at_zero():
    assert (paths(n_paths=500, steps=10).levels[:, 0] == 0.0).all()


def test_the_default_nu_is_fat_but_finite():
    assert 3.0 < DEFAULT_NU < 10.0


# --------------------------------------------------------------------------- #
# gaps a mutation test found
# --------------------------------------------------------------------------- #
def test_the_default_is_to_extrapolate_not_to_clamp():
    """Every tail test above passes `extrapolate_tails` explicitly, so flipping
    the default changed nothing and none of them noticed. The default is the
    thing every caller actually gets."""
    c = calibrate(paths(), TARGET)  # no keyword: whatever the default is

    assert "CLAMPED" not in c.tail_method
    assert (
        float(c.terminal.min()) < TARGET.quantiles[0.05] * 1.5
    ), "the default calibration clamped the tail; CVaR from it is optimistic"


def test_a_calibrated_path_still_starts_at_zero():
    """The drift adjustment is a ramp from 0 at entry to 1 at the horizon.
    Applying the shift flat instead moves the whole path, including its first
    point - so every path starts somewhere other than the price you entered at,
    and every P&L computed along it is offset by a constant nobody can see.
    """
    c = calibrate(paths(), TARGET)
    assert (c.levels[:, 0] == 0.0).all()
    assert c.terminal is not None


def test_the_first_step_variance_is_exactly_the_forecast():
    """ "Conditioned on M55" pinned directly. Comparing terminal spreads across
    two forecasts is not enough: omega is derived from variance0, so a sampler
    that ignored variance0 for the initial level still drifts to roughly the
    right place over sixty steps and the spread test passes anyway.
    """
    for v0 in (VAR0, VAR0 * 9.0, VAR0 * 0.25):
        p = simulate(n_paths=500, steps=30, variance0=v0, seed=SEED)
        assert np.allclose(p.variance[:, 0], v0, rtol=0, atol=0)
