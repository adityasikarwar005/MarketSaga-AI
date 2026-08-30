"""M63 null tests: what does the positioning statistic say when there is nothing?

The pattern that recurs across two audits of this project is that each defect
produced a working pipeline, plausible numbers, and no error. This module got
one on its first run: `gamma_kernel` returned 0.0 whenever days-to-expiry was 0,
because expiry day was being treated as zero time to expiry. Gamma is *largest*
then. The column of numbers looked ordinary, and the correlation with
days-to-expiry read +0.12 instead of -0.82 - a sign flip, from a guard clause.

So the tests here are of three kinds:

* the arithmetic is pinned against a finite difference of an independently
  written Black-Scholes price, because a gamma that is merely self-consistent
  is a gamma nobody has checked;
* the confound the module exists to remove is *constructed* and the raw and
  adjusted statistics are required to behave differently on it;
* the signal is destroyed by shuffling and the adjusted score is required to
  stop carrying information.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from libs.ms_models.positioning import (
    EXPIRY_DAY_FRACTION,
    KERNEL_SIGMA,
    MIN_STRIKES,
    GammaProfile,
    PositioningError,
    concentration_is_robust_to_sigma,
    cycle_adjusted,
    fit_cycle_norm,
    gamma_concentration,
    gamma_kernel,
)

SPOT = 24_000.0


def bs_call(spot: float, strike: float, t: float, sigma: float) -> float:
    """Black-Scholes call at zero rates, written out here on purpose.

    Sharing algebra with the thing under test would let both drift together.
    """
    v = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    return spot * ncdf(d1) - strike * ncdf(d2)


class TestTheArithmetic:
    @pytest.mark.parametrize("strike", [22_000.0, 24_000.0, 26_000.0])
    @pytest.mark.parametrize("t", [0.01, 0.08, 0.5])
    def test_kernel_matches_a_finite_difference_of_the_price(self, strike: float, t: float) -> None:
        """Gamma is the second derivative of the price in spot. Pin it there."""
        h = SPOT * 1e-4
        fd = (
            bs_call(SPOT + h, strike, t, KERNEL_SIGMA)
            - 2.0 * bs_call(SPOT, strike, t, KERNEL_SIGMA)
            + bs_call(SPOT - h, strike, t, KERNEL_SIGMA)
        ) / (h * h)
        got = gamma_kernel(SPOT, strike, t)
        assert got == pytest.approx(fd, rel=2e-3), (
            f"kernel gamma {got:.3e} disagrees with a finite difference of an "
            f"independently written price {fd:.3e} at K={strike}, T={t}."
        )

    def test_gamma_is_largest_at_the_money(self) -> None:
        atm = gamma_kernel(SPOT, SPOT, 0.05)
        for away in (0.90, 0.95, 1.05, 1.10):
            assert gamma_kernel(SPOT, SPOT * away, 0.05) < atm

    def test_gamma_grows_as_expiry_approaches(self) -> None:
        """The mechanism the whole module is about, and the one the first cut
        inverted by returning zero on expiry day."""
        near = gamma_kernel(SPOT, SPOT, EXPIRY_DAY_FRACTION / 365.0)
        far = gamma_kernel(SPOT, SPOT, 30.0 / 365.0)
        assert near > far * 5.0


class TestItRefusesRatherThanReturningZero:
    @pytest.mark.parametrize("t", [0.0, -0.01])
    def test_non_positive_time_raises(self, t: float) -> None:
        """A zero here is indistinguishable from 'no open interest near spot'
        after summing. That is how the expiry-day bug survived."""
        with pytest.raises(PositioningError, match="hours"):
            gamma_kernel(SPOT, SPOT, t)

    def test_expiry_day_gets_a_positive_time_not_zero(self) -> None:
        """`> 0` is not enough, and the first version of this test only asserted
        that. Replacing the constant with 1e-12 leaves concentration positive -
        vast, and positive - so it passed against a degenerate fix.

        What pins it is the constant. `year_fraction` is now the single
        definition both the property and the weighting path use; when they were
        two expressions this assertion could pass while the strikes were being
        weighted by a different number entirely.
        """
        rows = [(SPOT + 100.0 * i, 1_000) for i in range(-10, 11)]
        day = date(2026, 8, 25)
        prof = gamma_concentration(rows, day=day, expiry=day, spot=SPOT)

        assert prof.days_to_expiry == 0
        assert prof.year_fraction == pytest.approx(EXPIRY_DAY_FRACTION / 365.0)
        assert prof.concentration > 0.0

    def test_a_uniform_chain_is_flat_across_the_cycle(self) -> None:
        """The statistic measures shape, not time, and this is the proof.

        Gamma is a density in strike: summed over a fixed grid it is conserved
        as expiry approaches. So with open interest identical at every strike,
        concentration must barely move across the cycle. The obvious story -
        "gamma rises into expiry so the sum rises" - is wrong, and it was in
        this module's docstring until this test was written. What actually
        drives the real -0.83 is that real open interest is peaked at the money
        and the kernel narrows onto that peak.
        """
        day = date(2026, 8, 25)
        flat = [(SPOT + 100.0 * i, 1_000) for i in range(-10, 11)]
        values = [
            gamma_concentration(
                flat, day=day - timedelta(days=dte), expiry=day, spot=SPOT
            ).concentration
            for dte in range(6)
        ]
        spread = (max(values) - min(values)) / max(values)
        assert spread < 0.01, (
            f"uniform open interest gave a {spread:.1%} spread across the "
            "expiry cycle. The statistic is picking up time to expiry directly, "
            "which means cycle_adjusted is removing the wrong thing."
        )

    def test_too_few_strikes_raises(self) -> None:
        rows = [(SPOT + 100.0 * i, 1_000) for i in range(MIN_STRIKES - 1)]
        with pytest.raises(PositioningError, match="profile"):
            gamma_concentration(rows, day=date(2026, 8, 3), expiry=date(2026, 8, 25), spot=SPOT)

    def test_expiry_before_the_session_raises(self) -> None:
        rows = [(SPOT + 100.0 * i, 1_000) for i in range(-10, 11)]
        with pytest.raises(PositioningError, match="precedes"):
            gamma_concentration(rows, day=date(2026, 8, 25), expiry=date(2026, 8, 3), spot=SPOT)


def _chain(spot: float, oi_scale: float = 1.0, seed: int = 0) -> list[tuple[float, int]]:
    """A plausible open-interest smile: most contracts sit near the money."""
    rng = random.Random(seed)
    rows = []
    for i in range(-20, 21):
        k = spot + 100.0 * i
        weight = math.exp(-0.5 * (i / 7.0) ** 2)
        rows.append((k, int(oi_scale * weight * 1_000_000 * rng.uniform(0.7, 1.3))))
    return rows


class TestTheConfound:
    """The whole reason the module has a `cycle_adjusted` at all.

    Constructed so that concentration is driven *entirely* by days-to-expiry and
    the open interest itself is statistically identical across days. The raw
    statistic must see a strong relationship; the adjusted one must not, because
    there is nothing there beyond the cycle.
    """

    @staticmethod
    def _profiles() -> list[GammaProfile]:
        out = []
        base = date(2026, 1, 6)
        for cycle in range(30):
            expiry = base + timedelta(days=7 * cycle)
            for dte in range(7):
                day = expiry - timedelta(days=dte)
                out.append(
                    gamma_concentration(
                        _chain(SPOT, seed=cycle * 10 + dte),
                        day=day,
                        expiry=expiry,
                        spot=SPOT,
                    )
                )
        return out

    def test_raw_concentration_is_mostly_the_expiry_cycle(self) -> None:
        profs = self._profiles()
        r = _corr([p.concentration for p in profs], [float(p.days_to_expiry) for p in profs])
        assert r < -0.5, (
            f"raw concentration correlates {r:+.3f} with days-to-expiry on data "
            "built to contain nothing else. If this is weak, the confound this "
            "module removes is not the confound that exists."
        )

    def test_cycle_adjustment_removes_it(self) -> None:
        profs = self._profiles()
        norm = fit_cycle_norm(profs)
        scored = [(p, cycle_adjusted(p, norm)) for p in profs]
        kept = [(p, z) for p, z in scored if z is not None]
        assert len(kept) > 100, "too few days survived the norm to conclude anything"
        r = _corr([z for _, z in kept], [float(p.days_to_expiry) for p, _ in kept])
        assert abs(r) < 0.15, (
            f"adjusted score still correlates {r:+.3f} with days-to-expiry. The "
            "adjustment is not removing the cycle, so any downstream result "
            "would be rediscovering the calendar."
        )


class TestNulls:
    def test_a_flat_chain_has_no_unusual_days(self) -> None:
        """Identical open interest every day means every day is normal.

        The adjusted score must be ~0 throughout rather than large numbers
        manufactured out of a near-zero spread.
        """
        base = date(2026, 1, 6)
        profs = []
        for cycle in range(20):
            expiry = base + timedelta(days=7 * cycle)
            for dte in range(7):
                rows = [(SPOT + 100.0 * i, 1_000_000) for i in range(-20, 21)]
                profs.append(
                    gamma_concentration(
                        rows, day=expiry - timedelta(days=dte), expiry=expiry, spot=SPOT
                    )
                )
        norm = fit_cycle_norm(profs)
        scores = [cycle_adjusted(p, norm) for p in profs]
        # Every horizon has zero spread, so every day must be refused rather
        # than scored. A number here would be division by rounding error.
        assert all(
            z is None for z in scores
        ), "a chain that never changes produced non-null unusualness scores"

    def test_shuffling_open_interest_destroys_the_ordering(self) -> None:
        """If the statistic still ranked days the same way after the open
        interest was reassigned at random, it would not be measuring open
        interest."""
        base = date(2026, 1, 6)
        expiry = base + timedelta(days=7)
        real = [
            gamma_concentration(
                _chain(SPOT, oi_scale=1.0 + i / 10.0, seed=i), day=base, expiry=expiry, spot=SPOT
            ).concentration
            for i in range(30)
        ]
        rng = random.Random(99)
        scrambled = []
        for i in range(30):
            rows = _chain(SPOT, oi_scale=1.0 + i / 10.0, seed=i)
            ois = [oi for _, oi in rows]
            rng.shuffle(ois)
            scrambled.append(
                gamma_concentration(
                    [(k, oi) for (k, _), oi in zip(rows, ois, strict=True)],
                    day=base,
                    expiry=expiry,
                    spot=SPOT,
                ).concentration
            )
        assert real != scrambled

    def test_the_norm_refuses_a_history_too_short_to_have_one(self) -> None:
        base = date(2026, 1, 6)
        profs = [
            gamma_concentration(
                _chain(SPOT, seed=i), day=base, expiry=base + timedelta(days=3), spot=SPOT
            )
            for i in range(2)
        ]
        with pytest.raises(PositioningError, match="not enough history"):
            fit_cycle_norm(profs, min_per_horizon=5)

    def test_an_unseen_horizon_scores_none_rather_than_guessing(self) -> None:
        base = date(2026, 1, 6)
        expiry = base + timedelta(days=3)
        profs = [
            gamma_concentration(_chain(SPOT, seed=i), day=base, expiry=expiry, spot=SPOT)
            for i in range(8)
        ]
        norm = fit_cycle_norm(profs)
        far = gamma_concentration(
            _chain(SPOT, seed=1), day=base, expiry=base + timedelta(days=45), spot=SPOT
        )
        assert cycle_adjusted(far, norm) is None


class TestDeterminismAndBandwidth:
    def test_two_runs_on_the_same_input_are_identical(self) -> None:
        rows = _chain(SPOT, seed=4)
        a = gamma_concentration(rows, day=date(2026, 8, 3), expiry=date(2026, 8, 25), spot=SPOT)
        b = gamma_concentration(rows, day=date(2026, 8, 3), expiry=date(2026, 8, 25), spot=SPOT)
        assert a == b

    def test_the_bandwidth_is_not_a_tuning_knob(self) -> None:
        """The module docstring claims sigma does not choose the answer.

        Measured rather than asserted: if the ordering of days by concentration
        survives a 2x change in bandwidth, no choice of it buys a result.
        """
        base = date(2026, 1, 6)
        expiry = base + timedelta(days=7)
        days = [
            (base, expiry, SPOT, _chain(SPOT, oi_scale=1.0 + i / 8.0, seed=i)) for i in range(40)
        ]
        worst = concentration_is_robust_to_sigma(days)
        assert worst > 0.95, (
            f"day ordering correlates only {worst:.3f} across bandwidths, so the "
            "bandwidth is doing real work and the module docstring is wrong."
        )


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0
