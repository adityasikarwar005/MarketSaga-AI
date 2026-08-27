"""What each model does when there is nothing to find.

Every other test in this suite asks whether a model works when there is a
signal. These ask the harder question: what does it report when there is not
one. A model that finds structure in noise will find structure in a live market
too, and it will be wrong in exactly the same way - except that there the
mistake costs money instead of a red test.

Each test here corresponds to a defect that was actually present and that the
rest of the suite passed straight through.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from libs.ms_core.schemas import Candle
from libs.ms_impact.baseline import Regime
from libs.ms_impact.study import EventObservation, severity_response
from libs.ms_impact.transfer import best_lag, estimate_transfer
from libs.ms_models.quantile import QuantileRegressor

TRIALS = 40


def noise_bars(n: int, seed: int) -> list[Candle]:
    rng = random.Random(seed)
    t0 = datetime(2026, 3, 2, 4, 0, tzinfo=UTC)
    out, px = [], 100.0
    for i in range(n):
        px *= 1 + rng.gauss(0, 0.0008)
        out.append(
            Candle(
                instrument_id=1,
                ts=t0 + timedelta(minutes=i),
                tf=1,
                o=px,
                h=px * 1.001,
                low=px * 0.999,
                c=px,
                v=100,
            )
        )
    return out


# ----------------------------------------------------- searching over lags
def test_best_lag_pays_for_its_own_search():
    """Picking the best of six lags and reporting a 95% interval is reporting a
    68% interval with a 95% label on it.

    Measured before this was corrected: 8% of single fixed lags looked
    significant on independent noise, against 30% for the best of six.
    """
    single = 0
    winner = 0
    for s in range(TRIALS):
        src, tgt = noise_bars(400, 1000 + s), noise_bars(400, 5000 + s)
        one = estimate_transfer(src, tgt, source_name="NOISE", lag_minutes=0)
        best = best_lag(src, tgt, source_name="NOISE")
        single += bool(one and one.significant)
        winner += bool(best and best.significant)

    assert winner <= single + TRIALS * 0.05, (
        f"best-of-six was significant {winner}/{TRIALS} times against "
        f"{single}/{TRIALS} for one fixed lag - the search is not being paid for"
    )
    assert winner / TRIALS <= 0.15, f"false-positive rate {winner / TRIALS:.0%} on pure noise"


def test_best_lag_still_returns_the_coefficient_it_cannot_certify():
    """Correcting significance must not silently discard the estimate. A real
    relationship at an unproven lag is worth seeing, clearly marked."""
    src, tgt = noise_bars(400, 1), noise_bars(400, 2)
    got = best_lag(src, tgt, source_name="NOISE")
    assert got is not None
    assert got.fit.searched > 1, "the winner does not record how many lags were compared"


def test_a_single_lag_is_not_penalised():
    src, tgt = noise_bars(400, 1), noise_bars(400, 2)
    one = estimate_transfer(src, tgt, source_name="NOISE", lag_minutes=0)
    assert one is not None and one.fit.searched == 1


# ------------------------------------------------------- severity scaling
def observations(seed: int, n: int = 60, gamma: float | None = None):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        sev = rng.uniform(1, 10)
        ar = rng.gauss(0, 0.004) if gamma is None else ((sev / 5) ** gamma) * 0.003
        out.append(
            EventObservation(
                event_id=f"e{i}",
                event_class="c",
                t0=datetime(2026, 3, 2, tzinfo=UTC),
                severity=sev,
                regime=Regime.NORMAL,
                abnormal={30: ar},
            )
        )
    return out


def test_severity_scaling_is_not_invented_from_noise():
    """A zero slope used to be clamped to 0.5, turning "severity does not
    matter" into "impact scales as its square root" - in 80 of 80 noise trials.
    The neutral answer is 1.0, the same as having no data at all."""
    got = [severity_response(observations(s), 30) for s in range(TRIALS)]
    invented = [g for g in got if abs(g - 1.0) > 1e-9]
    assert len(invented) <= TRIALS * 0.1, (
        f"{len(invented)}/{TRIALS} noise samples produced a severity exponent; "
        f"first few: {invented[:5]}"
    )


def test_a_real_severity_exponent_is_still_recovered():
    """The correction must not have cost the estimator its job."""
    for gamma in (0.8, 1.5, 2.5):
        got = severity_response(observations(0, n=200, gamma=gamma), 30)
        assert abs(got - gamma) < 0.15, f"true {gamma} came back as {got}"


def test_no_data_and_no_relationship_agree():
    """Both mean the same thing - nothing is known - so both must say 1.0.
    They used to disagree, and the noise fit made the stronger claim."""
    assert severity_response(observations(0, n=5), 30) == 1.0
    assert severity_response(observations(0), 30) == 1.0


# --------------------------------------------------- quantile regularisation
def test_the_l2_default_is_not_decoration():
    """l2=1e-4 shrank the weights by 0.2% against no penalty at all. A knob that
    does nothing is worse than an absent one: it reads as a defence."""
    rng = random.Random(7)
    rows = [[rng.gauss(0, 1) for _ in range(20)] for _ in range(80)]
    targets = [rng.gauss(0, 0.005) for _ in range(80)]

    def mean_abs_weight(l2: float) -> float:
        w = QuantileRegressor(l2=l2).fit(rows, targets).weights[0.5]
        return sum(abs(x) for x in w) / len(w)

    unpenalised = mean_abs_weight(0.0)
    default = mean_abs_weight(QuantileRegressor().l2)
    assert default < unpenalised * 0.9, (
        f"the default penalty shrinks weights only "
        f"{1 - default / unpenalised:.1%} against none at all"
    )


def test_more_penalty_shrinks_more():
    rng = random.Random(3)
    rows = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(60)]
    targets = [rng.gauss(0, 0.005) for _ in range(60)]
    norms = [
        sum(abs(x) for x in QuantileRegressor(l2=l2).fit(rows, targets).weights[0.5])
        for l2 in (0.0, 0.1, 1.0, 10.0)
    ]
    assert norms == sorted(norms, reverse=True), f"penalty is not monotone: {norms}"


def test_the_training_objective_never_rises():
    """Pinball loss is convex. A descent that increases it is diverging, and
    every number downstream of it is describing an accident."""
    from statistics import fmean

    from libs.ms_models.quantile import DEFAULT_QUANTILES, pinball_loss

    rng = random.Random(5)
    rows = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(150)]
    targets = [0.4 * r[0] - 0.3 * r[1] + rng.gauss(0, 1) for r in rows]

    losses = []
    for epochs in (50, 200, 800):
        m = QuantileRegressor(epochs=epochs).fit(rows, targets)
        losses.append(
            fmean(
                pinball_loss(y, m.predict(x).quantiles[a], a)
                for x, y in zip(rows, targets, strict=True)
                for a in DEFAULT_QUANTILES
            )
        )
    assert losses == sorted(losses, reverse=True), f"training loss rose: {losses}"


# ------------------------------------------------- hierarchical shrinkage
def class_stats(seed: int, *, n_classes: int, n_obs: int, spread: float):
    """Classes drawn from one family. ``spread`` is the TRUE between-class
    spread: zero means the classes are genuinely indistinguishable."""
    from statistics import fmean, median, pstdev

    from libs.ms_impact.study import ClassStats

    rng = random.Random(seed)
    out = []
    for c in range(n_classes):
        true_effect = rng.gauss(0.0, spread)
        obs = [rng.gauss(true_effect, 0.004) for _ in range(n_obs)]
        out.append(
            ClassStats(
                event_class=f"c{c}",
                horizon=30,
                n=n_obs,
                median=median(obs),
                mean=fmean(obs),
                sigma=pstdev(obs),
                q25=min(obs),
                q75=max(obs),
                p_sign_correct=0.5,
                mean_mfe=0.0,
                mean_mae=0.0,
                mean_vol_ratio=1.0,
                median_time_to_peak=None,
                median_half_life=None,
            )
        )
    return out


def test_indistinguishable_classes_borrow_almost_everything():
    """Twenty classes from one distribution: nothing separates them, so the
    family mean is the best estimate for every one and each should keep almost
    none of its own eight-observation average.

    Before the sampling-noise correction this shrank only 0.46 of the way.
    """
    from libs.ms_impact.study import family_prior, shrink

    shrinkages = []
    for seed in range(30):
        stats = class_stats(seed, n_classes=20, n_obs=8, spread=0.0)
        fm, fs = family_prior(stats)
        shrinkages += [shrink(s, fm, fs, family_size=len(stats)).shrinkage for s in stats]

    mean_shrinkage = sum(shrinkages) / len(shrinkages)
    assert mean_shrinkage > 0.7, f"borrowed only {mean_shrinkage:.2f} of the way to the family"


def test_genuinely_different_classes_keep_their_own_estimates():
    """The correction must not have turned shrinkage into a blunt instrument.
    When classes really do differ, they must be allowed to say so."""
    from libs.ms_impact.study import family_prior, shrink

    shrinkages = []
    for seed in range(30):
        stats = class_stats(seed, n_classes=12, n_obs=60, spread=0.02)
        fm, fs = family_prior(stats)
        shrinkages += [shrink(s, fm, fs, family_size=len(stats)).shrinkage for s in stats]

    mean_shrinkage = sum(shrinkages) / len(shrinkages)
    assert mean_shrinkage < 0.2, f"a real family effect was shrunk away ({mean_shrinkage:.2f})"


def test_a_family_of_one_cannot_borrow():
    """Zero spread means two opposite things. A lone class has no siblings, so
    it must stand on its own rather than shrink to a family that is itself."""
    from libs.ms_impact.study import family_prior, shrink

    stats = class_stats(0, n_classes=1, n_obs=30, spread=0.0)
    fm, fs = family_prior(stats)
    post = shrink(stats[0], fm, fs, family_size=1)
    assert post.shrinkage == 0.0
    assert post.mean == stats[0].mean


def test_nothing_looks_significant_when_nothing_differs():
    """The number that matters downstream: how often a credible interval
    excludes zero when no class has any real effect at all."""
    from libs.ms_impact.study import family_prior, shrink

    total = flagged = 0
    for seed in range(50):
        stats = class_stats(seed, n_classes=20, n_obs=8, spread=0.0)
        fm, fs = family_prior(stats)
        for s in stats:
            total += 1
            flagged += shrink(s, fm, fs, family_size=len(stats)).excludes_zero

    assert flagged / total <= 0.05, f"{flagged}/{total} classes looked real ({flagged / total:.1%})"
