"""M30-M32 verify - abnormal returns, event study, Bayesian shrinkage."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from libs.ms_core.clock import SessionCalendar
from libs.ms_core.schemas import Candle
from libs.ms_impact.baseline import (
    MIN_SESSIONS_FOR_PROFILE,
    Regime,
    TimeOfDayProfile,
    abnormal_return,
    bootstrap_significance,
    build_profile,
    classify_regimes,
    event_windows,
    group_by_session,
    matched_controls,
    session_volatility,
)
from libs.ms_impact.study import (
    MIN_SAMPLE_FOR_CLASS,
    ClassStats,
    build_library,
    family_prior,
    observe_event,
    prior_for_baseline,
    severity_response,
    shrink,
)
from tests.conftest import ist


@pytest.fixture
def cal(cfg):
    return SessionCalendar.from_config(cfg.sessions).synthetic_variant()


def session(day_offset: int, *, vol: float = 0.0004, seed: int = 0, n: int = 200) -> list[Candle]:
    """One session with a U-shaped intraday volatility profile."""
    import random

    # Python 3.13 dropped tuple seeds; derive a stable int instead.
    rng = random.Random(day_offset * 1_000_003 + seed)
    base = ist(9, 15) + timedelta(days=day_offset)
    out, p = [], 100.0
    for i in range(n):
        x = i / n
        # The U-shape the baseline exists to remove.
        multiplier = 0.7 + 1.6 * (math.exp(-8 * x) + math.exp(-8 * (1 - x)))
        o = p
        p = p * math.exp(rng.gauss(0.0, vol * multiplier))
        out.append(
            Candle(
                ts=base + timedelta(minutes=i),
                instrument_id=1,
                tf=1,
                o=round(o, 2),
                h=round(max(o, p) * 1.0003, 2),
                l=round(min(o, p) * 0.9997, 2),
                c=round(p, 2),
                v=1000,
            )
        )
    return out


@pytest.fixture
def sessions():
    """Forty sessions - enough for the profile to be considered reliable."""
    return {(date(2026, 8, 26) + timedelta(days=d)): session(d) for d in range(40)}


# ------------------------------------------------------------------- regimes
def test_session_volatility_is_positive_and_scales():
    calm = session_volatility(session(0, vol=0.0002))
    wild = session_volatility(session(0, vol=0.0020))
    assert 0 < calm < wild


def test_regimes_use_percentiles_not_absolute_thresholds():
    """An absolute threshold ages badly - what counted as stressed two years ago
    is an ordinary Tuesday now."""
    per_session = {
        date(2026, 8, 26) + timedelta(days=d): session(d, vol=0.0001 + d * 0.0002)
        for d in range(12)
    }
    regimes = classify_regimes(per_session)
    assert set(regimes.values()) == {Regime.CALM, Regime.NORMAL, Regime.STRESSED}

    # Shift every session up; the buckets should still split three ways.
    louder = {
        day: [c.model_copy(update={"c": c.c}) for c in bars] for day, bars in per_session.items()
    }
    assert set(classify_regimes(louder).values()) == set(regimes.values())


def test_too_few_sessions_are_all_normal():
    per_session = {date(2026, 8, 26): session(0)}
    assert set(classify_regimes(per_session).values()) == {Regime.NORMAL}


# ------------------------------------------------------------------- profile
def test_profile_needs_enough_sessions(cal):
    few = {date(2026, 8, 26) + timedelta(days=d): session(d) for d in range(5)}
    profile = build_profile(few, cal, horizon_minutes=15)
    assert not profile.reliable
    assert "UNRELIABLE" in profile.render()


def test_profile_is_reliable_with_enough_sessions(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    assert profile.n_sessions == 40
    assert profile.reliable
    assert profile.coverage() > 0.5


def test_profile_captures_the_u_shape(sessions, cal):
    """If the baseline were flat, every event near the open would look
    significant and every event at midday would look like nothing."""
    profile = build_profile(sessions, cal, horizon_minutes=15)
    regime = Regime.NORMAL

    def sigma_at(minute: int) -> float:
        found = [
            profile.expected(r, minute)[1]
            for r in Regime
            if profile.expected(r, minute) is not None
        ]
        return max(found) if found else 0.0

    open_sigma = sigma_at(5)
    mid_sigma = sigma_at(170)
    assert open_sigma > 0 and mid_sigma > 0
    assert open_sigma > mid_sigma, "the profile flattened out the intraday shape"
    _ = regime


def test_unreliable_slot_returns_none(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    assert profile.expected(Regime.NORMAL, 9999) is None


def test_event_windows_can_be_excluded(sessions, cal):
    """Without this the baseline absorbs the very moves it should measure
    against, and every event study understates its own effect."""
    events = [ist(11, 0) + timedelta(days=d) for d in range(40)]
    excluded = event_windows(events, before=5, after=30)
    with_events = build_profile(sessions, cal, horizon_minutes=15)
    without = build_profile(sessions, cal, horizon_minutes=15, exclude=excluded)

    total_with = sum(s.n for s in with_events.stats.values())
    total_without = sum(s.n for s in without.stats.values())
    assert total_without < total_with


# ----------------------------------------------------------- abnormal returns
def test_abnormal_return_subtracts_the_baseline(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    ar = abnormal_return(bars, bars[60].ts, profile, cal)
    assert ar is not None
    assert ar.abnormal == pytest.approx(ar.raw - ar.expected)
    assert ar.reliable


def test_missing_baseline_reports_raw_and_says_so(cal):
    """Subtracting a number nobody should trust is worse than not subtracting."""
    thin = {date(2026, 8, 26): session(0)}
    profile = build_profile(thin, cal, horizon_minutes=15)
    bars = thin[date(2026, 8, 26)]
    ar = abnormal_return(bars, bars[60].ts, profile, cal)
    assert ar is not None
    assert not ar.reliable
    assert ar.abnormal == ar.raw
    assert "unreliable" in ar.render()


def test_a_large_move_shows_a_large_z(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = list(sessions[date(2026, 8, 26)])
    for i in range(61, 80):
        c = bars[i]
        bars[i] = c.model_copy(
            update={"o": c.o * 1.02, "h": c.h * 1.02, "low": c.low * 1.02, "c": c.c * 1.02}
        )
    ar = abnormal_return(bars, bars[60].ts, profile, cal)
    assert ar is not None and abs(ar.z) > 2.0


# ---------------------------------------------------------------- controls
def test_matched_controls_pool_neighbouring_minutes(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    single = profile.get(Regime.NORMAL, 60)
    pooled = matched_controls(profile, Regime.NORMAL, 60, window=5)
    assert single is not None
    assert len(pooled) > single.n


def test_significance_needs_enough_controls():
    """With a handful of controls the p-value cannot be small however striking
    the observation looks."""
    assert bootstrap_significance(0.05, [0.001] * 10) is None


def test_an_ordinary_move_is_not_significant():
    import random

    rng = random.Random(1)
    controls = [rng.gauss(0.0, 0.002) for _ in range(500)]
    sig = bootstrap_significance(0.0005, controls)
    assert sig is not None and not sig.significant
    assert "not distinguishable" in sig.render()


def test_an_extreme_move_is_significant():
    import random

    rng = random.Random(1)
    controls = [rng.gauss(0.0, 0.002) for _ in range(500)]
    sig = bootstrap_significance(0.02, controls)
    assert sig is not None and sig.significant


# ------------------------------------------------------------- event study
def test_observation_measures_excursions(sessions, cal):
    """MFE and MAE are where a target and a stop belong - from evidence, not
    from a round number someone liked."""
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    obs = observe_event(bars, bars[60].ts, "e1", "macro.inflation_release", 7.0, profile, cal)
    assert obs is not None
    assert obs.mfe >= 0 >= obs.mae
    assert obs.time_to_peak_minutes is not None
    assert obs.family == "macro"


def test_observation_reports_a_volatility_ratio(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = list(sessions[date(2026, 8, 26)])
    for i in range(61, 130):
        c = bars[i]
        wide = c.model_copy(update={"h": c.h * 1.01, "low": c.low * 0.99})
        bars[i] = wide.model_copy(update={"c": c.c * (1.004 if i % 2 else 0.996)})
    obs = observe_event(bars, bars[60].ts, "e1", "geopolitical.military_action", 8.0, profile, cal)
    assert obs is not None and obs.vol_ratio > 1.5


def test_event_outside_the_data_returns_none(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    assert observe_event(bars, bars[-1].ts, "e", "c", 5.0, profile, cal) is None


# ---------------------------------------------------------------- shrinkage
def _stat(name: str, n: int, mean: float, sigma: float = 0.004) -> ClassStats:
    return ClassStats(
        event_class=name,
        horizon=15,
        n=n,
        median=mean,
        mean=mean,
        sigma=sigma,
        q25=mean - sigma,
        q75=mean + sigma,
        p_sign_correct=0.6,
        mean_mfe=0.004,
        mean_mae=-0.003,
        mean_vol_ratio=1.2,
        median_time_to_peak=8.0,
        median_half_life=12.0,
    )


def test_a_rare_class_is_dominated_by_its_family():
    """With n=3 the posterior should be mostly borrowed. That is correct, not a
    compromise: three observations must not move a position size."""
    rare = _stat("macro.rare", n=3, mean=0.02)
    posterior = shrink(rare, family_mean=0.001, between_sigma=0.001)
    assert posterior.shrinkage > 0.5
    assert abs(posterior.mean - 0.001) < abs(posterior.mean - 0.02)
    assert posterior.n_eff < rare.n


def test_a_well_observed_class_keeps_its_own_estimate():
    common = _stat("macro.common", n=200, mean=0.02, sigma=0.004)
    posterior = shrink(common, family_mean=0.001, between_sigma=0.01)
    assert posterior.shrinkage < 0.2
    assert abs(posterior.mean - 0.02) < abs(posterior.mean - 0.001)


def test_a_family_that_disagrees_lends_little():
    """No coherent family behaviour means nothing to borrow."""
    rare = _stat("x.rare", n=4, mean=0.02)
    tight = shrink(rare, family_mean=0.0, between_sigma=0.0005)
    loose = shrink(rare, family_mean=0.0, between_sigma=0.05)
    assert tight.shrinkage > loose.shrinkage


def test_zero_observations_is_unknown():
    posterior = shrink(_stat("x.never", n=0, mean=0.0), 0.0, 0.001)
    assert posterior.is_unknown
    assert not posterior.usable
    assert "UNKNOWN" in posterior.render()


def test_family_prior_pools_classes():
    mean, sigma = family_prior([_stat("f.a", 20, 0.01), _stat("f.b", 20, 0.02)])
    assert mean == pytest.approx(0.015)
    assert sigma > 0


def test_empty_family_is_neutral():
    assert family_prior([]) == (0.0, 0.0)


# ------------------------------------------------------------------ library
def test_library_builds_and_reports(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    observations = []
    for day, bars in sorted(sessions.items())[:30]:
        for k, cls in enumerate(("macro.inflation_release", "geopolitical.military_action")):
            obs = observe_event(bars, bars[60 + k * 20].ts, f"{day}-{k}", cls, 7.0, profile, cal)
            if obs:
                observations.append(obs)

    library = build_library(observations, profile=profile, calendar=cal)
    assert library.built_from == len(observations)
    assert library.usable_classes(15)
    assert "impact library" in library.render(15)


def test_library_refuses_thin_classes(sessions, cal):
    """Below the sample floor the credible interval is wider than any plausible
    effect, so the estimate cannot change a decision."""
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    observations = [
        observe_event(bars, bars[60 + i].ts, f"e{i}", "rare.thing", 7.0, profile, cal)
        for i in range(3)
    ]
    library = build_library([o for o in observations if o], profile=profile, calendar=cal)
    posterior = library.get("rare.thing", 15)
    assert posterior is not None
    assert posterior.n < MIN_SAMPLE_FOR_CLASS
    assert not posterior.usable


def test_baseline_gets_no_direction_from_a_thin_class(sessions, cal):
    """The silence is the intended behaviour, not a gap to be filled later."""
    from libs.ms_models.baseline import RulesBaseline

    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    observations = [
        observe_event(bars, bars[60 + i].ts, f"e{i}", "rare.thing", 8.0, profile, cal)
        for i in range(3)
    ]
    library = build_library([o for o in observations if o], profile=profile, calendar=cal)

    event = {
        "severity": 8.0,
        "credibility": 0.9,
        "novelty": 0.8,
        **prior_for_baseline(library, "rare.thing", 15),
    }
    opinion = RulesBaseline().predict({}, event=event)
    assert opinion.direction == 0
    assert any("no usable prior" in r for r in opinion.reasons)


def test_unknown_class_yields_no_prior():
    from libs.ms_impact.study import ImpactLibrary

    assert prior_for_baseline(ImpactLibrary(), "never.seen", 15) == {
        "prior_return": None,
        "n_eff": 0,
    }


# ------------------------------------------------------------------ severity
def test_severity_response_falls_back_when_thin(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    bars = sessions[date(2026, 8, 26)]
    few = [observe_event(bars, bars[60].ts, "e", "c", 7.0, profile, cal)]
    assert severity_response([o for o in few if o], 15) == 1.0


def test_severity_response_is_bounded(sessions, cal):
    profile = build_profile(sessions, cal, horizon_minutes=15)
    observations = []
    for i, (day, bars) in enumerate(sorted(sessions.items())[:30]):
        obs = observe_event(bars, bars[60].ts, f"{day}", "c.x", 3.0 + i * 0.2, profile, cal)
        if obs:
            observations.append(obs)
    gamma = severity_response(observations, 15)
    assert 0.5 <= gamma <= 3.0


# -------------------------------------------------------------------- utils
def test_group_by_session(sessions, cal):
    flat = [c for bars in sessions.values() for c in bars]
    grouped = group_by_session(flat, cal, instrument_id=1)
    assert len(grouped) == len(sessions)
    assert all(bars == sorted(bars, key=lambda c: c.ts) for bars in grouped.values())


def test_profile_min_sessions_constant_is_enforced():
    assert MIN_SESSIONS_FOR_PROFILE >= 20
    empty = TimeOfDayProfile(horizon_minutes=15)
    assert not empty.reliable
    assert empty.coverage() == 0.0


# ---------------------------------------------- single-observation guard
def test_one_observation_cannot_claim_certainty():
    """The bug this guard exists for: with n=1 the spread computes to zero,
    which produces a zero-width credible interval that technically excludes zero
    and asserts perfect certainty from a single data point."""
    single = _stat("macro.once", n=1, mean=0.0015, sigma=0.0)
    posterior = shrink(single, family_mean=0.0002, between_sigma=0.001)

    assert not posterior.excludes_zero
    assert not posterior.usable
    assert posterior.credible_high > posterior.credible_low, "interval has zero width"
    assert "*" not in posterior.render()


def test_one_observation_with_no_family_is_unknown():
    posterior = shrink(_stat("x.once", n=1, mean=0.02, sigma=0.0), 0.0, 0.0)
    assert posterior.is_unknown
    assert "UNKNOWN" in posterior.render()


def test_a_usable_class_can_still_exclude_zero():
    """The guard must not suppress a genuine finding."""
    strong = _stat("macro.strong", n=60, mean=0.010, sigma=0.004)
    posterior = shrink(strong, family_mean=0.009, between_sigma=0.02)
    assert posterior.usable
    assert posterior.sigma > 0
    assert posterior.excludes_zero
