"""Fit and evaluate the overnight gap forecaster on real daily sessions.

    python jobs/gap_backfill.py --fetch          download and cache two years
    python jobs/gap_backfill.py                  evaluate from the cache
    python jobs/gap_backfill.py --forecast       the next session's gap

This is the first component in the project to run on real market data. It needs
no broker, no subscription and no low-latency infrastructure - which is exactly
why it comes first.

What it reports is a genuine out-of-sample result, and it may well be that the
gate does not pass. That is a finding, not a failure: it would say the daily
cross-asset state does not forecast the Indian open well enough to beat a
univariate regression on the strongest driver, which is worth knowing before
anything is built on top of it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.obs import configure
from libs.ms_data.crossasset import registry_from_config
from libs.ms_data.daily import (
    DailyStore,
    approximate_window_hours,
    build_overnight_series,
    coverage_report,
    session_span,
)
from libs.ms_models.cv import Experiment, ExperimentLog
from libs.ms_models.gap import (
    MIN_SESSIONS_TO_FIT,
    GapForecaster,
    GapSample,
    NaiveGapBaseline,
    feature_names,
    walk_forward_evaluate,
)
from libs.ms_models.quantile import score_forecasts

INDEX = "NIFTY"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def lean_features(registry, coverage: dict[str, float], *, min_coverage: float = 0.9) -> list[str]:
    """One column per series that is actually present, plus the window shape.

    The full set gives every configured series three columns - return, range and
    a seen flag - which is 27 features against roughly a hundred training rows in
    the first fold. The seen flags are near-constant once a series is in the
    cache, the ranges are collinear with the returns, and a series at 0% coverage
    contributes three columns of zeros that the model can still fit noise to.

    Chosen from measured coverage rather than from trying several sets and
    keeping the best, which would make the comparison meaningless.
    """
    names = [
        f"{s.symbol}_ret" for s in registry.all() if coverage.get(s.symbol, 0.0) >= min_coverage
    ]
    return [*names, "window_hours", "spans_weekend", "prior_session_return"]


def dataset_version(series) -> str:
    """Identifies the exact data an attempt was scored on.

    Two runs reporting different numbers are only comparable if this string
    matches; if it does not, the model may not have changed at all - the cache
    just grew by a week.
    """
    span = session_span(series)
    return f"{len(series)}@{span[0]}..{span[1]}" if span else "empty"


def load_series(store: DailyStore, symbols: list[str], *, years: int, refresh: bool):
    index_bars = store.get(INDEX, years=years, refresh=refresh)
    foreign = {}
    for symbol in symbols:
        try:
            bars = store.get(symbol, years=years, refresh=refresh)
        except Exception as exc:
            print(f"    {symbol:<12} unavailable ({type(exc).__name__})")
            continue
        if bars:
            foreign[symbol] = bars
    return index_bars, foreign


def to_samples(series, registry) -> list[GapSample]:
    """Overnight state to model input, in the shape the forecaster expects."""
    names = feature_names(registry)
    out: list[GapSample] = []
    for entry in series:
        features: dict[str, float] = {}
        for s in registry.all():
            ret = entry.foreign_returns.get(s.symbol)
            features[f"{s.symbol}_ret"] = ret if ret is not None else 0.0
            features[f"{s.symbol}_range"] = entry.foreign_ranges.get(s.symbol, 0.0)
            features[f"{s.symbol}_seen"] = 1.0 if ret is not None else 0.0
        hours = approximate_window_hours(entry)
        features["window_hours"] = hours
        features["spans_weekend"] = 1.0 if entry.spans_multiple_days else 0.0
        features["day_of_week"] = float(entry.session.weekday())
        features["prior_session_return"] = entry.prior_session_return

        out.append(
            GapSample(
                session=entry.session,
                features={k: features.get(k, 0.0) for k in names},
                target=entry.gap,
                window_hours=hours,
                spans_weekend=entry.spans_multiple_days,
            )
        )
    return out


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Gap forecaster on real daily data (M35)")
    p.add_argument("--fetch", action="store_true", help="refresh the cached series")
    p.add_argument("--years", type=int, default=2)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--forecast", action="store_true", help="predict the next session")
    p.add_argument(
        "--name",
        default="gap_forecaster",
        help="what to call this attempt in the trial log",
    )
    p.add_argument(
        "--note",
        default="",
        help="why this attempt differs from the last one",
    )
    p.add_argument(
        "--lean",
        action="store_true",
        help="one column per covered series instead of three per configured series",
    )
    args = p.parse_args()

    cfg = cfgmod.load(warn=False)
    registry = registry_from_config(cfg.crossasset.series)
    store = DailyStore()
    symbols = [s.symbol for s in registry.all()]

    print(f"\n  {'fetching' if args.fetch else 'loading'} {len(symbols) + 1} daily series\n")
    index_bars, foreign = load_series(store, symbols, years=args.years, refresh=args.fetch)

    if not index_bars:
        print("  no index data. Run with --fetch (needs network).")
        return 1
    print(f"    {INDEX:<12} {len(index_bars):>4} sessions")
    for symbol, bars in sorted(foreign.items()):
        print(f"    {symbol:<12} {len(bars):>4} sessions")

    series = build_overnight_series(index_bars, foreign)
    span = session_span(series)
    print(f"\n  {len(series):,} overnight windows" + (f", {span[0]} .. {span[1]}" if span else ""))

    coverage = coverage_report(series, symbols)
    thin = {s: c for s, c in coverage.items() if c < 0.9}
    if thin:
        print("  thin coverage (a series present on half the sessions is not a feature):")
        for symbol, share in sorted(thin.items(), key=lambda kv: kv[1]):
            print(f"    {symbol:<12} {share:.0%}")

    samples = to_samples(series, registry)
    gaps = [s.target for s in samples]
    if gaps:
        ordered = sorted(abs(g) for g in gaps)
        print(
            f"\n  realised gaps: median |gap| {ordered[len(ordered) // 2]:.3%}, "
            f"90th pct {ordered[int(0.9 * len(ordered))]:.3%}, "
            f"max {ordered[-1]:.3%}"
        )

    if len(samples) < MIN_SESSIONS_TO_FIT:
        print(f"\n  only {len(samples)} sessions - need {MIN_SESSIONS_TO_FIT} to fit")
        return 1

    print()
    features = lean_features(registry, coverage) if args.lean else feature_names(registry)
    if args.lean:
        print(
            f"  lean feature set: {len(features)} columns instead of "
            f"{len(feature_names(registry))}"
        )
    result = walk_forward_evaluate(samples, registry, n_folds=args.folds, features=features)
    if result is None:
        print("  not enough sessions for a walk-forward evaluation")
        return 1
    print(result.render())

    # Record the attempt before reading the result, not after. An evaluation
    # that is only logged when the author likes the number is not a trial count,
    # it is a highlight reel - and the deflated Sharpe computed from it would
    # overstate every subsequent claim.
    log = ExperimentLog()
    entry = log.record(
        Experiment(
            name=args.name,
            ts=datetime.now(UTC).isoformat(),
            config_hash=cfg.config_hash,
            dataset_version=dataset_version(series),
            feature_hash=_hash(",".join(feature_names(registry))),
            metrics={
                "crps": result.comparison.challenger.crps,
                "crps_naive": result.comparison.baseline.crps,
                "improvement": result.comparison.crps_improvement,
                "cov90": result.comparison.challenger.coverage_90,
                "passes_g4": float(result.passes_g4),
                "sessions": float(len(samples)),
                "features": float(len(features)),
            },
            notes=args.note,
        )
    )
    print()
    print(f"  logged as trial {log.trial_count()} ({entry.name})")

    # Climatology floor: what a forecaster that ignores every feature achieves.
    split = int(len(samples) * 0.75)
    clim_model = GapForecaster(registry, epochs=200).fit(samples[:split])
    clim = clim_model.climatology
    clim_score = score_forecasts(
        "climatology",
        [clim.predict(s.vector(clim_model.names)) for s in samples[split:]],
        [s.target for s in samples[split:]],
    )
    print("\n  floor for reference:")
    print(clim_score.render())

    if args.forecast:
        model = GapForecaster(registry).fit(samples[:-1])
        print()
        print(model.predict(samples[-1]).render())
        naive = NaiveGapBaseline().fit(samples[:-1])
        print(f"    naive    : {naive.predict(samples[-1]).render()}")
        print(f"    realised : {samples[-1].target:+.3%}")

    print(
        "\n  Daily bars only, so the overnight window is inferred rather than"
        "\n  measured, and every foreign series contributes its D-1 session."
        "\n  A minute-level feed would recover the hours this discards."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
