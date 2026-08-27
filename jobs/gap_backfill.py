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
import sys
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
    result = walk_forward_evaluate(samples, registry, n_folds=args.folds)
    if result is None:
        print("  not enough sessions for a walk-forward evaluation")
        return 1
    print(result.render())

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
