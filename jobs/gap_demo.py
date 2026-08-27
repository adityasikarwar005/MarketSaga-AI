"""Gate G4 discrimination check for the overnight gap forecaster.

    python jobs/gap_demo.py

Runs the walk-forward evaluation twice on synthetic sessions: once where the gap
genuinely follows the overnight driver, and once where it is pure noise. The
model should pass the gate on the first and fail on the second.

That second run is the one that matters. A gate that passes on noise is not a
gate, and the only way to know is to point it at noise and watch it refuse. Real
data is never labelled "this contains no signal", so this is the only setting in
which the check can actually be made.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import configure
from libs.ms_data.crossasset import SeriesMove, overnight_window, registry_from_config
from libs.ms_data.holidays import HolidayStore
from libs.ms_models.gap import (
    GapForecaster,
    GapSample,
    build_features,
    feature_names,
    walk_forward_evaluate,
)


def make_sessions(
    registry,
    calendar,
    n: int,
    *,
    signal: float,
    noise: float,
    seed: int,
    heteroscedastic: bool = False,
) -> list[GapSample]:
    rng = random.Random(seed)
    names = feature_names(registry)
    out: list[GapSample] = []
    day = date(2026, 1, 5)
    while len(out) < n:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        window = overnight_window(calendar, day)
        driver = rng.gauss(0.0, 0.006)
        moves = {
            s.symbol: SeriesMove(
                symbol=s.symbol,
                instrument_id=s.instrument_id,
                first=100.0,
                last=100.0 * (1 + (driver if s.symbol == "US_EQ_FUT" else rng.gauss(0, 0.004))),
                bars=120,
                high=101.0,
                low=99.0,
            )
            for s in registry.all()
        }
        features = build_features(window, moves, registry, prior_session_return=rng.gauss(0, 0.005))
        # A longer window means more has happened, so the gap distribution is
        # genuinely wider. The naive model carries one fixed set of residual
        # quantiles and cannot express that; a quantile model conditioned on the
        # window can. This is the kind of structure a distributional model is
        # actually for.
        spread = noise
        if heteroscedastic:
            spread = noise * (1.0 + 1.5 * (window.hours / 17.0 - 1.0))
        out.append(
            GapSample(
                session=day,
                features={k: features.get(k, 0.0) for k in names},
                target=signal * driver + rng.gauss(0.0, max(1e-6, spread)),
                window_hours=window.hours,
                spans_weekend=window.spans_multiple_days,
            )
        )
        day += timedelta(days=1)
    return out


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Gate G4 discrimination check (M35)")
    p.add_argument("--sessions", type=int, default=300)
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args()

    cfg = cfgmod.load(warn=False)
    calendar = SessionCalendar.from_config(cfg.sessions).with_holidays(
        HolidayStore().as_calendar_dict()
    )
    registry = registry_from_config(cfg.crossasset.series)

    print(f"\n  {args.sessions} synthetic sessions, {args.folds}-fold walk-forward\n")

    scenarios = (
        # The naive model IS the true model here, so the best the gap model can
        # do is match it minus the cost of estimating coefficients it does not
        # need. Failing this scenario is correct.
        ("driver only - the naive model is the true model", 0.6, 0.004, 3, False),
        # Structure the naive model cannot express: the spread widens with the
        # length of the overnight window.
        ("driver plus a window-dependent spread", 0.6, 0.004, 5, True),
        # Nothing to find. The gate must refuse.
        ("no relationship at all - pure noise", 0.0, 0.006, 77, False),
    )
    for label, signal, noise, seed, hetero in scenarios:
        samples = make_sessions(
            registry,
            calendar,
            args.sessions,
            signal=signal,
            noise=noise,
            seed=seed,
            heteroscedastic=hetero,
        )
        result = walk_forward_evaluate(samples, registry, n_folds=args.folds)
        print(f"  {label}")
        if result is None:
            print("    not enough sessions to evaluate\n")
            continue
        print(result.render())
        print()

    # One worked example, so the output is a forecast a person can read.
    samples = make_sessions(registry, calendar, 200, signal=0.6, noise=0.004, seed=3)
    model = GapForecaster(registry, epochs=300).fit(samples[:150])
    print(model.predict(samples[160]).render())

    print(
        "\n  The second run is the point. A gate that passes on noise is not a"
        "\n  gate, and real data never arrives labelled 'no signal here'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
