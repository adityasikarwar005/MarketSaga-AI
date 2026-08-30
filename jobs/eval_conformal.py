"""Does conformal calibration fix M54's under-coverage, out of sample?

    python jobs/eval_conformal.py
    python jobs/eval_conformal.py --horizon 5 --no-log

`jobs/eval_quantiles.py` found a nominal 90% band holding 83-85% of outcomes on
ten years of real NIFTY bars. `jobs/sizing_sensitivity.py` carried that to M75:
positions 1.09x to 11.38x too large. This asks whether the width can be
corrected without the correction being fitted to the number it is judged on.

Three chronological blocks
----------------------------

    train      fit the quantile heads
    calibrate  choose the conformal scale, on data the fit never saw
    test       measure, on data neither of the above touched

The scale is an order statistic of the calibration block's conformity scores, so
the coverage reported on the test block is a result rather than a restated
objective.

**One honest limit on that claim.** The earlier M54 evaluation walked forward
over this entire series, so no part of it is untouched by the *project's*
history - only by *this procedure's*. A genuinely virgin period would need data
that has not been used yet, and the strongest available statement is the weaker
one: the scale was not chosen on the rows it is scored against.

Two numbers, and the second is the one that usually breaks
------------------------------------------------------------

Marginal coverage is what split conformal guarantees, and it guarantees it under
exchangeability. Daily index returns are not exchangeable - volatility clusters -
so the marginal number can be correct while a regime underneath it is not. M54's
own audit already recorded that shape: 97.1% quiet against 84.6% busy, averaging
to something that read as calibrated. Coverage by volatility tercile is printed
next to the marginal figure for that reason, and a run that fixes the first
while leaving the second split is not a fix.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.eval_quantiles import build_dataset  # noqa: E402
from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.conformal import (  # noqa: E402
    ConformalError,
    apply_conformal,
    coverage,
    coverage_by_regime,
    fit_conformal,
    spread,
)
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_models.quantile import DEFAULT_QUANTILES  # noqa: E402
from libs.ms_models.quantile_heads import QuantileHeads, mean_pinball  # noqa: E402

NOMINAL = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES)

# What `jobs/eval_quantiles.py` measured at horizon 1 under a four-fold
# walk-forward. Kept here so this job can say when it disagrees with it, rather
# than quietly reporting a different number for the same quantity.
WALK_FORWARD_COVERAGE = 0.854


def dataset_version(bars) -> str:
    payload = ";".join(f"{b.day}:{b.o}:{b.h}:{b.low}:{b.c}" for b in bars)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train", type=float, default=0.5, help="share of rows for fitting")
    p.add_argument("--calibrate", type=float, default=0.25, help="share for the scale")
    p.add_argument("--name", default="m54_conformal_real_bars")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)

    configure()
    cfg = cfgmod.load(warn=False)
    clock = RealClock()

    bars = DailyStore().load(args.symbol)
    if not bars:
        print(f"  no cached daily bars for {args.symbol!r}.")
        return 1

    rows, targets, sigmas = build_dataset(bars, horizon=args.horizon)
    n = len(rows)
    a = int(n * args.train)
    b = a + int(n * args.calibrate)
    # At horizon > 1 neighbouring rows share outcome windows, so each block
    # starts horizon-1 rows after the previous one ends.
    embargo = args.horizon - 1
    train = slice(0, a)
    cal = slice(a + embargo, b)
    test = slice(b + embargo, n)

    print()
    print(f"  REAL daily bars: {args.symbol}, {len(bars):,} sessions, horizon {args.horizon}d")
    print(f"  {n:,} rows -> train {a:,} | calibrate {b - a - embargo:,} | test {n - b - embargo:,}")
    print(f"  nominal band {NOMINAL:.0%}")
    print()

    heads = QuantileHeads(quantiles=DEFAULT_QUANTILES, horizons=(args.horizon,))
    heads.fit(
        rows[train],
        {args.horizon: targets[train]},
        sigma_by_horizon={args.horizon: fmean(sigmas[train])},
    )

    def predict(sl: slice):
        return [heads.predict(r, horizon=args.horizon) for r in rows[sl]]

    cal_fc, cal_actual = predict(cal), targets[cal]
    test_fc, test_actual = predict(test), targets[test]
    test_vol = sigmas[test]

    try:
        band = fit_conformal(cal_fc, cal_actual, nominal=NOMINAL)
    except ConformalError as exc:
        print(f"  conformal calibration refused: {exc}")
        return 1
    print(band.render())
    print()

    after_fc = [apply_conformal(f, band) for f in test_fc]
    before_cov = coverage(test_fc, test_actual, nominal=NOMINAL)
    after_cov = coverage(after_fc, test_actual, nominal=NOMINAL)
    before_pin = mean_pinball(test_fc, test_actual, quantiles=DEFAULT_QUANTILES)
    after_pin = mean_pinball(after_fc, test_actual, quantiles=DEFAULT_QUANTILES)

    print(f"    {'':<12} {'cov' + f'{NOMINAL:.0%}':>8} {'width':>10} {'pinball':>10}")
    print(
        f"    {'raw heads':<12} {before_cov:>8.1%} "
        f"{spread(test_fc, nominal=NOMINAL):>10.5f} {before_pin:>10.6f}"
    )
    print(
        f"    {'conformal':<12} {after_cov:>8.1%} "
        f"{spread(after_fc, nominal=NOMINAL):>10.5f} {after_pin:>10.6f}"
    )
    print()

    print("  coverage by volatility tercile - the number conformal does NOT guarantee:")
    print(f"    {'regime':<8} {'n':>6} {'raw':>8} {'conformal':>10}")
    raw_regimes = dict(
        (lab, cov)
        for lab, _n, cov in coverage_by_regime(test_fc, test_actual, test_vol, nominal=NOMINAL)
    )
    conf_regimes = coverage_by_regime(after_fc, test_actual, test_vol, nominal=NOMINAL)
    for label, count, cov in conf_regimes:
        print(f"    {label:<8} {count:>6,} {raw_regimes.get(label, 0):>7.1%} {cov:>10.1%}")
    spread_before = max(raw_regimes.values()) - min(raw_regimes.values())
    spread_after = max(c for _l, _n, c in conf_regimes) - min(c for _l, _n, c in conf_regimes)
    print()
    print(f"    quiet-to-busy gap: {spread_before:.1%} raw -> {spread_after:.1%} conformal")
    print()

    marginal_ok = abs(after_cov - NOMINAL) <= 0.03
    regime_ok = spread_after <= spread_before
    was_broken = abs(before_cov - NOMINAL) > 0.03
    print("  ---")
    if not was_broken:
        print("  NOTHING WAS BROKEN ON THIS SPLIT. The raw heads already cover")
        print(f"  {before_cov:.1%} against a nominal {NOMINAL:.0%}, so the conformal scale")
        print(f"  of x{band.scale:.3f} is correcting an error that is not here.")
        print()
        print("  That is a finding about the earlier measurement, not about this one.")
        print(f"  `eval_quantiles.py` reported {WALK_FORWARD_COVERAGE:.1%} under a four-fold")
        print("  walk-forward on the same series, where the first fold trains on 500")
        print("  rows; this trains on half the series before predicting anything.")
        print("  The under-coverage is therefore at least partly a training-size")
        print("  effect rather than a systematic width bug, and a single number for")
        print("  'the coverage of M54' does not exist - it depends on the split.")
        print()
        print("  Conformal is not shown to fix anything here. It is shown not to")
        print("  break anything, which is the weaker claim and the true one.")
    elif marginal_ok and regime_ok:
        print(f"  Marginal coverage {before_cov:.1%} -> {after_cov:.1%}, within three points")
        print("  of nominal, and the regime gap did not widen. The correction holds")
        print("  out of sample.")
    elif marginal_ok:
        print(f"  Marginal coverage {after_cov:.1%} is right and the regime gap WIDENED")
        print(f"  ({spread_before:.1%} -> {spread_after:.1%}). A single scale cannot fix a")
        print("  band that is wrong by different amounts in different regimes; it")
        print("  moves the error around. Not a pass.")
    else:
        print(f"  Marginal coverage {after_cov:.1%} is still not within three points of")
        print(f"  {NOMINAL:.0%} out of sample. The scale that worked on the calibration")
        print("  block did not transfer, which is what non-exchangeable data does")
        print("  to this method. Recorded rather than retuned.")

    print()
    print("  Width is the price. Coverage bought by widening is not a better")
    print("  forecast, and M75 sizes smaller on a wider band - correctly, but the")
    print("  positions it declines are the cost of having been wrong before.")

    if args.no_log:
        print("\n  --no-log: nothing appended to the experiment log.")
        return 0

    log = ExperimentLog()
    entry = log.record(
        Experiment(
            name=args.name,
            ts=clock.now().astimezone(UTC).isoformat(),
            config_hash=cfg.config_hash,
            dataset_version=dataset_version(bars),
            feature_hash=hashlib.sha256(b"rv_daily,rv_weekly,rv_monthly").hexdigest()[:16],
            metrics={
                "horizon": float(args.horizon),
                "scale": band.scale,
                "calibration_n": float(band.calibration_n),
                "test_n": float(len(test_actual)),
                "cov_before": before_cov,
                "cov_after": after_cov,
                "pinball_before": before_pin,
                "pinball_after": after_pin,
                "regime_gap_before": spread_before,
                "regime_gap_after": spread_after,
                "marginal_ok": float(marginal_ok),
                "regime_ok": float(regime_ok),
                "was_broken": float(was_broken),
            },
            notes=args.note
            or (f"{args.symbol} daily bars, split conformal, horizon {args.horizon}"),
        )
    )
    print(f"\n  logged as trial {log.trial_count()} ({entry.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
