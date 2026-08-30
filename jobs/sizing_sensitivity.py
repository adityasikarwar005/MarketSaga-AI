"""What the measured under-coverage does to position size (M54 -> M75).

    python jobs/sizing_sensitivity.py

`jobs/eval_quantiles.py` found that a nominal 90% band from either the fitted
heads or their Gaussian fallback holds the outcome only 80-85% of the time on
ten years of real NIFTY bars. That is a statement about M54. This job asks the
question that makes it matter: **M75 sizes against those intervals, so how much
too large is every position?**

Two channels, and they compound rather than cancel.

**The conservative substitution stops being conservative.** `KellyInput.p_lower`
exists so the formula runs on the pessimistic end of the model's own interval
instead of its point estimate. If the interval is too narrow, `p_lower` sits too
close to `p_win`, the substitution gives back less than it was meant to, and the
Kelly fraction comes out high.

**The uncertainty modifier under-shrinks.** `uncertainty_modifier` reads the
interval half-width and scales the position down as it widens. A width that is
understated reads as a confident model, and the multiplier stays near 1.0 when
it should have been cutting.

Both push the same way. This job measures the widening factor the real data
demands, then applies it through both channels and reports the ratio.

**It changes nothing.** Widening the heads until coverage hits 90% would be
tuning a number until it passes the check that exists to catch it. The output
here is a measurement of an existing error, not a fix for it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.eval_quantiles import build_dataset, coverage, walk_forward  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.quantile import DEFAULT_QUANTILES  # noqa: E402
from libs.ms_options.sizing import KellyInput, kelly_fraction, uncertainty_modifier  # noqa: E402

# Representative bets. The point is the ratio, which is insensitive to these -
# but a single hand-picked example would invite the reading that the number was
# chosen, so a spread is shown and the range is reported.
BETS = (
    ("even money, modest edge", 0.55, 1.0),
    ("even money, strong edge", 0.62, 1.0),
    ("2:1 payoff, modest edge", 0.42, 2.0),
    ("1:2 payoff, high hit rate", 0.72, 0.5),
)

# Interval half-widths a direction model plausibly reports, in probability.
WIDTHS = (0.03, 0.06, 0.10)


def widening_for_nominal(forecasts, actuals, *, nominal: float, hi: float = 4.0) -> float:
    """Factor the 5-95 band must be scaled by to reach nominal coverage.

    Bisection on the scale rather than a closed form: the band is not symmetric
    around the median once the heads have fitted, so widening is applied to each
    side about the median and the coverage is recounted.
    """
    lo_a, hi_a = min(DEFAULT_QUANTILES), max(DEFAULT_QUANTILES)
    mid = 0.5

    def cov_at(scale: float) -> float:
        inside = 0
        for f, actual in zip(forecasts, actuals, strict=True):
            centre = f.quantiles[mid]
            lo = centre + (f.quantiles[lo_a] - centre) * scale
            high = centre + (f.quantiles[hi_a] - centre) * scale
            if lo <= actual <= high:
                inside += 1
        return inside / len(actuals)

    if cov_at(1.0) >= nominal:
        return 1.0
    lo_s, hi_s = 1.0, hi
    if cov_at(hi_s) < nominal:
        return hi_s  # even four times as wide does not reach it; report the cap
    for _ in range(40):
        mid_s = (lo_s + hi_s) / 2.0
        if cov_at(mid_s) < nominal:
            lo_s = mid_s
        else:
            hi_s = mid_s
    return hi_s


def size_ratio(p_win: float, odds_win: float, width: float, k: float) -> tuple[float, float]:
    """Kelly x uncertainty, under the narrow interval and under the honest one.

    ``width`` is the half-width the model reports; ``k * width`` is what the
    coverage measurement says it should have been.
    """
    out = []
    for w in (width, k * width):
        p_lower = max(0.0, p_win - w)
        f = kelly_fraction(
            KellyInput(p_win=p_win, win_amount=odds_win, loss_amount=1.0, p_lower=p_lower)
        )
        out.append(f * uncertainty_modifier(p_win, p_lower).value)
    return out[0], out[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--min-train", type=int, default=500)
    args = p.parse_args(argv)

    configure()
    bars = DailyStore().load(args.symbol)
    if not bars:
        print(f"  no cached daily bars for {args.symbol!r}.")
        return 1

    nominal = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES)
    rows, targets, sigmas = build_dataset(bars, horizon=args.horizon)
    out = walk_forward(
        rows, targets, sigmas, horizon=args.horizon, folds=args.folds, min_train=args.min_train
    )
    if out is None:
        print("  not enough rows to walk forward.")
        return 1
    head_fc, base_fc, actual, _fallbacks = out

    print()
    print(f"  REAL daily bars: {args.symbol}, {len(bars):,} sessions, horizon {args.horizon}d")
    print(f"  tested on {len(actual):,} out-of-sample rows, nominal band {nominal:.0%}")
    print()

    k_head = widening_for_nominal(head_fc, actual, nominal=nominal)
    k_base = widening_for_nominal(base_fc, actual, nominal=nominal)
    print(
        f"    heads     coverage {coverage(head_fc, actual):.1%}  "
        f"-> band must widen x{k_head:.2f} to reach {nominal:.0%}"
    )
    print(
        f"    Gaussian  coverage {coverage(base_fc, actual):.1%}  "
        f"-> band must widen x{k_base:.2f} to reach {nominal:.0%}"
    )
    print()

    k = k_head
    print(f"  Applying x{k:.2f} to the interval M75 sizes against.")
    print("  Kelly x uncertainty modifier, narrow interval vs honest one:")
    print()
    print(f"    {'bet':<28} {'width':>6} {'as sized':>10} {'honest':>10} {'too large by':>13}")
    ratios: list[float] = []
    flipped = 0  # sized > 0 under the narrow interval, 0 under the honest one
    already_flat = 0  # no position either way - not an error, and not evidence
    for label, p_win, odds in BETS:
        for width in WIDTHS:
            narrow, honest = size_ratio(p_win, odds, width, k)
            if narrow <= 0.0 and honest <= 0.0:
                already_flat += 1
                verdict = "flat either way"
            elif honest <= 0.0:
                flipped += 1
                verdict = "SHOULD BE FLAT"
            else:
                r = narrow / honest
                ratios.append(r)
                verdict = f"{r:.2f}x"
            print(f"    {label:<28} {width:>6.2f} {narrow:>10.4f} {honest:>10.4f} {verdict:>13}")
    print()

    total = len(BETS) * len(WIDTHS)
    if ratios:
        print(
            f"  Where a position is taken either way ({len(ratios)} of {total} cases), it is "
            f"{min(ratios):.2f}x to {max(ratios):.2f}x too large."
        )
    if flipped:
        print(f"  In {flipped} of {total}, the honest interval removes the edge entirely:")
        print("  Kelly goes to zero, the correct size is no position, and the system")
        print("  currently takes one.")
    if already_flat:
        print(f"  In {already_flat} of {total}, both intervals give no position. Those")
        print("  are not evidence of anything - the edge was already too thin for")
        print("  the modifier to matter, and counting them as failures would inflate")
        print("  the finding.")
    print()
    print("  Read this as the size of an existing error, not as a correction.")
    print("  Nothing here widens any interval: the heads are recalibrated by a")
    print("  design decision with its own justification and its own held-out")
    print("  period, or not at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
