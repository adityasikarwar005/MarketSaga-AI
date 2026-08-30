"""What the calibration error costs M75, per regime and with a sign.

    python jobs/sizing_by_regime.py --horizon 5

`jobs/sizing_sensitivity.py` reported this as "positions 1.09x to 11.38x too
large" from a single widening factor. **That framing was one-directional and it
was wrong.** Mondrian conformal fits a scale per volatility bucket and the
scales came back x0.803, x1.035, x1.188 - so the band is too *wide* in the quiet
tercile and too *narrow* in the busy one, and the sizing error changes sign
between them.

An over-wide band in a quiet regime makes every position smaller than it should
be. That costs opportunity and nothing else, and it is invisible in a P&L. An
under-wide band in a busy regime makes every position larger than it should be,
on exactly the days the tail arrives. Averaging the two into one number - which
is what the earlier job did - hides that one of them is expensive and the other
is not.

**Two channels, still compounding.** `KellyInput.p_lower` runs the formula on
the pessimistic end of the model's interval, and `uncertainty_modifier` shrinks
as the interval widens. A band that is wrong in width is wrong through both, in
the same direction, at once.

**This changes nothing.** It is a measurement of an error that is live, with the
per-regime scales taken from a conformal fit on a calibration block the
measurement does not include.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.eval_quantiles import build_dataset  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_data.daily import DailyStore  # noqa: E402
from libs.ms_models.conformal import ConformalError, fit_mondrian  # noqa: E402
from libs.ms_models.quantile import DEFAULT_QUANTILES  # noqa: E402
from libs.ms_models.quantile_heads import QuantileHeads  # noqa: E402
from libs.ms_options.sizing import KellyInput, kelly_fraction, uncertainty_modifier  # noqa: E402

NOMINAL = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES)

# Representative bets. The ratio is what matters and it is insensitive to these,
# but one hand-picked example would invite the reading that it was chosen.
BETS = (
    ("even money, modest edge", 0.55, 1.0),
    ("even money, strong edge", 0.62, 1.0),
    ("2:1 payoff, modest edge", 0.42, 2.0),
)
WIDTHS = (0.03, 0.06, 0.10)


def sized(p_win: float, odds_win: float, width: float) -> float:
    """Kelly x the uncertainty modifier, which is what M75 actually applies."""
    p_lower = max(0.0, p_win - width)
    f = kelly_fraction(
        KellyInput(p_win=p_win, win_amount=odds_win, loss_amount=1.0, p_lower=p_lower)
    )
    return f * uncertainty_modifier(p_win, p_lower).value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--train", type=float, default=0.5)
    p.add_argument("--calibrate", type=float, default=0.25)
    p.add_argument("--buckets", type=int, default=3)
    args = p.parse_args(argv)

    configure()
    bars = DailyStore().load(args.symbol)
    if not bars:
        print(f"  no cached daily bars for {args.symbol!r}.")
        return 1

    rows, targets, sigmas = build_dataset(bars, horizon=args.horizon)
    n = len(rows)
    a = int(n * args.train)
    b = a + int(n * args.calibrate)
    embargo = args.horizon - 1

    heads = QuantileHeads(quantiles=DEFAULT_QUANTILES, horizons=(args.horizon,))
    heads.fit(
        rows[:a],
        {args.horizon: targets[:a]},
        sigma_by_horizon={args.horizon: fmean(sigmas[:a])},
    )
    cal = slice(a + embargo, b)
    cal_fc = [heads.predict(r, horizon=args.horizon) for r in rows[cal]]
    try:
        bands = fit_mondrian(
            cal_fc, targets[cal], sigmas[cal], nominal=NOMINAL, buckets=args.buckets
        )
    except ConformalError as exc:
        print(f"  mondrian not fitted: {exc}")
        return 1

    print()
    print(f"  REAL daily bars: {args.symbol}, horizon {args.horizon}d")
    for line in bands.render():
        print(line)
    print()
    print("  A scale above 1 means the band was too NARROW there, so M75 sized")
    print("  too large. Below 1 means too wide, and it sized too small.")
    print()

    labels = ["quiet"] + [f"mid{i}" for i in range(1, args.buckets - 1)] + ["busy"]
    worst_over = 0.0
    for label, band in zip(labels, bands.bands, strict=True):
        k = band.scale
        direction = "TOO LARGE" if k > 1 else "too small"
        print(f"  {label}  scale x{k:.3f}  ->  positions {direction}")
        print(f"    {'bet':<28} {'width':>6} {'as sized':>10} {'honest':>10} {'ratio':>9}")
        for name, p_win, odds in BETS:
            for width in WIDTHS:
                narrow, honest = sized(p_win, odds, width), sized(p_win, odds, k * width)
                if narrow <= 0.0 and honest <= 0.0:
                    verdict = "flat"
                elif honest <= 0.0:
                    # The band was so narrow that the honest one removes the
                    # edge entirely, and the system takes the position anyway.
                    verdict = "SHOULD BE FLAT"
                elif narrow <= 0.0:
                    # The mirror case, and the one the earlier job had no name
                    # for: an over-wide band shrinks the edge to nothing, so
                    # the system sits out a trade the honest interval would
                    # have taken. It costs nothing measurable, which is exactly
                    # why it goes uncorrected.
                    verdict = "MISSED"
                else:
                    ratio = narrow / honest
                    verdict = f"{ratio:.2f}x"
                    if k > 1:
                        worst_over = max(worst_over, ratio)
                print(f"    {name:<28} {width:>6.2f} {narrow:>10.4f} {honest:>10.4f} {verdict:>9}")
        print()

    print("  ---")
    quiet_scale, busy_scale = bands.scales[0], bands.scales[-1]
    if busy_scale > 1.0:
        print(f"  In the busy tercile the band is {(busy_scale - 1):.1%} too narrow and every")
        print(f"  position is up to {worst_over:.2f}x too large - on the days the tail")
        print("  arrives. That is the expensive half.")
    if quiet_scale < 1.0:
        print(f"  In the quiet tercile it is {(1 - quiet_scale):.1%} too wide and every position")
        print("  is smaller than it should be. That costs opportunity and shows up")
        print("  in no P&L, which is why it is easy to leave uncorrected.")
    print()
    print("  The earlier `sizing_sensitivity.py` reported one number for all of")
    print("  this and gave it one direction. Both halves are real and only one")
    print("  of them costs money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
