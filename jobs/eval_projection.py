"""M62 - the projection scorer, and Gate G6.

    python jobs/eval_projection.py --walk-forward --report
    python jobs/eval_projection.py --walk-forward --report --headroom

Gate G6 asks for three things at once: 90% coverage within three points, a
uniform PIT, and CRPS beating **every** baseline by 8%.

``--headroom`` answers the question that should be asked before any of them:
how much room is there between a fitted baseline and a perfect forecast? An
oracle handed the true conditional variance beats a fitted GARCH by roughly a
fifth of a percent on GARCH-like data, so an 8% margin there would say more
about the baseline than the model.

Nothing here has run on real market data. The series is synthetic and its
generating process is known, which is what lets the oracle exist at all.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_models.projection import (
    MIN_CRPS_IMPROVEMENT,
    gaussian_forecast,
    walk_forward,
)

WARMUP = 600
N = 1400


def synthetic_series(
    n: int = N,
    seed: int = 9,
    *,
    a: float = 0.10,
    b: float = 0.85,
    burst_every: int = 0,
    burst_size: float = 9.0,
) -> tuple[list[float], list[float]]:
    """A GARCH series, optionally with event bursts a GARCH cannot anticipate.

    The bursts are the point of ``--headroom``: they are the structure a
    conditional model could know about in advance and a GARCH baseline could
    not, which is the only place an 8% CRPS margin can come from.
    """
    rng = random.Random(seed)
    h = 1e-4
    series: list[float] = []
    variances: list[float] = []
    for i in range(n):
        # A scheduled event: known in advance, and invisible to a variance
        # recursion driven only by past returns.
        v = h * burst_size if (burst_every and i % burst_every == 0) else h
        variances.append(v)
        r = math.sqrt(v) * rng.gauss(0.0, 1.0)
        series.append(r)
        h = 1e-5 + a * r * r + b * h
    return series, variances


def oracle(variances: list[float], warmup: int) -> list:
    return [
        gaussian_forecast(math.sqrt(variances[warmup + i])) for i in range(len(variances) - warmup)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--walk-forward", action="store_true", help="run the walk-forward scoring")
    parser.add_argument("--report", action="store_true", help="print the full report")
    parser.add_argument(
        "--headroom",
        action="store_true",
        help="show what an oracle can win against each baseline, with and without "
        "structure the baselines cannot see",
    )
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args(argv)

    if not args.walk_forward:
        parser.error("--walk-forward is the only mode; there is no in-sample scoring here")

    configure()
    print("  SYNTHETIC series with a known generating process - not real market data.\n")

    series, variances = synthetic_series(seed=args.seed)
    result = walk_forward(
        series,
        oracle(variances, WARMUP),
        warmup=WARMUP,
        implied_vols=[0.16] * (len(series) - WARMUP),
        horizon_years=1 / 252,
        name="oracle-vol",
    )

    if args.report:
        print(result.render())

    if args.headroom:
        print("\n  headroom: what a PERFECT forecast can win against each baseline")
        print("  (an oracle handed the true conditional variance of the process)\n")
        print(f"    {'predictable structure':<28}{'vs random-walk':>16}{'vs garch':>11}")
        for every, size in ((0, 1.0), (37, 9.0), (15, 9.0), (7, 9.0), (3, 25.0)):
            s_, v_ = synthetic_series(seed=args.seed, burst_every=every, burst_size=size)
            r = walk_forward(s_, oracle(v_, WARMUP), warmup=WARMUP, name="oracle-vol")
            rw = (r.random_walk.crps - r.challenger.crps) / r.random_walk.crps
            gr = (r.garch.crps - r.challenger.crps) / r.garch.crps
            label = "none (plain GARCH)" if not every else f"1 step in {every}, x{size:g} var"
            mark = "  <- clears 8%" if gr >= MIN_CRPS_IMPROVEMENT else ""
            print(f"    {label:<28}{rw:>15.2%}{gr:>11.2%}{mark}")
        print(
            "\n    The gate wants +8% against each baseline. On a plain GARCH series a\n"
            "    perfect forecast gets 0.2% against a fitted GARCH - because there the\n"
            "    fitted GARCH *is* the oracle. The margin only appears once a large\n"
            "    share of the variance is scheduled and known in advance: on this grid\n"
            "    it takes roughly one step in seven carrying nine times the variance.\n\n"
            "    Two things follow. The 8% bar is demanding rather than routine, and a\n"
            "    comfortable pass on a series without that structure is evidence about\n"
            "    the baseline rather than about the model."
        )

    return 0 if result.gate.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
