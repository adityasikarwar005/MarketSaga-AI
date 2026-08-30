"""Split-conformal calibration for the quantile heads.

`jobs/eval_quantiles.py` measured a nominal 90% band holding the outcome 83-85%
of the time on ten years of real NIFTY bars, and `jobs/sizing_sensitivity.py`
carried that through to M75: where a position is taken at all it is 1.09x to
11.38x too large, worst exactly where the interval is widest and the model least
sure.

The obvious response is to widen the heads until coverage reaches 90%. That is
tuning the number the check exists to catch, and it would report a pass on the
data it was tuned on. This module is the version that cannot do that.

Three blocks, and the third is never touched by the first two
---------------------------------------------------------------

Fit the heads on block one. Compute the conformity scores on block two, which
the fit has never seen. Measure on block three, which neither has. The scale
that comes out is chosen on data the measurement does not include, so the
coverage reported at the end is an out-of-sample number rather than a restated
objective.

The conformity score
----------------------

For one observation, the smallest factor the band would have needed to contain
it, measured about the median so an asymmetric band stays asymmetric:

    actual above the median   ->  (actual - median) / (upper - median)
    actual below the median   ->  (median - actual) / (median - lower)

The ``nominal`` empirical quantile of those scores is the multiplier. A value
above 1 widens, below 1 narrows - and narrowing is a legitimate outcome that the
implementation does not prevent, because a floor at 1 would turn this into a
one-way ratchet that can only ever agree with the diagnosis.

The guarantee is weaker here than the textbook says
-----------------------------------------------------

Split conformal gives finite-sample marginal coverage **under exchangeability**,
and daily index returns are not exchangeable: volatility clusters, so a
calibration block drawn from a quiet period under-scales for a busy one. The
guarantee that survives is marginal - about the average over the test block - and
says nothing about any particular regime.

That is exactly the failure M54's own audit already recorded, where a fitted band
covered 97.1% in the quiet regime and 84.6% in the busy one and averaged to
something that read as calibrated. So :func:`coverage_by_regime` exists and any
report that quotes the marginal number without it is quoting half the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_models.quantile import DEFAULT_QUANTILES, QuantileForecast

log = get_logger(__name__)

# Below this the calibration quantile is being read off too few points for its
# own order statistic to mean anything: at n=20 the 90th percentile is the
# second-largest value, and one outlier moves it entirely.
MIN_CALIBRATION_ROWS = 100


# Quantile levels are dict keys, and deriving them by arithmetic does not
# produce the keys that are there: (1.0 - 0.9) / 2 is 0.05000000000000002, not
# 0.05. Matched by proximity instead, within a tolerance far tighter than any
# two levels a caller would sensibly ask for.
LEVEL_TOLERANCE = 1e-6


class ConformalError(RuntimeError):
    pass


def band_edges(forecast: QuantileForecast, nominal: float) -> tuple[float, float]:
    """The two keys in ``forecast`` that bound a ``nominal`` interval.

    Returns the actual keys rather than the computed levels, so every later
    lookup uses a value that is present rather than one that is nearly present.
    """
    want_lo = (1.0 - nominal) / 2.0
    want_hi = 1.0 - want_lo
    levels = sorted(forecast.quantiles)
    found: list[float] = []
    for want in (want_lo, want_hi):
        closest = min(levels, key=lambda a: abs(a - want)) if levels else None
        if closest is None or abs(closest - want) > LEVEL_TOLERANCE:
            raise ConformalError(
                f"forecast has no quantile at {want:.6g} for a {nominal:.0%} band; "
                f"it carries {levels}"
            )
        found.append(closest)
    if 0.5 not in forecast.quantiles:
        raise ConformalError(f"forecast has no median; it carries {levels}")
    return found[0], found[1]


@dataclass(frozen=True)
class ConformalBand:
    """The multiplier, and enough context to know whether to believe it."""

    scale: float
    nominal: float
    calibration_n: int
    # Where the scale sits among the scores it was chosen from. A scale at the
    # very top of the calibration set is an extrapolation dressed as a quantile.
    score_max: float

    @property
    def extrapolating(self) -> bool:
        return self.scale >= self.score_max

    def render(self) -> str:
        direction = "widen" if self.scale > 1 else "narrow"
        note = "  (at the top of the calibration set)" if self.extrapolating else ""
        return (
            f"    conformal scale x{self.scale:.3f} to {direction}, "
            f"chosen on {self.calibration_n:,} rows for {self.nominal:.0%}{note}"
        )


def conformity_score(forecast: QuantileForecast, actual: float, *, nominal: float) -> float:
    """The smallest factor this band needed to contain ``actual``.

    Measured about the median so an asymmetric band scales asymmetrically - a
    skewed forecast that is right about the skew should keep it.
    """
    lo_a, hi_a = band_edges(forecast, nominal)
    q = forecast.quantiles
    median = q[0.5]
    half = q[hi_a] - median if actual >= median else median - q[lo_a]
    if half <= 0:
        # A degenerate band contains nothing above or below its median. Reported
        # as infinite rather than skipped: dropping it would quietly improve the
        # calibration set by removing its worst members.
        return math.inf
    return abs(actual - median) / half


def fit_conformal(
    forecasts: Sequence[QuantileForecast],
    actuals: Sequence[float],
    *,
    nominal: float = max(DEFAULT_QUANTILES) - min(DEFAULT_QUANTILES),
) -> ConformalBand:
    """The scale that would have given ``nominal`` coverage on this block.

    Uses the ceil((n+1) * nominal) order statistic - the finite-sample form,
    not the plain percentile. On 100 rows at 90% the difference is the 91st
    value rather than the 90th, and the correction is the whole reason the
    method has a guarantee rather than an asymptotic hope.
    """
    if len(forecasts) != len(actuals):
        raise ConformalError("forecasts and actuals must be the same length")
    if len(forecasts) < MIN_CALIBRATION_ROWS:
        raise ConformalError(
            f"{len(forecasts)} calibration rows is too few (need "
            f"{MIN_CALIBRATION_ROWS}). The quantile would be read off an order "
            "statistic that one outlier moves entirely."
        )
    scores = sorted(
        conformity_score(f, a, nominal=nominal) for f, a in zip(forecasts, actuals, strict=True)
    )
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * nominal))
    scale = scores[rank - 1]
    if not math.isfinite(scale):
        raise ConformalError(
            "the calibration quantile is infinite: more than "
            f"{(1 - nominal):.0%} of these bands have zero width on one side, "
            "so no finite multiplier reaches nominal coverage."
        )
    band = ConformalBand(scale=scale, nominal=nominal, calibration_n=n, score_max=scores[-1])
    METRICS.observe("conformal_scale", scale)
    log.info(
        "conformal band fitted",
        fields={"scale": scale, "n": n, "nominal": nominal, "extrapolating": band.extrapolating},
    )
    return band


def apply_conformal(forecast: QuantileForecast, band: ConformalBand) -> QuantileForecast:
    """Scale every quantile about the median. The median itself does not move.

    Only the width was found to be wrong; moving the centre would be a second,
    unmeasured correction riding along with the first.
    """
    median = forecast.quantiles[0.5]
    return QuantileForecast(
        quantiles={a: median + (v - median) * band.scale for a, v in forecast.quantiles.items()}
    )


def coverage(
    forecasts: Sequence[QuantileForecast], actuals: Sequence[float], *, nominal: float
) -> float:
    if not actuals:
        return 0.0
    lo_a, hi_a = band_edges(forecasts[0], nominal)
    inside = sum(
        1
        for f, a in zip(forecasts, actuals, strict=True)
        if f.quantiles[lo_a] <= a <= f.quantiles[hi_a]
    )
    return inside / len(actuals)


def coverage_by_regime(
    forecasts: Sequence[QuantileForecast],
    actuals: Sequence[float],
    volatility: Sequence[float],
    *,
    nominal: float,
    buckets: int = 3,
) -> list[tuple[str, int, float]]:
    """Coverage within volatility terciles, as ``(label, n, coverage)``.

    The marginal number is the one conformal guarantees and the one that hides
    the failure this project has already seen once: 97.1% in the quiet regime
    and 84.6% in the busy one average to something that reads as calibrated,
    and size a position wrongly in both directions.
    """
    if not actuals:
        return []
    order = sorted(range(len(actuals)), key=lambda i: volatility[i])
    size = len(order) // buckets
    labels = {0: "quiet", buckets - 1: "busy"}
    out: list[tuple[str, int, float]] = []
    for b in range(buckets):
        idx = order[b * size : (b + 1) * size] if b < buckets - 1 else order[b * size :]
        if not idx:
            continue
        label = labels.get(b, f"mid{b}")
        cov = coverage([forecasts[i] for i in idx], [actuals[i] for i in idx], nominal=nominal)
        out.append((label, len(idx), cov))
    return out


def spread(forecasts: Sequence[QuantileForecast], *, nominal: float) -> float:
    """Mean band width. Coverage bought with width is not free, and the two
    numbers only mean something together."""
    if not forecasts:
        return 0.0
    lo_a, hi_a = band_edges(forecasts[0], nominal)
    return fmean(f.quantiles[hi_a] - f.quantiles[lo_a] for f in forecasts)
