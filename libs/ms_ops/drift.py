"""MA3 - drift monitors.

Three measurements and one consequence: **a breach blocks auto-execution.**

A model does not announce that the world changed under it. It keeps returning
probabilities in the same range, with the same confidence, and the only outward
sign is that they stop being right - which is visible in the attribution weeks
later, after the losses. These monitors are the earlier signal, and the reason
they matter is that they fire on the *inputs* before the outcomes arrive.

The reference is fixed, and this is the whole trick
---------------------------------------------------

Population Stability Index compares a new distribution against a **reference**
one, and the bin edges must come from the reference. Recomputing them on the new
data is the mistake that makes the metric useless in the exact way that is
hardest to notice: PSI comes back near zero forever, because the bins have
quietly followed the drift they were supposed to detect.

The reference is snapshotted once, from the training distribution, and every
comparison after that uses those edges - including on data that lies entirely
outside them, which is itself the finding.

What each one catches
---------------------

* **Feature PSI** - the inputs changed. The market is not the one the model was
  fitted on, and nothing about the model's output has to look wrong yet.
* **Prediction KL** - the outputs changed shape. This can drift while the inputs
  look stable, and usually means a feature the model leaned on has gone flat.
* **Rolling Brier** - the outcomes got worse. This is the last to move and the
  only one that proves harm, which is why it cannot be the only monitor.

Blocking, not alerting
----------------------

A drift alert that only notifies is a drift alert that gets acknowledged during
a busy morning. :class:`DriftGate` returns a decision the execution path has to
consult, and the breach clears when the metric recovers - not when someone
dismisses it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# PSI thresholds are the industry-standard reading, and worth naming rather than
# leaving as bare numbers: below 0.1 the distributions are the same population,
# 0.1-0.25 is a shift worth watching, above 0.25 is a different population.
PSI_WATCH = 0.10
PSI_BREACH = 0.25

# KL on the prediction histogram. Less standardised than PSI, so the number is
# calibrated to the same intent: a value this size means the output distribution
# is no longer the one the calibration was fitted against.
KL_BREACH = 0.20

# How much worse the rolling Brier may get before it counts as harm rather than
# noise, as a fraction of the reference.
BRIER_DEGRADATION_BREACH = 0.15

# Below this a histogram is mostly empty bins and every divergence is noise.
MIN_SAMPLES = 200

# Added to every bin before normalising. Without it a single empty bin makes PSI
# and KL infinite, and an infinite metric is indistinguishable from a broken one.
LAPLACE = 1e-6


class Severity(StrEnum):
    OK = "ok"
    WATCH = "watch"
    BREACH = "breach"
    UNKNOWN = "unknown"  # not enough data to say, which is not the same as ok


@dataclass(frozen=True)
class Reference:
    """The distribution a monitor compares against. Snapshotted, never refitted.

    ``edges`` are the bin boundaries from the training distribution. Everything
    afterwards is bucketed with these, including values that fall outside them -
    which land in the end bins and are exactly what a shift looks like.
    """

    name: str
    edges: tuple[float, ...]
    proportions: tuple[float, ...]
    n: int

    @property
    def bins(self) -> int:
        return len(self.proportions)


def build_reference(name: str, values: Sequence[float], *, bins: int = 10) -> Reference:
    """Snapshot a distribution as quantile bins.

    Quantile rather than equal-width, so each reference bin holds roughly the
    same count. Equal-width bins on a skewed feature put ninety percent of the
    reference in one bucket, and then every PSI is dominated by noise in the
    empty ones.
    """
    if len(values) < MIN_SAMPLES:
        raise ValueError(
            f"{len(values)} samples is not enough for a reference distribution "
            f"(need {MIN_SAMPLES}). A reference built on too little data makes "
            "every later comparison a comparison against noise."
        )
    ordered = sorted(values)
    step = len(ordered) / bins
    edges = tuple(ordered[int(i * step)] for i in range(1, bins))
    counts = _bucket(values, edges)
    total = sum(counts)
    return Reference(
        name=name,
        edges=edges,
        proportions=tuple(c / total for c in counts),
        n=len(values),
    )


def _bucket(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Count into bins defined by ``edges``. Values outside land in the ends."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v > edges[i]:
            i += 1
        counts[i] += 1
    return counts


def psi(reference: Reference, values: Sequence[float]) -> float:
    """Population Stability Index against a fixed reference.

    ``sum((a - e) * ln(a / e))`` over bins. Symmetric in the sense that matters:
    a bin that gained mass and one that lost the same mass contribute equally,
    so a shift is a shift whichever way it went.
    """
    counts = _bucket(values, reference.edges)
    total = sum(counts) or 1
    out = 0.0
    for actual_count, expected in zip(counts, reference.proportions, strict=True):
        a = actual_count / total + LAPLACE
        e = expected + LAPLACE
        out += (a - e) * math.log(a / e)
    return out


def kl_divergence(reference: Sequence[float], actual: Sequence[float], *, bins: int = 10) -> float:
    """KL(actual || reference) over a shared [0, 1] grid.

    Used on prediction distributions, which already live on [0, 1], so the grid
    is fixed rather than derived - two models' outputs must be comparable
    without either one defining the axis.
    """
    ref_counts = _histogram_01(reference, bins)
    act_counts = _histogram_01(actual, bins)
    ref_total = sum(ref_counts) or 1
    act_total = sum(act_counts) or 1

    out = 0.0
    for r, a in zip(ref_counts, act_counts, strict=True):
        p = a / act_total + LAPLACE
        q = r / ref_total + LAPLACE
        out += p * math.log(p / q)
    return out


def _histogram_01(values: Sequence[float], bins: int) -> list[int]:
    counts = [0] * bins
    for v in values:
        counts[min(int(max(0.0, min(1.0, v)) * bins), bins - 1)] += 1
    return counts


def rolling_brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes, strict=True)) / len(probs)


# --------------------------------------------------------------------------- #
# The monitors
# --------------------------------------------------------------------------- #
@dataclass
class DriftReading:
    """One measurement, its verdict, and enough context to argue with it."""

    monitor: str
    value: float
    severity: Severity
    n: int
    detail: str = ""

    @property
    def blocks(self) -> bool:
        return self.severity is Severity.BREACH

    def render(self) -> str:
        mark = {
            Severity.OK: "ok",
            Severity.WATCH: "WATCH",
            Severity.BREACH: "BREACH",
            Severity.UNKNOWN: "unknown",
        }[self.severity]
        return f"    {self.monitor:<26} {self.value:>9.4f}  n={self.n:<6} {mark}  {self.detail}"


def check_feature_psi(reference: Reference, values: Sequence[float]) -> DriftReading:
    if len(values) < MIN_SAMPLES:
        return DriftReading(
            f"psi[{reference.name}]",
            float("nan"),
            Severity.UNKNOWN,
            len(values),
            "too few samples to compare - not the same as no drift",
        )
    value = psi(reference, values)
    if value >= PSI_BREACH:
        sev, detail = Severity.BREACH, "a different population from the training data"
    elif value >= PSI_WATCH:
        sev, detail = Severity.WATCH, "a shift worth watching"
    else:
        sev, detail = Severity.OK, "same population"
    return DriftReading(f"psi[{reference.name}]", value, sev, len(values), detail)


def check_prediction_kl(reference: Sequence[float], actual: Sequence[float]) -> DriftReading:
    if len(actual) < MIN_SAMPLES:
        return DriftReading(
            "prediction_kl", float("nan"), Severity.UNKNOWN, len(actual), "too few predictions"
        )
    value = kl_divergence(reference, actual)
    sev = Severity.BREACH if value >= KL_BREACH else Severity.OK
    detail = (
        "the output distribution is no longer the one calibration was fitted against"
        if sev is Severity.BREACH
        else "output shape stable"
    )
    return DriftReading("prediction_kl", value, sev, len(actual), detail)


def check_rolling_brier(
    reference_brier: float, probs: Sequence[float], outcomes: Sequence[int]
) -> DriftReading:
    if len(probs) < MIN_SAMPLES:
        return DriftReading(
            "rolling_brier", float("nan"), Severity.UNKNOWN, len(probs), "too few outcomes"
        )
    current = rolling_brier(probs, outcomes)
    degradation = (current - reference_brier) / reference_brier if reference_brier > 0 else 0.0
    sev = Severity.BREACH if degradation >= BRIER_DEGRADATION_BREACH else Severity.OK
    return DriftReading(
        "rolling_brier",
        current,
        sev,
        len(probs),
        f"{degradation:+.1%} against the reference {reference_brier:.4f}",
    )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@dataclass
class DriftGate:
    """Whether auto-execution may proceed.

    A drift alert that only notifies is one that gets acknowledged during a busy
    morning and forgotten by the afternoon. This returns a decision the
    execution path has to consult, and the breach clears when the metric
    recovers rather than when someone dismisses it.
    """

    readings: list[DriftReading] = field(default_factory=list)

    def evaluate(self, readings: Sequence[DriftReading]) -> DriftGate:
        self.readings = list(readings)
        blocked = [r for r in self.readings if r.blocks]
        if blocked:
            log.info(
                "drift gate blocking auto-execution",
                fields={"breached": [r.monitor for r in blocked]},
            )
        return self

    @property
    def breaches(self) -> list[DriftReading]:
        return [r for r in self.readings if r.blocks]

    @property
    def unknown(self) -> list[DriftReading]:
        return [r for r in self.readings if r.severity is Severity.UNKNOWN]

    @property
    def allows_auto_execution(self) -> bool:
        """Blocked by any breach.

        Unknown does not block. A monitor that cannot measure is not evidence of
        drift, and treating it as such would halt trading every quiet morning -
        which teaches an operator to route around the gate.
        """
        return not self.breaches

    def render(self) -> str:
        lines = ["  drift monitors:"] + [r.render() for r in self.readings]
        if self.breaches:
            lines.append(
                f"  -> AUTO-EXECUTION BLOCKED by {len(self.breaches)} breach(es). "
                "It clears when the metric recovers, not when it is acknowledged."
            )
        else:
            lines.append("  -> auto-execution allowed")
        if self.unknown:
            lines.append(
                f"  {len(self.unknown)} monitor(s) could not measure. Not evidence of "
                "drift, and not evidence of its absence either."
            )
        return "\n".join(lines)
