"""M53 - the direction model.

The first learned model that has to beat the rules baseline on its own terms.
It is a boosted-tree classifier behind the same ``Model`` interface the rules
baseline implements, so the risk gate, the simulator and the recommendation
assembler cannot tell them apart and never have to be rewritten when one
replaces the other.

Three things in here are not the model, and matter more than the model.

**The feature order is stored, not assumed.** A model that takes a dict and
iterates it is one dictionary-ordering change away from silently scoring the
wrong columns - the training run stays green and every live prediction is
garbage. The names are fixed at fit time and every later vectorisation is
checked against them.

**Missing is a value, not a zero.** Filling an absent feature with 0.0 puts it
at the middle of a standardised distribution, which is a confident claim that
the feature was average. Here missingness is passed to the trees as a sentinel
far below the observed range, so a split can isolate it, plus an explicit
indicator column when the feature is missing often enough to matter. What must
never happen is the silent zero.

**A probability is a claim about frequency.** Boosted trees are systematically
overconfident near 0 and 1, and an uncalibrated 0.85 that is right 70% of the
time will size a position as though it were right 85% of the time. Isotonic
calibration is fitted on a held-out slice, and the model records whether it was
calibrated at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from libs.ms_core.obs import get_logger
from libs.ms_models.baseline import Model, Opinion
from libs.ms_models.trees import GradientBoostedTrees, log_loss, select_depth

log = get_logger(__name__)

# Missing features are handed to the trees as this, far below anything a real
# standardised feature reaches, so a split can separate "absent" from "small".
MISSING = -1e6

# Below this share of rows, a dedicated is-missing column is more noise than
# signal; above it, the pattern of absence is usually informative on its own.
MISSING_INDICATOR_THRESHOLD = 0.02

MIN_ROWS_TO_FIT = 200
MIN_CALIBRATION_ROWS = 60


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class IsotonicCalibrator:
    """Monotone probability mapping fitted by pool-adjacent-violators.

    Isotonic rather than Platt scaling because the miscalibration of a boosted
    tree is not a shape a sigmoid can express - it is typically too confident at
    both ends and roughly right in the middle. Isotonic can bend that back;
    Platt can only stretch it.

    Isotonic left unregularised is dangerous in exactly the way this project
    keeps finding. Fitted on 400 rows of pure noise - raw probabilities spanning
    [0.442, 0.571] against coin-flip labels - plain PAV produced nine blocks, six
    of them valued at exactly 0.0 or 1.0. It turned a model with no opinion into
    one asserting certainty, from blocks of a few lucky rows, and it *improved*
    log loss while doing it.

    Two guards, both necessary:

    ``min_block`` merges neighbouring blocks until each rests on enough rows to
    mean something, and a Beta prior of strength ``prior_strength`` keeps any
    single block from claiming 0 or 1.

    Those two are not sufficient on their own, and the reason is worth stating.
    Fitted on a booster that had learned nothing - raw output spanning
    [0.475, 0.496] - the calibrator produced values spanning [0.046, 0.659].
    Shrinking the blocks harder narrowed that but cost real accuracy when there
    *was* a signal, which is the wrong trade.

    The real fault is upstream: isotonic assumes the ranking it is given is
    meaningful and only the *levels* are wrong. When the base model has no
    discrimination, its ordering is noise, and a monotone map fitted to noise
    manufactures an opinion out of nothing. So the caller checks discrimination
    first - see ``has_discrimination`` - and skips calibration entirely when
    there is none, which is also the honest answer: a model with no ranking
    ability should report the base rate, not a calibrated version of an
    accident.
    """

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    fitted: bool = False
    min_block: int = 20
    prior_strength: float = 5.0
    weights: list[float] = field(default_factory=list)

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        if len(probs) != len(labels):
            raise ValueError("probs and labels must be the same length")
        pairs = sorted(zip(probs, labels, strict=True))
        # Each block is [mean, weight, right edge]; adjacent violators are merged
        # until the sequence is non-decreasing.
        blocks: list[list[float]] = []
        for p, y in pairs:
            blocks.append([float(y), 1.0, p])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                b = blocks.pop()
                a = blocks.pop()
                total = a[1] + b[1]
                blocks.append([(a[0] * a[1] + b[0] * b[1]) / total, total, b[2]])

        # At most ten blocks on any sample: more than that and the later ones
        # are describing individual rows rather than a shape.
        floor = max(self.min_block, len(pairs) // 10)
        blocks = _merge_thin_blocks(blocks, floor)

        base = sum(labels) / len(labels)
        k = self.prior_strength
        self.xs = [b[2] for b in blocks]
        self.ys = [(mean * weight + base * k) / (weight + k) for mean, weight, _ in blocks]
        # How many calibration rows stand behind each mapped value. Kept because
        # a probability backed by 200 observations and one backed by 20 are not
        # the same claim, and everything downstream that sizes a position needs
        # to be able to tell them apart.
        self.weights = [w for _, w, _ in blocks]
        self.fitted = True
        return self

    def _block_of(self, p: float) -> int:
        if p <= self.xs[0]:
            return 0
        if p >= self.xs[-1]:
            return len(self.xs) - 1
        lo, hi = 0, len(self.xs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.xs[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def interval(self, p: float, *, z: float = 1.96) -> tuple[float, float]:
        """A Wilson score interval on the calibration block this lands in.

        Wilson rather than the normal approximation because the normal one is
        nonsense near 0 and 1 - it happily returns bounds outside [0, 1] and is
        badly wrong at exactly the confident end where a sizing rule is most
        dangerous.

        The width is the honest statement of how much evidence backs this
        probability. A block resting on 200 calibration rows earns a tight
        interval; one resting on 20 does not, and a position sized on the point
        estimate would treat them identically.
        """
        if not self.fitted or not self.xs:
            return (p, p)
        i = self._block_of(p)
        n = max(1.0, self.weights[i])
        phat = self.ys[i]
        denom = 1.0 + z * z / n
        centre = (phat + z * z / (2 * n)) / denom
        half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def apply(self, p: float) -> float:
        if not self.fitted or not self.xs:
            return p
        if p <= self.xs[0]:
            return self.ys[0]
        if p >= self.xs[-1]:
            return self.ys[-1]
        lo, hi = 0, len(self.xs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.xs[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        return self.ys[lo]


def has_discrimination(probs: Sequence[float], labels: Sequence[int]) -> bool:
    """Does this ranking separate the classes better than chance would?

    The gate in front of calibration. Isotonic corrects the *levels* of a
    ranking and takes the ranking itself on trust, so fitting it to a model with
    no ranking ability turns noise into confident-looking probabilities. Under
    the null the Mann-Whitney statistic has standard error
    ``sqrt((n1 + n2 + 1) / (12 n1 n2))``, so the bar is two of those - the
    ordinary 5% level.

    One standard error was tried first and is not enough: it passes on noise
    about a third of the time, which left the null case still producing opinions
    0.30 away from a coin flip. The threshold has to be a significance level,
    not a gesture at one.
    """
    area = auc(probs, labels)
    if area is None:
        return False
    n1 = sum(1 for y in labels if y == 1)
    n2 = len(labels) - n1
    if n1 == 0 or n2 == 0:
        return False
    se = math.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    return abs(area - 0.5) > 2.0 * se


def _merge_thin_blocks(blocks: list[list[float]], min_block: int) -> list[list[float]]:
    """Fold blocks resting on too few rows into their neighbour.

    Monotonicity is preserved because merging adjacent blocks can only move the
    combined mean between the two it replaces.
    """
    if min_block <= 1:
        return blocks
    merged: list[list[float]] = []
    for block in blocks:
        merged.append(list(block))
        while len(merged) > 1 and merged[-2][1] < min_block:
            b = merged.pop()
            a = merged.pop()
            total = a[1] + b[1]
            merged.append([(a[0] * a[1] + b[0] * b[1]) / total, total, b[2]])
    # The last block can still be short; fold it back if there is anything to
    # fold it into.
    while len(merged) > 1 and merged[-1][1] < min_block:
        b = merged.pop()
        a = merged.pop()
        total = a[1] + b[1]
        merged.append([(a[0] * a[1] + b[0] * b[1]) / total, total, b[2]])
    return merged


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    """Average gap between claimed probability and realised frequency.

    The number that says whether a 0.8 means anything. Log loss can be improved
    by a model that is sharper *and* worse calibrated; this cannot.
    """
    if not probs:
        return float("nan")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        claimed = sum(p for p, _ in bucket) / len(bucket)
        realised = sum(y for _, y in bucket) / len(bucket)
        total += len(bucket) * abs(claimed - realised)
    return total / len(probs)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class DirectionReport:
    rows: int
    features: int
    trees: int
    depth: int
    calibrated: bool
    train_log_loss: float
    calibration_log_loss: float
    ece_before: float
    ece_after: float

    def render(self) -> str:
        cal = (
            f"calibrated on {self.calibration_log_loss:.5f}, "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f}"
            if self.calibrated
            else "UNCALIBRATED - the probabilities are not frequencies"
        )
        return (
            f"  direction model: {self.rows:,} rows, {self.features} features, "
            f"{self.trees} trees (depth {self.depth})\n"
            f"    train log loss {self.train_log_loss:.5f}, {cal}"
        )


class DirectionModel(Model):
    """Boosted trees over the feature catalogue, calibrated, behind ``Model``."""

    name = "M1_direction"
    version = "1"

    def __init__(
        self,
        *,
        calibration_share: float = 0.25,
        depth: int | None = None,
        max_edge: float = 0.35,
        **booster: Any,
    ) -> None:
        self.calibration_share = calibration_share
        self.depth = depth
        # Even a good model on this problem is nowhere near 0.9. The cap is a
        # statement about the problem, not about the model: a direction call on
        # an index over a short horizon that claims 90% is describing a bug.
        self.max_edge = max_edge
        self.booster_kwargs = booster

        self.names: list[str] = []
        self.indicator_for: list[str] = []
        self.booster: GradientBoostedTrees | None = None
        self.calibrator: IsotonicCalibrator | None = None
        self.report: DirectionReport | None = None
        # True when the booster could not rank the calibration rows at all.
        self.no_discrimination = False

    # ------------------------------------------------------------ vectorising
    def _vector(self, features: Mapping[str, float | None]) -> list[float]:
        row = [MISSING if (v := features.get(n)) is None else float(v) for n in self.names]
        row += [1.0 if features.get(n) is None else 0.0 for n in self.indicator_for]
        return row

    def _learn_schema(self, rows: Sequence[Mapping[str, float | None]]) -> None:
        """Fix the column order and decide which absences get their own column."""
        seen: dict[str, int] = {}
        for r in rows:
            for k, v in r.items():
                seen.setdefault(k, 0)
                if v is None:
                    seen[k] += 1
        # Sorted, so the order depends on the feature names and nothing else -
        # not on dictionary insertion, not on which row happened to be first.
        self.names = sorted(seen)
        self.indicator_for = [
            n for n in self.names if seen[n] / len(rows) >= MISSING_INDICATOR_THRESHOLD
        ]

    @property
    def columns(self) -> list[str]:
        return self.names + [f"{n}__missing" for n in self.indicator_for]

    # ---------------------------------------------------------------- fitting
    def fit(
        self,
        rows: Sequence[Mapping[str, float | None]],
        labels: Sequence[int],
    ) -> DirectionModel:
        """``labels`` are +1 / -1 / 0, as the triple-barrier builder emits them.

        Timeout rows (0) are dropped from fitting: a barrier that was never
        touched is not a direction, and coding it as either class teaches the
        model something that did not happen. They stay in *scoring*, where the
        model still pays for having been confident about nothing.
        """
        if len(rows) != len(labels):
            raise ValueError("rows and labels must be the same length")

        kept = [(r, 1 if y > 0 else 0) for r, y in zip(rows, labels, strict=True) if y != 0]
        if len(kept) < MIN_ROWS_TO_FIT:
            raise ValueError(
                f"{len(kept)} directional rows is not enough to fit "
                f"(need {MIN_ROWS_TO_FIT}). Timeouts do not count."
            )

        self._learn_schema([r for r, _ in kept])
        vectors = [self._vector(r) for r, _ in kept]
        ys = [y for _, y in kept]

        n_calib = int(len(vectors) * self.calibration_share)
        calibrated = n_calib >= MIN_CALIBRATION_ROWS
        if calibrated:
            # Chronological: callers pass rows in time order, and a randomly
            # drawn calibration slice is interleaved with training data.
            cut = len(vectors) - n_calib
            fit_x, cal_x = vectors[:cut], vectors[cut:]
            fit_y, cal_y = ys[:cut], ys[cut:]
        else:
            fit_x, cal_x, fit_y, cal_y = vectors, [], ys, []

        depth = self.depth if self.depth is not None else select_depth(fit_x, fit_y)
        self.booster = GradientBoostedTrees(max_depth=depth, **self.booster_kwargs).fit(
            fit_x, fit_y
        )

        ece_before = ece_after = float("nan")
        cal_loss = float("nan")
        self.no_discrimination = False
        if calibrated:
            raw = self.booster.predict_all(cal_x)
            if has_discrimination(raw, cal_y):
                ece_before = expected_calibration_error(raw, cal_y)
                self.calibrator = IsotonicCalibrator().fit(raw, cal_y)
                adjusted = [self.calibrator.apply(p) for p in raw]
                ece_after = expected_calibration_error(adjusted, cal_y)
                cal_loss = log_loss(adjusted, cal_y)
            else:
                # The booster cannot rank the held-out rows at all. There is no
                # shape to correct, and calibrating anyway would invent one.
                calibrated = False
                self.no_discrimination = True
                log.info(
                    "calibration skipped: the model has no discrimination",
                    fields={"calibration_rows": len(cal_y)},
                )

        self.report = DirectionReport(
            rows=len(kept),
            features=len(self.columns),
            trees=len(self.booster.trees),
            depth=depth,
            calibrated=calibrated,
            train_log_loss=log_loss(self.booster.predict_all(fit_x), fit_y),
            calibration_log_loss=cal_loss,
            ece_before=ece_before,
            ece_after=ece_after,
        )
        log.info(
            "direction model fitted",
            fields={
                "rows": len(kept),
                "features": len(self.columns),
                "trees": len(self.booster.trees),
                "depth": depth,
                "calibrated": calibrated,
            },
        )
        return self

    # ------------------------------------------------------------ predicting
    def predict(
        self, features: Mapping[str, float | None], *, event: dict[str, Any] | None = None
    ) -> Opinion:
        if self.booster is None:
            raise RuntimeError("fit() before predict()")

        raw = self.booster.predict_proba(self._vector(features))
        p = self.calibrator.apply(raw) if self.calibrator else raw
        # The interval comes from the calibration block this landed in: how much
        # evidence stands behind a probability at this level.
        lower = upper = None
        if self.calibrator is not None:
            lower, upper = self.calibrator.interval(raw)

        # Clamp toward the coin flip rather than clipping at the ends: an edge
        # this model cannot support should read as less certainty, not as a
        # different probability.
        #
        # The same clamp applies to the interval, and that is not tidiness. The
        # first version capped the point estimate and left the bounds where they
        # were, which produced p_up=0.8500 with an interval of [0.8424, 0.9570]
        # - the reported probability sitting on the very floor of its own
        # interval, and a lower bound asserting near-certainty about a level the
        # cap existed precisely to disbelieve. A caller sizing on that p_lower
        # would have been handed the uncapped opinion through the back door.
        def clamp(x: float) -> float:
            return 0.5 + max(-self.max_edge, min(self.max_edge, x - 0.5))

        p = clamp(p)
        if lower is not None and upper is not None:
            lower, upper = clamp(lower), clamp(upper)

        missing = [n for n in self.names if features.get(n) is None]
        depth = self.report.depth if self.report else self.booster.max_depth
        reasons = [f"{len(self.booster.trees)} trees, depth {depth}"]
        if missing:
            reasons.append(f"{len(missing)} of {len(self.names)} features missing")
        if self.no_discrimination:
            reasons.append("no discrimination on held-out rows - reporting the base rate")
        elif self.calibrator is None:
            reasons.append("UNCALIBRATED")

        return Opinion(
            p_up=p,
            p_lower=lower,
            p_upper=upper,
            # Missing inputs reduce confidence proportionally. A model asked to
            # opine on half a feature vector should say so rather than answer at
            # full strength.
            confidence=(1.0 - len(missing) / len(self.names)) if self.names else 0.0,
            reasons=reasons,
            model=self.name,
            version=self.version,
        )


def reliability_curve(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> list[tuple[float, float, int]]:
    """(claimed, realised, count) per bin - the input to the G5 slope check.

    A slope well below 1 means overconfidence: the model says 0.8 and gets 0.65.
    That is the failure that survives every accuracy metric and destroys any
    sizing rule built on top of it.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    out = []
    for bucket in buckets:
        if not bucket:
            continue
        out.append(
            (
                sum(p for p, _ in bucket) / len(bucket),
                sum(y for _, y in bucket) / len(bucket),
                len(bucket),
            )
        )
    return out


def reliability_slope(curve: Sequence[tuple[float, float, int]]) -> float | None:
    """Weighted least-squares slope of realised against claimed. 1.0 is perfect."""
    points = [(c, r, n) for c, r, n in curve if n > 0]
    if len(points) < 3:
        return None
    total = sum(n for _, _, n in points)
    mx = sum(c * n for c, _, n in points) / total
    my = sum(r * n for _, r, n in points) / total
    sxx = sum(n * (c - mx) ** 2 for c, _, n in points)
    if sxx <= 0:
        return None
    sxy = sum(n * (c - mx) * (r - my) for c, r, n in points)
    return sxy / sxx


def brier_skill_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Improvement in Brier score over always predicting the base rate."""
    if not probs:
        return float("nan")
    base = sum(labels) / len(labels)
    ref = sum((base - y) ** 2 for y in labels) / len(labels)
    got = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)
    return (ref - got) / ref if ref > 0 else 0.0


def auc(probs: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based discrimination, ties counted as half. None if one class only."""
    pos = [p for p, y in zip(probs, labels, strict=True) if y == 1]
    neg = [p for p, y in zip(probs, labels, strict=True) if y == 0]
    if not pos or not neg:
        return None
    ranked = sorted(zip(probs, labels, strict=True))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and math.isclose(ranked[j][0], ranked[i][0]):
            j += 1
        average_rank = (i + j + 1) / 2.0
        rank_sum += sum(average_rank for k in range(i, j) if ranked[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
