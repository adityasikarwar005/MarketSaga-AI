"""M57 - the meta-labeller.

The primary model says which way. This one says whether to act at all, and its
job is to say **no**. A meta-labeller that approves most of what it sees is not
filtering anything; it is a rubber stamp with a log loss.

Why the split is worth the extra model
--------------------------------------

A direction model is trained to be right about direction, and it is scored on
every row whether or not that row was worth trading. Most rows are not. The
primary is asked one question - which way - and answering it well says nothing
about whether the move was large enough to clear the spread, whether the option
was liquid enough to get out of, or whether the edge survived the four orders it
took to express it.

Splitting them means the primary can stay focused on direction while this model
learns the separate and much more asymmetric question: *of the calls the primary
made, which ones actually paid?* That is Lopez de Prado's meta-labelling, and
the reason it raises precision is that it is allowed to decline.

The target is post-cost, and that is the whole point
----------------------------------------------------

``build_meta_labels`` takes realised P&L **and** the cost of the structure that
would have expressed the trade, and labels the row 1 only if the trade was
profitable net. Training on gross P&L produces a model that approves trades
which were directionally correct and financially pointless - which is the
failure this module exists to catch, and it is invisible in every metric that
does not include costs.

Asymmetry lives in the threshold, not in the training data
----------------------------------------------------------

Skipping a good trade costs an opportunity. Taking a bad one costs money and the
capacity to take the next good trade, and that is the loss that compounds. The
model has to encode that asymmetry somewhere.

The obvious place is class weighting - duplicate the "skip" rows so the booster
feels them more - and it is the wrong place, measurably. Weighting the fit set
and then calibrating on the unweighted set puts two mechanisms in direct
opposition: calibration's entire job is to make the scores match observed
frequencies, so it undoes the distortion the weighting introduced. Measured
here, it did worse than undo it. Raising ``skip_weight`` from 1 to 4 moved the
approval rate from 11% to 30% - the opposite of the intent, silently.

So the asymmetry lives in the decision threshold, which is where it belongs.
Calibrate the probability honestly, then take the trade only when

    P(pays) > loss_ratio / (1 + loss_ratio)

which is the Bayes-optimal cut when a bad trade costs ``loss_ratio`` times what
a missed good one does. ``loss_ratio = 2`` gives a threshold of 0.67. The
parameter now means something checkable instead of being a knob whose direction
was assumed.

``MetaReport`` reports the **approval rate** as a first-class number, because a
meta-labeller drifting toward approving everything is the failure mode and it
looks like improving recall while it happens.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from libs.ms_core.obs import get_logger
from libs.ms_models.direction import MISSING, IsotonicCalibrator, has_discrimination
from libs.ms_models.trees import GradientBoostedTrees, log_loss

log = get_logger(__name__)

MIN_ROWS_TO_FIT = 200
MIN_CALIBRATION_ROWS = 60

# How much more a bad trade costs than a missed good one. Two is deliberately
# modest; the point is that it is greater than one and stated rather than
# implied.
DEFAULT_LOSS_RATIO = 2.0


def threshold_from_loss_ratio(loss_ratio: float) -> float:
    """The Bayes-optimal cut under asymmetric costs.

    Take the trade when ``P(pays) * gain > (1 - P(pays)) * loss``, which
    rearranges to ``P > loss / (gain + loss)``. With gain normalised to one that
    is ``loss_ratio / (1 + loss_ratio)``.

    Worth deriving rather than picking: a threshold chosen by hand encodes an
    asymmetry nobody wrote down, and it is then impossible to tell whether a
    later change to it reflects new information or someone wanting more trades.
    """
    if loss_ratio <= 0:
        raise ValueError("loss_ratio must be positive")
    return loss_ratio / (1.0 + loss_ratio)


DEFAULT_APPROVAL_THRESHOLD = threshold_from_loss_ratio(DEFAULT_LOSS_RATIO)

# A meta-labeller approving more than this is not filtering. It is not an error
# - it is a finding, reported rather than corrected, because forcing the rate
# down by moving the threshold would hide it rather than fix it.
SUSPICIOUS_APPROVAL_RATE = 0.75


@dataclass(frozen=True)
class MetaSample:
    """One primary call, its context, and whether it paid after costs."""

    features: dict[str, float | None]
    primary_prob: float
    gross_pnl: float
    cost: float
    taken: bool = True

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost

    @property
    def paid(self) -> int:
        return 1 if self.net_pnl > 0 else 0

    @property
    def cost_to_edge(self) -> float:
        """How much of the gross move the costs would have eaten.

        The single most informative feature this model gets, and one the primary
        model never sees: a call can be perfectly correct and still not clear the
        four orders it takes to express.
        """
        return self.cost / abs(self.gross_pnl) if self.gross_pnl != 0 else math.inf


def build_meta_labels(
    primary_probs: Sequence[float],
    gross_pnl: Sequence[float],
    costs: Sequence[float],
    features: Sequence[Mapping[str, float | None]],
    *,
    min_edge: float = 0.0,
) -> list[MetaSample]:
    """Label each primary call by whether it paid **after costs**.

    ``min_edge`` raises the bar above breakeven. Zero is the honest default -
    "did this make money" - but a caller that only wants trades worth the
    capital and attention can require more, and then the model is learning that
    question instead. Either way the threshold is explicit rather than implied.
    """
    out = []
    for p, pnl, cost, feats in zip(primary_probs, gross_pnl, costs, features, strict=True):
        enriched = dict(feats)
        enriched.setdefault("primary_prob", p)
        enriched.setdefault("primary_edge", abs(p - 0.5))
        sample = MetaSample(features=enriched, primary_prob=p, gross_pnl=pnl, cost=cost + min_edge)
        enriched.setdefault(
            "cost_to_edge", min(sample.cost_to_edge, 10.0) if sample.gross_pnl != 0 else 10.0
        )
        out.append(sample)
    return out


@dataclass
class MetaReport:
    rows: int
    features: int
    trees: int
    base_rate: float  # how often the primary's calls actually paid
    approval_rate: float
    threshold: float
    calibrated: bool
    train_log_loss: float
    notes: list[str] = field(default_factory=list)

    @property
    def is_filtering(self) -> bool:
        return self.approval_rate < SUSPICIOUS_APPROVAL_RATE

    def render(self) -> str:
        verdict = (
            f"approves {self.approval_rate:.0%} of the primary's calls"
            if self.is_filtering
            else f"approves {self.approval_rate:.0%} - NOT FILTERING, this is a rubber stamp"
        )
        lines = [
            f"  meta-labeller: {self.rows:,} rows, {self.features} features, "
            f"{self.trees} trees",
            f"    the primary's calls paid {self.base_rate:.0%} of the time after costs",
            f"    at threshold {self.threshold:.2f} it {verdict}",
        ]
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class PrecisionRecall:
    """What the gate actually asks: more precision at the same recall."""

    threshold: float
    precision: float
    recall: float
    approved: int
    total: int
    net_pnl: float
    expectancy: float

    def render(self) -> str:
        return (
            f"    t={self.threshold:.2f}  approved {self.approved:>4}/{self.total}  "
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  "
            f"net {self.net_pnl:>10,.0f}  per-trade {self.expectancy:>8,.1f}"
        )


def evaluate_at(
    samples: Sequence[MetaSample], scores: Sequence[float], threshold: float
) -> PrecisionRecall:
    """Precision, recall and realised post-cost expectancy at one threshold.

    Expectancy is per **approved** trade, not per row. A filter that approves
    three trades a year with a huge average is a different animal from one that
    approves three hundred, and dividing by the row count would hide that.
    """
    approved = [(s, sc) for s, sc in zip(samples, scores, strict=True) if sc >= threshold]
    positives = sum(s.paid for s in samples)
    hits = sum(s.paid for s, _ in approved)
    net = sum(s.net_pnl for s, _ in approved)
    return PrecisionRecall(
        threshold=threshold,
        precision=hits / len(approved) if approved else 0.0,
        recall=hits / positives if positives else 0.0,
        approved=len(approved),
        total=len(samples),
        net_pnl=net,
        expectancy=net / len(approved) if approved else 0.0,
    )


def threshold_for_recall(
    samples: Sequence[MetaSample], scores: Sequence[float], target_recall: float
) -> float:
    """The highest threshold that still reaches ``target_recall``.

    The gate compares precision *at fixed recall*, because precision alone is
    trivially maximised by approving one trade. Fixing recall first is what
    makes the comparison mean something.
    """
    candidates = sorted({round(s, 4) for s in scores}, reverse=True)
    best = 0.0
    for t in candidates:
        if evaluate_at(samples, scores, t).recall >= target_recall:
            best = t
            break
    return best


class MetaLabeller:
    """Boosted trees over the primary's call and its context, tilted toward no.

    Falls back to a threshold on the primary probability when it cannot fit or
    cannot discriminate - which is a real answer, not an error path: "take the
    confident ones" is a legitimate filter, and it is the thing this model has
    to beat.
    """

    name = "M57_meta"
    version = "1"

    def __init__(
        self,
        *,
        loss_ratio: float = DEFAULT_LOSS_RATIO,
        threshold: float | None = None,
        calibration_share: float = 0.25,
        fallback_edge: float = 0.05,
        **booster: object,
    ) -> None:
        # A threshold can be given directly, but the derived one is the default
        # so that the asymmetry is always traceable to a stated loss ratio.
        self.loss_ratio = loss_ratio
        self.threshold = (
            threshold if threshold is not None else threshold_from_loss_ratio(loss_ratio)
        )
        self.calibration_share = calibration_share
        self.fallback_edge = fallback_edge
        self.booster_kwargs = booster

        self.names: list[str] = []
        self.booster: GradientBoostedTrees | None = None
        self.calibrator: IsotonicCalibrator | None = None
        self.report: MetaReport | None = None
        self.using_fallback = False

    def _vector(self, features: Mapping[str, float | None]) -> list[float]:
        return [MISSING if (v := features.get(n)) is None else float(v) for n in self.names]

    def fit(self, samples: Sequence[MetaSample]) -> MetaLabeller:
        if len(samples) < MIN_ROWS_TO_FIT:
            raise ValueError(
                f"{len(samples)} rows is not enough to fit a meta-labeller "
                f"(need {MIN_ROWS_TO_FIT})"
            )
        labels = [s.paid for s in samples]
        base_rate = sum(labels) / len(labels)
        notes: list[str] = []

        if len(set(labels)) < 2:
            # Every primary call paid, or none did. There is nothing to learn
            # and pretending otherwise would produce a filter that approves or
            # refuses everything for no reason.
            self.using_fallback = True
            self.report = MetaReport(
                rows=len(samples),
                features=0,
                trees=0,
                base_rate=base_rate,
                approval_rate=self._fallback_rate(samples),
                threshold=self.threshold,
                calibrated=False,
                train_log_loss=float("nan"),
                notes=["every outcome was the same - nothing to learn, using the fallback"],
            )
            return self

        self.names = sorted({k for s in samples for k in s.features})
        vectors = [self._vector(s.features) for s in samples]

        n_calib = int(len(samples) * self.calibration_share)
        calibrated = n_calib >= MIN_CALIBRATION_ROWS
        cut = len(samples) - n_calib if calibrated else len(samples)
        fit_x, fit_y = vectors[:cut], labels[:cut]

        # No class weighting. The booster is fitted on the rows as they happened
        # and calibrated to say what it honestly believes; the asymmetry is
        # applied afterwards, at the threshold. Duplicating skip rows here and
        # then calibrating on the unweighted set puts the two in opposition -
        # measured, raising the weight from 1 to 4 moved the approval rate from
        # 11% to 30%, the opposite of the intent.
        try:
            self.booster = GradientBoostedTrees(**self.booster_kwargs).fit(fit_x, fit_y)  # type: ignore[arg-type]
        except ValueError as exc:
            self.using_fallback = True
            notes.append(f"booster refused to fit ({exc}) - using the fallback")

        if self.booster is not None and calibrated:
            cal_x, cal_y = vectors[cut:], labels[cut:]
            raw = self.booster.predict_all(cal_x)
            if has_discrimination(raw, cal_y):
                self.calibrator = IsotonicCalibrator().fit(raw, cal_y)
            else:
                notes.append(
                    "no discrimination on held-out rows - scores are uncalibrated "
                    "and the fallback may be the better filter"
                )
                calibrated = False

        scores = [self.score(s) for s in samples]
        approved = sum(1 for sc in scores if sc >= self.threshold)
        approval_rate = approved / len(samples)
        if approval_rate >= SUSPICIOUS_APPROVAL_RATE:
            notes.append(
                f"approves {approval_rate:.0%} of calls - a meta-labeller that "
                "approves nearly everything is not filtering anything"
            )

        self.report = MetaReport(
            rows=len(samples),
            features=len(self.names),
            trees=len(self.booster.trees) if self.booster else 0,
            base_rate=base_rate,
            approval_rate=approval_rate,
            threshold=self.threshold,
            calibrated=calibrated and self.calibrator is not None,
            train_log_loss=log_loss(scores, labels),
            notes=notes,
        )
        log.info(
            "meta-labeller fitted",
            fields={
                "rows": len(samples),
                "base_rate": round(base_rate, 4),
                "approval_rate": round(approval_rate, 4),
                "fallback": self.using_fallback,
            },
        )
        return self

    def _fallback_rate(self, samples: Sequence[MetaSample]) -> float:
        return sum(1 for s in samples if self._fallback_score(s) >= self.threshold) / len(samples)

    def _fallback_score(self, sample: MetaSample) -> float:
        """Take the confident ones. The thing the model has to beat.

        Mapped onto the same 0-1 scale as a fitted score so the threshold means
        the same thing either way - otherwise switching to the fallback silently
        changes what the threshold does.
        """
        edge = abs(sample.primary_prob - 0.5)
        return min(1.0, 0.5 + 0.5 * (edge / self.fallback_edge)) if self.fallback_edge > 0 else 0.5

    def score(self, sample: MetaSample) -> float:
        """Probability that taking this call pays after costs."""
        if self.booster is None:
            return self._fallback_score(sample)
        raw = self.booster.predict_proba(self._vector(sample.features))
        return self.calibrator.apply(raw) if self.calibrator else raw

    def approve(self, sample: MetaSample) -> bool:
        return self.score(sample) >= self.threshold

    def explain(self, sample: MetaSample) -> str:
        score = self.score(sample)
        verdict = "TAKE" if score >= self.threshold else "SKIP"
        how = "fallback (primary confidence only)" if self.booster is None else "fitted"
        return f"{verdict} at {score:.3f} vs threshold {self.threshold:.2f} [{how}]"


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@dataclass
class MetaEvaluation:
    """Does the meta-labeller earn its place against taking everything?"""

    take_everything: PrecisionRecall
    fallback: PrecisionRecall
    model: PrecisionRecall
    target_recall: float

    @property
    def precision_gain(self) -> float:
        return self.model.precision - self.fallback.precision

    @property
    def expectancy_gain(self) -> float:
        return self.model.expectancy - self.take_everything.expectancy

    @property
    def passes(self) -> bool:
        """More precision than the fallback at the same recall, and better
        post-cost expectancy than taking everything.

        Both, because either alone is gameable. Precision without expectancy can
        mean approving only tiny sure things; expectancy without precision can
        mean one lucky trade.
        """
        return self.precision_gain > 0 and self.expectancy_gain > 0

    @property
    def net_positive(self) -> bool:
        """Whether the approved trades actually made money.

        Deliberately separate from :attr:`passes`, because they answer different
        questions and conflating them is how a filter gets shipped on top of a
        strategy that has no edge. On the first synthetic run this model cut the
        loss per trade from -254 to -109 and raised precision from 0.23 to 0.62
        - it earned its place completely, and the filtered book still lost money,
        because the underlying calls did not have an edge to filter.

        A meta-labeller cannot manufacture one. It can only stop you paying
        costs on calls that were never going to cover them.
        """
        return self.model.net_pnl > 0

    def render(self) -> str:
        return "\n".join(
            [
                f"  at fixed recall {self.target_recall:.0%}:",
                f"    take everything {self.take_everything.render().strip()}",
                f"    fallback        {self.fallback.render().strip()}",
                f"    model           {self.model.render().strip()}",
                f"  precision {self.precision_gain:+.3f} vs fallback, "
                f"expectancy {self.expectancy_gain:+,.1f} vs taking everything  ->  "
                f"{'earns its place' if self.passes else 'does not earn its place'}",
                (
                    f"  approved book nets {self.model.net_pnl:+,.0f}"
                    + (
                        ""
                        if self.net_positive
                        else " - the filter works and the underlying calls still "
                        "lose; a meta-labeller cannot manufacture an edge"
                    )
                ),
            ]
        )


def evaluate_meta(
    model: MetaLabeller,
    samples: Sequence[MetaSample],
    *,
    target_recall: float = 0.5,
) -> MetaEvaluation:
    """Compare the model, the fallback, and taking every call.

    Against both, always. A meta-labeller that beats "take everything" but not
    "take the confident ones" has demonstrated that primary confidence is
    informative, which was already known.
    """
    scores = [model.score(s) for s in samples]
    fallback_scores = [model._fallback_score(s) for s in samples]

    return MetaEvaluation(
        take_everything=evaluate_at(samples, [1.0] * len(samples), 0.0),
        fallback=evaluate_at(
            samples, fallback_scores, threshold_for_recall(samples, fallback_scores, target_recall)
        ),
        model=evaluate_at(samples, scores, threshold_for_recall(samples, scores, target_recall)),
        target_recall=target_recall,
    )
