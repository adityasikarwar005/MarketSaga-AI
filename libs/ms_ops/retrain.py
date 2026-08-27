"""MA2 - the retraining pipeline.

Weekly incremental, quarterly full, and one rule that does the work:

    promote only if it beats the incumbent on the last four weeks out-of-sample

Every part of that sentence is load-bearing.

"Beats the incumbent", not "is good"
------------------------------------

The incumbent is already live and already costs nothing to keep. A challenger
that is merely acceptable is a change with all of the risk and none of the
benefit, and the natural comparison - "is the new model good?" - has no way to
answer it. The comparison is always against what is running.

"Out-of-sample", and the challenger defines the boundary
---------------------------------------------------------

The four weeks must be data the **challenger** was not fitted on. The incumbent
was fitted even earlier, so the same window is out-of-sample for both - but
asymmetrically, and in the incumbent's disfavour. That asymmetry is correct and
worth naming: the incumbent is being asked to hold up on data further from its
training set, which is exactly the question that matters, because that is what
it will keep doing if nothing changes.

Getting this backwards - scoring both on the incumbent's boundary - lets a
challenger win on weeks it was trained on, and the win is large, clean and
entirely fake.

"Four weeks", not two
---------------------

The challenger was fitted up to the start of the window. A shorter window is
mostly the tail of its training set, where any model looks excellent.

Every retrain is a trial
-------------------------

Fifty-two retrains a year is fifty-two attempts at beating the incumbent, and a
challenger that wins once in that sequence has not necessarily found anything.
:func:`retrain` writes to the experiment log for the same reason the gap
forecaster does, and the deflated Sharpe downstream needs that count to mean
anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

from libs.ms_core.clock import Clock, RealClock
from libs.ms_core.obs import get_logger
from libs.ms_models.cv import Experiment, ExperimentLog
from libs.ms_ops.registry import (
    MIN_SHADOW_SESSIONS,
    OOS_WEEKS,
    ModelVersion,
    PromotionCheck,
    Registry,
    Stage,
)

log = get_logger(__name__)

# How much better the challenger must be. Not zero: a challenger ahead by a
# thousandth of a log-loss has won a coin toss, and promoting on it means
# promoting every week for no reason.
MIN_IMPROVEMENT = 0.005

# Below this the comparison is noise whatever it says.
MIN_OOS_PREDICTIONS = 200


class RetrainKind(StrEnum):
    INCREMENTAL = "incremental"  # weekly
    FULL = "full"  # quarterly


@dataclass(frozen=True)
class Scored:
    """One model's performance on a window, and which window it was."""

    version: str
    window_start: date
    window_end: date
    n: int
    log_loss: float
    brier: float

    def render(self) -> str:
        return (
            f"    {self.version:<20} n={self.n:<5} log_loss={self.log_loss:.5f}  "
            f"brier={self.brier:.5f}   [{self.window_start} .. {self.window_end}]"
        )


def oos_window(challenger: ModelVersion, *, weeks: int = OOS_WEEKS) -> tuple[date, date]:
    """The evaluation window, defined by the challenger's training boundary.

    Starts the day after the challenger's last training day. Defining it from
    "today" instead would let a challenger trained through last Friday be scored
    on last Thursday, and the win would be large, clean and entirely fake.
    """
    start = challenger.trained_through + timedelta(days=1)
    return start, start + timedelta(weeks=weeks)


def evaluate_promotion(
    challenger: ModelVersion,
    incumbent: ModelVersion | None,
    *,
    challenger_score: Scored,
    incumbent_score: Scored | None,
    today: date,
    min_improvement: float = MIN_IMPROVEMENT,
) -> PromotionCheck:
    """Every reason a challenger may or may not go live. All of them, not the first.

    Returning the full list rather than short-circuiting matters: a challenger
    that fails on three counts and is fixed for one comes back and fails on the
    other two, and each round trip is a week.
    """
    check = PromotionCheck(
        challenger=challenger.version, incumbent=incumbent.version if incumbent else None
    )

    # --- the shadow requirement --------------------------------------------
    if challenger.stage is not Stage.SHADOW:
        check.blockers.append(
            f"stage is {challenger.stage.name}, not SHADOW - a model goes live only "
            "from shadow, where its predictions were recorded and traded nothing"
        )
    elif not challenger.shadow_complete:
        check.blockers.append(
            f"{challenger.shadow_sessions} shadow sessions, needs {MIN_SHADOW_SESSIONS} - "
            "shorter than that has not met a bad Monday, an expiry and a quiet Friday"
        )
    else:
        check.evidence.append(f"{challenger.shadow_sessions} shadow sessions completed")

    # --- the window --------------------------------------------------------
    start, end = oos_window(challenger)
    if challenger_score.window_start < start:
        check.blockers.append(
            f"scored from {challenger_score.window_start}, but the challenger was "
            f"trained through {challenger.trained_through} - the window overlaps its "
            "own training data and the result is not out-of-sample"
        )
    else:
        check.evidence.append(f"scored on {start} .. {end}, after its training boundary")

    if challenger_score.n < MIN_OOS_PREDICTIONS:
        check.blockers.append(
            f"{challenger_score.n} out-of-sample predictions, needs "
            f"{MIN_OOS_PREDICTIONS} - below that the comparison is noise whatever "
            "it says"
        )

    # --- against the incumbent ---------------------------------------------
    if incumbent is None:
        check.evidence.append("nothing in production - the challenger only has to be valid")
        return check

    if incumbent_score is None:
        check.blockers.append(
            "the incumbent was not scored on the same window. A challenger "
            "compared against nothing wins every time."
        )
        return check

    if incumbent_score.window_start != challenger_score.window_start:
        check.blockers.append(
            f"the two were scored on different windows "
            f"({incumbent_score.window_start} vs {challenger_score.window_start}) - "
            "the comparison is between a model and a different question"
        )
        return check

    improvement = incumbent_score.log_loss - challenger_score.log_loss
    if improvement < min_improvement:
        check.blockers.append(
            f"log loss {challenger_score.log_loss:.5f} against the incumbent's "
            f"{incumbent_score.log_loss:.5f} - an improvement of {improvement:+.5f}, "
            f"below the {min_improvement} needed. A challenger ahead by a thousandth "
            "has won a coin toss."
        )
    else:
        check.evidence.append(
            f"log loss {improvement:+.5f} better than the incumbent on the same window"
        )
    return check


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
@dataclass
class RetrainResult:
    kind: RetrainKind
    ran_on: date
    challenger: ModelVersion | None
    check: PromotionCheck | None
    promoted: bool
    trial_number: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"  {self.kind.value} retrain on {self.ran_on}"]
        if self.challenger is None:
            lines.append("    no challenger produced")
        else:
            lines.append(f"    challenger {self.challenger.version}")
        if self.check:
            lines.append(self.check.render())
        lines.append(f"    promoted: {self.promoted}")
        if self.trial_number:
            lines.append(
                f"    trial {self.trial_number} in the log. Fifty-two retrains a year "
                "is fifty-two attempts at beating the incumbent, and a challenger "
                "that wins once has not necessarily found anything."
            )
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def retrain(
    registry: Registry,
    *,
    kind: RetrainKind,
    today: date,
    challenger: ModelVersion | None,
    challenger_score: Scored | None,
    incumbent_score: Scored | None,
    apply: bool = False,
    experiment_log: ExperimentLog | None = None,
    config_hash: str = "",
    clock: Clock | None = None,
) -> RetrainResult:
    """Score a challenger, decide, and record the attempt either way.

    ``apply`` gates the promotion. A pipeline that promotes on every scheduled
    run changes what is trading without anyone deciding to, and the change looks
    like routine maintenance in the log.

    The attempt is logged **whether or not it is promoted**. A trial count that
    only includes the winners is not a trial count, and every deflated statistic
    downstream reads it.
    """
    incumbent = registry.production()
    result = RetrainResult(
        kind=kind, ran_on=today, challenger=challenger, check=None, promoted=False
    )

    if challenger is None or challenger_score is None:
        result.notes.append("nothing to evaluate")
        return result

    check = evaluate_promotion(
        challenger,
        incumbent,
        challenger_score=challenger_score,
        incumbent_score=incumbent_score,
        today=today,
    )
    result.check = check

    if experiment_log is not None:
        experiment_log.record(
            Experiment(
                name=f"retrain_{kind.value}_{challenger.version}",
                # From the clock, not the wall. R6 caught this: a retrain
                # replayed over a past quarter would stamp its trial with
                # today's date, and the experiment log would then show
                # fifty-two attempts that all happened on one afternoon.
                ts=(clock or RealClock()).now().isoformat(),
                config_hash=config_hash,
                dataset_version=f"{challenger_score.window_start}..{challenger_score.window_end}",
                feature_hash=challenger.version,
                metrics={
                    "log_loss": challenger_score.log_loss,
                    "brier": challenger_score.brier,
                    "n": float(challenger_score.n),
                    "incumbent_log_loss": incumbent_score.log_loss if incumbent_score else math.nan,
                    "approved": float(check.approved),
                },
                notes="; ".join(check.blockers) or "passed",
            )
        )
        result.trial_number = experiment_log.trial_count("retrain_")

    if not check.approved:
        return result
    if not apply:
        result.notes.append(
            "the check passed but --apply was not given. A pipeline that promotes "
            "on every scheduled run changes what is trading without anyone "
            "deciding to."
        )
        return result

    registry.promote(challenger.version, check, today=today)
    result.promoted = True
    return result


def due(kind: RetrainKind, *, today: date, last_run: date | None) -> bool:
    """Whether this retrain is due.

    Weekly for incremental, quarterly for full. A missed weekly is caught by the
    next one; a missed quarterly is not, so it is checked against the calendar
    rather than against a counter.
    """
    if last_run is None:
        return True
    gap = today - last_run
    return (
        gap >= timedelta(days=7) if kind is RetrainKind.INCREMENTAL else gap >= timedelta(days=90)
    )


def summarise(scores: Sequence[Scored]) -> str:
    return "\n".join(["  scored:"] + [s.render() for s in scores])
