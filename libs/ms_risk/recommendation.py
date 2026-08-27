"""M77 - the recommendation assembler.

Everything upstream of here produces a number. This produces the **record**: one
immutable object that says what was recommended, what every model believed at the
time, which gates it passed, and why - in enough detail that the decision can be
reconstructed months later by someone who was not there.

Why the explanation is a first-class field
------------------------------------------

Six months from now the question will be "why did this system lose money on this
class of event", and the only way to answer it is to read what the system
believed when it acted. A stored, structured reason for every decision is what
makes that investigation possible; without it the system cannot improve, because
nobody can tell a bad model from a bad input from a bad rule.

So ``explanation`` is ordered, non-empty, and validated. A recommendation that
cannot say why it exists does not get built - :func:`assemble` raises rather than
emitting one, because an empty explanation is exactly the record that will be
needed most and noticed least.

Immutable, and stored before published
--------------------------------------

The record is frozen after construction. Not as ceremony: a mutable
recommendation is one that can be edited after the fact to match what happened,
and the edit will be well-intentioned - correcting a field that "was obviously
wrong" - and it destroys the only evidence of what was actually believed.

``store_then_publish`` writes before it emits, and if the write fails it does not
emit. The alternative ordering produces recommendations that were acted on and
never recorded, which is the exact population you most need to study.

Mode is never absent
--------------------

``mode`` has no default. PAPER and LIVE differ by whether real money moves, and a
field that defaults to one of them will eventually default to the wrong one on a
code path nobody tested. It must be passed explicitly every time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from libs.ms_core.obs import get_logger
from libs.ms_risk.gate import KillLevel, RiskDecision, Verdict

log = get_logger(__name__)

REC_ROOT = Path(__file__).resolve().parents[2] / "data" / "recommendations"


class Mode(StrEnum):
    PAPER = "PAPER"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"


class Action(StrEnum):
    TAKE = "TAKE"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Trigger:
    """What caused this to be considered at all."""

    event_id: str
    event_class: str
    severity: float
    novelty: float
    credibility: float
    sources: tuple[str, ...] = ()
    seconds_since_event: float = 0.0


@dataclass(frozen=True)
class Forecast:
    """What the models believed, with their uncertainty attached.

    ``p_up_ci`` is not optional decoration. A probability without an interval is
    a point estimate, and the sizing layer treats those very differently - see
    ``ms_options.sizing``, where the interval is the difference between running
    Kelly on evidence and running it on a guess.
    """

    horizon_min: int
    p_up: float
    p_up_ci: tuple[float, float] | None = None
    quantiles: Mapping[str, float] = field(default_factory=dict)
    sigma_forecast: float | None = None
    d_iv_forecast: float | None = None
    implied_move: float | None = None

    @property
    def edge_ratio(self) -> float | None:
        """Forecast move against what the market has already priced.

        Below 1.0 there is nothing to buy: the move being predicted is smaller
        than the one already in the premium.
        """
        if self.sigma_forecast is None or not self.implied_move:
            return None
        return self.sigma_forecast / self.implied_move

    @property
    def has_interval(self) -> bool:
        return self.p_up_ci is not None


@dataclass(frozen=True)
class Leg:
    side: str
    strike: float
    opt: str
    qty: int


@dataclass(frozen=True)
class Structure:
    type: str
    legs: tuple[Leg, ...]
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven: float


@dataclass(frozen=True)
class Sizing:
    kelly_frac: float
    used_frac: float
    lots: int
    capital_at_risk: float


@dataclass(frozen=True)
class Exits:
    """Three exits, always. A position with only a target is a position with no
    plan for being wrong, and the time stop is the one people leave out."""

    target: float
    stop: float
    time_stop: str
    vol_stop: str | None = None


@dataclass(frozen=True)
class Recommendation:
    """The immutable record. Frozen, complete, and reconstructable."""

    reco_id: str
    trace_id: str
    ts_emitted: datetime
    mode: Mode
    action: Action
    trigger: Trigger
    forecast: Forecast
    explanation: tuple[str, ...]
    gates_passed: tuple[str, ...]
    model_versions: Mapping[str, str]
    structure: Structure | None = None
    greeks_net: Mapping[str, float] = field(default_factory=dict)
    sizing: Sizing | None = None
    exits: Exits | None = None
    rejected_because: tuple[str, ...] = ()
    kill_level: str = KillLevel.NONE.name

    @property
    def is_actionable(self) -> bool:
        return self.action is Action.TAKE and self.sizing is not None and self.sizing.lots > 0

    def to_json(self) -> str:
        def encode(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, StrEnum):
                return o.value
            raise TypeError(f"cannot serialise {type(o).__name__}")

        return json.dumps(asdict(self), default=encode, sort_keys=True, indent=1)

    def render(self) -> str:
        head = f"  [{self.mode.value}] {self.action.value} {self.reco_id[:8]}"
        if self.structure:
            head += f"  {self.structure.type}"
        if self.sizing:
            head += f"  {self.sizing.lots} lot(s), {self.sizing.capital_at_risk:,.0f} at risk"
        lines = [head, f"    trigger: {self.trigger.event_class} sev {self.trigger.severity:.1f}"]

        ci = ""
        if self.forecast.p_up_ci:
            lo, hi = self.forecast.p_up_ci
            ci = f" [{lo:.3f}, {hi:.3f}]"
        else:
            ci = "  (NO INTERVAL - sized on a point estimate)"
        lines.append(f"    p_up {self.forecast.p_up:.3f}{ci}")

        lines += [f"    why: {e}" for e in self.explanation]
        if self.rejected_because:
            lines += [f"    BLOCKED: {r}" for r in self.rejected_because]
        lines.append(f"    gates: {', '.join(self.gates_passed) or 'none recorded'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
class IncompleteRecommendation(ValueError):
    """Raised rather than emitting a record that cannot be reconstructed."""


def assemble(
    *,
    reco_id: str,
    trace_id: str,
    ts_emitted: datetime,
    mode: Mode,
    trigger: Trigger,
    forecast: Forecast,
    explanation: Sequence[str],
    gates_passed: Sequence[str],
    model_versions: Mapping[str, str],
    risk: RiskDecision,
    structure: Structure | None = None,
    greeks_net: Mapping[str, float] | None = None,
    sizing: Sizing | None = None,
    exits: Exits | None = None,
) -> Recommendation:
    """Build the record, refusing to build an unusable one.

    The refusals below are all of the same kind: a field that is easy to leave
    empty, and whose emptiness only becomes a problem long after the run, when
    the record is the only thing left.
    """
    reasons = [e.strip() for e in explanation if e and e.strip()]
    if not reasons:
        raise IncompleteRecommendation(
            "explanation is empty. A recommendation that cannot say why it "
            "exists is the record you will need most and notice least - six "
            "months from now it is the only evidence of what was believed."
        )
    if not model_versions:
        raise IncompleteRecommendation(
            "model_versions is empty. Without it a stored decision cannot be "
            "reproduced, because nobody knows which models made it."
        )

    took = risk.verdict is not Verdict.REJECT
    if took and sizing is None:
        raise IncompleteRecommendation(
            "the risk gate allowed this and no sizing was supplied - an "
            "actionable recommendation with no size is not actionable"
        )
    if took and exits is None:
        raise IncompleteRecommendation(
            "no exits supplied. A position with a target and no stop and no "
            "time stop has no plan for being wrong, which is the case it will "
            "spend most of its life in."
        )
    if took and structure is None:
        raise IncompleteRecommendation("allowed to trade with no structure to trade")

    if not forecast.has_interval:
        # Not fatal - a model that cannot produce an interval is a real state -
        # but it goes on the record, because the sizing layer treats a point
        # estimate very differently and the reader has to be able to see which
        # happened.
        reasons.append(
            "NOTE: the forecast carried no interval, so any sizing downstream "
            "used a point estimate"
        )

    rec = Recommendation(
        reco_id=reco_id,
        trace_id=trace_id,
        ts_emitted=ts_emitted,
        mode=mode,
        action=Action.TAKE if took else Action.SKIP,
        trigger=trigger,
        forecast=forecast,
        explanation=tuple(reasons),
        gates_passed=tuple(gates_passed),
        model_versions=dict(model_versions),
        structure=structure if took else None,
        greeks_net=dict(greeks_net or {}),
        sizing=sizing if took else None,
        exits=exits if took else None,
        rejected_because=tuple(risk.breaches),
        kill_level=risk.kill_level.name,
    )
    log.info(
        "recommendation assembled",
        fields={
            "reco_id": reco_id,
            "mode": mode.value,
            "action": rec.action.value,
            "lots": sizing.lots if sizing else 0,
            "gates": len(gates_passed),
        },
    )
    return rec


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class RecommendationStore:
    """Append-only, one file per recommendation, named by id.

    One file rather than one log, because a recommendation is read individually
    - "what did we think on the 14th" - and a single growing file makes that a
    scan. Named by id so the file that a trace refers to can be found without an
    index.
    """

    def __init__(self, root: Path = REC_ROOT) -> None:
        self.root = root

    def path(self, reco_id: str) -> Path:
        return self.root / f"{reco_id}.json"

    def save(self, rec: Recommendation) -> Path:
        p = self.path(rec.reco_id)
        if p.exists():
            raise FileExistsError(
                f"{rec.reco_id} is already stored. Recommendations are immutable; "
                "a corrected decision is a new record that supersedes this one, "
                "not an edit to it - otherwise the only evidence of what was "
                "actually believed is gone."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        p.write_text(rec.to_json(), encoding="utf-8")
        return p

    def load(self, reco_id: str) -> dict[str, Any]:
        p = self.path(reco_id)
        if not p.exists():
            raise FileNotFoundError(f"no recommendation stored as {reco_id}")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    def ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))


def store_then_publish(
    rec: Recommendation,
    store: RecommendationStore,
    publish: Any,
) -> Path:
    """Write first. Emit only if the write succeeded.

    The other ordering produces recommendations that were acted on and never
    recorded - which is precisely the population that most needs studying, and
    precisely the one that would be missing. If the store fails, nothing is
    published and the failure propagates.
    """
    path = store.save(rec)
    try:
        publish(rec)
    except Exception:
        log.info(
            "recommendation stored but publishing failed - the record survives",
            fields={"reco_id": rec.reco_id, "path": str(path)},
        )
        raise
    return path
