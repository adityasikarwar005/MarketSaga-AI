"""M5A - the model registry, and the gate a challenger has to pass.

MLflow and ONNX serving are deployment concerns and are deferred. What is here
is the part with correctness requirements: **which model is in production, how a
new one gets there, and what it had to prove first.** That has to be testable
without standing up a registry service, and must not change behaviour when the
service is swapped.

Four stages, and the order is not decoration
--------------------------------------------

``staging → shadow → production → archived``

A model that goes from a good backtest straight to production has been evaluated
on data it was fitted near, by a person who wanted it to work. Shadow mode is
the only stage that produces evidence nobody chose: the challenger predicts on
live data, its predictions are recorded, and it trades nothing. Fifteen sessions
is the minimum, and the number is a floor rather than a default.

Every prediction carries its model version
------------------------------------------

Without it, attribution cannot attribute. A week that mixes two model versions
and reports one hit rate is reporting a number about nothing, and the mixing is
invisible - the report looks exactly like a single-model week.

Production loads from the registry, never from a path
------------------------------------------------------

A file path is a claim about what is running that nobody checks. The registry is
the claim, and :func:`production_model` is the only way to ask.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Sessions a challenger must run in shadow before it may be promoted. A floor,
# not a default: fifteen sessions is roughly three weeks, which is the shortest
# window in which a model can meet a bad Monday, an expiry and a quiet Friday.
MIN_SHADOW_SESSIONS = 15

# Weeks of out-of-sample data the challenger must beat the incumbent on. Four,
# because the challenger was fitted up to the start of it - a shorter window is
# mostly the tail of the training set.
OOS_WEEKS = 4


class Stage(IntEnum):
    """Ordered so "at least shadow" is a comparison."""

    STAGING = 0
    SHADOW = 1
    PRODUCTION = 2
    ARCHIVED = 3


class RegistryError(RuntimeError):
    pass


@dataclass
class ModelVersion:
    """One trained model, and everything needed to decide whether to trust it."""

    version: str
    trained_on: date
    trained_through: date  # the last day in its training data
    stage: Stage = Stage.STAGING
    shadow_sessions: int = 0
    promoted_on: date | None = None
    archived_on: date | None = None
    notes: str = ""

    @property
    def is_live(self) -> bool:
        return self.stage is Stage.PRODUCTION

    @property
    def shadow_complete(self) -> bool:
        return self.shadow_sessions >= MIN_SHADOW_SESSIONS

    def render(self) -> str:
        extra = ""
        if self.stage is Stage.SHADOW:
            extra = f"  ({self.shadow_sessions}/{MIN_SHADOW_SESSIONS} sessions)"
        elif self.stage is Stage.PRODUCTION:
            extra = f"  (live since {self.promoted_on})"
        return f"    {self.version:<20} {self.stage.name:<11} trained through {self.trained_through}{extra}"


@dataclass
class PromotionCheck:
    """Why a challenger may or may not be promoted. Every reason, not the first."""

    challenger: str
    incumbent: str | None
    blockers: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        head = f"  {self.challenger} vs {self.incumbent or 'nothing in production'}"
        lines = [head]
        lines += [f"    ok: {e}" for e in self.evidence]
        lines += [f"    BLOCKED: {b}" for b in self.blockers]
        lines.append("    -> promote" if self.approved else "    -> stays where it is")
        return "\n".join(lines)


@dataclass
class Registry:
    """What exists, what is live, and what it took to get there."""

    versions: dict[str, ModelVersion] = field(default_factory=dict)

    def register(self, model: ModelVersion) -> ModelVersion:
        if model.version in self.versions:
            raise RegistryError(
                f"{model.version} is already registered. A retrain that produces "
                "the same version string as an existing model makes every "
                "prediction stamped with it ambiguous."
            )
        self.versions[model.version] = model
        log.info("model registered", fields={"version": model.version})
        return model

    def production(self) -> ModelVersion | None:
        """The single live model, or None.

        Returns None rather than raising: "nothing is in production" is a real
        state on day one and after an emergency archive, and a caller that has
        to catch an exception for it will eventually catch it and continue.
        """
        live = [m for m in self.versions.values() if m.is_live]
        if len(live) > 1:
            raise RegistryError(
                f"{len(live)} models are in production: {[m.version for m in live]}. "
                "Two live models means predictions are coming from whichever one "
                "the caller happened to load, and attribution is meaningless."
            )
        return live[0] if live else None

    def in_shadow(self) -> list[ModelVersion]:
        return [m for m in self.versions.values() if m.stage is Stage.SHADOW]

    # ------------------------------------------------------------- transitions
    def start_shadow(self, version: str) -> ModelVersion:
        m = self._get(version)
        if m.stage is not Stage.STAGING:
            raise RegistryError(
                f"{version} is {m.stage.name}, not STAGING. Shadow is a stage a "
                "model enters once; restarting it would reset a session count "
                "that is supposed to be evidence."
            )
        m.stage = Stage.SHADOW
        m.shadow_sessions = 0
        return m

    def record_shadow_session(self, version: str) -> ModelVersion:
        """One more session of live predictions that traded nothing."""
        m = self._get(version)
        if m.stage is not Stage.SHADOW:
            raise RegistryError(f"{version} is {m.stage.name} and is not shadowing")
        m.shadow_sessions += 1
        return m

    def promote(self, version: str, check: PromotionCheck, *, today: date) -> ModelVersion:
        """Make a challenger live, archiving whatever it replaces.

        Takes the check rather than performing it, so the decision and the
        action are separable: a caller can print the check, sleep on it, and
        promote later - and cannot promote without having one.
        """
        if not check.approved:
            raise RegistryError(
                f"{version} has not passed its promotion check: " f"{'; '.join(check.blockers)}"
            )
        if check.challenger != version:
            raise RegistryError(
                f"the check is for {check.challenger!r} and the promotion is for "
                f"{version!r}. Promoting on someone else's evidence is how the "
                "wrong model goes live."
            )

        m = self._get(version)
        outgoing = self.production()
        if outgoing is not None:
            outgoing.stage = Stage.ARCHIVED
            outgoing.archived_on = today
        m.stage = Stage.PRODUCTION
        m.promoted_on = today
        log.info(
            "model promoted",
            fields={"version": version, "replaced": outgoing.version if outgoing else None},
        )
        return m

    def archive(self, version: str, *, today: date, reason: str) -> ModelVersion:
        """Take a model out of service. Requires a reason, always.

        An archived model with no reason is one nobody can learn from, and the
        reason a model was pulled is the most useful thing about it later.
        """
        if not reason.strip():
            raise RegistryError("archiving a model needs a reason")
        m = self._get(version)
        m.stage = Stage.ARCHIVED
        m.archived_on = today
        m.notes = reason.strip()
        return m

    def _get(self, version: str) -> ModelVersion:
        if version not in self.versions:
            raise RegistryError(f"{version} is not registered")
        return self.versions[version]

    def render(self) -> str:
        if not self.versions:
            return "  registry is empty"
        order = sorted(self.versions.values(), key=lambda m: (m.stage, m.version))
        return "\n".join(["  registry:"] + [m.render() for m in order])


def production_model(registry: Registry) -> ModelVersion:
    """The live model, or a refusal.

    The only sanctioned way to ask. A file path is a claim about what is running
    that nobody checks; this raises when nothing is live rather than falling back
    to a default, because a system trading on a default model is a system whose
    operator does not know what it is trading on.
    """
    live = registry.production()
    if live is None:
        raise RegistryError(
            "nothing is in production. Refusing rather than loading a default - "
            "a system trading on a fallback model is one whose operator does not "
            "know what it is trading on."
        )
    return live


def stamp(version: str, payload: dict[str, object]) -> dict[str, object]:
    """Attach the model version to a prediction.

    Without it a week that mixes two versions reports one hit rate about
    nothing, and the mixing is invisible - the report looks exactly like a
    single-model week.
    """
    if not version:
        raise ValueError("a prediction with no model version cannot be attributed")
    return {**payload, "model_version": version}


def versions_in(predictions: Sequence[dict[str, object]]) -> set[str]:
    """Which models produced this set. More than one is a finding."""
    return {str(p.get("model_version", "")) for p in predictions}
