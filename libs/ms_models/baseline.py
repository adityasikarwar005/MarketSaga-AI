"""M52 - The M0 rules baseline.

Every model in this project has to beat this one before it ships. That is the
whole purpose: a gradient-booster reporting 54% directional accuracy sounds like
a result until you discover the rule "follow the event prior when severity is
high, otherwise say nothing" gets 53.5%. The difference is the model's actual
contribution, and without a baseline nobody ever computes it.

It has a second job. When the model server is unhealthy, this runs instead: no
training, no dependencies, no state, fully deterministic. The system degrades to
a simpler opinion rather than to silence or to a stale prediction.

Its design principle is timidity. A rules baseline that outputs 0.85 is not
confident, it is broken - so the output is clamped to a narrow band around a
coin flip. Anything genuinely more certain than that has to be earned by a model
that was actually fitted to data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.ms_core.obs import METRICS, get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# The interface every model implements
# --------------------------------------------------------------------------- #
@dataclass
class Opinion:
    """What a model says about one instant."""

    p_up: float
    confidence: float = 0.0  # 0 = no opinion, 1 = maximally certain
    reasons: list[str] = field(default_factory=list)
    model: str = ""
    version: str = ""

    @property
    def direction(self) -> int:
        if self.p_up > 0.5:
            return 1
        return -1 if self.p_up < 0.5 else 0

    @property
    def edge(self) -> float:
        """Distance from a coin flip. The quantity sizing actually cares about."""
        return abs(self.p_up - 0.5)


class Model(ABC):
    """Everything downstream depends on this, and nothing else.

    The rules baseline, a gradient booster and an ensemble are interchangeable
    here, which is what lets the risk gate and the simulator be written once.
    """

    name: str = "unnamed"
    version: str = "0"

    @abstractmethod
    def predict(
        self, features: dict[str, float | None], *, event: dict[str, Any] | None = None
    ) -> Opinion: ...

    def predict_batch(
        self, rows: list[tuple[dict[str, float | None], dict[str, Any] | None]]
    ) -> list[Opinion]:
        return [self.predict(f, event=e) for f, e in rows]


# --------------------------------------------------------------------------- #
# M0
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineConfig:
    """Every threshold here is a candidate for replacement by evidence.

    They are starting values, not findings. The point of the baseline is to be
    beaten, and a baseline tuned until it wins is a baseline that has stopped
    doing its job.
    """

    min_severity: float = 5.0
    min_credibility: float = 0.75
    min_novelty: float = 0.45
    # Deliberately narrow. A rules baseline that claims 85% is not confident,
    # it is broken.
    max_edge: float = 0.12
    event_weight: float = 0.6
    momentum_weight: float = 0.25
    trend_quality_weight: float = 0.15
    # A move already several sigma old is information the market has priced.
    chase_guard_z: float = 2.0


class RulesBaseline(Model):
    """Event prior, confirmed by momentum, damped by everything suspicious."""

    name = "M0_rules"
    version = "1"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()

    def predict(
        self, features: dict[str, float | None], *, event: dict[str, Any] | None = None
    ) -> Opinion:
        cfg = self.config
        reasons: list[str] = []
        score = 0.0  # in [-1, 1]; positive is up

        event_score, event_reasons = self._event_component(event)
        score += cfg.event_weight * event_score
        reasons.extend(event_reasons)

        momentum, momentum_reasons = self._momentum_component(features)
        score += cfg.momentum_weight * momentum
        reasons.extend(momentum_reasons)

        quality, quality_reasons = self._trend_quality(features)
        score += cfg.trend_quality_weight * quality * (1 if momentum >= 0 else -1)
        reasons.extend(quality_reasons)

        score, damp_reasons = self._apply_dampers(score, features)
        reasons.extend(damp_reasons)

        p_up = 0.5 + max(-1.0, min(1.0, score)) * cfg.max_edge
        opinion = Opinion(
            p_up=p_up,
            confidence=min(1.0, abs(score)),
            reasons=reasons or ["no signal - default to a coin flip"],
            model=self.name,
            version=self.version,
        )
        METRICS.inc("baseline_prediction_total", direction=str(opinion.direction))
        return opinion

    # -- components ---------------------------------------------------------
    def _event_component(self, event: dict[str, Any] | None) -> tuple[float, list[str]]:
        cfg = self.config
        if not event:
            return 0.0, []

        severity = float(event.get("severity", 0.0) or 0.0)
        credibility = float(event.get("credibility", 0.0) or 0.0)
        novelty = float(event.get("novelty", 0.0) or 0.0)
        prior = event.get("prior_return")
        n_eff = int(event.get("n_eff", 0) or 0)

        if severity < cfg.min_severity:
            return 0.0, [f"severity {severity:.1f} below gate {cfg.min_severity}"]
        if credibility < cfg.min_credibility:
            return 0.0, [f"credibility {credibility:.2f} below gate - alert only, no direction"]
        if novelty < cfg.min_novelty:
            return 0.0, [f"novelty {novelty:.2f} below gate - likely already priced"]
        if prior is None or n_eff < 8:
            # Refusing to guess is the correct behaviour, and it is what the
            # impact library was built to make possible.
            return 0.0, [f"no usable prior (n_eff={n_eff}) - unknown event, no direction"]

        direction = 1.0 if float(prior) > 0 else -1.0
        strength = min(1.0, abs(float(prior)) / 0.005)  # 0.5% is a strong prior
        severity_scale = min(1.0, (severity / 10.0) ** 1.4 * 2.0)
        return direction * strength * severity_scale * credibility * novelty, [
            f"event prior {float(prior):+.3%} over {n_eff} historical cases, severity {severity:.1f}"
        ]

    def _momentum_component(self, features: dict[str, float | None]) -> tuple[float, list[str]]:
        z = features.get("ret_5m_z")
        if z is None:
            return 0.0, []
        # Volatility-normalised, so this means the same thing at 09:20 and 13:30.
        capped = max(-2.0, min(2.0, z)) / 2.0
        return capped, [f"5m momentum {z:+.2f} sigma"]

    def _trend_quality(self, features: dict[str, float | None]) -> tuple[float, list[str]]:
        er = features.get("efficiency_ratio_30m")
        if er is None:
            return 0.0, []
        if er < 0.3:
            return -0.5, [f"efficiency ratio {er:.2f} - choppy, momentum is unreliable"]
        return er, [f"efficiency ratio {er:.2f} - directional"]

    def _apply_dampers(
        self, score: float, features: dict[str, float | None]
    ) -> tuple[float, list[str]]:
        cfg = self.config
        reasons: list[str] = []

        z = features.get("ret_5m_z")
        if z is not None and abs(z) > cfg.chase_guard_z:
            # The move already happened. Following it now is chasing.
            score *= 0.3
            reasons.append(f"move already {z:+.1f} sigma - damped to avoid chasing")

        synthetic = features.get("synthetic_share_15m")
        if synthetic is not None and synthetic > 0.3:
            score *= 1.0 - synthetic
            reasons.append(f"{synthetic:.0%} of recent candles were carried forward - thin data")

        opening = features.get("is_opening_range")
        closing = features.get("is_closing_range")
        if opening:
            score *= 0.5
            reasons.append("opening range - price formation is not comparable")
        if closing:
            score *= 0.5
            reasons.append("closing range - squaring-off flow dominates")

        stale = features.get("chain_age_seconds")
        if stale is not None and stale > 60:
            reasons.append(f"chain is {stale:.0f}s stale")

        return score, reasons


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class ScoreCard:
    """How a model did. Log loss first, because accuracy hides miscalibration."""

    model: str
    n: int = 0
    log_loss: float = 0.0
    brier: float = 0.0
    accuracy: float = 0.0
    directional_n: int = 0

    def render(self) -> str:
        return (
            f"  {self.model:<12} n={self.n:,}  log_loss={self.log_loss:.5f}  "
            f"brier={self.brier:.5f}  acc={self.accuracy:.3f} (on {self.directional_n:,})"
        )


def score(model_name: str, probs: list[float], outcomes: list[int]) -> ScoreCard:
    """Score predictions against realised direction.

    Timeout labels are excluded from accuracy but kept for log loss: a model that
    was confident when nothing happened should still pay for that confidence.
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    card = ScoreCard(model=model_name, n=len(probs))
    if not probs:
        return card

    eps = 1e-12
    ll = 0.0
    brier = 0.0
    correct = 0
    directional = 0
    for p, y in zip(probs, outcomes, strict=True):
        target = 1.0 if y > 0 else 0.0 if y < 0 else 0.5
        p = min(1 - eps, max(eps, p))
        ll -= target * math.log(p) + (1 - target) * math.log(1 - p)
        brier += (p - target) ** 2
        if y != 0:
            directional += 1
            if (p > 0.5 and y > 0) or (p < 0.5 and y < 0):
                correct += 1

    card.log_loss = ll / len(probs)
    card.brier = brier / len(probs)
    card.directional_n = directional
    card.accuracy = correct / directional if directional else 0.0
    return card


@dataclass
class Comparison:
    baseline: ScoreCard
    challenger: ScoreCard
    delta_log_loss: float
    bootstrap_ci: tuple[float, float] | None = None

    @property
    def beats_baseline(self) -> bool:
        """Requires the confidence interval to exclude zero, not just a better mean.

        A challenger that wins on average but whose interval straddles zero has
        not been shown to beat anything, and shipping it on the point estimate
        is how a project accumulates models that do nothing.
        """
        if self.bootstrap_ci is None:
            return False
        lo, hi = self.bootstrap_ci
        return lo > 0 and hi > 0

    def render(self) -> str:
        lines = [self.baseline.render(), self.challenger.render()]
        ci = (
            f"  [{self.bootstrap_ci[0]:+.5f}, {self.bootstrap_ci[1]:+.5f}]"
            if self.bootstrap_ci
            else "  (no interval computed)"
        )
        verdict = "BEATS baseline" if self.beats_baseline else "does NOT beat baseline"
        lines.append(f"  log-loss improvement {self.delta_log_loss:+.5f}{ci}  ->  {verdict}")
        return "\n".join(lines)


def compare(
    baseline_probs: list[float],
    challenger_probs: list[float],
    outcomes: list[int],
    *,
    baseline_name: str = "M0_rules",
    challenger_name: str = "challenger",
    bootstrap: int = 2000,
    seed: int = 7,
) -> Comparison:
    """Bootstrap the per-sample log-loss difference.

    Paired resampling, not two independent samples: both models saw the same
    instants, and pairing removes the variance that comes from which instants
    happened to be in the set.
    """
    import random

    base = score(baseline_name, baseline_probs, outcomes)
    chall = score(challenger_name, challenger_probs, outcomes)

    eps = 1e-12

    def per_sample_loss(p: float, y: int) -> float:
        target = 1.0 if y > 0 else 0.0 if y < 0 else 0.5
        p = min(1 - eps, max(eps, p))
        return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    diffs = [
        per_sample_loss(b, y) - per_sample_loss(c, y)
        for b, c, y in zip(baseline_probs, challenger_probs, outcomes, strict=True)
    ]
    delta = sum(diffs) / len(diffs) if diffs else 0.0

    ci: tuple[float, float] | None = None
    if diffs and bootstrap > 0:
        rng = random.Random(seed)
        n = len(diffs)
        means = []
        for _ in range(bootstrap):
            means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        ci = (means[int(0.025 * bootstrap)], means[int(0.975 * bootstrap) - 1])

    return Comparison(baseline=base, challenger=chall, delta_log_loss=delta, bootstrap_ci=ci)


def climatology(outcomes: list[int]) -> float:
    """The base rate - what a model that knows nothing but the average would say.

    The floor under every skill score. A model that cannot beat this has learned
    nothing at all, as opposed to merely having learned less than the baseline.
    """
    directional = [y for y in outcomes if y != 0]
    if not directional:
        return 0.5
    return sum(1 for y in directional if y > 0) / len(directional)


@dataclass
class PredictionRun:
    """A model's output over a labelled set, ready for scoring."""

    model: str
    version: str
    timestamps: list[datetime] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)

    def add(self, ts: datetime, opinion: Opinion, outcome: int) -> None:
        self.timestamps.append(ts)
        self.probs.append(opinion.p_up)
        self.outcomes.append(outcome)

    def scorecard(self) -> ScoreCard:
        return score(self.model, self.probs, self.outcomes)
