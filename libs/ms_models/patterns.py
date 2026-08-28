"""Candlestick pattern detection, and the measurement that has to come with it.

TradingView's own page for this feature says candlestick patterns "are used to
predict future price movements". That is a description of what traders do, not a
claim that it works, and the page makes no claim about accuracy either way. The
detector and the question of whether the detector predicts anything are two
different pieces of work, and shipping the first without the second is how a
chart annotation becomes a trading signal by accident.

So this module is in two halves. :class:`Detector` finds the patterns. Everything
below :func:`evaluate` exists to answer whether finding them is worth anything,
and it is built to make the answer come out negative when it should.

Precise definitions, not resemblances
--------------------------------------

"Looks like a hammer" is not testable. Every pattern here is a predicate over
numbers with named thresholds, so two people reading the code agree on what was
detected, and a threshold that was chosen to make a result appear can be seen
being chosen.

The thresholds are conventional rather than fitted. Fitting them to maximise
detected edge is the same act as searching specifications until a gate passes -
it produces a number that describes the search, not the market.

Testing twenty patterns is twenty tests
-----------------------------------------

The failure this module is most likely to produce is a real-looking finding on
the best of twenty patterns. With twenty independent tests at p<0.05, the chance
of at least one false positive is 64%. :func:`evaluate` applies a Sidak
correction across every pattern tested, and reports the count it corrected for,
because a corrected p-value whose n is not stated is an uncorrected p-value.

The null test is the point
----------------------------

:func:`null_check` re-runs the whole evaluation against returns that have been
shuffled against the candles - the shapes are real, the outcomes are randomised,
and every genuine relationship is destroyed while every artefact of the method
survives. Whatever it reports as significant is what this pipeline finds when
there is nothing to find.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from statistics import NormalDist, fmean, pstdev

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

_ND = NormalDist()

# --- shape thresholds ------------------------------------------------------
# Conventional values, not fitted ones. Tuning these until a pattern looks
# predictive is the same act as searching specifications until a gate passes.
DOJI_BODY = 0.10  # body <= 10% of the range
LONG_BODY = 0.60  # body >= 60% of the range
SHADOW_RATIO = 2.0  # the long shadow is >= 2x the body
SMALL_SHADOW = 0.10  # the opposite shadow is <= 10% of the range
MARUBOZU_SHADOW = 0.03

# Minimum detections before a pattern's mean forward return means anything.
MIN_OCCURRENCES = 30

# Significance before correction. The corrected threshold is derived from this
# and the number of patterns tested.
ALPHA = 0.05


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candle:
    day: date
    o: float
    h: float
    low: float
    c: float

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def range_(self) -> float:
        return self.h - self.low

    @property
    def upper(self) -> float:
        return self.h - max(self.o, self.c)

    @property
    def lower(self) -> float:
        return min(self.o, self.c) - self.low

    @property
    def bullish(self) -> bool:
        return self.c > self.o

    @property
    def mid(self) -> float:
        return (self.o + self.c) / 2.0

    def body_share(self) -> float:
        """Body as a fraction of range. Zero range is a halted or untraded day."""
        return self.body / self.range_ if self.range_ > 0 else 0.0


# --------------------------------------------------------------------------- #
# The predicates
# --------------------------------------------------------------------------- #
# Each takes the candles up to and including index i and returns True if the
# pattern completes at i. Lookback is stated per pattern so the evaluator can
# skip the leading window rather than silently reading a negative index.
Predicate = Callable[[Sequence[Candle], int], bool]


def _trend(c: Sequence[Candle], i: int, n: int = 5) -> int:
    """-1 down, +1 up, 0 flat, over the n closes before i.

    Several patterns are only defined as reversals, which means the preceding
    trend is part of the definition rather than context. A hammer in an uptrend
    is a hanging man, and calling both a hammer merges two opposite signals.
    """
    if i < n:
        return 0
    before, now = c[i - n].c, c[i - 1].c
    if now > before * 1.005:
        return 1
    if now < before * 0.995:
        return -1
    return 0


def is_doji(c: Sequence[Candle], i: int) -> bool:
    return c[i].range_ > 0 and c[i].body_share() <= DOJI_BODY


def is_marubozu_bull(c: Sequence[Candle], i: int) -> bool:
    k = c[i]
    return (
        k.range_ > 0
        and k.bullish
        and k.body_share() >= 1 - 2 * MARUBOZU_SHADOW
        and k.upper <= MARUBOZU_SHADOW * k.range_
        and k.lower <= MARUBOZU_SHADOW * k.range_
    )


def is_marubozu_bear(c: Sequence[Candle], i: int) -> bool:
    k = c[i]
    return (
        k.range_ > 0
        and not k.bullish
        and k.body_share() >= 1 - 2 * MARUBOZU_SHADOW
        and k.upper <= MARUBOZU_SHADOW * k.range_
        and k.lower <= MARUBOZU_SHADOW * k.range_
    )


def _hammer_shape(k: Candle) -> bool:
    return (
        k.range_ > 0
        and k.body > 0
        and k.lower >= SHADOW_RATIO * k.body
        and k.upper <= SMALL_SHADOW * k.range_
    )


def _star_shape(k: Candle) -> bool:
    return (
        k.range_ > 0
        and k.body > 0
        and k.upper >= SHADOW_RATIO * k.body
        and k.lower <= SMALL_SHADOW * k.range_
    )


def is_hammer(c: Sequence[Candle], i: int) -> bool:
    """Long lower shadow after a decline. The trend is part of the definition."""
    return _hammer_shape(c[i]) and _trend(c, i) < 0


def is_hanging_man(c: Sequence[Candle], i: int) -> bool:
    """The same shape after a rise, and the opposite signal."""
    return _hammer_shape(c[i]) and _trend(c, i) > 0


def is_inverted_hammer(c: Sequence[Candle], i: int) -> bool:
    return _star_shape(c[i]) and _trend(c, i) < 0


def is_shooting_star(c: Sequence[Candle], i: int) -> bool:
    return _star_shape(c[i]) and _trend(c, i) > 0


def is_bullish_engulfing(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        not p.bullish
        and k.bullish
        and k.c >= p.o
        and k.o <= p.c
        and k.body > p.body
        and _trend(c, i) < 0
    )


def is_bearish_engulfing(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        p.bullish
        and not k.bullish
        and k.o >= p.c
        and k.c <= p.o
        and k.body > p.body
        and _trend(c, i) > 0
    )


def is_bullish_harami(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        not p.bullish
        and k.bullish
        and k.o > p.c
        and k.c < p.o
        and p.body > 0
        and k.body < p.body * 0.6
    )


def is_bearish_harami(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        p.bullish
        and not k.bullish
        and k.o < p.c
        and k.c > p.o
        and p.body > 0
        and k.body < p.body * 0.6
    )


def is_piercing_line(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        not p.bullish
        and k.bullish
        and p.body > 0
        and k.o < p.low
        and p.mid < k.c < p.o
        and _trend(c, i) < 0
    )


def is_dark_cloud(c: Sequence[Candle], i: int) -> bool:
    if i < 1:
        return False
    p, k = c[i - 1], c[i]
    return (
        p.bullish
        and not k.bullish
        and p.body > 0
        and k.o > p.h
        and p.o < k.c < p.mid
        and _trend(c, i) > 0
    )


def is_morning_star(c: Sequence[Candle], i: int) -> bool:
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return (
        not a.bullish
        and a.body_share() >= LONG_BODY
        and b.body < a.body * 0.4
        and k.bullish
        and k.c > a.mid
        and _trend(c, i) < 0
    )


def is_evening_star(c: Sequence[Candle], i: int) -> bool:
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return (
        a.bullish
        and a.body_share() >= LONG_BODY
        and b.body < a.body * 0.4
        and not k.bullish
        and k.c < a.mid
        and _trend(c, i) > 0
    )


def is_abandoned_baby_bull(c: Sequence[Candle], i: int) -> bool:
    """TradingView's own example pattern. Rare by construction: it needs two gaps."""
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return not a.bullish and is_doji(c, i - 1) and b.h < a.low and k.low > b.h and k.bullish


def is_abandoned_baby_bear(c: Sequence[Candle], i: int) -> bool:
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return a.bullish and is_doji(c, i - 1) and b.low > a.h and k.h < b.low and not k.bullish


def is_three_white_soldiers(c: Sequence[Candle], i: int) -> bool:
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return (
        all(x.bullish and x.body_share() >= 0.5 for x in (a, b, k))
        and b.o > a.o
        and k.o > b.o
        and b.c > a.c
        and k.c > b.c
    )


def is_three_black_crows(c: Sequence[Candle], i: int) -> bool:
    if i < 2:
        return False
    a, b, k = c[i - 2], c[i - 1], c[i]
    return (
        all((not x.bullish) and x.body_share() >= 0.5 for x in (a, b, k))
        and b.o < a.o
        and k.o < b.o
        and b.c < a.c
        and k.c < b.c
    )


@dataclass(frozen=True)
class Pattern:
    name: str
    direction: Direction
    lookback: int
    predicate: Predicate


CATALOGUE: tuple[Pattern, ...] = (
    Pattern("doji", Direction.NEUTRAL, 0, is_doji),
    Pattern("marubozu_bull", Direction.BULLISH, 0, is_marubozu_bull),
    Pattern("marubozu_bear", Direction.BEARISH, 0, is_marubozu_bear),
    Pattern("hammer", Direction.BULLISH, 5, is_hammer),
    Pattern("hanging_man", Direction.BEARISH, 5, is_hanging_man),
    Pattern("inverted_hammer", Direction.BULLISH, 5, is_inverted_hammer),
    Pattern("shooting_star", Direction.BEARISH, 5, is_shooting_star),
    Pattern("bullish_engulfing", Direction.BULLISH, 5, is_bullish_engulfing),
    Pattern("bearish_engulfing", Direction.BEARISH, 5, is_bearish_engulfing),
    Pattern("bullish_harami", Direction.BULLISH, 1, is_bullish_harami),
    Pattern("bearish_harami", Direction.BEARISH, 1, is_bearish_harami),
    Pattern("piercing_line", Direction.BULLISH, 5, is_piercing_line),
    Pattern("dark_cloud_cover", Direction.BEARISH, 5, is_dark_cloud),
    Pattern("morning_star", Direction.BULLISH, 5, is_morning_star),
    Pattern("evening_star", Direction.BEARISH, 5, is_evening_star),
    Pattern("abandoned_baby_bull", Direction.BULLISH, 2, is_abandoned_baby_bull),
    Pattern("abandoned_baby_bear", Direction.BEARISH, 2, is_abandoned_baby_bear),
    Pattern("three_white_soldiers", Direction.BULLISH, 2, is_three_white_soldiers),
    Pattern("three_black_crows", Direction.BEARISH, 2, is_three_black_crows),
)


def detect(
    candles: Sequence[Candle], *, catalogue: Sequence[Pattern] = CATALOGUE
) -> dict[str, list[int]]:
    """Every index at which each pattern completes."""
    found: dict[str, list[int]] = {p.name: [] for p in catalogue}
    for p in catalogue:
        for i in range(p.lookback, len(candles)):
            if p.predicate(candles, i):
                found[p.name].append(i)
    return found


# --------------------------------------------------------------------------- #
# Whether any of it predicts anything
# --------------------------------------------------------------------------- #
def sidak_alpha(alpha: float, tests: int) -> float:
    """Per-test threshold so the family-wise error rate stays at ``alpha``.

    Twenty independent tests at 0.05 give a 64% chance of at least one false
    positive. Sidak rather than Bonferroni because the tests here are close to
    independent and Bonferroni is needlessly conservative at this count.
    """
    if tests < 1:
        raise PatternError("cannot correct for fewer than one test")
    return 1.0 - (1.0 - alpha) ** (1.0 / tests)


@dataclass
class PatternResult:
    name: str
    direction: Direction
    n: int
    mean_forward: float
    unconditional: float
    t_stat: float
    p_value: float
    hit_rate: float

    @property
    def edge(self) -> float:
        """Mean forward return in the pattern's own claimed direction, net of
        the unconditional drift.

        Against the unconditional mean, not against zero. An index that drifts
        up makes every bullish pattern look right and every bearish one look
        wrong, and the drift is available without detecting anything.
        """
        signed = self.mean_forward - self.unconditional
        return -signed if self.direction is Direction.BEARISH else signed

    def significant(self, threshold: float) -> bool:
        return self.n >= MIN_OCCURRENCES and self.p_value < threshold

    def render(self, threshold: float) -> str:
        mark = "  <- significant" if self.significant(threshold) else ""
        thin = "  (too few)" if self.n < MIN_OCCURRENCES else ""
        return (
            f"    {self.name:<24}{self.direction.value:<9}n={self.n:<5}"
            f"edge={self.edge * 100:+7.3f}%  hit={self.hit_rate:5.1%}  "
            f"p={self.p_value:.4f}{mark}{thin}"
        )


@dataclass
class Evaluation:
    results: list[PatternResult] = field(default_factory=list)
    tests: int = 0
    alpha: float = ALPHA
    horizon: int = 1
    label: str = ""

    @property
    def threshold(self) -> float:
        return sidak_alpha(self.alpha, self.tests) if self.tests else self.alpha

    @property
    def significant(self) -> list[PatternResult]:
        return [r for r in self.results if r.significant(self.threshold)]

    def render(self) -> str:
        lines = [
            f"  {self.label or 'patterns'} · {self.horizon}-bar forward return",
            f"  {self.tests} patterns tested, Sidak threshold p < {self.threshold:.5f} "
            f"(uncorrected {self.alpha})",
        ]
        lines += [
            r.render(self.threshold) for r in sorted(self.results, key=lambda r: -abs(r.edge))
        ]
        sig = self.significant
        lines.append(
            f"  -> {len(sig)} of {self.tests} significant after correction"
            + (": " + ", ".join(r.name for r in sig) if sig else "")
        )
        return "\n".join(lines)


def forward_returns(candles: Sequence[Candle], horizon: int) -> list[float]:
    """Close-to-close return over ``horizon`` bars, NaN where it runs off the end."""
    out: list[float] = []
    for i in range(len(candles)):
        j = i + horizon
        out.append(candles[j].c / candles[i].c - 1.0 if j < len(candles) else math.nan)
    return out


def evaluate(
    candles: Sequence[Candle],
    *,
    horizon: int = 1,
    catalogue: Sequence[Pattern] = CATALOGUE,
    alpha: float = ALPHA,
    label: str = "",
    returns: Sequence[float] | None = None,
) -> Evaluation:
    """Mean forward return after each pattern, against the unconditional mean.

    ``returns`` can be supplied to score the same detections against a different
    outcome series - which is how :func:`null_check` destroys the relationship
    while keeping every artefact of the method.
    """
    fwd = list(returns) if returns is not None else forward_returns(candles, horizon)
    usable = [r for r in fwd if not math.isnan(r)]
    if len(usable) < MIN_OCCURRENCES:
        raise PatternError(f"{len(usable)} usable bars is not enough to evaluate anything")

    base_mean = fmean(usable)
    base_sd = pstdev(usable) or 1e-12

    found = detect(candles, catalogue=catalogue)
    ev = Evaluation(tests=len(catalogue), alpha=alpha, horizon=horizon, label=label)

    for p in catalogue:
        hits = [fwd[i] for i in found[p.name] if not math.isnan(fwd[i])]
        if not hits:
            ev.results.append(PatternResult(p.name, p.direction, 0, 0.0, base_mean, 0.0, 1.0, 0.0))
            continue
        m = fmean(hits)
        # Two-sided t against the unconditional mean, normal approximation.
        t = (m - base_mean) / (base_sd / math.sqrt(len(hits)))
        pv = 2.0 * (1.0 - _ND.cdf(abs(t)))
        up = sum(1 for h in hits if h > base_mean) / len(hits)
        ev.results.append(
            PatternResult(
                name=p.name,
                direction=p.direction,
                n=len(hits),
                mean_forward=m,
                unconditional=base_mean,
                t_stat=t,
                p_value=pv,
                hit_rate=up if p.direction is not Direction.BEARISH else 1.0 - up,
            )
        )
    return ev


def null_check(
    candles: Sequence[Candle],
    *,
    horizon: int = 1,
    trials: int = 20,
    seed: int = 0,
    catalogue: Sequence[Pattern] = CATALOGUE,
) -> tuple[float, int]:
    """Shuffle the outcomes against the candles and re-run the whole evaluation.

    The shapes stay real and the returns are randomised, so every genuine
    relationship is destroyed while every artefact of the method survives.
    Whatever comes back significant is what this pipeline finds when there is
    nothing to find.

    Returns the mean number of false positives per trial and the worst trial.
    """
    fwd = [r for r in forward_returns(candles, horizon) if not math.isnan(r)]
    rng = random.Random(seed)
    counts: list[int] = []
    for _ in range(trials):
        shuffled = fwd[:]
        rng.shuffle(shuffled)
        padded = shuffled + [math.nan] * (len(candles) - len(shuffled))
        ev = evaluate(candles, horizon=horizon, catalogue=catalogue, returns=padded)
        counts.append(len(ev.significant))
    return fmean(counts), max(counts)
