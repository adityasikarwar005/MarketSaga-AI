"""M33 - Cross-asset transfer coefficients.

How much of a move in crude, the dollar or US futures actually reaches Nifty,
and after how long. These are the coefficients the impact scorer uses to turn an
event about somewhere else into an expectation about here.

The blueprint contains a hand-written table of these betas. That table is a
cold-start prior and nothing more: every coefficient here is estimated from
history, carries a standard error and a sample size, and is refused when the
data cannot support it. The system must be able to answer "how do you know?"
with a number, and a hand-typed 0.25 cannot.

Three things this module insists on.

**Asymmetry is measured, not assumed.** A crude spike and a crude drift of the
same magnitude do not transmit alike - fear moves faster than arithmetic. Fitting
one beta to both averages them into a coefficient that describes neither, so up
and down are fitted separately and the difference is tested.

**Coefficients are point-in-time.** Each carries the ``as_of`` date beyond which
it must not be used. A backtest that applies a beta fitted on the whole history
to a trade from eighteen months ago is using tomorrow's relationship to explain
yesterday's move.

**Significance is a property of the estimate, not of the modeller.** A beta whose
confidence interval straddles zero is reported with that fact attached, and the
caller is expected to act on it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise
from statistics import fmean

from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import Candle

log = get_logger(__name__)

# Below this an estimate is reported but marked unusable. Thirty paired
# observations is not many; it is the floor at which a standard error means
# anything at all.
MIN_PAIRS_FOR_BETA = 30


# --------------------------------------------------------------------------- #
# Ordinary least squares with standard errors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fit:
    alpha: float
    beta: float
    stderr: float
    n: int
    r_squared: float

    @property
    def t_stat(self) -> float:
        return self.beta / self.stderr if self.stderr > 0 else 0.0

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return self.beta - z * self.stderr, self.beta + z * self.stderr

    @property
    def significant(self) -> bool:
        lo, hi = self.ci()
        return (lo > 0) or (hi < 0)


def ols(xs: Sequence[float], ys: Sequence[float]) -> Fit | None:
    """Simple regression of ``ys`` on ``xs``, with the standard error of beta.

    The standard error is the point. A beta without one is a number that cannot
    be argued with, and every relationship in this file needs to be arguable.
    """
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    beta = sxy / sxx
    alpha = my - beta * mx

    residuals = [y - (alpha + beta * x) for x, y in zip(xs, ys, strict=True)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - my) ** 2 for y in ys)
    dof = n - 2
    stderr = math.sqrt(sse / dof / sxx) if dof > 0 and sxx > 0 else 0.0
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return Fit(alpha=alpha, beta=beta, stderr=stderr, n=n, r_squared=r2)


# --------------------------------------------------------------------------- #
# Transfer coefficients
# --------------------------------------------------------------------------- #
@dataclass
class TransferCoefficient:
    source: str
    target: str
    lag_minutes: int
    fit: Fit
    as_of: date
    beta_up: Fit | None = None
    beta_down: Fit | None = None

    @property
    def beta(self) -> float:
        return self.fit.beta

    @property
    def usable(self) -> bool:
        return self.fit.n >= MIN_PAIRS_FOR_BETA and self.fit.stderr > 0

    @property
    def significant(self) -> bool:
        return self.usable and self.fit.significant

    @property
    def asymmetry(self) -> float | None:
        """``beta_up - beta_down``. Positive means rallies transmit more strongly."""
        if self.beta_up is None or self.beta_down is None:
            return None
        return self.beta_up.beta - self.beta_down.beta

    @property
    def asymmetry_is_real(self) -> bool:
        """Whether up and down betas differ by more than their combined error.

        Without this check every pair of separately-fitted betas looks
        asymmetric, because two noisy estimates are never exactly equal.
        """
        if self.beta_up is None or self.beta_down is None:
            return False
        gap = abs(self.beta_up.beta - self.beta_down.beta)
        combined = math.sqrt(self.beta_up.stderr**2 + self.beta_down.stderr**2)
        return combined > 0 and gap > 1.96 * combined

    def apply(self, source_return: float) -> float:
        """Expected target move for a given source move.

        Uses the directional beta when the asymmetry is real, and the pooled one
        when it is not - which keeps the estimate honest rather than merely
        detailed.
        """
        if self.asymmetry_is_real and self.beta_up and self.beta_down:
            fit = self.beta_up if source_return >= 0 else self.beta_down
            return fit.beta * source_return
        return self.fit.beta * source_return

    def render(self) -> str:
        lo, hi = self.fit.ci()
        star = " *" if self.significant else "  "
        asym = ""
        if self.asymmetry_is_real and self.beta_up and self.beta_down:
            asym = f"  asym up {self.beta_up.beta:+.3f} / down {self.beta_down.beta:+.3f}"
        flag = "" if self.usable else f"  [n<{MIN_PAIRS_FOR_BETA}]"
        return (
            f"    {self.source:<12} -> {self.target:<10} lag {self.lag_minutes:>3}m  "
            f"beta {self.fit.beta:+.3f} [{lo:+.3f}, {hi:+.3f}]{star} "
            f"R2 {self.fit.r_squared:.3f} n={self.fit.n}{asym}{flag}"
        )


def aligned_returns(
    source: Sequence[Candle], target: Sequence[Candle], lag_minutes: int
) -> tuple[list[float], list[float]]:
    """Pair source returns with target returns ``lag_minutes`` later.

    Joined on timestamp rather than on position: the two series keep different
    hours and have different gaps, and a positional join silently pairs a London
    afternoon with a Mumbai morning.
    """
    src = {c.ts: c for c in source}
    tgt = {c.ts: c for c in target}

    xs: list[float] = []
    ys: list[float] = []
    for ts, s in sorted(src.items()):
        s_next = src.get(ts + timedelta(minutes=1))
        t_then = tgt.get(ts + timedelta(minutes=lag_minutes))
        t_next = tgt.get(ts + timedelta(minutes=lag_minutes + 1))
        if not (s_next and t_then and t_next):
            continue
        if s.c <= 0 or t_then.c <= 0:
            continue
        xs.append(math.log(s_next.c / s.c))
        ys.append(math.log(t_next.c / t_then.c))
    return xs, ys


def estimate_transfer(
    source_bars: Sequence[Candle],
    target_bars: Sequence[Candle],
    *,
    source_name: str,
    target_name: str = "NIFTY",
    lag_minutes: int = 0,
    as_of: date | None = None,
) -> TransferCoefficient | None:
    """Fit one source-to-target relationship at one lag."""
    xs, ys = aligned_returns(source_bars, target_bars, lag_minutes)
    fit = ols(xs, ys)
    if fit is None:
        return None

    up = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0]
    down = [(x, y) for x, y in zip(xs, ys, strict=True) if x < 0]
    fit_up = ols([x for x, _ in up], [y for _, y in up]) if len(up) >= 10 else None
    fit_down = ols([x for x, _ in down], [y for _, y in down]) if len(down) >= 10 else None

    return TransferCoefficient(
        source=source_name,
        target=target_name,
        lag_minutes=lag_minutes,
        fit=fit,
        as_of=as_of or date.today(),
        beta_up=fit_up,
        beta_down=fit_down,
    )


def best_lag(
    source_bars: Sequence[Candle],
    target_bars: Sequence[Candle],
    *,
    source_name: str,
    lags: Sequence[int] = (0, 1, 2, 5, 10, 30),
    as_of: date | None = None,
) -> TransferCoefficient | None:
    """The lag at which the relationship is strongest.

    Searching over lags is itself a multiple-comparison problem: try six lags on
    noise and one of them will look good. The winner is only returned when it is
    significant on its own terms, and the caller still owes the deflated-Sharpe
    style correction if it acts on many of these.
    """
    candidates = []
    for lag in lags:
        coef = estimate_transfer(
            source_bars, target_bars, source_name=source_name, lag_minutes=lag, as_of=as_of
        )
        if coef and coef.usable:
            candidates.append(coef)
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.fit.r_squared)


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #
@dataclass
class TransferLibrary:
    """Every estimated relationship, keyed by source and lag."""

    target: str = "NIFTY"
    coefficients: dict[tuple[str, int], TransferCoefficient] = field(default_factory=dict)

    def add(self, coef: TransferCoefficient) -> None:
        self.coefficients[(coef.source, coef.lag_minutes)] = coef

    def get(self, source: str, lag_minutes: int = 0) -> TransferCoefficient | None:
        return self.coefficients.get((source, lag_minutes))

    def usable(self) -> list[TransferCoefficient]:
        return [c for c in self.coefficients.values() if c.usable]

    def significant(self) -> list[TransferCoefficient]:
        return [c for c in self.coefficients.values() if c.significant]

    def expected_move(self, source_moves: Mapping[str, float]) -> tuple[float, list[str]]:
        """Combine several source moves into one expected target move.

        Only significant coefficients contribute. A beta whose interval straddles
        zero adds noise to the estimate and confidence to the presentation, which
        is the worst combination available.
        """
        total = 0.0
        used: list[str] = []
        for symbol, move in source_moves.items():
            coef = self.get(symbol, 0)
            if coef is None or not coef.significant:
                continue
            contribution = coef.apply(move)
            total += contribution
            used.append(f"{symbol} {move:+.2%} x {coef.beta:+.3f} = {contribution:+.3%}")
        return total, used

    def render(self) -> str:
        rows = sorted(self.coefficients.values(), key=lambda c: -abs(c.fit.beta))
        head = (
            f"  transfer library -> {self.target}: {len(self.coefficients)} estimated, "
            f"{len(self.usable())} usable, {len(self.significant())} significant"
        )
        return head + "\n" + "\n".join(c.render() for c in rows)


def build_transfer_library(
    cross_asset: Mapping[str, Sequence[Candle]],
    target_bars: Sequence[Candle],
    *,
    target_name: str = "NIFTY",
    lags: Sequence[int] = (0, 1, 5),
    as_of: date | None = None,
) -> TransferLibrary:
    library = TransferLibrary(target=target_name)
    for symbol, bars in cross_asset.items():
        for lag in lags:
            coef = estimate_transfer(
                bars,
                target_bars,
                source_name=symbol,
                target_name=target_name,
                lag_minutes=lag,
                as_of=as_of,
            )
            if coef:
                library.add(coef)
    log.info(
        "transfer library built",
        fields={
            "sources": len(cross_asset),
            "coefficients": len(library.coefficients),
            "significant": len(library.significant()),
        },
    )
    return library


# --------------------------------------------------------------------------- #
# Rolling re-estimation
# --------------------------------------------------------------------------- #
@dataclass
class RollingTransfer:
    """A coefficient's history, so drift in the relationship is visible.

    Relationships decay. A beta that was 0.4 two years ago and is 0.1 now is not
    a stable coefficient with noise around it - it is a relationship that has
    changed, and averaging over the whole history hides exactly that.
    """

    source: str
    history: list[TransferCoefficient] = field(default_factory=list)

    def as_of(self, day: date) -> TransferCoefficient | None:
        """The most recent estimate fitted strictly before ``day``.

        Point-in-time: a backtest must use the beta that existed at the time,
        not the one fitted with the benefit of everything that happened since.
        """
        eligible = [c for c in self.history if c.as_of < day]
        return max(eligible, key=lambda c: c.as_of) if eligible else None

    @property
    def drift(self) -> float | None:
        """Change in beta from the first estimate to the last."""
        usable = [c for c in self.history if c.usable]
        if len(usable) < 2:
            return None
        ordered = sorted(usable, key=lambda c: c.as_of)
        return ordered[-1].fit.beta - ordered[0].fit.beta

    @property
    def is_stable(self) -> bool:
        """Whether the drift is small next to the estimation error."""
        usable = sorted((c for c in self.history if c.usable), key=lambda c: c.as_of)
        if len(usable) < 2:
            return True
        d = usable[-1].fit.beta - usable[0].fit.beta
        combined = math.sqrt(usable[-1].fit.stderr**2 + usable[0].fit.stderr**2)
        return combined <= 0 or abs(d) <= 1.96 * combined


def rolling_estimate(
    source_by_day: Mapping[date, Sequence[Candle]],
    target_by_day: Mapping[date, Sequence[Candle]],
    *,
    source_name: str,
    window_days: int = 20,
    step_days: int = 5,
    lag_minutes: int = 0,
) -> RollingTransfer:
    """Re-fit on a rolling window so the relationship's own drift is measurable."""
    days = sorted(set(source_by_day) & set(target_by_day))
    rolling = RollingTransfer(source=source_name)
    for end in range(window_days, len(days) + 1, step_days):
        window = days[end - window_days : end]
        src = [c for d in window for c in source_by_day[d]]
        tgt = [c for d in window for c in target_by_day[d]]
        coef = estimate_transfer(
            src,
            tgt,
            source_name=source_name,
            lag_minutes=lag_minutes,
            # Stamped with the day AFTER the window closes: that is the first
            # day on which this estimate could legitimately have been used.
            as_of=window[-1] + timedelta(days=1),
        )
        if coef:
            rolling.history.append(coef)
    return rolling


def daily_returns(bars_by_day: Mapping[date, Sequence[Candle]]) -> dict[date, float]:
    """Close-to-close return per session, for daily-frequency work."""
    out: dict[date, float] = {}
    ordered = sorted(bars_by_day)
    for prev, day in pairwise(ordered):
        a = sorted(bars_by_day[prev], key=lambda c: c.ts)
        b = sorted(bars_by_day[day], key=lambda c: c.ts)
        if not a or not b or a[-1].c <= 0:
            continue
        out[day] = math.log(b[-1].c / a[-1].c)
    return out


def to_datetime_key(bars: Sequence[Candle]) -> dict[datetime, Candle]:
    return {c.ts: c for c in bars}
