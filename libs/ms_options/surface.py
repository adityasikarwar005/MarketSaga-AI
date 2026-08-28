"""M70 - the implied volatility surface.

Everything downstream prices off this: the Greeks engine, the structure
enumerator, the Monte Carlo P&L. A surface that is subtly wrong does not produce
an error anywhere - it produces a book that looks correctly hedged.

Three decisions do most of the work, and each one has a plausible-looking
alternative that fails silently.

Mid, never last-traded
-----------------------

An LTP is a print from whenever the last trade happened, which on a quiet strike
is minutes ago and on the wrong side of a spread that has since moved. A surface
built from LTPs is built from prices **that were never simultaneously
available**, so it is not a snapshot of anything - and it fits beautifully,
because stale prices are smooth.

:func:`iv_from_quote` reads ``quote.mid`` and there is no fallback. A quote with
no two-sided market is dropped, and the drop is counted.

The forward comes from the chain, not from the spot
----------------------------------------------------

``F = S·e^(rT)`` is the textbook forward and it is wrong for an index, by the
dividend yield you do not know. Using it puts a systematic wedge between the
call-implied and put-implied vols at the same strike, which then looks like
skew - a real, stable, entirely manufactured skew that a model will happily
learn.

:func:`synthetic_forward` recovers the forward the market is actually using, from
put-call parity on the most liquid strikes. It is the one number the whole
surface hangs on.

No-arbitrage is enforced twice, and that is not redundant
-----------------------------------------------------------

An SVI fitted with a penalty term for arbitrage will, whenever the penalty is
outweighed by fit error, hand back a surface with excellent RMSE that implies a
**negative probability density**. Every option price on it is positive. Every
individual number is plausible. The butterfly spread it prices is negative.

So the penalty steers the optimiser, and :meth:`Surface.arbitrage_report` checks
the result independently. A violation that survives the penalty is refused, not
served. Checking only inside the objective is the version of this that looks
identical right up until someone trades on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import pairwise

from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import ChainSnapshot, OptionQuote
from libs.ms_options.pricing import OptType, black76_greeks, implied_vol

log = get_logger(__name__)

# --- quote filters --------------------------------------------------------- #
# Deltas outside this band are almost all bid-ask noise: a 2-delta option worth
# 0.35 with a 0.30 spread carries an implied vol that moves twenty points when
# the quote moves one tick, and fitting to it drags the wings around.
MIN_ABS_DELTA = 0.05
MAX_ABS_DELTA = 0.95
MAX_SPREAD_PCT = 0.20  # of mid
MIN_OI = 100

# --- staleness ------------------------------------------------------------- #
# A surface is a photograph. Serving one from five minutes ago during a move is
# how a position gets sized against a volatility that no longer exists.
MAX_SURFACE_AGE = timedelta(seconds=90)

# Fitting fewer than this many quotes fits noise. Five parameters need more than
# five points to mean anything, and a smile needs points on both wings.
MIN_QUOTES_PER_EXPIRY = 8


class RejectReason(StrEnum):
    NO_TWO_SIDED_MARKET = "no_two_sided_market"
    SPREAD_TOO_WIDE = "spread_too_wide"
    OI_TOO_LOW = "oi_too_low"
    DELTA_OUT_OF_BAND = "delta_out_of_band"
    IV_DID_NOT_SOLVE = "iv_did_not_solve"
    EXPIRED = "expired"


class SurfaceError(RuntimeError):
    pass


class StaleSurfaceError(SurfaceError):
    """Asked for a surface older than the staleness bound."""


class ArbitrageError(SurfaceError):
    """The fitted surface admits arbitrage and will not be served."""


# --------------------------------------------------------------------------- #
# The forward
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForwardEstimate:
    forward: float
    strike_used: float
    n_candidates: int
    method: str

    def render(self) -> str:
        return (
            f"    forward {self.forward:.2f} via {self.method} at strike "
            f"{self.strike_used:.0f} ({self.n_candidates} candidate strikes)"
        )


def synthetic_forward(
    quotes: Sequence[OptionQuote], *, spot: float, r: float, t: float
) -> ForwardEstimate:
    """The forward the market is using, from put-call parity.

    ``C - P = e^(-rT)(F - K)`` rearranges to ``F = K + e^(rT)(C - P)``, which is
    the forward implied by the options themselves - dividends, repo and all,
    without needing to know any of them.

    The strike closest to the money is used, because parity is a statement about
    every strike but only the near-the-money ones have two liquid sides. Deep
    strikes give the same equation with the difference of two numbers dominated
    by the spread of the illiquid leg.

    Falls back to ``S·e^(rT)`` only when no strike has both legs quoted, and says
    so in ``method`` - a caller that cannot tell which it got will not notice the
    surface acquiring a manufactured skew.
    """
    by_strike: dict[float, dict[OptType, float]] = {}
    for q in quotes:
        mid = q.mid
        if mid is None or mid <= 0:
            continue
        kind = OptType.CALL if str(q.opt_type).upper().startswith("C") else OptType.PUT
        by_strike.setdefault(q.strike, {})[kind] = mid

    both = {k: v for k, v in by_strike.items() if OptType.CALL in v and OptType.PUT in v}
    if not both:
        return ForwardEstimate(
            forward=spot * math.exp(r * t),
            strike_used=float("nan"),
            n_candidates=0,
            method="spot*exp(rT) - NO parity strike available, skew may be manufactured",
        )

    atm = min(both, key=lambda k: abs(k - spot))
    legs = both[atm]
    forward = atm + math.exp(r * t) * (legs[OptType.CALL] - legs[OptType.PUT])
    return ForwardEstimate(
        forward=forward, strike_used=atm, n_candidates=len(both), method="put-call parity"
    )


# --------------------------------------------------------------------------- #
# Quote filtering
# --------------------------------------------------------------------------- #
@dataclass
class FilterStats:
    """How many quotes survived, and exactly why the rest did not.

    The rejection *rate* is the number that matters operationally - the plan asks
    the surface report to print it. A chain that suddenly rejects 60% of its
    quotes is a feed problem, and the surface fitted from the survivors will look
    entirely normal.
    """

    kept: int = 0
    rejected: dict[RejectReason, int] = field(default_factory=dict)

    @property
    def seen(self) -> int:
        return self.kept + sum(self.rejected.values())

    @property
    def reject_rate(self) -> float:
        return (self.seen - self.kept) / self.seen if self.seen else 0.0

    def reject(self, reason: RejectReason) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def render(self) -> str:
        lines = [f"    kept {self.kept}/{self.seen} ({self.reject_rate:.1%} rejected)"]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {reason.value:<24} {n}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Observation:
    """One usable point on the surface: log-moneyness and total variance."""

    strike: float
    k: float  # log-moneyness, ln(K/F)
    iv: float
    total_variance: float  # w = iv^2 * T
    kind: OptType
    delta: float
    weight: float = 1.0


def iv_from_quote(
    quote: OptionQuote,
    *,
    forward: float,
    t: float,
    r: float,
    stats: FilterStats,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_oi: int = MIN_OI,
    min_abs_delta: float = MIN_ABS_DELTA,
    max_abs_delta: float = MAX_ABS_DELTA,
) -> Observation | None:
    """One quote to one observation, or a counted rejection.

    Reads ``quote.mid``. There is deliberately no LTP path: a surface built from
    last-traded prices is built from prices that were never simultaneously
    available, and it fits beautifully because stale prices are smooth.
    """
    if t <= 0.0:
        stats.reject(RejectReason.EXPIRED)
        return None

    mid = quote.mid  # bid > 0 and ask present, or None
    if mid is None or mid <= 0.0:
        stats.reject(RejectReason.NO_TWO_SIDED_MARKET)
        return None

    spread = quote.spread_pct_of_mid
    if spread is None or spread > max_spread_pct:
        stats.reject(RejectReason.SPREAD_TOO_WIDE)
        return None

    if quote.oi is not None and quote.oi < min_oi:
        stats.reject(RejectReason.OI_TOO_LOW)
        return None

    kind = OptType.CALL if str(quote.opt_type).upper().startswith("C") else OptType.PUT
    iv = implied_vol(mid, forward, quote.strike, t, r, kind)
    if iv is None or iv <= 0.0:
        stats.reject(RejectReason.IV_DID_NOT_SOLVE)
        return None

    delta = black76_greeks(forward, quote.strike, iv, t, r, kind).delta
    if not (min_abs_delta <= abs(delta) <= max_abs_delta):
        stats.reject(RejectReason.DELTA_OUT_OF_BAND)
        return None

    stats.kept += 1
    return Observation(
        strike=quote.strike,
        k=math.log(quote.strike / forward),
        iv=iv,
        total_variance=iv * iv * t,
        kind=kind,
        delta=delta,
        # Vega-ish weighting: near-the-money quotes carry more information about
        # the level, and weighting them equally with the wings lets two noisy
        # 5-delta points tilt the whole smile.
        weight=math.exp(-0.5 * (math.log(quote.strike / forward) ** 2) / 0.04),
    )


# --------------------------------------------------------------------------- #
# SVI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVIParams:
    """Raw SVI, in total implied variance.

        w(k) = a + b·( rho·(k - m) + sqrt((k - m)² + sigma²) )

    ``w`` is total variance (``iv² · T``), not vol - because total variance is
    what the no-arbitrage conditions are stated in, and converting at every step
    invites the one place someone forgets to.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: float) -> float:
        y = k - self.m
        return self.a + self.b * (self.rho * y + math.hypot(y, self.sigma))

    def d1(self, k: float) -> float:
        """dw/dk."""
        y = k - self.m
        return self.b * (self.rho + y / math.hypot(y, self.sigma))

    def d2(self, k: float) -> float:
        """d²w/dk²."""
        y = k - self.m
        s = math.hypot(y, self.sigma)
        return self.b * self.sigma * self.sigma / (s * s * s)

    def vol(self, k: float, t: float) -> float:
        w = self.total_variance(k)
        if w <= 0.0 or t <= 0.0:
            raise ArbitrageError(
                f"total variance {w:.6f} at k={k:.4f} is not positive; this surface "
                "cannot produce a volatility and must not be used to price anything"
            )
        return math.sqrt(w / t)

    @property
    def parameters_admissible(self) -> bool:
        """The necessary conditions for ``w(k) > 0`` everywhere.

        Necessary, not sufficient - butterfly arbitrage lives inside this region
        too, which is what :func:`durrleman_g` is for.
        """
        return (
            self.b >= 0.0
            and abs(self.rho) < 1.0
            and self.sigma > 0.0
            and self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho) >= 0.0
        )

    def render(self) -> str:
        return (
            f"a={self.a:+.6f} b={self.b:.6f} rho={self.rho:+.4f} "
            f"m={self.m:+.4f} sigma={self.sigma:.4f}"
        )


def durrleman_g(p: SVIParams, k: float) -> float:
    """Durrleman's function. ``g(k) ≥ 0`` for every k is butterfly arbitrage-free.

        g = (1 - k·w'/(2w))² - (w'²/4)·(1/w + 1/4) + w''/2

    Negative g means the risk-neutral density is negative somewhere - a butterfly
    spread with negative value. Nothing about the individual option prices looks
    wrong when this happens; they are all positive, all monotone in strike, and
    all consistent with a smile someone would call reasonable.
    """
    w = p.total_variance(k)
    if w <= 0.0:
        return -1.0
    wp = p.d1(k)
    wpp = p.d2(k)
    term = 1.0 - k * wp / (2.0 * w)
    return term * term - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def butterfly_violations(
    p: SVIParams, *, lo: float = -1.5, hi: float = 1.5, n: int = 201
) -> list[tuple[float, float]]:
    """Every grid point where ``g < 0``, with its value.

    A grid rather than an analytic minimum: g is not convex in k, and a solver
    that finds one stationary point can report a surface clean when a second
    trough elsewhere is deeply negative.
    """
    out: list[tuple[float, float]] = []
    for i in range(n):
        k = lo + (hi - lo) * i / (n - 1)
        g = durrleman_g(p, k)
        if g < 0.0:
            out.append((k, g))
    return out


def calendar_violations(
    earlier: SVIParams, later: SVIParams, *, lo: float = -1.5, hi: float = 1.5, n: int = 201
) -> list[tuple[float, float, float]]:
    """Where total variance decreases with maturity. ``(k, w_earlier, w_later)``.

    Total variance must be non-decreasing in T at every log-moneyness. Where it
    is not, a calendar spread has negative value - you are paid to hold optionality
    for longer.
    """
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        k = lo + (hi - lo) * i / (n - 1)
        w1, w2 = earlier.total_variance(k), later.total_variance(k)
        if w2 < w1 - 1e-12:
            out.append((k, w1, w2))
    return out


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
ARB_PENALTY = 1e4  # steers the optimiser; never the only line of defence


def _objective(p: SVIParams, obs: Sequence[Observation]) -> float:
    if not p.parameters_admissible:
        return float("inf")
    err = 0.0
    for o in obs:
        d = p.total_variance(o.k) - o.total_variance
        err += o.weight * d * d
    # The penalty steers; it does not decide. Whatever survives it is checked
    # again independently, because a large enough fit error can outweigh it and
    # hand back an arbitrageable surface with an excellent RMSE.
    penalty = sum(-g for _k, g in butterfly_violations(p, n=61))
    return err + ARB_PENALTY * penalty


def _nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    step: Sequence[float],
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[list[float], float]:
    """Deterministic Nelder-Mead. No scipy, and no randomness.

    Determinism is a requirement rather than a nicety: two fits of the same chain
    must produce the same surface, or a recommendation cannot be reproduced from
    its inputs.
    """
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step[i]
        simplex.append(pt)
    values = [f(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tol * (abs(values[0]) + tol):
            break

        centroid = [sum(p[i] for p in simplex[:-1]) / n for i in range(n)]
        worst = simplex[-1]

        refl = [centroid[i] + (centroid[i] - worst[i]) for i in range(n)]
        f_refl = f(refl)
        if values[0] <= f_refl < values[-2]:
            simplex[-1], values[-1] = refl, f_refl
            continue
        if f_refl < values[0]:
            exp = [centroid[i] + 2.0 * (centroid[i] - worst[i]) for i in range(n)]
            f_exp = f(exp)
            simplex[-1], values[-1] = (exp, f_exp) if f_exp < f_refl else (refl, f_refl)
            continue
        con = [centroid[i] + 0.5 * (worst[i] - centroid[i]) for i in range(n)]
        f_con = f(con)
        if f_con < values[-1]:
            simplex[-1], values[-1] = con, f_con
            continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j]) for j in range(n)]
            values[i] = f(simplex[i])

    best_i = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_i], values[best_i]


def fit_svi(obs: Sequence[Observation], *, t: float) -> SVIParams:
    """Fit one expiry.

    Seeded from the data rather than from constants: the ATM total variance sets
    ``a``, the observed spread of variance across the wings sets ``b``, and ``m``
    starts at the observed minimum. A fixed seed works on a calm day and walks
    into a local minimum on the day the smile is steep, which is the day it
    matters.
    """
    if len(obs) < MIN_QUOTES_PER_EXPIRY:
        raise SurfaceError(
            f"{len(obs)} usable quotes for this expiry, need {MIN_QUOTES_PER_EXPIRY}. "
            "Five SVI parameters fitted to fewer points than that is a curve drawn "
            "through noise, and it will look like a smile."
        )

    ws = [o.total_variance for o in obs]
    ks = [o.k for o in obs]
    atm = min(obs, key=lambda o: abs(o.k))
    w_min, w_max = min(ws), max(ws)
    k_at_min = ks[ws.index(w_min)]

    a0 = max(w_min * 0.5, 1e-8)
    b0 = max((w_max - w_min) / max(max(ks) - min(ks), 0.1), 1e-4)
    seed = (a0, b0, -0.3, k_at_min, 0.1)
    steps = (max(atm.total_variance * 0.3, 1e-4), max(b0 * 0.5, 1e-3), 0.2, 0.05, 0.05)

    best, _ = _nelder_mead(
        lambda v: _objective(SVIParams(v[0], v[1], v[2], v[3], v[4]), obs),
        seed,
        step=steps,
    )
    params = SVIParams(best[0], best[1], best[2], best[3], best[4])

    if not params.parameters_admissible:
        raise SurfaceError(
            f"the fit landed outside the admissible SVI region ({params.render()}). "
            "Serving it would produce negative total variance somewhere on the "
            "surface, and the first symptom would be a NaN in a Greek."
        )
    return params


def fit_rmse(p: SVIParams, obs: Sequence[Observation], *, t: float) -> float:
    """RMSE in **vol** points, not total variance.

    Total-variance RMSE is unreadable - it changes scale with maturity, so the
    same quality of fit reports a different number for a weekly and a monthly.
    """
    if not obs:
        return float("nan")
    total = 0.0
    for o in obs:
        w = max(p.total_variance(o.k), 1e-12)
        total += (math.sqrt(w / t) - o.iv) ** 2
    return math.sqrt(total / len(obs))


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpirySlice:
    expiry: datetime
    t: float  # year fraction
    forward: ForwardEstimate
    params: SVIParams
    n_quotes: int
    rmse: float
    stats: FilterStats

    @property
    def atm_total_variance(self) -> float:
        return self.params.total_variance(0.0)

    @property
    def level(self) -> float:
        """ATM implied vol. The feature everything else is quoted relative to."""
        return math.sqrt(max(self.atm_total_variance, 0.0) / self.t)

    @property
    def slope(self) -> float:
        """dw/dk at the money - the skew, in total-variance terms."""
        return self.params.d1(0.0)

    @property
    def curvature(self) -> float:
        """d²w/dk² at the money - the smile."""
        return self.params.d2(0.0)

    def vol(self, strike: float) -> float:
        return self.params.vol(math.log(strike / self.forward.forward), self.t)

    def render(self) -> str:
        return (
            f"    {self.expiry.date()}  T={self.t:.4f}  n={self.n_quotes:<4} "
            f"rmse={self.rmse * 100:.2f}vp  atm={self.level * 100:.2f}%  "
            f"slope={self.slope:+.5f}  curv={self.curvature:.5f}\n"
            f"      {self.params.render()}\n"
            f"{self.stats.render()}"
        )


@dataclass
class ArbitrageReport:
    butterfly: dict[datetime, list[tuple[float, float]]] = field(default_factory=dict)
    calendar: dict[tuple[datetime, datetime], list[tuple[float, float, float]]] = field(
        default_factory=dict
    )

    @property
    def clean(self) -> bool:
        return not self.butterfly and not self.calendar

    def render(self) -> str:
        if self.clean:
            return "    no-arbitrage: butterfly and calendar both clean"
        lines = ["    NO-ARBITRAGE VIOLATIONS"]
        for expiry, points in self.butterfly.items():
            worst_k, worst_g = min(points, key=lambda kg: kg[1])
            lines.append(
                f"      butterfly {expiry.date()}: g<0 at {len(points)} grid points, "
                f"worst g={worst_g:.6f} at k={worst_k:+.3f} - the implied density is "
                "negative there and a butterfly spread prices below zero"
            )
        for (e1, e2), cal_points in self.calendar.items():
            worst = min(cal_points, key=lambda p: p[2] - p[1])
            lines.append(
                f"      calendar {e1.date()} -> {e2.date()}: total variance falls at "
                f"{len(cal_points)} points, worst at k={worst[0]:+.3f} "
                f"({worst[1]:.6f} -> {worst[2]:.6f}) - a calendar spread has negative value"
            )
        return "\n".join(lines)


@dataclass
class Surface:
    """A fitted surface, and the time it was a photograph of."""

    ts: datetime
    underlying_id: int
    spot: float
    slices: dict[datetime, ExpirySlice] = field(default_factory=dict)
    max_age: timedelta = MAX_SURFACE_AGE

    def age(self, now: datetime) -> timedelta:
        return now - self.ts

    def is_stale(self, now: datetime) -> bool:
        return self.age(now) > self.max_age

    def slice_at(self, expiry: datetime, *, now: datetime) -> ExpirySlice:
        """The slice, or a refusal if the surface has gone stale.

        Refusing rather than serving with a warning: a warning gets logged, and
        the position gets sized against a volatility that stopped existing four
        minutes ago.
        """
        if self.is_stale(now):
            raise StaleSurfaceError(
                f"surface is {self.age(now).total_seconds():.0f}s old, limit is "
                f"{self.max_age.total_seconds():.0f}s. Refusing rather than serving: "
                "a stale surface prices a position against a volatility that no "
                "longer exists, and nothing downstream can tell."
            )
        if expiry not in self.slices:
            raise SurfaceError(f"no fitted slice for expiry {expiry}")
        return self.slices[expiry]

    def arbitrage_report(self) -> ArbitrageReport:
        """Check the fitted result, independently of whatever the fit believed."""
        report = ArbitrageReport()
        for expiry, sl in self.slices.items():
            violations = butterfly_violations(sl.params)
            if violations:
                report.butterfly[expiry] = violations

        ordered = sorted(self.slices.items(), key=lambda kv: kv[1].t)
        for (e1, s1), (e2, s2) in pairwise(ordered):
            cal = calendar_violations(s1.params, s2.params)
            if cal:
                report.calendar[(e1, e2)] = cal
        return report

    def require_arbitrage_free(self) -> Surface:
        report = self.arbitrage_report()
        if not report.clean:
            raise ArbitrageError(
                "the fitted surface admits arbitrage and will not be served:\n" + report.render()
            )
        return self

    def features(self, expiry: datetime, *, now: datetime) -> dict[str, float]:
        """Level, slope and curvature - the surface as three numbers.

        Parameters are stored; these are derived. Storing the derived features
        instead would make a surface impossible to re-price from, and the whole
        point of keeping parameters is that a past recommendation can be rebuilt.
        """
        sl = self.slice_at(expiry, now=now)
        return {
            "iv_level": sl.level,
            "iv_slope": sl.slope,
            "iv_curvature": sl.curvature,
            "iv_atm_total_variance": sl.atm_total_variance,
            "surface_rmse": sl.rmse,
            "surface_reject_rate": sl.stats.reject_rate,
        }

    def render(self) -> str:
        lines = [
            f"  surface for {self.underlying_id} at {self.ts.isoformat()} "
            f"(spot {self.spot:.2f}, {len(self.slices)} expiries)"
        ]
        for _expiry, sl in sorted(self.slices.items(), key=lambda kv: kv[1].t):
            lines.append(sl.render())
        lines.append(self.arbitrage_report().render())
        return "\n".join(lines)


def fit_surface(
    chain: ChainSnapshot,
    *,
    now: datetime,
    r: float = 0.065,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_oi: int = MIN_OI,
    max_age: timedelta = MAX_SURFACE_AGE,
) -> Surface:
    """Fit every expiry in a chain snapshot.

    One expiry failing does not fail the surface. A weekly with four quotes on a
    holiday-shortened week is a normal condition, and refusing the whole surface
    for it would take the monthlies down with it.
    """
    surface = Surface(
        ts=chain.ts, underlying_id=chain.underlying_id, spot=chain.spot, max_age=max_age
    )

    by_expiry: dict[datetime, list[OptionQuote]] = {}
    for q in chain.quotes:
        by_expiry.setdefault(q.expiry, []).append(q)

    for expiry, quotes in sorted(by_expiry.items()):
        t = (expiry - chain.ts).total_seconds() / (365.0 * 24 * 3600)
        if t <= 0.0:
            continue

        fwd = synthetic_forward(quotes, spot=chain.spot, r=r, t=t)
        stats = FilterStats()
        obs = [
            o
            for o in (
                iv_from_quote(
                    q,
                    forward=fwd.forward,
                    t=t,
                    r=r,
                    stats=stats,
                    max_spread_pct=max_spread_pct,
                    min_oi=min_oi,
                )
                for q in quotes
            )
            if o is not None
        ]
        try:
            params = fit_svi(obs, t=t)
        except SurfaceError as exc:
            log.info(
                "expiry not fitted",
                fields={"expiry": expiry.isoformat(), "reason": str(exc)[:120]},
            )
            continue

        surface.slices[expiry] = ExpirySlice(
            expiry=expiry,
            t=t,
            forward=fwd,
            params=params,
            n_quotes=len(obs),
            rmse=fit_rmse(params, obs, t=t),
            stats=stats,
        )
    return surface
