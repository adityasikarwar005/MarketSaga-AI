"""M75 - position sizing.

The module that decides how much, given that something else has already decided
whether. It is the last place before the risk gate where a good idea can still
be turned into a ruinous one.

Kelly, and why it is used at a quarter strength
-----------------------------------------------

The Kelly fraction maximises the long-run growth rate of capital, and its
derivation assumes you **know** the probability. You do not. You have an
estimate from a model fitted on a few thousand rows, and Kelly is savagely
sensitive to that estimate: the growth-rate curve is roughly flat just below the
optimum and falls off a cliff just above it, so overestimating the edge is far
more expensive than underestimating it. Betting full Kelly on an estimated
probability is not aggressive, it is a different and much worse bet than the one
the formula describes.

Two defences, and they are separate:

**Fractional Kelly** (``kelly_fraction``, 0.25 from config) scales the whole
answer down. It costs a quarter of the theoretical growth rate and removes most
of the variance, which is the trade every practitioner takes.

**A conservative probability** matters more, and is the one usually skipped.
``KellyInput.p_lower`` takes the lower end of the model's own uncertainty rather
than its point estimate. A model saying "0.58, plus or minus 0.06" is not the
same bet as one saying "0.58" flatly, and sizing them identically throws away
the only thing the interval was for. When no interval is supplied the point
estimate is used and ``uncertain`` is False, so the output says which happened.

Odds, not just probability
--------------------------

Kelly needs the payoff, not only the hit rate. ``f* = (p*b - q) / b`` where
``b`` is the win/loss ratio. For options this is not a detail: a debit spread
risking 85 to make 115 and a naked long risking 100 to make 400 have completely
different optimal sizes at the same probability, and code that assumes even
money silently sizes both as though they were the first.

Lots, and rounding down
-----------------------

Sizes are in **lots**. A fractional lot is not a thing that can be traded, and
rounding to nearest would let a 0.6-lot answer become a 1-lot position that is
67% larger than intended - through a cap, not up against it. Rounding down is
the only direction that cannot breach a limit.

The consequence is that a small account sizes to **zero lots** on a normal
signal, and that is reported rather than swallowed. "Your account is too small
for this trade at your risk limit" is information; a silent skip is a bug that
looks like a quiet day.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from libs.ms_core.config import RiskLimits
from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Below this the modifier stack has effectively vetoed the trade, and returning
# a token position is worse than returning none: it pays four orders of costs to
# express an opinion the sizing logic just said it did not hold.
MIN_MEANINGFUL_FRACTION = 1e-4


@dataclass(frozen=True)
class KellyInput:
    """Everything the formula needs, including how much to distrust it."""

    p_win: float
    win_amount: float  # rupees gained per lot if right
    loss_amount: float  # rupees lost per lot if wrong, positive
    p_lower: float | None = None  # lower end of the model's own interval

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_win <= 1.0:
            raise ValueError("p_win must be a probability")
        if self.win_amount <= 0:
            raise ValueError("win_amount must be positive")
        if self.loss_amount <= 0:
            raise ValueError(
                "loss_amount must be positive and finite - an unbounded loss has "
                "no Kelly fraction, and a position with one should not be sized "
                "by this function at all"
            )
        if self.p_lower is not None and not 0.0 <= self.p_lower <= self.p_win:
            raise ValueError("p_lower must be a probability no greater than p_win")

    @property
    def odds(self) -> float:
        """Win/loss ratio - the ``b`` in the formula."""
        return self.win_amount / self.loss_amount

    @property
    def p_used(self) -> float:
        """The conservative end of the estimate when one was supplied."""
        return self.p_lower if self.p_lower is not None else self.p_win

    @property
    def uncertain(self) -> bool:
        return self.p_lower is not None

    @property
    def expected_value(self) -> float:
        """Per lot, at the probability actually used."""
        p = self.p_used
        return p * self.win_amount - (1 - p) * self.loss_amount


def kelly_fraction(inputs: KellyInput) -> float:
    """``f* = (p*b - q) / b``, floored at zero.

    Zero rather than negative when the edge is absent. A negative Kelly means
    "take the other side", and acting on that here would silently invert a
    signal the direction model never gave - the caller asked how much, not
    which way.
    """
    p = inputs.p_used
    b = inputs.odds
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


# --------------------------------------------------------------------------- #
# Regime-conditional interval widening (ADR 0001)
# --------------------------------------------------------------------------- #
# M54's quantile heads are miscalibrated, and not by one factor. Mondrian
# conformal fitted on real NIFTY bars returns x0.803 in the quiet tercile,
# x1.035 in the middle and x1.188 in the busy one: the interval is too WIDE
# when the market is calm and too NARROW when it is not. `p_lower` feeds the
# Kelly formula and `uncertainty_modifier` shrinks with the width, so a wrong
# width is wrong through both channels at once.
#
# Only the widening half is applied. The clamp is not conservatism for its own
# sake - it closes a specific failure mode. Bucket assignment uses TRAILING
# volatility, which is a lagging estimate: the first day of a volatility shift
# still carries the previous regime's label. Honouring a scale below one would
# mean narrowing the interval, and therefore sizing UP, on precisely the day
# the regime turned and the label had not caught up yet. The correction would
# be largest exactly where it is most wrong.
#
# The cost is real and is stated in the ADR: positions stay smaller than they
# need to be on genuinely quiet days. That costs opportunity, shows up in no
# P&L, and is the half that is safe to get wrong.

WIDENING_FLOOR = 1.0


@dataclass(frozen=True)
class RegimeWidening:
    """What the conformal scale did to the interval, and whether it was clamped."""

    raw_scale: float
    applied_scale: float

    @property
    def clamped(self) -> bool:
        return self.applied_scale != self.raw_scale

    @property
    def reason(self) -> str:
        if self.clamped:
            return (
                f"conformal scale x{self.raw_scale:.3f} would narrow the interval; "
                f"clamped to x{self.applied_scale:.3f} because the regime label lags"
            )
        return f"interval widened x{self.applied_scale:.3f} for this regime"


def widen_for_regime(inputs: KellyInput, scale: float) -> tuple[KellyInput, RegimeWidening]:
    """Re-state the interval at its conformally-corrected width.

    This is deliberately NOT a ``Modifier``. A modifier multiplies the finished
    size and is forbidden from exceeding one, which is what keeps any single
    code path from becoming a leverage knob. Widening an interval is a
    different operation: it restates the model's own uncertainty before the
    formula sees it, and it reaches the size through ``p_lower`` and
    ``uncertainty_modifier`` together. Expressing it as a modifier would apply
    it once and hide the second channel.

    Returns the input unchanged when there is no interval to widen - a caller
    with no ``p_lower`` has not claimed an uncertainty for this to correct.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"conformal scale must be finite and positive, got {scale!r}")
    applied = max(WIDENING_FLOOR, scale)
    if inputs.p_lower is None:
        return inputs, RegimeWidening(raw_scale=scale, applied_scale=applied)
    half_width = (inputs.p_win - inputs.p_lower) * applied
    widened = replace(inputs, p_lower=max(0.0, inputs.p_win - half_width))
    return widened, RegimeWidening(raw_scale=scale, applied_scale=applied)


# --------------------------------------------------------------------------- #
# Modifiers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Modifier:
    """One named multiplier, with the reason attached.

    Named because a size that came out small needs to say *why*. A single
    combined number is unarguable, which in a sizing path is a defect.
    """

    name: str
    value: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"modifier {self.name!r}={self.value} is outside [0, 1] - "
                "modifiers may only reduce size. A multiplier above one turns "
                "a risk control into a leverage knob."
            )


def uncertainty_modifier(p_win: float, p_lower: float | None) -> Modifier:
    """Shrink when the model's own interval is wide.

    Distinct from using ``p_lower`` in the formula, and both are wanted: the
    lower bound changes *which bet* is being sized, this changes *how much* of
    it to take. A model that is 0.60 give or take 0.02 and one that is 0.60 give
    or take 0.15 are not equally worth acting on even after the conservative
    substitution.
    """
    if p_lower is None:
        return Modifier(
            "uncertainty",
            0.5,
            "no interval supplied - halved, because an unquantified estimate is "
            "not the same as a confident one",
        )
    width = max(0.0, p_win - p_lower)
    # Full size at a tight interval, decaying to a third at 0.15 wide.
    value = max(1.0 / 3.0, 1.0 - width / 0.225)
    return Modifier("uncertainty", value, f"interval half-width {width:.3f}")


def loss_streak_modifier(consecutive_losses: int, *, halve_after: int = 3) -> Modifier:
    """Halve after a run of losses, and again after twice as many.

    Not because losses predict losses - they mostly do not - but because a run
    is the cheapest available evidence that the model's current estimate of its
    own edge is wrong. Reducing size is the response that costs least if the run
    was luck and saves most if it was not.
    """
    if consecutive_losses < halve_after:
        return Modifier("loss_streak", 1.0, f"{consecutive_losses} consecutive losses")
    steps = consecutive_losses // halve_after
    return Modifier(
        "loss_streak",
        max(0.125, 0.5**steps),
        f"{consecutive_losses} consecutive losses - halved {steps}x",
    )


def expiry_modifier(is_expiry_day: bool, limits: RiskLimits) -> Modifier:
    """Gamma on expiry day is not a bigger version of gamma on any other day."""
    if not is_expiry_day:
        return Modifier("expiry", 1.0, "not expiry day")
    return Modifier(
        "expiry",
        limits.expiry_day_size_multiplier,
        "expiry day - gamma risk is extreme and the position can invert in minutes",
    )


def time_of_day_modifier(minutes_from_open: int, minutes_to_close: int) -> Modifier:
    """The first and last minutes of a session are a different market."""
    if minutes_from_open < 15:
        return Modifier("time_of_day", 0.5, "opening auction noise still resolving")
    if minutes_to_close < 30:
        return Modifier("time_of_day", 0.5, "closing - less time for the thesis to work")
    return Modifier("time_of_day", 1.0, "mid-session")


def correlation_modifier(
    open_positions: Sequence[str],
    proposed: str,
    correlations: Mapping[tuple[str, str], float] | None = None,
    *,
    default_correlation: float = 0.7,
) -> Modifier:
    """Shrink for what is already on the book.

    Sizing every position independently is the most expensive mistake available
    here. Five positions each risking 1.5% of capital, all long the same index
    through different strikes, is not five 1.5% bets - it is one 7.5% bet
    wearing a diversified costume, and it is discovered on the day they all move
    together.

    The default correlation between two index-derivative positions is 0.7, not
    0. Assuming independence is the wrong prior for a book that is, by
    construction, all on one underlying.
    """
    if not open_positions:
        return Modifier("correlation", 1.0, "no open positions")

    def rho(a: str, b: str) -> float:
        if correlations is None:
            return default_correlation
        return correlations.get((a, b), correlations.get((b, a), default_correlation))

    total = sum(abs(rho(proposed, held)) for held in open_positions)
    # Effective count of independent positions this would be the (n+1)-th of.
    value = 1.0 / (1.0 + total)
    return Modifier(
        "correlation",
        value,
        f"{len(open_positions)} open, summed correlation {total:.2f}",
    )


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
@dataclass
class SizingDecision:
    """How many lots, and every reason it is not more."""

    lots: int
    capital: float
    lot_size: int
    loss_per_lot: float
    raw_kelly: float
    fractional_kelly: float
    modified_fraction: float
    capped_fraction: float
    cap_reason: str
    modifiers: list[Modifier] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def capital_at_risk(self) -> float:
        return self.lots * self.loss_per_lot

    @property
    def capital_at_risk_pct(self) -> float:
        return self.capital_at_risk / self.capital if self.capital > 0 else 0.0

    @property
    def rejected(self) -> bool:
        return self.lots == 0

    @property
    def combined_modifier(self) -> float:
        value = 1.0
        for m in self.modifiers:
            value *= m.value
        return value

    def render(self) -> str:
        lines = [
            f"  {self.lots} lot(s) = {self.capital_at_risk:,.0f} at risk "
            f"({self.capital_at_risk_pct:.2%} of {self.capital:,.0f})",
            f"    Kelly {self.raw_kelly:.4f} -> fractional {self.fractional_kelly:.4f} "
            f"-> modified {self.modified_fraction:.4f} -> capped {self.capped_fraction:.4f}",
            f"    binding constraint: {self.cap_reason}",
        ]
        lines += [f"    x{m.value:.3f} {m.name}: {m.reason}" for m in self.modifiers]
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def size_position(
    inputs: KellyInput,
    *,
    capital: float,
    lot_size: int,
    limits: RiskLimits,
    modifiers: Sequence[Modifier] = (),
    open_positions: int = 0,
) -> SizingDecision:
    """Kelly, scaled, modified, capped, and floored to whole lots.

    The order is deliberate and each step can only reduce: Kelly gives an
    optimum for a bet nobody actually has, the fraction accounts for that, the
    modifiers account for context, the cap is the limit that must not be
    breached regardless, and the floor to whole lots is the only rounding that
    cannot undo any of the above.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")

    notes: list[str] = []
    mods = list(modifiers)

    raw = kelly_fraction(inputs)
    if raw <= 0:
        notes.append(
            f"no Kelly edge at p={inputs.p_used:.3f} and odds {inputs.odds:.2f} - "
            "the expected value is not positive"
        )
    if not inputs.uncertain:
        notes.append("sized on a point estimate with no interval - the least reliable case")

    fractional = raw * limits.kelly_fraction
    combined = 1.0
    for m in mods:
        combined *= m.value
    modified = fractional * combined

    hard_cap = limits.per_trade_capital_at_risk_pct / 100.0
    if modified > hard_cap:
        capped, cap_reason = hard_cap, f"per-trade cap {limits.per_trade_capital_at_risk_pct}%"
    elif raw <= 0:
        capped, cap_reason = 0.0, "no edge"
    else:
        capped, cap_reason = modified, "Kelly after modifiers"

    if 0 < capped < MIN_MEANINGFUL_FRACTION:
        notes.append(
            f"fraction {capped:.6f} is below the meaningful floor - a token "
            "position pays four orders of costs to express an opinion the "
            "sizing just declined to hold"
        )
        capped = 0.0
        cap_reason = "below the meaningful floor"

    loss_per_lot = inputs.loss_amount
    affordable = (capped * capital) / loss_per_lot if loss_per_lot > 0 else 0.0
    lots = math.floor(affordable)

    if lots == 0 and capped > 0:
        budget = capped * capital
        # Which of the two produced the zero matters, and the naive message gets
        # it wrong. "Your account is too small" is right when the cap is binding
        # and the account cannot afford one lot at it. When the modifier stack
        # is what shrank the budget, the account is fine and the *context* is
        # what declined - and telling a user to fund the account would be advice
        # to override a risk control.
        if cap_reason.startswith("per-trade cap"):
            notes.append(
                f"{capped:.4%} of {capital:,.0f} is {budget:,.0f}, and one lot risks "
                f"{loss_per_lot:,.0f} - the account is too small for this trade at "
                "this risk limit"
            )
        else:
            shrunk = combined
            notes.append(
                f"the modifier stack cut the size to {shrunk:.1%} of fractional Kelly, "
                f"leaving a budget of {budget:,.0f} against {loss_per_lot:,.0f} per lot - "
                "the account is adequate, the context is not"
            )
    if open_positions >= limits.max_concurrent_positions:
        lots = 0
        cap_reason = f"already at {open_positions} concurrent positions"
        notes.append("the position limit is a veto, not a modifier")

    decision = SizingDecision(
        lots=lots,
        capital=capital,
        lot_size=lot_size,
        loss_per_lot=loss_per_lot,
        raw_kelly=raw,
        fractional_kelly=fractional,
        modified_fraction=modified,
        capped_fraction=capped,
        cap_reason=cap_reason,
        modifiers=mods,
        notes=notes,
    )
    log.info(
        "position sized",
        fields={
            "lots": lots,
            "capital_at_risk_pct": round(decision.capital_at_risk_pct, 5),
            "raw_kelly": round(raw, 5),
            "cap_reason": cap_reason,
        },
    )
    return decision


def standard_modifiers(
    inputs: KellyInput,
    *,
    limits: RiskLimits,
    is_expiry_day: bool = False,
    minutes_from_open: int = 60,
    minutes_to_close: int = 60,
    consecutive_losses: int = 0,
    open_symbols: Sequence[str] = (),
    proposed_symbol: str = "",
    correlations: Mapping[tuple[str, str], float] | None = None,
) -> list[Modifier]:
    """The full stack, in one call, so none of them is quietly forgotten.

    Every one of these can only shrink the position. That is enforced by
    ``Modifier`` rather than trusted, because a multiplier above one turns a
    risk control into a leverage knob and it would be an easy thing to add by
    accident.
    """
    return [
        uncertainty_modifier(inputs.p_win, inputs.p_lower),
        expiry_modifier(is_expiry_day, limits),
        time_of_day_modifier(minutes_from_open, minutes_to_close),
        loss_streak_modifier(consecutive_losses),
        correlation_modifier(open_symbols, proposed_symbol, correlations),
    ]


# --------------------------------------------------------------------------- #
# Book-level check
# --------------------------------------------------------------------------- #
@dataclass
class BookRisk:
    """What the whole book risks, which is not the sum of what each leg risks."""

    positions: dict[str, float]  # symbol -> capital at risk
    correlations: Mapping[tuple[str, str], float] | None = None
    default_correlation: float = 0.7

    def _rho(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if self.correlations is None:
            return self.default_correlation
        return self.correlations.get(
            (a, b), self.correlations.get((b, a), self.default_correlation)
        )

    @property
    def naive_total(self) -> float:
        """What a per-position limit believes the book risks."""
        return sum(self.positions.values())

    @property
    def correlated_total(self) -> float:
        """What it actually risks when the positions move together.

        ``sqrt(sum_ij rho_ij * r_i * r_j)``. With every correlation at 1 this
        equals the naive sum; with genuine diversification it is smaller. On a
        book that is all one underlying it will sit very close to the naive
        sum - which is the point: the per-position limit was never buying the
        diversification it appeared to.
        """
        symbols = list(self.positions)
        total = 0.0
        for a in symbols:
            for b in symbols:
                total += self._rho(a, b) * self.positions[a] * self.positions[b]
        return math.sqrt(max(0.0, total))

    @property
    def diversification_benefit(self) -> float:
        """1.0 means none at all. Expect it to be close to 1 on one underlying."""
        return self.correlated_total / self.naive_total if self.naive_total > 0 else 0.0

    def render(self) -> str:
        return (
            f"  {len(self.positions)} positions risking {self.naive_total:,.0f} "
            f"by the per-position view\n"
            f"    correlated exposure {self.correlated_total:,.0f} "
            f"({self.diversification_benefit:.0%} of the naive sum)\n"
            f"    a book on one underlying is one bet wearing a diversified costume"
        )
