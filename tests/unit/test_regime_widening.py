"""ADR 0001: the conformal correction reaches sizing, and only in one direction.

M54's interval is miscalibrated by a factor that changes sign across regimes -
x0.803 quiet, x1.188 busy, measured on real NIFTY bars. Applying it whole would
narrow the interval on quiet days and size UP. These tests pin that it cannot,
and that the widening half genuinely arrives.

The failure mode the clamp exists for is specific and is worth stating as a
test rather than a comment: bucket assignment uses trailing volatility, so the
first day of a volatility shift still wears the previous regime's label. An
unclamped correction would size up hardest exactly there.
"""

from __future__ import annotations

import pytest

from libs.ms_options.sizing import (
    WIDENING_FLOOR,
    KellyInput,
    kelly_fraction,
    uncertainty_modifier,
    widen_for_regime,
)

# Measured, not chosen: jobs/sizing_by_regime.py on ten years of real bars.
QUIET, MID, BUSY = 0.803, 1.035, 1.188


def sized(inputs: KellyInput) -> float:
    """Kelly x the uncertainty modifier - both channels the width reaches."""
    return kelly_fraction(inputs) * uncertainty_modifier(inputs.p_win, inputs.p_lower).value


@pytest.fixture
def bet() -> KellyInput:
    return KellyInput(p_win=0.60, win_amount=1.0, loss_amount=1.0, p_lower=0.54)


class TestOnlyWidens:
    @pytest.mark.parametrize("scale", [0.5, 0.803, 0.999])
    def test_a_narrowing_scale_is_clamped(self, bet: KellyInput, scale: float) -> None:
        out, w = widen_for_regime(bet, scale)
        assert w.clamped
        assert out.p_lower == bet.p_lower
        assert sized(out) == sized(bet), (
            f"scale x{scale} narrowed the interval and grew the position. The "
            "regime label comes from trailing volatility and lags a shift, so "
            "this would size up on the first day of the move."
        )

    @pytest.mark.parametrize("scale", [1.035, 1.188, 3.0])
    def test_a_widening_scale_shrinks_the_position(self, bet: KellyInput, scale: float) -> None:
        out, w = widen_for_regime(bet, scale)
        assert not w.clamped
        assert out.p_lower is not None and bet.p_lower is not None
        assert out.p_lower < bet.p_lower
        assert sized(out) < sized(bet)

    def test_the_busy_correction_is_not_cosmetic(self, bet: KellyInput) -> None:
        """If the measured scale barely moves the size, it is not worth the ADR."""
        out, _ = widen_for_regime(bet, BUSY)
        assert sized(out) < 0.8 * sized(bet), (
            f"x{BUSY} changed the position by less than 20%; check the two "
            "channels are both live."
        )

    def test_scale_one_is_a_no_op(self, bet: KellyInput) -> None:
        out, w = widen_for_regime(bet, WIDENING_FLOOR)
        assert out == bet and not w.clamped


class TestEdges:
    def test_no_interval_means_nothing_to_widen(self) -> None:
        """A caller with no p_lower has claimed no uncertainty to correct."""
        flat = KellyInput(p_win=0.6, win_amount=1.0, loss_amount=1.0)
        out, w = widen_for_regime(flat, BUSY)
        assert out is flat and w.applied_scale == BUSY

    def test_a_huge_scale_floors_p_lower_at_zero_rather_than_raising(self) -> None:
        """KellyInput rejects a negative p_lower, and a wide band is not an error.

        Flooring at zero is the honest reading: the interval now covers the
        whole range, the edge is gone, and kelly_fraction returns zero. That is
        a refusal to trade, which is the correct output, not an exception.
        """
        bet = KellyInput(p_win=0.60, win_amount=1.0, loss_amount=1.0, p_lower=0.54)
        out, _ = widen_for_regime(bet, 500.0)
        assert out.p_lower == 0.0
        assert kelly_fraction(out) == 0.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_meaningless_scale_raises(self, bet: KellyInput, bad: float) -> None:
        """A zero or NaN scale is a bug upstream, and silently clamping it to
        1.0 would let a broken conformal fit look like a calibrated one."""
        with pytest.raises(ValueError):
            widen_for_regime(bet, bad)


class TestTheLaggingLabelFailureMode:
    def test_a_misclassified_busy_day_cannot_size_up(self) -> None:
        """The whole reason for the clamp, stated as a test.

        A day that has just turned busy still carries the quiet label. Under
        the clamp the worst outcome is that it is sized as if uncorrected -
        never larger than that.
        """
        bet = KellyInput(p_win=0.60, win_amount=1.0, loss_amount=1.0, p_lower=0.54)
        mislabelled, _ = widen_for_regime(bet, QUIET)
        correct, _ = widen_for_regime(bet, BUSY)
        assert sized(mislabelled) <= sized(bet)
        assert sized(correct) < sized(mislabelled)


class TestItIsNotAModifier:
    def test_the_modifier_stack_still_only_shrinks(self) -> None:
        """`Modifier` forbids a multiplier above one, which is what keeps any
        single path from becoming a leverage knob. Widening reaches the size
        through p_lower AND uncertainty_modifier, so expressing it as a
        modifier would apply half of it and hide the other half.

        The first version of this test called `dir()` on the function object
        and asserted nothing at all - it passed against every implementation.
        This one drives the real stack.
        """
        from libs.ms_core.config import load
        from libs.ms_options.sizing import standard_modifiers

        limits = load(warn=False).risk_limits
        bet = KellyInput(p_win=0.60, win_amount=1.0, loss_amount=1.0, p_lower=0.54)
        mods = standard_modifiers(bet, limits=limits)
        assert mods, "the modifier stack came back empty"
        assert not any(
            "widen" in m.name for m in mods
        ), f"widening leaked into the modifier stack as {[m.name for m in mods]}"
        for m in mods:
            assert m.value <= 1.0, f"modifier {m.name} is {m.value}, a leverage knob"
