# 0001 — Route the Mondrian conformal scale into position sizing, widening only

**Date:** 2026-08-30
**Field:** sizing input `KellyInput.p_lower` — now restated at the conformally
corrected width before `kelly_fraction` and `uncertainty_modifier` see it.
Affects `kelly_fraction` sizing for every instrument.
**Status:** accepted

## What changed

`libs/ms_options/sizing.py` gains `widen_for_regime(inputs, scale)`. Callers
that hold a Mondrian fit look up the scale for the row's volatility bucket and
pass it in; the interval is restated at `p_win - (p_win - p_lower) * scale`.

Scales below 1.0 are clamped to 1.0. The correction can only widen an interval,
never narrow one.

It is deliberately not a `Modifier`. A modifier multiplies the finished size and
is forbidden from exceeding one, which is what stops any single path becoming a
leverage knob. Interval width is not a multiplier on size — it reaches the size
through `p_lower` inside the Kelly formula *and* through `uncertainty_modifier`,
which shrinks as the interval widens. Wiring it as a modifier would apply one
channel and silently drop the other.

## Why now

M54's quantile heads are miscalibrated, and `jobs/eval_conformal.py` showed a
single global scale does not fix it: the raw heads cover 98.0% in the quiet
tercile and 88.1% in the busy one, and global conformal leaves that 9.9-point
gap at 9.9. Mondrian conformal, fitted on ten years of real NIFTY daily bars at
horizon 5, returns **x0.803 quiet, x1.035 middle, x1.188 busy**.

The busy tercile is the one that costs money. A band 18.8% too narrow there
means every position is sized as if the model were more certain than it is, on
exactly the days the tail arrives. `jobs/sizing_by_regime.py` measures the
effect: at p_win 0.60 with a 0.06 interval, the position drops from 0.0587 to
0.0392 once the interval is stated honestly — a third smaller.

That measurement has been sitting in the repo, unrouted, since it was taken.
The number was known and the sizing code did not use it.

**Why the clamp, specifically.** Bucket assignment uses trailing volatility,
which lags. The first day of a volatility shift still carries the previous
regime's label. Honouring a scale below one would narrow the interval and size
*up* on precisely the day the regime turned and the label had not caught up —
the correction would be largest exactly where it is most wrong. The asymmetry
is not caution for its own sake; it is the shape of the estimator's error.

## What this makes worse

Positions stay smaller than they should be on genuinely quiet days. The quiet
tercile is over-covered at 98.0% against a 90% nominal, the honest interval
there is 19.7% narrower, and we are declining to use that. On the three
representative bets in `jobs/sizing_by_regime.py` the quiet-regime clamp leaves
trades on the table that an honest narrow interval would have taken — the
`MISSED` rows in that job's output.

This costs opportunity and it appears in no P&L, which is exactly why it is the
half that is safe to get wrong and exactly why it will never nag anyone into
being revisited. Recording it here is the only thing that will.

Second cost: the scales come from a calibration block, and re-fitting on new
data will move them. Nothing yet re-fits on a schedule, so the numbers in the
code are as of the fit that produced them and will go stale silently.

## How we would know this was a mistake

- **The clamp is wrong** if a Mondrian re-fit on out-of-sample data returns a
  quiet-tercile scale *above* 1.0, i.e. the sign of the miscalibration is not
  stable across regimes and the whole per-bucket story was noise. Check at the
  next re-fit.
- **The routing is wrong** if realised coverage in the busy tercile overshoots
  materially past nominal once widening is live — that would mean the
  correction is being applied twice, once here and once upstream.
- **The whole thing is unnecessary** if `MondrianBands.spread_of_scales` falls
  toward zero on fresh data. That is the built-in null: near-zero spread means
  the buckets found nothing and the global scale would have done the same job.

Next look: whenever the conformal fit is next run on data that includes
sessions after the current calibration block.
