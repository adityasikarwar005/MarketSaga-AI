# Model audit — 2026-08-27

Every fitted component, checked for the two failure modes that produce working
code and believable numbers: learning noise (overfitting), and failing to learn
what is there (underfitting).

The method throughout is the **null test** — give the model data with no signal
in it and see what it reports. Every other test in the suite asks whether a
model works when there is something to find. These ask what it says when there
is not, which is the question that matters, because a model that finds structure
in noise will find it in a live market too and be wrong the same way.

All six defects below were present in code where the full test suite passed.

---

## 1. `best_lag` did not pay for its search — fixed

`best_lag` tries six lags and returns the highest r². The docstring said the
winner "is only returned when it is significant on its own terms". It was not;
it was filtered on `usable`, which checks the sample count and a non-zero
standard error and is not a significance test at all.

Measured on 60 pairs of independent random walks:

| | significant |
|---|---|
| one fixed lag | 8% |
| best of six | **30%** |

A 95% interval on the winner of a six-way search is a 68% interval with a 95%
label on it. Any transfer coefficient chosen this way was about one in three
likely to be nothing.

**Fix.** `Fit` now carries `searched`, the number of candidates compared, and
`significant` widens the interval by the Šidák critical value for that count.
Šidák rather than Bonferroni because lag candidates are correlated, so
Bonferroni would overcorrect. After the fix the best-of-six rate is 8% — the
same as a single lag, which is the point.

The coefficient is still returned when it fails the test. A real relationship at
an unproven lag is worth seeing; it just may not be treated as established.

## 2. `severity_response` invented a scaling law from noise — fixed

Impact is assumed convex in severity, `g(s) = (s/5)^gamma`, with gamma fitted.
The result was clamped to `[0.5, 3.0]`.

When severity carries no information the true slope is zero — and `max(0.5, 0)`
turned that into **0.5**, i.e. "impact scales as the square root of severity".
On pure noise the old version returned exactly 0.500 in **80 of 80** trials.

Worse, the documented fallback for *too little data* was 1.0 (linear). So a fit
on pure noise made a **stronger** claim than having no data at all.

**Fix.** The slope is tested against its own standard error before it is
trusted. Anything not distinguishable from zero returns the neutral 1.0, the
same answer as no data. Real exponents are still recovered (0.8 → 0.88,
1.5 → 1.30, 2.5 → 1.73 under realistic noise; exact on noiseless data).

## 3. The L2 penalty was inert — fixed

`QuantileRegressor(l2=1e-4)`. Weight decay per epoch is `lr * l2`, so at
`lr=0.05` this is 5e-6 per step, about 0.2% over a whole training run.

Mean |w| on 28 features fitted to pure noise:

| l2 | mean abs weight |
|---|---|
| 0.0 | 0.102001 |
| 1e-4 (the default) | 0.101807 |
| 1.0 | 0.024782 |
| 10.0 | 0.003042 |

A knob that does nothing is worse than an absent one, because it reads as a
defence that is not there.

Sweeping it on the real gap data, every point logged as a trial:

| l2 | CRPS vs naive | cov90 |
|---|---|---|
| 0.0001 | −22.2% | 93.7% |
| 0.01 | −21.9% | 93.7% |
| 0.05 | −20.2% | 92.7% |
| 0.25 | −13.9% | 91.4% |
| **1.0** | **−11.5%** | 91.7% |
| 2.0 | −11.6% | 91.4% |
| 4.0 | −12.0% | 91.2% |
| 16.0 | −12.2% | 91.2% |

Monotone improvement to a plateau — an eleven-point swing from a constant that
was doing nothing.

**Fix.** `GapForecaster` now chooses the penalty per fit on an *inner*
chronological split of the training data (`gap.select_l2`), which never sees the
calibration set or the test fold. The raw `QuantileRegressor` default is 0.1,
documented explicitly as a fallback rather than a recommendation.

## 4. Zero family spread meant the opposite of what it should — fixed

`family_prior` estimated the between-class spread as the variance of the
observed class means. Each observed mean is a true effect *plus its own sampling
error*, so that variance is the sum of both, and the family looks more varied
than it is. Rare classes then borrow less than they should and keep more of
their own noise.

On 20 classes of 8 observations drawn from one identical distribution, where the
true between-class spread is exactly zero:

| | before | after |
|---|---|---|
| estimated between-class sigma | 0.001367 | 0.000408 |
| (pure sampling-noise floor) | 0.001276 | 0.001276 |
| mean shrinkage toward family | 0.46 | **0.85** |
| classes whose CI excluded zero | 0.2% | 0.0% |

Subtracting the average within-class sampling variance exposed a second bug
underneath. When the estimated spread reached zero, `shrink` took the branch
commented *"one class, or all identical — nothing to borrow, so the class stands
on its own"* and set `weight = 1.0`, meaning **no borrowing at all**. Those two
cases need opposite answers: a family of one has no siblings, but a family whose
classes are indistinguishable is the strongest possible case for borrowing.
Correcting the spread alone would have turned maximum shrinkage into none.

**Fix.** `shrink` takes `family_size` and distinguishes the two.

## 5. The gap model's problem was not mainly overfitting — record corrected

The previous commit reported the shortfall as "27 features on ~100 training rows
overfits". That was one third right and the framing was wrong.

The train/test CRPS gap looked damning at +92%, until the naive baseline — which
has **one** parameter and cannot overfit by 80% — showed +80% on the same split.
The test period is simply harder: mean |gap| is 79% higher than in training
(0.00294 → 0.00527), which accounts for nearly all of it.

The clean measure is skill against naive on each side:

| | train | test | overfit component |
|---|---|---|---|
| full, p=28 | −6.6% | −13.9% | 7.2 pts |
| lean, p=10 | −8.9% | −16.4% | 7.6 pts |

**The model loses to naive on the data it was fitted to.** That is not
overfitting — a model that overfits at least wins on its training set. Roughly
half the deficit was a fitting failure, and the cause was defect 3.

The lean-vs-full claim did hold up: lean wins at 3, 4, 5, 6 and 8 folds.

### Result after the fixes

| | CRPS | vs naive | cov90 |
|---|---|---|---|
| before | 0.003216 | −29.3% | 95.5% MISCALIBRATED |
| lean features | 0.003040 | −22.2% | 93.7% MISCALIBRATED |
| **+ selected penalty** | **0.002827** | **−13.7%** | **90.9% calibrated** |

Half the deficit closed, and the intervals mean what they say for the first
time. **G4 still does not pass** and is not close.

The remaining gap now has a clean explanation. The regularisation sweep
plateaus where the feature weights are essentially zero, so the best available
version of this model is one that uses almost nothing — while the naive baseline
uses the one variable that genuinely moves the Indian open. A uniform penalty
cannot express that: it taxes the one good feature exactly as hard as the
twenty-seven useless ones. That is a specification problem, not a tuning one,
and the honest fix is intraday data rather than more search.

## 6. What was checked and found sound

- **Split machinery.** `purged_kfold`, `walk_forward` and
  `combinatorial_purged_cv` produce zero leaks on samples with deliberately
  overlapping outcome windows, and `audit_split` catches hand-built leaky splits
  (adjacent-unpurged and index-overlapping) every time. No leakage anywhere.
- **The optimiser.** Pinball loss is convex; the training objective decreases
  monotonically with epochs (0.000897 → 0.000849 over 50 → 4000). An earlier
  reading that suggested divergence was the conformal layer correctly widening
  intervals as the base fit tightened, not a broken descent.
- **Epochs.** Test CRPS bottoms near 100 epochs against a default of 400, but the
  difference is small (0.003325 vs 0.003334) and it is now second-order to the
  penalty. Left alone rather than tuned.
- **`severity_response` attenuation.** True 2.0 came back as 1.51 under noisy
  data, but exactly 2.000 on noiseless data — errors-in-variables in the log
  domain, a property of the estimator's setting rather than a defect in it.

---

## What this is worth

Six defects, five of them in code the suite passed clean. The pattern is the one
this project keeps rediscovering: **each defect produced a working pipeline,
plausible numbers, and no error.** A 30% false-positive rate looks like a
finding. A gamma of 0.5 looks like a measurement. An inert penalty looks like
regularisation.

`tests/ml/test_no_signal.py` now holds the null tests permanently, one per
defect, so none of them can come back quietly.
