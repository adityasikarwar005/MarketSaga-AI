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

---

# Addendum — the direction model, 2026-08-27

The same null test applied to new code, before it went anywhere near real data.

`libs/ms_models/trees.py` is a from-scratch gradient booster: Newton boosting on
the log loss, histogram-binned splits, depth chosen on an inner split, and the
tree count decided by early stopping rather than a constant. Boosted trees, not
a neural network, because on tabular features with a few hundred to a few
hundred thousand rows the choice is not close. Deep learning earns its place in
this project on sequences and on text — raw candle paths and news — which are
separate modules with separate data requirements.

## What the null test caught

**A booster memorises noise far better than a linear model can.** Random labels,
30 random features:

| | trees grown | train skill | test skill |
|---|---|---|---|
| early stopping on (default) | **8** | — | −0.6% |
| guard removed | 400 | **+70.4%** | −20.4% |

Seventy percent apparent skill, on labels that were coin flips. That is the
clearest single illustration of overfitting anywhere in this project, and the
reason the tree count is not allowed to be a constant.

**Two defects in code written today**, both found by asking what the model says
when there is nothing to say.

*Isotonic calibration manufactured certainty.* Plain pool-adjacent-violators on
400 rows of noise produced nine blocks, six of them valued at exactly 0.0 or
1.0 — and it improved log loss while doing so. Same family as the zero-width
credible interval fixed earlier in the impact module: perfect confidence from a
handful of rows.

*Calibration invented an opinion the model did not have.* Even with blocks
merged and smoothed, a booster whose raw output spanned [0.475, 0.496] came out
spanning [0.046, 0.659] — a thirty-fold amplification of nothing.

The second is the more instructive. Shrinking the blocks harder did narrow it,
but cost real accuracy when there *was* a signal, which is the wrong trade. The
fault was upstream: isotonic corrects the *levels* of a ranking and takes the
ranking itself on trust. Given a model with no ranking ability, a monotone map
fitted to noise produces confident-looking probabilities out of nothing.

So calibration is now gated on discrimination — Mann-Whitney against its own
null — and skipped entirely when there is none, which is also the honest answer:
a model that cannot rank should report the base rate.

The gate had to be a real significance level. One standard error passes on noise
about a third of the time, and left the null case still opining 0.30 away from a
coin flip. Two standard errors is the ordinary 5% bar:

| | with a real signal | random labels |
|---|---|---|
| AUC | 0.784 | 0.504 |
| Brier skill | +0.209 | — |
| skill vs base rate | — | **+0.00%** |
| widest opinion | — | **0.016** (was 0.304) |
| ECE after calibration | 0.0124 | calibration skipped |
| reliability slope | 1.11 | — |

## Also built in

- **Column order is stored, not inherited from a dict.** A model that iterates a
  mapping is one ordering change away from scoring the wrong columns with the
  training run still green.
- **Missing is a value, not a zero.** Filling an absent feature with 0.0 claims
  it was average. Absence goes to the trees as a sentinel a split can isolate,
  with an indicator column when it is common, and confidence falls in proportion
  to how much of the vector is missing.
- **Timeout labels are dropped from fitting, kept in scoring.** A barrier never
  touched is not a direction; coding it as either class teaches something that
  did not happen. But a model confident when nothing happened should still pay
  for that confidence.
- **The edge is capped.** A short-horizon direction call on an index claiming
  90% is describing a bug.

Everything here runs on synthetic data with a known answer. Real numbers wait on
a broker feed — and when it arrives, the model class is a swap behind `Model`,
not a rewrite.

---

# Addendum — M55 volatility and M59 ensemble, 2026-08-27

Two more modules, the same discipline, three more defects found in code written
the same day.

## M55 · volatility

Range-based estimators (Parkinson, Garman-Klass) rather than close-to-close: a
bar that opened, swung two percent either way and closed flat reports **zero**
volatility under squared returns, and the high and low are already in the bar.

HAR augmented, never replaced. The booster fits HAR's log-residual, so the worst
case degrades to HAR rather than to noise. Scored on **QLIKE**, not MSE — MSE on
variance is dominated by the largest few observations, so a model can look good
by getting a handful of crisis days right and being useless the rest of the time.

### The retransformation trap

Volatility is modelled in logs. Exponentiating the fit returns the **median**,
not the mean, and for a right-skewed target that is a permanent downward bias no
relative metric can see — the model can rank perfectly and still be wrong by a
constant factor every single time. Duan's smearing estimator corrects it
non-parametrically, which matters because log-volatility residuals are visibly
fat-tailed and the `exp(σ²/2)` correction assumes they are not.

### Two defects the null test caught

*The booster was kept on a win rather than on evidence.* On a series with no
driver at all, it still beat HAR on the validation slice in one fold out of four
and cost 1.1% on the test rows. A finite validation slice makes a small win a
coin toss. The comparison is now **paired per observation** and has to clear two
standard errors — paired because both models see the same volatile days, and the
common component of their errors dwarfs the difference between them.

*Discarding the booster left a handicap behind.* HAR stayed fitted on 75% of the
training rows even after the booster was dropped, because that split existed to
hold out a validation slice. So the "augmented" model scored **0.8% worse than
plain HAR** on a series with no signal — purely from the smaller fit. HAR is now
refitted on every row once the keep-or-drop decision is made. Not leakage: the
validation rows informed a discrete decision, not the coefficients.

| driver strength | QLIKE change vs HAR | booster kept |
|---|---|---|
| none | **+0.0%** | 0/4 folds |
| 0.3 | +22.0% | 3/4 |
| 0.8 | +36.1% | 3/4 |
| 1.5 | +26.1% | 3/4 |

Exactly what the residual formulation is supposed to buy: it cannot be worse
than its own baseline, and it earns its place when there is something to earn.

## M59 · ensemble, calibration, conformal

The design rule is the opposite of the usual instinct: **keep the combiner
smaller than the things it combines.** A stacker sees a few hundred out-of-fold
rows whose every column is already a fitted model's output. It is the easiest
place in the system to overfit and the damage is invisible, because the stack is
validated on the same folds that produced its inputs.

So: one weight per model plus an intercept, an L2 prior centred on **equal
weighting** rather than zero, and it must beat the best single member by a
margin or it does not ship at all.

### The defect

*The meta-learner diverged.* Base probabilities clamped at [0.01, 0.99] give
log-odds reaching ±4.6, and a fixed-step gradient descent on those blew up:
stacked log loss **14.9** against 0.72 for its best single member. With a strong
penalty it oscillated instead of shrinking, because `lr * l2` exceeded one.

Replaced with IRLS — Newton's method — which has no step size to get wrong and
converges in a handful of iterations on a problem this small. A second, quieter
bug surfaced underneath: the ridge was scaled by `len(rows)`, making `l2=1` worth
five hundred pseudo-observations on a five-hundred-row fit. That pinned every
weight to the prior and produced a stack measurably worse than its own best
member.

### And one piece of arithmetic that was simply incomplete

Murphy's Brier decomposition — `reliability − resolution + uncertainty` — is
exact only when the forecast is **constant within each bin**. On continuous
probabilities there is a within-bin variance term that does not vanish, and the
identity missed by a few thousandths. Small enough to dismiss as rounding, which
is exactly how a real arithmetic error would hide.

The decomposition now reconciles the *binned* Brier score exactly and reports
the discretisation cost as its own named quantity, so the residual means
something: if it is not zero, something is wrong.

### Also enforced

- **Out-of-fold means out-of-fold.** `OutOfFoldMatrix` refuses a row whose fold
  is not recorded. An in-sample base prediction turns the stack into a machine
  for amplifying whichever member memorised hardest — and that failure produces
  excellent backtests.
- **Walk-forward, not K-fold.** A K-fold prediction for an early row comes from a
  model fitted partly on later rows, which cannot happen in production.
- **Rows a model never saw are dropped, not filled with 0.5.** Filling says the
  model was undecided when it was never asked, and the weight learned is a
  weight on an absence.
- **The conformal `n+1`.** `ceil((n+1)(1−α))`, not `ceil(n(1−α))`. That term is
  the entire finite-sample guarantee; without it intervals are quietly a few
  percent narrow and the shortfall looks like bad luck.
- **Coverage is marginal, not conditional.** 90% overall can be 98% on quiet days
  and 60% on volatile ones — and volatile days are when the interval was
  supposed to help. `coverage_by_group` exists to show it.

---

# Addendum — M74 costs and M57 meta-labeller, 2026-08-27

## M74 · the cost model

Where most retail quant projects quietly die. The components are levied on three
different bases and using the wrong one is a large, silent error:

- **Brokerage** — flat per *order*. A four-order spread pays it four times
  regardless of size.
- **Exchange fee** — on **premium** turnover, not notional. Charging it on
  notional overstates costs by roughly the inverse of moneyness — an order of
  magnitude, not a rounding error, and it is the more common mistake.
- **STT** — sell side only, on premium, when squared off.
- **GST** — on the *charges*, not the turnover. An ~18% surcharge on the fee
  stack; STT and stamp duty are themselves taxes and are not taxed again.

A two-leg spread is **four orders**, not two. Costing only the entry makes a
strategy look viable at twice its real edge, and it is an easy mistake because
the exit legs do not exist yet when the trade is being evaluated.

### The claim that was wrong, and how it got caught

The first draft's docstring asserted the folklore: that letting an ITM option
expire is always ruinous and squaring off is always cheaper. **Its own arithmetic
contradicted it on the first realistic example.**

The size of the trap depends entirely on what base the exercise STT is levied
on, and that base has not always been the same:

| base | exercise STT | square-off cost | verdict |
|---|---|---|---|
| settlement value | ₹2,205 | ₹110 | square off — 20× |
| intrinsic value | ₹49 | ₹110 | let it expire |

Both computed on the same position — 75 units, 520 intrinsic, 23,520 settlement.
`ExerciseBase` now makes the choice explicit and `should_square_off` costs both
sides and names the base it assumed, because a backtest has to price an old
trade under the rule in force at the time.

Asserting the folklore would have been easy, plausible, and wrong in one of the
two regimes. Computing it is what caught that.

### And the rates are still placeholders

`audit_cost_config` refuses to certify a profitability claim while
`config/costs.yaml` still says TODO, and separately while slippage is
uncalibrated. A total that is exact to the paisa in five terms and a guess in
the sixth is a guess, and `CostBreakdown` propagates that: one uncalibrated
component makes the whole sum uncalibrated.

## M57 · the meta-labeller

Its job is to say no. The target is **profit after costs**, not direction — a
directionally correct call that does not clear its four orders is a zero, and
that failure is invisible to every metric stopping at direction.

| | approved | precision | expectancy/trade |
|---|---|---|---|
| take everything | 1200/1200 | 0.226 | −254 |
| take the confident ones | 950/1200 | 0.231 | −245 |
| **the model** | **313/1200** | **0.617** | **−109** |

On random outcomes it approves **0%**.

### The defect: the asymmetry was in the wrong place

Skipping a good trade costs an opportunity; taking a bad one costs money and the
capacity to take the next good trade. The obvious way to encode that is class
weighting — duplicate the "skip" rows so the booster feels them more.

It is measurably the wrong place. Weighting the fit set and then calibrating on
the unweighted set puts two mechanisms in direct opposition: calibration's whole
job is to make scores match observed frequencies, so it undoes the distortion
the weighting introduced. Here it did worse than undo it —

| skip weight | approval rate |
|---|---|
| 1 | 11% |
| 4 | **30%** |

— the exact opposite of the intent, silently, with every test still passing.

The asymmetry now lives in the decision threshold, where it belongs: calibrate
honestly, then take the trade when `P(pays) > loss_ratio / (1 + loss_ratio)`,
the Bayes-optimal cut. `loss_ratio = 2` gives 0.67. The parameter now means
something checkable, and a test asserts the direction rather than assuming it.

### One more distinction worth keeping

`passes` and `net_positive` are separate properties. On the synthetic world above
the model earns its place completely — precision 0.23 → 0.62, loss per trade
halved — and **the filtered book still loses money**, because the underlying
calls had no edge to filter.

A meta-labeller cannot manufacture an edge. It can only stop you paying costs on
calls that were never going to cover them. Conflating those two is how a filter
gets shipped on top of a strategy that has nothing underneath it.

---

# Addendum — M75 position sizing, 2026-08-27

The last place before the risk gate where a good idea can be turned into a
ruinous one.

## Kelly assumes you know p. You do not.

Kelly's derivation maximises long-run growth **given the true probability**. What
this system has is an estimate from a model fitted on a few thousand rows, and
the growth-rate curve is roughly flat just below the optimum and falls off a
cliff just above it — so overestimating the edge costs far more than
underestimating it. Full Kelly on an estimated p is not aggressive; it is a
different and much worse bet than the one the formula describes.

Two defences, deliberately separate:

**Fractional Kelly** (0.25 from config) scales the whole answer down.

**A conservative probability** matters more and is the one usually skipped. The
formula runs on the *lower end* of the model's own interval, not the point
estimate:

| interval | p used | raw Kelly |
|---|---|---|
| none given | 0.58 | 0.264 |
| 0.58 ± 0.02 | 0.56 | 0.229 |
| 0.58 ± 0.08 | 0.50 | 0.124 |
| 0.58 ± 0.14 | 0.44 | 0.018 |

A model saying "0.58, give or take 0.06" is not the same bet as one saying
"0.58" flatly. Sizing them identically throws away the only thing the interval
was ever for — and every calibration and conformal check in this project exists
to produce that interval.

## Odds, not just probability

`f* = (p·b − q) / b`. A debit spread risking 85 to make 115 and a naked long
risking 100 to make 400 have completely different optimal sizes at the same hit
rate, and code that assumes even money sizes both as though they were the first.
Pinned against the closed form `2p − 1` at even odds, because a wrong formula
still returns plausible-looking fractions.

## Every reduction is named

A size that came out small has to say why. `Modifier` carries a name and a
reason, and enforces `0 ≤ value ≤ 1` — **a multiplier above one turns a risk
control into a leverage knob**, and it would be an easy thing to add by accident.

```
0 lot(s) = 0 at risk (0.00% of 500,000)
  Kelly 0.1586 -> fractional 0.0396 -> modified 0.0043 -> capped 0.0043
  x0.733 uncertainty: interval half-width 0.060
  x0.500 expiry: gamma risk is extreme and the position can invert in minutes
  x0.500 loss_streak: 3 consecutive losses - halved 1x
  x0.588 correlation: 1 open, summed correlation 0.70
```

## Correlation defaults to 0.7, not 0

Sizing each position independently is the most expensive available mistake here.
Five positions each risking 1.5%, all long the same index through different
strikes, is not five 1.5% bets — it is one 7.5% bet wearing a diversified
costume, discovered on the day they all move together. `BookRisk` measures it:
four strikes on one underlying show **88%** of the naive sum, i.e. almost no
diversification at all.

Independence is the wrong prior for a book that is, by construction, all on one
underlying.

## Two messages that had to be told apart

The first version reported "your account is too small for this trade" whenever
the answer was zero lots. That is correct when the per-trade cap is binding and
the account cannot afford one lot at it. It is **wrong** when the modifier stack
is what shrank the budget — the account is fine and the context declined — and
in that case the message is advice to fund the account in order to override a
risk control. The two are now distinguished by which constraint was binding.

Rounding is always **down**. A 0.6-lot answer rounded to nearest becomes a
position 67% larger than intended, through the cap rather than up against it.
The consequence is that a small account sizes to zero on a normal signal, and
that is reported rather than swallowed: a silent skip is a bug that looks like a
quiet day.

---

# Addendum — M76 risk gate and kill switches, 2026-08-27

The module that sits between the brain and the money. All ten blueprint stress
scenarios are automated in `tests/stress/test_risk_gate.py`, each one naming its
expected behaviour **before** the implementation — a scenario test whose
expectation is written after watching what the code does is a description, not a
test.

## Deny by default

The gate starts from reject and requires every check to pass. Starting from
allow and rejecting on breach **fails open**: a bug in a check silently permits
the trade it was written to stop. For the one component whose job is to say no,
that is the difference between a missing check being a missing veto and a
missing check being an unlimited allowance.

## Flatten cannot depend on what failed

The reason to flatten is usually that something broke. A flatten path that needs
the model server, the feature store or the feed does not work on the day it is
needed — and it passes every test written against a healthy system.

`flatten_all` takes a callable and a list of positions. Nothing else. There is a
test that parses the module's imports and fails if `ms_models`, `ms_features`,
`ms_data` or `ms_broker` ever appear.

It also assumes the broker will fight back, because "the broker API returns an
error during the flatten" is the scenario that actually costs money:

- each position is retried **independently** — a single try/except around the
  loop would leave every position after the first error open, and the log would
  show one failure instead of all of them
- a position that closed is never retried, so a duplicate acknowledgement cannot
  become a second order in the opposite direction
- what could not be closed is **returned**, not logged and forgotten. A flatten
  that reports success while a position is open is worse than one that fails
  loudly, because the operator stops looking

## Escalation is automatic, de-escalation is not

The kill level never falls on its own. A feed that comes back does not undo the
fact that it went away with a position open.

Two L3 events in a week become an L4 lockout — the second one is evidence the
first was not understood — and clearing a lockout **requires a written
root-cause note**. The failure being prevented is a system that recovers by
itself and takes the same loss again on the same undiagnosed cause. That is not
hypothetical; it is what happens when clearing is one keypress.

## The bug: a soft kill became a hard stop

L1 is "reduce size by half; notify" — explicitly not a stop. But a 4-lot trade
capped down to 1 lot, then halved by L1, produced **zero lots and a rejection**.

For any account whose positions are already at the minimum tradeable size, that
silently converts the mildest kill level into the most severe one. At one lot
there is no half; the choices are one or none, and L1 says one.

The trade now stands and the warning says plainly that the reduction could not
be expressed — *"L1 soft kill could NOT reduce size — already at one lot. The
soft kill is in force but is not shrinking anything."* An operator whose soft
kill is not actually reducing anything needs to know that, and a silent
rejection tells them the opposite.

## The drill

`python tasks.py kill-drill` runs against a broker that **fails on purpose** —
one position times out and recovers, one refuses permanently. A drill against a
cooperative broker proves the drill runs, not that the flatten does.

It exercises: escalate on a dead feed → flatten → discover what did not close →
escalate again → two L3s become a lockout → the lockout refuses to clear without
a written cause. Results append to `research/kill_drills.jsonl`, versioned like
the trial log, for the same reason.

## Limit changes need a decision record

A risk limit that can be edited in a commit is one that gets edited at the worst
moment — after a losing week, by the person who lost it, with a reason that
sounds convincing at the time. `limit_change_requires_adr` reports which changed
fields have no record.

The ADR template asks two questions that are easy to skip: **what this makes
worse**, and **how we would know it was a mistake**. If the first is empty the
change has not been thought through — a limit with only upside was set wrong
originally, and that is the decision to write down instead.

---

# Second full audit — 2026-08-27

Three defects, one of them in arithmetic that eighty-two tests had been passing
over, and one a composition failure where every module was individually correct.

## 1. Black-76 theta had a sign error — fixed

Differentiating `c = e^(-rT)(F·N(d₁) − K·N(d₂))` and using `F·φ(d₁) = K·φ(d₂)`:

```
dc/dT  = −r·c + e^(-rT)·F·φ(d₁)·σ / (2√T)
theta  = −dc/dT = −e^(-rT)·F·φ(d₁)·σ / (2√T)  +  r·c
```

The `r·c` term carried a **minus**. Every other greek matched a finite-difference
check to eight decimal places while theta was 2.2% out — which is what made it
findable at all:

| greek | code | finite difference | relative error |
|---|---|---|---|
| delta | 0.505299 | 0.505299 | 8e-09 |
| gamma | 0.000411859 | 0.000411859 | 8e-08 |
| vega | 2729.7812 | 2729.7812 | 2e-11 |
| **theta** | **−2349.7080** | **−2300.0193** | **2.2%** |

The discrepancy was exactly `2·r·c = 49.6888`. Too small to be a units mistake,
far too large to be numerical noise beside the others.

**A wrong sign on a small term does not look wrong.** It looks like a slightly
faster decay — and it grows with rates and with time to expiry, so it would have
been worst exactly where theta matters most.

Thirty pricing tests passed throughout. They checked signs, magnitudes,
monotonicity, put-call parity and IV round-trips — everything except the one
thing a derivative has a definition for. Every greek is now pinned against a
finite difference of the price it differentiates, across a grid that includes
`r = 0` (where the bad term vanishes and the error is invisible).

Also verified sound, at machine precision: put-call parity (1e-12), boundary
behaviour at `T=0` and `σ=0`, price ≥ intrinsic at every strike, monotonicity in
strike / vol / time, `Δc − Δp = e^(-rT)`, `Γc = Γp`, `νc = νp`, IV round-trip to
1e-13, and `None` rather than a guess for unreachable prices.

## 2. The sizing module's best defence was unreachable — fixed

`KellyInput.p_lower` — running Kelly on the lower end of the model's own
interval — appeared **only inside `sizing.py` and its tests**. Nothing in the
model layer produced an interval, so every real call path would have passed
`None` and sized on the point estimate: the case the module itself flags as
least reliable.

Worse, `Opinion.confidence` reads like it fills that role and does not. It
measures **input completeness** — what fraction of the feature vector was
present. A model with every feature present and enormous uncertainty about the
probability reported `confidence = 1.00`.

Each module was correct in isolation. The composition silently disabled the
defence, which is the same shape as the class-weighting-versus-calibration bug
found in the first audit.

**Fix.** `IsotonicCalibrator` already knew how many calibration rows stood behind
each mapped value; it now keeps them and returns a **Wilson score interval** on
that block. Wilson rather than the normal approximation because the normal one
returns bounds outside [0, 1] near the ends — exactly the confident region where
a sizing rule is dangerous.

| calibration rows | mean interval width |
|---|---|
| 400 | 0.221 |
| 1,600 | 0.145 |
| 6,000 | 0.102 |

`Opinion` now carries `p_lower` / `p_upper`, and `confidence` is documented as
what it actually is.

### And a bug inside the fix

The first version computed the interval from the raw score and capped only the
point estimate, producing `p_up = 0.8500` with an interval of
**[0.8424, 0.9570]** — the reported probability sitting on the floor of its own
interval, and a lower bound asserting near-certainty about a level the edge cap
existed precisely to disbelieve. A caller sizing on that `p_lower` would have
been handed the uncapped opinion through the back door. The cap now applies to
the bounds as well.

## 3. A neutral model was reported as wrong every time — fixed

`score()` counted a hit as `(p > 0.5 and y > 0) or (p < 0.5 and y < 0)`. At
`p == 0.5` neither branch fires, so an abstention counted as an error:

```
always_neutral  n=1,000  log_loss=0.69315  brier=0.25000  acc=0.000 (on 1,000)
```

**Accuracy 0.000** — the most alarming number available, for the least alarming
behaviour, and indistinguishable in the output from a model that was wrong every
single time. Now:

```
always_neutral  n=1,000  log_loss=0.69315  brier=0.25000  acc=n/a (on 0, abstained on 1,000)
```

Accuracy is over the rows where a side was taken. An abstention still pays in log
loss and Brier, which is where indecision belongs — it does not belong in a hit
rate.

## What was checked and found sound

- **Determinism.** Every fitted model — boosted trees (including with
  subsampling), quantile regressor, direction model, volatility model, stacked
  ensemble, meta-labeller — produces identical output from identical input across
  separate fits. A fit that cannot be rebuilt makes every claim from it
  unfalsifiable.
- **The M0 rules baseline on noise.** Log loss 0.69328 against a coin flip's
  0.69315, opinions spanning [0.401, 0.585] inside the 0.12 edge cap. It forms
  opinions from noisy features, as a rules engine must, and gains no skill from
  them.
- **Conformal coverage with a useless model.** Nominal 90% delivered 90.0% on a
  target independent of every feature. That is the guarantee working as designed:
  conformal coverage does not depend on the model being any good.
- **`climatology()`** returns the base rate, not a log loss. Reading it as one
  made it look wrong; it is not. Recorded here because the misreading was mine
  and the next person will make it too.

---

# M80 + M81 — the paper broker and the fill model, 2026-08-27

Sprint order says S16 is M76 M77 **M80 M81**. These two are a pair: a paper
broker without a fill model is a paper broker that lies.

## The tie rule

A bar has a high and a low and does not record which came first. When a
position's target and stop both sit inside one bar, there are three options:
assume the target (flatters everything), assume the stop (pessimistic), or model
the path (needs tick data a bar-resolution backtest does not have).

This takes the stop. Not because stops always fill first, but because **the
error is bounded and one-directional** — a real day can only be better than
reported, and a backtest that can only understate is one whose number means
something.

The opposite convention is the one a naive implementation reaches by accident,
because targets tend to get checked first. It turns the most volatile sessions —
exactly the ones where both levels are hit — into the most profitable ones, and
the equity curve simply looks good.

`worst_case_sequence` counts how many exits were resolved by the tie rather than
by the data. If that share is large, the result is largely a statement about the
convention and not about the strategy, and that is worth knowing before quoting
the number.

## Fills at the far touch, never the mid

A marketable order pays what someone else is willing to trade at. Filling at the
mid claims half the spread as free profit **on every leg** — on a four-order
option spread that is most of the edge, and it is the single most common way a
backtest lies.

A resting limit order is different again: it fills only when the market **trades
through** it, and even then the queue discount applies. Assuming a limit fills
the moment price touches it assumes you were first in the queue every time.
`queue_ahead = 1.0` assumes you were last, which is the assumption reality cannot
beat.

Impact is zero inside the displayed size and grows with the square root of the
excess, scaled by the spread — a wide market punishes size harder than a tight
one, which is why liquidity is a feature in the sizing layer. An empty book is
treated as expensive rather than free: nothing displayed is not infinite
liquidity.

## The interface is the point

"Same interface as the live adapter" is the M80 spec's first line, and there was
no such interface — `ms_broker` had a feed and an auth flow and nothing that
places an order. `protocol.py` now defines it once, and a test asserts
`PaperBroker` implements every abstract method and has no unimplemented ones.

A paper broker with a convenient extra method is one the strategy learns to
depend on and production does not have.

Three things the interface deliberately does **not** promise:

- **That an order fills.** `place` returns an acknowledgement. Real brokers
  accept an order and report the fill later, sometimes never; an interface that
  returns a fill synchronously teaches a habit that breaks in production.
- **That the broker's positions are the system's positions.** Reconciling those
  is a separate problem, and pretending they are the same is how a phantom
  position gets doubled.
- **That cancel succeeds.** It returns whether the request was accepted. An order
  can fill in the microsecond between deciding to cancel and the cancel arriving.

## Position arithmetic — the case that gets written wrong

Adding averages the price. Reducing realises against the average and leaves it
unchanged. **Crossing through zero does both**, in that order, and the new side
starts at the crossing fill's price *alone*.

Blending them produces an average that is a mixture of a long and a short, which
is not the price of anything. It shows up as P&L drifting slowly from the truth
and stays invisible until a reconciliation.

Also enforced: every fill pays real costs through `ms_options.costs`, so a
round trip at an unchanged price **loses money** — which is the most important
sanity check a paper broker has. Equity refuses to mark a position it has no
price for, because marking at zero turns a data gap into a loss that looks real.
Marks come from the mid, not the touch: marking at the liquidation price
conflates valuation with exit cost, and charging that twice understates every
open position.

---

# M82 — the event-driven simulator, 2026-08-27

Owns the only number anyone is allowed to quote about profitability.

Every other module can be wrong and produce a *worse* model. This one can be
wrong and produce a **better-looking** one, which is the failure that ends
projects. So the whole design is one sentence:

> **A decision made at time T cannot be filled at time T's price.**

Violating that is the most profitable bug in quantitative finance and the hardest
to see. Nothing errors, nothing looks out of place, and the equity curve simply
goes up.

## Three delays, kept separate

| delay | why it is its own thing |
|---|---|
| **Discovery** — event at T, known at T+3s | A headline is not on your screen the instant it is filed. **Floored at 3s and not configurable to zero** — `SimConfig` raises. |
| **Decision latency** | Sampled from a log-normal, not assumed constant. The slow decisions happen during bursts, and bursts are when prices move. |
| **Round trip** | Submission, ack, fill. |

They behave differently under load: discovery does not change when your machine
is busy, decision latency doubles. Collapsing them into one constant hides that
**the worst latency arrives at the worst moment**.

Latency is right-skewed on purpose — symmetric noise around a median produces
fast decisions as often as slow ones, and there is no such thing as a
negative-latency decision. `from_samples` fits the median and p99 rather than
mean and standard deviation, because a mean is dragged around by the tail it is
supposed to summarise, and it **refuses fewer than 30 timings**: an unmeasured
model is honest, a badly measured one is not.

## Lookahead is refused, not merely discouraged

`book_for_decision` and `book_for_fill` are two separate fetches at two separate
times, and the second **raises `LookaheadError`** if asked for a price from before
its own decision. Enforced by construction rather than by care, because care does
not survive a refactor.

The exit walk only sees bars strictly *after* the fill. A bar already in progress
when the order arrived has a high and a low that partly predate the position, and
crediting those is the same lookahead in a smaller costume.

## The report is written to be hard to over-read

```
190 trades, hit rate 37%
gross -75  costs 399  net -474 (-0.047% of capital)
profit factor 0.04   max drawdown 0.047%
by event class, worst first:
  corp.earnings         n=78  hit=29%  gross=-55  net=-219  cost drag 296%
  macro.rate_decision   n=59  hit=39%  gross=-16  net=-140  cost drag 792%
  geo.military          n=53  hit=45%  gross= -4  net=-115  cost drag 2900%
```

Attribution is ordered **worst class first** — the number that decides what gets
disabled belongs at the top, not buried under the winners. `cost_drag` above 1.0
says the edge never existed.

Net is printed as a fraction of capital as well as in rupees. A net of −474 beside
a drawdown of "0.0%" reads as a contradiction until you notice the capital is a
million; printing only one invites the wrong reading.

`tie_broken_share` is reported prominently and **warns above 20%**: a backtest
whose exits are mostly ties is largely a statement about the tie convention, and
its profit factor is a property of that choice rather than of the strategy.

And every run prints what would make the number an artefact — an unmeasured
latency model, the discovery delay being at its floor rather than a real wire,
and that one equity curve cannot say whether it was luck.

## What the demo actually demonstrates

The synthetic world plants a real directional edge (58% for `geo.military`) and
costs still eat it: hit rate 45% after the exit rules, cost drag 2900%. That is
the useful lesson rather than a disappointing one — **a directional edge is not a
trading edge**, and the gap between them is exactly what M74 and M81 measure.

Synthetic on purpose, and it says so: this job exists before there is intraday
data, and a demo that pretended otherwise would be the first lie in a module
built to prevent them.

---

# M85 — the ten scenarios, end to end, 2026-08-27

M76 already ran these ten against the risk gate. This is a different question and
the more important one.

**A gate deciding correctly and the system behaving correctly are not the same
claim.** A gate can escalate to L3 while nothing actually flattens. It can refuse
a trade that was already sent. It can be consulted *after* the position was
opened. Every one of those passes a gate test and fails in production, because
the defect is in the wiring rather than in the rule.

So these drive the whole chain — event, gate, broker, account — and every
assertion is about **what the account looks like afterwards**, not about a
verdict.

## What the end-to-end version catches that the gate version could not

| | gate test asks | scenario test asks |
|---|---|---|
| 1 — circuit breaker | did it escalate to L3? | **is the account flat?** |
| 1b — broker refuses | — | does a failed flatten report failure, or claim success? |
| 4 — duplicate ack | — | is the position doubled? |
| 5 — stale chain | is it L2 not L3? | **did the existing position survive?** |
| 8 — forced flatten | is a new entry refused? | **is everything actually closed?** |
| 10 — stale model | is it L1? | did the position survive *and* the new one get halved? |

Scenario 1 is the clearest: escalating to L3 and *being flat* are different
states, and only one of them is safe. A gate test cannot tell them apart because
it never touches an account.

And one that only exists here: `test_the_harness_asks_the_gate_before_the_broker`.
A system that places the order and then asks is a system whose risk gate is a
logger.

## Two guards on the suite itself

`test_every_scenario_has_a_test` fails when a scenario is added to the list and
forgotten. A stress suite that silently covers nine of ten is worse than one that
covers nine and says so.

`test_every_scenario_states_its_expectation` requires each scenario to carry its
expected behaviour as data, and to state it in more than a few words. A scenario
whose expectation is written *after* watching what the code does is a description
of a bug, phrased as a requirement.

## A design smell fixed on the way

`flatten_all` demanded `ms_risk.gate.Position`, and the paper broker has its own
`ms_broker.protocol.Position` — two types with the same name, structurally
different (one carries delta and vega, the other average price and realised P&L).

The flatten path only ever reads `.symbol`. Demanding a concrete type would force
a conversion **at the exact moment the system is failing**, which is the worst
possible time to be constructing objects. It now takes a `Closeable` protocol —
anything with a symbol — so the caller passes whatever it already has.

Where a real conversion is needed (broker positions into the gate's book state)
it is explicit and commented: the greeks are dropped because the broker does not
know them and **must not invent zeros that read as real exposure**.

---

# M93 ledger and M92 alerting, 2026-08-28

Two S18 modules that need no broker and no web framework.

## M93 — the audit spine

**Predictions and outcomes live in the same row.** Split across two tables joined
by an id, three very different situations collapse into one empty join: a
prediction whose outcome was never written, one whose outcome row was lost, and
one that was never meant to have an outcome. In one row, "unresolved" is a state
the row carries, with a reason attached.

**The file is the source of truth; Postgres is an index.** The ledger must never
lose a write, and the moment it most needs writing is the moment something has
gone wrong — which is also when the database is most likely unreachable. Same
shape the recorder already uses, for the same reason: the durable thing should
have the fewest moving parts.

**Resolution appends, it does not edit.** Writing the outcome adds a second
record for the same `entry_id`; the reader takes the last one and the earlier
stays on disk. Editing in place would mean the file no longer shows what was
believed at prediction time, which is the one thing it exists to preserve — and
that is exactly the edit that gets made with good intentions.

**Resolving early is refused.** An outcome resolved before its horizon measures a
different quantity than the one predicted, and the resulting number would be
perfectly plausible. Late resolutions are recorded with their lag, because a
systematically late resolution is a systematically different measurement and
nobody notices unless it is written down.

**`was_right` is `None` while pending, never `False`.** Defaulting it to False
makes every hit rate drift downward as the day goes on, which reads as a model
decaying.

The audit is three-way — resolved, unresolvable-with-a-reason, or still inside
its horizon — because a count of "resolved" alone cannot tell a healthy day from
one where half the outcomes were quietly never written. An unresolvable
prediction **must** state why: a row with no outcome and no explanation cannot be
told apart from one that was lost, and that single ambiguity makes every report
built on the table unauditable. The database constraint enforces it too.

## M92 — alerting

Two rules that pull against each other.

**Rate-limit everything**, so the alerts stay worth reading. Alert fatigue is not
a comfort problem; it is the mechanism by which a working alarm fails — the
operator learns that most notifications need no action, and the one that did
arrives into that habit.

**Except the ones that cannot be missed.** A risk breach is not a notification,
it is an instruction, and a rate limiter that can swallow it has converted a
safety control into a convenience. `CRITICAL` bypasses both the rate limit and
every mute, and there is **no parameter that changes that** — a test asserts
`emit` has not grown an override argument, because a configurable override would
be configured on the day it mattered, by someone trying to get some sleep.

The bypass keys off **severity, not kind**, so a new critical alert is protected
the moment it enters the table rather than when someone remembers the special
case.

The limit is **per alert key**, not global: a global budget means a burst of one
kind exhausts the allowance for every other, which is precisely backwards — the
burst is what should be throttled.

Mutes are **per channel and time-bounded**. "Mute everything" is the button
pressed during the noisy hour and forgotten by the quiet one, so the failure mode
here is a mute that lapses rather than one that persists. A mute needs a reason,
because an unexplained silence is indistinguishable from a broken alerter.

And the suppressed list is kept rather than dropped: *nothing happened* and
*forty things happened and were suppressed* look identical from the receiving
end, and only one of them is a quiet day.

---

# M90 — the API gateway, 2026-08-28

REST for what has already happened, a WebSocket for what is happening now. The
split is not stylistic: history is a question with an answer, live state is a
stream with no end, and serving the second over the first produces a dashboard
that polls and a system that spends its time answering.

## Backpressure is the requirement

**A dashboard that cannot keep up must be dropped, not waited for.** If a slow
client can make the publisher block, then opening a browser tab on bad wifi slows
down the thing that places orders — a trading system whose latency depends on who
is watching it.

Both alternative designs fail: an unbounded queue turns a slow client into
unbounded memory, a blocking send turns it into unbounded latency. So every
client gets a bounded queue and a slow one loses frames.

Dropping is also *correct* here, not merely pragmatic. The live channel carries
**state**: a client that missed three updates does not need all three, it needs
the current one. What it must never do is silently believe it is up to date —
so drops are counted and a `gap` frame is sent with the next message. Silently
resuming is how a dashboard ends up confidently displaying a stale position, and
that is worse than a gap it can see.

The queue is small (32) on purpose, and a test asserts it stays small: a client
twenty frames behind wants the newest frame, not the backlog.

## Auth, including in single-user mode

The blueprint says "habits matter later" and it is right for a specific reason:
the single-user version is the one that gets deployed to a laptop, then to a VPS
"temporarily", and **the moment auth is optional it is absent**.

- A missing token is a **startup failure**, not a generated default. A generated
  token gets logged, copied once, and becomes permanent.
- The comparison is `secrets.compare_digest`, because `==` leaks the length of
  the matching prefix through timing.
- The rejection says nothing about *which* part was wrong. "Invalid token" and
  "no token" are different messages to a person and the same information to a
  scanner.
- The WebSocket is authenticated **before accept**, with close code 4401 rather
  than a generic 1008 — a client reconnecting on a generic policy violation would
  retry forever against a token that will never work.
- `/health` is the only open route, because a health check that needs a
  credential fails for the wrong reason when the credential is what broke.

## The bug: postponed annotations broke FastAPI's parameter resolution

The WebSocket route closed immediately with `1008 Field required`, which reads
exactly like an auth rejection. It was not.

This module uses `from __future__ import annotations`, so every annotation is a
**string** that FastAPI resolves against the *module* globals. The `fastapi`
imports were inside `create_app`, so `websocket: WebSocket` resolved to nothing
and FastAPI reinterpreted the parameter as a required **query field**.

The imports are now at module scope with a comment saying why they cannot move.
A dependency-injection framework plus postponed annotations plus a factory
function is a combination that fails quietly and blames the wrong layer.

## And the project's own guard caught the next one

The first version of this module used `datetime.now(UTC)` twice — once for a gap
frame's timestamp, once for the ledger route's overdue check.
`test_no_wall_clock_calls_outside_clock_module` failed, which is rule R6: library
code never reads the wall clock, it takes a `Clock`.

Worth stating why the API is not an exception. **Replay serves a recorded day
through these same routes.** The ledger's "is this prediction overdue" answer
depends entirely on what "now" is, so a route reaching for the wall clock would
mark every prediction in a replayed day overdue — and would do it while looking
completely correct.

The clock is now injected into `create_app`, and a test in the API's own file
asserts the module has no wall-clock reads, so the failure surfaces next to the
code rather than in a distant architectural test.
