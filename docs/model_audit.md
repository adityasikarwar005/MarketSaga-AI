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

---

# MA0 + MA1 — the daily rhythm, 2026-08-28

Seven jobs at seven times. **Six can fail and be retried tomorrow. One cannot**,
and most of these two modules exist because of that one.

## The forced flatten is not a scheduled job

15:20 squares off every intraday position, no exceptions. Treating it as one
entry in a job list makes it as skippable as the nightly archive, and the failure
modes are not comparable: a missed archive costs a rerun, a missed flatten
**carries a position overnight through a gap nobody modelled**.

Three rules follow that nothing else gets:

- it runs **even when every earlier job failed** — a broken preflight is a reason
  to hold no position, not a reason to keep one;
- a missed window is an **incident**, not a skipped job;
- it cannot be disabled by configuration. A flag that turns it off is a flag
  someone sets during a debugging session on a Tuesday.

## The missed window is the interesting case

If the process was down from 15:00 to 16:00, the flatten never fired. A naive
scheduler does one of two things and **both are wrong**: run it late — squaring
off at 16:00 into a shut market where the order will not fill — or skip it,
leaving the position open overnight and reporting a clean run.

Neither is safe alone, so a missed critical job becomes an alert **plus a
position check**. That is the only response that distinguishes *the moment passed
harmlessly* from *something is still open*, and it is why `positions_open` is
injected: the runner cannot answer that question itself, and assuming either way
is wrong. Assuming open cries wolf every weekend; assuming flat is how a position
spends the night unnoticed.

A missed flatten with a flat book is still recorded. **A flatten that keeps not
running is a fault even on the days it happens not to matter.**

## Ordering is a dependency, not a time

Post-session runs after the flatten because it computes outcomes for positions
the flatten closed — not because 15:45 is later than 15:20. Encoded as a time, a
slow flatten produces attribution over positions that are still open and the
numbers look fine.

And a job whose dependency failed is **skipped with a reason, not failed**.
Counting it as a failure buries the one thing that actually broke under the six
that followed it, and the post-mortem then starts in the wrong place.

Preflight has **zero retries** on purpose: a retried preflight is a preflight
nobody read. The first failure is the signal, and quietly succeeding on the
second attempt hides that the feed took two tries to connect on the morning it
later dropped mid-session. The flatten has three, because there the retry *is*
the point — the broker returning an error is the expected case.

## A deliberate deviation from the plan

The plan says "Prefect flows". This puts the semantics in plain Python and makes
Prefect a deployment adapter, for a stated reason: the part of orchestration with
correctness requirements is *which job runs, after what, and what happens when
one is missed*, and a missed flatten carries overnight risk. That logic has to be
testable without standing up a workflow engine and must not change behaviour when
the engine is upgraded.

The engine gets what it is good at — cron, UI, history — and does not get the job
of deciding whether it is safe to skip a flatten. A test asserts the suite never
imports Prefect at all.

## The bug: naive times against an aware clock

`scheduled_for` returned `datetime.combine(day, self.at)` — **naive** — and the
Clock reports timezone-aware UTC. Comparing them raised `TypeError`, which is the
lucky outcome.

The unlucky one is worse and nearly happened: the rhythm's times are
**exchange-local IST**, so a naive comparison that *did* succeed would be off by
five and a half hours and fire the forced flatten at **20:50 IST** — hours after
the market shut, into a book that had been open the whole evening.

`scheduled_for` is now timezone-aware in `Asia/Kolkata`, and three tests pin it:
that the offset is +05:30, that 15:20 IST is 09:50 UTC, and that every job in the
table schedules to an aware instant.

---

# M94 — attribution and the class-level kill policy, 2026-08-28

The plan's note says this policy does more for long-run profitability than any
model improvement, and it is worth taking literally.

**A model that is right on eight event classes and quietly bleeds on the ninth
does not look broken.** It looks like a model with a mediocre average, and the
instinct is to improve the model. The cheaper and larger win is to stop trading
the ninth.

## Two conditions, both necessary

A class is disabled at **≥25 trades AND negative expectancy** over a rolling
quarter.

The trade count is not a formality. With five trades a losing streak is the
overwhelmingly likely outcome of a coin flip *with* an edge, and a policy that
disables on it will disable half the working classes — then look effective,
because the survivors are the ones that got lucky. The threshold separates
*"this does not work"* from *"this has not had a chance"*.

The window is rolling rather than since-inception: a class that lost for six
months and has worked for three is a class that **changed**, and an all-time
average hides exactly that.

## Re-enabling requires a change, not patience

This is the part that makes the policy real.

A disabled class that comes back after a cooling-off period comes back to the
same model, the same features and the same market, and loses money again. **The
only thing that elapsed was the operator's memory.**

`reenable` demands a changed model version *or* a written justification of at
least 30 characters. Time alone does nothing, and there is no parameter that
makes it do something. A test asserts that 180 days of waiting leaves the class
disabled.

The register is **state**, not a derived value. Recomputing it each week would
re-enable a class the moment a quiet quarter pushed its rolling expectancy back
above zero — the same class, the same model, the same reason it lost money. A
test pins that too.

## Metrics that say different things

- **avg R** rather than only rupee expectancy. Expectancy says a class that
  trades big is important; avg R says whether it is any *good*. Reporting only
  the first is how a large, mediocre class hides behind a small, excellent one —
  and a test constructs exactly that pair.
- **Edge realisation slope**: how much of the claimed move actually arrived. A
  slope near zero means confidence carries no information about magnitude, which
  is perfectly compatible with a good hit rate. A slope well below one means the
  edge is real but consistently smaller than claimed — a *sizing* problem, not a
  model problem.
- **Brier decomposition**, reused from the ensemble rather than reimplemented.
  High reliability error means the probabilities are mislabelled and calibration
  fixes it; low resolution means the model is not separating outcomes at all and
  no calibration will help.

## Two refusals in the reporting

A week with fewer than 20 resolved predictions reports **n/a** for the
reliability slope rather than a number. Twelve trades cannot support one, and
printing it invites acting on it.

And `--apply` is required to disable anything. Without it the run is a dry read:
a policy that silently changes what trades on every scheduled run is one nobody
remembers agreeing to.

Only **resolved** entries are attributed. Counting unresolvable ones as zeros
would dilute every average toward nothing and make a class with missing outcomes
look *safer* than one that was actually measured.

---

# MA2 + M5A — retraining and the promotion gate, 2026-08-28

MA2 depends on M5A, which was not built. Rather than inventing a registry inline,
M5A's **stage machine and promotion gate** are built here and MLflow/ONNX serving
are deferred — the same call made for Prefect, for the same reason: the part with
correctness requirements must be testable without standing up a service.

M5A is marked **partial**, not done, because that is what it is.

## The rule, and why every word is load-bearing

> promote only if it beats the incumbent on the last four weeks out-of-sample

**"Beats the incumbent", not "is good."** The incumbent is already live and costs
nothing to keep. A challenger that is merely acceptable is a change with all of
the risk and none of the benefit — and the natural question, *"is the new model
good?"*, has no way to answer that.

**"Out-of-sample", with the challenger defining the boundary.** The window starts
the day after the challenger's last training day. Defining it from *today*
instead would let a model trained through last Friday be scored on last Thursday,
and the win would be large, clean and entirely fake.

The incumbent is then scored on the same window — which is out-of-sample for it
too, but *further* from its training set. That asymmetry is correct and worth
naming: it is exactly the question that matters, because holding up on
increasingly distant data is what the incumbent will keep having to do if nothing
changes.

**"Four weeks", not two.** The challenger was fitted up to the start of the
window; a shorter one is mostly the tail of its training set, where any model
looks excellent.

## What the gate refuses

Every blocker is reported, not just the first — a challenger that fails on three
counts and is fixed for one comes back and fails on the other two, and each round
trip is a week.

- not in **shadow**, or fewer than 15 shadow sessions. Shadow is the only stage
  producing evidence nobody chose: live predictions, recorded, trading nothing.
- a scoring window that **overlaps its own training data**
- fewer than 200 out-of-sample predictions — below that the comparison is noise
  whatever it says
- the incumbent **not scored**, or scored on a *different window*: a challenger
  compared against nothing wins every time
- an improvement below 0.005 log-loss. **A challenger ahead by a thousandth has
  won a coin toss**, and promoting on it means promoting every week for no reason

## Every retrain is a trial, win or lose

Fifty-two retrains a year is fifty-two attempts at beating the incumbent, and one
that wins in that sequence has not necessarily found anything. `retrain` writes
to the experiment log **whether or not it promotes** — a trial count that includes
only the winners is not a trial count, and every deflated statistic downstream
reads it.

## Registry invariants

- **One live model.** Two means predictions come from whichever the caller
  happened to load, and attribution is meaningless.
- **Nothing live is a state, not an error** — day one and post-emergency-archive
  are both real, and a caller forced to catch an exception will eventually catch
  it and continue. But `production_model` *refuses* rather than falling back:
  a system trading on a default model is one whose operator does not know what it
  is trading on.
- **Promotion takes the check rather than performing it**, so the decision and
  the action are separable — and a check for a *different* model is refused,
  because promoting on someone else's evidence is how the wrong model goes live.
- **Shadow cannot be restarted.** It would reset a session count that is meant to
  be evidence.
- **Archiving needs a reason.** The reason a model was pulled is the most useful
  thing about it later.
- **Every prediction is stamped with its version.** A week mixing two versions and
  reporting one hit rate is reporting a number about nothing, and the mixing is
  invisible — it looks exactly like a single-model week.

## A small bug in my own tracker update

The tracker script marked MA2 partial instead of done, because the condition was
`"M5A" in old` and MA2's row contains `M5A` **in its dependency column**. Same
class as everything else in this document: a substring test that matched
somewhere unintended, produced a plausible result, and raised nothing.

## R6 caught me again

`test_no_wall_clock_calls_outside_clock_module` failed on `retrain.py`, which
stamped its experiment-log entry with `datetime.now(UTC)`. Second time in one
session — the API gateway was the first.

The consequence here is specific: a retrain replayed over a past quarter would
stamp every trial with **today's** date, and the experiment log would then show
fifty-two attempts that all happened on one afternoon. The trial count would
survive; the chronology, which is what makes it auditable, would not.

Fixed by injecting the clock, and `ms_ops` now carries its own guard so the
failure surfaces beside the code rather than in a distant architectural test.

Two hits from the same rule in one session is worth noting rather than
brushing past. The rule is not fussy — it is load-bearing in a project whose
entire correctness story is point-in-time, and I keep reaching for the wall
clock because it is what one reaches for.

## The scrip master had never been downloaded

`InstrumentMaster` could parse a payload and save one, and had no way to fetch
one. Every test ran against a three-row synthetic fixture. That is not an
oversight in the parser — it is the thing this document keeps recording: the
component had a green test suite because the test supplied data shaped the way
the parser expected.

The first real download raised on row one:

```
PriceScaleError: 972500000.0 decodes to 9,725,000.00, which is not a
plausible price on these exchanges (0.01 to 1,000,000).
```

**Strike is not in paise across the whole file.** The currency and
interest-rate segment (`CDS`) publishes it scaled by ten million; NFO, BFO, MCX
and NCO use paise. `USDINR26O2397.25PE` — strike 97.25 — arrives as
`972500000`.

Two things are worth separating here.

**Why it was invisible.** Nothing in the pipeline would have raised. A currency
strike of 9,725,000 is a number; moneyness computed from it is a number; an SVI
fit over those strikes converges. The failure would have surfaced as a
volatility surface for the CDS segment that was simply wrong, months later,
with no error anywhere — the same shape as the theta sign error and the inert
regularisation constant. The only reason it stopped at the boundary is that
`MAX_PLAUSIBLE_RUPEES` was already there, guarding a *different* mistake.

**How the scale was established.** Not by picking the divisor that made the
numbers look right — that is the anti-pattern this whole layer exists to
prevent. Angel's option symbols carry their own strike, so the file checks
itself: split each symbol on the expiry taken from that row's own `expiry`
field, and compare the tail against the `strike` field. That gives 9,290 CDS
rows at 1e7 with no exceptions, and NFO, BFO, MCX and NCO uniformly at 1e2.
The remaining rows are ones where the symbol's expiry encoding is ambiguous
(`ASIANPAINT26NOV2620PE` splits two ways), not ones that disagree.

Why CDS is 1e7 is not documented anywhere I could find, so the constant records
it as measured rather than explaining it.

**The second finding, from the same download.** `strike` is `-1` for anything
that does not have one — equities, futures, the underlying rows — 24,035 of
them, and `-1.000000` is the *only* negative value in the file. Sent through
the scale check it raises on the most ordinary rows there are. A sentinel is
not a price and now has its own path.

**What the download's own guard is for.** A truncated response or an HTML
maintenance page served with a 200 parses to zero rows without raising. The
feed would then subscribe to no tokens and go quiet, which is indistinguishable
from a closed market. `MIN_PLAUSIBLE_SCRIP_ROWS` refuses it at the boundary.

The master is also refetched when it is not today's. That is a correctness rule
rather than a cache policy: Angel republishes daily, an option's token belongs
to one expiry, and a master from before a roll resolves a live symbol to a
token that now trades something else. `resolve()` cannot tell — the row it
finds is perfectly well formed.

## M70 on a real chain: the data cannot support the fit

Section 7 said to point the SVI fitter at a real option chain, and that the
bhavcopy loader made it possible with no account and no credential. The fitter
ran. The answer is that the question cannot be settled this way, and the reason
is worth recording in full because I asserted the opposite in a docstring first
and the data disproved it.

**What I claimed.** `iv_from_quote` reads `quote.mid` and has no LTP path, for a
good reason: an LTP is a print from whenever the last trade happened, so a
surface built from LTPs is built from prices that were never simultaneously
available. A bhavcopy has no book at all. I added a `PriceBasis.SETTLEMENT` path
and argued it escaped that objection — a settlement price is the exchange's
official mark, struck for every contract as of one close, so simultaneity holds
by construction.

**What the data says.** On the NIFTY chain for 2026-08-26:

| | settlement == close |
|---|---|
| volume >= 500 | **100%** (423/423) |
| volume >= 100 | 99.6% |
| volume == 0 | **0%** (0/857) |

NSE sets settlement **to the close** on anything that genuinely traded, and
computes it from its own model on anything that did not. There is no third
category. So every row a fit may legitimately keep carries the last traded
price — the banned path exactly — and every row carrying a true exchange mark is
one whose price the exchange modelled. The claim was wrong, and it was wrong in
the direction that would have let the work proceed.

**The cost, measured before any model runs.** `negative_butterflies` checks
convexity in strike on the raw marks: **202 of 547 equally-spaced triples price
a negative butterfly (36.9%)**. Filtering harder barely moves it — 27.2% at
volume >= 500 — because the problem is not thin contracts, it is that adjacent
strikes last traded at different minutes with the index moving in between:

```
2026-09-01 CE 23600/23650/23700 -> 752.20/710.75/642.75   volume 700/513/2,292
```

That is not a smile with noise on it. Those three prices never existed together.

**The worst case is worse than non-convex.** `2026-10-27 PE 24800/24850/24900`
settles at `485.30/850.84/551.70` — not monotonic in strike, which a put price
must be. The middle contract traded five times all day; its neighbours traded
349 and 76. It passed `traded_only=True` because volume was not zero.
`traded_only` excludes the model-derived half and is necessary; the numbers
above are what "necessary and nowhere near sufficient" looks like.

**What this does and does not mean.**

- It is *not* a finding about the fitter. The fit inherits the arbitrage in its
  inputs. `require_arbitrage_free()` refuses the result, which is correct
  behaviour, and blaming the optimiser for it would be the wrong lesson.
- The fitted surface does admit butterfly violations inside the traded strike
  range — 12 grid points across 3 of 10 expiries. But of 820 *calendar*
  violations, only 18 fall inside the range where quotes actually exist: the
  arbitrage grid is hardcoded to k in [-1.5, 1.5], which is 0.22x to 4.5x the
  forward, while real traded strikes span roughly [-0.10, +0.11]. Reporting
  those 820 as a verdict on the fit would be reporting on extrapolation.
- So M71 Greeks, M72 structures and M75 sizing remain verified only against
  synthetic data. That was the thing this exercise was meant to change, and it
  did not change.

**What would settle it** is a chain captured with a book — bid and ask, at one
instant. That needs a live feed, which is what the recorder is for. The daily
archive can do many things; it cannot do this one, and the eight years of free
history that made M14 worth building do not help here.

`PriceBasis.SETTLEMENT` is kept, with its docstring corrected to say all of the
above, because the measurement is worth being able to reproduce. It is for
measuring what daily marks can support, not for producing a surface anything
downstream should consume.

## Every log line claimed UTC and carried IST

Found while checking something else: the Angel session cache recorded
`issued_on: 2026-08-29` while the log line written by the same call said
`2026-08-30T01:07:45Z`. One of them was wrong. The session was right.

```
real UTC now : 2026-08-29T19:38:52.327742+00:00
log says     : 2026-08-30T01:08:52.327Z
```

Exactly +5:30. `JsonFormatter` builds `ts` from
`logging.Formatter.formatTime(...) + "Z"`, and `formatTime` uses
`time.localtime` unless `converter` is set. It was never set. So the formatter
rendered local time and appended a character asserting it was UTC.

**Why it was invisible.** Three reasons, and the third is the one worth
remembering.

1. A log timestamp is not checked against anything. Nothing downstream parses
   it, so nothing could disagree with it.
2. It is self-consistent. Every line is wrong by the same amount, so ordering,
   durations and latencies between lines are all correct. Only an absolute
   comparison against a second clock reveals it — which is exactly what happened
   here, by accident.
3. **CI cannot reproduce it.** GitHub Actions runs with `TZ=UTC`, where
   `localtime` and `gmtime` are the same function. The bug is not merely
   untested there; it is *unobservable* there. A test asserting the rendered
   output would pass on CI whether the fix existed or not.

Point 3 dictated the test. `test_the_converter_is_gmtime` asserts the
*mechanism* — `JsonFormatter.converter is time.gmtime` — because that is the
only form that fails on the machine where the fault is invisible. A companion
test proves `formatTime` actually routes through `converter`, so the pin stays
load-bearing rather than decorative.

**The consequence that would have been worst.** 20:00 IST on the 29th is 14:30
UTC on the 29th; the old code wrote `2026-08-30`. Everything logged after 18:30
IST carried the next day's date, so a single trading day's logs split across two
dates. Anything correlating a decision record against its log line — which is
the audit story — would have found the evening's entries filed under tomorrow.

This is not a breach of any of the ten rules, which is part of why it lasted.
**R6** ("the clock is injected - no `datetime.now()` in library code") is the one
that should have been near it, and its scanner had nothing to see: `obs.py`
never calls `datetime.now`, it hands the problem to `logging` and inherits a
default. The failure is the same *class* as the naive-time convention enforced
in `ms_ops/rhythm.py` - "a `TypeError` on a good day and a 5.5-hour error on a
bad one" - arriving in the logs instead of the schedule, and as a wrong label
rather than a wrong type, so nothing could raise.

## The secret scanner had never run

`megaplan.md` gives M00 the verify `pre-commit run --all-files # exits 0`, and
`.pre-commit-config.yaml` opens by naming the hook that matters:

> The secret scanner is the one that matters. A broker API key committed once is
> a key that must be rotated, and if it reaches a public remote it is a key that
> is already being used by someone else.

It was the one hook that did not work. The config passes
`--baseline .secrets.baseline`, and that file did not exist, so every invocation
died in argparse:

```
detect-secrets-hook.EXE: error: argument --baseline: Invalid path: .secrets.baseline
```

**Why it was invisible.** The failure is loud and looks like configuration
noise rather than an unprotected repository. It scrolls past in a wall of hook
output, it says nothing about secrets, and the exit code is lost to whatever ran
`pre-commit` last. And nothing downstream depends on the scanner having run, so
there was no second signal - the same shape as the log timestamp above: wrong,
self-consistent, and unreferenced.

The timing is the part worth recording. This was found on the day live Angel One
credentials - a client code, a login PIN and a TOTP secret - were first put in
`.env`. `.env` is gitignored and `jobs/deploy_space.py` independently refuses to
read a token from it unless git genuinely cannot see the file, so nothing leaked.
But the guard specifically built for that hazard had been inert the whole time.

**Fixed, and then proved rather than assumed.** A scanner that runs and finds
nothing is indistinguishable from a scanner that does not run, so the fix is not
finished when the hook reports Passed:

```
repo as it stands ............ Passed
with a planted AWS key ....... Failed - Secret Type: Secret Keyword
```

The baseline stores `hashed_secret` only, never a value, so committing it does
not itself leak anything.

**And the baseline was only half the problem.** `.git/hooks/pre-commit` does not
exist - `pre-commit install` was never run in this clone. So even with a working
baseline, none of these hooks fire on `git commit`: not the secret scanner, not
`detect-private-key`, not `no-commit-to-branch`. They run only when somebody
invokes `pre-commit` by hand, which is how a repository ends up with a
carefully written hook config and no hooks. Installing them is a change to how
every future commit in this clone behaves - including blocking commits to `main`,
which is where this project's entire history sits - so it is left to the owner
rather than done quietly. Until then the scanner is armed but not wired.

**Still open, and deliberately not fixed here.** `risk-limits-need-adr` fails
too, and for a real reason: `docs/decisions/` contains a README and no ADRs at
all. The hook does not check whether an ADR covers the change - it checks
whether any ADR exists - so it blocks the first risk-limit change and every one
after it identically. That is a question about risk governance rather than about
tooling, and weakening it quietly is exactly the wrong move; it is written down
here instead. As it stands `pre-commit run --all-files` cannot exit 0, so M00's
verify block does not pass.

## `pre-commit run --all-files` leaves the tree failing `tasks.py lint`

Running M00's verify command modified sixteen files that had nothing to do with
the change in hand, and left the repository unable to pass its own lint. Two
separate causes, both version or tool skew between the hooks and the venv.

**`ruff-format` is in the hook config, and the project's formatter is black.**
`CLAUDE.md` says so plainly - "the formatter is black, not `ruff format` - they
disagree, and `lint` checks black" - and `.pre-commit-config.yaml` runs
`ruff-format` anyway, pinned at v0.5.7. That ruff predates PEP 695 support and
rewrites a generic signature into something no formatter in this repo wants:

```
-def flatten_all[P: Closeable](
+def flatten_all[
+    P: Closeable
+](
```

**The two blacks are different blacks.** pre-commit pins `black` 24.8.0; the
venv has 26.5.1, and `tasks.py lint` checks with the venv's. They format
keyword-argument calls differently, so a file formatted by the hook fails the
lint that CI runs, and a file formatted by `tasks.py fmt` would fail the hook.
Whichever one ran last wins, and the loser is whoever runs the other next.

Nothing here is subtle once seen; what makes it worth writing down is that the
verify command for M00 is the thing that causes it. Running the check leaves the
repository in a state the check's sibling rejects, and the sixteen touched files
look like real edits in `git status` - they were nearly committed alongside
genuine work, which is the actual hazard.

Reverted rather than committed. The hook config needs `ruff-format` dropped and
`black` pinned to the version `tasks.py` uses, but changing what formats every
file in the repository is not something to slip into an unrelated branch.

## The linters float, so CI and the working copy disagree by construction

The first push of this work went red on `lint + typecheck` in nineteen seconds,
having passed `python tasks.py verify` locally minutes earlier. The finding was
real - `RUF012 Mutable default value for class attribute` on a list used as a
`parametrize` source - and the local ruff simply did not have it.

    local  ruff 0.16.4  -> All checks passed
    CI     ruff 0.16.5  -> Found 1 error

One patch version apart. `pyproject.toml` asks for `ruff>=0.5` and CI runs
`pip install -e ".[dev]"` on a clean runner, so CI always resolves to whatever
was released most recently, while a working copy keeps whatever it installed on
the day it was set up. The gap widens on its own, silently, and the first
symptom is a red pipeline on a change that has nothing to do with the rule that
fired.

That is the third instance of the same shape in one session, which is what makes
it worth a section rather than a line:

| | working copy | CI / hooks |
|---|---|---|
| ruff | 0.16.4 | 0.16.5 |
| black | 26.5.1 | 24.8.0 (pre-commit) |
| formatter | black | `ruff-format` also runs (pre-commit) |

A floating lower bound is right for a library a project imports and wrong for a
tool that decides whether the build passes. `mypy>=1.10`, `black>=24.0` and
`ruff>=0.5` all carry it. Pinning them is a small change with a wide blast
radius - every contributor's next `pip install -e ".[dev]"` moves - so it is
recorded here rather than done in passing.

Local ruff was upgraded to 0.16.5 to match, and the failure now reproduces in
the working copy before it can reach a runner. That fixes today, not the
mechanism.

## M55 on real bars: HAR-RV is not beaten, and the model says so itself

`python jobs/eval_volatility.py` is the first time the volatility model has seen
a real price series. Ten years of NIFTY daily bars, 2016-08-29 to 2026-08-28,
walk-forward over four folds, HAR-RV as the baseline.

| estimator | HAR QLIKE | model QLIKE | improvement | booster kept |
|---|---|---|---|---|
| Parkinson | 0.41591 | 0.41591 | **+0.0%** | 0 of 4 folds |
| Garman-Klass | 0.36085 | 0.36106 | **-0.1%** | 1 of 4 folds |

At horizon 5, with the rows thinned to remove overlapping outcome windows, both
estimators give +0.0% and the booster is kept in 0 of 4 folds. Logged as trial
22.

**The interesting column is the last one.** `VolatilityModel` fits a residual
booster on top of HAR and keeps it only when it improves QLIKE on a held-out
split. On real data it mostly declines to keep it at all - the model's own
acceptance test is what produced the +0.0%, not a tie. That is the guard
behaving exactly as designed, and it is worth more than the headline number: an
earlier version of this project would have shipped a residual booster that
"improved" things because nothing was checking.

**HAR-RV being hard to beat is not a surprise, and that is the point.** Three
terms - yesterday, last week, last month - out-of-sample, on daily data. The
literature has said for twenty years that this is a strong baseline. What was
unknown was whether *this* implementation cleared it, and now that is measured
rather than assumed.

**What it does not say.** It does not say the module is wrong; every arithmetic
property it claims is still pinned by its tests. It says the residual booster
adds nothing on daily NIFTY bars, which is a statement about the data available
to it. The features it has are three lags of realised variance. The model was
designed to accept extra features per row (`HARRow.extra`), and none were
supplied here - so the honest reading is that HAR's own three terms exhaust what
daily OHLC carries about tomorrow's variance, and improving on that needs
information this series does not contain.

**The realised variance is itself an estimate.** Daily bars give one high, one
low, one open, one close, so Parkinson and Garman-Klass infer variance from a
range rather than summing intraday squared returns. They disagree by design
about jumps. Both were run because agreement between them is weak evidence and
disagreement would have been a finding; they agreed.

## M54 on real bars: both models under-cover, which matters more than which won

`python jobs/eval_quantiles.py` is the first time the quantile heads have seen a
real return series. Ten years of NIFTY daily bars, walk-forward over four folds,
baseline the module's own documented fallback - Gaussian quantiles whose sigma is
the trailing realised volatility scaled to the horizon. Trial 23.

| horizon | Gaussian pinball | heads pinball | improvement | Gaussian cov90 | heads cov90 |
|---|---|---|---|---|---|
| 1d | 0.002372 | 0.002424 | **-2.2%** | 82.7% | **85.4%** |
| 5d | 0.005540 | 0.005611 | **-1.3%** | 80.2% | **83.4%** |

Every head fitted; the fallback was used zero times.

**The headline is a loss, and the sub-plot is not.** The heads lose on pinball
at both horizons and are closer to nominal coverage at both. They are wider and
more honest; the Gaussian is sharper and more confident than the data supports.
Pinball is the metric this module is judged on, so it does not pass - but
reporting that as "worse" would hide the only thing the extra machinery
demonstrably bought, so the job now says both.

**The number that matters more than either model.** A 90% band that holds the
outcome 83.4% of the time is not a 90% band. Both models under-cover, at both
horizons, by 5-10 points **under this walk-forward** - and that qualifier was
missing when this was written. A later single-split run put the same heads at
89.9% and 92.6%; see "Conformal calibration" below. The numbers here are real
and they are conditional on the split:

    nominal 90%   ->   heads 85.4% / 83.4%,   Gaussian 82.7% / 80.2%

Real index returns have fatter tails than either representation carries. The
Gaussian cannot fix this in principle - it is symmetric and thin-tailed by
construction. The heads under-cover for a different reason: pinball-optimal
quantiles fitted on a sample with few tail events place the 5% and 95% heads
inside where the true tails are, and the walk-forward folds contain roughly two
years of test data each.

The consequence is downstream and specific. **M75 sizes against these
intervals.** A band 6 points too narrow makes every position slightly too large,
and the error is not uniform: it concentrates on the days the tail arrives,
which are the days a sizing mistake is expensive. That is a finding about the
whole projection layer rather than about M54, and it is the reason this
evaluation was worth running even though the model lost.

**What was not done.** The obvious response - widen the outer heads until
coverage hits 90% - was not taken. That would be tuning a number until it passes
the check that exists to catch it, which is the anti-pattern the entire
measurement layer is here to prevent. If the heads should be recalibrated, it
needs to be a design decision with its own justification, made against a
held-out period that this evaluation has not already consumed.

## What the under-coverage costs: M75 sizes 1.1x to 11x too large

The M54 evaluation above is a statement about a model. `jobs/sizing_sensitivity.py`
asks the question that makes it matter, because M75 sizes against exactly those
intervals.

**The error has two channels and they compound.** `KellyInput.p_lower` exists so
the formula runs on the pessimistic end of the model's interval rather than its
point estimate - a narrow interval makes that substitution give back less than
it was meant to. `uncertainty_modifier` reads the interval half-width and shrinks
the position as it widens - an understated width reads as a confident model and
the multiplier stays near 1.0. One makes the bet look better; the other makes it
look surer. Both push the same way.

**How much narrow.** Bisecting on a scale factor applied about the median, on
1,944 out-of-sample rows:

    heads     coverage 85.4%  ->  band must widen x1.18 to reach 90%
    Gaussian  coverage 82.7%  ->  band must widen x1.22

**What that does to size**, running Kelly x uncertainty at x1.18:

| bet | half-width | as sized | honest | too large by |
|---|---|---|---|---|
| even money, modest edge | 0.03 | 0.0347 | 0.0246 | 1.41x |
| even money, strong edge | 0.03 | 0.1560 | 0.1426 | 1.09x |
| even money, strong edge | 0.06 | 0.0880 | 0.0675 | 1.30x |
| even money, strong edge | 0.10 | 0.0222 | 0.0020 | **11.38x** |
| 2:1 payoff, modest edge | 0.03 | 0.0737 | 0.0648 | 1.14x |
| 2:1 payoff, modest edge | 0.06 | 0.0293 | 0.0163 | 1.79x |
| 1:2 payoff, high hit rate | 0.03 | 0.0607 | 0.0454 | 1.34x |

The 11x is not an outlier to be explained away, and it is not typical either. It
is what happens near the edge boundary: an 18% wider interval moves `p_used`
just far enough that Kelly nearly vanishes, so the ratio between a small number
and a much smaller one is large. The honest summary is the one the job prints -
**1.09x to 11.38x, worst where the interval is widest**, which is to say worst
exactly when the model is least sure.

**A correction I had to make to my own output.** The first version of this job
reported "in 5 of 12 cases the honest interval removes the edge entirely -
those are bets the system currently takes." That was wrong. In all five, the
narrow interval also sizes to zero: the system does not take them, and they are
not evidence of anything. Counting a case where both sides agree on no position
as a failure of one side inflates the finding, which is the exact move this
document exists to catch. The job now separates "flat either way" from "should
be flat" and reports zero of the latter.

**Nothing was changed.** Widening the heads until coverage reaches 90% would be
tuning a number until it passes the check that exists to catch it. This is a
measurement of an error that is currently live, recorded so the recalibration -
if it happens - is a design decision with its own justification and its own
held-out period rather than a reflex.

*Later correction.* The single widening factor used here averages over regimes
that are wrong in opposite directions - the band is too wide in the quiet
tercile and too narrow in the busy one - so "too large" describes the busy half
and inverts the quiet one. See "The sizing error changes sign between regimes"
below.

## M53 had no Verify block, and has now been measured against its own baseline

`direction.py` opens by calling itself "the first learned model that has to beat
the rules baseline on its own terms". It never had. The module was marked ☑ in
the tracker with 23 tests, and its section in `megaplan.md` was a single header
line: no Tasks, no Built, and **no Verify block at all**.

That is the quiet version of the failure R1 exists to prevent. R1 says a module
is not done until its Verify block passes; a module with no Verify block cannot
fail it, and gets marked done on the strength of a green test file. The tests
were good - a null test, column-order, missing-is-not-zero, isotonic
monotonicity - and none of them asks the question the module's own first
sentence asks.

**Measured, on ten years of real NIFTY bars, walk-forward over four folds.**
Trial 24.

| | log loss | brier | accuracy |
|---|---|---|---|
| M0_rules | 0.69289 | 0.24973 | 0.505 |
| M1_trees | 0.68882 | 0.24765 | 0.552 |

Delta log loss +0.00408, bootstrap CI **[-0.00067, +0.00902]**. The interval
straddles zero, so the challenger has not been shown to beat anything -
`compare()` requires the interval to exclude zero precisely so a point-estimate
win cannot ship. Brier skill 0.0024 against G5's >0.02; AUC 0.5292 against
>=0.55; reliability slope 1.024, the one criterion that passes.

**The model's own gate got there first, and that is the part worth keeping.** In
4 folds of 4, `has_discrimination()` found the booster could not rank the
held-out rows at all, so isotonic calibration was skipped rather than inventing
a shape to correct. An internal refusal, reached before any external metric.
This is now the third time a model in this repository has declined itself: M55's
residual booster refused to be kept in three folds of four, M54's heads sat
inside the true tails, and M53 cannot rank. Different mechanisms, same answer.

The consequence propagates and is correct: no discrimination means no
calibrator, which means no `p_lower`, which means `uncertainty_modifier` halves
the position for an unquantified estimate. The chain behaves properly on a model
with nothing to say. That is worth verifying once rather than assuming forever.

**What the baseline scored is the context that stops this being a story about a
strong opponent.** `M0_rules` logs 0.69289 where a coin flip is 0.69315. The
rules baseline has essentially no skill on this data either. Two models, neither
with anything to say about next-day direction.

**What this is not.** M53 is an intraday model and there is no intraday history,
so both models were fed daily analogues of the five- and thirty-minute features
`RulesBaseline` reads. G5 asks about an intraday model on intraday data and this
does not answer it. A loss is still informative - same code, same baseline, no
edge on the easier-to-sample horizon - but a win would have proved nothing, and
the job says so above its own output so the number cannot be quoted loose.

## M23: two defects the tests found in what I had just written

Both were in code that lint, mypy and my own reading had already passed, and
both were found by tests failing rather than by review.

**The docstring asserted a distance I had not measured.** It said the guard
exists because "RBI keeps the repo rate unchanged" and "RBI cuts the repo rate
by 25 bps" have a *small* Hamming distance and would otherwise be merged.
Measured, they are 27 apart against a threshold of 3, and so is every other pair
in the labelled set - 12 to 28. SimHash was never going to merge them.

The real finding is worse than the one I claimed. On headline-length text the
two distributions overlap completely:

    genuine duplicates   11, 16, 24, 26
    opposite events      12, 15, 17, 20, 21, 27, 28

There is no threshold that admits one and excludes the other. A 3-shingle set
over an eight-word headline has five members and one changed word moves three of
them, so the distance stops carrying information at that length. On a 47-word
body it behaves as intended - byline appended 2, one word changed 7 - which is
what the layer is actually for.

So `Hamming <= 3` is inherited from web-scale document dedup and does not
transfer. It is kept as the conservative end of a scale with no safe end, and
the guard is no longer described as a supplement to it: the guard is the only
thing that makes any threshold safe to loosen. The docstring now carries the
measurement instead of the assumption.

**The decisive-word list had a hole shaped like a missing plural.** It was typed
as inflections and contained "halt" and "halted" but not "halts", so:

    "NSE halts trading in XYZ Ltd"   and   "NSE resumes trading in XYZ Ltd"

were allowed to merge - a halt and its resumption becoming one event, with
whichever arrived first surviving. Worse, the neighbouring pair in the set
passed only because one of *its* two words happened to be in the table, which
looks exactly like the guard working.

Fixed by stemming rather than by adding the missing words, because a
hand-maintained inflection table is precisely the thing that rots. Two more
holes surfaced immediately from the same tests - `resumed`, `surged` and
`plunged` need the silent-e rule - and a meta-test now walks every entry in
`DECISIVE_STEMS` and asserts its own inflections reduce to it, so adding a word
without thinking about its endings fails rather than ships.

The pattern is the one this document keeps recording. Neither defect raised
anything. The first produced a module whose stated justification was false while
its behaviour was fine; the second produced a guard with a hole that fired
correctly on four of five test pairs.

## Angel historical candles: three vendor traps and one of mine

M53 is an intraday model and had only ever been measured on daily analogues,
because there is no intraday history. Angel's historical endpoint is the way to
get some. Implementing it turned up four things, and the first is mine.

**The Bearer prefix, and why it looked like an expired session.**
`smartapi-python` stores the raw token on the client and then rewrites the dict
it hands back:

```python
self.setAccessToken(jwtToken)                     # raw
user["data"]["jwtToken"] = "Bearer " + jwtToken   # prefixed
```

So the value a caller reads out of the response is *not* the value the client is
using. `AngelSession` stored the prefixed one. Feeding it back through
`setAccessToken` built `Authorization: Bearer Bearer eyJ...`, and the server
answered `Invalid Token` (AG8001) - which reads as a stale session and sends the
investigation to the credentials. It did, for a while.

The tell was that a *fresh login followed by an immediate call on the same
client* worked, while a cached session on a new client did not. That is not a
token-lifetime story; it is a token-contents story. `strip_bearer()` now
normalises at the source, so every later use adds exactly one prefix.

**Prices here are rupees. The websocket sends paise.** Confirmed against real
data rather than reasoned about:

```
RELIANCE-EQ  ['2026-08-21T00:00:00+05:30', 1314.0, 1316.0, 1308.0, 1316.0, 5434871]
```

RELIANCE trades near 1,300. In paise that row would read 131,400. The
plausibility band cannot separate the two cases - 24,180 and 241.80 are both
inside it - so `jobs/angel_history.py --cross-check` exists to compare a fetched
daily series against the ten years of NIFTY bars this project already holds from
an unrelated provider. Two sources agreeing on the level is the only evidence
available here.

**The NIFTY index has no candles; equities do.**

```
NIFTY index (26000, NSE)  -> status True, 0 rows
RELIANCE-EQ (2885, NSE)   -> status True, 6 rows
```

Note the status: **True, with nothing in it**. An empty series and a working one
are the same shape, so an index backfill would have produced a silent hole. The
front-month future is the usual proxy and its token resolves; whether it returns
candles is not yet known, because of the next item.

**The throttle answers with an empty body, not a status.** After roughly four
logins and five candle requests inside ninety seconds, every later request came
back as `b""` - no code, no message. The SDK reports that as "Couldn't parse the
JSON response received from the server", which reads as a broken response format
rather than as throttling, and it persisted for at least half an hour.

A backfill loop is exactly the shape that trips this. `HISTORY_MIN_INTERVAL`
now paces chunked requests, and the pacing is deliberately **not** a retry: the
block never says stop, so a retry loop would keep firing at silence. The
measurement is in the constant's comment so the next person does not have to
rediscover it by getting blocked.

## Conformal calibration, and the finding that the under-coverage moves with the split

M54's bands were measured at 83-85% against a nominal 90%, and M75 sizes 1.09x
to 11.38x too large as a result. The obvious response - widen until coverage
reaches 90% - reports a pass on the data it was tuned on, so the correction was
built as split conformal instead: fit on block one, choose the scale on block
two, measure on block three.

**It worked, and then it turned out there was nothing to fix on that split.**

| | walk-forward (`eval_quantiles`) | single split (`eval_conformal`) |
|---|---|---|
| horizon 1 | 85.4% | **89.9%** |
| horizon 5 | 83.4% | **92.6%** |

Same series, same model, same nominal. Under a four-fold walk-forward the bands
are short; under a 50/25/25 chronological split they are at nominal on one
horizon and *over*-covering on the other. Trials 25 and 26.

The difference is training size. The walk-forward's first fold fits on 500 rows
and predicts the next 486; the single split fits on 1,222 before predicting
anything. So a meaningful part of what was reported as a calibration defect is a
small-sample effect, and **"the coverage of M54" is not a number that exists** -
it is a number per split, and the earlier write-up should have said so.

That is a correction to this document, not to the model. The M54 entry above
reports 83-85% without qualifying it, and the sizing consequence computed from
it inherits the same conditioning.

**What conformal actually demonstrated.** On a split where the band was already
right, a scale of x1.070 moved coverage from 89.9% to 91.2% and cost 7% of
width. That is not a fix; it is the method declining to break something. The job
now says exactly that, because the first version printed "the correction holds
out of sample" while the raw band underneath it was already at nominal - the
verdict was true and the sentence was misleading.

**The problem a single scale cannot touch.** At horizon 5 the regime split is
quiet 98.0%, busy 88.1% - a 9.9-point gap that conformal leaves at 9.9. One
global multiplier moves both terciles together by construction. This is the
shape M54's own audit already recorded (97.1% quiet, 84.6% busy averaging to
something that read as calibrated), and it is the part that matters for sizing:
the busy tercile is where the tail arrives and where a short band is expensive.

Conditional coverage needs a scale that varies with the regime - Mondrian
conformal on volatility buckets, or a normalised conformity score. That is a
design decision with its own justification and is not made here.

**A float bug the tests caught on the way.** `(1.0 - 0.9) / 2` is
`0.04999999999999999`, and a forecast keyed on `0.05` does not contain it, so
deriving quantile-lookup keys by arithmetic raised on every forecast. Levels are
now matched to the nearest existing key within a tolerance, with a test that
pins both the proximity match and its refusal to match something genuinely
absent.

## Mondrian conformal: the regime gap closes, and the marginal number never saw it

The previous entry left one thing open. Global conformal fixes marginal
coverage and leaves the regime split untouched, because a single multiplier
moves every tercile together by construction. On real bars at horizon 5 that was
quiet 98.0%, busy 88.1% - a 9.9-point gap that global conformal left at 9.9.

Mondrian conformal fits one scale per volatility bucket, chosen on the
calibration block. On the same split, horizon 5:

| | quiet | mid | busy | widest gap | width | pinball |
|---|---|---|---|---|---|---|
| raw heads | 98.0% | 91.6% | 88.1% | **9.9%** | 0.06546 | 0.004607 |
| global | 98.0% | 92.1% | 88.1% | **9.9%** | 0.06600 | 0.004609 |
| mondrian | 93.6% | 92.6% | 91.1% | **2.5%** | 0.06506 | **0.004592** |

Scales x0.803 / x1.035 / x1.188 - it narrows the quiet bucket and widens the
busy one, which is the move a single scale cannot make in either direction. And
it is not bought with width: the mean band is *narrower* than the raw one and the
pinball loss slightly better. At horizon 1, where the raw gap is only 5.9%, the
improvement is correspondingly smaller: 5.9% -> 4.5%.

**The synthetic case makes the mechanism unmissable.** On data whose scale
switches by regime, a global scale lands marginal coverage at 90.8% against a
nominal 90% - a number that reads as perfectly calibrated - while the quiet
tercile sits at 100% and the busy one at 81.6%. Both wrong, opposite directions,
and the average conceals it. Mondrian: 91.2% and 92.2%. That pair is now a test.

**Why this matters for sizing rather than for scoring.** The busy tercile is
where the tail arrives. A band that is short there is short exactly when being
short is expensive, and M75 reads the interval width through two channels that
compound. An over-wide quiet band costs opportunity; a short busy band costs
money.

**Two corrections to my own output on the way there.**

The verdict logic only looked at marginal coverage, so on the horizon-5 run it
printed "NOTHING WAS BROKEN ON THIS SPLIT" while the regime gap underneath was
9.9 points. That is the same mistake the whole entry is about, made by the job
written to detect it. It now weighs both, and `MAX_REGIME_GAP` says why five
points is the line.

And the report labelled its own number "quiet-to-busy gap" while computing
max-minus-min across all three terciles - at horizon 1 the lowest is the middle
bucket, not the busy one, so the label named a comparison the code was not
making.

**What is still not addressed.** Bucket edges are fitted on the calibration
block, so a test row outside that range lands in an end bucket - an
extrapolation that is flagged but not solved. And three buckets over a 600-row
calibration block is 200 rows each, which is above the floor and not comfortably
so. Splitting a calibration set does not create information, and the refusal
below `MIN_CALIBRATION_ROWS` per bucket is the guard for that.

## The sizing error changes sign between regimes, and I reported one direction

`jobs/sizing_sensitivity.py` reported "positions 1.09x to 11.38x too large" from
a single widening factor of x1.18. Mondrian's per-bucket scales say that number
was an average over two errors pointing opposite ways.

At horizon 5 on real bars, scales x0.803 / x1.035 / x1.188:

| regime | band is | positions | worst ratio |
|---|---|---|---|
| quiet | 19.7% too **wide** | too **small** | 0.44x, and one trade missed entirely |
| mid | 3.5% too narrow | slightly large | 1.25x |
| busy | 18.8% too **narrow** | too **large** | **20.42x** |

**Only one half costs money.** An over-wide band in a quiet regime declines
positions it should have taken; that is opportunity, it never appears in a P&L,
and it is therefore the half that goes uncorrected indefinitely. An under-wide
band in a busy regime sizes up on exactly the days the tail arrives.

The single-factor version could not say this. It took a global widening factor
and applied it everywhere, so the quiet tercile's over-width was folded into the
same "too large" verdict as the busy tercile's under-width - and the direction
that was reported is the one that happened to dominate the average.

**A case the earlier job had no name for.** It separated "flat either way" from
"should be flat" - the band so narrow that the honest interval removes the edge
while the system takes the position anyway. The mirror is now labelled `MISSED`:
an over-wide band shrinking the edge to nothing, so the system sits out a trade
the honest interval would have taken. It costs nothing measurable, which is
exactly why it needed a name.

**What has not changed.** No interval is widened, no scale is applied in
production, and M75 still sizes against the raw heads. This is a measurement of
a live error with the per-regime factors taken from a calibration block that the
measurement does not include. Whether to route Mondrian scales into M75 is a
decision about the live sizing path and belongs with the risk limits, not with a
diagnostic job.
