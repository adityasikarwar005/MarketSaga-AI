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
