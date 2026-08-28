# MarketSage AI — Phase Gate Record

> The spine of the project. **No entry here = no phase advance.**
> Paste the *measured numbers*, not adjectives. Sign with date + commit hash.

| Gate | Phase | Target | Measured | Date | Commit | Signed |
|------|-------|--------|----------|------|--------|--------|
| G0 | Foundation & Legal | 5 consecutive complete days recorded; compliance memo signed | — | — | — | ☐ |
| G1 | Market Data Spine | Byte-identical replay ×2; 20 sessions ≥99.5% candles; chain ≥95%; restore verified | — | — | — | ☐ |
| G2 | News & PIT Store | 30 days collected; dup rate <3%; publish-ts within 5s for ≥90% of 200 sampled | — | — | — | ☐ |
| G3 | Event Classification | macro-F1 ≥0.80; recall(sev≥7) ≥0.95; noise FP ≤2%; p95 ≤120ms; severity↔\|AR₃₀\| Spearman ≥0.35; AP-hack test passes | — | — | — | ☐ |
| G4 | Impact Library + Gap | >=15 classes with n_eff >=8; >=5 with CI excluding 0; gap CRPS beats naive by >=10%; 20 auto briefings | gap: -13.7% vs naive, calibrated (12 trials) | 2026-08-27 | — | NOT PASSED |
| **G5** | **Features + Baseline** | **M1 > M0 on log loss (bootstrap CI excl. 0); reliability slope ∈[0.85,1.15]; Brier skill >0.02; AUC ≥0.55 on high-sev; M5 beats HAR-RV on QLIKE; edge-realisation slope >0.5** | — | — | — | ☐ |
| G6 | Candle Projection | 90% coverage = 90%±3%; PIT uniform; CRPS beats RW-GARCH and implied by ≥8%; render <120ms | — | — | — | ☐ |
| G7 | Options + Simulator | ≥12 months L2: PF ≥1.2; positive in ≥3 of 4 quarters; maxDD ≤12%; deflated Sharpe >0; all 10 stress scenarios pass | — | — | — | ☐ |
| G8 | Live Paper | ≥60 sessions / ≥150 trades; positive net expectancy; PF ≥1.15; maxDD ≤10%; reliability slope ∈[0.85,1.15]; live within 1 SE of L2; zero unhandled exceptions | — | — | — | ☐ |
| R1→R4 | Live Capital Ladder | See Chapter 21 §Phase 9 rungs | — | — | — | ☐ |

## Kill criteria log

Record here every time a gate fails, what was tried, and the decision taken
(retry / pivot / stop). Three failed attempts at **G5** means the honest answer
is that no intraday directional edge exists at this data tier — pivot to
volatility-only or the overnight gap product.

| Date | Gate | Failure | Root cause | Decision |
|------|------|---------|-----------|----------|
| | | | | |

## Disabled event classes

Auto-populated by the class-level kill policy (≥25 trades, negative expectancy
over a rolling quarter). Re-enabling requires a documented model change **and** a
fresh out-of-sample period.

| Class | Disabled on | n | Expectancy | Re-enabled on | Justification |
|-------|-------------|---|-----------|---------------|---------------|
| | | | | | |

---

## G4, gap component — measured 2026-08-27, not passed

Two attempts, both in `research/experiments.jsonl` (`python jobs/experiment_log.py`).

| trial | features | CRPS | vs naive | cov90 |
|---|---|---|---|---|
| gap_v1_all_features | 28 | 0.003216 | −29.3% | 95.5% |
| gap_v2_lean_features | 10 | 0.003040 | −22.2% | 93.7% |

Baseline `naive_driver` (univariate regression on the overnight US equity futures
move): CRPS 0.002487, cov90 84.1%. Climatology floor: CRPS 0.003800, cov90 71.0%.
496 overnight windows, 2024-08-28 to 2026-08-27, 396 walk-forward test sessions.

**Finding.** The model beats climatology, so the features carry information, and
loses to the naive driver by a wide margin. Trimming 28 columns to 10 recovered
about a quarter of the shortfall and narrowed the 90% band from 3.06% to 2.40%,
which supports the overfitting explanation without rescuing the result.

The remaining gap is most likely not a tuning problem. The naive baseline
regresses on one variable that genuinely moves the Indian open; the model spends
capacity on eight series whose daily bars are, at this resolution, mostly noise
against that one signal. Daily data also forces every foreign series back to its
D−1 session, discarding the Asian hours that are legitimately knowable at 08:45.

**Not being pursued further on daily bars.** Continuing to search specifications
until the gate passes is the failure mode this ledger exists to prevent, and the
trial count above is what would make any later "improvement" honest. The gate is
re-opened when intraday data exists, which is the change that would alter the
answer rather than the estimate.

### Candlestick patterns, measured 2026-08-28 — nothing survives correction

Added because TradingView's automatic pattern detection was raised as a model.
Their own page describes patterns as things that "are used to predict future
price movements" — a statement about what traders do, not a claim that it works
— and makes no accuracy claim either way. The detector and the question of
whether the detector predicts anything are separate pieces of work, so both were
built.

Nineteen classical patterns, precise predicates rather than resemblances,
against **2 466 real NIFTY daily bars, 2016-08-29 .. 2026-08-28**.

Forward return after each pattern, measured against the **unconditional** mean
rather than against zero — an index that drifts up makes every bullish pattern
look right for free:

| horizon | patterns tested | Sidak threshold | significant |
|---|---|---|---|
| 1 bar | 19 | p < 0.00270 | **0** |
| 5 bar | 19 | p < 0.00270 | **0** |

Three would have looked like findings on their own:

| pattern | horizon | uncorrected p | after correction |
|---|---|---|---|
| `three_white_soldiers` | 5-bar | 0.0188 | not significant |
| `three_black_crows` | 5-bar | 0.0311 | not significant |
| `shooting_star` | 1-bar | 0.0483 | not significant |

Nineteen tests at an uncorrected 0.05 give a **62% chance of at least one false
positive**, which is why the correction is not optional and why the report
states the count it corrected for. A corrected p-value whose n is not stated is
an uncorrected p-value.

Several patterns carry the **opposite sign to their own claimed direction**:
`hammer` (bullish) at −0.219% on 1 bar, `bullish_engulfing` at −0.707% on 5,
`evening_star` (bearish) at +0.791%. None of those are significant either — they
are noise, and reporting them as contrarian signals would be the same error in
the other direction.

**Null check.** The same shapes with the outcomes shuffled against them: **0.05
false positives per trial over 20 shuffles, worst 1.** The pipeline finds
approximately nothing when there is nothing to find, which is what makes the
zero above worth anything.

The detector ships (`libs/ms_models/patterns.py`) because drawing patterns on a
chart is useful annotation. It does not feed the decision chain, and this table
is why.

### Update, 2026-08-28: ten years of real bars, and three more attempts

The cached series were two years deep. Refetched at ten:

| | sessions | span |
|---|---|---|
| before | 496 | 2024-08-27 .. 2026-08-27 |
| after | **2 465** | 2016-08-30 .. 2026-08-28 |

**Most of the deficit was data, not specification.** The same model, unchanged,
on five times the history:

| | CRPS | vs naive | cov90 |
|---|---|---|---|
| 2 years (previous best) | 0.002827 | **-13.7%** | 90.9% |
| 10 years | 0.002421 | **-2.4%** | 86.8% |

Eleven of the 13.7 points closed by having more of the same data. That is worth
recording plainly: the earlier conclusion named a specification problem, and
four fifths of it was sample size.

The intervals stopped being calibrated in the process, and both models are
affected - the naive baseline reads 86.5%. That is a real finding rather than a
regression: two years of data was not enough to see it.

#### Where the miscalibration actually lives

Sliced by trailing volatility, on the model's own test forecasts:

| regime | outcomes widen | model band | naive band | model cov90 | naive cov90 |
|---|---|---|---|---|---|
| quiet third | - | - | - | 89.1% | 93.8% |
| busy third | **2.39x** | **1.50x** | **1.02x** | **81.4%** | **78.5%** |

Both bands are far too rigid, and the naive one barely moves at all. The
marginal 86% hides an interval that over-covers when it is quiet and misses
badly when it is not - which is the direction that costs money, because a
position sized off it is largest exactly when the band is most wrong.

#### Three attempts on that, all logged, none passing

**gap_v7 - volatility-scaled target.** Fit quantiles of `gap / sigma_hat` and
rescale. It fixed the thing it targeted, busy-third coverage 77.5% -> 86.1%, and
cost **26% of CRPS**: a 60-session trailing sigma is noisy and lagging, and
dividing by it adds estimation variance that multiplying back does not remove.

The same transform cost the naive baseline **47.6%**. So "the scaled model beats
the scaled naive by 13.4%" is true and means nothing - it is a comparison
between two models the same transform degraded, and it is exactly the shape a
gate pass would take if nobody checked what the transform did to the baseline.

**gap_v8 - adaptive penalty. VOID, measured nothing.** The audit's own diagnosis
says a uniform penalty "taxes the one good feature exactly as hard as the
twenty-seven useless ones", so a per-feature penalty was the obvious next test.
It was faked by scaling feature columns - and `QuantileRegressor` standardises
each column by its own training sigma, undoing the scaling exactly. The results
were byte-identical to the uniform fit. Logged as an attempt because it was one.

**gap_v9 - feature subsets.** The direct version of the same question. Top-k
features by first-pass usage, ranked on the first training block only:

| k | CRPS | vs naive |
|---|---|---|
| **1** (`US_EQ_FUT_ret`) | 0.002317 | **+1.0%** |
| 2 | 0.002346 | -0.2% |
| 3 | 0.002373 | -1.4% |
| 6 | 0.002440 | -4.2% |
| 12 | 0.002369 | -1.2% |

**Skill peaks at one feature and decays monotonically.** The single-feature model
beats the naive baseline for the first time, by 1.0%.

That settles the penalty question without needing a penalty. The eleven other
features are not underweighted - they are harmful, and no regulariser reaches a
better answer than not having them. It also settles the loss-function question:
CRPS ranked every variant correctly, including preferring the sharp
miscalibrated model over the wide calibrated one, which is a proper scoring
rule doing its job.

**+1.0% against the +10% the gate asks for.** The conclusion is unchanged and
now has a number behind it: at daily resolution this problem has roughly one
variable's worth of signal in it, and the naive baseline already uses that
variable. **The gate stays open and the answer is still intraday data.**

Sixteen gap trials are now in `research/experiments.jsonl`. Pure noise tried
sixteen times produces a best Sharpe near 1.8, and every later claim about this
model is measured against that.

### Update, same day, after the model audit

`docs/model_audit.md` found that the diagnosis above was wrong in its framing.
The dominant fixable defect was not overfitting but an **inert L2 penalty**:
`l2=1e-4` shrank the weights by 0.2% against no penalty at all, so the model was
effectively unregularised no matter what the constant said.

Choosing the penalty per fit on an inner split of the training data:

| | CRPS | vs naive | cov90 |
|---|---|---|---|
| 28 features, inert penalty | 0.003216 | -29.3% | 95.5% MISCALIBRATED |
| 10 features, inert penalty | 0.003040 | -22.2% | 93.7% MISCALIBRATED |
| 10 features, selected penalty | 0.002827 | **-13.7%** | **90.9% calibrated** |

Half the deficit closed and the intervals are calibrated for the first time.
The gate still does not pass, and the reason is now specific rather than vague:
the regularisation curve plateaus where the feature weights are near zero, so
the best available version of this model uses almost nothing, while the naive
baseline uses the one variable that actually moves the open. A uniform penalty
taxes that feature as hard as the twenty-seven useless ones. Specification, not
tuning - so the gate stays shut until intraday data exists.

Twelve trials, all in `research/experiments.jsonl`.
