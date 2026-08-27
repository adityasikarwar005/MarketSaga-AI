# MarketSage AI — Phase Gate Record

> The spine of the project. **No entry here = no phase advance.**
> Paste the *measured numbers*, not adjectives. Sign with date + commit hash.

| Gate | Phase | Target | Measured | Date | Commit | Signed |
|------|-------|--------|----------|------|--------|--------|
| G0 | Foundation & Legal | 5 consecutive complete days recorded; compliance memo signed | — | — | — | ☐ |
| G1 | Market Data Spine | Byte-identical replay ×2; 20 sessions ≥99.5% candles; chain ≥95%; restore verified | — | — | — | ☐ |
| G2 | News & PIT Store | 30 days collected; dup rate <3%; publish-ts within 5s for ≥90% of 200 sampled | — | — | — | ☐ |
| G3 | Event Classification | macro-F1 ≥0.80; recall(sev≥7) ≥0.95; noise FP ≤2%; p95 ≤120ms; severity↔\|AR₃₀\| Spearman ≥0.35; AP-hack test passes | — | — | — | ☐ |
| G4 | Impact Library + Gap | >=15 classes with n_eff >=8; >=5 with CI excluding 0; gap CRPS beats naive by >=10%; 20 auto briefings | gap: -22.2% vs naive (best of 2 trials) | 2026-08-27 | — | NOT PASSED |
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
