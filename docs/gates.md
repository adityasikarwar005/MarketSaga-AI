# MarketSage AI — Phase Gate Record

> The spine of the project. **No entry here = no phase advance.**
> Paste the *measured numbers*, not adjectives. Sign with date + commit hash.

| Gate | Phase | Target | Measured | Date | Commit | Signed |
|------|-------|--------|----------|------|--------|--------|
| G0 | Foundation & Legal | 5 consecutive complete days recorded; compliance memo signed | — | — | — | ☐ |
| G1 | Market Data Spine | Byte-identical replay ×2; 20 sessions ≥99.5% candles; chain ≥95%; restore verified | — | — | — | ☐ |
| G2 | News & PIT Store | 30 days collected; dup rate <3%; publish-ts within 5s for ≥90% of 200 sampled | — | — | — | ☐ |
| G3 | Event Classification | macro-F1 ≥0.80; recall(sev≥7) ≥0.95; noise FP ≤2%; p95 ≤120ms; severity↔\|AR₃₀\| Spearman ≥0.35; AP-hack test passes | — | — | — | ☐ |
| G4 | Impact Library + Gap | ≥15 classes with n_eff ≥8; ≥5 with CI excluding 0; gap CRPS beats naive by ≥10%; 20 auto briefings | — | — | — | ☐ |
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
