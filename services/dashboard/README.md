---
title: MarketSage Desk
emoji: 🎛️
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Options research that reports what it found, losses included
---

# MarketSage Desk

The monitoring workspace for an event-driven options research system for Indian
index derivatives. Event stream, a forecast distribution drawn against the
option-implied move, book Greeks, candidate structures, and the gate ledger.

## Read this before reading the numbers

**Every figure on this page came from synthetic inputs whose answers were known
in advance.** They were produced by the actual pipeline — the SVI surface
fitter, the path sampler, the quantile heads, the Greeks engine, the structure
enumerator — but on generated data, not on a market.

That establishes the arithmetic and says nothing about edge. The fitted surface
parameters return to six decimals and the calibrated paths land on the quantile
heads to five, which means the code computes what it claims to compute. It does
not mean the system makes money.

**Five components have ever consumed real market data, and exactly one *model*
has been measured on it**: the overnight gap forecaster, on ten years - 2,465
sessions - of free daily bars. Measured against a naive baseline
that regresses on a single variable, it is **2.4% worse**. Its direction hit
rate is 74%, of which 67 points come from the index simply gapping up more often
than down — the model contributes about seven, and transaction costs consume
them.

The gate ledger at the bottom of the page shows **G4 and G7 failing**. That is
not a work-in-progress note; it is the current measured state, and it is on the
page because a dashboard that only displays successes is a dashboard nobody can
use to decide anything.

## What is on the page

- **Event stream** — seven events, four of them suppressed and all four still
  listed with the specific rule that stopped them. A stream that shows only what
  was acted on cannot answer the question people ask after a bad day.
- **Forecast vs implied** — per-minute quantile bands with the option-implied
  one-sigma cone laid under them. Hover for a crosshair and a live readout.
- **Every derived number is hoverable** and explains how it was computed, not
  what it is called.
- **PAPER indicator** — structural rather than a badge. No order in this system
  has ever reached a broker.

## Two pages

`index.html` is the overview: what the system is, the five models measured on
real data, the gate ledger with G4 and G7 failing, and the machinery that makes
a negative result trustworthy. It exists because landing directly in a dense
instrument panel with no context tells a first-time reader nothing.

`desk.html` is the workspace itself. It is deliberately dense and stays that
way; the front page is what carries the air and the explanation.

## Running it locally

```bash
python tasks.py dashboard      # http://127.0.0.1:8787
```

The workspace reads a session snapshot from `data.js`, so it also opens straight
from the filesystem. It is not wired to a live feed.
