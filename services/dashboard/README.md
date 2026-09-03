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

## Three pages

`index.html` is the overview: what the system is, the five models measured on
real data, the gate ledger with G4 and G7 failing, and the machinery that makes
a negative result trustworthy. It exists because landing directly in a dense
instrument panel with no context tells a first-time reader nothing.

`desk.html` is the workspace itself. It is deliberately dense and stays that
way; the front page is what carries the air and the explanation.

`chart.html` is the chain, and it is the one page here built on **real market
data throughout**. Every call and every put listed on NIFTY and BankNifty - every
strike, every live expiry, weeklies out to the far-dated contracts nobody trades
- with real one-minute candles from the exchange's own history for each one.

It is read the way an Indian chain is read: calls left, strikes down the centre,
puts right, expiries on a rail above. Picking a rung opens its call and its put
together, because a strike is one position seen from two directions and they move
against each other.

Two things on it are drawn to be impossible to confuse with the candles. The
**forecast** is violet - a colour no traded price uses - hollow, and a bar a
minute for the next hour; it is not a price that happened and it is not a range
anything traded in, it is the 10th to 90th percentile of a distribution. Its only
mechanism is time decay: each contract's implied volatility is inverted from its
own price against the future's price from the same minute, and the forward is
then diffused with no drift, because nothing in this repository has earned a view
on direction. What that is worth is measured on the same day it is drawn on,
per expiry, against the only baseline that matters - "it will be what it is now" -
and whatever the comparison says is printed under the chart, including the days
it loses.

And **an expiry with no future behind it gets no forecast at all**. NIFTY lists
options to 2031 and futures three months out, so most expiries have nothing to
price against; extending the last future out three years would be assumed
dividends dressed as a traded price. Those keep their ladder and their real
candles and are marked on the rail.

## Running it locally

```bash
python tasks.py dashboard      # http://127.0.0.1:8787
```

The workspace reads a session snapshot from `data.js`, so it also opens straight
from the filesystem. It is not wired to a live feed.
