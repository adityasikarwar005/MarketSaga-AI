# MarketSage AI

Event-driven, distributional forecasting for Indian index derivatives.

Detect a market-moving event within milliseconds, estimate the **probability
distribution** of Nifty / Bank Nifty returns over the next 1–30 minutes
conditioned on that event and current market state, convert it into an optimal
options structure, size under a strict risk budget, and score its own
calibration in public.

> Not investment advice. Every performance number in this repository is a design
> target subject to falsification, not a promise.

---

## Documents

| File | What it is |
|---|---|
| `docs/pdf/marketsage_master_plan.pdf` | The blueprint — 69 pages. Architecture, ML strategy, risk, roadmap. **Read Chapter 3 (Reality Calibration) first.** |
| `docs/pdf/part04..10.pdf` | The same content split into standalone parts |
| `megaplan.md` | The working plan — 79 modules, each with a contract and a verify command |
| `docs/gates.md` | Phase gate ledger. No entry here, no phase advance. |

---

## Quick start

```bash
python tasks.py setup     # .venv + dependencies
python tasks.py test      # suite (infra tests skip if Docker is down)
python tasks.py status    # what still blocks Gate G0
```

Optional, once Docker Desktop is installed:

```bash
python tasks.py up        # Postgres/TimescaleDB + Redis
```

---

## Layout

```
config/     every tunable number, versioned and hashed. No magic numbers in code.
libs/       ms_core, ms_data, ms_news, ms_events, ms_impact, ms_features,
            ms_models, ms_options, ms_risk, ms_sim, ms_broker
services/   long-running processes (ingest, engines, api)
jobs/       scheduled and one-shot jobs
tests/      unit · integration · replay · parity · stress · ml
data/       captured market and news data. Never committed.
docs/       blueprint (LaTeX + PDF), gates, architecture decision records
```

---

## The ten rules

1. A module is not done until its `Verify` block passes.
2. One owner concept per module.
3. Modules talk through schemas, never by importing internals.
4. **Point-in-time or it does not exist** — every record carries `t_event`, `t_ingest`, `t_available`.
5. No magic numbers; all thresholds in `config/`, hash logged with every decision.
6. The clock is injected — no `datetime.now()` in library code.
7. Any order-placing function takes `live: bool = False`.
8. Build the boring module before the fun one.
9. When in doubt, emit nothing. No-trade is the correct default.
10. Write the verify test before the implementation.

---

## Status — Sprint 1

| Module | | Verify |
|---|---|---|
| M00 repo skeleton | ◐ | `pytest tests/unit/test_smoke.py` — infra parts pending Docker |
| M01 config registry | ☑ | `pytest tests/unit/test_config.py` |
| M02 clock & calendar | ☑ | `pytest tests/unit/test_clock.py` |
| M03 schemas | ☑ | `pytest tests/unit/test_schemas.py` |
| M06 env isolation | ☑ | `pytest tests/unit/test_env_isolation.py` |
| M10 instrument master | ◐ | reference data loads; **not yet verified against the exchange** |
| M11 market feed | ☐ | needs a broker decision |
| M12 recorder | ☐ | **the next thing that matters** |

`python tasks.py status` prints the live version of this list.

---

## What matters right now

Everything in Sprint 1 exists to make **M12, the recorder**, correct and
restartable. Models can be rebuilt in a weekend. History cannot be recovered at
any price, and the binding constraint on whether this project ever passes Gate G5
is how many event-conditioned samples were captured before then.

Every day the recorder is not running is a day of that constraint getting worse.
