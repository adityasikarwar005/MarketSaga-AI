# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

`tasks.py` is the entry point for everything. It uses `.venv` directly, so no activation is needed.

```bash
python tasks.py verify        # what CI runs: lint + typecheck + test. A module is not done until this is green.
python tasks.py test          # pytest -q
python tasks.py lint          # ruff check + black --check
python tasks.py fmt           # black + ruff --fix
# The formatter is black, not `ruff format` - they disagree, and `lint` checks black.
python tasks.py typecheck     # mypy (strict on libs/, relaxed annotations on jobs/ and tasks.py)
python tasks.py status        # config health and what blocks the current gate
python tasks.py ci            # the six CI stages separately: unit, determinism, parity, leakage-lint, ap-hack, stress
# GitHub Actions runs the same stages plus lint/typecheck, an audit-chain check and infra tests.
# After pushing: `gh run watch <id> --exit-status`. The pipeline was red for four pushes once
# because nobody looked; it is not done when it is written, it is done when a run is green.
python tasks.py restore-drill # prove the restore drill still detects a damaged backup
```

Running a single test or file:

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_costs.py -q
.venv/Scripts/python.exe -m pytest tests/ml/test_no_signal.py -k "null or noise" -q
.venv/Scripts/python.exe -m pytest -m infra -q      # only the tests needing Postgres/Redis
```

The suite takes **8–11 minutes** — the pure-Python gradient booster dominates. Run it in the background rather than blocking on it. Markers: `infra` (needs Docker), `slow`, `manual`.

### Infrastructure

```bash
docker compose up -d          # TimescaleDB on 5433, Redis on 6379
pip install -e ".[infra]"     # redis, psycopg, pyarrow, duckdb, pandas
python tasks.py migrate       # hypertables and indexes
```

**Postgres is published on 5433, not 5432.** A stock Windows PostgreSQL service owns 5432 and starts at boot; connecting there reaches *that* server and reports `password authentication failed`, which reads like a wrong credential and is a wrong server. See `docs/editor-setup.md`.

Docker Desktop puts `docker.exe` in the **user** PATH, so a shell opened before installing it will not see the command.

Optional extras: `.[api]` (FastAPI, uvicorn, httpx, msgpack), `.[dev]`.

### Other jobs

```bash
python tasks.py trials        # the experiment log: how many attempts a result is one of
python tasks.py trials --verify   # is the log's hash chain intact
python tasks.py kill-drill    # monthly kill-switch drill, against a broker that fails on purpose
python jobs/backtest.py --demo --events 400
python jobs/weekly_report.py --week-ending YYYY-MM-DD --report
```

## Where the plan lives

- **`megaplan.md`** — 81 modules (M00…MA5), each with a contract and a **Verify** command. Contains the sprint build order, the per-layer progress table, and an honest snapshot of what the module count does *not* say. **Update the tracker table when a module completes** — regenerate the per-layer counts from the module rows rather than editing them by hand.
- **`docs/gates.md`** — the phase gate ledger. Gate results are recorded here whether or not they pass.
- **`docs/model_audit.md`** — the running record of every defect found, why it was invisible, and the design rationale behind each fix. Read this before changing anything in `ms_models`, `ms_options` or `ms_risk`; several decisions that look arbitrary are load-bearing and this file says why.
- **`docs/pdf/marketsage_master_plan.pdf`** — the 69-page blueprint. Chapter 12 carries the risk-limit table and the ten stress scenarios.

`README.md`'s "Status — Sprint 1" section is stale; `megaplan.md` is authoritative.

## Architecture

### The spine everything depends on

`libs/ms_core` is the floor: `config` (frozen, hashed, point-in-time), `clock` (injected `Clock`, `SessionCalendar` that refuses to guess unknown years), `schemas` (frozen pydantic types, `deterministic_ids()`), `env` (research env cannot hold credentials; live needs a double-arm), `bus` (in-memory + Redis Streams with per-topic schema validation), `obs`.

Nothing above it may hard-code a threshold — every tunable number is in `config/*.yaml`, and `Config.config_hash` is logged with decisions so a past recommendation can be reproduced with the exact settings that produced it.

### Adapters, not implementations

External systems sit behind protocols so swapping one is a one-file change: `MarketFeed`, `Broker` (`ms_broker/protocol.py` — paper and live implement the same surface, and a test asserts they have not diverged), `DailyBarProvider`, `HolidayProvider`, `CrossAssetProvider`, `Model` (`ms_models/baseline.py` — the rules baseline, a booster and an ensemble are interchangeable).

Two modules deliberately keep their **semantics in plain Python and treat the named tool as a deployment adapter**: `ms_ops/runner.py` (plan says Prefect) and `ms_ops/registry.py` (plan says MLflow). The reasoning is in each module's docstring — the part with correctness requirements has to be testable without standing up a service.

### The decision chain

```
event → ms_models (direction, volatility, ensemble, meta) → ms_options.sizing
      → ms_risk.gate → ms_broker → ms_risk.recommendation (immutable record)
      → ms_risk.ledger (prediction + outcome, one row) → ms_ops.attribution
```

`ms_sim` holds the simulator (`simulator.py`), the fill model (`fills.py`) and the stress-scenario harness (`scenarios.py`).

### The audit apparatus

This is the part that distinguishes the project, and it is easy to break by accident:

- **`research/experiments.jsonl`** — every model evaluation, hash-chained so deleting a disappointing run breaks the chain at a reportable index. Versioned in git, deliberately *not* under `build/`. Never edit it; append via `ExperimentLog.record`. A bare `ExperimentLog()` writes to this real file — correct in `jobs/`, a bug in a test, so tests must pass an explicit `tmp_path`. An autouse fixture in `tests/conftest.py` fails any test that touches it, because one test did for several sessions and the only symptom was a modified file in `git status`.
- **`research/kill_drills.jsonl`** — kill-switch drill results.
- **`data/ledger/*.ndjson`** — the audit spine. Append-only; resolution writes a *superseding* record rather than editing. The file is the source of truth and Postgres (`paper_ledger`) is an index built from it.
- **`docs/decisions/`** — ADRs. Changing a risk limit requires one; `limit_change_requires_adr` reports which changed fields have none.

## Conventions that are enforced, not suggested

The ten rules are in `README.md`. Three have teeth in the test suite:

1. **The clock is injected.** `test_no_wall_clock_calls_outside_clock_module` scans `libs/` and `services/` for `datetime.now(` and `time.time(`. This has caught real bugs twice — an API route that would have marked every prediction in a replayed day overdue, and a retrain that would have stamped fifty-two trials with one afternoon's date. Take a `Clock`; `RealClock()` is a fine default argument.
2. **Point-in-time or it does not exist.** Every record carries `t_event`, `t_ingest`, `t_available`; the replay engine releases nothing before `t_available`, and `Simulator.book_for_fill` raises `LookaheadError` if asked for a price from before its own decision.
3. **Schedule times are exchange-local IST and timezone-aware.** `ms_ops/rhythm.py` compares them against a UTC clock; a naive value is a `TypeError` on a good day and a 5½-hour error on a bad one.

## The testing discipline

Ordinary tests ask whether a model works when there *is* a signal. This codebase also requires **null tests**: what does it report when there is nothing to find? `tests/ml/test_no_signal.py` holds one per known defect class.

Every fitted component must have one. The pattern that recurs across two full audits: **each defect produced a working pipeline, plausible numbers, and no error.** An inert regularisation constant looks like regularisation; a 30% false-positive rate looks like a finding; a gamma of 0.5 looks like a measurement; a theta sign error hid under eighty-two passing tests.

When adding a model:
- give it a null test before believing any result from it
- pin arithmetic against a closed form or a finite difference where one exists
- assert determinism — two fits on the same input must be identical
- if you search over a hyperparameter, do it on an **inner** split and log the trial

Tests are written to fail loudly with a sentence explaining what the failure means, not just what value was wrong. Match that style.

## Reporting results

Never report a gate as passing that did not pass. Gate G4 currently does **not** pass (−13.7% vs a naive baseline on real daily bars) and that is recorded in `docs/gates.md` with all twelve trials named. Tuning until a gate passes is the anti-pattern the whole measurement layer exists to prevent.

Exactly one component has ever run on **real** market data (the overnight gap forecaster, on free daily bars). Everything else has run on synthetic data where the answer was known in advance. Be precise about which is which when describing status.

## Git

Commits are authored by the repository owner. **Never add Claude, Anthropic, `Co-Authored-By`, or a session trailer to any commit message or author field** — this was an explicit instruction and it applies to every commit.
