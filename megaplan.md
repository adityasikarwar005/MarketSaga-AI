# MarketSage AI — MEGAPLAN

> **Working execution document.** The LaTeX blueprint in `docs/` says *what to build and why*.
> This file says *what to build next, in what order, and how to prove it works*.
>
> Companion files: `docs/pdf/marketsage_master_plan.pdf` (strategy) · `docs/gates.md` (phase sign-off)

---

## 0. Rules of engagement

Read these once. They are the difference between a system and a pile of scripts.

| # | Rule | Why |
|---|------|-----|
| R1 | **A module is not done until its `Verify` block passes.** Not "it runs" — the verify command exits 0. | Prevents the 80%-done graveyard |
| R2 | **Every module has exactly one owner concept.** If you can't state it in one line, split it. | Keeps interfaces clean |
| R3 | **Modules talk through schemas, never by importing each other's internals.** | Lets you rewrite any module without touching the rest |
| R4 | **Point-in-time or it doesn't exist.** Every record carries `t_event`, `t_ingest`, `t_available`. | This single rule prevents the most expensive class of bug |
| R5 | **No magic numbers.** All thresholds in `config/*.yaml`, hash logged with every decision. | Reproducibility |
| R6 | **Clock is injected, never `datetime.now()` in library code.** | The simulator and production must share one code path |
| R7 | **Any order-placing function takes `live: bool = False`.** | The one bug you cannot afford |
| R8 | **Build the boring module before the fun one.** Recorder before model. Baseline before transformer. | See kill criteria at G5 |
| R9 | **When in doubt, emit nothing.** No-trade is the correct default output. | |
| R10 | **Write the verify test before the implementation.** | Forces you to define "working" |

### Status legend
`☐` not started · `◐` in progress · `☑` verified · `⊘` deferred/descoped · `✖` blocked

---

## 1. System design — the module map

```mermaid
graph TD
    subgraph L0["L0 · Foundation"]
        M00[M00 repo skeleton]
        M03[M03 schemas]
        M04[M04 event bus]
        M02[M02 clock]
        M05[M05 observability]
    end

    subgraph L1["L1 · Market Data"]
        M11[M11 market feed]
        M13[M13 candle builder]
        M14[M14 chain snapshotter]
        M17[M17 parquet + duckdb]
        M18[M18 replay engine]
        M19[M19 data QC]
    end

    subgraph L2["L2 · News & Events"]
        M21[M21 collectors]
        M23[M23 dedup]
        M25[M25 PIT event store]
        M27["M27 classifier ⚙AI"]
        M29[M29 novelty + confirm]
    end

    subgraph L3["L3 · Impact"]
        M31[M31 event study]
        M32["M32 Bayesian priors ⚙AI"]
        M34[M34 live impact scorer]
    end

    subgraph L4["L4 · Features"]
        M40[M40 feature registry]
        M43[M43 parity + leakage lint]
        M44[M44 feature groups A-F]
        M45["M45 regime model ⚙AI"]
    end

    subgraph L5["L5 · Models"]
        M50[M50 labels]
        M51[M51 CV harness]
        M53["M53 direction ⚙AI"]
        M55["M55 volatility ⚙AI"]
        M56["M56 IV change ⚙AI"]
        M59["M59 ensemble + calibration ⚙AI"]
    end

    subgraph L6["L6 · Projection"]
        M61[M61 fan chart]
    end

    subgraph L7["L7 · Decision"]
        M70[M70 IV surface]
        M73[M73 monte carlo PnL]
        M76[M76 risk gate]
        M77[M77 recommendation]
    end

    subgraph L8["L8 · Execution"]
        M82[M82 L2 simulator]
        M80[M80 paper broker]
        M84[M84 live broker]
    end

    subgraph L9["L9 · Surface"]
        M91[M91 dashboard]
        M93[M93 ledger]
        M94[M94 attribution]
        M95[M95 gap product]
    end

    M00 --> M03 --> M04
    M02 --> M04
    M04 --> M11 & M21
    M11 --> M13 --> M17 --> M18
    M11 --> M14 --> M70
    M13 --> M19
    M21 --> M23 --> M25 --> M27 --> M29
    M25 --> M31 --> M32 --> M34
    M29 --> M34
    M13 & M14 & M34 --> M40 --> M44 --> M43
    M44 --> M45 & M50
    M50 --> M51 --> M53 & M55 & M56
    M53 & M55 --> M59 --> M61
    M59 & M70 --> M73 --> M76 --> M77
    M77 --> M80 --> M82
    M77 --> M91 & M93 --> M94
    M31 --> M95
    M82 --> M84
    M05 -.-> M91
```

### Design invariants

| Invariant | Enforced by |
|---|---|
| No module reads another module's database tables directly | Code review + `libs/` boundaries |
| Every bus message validates against a versioned schema | M03 + M04 runtime validation |
| Offline and online feature values are identical | M43 parity test in CI |
| A recommendation can be traced back to the raw news item | M05 trace-id propagation |
| Replaying a day produces byte-identical output | M18 determinism test |
| Nothing can place a real order except M84, and only when `live=True` | M76 risk gate + code convention R7 |

---

## 2. Module index

Phase column maps to the gates in `docs/gates.md`. Build strictly top-down within a phase.

| ID | Module | Layer | Phase | Depends on | AI/ML | Status |
|----|--------|-------|-------|-----------|:-----:|:------:|
| **M00** | Repo skeleton & Docker stack | L0 | P0 | — | | ☑ |
| **M01** | Config registry (YAML + hashing) | L0 | P0 | M00 | | ☑ |
| **M02** | Clock & session calendar | L0 | P0 | M00 | | ☑ |
| **M03** | Schemas & contracts | L0 | P0 | M00 | | ☑ |
| **M04** | Event bus (Redis Streams) | L0 | P0 | M03 | | ☑ |
| **M05** | Observability (logs, traces, metrics) | L0 | P0 | M00 | | ☑ |
| **M06** | Secrets & environment isolation | L0 | P0 | M00 | | ☑ |
| **M10** | Instrument master & calendars | L1 | P0 | M01 | ✅ | ☑ |
| **M11** | Market feed adapter | L1 | P0 | M04, M10 | | ☑ |
| **M12** | Raw tick recorder | L1 | P0 | M11 | | ☑ |
| **M13** | Candle builder | L1 | P1 | M12, M02 | | ☑ |
| **M14** | Option chain snapshotter | L1 | P0 | M11 | | ☑ |
| **M15** | Cross-asset collector | L1 | P1 | M04 | | ☑ |
| **M15b** | Daily bar loader (free source) | L1 | P4 | M15 | | ☑ |
| **M16** | TimescaleDB schema & loaders | L1 | P1 | M03 | | ☑ |
| **M17** | Parquet archive + DuckDB research layer | L1 | P1 | M16 | | ☑ |
| **M18** | Replay engine | L1 | P1 | M17, M02, M04 | | ☑ |
| **M19** | Data quality gates | L1 | P1 | M16 | | ☑ |
| **M20** | Source registry & credibility weights | L2 | P2 | M01 | ✅ | ☑ |
| **M21** | Collector framework | L2 | P2 | M04, M20 | ✅ | ☑ |
| **M22** | Normaliser & publish-time resolver | L2 | P2 | M21 | | ☐ |
| **M23** | Deduplication | L2 | P2 | M22 | ⚙ | ☐ |
| **M24** | Economic calendar & consensus | L2 | P2 | M21 | | ☐ |
| **M25** | Point-in-time event store | L2 | P2 | M22, M16 | | ☐ |
| **M26** | Relevance gate (tier 1) | L2 | P3 | M25 | ⚙ | ☐ |
| **M27** | Event classifier (tier 2) | L2 | P3 | M26 | ⚙ | ☐ |
| **M28** | Entity/geo extraction & KG routing | L2 | P3 | M27 | ⚙ | ☐ |
| **M29** | Novelty, confirmation, credibility fusion | L2 | P3 | M27, M13 | ⚙ | ☐ |
| **M30** | Abnormal-return engine | L3 | P4 | M13, M17 | | ☑ |
| **M31** | Event-study runner | L3 | P4 | M30, M25 | | ☑ |
| **M32** | Hierarchical Bayesian impact priors | L3 | P4 | M31 | ⚙ |☑ |
| **M33** | Cross-asset transfer estimator | L3 | P4 | M15, M30 | ⚙ |☑ |
| **M34** | Live impact scorer | L3 | P4 | M32, M29 | | ☐ |
| **M35** | Overnight gap forecaster | L3 | P4 | M33, M31 | ⚙ |☑ |
| **M35a** | Model audit: null tests for every fitted component | L5 | P4 | M32, M33, M35 | | ☑ |
| **M40** | Feature registry & framework | L4 | P5 | M03, M17 | | ☑ |
| **M41** | Offline materialiser | L4 | P5 | M40 | | ☑ |
| **M42** | Online feature server | L4 | P5 | M40, M04 | | ☑ |
| **M43** | Parity test & leakage linter | L4 | P5 | M41, M42 | | ☑ |
| **M44** | Feature groups A–F | L4 | P5 | M43 | | ◐ |
| **M45** | Regime model | L4 | P5 | M44 | ⚙ | ☐ |
| **M50** | Label builder (triple barrier) | L5 | P5 | M13, M44 | | ☑ |
| **M51** | CV harness (purged / embargo / CPCV) | L5 | P5 | M50 | | ☑ |
| **M52** | M0 rules baseline | L5 | P5 | M34, M44 | | ☑ |
| **M53** | Direction model (boosted trees + isotonic) | L5 | P5 | M51 | ⚙ | ☑ |
| **M54** | Quantile model | L5 | P6 | M51 | ✅ | ☑ |
| **M55** | Volatility model (HAR + boosted residual) | L5 | P5 | M51 | ⚙ | ☑ |
| **M56** | IV-change model | L5 | P7 | M51, M70 | ⚙ | ☐ |
| **M57** | Meta-labeller (post-cost, threshold from loss ratio) | L5 | P5 | M53, M74 | ⚙ | ☑ |
| **M58** | Sequence models (TCN / transformer) | L5 | P6 | M51, M13 | ⚙ | ⊘ |
| **M59** | Ensemble, calibration, conformal | L5 | P5 | M53, M55 | ⚙ | ☑ |
| **M5A** | Model registry & ONNX serving (stages + gate done; MLflow/ONNX deferred) | L5 | P5 | M59 | | ◐ |
| **M60** | Path sampler | L6 | P6 | M55, M54 | ✅ | ☑ |
| **M61** | Fan-chart producer | L6 | P6 | M60 | ✅ | ☑ |
| **M62** | Projection scorer (pinball/CRPS/PIT) | L6 | P6 | M61 | ✅ | ☑ |
| **M70** | IV surface fitter | L7 | P7 | M14 | ✅ | ☑ |
| **M71** | Greeks engine | L7 | P7 | M70 | ✅ | ☑ |
| **M72** | Structure enumerator & liquidity filter | L7 | P7 | M71 | ✅ | ☑ |
| **M73** | Monte Carlo P&L engine | L7 | P7 | M72, M60, M56 | | ☐ |
| **M74** | Cost model (PIT rates, STT bases, slippage) | L7 | P7 | M01 | | ☑ |
| **M75** | Position sizing (fractional Kelly on a lower bound) | L7 | P7 | M59, M74 | | ☑ |
| **M76** | Risk gate & kill switches ⭐ | L7 | P7 | M75 | | ☑ |
| **M77** | Recommendation assembler | L7 | P7 | M76 | | ☑ |
| **M80** | Paper broker adapter | L8 | P7 | M77 | | ☑ |
| **M81** | Fill & slippage model | L8 | P7 | M80 | ⚙ | ☑ |
| **M82** | L2 event-driven simulator ⭐ | L8 | P7 | M18, M81 | | ☑ |
| **M83** | Order state machine & reconciliation | L8 | P8 | M80 | ✅ | ☑ |
| **M84** | Live broker adapter | L8 | P9 | M83, M76 | | ☐ |
| **M85** | Stress-scenario suite | L8 | P7 | M82 | | ☑ |
| **M90** | API gateway & WebSocket | L9 | P8 | M04 | | ☑ |
| **M91** | Dashboard UI | L9 | P8 | M90 | ⚙ | ◐ |
| **M92** | Alerting | L9 | P8 | M90 | | ☑ |
| **M93** | Paper ledger & outcome resolver | L9 | P8 | M80 | | ☑ |
| **M94** | Attribution & calibration reports | L9 | P8 | M93 | | ☑ |
| **M95** | Morning briefing product | L9 | P4 | M35 | ✅ | ☑ |
| **MA0** | Orchestration (Prefect) | LX | P5 | M00 | | ☑ |
| **MA1** | Daily rhythm jobs | LX | P8 | MA0 | | ☑ |
| **MA2** | Retraining pipeline | LX | P8 | M5A, MA0 | ⚙ | ☑ |
| **MA3** | Drift monitors | LX | P8 | M94 | ✅ | ☑ |
| **MA4** | CI pipeline | LX | P1 | M18, M43 | ✅ | ☑ |
| **MA5** | Backup & restore | LX | P1 | M16 | ✅ | ☑ |

**Count:** 79 modules · 20 carry AI/ML · 16 model cards in section 4 · every module has a verify block below.

---

## 3. Module specifications

Each spec is deliberately short. The pattern is always the same:
**Owns → Contract → Tasks → Verify → Done → Traps.**

---

### L0 · FOUNDATION

#### M00 · Repo skeleton & Docker stack
**Owns:** the shell everything else lives in.
**Contract:** `docker compose up` yields Postgres+TimescaleDB, Redis, and a health-checked service stub.

**Tasks**
- [ ] Monorepo layout per blueprint Ch.18 (`libs/`, `services/`, `jobs/`, `tests/`, `config/`)
- [ ] `pyproject.toml`, one virtualenv, pinned lockfile
- [ ] `docker-compose.yml`: timescaledb, redis, adminer (dev only)
- [ ] Pre-commit: ruff, black, mypy (libs only), **secret scanner**
- [ ] `Makefile` / `tasks.py`: `up`, `down`, `test`, `lint`, `record`

**Verify**
```bash
docker compose up -d && sleep 10
pytest tests/unit/test_smoke.py       # asserts DB + Redis reachable
pre-commit run --all-files            # exits 0
```
**Done:** clean machine → clone → `make up` → smoke test green in under 5 minutes.
**Traps:** don't add Kubernetes. Don't add a web framework yet.

---

#### M01 · Config registry
**Owns:** all tunable numbers, with provenance.
**Contract:** `Config.load()` → frozen object + `config_hash` string logged with every decision.

**Tasks**
- [ ] `config/{instruments,costs,risk_limits,sources,taxonomy,models}.yaml`
- [ ] Pydantic models for each; fail loudly on unknown keys
- [ ] `config_hash` = sha256 of the resolved config
- [ ] Cost tables are **dated** — `effective_from` per row, resolved point-in-time

**Verify**
```bash
pytest tests/unit/test_config.py -k "hash_stable or rejects_unknown_key or cost_is_pit"
grep -rn "0\.\|[0-9]\{2,\}" libs/ --include=*.py | grep -v test | grep -vE "^\s*#"  # review for magic numbers
```
**Done:** changing any threshold requires touching only YAML.

---

#### M02 · Clock & session calendar
**Owns:** what time it is, and whether the market is open.
**Contract:** `Clock.now()`, `Session.state(ts) -> PREOPEN|OPEN|CLOSED|HOLIDAY`, `Session.minutes_since_open(ts)`.

**Tasks**
- [ ] `RealClock` and `SimClock` behind one interface — **injected everywhere**
- [ ] Exchange timezone handling; store UTC, present exchange time
- [ ] Holiday + expiry calendar loaded from `config/instruments.yaml`
- [ ] Special sessions (muhurat, extended) flagged, never mixed into stats
- [ ] NTP drift monitor → raises above 50 ms

**Verify**
```bash
pytest tests/unit/test_clock.py    # DST-free but timezone-correct, holiday edges, 09:15:00.000 boundary
```
**Done:** no `datetime.now()` anywhere outside `RealClock`.
```bash
grep -rn "datetime.now\|time.time()" libs/ services/ | grep -v "clock.py" | wc -l   # must be 0
```

---

#### M03 · Schemas & contracts
**Owns:** the shape of every message. This is the most important file in the repo.

**Contract:** versioned Pydantic/msgspec models, one module, importable by all.

Core types:
```
Tick, Candle, OptionQuote, ChainSnapshot,
RawNewsItem, ClassifiedEvent, ImpactEstimate,
FeatureSnapshot, Prediction, ProjectionCone,
OptionStructure, Recommendation, Order, Fill, Outcome
```
Every one carries: `schema_version`, `trace_id`, `t_event`, `t_ingest`, `t_available`.

**Tasks**
- [ ] Define all types with explicit units in field names (`ltp_paise`? no — document units in docstring)
- [ ] Serialisation: msgpack for bus, JSON for storage/debug
- [ ] Version bump rules: additive = minor, semantic change = major + new topic
- [ ] Golden-file tests: a serialized sample of each type, checked into `tests/fixtures/`

**Verify**
```bash
pytest tests/unit/test_schemas.py -k "roundtrip or golden or required_timestamps"
```
**Traps:** resist "just add a dict field". Once `extra: dict` exists, the contract is gone.

---

#### M04 · Event bus
**Owns:** ordered, durable, replayable message transport.
**Contract:** `bus.publish(topic, msg)` / `bus.subscribe(topic, group) -> iterator`.

Topics: `raw.news` · `event.classified` · `market.tick` · `market.candle` · `market.chain` · `feature.snapshot` · `signal.pred` · `trade.reco` · `order.*` · `audit.*`

**Tasks**
- [ ] Redis Streams wrapper: consumer groups, ack, claim-stale, max-len trimming
- [ ] Schema validation on publish **and** consume
- [ ] Backpressure: bounded local queues, drop policy per topic (never drop `order.*`)
- [ ] `bus.replay(topic, from_ts, to_ts)` for the simulator

**Verify**
```bash
pytest tests/unit/test_bus.py -k "ordering or at_least_once or consumer_group_resume or replay_matches"
```
**Done:** kill a consumer mid-stream, restart, no message lost, none processed twice without ack.

---

#### M05 · Observability
**Owns:** the ability to answer "why did it do that?" six weeks later.

**Tasks**
- [ ] Structured JSON logging, one logger factory, `trace_id` in every record
- [ ] `trace_id` generated at news ingest / tick arrival, propagated through **every** stage
- [ ] Prometheus metrics: per-stage latency histogram, throughput, error rate, queue depth
- [ ] Grafana dashboards: golden signals + business alarms
- [ ] OpenTelemetry spans on the hot path

**Verify**
```bash
pytest tests/unit/test_tracing.py   # one synthetic news item -> assert trace_id present at 6 stages
curl -s localhost:9090/metrics | grep marketsage_stage_latency_seconds
```
**Done:** given a `reco_id`, one query returns the full chain: headline → classification → features → prediction → recommendation.

---

#### M06 · Secrets & environment isolation
**Tasks**
- [ ] `.env` never committed; secret scanner in pre-commit (M00)
- [ ] Four environments: `research` / `replay` / `paper` / `live` with separate credentials
- [ ] **`research` env has no broker credentials at all** — enforced by config loader
- [ ] `live` requires an explicit env var *and* a file on disk to arm

**Verify**
```bash
pytest tests/unit/test_env_isolation.py -k "research_cannot_reach_broker or live_requires_double_arm"
```

---

### L1 · MARKET DATA

#### M10 · Instrument master & calendars ✅
**Owns:** the canonical identity of every tradable thing.

**Tasks**
- [x] `instrument` table: id, symbol, exchange token, type, lot size, tick size, expiry, strike, underlying
- [x] Expiry calendar (weekly + monthly), holiday calendar
- [x] Symbol → id resolution with historical validity intervals (symbols get reused!)

**Built** `libs/ms_data/instruments.py`, `tests/unit/test_instruments.py` (35 tests).
The holiday calendar was already in `ms_data/holidays.py` and the lot-size history
in `Instrument.lot_size_on`; what was missing was contract identity.

The trap above is stated for lot sizes and applies more sharply to the symbols
themselves. A stale lot size scales a number; a reused symbol attaches it to the
wrong contract entirely. So `resolve_at_date` takes a date and there is no
overload without one — a default of "today" silently re-points every historical
lookup at whatever currently holds the string, and the answer is still a real
contract with a real price.

The indexes are **lists, not dicts**. A dict keyed by tradingsymbol is the
structure that makes reuse invisible: the second contract overwrites the first,
the master reports one entry, and every earlier trade resolves to the later
contract without anything failing. Exchange tokens get the same treatment —
brokers recycle them — which is why stored records reference `instrument_id`,
the one identifier that may never be reused.

Expiry rolls **backwards** off a holiday, never forwards. Forwards settles a
contract after the date printed on it, and a daily-close backtest never shows
it: the position simply appears to survive a day longer than it could have.

Mutation-tested: date-blind resolution, forward-rolling expiry, and date-blind
lot size each break three tests.

**Verify**
```bash
pytest tests/unit/test_instruments.py -k "resolve_at_date or lot_size_history or expiry_calendar"
```
**Traps:** lot sizes change over time. Store them with validity intervals or every historical P&L is wrong.

---

#### M11 · Market feed adapter
**Owns:** the connection to the broker/vendor WebSocket.
**Contract:** emits `Tick` to `market.tick`. Nothing else.

**Tasks**
- [ ] Subscribe manager (instrument limits are finite — budget them explicitly)
- [ ] Reconnect with exponential backoff, resubscribe on reconnect
- [ ] Sequence tracking + gap detection → `audit.gap` event
- [ ] Heartbeat: no message in 3 s during market hours → `HALT` signal
- [ ] Adapter interface so a second vendor can be swapped in without touching consumers

**Verify**
```bash
pytest tests/integration/test_feed.py -k "reconnect_no_loss or gap_detected or heartbeat_halt"
python -m services.ingest_market --dry-run --duration 300   # 5 min live, assert 0 unexplained gaps
```
**Done:** forcibly kill the network for 20 s mid-session → reconnects, gap logged, no silent data loss.

**Three adapters, one protocol.** `libs/ms_broker/kite.py` and
`libs/ms_data/angelone.py` both implement `MarketFeed` and no consumer imports
either. Angel One is the one credentials exist for. Its four vendor-specific
traps are documented in the module docstring and each has a mutation-tested
guard: prices arrive in **paise** (a 100x that stays internally consistent all
the way downstream), the websocket wants the **feed token** and not the JWT (the
wrong one connects and then delivers nothing), the scrip master's tokens belong
to **one expiry** (M10's lookahead bug in vendor clothing), and there are **no
sequence numbers**, so `GapDetector` is genuinely blind here rather than
reporting zero gaps - `AngelOneFeed.gap_detection` returns `False` so that
distinction is asked for rather than inferred.

`libs/ms_data/upstox.py` is the third, and the one to reach for: the Analytics
Token is valid for a year and is **read-only**, so an unattended recorder needs
no daily login and physically cannot place an order. That last part is a
stronger guarantee than `ms_core.env` provides, because it is enforced at the
venue rather than by this repository. Its traps are the inverse of Angel One's
and are documented and mutation-tested the same way: prices are `double` and
already in **rupees** (the danger is carrying the Angel paise fix across), int64
fields arrive as **strings** from protobuf's JSON mapping, `ltt` is the last
*traded* time and not the feed clock (`currentTs` would claim a trade happened
now), the first message after subscribing is a **snapshot and not a tick**, and
`feeds` is keyed by a vendor **string** where `Tick.instrument_id` is an int -
resolved through the instrument master, never hashed.

One thing Upstox gives that neither of the others does: `MarketInfo.segmentStatus`
reports whether the segment is open, which is the question `FeedHealth` is
otherwise forced to guess at when the socket goes quiet.

Known risk, carried deliberately: the documentation lists the websocket as
supported for Analytics Tokens, and there are open community reports of the feed
authorize endpoint returning 403 for exactly that combination. Unresolved as of
2026-08-28. `authorize()` says so in its error rather than leaving somebody to
regenerate a token that was never the problem; the fallback is a daily OAuth
token in the same field.

The tick counts as ingested when a real one has been recorded from a real
session. Nothing in this repository has done that yet.

---

#### M12 · Raw tick recorder
**Owns:** writing history to disk. **Build this on day 3. It never stops again.**

**Tasks**
- [ ] Append-only newline-delimited msgpack/JSON per instrument per day
- [ ] Fsync policy: durability over throughput
- [ ] Daily rotation + integrity manifest (row count, hash)
- [ ] Runs as a supervised service, restarts on crash

**Verify**
```bash
python jobs/verify_recording.py --date $(date +%F)
# asserts: file exists, row count > threshold, no time-travel, manifest hash matches
```
**Done:** 5 consecutive complete sessions on disk. **This is Gate G0.**

---

#### M13 · Candle builder
**Owns:** ticks → OHLCV. Deceptively subtle — see blueprint Ch.5.

**Tasks**
- [ ] Left-closed / right-open minute buckets
- [ ] Synthetic zero-volume candles for silent minutes, flagged
- [ ] Session tagging (preopen vs continuous)
- [ ] Late-tick correction log; candles are **versioned**, never mutated
- [ ] Roll-ups: 1m → 5m → 15m derived, not recomputed from ticks
- [ ] VWAP, trade count, OI

**Verify**
```bash
pytest tests/replay/test_candle_determinism.py     # same input twice -> byte-identical output
pytest tests/unit/test_candles.py -k "boundary_tick or synthetic_gap or session_split or rollup_consistency"
python jobs/verify_candles.py --date <day>          # asserts exactly 375 continuous-session candles
```
**Done:** determinism test in CI, green.

---

#### M14 · Option chain snapshotter
**Owns:** periodic full-chain capture. The scarcest data you will ever own.

**Tasks**
- [ ] Both indices, nearest two expiries, full liquid strike range
- [ ] 1–5 s cadence with completeness metric per snapshot
- [ ] Capture bid/ask/LTP/volume/OI/(vendor IV) **and the spot at the same instant**
- [ ] Never discard: even incomplete snapshots are stored, flagged

**Verify**
```bash
python jobs/verify_chain.py --date <day>
# asserts: >= 95% expected strikes, spot attached, monotone timestamps, no duplicate (ts,strike,type)
```
**Traps:** the spot price must be captured *with* the chain, not joined later. Joining introduces a lag you cannot undo.

**A historical source, until a live one exists.** `libs/ms_data/bhavcopy.py`
loads NSE's own F&O bhavcopy: every listed contract's strike, expiry, close,
settlement, OI and volume, for every session back to 2018, free and with no
broker account. It is **daily, not intraday**, and it has **no book at all** -
`OptionQuote.mid` is correctly `None` on every quote it produces, because a
bhavcopy cannot supply one and manufacturing a mid from a single price is worse
than not having one.

The defect it exists to prevent, measured on the real NIFTY chain for
2026-08-26: of 1,584 listed contracts **857 (54%) never traded**, and every one
of them still carries both a close and a settlement price computed by the
exchange. On those, the two numbers disagree by a **median 15.9%**, 78% at p90,
2273% at worst. Fitting SVI to all 1,584 would fit NSE's own model on more than
half the points, converge, and report a smile nothing traded at - a working
pipeline, plausible numbers, no error. So `traded_only` defaults to True, the
dropped count is *returned* rather than logged, and there is no `price` property
that silently picks close or settlement.

Two formats, and the trap sits in the difference: UDiFF (2024 onward) carries
`UndrlygPric`, the legacy format carries no spot at all. A chain built from a
legacy file therefore **refuses to construct** without an explicit spot rather
than joining one from elsewhere - the trap named directly above this paragraph.
Verified on real files: 2026-08-26 builds, 2023-06-09 refuses.

`available_at()` returns the morning after the session, deliberately
conservative: too early is lookahead that silently inflates a backtest, too late
costs one day.

Not a new module - M14's contract is unchanged and this is a second source
behind it. The tick still counts as ingested only when a real one has been
recorded live, and nothing here has done that.

---

#### M15 · Cross-asset collector
**Tasks**
- [ ] GIFT/SGX Nifty, global equity futures, crude, gold, DXY, USD/INR, US yields, Asian indices
- [ ] 1-minute bars minimum; tick where cheap
- [ ] Runs **24×7**, not just Indian market hours — the overnight window is the point
- [ ] Source-lag annotation per series

**Verify**
```bash
python jobs/verify_crossasset.py --date <day> --expect-24h
```

---

#### M16 · TimescaleDB schema & loaders
**Tasks**
- [ ] Hypertables per blueprint Ch.5 listing; chunk = 1 day
- [ ] Compression policy after 7 days; retention policy for ticks
- [ ] Continuous aggregates for candle roll-ups
- [ ] Idempotent loaders: re-running a day's load changes nothing
- [ ] Migrations under version control (alembic or plain SQL + a version table)

**Verify**
```bash
pytest tests/integration/test_loaders.py -k "idempotent or compression or continuous_agg"
psql -c "SELECT count(*) FROM candle WHERE tf=1 AND ts::date='<day>'"   # = 375 per instrument
```

---

#### M17 · Parquet archive + DuckDB research layer
**Tasks**
- [ ] Nightly export: `data/parquet/{table}/date=YYYY-MM-DD/instrument=NNN/*.parquet`
- [ ] DuckDB helper: `research.candles(start, end, instruments) -> DataFrame`
- [ ] Dataset manifest = list of file hashes → a "dataset version"

**Verify**
```bash
python -c "from libs.ms_data.research import candles; df=candles('2026-07-01','2026-07-31',['NIFTY']); print(len(df))"
# under 1 second for a month; row count matches TimescaleDB
pytest tests/unit/test_parquet_parity.py    # Parquet == Timescale for a sample day
```

---

#### M18 · Replay engine
**Owns:** turning history back into a live-looking stream. **This module makes everything else testable.**

**Contract:** `replay(date_range, speed=inf) -> publishes to the same bus topics in true timestamp order`.

**Tasks**
- [ ] Merge ticks + chain snapshots + news + calendar into one ordered stream
- [ ] Drives `SimClock`; consumers cannot tell the difference from live
- [ ] Speed control: `inf` for backtests, `1.0` for demos
- [ ] Golden-day fixture checked into `tests/replay/golden_day_*/`

**Verify**
```bash
pytest tests/replay/test_golden_day.py    # full pipeline on a fixed day, output hash matches
python jobs/replay.py --date <day> --speed inf --assert-deterministic
```
**Done:** two runs → identical output hash. **This is half of Gate G1.**

---

#### M19 · Data quality gates
**Tasks** — implement each check from blueprint Ch.5 §Data quality gates:
- [ ] Tick sequence gaps · candle count · chain completeness · IV sanity · put-call parity
- [ ] News timestamp sanity · duplicate rate · feature null rate · train/serve PSI
- [ ] Each check returns `PASS|WARN|FAIL` + a number; FAIL blocks downstream jobs
- [ ] Nightly report to `build/qc/YYYY-MM-DD.json` + alert on FAIL

**Verify**
```bash
python jobs/qc.py --date <day> --strict     # exit 0 only if all PASS
pytest tests/unit/test_qc.py -k "detects_injected_gap or detects_bad_iv"
```
**Traps:** write the tests by *injecting* corruption into a good day. A QC check nobody has seen fail is not a check.

---

### L2 · NEWS & EVENTS

#### M20 · Source registry & credibility ✅
**Tasks**
- [x] `config/sources.yaml`: id, name, class, credibility `c`, expected lag, terms-of-service notes
- [x] Credibility tiers per blueprint Ch.5 §Source catalogue
- [x] Per-source rate limits and quota tracking

**Built** `libs/ms_news/sources.py`, `tests/unit/test_sources.py` (28 tests).
The schema and the low-credibility rule were already in `ms_core/config.py`;
this adds tiers, rate limits, quota and readiness.

**A rate limit that records calls after making them is a log, not a limiter.**
`Budget.take` refuses *before* the request. The block it fails to prevent
arrives on the morning the news matters, because that is the morning the poller
is busiest. Quota windows are **wall-clock aligned** — a provider's "1 000 per
day" resets at their midnight, and a rolling window silently allows nearly
double across the boundary while looking like ordinary traffic. `min_interval`
is enforced separately from the quota: honouring only the quota fires a hundred
requests in the window's first second and gets blocked having used a tenth of
its allowance.

**Tier comes from the credibility number, not the `source_class` string.** A
class string is a label somebody typed; the credibility is the weight
everything downstream applies, and letting them disagree means the tier in a
report is not the tier the scorer used.

`can_open_directional` re-checks credibility itself rather than trusting the
config flag. **A mutation test found this**: the cross-field rule lives on
`SourcesConfig`, so a bare `Source` can be constructed claiming
`may_trigger_directional=True` at 0.25 — by a fixture, by hand, by a future
loader. Dropping the credibility half of the check broke nothing, because every
other test used a source that already said `False`.

All six shipped sources remain `enabled: false` with `TODO` ToS notes, and
`readiness()` says so. **That is the configured state, not a fault** — reading a
provider's terms is not a coding task.

**Verify** `pytest tests/unit/test_sources.py -k "weights_in_range or all_have_tos_note"`

---

#### M21 · Collector framework ✅
**Tasks**
- [x] Pluggable `Collector` base: `poll()` / `stream()`, retries, backoff, health heartbeat
- [x] One collector per source; failure in one never affects others
- [x] Emits `RawNewsItem` to `raw.news`
- [x] Per-source metrics: items/min, error rate, last-success age

**Built** `libs/ms_news/collectors.py`, `tests/unit/test_collectors.py` (26 tests).

**Isolation is the whole point, and its failure is misdiagnosed.** A single loop
letting one exception escape stops every source after the one that threw — and
the symptom is that the *others* went quiet, which reads as a network problem
and sends the investigation somewhere else entirely.

**Backoff resets on success**, or a source that failed twice at breakfast is
still throttled at noon and the courtesy has become an outage of our own making.
It is **jittered** so N collectors that failed during one provider outage do not
all return the instant it ends and re-create the load that caused it — and the
jitter is seeded, because a test that cannot reproduce a backoff schedule cannot
check one. Backing off *skips the poll* rather than recording that it made one.

**The heartbeat measures silence since the last success, not the last attempt.**
A collector erroring every thirty seconds is not silent; one succeeding-but-empty
is, and it is the second that looks healthy. Staleness is reported as a number
for a human rather than a halt: forty minutes of silence at 10 a.m. is an
outage and the same forty minutes at 3 a.m. is a Sunday, and the framework
cannot tell.

An item stamped with the wrong `source_id` is refused, not published — it would
inherit the wrong credibility, and credibility decides whether it can open a
position. Two collectors on one source are refused: they double every item and
each copy looks like corroboration for the other.

`FailingCollector` ships in the library, not the tests — same reasoning as the
kill drill's stubborn broker.

Mutation-tested seven guards, all caught.

**Verify**
```bash
pytest tests/unit/test_collectors.py -k "isolation or backoff or heartbeat"
python -m services.ingest_news --duration 3600 --report   # 1 hour, per-source counts sane
```

---

#### M22 · Normaliser & publish-time resolver
**Owns:** the three timestamps. **This module determines whether your research is real.**

**Tasks**
- [ ] Encoding normalisation, language ID, boilerplate strip
- [ ] Publish-time extraction: structured metadata → HTML meta → in-text date → `ts_uncertain`
- [ ] Record *which method* produced `t_event` (you will need this later)
- [ ] `t_available` stamped after the full pipeline, not at ingest

**Verify**
```bash
python jobs/audit_timestamps.py --sample 200 --manual-review
# manual pass required: >= 90% within 5 s of truth  -> part of Gate G2
pytest tests/unit/test_publish_time.py -k "prefers_structured or flags_uncertain"
```

---

#### M23 · Deduplication ⚙
**Tasks**
- [ ] Layer 1: exact content hash
- [ ] Layer 2: SimHash, 64-bit, Hamming ≤ 3
- [ ] Layer 3: sentence-embedding cosine > 0.93 within a 30-min window (ANN index)
- [ ] Cluster head = first-seen item; later items attach as `duplicate_of`
- [ ] Cluster metadata: source count, credibility max — feeds M29 confirmation

**Verify**
```bash
pytest tests/unit/test_dedup.py -k "exact or near_dup or paraphrase or false_merge_guard"
python jobs/dedup_report.py --days 30    # duplicate rate < 3%, false-merge spot check
```
**Traps:** over-aggressive dedup merges two *different* events on the same topic. Keep a labelled false-merge test set.

---

#### M24 · Economic calendar & consensus
**Tasks**
- [ ] Scheduled events loaded ahead of time with expected value + consensus dispersion
- [ ] At release: compute `surprise = (actual − consensus) / σ_consensus`
- [ ] Revision handling: a revised prior figure is a **new row**, not an update

**Verify** `pytest tests/unit/test_calendar.py -k "surprise_math or revision_is_append_only"`
**Note:** surprise is the single most powerful macro feature. Do not skip this module.

---

#### M25 · Point-in-time event store
**Contract:** `store.as_of(t) -> everything knowable at time t`. Nothing more, nothing less.

**Tasks**
- [ ] Append-only tables per blueprint Ch.5 listing
- [ ] `pgvector` (or FAISS) index on embeddings
- [ ] Query API: by class, by time window, by entity, by `as_of`
- [ ] Backfill kept in a **separate partition** from live-collected data (timestamp quality differs)

**Verify**
```bash
pytest tests/unit/test_pit_store.py -k "as_of_excludes_future or backfill_partition_separate"
python jobs/pit_query.py --as-of "2026-07-14T11:03:00+05:30" --hash   # reproducible hash across runs
```
**Done:** 30 days collected, dup rate < 3%, timestamps audited. **This is Gate G2.**

---

#### M26 · Relevance gate (tier 1) ⚙
See model card **[AI-2]**.
**Tasks**
- [ ] Keyword/entity bloom filter (recall-first)
- [ ] Tiny logistic/linear model on cached embeddings
- [ ] Threshold tuned for **recall ≥ 0.99** on a labelled set; precision is irrelevant here
- [ ] p95 latency under 5 ms

**Verify**
```bash
pytest tests/ml/test_relevance_gate.py -k "recall_above_99 or latency_p95"
```

---

#### M27 · Event classifier (tier 2) ⚙
See model cards **[AI-3] [AI-4]**. The heart of L2.

**Tasks**
- [ ] Freeze `config/taxonomy.yaml` v1 **before** labelling
- [ ] Gold set: 2 000–3 000 items, LLM pre-label → human correct, 20% double-annotated
- [ ] Fine-tune encoder: multi-label head + severity regression head + polarity head
- [ ] Temperature calibration on a held-out slice
- [ ] Export to ONNX, quantise, p95 ≤ 120 ms
- [ ] **Empirical severity recalibration** against realised |AR₃₀| (blueprint Ch.6 §Severity step 3)

**Verify**
```bash
pytest tests/ml/test_classifier.py -k "macro_f1 or high_sev_recall or noise_fp or latency"
python jobs/eval_severity_correlation.py     # Spearman(severity, |AR_30|) >= 0.35
```
**Done:** macro-F1 ≥ 0.80 · recall(sev≥7) ≥ 0.95 · noise FP ≤ 2% · p95 ≤ 120 ms · Spearman ≥ 0.35. **Gate G3.**
**Traps:** a classifier that scores well on labels but whose severity does not correlate with market movement is useless. The Spearman check is the real gate.

---

#### M28 · Entity/geo extraction & KG routing ⚙
**Tasks**
- [ ] NER → canonical entity IDs (knowledge graph), not free text
- [ ] Geo extraction with an India-exposure weight
- [ ] Exposure routing: entity → factor → instrument betas
- [ ] Betas **estimated from rolling regressions**; the hand-written dict is only a cold-start prior

**Verify**
```bash
pytest tests/unit/test_kg_routing.py -k "canonical_ids or beta_has_ci or no_hardcoded_in_prod"
python jobs/refresh_betas.py --window 250d --report   # every beta has n and CI
```

---

#### M29 · Novelty, confirmation & credibility fusion ⚙
**Tasks**
- [ ] Novelty per blueprint Ch.6 formula — including the **already-moved** term (needs M13 live)
- [ ] Confirmation state machine: single-source → corroborated → official → **retracted**
- [ ] Retraction handling: emit `RETRACTION`, poison the event, force-close any position opened on it
- [ ] Severity ≥ 9 requires two independent sources with `c ≥ 0.8`
- [ ] Market-disagreement suppressor

**Verify**
```bash
pytest tests/ml/test_confirmation.py -k "single_source_no_directional or retraction_closes or sev9_needs_two"
pytest tests/stress/test_ap_hack.py    # inject a false catastrophic headline -> NO directional signal
```
**Done:** `test_ap_hack.py` is in CI **forever**. It never gets deleted, never gets skipped.

---

### L3 · IMPACT

#### M30 · Abnormal-return engine
**Tasks**
- [ ] Time-of-day baseline: expected return and σ per minute-of-day per volatility regime
- [ ] `AR_h = r_h − E[r_h | time-of-day, regime, day-type]`
- [ ] Matched control sampler (same minute-of-day, same regime, non-event days)
- [ ] Bootstrap significance, not normal approximation

**Verify**
```bash
pytest tests/unit/test_abnormal_returns.py -k "tod_normalisation or control_matching or bootstrap_ci"
python jobs/plot_tod_profile.py   # eyeball the U-shape; it must exist, or your data is wrong
```

---

#### M31 · Event-study runner
**Tasks**
- [ ] For every historical event: AR at h ∈ {1,5,15,30,60}, RV ratio, ΔIV, MFE/MAE, volume ratio, time-to-peak, half-life
- [ ] Store results in a queryable table keyed by event id + class
- [ ] Re-runnable end to end; results versioned by dataset version

**Verify**
```bash
python jobs/event_study.py --rebuild --report
pytest tests/unit/test_event_study.py -k "no_lookahead or reproducible_with_seed"
```

---

#### M32 · Hierarchical Bayesian impact priors ⚙
See model card **[AI-5]**.
**Tasks**
- [ ] Two-level model: class nested in family, shrinkage per blueprint Ch.7
- [ ] Output: posterior mean, credible interval, `n_eff` per (class, horizon)
- [ ] Severity response `g(s) = (s/5)^γ`, γ fitted
- [ ] **Refuse to answer** when `n_c < 8` and no family prior — return `UNKNOWN`, not a guess

**Verify**
```bash
pytest tests/ml/test_impact_priors.py -k "shrinkage_toward_family or refuses_small_n or ci_coverage"
python jobs/impact_report.py    # >=15 classes with n_eff>=8; >=5 with CI excluding 0
```
**Done:** part of **Gate G4**.

---

#### M33 · Cross-asset transfer estimator ⚙
**Tasks**
- [ ] Lagged regressions / VAR for each channel in blueprint Ch.7 §Cross-asset
- [ ] Asymmetric handling: spikes vs drifts transmit differently
- [ ] Rolling re-estimation with CIs; store coefficients with `as_of`

**Verify** `pytest tests/ml/test_transfer.py -k "lag_selection or rolling_stability or asymmetry"`

---

#### M34 · Live impact scorer
**Contract:** `ClassifiedEvent + market state → ImpactEstimate` (distribution, not scalar).

**Tasks**
- [ ] The multiplicative formula from blueprint Ch.7 §Live scoring
- [ ] All terms: severity, novelty, credibility, regime κ, time-of-day λ, already-priced π
- [ ] Output triple `(r̂_h, quantiles, n_eff)` per horizon + volatility/ΔIV forecast
- [ ] p95 ≤ 20 ms (priors precomputed in Redis)

**Verify**
```bash
pytest tests/unit/test_impact_scorer.py -k "already_priced_damps or unknown_class_falls_back or latency"
```
**Traps:** the `already-priced` term is what stops the system chasing a move that is already over. Test it explicitly.

---

#### M35 · Overnight gap forecaster ⚙
See model card **[AI-6]**. **This is the first module that produces standalone value.**

**Tasks**
- [ ] Features: overnight global moves, GIFT Nifty, overnight events + severities, prior-day close context, regime
- [ ] Target: next-session opening gap, as a **distribution** (quantile regression)
- [ ] Baseline to beat: `gap = β × overnight global futures move`
- [ ] Backtest with strict point-in-time cutoff at 08:45 IST

**Verify**
```bash
python jobs/eval_gap_model.py --walk-forward --report
# CRPS beats naive baseline by >= 10% out of sample
pytest tests/ml/test_gap_model.py -k "pit_cutoff_respected or beats_baseline"
```
**Done:** part of **Gate G4**. Ship it via M95.

---

### L4 · FEATURES

#### M40 · Feature registry & framework
**Owns:** the one-definition/two-materialisation contract. Get this right or train/serve skew will eat you.

**Tasks**
- [ ] `@feature(name, version, deps, window, ttl, fill)` decorator
- [ ] Registry: name → definition, with dependency graph
- [ ] Every feature computes from an `as_of` timestamp; reading future data is structurally impossible
- [ ] Feature vector hash stored with every prediction

**Verify** `pytest tests/unit/test_feature_registry.py -k "as_of_enforced or version_required or dep_graph_acyclic"`

---

#### M41 · Offline materialiser
**Tasks**
- [ ] Run any feature over a date range from Parquet → columnar output
- [ ] Incremental: only recompute changed dates
- [ ] Parallel over instruments/dates

**Verify** `python jobs/materialise.py --features all --start <d1> --end <d2> --verify-nulls`

---

#### M42 · Online feature server
**Tasks**
- [ ] Redis-backed rolling windows updated on each tick/candle/chain event
- [ ] `serve(feature, as_of) -> value` with TTL/staleness enforcement
- [ ] Pipeline reads: whole vector in one round trip, p95 ≤ 40 ms

**Verify** `pytest tests/integration/test_online_features.py -k "staleness_rejected or latency_p95"`

---

#### M43 · Parity test & leakage linter ⭐
**This module is why your live results will match your backtest.**

**Tasks**
- [ ] Parity: replay a recorded day through the online path, materialise the same day offline, assert equality within float tolerance — **per feature**
- [ ] Leakage linter: static check that every feature's dependency timestamps ≤ `as_of`
- [ ] Both run in CI on every commit touching `libs/ms_features/`

**Verify**
```bash
pytest tests/parity/test_offline_online.py -v    # per-feature diff report, all pass
python jobs/leakage_lint.py --strict             # exit 0
```
**Done:** a new feature cannot be merged without passing both. No exceptions, ever.

---

#### M44 · Feature groups A–F
Implement per blueprint Ch.8 catalogue. Build in this order:

- [ ] **Group A** price/trend (+ time-of-day z-scores — do not skip these)
- [ ] **Group F** calendar/session/regime (cheap, high value)
- [ ] **Group C** options-derived (the differentiated group: skew, term structure, GEX, implied move, VRP)
- [ ] **Group E** event features (severity, novelty, surprise, prior posterior, seconds-since-event)
- [ ] **Group D** cross-asset
- [ ] **Group B** microstructure — **only if the feed provides depth**; train a with/without variant

**Verify**
```bash
pytest tests/parity/test_offline_online.py       # all groups pass parity
python jobs/feature_report.py --null-rates --psi # null rate < 1%/feature/day
```
**Traps:** never train on a feature you will not have live at 09:16 on a Tuesday. Feed capability determines the feature set, not the other way round.

---

#### M45 · Regime model ⚙
See model card **[AI-7]**.
**Tasks**
- [ ] Features: trailing RV percentile, trend efficiency, spread/liquidity, IV term slope, GEX sign
- [ ] Gaussian HMM (3–5 states) or clustering; label states by their statistics, not by intuition
- [ ] Regime label becomes a feature **and** a conditioning variable for ensembling and sizing

**Verify**
```bash
pytest tests/ml/test_regime.py -k "state_persistence or no_lookahead_in_state or states_are_distinct"
python jobs/regime_report.py    # per-state return/vol stats must actually differ
```

---

### L5 · MODELS

#### M50 · Label builder
**Tasks**
- [ ] Triple barrier: `±k·σ_t` barriers with σ from the **current** volatility estimate, plus a time barrier
- [ ] Never let a barrier cross the session close
- [ ] Sample uniqueness weights (overlapping outcome windows are not independent)
- [ ] Meta-labels: was the primary model's call correct?
- [ ] Event grouping key so CV can split by event, not by row

**Verify**
```bash
pytest tests/ml/test_labels.py -k "barrier_before_close or uniqueness_weights or vol_scaled_barriers"
python jobs/label_report.py    # class balance per horizon; sanity-check against raw returns
```

---

#### M51 · CV harness
**Owns:** the honesty of every number you will ever report.

**Tasks**
- [ ] Purged K-fold + embargo (1 trading day)
- [ ] Walk-forward: anchored and rolling
- [ ] Combinatorial purged CV → a **distribution** of Sharpe, not one number
- [ ] Deflated Sharpe ratio with a trial counter
- [x] **Experiment log**: every run appends config hash, dataset version, metrics — hash-chained and versioned in `research/`, not `build/`
- [ ] **Locked holdout** guard: last 3 months, refuses to load without an explicit unlock flag; logs every access

**Verify**
```bash
pytest tests/ml/test_cv.py -k "purging_removes_overlap or embargo_applied or holdout_locked"
python jobs/experiment_log.py --count     # must be non-zero and growing; feeds deflated Sharpe
python jobs/experiment_log.py --verify    # chain intact = no run was quietly deleted
```
**Traps:** if you evaluate on the locked holdout more than once, delete the project's performance claims. The guard exists to protect you from yourself.

---

#### M52 · M0 rules baseline
**Owns:** the bar. Also the degraded-mode fallback.

**Tasks**
- [ ] Rule: `impact prior direction × severity gate × momentum confirmation` → long/short/none
- [ ] Same interface as every ML model, so it can be swapped in at runtime
- [ ] Fully deterministic, no training

**Verify** `python jobs/eval_model.py --model M0 --walk-forward --report`
**Done:** every ML model below must beat this on log loss with a bootstrap CI excluding zero, or it does not ship.

---

#### M53 · Direction model ⚙ → card **[AI-8]**
#### M54 · Quantile model ✅ → card **[AI-9]**

**Built** `libs/ms_models/quantile_heads.py`, `tests/ml/test_quantile_heads.py`
(36 tests). One boosted head per (alpha, horizon), monotonicity enforced
post-hoc, Gaussian fallback from M55's sigma per head.

**Pinball loss has no second derivative, and Newton boosting divides by it.**
With `h = 0` the leaf value collapses to `-sum(g)/l2`, and the pinball gradient
only counts which side of the prediction each point is on — never how far. The
step becomes a function of the sign imbalance and the regularisation constant.

It does not fail the way you would expect. **Marginal coverage still looks
right.** What breaks is the conditional part: on heteroscedastic data whose true
5–95 half-width triples between regimes, the fitted band widens **1.28x without
leaf renewal against 1.89x with it** — 97.1% coverage in the quiet regime and
84.6% in the busy one, which average to something that reads as calibrated. Too
small a position on quiet days and too large a one on exactly the days that
matter. It also never early-stops, running to the tree cap.

The fix is leaf-value renewal: constant hessian to choose the splits, then each
leaf's value set to the empirical alpha-quantile of its residuals. `renew_leaves`
is exposed so a test can measure what it buys — a defence nobody can switch off
is a defence nobody has measured.

**Verify** `pytest tests/ml/test_quantile_heads.py`
#### M55 · Volatility model ⚙ → card **[AI-10]**
#### M56 · IV-change model ⚙ → card **[AI-11]**
#### M57 · Meta-labeller ⚙ → card **[AI-12]**
#### M58 · Sequence models ⚙ → card **[AI-13]** *(deferred until M53 is beaten)*

**Where deep learning and RL belong in this project, and where they do not.**

Both were considered for M53 and both were rejected there on evidence, not
preference. The record is here so the decision is re-openable rather than
re-argued.

*Not the gap forecaster.* The regularisation sweep in `docs/model_audit.md`
improves monotonically to a plateau where the feature weights are essentially
zero - the data is telling us the daily cross-asset features carry almost
nothing. A neural network is more capacity applied to the same absent signal,
which is the failure this project keeps catching, not the fix for it.

*Not the direction model, for now.* On tabular features at a few hundred to a
few hundred thousand rows, gradient boosted trees beat neural networks
consistently and by enough that the choice is not close. M53 is boosted trees
for that reason. When it is beaten, this is the module that beats it.

*Sequences (this module).* A TCN or a small transformer over raw candle paths is
the honest case for deep learning here: the pattern is in the ordering and the
shape, which is exactly what hand-built features throw away. It needs intraday
history that does not exist yet, and it needs M53 as the thing to beat -
otherwise there is no way to tell whether the extra machinery bought anything.

*Text (M27).* Classifying a headline into an event class is the one place in
this system where a pretrained transformer is straightforwardly the right tool
and nothing else comes close. Blocked on the news collectors (M21-M25).

*Reinforcement learning (M75, sizing and execution).* Sequential decisions with
a reward is genuinely RL's shape. It is also the single most backtest-overfitting
-prone technique in quantitative finance: an RL agent will learn the simulator,
including its bugs, and report a beautiful equity curve for behaviour that
cannot survive a real fill. It is not attempted before the L2 event-driven
simulator (M82) and the stress suite (M85) exist, and when it is, the null test
is the same as everywhere else - an agent trained on shuffled returns must learn
nothing.

Shared tasks for all of the above:
- [ ] Train through M51 only — no ad-hoc `train_test_split` anywhere
- [ ] Log to MLflow: params, metrics, dataset version, feature list, config hash
- [ ] SHAP report; any feature with near-zero importance across folds gets dropped
- [ ] Export ONNX + record p99 batch-1 latency
- [ ] Register in M5A with stage `staging`

**Verify (each)**
```bash
python jobs/train.py --model <ID> --cv purged --report
python jobs/eval_model.py --model <ID> --vs M0 --bootstrap 2000    # CI excludes 0
python jobs/latency_check.py --model <ID> --p99-budget-ms <n>
```

---

#### M59 · Ensemble, calibration & conformal ⚙ → card **[AI-14]**
**Tasks**
- [ ] Stacking with time-aware out-of-fold predictions; **simple** meta-learner (logistic regression)
- [ ] Regime-conditional weights from M45
- [ ] Isotonic/Platt calibration fitted on a chronologically later slice
- [ ] Split conformal intervals for magnitude
- [ ] Reliability diagram + Brier decomposition generated on every eval

**Verify**
```bash
python jobs/eval_calibration.py --model ensemble --report
# reliability slope in [0.85, 1.15]; Brier skill > 0.02 vs climatology
pytest tests/ml/test_conformal.py -k "coverage_within_tolerance"
```
**Done:** with M53/M55 this closes **Gate G5 — the decision gate of the whole project.**

> ### ⚠ If G5 fails
> Do **not** proceed to L6/L7. Pivot options, in order of preference:
> 1. **Volatility-only** strategies (M55/M56 targets are far more forecastable — the gate is easier)
> 2. **Ship the gap product** (M35 + M95) which already passed G4
> 3. Buy better data (order-book depth, faster sources) and retry **once**
>
> Three failed attempts = the honest answer is that this edge does not exist at your data tier.
> Knowing that at week 24 instead of week 80 is a **success**.

---

#### M5A · Model registry & ONNX serving
**Tasks**
- [ ] MLflow registry; stages `staging → shadow → production → archived`
- [ ] Production code loads **only** from the registry, never from a file path
- [ ] ONNX Runtime server, warm, batch-1 optimised, health endpoint
- [ ] Model version stamped on every prediction
- [ ] Shadow mode: challenger runs on live data for ≥15 sessions before promotion

**Verify**
```bash
pytest tests/integration/test_serving.py -k "loads_from_registry or version_stamped or warm_latency"
python jobs/shadow_report.py --challenger <ID> --sessions 15
```

---

### L6 · PROJECTION

#### M60 · Path sampler ✅ → card **[AI-15]**
**Tasks**
- [x] Volatility-clustering simulator (student-t innovations, GARCH-style) conditioned on M55
- [x] Calibrate so terminal quantiles match M54's heads
- [x] 5 000 paths in < 60 ms (vectorised)

**Built** `libs/ms_models/paths.py`, `tests/ml/test_path_sampler.py` (30 tests).

**This is the one module that uses numpy**, and the reason is measured: 5 000
paths of 60 steps is **443 ms** in pure Python against a 60 ms budget. Vectorised
it runs at a **49 ms median** (min 36, max 58), of which ~22 ms is drawing the
student-t innovations. An earlier shape cost 79 ms purely from writing
`levels[:, s] = x` down the column of a row-major array; the arrays are built
step-major and transposed once. The latency test guards at 2x the budget rather
than at the budget — a timing assertion that flakes is one somebody deletes.

**Two distributions that are not the same distribution.** GARCH gives the path
*shape* (clustering, fat tails); the heads give a *terminal distribution* that
was fitted and scored. Using the simulated terminal throws the fitted model
away at the last step. So paths are simulated for shape and bent to land where
the heads say — each keeps its increments and takes a drift adjustment ramped
from 0 at entry to 1 at the horizon. Flat instead of ramped moves the path's
first point too, so every P&L along it is offset by a constant nobody can see.

**The tails are where it goes wrong quietly.** Clamping everything past the
outermost head onto that head flattens the tail onto one value, so CVaR — the
*mean of the tail* — becomes the 5th percentile, which is the tail's boundary.
Measured: **CVaR₅ is 30% optimistic** clamped, and the worst path collapses from
−0.133 to −0.020. The number still computes and is still negative.

Clustering is verified against an iid control (alpha=beta=0): ACF of squared
returns **0.157 vs 0.0007**. Squared returns, because a GARCH process has
uncorrelated returns by construction and an ACF on returns reads zero either way.

Mutation-tested: unscaled innovations, clamped-by-default tails, flat instead of
ramped drift, ACF on returns rather than squares, and ignoring `variance0`.
Three survived the first pass and needed tests written for them.

**Verify** `pytest tests/ml/test_path_sampler.py -k "terminal_matches_quantiles or vol_clustering_present or latency"`

---

#### M61 · Fan-chart producer ✅
**Tasks**
- [x] Per-minute empirical OHLC quantiles from sampled paths
- [x] Body = [q25,q75] of close, wick = [q05,q95], tint by `P(up)`
- [x] Implied-move band overlay from M70
- [x] `edge_ratio = forecast |move| / implied move` computed and surfaced

**Built** `libs/ms_models/fanchart.py`, `tests/unit/test_fanchart.py` (29 tests).

**The two moves have to be measured against the same year, and they were not.**
`pricing.MINUTES_PER_YEAR` is **94 500** (252 x 375, trading time).
`surface.fit_surface` computes its year fraction from `365 * 24 * 3600` —
**525 600** minutes, calendar time. They differ by 5.56x, so any move computed
from a volatility differs by **2.358x**. A 13-vol name over sixty minutes
implies 0.3276% on trading time and 0.1389% on calendar time, and both are
numbers somebody would print without blinking.

That is not a rescaling, it can flip the decision. With a forecast band sitting
between the two implied moves, calendar time gives `edge_ratio` **1.500** — buy
volatility — and trading time gives **0.636** — sell it. Same fan, same IV, same
horizon; a test pins exactly that pair.

So `TimeBasis` has **no default anywhere in the module**, and
`edge_ratio_from_surface` uses `SURFACE_BASIS` so a change to what M70 fits
breaks here rather than silently rescaling every edge estimate. Whether calendar
time is right for an intraday horizon is a real open question — options do not
decay over a weekend the way it implies — but it has to be settled by changing
what the surface fits, not by disagreeing with it downstream.

Two smaller ones: both sides of the ratio are 1.645 sd, because comparing a 90%
band against a one-sd move returns 1.645 for a model that agrees exactly with
the market and somebody reads that as a large edge. And `P(up)` counts paths
above **the entry**, not above the ensemble median — against the median it is
50% at every minute by construction and the tint varies only with sampling noise
while looking exactly like information.

Mutation-tested: a default basis, the surface basis flipped, `P(up)` against the
median, the 1.645 factor dropped, monotonicity unchecked, and a narrowing fan
not reported. All six caught.

**Verify** `pytest tests/unit/test_fanchart.py -k "monotone_quantiles or implied_overlay_present"`

---

#### M62 · Projection scorer ✅
**Tasks**
- [x] Pinball loss, CRPS, PIT histogram, empirical coverage, interval width
- [x] Baselines: random-walk + GARCH, and the option-implied distribution
- [x] Daily rolling report

**Built** `libs/ms_models/projection.py`, `tests/ml/test_projection.py` (34 tests),
`jobs/eval_projection.py`.

**The PIT test was measuring against the wrong buckets.** A five-quantile grid
partitions outcomes into six buckets with nominal probabilities 5, 20, 25, 25,
20 and 5 percent — not equal. `pit_uniformity` compared observed counts against
*equal* expected counts, so a **perfectly calibrated model scored 0.150 rather
than 0**, and discrimination near that floor was almost nil: a model whose 90%
band covered **63%** scored 0.152, which is 1.4% worse while being catastrophically
miscalibrated. Against the correct buckets the same three score **3.9, 339 and
16 513**. Fixed in `quantile.py`; only one test used it, and it asserted `> 0`,
which the broken floor made trivially true.

**A baseline nobody fitted is one the model beats by construction.**
`GarchBaseline` fits alpha and beta by maximising the training log-likelihood
over a grid, and a test checks the fit beats plausible fixed constants.

**The option-implied baseline is risk-neutral, not physical.** Beating it on
CRPS may only be the variance risk premium — a volatility-selling edge with a
volatility seller's tail, not distributional skill. That caveat prints in the
comparison output, because that result is the one most likely to look strongest.

**How much headroom the 8% actually is** — measured before anyone tunes toward it.
An oracle handed the *true* conditional variance:

| predictable structure | vs random-walk | vs GARCH |
|---|---|---|
| none (plain GARCH) | +2.28% | **+0.19%** |
| 1 step in 15, x9 var | +11.04% | +3.39% |
| 1 step in 7, x9 var | +23.34% | **+11.05%** |

A perfect forecast gets 0.2% against a fitted GARCH on GARCH-generated data —
because there the fitted GARCH *is* the oracle. The 8% margin only appears once
roughly one step in seven carries nine times the variance and is known in
advance. So the bar is demanding rather than routine, and **a comfortable pass
on a structureless series is evidence about the baseline, not the model.**

Every baseline must be beaten separately, not on average: beating a mean lets a
model lose badly to one, and the one it loses to is the informative one.

**Verify**
```bash
python jobs/eval_projection.py --walk-forward --report
# 90% coverage = 90% ± 3%; PIT uniform; CRPS beats both baselines by >= 8%
```
**Done:** **Gate G6.**

---

### L7 · DECISION

#### M70 · IV surface fitter ✅
**Tasks**
- [x] IV from **mid** prices only (never LTP); Black-76 on futures with the right forward
- [x] Quote filters: bid>0, spread cap, OI floor, |δ| ∈ [0.05, 0.95]
- [x] SVI (or SABR) fit per expiry with no-arbitrage constraints (calendar + butterfly)
- [x] Store surface **parameters**; expose level/slope/curvature as features
- [x] Staleness: refuse to serve a surface older than N seconds

**Built** `libs/ms_options/surface.py`, `tests/unit/test_iv_surface.py` (39 tests),
`jobs/surface_report.py`. Raw SVI in total variance, fitted with a pure-Python
deterministic Nelder-Mead (no scipy; two fits of one chain must be identical or
a recommendation cannot be reproduced). Recovers known parameters to 1.3e-12.

**The forward comes from the chain, not the spot.** `S·e^(rT)` is wrong for an
index by the dividend yield you do not know, and the error appears as a stable,
entirely manufactured skew between call- and put-implied vols that a model will
happily learn. `synthetic_forward` recovers it from put-call parity at the
nearest-the-money strike, and *says so in `method`* when it had to fall back —
a caller that cannot tell which forward it got will not notice the skew.

**No-arbitrage is enforced twice, and that is not redundant.** A penalty inside
the objective steers the optimiser; whenever fit error outweighs it, the result
is a surface with excellent RMSE that implies a negative probability density.
Every option price on it is positive. So `arbitrage_report()` re-checks the
fitted result independently and `require_arbitrage_free()` refuses rather than
serves.

Durrleman's g was verified against an independent route — the second difference
of Black-76 call prices in strike, i.e. the discounted risk-neutral density —
in both directions.

**A mutation test found a real gap.** Deleting the `w''/2` term from g left all
six butterfly tests passing: the truncated formula is still right on obviously
broken surfaces and obviously clean ones, and wrong in the middle, which is
where every fitted surface lives. A search found 14 parameter sets where the
term flips the verdict; one is now pinned as a test, with its density confirmed
positive everywhere.

Also mutation-tested: LTP fallback, spot-derived forward, staleness downgraded
to a warning, and calendar checks in insertion rather than time order.

**Verify**
```bash
pytest tests/unit/test_iv_surface.py -k "no_butterfly_arb or no_calendar_arb or rejects_stale or mid_not_ltp"
python jobs/surface_report.py --demo    # synthetic chain; --date needs recorded option data
```

---

#### M71 · Greeks engine ✅
**Tasks**
- [x] Delta, gamma, vega, theta, rho — from the fitted surface, not vendor values
- [x] Portfolio aggregation across legs and positions
- [x] Validate against an independent reference implementation

**Built** `libs/ms_options/greeks.py`, `tests/unit/test_greeks.py` (37 tests).
Black-76 itself was already in `pricing.py` — that was the ◐. What was missing
is the part that makes those numbers hedge correctly.

**Black-Scholes delta holds volatility constant; a real surface does not.**
The strike stays put, `k = ln(K/F)` shifts when the forward moves, and the smile
returns a different vol at the same strike. The delta that hedges is
`dV/dF = dV/dF|sigma + vega * dsigma/dF`, and the second term is not small: on
this Nifty-shaped smile a 500-point-OTM call has a Black-76 delta of **0.2839**
and a true delta of **0.2940** — 3.6% more. Ten lots hedged on the first number
sit about seven units of forward short, every day, in the same direction. It
presents as slippage, which is a thing that exists and that everyone expects to
see some of.

Gamma carries the same chain rule with four terms
(`Gamma + 2*vanna*sigma_F + volga*sigma_F^2 + vega*sigma_FF`), where one wrong
sign yields a number that is smooth, positive and wrong. So every derivative is
pinned against a **full reprice**: bump the forward, recompute k, ask the SVI
slice for the vol there, price again. That route shares no algebra with the
analytic one, which is what "independent reference implementation" has to mean
to be worth anything.

**Vega is never one number.** A 7-day vega and a 60-day vega are different risks
sharing a unit — front vol moves two to three times as much on any day worth
hedging. Summed, a book reports vega-neutral while short the front and long the
back: a real position, real P&L, and no number describing it. `PortfolioGreeks`
buckets vega by expiry and exposes the total only as `vega_gross`, named so it
cannot be mistaken for a net.

Mutation-tested: dropping the smile term, flipping the sign of `dsigma/dF`,
dropping volga from gamma, collapsing vega into one bucket, and bypassing the
staleness check during aggregation. All five caught.

**Verify** `pytest tests/unit/test_greeks.py -k "matches_reference_within_tol or portfolio_aggregation"`

---

#### M72 · Structure enumerator & liquidity filter ✅
**Tasks**
- [x] Candidates: naked, vertical debit, straddle/strangle, iron condor, calendar *(later)*, no-trade
- [x] **Liquidity eligibility**: OI floor, recent volume floor, spread ≤ x% of mid
- [x] Strike grid limited to eligible strikes only

**Built** `libs/ms_options/structures.py`, `tests/unit/test_structures.py` (32 tests).
Legs are `greeks.Leg`, so any candidate can be handed to `greeks.aggregate` and
priced without a second representation.

**No-trade is a candidate, not a fallback.** It is returned on every chain,
including an empty one, and it is first in the list. An enumerator that returns
only tradeable structures hands the ranker a set where *something* always wins,
and "the best of a bad set" arrives in exactly the same shape as "a good
opportunity" — a structure with a score. Making doing nothing compete on the
same axis is what lets a quiet Tuesday recommend no position for a reason,
rather than through an exception path someone has to remember to write.

**Per-leg liquidity passing does not make a structure tradeable.** Crossing cost
compounds with leg count: four legs each inside the spread cap is a round trip
crossing four spreads, and on a condor whose maximum profit is the credit
received that routinely exceeds everything the structure could ever make — while
every individual leg passes every individual filter. `viable_after_crossing`
compares crossing cost against what the structure can actually win, and an
unbounded structure is judged against its own debit, because "can make unlimited
money" is not an answer to "can it clear its spread".

The grid is filtered **before** structures are built, and `Grid.nearest` searches
only the grid — a nearest-strike helper over the full ladder returns a strike
that looks right, is not eligible, and fails at the broker rather than at the
filter. Missing `oi` or `volume` is a rejection, not a pass: an absent field is
not evidence of liquidity.

Mutation-tested: dropping no-trade, treating missing fields as fine, letting
`nearest` reach past the grid, disabling the crossing filter, and reversing the
call vertical into a credit spread. All five caught.

**Verify** `pytest tests/unit/test_structures.py -k "illiquid_excluded or no_trade_always_candidate"`

---

#### M73 · Monte Carlo P&L engine
**Tasks**
- [ ] Reprice every candidate along M60's paths, minute by minute
- [ ] Evolve IV using M56 + a stochastic component negatively correlated with spot
- [ ] Apply exit rules (target / stop / time stop / vol stop) per path
- [ ] Subtract full costs (M74) on entry and exit, both legs
- [ ] Output: `E[P&L]`, `P(profit)`, `CVaR₅`, path max-drawdown
- [ ] p99 ≤ 90 ms for the full candidate set

**Verify**
```bash
pytest tests/unit/test_montecarlo.py -k "costs_applied_both_sides or vega_scenario_present or latency_p99"
python jobs/mc_sanity.py    # a zero-edge input must produce E[PnL] < 0 after costs
```
**Traps:** if a zero-edge scenario shows positive expected P&L, your cost model is wrong. Test that explicitly.

---

#### M74 · Cost model
**Tasks**
- [ ] Brokerage, exchange charge, STT/CTT, regulator fee, stamp duty, GST — all from dated config
- [ ] **Never let an ITM option expire** — always square off (STT trap)
- [ ] Slippage model calibrated from your own paper/live fills, refreshed weekly
- [ ] `breakeven_pct_of_premium` exposed to the UI

**Verify**
```bash
pytest tests/unit/test_costs.py -k "pit_rates or four_orders_for_spread or itm_expiry_flagged"
python jobs/cost_calibration.py --from-fills --weeks 4
```

---

#### M75 · Position sizing
**Tasks**
- [ ] Fractional Kelly, `f = min(0.25·f*, f_max)`
- [ ] Modifiers: prior uncertainty, regime, time-of-day, loss streak, expiry day (×0.5), book correlation
- [ ] Sizes in **lots**, never fractional; rounds down

**Verify** `pytest tests/unit/test_sizing.py -k "never_exceeds_cap or shrinks_on_high_variance or rounds_down"`

---

#### M76 · Risk gate & kill switches ⭐
**Owns:** the veto. Sits between the brain and the money. Nothing bypasses it.

**Tasks**
- [ ] Every limit from blueprint Ch.12 table, loaded from `config/risk_limits.yaml`
- [ ] Kill-switch hierarchy L1 soft → L2 pause → L3 flatten → L4 lockout
- [ ] `flatten_all()` that works even when the model server and feed are down
- [ ] Changing any limit requires an ADR file — enforced by a CI check
- [ ] Manual kill: one CLI command **and** one dashboard button

**Verify**
```bash
pytest tests/stress/test_risk_gate.py -v     # all 10 scenarios from blueprint Ch.12
python jobs/kill_drill.py --dry-run          # monthly drill, logged
```
**Done:** all ten stress scenarios pass with the specified behaviour. Part of **Gate G7**.

---

#### M77 · Recommendation assembler
**Tasks**
- [ ] Emit the full record from blueprint Ch.11 §Recommendation schema
- [ ] `explanation[]` ordered list — a first-class requirement, not UI decoration
- [ ] `gates_passed[]` and `model_versions{}` populated
- [ ] `mode: PAPER|LIVE|DEGRADED` always present
- [ ] Immutable; stored before it is published

**Verify**
```bash
pytest tests/unit/test_recommendation.py -k "schema_complete or explanation_non_empty or immutable"
python jobs/replay_reco.py --reco-id <id>    # full reconstruction from stored record
```

---

### L8 · EXECUTION & SIMULATION

#### M80 · Paper broker adapter
**Tasks**
- [ ] Same interface as the live adapter — this is the point
- [ ] Fills at live prices via M81; maintains positions, margin estimate, P&L
- [ ] Runs every session from Phase 5 onward, forever

**Verify** `pytest tests/integration/test_paper_broker.py -k "same_interface_as_live or position_math"`

---

#### M81 · Fill & slippage model ⚙
**Tasks**
- [ ] Marketable orders fill at far touch + slippage(spread, size/depth)
- [ ] Limit orders fill only on trade-through, with a queue-position discount
- [ ] Partial fills
- [ ] **Conservative tie rule**: target and stop touched in the same bar → assume stop filled first
- [ ] Recalibrate from real fills once live

**Verify** `pytest tests/unit/test_fills.py -k "conservative_tie or partial_fill or slippage_scales_with_spread"`

---

#### M82 · L2 event-driven simulator ⭐
**Owns:** the only number you are allowed to quote about profitability.

**Tasks**
- [ ] Merged timestamp-ordered stream through the **production code path** (M18 + real engine)
- [ ] Injected decision latency sampled from the measured live distribution
- [ ] **Forced 3-second delay from event timestamp** in every run
- [ ] Full cost and slippage model
- [ ] Output: trade blotter, equity curve, per-class attribution

**Verify**
```bash
python jobs/backtest.py --start <d1> --end <d2> --delay 3s --costs full --report
pytest tests/replay/test_sim_matches_production.py   # sim and live produce identical signals on a replayed day
```
**Done:** ≥12 months, PF ≥ 1.2, positive in ≥3 of 4 quarters, maxDD ≤ 12%, deflated Sharpe > 0. **Gate G7.**

---

#### M83 · Order state machine & reconciliation ✅
**Tasks**
- [x] States: `intent → submitted → acked → partial → filled | rejected | cancelled`
- [x] Reconciliation against broker positions at startup and every N minutes
- [x] Handles: ack lost, duplicate ack, success-but-not-placed, rejection mid-flatten
- [x] Immutable audit log shipped off-host

**Built** `libs/ms_broker/state.py`, `tests/stress/test_order_states.py` (35 tests).

The lifecycle in the task list is missing the state that does the work. A submit
that times out has not failed — it is **UNKNOWN**, and both guesses are wrong:
call it rejected and the retry doubles the position, call it filled and the
system believes it is hedged while flat. So a timeout goes to UNKNOWN and *only
reconciliation* may move it out — enforced, not documented.

Reconciliation compares the **union** of both books, never just ours: iterating
our own positions makes an untracked broker position structurally invisible, and
that is the one mismatch nothing else counts, flattens or watches. It reports
and never trades — a broker snapshot can be stale by seconds, and correcting one
by trading turns a data gap into a real position.

Each of the four guards was mutation-tested (remove the guard, confirm a test
fails): duplicate-fill dedup, union iteration, the UNKNOWN escape, and the
flatten that continues past a rejection. All four caught.

**Verify** `pytest tests/stress/test_order_states.py -k "duplicate_ack or lost_ack or reconcile_mismatch"`

---

#### M84 · Live broker adapter 🔒
**Gated behind G8. Do not build early.**
- [ ] Requires: broker algo registration complete, compliance memo updated
- [ ] `live=True` double-arm (env var + on-disk file)
- [ ] Separate host, separate credentials, stricter limits
- [ ] Runs the paper engine **in parallel** on identical signals, forever

**Verify** `python jobs/live_vs_paper.py --sessions 20 --report   # divergence within tolerance`

---

#### M85 · Stress-scenario suite ✅
- [x] All ten scenarios from blueprint Ch.12 as automated replay tests
- [x] Each defines expected behaviour before implementation

**Verify** `pytest tests/stress/test_scenarios.py -v   # all ten scenarios`

The original line here read ``pytest tests/stress/ -v   # 10/10 pass`` and is
now wrong in a way worth naming rather than quietly editing: ``tests/stress/``
holds 92 tests across three files, so a future reader running it and seeing 92
would conclude the suite had drifted from the plan. The ten scenarios live in
``test_scenarios.py`` (16 tests — several scenarios assert more than one thing),
and that is the file the verify command should point at.

---

### L9 · SURFACE

#### M90 · API gateway & WebSocket
- [ ] FastAPI; REST for history/reports, WebSocket for live state
- [ ] MessagePack framing on the live channel
- [ ] Auth even in single-user mode (habits matter later)

**Verify** `pytest tests/integration/test_api.py -k "ws_reconnect or backpressure or auth_required"`

---

#### M91 · Dashboard UI ◐ *partial*
- [x] Five panels per blueprint Ch.16: event stream · fan chart vs implied · recommendation · calibration/attribution · status bar
- [x] Suppressed events shown greyed — the user must see what was ignored
- [x] Every number hoverable → computation + as-of timestamp
- [ ] Click any recommendation → full replay view
- [x] **Unmissable PAPER/LIVE indicator**

**Built** `services/dashboard/` — `index.html`, `app.css`, `app.js`, `data.js`,
plus `tests/manual/dashboard_checklist.md`. Serve with `python tasks.py dashboard`.

A workspace rather than a page: toolbar, tool rail, canvas chart with a
crosshair, a tabbed drawer, a side panel, a status bar. Canvas rather than SVG
because the crosshair redraws on every pointer move, and swapping attributes on
sixty SVG nodes at 60fps is how a dashboard starts to feel heavy.

**PAPER is structural, not a badge** — the chip carries diagonal hatching and
says what the mode *means*. If you can look at the screen for three seconds and
not know which mode you are in, the requirement has failed regardless of whether
the element is present.

Toggling a chart series **redraws** rather than hiding a layer, so the y-axis
re-fits to what is actually shown; a hidden layer leaves the old scale and the
remaining series then reads against a range that includes something invisible.

All seven events are listed, four dimmed, each naming the specific rule that
suppressed it — including the 10:44 anonymous-social severity-9 item, which is
the AP-hack case: loud, high-severity, low-credibility, shown and not acted on.

Every derived number carries a tooltip saying **how it is computed**, not what
it is called.

**Still open, and marked partial for it:**
- The **Playwright smoke test is not written**. Until it is, nothing here is
  checked on a commit and the manual checklist is the whole verification.
- **Click a recommendation → full replay view** does not exist. The panel shows
  the decision, not the inputs that produced it.
- The workspace reads a session snapshot from `data.js`; it is not wired to the
  API (M90) and does not update.

**Verify** manual checklist in `tests/manual/dashboard_checklist.md` + Playwright smoke test

---

#### M92 · Alerting
- [ ] Channels and rules per blueprint Ch.16 §Alerting; rate-limited
- [ ] Risk breaches go to a channel that cannot be muted

**Verify** `pytest tests/unit/test_alerts.py -k "rate_limited or risk_breach_not_mutable"`

---

#### M93 · Paper ledger & outcome resolver
- [ ] Every signal recorded with simulated fills at live prices
- [ ] Outcome resolved at horizon and written back — predictions and outcomes in one table
- [ ] Never deleted. This is the audit spine.

**Verify** `python jobs/postsession.py --date <day> --verify   # every prediction has an outcome or a documented reason`

---

#### M94 · Attribution & calibration reports
- [ ] Per-class attribution table (n, hit%, avg R, net, verdict)
- [ ] Reliability curve, Brier skill, coverage, edge-realisation slope, cost drag
- [ ] **Class-level kill policy**: ≥25 trades + negative expectancy over a rolling quarter → auto-disable
- [ ] Weekly report generated and archived

**Verify**
```bash
python jobs/weekly_report.py --week <n> --report
pytest tests/unit/test_kill_policy.py -k "auto_disables or requires_change_to_reenable"
```
**Note:** this auto-disable policy does more for long-run profitability than any model improvement.

---

#### M95 · Morning briefing product ✅
**The first thing you ship to a real user (yourself).**
- [x] Runs 06:30 IST daily: overnight events, gap forecast distribution, key levels, today's calendar
- [x] Delivered as a push/email/markdown file
- [x] Scored daily against the actual open

**Built** `libs/ms_ops/briefing.py`, `tests/unit/test_briefing.py` (28 tests),
`jobs/morning_briefing.py`.

**The gate says *20 consecutive* briefings, and "consecutive" is the word that
does the work.** Counting records gives twenty for a system that ran on twenty
scattered mornings across three months — which satisfies the sentence and
answers a different question. `consecutive_streak` walks trading days backwards
from the latest session and stops at the first with no briefing. A holiday is
skipped rather than counted as a miss: a streak that reset every Diwali would
never reach twenty, which is the failure people notice — unlike a scattered
count reaching it while proving nothing.

**Published before the open, scored after it.** `publish` refuses once the
session has opened, against an **injected clock** (R6): a replayed morning must
produce the briefing it would have produced, and one stamped with today's time
while forecasting March is not a record of a forecast. The score is a
**separate later record** pointing back by id — writing it into the briefing
would make that file a place where a published forecast can be edited once the
answer is known.

The gate also fails while any briefing is unscored. A forecast nobody checked
is a claim nobody tested, and a run of those is the shape of a system that looks
operational and is not.

`GapForecast.calibrated` survives into the text a person reads. A 90% band that
is not a 90% band is the most misleading thing a briefing can carry — it is
wrong in the direction that makes the reader more confident.

`--dry-run` records nothing, so a rehearsal cannot pad the streak.

Mutation-tested: counting records instead of walking days, publishing after the
open, a holiday breaking the streak, unscored briefings ignored, a second
briefing for one morning, and the uncalibrated label dropped. All six caught.

**Verify** `python jobs/morning_briefing.py --dry-run` · 20 consecutive automated briefings → part of **Gate G4**

---

### LX · OPS

| ID | Module | Key tasks | Verify |
|----|--------|-----------|--------|
| **MA0** | Orchestration | Prefect flows for nightly/weekly jobs; retries; alerting on failure | `prefect deployment ls` + a forced-failure test |
| **MA1** | Daily rhythm | 06:30 briefing · 08:45 preflight · session · 15:20 forced flatten · 15:45 post-session · 18:00 nightly | `pytest tests/integration/test_daily_rhythm.py` |
| **MA2** | Retraining | Weekly incremental, quarterly full; promote only if it beats incumbent on last 4 weeks OOS | `python jobs/retrain.py --dry-run --report` |
| **MA3** ✅ | Drift monitors | Feature PSI, prediction-distribution KL, rolling Brier; breach blocks auto-execution | `pytest tests/unit/test_drift.py -k "psi_breach_blocks_auto"` |
| **MA4** ✅ | CI pipeline | On every commit: unit + determinism + parity + leakage lint + AP-hack + stress | `python tasks.py ci` · run 33145124573 green 9/9 |
| **MA5** ✅ | Backup & restore | Nightly dump + Parquet sync; **monthly restore drill on a clean machine** | `python jobs/restore_drill.py --target clean-vm` · `--self-test` |


**MA4 was marked done before the pipeline was ever watched, and it was red for
four pushes.** Two causes, both in code written for it: the stage runner assumed
a `.venv` that a CI runner does not have, and mypy failed on stubs this machine
happens to have installed while the declared extras never install them. The
second is the more useful finding — the local environment had drifted from
`pyproject.toml`, and nothing could see that until something built the
environment from the file. A third followed: the `unit` stage ran a smaller
test set than a developer does, because `pyarrow`/`duckdb` sit in the `infra`
extra and CI was not installing it.

The lesson is procedural rather than technical: **a pipeline is not done when it
is written, it is done when a run of it is green.** `gh run watch <id>
--exit-status` after pushing.


---

## 4. AI/ML — the deep section

> Everything in this section obeys eight non-negotiable rules. Read them before writing a training script.

### 4.1 ML engineering laws

| # | Law | Enforcement |
|---|-----|-------------|
| L1 | **Point-in-time only.** Any feature computable only after the decision moment is banned. | M43 leakage linter in CI |
| L2 | **One feature definition, two materialisations.** | M43 parity test in CI |
| L3 | **Purged + embargoed CV, grouped by event.** No `train_test_split` in the repo. | M51; grep check in CI |
| L4 | **Beat M0 or don't ship.** Bootstrap CI on the log-loss delta must exclude zero. | M52 + `eval_model.py --vs M0` |
| L5 | **Calibration is the deliverable.** Reliability slope ∈ [0.85, 1.15] or the model is not usable for sizing. | M59 |
| L6 | **Distributions, not points.** Every output has quantiles or a probability with an interval. | M03 schemas |
| L7 | **Registry only.** Production never loads a model from a file path. | M5A |
| L8 | **Log every trial.** Deflated Sharpe needs the trial count; an unreported count makes a Sharpe meaningless. | M51 experiment log |

### 4.2 Where each model sits in the pipeline

```
news ──▶ [AI-1 embed] ──▶ [AI-2 relevance] ──▶ [AI-3 classify] ──▶ [AI-4 severity]
                                                      │
                                                      ▼
                                            [AI-5 impact priors]
                                                      │
market ──▶ features ◀────────────────────────────────┘
              │
              ├──▶ [AI-7 regime] ──────┐
              ├──▶ [AI-8 direction] ───┤
              ├──▶ [AI-9 quantiles] ───┼──▶ [AI-14 ensemble + calibration] ──▶ [AI-15 paths] ──▶ fan chart
              ├──▶ [AI-10 volatility] ─┤                     │
              └──▶ [AI-11 IV change] ──┘                     ▼
                                                    [AI-12 meta-label] ──▶ trade / no-trade
overnight ──▶ [AI-6 gap forecaster] ──▶ morning briefing
```

### 4.3 Model cards

Format: **Task → Target → Features → Data needed → Algorithm → Training → Output → Gate → Latency → Fallback**

---

**[AI-1] Sentence embedder** · module M23/M26
- **Task** dense representation of headlines for dedup, novelty, retrieval
- **Algorithm** pre-trained multilingual sentence encoder; **no fine-tuning in v1**
- **Training** none. Cache embeddings keyed by content hash.
- **Output** 384–768-d vector · **Latency** p95 ≤ 15 ms batched
- **Gate** dedup recall/precision downstream (M23)
- **Fallback** SimHash only

---

**[AI-2] Relevance gate** · module M26
- **Task** binary: could this item conceivably move Indian markets?
- **Target** human-labelled `relevant / noise` (from the M27 gold set)
- **Features** embedding + bloom-filter hits + source credibility
- **Data needed** ~3 000 labelled items
- **Algorithm** logistic regression on embeddings. Deliberately tiny.
- **Training** class-weighted, threshold tuned for **recall ≥ 0.99**
- **Gate** recall ≥ 0.99 · p95 ≤ 5 ms · rejects ≥ 95% of traffic
- **Fallback** keyword filter alone
- **Trap** optimising this for precision destroys the funnel. It is a recall device.

---

**[AI-3] Event classifier** · module M27
- **Task** multi-label event class + polarity
- **Target** frozen taxonomy (`config/taxonomy.yaml`)
- **Features** headline + lead paragraph tokens; source class as a categorical
- **Data needed** 2 000–3 000 gold items, ≥50 per class for the common ones; LLM pre-label → human correct
- **Algorithm** fine-tuned financial/multilingual encoder, multi-label head (BCE)
- **Training** class-balanced sampling; 5-fold **temporal** split; temperature calibration; early stop on macro-F1
- **Output** class probabilities + polarity ∈ [−1,1]
- **Gate** macro-F1 ≥ 0.80 · recall(sev≥7) ≥ 0.95 · noise FP ≤ 2% · p95 ≤ 120 ms
- **Fallback** rules + keyword taxonomy, flagged `DEGRADED`
- **Trap** polarity is *risk-on/risk-off for Indian equities*, not generic sentiment. A rate cut is negative news about growth and positive polarity for equities. Label it that way.

---

**[AI-4] Severity head** · module M27
- **Task** regress 0–10 market-impact severity
- **Target** human anchors → **then empirically recalibrated against realised |AR₃₀|**
- **Features** shared encoder trunk with AI-3
- **Data needed** 400–600 anchored items with a written rubric, 2 annotators, IAA measured
- **Algorithm** regression head + isotonic remap `severity_cal = F⁻¹_{|AR30|}(F̂_sev(raw))`
- **Gate** **Spearman(severity_cal, |AR₃₀|) ≥ 0.35** ← the real gate
- **Fallback** class-level median severity
- **Trap** everyone skips the recalibration step. It is the step that makes severity mean something.

---

**[AI-5] Hierarchical Bayesian impact prior** · module M32
- **Task** conditional distribution of abnormal return given event class + severity
- **Target** `AR_h` from the event study (M31)
- **Algorithm** two-level normal hierarchical model, class nested in family, with shrinkage
- **Training** MCMC or conjugate closed form; fit γ for `g(s)=(s/5)^γ`
- **Output** posterior mean, credible interval, `n_eff` per (class, horizon)
- **Gate** ≥15 classes with `n_eff ≥ 8`; ≥5 with CI excluding 0; CI coverage validated by simulation
- **Fallback** family prior; below that, `UNKNOWN` → volatility-only response
- **Trap** with n=3 the posterior *should* be dominated by the family prior. If it isn't, your τ is wrong.

---

**[AI-6] Overnight gap forecaster** · module M35 — **first shippable model**
- **Task** distribution of tomorrow's opening gap
- **Target** `(open_t+1 − close_t) / close_t`
- **Features** overnight global futures, GIFT Nifty, crude, DXY, USD/INR, overnight event severities/classes, prior-day context, regime, day-of-week
- **Data needed** ≥2 years of sessions (~500 samples) — small but the signal is strong
- **Algorithm** quantile gradient boosting (pinball loss) at α ∈ {.05,.25,.5,.75,.95}
- **Training** walk-forward; **hard PIT cutoff at 08:45 IST**
- **Gate** CRPS beats `gap = β × overnight futures` by ≥10% OOS; 90% coverage within ±3%
- **Latency** irrelevant (batch, once daily) — which is exactly why this ships first
- **Fallback** the naive baseline itself

---

**[AI-7] Regime model** · module M45
- **Task** label the current market state
- **Features** RV percentile, trend efficiency ratio, spread/liquidity, IV term slope, GEX sign
- **Algorithm** Gaussian HMM (3–5 states); alternative: k-means on standardised features
- **Training** fit on history; **filtered (not smoothed) state** at serving — smoothing peeks at the future
- **Gate** states are statistically distinct (return/vol/persistence differ); no-lookahead test passes
- **Fallback** volatility-percentile buckets
- **Trap** using the smoothed state in backtests is a subtle and devastating leak. Test for it.

---

**[AI-8] Direction model** · module M53 — the workhorse
- **Task** `P(triple-barrier label = +1)` at h ∈ {5,15,30} min
- **Features** groups A, C, D, E, F (+B if available); ~120–250 total
- **Data needed** as many **event-conditioned** samples as possible; unconditional samples for pretraining the base rate
- **Algorithm** LightGBM, multiclass or one-vs-rest per horizon
- **Training** purged CV grouped by event · sample uniqueness weights · early stopping · modest depth (overfitting is the default) · SHAP-driven feature pruning
- **Output** calibrated probability + interval
- **Gate** beats M0 on log loss (bootstrap CI excludes 0) · AUC ≥ 0.55 on high-severity samples · reliability slope ∈ [0.85,1.15]
- **Latency** p99 ≤ 20 ms
- **Fallback** M0
- **Trap** if AUC looks like 0.65+, you have a leak. Go back to L3 of the leakage checklist before celebrating.

---

**[AI-9] Quantile model** · module M54
- **Task** `q_α(r_h)` for α ∈ {.05,.25,.5,.75,.95}
- **Algorithm** LightGBM with pinball objective, one model per (α, horizon); enforce monotonicity post-hoc
- **Gate** pinball beats RW+GARCH by ≥8%; coverage within ±3%; PIT uniform
- **Fallback** Gaussian quantiles from AI-10's σ forecast

---

**[AI-10] Volatility model** · module M55 — **the most forecastable target in the system**
- **Task** realised volatility over the next h minutes
- **Target** range-based RV (Garman–Klass / Parkinson) — far more efficient than close-to-close
- **Features** HAR components (daily/weekly/monthly RV), intraday RV path, IV level, event features, time-of-day
- **Algorithm** HAR-RV baseline **+ LightGBM on the residual** (do not replace HAR, augment it)
- **Training** log-target regression; evaluate with QLIKE, not MSE
- **Gate** beats HAR-RV alone on QLIKE OOS
- **Fallback** HAR-RV
- **Note** if the direction model fails G5, this model plus AI-11 is your pivot. Invest in it accordingly.

---

**[AI-11] IV-change model** · module M56 — **decides options profitability**
- **Task** `Δσ_IV` over the next 15–30 min
- **Features** IV rank/percentile, term-structure slope, skew, time to expiry, VRP (IV − forecast RV), event class + severity, scheduled-event proximity, realised-implied spread
- **Data needed** requires the purchased historical option chain — this is why that purchase is a hard dependency
- **Algorithm** gradient boosting; separate models for scheduled vs unscheduled events
- **Patterns to learn** pre-event IV expansion · post-release collapse · post-shock overshoot and mean reversion · expiry-day decay
- **Gate** sign accuracy ≥ 65% OOS; beats "no change" on MAE
- **Fallback** class-level mean ΔIV from the impact library (AI-5)
- **Trap** a directionally correct trade that loses to IV crush is the single most common way this project loses money. This model is the defence.

---

**[AI-12] Meta-labeller** · module M57
- **Task** given the primary model's call, should we take it?
- **Target** was the primary call profitable after costs?
- **Features** primary probability + its interval, regime, liquidity, time-of-day, event agreement, prior `n_eff`, cost-to-edge ratio
- **Algorithm** LightGBM binary; heavily class-weighted toward "skip"
- **Gate** raises precision at fixed recall; improves post-cost expectancy in the L2 simulator
- **Fallback** threshold rule on primary probability
- **Note** its job is to say **no**. A meta-labeller that approves most trades is not doing its job.

---

**[AI-13] Sequence models (TCN / transformer)** · module M58 — **deferred**
- **Task** direction + multi-horizon distribution from raw sequences
- **Build only after** M53 is trained, evaluated and beaten-or-not by a simple model
- **Data needed** substantially more than gradient boosting; highest overfitting risk in the project
- **Gate** must improve the *ensemble*, not just its own standalone metric
- **Decision rule** if it does not improve the ensemble after two honest attempts, **drop it without sentiment**

---

**[AI-14] Ensemble, calibration & conformal** · module M59
- **Task** combine, calibrate, quantify uncertainty
- **Algorithm** stacking with time-aware OOF predictions → **logistic regression** meta-learner (keep it simple); regime-conditional weights; isotonic calibration on a later slice; split conformal for magnitude
- **Gate** reliability slope ∈ [0.85,1.15] · Brier skill > 0.02 vs climatology · conformal coverage within tolerance
- **Fallback** best single model with its own calibration
- **Trap** a complex stacker overfits instantly. If the meta-learner has more than a handful of parameters, simplify it.

---

**[AI-15] Path sampler** · module M60
- **Task** simulate intraday paths whose terminal distribution matches AI-9
- **Algorithm** GARCH-style simulator with student-t innovations, conditioned on AI-10's σ forecast; optional conditional generative model later
- **Calibration** rescale so terminal quantiles match AI-9's heads
- **Gate** terminal quantiles match within tolerance; volatility clustering statistically present; 5 000 paths < 60 ms
- **Fallback** Gaussian random walk with AI-10's σ

---

**[AI-16] Slippage model** · module M81
- **Task** expected slippage given spread, size, depth, volatility, time-of-day
- **Target** your own realised fills (paper first, then live)
- **Algorithm** simple regression / gradient boosting — deliberately unsophisticated
- **Gate** predicted vs realised slippage MAE within tolerance on live fills
- **Note** the most valuable model you will build **after** going live. Refresh weekly.

---

### 4.4 Data volume reality check

| Model | Minimum viable samples | Where they come from | If you don't have enough |
|---|---|---|---|
| AI-2 relevance | ~3 000 labelled | gold set | LLM pre-labelling |
| AI-3 classifier | 2 000–3 000, ≥50/class | gold set | merge rare classes into families |
| AI-4 severity | 400–600 anchored | manual, 2 annotators | keep the taxonomy coarser |
| AI-5 impact priors | ≥8 events/class | historical archive | family-level prior; `UNKNOWN` |
| AI-6 gap | ~500 sessions (2 yr) | daily data | usable from ~250 |
| AI-8 direction | tens of thousands of minutes; **event-conditioned samples are the constraint** | tick history | this is what P1–P2 exist to accumulate |
| AI-10 volatility | same, but easier — volatility is smoother | tick history | HAR baseline works from day 1 |
| AI-11 IV change | needs the purchased chain history | vendor | **blocked without it** — buy at P5 |
| AI-13 sequence | 5–10× AI-8 | time | deferred, correctly |

> **The binding constraint on this entire project is event-conditioned sample count.**
> This is why M12 (the recorder) is day 3 and not month 6.

---

## 5. Build order

Sprints are two weeks. Gate references are `docs/gates.md`.

| Sprint | Weeks | Modules | Gate |
|---|---|---|---|
| S1 | 1–2 | M00 M01 M02 M03 M06 M10 M11 **M12** M14 | **G0** |
| S2 | 3–4 | M04 M05 M16 M13 | |
| S3 | 5–6 | M17 M18 M19 M15 MA4 MA5 | **G1** · MA4/MA5 ✅ |
| S4 | 7–8 | M20 M21 M22 | |
| S5 | 9–10 | M23 M24 M25 | **G2** |
| S6 | 11–12 | M26 gold-set labelling (start early, it is slow) | |
| S7 | 13–14 | M27 M28 M29 | **G3** |
| S8 | 15–16 | M30 M31 M32 M33 | |
| S9 | 17–18 | M35 M34 M95 | **G4** ← *first shipped product* |
| S10 | 19–20 | M40 M41 M42 **M43** | |
| S11 | 21–22 | M44 M45 M50 M51 M52 · **order the historical option chain now** | |
| S12 | 23–24 | M53 M55 M57 M59 M5A | **G5** ← *the decision gate* |
| S13 | 25–26 | M54 M60 M61 M62 | **G6** |
| S14 | 27–28 | M70 M71 M56 | |
| S15 | 29–30 | M72 M73 M74 M75 | |
| S16 | 31–32 | M76 M77 M80 M81 | |
| S17 | 33–34 | M82 M85 · locked-holdout evaluation | **G7** |
| S18 | 35–36 | M90 M91 M92 M93 MA0 MA1 | |
| S19 | 37–38 | M94 MA2 MA3 M83 | ✅ complete |
| S20+ | 39–45 | **live paper trading, 60 sessions** · weekly attribution reviews | **G8** |
| — | 46+ | M84 · capital ladder R1→R4 | live |

### Parallelisation (if a second person joins)
- **Track A (data/infra):** S1–S5, then M43, M82, M90–M92, MA0–MA5
- **Track B (ML/research):** joins at S6, owns M26–M35, M45, M50–M62, all AI cards
- Sync point: **M43 parity test** — neither track may proceed past it independently

---

## 6. Progress tracker

Update this table at the end of every sprint. It is the honest picture.

| Layer | Modules | ☑ Verified | ◐ In progress | ☐ Not started |
|---|---:|---:|---:|---:|
| L0 Foundation | 7 | 7 | 0 | 0 |
| L1 Market Data | 11 | 10 | 1 | 0 |
| L2 News & Events | 10 | 0 | 0 | 10 |
| L3 Impact | 6 | 5 | 0 | 1 |
| L4 Features | 6 | 4 | 1 | 1 |
| L5 Models | 12 | 8 | 1 | 3 |
| L6 Projection | 3 | 0 | 0 | 3 |
| L7 Decision | 8 | 4 | 1 | 3 |
| L8 Execution | 6 | 4 | 0 | 2 |
| L9 Surface | 6 | 4 | 0 | 2 |
| LX Ops | 6 | 3 | 0 | 3 |
| **Total** | **81** | **49** | **4** | **28** |

### What the module count does not say

**49 of 81 is 60%, and read alone it is misleading in both directions.**

*It understates the foundation.* L0 and L1 are effectively finished, and they are
the layers everything else stands on: point-in-time correctness, deterministic
replay, a leak-free CV harness, quality gates. L5 at 8/12 includes the whole
measurement apparatus - calibration, conformal intervals, the trial log, null
tests for every fitted component.

*It overstates the distance travelled.* Four layers have not started at all -
News & Events (0/10), Projection (0/3), Execution (0/6), Surface (0/6) - and two
of those are real work rather than plumbing. The remaining 44 modules are not
evenly sized and are not evenly hard.

*And it says nothing about the only question that matters.* **One component has
run on real market data**: the overnight gap forecaster, on 496 sessions of free
daily bars. It did not pass its gate. Everything else - direction, volatility,
ensemble, meta-labeller, sizing, the risk gate - has run only on synthetic data
where the answer was known in advance.

That is not a criticism of those modules. It is the honest statement of what has
been *demonstrated* versus what has been *built*, and the two are far apart.

**What exists is the scale, not the goods.** The scale now works: two full audits
have found nine defects, every one of which produced a working pipeline,
plausible numbers and no error - a 30% false-positive rate that looked like a
finding, a gamma of 0.5 invented from noise, an inert regularisation constant
that read as a defence, a theta sign error hiding under eighty-two passing tests,
a sizing defence that no code path could reach.

Real progress resumes when a broker feed arrives. Until then every new module is
another thing waiting on the same input.

### Infrastructure status

| | |
|---|---|
| Docker stack | up - TimescaleDB 2.29.2 on **5433**, Redis 7 on 6379, both healthy |
| Migrations | applied; hypertables live |
| Test suite | 814 passing, **zero skipped** (the infra tests ran for the first time) |
| Broker feed | **not connected** - the one remaining external dependency |

Postgres is published on 5433 rather than 5432 because a stock Windows
PostgreSQL service owns 5432 and starts at boot; connecting there reports a
password failure against a database that exists, which reads like a wrong
credential and is a wrong server. See `docs/editor-setup.md`.

### Gate ledger
| Gate | Blocks | Status |
|---|---|---|
| G0 Foundation | everything | ☐ |
| G1 Data spine | all research | ☐ |
| G2 Event store | all event work | ☐ |
| G3 Classification | impact library | ☐ |
| G4 Impact + gap | features | ☐ |
| **G5 Baseline models** | **the entire options stack** | ☐ |
| G6 Projection | fan chart | ☐ |
| G7 Options + sim | paper trading | ☐ |
| G8 Live paper | real money | ☐ |

---

## 7. What to do right now

1. `M00` — repo skeleton, docker compose up.
2. `M10` `M11` — instrument master + feed adapter.
3. **`M12` — the recorder. Start it tonight.**
4. `M14` — chain snapshotter. Start it tonight too.
5. Everything else, in sprint order.

> **The recorder is the whole project's foundation.** Models can be rebuilt in a weekend.
> History cannot be recovered at any price. Every day it isn't running is a day of
> event-conditioned samples you will never have — and event-conditioned sample count
> is the binding constraint on whether G5 ever passes.
