-- M16 - core time-series schema.
--
-- Applied once by libs/ms_data/store.py and recorded in schema_migration.
-- Never edit an applied migration; add a new numbered one.

-- TimescaleDB is optional at this stage. The tables and every query work on
-- plain Postgres; the extension adds chunking, compression and continuous
-- aggregates. Making it required here would block development on a machine
-- without it, for no benefit until the archive is large.
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;


-- ---------------------------------------------------------------- ticks ----
CREATE TABLE IF NOT EXISTS tick (
    ts            TIMESTAMPTZ   NOT NULL,
    instrument_id INT           NOT NULL,
    ltp           NUMERIC(14,2) NOT NULL,
    qty           INT,
    bid           NUMERIC(14,2),
    ask           NUMERIC(14,2),
    bid_qty       INT,
    ask_qty       INT,
    oi            BIGINT,
    seq           BIGINT,                 -- vendor sequence, for gap detection
    src           SMALLINT      NOT NULL DEFAULT 0,
    trace_id      TEXT
);

CREATE INDEX IF NOT EXISTS tick_instrument_ts_idx ON tick (instrument_id, ts DESC);


-- --------------------------------------------------------------- candles ---
-- The primary key includes `version`: a late tick creates version 2 beside
-- version 1 rather than overwriting it. A backtest asks what the candle looked
-- like at decision time; a report asks what it looks like now. Both must be
-- answerable, and an UPDATE would destroy the first question permanently.
CREATE TABLE IF NOT EXISTS candle (
    ts            TIMESTAMPTZ   NOT NULL,
    instrument_id INT           NOT NULL,
    tf            SMALLINT      NOT NULL,
    o             NUMERIC(14,2) NOT NULL,
    h             NUMERIC(14,2) NOT NULL,
    l             NUMERIC(14,2) NOT NULL,
    c             NUMERIC(14,2) NOT NULL,
    v             BIGINT        NOT NULL DEFAULT 0,
    oi            BIGINT,
    trades        INT,
    vwap          NUMERIC(14,4),
    synthetic     BOOLEAN       NOT NULL DEFAULT FALSE,
    version       SMALLINT      NOT NULL DEFAULT 1,
    trace_id      TEXT,
    PRIMARY KEY (instrument_id, tf, ts, version)
);

CREATE INDEX IF NOT EXISTS candle_ts_idx ON candle (ts DESC);


-- ---------------------------------------------------------- option chain ---
-- `spot` is stored on every quote row rather than in a separate snapshot table.
-- Denormalised deliberately: the spot must be the one captured WITH the chain,
-- and a join against a separately timestamped spot series is exactly the lag
-- that manufactures a phantom edge in an options backtest.
CREATE TABLE IF NOT EXISTS option_snap (
    ts            TIMESTAMPTZ   NOT NULL,
    underlying_id INT           NOT NULL,
    spot          NUMERIC(14,2) NOT NULL,
    completeness  NUMERIC(5,4)  NOT NULL DEFAULT 1.0,
    expiry        TIMESTAMPTZ   NOT NULL,
    strike        NUMERIC(12,2) NOT NULL,
    opt_type      CHAR(2)       NOT NULL,
    ltp           NUMERIC(12,2),
    bid           NUMERIC(12,2),
    ask           NUMERIC(12,2),
    volume        BIGINT,
    oi            BIGINT,
    oi_change     BIGINT,
    iv            NUMERIC(8,5),           -- NULL when inversion had no opinion
    trace_id      TEXT
);

CREATE INDEX IF NOT EXISTS option_snap_lookup_idx
    ON option_snap (underlying_id, ts DESC, expiry, strike);


-- ------------------------------------------------------------- decisions ---
-- Never deleted. This is the audit spine: predictions and their outcomes live
-- in one table so the class-level kill policy (M94) can always be recomputed.
CREATE TABLE IF NOT EXISTS recommendation (
    reco_id     TEXT PRIMARY KEY,
    ts_emitted  TIMESTAMPTZ NOT NULL,
    mode        TEXT        NOT NULL,
    event_id    TEXT,
    structure   TEXT        NOT NULL,
    payload     JSONB       NOT NULL,     -- the full immutable record
    config_hash TEXT,
    trace_id    TEXT
);

CREATE TABLE IF NOT EXISTS outcome (
    reco_id         TEXT PRIMARY KEY REFERENCES recommendation(reco_id),
    resolved_at     TIMESTAMPTZ NOT NULL,
    realised_return NUMERIC(10,6) NOT NULL,
    predicted_p_up  NUMERIC(6,5)  NOT NULL,
    pnl_gross       NUMERIC(14,2) NOT NULL DEFAULT 0,
    pnl_net         NUMERIC(14,2) NOT NULL DEFAULT 0,
    exit_reason     TEXT          NOT NULL,
    event_class     TEXT
);


-- ------------------------------------------------------------ hypertables --
-- Wrapped so the migration succeeds on plain Postgres too. The tables are
-- correct either way; TimescaleDB adds chunking and compression on top.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('tick', 'ts',
                                  chunk_time_interval => INTERVAL '1 day',
                                  if_not_exists => TRUE);
        PERFORM create_hypertable('candle', 'ts',
                                  chunk_time_interval => INTERVAL '7 days',
                                  if_not_exists => TRUE);
        PERFORM create_hypertable('option_snap', 'ts',
                                  chunk_time_interval => INTERVAL '1 day',
                                  if_not_exists => TRUE);
    END IF;
END
$$;
