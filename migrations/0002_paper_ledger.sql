-- M93 - the paper ledger index.
--
-- The ndjson files under data/ledger/ are the source of truth. This table is a
-- projection of them, loaded by jobs/postsession.py, and it exists so the
-- ledger can be queried rather than scanned.
--
-- If the two ever disagree, the file wins. The ledger must never lose a write,
-- and the moment it most needs to be written is the moment something has gone
-- wrong - which is also when this database is most likely to be unreachable.
--
-- Predictions and outcomes are in ONE table on purpose. Split across two joined
-- by an id, a prediction whose outcome was never written looks identical to one
-- whose outcome row was lost, and both look identical to one that was never
-- meant to have an outcome. Three different situations, one empty join.

CREATE TABLE IF NOT EXISTS paper_ledger (
    entry_id          TEXT          PRIMARY KEY,
    reco_id           TEXT          NOT NULL,

    -- the prediction
    ts_predicted      TIMESTAMPTZ   NOT NULL,
    symbol            TEXT          NOT NULL,
    event_class       TEXT          NOT NULL,
    horizon_minutes   INT           NOT NULL,
    p_up              NUMERIC(6,5)  NOT NULL,
    p_lower           NUMERIC(6,5),          -- NULL means the model gave no interval
    entry_price       NUMERIC(14,2) NOT NULL,
    lots              INT           NOT NULL,
    is_long           BOOLEAN       NOT NULL,
    mode              TEXT          NOT NULL,

    -- the outcome, written back at the horizon
    resolution        TEXT          NOT NULL DEFAULT 'PENDING',
    ts_resolved       TIMESTAMPTZ,
    exit_price        NUMERIC(14,2),
    gross_pnl         NUMERIC(14,2),
    costs             NUMERIC(14,2),
    net_pnl           NUMERIC(14,2),
    was_right         BOOLEAN,               -- NULL while unresolved, never FALSE

    -- why there is no outcome, when there is no outcome
    unresolved_reason TEXT          NOT NULL DEFAULT '',

    CONSTRAINT resolution_is_known
        CHECK (resolution IN ('PENDING', 'RESOLVED', 'UNRESOLVABLE')),

    -- An unresolvable prediction must say why. A row with no outcome and no
    -- explanation cannot be told apart from one that was lost, and that single
    -- ambiguity makes every report built on this table unauditable.
    CONSTRAINT unresolvable_has_a_reason
        CHECK (resolution <> 'UNRESOLVABLE' OR length(trim(unresolved_reason)) > 0),

    -- A resolved prediction must have a price to have been resolved against.
    CONSTRAINT resolved_has_an_exit
        CHECK (resolution <> 'RESOLVED' OR exit_price IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS paper_ledger_day_idx
    ON paper_ledger (ts_predicted DESC);

CREATE INDEX IF NOT EXISTS paper_ledger_class_idx
    ON paper_ledger (event_class, ts_predicted DESC);

-- The query the post-session check runs: what is past its horizon and still
-- has no outcome.
CREATE INDEX IF NOT EXISTS paper_ledger_pending_idx
    ON paper_ledger (ts_predicted)
    WHERE resolution = 'PENDING';
