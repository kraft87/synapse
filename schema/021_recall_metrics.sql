-- 021: recall telemetry table.
--
-- One row per recall() / recall_episodes() call, written fire-and-forget from the
-- recall engine (mcp_server/recall.py::_do_record) — local + SQL-queryable, NOT
-- logfire. Captures per-leg timing, served-payload size (the context cost each
-- recall imposes), pool sizes, rerank model + top relevance score, and the call
-- origin. recall_episodes rows fill only the common columns; the recall-only
-- columns stay NULL (distinguish by `kind`).
--
-- This is an append-only metrics log; prune by created_at when it grows.

CREATE TABLE IF NOT EXISTS recall_metrics (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind             TEXT NOT NULL,            -- 'recall' | 'episodes'
    source           TEXT,                     -- mcp-tool | http | recall-hook:<mode> | ...
    query            TEXT,                     -- truncated to 200 chars
    group_id         TEXT,
    write_feedback   BOOLEAN,
    -- timing (ms)
    ms_total         REAL,
    ms_embed         REAL,
    ms_bm25          REAL,
    ms_vector        REAL,
    ms_kg            REAL,
    ms_web           REAL,
    ms_rerank        REAL,
    -- served-payload shape
    n_facts          INT,
    n_episodes       INT,
    n_entities       INT,
    n_web            INT,
    n_history        INT,
    chars            INT,
    est_tokens       INT,
    -- pool health
    pool_bm25        INT,
    pool_vector      INT,
    pool_fused       INT,
    kg_candidates    INT,
    -- rerank
    rerank_model     TEXT,
    rerank_top_score REAL,
    emb_ok           BOOLEAN
);

CREATE INDEX IF NOT EXISTS recall_metrics_created_idx ON recall_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS recall_metrics_source_idx ON recall_metrics (source);
