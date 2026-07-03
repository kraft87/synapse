CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
    id             BIGSERIAL PRIMARY KEY,
    session_id     TEXT NOT NULL,
    sequence       INT  NOT NULL,
    project        TEXT,
    platform       TEXT,  -- claude_code | cursor | claude_ai
    model          TEXT,
    human_turn     TEXT,
    assistant_turn TEXT,
    content        TEXT NOT NULL,
    content_tsv    TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding      VECTOR(3072),
    is_embedded    BOOLEAN NOT NULL DEFAULT FALSE,
    metadata       JSONB,
    source         TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sequence)
);

CREATE INDEX IF NOT EXISTS episodes_content_tsv_idx ON episodes USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS episodes_project_idx     ON episodes (project);
CREATE INDEX IF NOT EXISTS episodes_session_idx     ON episodes (session_id);
CREATE INDEX IF NOT EXISTS episodes_created_at_idx  ON episodes (created_at DESC);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id               TEXT PRIMARY KEY,
    project                  TEXT,
    platform                 TEXT,
    summary                  TEXT NOT NULL,
    last_summarized_sequence INT  NOT NULL DEFAULT 0,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS session_summaries_project_idx    ON session_summaries (project);
CREATE INDEX IF NOT EXISTS session_summaries_updated_at_idx ON session_summaries (updated_at DESC);

CREATE TABLE IF NOT EXISTS search_cache (
    id          BIGSERIAL PRIMARY KEY,
    query       TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    title       TEXT,
    content     TEXT NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || content)) STORED,
    embedding   VECTOR(3072),
    is_embedded BOOLEAN NOT NULL DEFAULT FALSE,
    project     TEXT,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (query, source_url)
);

CREATE INDEX IF NOT EXISTS search_cache_content_tsv_idx ON search_cache USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS search_cache_query_idx       ON search_cache (query);
CREATE INDEX IF NOT EXISTS search_cache_project_idx     ON search_cache (project);

CREATE TABLE IF NOT EXISTS ingestion_state (
    source           TEXT PRIMARY KEY,
    last_ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Extraction queue: episodes/summaries waiting for KG extraction
CREATE TABLE IF NOT EXISTS extraction_queue (
    id          BIGSERIAL PRIMARY KEY,
    episode_id  BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
    session_id  TEXT,   -- for summary extraction (no episode_id)
    content     TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'episode',  -- episode | summary | manual
    project     TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | done | failed
    attempts    INT  NOT NULL DEFAULT 0,
    error       TEXT,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS extraction_queue_status_idx ON extraction_queue (status, enqueued_at);
