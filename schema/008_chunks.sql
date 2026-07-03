-- Sliding-window chunk documents for richer semantic search.
-- Each chunk spans 3-5 consecutive episodes within a session.

CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    start_sequence  INTEGER NOT NULL,
    end_sequence    INTEGER NOT NULL,
    episode_ids     JSONB NOT NULL,       -- ordered list of episode PKs
    project         TEXT,
    content         TEXT NOT NULL,
    embedding       vector(2048),
    is_embedded     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, start_sequence, end_sequence)
);

CREATE INDEX IF NOT EXISTS chunks_session_idx    ON chunks (session_id);
CREATE INDEX IF NOT EXISTS chunks_project_idx    ON chunks (project);
CREATE INDEX IF NOT EXISTS chunks_embedded_idx   ON chunks (is_embedded) WHERE is_embedded = FALSE;

-- BM25 index for keyword search (ParadeDB)
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks
    USING bm25 (id, content, session_id, project)
    WITH (key_field = 'id');
