-- 012_web_chunks.sql
-- Per-chunk embeddings for web_artifacts.kind = 'web_scrape'.
--
-- Sidecar table (1:N from web_artifacts) rather than reusing chunks (which is
-- scoped to episodes) or synth_documents (whose semantic category — LLM-derived
-- from the user's own content — doesn't fit external scrapes). See architect's
-- review in the council distillation.
--
-- voyage-4-large at 2048 dims, matching the rest of the synapse embedding stack.

CREATE TABLE IF NOT EXISTS web_chunks (
    id              BIGSERIAL PRIMARY KEY,
    web_artifact_id BIGINT NOT NULL REFERENCES web_artifacts(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,
    content         TEXT NOT NULL,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,
    embedding       vector(2048),
    is_embedded     BOOLEAN NOT NULL DEFAULT false,
    content_ts      TIMESTAMPTZ,                    -- inherited from web_artifacts.fetched_at at write
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (web_artifact_id, idx)
);

CREATE INDEX IF NOT EXISTS web_chunks_artifact_idx
    ON web_chunks (web_artifact_id);

CREATE INDEX IF NOT EXISTS web_chunks_embedded_idx
    ON web_chunks (is_embedded)
    WHERE is_embedded = false;

CREATE INDEX IF NOT EXISTS web_chunks_content_ts_idx
    ON web_chunks (content_ts DESC);

-- BM25 over chunk content for hybrid retrieval
CREATE INDEX IF NOT EXISTS web_chunks_search_idx
    ON web_chunks
    USING bm25 (id, content)
    WITH (key_field = 'id');

-- No HNSW: pgvector HNSW caps at 2000 dims; voyage-4-large is 2048. Vector
-- search uses sequential scan, which is fine at our scale (<10k chunks).
-- Revisit when corpus exceeds ~50k chunks — at that point switch to IVFFlat
-- or reduce dimensions via Voyage's `output_dimension` parameter.
