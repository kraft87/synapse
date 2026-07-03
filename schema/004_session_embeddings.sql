-- Add embedding support to session_summaries for hierarchical retrieval.
-- Pass 1 of two-pass search: ANN over session embeddings to find top-N sessions.
-- Pass 2: ANN over episodes within those sessions.
ALTER TABLE session_summaries ADD COLUMN IF NOT EXISTS embedding VECTOR(2048);
ALTER TABLE session_summaries ADD COLUMN IF NOT EXISTS is_embedded BOOLEAN NOT NULL DEFAULT FALSE;

-- Note: pgvector HNSW is limited to 2000 dims. Exact scan is fine at personal scale.
-- CREATE INDEX IF NOT EXISTS session_summaries_embedding_idx ON session_summaries
--     USING hnsw (embedding vector_cosine_ops)
--     WHERE is_embedded = TRUE;
