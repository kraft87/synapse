-- Migrate episodes embedding column from VECTOR(3072) to VECTOR(2048) for voyage-4-large.
-- All existing rows have is_embedded=FALSE so no embedding data is lost.
ALTER TABLE episodes ALTER COLUMN embedding TYPE VECTOR(2048);

-- Drop old index if it exists.
DROP INDEX IF EXISTS episodes_embedding_idx;

-- Note: pgvector HNSW is limited to 2000 dims. For 2048 dims use either:
--   (a) halfvec type (HNSW up to 4000 half-precision dims), or
--   (b) exact scan (fine for < 50K rows at personal scale).
-- At current volume, exact scan is instant. Revisit if corpus exceeds 50K episodes.
-- CREATE INDEX episodes_embedding_idx ON episodes
--     USING hnsw (embedding vector_cosine_ops)
--     WHERE is_embedded = TRUE;
