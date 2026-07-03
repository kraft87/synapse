-- Migrate from tsvector/GIN to ParadeDB BM25 via pg_search
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Drop generated tsvector columns and their GIN indexes
ALTER TABLE episodes     DROP COLUMN IF EXISTS content_tsv;
ALTER TABLE search_cache DROP COLUMN IF EXISTS content_tsv;
DROP INDEX IF EXISTS episodes_content_tsv_idx;
DROP INDEX IF EXISTS search_cache_content_tsv_idx;

-- BM25 index on episodes: search over content + filter fields
CREATE INDEX IF NOT EXISTS episodes_search_idx ON episodes
    USING bm25 (id, content, session_id, project)
    WITH (key_field = 'id');

-- BM25 index on search_cache: search over title + content + filter fields
CREATE INDEX IF NOT EXISTS search_cache_search_idx ON search_cache
    USING bm25 (id, content, title, query, project)
    WITH (key_field = 'id');
