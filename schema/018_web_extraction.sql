-- 018_web_extraction.sql
-- Web → KG extraction lane (task #68 / spec: synapse-web-to-graph-spec.md).
--
-- Three pieces:
--   1. extraction_queue grows a nullable web_chunk_id so web chunks ride the
--      same claim/drain machinery as episode chunks (content_type='web_chunk').
--      The partial UNIQUE index makes enqueue idempotent via ON CONFLICT.
--   2. kg_relationships (the Postgres KG mirror, schema 017) grows a nullable
--      web_artifact_id — edge provenance for web-derived facts. FalkorDB is
--      intentionally NOT touched: provenance lives in the mirror (the future
--      canonical store per #67); rollback joins uuid → FalkorDB delete.
--   3. web_artifacts.kind gains 'research_brief' — run-research briefs ingested
--      as first-class docs (synthesized, zero-chrome) that ride the same
--      chunk/embed/contextualize/extract lane.

ALTER TABLE extraction_queue
    ADD COLUMN IF NOT EXISTS web_chunk_id BIGINT REFERENCES web_chunks(id) ON DELETE CASCADE;

CREATE UNIQUE INDEX IF NOT EXISTS extraction_queue_web_chunk_uq
    ON extraction_queue (web_chunk_id)
    WHERE web_chunk_id IS NOT NULL;

ALTER TABLE kg_relationships
    ADD COLUMN IF NOT EXISTS web_artifact_id BIGINT;

-- Rollback / audit surface: "all edges derived from web content" without a seq scan.
CREATE INDEX IF NOT EXISTS kg_rel_web_artifact
    ON kg_relationships (web_artifact_id)
    WHERE web_artifact_id IS NOT NULL;

-- Widen the kind CHECK (inline CHECK from 011 auto-named web_artifacts_kind_check).
ALTER TABLE web_artifacts DROP CONSTRAINT IF EXISTS web_artifacts_kind_check;
ALTER TABLE web_artifacts ADD CONSTRAINT web_artifacts_kind_check
    CHECK (kind IN ('web_scrape', 'search_result_set', 'research_job_ref', 'research_brief'));
