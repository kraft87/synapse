-- schema-check: optional
-- Vector index over chunks.embedding, for recall's chunk candidate source.
--
-- With SYNAPSE_RECALL_CHUNK_CANDIDATES=1 recall also searches the chunks (4 consecutive
-- turns of a session) by embedding, and feeds the episodes of the nearest ones into the
-- pool the cross-encoder reranks. Chunks were embedded all along but, like the episodes
-- before 014, had only the exact scan to be searched with: ~2048 floats per chunk, read on
-- every recall. This is the same halfvec HNSW index 014 built for episodes, on the same
-- expression the query orders by.
--
-- Optional: the setting is off by default, and with it on but this index missing (or an
-- interrupted build, which leaves an INVALID one) the vector source says so in the
-- response's `warnings` and serves nothing, while the keyword source keeps working. The
-- 4000-dim halfvec cap that applies to episodes applies here too (apply_schema.sh
-- substitutes the dimensions this database was provisioned with).
--
-- On a POPULATED table, build it out-of-band, CONCURRENTLY, exactly as below: a plain
-- CREATE INDEX takes a lock that blocks chunk inserts for the whole build, and the parallel
-- build needs more /dev/shm than the container has.
--   SET maintenance_work_mem = '512MB';
--   SET max_parallel_maintenance_workers = 0;
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw
--       ON chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
--       WHERE is_embedded = TRUE;
-- This file does the same. If a CONCURRENTLY build fails, drop the INVALID index
-- (DROP INDEX CONCURRENTLY chunks_embedding_hnsw) and run this file again.

SET maintenance_work_mem = '512MB';
SET max_parallel_maintenance_workers = 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE is_embedded = TRUE;
