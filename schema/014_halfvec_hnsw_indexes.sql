-- 014_halfvec_hnsw_indexes.sql
-- Vector ANN indexes for recall(). The embeddings are 2048-dim (voyage-4-large),
-- which exceeds pgvector's 2000-dim limit for HNSW on the full `vector` type
-- (see 003_voyage_dims.sql). That migration deferred the index with a documented
-- trigger: "exact scan is instant at this volume; revisit if corpus exceeds 50K
-- episodes." Triggered 2026-06-03 at 62,866 episodes — the brute-force seq scan
-- that was instant at small scale became ~0.9-1.3s per vector search (EXPLAIN
-- ANALYZE: Parallel Seq Scan over all episodes, then top-N sort).
--
-- Fix: HNSW indexes on `embedding::halfvec(2048)` (half precision; pgvector's
-- halfvec index limit is 4000 dims). Loss-free for what recall() serves:
-- recall@10 vs exact = 1.000, recall@100 = 0.981, and needle keyword recall on
-- the served context is IDENTICAL to exact (delta +0.000, 42-query A/B). Latency:
-- episode vector search 878ms -> ~15ms (38x); recall_episodes() 2.3s -> 0.77s.
--
-- Queries must ORDER BY `embedding::halfvec(2048) <=> $q::halfvec(2048)` to be
-- served from these indexes (see mcp_server/recall.py::_vector_table), and the
-- session must `SET hnsw.ef_search = 200` (default 40 under-recalls a 100-deep
-- fetch; set in _ensure_pg). ef_search beyond ~400 degrades back toward a scan.
--
-- NOTE: on a POPULATED table, build these CONCURRENTLY out-of-band — a plain
-- CREATE INDEX takes a SHARE lock that blocks writes for the multi-minute build,
-- and the parallel build needs more /dev/shm than the container has, so disable
-- parallelism:
--   SET maintenance_work_mem = '512MB';
--   SET max_parallel_maintenance_workers = 0;
--   CREATE INDEX CONCURRENTLY episodes_embedding_hnsw
--       ON episodes USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
--       WHERE is_embedded = TRUE;
-- The IF NOT EXISTS form below is instant on a fresh/empty DB and a no-op if the
-- index already exists (as it does in production, built 2026-06-03).

CREATE INDEX IF NOT EXISTS episodes_embedding_hnsw
    ON episodes USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE is_embedded = TRUE;

CREATE INDEX IF NOT EXISTS web_chunks_embedding_hnsw
    ON web_chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE is_embedded = TRUE;

-- synth_documents (669 rows) and chunks (not searched by recall() since the
-- episode-blend swap) are left on exact scan — tiny / unused, indexing not worth it.
