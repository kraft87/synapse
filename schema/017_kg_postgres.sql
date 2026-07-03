-- 017_kg_postgres.sql
-- Knowledge-graph entities + relationships in Postgres, mirroring the live
-- FalkorDB write path (ingestion/falkordb_client.py: upsert_node / create_edge /
-- invalidate_edge). This is the SHADOW target for the FalkorDB->Postgres KG
-- migration: FalkorDB stays the source of truth until read-cutover, Postgres is
-- dual-written (best-effort) so it can be benchmarked on live traffic before the
-- switch.
--
-- Forward-compatible with per-tenant isolation via owner_id (one real owner today,
-- the #49 owner-axis / multi-user seam). owner_id is mandatory-with-a-default so a
-- write can never land unscoped (the soft-scope leak that bit `project`).
--
-- Bitemporal lifecycle mirrors the edge model exactly: t_created (write time),
-- t_valid (fact became true), t_invalid (fact stopped being true; NULL = live),
-- t_expired (reserved for selective forgetting). Legacy created_at/valid_at/
-- invalid_at mirrors are carried so a reader port is a column rename, not a remodel.
--
-- All statements are idempotent (CREATE ... IF NOT EXISTS). Reverse = DROP the two
-- tables; nothing else references them.

CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------------ entities
CREATE TABLE IF NOT EXISTS kg_entities (
    id              BIGSERIAL PRIMARY KEY,
    uuid            TEXT NOT NULL,
    owner_id        TEXT NOT NULL DEFAULT 'default',
    group_id        TEXT NOT NULL,
    project         TEXT,
    name            TEXT,
    normalized_name TEXT,
    entity_type     TEXT,
    summary         TEXT,
    embedding       vector(2048),
    -- live-edge degree; populated by the backfill, recomputed before cutover.
    -- (dual-write leaves this stale -- reads still come from FalkorDB until then.)
    degree          INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ,
    t_created       TIMESTAMPTZ,
    valid_at        TIMESTAMPTZ,
    t_valid         TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS kg_entities_uuid ON kg_entities (uuid);
-- dedup / alias lookup is owner-scoped (normalized_name is intentionally NOT unique:
-- the live graph still carries duplicate-name nodes -- that's the #49 fragmentation
-- this migration's resolution pass will collapse, not something the schema rejects).
CREATE INDEX IF NOT EXISTS kg_entities_norm  ON kg_entities (owner_id, group_id, normalized_name);
CREATE INDEX IF NOT EXISTS kg_entities_owner ON kg_entities (owner_id, group_id);
-- entity vector seed search (recall._search_kg seed leg). halfvec: 2048 > pgvector's
-- 2000-dim vector-HNSW limit, so the index is on the half-precision cast (loss-free
-- in practice; see schema/014). Full vector(2048) stays in the heap for exact rescore.
CREATE INDEX IF NOT EXISTS kg_entities_hnsw ON kg_entities
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE embedding IS NOT NULL;

-- ------------------------------------------------------------- relationships
-- One row per RELATES_TO edge. src/tgt are entity uuids (not FKs -- an edge can be
-- written before its endpoints in a concurrent pipeline; match by uuid at read time).
CREATE TABLE IF NOT EXISTS kg_relationships (
    id              BIGSERIAL PRIMARY KEY,
    uuid            TEXT NOT NULL,
    owner_id        TEXT NOT NULL DEFAULT 'default',
    group_id        TEXT NOT NULL,
    src_uuid        TEXT NOT NULL,
    tgt_uuid        TEXT NOT NULL,
    name            TEXT,                 -- relationship verb (r.name)
    fact            TEXT,
    fact_embedding  vector(2048),
    episodes        JSONB,                -- ordered list of source episode ids
    retrieval_count INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ,
    t_created       TIMESTAMPTZ,
    valid_at        TIMESTAMPTZ,
    t_valid         TIMESTAMPTZ,
    invalid_at      TIMESTAMPTZ,          -- legacy mirror of t_invalid
    t_invalid       TIMESTAMPTZ,          -- NULL = live edge (the lifecycle sentinel)
    t_expired       TIMESTAMPTZ           -- reserved: selective forgetting
);

CREATE UNIQUE INDEX IF NOT EXISTS kg_rel_uuid  ON kg_relationships (uuid);
CREATE INDEX IF NOT EXISTS kg_rel_src   ON kg_relationships (owner_id, group_id, src_uuid);
CREATE INDEX IF NOT EXISTS kg_rel_tgt   ON kg_relationships (owner_id, group_id, tgt_uuid);
-- per-pair lookup drives the contradiction detector (find_edges_by_pair) and is the
-- lock target for the write-path race fix (SELECT ... FOR UPDATE on the pair).
CREATE INDEX IF NOT EXISTS kg_rel_pair  ON kg_relationships (owner_id, group_id, src_uuid, tgt_uuid);
CREATE INDEX IF NOT EXISTS kg_rel_owner ON kg_relationships (owner_id, group_id);
-- live-fact vector search: ONLY currently-true facts are ANN-searchable, so the
-- HNSW index is partial on t_invalid IS NULL (avoids the HNSW-prefilter-then-filter
-- under-return on dead edges).
CREATE INDEX IF NOT EXISTS kg_rel_hnsw ON kg_relationships
    USING hnsw ((fact_embedding::halfvec(2048)) halfvec_cosine_ops)
    WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL;

-- BM25 over fact text (ParadeDB / pg_search), mirroring the FalkorDB fulltext leg.
CREATE INDEX IF NOT EXISTS kg_rel_bm25 ON kg_relationships
    USING bm25 (id, fact) WITH (key_field = 'id');
