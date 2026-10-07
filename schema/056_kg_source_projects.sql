-- 056_kg_source_projects.sql
-- Provenance-scoped KG facts for restricted surfaces.
--
-- Restricted surfaces (053/054: trust='restricted' + an allowed_projects allowlist) get
-- episodes filtered by `project = ANY(allowed_projects)`, but they got ZERO KG facts:
-- kg_relationships has no project column, so nothing proved a fact was confined to the
-- allowlist and serving none was the only fail-closed answer. This migration gives every
-- fact a cached, provable provenance set so a restricted caller can be served the facts
-- that came ONLY from projects it may see.
--
--   source_projects  the DISTINCT projects of the fact's source episodes, sorted. NULL
--                    means UNKNOWN, and an unknown set is never served restricted. A set
--                    is unknown when:
--                      * episodes is NULL, not a JSON array, or empty (web-artifact facts)
--                      * any element is not an episode id (JSON integer or digit string)
--                      * any referenced episode no longer exists
--                      * any referenced episode has a NULL project
--
-- The serving rule (mcp_server/kg_pg.py, recall_sources.py) is then a pure set test:
--   source_projects IS NOT NULL AND cardinality(source_projects) > 0
--   AND source_projects <@ allowed_projects
-- Mixed provenance (one allowed project, one not) fails the subset test and is excluded.
--
-- The column is maintained by triggers, not by the application, so every writer is
-- covered without code changes: the extractor's INSERT, reinforce_edges' episodes UNION,
-- the dashboard's episode-delete unlink, and anything written by hand. The value cannot
-- be forged either: an UPDATE that names source_projects is recomputed like one that
-- names episodes.
--
-- The cache also depends on the EPISODES table, so two companion triggers keep it honest
-- when a source episode changes underneath a fact:
--   * an episode's project is relabelled (the ingest upsert can COALESCE a new project
--     onto an existing turn) -> every fact citing it is recomputed;
--   * an episode is deleted or episodes is truncated -> every fact citing it goes NULL
--     (a missing source is unknown, by the rule above).
-- Both recompute through the same function, so there is one definition of the rule.
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the
-- column does not exist and the restricted KG paths serve nothing, exactly as before 056.
-- The marker below tells the boot-time schema guard (ingestion/schema_check.py) that a
-- database behind only by this file is fine, so the new image boots instead of exiting:
-- schema-check: optional
-- Every statement is guarded (IF NOT EXISTS / OR REPLACE / DROP ... IF EXISTS) and the
-- backfill only rewrites rows whose value differs, so a re-run is safe and cheap.

ALTER TABLE kg_relationships ADD COLUMN IF NOT EXISTS source_projects TEXT[];

-- The rule, in one place. STABLE: it reads episodes, so it must never back an index
-- expression (an index would freeze a value the companion triggers need to change).
CREATE OR REPLACE FUNCTION kg_source_projects(eps JSONB)
RETURNS TEXT[]
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    ids     BIGINT[];
    n_found INTEGER;
    n_null  INTEGER;
    projs   TEXT[];
BEGIN
    IF eps IS NULL OR jsonb_typeof(eps) <> 'array' OR jsonb_array_length(eps) = 0 THEN
        RETURN NULL;
    END IF;
    -- Every element must name an episode: a JSON integer or a string of digits (both
    -- shapes exist in the wild; the dashboard's delete route matches either). Anything
    -- else names nothing provable, so the whole set is unknown. {1,18} keeps the cast
    -- inside bigint.
    IF EXISTS (
        SELECT 1
          FROM jsonb_array_elements(eps) AS e(v)
         WHERE NOT (jsonb_typeof(e.v) IN ('number', 'string')
                    AND (e.v #>> '{}') ~ '^[0-9]{1,18}$')
    ) THEN
        RETURN NULL;
    END IF;
    SELECT array_agg(DISTINCT (e.v #>> '{}')::BIGINT)
      INTO ids
      FROM jsonb_array_elements(eps) AS e(v);
    SELECT count(*),
           count(*) FILTER (WHERE ep.project IS NULL),
           array_agg(DISTINCT ep.project ORDER BY ep.project)
               FILTER (WHERE ep.project IS NOT NULL)
      INTO n_found, n_null, projs
      FROM episodes ep
     WHERE ep.id = ANY (ids);
    IF n_found <> cardinality(ids) OR n_null > 0 THEN
        RETURN NULL;
    END IF;
    RETURN projs;
END
$$;

-- Writer-side trigger: every INSERT, and every UPDATE that names episodes or
-- source_projects, recomputes from the episodes table.
CREATE OR REPLACE FUNCTION kg_relationships_source_projects_trg()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.source_projects := kg_source_projects(NEW.episodes);
    RETURN NEW;
END
$$;

-- Episode relabel: recompute every fact that cites the episode. Setting source_projects
-- routes the row through the writer-side trigger above, so the rule stays in one place.
-- Both id shapes are matched, as in kg_source_projects.
CREATE OR REPLACE FUNCTION episodes_kg_source_projects_relabel_trg()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE kg_relationships
       SET source_projects = NULL
     WHERE episodes @> jsonb_build_array(NEW.id)
        OR episodes @> jsonb_build_array(NEW.id::TEXT);
    RETURN NULL;
END
$$;

-- Episode delete: statement-level with a transition table, so a bulk delete costs one
-- pass over the facts rather than one per deleted row. Only rows that are currently
-- servable can change (unknown stays unknown). The CASE guard keeps a malformed
-- (non-array) episodes value from failing the DELETE itself.
CREATE OR REPLACE FUNCTION episodes_kg_source_projects_delete_trg()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE kg_relationships r
       SET source_projects = NULL
     WHERE r.source_projects IS NOT NULL
       AND EXISTS (
           SELECT 1
             FROM jsonb_array_elements_text(
                      CASE WHEN jsonb_typeof(r.episodes) = 'array'
                           THEN r.episodes ELSE '[]'::JSONB END) AS e(v)
             JOIN deleted_episodes d ON e.v = d.id::TEXT
       );
    RETURN NULL;
END
$$;

-- Episode truncate: no rows survive, so no cached set can be proven.
CREATE OR REPLACE FUNCTION episodes_kg_source_projects_truncate_trg()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE kg_relationships SET source_projects = NULL WHERE source_projects IS NOT NULL;
    RETURN NULL;
END
$$;

-- DROP + CREATE (not CREATE OR REPLACE TRIGGER, which needs PG14+) inside one
-- transaction, so a re-run never leaves a window with a trigger missing.
BEGIN;

DROP TRIGGER IF EXISTS kg_rel_source_projects ON kg_relationships;
CREATE TRIGGER kg_rel_source_projects
    BEFORE INSERT OR UPDATE OF episodes, source_projects ON kg_relationships
    FOR EACH ROW EXECUTE FUNCTION kg_relationships_source_projects_trg();

-- WHEN keeps the ingest upsert (which names project on every re-ingest) free unless the
-- label actually changed.
DROP TRIGGER IF EXISTS episodes_kg_source_projects_relabel ON episodes;
CREATE TRIGGER episodes_kg_source_projects_relabel
    AFTER UPDATE OF project ON episodes
    FOR EACH ROW
    WHEN (OLD.project IS DISTINCT FROM NEW.project)
    EXECUTE FUNCTION episodes_kg_source_projects_relabel_trg();

DROP TRIGGER IF EXISTS episodes_kg_source_projects_delete ON episodes;
CREATE TRIGGER episodes_kg_source_projects_delete
    AFTER DELETE ON episodes
    REFERENCING OLD TABLE AS deleted_episodes
    FOR EACH STATEMENT EXECUTE FUNCTION episodes_kg_source_projects_delete_trg();

DROP TRIGGER IF EXISTS episodes_kg_source_projects_truncate ON episodes;
CREATE TRIGGER episodes_kg_source_projects_truncate
    AFTER TRUNCATE ON episodes
    FOR EACH STATEMENT EXECUTE FUNCTION episodes_kg_source_projects_truncate_trg();

COMMIT;

-- Backfill. Idempotent: only rows whose cached value differs are rewritten, so a re-run
-- is a read-only pass. On a populated database the first run rewrites every fact with a
-- provable set, and each rewrite adds entries to every index on the table (the HNSW one
-- included), so expect minutes, not seconds (measured: ~6 min per 30k rows on a 1 GB
-- test container). Safe against the live system: the triggers above are already in
-- place, so concurrent writers compute their own values, and the work COMMITs in id
-- batches so no writer waits on a row lock for longer than one batch. Committing inside
-- DO needs the file to run outside a transaction block, which is how apply_schema.sh
-- (psql -f, autocommit) runs it; do not apply this file with psql -1.
DO $$
DECLARE
    batch_ids BIGINT[];
    last_id   BIGINT := 0;
    n_rows    BIGINT := 0;
    n         BIGINT;
BEGIN
    LOOP
        SELECT array_agg(id ORDER BY id)
          INTO batch_ids
          FROM (SELECT id FROM kg_relationships WHERE id > last_id ORDER BY id LIMIT 2000) s;
        EXIT WHEN batch_ids IS NULL;
        UPDATE kg_relationships
           SET source_projects = kg_source_projects(episodes)
         WHERE id = ANY (batch_ids)
           AND source_projects IS DISTINCT FROM kg_source_projects(episodes);
        GET DIAGNOSTICS n = ROW_COUNT;
        n_rows := n_rows + n;
        last_id := batch_ids[array_upper(batch_ids, 1)];
        COMMIT;
        RAISE NOTICE 'kg source_projects backfill: through id %, % row(s) updated so far',
            last_id, n_rows;
    END LOOP;
END
$$;

-- Restricted serving filters with `source_projects <@ $allowlist`; GIN array_ops
-- answers <@. Partial on IS NOT NULL (every restricted predicate carries it) so the
-- unknown rows, which are never served restricted, stay out of the index. Built after
-- the backfill so the build is one pass rather than per-row maintenance.
CREATE INDEX IF NOT EXISTS kg_rel_source_projects_gin
    ON kg_relationships USING gin (source_projects)
    WHERE source_projects IS NOT NULL;

ANALYZE kg_relationships (source_projects);
