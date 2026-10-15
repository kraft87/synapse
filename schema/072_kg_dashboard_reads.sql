-- schema-check: optional
-- 072_kg_dashboard_reads.sql
-- The dashboard reads the KG's own columns, t_created and entity_subtype, instead of
-- the FalkorDB-era mirrors (created_at, entity_type) that 017 and 020 kept for a cycle.
--
-- 1. Backfill. Copy the legacy value into t_created / entity_subtype where only the
--    legacy column was ever written, so no fact drops out of the feed and no entity
--    loses its type once readers stop looking at the mirrors. Only those rows change,
--    and a re-run changes nothing. Nothing in the previous image reads either column,
--    so the backfill is safe while it is still running.
-- 2. Index. The feed's facts leg pages kg_relationships newest-first by
--    (t_created, uuid). Without this index every feed page sorts the whole table.
--
-- On a POPULATED table (prod), don't run this file whole: its CREATE INDEX holds a write
-- lock on kg_relationships while it builds. Run the three UPDATEs below (they are safe on
-- the old image), then build the index by hand without the lock:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS kg_rel_t_created_idx
--     ON kg_relationships (t_created DESC, uuid DESC) WHERE t_created IS NOT NULL;
-- A failed CONCURRENTLY build leaves an INVALID index: drop it and build it again.
--
-- Optional: the code ships first and runs without this file, but apply it soon after
-- deploy. Until the UPDATEs have run, a legacy-only fact (t_created NULL) is absent from
-- the feed altogether and a legacy-only entity shows no type or date; until the index
-- exists, every feed page sorts the whole facts table.

UPDATE kg_relationships SET t_created = created_at
 WHERE t_created IS NULL AND created_at IS NOT NULL;

UPDATE kg_entities SET t_created = created_at
 WHERE t_created IS NULL AND created_at IS NOT NULL;

UPDATE kg_entities SET entity_subtype = entity_type
 WHERE entity_subtype IS NULL AND entity_type <> '';

CREATE INDEX IF NOT EXISTS kg_rel_t_created_idx
    ON kg_relationships (t_created DESC, uuid DESC) WHERE t_created IS NOT NULL;
