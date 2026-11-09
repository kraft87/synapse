-- 066_kg_fact_audience.sql
-- A sensitivity label on every KG fact, so a restricted surface (a work laptop) is
-- served only facts labelled work-safe, plus a flag naming the install owner's own
-- entity so recall never walks the 1-hop leg out of it.
--
--   kg_relationships.audience       'personal' | 'work-safe' (the notes/web vocabulary,
--                                   053/063). NULL = not labelled yet. A restricted
--                                   caller is served 'work-safe' only, plus NULL rows
--                                   that existed before this file ran (the legacy
--                                   clause, below). 'personal' is never served there.
--   kg_relationships.label_version  which labelling rule set the value: 0 = a rule in
--                                   this file (the personal-pile seed and its insert
--                                   trigger); 1+ = ingestion/fact_audience.LABEL_VERSION,
--                                   the prompt that produced it.
--   kg_relationships.label_model    the model id that produced a version >= 1 label.
--   kg_entities.is_self             the install owner's own entity: never a hop seed.
--                                   At most one per (owner_id, group_id). Set only by
--                                   `synapse-admin owner set <uuid>`, never by name.
--
-- Legacy clause. synapse_meta 'fact_audience_legacy_max_id' records the highest
-- kg_relationships.id at the moment this file first ran. A NULL-audience fact at or
-- below it is served exactly as before 066 (its provenance check still applies); a
-- NULL fact above it is not served restricted until the labeller decides. ids come
-- from the sequence, so no writer can place a new row inside the legacy window. The
-- clause goes once every fact is labelled (a later migration, per deployment).
--
-- Deploy order: the image may ship before this file is applied. Until it is, the code
-- probes for the columns (ingestion/surfaces.FACT_AUDIENCE_PROBE): restricted callers
-- get today's provenance-only facts, no hop seed is excluded and the poller labels
-- nothing. The marker below lets the boot-time schema guard start a build on a database
-- behind only by this file:
-- schema-check: optional
-- Every statement is guarded, so a re-run is a no-op; the cutover is stamped once and
-- never moved. Runs under psql autocommit (apply_schema.sh), not -1.

ALTER TABLE kg_relationships
    ADD COLUMN IF NOT EXISTS audience      TEXT,
    ADD COLUMN IF NOT EXISTS label_version SMALLINT,
    ADD COLUMN IF NOT EXISTS label_model   TEXT;

-- NOT VALID, then VALIDATE: the ADD takes its exclusive lock only for the catalog
-- change; the scan runs under SHARE UPDATE EXCLUSIVE, which writers do not wait on.
DO $$
BEGIN
    ALTER TABLE kg_relationships ADD CONSTRAINT kg_relationships_audience_chk
        CHECK (audience IN ('personal', 'work-safe')) NOT VALID;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;
ALTER TABLE kg_relationships VALIDATE CONSTRAINT kg_relationships_audience_chk;

-- The cutover, stamped before anything else writes the column. ON CONFLICT DO NOTHING:
-- a re-run must never move it, or rows written in between would become "legacy".
INSERT INTO synapse_meta (key, value)
SELECT 'fact_audience_legacy_max_id', COALESCE(max(id), 0)::text FROM kg_relationships
ON CONFLICT (key) DO NOTHING;

-- Facts still written to the personal pile (until writes go one-group) are personal
-- on insert, like the seeded rows below: the labeller never sees them, and a restricted
-- caller that names group 'personal' is served nothing from that pile.
CREATE OR REPLACE FUNCTION kg_relationships_personal_audience_trg()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.audience := 'personal';
    NEW.label_version := 0;
    RETURN NEW;
END
$$;
DROP TRIGGER IF EXISTS kg_rel_personal_audience ON kg_relationships;
CREATE TRIGGER kg_rel_personal_audience
    BEFORE INSERT ON kg_relationships
    FOR EACH ROW
    WHEN (NEW.group_id = 'personal' AND NEW.audience IS NULL)
    EXECUTE FUNCTION kg_relationships_personal_audience_trg();

-- Seed: every personal-pile fact, live and retired, is personal. The trigger is in
-- place first, so nothing inserted while this runs is missed.
UPDATE kg_relationships
   SET audience = 'personal', label_version = 0
 WHERE group_id = 'personal' AND audience IS NULL;

ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS is_self BOOLEAN NOT NULL DEFAULT false;
-- One flagged entity per (owner_id, group_id): an install whose facts still sit in two
-- piles flags the owner's node in each.
CREATE UNIQUE INDEX IF NOT EXISTS kg_entities_one_self
    ON kg_entities (owner_id, group_id) WHERE is_self;
