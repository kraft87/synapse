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
--   kg_relationships.label_model    the model id that produced a version >= 1 label;
--                                   'error:<model id>' marks a 'personal' fallback written
--                                   for a fact the labeller could not get a verdict on.
--   kg_entities.is_self             the install owner's own entity: never a hop seed.
--                                   At most one per owner (069). Set only by
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
--
-- Apply with the poller paused. The personal-pile seed below is one UPDATE over every
-- fact in that group, live and retired (the 056 backfill measured about 12 ms a row,
-- because each row's update writes to every index), and it holds those rows' locks until
-- it finishes: nothing else writes the personal group but the poller, so pausing it also
-- closes the window between the cutover stamp and the first labelled row.

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

-- While two piles existed, facts written to the personal pile were personal on insert,
-- like the seeded rows below: the labeller never saw them. 067 merged the piles and drops
-- the trigger and this function.
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
-- The trigger, the seed and the one-self index below name group_id, which 069 drops: each
-- runs only while the column exists.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'kg_relationships'::regclass AND attname = 'group_id' AND NOT attisdropped) THEN
        CREATE TRIGGER kg_rel_personal_audience
            BEFORE INSERT ON kg_relationships
            FOR EACH ROW
            WHEN (NEW.group_id = 'personal' AND NEW.audience IS NULL)
            EXECUTE FUNCTION kg_relationships_personal_audience_trg();
    END IF;
END
$$;

-- Seed: every personal-pile fact, live and retired, is personal. The trigger is in
-- place first, so nothing inserted while this runs is missed.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'kg_relationships'::regclass AND attname = 'group_id' AND NOT attisdropped) THEN
        UPDATE kg_relationships
           SET audience = 'personal', label_version = 0
         WHERE group_id = 'personal' AND audience IS NULL;
    END IF;
END
$$;

ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS is_self BOOLEAN NOT NULL DEFAULT false;
-- One flagged entity per (owner_id, group_id) while two piles existed; 069 builds the
-- index again on owner_id alone.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'kg_entities'::regclass AND attname = 'group_id' AND NOT attisdropped) THEN
        CREATE UNIQUE INDEX IF NOT EXISTS kg_entities_one_self
            ON kg_entities (owner_id, group_id) WHERE is_self;
    END IF;
END
$$;

-- The seed touches too few rows to trigger autovacuum's analyze, so without this `audience`
-- has no statistics until the labeller has written many rows, and the planner guesses
-- `audience IS NULL` at its default selectivity.
ANALYZE kg_relationships (audience);
