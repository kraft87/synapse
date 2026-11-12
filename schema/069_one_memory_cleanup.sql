-- 069_one_memory_cleanup.sql
-- What one memory left behind. Since 067 folded the knowledge graph's second group into the
-- first, no service reads or writes group_id (tests/test_no_group_sql.py), and every row of
-- the four tables that carry it says 'technical'. This file removes:
--
--   * group_id on kg_entities, kg_relationships, notes and preferences, and with it the nine
--     indexes that lead with it: kg_entities_norm, kg_entities_owner, kg_entities_one_self,
--     kg_rel_owner, kg_rel_pair, kg_rel_src, kg_rel_tgt, notes_live_idx and
--     preferences_live_idx. 067's owner-only indexes (kg_entities_norm_any, kg_rel_pair_any,
--     kg_rel_tgt_any) serve the reads the KG ones served. kg_entities_one_self,
--     notes_live_idx and preferences_live_idx are built again on owner_id alone, before the
--     column goes, so the one-self rule and the live-set lookups never lapse;
--   * the 067 undo ledger, the seven kg_pile_merge_067_* tables.
--
-- recall_metrics.group_id stays: it is telemetry history, and old rows carry real values.
--
-- Optional: no service needs this file, and the released image runs the same with or
-- without it, so it is applied by hand to a running system, nothing stopped:
--
--   psql "$DSN" -v ON_ERROR_STOP=1 -f schema/069_one_memory_cleanup.sql
--
-- under autocommit (never -1 or --single-transaction). There is no outer transaction: each
-- step below is one statement, so one transaction, each is idempotent, and a step that fails
-- leaves the earlier ones done and the file safe to run again. A step that finds its work
-- done takes no lock.
--
-- lock_timeout: dropping a column takes ACCESS EXCLUSIVE on its table. Queued behind a long
-- reader, that request would make every later reader of the table wait behind it; after
-- 5 s it fails instead, the step rolls back, and the file can be run again.
-- schema-check: optional

SET lock_timeout = '5s';

-- 0. Preflight: reads and raises, writes nothing. Each refusal names what to look at.
DO $cleanup$
DECLARE
    stamp text;
    t     text;
    n     bigint;
    bad   text;
BEGIN
    SELECT value INTO stamp FROM synapse_meta WHERE key = 'schema_version';
    IF stamp IS NULL OR stamp < '067' THEN
        RAISE EXCEPTION '069 needs 067 applied first (schema_version is %)', COALESCE(stamp, 'not set');
    END IF;

    -- Every row is technical, or dropping the column would merge a group 067 never folded.
    FOREACH t IN ARRAY ARRAY['kg_entities', 'kg_relationships', 'notes', 'preferences'] LOOP
        IF EXISTS (SELECT 1 FROM pg_attribute
                    WHERE attrelid = ('public.' || t)::regclass
                      AND attname = 'group_id' AND NOT attisdropped) THEN
            EXECUTE format('SELECT count(*) FROM public.%I WHERE group_id IS DISTINCT FROM ''technical''', t)
               INTO n;
            IF n > 0 THEN
                RAISE EXCEPTION '069: % has % rows whose group_id is not technical; 067 folds the personal ones (apply it again), any other value is yours to settle', t, n;
            END IF;
        END IF;
    END LOOP;

    SELECT string_agg(ix, ', ') INTO bad
      FROM unnest(ARRAY['kg_entities_norm_any', 'kg_rel_pair_any', 'kg_rel_tgt_any']) AS ix
     WHERE NOT EXISTS (SELECT 1
                         FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                        WHERE c.relnamespace = 'public'::regnamespace
                          AND c.relname = ix AND i.indisvalid);
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '069: index % is missing or INVALID; it serves the reads of the indexes this file drops (apply 067 again)', bad;
    END IF;

    -- A dropped column silently takes every index on it along: one this file does not name
    -- (a search index, say) must stop it. Views, triggers, constraints, statistics and
    -- policies on the column are in pg_depend too; its default is not a dependent to keep.
    SELECT string_agg(DISTINCT pg_describe_object(d.classid, d.objid, d.objsubid), '; ') INTO bad
      FROM pg_depend d
      JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
     WHERE d.refclassid = 'pg_class'::regclass
       AND d.refobjid IN ('public.kg_entities'::regclass, 'public.kg_relationships'::regclass,
                          'public.notes'::regclass, 'public.preferences'::regclass)
       AND a.attname = 'group_id' AND NOT a.attisdropped
       AND d.classid <> 'pg_attrdef'::regclass
       AND NOT (d.classid = 'pg_class'::regclass
                AND d.objid IN (SELECT c.oid FROM pg_class c
                                 WHERE c.relnamespace = 'public'::regnamespace
                                   AND c.relname IN ('kg_entities_norm', 'kg_entities_owner',
                                                     'kg_entities_one_self', 'kg_rel_owner',
                                                     'kg_rel_pair', 'kg_rel_src', 'kg_rel_tgt',
                                                     'notes_live_idx', 'preferences_live_idx')));
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '069: % depends on group_id and is not one of the nine indexes this file drops; drop or rewrite it first', bad;
    END IF;

    -- A function body is not in pg_depend; one that names the column would fail at its next
    -- call. Extension functions are not ours to judge.
    SELECT string_agg(p.oid::regprocedure::text, ', ') INTO bad
      FROM pg_proc p
     WHERE p.pronamespace NOT IN ('pg_catalog'::regnamespace, 'information_schema'::regnamespace)
       AND p.prosrc ~* '\mgroup_id\M'
       AND NOT EXISTS (SELECT 1 FROM pg_depend e
                        WHERE e.classid = 'pg_proc'::regclass AND e.objid = p.oid
                          AND e.deptype = 'e');
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '069: function % names group_id; drop or rewrite it first', bad;
    END IF;
END
$cleanup$;

-- 1. kg_entities. The owner-only one-self index is built under another name first, so there
-- is never a moment without it; dropping the column drops kg_entities_norm,
-- kg_entities_owner and the old kg_entities_one_self.
DO $cleanup$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.kg_entities'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        CREATE UNIQUE INDEX kg_entities_one_self_069 ON public.kg_entities (owner_id) WHERE is_self;
        ALTER TABLE public.kg_entities DROP COLUMN group_id;
        ALTER INDEX public.kg_entities_one_self_069 RENAME TO kg_entities_one_self;
    END IF;
END
$cleanup$;

-- 2. kg_relationships: drops kg_rel_owner, kg_rel_pair, kg_rel_src and kg_rel_tgt.
DO $cleanup$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.kg_relationships'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        ALTER TABLE public.kg_relationships DROP COLUMN group_id;
    END IF;
END
$cleanup$;

-- 3. notes: the live-set index, as in step 1.
DO $cleanup$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.notes'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        CREATE INDEX notes_live_idx_069 ON public.notes (owner_id) WHERE superseded_by IS NULL;
        ALTER TABLE public.notes DROP COLUMN group_id;
        ALTER INDEX public.notes_live_idx_069 RENAME TO notes_live_idx;
    END IF;
END
$cleanup$;

-- 4. preferences: the same.
DO $cleanup$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.preferences'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        CREATE INDEX preferences_live_idx_069 ON public.preferences (owner_id) WHERE t_invalid IS NULL;
        ALTER TABLE public.preferences DROP COLUMN group_id;
        ALTER INDEX public.preferences_live_idx_069 RENAME TO preferences_live_idx;
    END IF;
END
$cleanup$;

-- 5. The 067 ledger. Its undo was released on 2026-11-12, and nothing reads it now.
DROP TABLE IF EXISTS
    kg_pile_merge_067_runs, kg_pile_merge_067_entities, kg_pile_merge_067_relationships,
    kg_pile_merge_067_notes, kg_pile_merge_067_preferences, kg_pile_merge_067_entity_map,
    kg_pile_merge_067_exclude;

-- 6. Last, so a failure above leaves the stamp where it was. Guarded: a re-run never moves
-- it down.
INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '069')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
 WHERE synapse_meta.value < EXCLUDED.value;
