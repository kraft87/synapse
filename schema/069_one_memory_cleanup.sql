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
--   psql -X "$DSN" -v ON_ERROR_STOP=1 -f schema/069_one_memory_cleanup.sql
--
-- -X skips the psqlrc: one that sets AUTOCOMMIT off would run the whole file as one
-- transaction and roll it back, silently, when psql exits. Never -1 or --single-transaction
-- either. There is no outer transaction: each step below is one statement, so one
-- transaction, each is idempotent, and a step that fails leaves the earlier ones done. After
-- a failed step, run the whole file again, with no older schema file applied in between. A
-- step that finds its work done takes no lock.
--
-- Bound to its preflight. The file turns ON_ERROR_STOP on itself, and the preflight, once
-- every check has passed, sets synapse.cleanup_069 to 'ok' for the session; every later step
-- refuses without it. So a client that goes on after an error (the file pasted into a
-- prompt, psql without ON_ERROR_STOP) still changes nothing once the preflight has refused.
--
-- After 069, an image older than the one-memory release cannot write these tables: its
-- statements name group_id.
--
-- The preflight counts rows by group_id. On a table with a search index that count can
-- print a planner WARNING from the search extension; it is expected and changes nothing.
--
-- Locks. Each step that changes a table takes ACCESS EXCLUSIVE on it first, then builds its
-- index and drops the column under that one lock. Built first under SHARE, the index would
-- let a reader in that then queues a write behind it, while the drop waits for that reader:
-- a deadlock. Queued behind a long reader, the request would make every later reader of the
-- table wait behind it; after 5 s (lock_timeout) it fails instead, the step rolls back, and
-- the file can be run again.
-- schema-check: optional

\set ON_ERROR_STOP on
SET lock_timeout = '5s';
SET synapse.cleanup_069 = '';

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
    -- (a search index, say) must stop it. Views, triggers, constraints, statistics,
    -- policies and other columns' defaults or generated expressions over it are in pg_depend
    -- too; only the column's own default is not a dependent to keep.
    SELECT string_agg(DISTINCT pg_describe_object(d.classid, d.objid, d.objsubid), '; ') INTO bad
      FROM pg_depend d
      JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
     WHERE d.refclassid = 'pg_class'::regclass
       AND d.refobjid IN ('public.kg_entities'::regclass, 'public.kg_relationships'::regclass,
                          'public.notes'::regclass, 'public.preferences'::regclass)
       AND a.attname = 'group_id' AND NOT a.attisdropped
       AND NOT (d.classid = 'pg_attrdef'::regclass
                AND d.objid IN (SELECT ad.oid FROM pg_attrdef ad
                                 WHERE ad.adrelid = d.refobjid AND ad.adnum = d.refobjsubid))
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

    -- Every check passed: the steps below may run in this session.
    PERFORM set_config('synapse.cleanup_069', 'ok', false);
END
$cleanup$;

-- 1. kg_entities. The owner-only one-self index is built under another name first, so there
-- is never a moment without it; dropping the column drops kg_entities_norm,
-- kg_entities_owner and the old kg_entities_one_self.
DO $cleanup$
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.kg_entities'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        LOCK TABLE public.kg_entities IN ACCESS EXCLUSIVE MODE;
        CREATE UNIQUE INDEX kg_entities_one_self_069 ON public.kg_entities (owner_id) WHERE is_self;
        ALTER TABLE public.kg_entities DROP COLUMN group_id;
        ALTER INDEX public.kg_entities_one_self_069 RENAME TO kg_entities_one_self;
    END IF;
END
$cleanup$;

-- 2. kg_relationships: drops kg_rel_owner, kg_rel_pair, kg_rel_src and kg_rel_tgt.
DO $cleanup$
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.kg_relationships'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        LOCK TABLE public.kg_relationships IN ACCESS EXCLUSIVE MODE;
        ALTER TABLE public.kg_relationships DROP COLUMN group_id;
    END IF;
END
$cleanup$;

-- 3. notes: the live-set index, as in step 1.
DO $cleanup$
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.notes'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        LOCK TABLE public.notes IN ACCESS EXCLUSIVE MODE;
        CREATE INDEX notes_live_idx_069 ON public.notes (owner_id) WHERE superseded_by IS NULL;
        ALTER TABLE public.notes DROP COLUMN group_id;
        ALTER INDEX public.notes_live_idx_069 RENAME TO notes_live_idx;
    END IF;
END
$cleanup$;

-- 4. preferences: the same.
DO $cleanup$
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_attribute
                WHERE attrelid = 'public.preferences'::regclass
                  AND attname = 'group_id' AND NOT attisdropped) THEN
        LOCK TABLE public.preferences IN ACCESS EXCLUSIVE MODE;
        CREATE INDEX preferences_live_idx_069 ON public.preferences (owner_id) WHERE t_invalid IS NULL;
        ALTER TABLE public.preferences DROP COLUMN group_id;
        ALTER INDEX public.preferences_live_idx_069 RENAME TO preferences_live_idx;
    END IF;
END
$cleanup$;

-- 5. The 067 ledger. Its undo was released on 2026-11-12, and nothing reads it now.
DO $cleanup$
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;
    DROP TABLE IF EXISTS
        kg_pile_merge_067_runs, kg_pile_merge_067_entities, kg_pile_merge_067_relationships,
        kg_pile_merge_067_notes, kg_pile_merge_067_preferences, kg_pile_merge_067_entity_map,
        kg_pile_merge_067_exclude;
END
$cleanup$;

-- 6. Last, so a failure above leaves the stamp where it was, and only on the end state this
-- file builds: no group_id on the four tables, the one-self rule on owner_id, both live-set
-- indexes in place. A column dropped by hand, without its index built again, is not
-- stamped. Guarded: a re-run never moves the stamp down.
DO $cleanup$
DECLARE
    bad text;
BEGIN
    IF current_setting('synapse.cleanup_069', true) IS DISTINCT FROM 'ok' THEN
        RAISE EXCEPTION '069: the preflight did not pass in this session; nothing after it runs';
    END IF;

    SELECT string_agg(c.relname, ', ' ORDER BY c.relname) INTO bad
      FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
     WHERE a.attrelid IN ('public.kg_entities'::regclass, 'public.kg_relationships'::regclass,
                          'public.notes'::regclass, 'public.preferences'::regclass)
       AND a.attname = 'group_id' AND NOT a.attisdropped;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '069: % still carries group_id; run the file again', bad;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_index i
                    WHERE i.indexrelid = to_regclass('public.kg_entities_one_self')
                      AND i.indrelid = 'public.kg_entities'::regclass
                      AND i.indisunique AND i.indisvalid AND i.indnatts = 1
                      AND i.indkey[0] = (SELECT attnum FROM pg_attribute
                                          WHERE attrelid = 'public.kg_entities'::regclass
                                            AND attname = 'owner_id')
                      -- The predicate is read from its stored tree, a lone is_self: both
                      -- pg_get_expr and pg_get_indexdef lock the table, and a second run
                      -- waits for no lock.
                      AND i.indpred::text ~ ('^\(?\{VAR :varno 1 :varattno '
                            || (SELECT attnum FROM pg_attribute
                                 WHERE attrelid = 'public.kg_entities'::regclass
                                   AND attname = 'is_self')
                            || ' [^{}]*\}\)?$')) THEN
        RAISE EXCEPTION '069: kg_entities_one_self is not a valid UNIQUE (owner_id) WHERE is_self; build it so (CREATE UNIQUE INDEX kg_entities_one_self ON kg_entities (owner_id) WHERE is_self) and run the file again';
    END IF;

    SELECT string_agg(v.ix, ', ') INTO bad
      FROM (VALUES ('notes_live_idx', 'public.notes'),
                   ('preferences_live_idx', 'public.preferences')) AS v(ix, tbl)
     WHERE NOT EXISTS (SELECT 1 FROM pg_index i
                        WHERE i.indexrelid = to_regclass('public.' || v.ix)
                          AND i.indrelid = v.tbl::regclass AND i.indisvalid);
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '069: % is missing or INVALID; build it on owner_id and run the file again', bad;
    END IF;

    INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '069')
    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
     WHERE synapse_meta.value < EXCLUDED.value;
END
$cleanup$;
