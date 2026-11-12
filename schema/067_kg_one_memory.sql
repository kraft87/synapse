-- 067_kg_one_memory.sql
-- One knowledge graph, not two. The 'personal' and 'technical' group_id piles are merged
-- into one: every personal entity, fact, note and preference becomes 'technical' (the
-- stored value stays; 069 later drops the column and its old indexes), and a
-- personal entity that has a same-name technical twin is folded into that twin.
--
-- REQUIRED. This file carries no optional marker, so ingestion/schema_check.py makes a
-- build that needs it refuse to start on a database that has not applied it. Apply it
-- first, then release; the release must carry the change that stops naming group_id (one
-- pile for reads and writes). A build that required 067 and still routed a project to
-- 'personal' would write a new pile whose facts miss the personal stamp the 066 trigger
-- gave them (the trigger is dropped below), and that pile would stay until a re-run of
-- this file folded it.
--
-- THE TWIN RULE (computed from the live rows on every run):
--   key  COALESCE(NULLIF(normalized_name, ''), lower(trim(name))), within one owner_id:
--        the key upsert_node compares a node's name by, and the one entity_uuid_by_
--        normalized_name looks a name up by.
--   (a)  a flagged personal entity (is_self) folds into the owner's flagged technical one,
--        whatever the names, and whether or not it is listed for exclusion (below): its
--        flag cannot be regrouped beside the technical one.
--   (b)  any other personal entity with a technical row of the same key folds into the
--        first of them by SURVIVOR_ORDER (ingestion/kg_pg_read.py): is_self DESC, id. Type,
--        supertype and embedding are ignored, as Stage 4's exact-name match ignores them.
--        An entity listed in public.kg_pile_merge_067_exclude (owner_id, uuid; created by
--        hand, never by this file, and dropped by 069) is never folded this
--        way: it is regrouped as it is, with its edges untouched, and then ranks among the
--        technical rows of its key by SURVIVOR_ORDER like any other.
--   (c)  every other personal entity is regrouped as it is (uuid, id and columns kept).
--   (d)  an entity of any group that carries the uuid of an earlier run's folded entity
--        (a writer's name index kept the deleted uuid and wrote it again) folds into that
--        run's survivor, ahead of (b): the uuid already stood for that survivor.
-- Within-pile duplicates are left alone. A folded entity's facts are re-pointed to the
-- survivor; the survivor keeps its uuid, id, name and type and takes the longest summary,
-- COALESCE(embedding, supertype, subtype) and the summed degree. No fact row is deleted
-- and no fact uuid changes; every personal fact becomes group 'technical' and keeps its
-- audience label (a NULL one is stamped 'personal', label_version 0, first: the rule the
-- 066 insert trigger applied).
--
-- TWO TRANSACTIONS, piped as this file ALONE:
--     psql -v ON_ERROR_STOP=1 < 067_kg_one_memory.sql
-- never wrapped in BEGIN ... COMMIT and never run with --single-transaction or -1 (the
-- transactions below are its own; scripts/apply_schema.sh runs it under autocommit, like
-- every file).
--   A  the merge. Takes SHARE ROW EXCLUSIVE on kg_entities and kg_relationships (readers
--      are never blocked; writers wait) and nothing stronger on any live table: no DROP
--      TRIGGER, no ALTER TABLE. It builds the three owner-only btrees the code without a
--      group predicate needs, under that same lock. Applied with the pollers stopped it
--      takes minutes. With nothing to merge (a fresh install, a re-run) it returns before
--      it takes a lock stronger than the ACCESS SHARE of its own reads, and writes no
--      ledger row; the indexes are then built only where one is missing, which a fresh
--      install does on its empty tables under SHARE, and a database that has them does not
--      do at all. A folded uuid that a writer wrote again after its survivor was deleted
--      is not something to merge: it counts as nothing. (The ghost query's other conditions, a
--      flagged entity, another owner, a survivor a later run folded too, a returned uuid
--      in a third group, are only tested under the lock; while a row they reject exists,
--      every re-run takes it, finds nothing to do and returns.) A re-run with only notes
--      or preferences left flips them under row locks and leaves the KG tables alone.
--   B  the tail. SET DEFAULT 'technical' on the four group_id columns (the code stops
--      naming it), DROP the 066 insert trigger and its function, and stamp 067. It checks
--      its two settings (below) and reads what is still to do from the catalog first, so a
--      re-run of a finished file takes no lock on the four tables. It reads the stamp row
--      FOR UPDATE, so an undo that is writing the stamp at that moment is waited for and
--      read, not stamped over. Until the stamp is 067 it takes SHARE ROW EXCLUSIVE on the
--      two KG tables while it checks the merge is complete (no personal row left, the three
--      indexes valid, 066 applied), and keeps it to the commit, through every sleep below:
--      writers wait for the whole of B, readers never wait for that lock. Then one LOCK
--      TABLE ... ACCESS EXCLUSIVE NOWAIT names every table it changes. NOWAIT never queues,
--      so no reader waits behind a pending request, and what the lock lets B do (the
--      defaults, the trigger, the stamp) is a few catalog changes before the commit. An
--      attempt is refused while another transaction holds any lock on one of those tables,
--      and B then sleeps a second and tries again, 60 times. The settings
--      synapse.pm067_lock_attempts (a whole number from 1 to 3600) and
--      synapse.pm067_lock_pause (seconds, from 0 to 60) change both; a value outside those
--      ranges, or not a number, stops B before it takes a lock. The 60th refusal fails the
--      run ("could not obtain lock"). A database that never has a moment without a reader
--      does that; so does a writer that locks a row of one of those tables (SELECT ... FOR
--      UPDATE, as the dashboard's fact and episode deletes do) before B's ACCESS EXCLUSIVE
--      is granted, B's sleeps included, and then waits for a write lock B's SHARE ROW
--      EXCLUSIVE withholds: that writer cannot finish while B holds its lock, so every
--      remaining attempt is refused until B gives up. Then B rolls back, the writer goes on,
--      the stamp stays at 066, a build that requires 067 still refuses to start, and
--      re-running the file finishes. A writer that holds nothing yet and only queues behind
--      B's SHARE ROW EXCLUSIVE does not refuse an attempt. B sets statement_timeout to 0,
--      because that loop is one long statement; A does too. After the loop B checks the
--      catalog that the defaults are set and the trigger is gone, and fails if not.
--      The stamp is the last statement: if B fails, the merged data stays, the stamp stays
--      at 066, and a build that requires 067 still refuses to start. Re-run the file.
--
-- OPERATOR NOTES. Apply with the pollers stopped, which are the writers that add graph rows
-- for long, and with no dashboard fact or episode deletes while the migration runs. A
-- delete there takes its rows FOR UPDATE, which the SHARE ROW EXCLUSIVE locks above let
-- through, and then waits for those locks at its UPDATE or DELETE. If A then re-points or
-- stamps one of the same rows, A waits for the delete's row lock and the two wait for each
-- other; B's attempts are refused for as long as the delete is open (see above). Two
-- symptoms, both safe: "deadlock detected" in A (the server aborts one of the two, usually
-- the migration), and "could not obtain lock" in B after its last attempt. Neither one
-- stamps anything, so the stamp stays at 066 and a build that requires 067 still refuses
-- to start: re-run the file. The other writer that lives while the pollers are stopped is
-- the recall fact-count bump (one UPDATE of kg_relationships); it holds nothing before it
-- waits, so it queues behind the locks above and refuses no attempt. A poller that wrote
-- personal rows in between, or wrote a folded uuid again, is folded by a re-run, under a
-- new run; restart the pollers after it, because a poller keeps a deleted uuid in its name
-- index for up to 900 s and would write it again.
--
-- THE LEDGER. Every run is written to kg_pile_merge_067_* in the same transaction: the
-- before image of each row it changes (full rows), the twin map, and counts. The script
-- that previewed, checked and undid a run went with schema 069, which drops these tables:
-- the merge is this file alone, and it has no undo.
-- Restores that write group_id, src_uuid or tgt_uuid from an older backup table must not
-- run after this file: `repair_numbered_entity_magnets.py --restore` refuses a backup that
-- carries group_id once 067 is applied, and a restore from kg_fact_tzfix_20261011 must not
-- be run.
--
-- Needs 066 (audience, label_version, is_self). The tail checks the merged graph before it
-- stamps 067.

BEGIN;
-- txn-a:begin
SET LOCAL lock_timeout = '15s';
-- The merge is one long statement; a statement_timeout set on the role must not cut it off.
SET LOCAL statement_timeout = 0;

-- Refuse early, before any lock or write, on what would only fail later or never be noticed:
-- no 066 columns, or an owner-only index left INVALID by a cancelled CREATE INDEX
-- CONCURRENTLY (CREATE INDEX IF NOT EXISTS would skip it and the planner would ignore it).
DO $pm$
DECLARE
    bad text;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'kg_entities'
                      AND column_name = 'is_self') THEN
        RAISE EXCEPTION '067 needs 066 applied first (kg_entities.is_self is missing)';
    END IF;
    SELECT string_agg(c.relname, ', ') INTO bad
      FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
     WHERE c.relnamespace = 'public'::regnamespace
       AND c.relname IN ('kg_rel_pair_any', 'kg_rel_tgt_any', 'kg_entities_norm_any')
       AND NOT i.indisvalid;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '067: index % is INVALID; DROP INDEX CONCURRENTLY it and re-run', bad;
    END IF;
END
$pm$;

DO $pm$
DECLARE
    v_run     bigint;
    v_counts  jsonb;
    v_pair    record;
    v_col     record;
    v_cols    text;
    v_n       bigint;
    v_notes   bigint := 0;
    v_prefs   bigint := 0;
    v_ledger  boolean;
    v_kg      boolean;
    v_np      boolean;
BEGIN
    -- Past 069 the four tables have no group_id: there is no pile left to merge.
    IF (SELECT count(*) FROM pg_attribute
         WHERE attrelid IN ('public.kg_entities'::regclass, 'public.kg_relationships'::regclass,
                            'public.notes'::regclass, 'public.preferences'::regclass)
           AND attname = 'group_id' AND NOT attisdropped) < 4 THEN
        RETURN;
    END IF;

    -- Folded entities of the runs not undone, and the survivor each stood for. A writer
    -- whose name index still held a deleted uuid writes it again; those rows are folded by
    -- rule (d). Filled below, under the lock.
    CREATE TEMP TABLE _pm_ghost (owner_id text, uuid text, survivor_uuid text) ON COMMIT DROP;
    v_ledger := to_regclass('public.kg_pile_merge_067_entity_map') IS NOT NULL
            AND to_regclass('public.kg_pile_merge_067_runs') IS NOT NULL;

    -- Nothing to merge anywhere: a fresh install, CI, a re-run on a merged database. No
    -- lock beyond these reads, no ledger table, no run row. A folded uuid written again is
    -- work only while its survivor is in technical; with the survivor gone there is nothing
    -- to fold it into (the ghost query below needs the same). This asks for less than the
    -- ghost query does (not the flag, the owner, the group of the returned uuid or the chain
    -- of runs), so it can say yes to a row the ghost query then rejects; the check under the
    -- lock settles that.
    v_kg := EXISTS (SELECT 1 FROM kg_entities WHERE group_id = 'personal')
         OR EXISTS (SELECT 1 FROM kg_relationships WHERE group_id = 'personal');
    IF NOT v_kg AND v_ledger THEN
        v_kg := EXISTS (SELECT 1
                          FROM kg_pile_merge_067_entity_map m
                          JOIN kg_pile_merge_067_runs r ON r.run_id = m.run_id AND r.undone_at IS NULL
                          JOIN kg_entities e ON e.uuid = m.loser_uuid
                          JOIN kg_entities s ON s.uuid = m.survivor_uuid AND s.group_id = 'technical');
    END IF;
    v_np := EXISTS (SELECT 1 FROM notes WHERE group_id = 'personal')
         OR EXISTS (SELECT 1 FROM preferences WHERE group_id = 'personal');
    IF NOT v_kg AND NOT v_np THEN
        RETURN;
    END IF;

    IF v_kg THEN
        -- Blocks the KG writers (the poller, the orphan sweep), never a reader.
        LOCK TABLE kg_entities, kg_relationships IN SHARE ROW EXCLUSIVE MODE;
        IF v_ledger THEN
            INSERT INTO _pm_ghost
            SELECT DISTINCT ON (m.loser_uuid) e.owner_id, e.uuid, m.survivor_uuid
              FROM kg_pile_merge_067_entity_map m
              JOIN kg_pile_merge_067_runs r ON r.run_id = m.run_id AND r.undone_at IS NULL
              JOIN kg_entities e ON e.uuid = m.loser_uuid AND e.owner_id = m.owner_id
                                AND e.group_id IN ('personal', 'technical') AND NOT e.is_self
              JOIN kg_entities s ON s.uuid = m.survivor_uuid AND s.owner_id = m.owner_id
                                AND s.group_id = 'technical'
             WHERE NOT EXISTS (SELECT 1
                                 FROM kg_pile_merge_067_entity_map x
                                 JOIN kg_pile_merge_067_runs xr
                                   ON xr.run_id = x.run_id AND xr.undone_at IS NULL
                                WHERE x.loser_uuid = m.survivor_uuid)
             ORDER BY m.loser_uuid, m.run_id DESC
            ;
        END IF;
        -- A concurrent apply may have merged everything while this one waited for the lock.
        v_kg := EXISTS (SELECT 1 FROM kg_entities WHERE group_id = 'personal')
             OR EXISTS (SELECT 1 FROM kg_relationships WHERE group_id = 'personal')
             OR EXISTS (SELECT 1 FROM _pm_ghost);
        v_np := EXISTS (SELECT 1 FROM notes WHERE group_id = 'personal')
             OR EXISTS (SELECT 1 FROM preferences WHERE group_id = 'personal');
        IF NOT v_kg AND NOT v_np THEN
            RETURN;
        END IF;
    END IF;

    -- ---- the ledger: fixed names, one run_id per run, full rows (LIKE: no dimension literal)
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_runs (
        run_id                bigserial PRIMARY KEY,
        started_at            timestamptz NOT NULL DEFAULT now(),
        schema_version_before text,
        counts                jsonb,
        undone_at             timestamptz
    );
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_entities (
        run_id bigint NOT NULL, backed_up_at timestamptz NOT NULL DEFAULT now(),
        image text NOT NULL CHECK (image IN ('before', 'after')), LIKE kg_entities);
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_relationships (
        run_id bigint NOT NULL, backed_up_at timestamptz NOT NULL DEFAULT now(),
        image text NOT NULL CHECK (image IN ('before', 'after')), LIKE kg_relationships);
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_notes (
        run_id bigint NOT NULL, backed_up_at timestamptz NOT NULL DEFAULT now(),
        image text NOT NULL CHECK (image IN ('before', 'after')), LIKE notes);
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_preferences (
        run_id bigint NOT NULL, backed_up_at timestamptz NOT NULL DEFAULT now(),
        image text NOT NULL CHECK (image IN ('before', 'after')), LIKE preferences);
    CREATE TABLE IF NOT EXISTS kg_pile_merge_067_entity_map (
        run_id bigint NOT NULL, owner_id text NOT NULL, loser_uuid text NOT NULL,
        survivor_uuid text NOT NULL, rule text NOT NULL, merge_key text,
        loser_flagged boolean NOT NULL, flag_moved boolean NOT NULL DEFAULT false,
        PRIMARY KEY (run_id, loser_uuid));
    CREATE INDEX IF NOT EXISTS kg_pile_merge_067_entities_uuid
        ON kg_pile_merge_067_entities (uuid, run_id);
    CREATE INDEX IF NOT EXISTS kg_pile_merge_067_relationships_uuid
        ON kg_pile_merge_067_relationships (uuid, run_id);

    -- A later migration may have added a column to a live table (or dropped one) since an
    -- earlier run created its ledger table: the inserts below name their columns, so line
    -- the ledger up first. A column the ledger gains takes the live column's default, so an
    -- older run's rows hold what a row of that time would have been given and an undo can
    -- put them back into a live column that is NOT NULL.
    FOR v_pair IN
        SELECT * FROM (VALUES
            ('kg_entities', 'kg_pile_merge_067_entities'),
            ('kg_relationships', 'kg_pile_merge_067_relationships'),
            ('notes', 'kg_pile_merge_067_notes'),
            ('preferences', 'kg_pile_merge_067_preferences')) AS t(live, ledger)
    LOOP
        FOR v_col IN
            SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS typ,
                   (SELECT pg_get_expr(d.adbin, d.adrelid)
                      FROM pg_attrdef d
                     WHERE d.adrelid = a.attrelid AND d.adnum = a.attnum) AS dflt
              FROM pg_attribute a
             WHERE a.attrelid = ('public.' || v_pair.live)::regclass
               AND a.attnum > 0 AND NOT a.attisdropped
               AND NOT EXISTS (SELECT 1 FROM pg_attribute b
                                WHERE b.attrelid = ('public.' || v_pair.ledger)::regclass
                                  AND b.attname = a.attname AND b.attnum > 0
                                  AND NOT b.attisdropped)
        LOOP
            EXECUTE format('ALTER TABLE public.%I ADD COLUMN %I %s%s',
                           v_pair.ledger, v_col.attname, v_col.typ,
                           CASE WHEN v_col.dflt IS NULL THEN '' ELSE ' DEFAULT ' || v_col.dflt END);
        END LOOP;
        FOR v_col IN
            SELECT b.attname
              FROM pg_attribute b
             WHERE b.attrelid = ('public.' || v_pair.ledger)::regclass
               AND b.attnum > 0 AND NOT b.attisdropped AND b.attnotnull
               AND b.attname NOT IN ('run_id', 'backed_up_at', 'image')
               AND NOT EXISTS (SELECT 1 FROM pg_attribute a
                                WHERE a.attrelid = ('public.' || v_pair.live)::regclass
                                  AND a.attname = b.attname AND a.attnum > 0
                                  AND NOT a.attisdropped)
        LOOP
            EXECUTE format('ALTER TABLE public.%I ALTER COLUMN %I DROP NOT NULL',
                           v_pair.ledger, v_col.attname);
        END LOOP;
    END LOOP;

    INSERT INTO kg_pile_merge_067_runs (schema_version_before)
    SELECT value FROM synapse_meta WHERE key = 'schema_version'
    RETURNING run_id INTO v_run;
    IF v_run IS NULL THEN
        INSERT INTO kg_pile_merge_067_runs (schema_version_before) VALUES (NULL)
        RETURNING run_id INTO v_run;
    END IF;

    IF v_kg THEN
        -- ---- the entity map
        -- Owner-chosen twins that must stay apart (optional; this file never creates it).
        CREATE TEMP TABLE _pm_excl (owner_id text, uuid text) ON COMMIT DROP;
        IF to_regclass('public.kg_pile_merge_067_exclude') IS NOT NULL THEN
            EXECUTE 'INSERT INTO _pm_excl SELECT owner_id, uuid FROM public.kg_pile_merge_067_exclude';
        END IF;

        CREATE TEMP TABLE _pm_map ON COMMIT DROP AS
        WITH pk AS (
            SELECT e.id, e.uuid, e.owner_id, e.is_self, e.group_id,
                   COALESCE(NULLIF(e.normalized_name, ''), lower(trim(e.name))) AS k,
                   EXISTS (SELECT 1 FROM _pm_excl x
                            WHERE x.owner_id = e.owner_id AND x.uuid = e.uuid) AS excluded
              FROM kg_entities e
             WHERE e.group_id = 'personal'
                OR e.uuid IN (SELECT uuid FROM _pm_ghost)
        ), tk AS (
            SELECT e.id, e.uuid, e.owner_id, e.is_self,
                   COALESCE(NULLIF(e.normalized_name, ''), lower(trim(e.name))) AS k
              FROM kg_entities e
             WHERE e.group_id = 'technical'
               AND e.uuid NOT IN (SELECT uuid FROM _pm_ghost)
        ), tself AS (
            SELECT DISTINCT ON (owner_id) owner_id, uuid
              FROM tk
             WHERE is_self
             ORDER BY owner_id, id
        ), tname AS (
            SELECT DISTINCT ON (owner_id, k) owner_id, k, uuid
              FROM tk
             WHERE k <> ''
             ORDER BY owner_id, k, is_self DESC, id
        )
        SELECT p.owner_id,
               p.uuid AS loser_uuid,
               COALESCE(f.uuid, g.survivor_uuid, n.uuid) AS survivor_uuid,
               CASE WHEN f.uuid IS NOT NULL THEN 'owner-flag'
                    WHEN g.uuid IS NOT NULL THEN 'ghost'
                    ELSE 'name' END AS rule,
               p.k AS merge_key,
               p.is_self AS loser_flagged
          FROM pk p
          LEFT JOIN tself f ON p.is_self AND f.owner_id = p.owner_id
          LEFT JOIN _pm_ghost g ON g.uuid = p.uuid
          LEFT JOIN tname n ON p.group_id = 'personal' AND NOT p.excluded
                           AND n.owner_id = p.owner_id AND n.k = p.k
         WHERE f.uuid IS NOT NULL OR g.uuid IS NOT NULL OR n.uuid IS NOT NULL
        ;
        CREATE UNIQUE INDEX ON _pm_map (loser_uuid);
        ANALYZE _pm_map;

        -- Every fact the merge touches: personal ones, and any fact (whatever its group) on a
        -- folded entity, with the endpoints it will have afterwards.
        CREATE TEMP TABLE _pm_fx ON COMMIT DROP AS
        SELECT r.id, r.uuid, r.owner_id, r.group_id, r.src_uuid, r.tgt_uuid,
               COALESCE(ms.survivor_uuid, r.src_uuid) AS src2,
               COALESCE(mt.survivor_uuid, r.tgt_uuid) AS tgt2,
               r.t_invalid IS NULL AS live
          FROM kg_relationships r
          LEFT JOIN _pm_map ms ON ms.loser_uuid = r.src_uuid
          LEFT JOIN _pm_map mt ON mt.loser_uuid = r.tgt_uuid
         WHERE r.group_id = 'personal' OR ms.loser_uuid IS NOT NULL OR mt.loser_uuid IS NOT NULL
        ;
        CREATE UNIQUE INDEX ON _pm_fx (id);
        ANALYZE _pm_fx;

        -- ---- counts, taken before anything is written
        SELECT
        jsonb_build_object(
            'entities_total',        (SELECT count(*) FROM kg_entities),
            'entities_personal',     (SELECT count(*) FROM kg_entities WHERE group_id = 'personal'),
            'entities_technical',    (SELECT count(*) FROM kg_entities WHERE group_id = 'technical'),
            'facts_total',           (SELECT count(*) FROM kg_relationships),
            'facts_personal',        (SELECT count(*) FROM kg_relationships
                                       WHERE group_id = 'personal'),
            'facts_personal_live',   (SELECT count(*) FROM kg_relationships
                                       WHERE group_id = 'personal' AND t_invalid IS NULL),
            'facts_technical',       (SELECT count(*) FROM kg_relationships
                                       WHERE group_id = 'technical'),
            'facts_technical_live',  (SELECT count(*) FROM kg_relationships
                                       WHERE group_id = 'technical' AND t_invalid IS NULL),
            'notes_personal',        (SELECT count(*) FROM notes WHERE group_id = 'personal'),
            'preferences_personal',  (SELECT count(*) FROM preferences
                                       WHERE group_id = 'personal'),
            'flagged_personal',      (SELECT count(*) FROM kg_entities
                                       WHERE group_id = 'personal' AND is_self),
            'flagged_technical',     (SELECT count(*) FROM kg_entities
                                       WHERE group_id = 'technical' AND is_self),
            'losers',                (SELECT count(*) FROM _pm_map),
            'losers_owner_flag',     (SELECT count(*) FROM _pm_map WHERE rule = 'owner-flag'),
            'losers_by_name',        (SELECT count(*) FROM _pm_map WHERE rule = 'name'),
            'losers_ghost',          (SELECT count(*) FROM _pm_map WHERE rule = 'ghost'),
            'survivors',             (SELECT count(DISTINCT survivor_uuid) FROM _pm_map),
            'regrouped',             (SELECT count(*) FROM kg_entities
                                       WHERE group_id = 'personal'
                                         AND uuid NOT IN (SELECT loser_uuid FROM _pm_map)),
            'twin_shapes',           (SELECT COALESCE(jsonb_object_agg(shape, n), '{}'::jsonb)
                                        FROM (SELECT shape, count(*) AS n
                                                FROM (SELECT CASE
                                                        WHEN np = 1 AND nt = 1 THEN '1:1'
                                                        WHEN np = 1 AND nt > 1 THEN '1:n'
                                                        WHEN np > 1 AND nt = 1 THEN 'n:1'
                                                        WHEN np > 1 AND nt > 1 THEN 'n:n'
                                                        WHEN np > 1 AND nt = 0 THEN 'n:0'
                                                      END AS shape
                                                        FROM (SELECT count(*) FILTER (WHERE group_id = 'personal') AS np,
                                                                     count(*) FILTER (WHERE group_id = 'technical') AS nt
                                                                FROM (SELECT owner_id, group_id,
                                                                             COALESCE(NULLIF(normalized_name, ''),
                                                                                      lower(trim(name))) AS k
                                                                        FROM kg_entities
                                                                       WHERE group_id IN ('personal', 'technical')) ek
                                                               WHERE k <> ''
                                                               GROUP BY owner_id, k) kc
                                                       WHERE np > 0) sh
                                               WHERE shape IS NOT NULL
                                               GROUP BY shape) shn),
            'supertype_differ',      (SELECT count(*)
                                        FROM _pm_map m
                                        JOIN kg_entities l ON l.uuid = m.loser_uuid
                                        JOIN kg_entities s ON s.uuid = m.survivor_uuid
                                       WHERE l.entity_supertype IS NOT NULL
                                         AND s.entity_supertype IS NOT NULL
                                         AND l.entity_supertype <> s.entity_supertype),
            'supertype_differ_person', (SELECT count(*)
                                        FROM _pm_map m
                                        JOIN kg_entities l ON l.uuid = m.loser_uuid
                                        JOIN kg_entities s ON s.uuid = m.survivor_uuid
                                       WHERE l.entity_supertype IS NOT NULL
                                         AND s.entity_supertype IS NOT NULL
                                         AND l.entity_supertype <> s.entity_supertype
                                         AND 'Person' IN (l.entity_supertype, s.entity_supertype)),
            'supertype_pairs',       (SELECT COALESCE(jsonb_object_agg(pair, n), '{}'::jsonb)
                                        FROM (SELECT l.entity_supertype || ' | ' || s.entity_supertype AS pair,
                                                     count(*) AS n
                                                FROM _pm_map m
                                                JOIN kg_entities l ON l.uuid = m.loser_uuid
                                                JOIN kg_entities s ON s.uuid = m.survivor_uuid
                                               WHERE l.entity_supertype IS NOT NULL
                                                 AND s.entity_supertype IS NOT NULL
                                                 AND l.entity_supertype <> s.entity_supertype
                                               GROUP BY 1) sp),
            'facts_repointed',       (SELECT count(*) FROM _pm_fx
                                       WHERE src2 <> src_uuid OR tgt2 <> tgt_uuid),
            'facts_repointed_live',  (SELECT count(*) FROM _pm_fx
                                       WHERE live AND (src2 <> src_uuid OR tgt2 <> tgt_uuid)),
            'technical_facts_repointed', (SELECT count(*) FROM _pm_fx
                                       WHERE group_id <> 'personal'
                                         AND (src2 <> src_uuid OR tgt2 <> tgt_uuid)),
            'personal_facts_on_technical_endpoint',
                                     (SELECT count(*) FROM kg_relationships r
                                       WHERE r.group_id = 'personal'
                                         AND (EXISTS (SELECT 1 FROM kg_entities e
                                                       WHERE e.uuid = r.src_uuid AND e.group_id = 'technical')
                                              OR EXISTS (SELECT 1 FROM kg_entities e
                                                          WHERE e.uuid = r.tgt_uuid AND e.group_id = 'technical'))),
            'self_loops_created',    (SELECT count(*) FROM _pm_fx
                                       WHERE src2 = tgt2 AND src_uuid <> tgt_uuid),
            'self_loops_created_live', (SELECT count(*) FROM _pm_fx
                                       WHERE live AND src2 = tgt2 AND src_uuid <> tgt_uuid),
            'missing_endpoints_before', (SELECT count(*) FROM kg_relationships r
                                       WHERE NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = r.src_uuid)
                                          OR NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = r.tgt_uuid)),
            'missing_endpoints_after', (SELECT count(*) FROM kg_relationships r
                                       LEFT JOIN _pm_fx x ON x.id = r.id
                                       WHERE NOT EXISTS (SELECT 1 FROM kg_entities e
                                                          WHERE e.uuid = COALESCE(x.src2, r.src_uuid))
                                          OR NOT EXISTS (SELECT 1 FROM kg_entities e
                                                          WHERE e.uuid = COALESCE(x.tgt2, r.tgt_uuid))),
            'dangling_invalidated_by', (SELECT count(*) FROM kg_relationships r
                                       WHERE r.invalidated_by IS NOT NULL
                                         AND NOT EXISTS (SELECT 1 FROM kg_relationships o
                                                          WHERE o.uuid = r.invalidated_by)),
            'live_duplicate_pairs',  (SELECT count(*)
                                        FROM _pm_fx p
                                        JOIN kg_relationships pr ON pr.id = p.id
                                        JOIN kg_relationships t
                                          ON t.owner_id = p.owner_id AND t.group_id = 'technical'
                                         AND t.src_uuid = p.src2 AND t.tgt_uuid = p.tgt2
                                         AND t.t_invalid IS NULL
                                       WHERE p.live AND p.group_id = 'personal'
                                         AND lower(btrim(regexp_replace(COALESCE(pr.fact, ''), '\s+', ' ', 'g')))
                                           = lower(btrim(regexp_replace(COALESCE(t.fact, ''), '\s+', ' ', 'g')))),
            'label_disagreements',   (SELECT count(*)
                                        FROM _pm_fx p
                                        JOIN kg_relationships pr ON pr.id = p.id
                                        JOIN kg_relationships t
                                          ON t.owner_id = p.owner_id AND t.group_id = 'technical'
                                         AND t.src_uuid = p.src2 AND t.tgt_uuid = p.tgt2
                                         AND t.t_invalid IS NULL
                                       WHERE p.live AND p.group_id = 'personal'
                                         AND lower(btrim(regexp_replace(COALESCE(pr.fact, ''), '\s+', ' ', 'g')))
                                           = lower(btrim(regexp_replace(COALESCE(t.fact, ''), '\s+', ' ', 'g')))
                                         AND pr.audience IS DISTINCT FROM t.audience),
            'stragglers_to_stamp',   (SELECT count(*) FROM kg_relationships
                                       WHERE group_id = 'personal' AND audience IS NULL),
            'exclude_listed',        (SELECT count(*) FROM _pm_excl),
            'exclude_personal',      (SELECT count(*) FROM kg_entities e
                                        JOIN _pm_excl x ON x.owner_id = e.owner_id AND x.uuid = e.uuid
                                       WHERE e.group_id = 'personal'),
            'exclude_twins',         (SELECT count(*) FROM kg_entities e
                                        JOIN _pm_excl x ON x.owner_id = e.owner_id AND x.uuid = e.uuid
                                       WHERE e.group_id = 'personal'
                                         AND COALESCE(NULLIF(e.normalized_name, ''), lower(trim(e.name))) <> ''
                                         AND EXISTS (SELECT 1 FROM kg_entities t
                                                      WHERE t.owner_id = e.owner_id AND t.group_id = 'technical'
                                                        AND COALESCE(NULLIF(t.normalized_name, ''),
                                                                     lower(trim(t.name)))
                                                          = COALESCE(NULLIF(e.normalized_name, ''),
                                                                     lower(trim(e.name))))),
            'exclude_owner_flag',    (SELECT count(*) FROM _pm_map m
                                       WHERE m.rule = 'owner-flag'
                                         AND EXISTS (SELECT 1 FROM _pm_excl x
                                                      WHERE x.owner_id = m.owner_id
                                                        AND x.uuid = m.loser_uuid)),
            'exclude_outranks_twin', (SELECT count(*) FROM kg_entities e
                                        JOIN _pm_excl x ON x.owner_id = e.owner_id AND x.uuid = e.uuid
                                       WHERE e.group_id = 'personal'
                                         AND COALESCE(NULLIF(e.normalized_name, ''), lower(trim(e.name))) <> ''
                                         AND EXISTS (SELECT 1 FROM kg_entities t
                                                      WHERE t.owner_id = e.owner_id AND t.group_id = 'technical'
                                                        AND COALESCE(NULLIF(t.normalized_name, ''),
                                                                     lower(trim(t.name)))
                                                          = COALESCE(NULLIF(e.normalized_name, ''),
                                                                     lower(trim(e.name))))
                                         AND NOT EXISTS (SELECT 1 FROM kg_entities t
                                                          WHERE t.owner_id = e.owner_id AND t.group_id = 'technical'
                                                            AND COALESCE(NULLIF(t.normalized_name, ''),
                                                                         lower(trim(t.name)))
                                                              = COALESCE(NULLIF(e.normalized_name, ''),
                                                                         lower(trim(e.name)))
                                                            AND (t.is_self AND NOT e.is_self
                                                                 OR t.is_self = e.is_self AND t.id < e.id)))
        )
        INTO v_counts;
        UPDATE kg_pile_merge_067_runs SET counts = v_counts WHERE run_id = v_run;

        INSERT INTO kg_pile_merge_067_entity_map
            (run_id, owner_id, loser_uuid, survivor_uuid, rule, merge_key, loser_flagged)
        SELECT v_run, owner_id, loser_uuid, survivor_uuid, rule, merge_key, loser_flagged
          FROM _pm_map;

        -- ---- before images: every row this run will change
        SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_cols
          FROM pg_attribute
         WHERE attrelid = 'public.kg_entities'::regclass AND attnum > 0 AND NOT attisdropped;
        EXECUTE format(
            'INSERT INTO kg_pile_merge_067_entities (run_id, image, %1$s) '
            'SELECT $1, ''before'', %1$s FROM kg_entities '
            'WHERE group_id = ''personal'' OR uuid IN (SELECT survivor_uuid FROM _pm_map) '
            '   OR uuid IN (SELECT loser_uuid FROM _pm_map)', v_cols)
            USING v_run;

        SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_cols
          FROM pg_attribute
         WHERE attrelid = 'public.kg_relationships'::regclass AND attnum > 0 AND NOT attisdropped;
        EXECUTE format(
            'INSERT INTO kg_pile_merge_067_relationships (run_id, image, %1$s) '
            'SELECT $1, ''before'', %1$s FROM kg_relationships WHERE id IN (SELECT id FROM _pm_fx)',
            v_cols)
            USING v_run;

        -- ---- facts: the straggler stamp (the rule the 066 trigger applied to a personal insert),
        -- then one UPDATE that re-points the endpoints and regroups. Nothing else on a fact moves.
        UPDATE kg_relationships
           SET audience = 'personal', label_version = 0
         WHERE group_id = 'personal' AND audience IS NULL;

        UPDATE kg_relationships r
           SET src_uuid = x.src2,
               tgt_uuid = x.tgt2,
               group_id = CASE WHEN r.group_id = 'personal' THEN 'technical' ELSE r.group_id END
          FROM _pm_fx x
         WHERE r.id = x.id;

        -- ---- survivors: the longest summary, the first non-null embedding / supertype /
        -- subtype among the folded rows (lowest id), and the summed degree
        UPDATE kg_entities s
           SET summary = CASE WHEN length(btrim(COALESCE(g.lsummary, ''))) > length(btrim(COALESCE(s.summary, '')))
                              THEN btrim(g.lsummary) ELSE s.summary END,
               embedding = COALESCE(s.embedding, g.lembedding),
               entity_supertype = COALESCE(s.entity_supertype, g.lsupertype),
               entity_subtype = COALESCE(s.entity_subtype, g.lsubtype),
               degree = s.degree + g.ldegree
          FROM (SELECT m.survivor_uuid,
                       (array_agg(l.summary ORDER BY length(btrim(l.summary)) DESC, l.id)
                           FILTER (WHERE btrim(COALESCE(l.summary, '')) <> ''))[1] AS lsummary,
                       (array_agg(l.embedding ORDER BY l.id)
                           FILTER (WHERE l.embedding IS NOT NULL))[1] AS lembedding,
                       (array_agg(l.entity_supertype ORDER BY l.id)
                           FILTER (WHERE l.entity_supertype IS NOT NULL))[1] AS lsupertype,
                       (array_agg(l.entity_subtype ORDER BY l.id)
                           FILTER (WHERE l.entity_subtype IS NOT NULL))[1] AS lsubtype,
                       sum(l.degree) AS ldegree
                  FROM _pm_map m
                  JOIN kg_entities l ON l.uuid = m.loser_uuid
                 GROUP BY m.survivor_uuid) g
         WHERE s.uuid = g.survivor_uuid;

        -- Deleted before any flag moves: the unique one-flag-per-(owner, group) index never sees
        -- two flagged rows in one group.
        DELETE FROM kg_entities WHERE uuid IN (SELECT loser_uuid FROM _pm_map);

        -- The owner's flag follows a flagged personal row that folded by name, when the owner
        -- has no technical flag yet.
        WITH moved AS (
            UPDATE kg_entities e SET is_self = true
              FROM _pm_map m
             WHERE m.rule = 'name' AND m.loser_flagged AND e.uuid = m.survivor_uuid
               AND NOT e.is_self
               AND NOT EXISTS (SELECT 1 FROM kg_entities o
                                WHERE o.owner_id = e.owner_id AND o.group_id = 'technical' AND o.is_self)
            RETURNING m.loser_uuid
        )
        UPDATE kg_pile_merge_067_entity_map
           SET flag_moved = true
         WHERE run_id = v_run AND loser_uuid IN (SELECT loser_uuid FROM moved);

        UPDATE kg_entities SET group_id = 'technical' WHERE group_id = 'personal';

        SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_cols
          FROM pg_attribute
         WHERE attrelid = 'public.kg_entities'::regclass AND attnum > 0 AND NOT attisdropped;
        EXECUTE format(
            'INSERT INTO kg_pile_merge_067_entities (run_id, image, %1$s) '
            'SELECT $1, ''after'', %1$s FROM kg_entities '
            'WHERE uuid IN (SELECT survivor_uuid FROM _pm_map)', v_cols)
            USING v_run;

        -- ---- self-checks: a failure rolls the whole transaction back
        IF EXISTS (SELECT 1 FROM kg_entities WHERE group_id = 'personal')
           OR EXISTS (SELECT 1 FROM kg_relationships WHERE group_id = 'personal') THEN
            RAISE EXCEPTION '067: personal rows are left after the merge';
        END IF;
        IF EXISTS (SELECT 1 FROM kg_entities WHERE uuid IN (SELECT loser_uuid FROM _pm_map))
           OR EXISTS (SELECT 1 FROM kg_relationships r
                       WHERE r.src_uuid IN (SELECT loser_uuid FROM _pm_map)
                          OR r.tgt_uuid IN (SELECT loser_uuid FROM _pm_map)) THEN
            RAISE EXCEPTION '067: a folded entity still exists or still has an edge';
        END IF;
        IF EXISTS (SELECT 1 FROM kg_entities WHERE is_self GROUP BY owner_id HAVING count(*) > 1) THEN
            RAISE EXCEPTION '067: an owner has more than one flagged entity after the merge';
        END IF;

    ELSE
        v_counts := jsonb_build_object(
            'notes_personal',       (SELECT count(*) FROM notes WHERE group_id = 'personal'),
            'preferences_personal', (SELECT count(*) FROM preferences WHERE group_id = 'personal'));
        UPDATE kg_pile_merge_067_runs SET counts = v_counts WHERE run_id = v_run;
    END IF;

    -- ---- notes and preferences: back up and flip in one statement each, so the set flipped
    -- is exactly the set backed up; only row locks are taken
    SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_cols
      FROM pg_attribute
     WHERE attrelid = 'public.notes'::regclass AND attnum > 0 AND NOT attisdropped;
    EXECUTE format(
        'WITH b AS (INSERT INTO kg_pile_merge_067_notes (run_id, image, %1$s) '
        '           SELECT $1, ''before'', %1$s FROM notes WHERE group_id = ''personal'' '
        '           RETURNING id) '
        'UPDATE notes SET group_id = ''technical'' WHERE id IN (SELECT id FROM b)', v_cols)
        USING v_run;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    v_notes := v_n;

    SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_cols
      FROM pg_attribute
     WHERE attrelid = 'public.preferences'::regclass AND attnum > 0 AND NOT attisdropped;
    EXECUTE format(
        'WITH b AS (INSERT INTO kg_pile_merge_067_preferences (run_id, image, %1$s) '
        '           SELECT $1, ''before'', %1$s FROM preferences WHERE group_id = ''personal'' '
        '           RETURNING id) '
        'UPDATE preferences SET group_id = ''technical'' WHERE id IN (SELECT id FROM b)', v_cols)
        USING v_run;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    v_prefs := v_n;

    UPDATE kg_pile_merge_067_runs
       SET counts = v_counts || jsonb_build_object('applied', jsonb_build_object(
               'notes_flipped', v_notes, 'preferences_flipped', v_prefs))
     WHERE run_id = v_run;
    IF v_kg THEN
        RAISE NOTICE '067 run %: % entities folded into % survivors, % regrouped; % facts moved, % notes and % preferences flipped',
            v_run, v_counts->>'losers', v_counts->>'survivors', v_counts->>'regrouped',
            v_counts->>'facts_personal', v_notes, v_prefs;
    ELSE
        RAISE NOTICE '067 run %: % notes and % preferences flipped (the knowledge graph had nothing to merge)',
            v_run, v_notes, v_prefs;
    END IF;
END
$pm$;

-- The owner-only btrees the code without a group predicate reads through (every existing
-- one leads with (owner_id, group_id)). Plain CREATE INDEX, built here under the lock the
-- merge holds (a SHARE lock never conflicts with the SHARE ROW EXCLUSIVE it already has)
-- so none is ever left INVALID. Guarded by the catalog: CREATE INDEX IF NOT EXISTS skips an
-- existing index only AFTER it has taken its SHARE lock, which would block the pollers'
-- writes on a re-run for nothing. Fresh installs build them on empty tables.
DO $pm$
BEGIN
    IF to_regclass('public.kg_rel_pair_any') IS NULL THEN
        CREATE INDEX IF NOT EXISTS kg_rel_pair_any
            ON kg_relationships (owner_id, src_uuid, tgt_uuid);
    END IF;
    IF to_regclass('public.kg_rel_tgt_any') IS NULL THEN
        CREATE INDEX IF NOT EXISTS kg_rel_tgt_any
            ON kg_relationships (owner_id, tgt_uuid);
    END IF;
    IF to_regclass('public.kg_entities_norm_any') IS NULL THEN
        CREATE INDEX IF NOT EXISTS kg_entities_norm_any
            ON kg_entities (owner_id, normalized_name);
    END IF;
END
$pm$;
-- txn-a:end
COMMIT;

BEGIN;
-- txn-b:begin
SET LOCAL lock_timeout = '15s';
-- The retry loop below is one statement that sleeps by design; a statement_timeout set on
-- the role must not cut it off.
SET LOCAL statement_timeout = 0;

DO $pm$
DECLARE
    attempt   int;
    raw_tries text   := COALESCE(NULLIF(current_setting('synapse.pm067_lock_attempts', true), ''), '60');
    raw_pause text   := COALESCE(NULLIF(current_setting('synapse.pm067_lock_pause', true), ''), '1');
    tries     int;
    pause_s   float8;
    t         text;
    need      text[] := '{}';
    trg       boolean;
    stamped   text;
    bad       text;
BEGIN
    -- Past 069 there is no group_id to give a default, and 066 builds no trigger to drop.
    IF (SELECT count(*) FROM pg_attribute
         WHERE attrelid IN ('public.kg_entities'::regclass, 'public.kg_relationships'::regclass,
                            'public.notes'::regclass, 'public.preferences'::regclass)
           AND attname = 'group_id' AND NOT attisdropped) < 4 THEN
        RETURN;
    END IF;

    -- The two retry settings, before any lock. A count of 0 or less would skip the loop
    -- without an error, and a huge count or an endless pause would keep SHARE ROW EXCLUSIVE
    -- for as long as the loop runs (statement_timeout is 0 here); neither may reach it.
    -- NaN and Infinity fall outside BETWEEN; text that is no number fails its cast.
    BEGIN
        tries := raw_tries::int;
    EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
        tries := NULL;
    END;
    IF tries IS NULL OR tries NOT BETWEEN 1 AND 3600 THEN
        RAISE EXCEPTION '067: synapse.pm067_lock_attempts is "%", outside the allowed range: a whole number from 1 to 3600', raw_tries;
    END IF;
    BEGIN
        pause_s := raw_pause::float8;
    EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
        pause_s := NULL;
    END;
    IF pause_s IS NULL OR pause_s NOT BETWEEN 0 AND 60 THEN
        RAISE EXCEPTION '067: synapse.pm067_lock_pause is "%", outside the allowed range: seconds from 0 to 60', raw_pause;
    END IF;

    -- What is still to change, from the catalog alone: no lock on the four tables it
    -- changes is taken to find out, so a re-run of a finished file takes none.
    FOREACH t IN ARRAY ARRAY['kg_entities', 'kg_relationships', 'notes', 'preferences'] LOOP
        IF NOT EXISTS (
            SELECT 1
              FROM pg_attrdef d
              JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
             WHERE d.adrelid = ('public.' || t)::regclass
               AND a.attname = 'group_id'
               AND pg_get_expr(d.adbin, d.adrelid) = '''technical''::text') THEN
            need := need || t;
        END IF;
    END LOOP;
    trg := EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgrelid = 'public.kg_relationships'::regclass
                      AND tgname = 'kg_rel_personal_audience' AND NOT tgisinternal);
    -- FOR UPDATE: an undo writes the stamp last and commits after it. B must not read a
    -- stamp that is about to go down and write 067 over it once that commits.
    SELECT value INTO stamped FROM synapse_meta WHERE key = 'schema_version' FOR UPDATE;

    -- Stamp 067 only on a merged graph and a finished merge transaction. A client that
    -- ignored ON_ERROR_STOP would otherwise run this transaction after a failed one and
    -- stamp a database that still has a pile or lacks the indexes. The scans below read
    -- both KG tables in full, so they run under SHARE ROW EXCLUSIVE, which keeps writers
    -- out until the commit (the retry sleeps below included) and never makes a reader wait.
    IF stamped IS NULL OR stamped < '067' THEN
        LOCK TABLE kg_entities, kg_relationships IN SHARE ROW EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM kg_entities WHERE group_id = 'personal')
           OR EXISTS (SELECT 1 FROM kg_relationships WHERE group_id = 'personal') THEN
            RAISE EXCEPTION '067: personal rows are still in the knowledge graph (the merge did not complete); apply this file again';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'kg_entities'
                          AND column_name = 'is_self') THEN
            RAISE EXCEPTION '067 needs 066 applied first (kg_entities.is_self is missing)';
        END IF;
        SELECT string_agg(ix, ', ') INTO bad
          FROM unnest(ARRAY['kg_rel_pair_any', 'kg_rel_tgt_any', 'kg_entities_norm_any']) AS ix
         WHERE NOT EXISTS (SELECT 1
                             FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                            WHERE c.relnamespace = 'public'::regnamespace
                              AND c.relname = ix AND i.indisvalid);
        IF bad IS NOT NULL THEN
            RAISE EXCEPTION '067: index % is missing or INVALID (the merge transaction did not complete); apply this file again', bad;
        END IF;
    END IF;

    -- ACCESS EXCLUSIVE: SET DEFAULT and DROP TRIGGER both take it. One statement names every
    -- table this run changes and NOWAIT never queues, so a pending request cannot make a
    -- reader wait behind it. A refused attempt (a subtransaction) gives back whatever it got
    -- and the next one starts clean. What stays held across the sleeps is the SHARE ROW
    -- EXCLUSIVE above (only while the stamp was still below 067): writers wait for it, and a
    -- writer that already holds a lock on one of these tables refuses every attempt.
    IF trg AND NOT 'kg_relationships' = ANY (need) THEN
        need := need || 'kg_relationships'::text;
    END IF;
    IF cardinality(need) > 0 THEN
        FOR attempt IN 1..tries LOOP
            BEGIN
                EXECUTE 'LOCK TABLE ' || (SELECT string_agg(format('public.%I', x), ', ')
                                            FROM unnest(need) AS x)
                     || ' IN ACCESS EXCLUSIVE MODE NOWAIT';
                FOREACH t IN ARRAY need LOOP
                    IF NOT EXISTS (
                        SELECT 1
                          FROM pg_attrdef d
                          JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                         WHERE d.adrelid = ('public.' || t)::regclass
                           AND a.attname = 'group_id'
                           AND pg_get_expr(d.adbin, d.adrelid) = '''technical''::text') THEN
                        EXECUTE format(
                            'ALTER TABLE public.%I ALTER COLUMN group_id SET DEFAULT ''technical''', t);
                    END IF;
                END LOOP;
                IF trg THEN
                    DROP TRIGGER IF EXISTS kg_rel_personal_audience ON public.kg_relationships;
                END IF;
                EXIT;
            EXCEPTION WHEN lock_not_available OR deadlock_detected THEN
                IF attempt = tries THEN
                    RAISE;
                END IF;
                PERFORM pg_sleep(pause_s);
            END;
        END LOOP;
    END IF;

    -- The loop ends by finishing its changes or by raising; say from the catalog that it
    -- did, so the stamp below can never follow a tail that did not run.
    SELECT string_agg(x, ', ') INTO bad
      FROM unnest(ARRAY['kg_entities', 'kg_relationships', 'notes', 'preferences']) AS x
     WHERE NOT EXISTS (
        SELECT 1
          FROM pg_attrdef d
          JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
         WHERE d.adrelid = ('public.' || x)::regclass
           AND a.attname = 'group_id'
           AND pg_get_expr(d.adbin, d.adrelid) = '''technical''::text');
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION '067: group_id has no ''technical'' default on % (the tail did not finish); apply this file again', bad;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_trigger
                WHERE tgrelid = 'public.kg_relationships'::regclass
                  AND tgname = 'kg_rel_personal_audience' AND NOT tgisinternal) THEN
        RAISE EXCEPTION '067: the 066 insert trigger is still in place (the tail did not finish); apply this file again';
    END IF;
END
$pm$;

DROP FUNCTION IF EXISTS kg_relationships_personal_audience_trg();

-- Last, so a failure above leaves the stamp at 066. Guarded: a re-run of an older file
-- never moves it down.
INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '067')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
 WHERE synapse_meta.value < EXCLUDED.value;
-- txn-b:end
COMMIT;

ANALYZE kg_entities;
ANALYZE kg_relationships;
