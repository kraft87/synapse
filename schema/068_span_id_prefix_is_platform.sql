-- 068_span_id_prefix_is_platform.sql
-- A turn's span id starts with the platform that produced it: one name per client.
--
-- Three parsers prefixed their span ids with the transcript format instead, so the same
-- client went by two names in one row (platform 'claude_code', span id 'jsonl:...'):
--
--   platform      span id was          span id is
--   claude_code   jsonl:<uuid>         claude_code:<uuid>
--   chatgpt       openai:<id>          chatgpt:<id>
--   cursor        cursor-sqlite:<...>  cursor:<...>
--
-- codex: and claude_ai: already matched their platform and do not change. `source` keeps
-- naming the transcript format ('jsonl', 'openai', 'cursor_sqlite'); nothing reads it as
-- an identity.
--
-- Two parts:
--
--   1. A trigger that rewrites the old spellings on every INSERT and on every UPDATE of
--      span_id. It stays for good. episodes_span_id_idx is the dedup: a build from before
--      this file writes 'jsonl:<uuid>', the trigger stores 'claude_code:<uuid>', and a
--      turn already stored under the new spelling is a unique violation the ingest route
--      has always treated as "already have it". Without the trigger an older build
--      (redeployed by a rollback, or still running while this file is applied) would
--      store every re-sent turn a second time under the old spelling.
--   2. One UPDATE that renames the stored span ids.
--
-- Deploy order: apply this file BEFORE the image that ships with it. The older image
-- keeps working against the renamed rows because of the trigger (the schema guard lets a
-- build start on a database that is ahead of it). The newer image must not run before
-- the rename: it would look for 'claude_code:<uuid>', miss the stored 'jsonl:<uuid>' and
-- store the turn again. So this file is NOT marked optional: the newer image refuses to
-- start on a database without it.
--
-- The UPDATE rewrites every renamed row (about three in five on a database fed mostly by
-- Claude Code), and each new row version is entered into every index on episodes,
-- including the BM25 and HNSW ones. Expect minutes on tens of thousands of episodes, and
-- run VACUUM (ANALYZE) episodes afterwards.
--
-- The file stamps schema_version 068 itself, last: it is applied by hand, ahead of the
-- image, and that image will not start on a database stamped lower.
--
-- Re-running is a no-op: the functions and the trigger are replaced, no row matches the
-- UPDATE once it has run, and the stamp only ever moves up.

CREATE OR REPLACE FUNCTION canonical_span_id(span TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $SPAN$
    SELECT CASE
        WHEN span LIKE 'jsonl:%'         THEN 'claude_code:' || substr(span, length('jsonl:') + 1)
        WHEN span LIKE 'openai:%'        THEN 'chatgpt:'     || substr(span, length('openai:') + 1)
        WHEN span LIKE 'cursor-sqlite:%' THEN 'cursor:'      || substr(span, length('cursor-sqlite:') + 1)
        ELSE span
    END
$SPAN$;

CREATE OR REPLACE FUNCTION episodes_canonical_span_id() RETURNS trigger
LANGUAGE plpgsql AS $SPAN$
BEGIN
    NEW.span_id := canonical_span_id(NEW.span_id);
    RETURN NEW;
END
$SPAN$;

-- Before the rename, not after: from here on nothing can add a row under an old spelling
-- behind the UPDATE's back. The WHEN clause keeps the trigger off every other row.
DROP TRIGGER IF EXISTS episodes_canonical_span_id ON episodes;
CREATE TRIGGER episodes_canonical_span_id
    BEFORE INSERT OR UPDATE OF span_id ON episodes
    FOR EACH ROW
    WHEN (
        NEW.span_id LIKE 'jsonl:%'
        OR NEW.span_id LIKE 'openai:%'
        OR NEW.span_id LIKE 'cursor-sqlite:%'
    )
    EXECUTE FUNCTION episodes_canonical_span_id();

-- A renamed span id that is already taken would fail the UPDATE on the unique index with
-- nothing to say which rows. That happens only where a build newer than the database has
-- already stored turns under the new spelling. Say so, and change nothing.
DO $SPAN$
DECLARE
    clashes BIGINT;
    example TEXT;
BEGIN
    SELECT count(*), min(old.span_id) INTO clashes, example
      FROM episodes old
      JOIN episodes new ON new.span_id = canonical_span_id(old.span_id)
     WHERE old.span_id LIKE 'jsonl:%'
        OR old.span_id LIKE 'openai:%'
        OR old.span_id LIKE 'cursor-sqlite:%';
    IF clashes > 0 THEN
        RAISE EXCEPTION
            '068: % episode(s) are stored under both the old and the new span id (for example %). '
            'A build newer than this database stored them a second time. Delete the newer copy of '
            'each pair, then run this file again.', clashes, example;
    END IF;
END
$SPAN$;

UPDATE episodes
   SET span_id = canonical_span_id(span_id)
 WHERE span_id LIKE 'jsonl:%'
    OR span_id LIKE 'openai:%'
    OR span_id LIKE 'cursor-sqlite:%';

-- Rows an operator moved aside by hand keep the span id they had. The table is not part
-- of this schema, so it is renamed only where it exists.
DO $SPAN$
BEGIN
    IF to_regclass('public.episodes_quarantine') IS NOT NULL THEN
        UPDATE episodes_quarantine
           SET span_id = canonical_span_id(span_id)
         WHERE span_id LIKE 'jsonl:%'
            OR span_id LIKE 'openai:%'
            OR span_id LIKE 'cursor-sqlite:%';
    END IF;
END
$SPAN$;

-- Last, so a failure above leaves the stamp where it was, and the build that needs this
-- file still refuses to start. Guarded: a re-run never moves the stamp down.
INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '068')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
 WHERE synapse_meta.value < EXCLUDED.value;
