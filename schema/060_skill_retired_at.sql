-- 060_skill_retired_at.sql
-- Record WHEN a skill was retired, so a retirement can stick.
--
-- Setting status='retired' never held: a retired skill drops out of /skills/list, so every
-- opted-in machine that still had it on disk saw "only on disk", published it back, and the
-- publish upsert made it active again. The two-way sync settles every disagreement by
-- newest edit wins, so a retirement has to be an edit with a time the clients can compare
-- their own copies against:
--
--   retired_at  when /skills/retire retired the skill. NULL while it is active: a publish
--               or an accept that (re)activates the row clears it. A publish whose
--               content_modified_at is not newer than retired_at is refused, and
--               /skills/list reports the retirement so a machine can remove the copy
--               Synapse delivered to it.
--
-- updated_at cannot serve: the nightly usage rollup bumps it on every row with fires,
-- retired ones included, so it says when the row was last touched, not when it was
-- retired. It is still the best trace a row retired before 060 has, so the backfill
-- stamps those rows with it.
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the
-- column does not exist; the code probes for it, /skills/retire answers that the schema is
-- missing, and list and publish behave exactly as before 060. The marker below lets the
-- boot-time schema guard (ingestion/schema_check.py) start a build on a database behind
-- only by this file:
-- schema-check: optional
-- Both statements are guarded, so a re-run is a no-op.

ALTER TABLE skills_lane.skill_registry
    ADD COLUMN IF NOT EXISTS retired_at TIMESTAMPTZ;

UPDATE skills_lane.skill_registry
   SET retired_at = updated_at
 WHERE status = 'retired' AND retired_at IS NULL;
