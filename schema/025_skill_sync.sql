-- 025_skill_sync.sql
-- Two-way skill sync support. Track when a skill's CONTENT (body + bundled files) was last
-- EDITED, so the sync hook orders disk vs PG by newest-edit rather than newest-sync:
--   * push (disk -> PG): set from the disk file mtime (newest among SKILL.md + bundled files)
--   * PG-side edit (e.g. dream): set to now()
--   * pull (PG -> disk): os.utime the written files to this value, so the clocks stay aligned
--     to the content version and "newest" keeps meaning "newest edit".
-- The sync compares a whole-skill content hash to decide IF a skill changed; this timestamp
-- only breaks the tie on WHICH side is newer.
ALTER TABLE skills_lane.skill_registry
    ADD COLUMN IF NOT EXISTS content_modified_at timestamptz;
