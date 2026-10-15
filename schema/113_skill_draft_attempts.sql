-- schema-check: optional
-- Skills ledger: remember a proposal's failed drafts.
--
-- The nightly draft pass drafts up to DRAFT_CAP proposed rows, best first. A row whose draft
-- failed its checks stayed body-less and was tried again, with an Opus call, every night
-- (proposed rows never decay), so a few permanently failing rows at the top of the ordering
-- took the whole budget and told the reviewer nothing but "no draft yet".
--
--   draft_attempts    failed drafts since the last good one (0 once a draft is stored)
--   last_draft_at     when the last attempt failed; the pass skips the row for 1, 2, 4...
--                     days (at most 16) after that and tries rows with fewer failures first
--   last_draft_error  why it failed, shown by `skill_review show` so the reviewer can
--                     hand-draft (accept --body-file) or reject
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the draft
-- pass probes for the columns and keeps its pre-migration behaviour (every row retried every
-- night, no reason stored). The marker on the first line lets the boot-time schema guard
-- (ingestion/schema_check.py) start a build on a database behind only by this file.
-- Additive and guarded: an older image ignores the columns, and a re-run is a no-op.

ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS draft_attempts   INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_draft_at    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_draft_error TEXT;
