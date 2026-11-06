-- 062_skill_review_queue.sql
-- The dream->skills review queue looks after itself (dream/skills/review_queue.py), and the
-- nightly run can see the proposals it raised.
--
--   proposed_at     when the row last became 'proposed': the ledger gate, the nightly re-gate
--                   of open derive rows, or a human `skill_review.py revive`. dream_runs
--                   counts the proposals a run raised by it (it counted rows CREATED that
--                   night, so a row that sat in observe and went proposed later, like #76
--                   on 10-29, was never counted), the 21-day expiry reads it, and the
--                   report, the review list and the morning briefing show days pending.
--   recommendation  the nightly judge's {verdict, reason, draft_hash, model, at} for the
--                   row's current draft. It is asked again when the draft changes
--                   (draft_hash no longer matches the draft).
--
-- Backfill: a row already proposed gets proposed_at = updated_at, capped at 7 days ago.
-- A row that has waited longer (on 2026-11-06, #142, proposed 10-11) gets a
-- recommendation and two weeks in the new queue instead of expiring the night this is
-- applied, before anyone saw a recommendation for it.
--
-- Deploy order: the image ships BEFORE this file is applied. Until it is, the lane probes
-- for both columns and skips what needs them (the expiry, the recommendations, proposed_at
-- stamping) with one log line; proposals_raised falls back to rows created that night. The
-- marker below lets the boot-time schema guard (ingestion/schema_check.py) start a build on
-- a database behind only by this file:
-- schema-check: optional
-- Both statements are guarded, so a re-run is a no-op.

ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS proposed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recommendation JSONB;

UPDATE skills_lane.skill_gap_candidates
   SET proposed_at = GREATEST(updated_at, now() - interval '7 days')
 WHERE status = 'proposed' AND proposed_at IS NULL;
