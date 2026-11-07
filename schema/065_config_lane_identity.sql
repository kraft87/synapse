-- 065_config_lane_identity.sql
-- The dream->config lane tells two corrections apart by meaning, and looks after the
-- proposals it raises (dream/config/nightly.py, dream/config/review_queue.py).
--
--   embedding       the rule text's embedding, taken once when the row is created and never
--                   moved: a new correction is "the same rule" when its cosine to this
--                   anchor clears the threshold. Rows written before this migration get it
--                   on the next nightly run (the lane embeds every NULL).
--   proposed_at     when the row last became 'proposed'. The 21-day expiry, the report, the
--                   review list and the morning briefing read it.
--   recommendation  the nightly judge's {verdict, reason, key, model, at} for the row's
--                   current rule and evidence; asked again when the evidence changes.
--   coverage        the pre-proposal check against the user's own rule files:
--                   {covered, by, registry_hash, model, at}. Asked again only when the
--                   files changed.
--
-- Backfill: a row already proposed gets proposed_at = updated_at, capped at 7 days ago,
-- so a row that has waited longer gets a week in the new queue instead of expiring the
-- night this is applied.
--
-- Deploy order: the image ships BEFORE this file is applied. Until it is, the lane probes
-- for the four columns, logs one line and idles (the scan watermark does not move, so
-- nothing is lost); the review list shows what it showed before. The marker below lets the
-- boot-time schema guard (ingestion/schema_check.py) start a build on a database behind
-- only by this file:
-- schema-check: optional
-- Every statement is guarded, so a re-run is a no-op.

ALTER TABLE config_lane.config_proposals
    ADD COLUMN IF NOT EXISTS embedding      halfvec(2048),
    ADD COLUMN IF NOT EXISTS proposed_at    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recommendation JSONB,
    ADD COLUMN IF NOT EXISTS coverage       JSONB;

UPDATE config_lane.config_proposals
   SET proposed_at = GREATEST(updated_at, now() - interval '7 days')
 WHERE status = 'proposed' AND proposed_at IS NULL;
