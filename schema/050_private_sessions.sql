-- 050_private_sessions.sql
-- private_sessions — the durable "off the record" list behind private mode.
-- One row per Claude Code session whose turns must NEVER become episodes. The
-- plugin's Stop hook already skips posting while a local marker file exists
-- (~/.synapse/private/<session_id>), but that marker is host-local and expires;
-- this table is the server-side enforcement, checked at the same chokepoint the
-- contamination predicate runs at (/ingest + the disk backfill). It covers every
-- path that bypasses the hook — catch-up sweeps, the bulk backfill, a re-scan of
-- transcripts that are still sitting on disk months later.
-- Rows are PERMANENT by design: the transcript stays on disk forever, so deleting
-- the row would let a later backfill ingest the very turns the user marked private.
-- Only the local markers expire (12h) — the row is the record of intent.
CREATE TABLE IF NOT EXISTS private_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
