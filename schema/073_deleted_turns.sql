-- schema-check: optional
-- Tombstones for episodes deleted from memory (the dashboard's delete, private mode's purge).
--
-- A delete removes the episode, so nothing marked it gone: the Stop hook re-ships the
-- last turn, and a catch-up sweep, synapse-import or ingestion.backfill re-reads
-- transcripts that stay on disk, and the turn came back under a new id to be chunked,
-- extracted and served again. The delete cascade now records one row per deleted turn
-- here, in its own transaction, and the ingest chokepoint (ingestion.episode_writer)
-- drops a turn that matches one:
--   * span_id: the turn's identity, which a re-ship of the same session repeats;
--   * content_md5 within the same project or working directory: a replayed session
--     carries fresh span ids (the same reason as the episodes content guard).
-- Rows are permanent by design, like private_sessions; there is no undelete.
--
-- Additive, and the code probes for the table: until it exists a delete leaves no
-- tombstone and ingest checks nothing, which is the previous behaviour, so the image
-- can land before this is applied. The previous image never reads the table.

CREATE TABLE IF NOT EXISTS deleted_turns (
    id          BIGSERIAL PRIMARY KEY,
    span_id     TEXT,
    session_id  TEXT        NOT NULL,
    content_md5 TEXT        NOT NULL,
    project     TEXT,
    cwd         TEXT,
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS deleted_turns_span_idx ON deleted_turns (span_id) WHERE span_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS deleted_turns_md5_idx ON deleted_turns (content_md5);
