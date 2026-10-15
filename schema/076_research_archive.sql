-- schema-check: optional
-- The cold research archive: the verbatim results of the sources a capture rule sends to
-- destination = 'archive' (X and Bluesky threads, YouTube transcripts; see
-- ingestion/capture/rules/). No FKs, never chunked, embedded, extracted or served by recall.
--
-- A deployment whose out-of-repo cron ran the old ResearchArchiveCapture already has this
-- table (it created it on the fly); IF NOT EXISTS leaves it as it is, and the ALTERs give it
-- the provenance a delete needs: the session and the stored turn the capture came from (a
-- soft pointer, no FK, like web_artifacts.parent_episode_id).
--
-- Optional: the code that writes the archive probes for parent_episode_id and, until this is
-- applied, skips archive-rule captures (reason 'archive_unavailable') instead of failing.

CREATE TABLE IF NOT EXISTS research_archive (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind        text        NOT NULL,
    source_id   text        NOT NULL UNIQUE,
    url         text        NOT NULL,
    title       text,
    author      text,
    brief_slug  text,
    captured_from text,
    content     text        NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE research_archive
    ADD COLUMN IF NOT EXISTS session_id text,
    ADD COLUMN IF NOT EXISTS parent_episode_id bigint;

CREATE INDEX IF NOT EXISTS research_archive_session_idx
    ON research_archive (session_id);
