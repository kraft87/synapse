-- 041_notes.sql
-- notes — the EXPLICIT-memory store behind the always-injected board (get_context).
-- One row per curated memory: a ~120-char `hook` (the board line; also the embed
-- target for dedup KNN) + a self-contained `body` (fetched on demand by id).
-- Kept out of the KG (same supernode rationale as preferences/timeline) and out of
-- episodes (episodes are the archive; notes are the index). Live set = rows with
-- superseded_by IS NULL; contradictions supersede (lineage), restatements UPDATE
-- in place (updated_at bump). Types: user/feedback = global scope; project =
-- project-scoped, stale-managed by the board's overflow policy; reference = pointers.
CREATE TABLE IF NOT EXISTS notes (
    id            BIGSERIAL PRIMARY KEY,
    owner_id      TEXT        NOT NULL,
    group_id      TEXT        NOT NULL,
    project       TEXT,
    type          TEXT        NOT NULL,
    hook          TEXT        NOT NULL,
    body          TEXT        NOT NULL,
    embedding     halfvec(2048),
    embed_model   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_by BIGINT      REFERENCES notes(id),
    source_ref    TEXT,
    CONSTRAINT notes_type_chk CHECK (type IN ('user','feedback','project','reference'))
);
CREATE INDEX IF NOT EXISTS notes_live_idx    ON notes (owner_id, group_id) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS notes_project_idx ON notes (project)            WHERE superseded_by IS NULL;
-- Vector index deferred (dozens-hundreds of rows; filtered seqscan is instant — 033/035 precedent).
