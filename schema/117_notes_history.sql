-- schema-check: optional
-- notes_history -- the text a note had before remember() rewrote it in place.
--
-- A restatement that the write-time judge calls a duplicate refreshes the existing note's
-- hook and body (the id, and so every board reference to it, stays). Without a record the
-- old wording was gone for good: only the archived manual episode still held it (source_ref
-- points there); update_note now moves source_ref to the new episode only when it writes a
-- history row.
--
-- update_note copies the row's hook, body and source_ref here in the same transaction as the
-- overwrite, and only when the text actually changes. Nothing reads the table at run time; it
-- is the undo copy, read by an operator:
--   SELECT * FROM notes_history WHERE note_id = N ORDER BY replaced_at DESC;
--
-- Optional: the code probes for the table and rewrites in place without a record until it
-- exists, which is how it behaved before this migration.

CREATE TABLE IF NOT EXISTS notes_history (
    id          BIGSERIAL   PRIMARY KEY,
    note_id     BIGINT      NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    hook        TEXT        NOT NULL,
    body        TEXT        NOT NULL,
    source_ref  TEXT,
    replaced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS notes_history_note_idx ON notes_history (note_id, replaced_at DESC);
