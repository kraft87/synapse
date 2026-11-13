-- 070_note_history.sql
-- schema-check: optional
--
-- Append-only history of the notes table. remember() updates a note in place when it
-- judges the new text a restatement of a stored one, and the nightly curation lane merges
-- and retires notes; before this file the text a write replaced was gone from the table
-- (2026-11-13: a new benchmark result was judged a restatement of an older, different
-- one and overwrote it). A trigger copies the row as it was BEFORE every change of a
-- note's text, type, project, audience or successor, and before every delete, so the
-- earlier version of any note can be read back, whichever writer changed it.
--
-- The embedding is not copied: it is derived from the hook and can be computed again.
-- note_id is not a foreign key, so the history of a deleted note outlives it.
--
-- Optional: no service reads this table, so the image runs the same before the file is
-- applied. Idempotent. Rollback: DROP the two triggers, the function, the table.

CREATE TABLE IF NOT EXISTS notes_history (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_id         bigint NOT NULL,
    owner_id        text NOT NULL,
    project         text,
    type            text NOT NULL,
    hook            text NOT NULL,
    body            text NOT NULL,
    audience        text,
    superseded_by   bigint,
    source_ref      text,
    note_created_at timestamptz,
    note_updated_at timestamptz,
    op              text NOT NULL,   -- 'updated' | 'deleted': what happened to this version
    recorded_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS notes_history_note_idx ON notes_history (note_id, recorded_at DESC);

CREATE OR REPLACE FUNCTION notes_history_record() RETURNS trigger AS $$
BEGIN
    INSERT INTO notes_history (note_id, owner_id, project, type, hook, body, audience,
                               superseded_by, source_ref, note_created_at, note_updated_at, op)
        VALUES (OLD.id, OLD.owner_id, OLD.project, OLD.type, OLD.hook, OLD.body, OLD.audience,
                OLD.superseded_by, OLD.source_ref, OLD.created_at, OLD.updated_at,
                CASE WHEN TG_OP = 'DELETE' THEN 'deleted' ELSE 'updated' END);
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Not on a change of updated_at, embedding or source_ref alone: those carry no content.
DROP TRIGGER IF EXISTS notes_history_update_trg ON notes;
CREATE TRIGGER notes_history_update_trg
    AFTER UPDATE ON notes
    FOR EACH ROW WHEN (
        OLD.hook IS DISTINCT FROM NEW.hook OR OLD.body IS DISTINCT FROM NEW.body
        OR OLD.type IS DISTINCT FROM NEW.type OR OLD.project IS DISTINCT FROM NEW.project
        OR OLD.audience IS DISTINCT FROM NEW.audience
        OR OLD.superseded_by IS DISTINCT FROM NEW.superseded_by
    )
    EXECUTE FUNCTION notes_history_record();

DROP TRIGGER IF EXISTS notes_history_delete_trg ON notes;
CREATE TRIGGER notes_history_delete_trg
    AFTER DELETE ON notes
    FOR EACH ROW
    EXECUTE FUNCTION notes_history_record();
