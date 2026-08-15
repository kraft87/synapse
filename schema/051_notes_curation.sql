-- 051_notes_curation.sql
-- notes_curation — the memo + audit table behind the dream→notes curation lane.
-- The notes store (041) has one inflow (remember()) and had no outflow: nothing
-- merged paraphrase duplicates, retired notes a later correction had already
-- replaced, or re-scoped project-specific content typed as global feedback, so
-- the always-injected board relied on silent truncation to stay bounded.
-- The nightly lane (dream/notes/) closes that loop. Every judgement it makes —
-- applied or not — lands here, which makes this table three things at once:
--   * the AUDIT log: what the lane did, when, and on which rows (detail jsonb
--     carries the winner/loser, the resolved project slug, the skip reason);
--   * the MEMO: a judged pair/note is never re-sent to the LLM, so cost per
--     night falls to whatever is genuinely new;
--   * the STALENESS cursor: judged_at is compared against notes.updated_at, so
--     a note edited after its verdict is re-judged rather than frozen.
-- One row per unit of work: op='pair' (note_a + note_b, order-insensitive) or
-- op='retype' (note_a only, note_b NULL). Re-judging UPSERTs onto the partial
-- unique indexes below rather than appending — the row IS the current verdict.
-- Nothing here is destructive: the lane's only writes to notes are
-- superseded_by (reversible, lineage kept) and type/project (reversible).
CREATE TABLE IF NOT EXISTS notes_curation (
    id        BIGSERIAL   PRIMARY KEY,
    op        TEXT        NOT NULL,
    note_a    BIGINT      NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    note_b    BIGINT      REFERENCES notes(id) ON DELETE CASCADE,
    verdict   TEXT        NOT NULL,
    applied   BOOLEAN     NOT NULL DEFAULT false,
    detail    JSONB,
    judged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT notes_curation_op_chk CHECK (op IN ('pair','retype')),
    CONSTRAINT notes_curation_shape_chk CHECK (
        (op = 'pair'   AND note_b IS NOT NULL AND note_b <> note_a) OR
        (op = 'retype' AND note_b IS NULL)
    )
);

-- Pair identity is unordered: (12,34) and (34,12) are the same judgement, so the
-- index keys on the normalized (least, greatest) form. The lane always inserts
-- with note_a < note_b, but the index is what makes that an invariant.
CREATE UNIQUE INDEX IF NOT EXISTS notes_curation_pair_idx
    ON notes_curation (least(note_a, note_b), greatest(note_a, note_b))
    WHERE op = 'pair';

CREATE UNIQUE INDEX IF NOT EXISTS notes_curation_retype_idx
    ON notes_curation (note_a)
    WHERE op = 'retype';
