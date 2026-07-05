-- 040: dedup_gate_shadow — threshold-picking telemetry for the Stage-6 gray-zone
-- gate (issue #14). One row per (new fact, dedup/contradiction candidate): the
-- gate's embedding similarity + would-be decision next to the LLM confirm's
-- actual verdict for the same pair.
--
-- Threshold analysis: agreement rate of decision='merge' rows with
-- llm_duplicate, and of decision='new' rows with (NOT llm_duplicate AND NOT
-- llm_contradicted), sliced by sim. llm_ran=false rows (batch confirm failed,
-- or the candidate was dropped by enforcement) carry NULL verdicts — exclude
-- them. Append-only; prune by created_at once enforcement ships.

CREATE TABLE IF NOT EXISTS dedup_gate_shadow (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    group_id         TEXT NOT NULL,
    fact             TEXT NOT NULL,         -- new fact text (truncated 500)
    candidate_uuid   TEXT NOT NULL,
    candidate_fact   TEXT,                  -- existing edge's fact (truncated 500)
    pool             TEXT NOT NULL,         -- 'pair' | 'semantic'
    sim              REAL,                  -- cosine similarity; NULL = BM25-only candidate
    decision         TEXT NOT NULL,         -- 'merge' | 'new' | 'gray'
    llm_duplicate    BOOLEAN,               -- LLM verdict (NULL when llm_ran = false)
    llm_contradicted BOOLEAN,
    llm_ran          BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS dedup_gate_shadow_created_idx
    ON dedup_gate_shadow (created_at DESC);
