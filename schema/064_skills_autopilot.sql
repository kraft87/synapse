-- 064_skills_autopilot.sql
-- The dream->skills lane decides the routine proposals itself (dream/skills/autoreview.py).
--
-- Policy (approved 2026-11-06): a proposal is applied without a person when an Opus
-- reviewer accepts it AND an independent verifier of another model family passes it in
-- both evidence orders, and it is either a new skill the lane created (a derive that met
-- the recurrence gate) or a change to a skill's frontmatter description only, on a skill
-- that is not pinned. Reviewer reject + verifier agreeing rejects it (no grounded -3, the
-- usual 30-day cooldown). Everything else stays 'proposed' for a person: body edits,
-- merges, retire/rescope, pinned skills, reviewer/verifier disagreement.
--
--   skill_auto_reviews      one row per judgement of one candidate's evidence + draft: the
--                           audit trail, the memo that stops a row being re-judged until
--                           its evidence or draft changes (evidence_hash, draft_hash), and,
--                           for an applied change, what an undo needs (previous/applied
--                           body hashes, the skill_history row holding the old body) and
--                           whether it was rolled back.
--   skill_gap_candidates    applied_by ('human'|'auto') and the auto review that applied it.
--   skill_registry          lane_body_hash: the body the lane itself last wrote. A skill
--                           whose body hash differs (or that has none) is hand-written.
--   grounded_required_to_apply
--                           'grounded' keeps meaning a person did it: a row reaches
--                           accepted/promoted with a grounded signal, or through a recorded
--                           auto review. The auto path never writes a grounded entry.
--
-- Deploy order: the image ships BEFORE this file is applied. Until it is, the lane probes
-- for the table and columns and skips the autopilot with one log line (the #229 queue
-- keeps working), and a human accept leaves applied_by unset. The marker below lets the
-- boot-time schema guard start a build on a database behind only by this file:
-- schema-check: optional
-- Every statement is guarded, so a re-run is a no-op.

CREATE TABLE IF NOT EXISTS skills_lane.skill_auto_reviews (
    id                 BIGSERIAL PRIMARY KEY,
    candidate_id       BIGINT NOT NULL
                         REFERENCES skills_lane.skill_gap_candidates(id) ON DELETE CASCADE,
    night              DATE NOT NULL,
    mode               TEXT NOT NULL CHECK (mode IN ('shadow', 'live')),
    evidence_hash      TEXT NOT NULL,
    draft_hash         TEXT,
    tier               TEXT NOT NULL,   -- new_skill | description | body_edit | pinned | ...
    preflight          JSONB NOT NULL DEFAULT '{}',
    reviewer           JSONB,           -- {verdict, reason, model}
    verifier           JSONB,           -- {result, orders:[{...},{...}], model}
    verdict            TEXT NOT NULL
                         CHECK (verdict IN ('applied', 'rejected', 'ask', 'capped',
                                            'would_apply', 'would_reject', 'failed')),
    skill              TEXT,
    previous_body_hash TEXT,
    applied_body_hash  TEXT,
    history_id         BIGINT,          -- skill_history row holding the body it replaced
    applied_at         TIMESTAMPTZ,
    rolled_back_at     TIMESTAMPTZ,
    rollback_reason    TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS skill_auto_reviews_candidate_idx
    ON skills_lane.skill_auto_reviews (candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS skill_auto_reviews_applied_idx
    ON skills_lane.skill_auto_reviews (applied_at DESC)
    WHERE verdict = 'applied';

ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS applied_by TEXT,
    ADD COLUMN IF NOT EXISTS auto_review_id BIGINT;

ALTER TABLE skills_lane.skill_registry
    ADD COLUMN IF NOT EXISTS lane_body_hash TEXT;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'skill_gap_candidates_applied_by_check') THEN
        ALTER TABLE skills_lane.skill_gap_candidates
            ADD CONSTRAINT skill_gap_candidates_applied_by_check
            CHECK (applied_by IS NULL OR applied_by IN ('human', 'auto'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'skill_gap_candidates_auto_review_fk') THEN
        ALTER TABLE skills_lane.skill_gap_candidates
            ADD CONSTRAINT skill_gap_candidates_auto_review_fk
            FOREIGN KEY (auto_review_id) REFERENCES skills_lane.skill_auto_reviews(id);
    END IF;
END $$;

-- Every row applied before this file was applied by a person.
UPDATE skills_lane.skill_gap_candidates
   SET applied_by = 'human'
 WHERE status IN ('accepted', 'promoted') AND applied_by IS NULL;

ALTER TABLE skills_lane.skill_gap_candidates
    DROP CONSTRAINT IF EXISTS grounded_required_to_apply;
ALTER TABLE skills_lane.skill_gap_candidates
    ADD CONSTRAINT grounded_required_to_apply
    CHECK (status NOT IN ('accepted', 'promoted')
           OR grounded_weight > 0
           OR (applied_by = 'auto' AND auto_review_id IS NOT NULL));
