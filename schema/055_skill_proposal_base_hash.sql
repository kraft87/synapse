-- 055_skill_proposal_base_hash.sql
-- Accept applies a skill proposal: the review accept writes proposal_body straight into
-- skills_lane.skill_registry (the same upsert /skills/publish runs), and the two-way sync
-- delivers it to whichever machine owns the skill. A drafted body is therefore only safe to
-- apply against the registry body it was drafted FROM; if the skill changed underneath the
-- draft (its owner edited and synced it), applying the draft would silently revert that edit.
--
--   base_body_hash  sha256 hex of the registry body the nightly drafted proposal_body
--                   against (retunes). NULL for a new-skill draft (derive), which is
--                   drafted against no existing skill, and for rows never drafted.
--
-- Additive and nullable: safe to apply before the code that reads it deploys, and the
-- running code ignores it until then.
ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS base_body_hash TEXT;
