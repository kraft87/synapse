-- 027_skill_proposal_body.sql
-- Store the drafted SKILL.md body in the candidate row, not only as a file path.
-- The dream container writes proposal drafts to its local disk (proposal_path), but the
-- review path now runs over HTTP via the mcp-server container, which can't read another
-- container's filesystem. Carrying the draft body in the DB makes the proposal fully
-- self-describing, so /skills/proposals can serve it to a DSN-free review client.
ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS proposal_body text;
