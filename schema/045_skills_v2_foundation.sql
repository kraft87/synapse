-- 045: dream->skills lane v2 foundation — richer candidates, retune directions,
--      and a skill_usage data repair.
--
-- The v1 observe->proposed gate (score >= 1.5) proved uncrossable at the real event
-- rate, so v2 moves to richer detector evidence (verbatim quote + scan_night recurrence
-- unit), a looser create gate, and junk control downstream via candidate decay + human
-- review. This migration is the schema half; the writer contract lives in
-- dream/skills/skill_ledger.py.
--
-- Idempotent: safe to re-run against an already-migrated database.

-- ---------------------------------------------------------------------------
-- (1) Candidate columns
-- ---------------------------------------------------------------------------
-- salience        1-5 pain/severity from the detector (persisted as max across sightings)
-- source_detector which lane found it: gap_scan | struggle_arc | post_fire | procedure_miner | ...
-- proposed_patch  retunes: the drafted concrete change as bulleted change-intent TEXT —
--                 never a unified diff (retune targets can live on another machine, so a
--                 line-anchored diff is infeasible); rendered by skill_review show
ALTER TABLE skills_lane.skill_gap_candidates
    ADD COLUMN IF NOT EXISTS salience        SMALLINT,
    ADD COLUMN IF NOT EXISTS source_detector TEXT,
    ADD COLUMN IF NOT EXISTS proposed_patch  TEXT;

ALTER TABLE skills_lane.skill_gap_candidates
    DROP CONSTRAINT IF EXISTS skill_gap_candidates_salience_check;
ALTER TABLE skills_lane.skill_gap_candidates
    ADD CONSTRAINT skill_gap_candidates_salience_check
    CHECK (salience IS NULL OR salience BETWEEN 1 AND 5);

-- Retune directions grow the update-first ladder: patch existing ('fix') > extend
-- existing ('extend') > new skill. 'widen'/'narrow' stay valid for existing rows.
ALTER TABLE skills_lane.skill_gap_candidates
    DROP CONSTRAINT IF EXISTS skill_gap_candidates_direction_check;
ALTER TABLE skills_lane.skill_gap_candidates
    ADD CONSTRAINT skill_gap_candidates_direction_check
    CHECK (direction IN ('widen', 'narrow', 'extend', 'fix'));

-- (Registry stale/archive lifecycle was CUT from v2 foundation on Oracle review:
--  skill_registry carries no provenance column, so a lifecycle could not distinguish
--  agent-derived skills from hand-authored ones and risked archiving the user's own
--  skills. Junk control stays candidate-side: observe rows decay at 28d, proposed
--  rows wait for human review.)

-- ---------------------------------------------------------------------------
-- (2) Data repair: ghost skill_usage rows
-- ---------------------------------------------------------------------------
-- A historical backfill scan emitted phantom fire rows named 'research',
-- 'multi-research' and 'deep-research' alongside the real 'run-research' fire they
-- were misread from — all four sharing one (session_id, fired_at). The ghosts are
-- detector artifacts (no such skills exist); delete any usage row for those names
-- that shares its exact (session_id, fired_at) with a real run-research row.
DELETE FROM skills_lane.skill_usage g
 USING skills_lane.skill_usage r
 WHERE g.skill IN ('research', 'multi-research', 'deep-research')
   AND r.skill = 'run-research'
   AND g.session_id = r.session_id
   AND g.fired_at = r.fired_at;
