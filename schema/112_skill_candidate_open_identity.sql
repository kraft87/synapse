-- schema-check: optional
-- Skills ledger: a retune's identity key is unique among the OPEN candidates only.
--
-- skill_gap_named_idx made (kind, name, direction) unique across every status, so once a
-- retune had been accepted its (skill, direction) stayed occupied for the life of the
-- database: every later sighting was folded into the closed row and never reached review. The
-- key now covers only the states that still wait on a decision (observe, proposed, rejected
-- inside or past its cooldown). A sighting after an accept or a retirement opens a fresh
-- candidate whose evidence is post-change only, and the closed row stays as history. The
-- consolidate kind left with its nightly pass, so it leaves the index too.
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the old index
-- still occupies the key of a closed row: the writer's INSERT then does nothing and
-- skill_ledger.merge_candidate reports the closed row untouched (the pre-migration
-- behaviour, minus the evidence merge). Its ON CONFLICT clause infers either index. The
-- marker on the first line lets the boot-time schema guard (ingestion/schema_check.py) start
-- a build on a database behind only by this file.
--
-- Apply it after the new image is running: an older image's INSERT names the old index
-- predicate, which cannot infer this one, so it would fail on every NEW retune key (merges
-- into existing rows still work).
--
-- Dropping and creating the index in one transaction leaves no window without the guard.
-- A re-run recreates the same index.

BEGIN;

DROP INDEX IF EXISTS skills_lane.skill_gap_named_idx;

CREATE UNIQUE INDEX skill_gap_named_idx
    ON skills_lane.skill_gap_candidates (kind, name, COALESCE(direction, '-'))
    WHERE kind = 'retune' AND status IN ('observe', 'proposed', 'rejected');

COMMIT;
