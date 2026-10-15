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
-- still occupies the key of a promoted or retired row: a re-sighting of that retune hits ON
-- CONFLICT DO NOTHING, its evidence is dropped (a promoted row used to take the sighting, a
-- retired one was reopened), and skill_ledger.merge_candidate reports the closed row
-- untouched, with a warning. Apply this file soon after the new image is running. Its ON
-- CONFLICT clause infers either index. The marker on the first line lets the boot-time schema
-- guard (ingestion/schema_check.py) start a build on a database behind only by this file.
--
-- Do not roll the image back past this file. Postgres picks the ON CONFLICT arbiter index when
-- it plans the statement, and an older image's clause (WHERE kind IN ('retune',
-- 'consolidate')) cannot infer this index, so once this file is applied that image's
-- candidate INSERT fails for every new candidate of any kind, derive included: the judge,
-- grounded and struggle_arc stages fail every night. No index can serve that clause and still
-- allow a closed and an open row on one key. A rollback needs DROP INDEX
-- skills_lane.skill_gap_named_idx, retiring or deleting the duplicate (kind, name, direction)
-- retune rows, and recreating the index from schema/022.
--
-- Dropping and creating the index in one transaction leaves no window without the guard.
-- A re-run recreates the same index.

BEGIN;

DROP INDEX IF EXISTS skills_lane.skill_gap_named_idx;

CREATE UNIQUE INDEX skill_gap_named_idx
    ON skills_lane.skill_gap_candidates (kind, name, COALESCE(direction, '-'))
    WHERE kind = 'retune' AND status IN ('observe', 'proposed', 'rejected');

COMMIT;
