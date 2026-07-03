-- 026_skill_history.sql
-- Append-only audit + conflict-recovery log for skill content. A trigger captures every
-- superseded or deleted skill BODY in PG, so the version a push (or a future dream edit)
-- overwrote stays recoverable and you get a full change trail. The sync hook additionally
-- records a local-disk body it overwrites on a pull -- the one data-loss case the trigger
-- can't see: a local edit that never reached PG. Body-level for v1 (bundled-file history
-- can follow). Prune later if it grows (keep last N per name).

CREATE TABLE IF NOT EXISTS skills_lane.skill_history (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name                text NOT NULL,
    scope               text,
    body                text,
    content_modified_at timestamptz,
    op                  text NOT NULL,   -- 'superseded' | 'deleted' | 'disk_overwrite'
    recorded_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS skill_history_name_idx
    ON skills_lane.skill_history (name, recorded_at DESC);

CREATE OR REPLACE FUNCTION skills_lane.skill_history_record() RETURNS trigger AS $$
BEGIN
    INSERT INTO skills_lane.skill_history (name, scope, body, content_modified_at, op)
        VALUES (OLD.name, OLD.scope, OLD.body, OLD.content_modified_at,
                CASE WHEN TG_OP = 'DELETE' THEN 'deleted' ELSE 'superseded' END);
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Only log when the body actually changed (skip no-op nightly re-stocks).
DROP TRIGGER IF EXISTS skill_history_update_trg ON skills_lane.skill_registry;
CREATE TRIGGER skill_history_update_trg
    AFTER UPDATE ON skills_lane.skill_registry
    FOR EACH ROW WHEN (OLD.body IS DISTINCT FROM NEW.body)
    EXECUTE FUNCTION skills_lane.skill_history_record();

DROP TRIGGER IF EXISTS skill_history_delete_trg ON skills_lane.skill_registry;
CREATE TRIGGER skill_history_delete_trg
    AFTER DELETE ON skills_lane.skill_registry
    FOR EACH ROW
    EXECUTE FUNCTION skills_lane.skill_history_record();
