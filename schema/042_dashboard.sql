-- 042_dashboard.sql
-- Operator-dashboard state (issue #12) — the ONLY writable surface the dashboard adds.
-- Everything else the dashboard shows is read-only over existing tables (episodes, the
-- KG, timeline, preferences, notes); flags are the one bit of operator intent it needs
-- to persist, so they get their own store rather than mutating the memory tables.
--
-- dashboard_flags — an operator's "look at this" mark on any memory item (a suspect
-- fact, a mis-ingested episode, a preference to revisit). Deliberately NOT a delete or
-- an edit: the memory item is untouched; a flag is a soft, reversible pointer the
-- operator toggles. active = removed_at IS NULL; a partial-unique index enforces at
-- most one live flag per (kind, item_id) so a re-flag after unflag is a fresh row, not
-- a duplicate. item_id is TEXT because the kinds mix id spaces: episode/timeline/
-- preference/note carry a numeric id as text, a fact carries its edge uuid.
--
-- dashboard_audit — append-only history of every flag toggle (and, in later phases,
-- proposal accept/reject decisions), so the operator's actions are reconstructable
-- independent of the current flag state. Never updated, never deleted; `detail` holds
-- the free-form envelope (the note text on a flag, decision context later).
CREATE TABLE IF NOT EXISTS dashboard_flags (
    id         BIGSERIAL PRIMARY KEY,
    kind       TEXT        NOT NULL,               -- episode|fact|timeline_event|preference|note
    item_id    TEXT        NOT NULL,               -- numeric id as text, or a fact's edge uuid
    note       TEXT,                                -- optional operator note captured at flag time
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    removed_at TIMESTAMPTZ,                          -- set on unflag; NULL = active flag
    CONSTRAINT dashboard_flags_kind_chk
        CHECK (kind IN ('episode', 'fact', 'timeline_event', 'preference', 'note'))
);

-- At most one LIVE flag per item — the toggle path relies on this to find the active
-- row. Superseded (unflagged) rows stay for history and are excluded by the predicate,
-- so a later re-flag inserts cleanly instead of colliding.
CREATE UNIQUE INDEX IF NOT EXISTS dashboard_flags_active_uniq
    ON dashboard_flags (kind, item_id) WHERE removed_at IS NULL;

CREATE TABLE IF NOT EXISTS dashboard_audit (
    id      BIGSERIAL PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    action  TEXT        NOT NULL,                   -- 'flag' | 'unflag' this phase; proposal decisions later
    kind    TEXT,                                    -- mirrors dashboard_flags.kind (NULL for non-flag actions)
    item_id TEXT,                                    -- mirrors dashboard_flags.item_id
    detail  JSONB                                    -- free-form context (note text now; decision payload later)
);
-- action is intentionally NOT CHECK-constrained: later phases append proposal-decision
-- actions here, and a CHECK would force a migration to widen it.
CREATE INDEX IF NOT EXISTS dashboard_audit_ts_idx ON dashboard_audit (ts DESC);
