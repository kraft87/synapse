-- 052: remember_intents — idempotency ledger for spooled remember() writes.
--
-- The Claude Code plugin spools a remember() intent to local disk whenever the MCP
-- tool is unavailable (plugin/scripts/remember_spool.py) and replays it over the
-- machine-token /remember/spool route once the server is reachable again. A replay
-- must be safe to retry forever: the flush only dequeues after a confirmed write, so
-- a flush that dies between "server wrote it" and "client dropped the line" WILL
-- re-post the same intent on the next session start.
--
-- The client-generated intent id is therefore the primary key: the route claims it
-- (INSERT ... ON CONFLICT DO NOTHING) BEFORE doing any write, and a claim that finds
-- an already-'done' row short-circuits to the recorded note/episode instead of
-- writing a second note. A row left 'pending' means a prior attempt crashed mid-write;
-- that one IS re-run (reconcile_note is itself convergent — the identical hook lands
-- on the same live note as an update, not a duplicate).
--
-- Rows are small and bounded by how often memory writes happen while the server is
-- down; no retention job, same call as private_sessions.

CREATE TABLE IF NOT EXISTS remember_intents (
    intent_id    TEXT        PRIMARY KEY,
    status       TEXT        NOT NULL DEFAULT 'pending',
    hook         TEXT,
    note_id      BIGINT,
    episode_id   BIGINT,
    outcome      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT remember_intents_status_chk CHECK (status IN ('pending', 'done'))
);

-- Operator view: "what did the spool replay, and is anything stuck pending".
CREATE INDEX IF NOT EXISTS remember_intents_pending_idx
    ON remember_intents (created_at DESC)
    WHERE status = 'pending';
