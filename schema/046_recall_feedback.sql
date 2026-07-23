-- 046: recall_feedback — labeled retrieval-quality reports from agents.
--
-- After acting on a recall(), an agent files ONE row here via the recall_feedback
-- MCP tool (or POST /feedback): which served ids were load-bearing (`helpful`),
-- which were irrelevant or distracting (`noise`), what the caller needed but was
-- never served (`missing`), plus a free-form improvement idea (`note`). The id
-- arrays hold the served forms — "e:N" episode, "n:N" note, "f:<uuid>" fact,
-- "t:N" timeline, "w:N" web, "p:N" preference — validated at the tool boundary
-- so eval tooling can trust them without re-parsing.
--
-- OFFLINE labeled data only — eval goldens and reranker tuning. Deliberately NOT
-- wired into live scoring: no ranking boost, no retrieval_count bump (that stays
-- schema 006's separate implicit signal). Append-only; prune by created_at.
--
-- Idempotent: safe to re-run against an already-migrated database.

CREATE TABLE IF NOT EXISTS recall_feedback (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    query      TEXT NOT NULL,
    helpful    JSONB NOT NULL DEFAULT '[]'::jsonb,
    noise      JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing    TEXT,
    note       TEXT,
    session_id TEXT,
    project    TEXT
);

CREATE INDEX IF NOT EXISTS recall_feedback_created_idx ON recall_feedback (created_at DESC);
