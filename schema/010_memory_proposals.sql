-- 010_memory_proposals.sql
-- Dream stage 3 mines behavioral patterns from recent dream summaries and
-- writes proposed auto-memory entries here. A client-side processor pulls
-- pending rows, surfaces them for review, and (later) applies them as memory
-- files in the agent's memory directory.

CREATE TABLE IF NOT EXISTS memory_proposals (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'posted', 'approved', 'rejected', 'applied')),
    kind TEXT NOT NULL
        CHECK (kind IN ('feedback', 'user', 'project', 'reference')),
    filename TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    body_md TEXT NOT NULL,
    evidence_dream_ids BIGINT[] NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5,
    rationale TEXT NOT NULL DEFAULT '',
    discord_channel TEXT,
    discord_message_id TEXT,
    posted_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,
    applied_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_memory_proposals_status
    ON memory_proposals (status, created_at DESC);

-- Stop the same proposal from being re-mined every dream cycle.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_proposals_filename_pending
    ON memory_proposals (filename)
    WHERE status IN ('pending', 'posted');
