-- Reinforcement signal: how many times a fact has been (re-)asserted across
-- conversations. Captured at the dedup-skip in extractor Stage 7 — when a newly
-- extracted fact is judged a duplicate of an existing edge, that edge's
-- mention_count is bumped (and its source episodes unioned) instead of the
-- assertion being silently dropped. Default 1 = asserted once (the create).
-- Forward-only: historical dedup hits are already gone. Read-side ranking boost
-- is a later phase; this migration only adds the column the write side populates.
ALTER TABLE kg_relationships ADD COLUMN IF NOT EXISTS mention_count INT NOT NULL DEFAULT 1;
