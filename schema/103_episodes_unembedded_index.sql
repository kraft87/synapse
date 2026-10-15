-- schema-check: optional
-- The poller asks every cycle which episodes still lack a vector (WHERE is_embedded =
-- FALSE). Chunks and web chunks have a partial index for that question; episodes did not,
-- so the usual answer, nothing, cost a scan of the whole table every five minutes.
--
-- Optional: the query is the same with or without the index. CONCURRENTLY, so no insert
-- waits for the build; a failed build leaves an INVALID index: drop it and run this again.

CREATE INDEX CONCURRENTLY IF NOT EXISTS episodes_unembedded_idx
    ON episodes (id)
    WHERE is_embedded = FALSE;
