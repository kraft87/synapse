-- 037: non-destructive timeline dedup (write-time confirm-merge).
--
-- A re-told happening (same real-world event narrated again in a later turn,
-- often with a differently-resolved date) no longer lands as a second row: the
-- chat gate finds near candidates, confirms "same happening?" with an LLM that
-- reads BOTH source turns (both-orders consensus), and bumps the canonical row
-- instead of inserting. reported_count keeps the re-assertion signal the merge
-- would otherwise lose; no rows are deleted, so a false merge stays recoverable.
ALTER TABLE timeline_events ADD COLUMN IF NOT EXISTS reported_count INT NOT NULL DEFAULT 1;
