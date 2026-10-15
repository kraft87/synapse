-- schema-check: optional
-- A timeline event a later turn re-told records WHICH turns re-told it.
--
-- Schema 037's confirm-merge folds a re-told happening into its existing row and bumped
-- reported_count, which nothing ever read, and which counted a retried item's merge twice
-- (the gate re-runs when the item after it fails). reported_refs is the append-only list
-- of the re-telling turns' source refs ('ep:<id>'): a retry finds its own ref already
-- there and changes nothing, and the dashboard's delete cascade, when it deletes the turn
-- the row's source_ref names, re-points the row at a surviving re-telling turn instead of
-- deleting an event another stored turn backs.
--
-- Additive, and the code probes for the column: until it exists a merge only moves
-- t_valid, which is idempotent, and the cascade deletes a re-told row with its owner turn,
-- the previous behaviour. Rows merged before this have an empty list (who re-told them was
-- never recorded). reported_count is no longer written; it stays (the previous image still
-- bumps it) until a later migration drops it.

ALTER TABLE timeline_events
    ADD COLUMN IF NOT EXISTS reported_refs TEXT[] NOT NULL DEFAULT '{}';
