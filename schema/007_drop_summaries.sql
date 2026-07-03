-- Drop the session_summaries table and related objects.
-- This table is no longer used after moving from a hierarchical retrieval
-- model to a flat, per-episode retrieval model.
DROP TABLE IF EXISTS session_summaries;

-- The functions and triggers that might have used this table are implicitly
-- removed or will be handled by the application logic being updated.
-- No other dependent objects were found in the schema files.
