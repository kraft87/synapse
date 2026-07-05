-- 039: recall_metrics — record WHICH results were served, not just how many (issue #10).
--
-- served_ids lets us analyze what actually gets served per query: feeds the
-- episode-validity overlay and retrieval-quality debugging. Shape:
--   {"episodes": ["e:123", ...], "facts": ["<edge-uuid>", ...],
--    "timeline": [<event-id>, ...], "prefs": [<pref-id>, ...]}
-- recall_episodes rows carry only the episodes key.
--
-- Also adds the timeline/prefs telemetry columns the writer has been passing
-- since the 033/035 legs shipped — they were silently dropped because
-- _do_record() inserts only _METRIC_COLS, which never gained them.

ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS n_timeline  INT;
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS ms_timeline REAL;
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS n_prefs     INT;
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS ms_prefs    REAL;
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS served_ids  JSONB;
