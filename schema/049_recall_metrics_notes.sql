-- 049_recall_metrics_notes.sql
-- Notes leg telemetry (recall()'s new "notes" bucket — the board's searchable other
-- half). Mirrors the 039 timeline columns: per-call served count + leg latency; the
-- served n: ids ride the existing served_ids JSONB envelope.
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS n_notes  INT;
ALTER TABLE recall_metrics ADD COLUMN IF NOT EXISTS ms_notes REAL;
