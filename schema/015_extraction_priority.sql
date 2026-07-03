-- 015: priority lane for the extraction queue.
--
-- The poller drains extraction_queue FIFO. A bulk backfill (e.g. #64 — re-extract
-- the ~10.7K historical chunks) would sit AHEAD of newly-ingested turns and starve
-- normal extraction for days. Add a priority column so new ingest (default 0) always
-- drains before low-priority backfill rows (enqueued with priority = 10).
--
-- Lower number = drained first. Backward compatible: existing code ignores the column,
-- and DEFAULT 0 means every current/new ingest row keeps today's behaviour. Apply this
-- to prod BEFORE deploying the db.py ORDER BY change.

ALTER TABLE extraction_queue
    ADD COLUMN IF NOT EXISTS priority SMALLINT NOT NULL DEFAULT 0;

-- Supports the claim/get ordering: WHERE status='pending' ORDER BY priority, enqueued_at.
CREATE INDEX IF NOT EXISTS idx_extraction_queue_claim
    ON extraction_queue (priority, enqueued_at)
    WHERE status = 'pending';
