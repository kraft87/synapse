-- Add span_id for Logfire deduplication (defense-in-depth alongside watermark)
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS span_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS episodes_span_id_idx ON episodes (span_id) WHERE span_id IS NOT NULL;
