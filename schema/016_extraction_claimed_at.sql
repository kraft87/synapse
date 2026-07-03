-- 016: record WHEN a queue row was claimed.
--
-- release_stale_claims previously reset ALL 'processing' rows and only ran at
-- worker startup, so orphaned claims (from a crashed or scaled-down worker)
-- sat stuck until the next restart — and a mid-run "release all" would clobber
-- rows a peer is actively processing. With claimed_at, the maintenance loop can
-- sweep periodically and release ONLY genuinely-stale claims (claimed long ago),
-- never a live one (an active claim is younger than the per-item processing time,
-- far under the staleness threshold).
--
-- Nullable; set to now() on claim. Pre-migration rows have NULL claimed_at and
-- are treated as stale (released on the next sweep). Backward compatible.

ALTER TABLE extraction_queue
    ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;
