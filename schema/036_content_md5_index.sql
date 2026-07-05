-- 036: cross-session content-dedup guard support.
--
-- Retried sessions (agent re-runs, session forks after compaction) re-ship
-- byte-identical turns under FRESH session ids and FRESH span ids, so the
-- per-session span guard in /ingest never sees them: 1,845 exact-copy episodes
-- (4.6% of the table) had accumulated by 2026-07-04, and each copy re-fed the
-- timeline gate. /ingest now probes for an identical-content episode in the
-- same project before inserting; this expression index makes that probe an
-- index hit instead of a seq scan.
CREATE INDEX IF NOT EXISTS episodes_content_md5_idx ON episodes (md5(content));
