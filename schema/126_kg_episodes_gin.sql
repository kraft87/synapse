-- schema-check: optional
-- GIN on kg_relationships.episodes, for the "facts that cite episode N" lookup.
--
-- Three paths ask `episodes @> [N]` (both id shapes, number and text) and, with no index
-- that can serve it, each reads the whole facts table: 056's relabel trigger once per
-- relabeled episode (a prod relabel of tens of thousands of episodes is that many full
-- scans, inside long transactions that hold row locks), and the dashboard's episode
-- drill-down and episode delete. 029's index is partial (superseded facts only) and
-- cannot serve them.
--
-- Built CONCURRENTLY, so applying this file never blocks fact writes. A failed
-- CONCURRENTLY build leaves an INVALID index: drop it and run this file again.
--
-- Optional: nothing reads the index by name; every lookup above is correct without it
-- and only slower, so the image can land before this is applied.

CREATE INDEX CONCURRENTLY IF NOT EXISTS kg_rel_episodes_gin
    ON kg_relationships USING gin (episodes jsonb_path_ops);
