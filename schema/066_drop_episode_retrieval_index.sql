-- schema-check: optional
-- Let a retrieval_count bump be a HOT update.
--
-- fetch(), fetch_session() and recall_full_turns() add 1 to episodes.retrieval_count for
-- every episode they serve. The column carried an index nothing reads (006: no query
-- orders or filters by it; merge_rrf reads the value off rows it already has), and an
-- indexed column cannot be updated in place: each bump wrote a new row version plus an
-- entry in every index of the table (HNSW, BM25, GIN, the btrees). With the index gone the
-- bump can stay on its page, given room there: new pages keep 10% free from now on (the
-- ALTER only sets the target; existing pages stay as full as they are until a table
-- rewrite, so the effect builds up as episodes are written).
--
-- Optional, and safe for the previous image: nothing in any image reads the index, and the
-- ALTER is a metadata change that blocks neither reads nor writes. The index is dropped
-- CONCURRENTLY, so no query waits for a table lock. Re-running is a no-op.

DROP INDEX CONCURRENTLY IF EXISTS idx_episodes_retrieval_count;

ALTER TABLE episodes SET (fillfactor = 90);
