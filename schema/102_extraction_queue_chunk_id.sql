-- schema-check: optional
-- A chunk's queue row names the chunk, and a worker reads the chunk's text from it.
--
-- extraction_queue copied the full text of every chunk (and every turn) and keyed a chunk
-- item by that text: the enqueue dedup compared the text of every live row, the worker
-- found the chunk's source episodes by comparing it with every chunk of the session, and
-- a dashboard delete could not reach a done row, so a deleted turn's words stayed in it.
-- chunk_id replaces all three, mirroring web_chunk_id (018): enqueue is ON CONFLICT DO
-- NOTHING on the unique index, the worker reads the chunk by primary key, and deleting
-- the chunk deletes the queue row. A row that names its chunk stores no text.
--
-- The episode_id index is for the cascade in the other direction: deleting an episode
-- (the dashboard's delete) walks the foreign key from episodes, and without an index it
-- scanned the whole queue, which keeps its finished rows.
--
-- Additive, and the code probes for the column: until it exists a chunk item keeps its
-- text in the row, which is the previous behaviour, so the image can land before this is
-- applied. Pending rows enqueued before it have no chunk_id and are still found by their
-- text. The previous image never names the column, and it processes a row that has one
-- only until the new image replaces it.
--
-- On a populated queue, build the index without the write lock (the plain form below
-- briefly blocks enqueues):
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS extraction_queue_episode_idx
--     ON extraction_queue (episode_id) WHERE episode_id IS NOT NULL;
-- A failed CONCURRENTLY build leaves an INVALID index: drop it and build it again.

ALTER TABLE extraction_queue
    ADD COLUMN IF NOT EXISTS chunk_id BIGINT REFERENCES chunks(id) ON DELETE CASCADE;

CREATE UNIQUE INDEX IF NOT EXISTS extraction_queue_chunk_uq
    ON extraction_queue (chunk_id)
    WHERE chunk_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS extraction_queue_episode_idx
    ON extraction_queue (episode_id)
    WHERE episode_id IS NOT NULL;
