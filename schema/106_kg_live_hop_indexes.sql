-- schema-check: optional
-- Live edges of an entity, newest first: the KG leg's graph hop asks for the 8 most
-- recently valid live edges of each seed entity, in either direction. With only the
-- (owner, group, endpoint) btrees to go on, an entity with thousands of edges (the User
-- node) is read whole and sorted on every recall; with these partial indexes each side
-- reads its 8 rows in order and stops. They also serve the seed gate's "has a live edge?"
-- probe.
--
-- Optional: the hop query is the same, and returns the same rows, with or without them.
-- Built CONCURRENTLY, so applying this file never blocks fact writes. A failed
-- CONCURRENTLY build leaves an INVALID index: drop it and run this file again.

CREATE INDEX CONCURRENTLY IF NOT EXISTS kg_rel_src_live
    ON kg_relationships (owner_id, group_id, src_uuid, t_valid DESC)
    WHERE t_invalid IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS kg_rel_tgt_live
    ON kg_relationships (owner_id, group_id, tgt_uuid, t_valid DESC)
    WHERE t_invalid IS NULL;
