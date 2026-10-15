-- schema-check: optional
-- Drop the dead search_cache table and the btree indexes that another index already
-- covers. Every insert into episodes and the KG tables maintained them for nothing.
--
--   search_cache              nothing reads or writes it; only the tests' truncate kept it
--   episodes_session_idx      (session_id) is the prefix of UNIQUE (session_id, sequence)
--   episodes_project_idx      (project) is the prefix of 061's (project, created_at)
--   kg_rel_src, kg_rel_owner  prefixes of kg_rel_pair (owner_id, group_id, src_uuid, ...)
--   kg_entities_owner         prefix of kg_entities_norm (owner_id, group_id, ...)
--
-- episodes_project_idx goes only once 061's index exists and is valid: 061 is optional,
-- and dropping the old index without it would leave project lookups with none. The
-- other drops need no such check, their covering index comes from the same file that
-- made them. Indexes are dropped CONCURRENTLY, so no query waits for a table lock.
--
-- Optional, and safe for the previous image: no code names the table or these indexes, and
-- a lookup that used one of them is served by the covering index. Re-running is a no-op.

DROP TABLE IF EXISTS search_cache;

DROP INDEX CONCURRENTLY IF EXISTS episodes_session_idx;
DROP INDEX CONCURRENTLY IF EXISTS kg_rel_src;
DROP INDEX CONCURRENTLY IF EXISTS kg_rel_owner;
DROP INDEX CONCURRENTLY IF EXISTS kg_entities_owner;

SELECT 'DROP INDEX CONCURRENTLY IF EXISTS episodes_project_idx'
 WHERE EXISTS (SELECT 1 FROM pg_index
                WHERE indexrelid = to_regclass('episodes_project_created_idx') AND indisvalid)
\gexec
