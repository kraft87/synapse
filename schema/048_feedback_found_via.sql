-- 048: recall_feedback.found_via — where the missing information was eventually found.
--
-- A `missing` report alone is hindsight with no diagnosis: it can't distinguish a
-- SERVING failure (memory had it, retrieval didn't surface it — e.g. found later via
-- recall_full_turns or fetch: exactly the lexical-miss tripwire that would justify
-- SYNAPSE_RECALL_BM25_RESERVE=1) from an INGEST gap or not-memory's-job (found via
-- filesystem / live system / the user), or from a genuinely unresolved miss
-- ("nowhere" — the silent-miss class, previously invisible). One short free-text
-- token, suggested vocabulary in the tool docstring, deliberately NOT an enum so
-- new sources don't need DDL.
--
-- Idempotent: safe to re-run against an already-migrated database.

ALTER TABLE recall_feedback ADD COLUMN IF NOT EXISTS found_via TEXT;
