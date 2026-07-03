-- 011_web_artifacts.sql
-- Capture web-research tool_results (WebFetch, Exa, Firecrawl, WebSearch) as
-- first-class rows so that recall can return source material — not just the
-- 500-char excerpt episodes.content carries today.
--
-- Single table with a `kind` discriminator. Three kinds:
--   web_scrape          — single-page content (URL + markdown body)
--   search_result_set   — list of search-result items in items JSONB
--   research_job_ref    — deep_researcher_start: a job id; report arrives via
--                         deep_researcher_check (parsed as web_scrape)
--
-- `tool_use_id` from the JSONL is the natural dedup key. Upserts are
-- idempotent via ON CONFLICT DO NOTHING.
--
-- No embedding column at v1: BM25 over (title, content_markdown, query) is
-- the retrieval surface. Embedding can be added later as a nullable column
-- without breaking writers.

CREATE TABLE IF NOT EXISTS web_artifacts (
    id              BIGSERIAL PRIMARY KEY,
    kind            TEXT NOT NULL
                    CHECK (kind IN ('web_scrape', 'search_result_set', 'research_job_ref')),
    tool_name       TEXT NOT NULL,
    tool_use_id     TEXT NOT NULL UNIQUE,

    -- web_scrape fields (NULL for other kinds)
    url             TEXT,
    url_canonical   TEXT,
    content_hash    TEXT,
    title           TEXT,
    content_markdown TEXT,
    synthesized     BOOLEAN,        -- True for WebFetch (LLM-mediated answer), False for raw scrapes
    prompt          TEXT,           -- WebFetch only
    author          TEXT,
    published_at    TIMESTAMPTZ,

    -- search_result_set fields (NULL for other kinds)
    query           TEXT,
    items           JSONB,          -- list of {url, title, snippet?, published_at?, author?, position}
    item_count      INTEGER,

    -- research_job_ref fields (NULL for other kinds)
    research_id     TEXT,
    research_instructions TEXT,
    research_model  TEXT,

    -- provenance + timing
    session_id      TEXT,
    parent_episode_id BIGINT,       -- soft pointer; no FK by design (re-ingestion safety)
    jsonl_path      TEXT,
    persisted_output_path TEXT,     -- when content came from /tool-results/<id>.{txt,json}
    raw_chars       INTEGER,
    fetched_at      TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS web_artifacts_kind_idx
    ON web_artifacts (kind, fetched_at DESC);

CREATE INDEX IF NOT EXISTS web_artifacts_tool_name_idx
    ON web_artifacts (tool_name);

CREATE INDEX IF NOT EXISTS web_artifacts_url_canonical_idx
    ON web_artifacts (url_canonical)
    WHERE url_canonical IS NOT NULL;

CREATE INDEX IF NOT EXISTS web_artifacts_session_idx
    ON web_artifacts (session_id);

CREATE INDEX IF NOT EXISTS web_artifacts_fetched_at_idx
    ON web_artifacts (fetched_at DESC);

CREATE INDEX IF NOT EXISTS web_artifacts_content_hash_idx
    ON web_artifacts (content_hash)
    WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS web_artifacts_metadata_gin_idx
    ON web_artifacts USING GIN (metadata);

CREATE INDEX IF NOT EXISTS web_artifacts_items_gin_idx
    ON web_artifacts USING GIN (items);

-- BM25 over the searchable columns. paradedb's bm25 index uses a fixed
-- column set; combine title + content + query so all kinds participate.
-- Coalesce so that NULL columns (e.g. content on a search_result_set) don't
-- break indexing.
CREATE INDEX IF NOT EXISTS web_artifacts_search_idx
    ON web_artifacts
    USING bm25 (id, title, content_markdown, query, tool_name, session_id)
    WITH (key_field = 'id');
