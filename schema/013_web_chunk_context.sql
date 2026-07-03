-- 013_web_chunk_context.sql
-- Contextual Retrieval (Anthropic, Sep 2024): each chunk gets a 50-100 token
-- LLM-generated context prefix describing where it fits in the parent doc.
-- The prefix is prepended to the chunk content before embedding (and BM25
-- indexing if re-enabled). When the contextualized chunk matches the user's
-- query, retrieval recall improves materially per Anthropic's measurements
-- (49% recall-failure reduction alone, 67% with rerank).
--
-- Filled by scripts/contextualize_web_chunks.py via a Haiku call per parent
-- artifact (all chunks contextualized in one prompt to amortize doc tokens).
-- Empty string means "trivial doc" (single-chunk page, no context needed).
-- NULL means "not yet contextualized".

ALTER TABLE web_chunks
    ADD COLUMN IF NOT EXISTS context_prefix TEXT;
