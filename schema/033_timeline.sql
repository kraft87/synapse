-- 033_timeline.sql
-- timeline_events — the EPISODIC store: an append-only, time-ordered log of dated events
-- that HAPPENED ("committed the dating fix", "decided X"), distinct from the semantic KG
-- (deduplicated bitemporal facts). Events are NEVER deduplicated — recurrence is signal —
-- and never live in the graph (a "User did everything" star node has nothing to traverse).
-- Dividing line vs the KG: states-with-duration (uses X, lives in Y) stay KG-bitemporal;
-- POINT events (shipped, decided, committed) go here. No content lives in both.
--
-- Design review (2026-07-01, opus-Oracle + Gemini): naked events (no actor field — the
-- verb carries decides-vs-executes), coarse salience, UNIQUE(source,source_ref) idempotency,
-- author-date for git, entity_refs/event_type deferred for later batch enrichment.
CREATE TABLE IF NOT EXISTS timeline_events (
    id          BIGSERIAL PRIMARY KEY,
    t_valid     TIMESTAMPTZ NOT NULL,               -- when it happened (git: AUTHOR-date)
    fact        TEXT        NOT NULL,               -- naked past-tense event text (no actor)
    source      TEXT        NOT NULL,               -- 'git:synapse' | 'chat' (doubles as provenance trust)
    source_ref  TEXT        NOT NULL,               -- sha | turn span_id (chat: hydrates the episode)
    project     TEXT,                                -- 'synapse' | 'homelab' | ...
    salience    SMALLINT    NOT NULL DEFAULT 1,      -- 0=low 1=med 2=high (coarse; tiebreaker not ranker)
    embedding   vector(2048),                        -- Voyage over "Project: X | <fact>"
    embed_model TEXT,                                -- model id, so a future re-embed is possible
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),  -- operational (the useful half of bitemporal)
    entity_refs JSONB,                               -- DEFERRED: KG entity UUIDs, batch-backfilled
    event_type  TEXT,                                -- DEFERRED: decision|action|finding|milestone
    CONSTRAINT timeline_source_ref_uniq UNIQUE (source, source_ref)  -- idempotency guard
);

CREATE INDEX IF NOT EXISTS timeline_t_valid_idx ON timeline_events (t_valid DESC);
CREATE INDEX IF NOT EXISTS timeline_project_idx ON timeline_events (project);

-- Lexical leg. Episodic queries are identifier-dense (PR #193, a SHA fragment, "N=6") where
-- short-text embeddings are weakest, so BM25 carries MORE weight here than in episode recall.
CREATE INDEX IF NOT EXISTS timeline_events_bm25 ON timeline_events
    USING bm25 (id, fact, project) WITH (key_field = 'id');

-- Vector index deferred until ~10k rows: 2048-dim exceeds vector-type HNSW's 2000 cap, so it
-- needs the halfvec expression-index pattern (see 018/#104); a time-filtered seqscan is fast
-- at prototype scale and adding the index later is non-breaking.
