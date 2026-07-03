-- Synthesized documents: segment summaries (ingestion-time) and dream documents (nightly).
-- Both are first-class search targets alongside episodes and chunks.

DO $$ BEGIN
    CREATE TYPE synth_doc_type AS ENUM ('summary', 'dream');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS synth_documents (
    id                  SERIAL PRIMARY KEY,
    doc_type            synth_doc_type NOT NULL,
    session_id          TEXT,                    -- owning session (for summaries)
    project             TEXT,
    start_sequence      INTEGER,                 -- first episode seq in segment (summaries)
    end_sequence        INTEGER,                 -- last episode seq in segment (summaries)
    source_ids          JSONB,                   -- episode IDs or summary IDs that produced this
    constituent_hash    TEXT UNIQUE,             -- hash(source_ids) — prevents duplicate generation
    content             TEXT NOT NULL,
    embedding           vector(2048),
    is_embedded         BOOLEAN NOT NULL DEFAULT FALSE,
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS synth_docs_session_idx   ON synth_documents (session_id);
CREATE INDEX IF NOT EXISTS synth_docs_project_idx   ON synth_documents (project);
CREATE INDEX IF NOT EXISTS synth_docs_type_idx      ON synth_documents (doc_type);
CREATE INDEX IF NOT EXISTS synth_docs_embedded_idx  ON synth_documents (is_embedded) WHERE is_embedded = FALSE;

-- BM25 index for keyword search
CREATE INDEX IF NOT EXISTS synth_docs_search_idx ON synth_documents
    USING bm25 (id, content, session_id, project)
    WITH (key_field = 'id');
