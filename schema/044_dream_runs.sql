-- 044: dream_runs — one row per nightly dream-pipeline run (issue #12, dashboard phase 4).
--
-- The nightly dream pipeline (dream/__main__.py::run_once) mines transcripts for skill +
-- config proposals. This table records ONE bookkeeping row per run so the Metrics ops page
-- ("last nightly-dream run") and the phase-5 Dream-report page can read what a run did
-- without scraping container logs.
--
-- Written fail-soft from dream/__main__.py: a row is INSERTed at run start (ok = NULL,
-- finished_at = NULL) and UPDATEd at finish with the per-lane outcome, cheap counts, a few
-- bounded samples, and any lane errors. Bookkeeping NEVER breaks the pipeline — a write
-- failure is swallowed and logged.
--
-- The jsonb columns are deliberately generous so phase 5 can drill in WITHOUT another
-- migration. Only what a lane cheaply exposes today is populated; absent keys are honestly
-- absent (the dream lanes propose; they do not extract facts, so facts_extracted /
-- superseded / dedup_merges / timeline_events stay absent until a lane reports them).
--
--   stages  {"skills": {"ran": true, "ok": true}, "config": {...}}   -- per-lane run/outcome
--   counts  {"proposals_raised": 3, "config_proposals": 2,           -- cheap aggregate counts
--            "config_corrections_found": 5, "config_sessions_scanned": 12, ...}
--   samples {"proposals": [{"id": "skill:12", "kind": "skill",       -- bounded (~10 each)
--            "name": "latency-triage"}, ...]}
--   errors  ["config lane: <str>", ...]                              -- lane error strings
--
-- Append-only; prune by started_at when it grows (a run/day, so this stays tiny).

CREATE TABLE IF NOT EXISTS dream_runs (
    id           BIGSERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    stages       JSONB NOT NULL DEFAULT '{}'::jsonb,
    counts       JSONB NOT NULL DEFAULT '{}'::jsonb,
    samples      JSONB NOT NULL DEFAULT '{}'::jsonb,
    errors       JSONB NOT NULL DEFAULT '[]'::jsonb,
    ok           BOOLEAN
);

CREATE INDEX IF NOT EXISTS dream_runs_started_idx ON dream_runs (started_at DESC);
