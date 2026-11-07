-- 063_web_audience.sql
-- An audience for web research, so a restricted surface can be served the work-safe part.
--
-- Until now the web bucket was trusted-hosts-only: web_artifacts carry no project and no
-- audience, and research briefs are often personal, so a restricted surface (a work
-- laptop) was served no web chunks at all. That kept personal research off untrusted
-- hardware, and it also kept every piece of plain technical research away from exactly
-- the machine doing technical work.
--
-- The filter is by SENSITIVITY, not project. Two similar projects should share the
-- research either of them paid for, so a project allowlist is the wrong key; what must
-- never reach a restricted host is personal research (health, money, relationships,
-- family, legal matters, identity, private life), whichever project asked for it.
--
--   web_artifacts.audience   the same two tiers as notes.audience (053): 'personal' or
--                            'work-safe'. NULL means not classified yet, and NULL is
--                            never served to a restricted surface. Unlike notes there is
--                            no DEFAULT 'personal': NULL keeps "nobody decided" apart
--                            from "decided personal", so the labeller knows what is left
--                            to do. Both are equally invisible to a restricted caller.
--
-- Who writes it (ingestion/web_audience.py, same precedence shape as derive_audience):
--   1. an explicit label (scripts/ingest_research_brief.py --audience),
--   2. an artifact captured in a session whose turns a restricted surface uploaded is
--      work-safe: it is already on that machine (057 provenance),
--   3. otherwise a conservative LLM classifier, run by the poller's maintenance cycle on
--      the pipeline's own provider. work-safe only on a confident yes; anything
--      uncertain or mixed is 'personal'.
--
-- No index: the restricted filter is a join predicate on a table of about a thousand
-- rows, read through web_chunks, and the poller's "what is unlabelled" scan reads the
-- same small table.
--
-- Deploy order: the image may ship before or after this file is applied. Until it is,
-- the code probes for the column (ingestion/surfaces.WEB_AUDIENCE_PROBE): restricted
-- callers keep getting no web chunks, the poller labels nothing and the brief script
-- inserts exactly as before. The marker below lets the boot-time schema guard
-- (ingestion/schema_check.py) start a build on a database behind only by this file:
-- schema-check: optional
-- Every statement is guarded, so a re-run is a no-op.

ALTER TABLE web_artifacts ADD COLUMN IF NOT EXISTS audience TEXT;

-- ADD CONSTRAINT has no IF NOT EXISTS; the DO block makes the file re-runnable.
DO $$
BEGIN
    ALTER TABLE web_artifacts ADD CONSTRAINT web_artifacts_audience_chk
        CHECK (audience IN ('personal', 'work-safe'));
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;
