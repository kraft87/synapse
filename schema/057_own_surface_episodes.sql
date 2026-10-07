-- 057_own_surface_episodes.sql
-- Provenance stamp so a restricted surface can read back what IT ingested.
--
-- 053/054 scope a restricted surface's episode reads by its project allowlist. That
-- allowlist exists to keep PERSONAL memory off a work host, but it also hid the host's
-- own conversations: a work laptop with an empty allowlist could not recall a single
-- turn it had uploaded itself. Everything such a host ingests is work by definition, so
-- its own uploads are safe to serve back to it without anyone maintaining an allowlist.
--
--   surface_id   the credential-resolved surface (054) that wrote the row. Stamped by
--                the server from the authenticated caller, NEVER from the request body.
--                NULL for every pre-057 row, for root-token/unknown callers, and for the
--                legacy self-reported hostname lane. NULL never matches, so legacy rows
--                stay exactly as reachable as they were (allowlist only).
--
-- Restricted reads become `project = ANY(allowed) OR surface_id = <caller>`; full-trust
-- reads are untouched. No backfill on purpose: provenance cannot be reconstructed
-- after the fact, and guessing it would be the one way to leak personal rows.
--
-- The stamp means "every byte of this row came from that surface". The ingest upsert
-- keeps it only while the same surface rewrites the row; a rewrite by anyone else
-- (another device, the root token) clears it, and a NULL row can never be claimed.
--
-- timeline_events carries the same column because the board digest serves it to
-- restricted surfaces under the same allowlist filter. It is copied from the source
-- episode by the extraction gate, so an event inherits exactly its turn's provenance.
--
-- The indexes are partial: only stamped rows are ever looked up by surface, and the
-- legacy corpus (all NULL) stays out of them entirely.
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the
-- columns do not exist; the code probes for them and keeps the pre-057 statements, so
-- ingest stamps nothing and a restricted surface is served its allowlist only, exactly
-- as before 057. The marker below lets the boot-time schema guard
-- (ingestion/schema_check.py) start a build on a database behind only by this file:
-- schema-check: optional
-- Every statement is guarded (IF NOT EXISTS), so a re-run is a no-op.

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS surface_id TEXT;

CREATE INDEX IF NOT EXISTS episodes_surface_idx
    ON episodes (surface_id) WHERE surface_id IS NOT NULL;

ALTER TABLE timeline_events ADD COLUMN IF NOT EXISTS surface_id TEXT;

CREATE INDEX IF NOT EXISTS timeline_events_surface_idx
    ON timeline_events (surface_id) WHERE surface_id IS NOT NULL;
