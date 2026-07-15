-- 043_dash_notify.sql
-- Operator-dashboard Phase 3 (issue #12) — LISTEN/NOTIFY plumbing for the /dash/api/stream
-- SSE live feed that replaces the client's 30s polling.
--
-- Design: an AFTER INSERT row trigger on each of the three feed source tables
-- (episodes, kg_relationships, timeline_events) fires pg_notify('dash_feed', <payload>).
-- The dashboard server holds a single LISTEN connection; on each notification it
-- HYDRATES the full FeedItem (reusing the exact per-type SQL the /dash/api/feed endpoint
-- uses) and pushes it into an in-process ring buffer that the SSE route streams to clients.
--
-- Why a TINY payload (type + id only): Postgres NOTIFY has an 8KB payload cap, and a full
-- hydrated FeedItem (episode content, joined entity names, tool traces) can blow past it.
-- So the trigger sends only {"type": ..., "id": ...} — the id is enough for the server to
-- re-SELECT the row with the canonical feed SQL. This keeps the trigger cheap on the hot
-- ingest path and means the WIRE shape a client sees is produced by ONE code path (the feed
-- hydration), not duplicated in SQL here.
--
--   type      id column (JSON)      hydration source (mcp_server/dashboard_routes.py)
--   episode         episodes.id (int)         _EP_SELECT  -> _episode_item
--   fact            kg_relationships.uuid (text)  _FACT_SELECT -> _fact_item
--   timeline_event  timeline_events.id (int)  _TL_SELECT  -> _timeline_item
--
-- Cost when no client is connected: the server starts LISTEN lazily (only while an SSE
-- client is attached), so with zero listeners the NOTIFY is discarded by Postgres almost
-- for free — the trigger is a single json_build_object + pg_notify per inserted row.
--
-- Idempotent: CREATE OR REPLACE the function, DROP ... IF EXISTS then CREATE each trigger,
-- so a re-run against an already-migrated database is a clean no-op-equivalent.

CREATE OR REPLACE FUNCTION dash_notify_feed() RETURNS trigger AS $$
DECLARE
    payload text;
BEGIN
    IF TG_TABLE_NAME = 'episodes' THEN
        payload := json_build_object('type', 'episode', 'id', NEW.id)::text;
    ELSIF TG_TABLE_NAME = 'kg_relationships' THEN
        payload := json_build_object('type', 'fact', 'id', NEW.uuid)::text;
    ELSIF TG_TABLE_NAME = 'timeline_events' THEN
        payload := json_build_object('type', 'timeline_event', 'id', NEW.id)::text;
    ELSE
        RETURN NULL;  -- defensive: never armed on any other table
    END IF;
    PERFORM pg_notify('dash_feed', payload);
    RETURN NULL;  -- AFTER trigger: return value is ignored
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS dash_notify_episodes ON episodes;
CREATE TRIGGER dash_notify_episodes
    AFTER INSERT ON episodes
    FOR EACH ROW EXECUTE FUNCTION dash_notify_feed();

DROP TRIGGER IF EXISTS dash_notify_relationships ON kg_relationships;
CREATE TRIGGER dash_notify_relationships
    AFTER INSERT ON kg_relationships
    FOR EACH ROW EXECUTE FUNCTION dash_notify_feed();

DROP TRIGGER IF EXISTS dash_notify_timeline ON timeline_events;
CREATE TRIGGER dash_notify_timeline
    AFTER INSERT ON timeline_events
    FOR EACH ROW EXECUTE FUNCTION dash_notify_feed();
