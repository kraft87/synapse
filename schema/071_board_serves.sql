-- 071_board_serves.sql
-- schema-check: optional
--
-- What each session start was served. GET /context renders the board (note hooks n:<id>,
-- "Last 7 days" events t:<id>) and the preferences block, and before this file nothing
-- kept which of those lines went to which session: the stored episodes do not hold the
-- injected text, so no board line could be measured. One row per /context request that
-- built the board or the preferences block, written by mcp_server/board_serves.py after
-- the reply is sent.
--
-- The ids are kept in served order, as the reply showed them. Text is not kept, only a
-- sha256 of each rendered block, except for the preferences block (at most 7 short
-- lines): a preference is merged and rewritten in place and has no history table, so its
-- id alone does not say what the line read. Note text can be read back through
-- notes_history (070), and timeline facts do not change. No foreign keys: a serve row
-- outlives the note, event or preference it names.
--
-- session_id is the harness's own session id, the value episodes.session_id holds for
-- that session's turns, sent by clients from plugin 0.24.21 as ?session=. NULL for an
-- older client. A session that is private (private_sessions, schema 050) when its board
-- is served gets no row. Claude Code serves again on resume, clear and compaction, so a
-- session can have several rows.
--
--   Was note 42 on session S's board?  SELECT 1 FROM board_serves
--                                       WHERE session_id = 'S' AND note_ids @> '{42}';
--   How many sessions was note 42 served to?
--                                       SELECT count(DISTINCT session_id) FROM board_serves
--                                       WHERE note_ids @> '{42}';
--
-- Retention: the writer deletes rows older than 180 days, at most once an hour per server
-- process. At the measured rate (tens of session starts a day) that is tens of thousands
-- of rows, so the arrays are scanned without an index.
--
-- Optional: the server runs the same without it (nothing is recorded until it exists,
-- and nothing reads it). Idempotent. Rollback: DROP TABLE board_serves.

CREATE TABLE IF NOT EXISTS board_serves (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    served_at     timestamptz NOT NULL DEFAULT now(),
    session_id    text,              -- the harness's session id (= episodes.session_id); NULL: not sent
    project       text,              -- the ?project= label the board was built for
    surface_id    text,              -- the caller's surface, from its credential only; NULL: unknown
    trust         text NOT NULL,     -- 'full' | 'restricted', the verdict the serve was filtered by
    client        text,              -- the request's User-Agent (harness and client version), cut to 120
    parts         text[] NOT NULL,   -- what was built: 'board', 'preferences'
    note_ids      bigint[] NOT NULL DEFAULT '{}',  -- n:<id> in the order the board shows them
    event_ids     bigint[] NOT NULL DEFAULT '{}',  -- t:<id>, "Last 7 days", top line first
    pref_ids      bigint[] NOT NULL DEFAULT '{}',  -- preferences.id, block order
    notes_dropped int NOT NULL DEFAULT 0,          -- notes that did not fit ("+ N notes not shown")
    board_sha256  text,              -- hex sha256 of the board text as served; NULL: not built
    board_lines   int,
    board_tokens  int,               -- chars // 4, the board's own estimate
    prefs_sha256  text,              -- of the preferences text; NULL: not built
    prefs_text    text,              -- the preferences block as served ('' when it had no line)
    ms_build      real               -- time to build the board, ms
);
CREATE INDEX IF NOT EXISTS board_serves_session_idx ON board_serves (session_id);
CREATE INDEX IF NOT EXISTS board_serves_served_at_idx ON board_serves (served_at);
