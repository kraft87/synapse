-- schema-check: optional
-- Dream lanes: an episode-id cursor per lane, and a memo of the sessions each lane judged.
--
--   dream_lane_cursors  where each lane's next run starts: the last episodes.id it covered.
--                       episodes.id is a BIGSERIAL, so it follows insertion order; the old
--                       per-lane last_scan_at compared episodes.created_at, which is the
--                       conversation's EVENT time, so a turn uploaded after the run that
--                       followed its timestamp (a catch-up retry, an offline laptop, an
--                       import) was never mined. A run advances its cursor only past the
--                       work it finished, so a capped run's overflow rolls to the next night.
--   dream_session_memo  each session a lane judged, at how many user turns. A session is
--                       judged again only once it has grown by a few user turns, so a long
--                       session open for days is not re-sent in full every night.
--
-- Deploy order: the application ships BEFORE this file is applied. Until it is, the code
-- (dream/lane.py) probes for both tables, starts each window from the lane's legacy
-- last_scan_at, stores no cursor and skips nothing as already judged: the pre-migration
-- behaviour. The lanes keep writing last_scan_at either way, so an older image still
-- finds a current cursor. The marker on the first line lets the boot-time schema guard
-- (ingestion/schema_check.py) start a build on a database behind only by this file.
-- Every statement is guarded, so a re-run is a no-op.

CREATE TABLE IF NOT EXISTS dream_lane_cursors (
    lane            TEXT PRIMARY KEY,
    last_episode_id BIGINT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dream_session_memo (
    lane       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_turns INT NOT NULL,
    judged_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (lane, session_id)
);
