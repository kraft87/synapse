"""Shared plumbing for the dream lanes: which episodes one nightly run covers, and the
whole-session views their judges read.

WINDOW. A run covers the episodes with ``after_id < id <= until_id``. episodes.id is a
BIGSERIAL, so it follows insertion order, while created_at is the conversation's EVENT time
(ingestion/db_episodes.py): a cursor on created_at skipped every turn uploaded after the run
that followed its timestamp (a catch-up retry, a laptop offline overnight, an import).
``until_id`` is the newest id when the run starts, so a turn ingested while the run is busy
waits for the next run instead of being scanned by two. Each lane keeps its own cursor in
dream_lane_cursors and advances it only past the work it finished, so a capped run's
overflow rolls to the next night.

An id is assigned when its INSERT runs, not when it commits, so a turn whose INSERT is in
flight at the instant a run reads max(id) can commit below that run's bound and be missed.
Every episode write is one autocommit statement, so that gap is milliseconds wide, once a
night; accepted.

SESSIONS. The judges' unit is the whole session (Oracle Q2: a partial tail re-fragments the
derive signature_key across runs), so :func:`sessions` returns full views of the sessions
the window touched, oldest last touch first, up to the ``limit`` a lane judges per run. A
session already judged waits until it has REJUDGE_TURNS more user turns (dream_session_memo),
so a session open for days is not re-sent in full every night it is touched.

Until the migration that adds dream_lane_cursors and dream_session_memo is applied, a window
starts from the lane's legacy last_scan_at, nothing is stored here (the lane's own legacy
write still moves its cursor to the run start, dropping overflow as before), and no session
is skipped as already judged.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg

from dream.skills import config
from ingestion.schema_check import ColumnProbe

#: A session view: {session, first_user, user_turns, user_msgs, tools, bash_heads, fired}.
View = dict[str, Any]

PLATFORM = "claude_code"
FIRST_RUN_DAYS = 30  # a lane with no cursor yet, or a --backfill, starts this many days back
REJUDGE_TURNS = 3  # a judged session is judged again once it has this many more user turns

TOOL_MARKER = re.compile(r"^\[tool:([\w-]+)\]\s*(.*)$")
SKILL_NAME = re.compile(r"skill'?\s*[:=]\s*'?([\w-]+)")

#: The dream lane cursor migration adds dream_lane_cursors and dream_session_memo. The
#: image ships before it is applied, so every read or write of either asks this probe first.
LANE_TABLES = ColumnProbe(
    "SELECT 1 WHERE to_regclass('dream_lane_cursors') IS NOT NULL "
    "AND to_regclass('dream_session_memo') IS NOT NULL",
    "the dream lane cursor migration (dream_lane_cursors, dream_session_memo)",
)


def _tables_ready(conn: psycopg.Connection[Any]) -> bool:
    return LANE_TABLES.ready(conn, conn.info.dsn)


def _scalar(conn: psycopg.Connection[Any], sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


@dataclass(frozen=True)
class Window:
    """The episodes one lane run covers: ``after_id < episodes.id <= until_id``."""

    lane: str
    after_id: int
    until_id: int


def _id_before(conn: psycopg.Connection[Any], since: datetime) -> int | None:
    """The id just below the first episode stored with created_at after `since`: where a
    window starts when all it has is a time. None when no episode is that recent."""
    v = _scalar(conn, "SELECT min(id) - 1 FROM episodes WHERE created_at > %s", (since,))
    return None if v is None else int(v)


def _stored_cursor(conn: psycopg.Connection[Any], lane: str) -> int | None:
    if not _tables_ready(conn):
        return None
    v = _scalar(conn, "SELECT last_episode_id FROM dream_lane_cursors WHERE lane = %s", (lane,))
    return None if v is None else int(v)


def open_window(
    conn: psycopg.Connection[Any],
    lane: str,
    *,
    legacy_since: datetime | None,
    backfill: bool = False,
) -> Window:
    """`lane`'s window for this run, up to the newest episode now. It starts at the lane's
    stored cursor; without one (the first run after the migration, or before it is applied)
    at the lane's legacy last_scan_at; without that, FIRST_RUN_DAYS back. ``backfill``
    ignores both and starts FIRST_RUN_DAYS back."""
    after = None if backfill else _stored_cursor(conn, lane)
    if after is None:
        since = legacy_since if legacy_since is not None and not backfill else None
        if since is None:
            since = datetime.now(UTC) - timedelta(days=FIRST_RUN_DAYS)
        after = _id_before(conn, since)
    until = int(_scalar(conn, "SELECT COALESCE(max(id), 0) FROM episodes"))
    return Window(lane, until if after is None else min(after, until), until)


def advance(conn: psycopg.Connection[Any], window: Window, to_id: int) -> None:
    """Store `to_id` as the lane's cursor, so its next run starts after it. Never moves the
    cursor back: a --backfill run that stops short leaves a further cursor where it was.
    The caller commits. Before the migration, a no-op."""
    if not _tables_ready(conn):
        return
    conn.execute(
        "INSERT INTO dream_lane_cursors AS c (lane, last_episode_id) VALUES (%s, %s) "
        "ON CONFLICT (lane) DO UPDATE SET updated_at = now(), "
        "last_episode_id = GREATEST(c.last_episode_id, EXCLUDED.last_episode_id)",
        (window.lane, to_id),
    )


# --------------------------------------------------------------------------- sessions
def build_views(rows: Iterable[Sequence[Any]]) -> list[View]:
    """Session views from (session_id, sequence, human_turn, content) rows ordered by
    session and sequence. `content` carries the ingest markers ([user] / [tool:NAME] /
    [result] / [assistant]); `human_turn` the full user text (up to 3000 chars)."""
    sessions: dict[str, View] = {}
    for sid, _seq, human, content in rows:
        s = sessions.get(str(sid))
        if s is None:
            s = sessions[str(sid)] = {
                "session": str(sid),
                "first_user": "",
                "user_turns": 0,
                "user_msgs": [],
                "tools": Counter(),
                "bash_heads": [],
                "fired": set(),
            }
        h = (human or "").strip()
        if h:
            s["user_turns"] += 1
            if not s["first_user"]:
                s["first_user"] = h[:600]
            s["user_msgs"].append(h[:200])
        for line in (content or "").splitlines():
            m = TOOL_MARKER.match(line)
            if not m:
                continue
            name, detail = m.group(1), m.group(2)
            s["tools"][name] += 1
            if name == "Bash" and detail:
                s["bash_heads"].append(detail[:80])
            elif name == "Skill":
                sk = SKILL_NAME.search(detail)
                if sk:
                    s["fired"].add(sk.group(1))
    for s in sessions.values():
        s["tools"] = dict(s["tools"])
        s["bash_heads"] = s["bash_heads"][:25]
        s["fired"] = sorted(s["fired"])
    return list(sessions.values())


def session_views(
    conn: psycopg.Connection[Any],
    session_ids: Sequence[str],
    until_id: int,
    exclude_projects: Sequence[str] = config.EXCLUDE_PROJECTS,
) -> dict[str, View]:
    """Whole-session views of `session_ids`, from their turns stored up to `until_id`."""
    rows = conn.execute(
        "SELECT session_id, sequence, human_turn, content FROM episodes "
        "WHERE session_id = ANY(%s) AND id <= %s AND platform = %s "
        "AND (project IS NULL OR project <> ALL(%s)) ORDER BY session_id, sequence",
        (list(session_ids), until_id, PLATFORM, list(exclude_projects)),
    ).fetchall()
    return {v["session"]: v for v in build_views(rows)}


def _judged_turns(
    conn: psycopg.Connection[Any], lane: str, session_ids: list[str]
) -> dict[str, int]:
    if not session_ids or not _tables_ready(conn):
        return {}
    rows = conn.execute(
        "SELECT session_id, user_turns FROM dream_session_memo "
        "WHERE lane = %s AND session_id = ANY(%s)",
        (lane, session_ids),
    ).fetchall()
    return {str(sid): int(n) for sid, n in rows}


def sessions(
    conn: psycopg.Connection[Any],
    window: Window,
    *,
    limit: int,
    wanted: Callable[[View], bool] | None = None,
    exclude_projects: Sequence[str] = config.EXCLUDE_PROJECTS,
) -> tuple[list[View], int]:
    """Views of the sessions to judge this run, and the episode id the lane's cursor may
    advance to.

    Walks the sessions the window touched in order of their last touch, oldest first. A
    session is passed over when ``wanted`` rejects it or it has not grown by REJUDGE_TURNS
    user turns since the lane last judged it; the rest are taken, up to ``limit``. At the
    first one past the limit the walk stops, and the cursor may advance only to the last
    touch of the session before it: every session the walk passed lies at or below that
    id, and every one it did not reach still has a turn above it, so the next run picks it
    up. Views are loaded `limit` sessions at a time, only for the sessions walked."""
    if limit < 1:
        raise ValueError(f"{window.lane} lane: session limit must be at least 1, got {limit}")
    touched = [
        (str(sid), int(last_id))
        for sid, last_id in conn.execute(
            "SELECT session_id, max(id) FROM episodes "
            "WHERE id > %s AND id <= %s AND platform = %s "
            "AND (project IS NULL OR project <> ALL(%s)) "
            "GROUP BY session_id ORDER BY max(id)",
            (window.after_id, window.until_id, PLATFORM, list(exclude_projects)),
        ).fetchall()
    ]
    judged = _judged_turns(conn, window.lane, [sid for sid, _ in touched])
    picked: list[View] = []
    passed = window.after_id
    for start in range(0, len(touched), limit):
        chunk = touched[start : start + limit]
        views = session_views(conn, [sid for sid, _ in chunk], window.until_id, exclude_projects)
        for sid, last_id in chunk:
            v = views[sid]
            done = judged.get(sid)
            if (wanted is None or wanted(v)) and (
                done is None or v["user_turns"] - done >= REJUDGE_TURNS
            ):
                if len(picked) == limit:
                    return picked, passed
                picked.append(v)
            passed = last_id
    return picked, window.until_id


def mark_judged(conn: psycopg.Connection[Any], lane: str, views: Sequence[View]) -> None:
    """Record that `lane` judged `views` at their current size. The caller commits.
    Before the migration, a no-op."""
    if not views or not _tables_ready(conn):
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO dream_session_memo (lane, session_id, user_turns) VALUES (%s, %s, %s) "
            "ON CONFLICT (lane, session_id) DO UPDATE "
            "SET user_turns = EXCLUDED.user_turns, judged_at = now()",
            [(lane, v["session"], v["user_turns"]) for v in views],
        )
