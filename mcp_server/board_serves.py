"""What each session start was served: one ``board_serves`` row per ``GET /context``.

The board and the preferences block are injected into every session, and the episodes
stored from that session do not contain them, so without a record no board line can be
measured. ``Serve`` is what one reply showed: the note, event and preference ids in the
order the text shows them, a hash of each block, its size, and who it went to (the
session id the client sent, the project, the surface its credential resolved to, its
User-Agent). Schema 071 holds it.

The write is FAIL-SOFT and off the request path: the route hands ``record`` to the
response as a background task, so it runs after the reply is sent, and any error is
logged at debug and dropped. A database without schema 071 records nothing (a cached
probe, so no failed INSERT per session start). No embedding or LLM call.

Private mode: a session in ``private_sessions`` (schema 050) when its board is served
gets no row; the check is in the INSERT itself, so it cannot be skipped by a failure.
A session made private later keeps the rows written before, as its earlier turns stay
stored.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ingestion import pg_pool
from ingestion.schema_check import ColumnProbe
from ingestion.surfaces import SurfaceTrust

logger = logging.getLogger(__name__)

TABLE_READY = ColumnProbe(
    "SELECT 1 WHERE to_regclass('board_serves') IS NOT NULL",
    "schema 071 (board_serves)",
)

#: A session id longer than this, or with a control character, is not recorded (the
#: row is written without one). The private-session routes take ids up to the same size.
MAX_SESSION_LEN = 200
_MAX_CLIENT_LEN = 120
_MAX_PROJECT_LEN = 200

#: Rows older than this are deleted by the writer, at most once per _PRUNE_EVERY_S.
RETENTION_DAYS = 180
_PRUNE_EVERY_S = 3600.0
_pruned_at: dict[str, float] = {}
_prune_lock = threading.Lock()

_INSERT = """
INSERT INTO board_serves (
    session_id, project, surface_id, trust, client, parts, note_ids, event_ids, pref_ids,
    notes_dropped, board_sha256, board_lines, board_tokens, prefs_sha256, prefs_text, ms_build
)
SELECT %(session_id)s, %(project)s, %(surface_id)s, %(trust)s, %(client)s, %(parts)s,
       %(note_ids)s, %(event_ids)s, %(pref_ids)s, %(notes_dropped)s, %(board_sha256)s,
       %(board_lines)s, %(board_tokens)s, %(prefs_sha256)s, %(prefs_text)s, %(ms_build)s
WHERE %(session_id)s::text IS NULL
   OR NOT EXISTS (SELECT 1 FROM private_sessions WHERE session_id = %(session_id)s::text)
"""


def session_param(raw: str | None) -> str | None:
    """The ``?session=`` value to record, or None: absent, empty, longer than
    ``MAX_SESSION_LEN`` or holding a control character."""
    if not raw or len(raw) > MAX_SESSION_LEN:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in raw):
        return None
    return raw


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def note_ids_in_board_order(notes: Iterable[Mapping[str, Any]], order: Iterable[str]) -> list[int]:
    """The ids of ``notes`` in the order the board shows them: section by section in
    ``order`` (board._render's), each section in list order. A note of a type not in
    ``order`` is not shown, so it is not listed."""
    by_type: dict[str, list[int]] = {}
    for n in notes:
        by_type.setdefault(str(n["type"]), []).append(int(n["id"]))
    return [i for t in order for i in by_type.get(t, [])]


@dataclass
class Serve:
    """What one ``/context`` reply showed, and to whom."""

    trust: SurfaceTrust
    project: str | None = None
    session_id: str | None = None
    client: str | None = None
    parts: list[str] = field(default_factory=list)
    board_text: str | None = None
    note_ids: list[int] = field(default_factory=list)
    event_ids: list[int] = field(default_factory=list)
    notes_dropped: int = 0
    ms_build: float | None = None
    prefs_text: str | None = None
    pref_ids: list[int] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        """The INSERT's parameters."""
        board = self.board_text
        prefs = self.prefs_text
        return {
            "session_id": self.session_id,
            "project": (self.project or None) and self.project[:_MAX_PROJECT_LEN],
            # Only a server-established identity names a surface; anything else is NULL.
            "surface_id": self.trust.surface_id if self.trust.credential_bound else None,
            "trust": self.trust.trust,
            "client": (self.client or None) and self.client[:_MAX_CLIENT_LEN],
            "parts": list(self.parts),
            "note_ids": list(self.note_ids),
            "event_ids": list(self.event_ids),
            "pref_ids": list(self.pref_ids),
            "notes_dropped": int(self.notes_dropped or 0),
            "board_sha256": None if board is None else _sha256(board),
            "board_lines": None if board is None else (board.count("\n") + 1 if board else 0),
            "board_tokens": None if board is None else len(board) // 4,
            "prefs_sha256": None if prefs is None else _sha256(prefs),
            "prefs_text": prefs,
            "ms_build": None if self.ms_build is None else round(self.ms_build, 2),
        }


def _maybe_prune(conn: Any, db_url: str) -> None:
    now = time.monotonic()
    with _prune_lock:
        last = _pruned_at.get(db_url)
        if last is not None and now - last < _PRUNE_EVERY_S:
            return
        _pruned_at[db_url] = now
    conn.execute(
        "DELETE FROM board_serves WHERE served_at < now() - make_interval(days => %s)",
        (RETENTION_DAYS,),
    )


def record(db_url: str, serve: Serve) -> None:
    """Write ``serve``'s row, unless the table is absent or the session is private, and
    prune old rows now and then. Never raises."""
    try:
        if not serve.parts:
            return
        with pg_pool.connection(db_url) as conn:
            if not TABLE_READY.ready(conn, db_url):
                return
            conn.execute(_INSERT, serve.row())
            _maybe_prune(conn, db_url)
    except Exception as e:
        logger.debug("board_serves write failed: %s", e)
