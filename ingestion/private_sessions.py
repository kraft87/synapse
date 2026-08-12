"""Drop turns from a private-mode session at the ingest boundary.

Private mode is a session-scoped "off the record" toggle: the user runs
``private_mode.py on <session_id>``, which writes a local marker file (the plugin's
Stop hook then stops posting that session at all) AND inserts the session id into
``private_sessions`` (schema/050). This module is the server side of that pair — the
enforcement that does not depend on the hook having run.

It exists because the marker is not enough. The transcript stays on disk, so anything
that re-scans disk later — the SessionStart catch-up sweep, ``ingestion.backfill``, a
manual re-import months from now — would ingest those turns even though the hook
skipped them at the time. Marker = fast local skip; table = durable guarantee.

Wired exactly like ``ingestion.contamination``: one predicate applied at the same
chokepoints where a parsed turn becomes an episode (``mcp_server.server.ingest_turns``
and ``ingestion.backfill``). Dropping there means chunks, KG extraction, dream, and the
timeline never see the turn — nothing downstream needs to know private mode exists.
"""

from __future__ import annotations

from typing import Any


class PrivateSessions:
    """Per-batch memo of which session ids are private (schema/050).

    An ingest batch usually carries one session and many turns, so the lookup is
    cached per instance: one SELECT per distinct session id, not per turn. Build one
    per batch/backfill run — never cache it across runs, or a session marked private
    mid-run would keep ingesting.

    Fail-closed on error, deliberately: a lookup that raises propagates out of the
    ingest chokepoint, which fails the POST (the hook's cursor stays put and retries)
    or the backfill run. Silently ingesting turns the user believes are private is the
    unacceptable failure; a deferred turn is not. The one tolerated error is a missing
    table (a deployment that has not applied schema/050) — see
    ``Database.is_private_session``.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        self._cache: dict[str, bool] = {}

    def is_private(self, session_id: str | None) -> bool:
        """True if this session is off the record — drop the turn, do not store it."""
        if not session_id:
            return False
        hit = self._cache.get(session_id)
        if hit is None:
            hit = bool(self._db.is_private_session(session_id))
            self._cache[session_id] = hit
        return hit
