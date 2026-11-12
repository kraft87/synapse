from __future__ import annotations

from typing import Any, cast

import orjson
import psycopg

from ingestion.db_connection import DatabaseConnection
from ingestion.models import Episode
from ingestion.surfaces import OWN_SURFACE_PROBE, own_surface_columns_ready
from ingestion.textsafe import strip_nul

_COLUMNS = """
    (session_id, sequence, project, platform, model,
     human_turn, assistant_turn, content, span_id, metadata, source,
     created_at{extra_col})"""

#: Every value after session_id and sequence, in column order: shared by the rewriting
#: and the appending statement so the two cannot drift apart.
_VALUES = """%(project)s, %(platform)s, %(model)s,
     %(human_turn)s, %(assistant_turn)s, %(content)s,
     %(span_id)s, %(metadata)s::jsonb, %(source)s,
     COALESCE(%(created_at)s, now()){extra_val}"""

_UPSERT_COLUMNS = (
    _COLUMNS
    + """
VALUES
    (%(session_id)s, %(sequence)s,
     """
    + _VALUES
    + """)
ON CONFLICT (session_id, sequence) DO UPDATE SET
    content        = EXCLUDED.content,
    human_turn     = EXCLUDED.human_turn,
    assistant_turn = EXCLUDED.assistant_turn,
    model          = EXCLUDED.model,
    span_id        = COALESCE(EXCLUDED.span_id, episodes.span_id),
    metadata       = EXCLUDED.metadata,
    project        = COALESCE(EXCLUDED.project, episodes.project),
    created_at     = CASE WHEN %(created_at)s IS NOT NULL
                          THEN %(created_at)s::timestamptz
                          ELSE episodes.created_at END{extra_set}
RETURNING id
"""
)

#: The append: the number is the session's highest plus one, read inside the INSERT. When
#: a writer that did not take the session lock holds that number, the INSERT waits on the
#: index, then does nothing and returns no row; the caller asks again. The other row is
#: never touched.
_APPEND_COLUMNS = (
    _COLUMNS
    + """
SELECT %(session_id)s,
    COALESCE((SELECT max(sequence) FROM episodes WHERE session_id = %(session_id)s), 0) + 1,
    """
    + _VALUES
    + """
ON CONFLICT (session_id, sequence) DO NOTHING
RETURNING id, sequence
"""
)

#: A caller that keeps losing the number to a writer outside the session lock gives up
#: after this many tries.
_APPEND_ATTEMPTS = 8

#: Appenders to one session take turns on this transaction-scoped advisory lock (the key
#: is the session id under this prefix), released when the attempt's transaction commits.
_SESSION_LOCK = "SELECT pg_advisory_xact_lock(hashtext(%s))"
_SESSION_LOCK_KEY = "episodes.session:"

#: The pre-057 statement: what runs until the provenance column exists.
_UPSERT = "INSERT INTO episodes" + _UPSERT_COLUMNS.format(extra_col="", extra_val="", extra_set="")
_APPEND = "INSERT INTO episodes" + _APPEND_COLUMNS.format(extra_col="", extra_val="")

#: Schema 057: also writes the provenance stamp (``episodes.surface_id``). A stamp means
#: "every byte of this row came from that surface", so on a rewrite it survives only when
#: the SAME surface wrote it again. A rewrite by anyone else (another device, the root
#: token, an unknown caller) clears it, because the content is no longer that surface's
#: alone, and a NULL row can never be claimed: NULL vs a stamp is DISTINCT, so the CASE
#: (no ELSE) yields NULL. Either way a rewrite can only make a row serve LESS, never more.
_UPSERT_STAMPED = "INSERT INTO episodes" + _UPSERT_COLUMNS.format(
    extra_col=", surface_id",
    extra_val=", %(surface_id)s",
    extra_set=(
        ",\n    surface_id     = CASE WHEN episodes.surface_id IS NOT DISTINCT FROM"
        " EXCLUDED.surface_id\n                          THEN episodes.surface_id END"
    ),
)
_APPEND_STAMPED = "INSERT INTO episodes" + _APPEND_COLUMNS.format(
    extra_col=", surface_id", extra_val=", %(surface_id)s"
)


class EpisodeStore(DatabaseConnection):
    def span_id_exists(self, span_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM episodes WHERE span_id = %s LIMIT 1", (span_id,)
            ).fetchone()
        return row is not None

    def content_dup_exists(self, project: str | None, content: str, cwd: str | None = None) -> bool:
        """True if an identical-content episode already exists in this project, or was
        stored from the same working directory under any label.

        Cross-session replay guard: retried sessions re-ship byte-identical turns
        under fresh session ids AND fresh span ids, which the per-session span
        index cannot catch. Byte-identical content across sessions is always a
        replay — a genuine repeat of the same user request differs in the
        assistant/tool half of the turn. The ``cwd`` leg catches a replay whose first
        copy was stored under an older label for the same folder (a subfolder or
        worktree name, a whole Windows path) until relabel_projects merges them.
        Probe is an index hit via episodes_content_md5_idx (schema 036)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM episodes WHERE md5(content) = md5(%s) "
                "AND (project IS NOT DISTINCT FROM %s OR metadata->>'cwd' = %s) LIMIT 1",
                (content, project, cwd or None),
            ).fetchone()
        return row is not None

    def is_private_session(self, session_id: str) -> bool:
        """True if this session is flagged private (schema/050) — its turns must never
        become episodes, on ANY path (live /ingest, catch-up sweep, bulk backfill).

        The one tolerated failure is a missing table: a deployment that has not applied
        schema/050 has no private sessions to honour, so it reads as not-private rather
        than blocking all ingestion. Every other error propagates — see
        ``ingestion.private_sessions.PrivateSessions`` for why this leg fails closed."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM private_sessions WHERE session_id = %s", (session_id,)
                ).fetchone()
        except psycopg.errors.UndefinedTable:
            return False
        return row is not None

    @staticmethod
    def _episode_params(ep: Episode) -> dict[str, Any]:
        # created_at is EVENT time (when the conversation happened), not ingest
        # time: the parser fills Episode.created_at from the transcript's own
        # timestamp, and dropping it here silently re-dated every imported
        # episode to import day — which then poisoned served dates, recency
        # ranking, and the KG's fact t_valid via get_episodes_valid_at. NULL
        # (no transcript ts) falls back to now(), right for live ingestion.
        #
        # strip_nul on the text/metadata fields: TEXT columns reject NUL bytes
        # and jsonb rejects the u0000 escape, so one stray byte in a transcript
        # would fail the whole INSERT.
        return {
            "session_id": ep.session_id,
            "sequence": ep.sequence,
            "project": ep.project,
            "platform": ep.platform,
            "model": ep.model,
            "human_turn": strip_nul(ep.human_turn),
            "assistant_turn": strip_nul(ep.assistant_turn),
            "content": strip_nul(ep.content),
            "span_id": ep.span_id,
            "metadata": orjson.dumps(strip_nul(ep.metadata)).decode(),
            "source": ep.source,
            "created_at": ep.created_at,
            "surface_id": ep.surface_id,
        }

    def _run_episode_sql(
        self, pre057: str, stamped: str, params: dict[str, Any], lock_session: bool = False
    ) -> dict[str, Any] | None:
        """One transaction, committed on return. Schema 057 names the provenance column:
        before it is applied the statement is exactly the pre-057 one and the stamp is
        dropped. With ``lock_session`` the transaction first takes the session's advisory
        lock, as its own statement: the INSERT's snapshot is then taken once the lock is
        granted, so it sees the row the previous holder committed."""
        try:
            with self._conn() as conn:
                if lock_session:
                    conn.execute(_SESSION_LOCK, (_SESSION_LOCK_KEY + params["session_id"],))
                sql = stamped if own_surface_columns_ready(conn, self._url) else pre057
                return cast(dict[str, Any] | None, conn.execute(sql, params).fetchone())
        except psycopg.errors.UndefinedColumn as e:
            # 057 rolled back under a running process: note it, write the pre-057 row.
            # The aborted transaction took the lock with it, so this one takes it again.
            OWN_SURFACE_PROBE.missing(self._url, e)
            with self._conn() as conn:
                if lock_session:
                    conn.execute(_SESSION_LOCK, (_SESSION_LOCK_KEY + params["session_id"],))
                return cast(dict[str, Any] | None, conn.execute(pre057, params).fetchone())

    def upsert_episode(self, ep: Episode) -> int:
        """Write the row at ``(ep.session_id, ep.sequence)``, replacing it if it exists.

        The live writers append (:meth:`append_episode`); this is for a deliberate
        rewrite of a known row."""
        row = self._run_episode_sql(_UPSERT, _UPSERT_STAMPED, self._episode_params(ep))
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def append_episode(self, ep: Episode) -> tuple[int, int]:
        """Add ``ep`` as the next row of its session and return ``(id, sequence)``.

        ``ep.sequence`` is ignored: the number is the session's highest plus one, taken
        inside the INSERT, so a writer never works one out from an earlier read. Appenders
        to one session take turns: each try is one transaction that first takes a
        per-session advisory lock and then runs the INSERT as a second statement, so the
        INSERT sees the row the previous appender committed. The INSERT also does nothing
        when the number is already held (it never rewrites that row) and is run again, up
        to a bound, for a writer that does not take the lock: ``upsert_episode`` at an
        explicit number, or an older process during a deploy. A violation of the span
        index propagates to the caller. The caller must not hold an outer ``_conn()``
        across the call, since the lock lasts to the end of the outermost transaction."""
        params = self._episode_params(ep)
        for _ in range(_APPEND_ATTEMPTS):
            row = self._run_episode_sql(_APPEND, _APPEND_STAMPED, params, lock_session=True)
            if row is not None:
                return cast(int, row["id"]), cast(int, row["sequence"])
        raise RuntimeError(
            f"append to session {ep.session_id!r} kept colliding on the sequence number "
            f"after {_APPEND_ATTEMPTS} attempts"
        )

    def get_episode(self, episode_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            result = conn.execute("SELECT * FROM episodes WHERE id = %s", (episode_id,)).fetchone()
        return cast(dict[str, Any] | None, result)

    def get_session_episodes(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT * FROM episodes WHERE session_id = %s ORDER BY sequence ASC",
                (session_id,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def get_session_span_index(self, session_id: str) -> set[str]:
        """Return the span_ids stored for a session.

        Lean companion to :meth:`get_session_episodes` for the /ingest hot path.
        The push keys turns by span_id (the stable identity — a turn's last record
        uuid): one already stored is skipped, a new one is appended. Pulls one
        column instead of full rows since that's all the dedup needs.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT span_id FROM episodes WHERE session_id = %s",
                (session_id,),
            ).fetchall()
        return {r["span_id"] for r in rows if r["span_id"]}

    def get_episodes_valid_at(self, episode_ids: list[int]) -> str | None:
        """Representative valid-time for a set of source episodes = MAX(created_at),
        the latest turn in the window (when the segment's content was actually said).

        Used as the default ``t_valid`` and the relative-date ``reference_time`` for
        facts extracted from a conversation segment, so a fact with no in-text date
        inherits the CONVERSATION timestamp instead of ingest wall-clock (``now()``).
        Correct for live ingestion (created_at ≈ now) and a real fix for backfilled /
        retro transcripts whose conversation happened in the past. Returns ISO or None.
        """
        if not episode_ids:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT max(created_at) AS m FROM episodes WHERE id = ANY(%s)",
                (list(episode_ids),),
            ).fetchone()
        m = row["m"] if row else None
        if m is None:
            return None
        return m.isoformat() if hasattr(m, "isoformat") else str(m)
