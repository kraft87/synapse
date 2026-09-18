from __future__ import annotations

from typing import Any, cast

import orjson
import psycopg

from ingestion.db_connection import DatabaseConnection
from ingestion.models import Episode
from ingestion.textsafe import strip_nul


class EpisodeStore(DatabaseConnection):
    def span_id_exists(self, span_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM episodes WHERE span_id = %s LIMIT 1", (span_id,)
            ).fetchone()
        return row is not None

    def content_dup_exists(self, project: str | None, content: str) -> bool:
        """True if an identical-content episode already exists in this project.

        Cross-session replay guard: retried sessions re-ship byte-identical turns
        under fresh session ids AND fresh span ids, which the per-session span
        index cannot catch. Byte-identical content across sessions is always a
        replay — a genuine repeat of the same user request differs in the
        assistant/tool half of the turn. Probe is an index hit via
        episodes_content_md5_idx (schema 036)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM episodes WHERE md5(content) = md5(%s) "
                "AND project IS NOT DISTINCT FROM %s LIMIT 1",
                (content, project),
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

    def upsert_episode(self, ep: Episode) -> int:
        # created_at is EVENT time (when the conversation happened), not ingest
        # time: the parser fills Episode.created_at from the transcript's own
        # timestamp, and dropping it here silently re-dated every imported
        # episode to import day — which then poisoned served dates, recency
        # ranking, and the KG's fact t_valid via get_episodes_valid_at. NULL
        # (no transcript ts) falls back to now(), right for live ingestion.
        sql = """
            INSERT INTO episodes
                (session_id, sequence, project, platform, model,
                 human_turn, assistant_turn, content, span_id, metadata, source,
                 created_at)
            VALUES
                (%(session_id)s, %(sequence)s, %(project)s, %(platform)s, %(model)s,
                 %(human_turn)s, %(assistant_turn)s, %(content)s,
                 %(span_id)s, %(metadata)s::jsonb, %(source)s,
                 COALESCE(%(created_at)s, now()))
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
                                      ELSE episodes.created_at END
            RETURNING id
        """
        # strip_nul on the text/metadata fields: TEXT columns reject NUL bytes
        # and jsonb rejects the u0000 escape, so one stray byte in a transcript
        # would fail the whole INSERT.
        params = {
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
        }
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()

        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

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

    def get_session_span_index(self, session_id: str) -> tuple[set[str], int]:
        """Return (stored span_ids, max sequence) for a session.

        Lean companion to :meth:`get_session_episodes` for the /ingest hot path.
        The push keys turns by span_id (the stable identity — a turn's last record
        uuid) and appends new ones at ``max(sequence) + 1``, so a bounded-tail POST
        renumbers from the DB high-water mark instead of the parser's positional
        counter. Pulls two columns instead of full rows since that's all the dedup
        needs.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT span_id, sequence FROM episodes WHERE session_id = %s",
                (session_id,),
            ).fetchall()
        span_ids: set[str] = {r["span_id"] for r in rows if r["span_id"]}
        max_seq = max((int(r["sequence"]) for r in rows), default=0)
        return span_ids, max_seq

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
