from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

import orjson
import psycopg
from psycopg.rows import dict_row

from ingestion.embedding import embed_dims
from ingestion.models import Episode, ExtractionItem

# Embedding width for the vector casts below — matches the provisioned schema.
# Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()


class Database:
    def __init__(self, url: str) -> None:
        self._url = url
        # Simple synchronous connection — ONE thread per instance. The commit/rollback
        # in _conn() spans the whole connection, so concurrent users would entangle
        # transactions; the poller's concurrent drain gives each worker thread its own
        # Database (see Poller._thread_worker) rather than sharing this one.
        self._connection: psycopg.Connection[Any] | None = None

    @contextmanager
    def _conn(self) -> Generator[psycopg.Connection[Any], None, None]:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self._url, row_factory=dict_row, autocommit=False)
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def close(self) -> None:
        if self._connection and not self._connection.closed:
            self._connection.close()

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

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
        params = {
            "session_id": ep.session_id,
            "sequence": ep.sequence,
            "project": ep.project,
            "platform": ep.platform,
            "model": ep.model,
            "human_turn": ep.human_turn,
            "assistant_turn": ep.assistant_turn,
            "content": ep.content,
            "span_id": ep.span_id,
            "metadata": orjson.dumps(ep.metadata).decode(),
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

    def insert_timeline_event(
        self,
        *,
        t_valid: str,
        fact: str,
        source: str,
        source_ref: str,
        project: str | None,
        salience: int,
        embedding: list[float] | None,
        embed_model: str | None,
        event_type: str | None = None,
        domain: str | None = None,
    ) -> int:
        """Append one event to the episodic timeline (schema 033). Idempotent on
        UNIQUE(source, source_ref) — re-processing a turn never duplicates. Returns
        rows inserted (0 = already present)."""
        vlit = (
            "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding is not None else None
        )
        with self._conn() as conn:
            return conn.execute(
                "INSERT INTO timeline_events "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(t_valid, fact, source, source_ref, project, salience, embedding, embed_model, "
                " event_type, domain) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s,%s) "
                "ON CONFLICT (source, source_ref) DO NOTHING",
                (
                    t_valid,
                    fact,
                    source,
                    source_ref,
                    project,
                    salience,
                    vlit,
                    embed_model if embedding is not None else None,
                    event_type,
                    domain,
                ),
            ).rowcount

    def timeline_near_candidates(
        self,
        embedding: list[float],
        project: str | None,
        t_valid: str,
        exclude_episode_ref: str,
        window_days: int = 14,
        max_dist: float = 0.20,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Nearest same-project chat events within ±window_days of ``t_valid`` and
        under ``max_dist`` cosine distance — the dedup confirm call's candidate pool.
        Excludes the new event's own turn (the ``ep:<id>`` base ref and its ``#k``
        siblings): a turn's multiple events are intentionally distinct, never dups."""
        vlit = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, fact, t_valid, source_ref, "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                f"       (embedding <=> %s::vector({_EMBED_DIMS})) AS dist "
                "FROM timeline_events "
                "WHERE source = 'chat' AND project IS NOT DISTINCT FROM %s "
                "AND embedding IS NOT NULL "
                "AND source_ref != %s AND source_ref NOT LIKE %s "
                "AND t_valid BETWEEN %s::timestamptz - make_interval(days => %s) "
                "                AND %s::timestamptz + make_interval(days => %s) "
                f"AND (embedding <=> %s::vector({_EMBED_DIMS})) < %s "
                "ORDER BY dist LIMIT %s",
                (
                    vlit,
                    project,
                    exclude_episode_ref,
                    exclude_episode_ref + "#%",
                    t_valid,
                    window_days,
                    t_valid,
                    window_days,
                    vlit,
                    max_dist,
                    limit,
                ),
            ).fetchall()
        return cast(list[dict[str, Any]], rows)

    def bump_timeline_reported(self, event_id: int, t_valid: str) -> None:
        """Record a re-assertion of an existing timeline event (dedup merge outcome):
        increment reported_count and keep the EARLIEST t_valid — the canonical date of
        a date-split re-telling is the first-resolved one, and a re-tell that resolves
        an earlier true date corrects the row."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE timeline_events SET reported_count = reported_count + 1, "
                "t_valid = LEAST(t_valid, %s::timestamptz) WHERE id = %s",
                (t_valid, event_id),
            )

    def timeline_ident_exists(
        self, idents: list[str], project: str | None, t_valid: str, window_hours: int
    ) -> bool:
        """True if any timeline event in the project/time window already carries one of
        these identifiers (PR ref / SHA) in its fact text. The write-time cross-source
        dedup key — exact identifier match, deliberately NOT embedding similarity."""
        if not idents:
            return False
        pats = ["%" + i + "%" for i in idents]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM timeline_events "
                "WHERE (project = %s OR %s::text IS NULL) "
                "AND t_valid BETWEEN %s::timestamptz - make_interval(hours => %s) "
                "                AND %s::timestamptz + make_interval(hours => %s) "
                "AND lower(fact) LIKE ANY(%s) LIMIT 1",
                (project, project, t_valid, window_hours, t_valid, window_hours, pats),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Preferences (schema 035) — the standing USER-preference store. Kept out of
    # the KG (every pref hangs off the User node → supernode); a flat time-log with
    # dedup-by-reassertion + supersession instead. See ingestion/preferences_gate.py.
    # ------------------------------------------------------------------

    def find_live_preferences(
        self, owner_id: str, group_id: str, embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Live preferences for one owner/group, nearest to ``embedding`` by cosine.
        Returns ``[{id, pref, polarity, sim}]`` in DESCENDING similarity (sim = 1 -
        cosine_distance in [0, 2]). The gate's dedup/supersession decision reads this."""
        vlit = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, pref, polarity, 1 - (embedding <=> %s::vector({_EMBED_DIMS})) AS sim "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM preferences "
                "WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
                "AND embedding IS NOT NULL "
                f"ORDER BY embedding <=> %s::vector({_EMBED_DIMS}) ASC LIMIT %s",
                (vlit, owner_id, group_id, vlit, limit),
            ).fetchall()
        return [
            {"id": r["id"], "pref": r["pref"], "polarity": r["polarity"], "sim": float(r["sim"])}
            for r in rows
        ]

    def insert_preference(
        self,
        *,
        owner_id: str,
        group_id: str,
        project: str | None,
        pref: str,
        polarity: str,
        embedding: list[float] | None,
        embed_model: str | None,
        source_ref: str | None,
    ) -> int:
        """Append one live preference (assert_count=1). Returns the new row id."""
        vlit = (
            "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding is not None else None
        )
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO preferences "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(owner_id, group_id, project, pref, polarity, embedding, embed_model, source_ref) "
                f"VALUES (%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s) RETURNING id",
                (
                    owner_id,
                    group_id,
                    project,
                    pref,
                    polarity,
                    vlit,
                    embed_model if embedding is not None else None,
                    source_ref,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def reassert_preference(self, pref_id: int) -> None:
        """A restated preference: bump assert_count + last_asserted, keep the older text
        (the first phrasing is the anchor; recurrence is the strength signal)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE preferences "
                "SET assert_count = assert_count + 1, last_asserted = now() WHERE id = %s",
                (pref_id,),
            )

    def supersede_preference(self, old_id: int, new_id: int) -> None:
        """A contradicting preference won: retire the old row (t_invalid=now, superseded_by
        = the new row) so the live set carries only the current stance, auditably linked."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE preferences SET t_invalid = now(), superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )

    def top_preferences(self, owner_id: str, limit: int = 8) -> list[dict[str, Any]]:
        """Live preferences for the session-start block: strongest first — most-reasserted,
        then most-recent. Across ALL groups (a standing preference shapes every session)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pref, polarity, assert_count, left(first_seen::text, 10) AS since "
                "FROM preferences WHERE owner_id = %s AND t_invalid IS NULL "
                "ORDER BY assert_count DESC, last_asserted DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Notes (schema 041) — the EXPLICIT-memory store behind the always-injected
    # board. Kept out of the KG (same supernode rationale as preferences/timeline)
    # and out of episodes (episodes are the archive; notes are the index). Live set
    # = superseded_by IS NULL; contradictions supersede (lineage), restatements
    # UPDATE in place (updated_at bump). See ingestion/notes.py.
    # ------------------------------------------------------------------

    def find_live_notes(
        self, owner_id: str, group_id: str, embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Live notes for one owner/group, nearest to ``embedding`` by cosine (over the
        HOOK — the embed target). Returns ``[{id, hook, body, type, project, sim}]`` in
        DESCENDING similarity. The reconcile path's dedup/supersession decision reads this."""
        vlit = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, hook, body, type, project, 1 - (embedding <=> %s::halfvec({_EMBED_DIMS})) AS sim "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM notes "
                "WHERE owner_id = %s AND group_id = %s AND superseded_by IS NULL "
                "AND embedding IS NOT NULL "
                f"ORDER BY embedding <=> %s::halfvec({_EMBED_DIMS}) ASC LIMIT %s",
                (vlit, owner_id, group_id, vlit, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "hook": r["hook"],
                "body": r["body"],
                "type": r["type"],
                "project": r["project"],
                "sim": float(r["sim"]),
            }
            for r in rows
        ]

    def insert_note(
        self,
        *,
        owner_id: str,
        group_id: str,
        project: str | None,
        type: str,
        hook: str,
        body: str,
        embedding: list[float] | None,
        embed_model: str | None,
        source_ref: str | None,
    ) -> int:
        """Append one live note. Returns the new row id. NULL embedding is allowed
        (keyless dev/test; dedup KNN simply skips such rows)."""
        vlit = (
            "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding is not None else None
        )
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO notes "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(owner_id, group_id, project, type, hook, body, embedding, embed_model, source_ref) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::halfvec({_EMBED_DIMS}),%s,%s) RETURNING id",
                (
                    owner_id,
                    group_id,
                    project,
                    type,
                    hook,
                    body,
                    vlit,
                    embed_model if embedding is not None else None,
                    source_ref,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def update_note(
        self,
        note_id: int,
        *,
        hook: str,
        body: str,
        embedding: list[float] | None,
        embed_model: str | None,
    ) -> None:
        """A restated note: refresh hook/body/embedding in place and bump updated_at
        (the note keeps its id — the board line just gets the newer phrasing)."""
        vlit = (
            "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding is not None else None
        )
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                f"SET hook = %s, body = %s, embedding = %s::halfvec({_EMBED_DIMS}), "
                "embed_model = %s, updated_at = now() WHERE id = %s",
                (hook, body, vlit, embed_model if embedding is not None else None, note_id),
            )

    def supersede_note(self, old_id: int, new_id: int) -> None:
        """A contradicting note won: retire the old row (superseded_by = the new row) so
        the live set carries only the current statement, auditably linked."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )

    def list_board_notes(self, owner_id: str, project: str | None) -> list[dict[str, Any]]:
        """Live notes for the board: every global-scope type (user/feedback/reference)
        plus the current project's project-notes. Ordered feedback -> user -> project ->
        reference, newest-updated first within each type. ``project=None`` serves the
        global set only (``project = NULL`` matches nothing)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, hook, type, project, updated_at FROM notes "
                "WHERE owner_id = %s AND superseded_by IS NULL "
                "AND (type IN ('user','feedback','reference') OR project = %s) "
                "ORDER BY CASE type WHEN 'feedback' THEN 0 WHEN 'user' THEN 1 "
                "WHEN 'project' THEN 2 ELSE 3 END, updated_at DESC",
                (owner_id, project),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_notes_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        """Fetch note bodies by id — the on-demand half of the board (hook on the board,
        body behind the id). Silently drops unknown ids."""
        if not ids:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, hook, body, type, project, updated_at FROM notes "
                "WHERE id = ANY(%s) ORDER BY id",
                (ids,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_note_by_source_ref(self, source_ref: str) -> dict[str, Any] | None:
        """Newest note carrying this provenance ref, live or retired — the seed
        importer's idempotency probe (re-imports must not duplicate)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, hook, body, type, project, superseded_by, updated_at "
                "FROM notes WHERE source_ref = %s ORDER BY id DESC LIMIT 1",
                (source_ref,),
            ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Recall feedback (schema 046)
    # ------------------------------------------------------------------

    def insert_recall_feedback(
        self,
        *,
        query: str,
        helpful: list[str],
        noise: list[str],
        missing: str | None,
        note: str | None,
        session_id: str | None,
        project: str | None,
    ) -> int:
        """One labeled retrieval-quality report (the recall_feedback tool).

        Offline data only — never read by live ranking; ids are pre-validated
        served forms ("e:N" / "n:N") at the tool boundary."""
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO recall_feedback "
                "    (query, helpful, noise, missing, note, session_id, project) "
                "VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s) RETURNING id",
                (
                    query,
                    orjson.dumps(helpful).decode(),
                    orjson.dumps(noise).decode(),
                    missing,
                    note,
                    session_id,
                    project,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def get_unembedded_episodes(self, limit: int = 96) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, content FROM episodes WHERE is_embedded = FALSE LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def set_episode_embedding(self, episode_id: int, embedding: list[float]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE episodes SET embedding = %s::vector, is_embedded = TRUE WHERE id = %s",
                (embedding, episode_id),
            )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def upsert_chunk(
        self,
        session_id: str,
        start_sequence: int,
        end_sequence: int,
        episode_ids: list[int],
        content: str,
        project: str | None,
    ) -> None:
        """Insert a chunk; skip if this (session, start, end) range already exists."""
        import orjson

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chunks
                    (session_id, start_sequence, end_sequence, episode_ids, content, project)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (session_id, start_sequence, end_sequence) DO NOTHING
                """,
                (
                    session_id,
                    start_sequence,
                    end_sequence,
                    orjson.dumps(episode_ids).decode(),
                    content,
                    project,
                ),
            )

    def get_unembedded_chunks(self, limit: int = 96) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, content FROM chunks WHERE is_embedded = FALSE LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def set_chunk_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE chunks SET embedding = %s::vector, is_embedded = TRUE WHERE id = %s",
                (embedding, chunk_id),
            )

    # ------------------------------------------------------------------
    # Synth documents (segment summaries + dream documents)
    # ------------------------------------------------------------------

    def upsert_synth_document(
        self,
        doc_type: str,
        content: str,
        constituent_hash: str,
        session_id: str | None = None,
        project: str | None = None,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        source_ids: list[int] | None = None,
    ) -> int | None:
        """Insert a synth document; skip if constituent_hash already exists. Returns id or None."""
        import orjson

        sql = """
            INSERT INTO synth_documents
                (doc_type, session_id, project, start_sequence, end_sequence,
                 source_ids, constituent_hash, content)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (constituent_hash) DO NOTHING
            RETURNING id
        """
        with self._conn() as conn:
            row = conn.execute(
                sql,
                (
                    doc_type,
                    session_id,
                    project,
                    start_sequence,
                    end_sequence,
                    orjson.dumps(source_ids or []).decode(),
                    constituent_hash,
                    content,
                ),
            ).fetchone()
        return cast(int | None, row["id"] if row else None)

    def sessions_with_pending_segments(self, every_n: int = 25) -> list[str]:
        """Return session IDs that have at least one un-summarized segment.

        A session has pending work when its episode count divided by every_n
        exceeds its existing summary count (one summary per complete window).
        Replaces a per-session fan-out scan that issued ~2 queries against
        every distinct session, even when no work was outstanding.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                WITH ep_counts AS (
                    SELECT session_id, COUNT(*) AS n FROM episodes GROUP BY session_id
                ),
                sum_counts AS (
                    SELECT session_id, COUNT(*) AS n FROM synth_documents
                    WHERE doc_type = 'summary' GROUP BY session_id
                )
                SELECT e.session_id
                FROM ep_counts e
                LEFT JOIN sum_counts s USING (session_id)
                WHERE (e.n / %s) > COALESCE(s.n, 0)
                """,
                (every_n,),
            ).fetchall()
        return [cast(str, r["session_id"]) for r in rows]

    def sessions_with_pending_chunks(self) -> list[str]:
        """Return session IDs that have a NEW complete chunk window available.

        Pending when either (a) the session has no chunks yet but >= 4 episodes
        (enough for one window), or (b) its max episode sequence is at least 3
        past its chunks' max ``end_sequence`` — i.e. enough new episodes arrived
        to form the next complete window (window=4, step=3, so each new window
        needs ``step`` more episodes). The 4/3 mirror ``ingestion.chunks``; kept
        inline so this stays one pre-filter query. A session with only a 1-2
        episode incomplete tail is NOT flagged (avoids a no-op rebuild every
        cycle). Mirrors ``sessions_with_pending_segments``.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                WITH ep AS (
                    SELECT session_id, COUNT(*) AS c, MAX(sequence) AS m
                    FROM episodes GROUP BY session_id
                ),
                ch AS (
                    SELECT session_id, MAX(end_sequence) AS m
                    FROM chunks GROUP BY session_id
                )
                SELECT e.session_id
                FROM ep e
                LEFT JOIN ch c USING (session_id)
                WHERE (c.m IS NULL AND e.c >= 4)
                   OR (c.m IS NOT NULL AND e.m >= c.m + 3)
                """
            ).fetchall()
        return [cast(str, r["session_id"]) for r in rows]

    def get_chunk_ranges(self, session_id: str) -> set[tuple[int, int]]:
        """Return {(start_sequence, end_sequence)} of a session's existing chunks.

        Lets ``ingestion.chunks.rebuild_chunks`` skip windows already present —
        no wasted upsert, no re-embedding, and an accurate new-chunk count.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT start_sequence, end_sequence FROM chunks WHERE session_id = %s",
                (session_id,),
            ).fetchall()
        return {(int(r["start_sequence"]), int(r["end_sequence"])) for r in rows}

    def get_chunk_episode_ids(self, session_id: str, content: str) -> list[int]:
        """Return the episode_ids of the chunk with this exact content (for edge backlink).

        Chunk extraction (task #63) enqueues a chunk's text for KG extraction; when its
        facts are written the edges must trace back to the episodes the chunk was built
        from. Chunk content is the episodes joined verbatim, so (session_id, content) is a
        stable key. Returns [] if no match (chunk since rebuilt/removed).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT episode_ids FROM chunks WHERE session_id = %s AND content = %s LIMIT 1",
                (session_id, content),
            ).fetchone()
        if not row or row["episode_ids"] is None:
            return []
        raw = row["episode_ids"]
        if isinstance(raw, str):
            raw = orjson.loads(raw)
        return [int(x) for x in raw]

    def get_web_chunk_provenance(self, web_chunk_id: int) -> dict[str, Any] | None:
        """Source metadata for a web chunk's parent artifact (task #68).

        Web-chunk extraction needs the page's identity (url/title), trust level
        (synthesized: LLM-mediated answer vs raw scrape), and dates (published_at
        falls back to fetched_at as the default t_valid for facts whose text
        carries no date of its own). Returns None if the chunk vanished
        (artifact deleted; ON DELETE CASCADE removed the chunk).
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT a.id AS web_artifact_id, a.url, a.title, a.kind,
                       a.synthesized, a.fetched_at, a.published_at
                FROM web_chunks c
                JOIN web_artifacts a ON a.id = c.web_artifact_id
                WHERE c.id = %s
                """,
                (web_chunk_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_unsummarized_segments(
        self, session_id: str, every_n: int = 25
    ) -> list[tuple[int, int, list[int]]]:
        """Return (start_seq, end_seq, episode_ids) for segments that need summaries.

        A segment is N=every_n consecutive episodes. Returns segments not yet covered
        by an existing synth_document summary.
        """
        with self._conn() as conn:
            eps = conn.execute(
                "SELECT id, sequence FROM episodes WHERE session_id = %s ORDER BY sequence ASC",
                (session_id,),
            ).fetchall()

            if not eps:
                return []

            # Existing summary ranges for this session
            covered = conn.execute(
                """SELECT start_sequence, end_sequence FROM synth_documents
                   WHERE session_id = %s AND doc_type = 'summary'""",
                (session_id,),
            ).fetchall()

        covered_ranges = {(r["start_sequence"], r["end_sequence"]) for r in covered}

        segments = []
        for i in range(0, len(eps), every_n):
            window = eps[i : i + every_n]
            if len(window) < every_n:
                break  # incomplete final segment — wait for more episodes
            start = window[0]["sequence"]
            end = window[-1]["sequence"]
            if (start, end) not in covered_ranges:
                segments.append((start, end, [e["id"] for e in window]))

        return segments

    def get_synth_document_source_ids(
        self, session_id: str, content: str, doc_type: str = "summary"
    ) -> list[int]:
        """Return source episode IDs for a synth document, looked up by (session_id, content).

        Used by the extraction pipeline to associate KG edges derived from a summary
        with the underlying episode IDs the summary covers. Returns [] if not found
        or if source_ids is empty/missing.
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT source_ids FROM synth_documents
                WHERE session_id = %s AND content = %s AND doc_type = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id, content, doc_type),
            ).fetchone()
        if not row or not row.get("source_ids"):
            return []
        raw = row["source_ids"]
        # JSONB columns deserialize to Python lists already; defend against str fallback
        if isinstance(raw, str):
            try:
                raw = orjson.loads(raw)
            except (orjson.JSONDecodeError, ValueError):
                return []
        if not isinstance(raw, list):
            return []
        return [int(x) for x in raw if isinstance(x, int | str) and str(x).lstrip("-").isdigit()]

    def get_unembedded_synth_docs(self, limit: int = 48) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, content FROM synth_documents WHERE is_embedded = FALSE LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def set_synth_doc_embedding(self, doc_id: int, embedding: list[float]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE synth_documents SET embedding = %s::vector, is_embedded = TRUE WHERE id = %s",
                (embedding, doc_id),
            )

    # ------------------------------------------------------------------
    # Ingestion watermark
    # ------------------------------------------------------------------

    def get_watermark(self, source: str) -> datetime | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_ingested_at FROM ingestion_state WHERE source = %s",
                (source,),
            ).fetchone()
        return cast(datetime, row["last_ingested_at"]) if row else None

    def set_watermark(self, source: str, ts: datetime) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_state (source, last_ingested_at)
                VALUES (%s, %s)
                ON CONFLICT (source) DO UPDATE SET last_ingested_at = EXCLUDED.last_ingested_at
                """,
                (source, ts),
            )

    # ------------------------------------------------------------------
    # Extraction queue
    # ------------------------------------------------------------------

    def enqueue_extraction(self, item: ExtractionItem) -> None:
        """Enqueue an item for KG extraction. Idempotent — ignores duplicates."""
        if item.episode_id is not None:
            # Deduplicate by episode_id (pending or processing only)
            with self._conn() as conn:
                exists = conn.execute(
                    """
                    SELECT id FROM extraction_queue
                    WHERE episode_id = %s AND status IN ('pending', 'processing')
                    """,
                    (item.episode_id,),
                ).fetchone()
                if exists:
                    return
                conn.execute(
                    """
                    INSERT INTO extraction_queue
                        (episode_id, session_id, content, content_type, project)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        item.episode_id,
                        item.session_id,
                        item.content,
                        item.content_type,
                        item.project,
                    ),
                )

        elif item.content_type == "chunk":
            # Chunks: MANY per session (unlike a single summary), so dedup by
            # exact content, not (session_id, content_type) — the latter would
            # collapse every chunk of a session into one queue row. Enqueued
            # once at birth (ingestion.chunks.rebuild_chunks on_new), this guards
            # only against a double-run re-enqueuing a still-pending chunk.
            with self._conn() as conn:
                exists = conn.execute(
                    """
                    SELECT id FROM extraction_queue
                    WHERE session_id = %s AND content_type = 'chunk' AND content = %s
                      AND status IN ('pending', 'processing')
                    """,
                    (item.session_id, item.content),
                ).fetchone()
                if exists:
                    return
                conn.execute(
                    """
                    INSERT INTO extraction_queue
                        (episode_id, session_id, content, content_type, project)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (None, item.session_id, item.content, item.content_type, item.project),
                )

        else:
            # Summary or manual — deduplicate by session_id + content_type
            with self._conn() as conn:
                exists = conn.execute(
                    """
                    SELECT id FROM extraction_queue
                    WHERE session_id = %s AND content_type = %s
                      AND status IN ('pending', 'processing')
                    """,
                    (item.session_id, item.content_type),
                ).fetchone()
                if exists:
                    return
                conn.execute(
                    """
                    INSERT INTO extraction_queue
                        (episode_id, session_id, content, content_type, project)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (None, item.session_id, item.content, item.content_type, item.project),
                )

    def get_pending_extractions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                """
                SELECT * FROM extraction_queue
                WHERE status = 'pending'
                -- priority lane: new ingest (0) drains before backfill (10);
                -- then oldest-first. (summaries retired #113, so no doc-type tiebreak.)
                ORDER BY priority ASC, enqueued_at ASC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def claim_pending_extractions(self, limit: int = 30) -> list[dict[str, Any]]:
        """Atomically claim up-to-N pending items for this worker.

        Uses ``FOR UPDATE SKIP LOCKED`` against the inner SELECT so multiple
        worker processes (e.g. scaled poller replicas) can call this
        concurrently without race or duplication: each call grabs a distinct
        slice of the pending queue, marks them ``status='processing'`` in the
        same transaction via ``UPDATE ... RETURNING``, and returns the rows.

        If a worker crashes after claiming but before marking done/failed,
        the rows are left ``processing`` indefinitely — see
        ``release_stale_claims`` for the startup-time recovery sweep.
        """
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE extraction_queue
                SET status = 'processing', claimed_at = now()
                WHERE id IN (
                    SELECT id FROM extraction_queue
                    WHERE status = 'pending'
                    -- priority lane: new ingest (0) before backfill (10), then oldest-first.
                    ORDER BY priority ASC, enqueued_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def release_claims(self, queue_ids: list[int]) -> int:
        """Reset specific claimed items back to ``pending``.

        Used when a worker decides to abort the current batch (e.g. on
        UsageLimitError). Only flips rows that are still ``processing`` —
        won't clobber rows that meanwhile became ``done`` or ``failed``.
        """
        if not queue_ids:
            return 0
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE extraction_queue
                SET status = 'pending'
                WHERE status = 'processing' AND id = ANY(%s)
                RETURNING id
                """,
                (queue_ids,),
            ).fetchall()
        return len(result)

    def release_stale_claims(self, older_than_minutes: int = 45) -> int:
        """Reset GENUINELY-STALE ``status='processing'`` rows back to ``pending``.

        A claim is stale if it was taken more than ``older_than_minutes`` ago
        (``claimed_at`` older than the threshold) or predates the claimed_at
        migration (``claimed_at IS NULL``). The default 45 min sits safely above
        the worst-case batch processing time (drain_batch_limit items x per-item
        time, all sharing one batch claimed_at), so this recovers orphans left by
        a crashed or scaled-down worker WITHOUT clobbering rows a live peer is
        still working through its batch.

        Run at startup AND periodically from the maintenance loop. Safe across
        concurrent peers — Postgres serializes the UPDATEs and each row converges
        to ``pending`` exactly once.
        """
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE extraction_queue
                SET status = 'pending'
                WHERE status = 'processing'
                  AND (claimed_at IS NULL OR claimed_at < now() - make_interval(mins => %s))
                RETURNING id
                """,
                (older_than_minutes,),
            ).fetchall()
        return len(result)

    def mark_extraction_done(self, queue_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE extraction_queue
                SET status = 'done', processed_at = NOW()
                WHERE id = %s
                """,
                (queue_id,),
            )

    def mark_extraction_failed(self, queue_id: int, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE extraction_queue
                SET status = 'failed', error = %s,
                    attempts = attempts + 1, processed_at = NOW()
                WHERE id = %s
                """,
                (error, queue_id),
            )

    def log_dedup_gate_shadow(self, rows: list[tuple[Any, ...]]) -> None:
        """Batch-insert Stage-6 gray-zone gate telemetry (schema 040, issue #14).

        Row shape matches ingestion.extractor._gate_shadow_rows: (group_id, fact,
        candidate_uuid, candidate_fact, pool, sim, decision, llm_duplicate,
        llm_contradicted, llm_ran). Best-effort analysis data — the caller wraps
        this in a try/except so a missing table pre-migration never blocks Stage 7.
        """
        if not rows:
            return
        with self._conn() as conn:
            conn.cursor().executemany(
                """
                INSERT INTO dedup_gate_shadow
                    (group_id, fact, candidate_uuid, candidate_fact, pool,
                     sim, decision, llm_duplicate, llm_contradicted, llm_ran)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
