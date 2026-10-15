from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

import orjson

from ingestion.db_connection import DatabaseConnection
from ingestion.vector import vec_literal
from ingestion.web_chunker import Chunk


def _episode_ids(raw: Any) -> list[int]:
    """A chunk's ``episode_ids`` JSONB as ints ([] when NULL)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = orjson.loads(raw)
    return [int(x) for x in raw]


class DocumentStore(DatabaseConnection):
    def insert_recall_feedback(
        self,
        *,
        query: str,
        helpful: list[str],
        noise: list[str],
        missing: str | None,
        found_via: str | None,
        note: str | None,
        session_id: str | None,
        project: str | None,
    ) -> int:
        """One labeled retrieval-quality report (the recall_feedback tool).

        Offline data only — never read by live ranking; ids are pre-validated
        served forms ("e:N", "n:N", "f:<uuid>", "t:N", "w:N") at the tool
        boundary. Historical rows may also carry "p:N" preference ids, served
        before the preferences recall leg was removed (2026-07-27)."""
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO recall_feedback "
                "    (query, helpful, noise, missing, found_via, note, session_id, project) "
                "VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s) RETURNING id",
                (
                    query,
                    orjson.dumps(helpful).decode(),
                    orjson.dumps(noise).decode(),
                    missing,
                    found_via,
                    note,
                    session_id,
                    project,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def get_unembedded_episodes(self, limit: int) -> list[dict[str, Any]]:
        """Episodes still without a vector, newest first: a turn that arrives during a
        history import is searchable before the import's backlog is (``ORDER BY id`` is
        also what the partial ``is_embedded = FALSE`` index serves)."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, content FROM episodes WHERE is_embedded = FALSE "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def set_embeddings(
        self,
        table: Literal["episodes", "chunks", "web_chunks"],
        rows: list[tuple[int, list[float]]],
    ) -> None:
        """Store ``(id, vector)`` pairs on ``table`` in one statement and one commit."""
        if not rows:
            return
        ids = [i for i, _ in rows]
        vectors = [vec_literal(v) for _, v in rows]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE {table} t SET embedding = v.e::vector, is_embedded = TRUE "  # nosec B608 — table is one of three literals
                "FROM unnest(%s::bigint[], %s::text[]) AS v(id, e) WHERE t.id = v.id",
                (ids, vectors),
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
    ) -> int | None:
        """Insert a chunk and return its id; None if this (session, start, end) range
        already exists (another worker wrote it first)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                INSERT INTO chunks
                    (session_id, start_sequence, end_sequence, episode_ids, content, project)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (session_id, start_sequence, end_sequence) DO NOTHING
                RETURNING id
                """,
                (
                    session_id,
                    start_sequence,
                    end_sequence,
                    orjson.dumps(episode_ids).decode(),
                    content,
                    project,
                ),
            ).fetchone()
        return None if row is None else cast(int, row["id"])

    def get_unembedded_chunks(self, limit: int) -> list[dict[str, Any]]:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, content FROM chunks WHERE is_embedded = FALSE "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def sessions_with_pending_chunks(self) -> list[str]:
        """Return session IDs that have a NEW chunk window available.

        Pending when (a) the session has no chunks yet but >= 4 episodes (enough for one
        window), (b) its max episode sequence is at least 3 past its chunks' max
        ``end_sequence`` — i.e. enough new episodes arrived to form the next complete
        window (window=4, step=3, so each new window needs ``step`` more episodes), or
        (c) the session has been idle for 2 hours and holds a turn past its chunks' max
        ``end_sequence``: its tail is chunked as it stands
        (``ingestion.chunks.rebuild_chunks``). A remember() archive (``source='manual'``)
        is no reason for (c): chunks leave it out. The 4/3/2h mirror ``ingestion.chunks``;
        kept inline so this stays one pre-filter query. A live session with only a 1-2
        episode incomplete tail is NOT flagged (avoids a no-op rebuild every cycle).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                WITH ep AS (
                    SELECT session_id, COUNT(*) AS c, MAX(sequence) AS m,
                           MAX(sequence) FILTER (WHERE source IS DISTINCT FROM 'manual') AS t,
                           MAX(created_at) AS last_at
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
                   OR (e.last_at < now() - interval '2 hours' AND e.t > COALESCE(c.m, 0))
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

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        """``{content, episode_ids}`` of a chunk, by key (what a ``chunk_id`` queue item
        extracts, and the episodes its edges trace back to). None when the chunk is gone
        (deleted with its episodes, or rebuilt)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content, episode_ids FROM chunks WHERE id = %s", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        return {"content": row["content"], "episode_ids": _episode_ids(row["episode_ids"])}

    def get_chunk_episode_ids(self, session_id: str, content: str) -> list[int]:
        """Return the episode_ids of the chunk with this exact content (for edge backlink).

        For a queue item that has no ``chunk_id`` (one enqueued before that column
        existed): chunk extraction (task #63) enqueued a chunk's text, and when its facts
        are written the edges must trace back to the episodes the chunk was built from.
        Chunk content is the episodes joined verbatim, so (session_id, content) is a
        stable key. Returns [] if no match (chunk since rebuilt/removed).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT episode_ids FROM chunks WHERE session_id = %s AND content = %s LIMIT 1",
                (session_id, content),
            ).fetchone()
        return _episode_ids(row["episode_ids"]) if row else []

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

    # ------------------------------------------------------------------
    # Web chunks: captured pages (web_artifacts) -> chunks -> context -> vectors
    # ------------------------------------------------------------------

    def artifacts_without_chunks(self, limit: int) -> list[dict[str, Any]]:
        """Captured pages not chunked yet, oldest first (id, content_markdown, fetched_at)."""
        with self._conn() as conn:
            result = conn.execute(
                """
                SELECT a.id, a.content_markdown, a.fetched_at
                FROM web_artifacts a
                WHERE a.kind IN ('web_scrape', 'research_brief')
                  AND a.content_markdown IS NOT NULL AND a.content_markdown <> ''
                  AND NOT EXISTS (SELECT 1 FROM web_chunks c WHERE c.web_artifact_id = a.id)
                ORDER BY a.id LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def insert_web_chunks(self, artifact_id: int, fetched_at: datetime, chunks: list[Chunk]) -> int:
        """Store a page's chunks (unembedded, no context prefix yet); a chunk already stored
        under the same (artifact, idx) is left alone. ``content_ts`` is the page's fetch time."""
        if not chunks:
            return 0
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO web_chunks "
                "(web_artifact_id, idx, content, char_start, char_end, content_ts) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (web_artifact_id, idx) DO NOTHING",
                [
                    (artifact_id, c.idx, c.content, c.char_start, c.char_end, fetched_at)
                    for c in chunks
                ],
            )
        return len(chunks)

    def artifacts_to_contextualize(self, limit: int) -> list[dict[str, Any]]:
        """Pages with a chunk that has no context prefix yet, oldest first (id, content_markdown)."""
        with self._conn() as conn:
            result = conn.execute(
                """
                SELECT a.id, a.content_markdown
                FROM web_artifacts a
                WHERE a.kind IN ('web_scrape', 'research_brief')
                  AND a.content_markdown IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM web_chunks c
                      WHERE c.web_artifact_id = a.id AND c.context_prefix IS NULL
                  )
                ORDER BY a.id LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def get_web_chunks(self, artifact_id: int) -> list[dict[str, Any]]:
        """A page's chunks in order (id, idx, content)."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, idx, content FROM web_chunks WHERE web_artifact_id = %s ORDER BY idx",
                (artifact_id,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def set_web_chunk_contexts(self, rows: list[tuple[int, str]]) -> None:
        """Store ``(chunk id, context prefix)`` pairs ('' is "no context") and mark the chunks
        for (re-)embedding: a vector is of the prefix and the chunk together."""
        if not rows:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE web_chunks c SET context_prefix = v.prefix, is_embedded = FALSE "
                "FROM unnest(%s::bigint[], %s::text[]) AS v(id, prefix) WHERE c.id = v.id",
                ([i for i, _ in rows], [p for _, p in rows]),
            )

    def get_unembedded_web_chunks(self, limit: int) -> list[dict[str, Any]]:
        """Web chunks ready for a vector, newest first: ``content`` is the context prefix
        and the chunk. A chunk still waiting for its prefix is not ready."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT id, "
                "CASE WHEN context_prefix = '' THEN content "
                "ELSE context_prefix || E'\\n\\n' || content END AS content "
                "FROM web_chunks WHERE NOT is_embedded AND context_prefix IS NOT NULL "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return cast(list[dict[str, Any]], result)

    def enqueue_web_extraction(self) -> int:
        """Queue the embedded chunks of pages marked for KG extraction (see
        ``ingestion.web_enqueue``). Returns the number queued."""
        from ingestion.web_enqueue import enqueue_web_chunks

        with self._conn() as conn:
            return enqueue_web_chunks(conn).enqueued
