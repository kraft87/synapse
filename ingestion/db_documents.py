from __future__ import annotations

from typing import Any, cast

import orjson

from ingestion.db_connection import DatabaseConnection


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
