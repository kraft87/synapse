from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from ingestion.db_connection import DatabaseConnection
from ingestion.models import ExtractionItem


class ExtractionQueueStore(DatabaseConnection):
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
        # Each path dedups on a different predicate against still-live rows
        # (status pending/processing); the exists-check + INSERT are otherwise
        # identical, so compute (where, params) per path and share the rest.
        if item.episode_id is not None:
            # Deduplicate by episode_id (pending or processing only)
            where = "episode_id = %s"
            params: tuple[Any, ...] = (item.episode_id,)
        elif item.content_type == "chunk":
            # Chunks: MANY per session (unlike a single summary), so dedup by
            # exact content, not (session_id, content_type) — the latter would
            # collapse every chunk of a session into one queue row. Enqueued
            # once at birth (ingestion.chunks.rebuild_chunks on_new), this guards
            # only against a double-run re-enqueuing a still-pending chunk.
            where = "session_id = %s AND content_type = 'chunk' AND content = %s"
            params = (item.session_id, item.content)
        else:
            # Summary or manual — deduplicate by session_id + content_type
            where = "session_id = %s AND content_type = %s"
            params = (item.session_id, item.content_type)

        with self._conn() as conn:
            exists = conn.execute(
                "SELECT id FROM extraction_queue "  # nosec B608 — where is built from static literals, not user input
                f"WHERE {where} AND status IN ('pending', 'processing')",
                params,
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

    #: A failed item is retried at most this many times before it stays failed. Bounded
    #: so a genuinely poisonous item (content the model will never parse) stops costing
    #: LLM calls, while an outage or a wrong model id — which is what failure normally
    #: means here — gets picked up again on its own.
    MAX_EXTRACTION_ATTEMPTS = 5
    #: How long a failed item waits before it is eligible again. Long enough that a
    #: broken config is a slow trickle rather than a hot loop, short enough that fixing
    #: the config drains the backlog the same afternoon.
    FAILED_RETRY_MINUTES = 30

    def claim_pending_extractions(self, limit: int = 30) -> list[dict[str, Any]]:
        """Atomically claim up-to-N claimable items for this worker.

        Uses ``FOR UPDATE SKIP LOCKED`` against the inner SELECT so multiple
        worker processes (e.g. scaled poller replicas) can call this
        concurrently without race or duplication: each call grabs a distinct
        slice of the queue, marks them ``status='processing'`` in the
        same transaction via ``UPDATE ... RETURNING``, and returns the rows.

        Claimable = ``pending``, plus ``failed`` rows that have not exhausted
        ``MAX_EXTRACTION_ATTEMPTS`` and last ran more than
        ``FAILED_RETRY_MINUTES`` ago. Without the second lane ``failed`` was
        terminal forever: nothing ever re-claimed it, so an item that failed
        during an outage (or against a wrong model id) stayed graph-less no
        matter what was fixed afterwards. Pending always sorts first, so a
        retry backlog can never starve new ingest.

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
                       OR (
                            status = 'failed'
                            AND attempts < %s
                            AND COALESCE(processed_at, enqueued_at)
                                < now() - make_interval(mins => %s)
                       )
                    -- new work first, retries after; then the priority lane
                    -- (new ingest 0 before backfill 10) and oldest-first.
                    ORDER BY (status <> 'pending'), priority ASC, enqueued_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (self.MAX_EXTRACTION_ATTEMPTS, self.FAILED_RETRY_MINUTES, limit),
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
