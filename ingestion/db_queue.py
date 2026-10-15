from __future__ import annotations

from typing import Any, cast

from ingestion.db_connection import DatabaseConnection
from ingestion.models import ExtractionItem
from ingestion.schema_check import ColumnProbe

#: ``extraction_queue.chunk_id`` (an optional migration): until it exists a chunk item
#: keeps its text in the row and is found by it, which is how every chunk item was keyed
#: before the column.
CHUNK_ID_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('extraction_queue') AND attname = 'chunk_id' "
    "AND NOT attisdropped",
    "extraction_queue.chunk_id",
)


class ExtractionQueueStore(DatabaseConnection):
    # ------------------------------------------------------------------
    # Extraction queue
    # ------------------------------------------------------------------

    def enqueue_extraction(self, item: ExtractionItem) -> None:
        """Enqueue an item for KG extraction. Idempotent — ignores duplicates.

        An item names its source (``ExtractionItem``) and the row stores no text for it:
        the worker reads the episode or chunk, so deleting the source takes the work with
        it. The exception is a chunk on a database without ``chunk_id`` (see
        ``CHUNK_ID_PROBE``), which keeps its text and dedups on it."""
        if item.episode_id is None and item.content_type != "chunk":
            raise ValueError(f"a {item.content_type!r} extraction item needs an episode_id")
        with self._conn() as conn:
            if item.episode_id is not None:
                # Deduplicate by episode_id (pending or processing only)
                where = "episode_id = %s"
                params: tuple[Any, ...] = (item.episode_id,)
                text = ""
            elif item.chunk_id is not None and CHUNK_ID_PROBE.ready(conn, self._url):
                # Once at birth (ingestion.chunks.rebuild_chunks on_new); the unique index
                # makes a double run a no-op whatever the first row's status.
                conn.execute(
                    "INSERT INTO extraction_queue (chunk_id, session_id, content, content_type, "
                    "project, priority) VALUES (%s, %s, '', 'chunk', %s, %s) "
                    "ON CONFLICT (chunk_id) WHERE chunk_id IS NOT NULL DO NOTHING",
                    (item.chunk_id, item.session_id, item.project, item.priority),
                )
                return
            elif item.content:
                # Before the migration: MANY chunks per session, so dedup by exact text
                # against the live rows (a double run re-enqueuing a pending chunk).
                where = "session_id = %s AND content_type = 'chunk' AND content = %s"
                params = (item.session_id, item.content)
                text = item.content
            else:
                raise ValueError("a chunk extraction item needs a chunk_id or its content")
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
                    (episode_id, session_id, content, content_type, project, priority)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    item.episode_id,
                    item.session_id,
                    text,
                    item.content_type,
                    item.project,
                    item.priority,
                ),
            )

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

    def try_maintenance_lock(self) -> bool:
        """Try for the session-level advisory lock that makes one poller replica the one
        that runs maintenance (chunk rebuild, embedding, the sweeps). It is held until this
        object's connection ends, so a replica that dies hands it over. Trying on the session
        that already holds it succeeds, so asking every cycle answers for the current
        connection: a holder that dropped and reconnected gets the lock back only if no
        other replica took it meanwhile. It needs the connection to be this object's own,
        so not a pooled one (the poller opens no pool)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(hashtext('poller.maintenance')) AS held"
            ).fetchone()
        return bool(row and row["held"])

    def release_stale_claims(self, older_than_minutes: int = 45) -> int:
        """Reset GENUINELY-STALE ``status='processing'`` rows back to ``pending``.

        A claim is stale if it was taken more than ``older_than_minutes`` ago
        (``claimed_at`` older than the threshold) or predates the claimed_at
        migration (``claimed_at IS NULL``). The default 45 min sits safely above
        the worst-case time of one item (a worker claims no more rows than it has
        threads), so this recovers orphans left by a crashed or scaled-down worker
        WITHOUT clobbering rows a live peer is still working on.

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

    def blank_finished_queue_text(self, older_than_days: int = 7, batch: int = 2000) -> int:
        """Empty the ``content`` of finished rows older than ``older_than_days``; returns
        how many it emptied.

        The queue used to copy every turn and chunk it queued and kept the copy for good
        (as much text again as the corpus). Nothing reads a finished row's text. The rows
        stay (the dashboard's enqueue and completion series, and web_enqueue's anti-join,
        read them), only the text goes, in batches so one statement never rewrites
        gigabytes of TOAST at once. Rows that name their source store no text from the start.
        """
        total = 0
        while True:
            with self._conn() as conn:
                n = conn.execute(
                    "UPDATE extraction_queue SET content = '' WHERE id IN ("
                    "  SELECT id FROM extraction_queue "
                    "  WHERE status = 'done' AND content <> '' "
                    "    AND processed_at < now() - make_interval(days => %s) LIMIT %s)",
                    (older_than_days, batch),
                ).rowcount
            total += n
            if n < batch:
                return total
