"""Maintenance loop — builds chunks, embeds pending docs, and drains the
extraction queue to populate the Postgres KG.

Episodes arrive via the ``/ingest`` push hook (``mcp_server.server``), not a
Logfire poll. This process owns the post-ingest work: chunk construction,
embeddings, and KG extraction (facts are extracted from chunks)."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ingestion.db import Database
from ingestion.models import ExtractionItem

if TYPE_CHECKING:
    from ingestion.extractor import ExtractionPipeline

logger = logging.getLogger(__name__)


def _drain_concurrency() -> int:
    """In-worker drain concurrency (issue #13). Per-item extraction time is ~all
    network wait (3-6 dependent LLM calls), so running the claimed batch
    concurrently is ~batch_limit-fold throughput per worker at the same RAM. Default 4;
    SYNAPSE_DRAIN_CONCURRENCY=1 is the serial kill switch."""
    try:
        return max(1, int(os.environ.get("SYNAPSE_DRAIN_CONCURRENCY", "4") or "1"))
    except ValueError:
        return 1


# run_loop touches this at the top of every iteration; the poller services' compose
# healthcheck reads its age. Its window (compose.yml) must outlast the longest honest
# iteration: one drain (45 min is the stale-claim limit) plus the 30 min quota backoff.
_HEARTBEAT = Path(tempfile.gettempdir()) / "poller.alive"

# Daily, on the master replica's full cycle: entities with no edges are deleted once a day
# old (KGClient.delete_orphan_entities), and the text of finished queue rows is emptied.
# Drain-only replicas never sweep.
_DAILY_SWEEP_INTERVAL_S = 24 * 3600
_ORPHAN_MIN_AGE_H = 24
# extraction_queue.priority of catch-up work; new ingest is 0 (lower drains first).
_CATCH_UP_PRIORITY = 10
_QUEUE_TEXT_KEEP_DAYS = 7


class Poller:
    def __init__(
        self,
        db: Database,
        extraction_pipeline: ExtractionPipeline | None = None,
        worker_factory: Callable[[], tuple[Database, ExtractionPipeline]] | None = None,
    ) -> None:
        self._db = db
        self._extraction = extraction_pipeline
        # Per-THREAD (db, pipeline) builder for the concurrent drain path: Database
        # holds one non-autocommit connection whose commit/rollback would entangle
        # transactions if shared across threads, so each drain worker thread gets its
        # own stack (lazily, on its first item). None -> drain always runs serial.
        self._worker_factory = worker_factory
        self._worker_local = threading.local()
        self._drain_pool: ThreadPoolExecutor | None = None
        # SIGTERM/SIGINT set this (see run_loop): the drain finishes its
        # in-flight item, releases the rest of the claimed batch back to
        # pending, and run_loop exits instead of sleeping — so `docker stop`
        # doesn't force-kill a worker mid-item.
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask the loop to exit at the next item boundary."""
        self._stop_requested.set()

    def sweep_orphan_entities(self) -> int:
        """Delete KG entities no edge references (see KGClient.delete_orphan_entities)."""
        if self._extraction is None:
            return 0
        return int(self._extraction._kg.delete_orphan_entities(_ORPHAN_MIN_AGE_H))

    def sweep_finished_queue_text(self) -> int:
        """Empty the text of extraction-queue rows finished a week ago (see
        ``blank_finished_queue_text``)."""
        return self._db.blank_finished_queue_text(_QUEUE_TEXT_KEEP_DAYS)

    def embed_pending(self, batch_size: int = 500, budget_s: float = 60.0) -> int:
        """Embed unembedded episodes, then chunks, until nothing is pending or ``budget_s``
        is spent. Returns the number embedded.

        The embedder packs requests by tokens, so ``batch_size`` only bounds the rows one
        UPDATE carries. Looping, rather than one small batch per cycle, is what lets a
        history import (tens of thousands of turns) get its vectors in hours instead of
        weeks, while live turns (newest first) never wait behind it.
        """
        if self._extraction is None:
            return 0

        embedder = self._extraction._embedder
        deadline = time.monotonic() + budget_s
        total = 0
        sources: tuple[
            tuple[Literal["episodes", "chunks"], Callable[[int], list[dict[str, Any]]]], ...
        ] = (
            ("episodes", self._db.get_unembedded_episodes),
            ("chunks", self._db.get_unembedded_chunks),
        )
        for table, fetch in sources:
            while time.monotonic() < deadline and (rows := fetch(batch_size)):
                embeddings = embedder.embed([r["content"] for r in rows], task="document")
                self._db.set_embeddings(
                    table, [(r["id"], e) for r, e in zip(rows, embeddings, strict=True)]
                )
                total += len(rows)
                logger.info("Embedded %d %s", len(rows), table)
        return total

    def rebuild_pending_chunks(self, sessions: set[str] | None = None) -> int:
        """(Re)build chunks for sessions whose episodes outgrew their chunks.

        A cheap pre-filter (``sessions_with_pending_chunks``) avoids re-scanning
        every session each cycle. Chunk creation used to be coupled to the
        now-removed Logfire poll path; restoring it here keeps the chunk retrieval
        layer current for /ingest-hook episodes. Embedding is handled separately by
        ``embed_pending``.
        """
        from ingestion.chunks import rebuild_chunks

        pending = sessions if sessions is not None else set(self._db.sessions_with_pending_chunks())
        if not pending:
            return 0

        # Chunks are the KG fact substrate (task #63): enqueue each freshly-built
        # chunk for extraction exactly once, at birth. rebuild_chunks only fires
        # on_new for genuinely-new complete windows, so no re-enqueue churn. Not
        # flag-gated — this is the architecture; summaries no longer extract facts.
        enqueued = 0

        def on_new(chunk: dict[str, Any]) -> None:
            nonlocal enqueued
            self._db.enqueue_extraction(
                ExtractionItem(
                    session_id=chunk["session_id"],
                    chunk_id=chunk["chunk_id"],
                    content=chunk["content"],
                    content_type="chunk",
                    project=chunk["project"],
                    # A flushed tail is catch-up work (the first pass after it shipped
                    # covers every old session at once): behind live ingest, like a backfill.
                    priority=_CATCH_UP_PRIORITY if chunk["tail"] else 0,
                )
            )
            enqueued += 1

        total = 0
        for session_id in pending:
            try:
                total += rebuild_chunks(self._db, session_id, on_new=on_new)
            except Exception as e:
                logger.warning("Chunk rebuild failed for %s: %s", session_id[:12], e)
        if total:
            logger.info(
                "Rebuilt %d chunk(s) across %d session(s); enqueued %d for extraction",
                total,
                len(pending),
                enqueued,
            )
        return total

    def drain_extraction_queue(self, batch_limit: int = 30) -> int:
        """Process pending extraction queue items. Returns number processed.

        Uses ``claim_pending_extractions`` so multiple poller replicas can
        run concurrently without racing: each replica gets a distinct slice
        of pending items via ``FOR UPDATE SKIP LOCKED`` and the rows flip
        to ``status='processing'`` for the duration.

        With a ``worker_factory`` and SYNAPSE_DRAIN_CONCURRENCY > 1 (issue #13) items run on
        a thread pool: per-item work is ~all LLM network wait, so a thread per in-flight
        item is ~concurrency-fold throughput at the same RAM. ``batch_limit`` then bounds
        how many items one call takes in total (so run_loop gets back to its other work),
        while rows are claimed only as threads free up. Failure stays item-isolated — one
        bad item marks failed without touching the rest.

        On UsageLimitError (Claude usage cap hit) the offending item and
        every still-claimed item are released back to ``pending`` so another
        replica (or this one on the next cycle) can retry once the cap resets.
        """
        if self._extraction is None:
            return 0

        concurrency = _drain_concurrency() if self._worker_factory is not None else 1
        if concurrency > 1:
            processed = self._drain_concurrent(batch_limit, concurrency)
        else:
            items = self._db.claim_pending_extractions(limit=batch_limit)
            if not items:
                return 0
            processed = self._drain_serial(items)

        if processed:
            logger.info("Drained %d extraction items", processed)
        return processed

    def _drain_serial(self, items: list[dict[str, Any]]) -> int:
        """The original strictly-serial batch path (SYNAPSE_DRAIN_CONCURRENCY=1)."""
        import logfire

        from ingestion.llm_client import UsageLimitError

        assert self._extraction is not None
        unfinished: list[int] = [int(item["id"]) for item in items]
        processed = 0
        for item in items:
            if self._stop_requested.is_set():
                logger.info(
                    "Stop requested — releasing %d unstarted claim(s) back to pending",
                    len(unfinished),
                )
                self._db.release_claims(unfinished)
                break
            queue_id = int(item["id"])
            try:
                with logfire.span(
                    "process_item qid={qid} {ct}",
                    qid=queue_id,
                    ct=item.get("content_type"),
                    project=item.get("project"),
                ):
                    self._extraction.process_item(dict(item))
                self._db.mark_extraction_done(queue_id)
                unfinished.remove(queue_id)
                processed += 1
                logger.debug("Extraction done for queue_id=%d", queue_id)
            except UsageLimitError as e:
                logger.warning(
                    "Usage limit hit at queue_id=%d (%s). Releasing %d "
                    "still-claimed items back to pending for retry.",
                    queue_id,
                    str(e)[:200],
                    len(unfinished),
                )
                self._db.release_claims(unfinished)
                unfinished = []
                raise  # signal run_loop to back off until the quota window resets
            except Exception as e:
                logger.error("Extraction failed for queue_id=%d: %s", queue_id, e, exc_info=True)
                self._db.mark_extraction_failed(queue_id, str(e)[:500])
                unfinished.remove(queue_id)
        return processed

    def _thread_worker(self) -> tuple[Database, ExtractionPipeline]:
        """This thread's own (db, pipeline), built lazily on first use. The pool's
        threads persist across drain cycles, so connections are reused, not churned."""
        assert self._worker_factory is not None
        stack = getattr(self._worker_local, "stack", None)
        if stack is None:
            stack = self._worker_factory()
            self._worker_local.stack = stack
        return stack

    def _drain_concurrent(self, batch_limit: int, concurrency: int) -> int:
        """Run up to ``batch_limit`` queue items through the persistent drain pool, keeping
        every thread busy: a row is claimed only when a thread is free to start it, so one
        slow item (they run minutes) neither idles the other threads nor holds claims it
        is not working on.

        Per-item failure isolation: an ordinary failure marks THAT item failed and
        never releases the rest. A UsageLimitError marks the quota_hit event so no
        further row is claimed and claimed-but-unstarted items skip their LLM attempt;
        every quota-hit item is released back to ``pending`` and UsageLimitError is
        re-raised so run_loop backs off — the same contract as the serial path.
        """
        import logfire

        from ingestion.llm_client import UsageLimitError

        if self._drain_pool is None:
            self._drain_pool = ThreadPoolExecutor(
                max_workers=concurrency, thread_name_prefix="drain"
            )
        pool = self._drain_pool
        quota_hit = threading.Event()

        def _one(item: dict[str, Any]) -> tuple[int, str]:
            queue_id = int(item["id"])
            if self._stop_requested.is_set():
                return queue_id, "stopped"  # release below; exit once the in-flight items end
            if quota_hit.is_set():
                return queue_id, "quota"  # don't burn an attempt on a known-dead window
            db, pipeline = self._thread_worker()
            try:
                with logfire.span(
                    "process_item qid={qid} {ct}",
                    qid=queue_id,
                    ct=item.get("content_type"),
                    project=item.get("project"),
                ):
                    pipeline.process_item(dict(item))
                db.mark_extraction_done(queue_id)
                logger.debug("Extraction done for queue_id=%d", queue_id)
                return queue_id, "done"
            except UsageLimitError as e:
                quota_hit.set()
                logger.warning("Usage limit hit at queue_id=%d (%s)", queue_id, str(e)[:200])
                return queue_id, "quota"
            except Exception as e:
                logger.error("Extraction failed for queue_id=%d: %s", queue_id, e, exc_info=True)
                try:
                    db.mark_extraction_failed(queue_id, str(e)[:500])
                except Exception:
                    logger.exception("Could not mark queue_id=%d failed", queue_id)
                return queue_id, "failed"

        running: set[Future[tuple[int, str]]] = set()
        results: list[tuple[int, str]] = []
        claimed = 0
        while True:
            free = min(concurrency - len(running), batch_limit - claimed)
            if free > 0 and not self._stop_requested.is_set() and not quota_hit.is_set():
                items = self._db.claim_pending_extractions(limit=free)
                claimed += len(items)
                running.update(pool.submit(_one, item) for item in items)
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            results.extend(f.result() for f in finished)

        stopped_ids = [qid for qid, status in results if status == "stopped"]
        if stopped_ids:
            logger.info(
                "Stop requested — releasing %d unstarted claim(s) back to pending",
                len(stopped_ids),
            )
            self._db.release_claims(stopped_ids)
        quota_ids = [qid for qid, status in results if status == "quota"]
        if quota_ids:
            logger.warning(
                "Releasing %d quota-hit item(s) back to pending for retry", len(quota_ids)
            )
            self._db.release_claims(quota_ids)
            raise UsageLimitError(f"usage limit hit on {len(quota_ids)} item(s) in batch")
        return sum(1 for _, status in results if status == "done")

    def run_loop(self, interval_seconds: int = 300, project: str | None = None) -> None:
        """Run continuously.

        Summaries + embed run every ``interval_seconds`` on the master
        replica. Extraction drain runs more aggressively: a short pause (3s)
        when the previous drain returned a full batch (more backlog likely),
        a longer pause (60s) otherwise — eats through extraction backlog
        without idle time.

        When ``SYNAPSE_DRAIN_ONLY=1`` is set in the environment, the full
        cycle (chunk rebuild + embed) is skipped entirely and this replica
        becomes a pure extraction-queue worker. Use this for scaled-out drain
        replicas alongside a single master. (``project`` is unused now that
        episodes arrive pre-projected via the /ingest hook; kept for call
        compatibility.)
        """
        import os as _os
        import random as _random
        import signal as _signal

        from ingestion.llm_client import UsageLimitError

        def _on_signal(signum: int, _frame: Any) -> None:
            logger.info(
                "Signal %d received — finishing the in-flight item, then exiting cleanly",
                signum,
            )
            self.request_stop()

        try:
            _signal.signal(_signal.SIGTERM, _on_signal)
            _signal.signal(_signal.SIGINT, _on_signal)
        except ValueError:
            # Signal registration only works on the main thread; callers that
            # run the loop elsewhere (tests) drive request_stop() directly.
            pass

        drain_only = _os.environ.get("SYNAPSE_DRAIN_ONLY", "").strip() in ("1", "true", "yes")
        logger.info(
            "Starting maintenance loop (full-cycle-interval=%ds, fast-drain enabled, drain_only=%s)",
            interval_seconds,
            drain_only,
        )
        # Release any extraction-queue items left in `processing` by a
        # previous run that crashed or was killed mid-extraction. Safe to
        # run on every startup, including concurrent peer starts — Postgres
        # serializes the UPDATE so all replicas converge to the same state.
        released = self._db.release_stale_claims()
        if released:
            logger.info("Released %d stale 'processing' claims back to pending", released)
        last_full_cycle = 0.0
        last_stale_sweep = time.monotonic()  # just swept above; next sweep one interval out
        last_daily_sweep = 0.0  # first full cycle sweeps, then daily
        # Rows are claimed only as a worker thread frees up (the serial kill switch claims
        # one at a time): a claimed row stays 'processing' until its item ends, so one that
        # merely waits for a thread would make the priority lane laggy and, when the worker
        # is killed mid-item (a deploy, an OOM), be stranded for the 45 min stale-claim
        # threshold along with the items that were running. The limit is how many items one
        # drain call takes before run_loop gets back to its chunk/embed cycle and the
        # priority check: two waves of the pool.
        concurrency = _drain_concurrency() if self._worker_factory is not None else 1
        drain_batch_limit = 2 * concurrency if concurrency > 1 else 1
        prev_drained = 0
        quota_backoff = 0  # seconds; grows while the Max quota window is exhausted, 0 when healthy
        while True:
            if self._stop_requested.is_set():
                logger.info("Maintenance loop stopped cleanly")
                return
            _HEARTBEAT.touch()
            drained = 0
            now_mono = time.monotonic()
            if not drain_only:
                ready_for_full = now_mono - last_full_cycle >= interval_seconds
                # Don't let the periodic chunk-rebuild/embed cycle starve a large
                # EXTRACTION backlog (the #29 case): if the previous drain filled
                # its batch, defer the full cycle so extraction runs uninterrupted
                # — bounded to 2*interval so chunk-rebuild/embed still fire periodically.
                if (
                    prev_drained >= drain_batch_limit
                    and now_mono - last_full_cycle < 2 * interval_seconds
                ):
                    ready_for_full = False
                if ready_for_full:
                    # Each stage in its own try block: one failing stage must not
                    # abort the others (embed must run even if chunk rebuild fails,
                    # or the unembedded backlog grows unbounded). (2026-05-18.)
                    try:
                        self.rebuild_pending_chunks()
                    except Exception as e:
                        logger.error("Chunk rebuild failed: %s", e, exc_info=True)
                    try:
                        self.embed_pending()
                    except Exception as e:
                        logger.error("Embed pending failed: %s", e, exc_info=True)
                    if now_mono - last_daily_sweep >= _DAILY_SWEEP_INTERVAL_S:
                        try:
                            n_orphans = self.sweep_orphan_entities()
                            if n_orphans:
                                logger.info("Deleted %d orphan KG entities", n_orphans)
                        except Exception as e:
                            logger.error("Orphan entity sweep failed: %s", e, exc_info=True)
                        try:
                            n_blanked = self.sweep_finished_queue_text()
                            if n_blanked:
                                logger.info("Emptied the text of %d finished queue rows", n_blanked)
                        except Exception as e:
                            logger.error("Queue text sweep failed: %s", e, exc_info=True)
                        last_daily_sweep = now_mono
                    last_full_cycle = now_mono
            try:
                drained = self.drain_extraction_queue(batch_limit=drain_batch_limit)
                quota_backoff = 0  # a successful drain means the quota window is open again
            except UsageLimitError as e:
                # Max-subscription window exhausted. Claims were already released back to
                # pending by drain(), so nothing is lost. Back off exponentially (5→10→…→30 min,
                # jittered to avoid a thundering herd of 30+ workers) instead of hammering the
                # dead quota every 60s; resume automatically when a drain next succeeds.
                quota_backoff = min((quota_backoff or 150) * 2, 1800)
                logger.warning(
                    "Quota exhausted; backing off %ds before retry (%s)",
                    quota_backoff,
                    str(e)[:120],
                )
                self._stop_requested.wait(quota_backoff + _random.uniform(0, 30))
                continue
            except Exception as e:
                logger.error("Extraction drain failed: %s", e, exc_info=True)
            prev_drained = drained
            # Periodically recover orphaned claims (crashed / scaled-down worker left
            # rows stuck in 'processing'). Time-bounded so live peer claims are untouched.
            if now_mono - last_stale_sweep >= interval_seconds:
                try:
                    swept = self._db.release_stale_claims()
                    if swept:
                        logger.info("Periodic sweep released %d stale claim(s)", swept)
                except Exception as e:
                    logger.error("Stale-claim sweep failed: %s", e, exc_info=True)
                last_stale_sweep = now_mono
            # Fast loop when extraction has likely-more-pending, slow loop
            # otherwise. Event.wait so a stop request interrupts the sleep.
            self._stop_requested.wait(3 if drained >= drain_batch_limit else 60)


def make_poller(
    db_url: str,
    voyage_api_key: str,
    llm_model: str | None = None,
) -> Poller:
    from ingestion.embedding import create_embedder
    from ingestion.extractor import ExtractionPipeline
    from ingestion.kg_client import KGClient
    from ingestion.llm_client import create_llm_client

    db = Database(db_url)
    llm = create_llm_client()
    embedder = create_embedder(voyage_api_key=voyage_api_key, db_url=db_url)
    pipeline = ExtractionPipeline(
        db=db,
        llm_client=llm,
        embedder=embedder,
        kg_client=KGClient(db_url),
        llm_model=llm_model,
    )

    def _worker() -> tuple[Database, ExtractionPipeline]:
        # A drain worker THREAD's own stack (issue #13): Database is single-
        # connection/non-autocommit and the pipeline writes through it, so neither
        # can be shared across concurrently-running items. All constructors are
        # cheap (lazy PG connects, HTTP clients) — built once per pool thread.
        wdb = Database(db_url)
        wpipe = ExtractionPipeline(
            db=wdb,
            llm_client=create_llm_client(),
            embedder=create_embedder(voyage_api_key=voyage_api_key, db_url=db_url),
            kg_client=KGClient(db_url),
            llm_model=llm_model,
        )
        return wdb, wpipe

    return Poller(db=db, extraction_pipeline=pipeline, worker_factory=_worker)
