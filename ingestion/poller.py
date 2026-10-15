"""The poller process: the work after ingest, on two threads.

Episodes arrive via the ``/ingest`` push hook (``mcp_server.server``), not a Logfire poll.
The MAINTENANCE thread builds chunks, embeds, and runs the sweeps, and carries captured web
pages (``ingestion.capture``) through chunking, context prefixes and embedding to recall; the
EXTRACTION thread drains the extraction queue to populate the Postgres KG (facts are extracted
from chunks).
Neither waits for the other: a backlog of extraction batches never delays embedding, and a
quota backoff pauses only extraction."""

from __future__ import annotations

import logging
import os
import random
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ingestion.db import Database
from ingestion.models import ExtractionItem
from ingestion.web_chunker import chunk_markdown
from ingestion.web_context import contextualize_pending

if TYPE_CHECKING:
    from ingestion.extractor import ExtractionPipeline

logger = logging.getLogger(__name__)


def _drain_concurrency() -> int:
    """In-worker drain concurrency (issue #13). Per-item extraction time is ~all
    network wait (3-6 dependent LLM calls), so running items concurrently is
    ~concurrency-fold throughput per worker at the same RAM. Default 4; the one
    per-deployment burst knob, and SYNAPSE_DRAIN_CONCURRENCY=1 is the serial kill switch."""
    try:
        return max(1, int(os.environ.get("SYNAPSE_DRAIN_CONCURRENCY", "4") or "1"))
    except ValueError:
        return 1


# Each thread touches its own file at the top of every iteration, and the poller service's
# compose healthcheck reads both ages, so one stalled thread fails it while the other runs.
# The window (compose.yml) must outlast the longest honest iteration: one drain (45 min is
# the stale-claim limit) plus the 30 min quota backoff. The extraction file keeps the name
# the single-heartbeat compose reads, so a compose not yet re-applied keeps passing.
_EXTRACTION_HEARTBEAT = Path(tempfile.gettempdir()) / "poller.alive"
_MAINTENANCE_HEARTBEAT = Path(tempfile.gettempdir()) / "poller.maintenance.alive"

# Daily, on the replica holding the maintenance lock: entities with no edges are deleted once
# a day old (KGClient.delete_orphan_entities), and the text of finished queue rows is emptied.
_DAILY_SWEEP_INTERVAL_S = 24 * 3600
_ORPHAN_MIN_AGE_H = 24
# extraction_queue.priority of catch-up work; new ingest is 0 (lower drains first).
_CATCH_UP_PRIORITY = 10
_QUEUE_TEXT_KEEP_DAYS = 7
# Captured pages one cycle chunks.
_WEB_PAGES_PER_CYCLE = 200


class Maintenance:
    """Chunk building, embedding and the sweeps, on a (db, pipeline) stack of their own:
    Database is one non-autocommit connection, so this thread cannot share the extraction
    thread's. One replica runs them at a time (see :meth:`run`)."""

    def __init__(self, db: Database, pipeline: ExtractionPipeline) -> None:
        self._db = db
        self._pipeline = pipeline
        self._llm: Any = None  # built on first use: only the web context step needs one

    def sweep_orphan_entities(self) -> int:
        """Delete KG entities no edge references (see KGClient.delete_orphan_entities)."""
        deleted = int(self._pipeline._kg.delete_orphan_entities(_ORPHAN_MIN_AGE_H))
        if deleted:
            logger.info("Deleted %d orphan KG entities", deleted)
        return deleted

    def sweep_finished_queue_text(self) -> int:
        """Empty the text of extraction-queue rows finished a week ago (see
        ``blank_finished_queue_text``)."""
        blanked = self._db.blank_finished_queue_text(_QUEUE_TEXT_KEEP_DAYS)
        if blanked:
            logger.info("Emptied the text of %d finished queue rows", blanked)
        return blanked

    def release_stale_claims(self) -> int:
        """Return rows a crashed or scaled-down worker left in ``processing`` to pending
        (see ``Database.release_stale_claims``; time-bounded, so live claims are untouched)."""
        released = self._db.release_stale_claims()
        if released:
            logger.info("Periodic sweep released %d stale claim(s)", released)
        return released

    def embed_pending(self, batch_size: int = 500, budget_s: float = 60.0) -> int:
        """Embed unembedded episodes, then chunks, then web chunks (once they have their
        context prefix), until nothing is pending or ``budget_s`` is spent. Returns the
        number embedded.

        The embedder packs requests by tokens, so ``batch_size`` only bounds the rows one
        UPDATE carries. Looping, rather than one small batch per cycle, is what lets a
        history import (tens of thousands of turns) get its vectors in hours instead of
        weeks, while live turns (newest first) never wait behind it.
        """
        embedder = self._pipeline._embedder
        deadline = time.monotonic() + budget_s
        total = 0
        sources: tuple[
            tuple[
                Literal["episodes", "chunks", "web_chunks"], Callable[[int], list[dict[str, Any]]]
            ],
            ...,
        ] = (
            ("episodes", self._db.get_unembedded_episodes),
            ("chunks", self._db.get_unembedded_chunks),
            ("web_chunks", self._db.get_unembedded_web_chunks),
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

    def chunk_web_artifacts(self) -> int:
        """Chunk the captured web pages that have no chunks yet. Returns the chunks built.

        A page enters recall here, in its own good time: chunks, then
        :meth:`contextualize_web_chunks`, then :meth:`embed_pending`. (Pages come from
        ``ingestion.capture``; result lists have no body and are not chunked.)"""
        built = 0
        for page in self._db.artifacts_without_chunks(_WEB_PAGES_PER_CYCLE):
            chunks = chunk_markdown(page["content_markdown"])
            built += self._db.insert_web_chunks(page["id"], page["fetched_at"], chunks)
        if built:
            logger.info("Built %d web chunk(s)", built)
        return built

    def contextualize_web_chunks(self) -> int:
        """Give the new web chunks their context prefix (``ingestion.web_context``), a few
        pages a cycle. Returns the chunks done."""
        from ingestion.llm_client import create_llm_client, stage_model

        if not self._db.artifacts_to_contextualize(1):
            return 0
        if self._llm is None:
            self._llm = create_llm_client()
        return contextualize_pending(self._db, self._llm, stage_model("CONTEXTUALIZE"))

    def enqueue_web_extraction(self) -> int:
        """Queue the embedded chunks of pages marked for KG extraction (a capture rule that
        trusts its source says ``kg_extract``; generic captures are searchable only).
        Returns the number queued."""
        return self._db.enqueue_web_extraction()

    def run(self, stop: threading.Event, interval_seconds: int) -> None:
        """Run a cycle every ``interval_seconds`` until ``stop`` is set.

        Only the replica holding the session-level maintenance advisory lock cycles, so
        ``docker compose up --scale poller=N`` replicas only extract. The others try for the
        lock again each interval and take over when the holder's connection ends. The daily
        sweeps run on the first cycle a replica leads, then once a day."""
        last_daily_sweep: float | None = None
        while not stop.is_set():
            _MAINTENANCE_HEARTBEAT.touch()
            try:
                leader = self._db.try_maintenance_lock()
            except Exception as e:
                logger.error("Maintenance lock attempt failed: %s", e, exc_info=True)
                leader = False
            if leader:
                now = time.monotonic()
                daily = (
                    last_daily_sweep is None or now - last_daily_sweep >= _DAILY_SWEEP_INTERVAL_S
                )
                self.cycle(daily=daily)
                if daily:
                    last_daily_sweep = now
            stop.wait(interval_seconds)

    def cycle(self, daily: bool) -> None:
        """One pass over every stage, each in its own try block: one failing stage must not
        abort the others (embed must run even if chunk rebuild fails, or the unembedded
        backlog grows unbounded). (2026-05-18.)"""
        stages: list[tuple[str, Callable[[], int]]] = [
            ("Chunk rebuild", self.rebuild_pending_chunks),
            ("Web chunking", self.chunk_web_artifacts),
            ("Web context", self.contextualize_web_chunks),
            ("Embed pending", self.embed_pending),
            ("Web extraction enqueue", self.enqueue_web_extraction),
            ("Stale-claim sweep", self.release_stale_claims),
        ]
        if daily:
            stages += [
                ("Orphan entity sweep", self.sweep_orphan_entities),
                ("Queue text sweep", self.sweep_finished_queue_text),
            ]
        for name, stage in stages:
            _MAINTENANCE_HEARTBEAT.touch()
            try:
                stage()
            except Exception as e:
                logger.error("%s failed: %s", name, e, exc_info=True)


class Poller:
    def __init__(
        self,
        db: Database,
        worker_factory: Callable[[], tuple[Database, ExtractionPipeline]],
        maintenance: Maintenance | None = None,
    ) -> None:
        # ``db`` is the extraction thread's connection (claims and releases). Each drain
        # worker thread gets its own (db, pipeline) from ``worker_factory`` (lazily, on its
        # first item): Database holds one non-autocommit connection whose commit/rollback
        # would entangle transactions if shared across threads. ``maintenance`` is the other
        # thread's stack; None runs extraction only.
        self._db = db
        self._worker_factory = worker_factory
        self._maintenance = maintenance
        self._concurrency = _drain_concurrency()
        self._worker_local = threading.local()
        self._drain_pool: ThreadPoolExecutor | None = None
        # SIGTERM/SIGINT set this (see run_loop): the drain finishes its in-flight items,
        # releases the claims it has not started, and both threads exit instead of
        # sleeping — so `docker stop` doesn't force-kill a worker mid-item.
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask both threads to exit at the next item boundary."""
        self._stop_requested.set()

    def drain_extraction_queue(self, batch_limit: int = 30) -> int:
        """Process pending extraction queue items. Returns number processed.

        Uses ``claim_pending_extractions`` so multiple poller replicas can
        run concurrently without racing: each replica gets a distinct slice
        of pending items via ``FOR UPDATE SKIP LOCKED`` and the rows flip
        to ``status='processing'`` for the duration.

        Items run on a thread pool of SYNAPSE_DRAIN_CONCURRENCY threads (issue #13): per-item
        work is ~all LLM network wait, so a thread per in-flight item is ~concurrency-fold
        throughput at the same RAM. ``batch_limit`` bounds how many items one call takes in
        total (so the extraction loop paces itself), while rows are claimed only as threads
        free up. Failure stays item-isolated — one bad item marks failed without touching
        the rest.

        On UsageLimitError (Claude usage cap hit) the offending item and
        every still-claimed item are released back to ``pending`` so another
        replica (or this one on the next cycle) can retry once the cap resets.
        """
        processed = self._drain_concurrent(batch_limit, self._concurrency)
        if processed:
            logger.info("Drained %d extraction items", processed)
        return processed

    def _thread_worker(self) -> tuple[Database, ExtractionPipeline]:
        """This thread's own (db, pipeline), built lazily on first use. The pool's
        threads persist across drain cycles, so connections are reused, not churned."""
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
        re-raised so the extraction loop backs off.
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

    def _extract(self) -> None:
        """The extraction thread: drain, then pause 3s when the drain filled its batch (more
        backlog likely) and 60s otherwise, which eats through a backlog without idle time.

        Rows are claimed only as a worker thread frees up: a claimed row stays 'processing'
        until its item ends, so one that merely waits for a thread would make the priority
        lane laggy and, when the worker is killed mid-item (a deploy, an OOM), be stranded for
        the 45 min stale-claim threshold along with the items that were running. The batch
        limit is how many items one drain call takes before the loop paces itself: two waves
        of the pool."""
        from ingestion.llm_client import UsageLimitError

        batch_limit = 2 * self._concurrency
        quota_backoff = 0  # seconds; grows while the Max quota window is exhausted, 0 when healthy
        while not self._stop_requested.is_set():
            _EXTRACTION_HEARTBEAT.touch()
            drained = 0
            try:
                drained = self.drain_extraction_queue(batch_limit=batch_limit)
                quota_backoff = 0  # a successful drain means the quota window is open again
            except UsageLimitError as e:
                # Max-subscription window exhausted. Claims were already released back to
                # pending by drain(), so nothing is lost. Back off exponentially (5→10→…→30 min,
                # jittered to avoid a thundering herd of 30+ workers) instead of hammering the
                # dead quota every 60s; resume automatically when a drain next succeeds.
                # Only this thread waits: maintenance keeps embedding.
                quota_backoff = min((quota_backoff or 150) * 2, 1800)
                logger.warning(
                    "Quota exhausted; backing off %ds before retry (%s)",
                    quota_backoff,
                    str(e)[:120],
                )
                self._stop_requested.wait(quota_backoff + random.uniform(0, 30))
                continue
            except Exception as e:
                logger.error("Extraction drain failed: %s", e, exc_info=True)
            # Event.wait so a stop request interrupts the sleep.
            self._stop_requested.wait(3 if drained >= batch_limit else 60)

    def run_loop(self, interval_seconds: int = 300) -> None:
        """Run until SIGTERM/SIGINT: the extraction thread, and the maintenance thread
        every ``interval_seconds`` (chunk rebuild, embed, sweeps), each on its own schedule.

        On a stop the in-flight items finish, the claims not yet started go back to pending,
        and both threads are joined before this returns. A thread that dies of an unexpected
        error stops the other and raises, so the container restarts instead of idling with
        half a poller.

        ``SYNAPSE_DRAIN_ONLY=1`` is the retired poller-drain service: a replica that runs
        only the extraction thread. It is honoured for one release, until every deployment
        has re-applied the compose file that drops that service.
        """

        def _on_signal(signum: int, _frame: Any) -> None:
            logger.info(
                "Signal %d received — finishing the in-flight items, then exiting cleanly",
                signum,
            )
            self.request_stop()

        try:
            signal.signal(signal.SIGTERM, _on_signal)
            signal.signal(signal.SIGINT, _on_signal)
        except ValueError:
            # Signal registration only works on the main thread; callers that
            # run the loop elsewhere (tests) drive request_stop() directly.
            pass

        drain_only = os.environ.get("SYNAPSE_DRAIN_ONLY", "").strip() in ("1", "true", "yes")
        if drain_only:
            logger.warning(
                "SYNAPSE_DRAIN_ONLY is deprecated and will be removed: every poller replica "
                "extracts, and one holds the maintenance lock. This replica runs no "
                "maintenance. Re-apply compose.yml with --remove-orphans to drop poller-drain."
            )
        threads: dict[str, Callable[[], None]] = {"extraction": self._extract}
        maintenance = self._maintenance
        if maintenance is not None and not drain_only:
            threads["maintenance"] = lambda: maintenance.run(self._stop_requested, interval_seconds)
        logger.info(
            "Starting poller (threads=%s, maintenance-interval=%ds, drain-concurrency=%d)",
            "+".join(threads),
            interval_seconds,
            self._concurrency,
        )
        # Release any extraction-queue items left in `processing` by a
        # previous run that crashed or was killed mid-extraction. Safe to
        # run on every startup, including concurrent peer starts — Postgres
        # serializes the UPDATE so all replicas converge to the same state.
        released = self._db.release_stale_claims()
        if released:
            logger.info("Released %d stale 'processing' claims back to pending", released)

        crashed: list[str] = []

        def _guarded(name: str, loop: Callable[[], None]) -> None:
            try:
                loop()
            except Exception:
                logger.exception("The %s thread died; stopping the poller", name)
                crashed.append(name)
                self.request_stop()

        running = [
            threading.Thread(target=_guarded, args=(name, loop), name=name)
            for name, loop in threads.items()
        ]
        for t in running:
            t.start()
        for t in running:
            t.join()
        if self._drain_pool is not None:
            self._drain_pool.shutdown()
        if crashed:
            raise RuntimeError(f"poller thread(s) died: {', '.join(crashed)}")
        logger.info("Poller stopped cleanly")


def make_poller(
    db_url: str,
    voyage_api_key: str,
    llm_model: str | None = None,
) -> Poller:
    from ingestion.embedding import create_embedder
    from ingestion.extractor import ExtractionPipeline
    from ingestion.kg_client import KGClient
    from ingestion.llm_client import create_llm_client

    def _stack() -> tuple[Database, ExtractionPipeline]:
        # One thread's own stack (issue #13): Database is single-connection/non-autocommit
        # and the pipeline writes through it, so neither can be shared across threads that
        # run concurrently. All constructors are cheap (lazy PG connects, HTTP clients) —
        # built once per drain pool thread, and once for the maintenance thread.
        db = Database(db_url)
        pipeline = ExtractionPipeline(
            db=db,
            llm_client=create_llm_client(),
            embedder=create_embedder(voyage_api_key=voyage_api_key, db_url=db_url),
            kg_client=KGClient(db_url),
            llm_model=llm_model,
        )
        return db, pipeline

    return Poller(db=Database(db_url), worker_factory=_stack, maintenance=Maintenance(*_stack()))
