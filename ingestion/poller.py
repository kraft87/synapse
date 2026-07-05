"""Maintenance loop — builds chunks, embeds pending docs, and drains the
extraction queue to populate the Postgres KG.

Episodes arrive via the ``/ingest`` push hook (``mcp_server.server``), not a
Logfire poll. This process owns the post-ingest work: chunk construction,
embeddings, and KG extraction (facts are extracted from chunks)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ingestion.db import Database
from ingestion.models import ExtractionItem

if TYPE_CHECKING:
    from ingestion.extractor import ExtractionPipeline

logger = logging.getLogger(__name__)


class Poller:
    def __init__(
        self,
        db: Database,
        extraction_pipeline: ExtractionPipeline | None = None,
    ) -> None:
        self._db = db
        self._extraction = extraction_pipeline

    def embed_pending(self, batch_size: int = 96) -> int:
        """Embed unembedded episodes and chunks.

        Returns total items embedded.
        """
        if self._extraction is None:
            return 0

        embedder = self._extraction._embedder
        total = 0

        episodes = self._db.get_unembedded_episodes(limit=batch_size)
        if episodes:
            texts = [e["content"] for e in episodes]
            embeddings = embedder.embed(texts, task="document")
            for ep, emb in zip(episodes, embeddings, strict=True):
                self._db.set_episode_embedding(ep["id"], emb)
            total += len(episodes)
            logger.info("Embedded %d episodes", len(episodes))

        # Chunks are ~4x longer than episodes — use smaller batch to stay under token limit
        chunk_batch = max(1, batch_size // 4)
        chunks = self._db.get_unembedded_chunks(limit=chunk_batch)
        if chunks:
            texts = [c["content"] for c in chunks]
            embeddings = embedder.embed(texts, task="document")
            for chunk, emb in zip(chunks, embeddings, strict=True):
                self._db.set_chunk_embedding(chunk["id"], emb)
            total += len(chunks)
            logger.info("Embedded %d chunks", len(chunks))

        synth_docs = self._db.get_unembedded_synth_docs(limit=chunk_batch)
        if synth_docs:
            texts = [d["content"] for d in synth_docs]
            embeddings = embedder.embed(texts, task="document")
            for doc, emb in zip(synth_docs, embeddings, strict=True):
                self._db.set_synth_doc_embedding(doc["id"], emb)
            total += len(synth_docs)
            logger.info("Embedded %d synth documents", len(synth_docs))

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
                    content=chunk["content"],
                    content_type="chunk",
                    project=chunk["project"],
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

        On UsageLimitError (Claude usage cap hit) the offending item and
        every still-claimed item in the current batch are released back to
        ``pending`` so another replica (or this one on the next cycle) can
        retry once the cap resets.
        """
        if self._extraction is None:
            return 0

        import logfire

        from ingestion.llm_client import UsageLimitError

        items = self._db.claim_pending_extractions(limit=batch_limit)
        if not items:
            return 0

        unfinished: list[int] = [int(item["id"]) for item in items]
        processed = 0
        for item in items:
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

        if processed:
            logger.info("Drained %d extraction items", processed)
        return processed

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

        from ingestion.llm_client import UsageLimitError

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
        # Small batches on purpose: a worker holds claimed rows in 'processing' for the
        # whole batch (~per-item time each, serial), so a LARGE batch (a) makes the priority
        # lane laggy — a worker won't re-check for new high-priority ingest until its batch
        # drains — and (b) keeps legit in-progress rows 'processing' long enough that the
        # stale-claim sweep would wrongly release them. 8 items x ~3-4 min ~= a ~30 min batch:
        # new ingest preempts within one batch, and the 45 min stale threshold stays safe.
        drain_batch_limit = 8
        prev_drained = 0
        quota_backoff = 0  # seconds; grows while the Max quota window is exhausted, 0 when healthy
        while True:
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
                time.sleep(quota_backoff + _random.uniform(0, 30))
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
            # Fast loop when extraction has likely-more-pending, slow loop otherwise.
            time.sleep(3 if drained >= drain_batch_limit else 60)


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
        kg_client=KGClient(),
        llm_model=llm_model,
    )
    return Poller(db=db, extraction_pipeline=pipeline)
