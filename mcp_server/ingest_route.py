"""ingest route registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    db_url: Callable[[], str],
    _machine_authorized: Callable[[Request], bool],
) -> Any:
    @mcp.custom_route("/ingest", methods=["POST"])
    async def ingest_turns(request: Request) -> JSONResponse:
        """Direct-push ingest endpoint — replaces the Logfire poll for Claude Code.

        A Claude Code ``Stop`` hook POSTs a bounded TAIL of the session transcript
        (raw JSONL records) here on every turn. We parse with the SAME ``JSONLParser``
        the disk sweep uses, then dedup by span_id (a turn's stable last-record uuid):
        already-stored turns are skipped, new turns append at ``max(sequence)+1``. The
        parser's positional sequence is NOT used as the key — a tail would renumber it
        from 1 and collide — so identity rides on span_id and the sweep/push still
        converge idempotently. A full-transcript POST stays correct too (all old turns
        skip), so this is backward compatible with the pre-tail hook.

        Any client that runs the real ``claude`` CLI fires this same hook.
        Body: {"records": [...], "project": optional, "source": optional,
        "format": optional ("claude_code" default | "codex"), "session_id": optional}.

        ``format: "codex"`` parses the records as Codex CLI rollout lines via
        ``CodexRolloutParser`` instead. A pushed tail usually lacks the rollout's
        session_meta line, so the Codex hook passes ``session_id`` (from the
        rollout filename) as a fallback identity hint. Everything downstream —
        span_id dedup, private sessions, contamination guards — is shared.
        """
        if not _machine_authorized(request):
            return unauthorized()

        from starlette.concurrency import run_in_threadpool

        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON body", 400)

        records = body.get("records")
        if not isinstance(records, list):
            return err("body must contain a 'records' list", 400)
        source_label = body.get("source") or "hook"
        project_override = body.get("project")
        record_format = body.get("format") or "claude_code"
        session_id_hint = body.get("session_id")
        if record_format not in ("claude_code", "codex"):
            return err(f"unknown format {record_format!r}", 400)

        def _work() -> int:
            from ingestion.contamination import is_harness_call, is_transcript_contamination
            from ingestion.db import Database
            from ingestion.jsonl_client import JSONLParser
            from ingestion.models import ExtractionItem
            from ingestion.private_sessions import PrivateSessions
            from ingestion.scratch_projects import is_scratch_project

            if record_format == "codex":
                from ingestion.codex_client import CodexRolloutParser

                episodes = CodexRolloutParser().parse_records(
                    records, source_label, project_override, session_id_hint=session_id_hint
                )
            else:
                episodes = JSONLParser().parse_records(records, source_label, project_override)
            if not episodes:
                return 0
            db = Database(db_url())
            private = PrivateSessions(db)  # per-batch memo; schema/050
            try:
                # The hook ships a bounded TAIL of the transcript, so the parser's
                # positional sequence is meaningless here — turn 50 arrives numbered 1
                # and would collide with the real turn 1. Identity is the span_id (a
                # turn's last record uuid), stable across full and tail parses. Per
                # session, load the stored span_ids + max sequence ONCE, then:
                #   * span_id already stored -> skip wholesale. Idempotent no-op; also
                #     stops a tail's truncated leading fragment (same span_id as the
                #     full turn it tails) from overwriting that turn.
                #   * new span_id            -> append at max_seq+1 and enqueue for KG.
                # Backward compatible with a full-transcript POST (old hook): every old
                # turn skips, only genuinely-new tail turns append at the same numbers
                # they had positionally — same final state, minus the O(turns^2)
                # re-upsert/re-enqueue churn the full re-ship used to cause.
                index: dict[str, tuple[set[str], int]] = {}
                written = 0
                dropped = 0
                private_dropped = 0
                content_dups = 0
                for ep in episodes:
                    # Reject transcribe_ai deposition payloads before they ever land — third-party
                    # PII must not enter memory. Dev conversations about the domain still flow in.
                    # Likewise reject Synapse's own extraction/judge calls (the eval must not
                    # eat itself — see contamination.is_harness_call).
                    if is_transcript_contamination(ep.content) or is_harness_call(ep.content):
                        dropped += 1
                        continue
                    # Throwaway smoke-test workspaces are not a memory source. The KG
                    # extractor cannot tell a story prompt from biography — see
                    # scratch_projects for the fiction that reached the personal graph.
                    if is_scratch_project(ep.project):
                        dropped += 1
                        continue
                    # Private mode (schema/050): the user took this session off the record.
                    # The plugin's hook already stops posting while its local marker exists;
                    # this is the durable half, so a catch-up sweep or a backfill months
                    # later can't ingest what the hook skipped.
                    if private.is_private(ep.session_id):
                        private_dropped += 1
                        continue
                    if ep.session_id not in index:
                        index[ep.session_id] = db.get_session_span_index(ep.session_id)
                    seen, max_seq = index[ep.session_id]
                    if not ep.span_id or ep.span_id in seen:
                        continue  # no identity key, or already stored — skip
                    # Cross-session replay guard (schema 036): a retried session ships the
                    # same turns under a new session id + new span ids, so the span index
                    # above can't see them. SYNAPSE_CONTENT_DEDUP=0 is the kill switch.
                    if os.environ.get(
                        "SYNAPSE_CONTENT_DEDUP", "1"
                    ) != "0" and db.content_dup_exists(ep.project, ep.content):
                        content_dups += 1
                        continue
                    max_seq += 1
                    ep.sequence = max_seq
                    seen.add(ep.span_id)
                    index[ep.session_id] = (seen, max_seq)
                    eid = db.upsert_episode(ep)
                    if ep.content and ep.content.strip():
                        db.enqueue_extraction(
                            ExtractionItem(
                                episode_id=eid,
                                session_id=ep.session_id,
                                content=ep.content,
                                content_type="episode",
                                project=ep.project,
                            )
                        )
                    written += 1
                if dropped:
                    logger.info(
                        "ingest dropped %d transcribe_ai transcript-payload turn(s)", dropped
                    )
                if private_dropped:
                    logger.info("ingest dropped %d private-session turn(s)", private_dropped)
                if content_dups:
                    logger.info(
                        "ingest skipped %d cross-session content-duplicate turn(s)", content_dups
                    )
                return written
            finally:
                db.close()

        try:
            n = await run_in_threadpool(_work)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ingest failed")
            return err(str(exc)[:200], 500)
        return JSONResponse({"status": "ok", "ingested": n})

    return ingest_turns
