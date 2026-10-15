"""ingest route registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.project_label import clean_label
from ingestion.surfaces import SurfaceTrust
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    db_url: Callable[[], str],
    _machine_authorized: Callable[[Request], bool],
    _request_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> Any:
    """``_request_trust`` resolves the caller from its BEARER (schema 054); its
    credential-bound surface id is stamped on every episode written (schema 057). Omitted,
    or for the root token / an unknown credential, episodes are stamped NULL."""

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
        rollout filename) as a fallback identity hint. Everything downstream — span_id
        dedup, contamination and scratch guards, private sessions, delete tombstones and
        the cross-session replay guard — is ``ingestion.episode_writer``, shared with
        every backfill.

        Provenance (schema 057): each stored turn is stamped with the surface the BEARER
        resolves to, which is what lets a restricted device read its own uploads back.
        Nothing in the body can set or influence that stamp.
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
        # The client resolves the label (marker file, git toplevel, folder name); without
        # one the parser labels from the transcript's cwd. Either way, never a path.
        project_override = clean_label(body.get("project"))
        # A batch spanning several projects (a session that moved between repositories)
        # carries each cwd's label instead; every turn is then labelled from its own cwd.
        raw_by_cwd = body.get("cwd_projects")
        project_by_cwd = {
            c: lb
            for c, v in (raw_by_cwd.items() if isinstance(raw_by_cwd, dict) else ())
            if isinstance(c, str) and (lb := clean_label(v))
        }
        record_format = body.get("format") or "claude_code"
        session_id_hint = body.get("session_id")
        if record_format not in ("claude_code", "codex"):
            return err(f"unknown format {record_format!r}", 400)

        def _work() -> int:
            from ingestion.db import Database
            from ingestion.episode_writer import store_episodes
            from ingestion.jsonl_client import JSONLParser

            if record_format == "codex":
                from ingestion.codex_client import CodexRolloutParser

                episodes = CodexRolloutParser().parse_records(
                    records, source_label, project_override, session_id_hint=session_id_hint
                )
            else:
                episodes = JSONLParser().parse_records(
                    records, source_label, project_override, project_by_cwd=project_by_cwd or None
                )
            if not episodes:
                return 0
            # Resolved from the credential only. Assigned unconditionally so nothing a
            # parser might ever lift out of the records can survive as a stamp.
            stamp = _request_trust(request).stamp_surface_id if _request_trust else None
            for ep in episodes:
                ep.surface_id = stamp
            db = Database(db_url())
            try:
                # Every guard, the span dedup, the append at max(sequence)+1 and the
                # extraction enqueue live in ingestion.episode_writer, shared with the
                # backfills; the POST is one transaction on one connection.
                stats = store_episodes(db, episodes)
            finally:
                db.close()
            stats.log_skipped()
            return stats.written

        try:
            n = await run_in_threadpool(_work)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ingest failed")
            return err(str(exc)[:200], 500)
        return JSONResponse({"status": "ok", "ingested": n})

    return ingest_turns
