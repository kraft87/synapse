"""recall route registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import logfire
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.scope import coerce_group
from ingestion.surfaces import SurfaceTrust
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    _get_recall: Callable[[], Any],
    _machine_authorized: Callable[[Request], bool],
    _request_trust: Callable[[Request, str | None], SurfaceTrust],
) -> Any:
    @mcp.custom_route("/recall", methods=["POST"])
    async def recall_http(request: Request) -> JSONResponse:
        """Plain-HTTP recall for non-MCP callers — the auto-recall memory hook.

        A shell/Python hook (UserPromptSubmit) has no MCP client: command hooks talk
        stdout/exit-code only and can't invoke an MCP tool. This route gives them the
        same recall() over plain HTTP, reusing the warm process-singleton engine
        (loaded embedder + pooled PG connections + warm HNSW cache) so a per-turn hook
        stays fast instead of cold-starting the pipeline each call.

        Body: {"query": str, "project"?: str, "group_id"?: str, "write_feedback"?: bool,
               "source"?: str, "debug"?: bool, "surface"?: str}.
        The caller gets the same enforcement the MCP tools do: a DEVICE token names its own
        surface and ``surface`` is ignored; a root-token caller may still pass the
        deprecated ``surface`` param for the migration window. Holding a credential is not
        the same as being trusted — the token says "a Synapse client", the surface row says
        what that client may read.
        write_feedback defaults FALSE here: automatic recalls must not bump the
        retrieval-count feedback signal (bench-grade discipline) — and the phase-2
        dashboard debug console relies on this default staying false so its diagnostic
        recalls never pollute the feedback signal. ``debug`` attaches the per-leg timing /
        pool-size / rerank envelope the engine already measures (see recall(debug=...)).
        Fail-soft like /ingest — never raises past the JSONResponse boundary.
        """
        if not _machine_authorized(request):
            return unauthorized()

        from starlette.concurrency import run_in_threadpool

        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON body", 400)

        query = (body.get("query") or "").strip()
        if not query:
            return err("missing 'query'", 400)
        project = body.get("project") or None
        group_id = coerce_group(body.get("group_id")) or "technical"
        write_feedback = bool(body.get("write_feedback", False))
        source = body.get("source") or "http"
        debug = bool(body.get("debug", False))
        trust = _request_trust(request, body.get("surface") or None)

        def _work() -> dict:
            return _get_recall().recall(
                query=query,
                project=project,
                group_id=group_id,
                write_feedback=write_feedback,
                source=source,
                debug=debug,
                trust=trust,
            )

        try:
            with logfire.span("http.recall {query!r}", query=query[:80], group_id=group_id):
                out = await run_in_threadpool(_work)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("http recall failed")
            return err(str(exc)[:200], 500)
        return JSONResponse(out)

    return recall_http
