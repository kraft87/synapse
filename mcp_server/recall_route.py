"""recall route registration with explicit runtime dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import logfire
from fastmcp import FastMCP
from starlette.requests import Request

from ingestion.scope import coerce_group
from ingestion.surfaces import SurfaceTrust, own_surface_served
from mcp_server.caller_trust import note_ignored_surface
from mcp_server.http_helpers import RouteError, route
from mcp_server.recall_warnings import serving_notice, with_notice


def register(
    mcp: FastMCP,
    _get_recall: Callable[[], Any],
    _machine_authorized: Callable[[Request], bool],
    _request_trust: Callable[[Request], SurfaceTrust],
    _authenticated: Callable[[Request], bool],
) -> Any:
    @mcp.custom_route("/recall", methods=["POST"])
    @route(_machine_authorized, body=True)
    def recall_http(request: Request, body: dict[str, Any]) -> Any:
        """Plain-HTTP recall for non-MCP callers (the dashboard's Recall console, scripts).

        Gives a caller with no MCP client the same recall() over plain HTTP, reusing the warm
        process-singleton engine (loaded embedder + pooled PG connections + warm HNSW cache).

        Body: {"query": str, "project"?: str, "group_id"?: str, "write_feedback"?: bool,
               "source"?: str, "debug"?: bool}.
        The caller gets the same enforcement the MCP tools do, resolved from the bearer: a
        DEVICE token names its own surface, and the root token resolves to unknown
        (restricted, empty allowlist). A ``surface`` body field is ignored. Holding a
        credential is not the same as being trusted — the token says "a Synapse client",
        the surface row says what that client may read.
        write_feedback defaults FALSE here: HTTP recalls are not agent-served, so they
        must not bump the served counters, and the dashboard's debug console relies on
        this default so its diagnostic recalls never pollute them. ``debug`` attaches the per-leg timing /
        pool-size / rerank envelope the engine already measures (see recall(debug=...)).
        Fail-soft like /ingest — never raises past the JSONResponse boundary.
        An authenticated caller whose verdict serves (almost) nothing gets the reason
        first in ``warnings`` (recall_warnings.serving_notice), same as the MCP tools.
        """
        query = (body.get("query") or "").strip()
        if not query:
            raise RouteError("missing 'query'")
        project = body.get("project") or None
        group_id = coerce_group(body.get("group_id")) or "technical"
        write_feedback = bool(body.get("write_feedback", False))
        source = body.get("source") or "http"
        debug = bool(body.get("debug", False))
        note_ignored_surface(body.get("surface"), "http.recall")
        trust = _request_trust(request)
        told = _authenticated(request)

        with logfire.span("http.recall {query!r}", query=query[:80], group_id=group_id):
            engine = _get_recall()
            result = engine.recall(
                query=query,
                project=project,
                group_id=group_id,
                write_feedback=write_feedback,
                source=source,
                debug=debug,
                trust=trust,
            )
            notice = None
            if told:
                own = own_surface_served(getattr(engine, "_db_url", ""), trust)
                notice = serving_notice(trust, own)
        return with_notice(result, notice)

    return recall_http
