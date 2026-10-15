"""Operator dashboard routes (issue #12) — the server half of the /dash surface.

Serves ONE React/esbuild single-page bundle (web/dist, built by the other agent's
web/ tree) plus a small read/flag API over the existing memory tables. The wire
contract is pinned in docs/dashboard-contract.md; this package builds exactly to it.

One module per surface, each mounting its own routes from a ``register(mcp, ...)``:
``static`` (the bundle), ``feed`` (catalog, feed, episode, session), ``stream`` (SSE),
``kg`` (entity dossier, graph explorer), ``search``, ``metrics`` (recall history and the
aggregates), ``curation`` (flags and the two deletes), ``review`` (proposals) and ``records``
(timeline, preferences, dream report, behavior files). ``common`` holds what they share.

Two auth postures, by design (mirrors the contract's Namespace & auth section):
  * The static routes (/dash, /dash/app.js, /dash/assets/*) are UNAUTHENTICATED — the
    bundle is public code carrying no memory, so gating it buys nothing and would break
    the paste-once login screen (which needs to load before it has a token).
  * Every /dash/api/* route rides the ADMIN gate (``http_auth.admin_authorized``: an
    approved full-trust device token, such as the one the dashboard login mints; the
    root token is refused), passed in as ``authorized``. Custom routes bypass FastMCP's
    auth enforcement by design (issue #3704), so each handler is wrapped in
    ``http_helpers.route``, which checks the gate once. 401 body is the contract's
    {"status":"error","detail":"unauthorized"}.

Shape rules shared by every api route: PG work runs in a threadpool on ONE connection
borrowed from the server's pool (ingestion/pg_pool) per request, and the boundary is
fail-soft — ``route`` turns an exception into a 500 JSON and logs it, never raising past
the JSONResponse. Reads are owner-agnostic (an operator sees the whole store); the only
group_id/project/source scoping is the per-endpoint filtering the contract spells out.

Writable surfaces: dashboard_flags + dashboard_audit (schema 042, the soft-flag), and —
since phase 7 — the two DELETE routes that hard-delete an episode (with its content
cascade) or a single KG edge. Everything else is read-only over episodes / the KG /
timeline / preferences / notes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.requests import Request

from mcp_server.dash import curation, feed, kg, metrics, records, review, search, static, stream

logger = logging.getLogger(__name__)


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount /dash (static bundle) + /dash/api/* (gated by ``authorized``, which the server
    wires to the admin gate). No-op w/o DB_URL, matching the sibling route modules."""
    if not db_url:
        logger.info("dashboard routes disabled (no DB_URL)")
        return
    static.register(mcp)
    for module in (feed, stream, kg, search, metrics, curation, review, records):
        module.register(mcp, db_url, authorized)
