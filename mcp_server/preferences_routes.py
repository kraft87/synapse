"""Preferences read route — the session-start block's server seam.

The plugin's SessionStart hook is a THIN CLIENT (no DSN, no Voyage key): it GETs the
top standing preferences here and prints a bounded block into context. The server owns
the DB. Mirrors the timeline milestones route: machine-token auth (custom routes bypass
FastMCP's auth middleware by design), PG work in a threadpool, fail-soft JSON.

Route (GET):
  /preferences/top?limit=8  -> {"status": "ok", "items": [{pref, polarity, assert_count, since}]}

Ranked by (assert_count DESC, last_asserted DESC): the strongest, most-recently-reasserted
preferences first. A restricted caller (schema 053/055) gets only preferences from its
project allowlist or its own ingested turns; an unknown caller gets none. Across ALL groups for the owner — a standing preference ("never use
tables") shapes every session, not just its originating project's.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

# Single-owner constant, mirroring kg_pg_write.OWNER / the recall leg.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")
_MAX_LIMIT = 50


#: Restricted-surface scope for preferences. The table has no surface_id of its own; its
#: provenance is ``source_ref = 'ep:<id>'``, so "own" means "first asserted in a turn this
#: surface ingested" (schema 055). Literal SQL; only the values are bound.
_SCOPE_SQL = (
    " AND (project = ANY(%s) OR source_ref IN ("
    "SELECT 'ep:' || id FROM episodes WHERE surface_id = %s))"
)


def _top_preferences(
    db_url: str,
    limit: int,
    allowed_projects: list[str] | None = None,
    own_surface: str | None = None,
) -> list[dict[str, Any]]:
    """Live preferences for the session-start block, strongest first. Degrades to []
    if migration 035 hasn't been applied on this deployment yet.

    ``allowed_projects`` is the restricted caller's allowlist (None = full trust, no
    filter). Preferences are mined from conversation, so a restricted host gets only
    those from an allowlisted project or first asserted in a turn it ingested itself
    (``own_surface``). An empty allowlist with no own surface matches nothing: a
    NULL-project preference has no provenance to clear it, same as the board digest."""
    sql = (
        "SELECT pref, polarity, assert_count, left(first_seen::text, 10) AS since "
        "FROM preferences WHERE owner_id = %s AND t_invalid IS NULL"
    )
    params: list[Any] = [_OWNER]
    if allowed_projects is not None:
        sql += _SCOPE_SQL
        # surface_id = NULL never matches, so a caller with no own surface gets the
        # allowlist alone.
        params.extend([allowed_projects, own_surface])
    sql += " ORDER BY assert_count DESC, last_asserted DESC LIMIT %s"
    params.append(limit)
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [{"pref": r[0], "polarity": r[1], "assert_count": r[2], "since": r[3]} for r in rows]
    except psycopg.errors.UndefinedTable:
        return []
    finally:
        conn.close()


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> None:
    """``resolve_trust`` maps a request to its bearer-resolved verdict. Not supplied, every
    caller is treated as an unknown surface and served nothing (fail closed)."""
    if not db_url:
        logger.info("preferences routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/preferences/top", methods=["GET"])  # type: ignore[misc]
    async def preferences_top(request: Request) -> JSONResponse:
        """Top standing user preferences for the plugin's session-start block. Bounded
        and time-agnostic by design — a small factual block, not query-blind recall."""
        if not authorized(request):
            return unauthorized()
        try:
            limit = min(int(request.query_params.get("limit", "8")), _MAX_LIMIT)
        except (TypeError, ValueError):
            limit = 8
        limit = max(limit, 1)
        try:
            st = (
                await run_in_threadpool(resolve_trust, request)
                if resolve_trust
                else UNKNOWN_SURFACE
            )
            items = await run_in_threadpool(
                _top_preferences, db_url, limit, st.project_filter, st.own_surface
            )
        except Exception as e:
            logger.warning("preferences top failed: %s", e)
            return err(str(e)[:200], 500)
        return JSONResponse({"status": "ok", "items": items})
