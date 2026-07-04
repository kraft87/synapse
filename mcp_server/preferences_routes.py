"""Preferences read route — the session-start block's server seam.

The plugin's SessionStart hook is a THIN CLIENT (no DSN, no Voyage key): it GETs the
top standing preferences here and prints a bounded block into context. The server owns
the DB. Mirrors the timeline milestones route: machine-token auth (custom routes bypass
FastMCP's auth middleware by design), PG work in a threadpool, fail-soft JSON.

Route (GET):
  /preferences/top?limit=8  -> {"status": "ok", "items": [{pref, polarity, assert_count, since}]}

Ranked by (assert_count DESC, last_asserted DESC): the strongest, most-recently-reasserted
preferences first. Across ALL groups for the owner — a standing preference ("never use
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

logger = logging.getLogger(__name__)

# Single-owner constant, mirroring kg_pg_write.OWNER / the recall leg.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")
_MAX_LIMIT = 50


def _top_preferences(db_url: str, limit: int) -> list[dict[str, Any]]:
    """Live preferences for the session-start block, strongest first. Degrades to []
    if migration 035 hasn't been applied on this deployment yet."""
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        rows = conn.execute(
            "SELECT pref, polarity, assert_count, left(first_seen::text, 10) AS since "
            "FROM preferences WHERE owner_id = %s AND t_invalid IS NULL "
            "ORDER BY assert_count DESC, last_asserted DESC LIMIT %s",
            (_OWNER, limit),
        ).fetchall()
        return [{"pref": r[0], "polarity": r[1], "assert_count": r[2], "since": r[3]} for r in rows]
    except psycopg.errors.UndefinedTable:
        return []
    finally:
        conn.close()


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    if not db_url:
        logger.info("preferences routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/preferences/top", methods=["GET"])  # type: ignore[misc]
    async def preferences_top(request: Request) -> JSONResponse:
        """Top standing user preferences for the plugin's session-start block. Bounded
        and time-agnostic by design — a small factual block, not query-blind recall."""
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            limit = min(int(request.query_params.get("limit", "8")), _MAX_LIMIT)
        except (TypeError, ValueError):
            limit = 8
        limit = max(limit, 1)
        try:
            items = await run_in_threadpool(_top_preferences, db_url, limit)
        except Exception as e:
            logger.warning("preferences top failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        return JSONResponse({"status": "ok", "items": items})
