# mypy: ignore-errors
# Deliberately untyped route-handler helper module (mirrors skill_sync_routes/config_sync_routes,
# already in the mypy pre-commit exclude via mcp_server/*).
"""Shared plain-HTTP route helpers for the machine-token custom routes.

Consolidates the error-envelope + auth/JSON/threadpool boilerplate that the DSN-free
route modules (server, skill_sync, config_sync, timeline, preferences, board, dashboard)
each hand-rolled. The contract's error body is ``{"status": "error", "detail": ...}`` at
some status; ``unauthorized()`` is the 401 special case.

Import discipline: this module imports ONLY starlette + stdlib, never any other mcp_server
module, so it can be imported from all of them without a cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def err(detail: str, status: int) -> JSONResponse:
    """The shared error envelope: ``{"status": "error", "detail": detail}`` at ``status``."""
    return JSONResponse({"status": "error", "detail": detail}, status_code=status)


def unauthorized() -> JSONResponse:
    """The 401 every machine-token-gated route returns when the token is absent/wrong."""
    return err("unauthorized", 401)


def _iso(dt: Any) -> str | None:
    """A datetime's ISO string, or None when it's absent."""
    return dt.isoformat() if dt is not None else None


async def guarded_json(
    request: Request,
    authorized: Callable[[Request], bool],
    work: Callable[[Any], Any],
    *,
    label: str = "route",
) -> JSONResponse:
    """Auth-gate + JSON-parse + threadpool boundary shared by the skill/config sync routes.

    Gate on ``authorized(request)`` (401), parse the JSON body (400 on failure), run
    ``work(body)`` in a threadpool (500 + logged on exception), else return ``JSONResponse(out)``.
    ``label`` names the route in the 500 log line (e.g. 'skill route' / 'config route')."""
    if not authorized(request):
        return unauthorized()
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON", 400)
    try:
        out = await run_in_threadpool(work, body)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("%s failed", label)
        return err(str(exc)[:200], 500)
    return JSONResponse(out)
