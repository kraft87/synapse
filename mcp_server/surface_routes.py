"""Surface-registration routes — the operator seam for audience scoping (schema 053).

A surface is one host that talks to Synapse, identified by the plugin's
``SYNAPSE_SURFACE`` / hostname constant. Trust is per-surface: ``full`` sees the whole
corpus, ``restricted`` sees only ``audience='work-safe'`` notes plus episodes and
timeline events inside its ``allowed_projects``. A host with NO row is restricted with
an empty allowlist, which is why these routes exist only to GRANT — the safe state is
the one you get by doing nothing.

Routes (machine-token gated, same lane as the private-session / preferences routes):
  GET    /surfaces                -> {"status": "ok", "surfaces": [...]}
  PUT    /surfaces/{surface_id}   -> {"status": "ok", "surface": {...}}
  DELETE /surfaces/{surface_id}   -> {"status": "ok", "deleted": N}

PUT takes ``{"trust": "full"|"restricted", "allowed_projects": [...]}`` and REPLACES
both fields. Full replacement rather than a patch is deliberate: a partial update of a
security allowlist is exactly how a stale grant survives a demotion. DELETE reverts a
surface to the unregistered default (restricted, empty) — always a tightening, so it
needs no confirmation dance.

These routes deliberately do not authenticate the surface itself. Surface ids are
self-reported under the shared machine token; the threat model is an employer READING a
transcript, not an employer attacking the API. Per-surface tokens through the existing
device-flow lane are the answer for the day that changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.surfaces import (
    SchemaMissing,
    delete_surface,
    list_surfaces,
    upsert_surface,
)
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_ID_LEN = 200
_MAX_PROJECTS = 100


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    if not db_url:
        logger.info("surface routes disabled (no DB_URL)")
        return

    def _surface_id(request: Request) -> str:
        return (request.path_params.get("surface_id") or "").strip()

    async def _run(request: Request, work: Callable[[], Any]) -> JSONResponse | Any:
        """Auth + threadpool, shared by the three verbs. Returns a JSONResponse on any
        failure, or the raw work() result on success."""
        if not authorized(request):
            return unauthorized()
        try:
            return await run_in_threadpool(work)
        except SchemaMissing:
            return err("surfaces table missing (apply schema/053)", 503)
        except ValueError as e:
            return err(str(e)[:200], 400)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("surface route failed: %s", e)
            return err(str(e)[:200], 500)

    @mcp.custom_route("/surfaces", methods=["GET"])  # type: ignore[misc]
    async def surfaces_list(request: Request) -> JSONResponse:
        """Every registered surface — "which hosts can see what", in one read."""
        out = await _run(request, lambda: list_surfaces(db_url))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surfaces": out})

    @mcp.custom_route("/surfaces/{surface_id}", methods=["PUT"])  # type: ignore[misc]
    async def surfaces_put(request: Request) -> JSONResponse:
        """Register or re-register one surface. Idempotent; replaces trust + allowlist."""
        if not authorized(request):
            return unauthorized()
        surface_id = _surface_id(request)
        if not surface_id or len(surface_id) > _MAX_ID_LEN:
            return err("surface_id required", 400)
        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON", 400)
        if not isinstance(body, dict):
            return err("body must be a JSON object", 400)
        # Default to restricted when the caller omits `trust`: the schema default and the
        # no-row default both say restricted, so the route must not be the one place
        # where a missing field means "full".
        trust = str(body.get("trust") or "restricted")
        projects = body.get("allowed_projects") or []
        if not isinstance(projects, list):
            return err("allowed_projects must be a list", 400)
        if len(projects) > _MAX_PROJECTS:
            return err(f"allowed_projects capped at {_MAX_PROJECTS}", 400)

        out = await _run(request, lambda: upsert_surface(db_url, surface_id, trust, projects))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface": out})

    @mcp.custom_route("/surfaces/{surface_id}", methods=["DELETE"])  # type: ignore[misc]
    async def surfaces_delete(request: Request) -> JSONResponse:
        """Unregister a surface — it falls back to restricted with an empty allowlist."""
        if not authorized(request):
            return unauthorized()
        surface_id = _surface_id(request)
        if not surface_id or len(surface_id) > _MAX_ID_LEN:
            return err("surface_id required", 400)
        out = await _run(request, lambda: delete_surface(db_url, surface_id))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface_id": surface_id, "deleted": int(out)})
