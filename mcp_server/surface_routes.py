"""Surface enrollment + registration — the trust seam for audience scoping (053/054).

A surface is one CREDENTIAL that talks to Synapse. Trust is per-credential: ``full``
sees the whole corpus, ``restricted`` sees only ``audience='work-safe'`` notes plus
episodes and timeline events inside its ``allowed_projects``. A caller with no matching
row is restricted with an empty allowlist, which is why these routes only ever GRANT —
the safe state is the one you get by doing nothing.

**Two gates, and the difference between them is the whole design.**

``authorized`` (client) admits the root token or any approved device token. ``admin``
admits ONLY an approved, FULL-TRUST device token — never the root token. Every machine
that installs the plugin has to hold the root token long enough to enroll, so if the
root token could also approve, a machine could self-approve and TOFU would be theatre.
Enrollment can enroll; it cannot bless.

Routes:
  POST   /surfaces/enroll         client  -> {"status", "surface", "token", "pair_code"?}
  POST   /surfaces/approve        admin   -> {"status": "ok", "surface": {...}}
  POST   /surfaces/mint           admin   -> {"status": "ok", "surface": {...}, "token"}
  GET    /surfaces                admin   -> {"status": "ok", "surfaces": [...]}
  PUT    /surfaces/{surface_id}   admin   -> {"status": "ok", "surface": {...}}
  DELETE /surfaces/{surface_id}   admin   -> {"status": "ok", "revoked": N}

**enroll** mints a device token and inserts a row. If this Synapse has no approved
device yet, that first row is approved at full trust — nobody exists who could approve
it, and the branch is unreachable ever after. Every later enrollment lands ``pending``
with a 6-character pair code and is served nothing until approved. The body may carry
the machine's self-declared role (``requested_trust``), which is stored as a REQUEST:
it becomes approve's default and grants nothing on its own.

**approve** takes the pair code and grants a trust level, defaulting to what the device
requested. The code is cleared on success, so it is single-use, and only ``pending``
rows are reachable — a revoked device is never resurrected by replaying its old code.

**mint** is the untrusted-machine path: pre-create an approved token from a machine you
already trust, and carry only that token over. The target machine never holds the
enrollment credential at all.

**PUT** still exists for the CREDENTIAL-LESS ids: ``oauth:<login>`` (the claude.ai
connector authenticates by verified identity, not by a device token) and legacy
hostname rows during the migration window. It cannot touch a device row — promoting a
known token to full trust without going through approve() would reopen the hole.

**DELETE** revokes: status='revoked', token cleared, trust reset to the floor. The row
survives as the audit record of what was once trusted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.surfaces import (
    NoSuchPairCode,
    SchemaMissing,
    approve_surface,
    enroll_surface,
    list_surfaces,
    mint_surface,
    revoke_surface,
    upsert_surface,
)
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_ID_LEN = 200
_MAX_PROJECTS = 100
_MAX_LABEL_LEN = 200


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    admin: Callable[[Request], bool] | None = None,
) -> None:
    """Mount the surface routes.

    ``admin`` defaults to ``authorized`` only so a caller that has not been updated
    still boots; every real deployment passes the stricter gate. Tests that exercise
    the no-self-approve property MUST pass it explicitly.
    """
    if not db_url:
        logger.info("surface routes disabled (no DB_URL)")
        return
    admin_gate = admin if admin is not None else authorized

    def _surface_id(request: Request) -> str:
        return (request.path_params.get("surface_id") or "").strip()

    async def _run(
        request: Request, work: Callable[[], Any], gate: Callable[[Request], bool]
    ) -> JSONResponse | Any:
        """Auth + threadpool, shared by every verb. Returns a JSONResponse on any
        failure, or the raw work() result on success."""
        if not gate(request):
            return unauthorized()
        try:
            return await run_in_threadpool(work)
        except SchemaMissing:
            return err("surfaces table missing (apply schema/053+054)", 503)
        except NoSuchPairCode as e:
            # 404, not 401: the caller is authorized, the CODE is what is wrong. A 401
            # here would send an operator hunting a credential problem that isn't one.
            return err(str(e)[:200], 404)
        except ValueError as e:
            return err(str(e)[:200], 400)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("surface route failed: %s", e)
            return err(str(e)[:200], 500)

    async def _body(request: Request) -> dict[str, Any] | JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON", 400)
        if not isinstance(body, dict):
            return err("body must be a JSON object", 400)
        return body

    def _projects(body: dict[str, Any], key: str = "allowed_projects") -> list[str] | JSONResponse:
        projects = body.get(key) or []
        if not isinstance(projects, list):
            return err(f"{key} must be a list", 400)
        if len(projects) > _MAX_PROJECTS:
            return err(f"{key} capped at {_MAX_PROJECTS}", 400)
        return [str(p) for p in projects]

    # -- enrollment ---------------------------------------------------------

    @mcp.custom_route("/surfaces/enroll", methods=["POST"])  # type: ignore[misc]
    async def surfaces_enroll(request: Request) -> JSONResponse:
        """Join this Synapse: mint a device token for the calling machine.

        Body ``{"label": "<hostname>", "requested_trust": "full"|"restricted",
        "requested_projects": [...]}``.

        The label is DISPLAY ONLY — it is never matched against an existing row, because
        binding an enrollment to a self-reported name is exactly the hostname spoofing
        054 removes ("enroll as `trusted-laptop`, inherit its grant"). The row's real id
        is server-generated.

        ``requested_trust`` is the machine's answer to "personal or work?" from the
        plugin's install prompt. It is a REQUEST: stored on the pending row, offered as
        the default when someone approves, and load-bearing for nothing else. A device
        that declares itself personal is served exactly as much as one that declares
        nothing — which is nothing — until an approved full-trust device approves it.
        """
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        req_projects = _projects(body, "requested_projects")
        if isinstance(req_projects, JSONResponse):
            return req_projects
        label = str(body.get("label") or "")[:_MAX_LABEL_LEN]
        req_trust = str(body.get("requested_trust") or "") or None
        out = await _run(
            request,
            lambda: enroll_surface(db_url, label, req_trust, req_projects),
            authorized,
        )
        if isinstance(out, JSONResponse):
            return out
        surface = out["surface"]
        return JSONResponse(
            {
                "status": surface["status"],  # "approved" (bootstrap) or "pending"
                "surface": surface,
                # Shown ONCE. Only the hash is stored, so nothing can hand it back later.
                "token": out["token"],
                "pair_code": surface.get("pair_code"),
            }
        )

    @mcp.custom_route("/surfaces/approve", methods=["POST"])  # type: ignore[misc]
    async def surfaces_approve(request: Request) -> JSONResponse:
        """Approve a pending device by pair code. ADMIN gate — the root token cannot.

        Body ``{"pair_code": "ABC123", "trust": "full"|"restricted",
        "allowed_projects": [...]}``. ``trust`` and ``allowed_projects`` are OPTIONAL:
        omitted, they fall back to what the device asked for at enrollment (and, for a
        restricted device that named no projects, to the scope other approved restricted
        devices already have). Supplying them overrides the request entirely.

        A device with no stated request lands on ``restricted`` with an empty allowlist,
        which serves nothing — the schema default, the no-row default and this default
        all agree, so an unstated role never becomes a grant by omission.
        """
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        projects = None
        if body.get("allowed_projects") is not None:
            projects = _projects(body)
            if isinstance(projects, JSONResponse):
                return projects
        code = str(body.get("pair_code") or "")
        trust = str(body.get("trust") or "") or None
        out = await _run(
            request, lambda: approve_surface(db_url, code, trust, projects), admin_gate
        )
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface": out})

    @mcp.custom_route("/surfaces/mint", methods=["POST"])  # type: ignore[misc]
    async def surfaces_mint(request: Request) -> JSONResponse:
        """Pre-mint an approved device token for a machine you do not want to trust
        with the enrollment credential. ADMIN gate.

        Body ``{"label": "...", "trust": ..., "allowed_projects": [...]}``.
        """
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        projects = _projects(body)
        if isinstance(projects, JSONResponse):
            return projects
        label = str(body.get("label") or "")[:_MAX_LABEL_LEN]
        trust = str(body.get("trust") or "restricted")
        out = await _run(request, lambda: mint_surface(db_url, label, trust, projects), admin_gate)
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface": out["surface"], "token": out["token"]})

    # -- operator views -----------------------------------------------------

    @mcp.custom_route("/surfaces", methods=["GET"])  # type: ignore[misc]
    async def surfaces_list(request: Request) -> JSONResponse:
        """Every surface — pending first — in one read: "who can see what, and who is
        asking". ADMIN gate: the pair codes in here are approval capabilities."""
        out = await _run(request, lambda: list_surfaces(db_url), admin_gate)
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surfaces": out})

    @mcp.custom_route("/surfaces/{surface_id}", methods=["PUT"])  # type: ignore[misc]
    async def surfaces_put(request: Request) -> JSONResponse:
        """Register or re-register one credential-less surface (``oauth:<login>``, or a
        legacy hostname row). Idempotent; replaces trust + allowlist wholesale, because
        a partial update of a security allowlist is how a stale grant survives a
        demotion. Rejects device rows — those go through approve/mint/revoke."""
        surface_id = _surface_id(request)
        if not surface_id or len(surface_id) > _MAX_ID_LEN:
            return err("surface_id required", 400)
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        projects = _projects(body)
        if isinstance(projects, JSONResponse):
            return projects
        # Default to restricted when the caller omits `trust`: the schema default and
        # the no-row default both say restricted, so the route must not be the one
        # place where a missing field means "full".
        trust = str(body.get("trust") or "restricted")
        out = await _run(
            request, lambda: upsert_surface(db_url, surface_id, trust, projects), admin_gate
        )
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface": out})

    @mcp.custom_route("/surfaces/{surface_id}", methods=["DELETE"])  # type: ignore[misc]
    async def surfaces_delete(request: Request) -> JSONResponse:
        """Revoke a surface: its token stops authenticating on the very next request
        and its trust resets to the floor. The row stays as the audit trail."""
        surface_id = _surface_id(request)
        if not surface_id or len(surface_id) > _MAX_ID_LEN:
            return err("surface_id required", 400)
        out = await _run(request, lambda: revoke_surface(db_url, surface_id), admin_gate)
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse(
            # "deleted" kept alongside "revoked" for the 0.16.x clients still reading it.
            {"status": "ok", "surface_id": surface_id, "revoked": int(out), "deleted": int(out)}
        )
