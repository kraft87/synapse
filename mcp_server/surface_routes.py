"""Surface enrollment + registration — the trust seam for audience scoping (053/054).

A surface is one CREDENTIAL that talks to Synapse. Trust is per-credential: ``full``
sees the whole corpus, ``restricted`` sees only ``audience='work-safe'`` notes plus
episodes and timeline events inside its ``allowed_projects``. A caller with no matching
row is restricted with an empty allowlist, which is why these routes only ever GRANT —
the safe state is the one you get by doing nothing.

**Enrollment is anchored to the owner's identity, not to a shared secret.** A new
machine gets its device token by completing the IdP's device flow (RFC 8628 — a short
code approved in a browser on any device, the same lane ``synapse login`` already uses)
and passing the resulting ``device_code`` here. The server polls the IdP, reads the
identity, and enforces the same allowlist as every other login. Only then does it mint.

That is why there is no pair code and no pending state. Trust-on-first-use exists to
compensate for an ANONYMOUS enrollment credential — if all a new machine can prove is
"I hold the shared token", something else has to vouch for it, so you get a pending row
and a second device that approves it. But an OAuth-verified owner standing at the new
machine has already answered the only question that ceremony was asking. A second
confirmation after an authenticated request confirms nothing, and a pending state is a
state in which a laptop silently has no memory. Accepted in the threat model: theft of
the owner's IdP credential (mitigated by the IdP's own 2FA), which would in any case be
enough to reach the dashboard and read everything directly.

**The root machine token cannot enroll.** It is the services' credential for ``/ingest``
and the internal write paths, and every machine that ever ran the plugin has held it. A
credential that widespread must not be able to create a *new* credential.

Routes:
  POST   /surfaces/enroll         OAuth device_code -> {"status":"ok","surface":{...},"token"}
  POST   /surfaces/mint           admin             -> {"status":"ok","surface":{...},"token"}
  GET    /surfaces                admin             -> {"status":"ok","surfaces":[...]}
  PUT    /surfaces/{surface_id}   admin             -> {"status":"ok","surface":{...}}
  DELETE /surfaces/{surface_id}   admin             -> {"status":"ok","revoked":N}

**enroll** takes ``{device_code, label, trust, allowed_projects}``. ``label`` is the
hostname and is DISPLAY ONLY — never matched against an existing row, because keying an
enrollment on a self-reported name is exactly the hostname spoofing 054 removes ("enroll
as `trusted-laptop`, inherit its grant"). ``trust`` is the answer to the plugin's
install-time "personal or work?" prompt, and it is authoritative here because the person
answering it just authenticated. Unstated ⇒ ``restricted``: a client that did not ask
the question must not resolve it to full access.

**mint** is the same operation for a machine that will never run a browser flow (a
headless box, a service). ``admin`` gated, so it needs a machine that is already trusted.

**PUT** exists for the CREDENTIAL-LESS ids: ``oauth:<login>`` (the claude.ai connector
authenticates by verified identity, not by a device token) and legacy hostname rows
during the migration window. It cannot touch a device row — handing an already-issued
token full trust by id would bypass enrollment entirely.

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
    SchemaMissing,
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

#: IdP device-flow states that mean "keep polling", passed straight back to the client
#: so it can distinguish "not yet" from "no".
_PENDING_ERRORS = {"authorization_pending", "slow_down"}


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    admin: Callable[[Request], bool] | None = None,
    idp: Any = None,
) -> None:
    """Mount the surface routes.

    ``admin`` defaults to ``authorized`` only so an un-updated caller still boots; every
    real deployment passes the stricter gate. ``idp`` enables ``/surfaces/enroll``:
    without an identity provider there is no identity to anchor an enrollment to, so
    that route reports 503 and the operator uses ``scripts/surface_admin.py mint``.
    """
    if not db_url:
        logger.info("surface routes disabled (no DB_URL)")
        return
    admin_gate = admin if admin is not None else authorized

    def _surface_id(request: Request) -> str:
        return (request.path_params.get("surface_id") or "").strip()

    async def _run(
        request: Request, work: Callable[[], Any], gate: Callable[[Request], bool] | None
    ) -> JSONResponse | Any:
        """Auth + threadpool, shared by every verb. Returns a JSONResponse on any
        failure, or the raw work() result on success. ``gate=None`` means the caller has
        already authenticated by another means (the enroll route's identity check)."""
        if gate is not None and not gate(request):
            return unauthorized()
        try:
            return await run_in_threadpool(work)
        except SchemaMissing:
            return err("surfaces table missing (apply schema/053+054)", 503)
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

    def _projects(body: dict[str, Any]) -> list[str] | None | JSONResponse:
        """The allowlist the caller stated, or None for "unstated".

        The distinction is load-bearing: unstated lets a restricted surface inherit the
        scope other restricted surfaces already have, while an explicit ``[]`` narrows
        it to nothing. An operator has to be able to say "no projects" and mean it.
        """
        if body.get("allowed_projects") is None:
            return None
        projects = body["allowed_projects"]
        if not isinstance(projects, list):
            return err("allowed_projects must be a list", 400)
        if len(projects) > _MAX_PROJECTS:
            return err(f"allowed_projects capped at {_MAX_PROJECTS}", 400)
        return [str(p) for p in projects]

    def _grant(body: dict[str, Any]) -> tuple[str | None, list[str] | None] | JSONResponse:
        projects = _projects(body)
        if isinstance(projects, JSONResponse):
            return projects
        return (str(body.get("trust") or "") or None), projects

    # -- enrollment ---------------------------------------------------------

    @mcp.custom_route("/surfaces/enroll", methods=["POST"])  # type: ignore[misc]
    async def surfaces_enroll(request: Request) -> JSONResponse:
        """Join this Synapse: prove an allowlisted identity, get this device's token.

        Body ``{"device_code": "...", "label": "<hostname>", "trust": "full"|"restricted",
        "allowed_projects": [...]}``.

        ``device_code`` comes from ``POST /device/code`` — the client prints the short
        user code, the human approves it in a browser anywhere (phone included), and the
        client polls here. Each poll re-polls the IdP, so ``authorization_pending`` and
        ``slow_down`` come straight back and the client keeps waiting. Approval plus an
        allowlisted identity is what authorizes the mint; the machine token is refused
        outright, because the credential every machine already holds must not be able to
        create new credentials.
        """
        if idp is None:
            return err(
                "enrollment needs an identity provider (GitHub or OIDC); "
                "use scripts/surface_admin.py mint on the database host instead",
                503,
            )
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        device_code = str(body.get("device_code") or "")
        if not device_code:
            return err("device_code required — start with POST /device/code", 400)
        grant = _grant(body)
        if isinstance(grant, JSONResponse):
            return grant
        trust, projects = grant
        label = str(body.get("label") or "")[:_MAX_LABEL_LEN]

        # Identity first, always. Nothing below this point runs for a caller the IdP
        # and the allowlist have not both admitted.
        try:
            tok = await idp.device_poll(device_code)
        except Exception as e:
            logger.warning("enroll: %s device poll failed: %s", idp.label, e)
            return err(f"{idp.label} device poll failed", 502)
        access = tok.get("access_token")
        if not access:
            reason = tok.get("error", "authorization_pending")
            # 202 for "still waiting" so a polling client can tell it apart from a
            # refusal without parsing prose; 403 once the IdP has said no.
            return JSONResponse(
                {"status": "pending", "error": reason},
                status_code=202 if reason in _PENDING_ERRORS else 403,
            )
        try:
            identity = await idp.fetch_identity(access)
        except Exception as e:
            logger.warning("enroll: %s identity lookup failed: %s", idp.label, e)
            return err(f"{idp.label} identity lookup failed", 502)
        if not identity or identity not in idp.allowed:
            logger.warning("enroll: %s user %r not in allowlist", idp.label, identity)
            return err(f"{idp.label} user {identity!r} not in allowlist", 403)

        out = await _run(request, lambda: enroll_surface(db_url, label, trust, projects), None)
        if isinstance(out, JSONResponse):
            return out
        logger.info(
            "device enrolled by %s user %r: %s trust=%s",
            idp.label,
            identity,
            out["surface"]["surface_id"],
            out["surface"]["trust"],
        )
        return JSONResponse(
            {
                "status": "ok",
                "surface": out["surface"],
                # Shown ONCE. Only the hash is stored, so nothing can hand it back later.
                "token": out["token"],
                "login": identity,
            }
        )

    @mcp.custom_route("/surfaces/mint", methods=["POST"])  # type: ignore[misc]
    async def surfaces_mint(request: Request) -> JSONResponse:
        """Create a device token for a machine that will never run a browser flow —
        a headless box, a service, a container. ADMIN gate: it needs a machine that is
        already trusted, since there is no identity in the request to anchor on.

        Body ``{"label": "...", "trust": ..., "allowed_projects": [...]}``.
        """
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        grant = _grant(body)
        if isinstance(grant, JSONResponse):
            return grant
        trust, projects = grant
        label = str(body.get("label") or "")[:_MAX_LABEL_LEN]
        out = await _run(request, lambda: mint_surface(db_url, label, trust, projects), admin_gate)
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surface": out["surface"], "token": out["token"]})

    # -- operator views -----------------------------------------------------

    @mcp.custom_route("/surfaces", methods=["GET"])  # type: ignore[misc]
    async def surfaces_list(request: Request) -> JSONResponse:
        """Every surface — "who can see what", in one read. ADMIN gated: the listing is
        the map of what each credential reaches."""
        out = await _run(request, lambda: list_surfaces(db_url), admin_gate)
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "surfaces": out})

    @mcp.custom_route("/surfaces/{surface_id}", methods=["PUT"])  # type: ignore[misc]
    async def surfaces_put(request: Request) -> JSONResponse:
        """Register or re-register one credential-less surface (``oauth:<login>``, or a
        legacy hostname row). Idempotent; replaces trust + allowlist wholesale, because
        a partial update of a security allowlist is how a stale grant survives a
        demotion. Rejects device rows — those go through enroll/mint/revoke."""
        surface_id = _surface_id(request)
        if not surface_id or len(surface_id) > _MAX_ID_LEN:
            return err("surface_id required", 400)
        body = await _body(request)
        if isinstance(body, JSONResponse):
            return body
        grant = _grant(body)
        if isinstance(grant, JSONResponse):
            return grant
        trust, projects = grant
        # Default to restricted when the caller omits `trust`: the schema default and
        # the no-row default both say restricted, so the route must not be the one
        # place where a missing field means "full".
        out = await _run(
            request,
            lambda: upsert_surface(db_url, surface_id, trust or "restricted", projects or []),
            admin_gate,
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
