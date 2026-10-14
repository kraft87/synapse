"""Credential gates for custom HTTP routes.

FastMCP enforces auth on /mcp only; every custom route gates itself through here. Its
app-wide auth middleware still VERIFIES each bearer, though, and the gates reuse that
verdict (:func:`request_trust`) rather than paying for a second lookup.
"""

import hmac

from starlette.requests import Request

from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust
from mcp_server import caller_trust
from mcp_server.auth_tokens import KIND_DEVICE, claims_of

#: ``request.state`` attribute holding the bearer's verdict once it has been resolved.
_TRUST_STATE = "synapse_trust"


def bearer(request: Request) -> str:
    """The raw bearer credential on this request, or "" when there isn't one."""
    authz = request.headers.get("authorization", "")
    return authz[len("Bearer ") :].strip() if authz.startswith("Bearer ") else ""


def is_root(request: Request, machine_token: str) -> bool:
    """Root token, constant-time, no DB touch (see mcp_server.auth_tokens)."""
    tok = bearer(request)
    return bool(machine_token and tok and hmac.compare_digest(tok, machine_token))


def authenticated(request: Request, machine_token: str) -> bool:
    """Did this request present a credential, on a server that checks credentials?

    Read it AFTER :func:`machine_authorized` admitted the request: past that gate a
    bearer is the root token or an approved device token, so this needs no second
    database read. On an open server (no machine token) nothing authenticates, and this
    is False for every caller. Decides who may be told WHY they are served nothing
    (recall_warnings.serving_notice): a caller with no credential has nothing to fix.
    """
    return bool(machine_token) and bool(bearer(request))


def request_trust(request: Request, machine_token: str, db_url: str) -> SurfaceTrust:
    """The bearer's trust verdict, resolved at most once per request.

    Both gates and every route's trust read go through here. On a credentialed server
    FastMCP's app-wide auth middleware has already run SynapseTokenVerifier on this
    request's bearer (off the event loop) and left the verdict on ``scope["user"]``, so
    this reads that result rather than repeating the lookup: device claims become the
    trust, and a bearer the verifier rejected (pending, revoked, unknown, or a database
    it could not reach) is UNKNOWN. Without that middleware (an open server, a bare test
    app) it falls back to :func:`caller_trust.request_trust`.

    The verdict is kept on ``request.state`` and dies with the request. Nothing is
    cached across requests, so a revoke or a demotion bites on the very next call
    (ingestion.surfaces explains why that matters).
    """
    st = getattr(request.state, _TRUST_STATE, None)
    if st is None:
        tok = bearer(request)
        if "user" in request.scope:
            # The verifier parsed this same header; a bearer these gates do not accept
            # (no "Bearer " prefix) is no credential here, whatever the verifier made of it.
            claims = claims_of(getattr(request.scope["user"], "access_token", None))
            is_device = bool(tok) and claims.get("kind") == KIND_DEVICE
            st = caller_trust.trust_from_claims(claims) if is_device else UNKNOWN_SURFACE
        else:
            st = caller_trust.request_trust(db_url=db_url, machine_token=machine_token, bearer=tok)
        setattr(request.state, _TRUST_STATE, st)
    return st


def machine_authorized(request: Request, machine_token: str, db_url: str) -> bool:
    """ "Is this a Synapse client?" — the CLIENT gate on the custom routes.

    FastMCP enforces auth on /mcp only (custom routes are exempt by design, issue
    #3704), so gate them here. Passes for the root token (the services on the Docker host, and any
    not-yet-enrolled client, which may ingest but is SERVED nothing: its read scope
    resolves to unknown in ``mcp_server/caller_trust``) and for an APPROVED device
    token. A pending enrollment fails it: a device that has not been approved holds a
    real token that authenticates as nothing, which is the whole point of pending. Its
    transcripts are not lost — the ingest hook's ``--catchup`` sweep re-posts them
    after approval.

    Open when no machine token is set (dev / pre-cutover).
    """
    if not machine_token:
        return True
    if is_root(request, machine_token):
        return True
    return request_trust(request, machine_token, db_url).known


def admin_authorized(request: Request, machine_token: str, db_url: str) -> bool:
    """ "May this caller change who is trusted?" — the ADMIN gate, and it excludes root.

    Requires an APPROVED, FULL-TRUST device token. The root token is deliberately NOT
    enough, and that asymmetry is the security property this whole change buys:

      the credential a new machine must hold in order to ENROLL cannot APPROVE.

    Every machine that runs the plugin ends up holding the enrollment credential at
    install time. If that credential also approved devices, an attacker who read it off
    any one machine could self-approve to full trust, and the device trust levels would
    be decoration. So approval, minting, revocation, the surface list and the dashboard
    all demand a credential that only an already-trusted machine has.

    The dashboard reaches this gate through the login flow, which mints a full-trust
    device token for an OAuth-allowlisted identity rather than handing out the root
    token. Break-glass, when no full-trust device exists (first deploy, or every device
    revoked): ``synapse-admin bootstrap "<label>"`` (in the container, or
    ``scripts/surface_admin.py`` on the host) talks to Postgres directly, which requires
    shell access on the DB host — a strictly higher bar than holding a bearer token.

    Open when no machine token is set (dev / pre-cutover), same as the client gate.
    """
    if not machine_token:
        return True
    if not bearer(request) or is_root(request, machine_token):
        return False
    st = request_trust(request, machine_token, db_url)
    return st.known and not st.restricted
