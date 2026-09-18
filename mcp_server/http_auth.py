"""Credential gates for custom HTTP routes, independent of FastMCP middleware."""

import hmac

from starlette.requests import Request

from ingestion.surfaces import resolve_caller, token_hash


def bearer(request: Request) -> str:
    """The raw bearer credential on this request, or "" when there isn't one."""
    authz = request.headers.get("authorization", "")
    return authz[len("Bearer ") :].strip() if authz.startswith("Bearer ") else ""


def is_root(request: Request, machine_token: str) -> bool:
    """Root token, constant-time, no DB touch (see mcp_server.auth_tokens)."""
    tok = bearer(request)
    return bool(machine_token and tok and hmac.compare_digest(tok, machine_token))


def machine_authorized(request: Request, machine_token: str, db_url: str) -> bool:
    """ "Is this a Synapse client?" — the CLIENT gate on the custom routes.

    Custom routes bypass FastMCP's auth middleware (by design, issue #3704), so gate
    them here. Passes for the root token (the services on the Docker host, and any
    client still in the migration window) and for an APPROVED device token. A pending
    enrollment fails it: a device that has not been approved holds a real token that
    authenticates as nothing, which is the whole point of pending. Its transcripts are
    not lost — the ingest hook's ``--catchup`` sweep re-posts them after approval.

    Open when no machine token is set (dev / pre-cutover).
    """
    if not machine_token:
        return True
    if is_root(request, machine_token):
        return True
    tok = bearer(request)
    return bool(tok) and resolve_caller(db_url, token_hash_hex=token_hash(tok)).known


def admin_authorized(request: Request, machine_token: str, db_url: str) -> bool:
    """ "May this caller change who is trusted?" — the ADMIN gate, and it excludes root.

    Requires an APPROVED, FULL-TRUST device token. The root token is deliberately NOT
    enough, and that asymmetry is the security property this whole change buys:

      the credential a new machine must hold in order to ENROLL cannot APPROVE.

    Every machine that runs the plugin ends up holding the enrollment credential at
    install time. If that credential also approved devices, an attacker who read it off
    any one machine could self-approve to full trust and the TOFU gate would be
    decoration. So approval, minting, revocation, the surface list and the dashboard all
    demand a credential that only an already-trusted machine has.

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
    tok = bearer(request)
    if not tok or is_root(request, machine_token):
        return False
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    return st.known and not st.restricted
