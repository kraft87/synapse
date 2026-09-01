"""The bearer-credential verifier: one root token plus N per-device tokens (schema 054).

Replaces ``StaticTokenVerifier``, which could only know one token and therefore could
only answer "is this a Synapse client" — never "WHICH client". Audience scoping needs
the second answer, and asking the caller (a ``surface`` param under the shared token)
made the untrusted machine the authority on its own trust level. A per-device token
makes the credential itself the answer.

Two lanes, deliberately asymmetric:

* **Root** (``SYNAPSE_MACHINE_TOKEN``) — constant-time compare, **no database touch**.
  This is load-bearing: the services on the Docker host authenticate with this token to
  reach ``/ingest`` and the internal write paths, and those must keep working through a
  Postgres blip. A verifier that had to read PG to admit the root token would turn
  every PG hiccup into a total auth outage on the one lane that repairs things. The
  root token identifies a *deployment*, not a machine, so it resolves to NO surface —
  ``{"kind": "root"}`` and nothing else.
* **Device** — ``sha256(token)`` looked up in ``surfaces``. Only ``status='approved'``
  verifies; pending and revoked are indistinguishable from garbage on the wire (401),
  which is the intended answer for a device that has not been approved yet and for one
  whose credential was pulled.

The claims dict is the single carrier of "who is calling" for everything downstream:
``mcp_server/server`` reads it to build the caller's :class:`~ingestion.surfaces.SurfaceTrust`
without a second lookup, and to decide which routes a caller may reach at all.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastmcp.server.auth.auth import AccessToken, TokenVerifier

from ingestion.surfaces import resolve_caller, token_hash

logger = logging.getLogger(__name__)

#: client_id on the root lane. Pre-existing value — the OAuth allowlist middleware
#: skips this client_id, so changing it would silently subject the machine token to
#: the human allowlist.
ROOT_CLIENT_ID = "synapse-machine"

#: client_id on the device lane. Distinct so the same allowlist skip applies (a device
#: token is a machine credential, not a human login) while remaining tellable apart.
DEVICE_CLIENT_ID = "synapse-device"

#: The two claim shapes. ``kind`` is the discriminator every consumer switches on.
KIND_ROOT = "root"
KIND_DEVICE = "device"


def claims_of(token: Any) -> dict[str, Any]:
    """The claims dict of an AccessToken, or ``{}`` for anything else. Never raises."""
    try:
        return dict(getattr(token, "claims", None) or {})
    except Exception:  # pragma: no cover - defensive
        return {}


class SynapseTokenVerifier(TokenVerifier):
    """Verify a bearer as either the root token or an approved device token.

    ``scopes`` must clear whichever interactive leg is active, because ``MultiAuth``
    applies the server's required scopes to ``/mcp``: "user" for the GitHub leg, the
    configured OIDC set otherwise. Carrying both is harmless.
    """

    def __init__(self, machine_token: str, db_url: str, scopes: list[str]) -> None:
        super().__init__()
        self._machine_token = machine_token or ""
        self._db_url = db_url or ""
        self._scopes = list(scopes)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None
        # Root FIRST and without any I/O — see the module docstring.
        if self._machine_token and hmac.compare_digest(token, self._machine_token):
            return AccessToken(
                token=token,
                client_id=ROOT_CLIENT_ID,
                scopes=list(self._scopes),
                claims={"kind": KIND_ROOT},
            )
        if not self._db_url:
            return None
        # resolve_caller never raises and never fails open: an unreachable DB, a missing
        # table, or a non-approved row all come back UNKNOWN, which has known=False.
        st = resolve_caller(self._db_url, token_hash_hex=token_hash(token))
        if not st.known:
            return None
        return AccessToken(
            token=token,
            client_id=DEVICE_CLIENT_ID,
            scopes=list(self._scopes),
            claims={
                "kind": KIND_DEVICE,
                "surface_id": st.surface_id,
                "trust": st.trust,
                "allowed_projects": list(st.allowed_projects),
            },
        )
