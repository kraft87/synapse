"""Credential-bound surface trust (schema 054): the credential IS the surface.

Schema 053 keyed trust on a hostname the client asserted under a shared machine token,
which put the trust boundary in the hands of the machine being bounded. These tests
cover the replacement, organised around the properties that make it worth the change:

  * **Only an authenticated owner can create a credential.** Enrollment is anchored to
    the IdP's device flow plus the login allowlist; the shared machine token — which
    every machine that ever ran the plugin has held — cannot mint anything.
  * **Only approved serves.** A revoked credential matches no row: on the routes, in
    trust resolution, and in the audience-derivation union.
  * **The token outranks the claim.** A device token names its own surface; a `surface`
    param from that caller is ignored outright, and a self-reported label binds nothing.
  * **The old lanes still work.** A root-token caller's legacy `surface` param resolves
    byte-for-byte as it did (the migration window), and the `oauth:<login>` identity
    lane is untouched.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from starlette.testclient import TestClient

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from ingestion.surfaces import (  # noqa: E402
    UNKNOWN_SURFACE,
    issue_dash_token,
    mint_surface,
    resolve_caller,
    token_hash,
)
from mcp_server.surface_routes import register  # noqa: E402
from tests.helpers.surfaces import clear_surfaces, register_device  # noqa: E402

_ROOT = "test-root-token"
_ADMIN = "admin-device-token"
_CODE = "device-code-abc"


@pytest.fixture()
def clean(conn):
    clear_surfaces(conn)
    yield conn
    clear_surfaces(conn)


class _FakeIdP:
    """Stands in for GitHubIdP / OIDCIdP — the three members the enroll route touches.

    Deliberately not a mock of the HTTP calls: the route's contract with the IdP is
    "poll, read identity, check the allowlist", and that is exactly what this models.
    """

    label = "fake"

    def __init__(self, identity: str = "owner", *, poll: dict | None = None, allowed=("owner",)):
        self.allowed = set(allowed)
        self._identity = identity
        self._poll = poll if poll is not None else {"access_token": "idp-access"}
        self.polls: list[str] = []

    async def device_poll(self, device_code: str) -> dict:
        self.polls.append(device_code)
        return self._poll

    async def fetch_identity(self, access_token: str) -> str:
        return self._identity


def _client(db_url, *, admin_token: str | None = _ADMIN, idp: object | None = None):
    """A surfaces-routes app with the real gates wired the way the server wires them.

    ``authorized`` = any bearer at all; ``admin`` = one specific full-trust device token.
    Keeping them separate is the point of the file: a test that cleared both gates with
    one credential would prove nothing about the property under test.
    """
    from fastmcp import FastMCP

    def authorized(request):
        return bool(_bearer(request))

    def admin(request):
        return admin_token is not None and _bearer(request) == admin_token

    test_mcp = FastMCP("test-credential-trust")
    register(test_mcp, db_url, authorized, admin, idp=idp)
    return TestClient(test_mcp.http_app())


def _bearer(request):
    authz = request.headers.get("authorization", "")
    return authz[len("Bearer ") :].strip() if authz.startswith("Bearer ") else ""


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _enroll_body(**over):
    return {"device_code": _CODE, "label": "my-laptop", **over}


def _rows(conn) -> int:
    return conn.execute("SELECT count(*) FROM surfaces").fetchone()[0]


# ---------------------------------------------------------------------------
# Enrollment is anchored to an allowlisted identity
# ---------------------------------------------------------------------------


def test_enrollment_mints_a_live_token_for_an_allowlisted_identity(clean, db_url):
    """No pending state and no second approval: the person who just authenticated is
    the authority for what their own machine is, so the grant lands live."""
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body(trust="full"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["login"] == "owner"
    assert body["surface"]["status"] == "approved" and body["surface"]["trust"] == "full"
    assert idp.polls == [_CODE]

    st = resolve_caller(db_url, token_hash_hex=token_hash(body["token"]))
    assert st.known and not st.restricted


def test_enrollment_refuses_an_identity_outside_the_allowlist(clean, db_url):
    """The IdP approving is not enough — the same allowlist every other login clears."""
    idp = _FakeIdP(identity="stranger", allowed=("owner",))
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body(trust="full"))
    assert r.status_code == 403 and "not in allowlist" in r.json()["detail"]
    assert _rows(clean) == 0


def test_enrollment_reports_pending_while_the_human_has_not_approved(clean, db_url):
    """202, not an error: the client is polling and must be able to tell "not yet" from
    "no" without parsing prose."""
    idp = _FakeIdP(poll={"error": "authorization_pending"})
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body())
    assert r.status_code == 202 and r.json()["error"] == "authorization_pending"
    assert _rows(clean) == 0


def test_enrollment_reports_a_refusal_distinctly(clean, db_url):
    idp = _FakeIdP(poll={"error": "access_denied"})
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body())
    assert r.status_code == 403
    assert _rows(clean) == 0


def test_enrollment_needs_a_device_code(clean, db_url):
    """No device code means no identity to anchor on, so there is nothing to check."""
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json={"label": "x", "trust": "full"})
    assert r.status_code == 400 and idp.polls == []
    assert _rows(clean) == 0


def test_enrollment_is_unavailable_without_an_identity_provider(clean, db_url):
    """A bearer-only deployment has no identity to anchor an enrollment to. 503 and a
    pointer at the break-glass CLI beats silently falling back to a weaker check."""
    with _client(db_url, idp=None) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body(trust="full"), headers=_h(_ROOT))
    assert r.status_code == 503 and "surface_admin" in r.json()["detail"]
    assert _rows(clean) == 0


def test_the_machine_token_alone_cannot_enroll(clean, db_url):
    """THE property. Every machine that ever ran the plugin has held the root token, so
    a credential that widespread must not be able to create new credentials — holding it
    changes nothing about whether an enrollment is authorized."""
    idp = _FakeIdP(poll={"error": "authorization_pending"})
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body(trust="full"), headers=_h(_ROOT))
    assert r.status_code == 202  # still just "the human has not approved"
    assert _rows(clean) == 0


def test_the_declared_role_lands_as_declared(clean, db_url):
    """The install prompt's answer is authoritative here, not a request: the person who
    gave it is the person who just authenticated."""
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post(
            "/surfaces/enroll",
            json=_enroll_body(trust="restricted", allowed_projects=["work-a"]),
        )
    st = resolve_caller(db_url, token_hash_hex=token_hash(r.json()["token"]))
    assert st.known and st.restricted and st.project_filter == ["work-a"]


def test_an_unstated_role_enrolls_restricted(clean, db_url):
    """A client that never asked "personal or work?" must not resolve it to full access.
    The human default lives in the install prompt; the machine default is narrow."""
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body())
    assert r.json()["surface"]["trust"] == "restricted"


def test_a_restricted_enrollment_inherits_the_existing_work_scope(clean, db_url):
    """A second work machine should see the same work projects as the first. Making the
    operator re-list them by hand is how a scope silently drifts between devices."""
    register_device(clean, "work1", trust="restricted", projects=["work-a", "work-b"])
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post("/surfaces/enroll", json=_enroll_body(trust="restricted"))
    assert r.json()["surface"]["allowed_projects"] == ["work-a", "work-b"]


def test_an_explicit_empty_allowlist_still_means_empty(clean, db_url):
    """Explicit [] must not be mistaken for "unstated" and refilled from the existing
    scope — narrowing a grant to nothing has to be sayable."""
    register_device(clean, "work1", trust="restricted", projects=["work-a"])
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        r = client.post(
            "/surfaces/enroll", json=_enroll_body(trust="restricted", allowed_projects=[])
        )
    assert r.json()["surface"]["allowed_projects"] == []


def test_enrollment_does_not_bind_to_the_self_reported_label(clean, db_url):
    """The label is display only. Enrolling under an existing surface's name must not
    inherit that surface's grant — that is hostname spoofing through the back door,
    which is the exact hole this schema closes."""
    register_device(clean, "trusted-tok", trust="full", surface_id="trusted-host", label="laptop")
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        out = client.post("/surfaces/enroll", json=_enroll_body(label="laptop")).json()
    assert out["surface"]["surface_id"] != "trusted-host"
    assert out["surface"]["trust"] == "restricted"  # not the row it named
    assert resolve_caller(db_url, token_hash_hex=token_hash("trusted-tok")).trust == "full"


def test_each_enrollment_gets_its_own_credential(clean, db_url):
    idp = _FakeIdP()
    with _client(db_url, idp=idp) as client:
        a = client.post("/surfaces/enroll", json=_enroll_body(label="a", trust="full")).json()
        b = client.post("/surfaces/enroll", json=_enroll_body(label="b", trust="full")).json()
    assert a["token"] != b["token"]
    assert a["surface"]["surface_id"] != b["surface"]["surface_id"]


# ---------------------------------------------------------------------------
# Mint: the no-browser path, and it needs an already-trusted machine
# ---------------------------------------------------------------------------


def test_mint_issues_a_ready_scoped_token(clean, db_url):
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    with _client(db_url) as client:
        r = client.post(
            "/surfaces/mint",
            json={"label": "headless", "trust": "restricted", "allowed_projects": ["w"]},
            headers=_h(_ADMIN),
        )
    assert r.status_code == 200
    st = resolve_caller(db_url, token_hash_hex=token_hash(r.json()["token"]))
    assert st.known and st.restricted and st.project_filter == ["w"]


def test_mint_refuses_the_machine_token(clean, db_url):
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    with _client(db_url) as client:
        r = client.post("/surfaces/mint", json={"label": "x", "trust": "full"}, headers=_h(_ROOT))
    assert r.status_code == 401
    assert _rows(clean) == 1  # only the admin device


def test_a_restricted_device_cannot_mint(clean, db_url):
    """Restricted is a READ scope, not a lesser admin. It gets no say in who joins."""
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    register_device(clean, "work-tok", trust="restricted", projects=["w"], surface_id="dev-work")
    with _client(db_url) as client:
        r = client.post(
            "/surfaces/mint", json={"label": "x", "trust": "full"}, headers=_h("work-tok")
        )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Revocation and listing
# ---------------------------------------------------------------------------


def test_revoke_kills_the_credential_on_the_next_call(clean, db_url):
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    with _client(db_url) as client:
        minted = client.post(
            "/surfaces/mint", json={"label": "gone", "trust": "full"}, headers=_h(_ADMIN)
        ).json()
        sid = minted["surface"]["surface_id"]
        assert resolve_caller(db_url, token_hash_hex=token_hash(minted["token"])).known

        r = client.delete(f"/surfaces/{sid}", headers=_h(_ADMIN))
    assert r.status_code == 200 and r.json()["revoked"] == 1
    assert resolve_caller(db_url, token_hash_hex=token_hash(minted["token"])) == UNKNOWN_SURFACE
    # The row SURVIVES as the audit trail, reset to the fail-closed floor.
    row = clean.execute(
        "SELECT status, trust, token_hash FROM surfaces WHERE surface_id = %s", (sid,)
    ).fetchone()
    assert row == ("revoked", "restricted", None)


def test_the_list_needs_the_admin_gate(clean, db_url):
    """The listing is the map of what every credential reaches."""
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin", label="admin box")
    with _client(db_url) as client:
        assert client.get("/surfaces", headers=_h(_ROOT)).status_code == 401
        rows = client.get("/surfaces", headers=_h(_ADMIN)).json()["surfaces"]
    assert rows[0]["surface_id"] == "dev-admin" and rows[0]["label"] == "admin box"
    # The credential never comes back out — only whether one exists.
    assert all("token" not in r and "token_hash" not in r for r in rows)
    assert rows[0]["has_token"] is True


def test_put_cannot_promote_a_device_row(clean, db_url):
    """PUT exists for credential-less ids (oauth:<login>, legacy hosts). Letting it
    reach a device row would hand an already-issued token full trust without an
    owner-authenticated enrollment."""
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    register_device(clean, "work-tok", trust="restricted", projects=["w"], surface_id="dev-work")
    with _client(db_url) as client:
        r = client.put("/surfaces/dev-work", json={"trust": "full"}, headers=_h(_ADMIN))
    assert r.status_code == 400 and "device surface" in r.json()["detail"]
    assert resolve_caller(db_url, token_hash_hex=token_hash("work-tok")).restricted


def test_put_still_registers_an_oauth_identity(clean, db_url):
    """The identity lane is untouched by 054 — it has no device token to bind to."""
    register_device(clean, _ADMIN, trust="full", surface_id="dev-admin")
    with _client(db_url) as client:
        r = client.put("/surfaces/oauth:someone", json={"trust": "full"}, headers=_h(_ADMIN))
    assert r.status_code == 200
    st = resolve_caller(db_url, legacy_surface_id="oauth:someone")
    assert st.known and not st.restricted


# ---------------------------------------------------------------------------
# Resolution: which credential wins, and what a revoked row resolves to
# ---------------------------------------------------------------------------


def test_a_revoked_row_resolves_to_unknown_not_to_restricted_known(clean):
    """`known` is what lets a restricted write default to work-safe. A revoked device
    must not get that — it would WIDEN a note's later audience."""
    register_device(clean, "tok", trust="full", surface_id="dev-x", status="revoked")
    assert resolve_caller(_DB_URL, token_hash_hex=token_hash("tok")) == UNKNOWN_SURFACE
    assert resolve_caller(_DB_URL, legacy_surface_id="dev-x") == UNKNOWN_SURFACE


def test_an_unmatched_token_reveals_nothing(clean):
    """A verdict that echoed a credential-derived id back would be an oracle."""
    st = resolve_caller(_DB_URL, token_hash_hex=token_hash("never-issued"))
    assert st == UNKNOWN_SURFACE and st.surface_id is None


def test_no_credential_at_all_is_restricted(clean):
    assert resolve_caller(_DB_URL) == UNKNOWN_SURFACE


def test_a_token_lookup_ignores_any_surface_id_supplied_alongside_it(clean):
    """Token first, and the id is not even consulted — a device that also sent a
    trusted host name must not get that host's grant."""
    register_device(clean, "work-tok", trust="restricted", projects=["w"], surface_id="dev-work")
    register_device(clean, "other", trust="full", surface_id="trusted-host")
    st = resolve_caller(
        _DB_URL, token_hash_hex=token_hash("work-tok"), legacy_surface_id="trusted-host"
    )
    assert st.restricted and st.surface_id == "dev-work"


def test_last_seen_is_stamped_on_use(clean):
    register_device(clean, "tok", trust="full", surface_id="dev-seen")
    assert _last_seen(clean, "dev-seen") is None
    resolve_caller(_DB_URL, token_hash_hex=token_hash("tok"))
    assert _last_seen(clean, "dev-seen") is not None


def _last_seen(conn, sid):
    return conn.execute(
        "SELECT last_seen_at FROM surfaces WHERE surface_id = %s", (sid,)
    ).fetchone()[0]


def test_the_schema_default_status_serves_nothing(clean):
    """A hand-written row that forgets `status` must not authenticate. 'revoked' reads
    odd as a default and is exactly right: "nobody granted this" and "this grant was
    pulled" should behave identically."""
    clean.execute(
        "INSERT INTO surfaces (surface_id, trust, token_hash) VALUES ('dev-oops', 'full', %s)",
        (token_hash("hand-written"),),
    )
    assert resolve_caller(_DB_URL, token_hash_hex=token_hash("hand-written")) == UNKNOWN_SURFACE


def test_invalid_trust_is_rejected_before_a_token_is_minted(clean, db_url):
    with pytest.raises(ValueError, match="invalid trust"):
        mint_surface(db_url, "x", "sorta")
    assert _rows(clean) == 0


# ---------------------------------------------------------------------------
# The dashboard token
# ---------------------------------------------------------------------------


def test_the_dashboard_login_mints_a_revocable_full_trust_device(clean, db_url):
    """The browser must not receive the root token: /dash/api sits behind the admin
    gate, and a credential living in a URL fragment should be revocable on its own."""
    tok = issue_dash_token(db_url, "Someone")
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    assert st.known and not st.restricted and st.surface_id == "dash:someone"


def test_signing_in_again_rotates_the_dashboard_token_in_place(clean, db_url):
    """One row per identity — repeated logins must not grow the surface list. Only the
    hash is stored, so the old plaintext genuinely stops working."""
    first = issue_dash_token(db_url, "someone")
    second = issue_dash_token(db_url, "someone")
    assert first != second
    assert resolve_caller(db_url, token_hash_hex=token_hash(first)) == UNKNOWN_SURFACE
    assert resolve_caller(db_url, token_hash_hex=token_hash(second)).known
    assert _rows(clean) == 1
