"""Credential-bound surface trust (schema 054): the credential IS the surface.

Schema 053 keyed trust on a hostname the client asserted under a shared machine token,
which put the trust boundary in the hands of the machine being bounded. These tests
cover the replacement, and they are organised around the four properties that make it
worth the change:

  * **Enrollment cannot bless.** The root token enrolls (first device bootstraps, every
    later one lands pending) but CANNOT approve. Nothing a fresh machine holds is
    enough to widen its own trust.
  * **Only approved serves.** A pending or revoked credential authenticates as nothing —
    on the routes, in the trust resolution, and in the audience-derivation union.
  * **The token outranks the claim.** A device token names its own surface; a `surface`
    param from that caller is ignored outright.
  * **The old lanes still work.** A root-token caller's legacy `surface` param resolves
    byte-for-byte as it did (the migration window), and the `oauth:<login>` identity
    lane is untouched.
  * **A self-declared role is a request.** The install prompt's "personal or work?"
    answer rides along with the enrollment and becomes approve's DEFAULT — never a
    grant. A machine calling itself personal is served exactly as much as one that says
    nothing, which is nothing, until someone approves it.
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
    enroll_surface,
    issue_dash_token,
    resolve_caller,
    token_hash,
)
from mcp_server.surface_routes import register  # noqa: E402
from tests.helpers.surfaces import clear_surfaces, register_device  # noqa: E402

_ROOT = "test-root-token"


@pytest.fixture()
def clean(conn):
    clear_surfaces(conn)
    yield conn
    clear_surfaces(conn)


def _client(db_url, admin_token: str | None = None):
    """A surfaces-routes app with the two real gates wired the way the server wires them.

    ``authorized`` = the root token or ANY bearer we've been told about; ``admin`` = one
    specific full-trust device token and nothing else. Keeping them separate here is the
    point of the file: a test that passed both gates with one credential would prove
    nothing about the property under test.
    """
    from fastmcp import FastMCP

    def authorized(request):
        return bool(_bearer(request))

    def admin(request):
        return admin_token is not None and _bearer(request) == admin_token

    test_mcp = FastMCP("test-credential-trust")
    register(test_mcp, db_url, authorized, admin)
    return TestClient(test_mcp.http_app())


def _bearer(request):
    authz = request.headers.get("authorization", "")
    return authz[len("Bearer ") :].strip() if authz.startswith("Bearer ") else ""


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Enrollment: bootstrap once, pending ever after
# ---------------------------------------------------------------------------


def test_the_first_device_bootstraps_to_approved_full_trust(clean, db_url):
    """Nobody exists who could approve it, so the first device in is trusted. Safe
    precisely because the branch is unreachable the moment it has been taken."""
    with _client(db_url) as client:
        r = client.post("/surfaces/enroll", json={"label": "first-laptop"}, headers=_h(_ROOT))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved" and body["pair_code"] is None
    assert body["surface"]["trust"] == "full"

    st = resolve_caller(db_url, token_hash_hex=token_hash(body["token"]))
    assert st.known and not st.restricted


def test_the_second_device_lands_pending_with_a_pair_code(clean, db_url):
    with _client(db_url) as client:
        first = client.post("/surfaces/enroll", json={"label": "first"}, headers=_h(_ROOT)).json()
        second = client.post("/surfaces/enroll", json={"label": "second"}, headers=_h(_ROOT)).json()

    assert first["status"] == "approved"
    assert second["status"] == "pending"
    assert len(second["pair_code"]) == 6
    # The token is real; it just authenticates as nothing yet.
    assert second["token"] and second["token"] != first["token"]
    assert resolve_caller(db_url, token_hash_hex=token_hash(second["token"])) == UNKNOWN_SURFACE


def test_enrollment_does_not_bind_to_the_self_reported_label(clean, db_url):
    """The label is display only. Enrolling as an existing surface's name must not
    inherit that surface's grant — that would be hostname spoofing through the back
    door, which is the exact hole this schema closes."""
    register_device(clean, "trusted-tok", trust="full", surface_id="trusted-host", label="laptop")
    with _client(db_url) as client:
        out = client.post("/surfaces/enroll", json={"label": "laptop"}, headers=_h(_ROOT)).json()
    assert out["status"] == "pending"
    assert out["surface"]["surface_id"] != "trusted-host"
    assert resolve_caller(db_url, token_hash_hex=token_hash(out["token"])) == UNKNOWN_SURFACE


def test_enroll_requires_some_credential(clean, db_url):
    with _client(db_url) as client:
        assert client.post("/surfaces/enroll", json={"label": "x"}).status_code == 401
    assert _rows(clean) == 0


def _rows(conn) -> int:
    return conn.execute("SELECT count(*) FROM surfaces").fetchone()[0]


# ---------------------------------------------------------------------------
# Approval: the enrollment credential is NOT enough
# ---------------------------------------------------------------------------


def test_approve_refuses_the_enrollment_token(clean, db_url):
    """THE property. Every machine holds the enrollment credential at install time, so
    if it could approve, a device could self-approve and TOFU would be decoration."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        pending = client.post("/surfaces/enroll", json={"label": "work"}, headers=_h(_ROOT)).json()
        code = pending["pair_code"]

        refused = client.post(
            "/surfaces/approve", json={"pair_code": code, "trust": "full"}, headers=_h(_ROOT)
        )
        assert refused.status_code == 401

    # And nothing moved: the device is still pending, still serving nothing.
    assert resolve_caller(db_url, token_hash_hex=token_hash(pending["token"])) == UNKNOWN_SURFACE


def test_approve_with_a_full_trust_device_token_grants_the_stated_scope(clean, db_url):
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        pending = client.post("/surfaces/enroll", json={"label": "work"}, headers=_h(_ROOT)).json()
        r = client.post(
            "/surfaces/approve",
            json={
                "pair_code": pending["pair_code"],
                "trust": "restricted",
                "allowed_projects": ["work-thing"],
            },
            headers=_h(admin_tok),
        )
    assert r.status_code == 200
    st = resolve_caller(db_url, token_hash_hex=token_hash(pending["token"]))
    assert st.known and st.restricted and st.project_filter == ["work-thing"]


def test_a_restricted_device_cannot_approve(clean, db_url):
    """Restricted is a READ scope, not a lesser admin. It gets no say in who joins."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    register_device(clean, "work-tok", trust="restricted", projects=["w"], surface_id="dev-work")
    with _client(db_url, admin_token=admin_tok) as client:
        pending = client.post("/surfaces/enroll", json={"label": "n"}, headers=_h(_ROOT)).json()
        r = client.post(
            "/surfaces/approve",
            json={"pair_code": pending["pair_code"], "trust": "full"},
            headers=_h("work-tok"),
        )
    assert r.status_code == 401


def test_a_pair_code_is_single_use(clean, db_url):
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        pending = client.post("/surfaces/enroll", json={"label": "w"}, headers=_h(_ROOT)).json()
        body = {"pair_code": pending["pair_code"], "trust": "full"}
        assert client.post("/surfaces/approve", json=body, headers=_h(admin_tok)).status_code == 200
        again = client.post("/surfaces/approve", json=body, headers=_h(admin_tok))
    # 404, not 401 — the caller is fine, the code is spent.
    assert again.status_code == 404


def test_approve_defaults_to_restricted_when_nothing_was_requested(clean, db_url):
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        pending = client.post("/surfaces/enroll", json={"label": "w"}, headers=_h(_ROOT)).json()
        r = client.post(
            "/surfaces/approve", json={"pair_code": pending["pair_code"]}, headers=_h(admin_tok)
        )
    assert r.json()["surface"]["trust"] == "restricted"


# ---------------------------------------------------------------------------
# Mint: the untrusted-machine path
# ---------------------------------------------------------------------------


def test_mint_issues_a_ready_scoped_token_without_the_target_enrolling(clean, db_url):
    """For a machine you do not want holding the enrollment credential at all: mint
    here, carry only the minted token over."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        r = client.post(
            "/surfaces/mint",
            json={"label": "work-laptop", "trust": "restricted", "allowed_projects": ["w"]},
            headers=_h(admin_tok),
        )
    assert r.status_code == 200
    st = resolve_caller(db_url, token_hash_hex=token_hash(r.json()["token"]))
    assert st.known and st.restricted and st.project_filter == ["w"]


def test_mint_refuses_the_enrollment_token(clean, db_url):
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        r = client.post("/surfaces/mint", json={"label": "x", "trust": "full"}, headers=_h(_ROOT))
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Revocation and listing
# ---------------------------------------------------------------------------


def test_revoke_kills_the_credential_on_the_next_call(clean, db_url):
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        minted = client.post(
            "/surfaces/mint", json={"label": "gone", "trust": "full"}, headers=_h(admin_tok)
        ).json()
        sid = minted["surface"]["surface_id"]
        assert resolve_caller(db_url, token_hash_hex=token_hash(minted["token"])).known

        r = client.delete(f"/surfaces/{sid}", headers=_h(admin_tok))
    assert r.status_code == 200 and r.json()["revoked"] == 1
    assert resolve_caller(db_url, token_hash_hex=token_hash(minted["token"])) == UNKNOWN_SURFACE
    # The row SURVIVES as the audit trail, reset to the fail-closed floor.
    row = clean.execute(
        "SELECT status, trust, token_hash FROM surfaces WHERE surface_id = %s", (sid,)
    ).fetchone()
    assert row == ("revoked", "restricted", None)


def test_the_list_shows_pending_devices_first(clean, db_url):
    """An enrollment nobody can see is an enrollment nobody approves."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin", label="admin box")
    with _client(db_url, admin_token=admin_tok) as client:
        client.post("/surfaces/enroll", json={"label": "waiting"}, headers=_h(_ROOT))
        rows = client.get("/surfaces", headers=_h(admin_tok)).json()["surfaces"]
    assert rows[0]["status"] == "pending" and rows[0]["label"] == "waiting"
    assert rows[0]["pair_code"]
    # The credential never comes back out — only whether one exists.
    assert all("token" not in r and "token_hash" not in r for r in rows)
    assert rows[0]["has_token"] is True


def test_the_list_needs_the_admin_gate(clean, db_url):
    """The pair codes in the listing are approval capabilities."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        assert client.get("/surfaces", headers=_h(_ROOT)).status_code == 401
        assert client.get("/surfaces", headers=_h(admin_tok)).status_code == 200


def test_put_cannot_promote_a_device_row(clean, db_url):
    """PUT exists for credential-less ids (oauth:<login>, legacy hosts). Letting it
    reach a device row would grant a known token full trust without an approval."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    register_device(clean, "work-tok", trust="restricted", projects=["w"], surface_id="dev-work")
    with _client(db_url, admin_token=admin_tok) as client:
        r = client.put("/surfaces/dev-work", json={"trust": "full"}, headers=_h(admin_tok))
    assert r.status_code == 400 and "device surface" in r.json()["detail"]
    assert resolve_caller(db_url, token_hash_hex=token_hash("work-tok")).restricted


def test_put_still_registers_an_oauth_identity(clean, db_url):
    """The identity lane is untouched by 054 — it has no device token to bind to."""
    admin_tok = "admin-device-token"
    register_device(clean, admin_tok, trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token=admin_tok) as client:
        r = client.put("/surfaces/oauth:someone", json={"trust": "full"}, headers=_h(admin_tok))
    assert r.status_code == 200
    st = resolve_caller(db_url, legacy_surface_id="oauth:someone")
    assert st.known and not st.restricted


# ---------------------------------------------------------------------------
# Resolution: which credential wins, and what a non-approved row resolves to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "revoked"])
def test_a_non_approved_row_resolves_to_unknown_not_to_restricted_known(clean, status):
    """`known` is what lets a restricted write default to work-safe. A device that has
    not been approved must not get that — it would WIDEN a note's later audience."""
    register_device(clean, "tok", trust="full", surface_id="dev-x", status=status)
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


def test_enroll_survives_a_missing_label(clean, db_url):
    with _client(db_url) as client:
        r = client.post("/surfaces/enroll", json={}, headers=_h(_ROOT))
    assert r.status_code == 200 and r.json()["surface"]["label"] is None


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


# ---------------------------------------------------------------------------
# Concurrency: two first-enrollments must not both bootstrap
# ---------------------------------------------------------------------------


def test_only_one_of_two_racing_enrollments_can_bootstrap(clean, db_url):
    """Without the advisory lock both would read "zero approved devices" and both would
    land at full trust — a silent double-grant nobody would ever notice."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = [f.result() for f in [ex.submit(enroll_surface, db_url, f"n{i}") for i in (1, 2)]]
    statuses = sorted(r["surface"]["status"] for r in results)
    assert statuses == ["approved", "pending"]


# ---------------------------------------------------------------------------
# The self-declared machine role: a REQUEST, never a grant
# ---------------------------------------------------------------------------


def test_the_requested_role_is_recorded_but_grants_nothing(clean, db_url):
    """The security invariant of the whole request feature. A machine declaring itself
    personal is served exactly as much as one declaring nothing — none — until an
    approved full-trust device says otherwise."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token="admin-tok") as client:
        out = client.post(
            "/surfaces/enroll",
            json={"label": "laptop", "requested_trust": "full"},
            headers=_h(_ROOT),
        ).json()
    assert out["status"] == "pending"
    assert out["surface"]["requested_trust"] == "full"
    assert out["surface"]["trust"] == "restricted"  # the ROW is still at the floor
    assert resolve_caller(db_url, token_hash_hex=token_hash(out["token"])) == UNKNOWN_SURFACE


def test_approve_defaults_to_the_requested_role(clean, db_url):
    """The point of the request: approving is a one-word confirmation of a role the
    operator can already read next to the pair code."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token="admin-tok") as client:
        pending = client.post(
            "/surfaces/enroll",
            json={"label": "my-desktop", "requested_trust": "full"},
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve", json={"pair_code": pending["pair_code"]}, headers=_h("admin-tok")
        )
    assert r.status_code == 200 and r.json()["surface"]["trust"] == "full"
    assert not resolve_caller(db_url, token_hash_hex=token_hash(pending["token"])).restricted


def test_an_explicit_grant_overrides_the_request_in_both_directions(clean, db_url):
    """The request never constrains the approver — it is a default, not a negotiation."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token="admin-tok") as client:
        asked_full = client.post(
            "/surfaces/enroll",
            json={"label": "a", "requested_trust": "full"},
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve",
            json={"pair_code": asked_full["pair_code"], "trust": "restricted"},
            headers=_h("admin-tok"),
        )
        assert r.json()["surface"]["trust"] == "restricted"

        asked_restricted = client.post(
            "/surfaces/enroll",
            json={"label": "b", "requested_trust": "restricted"},
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve",
            json={"pair_code": asked_restricted["pair_code"], "trust": "full"},
            headers=_h("admin-tok"),
        )
        assert r.json()["surface"]["trust"] == "full"


def test_an_unstated_role_falls_back_to_restricted_and_empty(clean, db_url):
    """A client too old to declare anything must not become a grant by omission."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token="admin-tok") as client:
        pending = client.post(
            "/surfaces/enroll", json={"label": "old-client"}, headers=_h(_ROOT)
        ).json()
        r = client.post(
            "/surfaces/approve", json={"pair_code": pending["pair_code"]}, headers=_h("admin-tok")
        )
    surface = r.json()["surface"]
    assert surface["trust"] == "restricted" and surface["allowed_projects"] == []


def test_an_unrecognised_role_is_treated_as_unstated_not_rejected(clean, db_url):
    """A typo in a machine's config must not lock it out of enrolling: the request
    grants nothing either way, so "unstated" is the safe reading."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    with _client(db_url, admin_token="admin-tok") as client:
        out = client.post(
            "/surfaces/enroll",
            json={"label": "typo", "requested_trust": "sorta-full"},
            headers=_h(_ROOT),
        )
    assert out.status_code == 200 and out.json()["surface"]["requested_trust"] is None


def test_a_restricted_approve_inherits_the_existing_work_scope(clean, db_url):
    """A second work machine should see the same work projects as the first. Making the
    operator re-list them by hand is how a scope silently drifts between devices."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    register_device(
        clean, "work1", trust="restricted", projects=["work-a", "work-b"], surface_id="dev-w1"
    )
    with _client(db_url, admin_token="admin-tok") as client:
        pending = client.post(
            "/surfaces/enroll",
            json={"label": "work-laptop-2", "requested_trust": "restricted"},
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve", json={"pair_code": pending["pair_code"]}, headers=_h("admin-tok")
        )
    assert r.json()["surface"]["allowed_projects"] == ["work-a", "work-b"]


def test_requested_projects_win_over_the_inherited_scope(clean, db_url):
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    register_device(clean, "work1", trust="restricted", projects=["work-a"], surface_id="dev-w1")
    with _client(db_url, admin_token="admin-tok") as client:
        pending = client.post(
            "/surfaces/enroll",
            json={
                "label": "w2",
                "requested_trust": "restricted",
                "requested_projects": ["work-c"],
            },
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve", json={"pair_code": pending["pair_code"]}, headers=_h("admin-tok")
        )
    assert r.json()["surface"]["allowed_projects"] == ["work-c"]


def test_an_explicit_empty_allowlist_still_means_empty(clean, db_url):
    """Explicit [] must not be mistaken for "unstated" and refilled from the request —
    an operator narrowing a grant to nothing has to be able to say so."""
    register_device(clean, "admin-tok", trust="full", surface_id="dev-admin")
    register_device(clean, "work1", trust="restricted", projects=["work-a"], surface_id="dev-w1")
    with _client(db_url, admin_token="admin-tok") as client:
        pending = client.post(
            "/surfaces/enroll",
            json={"label": "w2", "requested_trust": "restricted", "requested_projects": ["x"]},
            headers=_h(_ROOT),
        ).json()
        r = client.post(
            "/surfaces/approve",
            json={"pair_code": pending["pair_code"], "allowed_projects": []},
            headers=_h("admin-tok"),
        )
    assert r.json()["surface"]["allowed_projects"] == []


def test_a_first_device_that_declares_itself_work_does_not_bootstrap(clean, db_url):
    """The deliberate edge. Auto-approving a restricted first device would leave the
    deployment with no credential that can reach the admin routes and no way to mint one
    over HTTP — an unadministrable Synapse. It lands pending instead, and the operator
    uses `scripts/surface_admin.py bootstrap` on the DB host."""
    with _client(db_url) as client:
        out = client.post(
            "/surfaces/enroll",
            json={"label": "work-only", "requested_trust": "restricted"},
            headers=_h(_ROOT),
        ).json()
    assert out["status"] == "pending" and out["pair_code"]
    assert resolve_caller(db_url, token_hash_hex=token_hash(out["token"])) == UNKNOWN_SURFACE


def test_a_first_device_that_declares_itself_personal_still_bootstraps(clean, db_url):
    with _client(db_url) as client:
        out = client.post(
            "/surfaces/enroll",
            json={"label": "my-desktop", "requested_trust": "full"},
            headers=_h(_ROOT),
        ).json()
    assert out["status"] == "approved"
    assert not resolve_caller(db_url, token_hash_hex=token_hash(out["token"])).restricted
