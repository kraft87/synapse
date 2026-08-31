"""Surface-registration routes (schema 053) — the operator seam for host trust.

These routes are the ONLY way a host becomes trusted, so the tests are about the
grant, not the plumbing: that a PUT actually widens what that surface is served, that
DELETE narrows it back, that an omitted `trust` field defaults to restricted rather
than to full, and that the whole thing sits behind the machine token like its
private-session / preferences siblings.

Skips cleanly when no test DB is reachable, mirroring test_private_session_routes.py.
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

from ingestion.surfaces import lookup_surface  # noqa: E402
from mcp_server.surface_routes import register  # noqa: E402

_TOKEN = "test-surface-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_SID = "route-test-host"


def _client(db_url):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-surfaces")
    register(test_mcp, db_url, authorized)
    return TestClient(test_mcp.http_app())


@pytest.fixture()
def clean(conn):
    conn.execute("DELETE FROM surfaces")
    yield conn
    conn.execute("DELETE FROM surfaces")


def test_routes_require_the_machine_token(clean, db_url):
    with _client(db_url) as client:
        assert client.get("/surfaces").status_code == 401
        assert client.put(f"/surfaces/{_SID}", json={"trust": "full"}).status_code == 401
        assert client.delete(f"/surfaces/{_SID}").status_code == 401
    # And nothing was written by the rejected calls.
    assert lookup_surface(db_url, _SID).known is False


def test_put_grants_trust_and_the_lookup_sees_it(clean, db_url):
    with _client(db_url) as client:
        r = client.put(
            f"/surfaces/{_SID}",
            json={"trust": "restricted", "allowed_projects": ["alpha", "beta"]},
            headers=_H,
        )
        assert r.status_code == 200
        body = r.json()["surface"]
        assert body["surface_id"] == _SID and body["trust"] == "restricted"
        assert body["allowed_projects"] == ["alpha", "beta"]

    st = lookup_surface(db_url, _SID)
    assert st.known and st.trust == "restricted"
    assert st.project_filter == ["alpha", "beta"]


def test_put_replaces_rather_than_patches(clean, db_url):
    """Full replacement is deliberate: a partial update of a security allowlist is how
    a stale grant survives a demotion."""
    with _client(db_url) as client:
        client.put(
            f"/surfaces/{_SID}", json={"trust": "full", "allowed_projects": ["alpha"]}, headers=_H
        )
        assert lookup_surface(db_url, _SID).trust == "full"
        r = client.put(
            f"/surfaces/{_SID}",
            json={"trust": "restricted", "allowed_projects": ["beta"]},
            headers=_H,
        )
        assert r.status_code == 200
    st = lookup_surface(db_url, _SID)
    assert st.trust == "restricted" and st.project_filter == ["beta"]  # alpha is GONE


def test_omitted_trust_defaults_to_restricted(clean, db_url):
    """The schema default, the no-row default and the route default must agree.
    Promotion to full is always an explicit act."""
    with _client(db_url) as client:
        r = client.put(f"/surfaces/{_SID}", json={"allowed_projects": ["alpha"]}, headers=_H)
        assert r.status_code == 200
        assert r.json()["surface"]["trust"] == "restricted"


def test_invalid_trust_is_rejected(clean, db_url):
    with _client(db_url) as client:
        r = client.put(f"/surfaces/{_SID}", json={"trust": "sorta"}, headers=_H)
        assert r.status_code == 400 and "invalid trust" in r.json()["detail"]
    assert lookup_surface(db_url, _SID).known is False


def test_malformed_bodies_are_rejected(clean, db_url):
    with _client(db_url) as client:
        assert client.put(f"/surfaces/{_SID}", content=b"{not json", headers=_H).status_code == 400
        r = client.put(f"/surfaces/{_SID}", json={"allowed_projects": "alpha"}, headers=_H)
        assert r.status_code == 400 and "must be a list" in r.json()["detail"]
        over = client.put(
            f"/surfaces/{_SID}",
            json={"allowed_projects": [f"p{i}" for i in range(101)]},
            headers=_H,
        )
        assert over.status_code == 400


def test_list_returns_every_registration(clean, db_url):
    with _client(db_url) as client:
        client.put("/surfaces/host-a", json={"trust": "full"}, headers=_H)
        client.put(
            "/surfaces/host-b",
            json={"trust": "restricted", "allowed_projects": ["alpha"]},
            headers=_H,
        )
        rows = client.get("/surfaces", headers=_H).json()["surfaces"]
    assert [r["surface_id"] for r in rows] == ["host-a", "host-b"]
    assert rows[0]["trust"] == "full" and rows[1]["allowed_projects"] == ["alpha"]
    assert all({"created_at", "updated_at"} <= set(r) for r in rows)


def test_delete_reverts_to_the_unregistered_default(clean, db_url):
    with _client(db_url) as client:
        client.put(f"/surfaces/{_SID}", json={"trust": "full"}, headers=_H)
        assert lookup_surface(db_url, _SID).trust == "full"
        r = client.delete(f"/surfaces/{_SID}", headers=_H)
        assert r.status_code == 200 and r.json()["deleted"] == 1
        # Idempotent: deleting an absent row is a no-op, not an error.
        assert client.delete(f"/surfaces/{_SID}", headers=_H).json()["deleted"] == 0

    st = lookup_surface(db_url, _SID)
    assert st.restricted and not st.known and st.project_filter == []


def test_blank_surface_id_is_rejected(clean, db_url):
    with _client(db_url) as client:
        assert client.put("/surfaces/%20", json={"trust": "full"}, headers=_H).status_code == 400
        assert client.delete("/surfaces/%20", headers=_H).status_code == 400
