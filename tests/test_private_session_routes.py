"""DB-backed tests for private mode's server half (schema 050).

Two things only a database can prove:

  * the toggle routes (``mcp_server/private_session_routes``) round-trip the durable flag
    the CLI verifies — PUT is idempotent, GET reads it back, DELETE is the deliberate
    un-private escape hatch. PUT and GET sit behind the client gate like every sibling;
    DELETE sits behind the admin gate, so only a full-trust device can take a session
    back on the record;
  * ``/ingest`` actually DROPS turns from a flagged session. That's the chokepoint the
    whole design rests on: the marker file only covers the live hook, so a catch-up sweep,
    a backfill, or any other client posting the same transcript must still land nothing.

Skips cleanly when no test DB is reachable (the pure-logic coverage — marker handling,
predicate, toggle CLI — lives in test_private_mode.py).
"""

from __future__ import annotations

import importlib
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

from mcp_server.http_auth import admin_authorized, machine_authorized  # noqa: E402
from mcp_server.private_session_routes import register  # noqa: E402
from tests.helpers.surfaces import clear_surfaces, register_device  # noqa: E402

_TOKEN = "test-machine-token"
_FULL_TOKEN = "full-trust-device-token"
_WORK_TOKEN = "restricted-device-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_FULL_H = {"Authorization": f"Bearer {_FULL_TOKEN}"}
_WORK_H = {"Authorization": f"Bearer {_WORK_TOKEN}"}
_SID = "s-private-route-test"
_PUBLIC = "s-public-route-test"

_SERVER_AUTH_KEYS = (
    "SYNAPSE_MACHINE_TOKEN",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "ALLOWED_GITHUB_USERS",
    "SYNAPSE_OAUTH_SIGNING_KEY",
    "OIDC_CONFIG_URL",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "ALLOWED_OIDC_USERS",
)


def _client(db_url):
    """The routes behind the server's real gates: ``_TOKEN`` is the root token, and the
    devices a test registers (see ``devices``) are checked against the surfaces table."""
    from fastmcp import FastMCP

    def authorized(request):
        return machine_authorized(request, _TOKEN, db_url)

    def admin(request):
        return admin_authorized(request, _TOKEN, db_url)

    test_mcp = FastMCP("test-private")
    register(test_mcp, db_url, authorized, admin)
    return TestClient(test_mcp.http_app())


@pytest.fixture()
def clean(conn):
    def _wipe():
        conn.execute("DELETE FROM private_sessions WHERE session_id = ANY(%s)", ([_SID, _PUBLIC],))
        conn.execute("DELETE FROM episodes WHERE session_id = ANY(%s)", ([_SID, _PUBLIC],))
        clear_surfaces(conn)

    _wipe()
    yield
    _wipe()


@pytest.fixture()
def devices(clean, conn):
    """One full-trust and one restricted approved device, beside the root token."""
    register_device(conn, _FULL_TOKEN, trust="full", surface_id="dev-full")
    register_device(conn, _WORK_TOKEN, trust="restricted", projects=["w"], surface_id="dev-work")


def _rows(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM private_sessions WHERE session_id = %s", (_SID,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Toggle routes
# ---------------------------------------------------------------------------


def test_routes_require_machine_token(clean, db_url):
    with _client(db_url) as client:
        for send in (client.put, client.get, client.delete):
            r = send(f"/private-sessions/{_SID}")
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}


def test_put_get_delete_round_trip(devices, conn, db_url):
    with _client(db_url) as client:
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is False

        r = client.put(f"/private-sessions/{_SID}", headers=_H)
        assert r.status_code == 200 and r.json()["private"] is True
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is True

        # idempotent: a retried toggle is a no-op, not a duplicate-key 500
        assert client.put(f"/private-sessions/{_SID}", headers=_H).status_code == 200
        n = conn.execute(
            "SELECT count(*) FROM private_sessions WHERE session_id = %s", (_SID,)
        ).fetchone()[0]
        assert n == 1

        r = client.delete(f"/private-sessions/{_SID}", headers=_FULL_H)
        assert r.status_code == 200 and r.json() == {
            "status": "ok",
            "session_id": _SID,
            "private": False,
            "deleted": 1,
        }
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is False


def test_a_credential_that_matches_nothing_is_a_401_on_every_verb(devices, conn, db_url):
    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_H)
        for send in (client.put, client.get, client.delete):
            r = send(f"/private-sessions/{_SID}", headers={"Authorization": "Bearer nope"})
            assert r.status_code == 401
    assert _rows(conn) == 1


def test_a_restricted_device_cannot_delete(devices, conn, db_url):
    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_H)
        r = client.delete(f"/private-sessions/{_SID}", headers=_WORK_H)
        assert r.status_code == 403
        assert r.json() == {
            "status": "error",
            "detail": "un-privating a session needs a full-trust device",
        }
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is True
    assert _rows(conn) == 1


def test_the_root_token_cannot_delete(devices, conn, db_url):
    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_WORK_H)
        r = client.delete(f"/private-sessions/{_SID}", headers=_H)
        assert r.status_code == 403
        assert r.json()["detail"] == "un-privating a session needs a full-trust device"
    assert _rows(conn) == 1


def test_a_full_trust_device_can_delete(devices, conn, db_url):
    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_WORK_H)
        assert _rows(conn) == 1
        r = client.delete(f"/private-sessions/{_SID}", headers=_FULL_H)
        assert r.status_code == 200 and r.json()["deleted"] == 1
    assert _rows(conn) == 0


@pytest.mark.parametrize("headers", [_H, _WORK_H], ids=["root", "restricted"])
def test_put_and_get_stay_on_the_client_gate(devices, conn, db_url, headers):
    with _client(db_url) as client:
        r = client.put(f"/private-sessions/{_SID}", headers=headers)
        assert r.status_code == 200 and r.json()["private"] is True
        r = client.get(f"/private-sessions/{_SID}", headers=headers)
        assert r.status_code == 200 and r.json()["private"] is True
    assert _rows(conn) == 1


@pytest.fixture()
def real_server(devices, db_url, monkeypatch):
    """``mcp_server.server`` reloaded against the test database with ``_TOKEN`` as its root
    token, so the private-session routes are the ones ``auxiliary_routes`` registers with
    the server's own gates, then put back as it was."""
    monkeypatch.setenv("SYNAPSE_ENV_FILE", "/nonexistent/synapse-test.env")
    for key in _SERVER_AUTH_KEYS:
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("SYNAPSE_MACHINE_TOKEN", _TOKEN)
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    import mcp_server.server as s

    yield importlib.reload(s)
    monkeypatch.undo()
    importlib.reload(s)


def test_the_real_server_puts_the_delete_behind_the_full_trust_gate(real_server, conn):
    """The unit-level gates above are the test's own. Here the routes come from the server's
    registration, so a restricted device, which clears the ordinary device gate (its PUT
    lands), still has to be refused the DELETE: the server must hand the routes its
    full-trust gate, not the ordinary one."""
    with TestClient(real_server.mcp.http_app()) as client:
        r = client.put(f"/private-sessions/{_SID}", headers=_WORK_H)
        assert r.status_code == 200 and r.json()["private"] is True

        r = client.delete(f"/private-sessions/{_SID}", headers=_WORK_H)
        assert r.status_code == 403
        assert r.json()["detail"] == "un-privating a session needs a full-trust device"
        assert _rows(conn) == 1

        r = client.delete(f"/private-sessions/{_SID}", headers=_FULL_H)
        assert r.status_code == 200 and r.json()["deleted"] == 1
    assert _rows(conn) == 0


def test_db_predicate_sees_the_flag(clean, db_url):
    """``Database.is_private_session`` is what the ingest chokepoint calls."""
    from ingestion.db import Database
    from ingestion.private_sessions import PrivateSessions

    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_H)
    db = Database(db_url)
    try:
        assert PrivateSessions(db).is_private(_SID) is True
        assert PrivateSessions(db).is_private(_PUBLIC) is False
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The chokepoint: /ingest drops flagged sessions
# ---------------------------------------------------------------------------


class _Req:
    """Minimal stand-in for the Starlette Request ingest_turns consumes."""

    def __init__(self, body: dict) -> None:
        self._body = body
        self.headers: dict[str, str] = {}

    async def json(self) -> dict:
        return self._body


def _records(session_id: str) -> list[dict]:
    return [
        {
            "type": "user",
            "uuid": f"{session_id}-u1",
            "sessionId": session_id,
            "timestamp": "2026-08-01T10:00:00.000Z",
            "cwd": "/tmp/demo",
            "message": {"content": "what is the chunk window"},
        },
        {
            "type": "assistant",
            "uuid": f"{session_id}-a1",
            "sessionId": session_id,
            "timestamp": "2026-08-01T10:00:05.000Z",
            "cwd": "/tmp/demo",
            "message": {"content": [{"type": "text", "text": "four episodes, step three"}]},
        },
    ]


async def test_ingest_drops_private_session_turns(clean, conn, db_url, monkeypatch):
    """A flagged session posts fine and stores NOTHING; an unflagged one still lands.

    This is the guarantee the marker file cannot give: any client (catch-up sweep,
    re-import, another machine) that posts the same transcript is dropped here."""
    from mcp_server import server

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "MACHINE_TOKEN", "")  # open route in this harness

    with _client(db_url) as client:
        client.put(f"/private-sessions/{_SID}", headers=_H)

    resp = await server.ingest_turns(_Req({"records": _records(_SID), "source": "hook"}))
    assert resp.status_code == 200
    stored = conn.execute(
        "SELECT count(*) FROM episodes WHERE session_id = %s", (_SID,)
    ).fetchone()[0]
    assert stored == 0

    resp = await server.ingest_turns(_Req({"records": _records(_PUBLIC), "source": "hook"}))
    assert resp.status_code == 200
    stored = conn.execute(
        "SELECT count(*) FROM episodes WHERE session_id = %s", (_PUBLIC,)
    ).fetchone()[0]
    assert stored == 1
