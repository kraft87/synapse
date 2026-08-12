"""DB-backed tests for private mode's server half (schema 050).

Two things only a database can prove:

  * the toggle routes (``mcp_server/private_session_routes``) round-trip the durable flag
    the CLI verifies — PUT is idempotent, GET reads it back, DELETE is the deliberate
    un-private escape hatch — and they're gated by the machine token like every sibling;
  * ``/ingest`` actually DROPS turns from a flagged session. That's the chokepoint the
    whole design rests on: the marker file only covers the live hook, so a catch-up sweep,
    a backfill, or any other client posting the same transcript must still land nothing.

Skips cleanly when no test DB is reachable (the pure-logic coverage — marker handling,
predicate, toggle CLI — lives in test_private_mode.py).
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

from mcp_server.private_session_routes import register  # noqa: E402

_TOKEN = "test-machine-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_SID = "s-private-route-test"
_PUBLIC = "s-public-route-test"


def _client(db_url):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-private")
    register(test_mcp, db_url, authorized)
    return TestClient(test_mcp.http_app())


@pytest.fixture()
def clean(conn):
    def _wipe():
        conn.execute("DELETE FROM private_sessions WHERE session_id = ANY(%s)", ([_SID, _PUBLIC],))
        conn.execute("DELETE FROM episodes WHERE session_id = ANY(%s)", ([_SID, _PUBLIC],))

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Toggle routes
# ---------------------------------------------------------------------------


def test_routes_require_machine_token(clean, db_url):
    with _client(db_url) as client:
        for send in (client.put, client.get, client.delete):
            r = send(f"/private-sessions/{_SID}")
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}


def test_put_get_delete_round_trip(clean, conn, db_url):
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

        r = client.delete(f"/private-sessions/{_SID}", headers=_H)
        assert r.status_code == 200 and r.json() == {
            "status": "ok",
            "session_id": _SID,
            "private": False,
            "deleted": 1,
        }
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is False


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
