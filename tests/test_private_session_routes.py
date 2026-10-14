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
from types import SimpleNamespace

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

from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust  # noqa: E402
from mcp_server.private_session_routes import register  # noqa: E402

_TOKEN = "test-machine-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_SID = "s-private-route-test"
_PUBLIC = "s-public-route-test"

#: Who a request resolves to. The root token and any unrecognised bearer are UNKNOWN; a
#: device token yields the credential-bound verdict of its surfaces row.
_FULL = SurfaceTrust(surface_id="dev-full", trust="full", known=True, credential_bound=True)
_WORK = SurfaceTrust(surface_id="dev-work", known=True, credential_bound=True)


def _client(db_url, trust: SurfaceTrust = UNKNOWN_SURFACE):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-private")
    register(test_mcp, db_url, authorized, lambda request: trust)
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


# ---------------------------------------------------------------------------
# Turning private mode on mid-session: what was already stored
# ---------------------------------------------------------------------------


def _turn(conn, session_id: str, seq: int, surface_id: str | None = None) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content, surface_id) "
        "VALUES (%s, %s, 'demo', 'turn', %s) RETURNING id",
        (session_id, seq, surface_id),
    ).fetchone()[0]


@pytest.fixture()
def derived(clean, conn):
    """Two stored turns of the private session, one of a public one, and rows derived from
    them. Everything is tagged so the fixture can wipe it."""
    from psycopg.types.json import Json

    a, b = _turn(conn, _SID, 1), _turn(conn, _SID, 2)
    pub = _turn(conn, _PUBLIC, 1)
    conn.execute(
        "INSERT INTO kg_entities (uuid, group_id, name) VALUES ('pm-e', 'technical', 'pm-e')"
    )
    for uuid, eps in (
        ("pm-sole", [a]),
        ("pm-shared", [b, pub]),
        ("pm-pub", [pub]),
        # Cite both private turns: one fact each, however many of its turns are deleted.
        ("pm-both", [a, b]),
        ("pm-both-shared", [a, b, pub]),
    ):
        conn.execute(
            "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, name, fact, "
            "episodes) VALUES (%s, 'technical', 'pm-e', 'pm-e', 'rel', %s, %s)",
            (uuid, uuid, Json(eps)),
        )
    conn.execute(
        "INSERT INTO chunks (session_id, start_sequence, end_sequence, episode_ids, project, "
        "content) VALUES (%s, 1, 2, %s, 'demo', 'pm-chunk')",
        (_SID, Json([a, b])),
    )
    conn.execute(
        "INSERT INTO timeline_events (t_valid, fact, source, source_ref) "
        "VALUES (now(), 'pm-event', 'chat', %s), (now(), 'pm-pub-event', 'chat', %s)",
        (f"ep:{a}#2", f"ep:{pub}"),
    )
    prefs = {}
    for key, ref in (("private", f"ep:{b}"), ("public", f"ep:{pub}"), ("older", f"ep:{pub}")):
        prefs[key] = conn.execute(
            "INSERT INTO preferences (owner_id, group_id, pref, polarity, source_ref) "
            "VALUES ('default', 'technical', %s, 'like', %s) RETURNING id",
            (f"pm-{key}", ref),
        ).fetchone()[0]
    # The private turn's preference superseded an older one; the pointer must not dangle.
    conn.execute(
        "UPDATE preferences SET t_invalid = now(), superseded_by = %s WHERE id = %s",
        (prefs["private"], prefs["older"]),
    )
    yield {"turns": (a, b), "public": pub, "prefs": prefs}
    conn.execute("DELETE FROM chunks WHERE content = 'pm-chunk'")
    conn.execute("DELETE FROM kg_relationships WHERE uuid LIKE 'pm-%'")
    conn.execute("DELETE FROM kg_entities WHERE uuid = 'pm-e'")
    conn.execute("DELETE FROM timeline_events WHERE fact LIKE 'pm-%'")
    conn.execute("DELETE FROM preferences WHERE pref LIKE 'pm-%'")
    conn.execute("DELETE FROM dashboard_audit WHERE item_id = %s", (_SID,))


def test_put_reports_the_turns_already_stored_and_keeps_them(derived, conn, db_url):
    """Private mode covers what happens from now on. Turns stored before the toggle are
    counted so the client can say so, and stay unless a purge is asked for by name."""
    with _client(db_url) as client:
        r = client.put(f"/private-sessions/{_SID}", headers=_H)
        assert r.status_code == 200
        assert r.json()["private"] is True and r.json()["stored_turns"] == 2
        assert "purged" not in r.json()
        r = client.put(f"/private-sessions/{_PUBLIC}-unseen", headers=_H)
        assert r.json()["stored_turns"] == 0
    conn.execute("DELETE FROM private_sessions WHERE session_id = %s", (f"{_PUBLIC}-unseen",))
    n = conn.execute("SELECT count(*) FROM episodes WHERE session_id = %s", (_SID,)).fetchone()[0]
    assert n == 2


def test_put_with_purge_deletes_the_turns_and_what_was_derived_from_them(derived, conn, db_url):
    pub, prefs = derived["public"], derived["prefs"]
    with _client(db_url, _FULL) as client:
        r = client.put(f"/private-sessions/{_SID}?purge=1", headers=_H)
        assert r.status_code == 200, r.text
        body = r.json()
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is True
    assert body["stored_turns"] == 0
    assert body["purged"] == {
        "episodes_deleted": 2,
        "chunks_deleted": 1,
        "facts_deleted": 2,
        "facts_unlinked": 2,
        "events_deleted": 1,
        "preferences_deleted": 1,
    }

    def ids(sql):
        return {r[0] for r in conn.execute(sql).fetchall()}

    assert ids(f"SELECT session_id FROM episodes WHERE session_id IN ('{_SID}', '{_PUBLIC}')") == {
        _PUBLIC
    }
    assert ids("SELECT content FROM chunks WHERE content = 'pm-chunk'") == set()
    facts = dict(conn.execute("SELECT uuid, episodes FROM kg_relationships WHERE uuid LIKE 'pm-%'"))
    assert facts == {"pm-shared": [pub], "pm-both-shared": [pub], "pm-pub": [pub]}
    assert ids("SELECT fact FROM timeline_events WHERE fact LIKE 'pm-%'") == {"pm-pub-event"}
    assert ids("SELECT pref FROM preferences WHERE pref LIKE 'pm-%'") == {"pm-public", "pm-older"}
    older = conn.execute(
        "SELECT superseded_by FROM preferences WHERE id = %s", (prefs["older"],)
    ).fetchone()[0]
    assert older is None
    audit = conn.execute(
        "SELECT action, detail FROM dashboard_audit WHERE item_id = %s", (_SID,)
    ).fetchall()
    assert audit == [("purge_private_session", {**body["purged"], "surface_id": "dev-full"})]


def test_purge_must_be_asked_for_exactly(clean, conn, db_url):
    """A destructive flag is never guessed at: anything but 0 or 1 is refused before the
    session is touched."""
    _turn(conn, _SID, 1)
    with _client(db_url, _FULL) as client:
        r = client.put(f"/private-sessions/{_SID}?purge=yes", headers=_H)
        assert r.status_code == 400
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is False
    n = conn.execute("SELECT count(*) FROM episodes WHERE session_id = %s", (_SID,)).fetchone()[0]
    assert n == 1


def _stored(conn, session_id: str = _SID) -> int:
    return conn.execute(
        "SELECT count(*) FROM episodes WHERE session_id = %s", (session_id,)
    ).fetchone()[0]


def _audits(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM dashboard_audit WHERE item_id = %s", (_SID,)
    ).fetchone()[0]


def test_the_root_token_flags_but_cannot_purge(derived, conn, db_url):
    """The root token is on every machine that ever ran the plugin, and deleting memory is
    not a client-gate act: the purge is refused, the flag it came with stays set, and
    nothing is deleted or audited."""
    with _client(db_url, UNKNOWN_SURFACE) as client:
        r = client.put(f"/private-sessions/{_SID}?purge=1", headers=_H)
        assert r.status_code == 403
        assert r.json()["status"] == "error" and "nothing was deleted" in r.json()["detail"]
        assert client.get(f"/private-sessions/{_SID}", headers=_H).json()["private"] is True
        # The flag alone is still the client gate's to set.
        assert client.put(f"/private-sessions/{_SID}", headers=_H).json()["stored_turns"] == 2
    assert _stored(conn) == 2 and _audits(conn) == 0


def test_a_restricted_device_purges_only_sessions_it_wrote(clean, conn, db_url):
    """Provenance (057) is what lets a restricted work laptop clear its own session without
    being able to wipe one another device wrote, or one stamped before 057."""
    _turn(conn, _SID, 1, "dev-work")
    _turn(conn, _SID, 2, "dev-work")
    _turn(conn, _PUBLIC, 1, "dev-work")
    _turn(conn, _PUBLIC, 2, "dev-other")
    with _client(db_url, _WORK) as client:
        # Another surface's turn in the session: refused, nothing deleted.
        r = client.put(f"/private-sessions/{_PUBLIC}?purge=1", headers=_H)
        assert r.status_code == 403
        assert _stored(conn, _PUBLIC) == 2
        # An unstamped (pre-057) turn is nobody's to delete but a full-trust device's.
        conn.execute("UPDATE episodes SET surface_id = NULL WHERE session_id = %s", (_PUBLIC,))
        assert client.put(f"/private-sessions/{_PUBLIC}?purge=1", headers=_H).status_code == 403
        assert _stored(conn, _PUBLIC) == 2

        r = client.put(f"/private-sessions/{_SID}?purge=1", headers=_H)
        assert r.status_code == 200, r.text
        assert r.json()["purged"]["episodes_deleted"] == 2 and r.json()["stored_turns"] == 0
    assert _stored(conn) == 0 and _stored(conn, _PUBLIC) == 2
    audit = conn.execute(
        "SELECT detail FROM dashboard_audit WHERE item_id = %s", (_SID,)
    ).fetchall()
    assert [d["surface_id"] for (d,) in audit] == ["dev-work"]


def test_a_restricted_device_cannot_purge_before_provenance_exists(
    clean, conn, db_url, monkeypatch
):
    """Without schema 057 no stamp can prove a turn is the device's own."""
    from ingestion.surfaces import OWN_SURFACE_PROBE

    _turn(conn, _SID, 1, "dev-work")
    monkeypatch.setattr(OWN_SURFACE_PROBE, "ready", lambda *a, **k: False)
    with _client(db_url, _WORK) as client:
        assert client.put(f"/private-sessions/{_SID}?purge=1", headers=_H).status_code == 403
    assert _stored(conn) == 1


def test_a_self_reported_surface_is_not_provenance(clean, conn, db_url):
    """A known restricted row reached through the hostname lane is not credential-bound,
    so it carries no stamp to match."""
    _turn(conn, _SID, 1, "dev-work")
    legacy = SurfaceTrust(surface_id="dev-work", known=True)
    with _client(db_url, legacy) as client:
        assert client.put(f"/private-sessions/{_SID}?purge=1", headers=_H).status_code == 403
    assert _stored(conn) == 1


def test_purging_a_long_session_scans_the_provenance_tables_once(clean, conn, db_url, monkeypatch):
    """The cascade is set-based: statements against the big tables do not grow with the
    session, or a 200-turn purge outruns the plugin's request timeout. Counted by watching
    the SQL the purge sends."""
    import psycopg

    seen: list[str] = []
    real = psycopg.Cursor.execute

    def spy(self, query, *a, **k):
        seen.append(query if isinstance(query, str) else query.as_string())
        return real(self, query, *a, **k)

    monkeypatch.setattr(psycopg.Cursor, "execute", spy)
    for seq in range(1, 41):
        _turn(conn, _SID, seq)
    seen.clear()
    with _client(db_url, _FULL) as client:
        assert client.put(f"/private-sessions/{_SID}?purge=1", headers=_H).status_code == 200
    scans = [
        q
        for q in seen
        if q.startswith("SELECT") and ("FROM kg_relationships" in q or "FROM chunks" in q)
    ]
    assert len(scans) == 2, scans


def test_a_missing_private_sessions_table_is_reported_as_the_error_it_is(conn, db_url):
    """Schema 050 is not optional (the boot guard refuses to start without it). A
    deployment that got past the guard anyway sees the database's own error, never a
    success: the CLI must not tell the user "off the record" on a write that failed."""
    from tests.helpers.schemaless import schemaless_dsn

    with schemaless_dsn(conn, db_url) as dsn, _client(dsn) as client:
        r = client.put(f"/private-sessions/{_SID}", headers=_H)
    assert r.status_code == 500 and "private_sessions" in r.json()["detail"]


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
        self.scope: dict = {}
        self.state = SimpleNamespace()

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
