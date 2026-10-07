"""The root-token ``surface`` lane is closed, end to end (the live repro).

Every machine that ever ran the plugin has held the shared ROOT machine token. Until
this change, a root-token caller could name any surfaces row by id (``?surface=`` on
``GET /context``, a ``surface`` field on ``POST /recall`` or ``/remember/spool``, the
``surface`` argument on every MCP tool) and inherit that row's trust. A full-trust legacy
hostname row turned the shared token into a full read of personal memory.

These tests drive the REAL resolution functions in ``mcp_server.server`` (with the root
token and test database patched in) through the real route and tool code:

  * root + a full-trust row's id  -> UNKNOWN (restricted, empty allowlist), on HTTP and MCP;
  * root + no surface             -> UNKNOWN, unchanged;
  * device tokens                 -> their own row, whatever surface they name;
  * the write side                -> a body/arg surface cannot attribute a note.

OAuth identities are covered in test_audience_scoping.py and stay unchanged.
"""

from __future__ import annotations

import asyncio
import os

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from fastmcp import FastMCP  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from ingestion.db import Database  # noqa: E402
from ingestion.notes import _OWNER  # noqa: E402
from ingestion.surfaces import UNKNOWN_SURFACE  # noqa: E402
from mcp_server import board, recall_route, remember_routes, server  # noqa: E402
from mcp_server.auth_tokens import SynapseTokenVerifier  # noqa: E402
from tests.helpers.surfaces import (  # noqa: E402
    clear_surfaces,
    register_device,
    register_full,
    register_restricted,
)

_ROOT = "root-tok"
_FULL_DEVICE = "full-device-tok"
_WORK_DEVICE = "work-device-tok"
_FULL_ROW = "legacy-full-host"  # credential-less, trust='full': the schema 053-era row
_WORK_ROW = "legacy-work-host"  # credential-less, trust='restricted', allowlist ["alpha"]
_PERSONAL = "User keeps a personal journal"
_WORK_SAFE = "User prefers tabs over spaces"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _SpyEngine:
    """Records the trust verdict each serve was handed. Enforcement given a verdict is
    covered by test_audience_scoping.py; this file is about which verdict arrives."""

    def __init__(self) -> None:
        self.trusts: list = []

    def _seen(self, trust):
        self.trusts.append(trust)
        return {"episodes": [], "notes": []}

    def recall(self, **kw):
        return self._seen(kw["trust"])

    def recall_episodes(self, **kw):
        return self._seen(kw["trust"])

    def fetch(self, ids, **kw):
        return self._seen(kw["trust"])

    def fetch_session(self, **kw):
        return self._seen(kw["trust"])

    def record_event(self, *a, **kw):
        return None


@pytest.fixture()
def lane(conn, db_url, monkeypatch):
    """Two legacy id rows, two device tokens, a personal and a work-safe note, and the
    server's resolvers pointed at the test DB with ``root-tok`` as the root token."""

    def _wipe():
        conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
        conn.execute("DELETE FROM notes")
        clear_surfaces(conn)

    _wipe()
    register_full(conn, _FULL_ROW)
    register_restricted(conn, ["alpha"], _WORK_ROW)
    register_device(conn, _FULL_DEVICE, trust="full", surface_id="dev-home")
    register_device(conn, _WORK_DEVICE, trust="restricted", projects=["alpha"], surface_id="dev-w")
    db = Database(db_url)
    try:
        for hook, audience in ((_PERSONAL, "personal"), (_WORK_SAFE, "work-safe")):
            db.insert_note(
                owner_id=_OWNER,
                group_id="technical",
                project=None,
                type="user",
                hook=hook,
                body=f"Body of: {hook}",
                embedding=None,
                embed_model=None,
                source_ref=None,
                audience=audience,
            )
    finally:
        db.close()

    spy = _SpyEngine()
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "MACHINE_TOKEN", _ROOT)
    monkeypatch.setattr(server, "_recall_engine", spy)
    monkeypatch.setattr(server, "_notes_deps", lambda: (None, None))
    yield spy
    _wipe()


@pytest.fixture()
def http(lane, db_url):
    """The real /context, /recall and /remember/spool routes, wired to the server's own
    gate and bearer resolution exactly as server.py wires them."""

    def authenticated(request):
        return server.authenticated(request, server.MACHINE_TOKEN)

    m = FastMCP("test-root-surface-lane")
    board.register(
        m,
        db_url,
        server._machine_authorized,
        resolve_trust=server._request_trust,
        authenticated=authenticated,
    )
    recall_route.register(
        m,
        lambda: server._get_recall(),
        server._machine_authorized,
        server._request_trust,
        authenticated,
    )
    remember_routes.register(
        m,
        db_url,
        server._machine_authorized,
        server._remember_as,
        resolve_trust=server._request_trust,
    )
    with TestClient(m.http_app()) as client:
        yield client


# ---------------------------------------------------------------------------
# Plain HTTP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("named", [_FULL_ROW, "dev-home"], ids=["legacy-row", "device-row-id"])
def test_context_root_naming_a_full_trust_row_gets_the_unknown_board(http, named):
    """The live repro: root + a full-trust id returned the whole board, personal notes
    included. It now gets exactly what root + no surface gets."""
    named_board = http.get(f"/context?surface={named}", headers=_bearer(_ROOT)).json()
    bare_board = http.get("/context", headers=_bearer(_ROOT)).json()
    assert named_board["trust"] == bare_board["trust"] == "restricted"
    assert _PERSONAL not in named_board["text"]
    assert named_board["text"] == bare_board["text"]


def test_context_devices_are_served_their_own_row(http):
    full = http.get("/context", headers=_bearer(_FULL_DEVICE)).json()
    assert full["trust"] == "full" and _PERSONAL in full["text"]

    # A restricted device naming the full-trust row keeps its own scope (unchanged).
    work = http.get(f"/context?surface={_FULL_ROW}", headers=_bearer(_WORK_DEVICE)).json()
    assert work["trust"] == "restricted"
    assert _PERSONAL not in work["text"] and _WORK_SAFE in work["text"]


def test_recall_root_body_surface_resolves_unknown(http, lane):
    for body in ({"query": "q", "surface": _FULL_ROW}, {"query": "q"}):
        assert http.post("/recall", json=body, headers=_bearer(_ROOT)).status_code == 200
    assert lane.trusts == [UNKNOWN_SURFACE, UNKNOWN_SURFACE]

    lane.trusts.clear()
    http.post("/recall", json={"query": "q", "surface": _WORK_ROW}, headers=_bearer(_FULL_DEVICE))
    [device] = lane.trusts
    assert device.surface_id == "dev-home" and not device.restricted


def _spool(http, token, intent_id, **extra):
    payload = {"intent_id": intent_id, "hook": f"Spooled note {intent_id}", "body": "B."}
    r = http.post("/remember/spool", json={**payload, **extra}, headers=_bearer(token))
    assert r.status_code == 200, r.text
    return r.json()["audience"]


def test_spool_root_cannot_attribute_a_note_to_a_restricted_surface(http, conn):
    """Write side. Root + body surface naming a registered restricted row used to mark
    the note work-safe on that row's behalf. Now the root token writes as unknown."""
    conn.execute("DELETE FROM remember_intents")
    assert _spool(http, _ROOT, "root-claims-work", surface=_WORK_ROW) == "personal"
    # The restricted DEVICE still gets the symmetric default for its own writes...
    assert _spool(http, _WORK_DEVICE, "device-own") == "work-safe"
    # ...and a full-trust device naming a restricted row does not pick it up either.
    assert _spool(http, _FULL_DEVICE, "full-claims-work", surface=_WORK_ROW) == "personal"
    conn.execute("DELETE FROM remember_intents")


# ---------------------------------------------------------------------------
# MCP — tokens minted by the real verifier
# ---------------------------------------------------------------------------


def _verified(db_url, token):
    verifier = SynapseTokenVerifier(_ROOT, db_url, ["user"])
    access = asyncio.run(verifier.verify_token(token))
    assert access is not None
    return access


def test_mcp_root_token_tools_ignore_a_named_surface(lane, db_url, monkeypatch):
    root = _verified(db_url, _ROOT)
    monkeypatch.setattr(server, "get_access_token", lambda: root)

    server.recall("q", surface=_FULL_ROW)
    server.recall_full_turns("q", surface=_FULL_ROW)
    server.fetch(["e:1", "n:1"], surface=_FULL_ROW)
    server.fetch_session("s-1", surface=_FULL_ROW)
    assert lane.trusts == [UNKNOWN_SURFACE] * 4

    note = asyncio.run(
        server.remember(hook="Root claims work", body="B.", type="user", surface=_WORK_ROW)
    )
    assert note["audience"] == "personal"


def test_mcp_device_token_is_unchanged(lane, db_url, monkeypatch):
    device = _verified(db_url, _FULL_DEVICE)
    monkeypatch.setattr(server, "get_access_token", lambda: device)

    server.recall("q", surface=_WORK_ROW)
    [trust] = lane.trusts
    assert trust.surface_id == "dev-home" and not trust.restricted and trust.known
