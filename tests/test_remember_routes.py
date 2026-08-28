"""DB-backed tests for the spooled-remember replay route (schema 052).

Two things only a database can prove:

  * ``POST /remember/spool`` performs the SAME write the remember() MCP tool does — a real
    note row, a real episode, a real extraction enqueue — over the machine-token lane that
    stayed up through the 2026-08-25 MCP OAuth outage;
  * the replay is IDEMPOTENT on the client's intent id. The plugin's flush only dequeues
    after a confirm, so a flush that dies in the confirm window re-posts; that retry must
    resolve to the recorded note instead of minting a second one.

Same shape as test_private_session_routes.py: skips cleanly when no test DB is reachable,
LLM and embedder stubbed so nothing leaves the process. The local spool half is covered in
test_remember_spool.py.
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

import ingestion.notes as notes_mod  # noqa: E402
from ingestion.notes import _OWNER  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402
from mcp_server.remember_routes import register  # noqa: E402
from tests.helpers.embed import onehot  # noqa: E402

_TOKEN = "test-machine-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_ROUTE = "/remember/spool"


class _SlotEmb:
    """Deterministic embedder: every hook lands in its own slot unless told otherwise, so
    two distinct notes never accidentally dedup onto each other."""

    model_name = "test-embed"

    def __init__(self) -> None:
        self.mapping: dict[str, int] = {}
        self._next = 1

    def embed(self, texts, task):
        out = []
        for t in texts:
            if t not in self.mapping:
                self.mapping[t] = self._next
                self._next += 1
            out.append(onehot(self.mapping[t]))
        return out


def _client(db_url):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-remember-spool")
    register(test_mcp, db_url, authorized, server.remember)
    return TestClient(test_mcp.http_app())


@pytest.fixture()
def env(monkeypatch, conn, db_url):
    """Clean tables + a fully stubbed server (test DSN, fresh recall engine, slot embedder,
    LLM confirm collapsed to 'same'). No network anywhere."""
    conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM remember_intents")

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", Recall(db_url, ""))
    emb = _SlotEmb()  # ONE instance: a fresh one per call would re-slot every hook to 1
    monkeypatch.setattr(server, "_notes_deps", lambda: (emb, object()))
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "same")
    yield conn
    conn.execute("DELETE FROM remember_intents")


def _payload(intent_id="i-1", **kw):
    return {
        "intent_id": intent_id,
        "hook": "User prefers absolute dates in notes",
        "body": "Stated 2026-08-25: relative dates rot; always write 2026-08-25 style.",
        "type": "user",
        **kw,
    }


# ---------------------------------------------------------------------------
# Auth + probe
# ---------------------------------------------------------------------------


def test_route_requires_the_machine_token(env, db_url):
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=_payload())
        assert r.status_code == 401
        assert r.json() == {"status": "error", "detail": "unauthorized"}
    assert env.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


def test_probe_confirms_reachability_without_writing(env, db_url):
    """What the plugin's SessionStart step calls: it must answer 'can I write' truthfully
    while touching nothing — a /health ping would say 'up' with a dead bearer."""
    with _client(db_url) as client:
        r = client.post(_ROUTE, json={"probe": True}, headers=_H)
    assert r.status_code == 200 and r.json() == {"status": "ok", "probe": True}
    assert env.execute("SELECT count(*) FROM notes").fetchone()[0] == 0
    assert env.execute("SELECT count(*) FROM remember_intents").fetchone()[0] == 0


def test_probe_is_gated_too(env, db_url):
    """An unauthorized probe must read as 'cannot write', not as 'reachable'."""
    with _client(db_url) as client:
        assert client.post(_ROUTE, json={"probe": True}).status_code == 401


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


def test_replay_writes_a_real_note_and_episode(env, db_url):
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=_payload(project="synapse"), headers=_H)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok" and out["outcome"] == "created"
    assert out["intent_id"] == "i-1" and out["note_id"] and out["episode_id"]

    note = env.execute(
        "SELECT hook, body, type, project, owner_id FROM notes WHERE id = %s", (out["note_id"],)
    ).fetchone()
    assert note[0] == "User prefers absolute dates in notes"
    assert note[1].startswith("Stated 2026-08-25:")
    assert note[2] == "user" and note[3] == "synapse" and note[4] == _OWNER

    # the episode archive + KG extraction stay fed, exactly as via the MCP tool
    ep = env.execute(
        "SELECT content, source FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    assert ep[1] == "manual" and "absolute dates" in ep[0]
    queued = env.execute(
        "SELECT count(*) FROM extraction_queue WHERE episode_id = %s", (out["episode_id"],)
    ).fetchone()[0]
    assert queued == 1

    ledger = env.execute(
        "SELECT status, note_id, episode_id, outcome FROM remember_intents WHERE intent_id = 'i-1'"
    ).fetchone()
    assert ledger == ("done", out["note_id"], out["episode_id"], "created")


def test_legacy_content_form_replays(env, db_url):
    with _client(db_url) as client:
        r = client.post(
            _ROUTE,
            json={"intent_id": "i-legacy", "content": "Chose provider A because latency won."},
            headers=_H,
        )
    out = r.json()
    assert r.status_code == 200 and out["status"] == "ok"
    hook = env.execute("SELECT hook, type FROM notes WHERE id = %s", (out["note_id"],)).fetchone()
    assert hook[0] == "Chose provider A because latency won." and hook[1] == "project"


def test_session_id_is_honoured(env, db_url):
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=_payload(session_id="s-spooled"), headers=_H)
    assert r.json()["session_id"] == "s-spooled"
    assert (
        env.execute("SELECT count(*) FROM episodes WHERE session_id = 's-spooled'").fetchone()[0]
        == 1
    )


# ---------------------------------------------------------------------------
# Idempotency — the reason the intent id exists
# ---------------------------------------------------------------------------


def test_retrying_a_confirmed_intent_writes_no_second_note(env, db_url):
    with _client(db_url) as client:
        first = client.post(_ROUTE, json=_payload(), headers=_H).json()
        second = client.post(_ROUTE, json=_payload(), headers=_H).json()

    assert second["status"] == "ok" and second["outcome"] == "duplicate"
    assert second["note_id"] == first["note_id"]
    assert second["episode_id"] == first["episode_id"]
    assert env.execute("SELECT count(*) FROM notes").fetchone()[0] == 1
    assert env.execute("SELECT count(*) FROM episodes").fetchone()[0] == 1


def test_a_pending_claim_is_re_run_not_abandoned(env, db_url):
    """A row left 'pending' means a prior attempt died mid-write. Skipping it would lose the
    note; re-running it cannot fork one, because reconcile_note converges on the same hook."""
    env.execute("INSERT INTO remember_intents (intent_id) VALUES ('i-crashed')")
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=_payload("i-crashed"), headers=_H)
    out = r.json()
    assert r.status_code == 200 and out["outcome"] == "created" and out["note_id"]
    assert env.execute(
        "SELECT status FROM remember_intents WHERE intent_id = 'i-crashed'"
    ).fetchone()[0] == ("done")


def test_distinct_intents_both_land(env, db_url):
    with _client(db_url) as client:
        a = client.post(_ROUTE, json=_payload("i-a", hook="Fact A", body="Body A."), headers=_H)
        b = client.post(_ROUTE, json=_payload("i-b", hook="Fact B", body="Body B."), headers=_H)
    assert a.json()["note_id"] != b.json()["note_id"]
    assert env.execute("SELECT count(*) FROM notes").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Validation — a 4xx tells the client to drop the line, so it must be right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ({}, "intent_id required"),
        ({"intent_id": "  "}, "intent_id required"),
        ({"intent_id": "x" * 201, "hook": "h", "body": "b"}, "intent_id too long"),
        ({"intent_id": "i-x"}, "provide hook + body"),
        ({"intent_id": "i-x", "hook": "h"}, "provide hook + body"),
        ({"intent_id": "i-x", "body": "b"}, "provide hook + body"),
    ],
)
def test_unreplayable_payloads_are_rejected(env, db_url, body, expected):
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=body, headers=_H)
    assert r.status_code == 400
    assert expected in r.json()["detail"]
    assert env.execute("SELECT count(*) FROM remember_intents").fetchone()[0] == 0


def test_a_write_refused_by_remember_is_a_400(env, db_url):
    """An invalid note type is the client's bug, not an outage: 400 so the flush drops the
    line instead of retrying it at every session start forever."""
    with _client(db_url) as client:
        r = client.post(_ROUTE, json=_payload(type="bogus"), headers=_H)
    assert r.status_code == 400 and "invalid type" in r.json()["detail"]
    assert env.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


def test_invalid_json_body(env, db_url):
    with _client(db_url) as client:
        r = client.post(_ROUTE, content=b"not json", headers=_H)
    assert r.status_code == 400 and r.json()["detail"] == "invalid JSON"


def test_route_is_a_no_op_without_a_db_url(env):
    """Same guard every sibling route module has — dev/stdio boots with no DB."""
    from fastmcp import FastMCP

    m = FastMCP("no-db")
    register(m, "", lambda r: True, server.remember)
    with TestClient(m.http_app()) as client:
        assert client.post(_ROUTE, json={"probe": True}, headers=_H).status_code == 404
