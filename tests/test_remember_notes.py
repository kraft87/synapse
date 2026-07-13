"""DB-backed tests for the reworked remember() MCP tool (notes form, PR-1b).

Covers both call forms against the real Postgres test DB with the LLM and the
embedder stubbed out: the embedder maps hook text to deterministic one-hot
vectors (so KNN similarity is exact), and ``parse_with_retry`` inside
``ingestion.notes`` is monkeypatched so no model call ever leaves the process.
Asserted here: the episode archive + extraction enqueue stay fed on every call,
the note reconcile outcomes (created / updated / superseded), fail-open
semantics, the telemetry row, and the input-validation error dicts.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

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

import ingestion.notes as notes_mod  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.embedding import embed_dims  # noqa: E402
from ingestion.notes import _OWNER  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402

_DIMS = embed_dims()
GROUP = "technical"  # _group_for(None) routes project-less notes here


def _vec(slot: int) -> list[float]:
    """One-hot unit vector: identical slots -> cosine sim 1, distinct -> 0."""
    v = [0.0] * _DIMS
    v[slot % _DIMS] = 1.0
    return v


class _SlotEmb:
    """Deterministic embedder: hook text -> one-hot vector via ``mapping`` (default slot 0)."""

    model_name = "test-embed"

    def __init__(self) -> None:
        self.mapping: dict[str, int] = {}

    def embed(self, texts, task):
        return [_vec(self.mapping.get(t, 0)) for t in texts]


def _remember(**kw):
    """Drive the async tool the way FastMCP would — on an event loop."""
    return asyncio.run(server.remember(**kw))


@pytest.fixture()
def env(monkeypatch, conn, db_url):
    """Clean tables + a fully stubbed server: test-DB DSN, fresh recall engine,
    slot embedder, recorded/controllable LLM confirm. No network anywhere."""
    conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    conn.execute("DELETE FROM notes")

    engine = Recall(db_url, "")
    emb = _SlotEmb()
    llm_calls: list[dict] = []
    relation = {"value": "same", "raise": False}

    def _fake_parse(*a, **k):
        llm_calls.append(k)
        if relation["raise"]:
            raise RuntimeError("LLM down")
        return relation["value"]

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", engine)
    monkeypatch.setattr(server, "_notes_deps", lambda: (emb, object()))
    monkeypatch.setattr(notes_mod, "parse_with_retry", _fake_parse)

    db = Database(db_url)
    yield SimpleNamespace(
        conn=conn, db=db, engine=engine, emb=emb, llm_calls=llm_calls, relation=relation
    )
    db.close()


def _seed_note(env, *, hook="User prefers dark mode", type="user", slot=1, project=None) -> int:
    return env.db.insert_note(
        owner_id=_OWNER,
        group_id=GROUP,
        project=project,
        type=type,
        hook=hook,
        body="Existing body.",
        embedding=_vec(slot),
        embed_model="test-embed",
        source_ref=None,
    )


def _note_row(env, note_id):
    return env.conn.execute(
        "SELECT hook, body, type, project, source_ref, superseded_by FROM notes WHERE id = %s",
        (note_id,),
    ).fetchone()


# ---------------------------------------------------------------------------
# Legacy content form
# ---------------------------------------------------------------------------


def test_legacy_content_writes_episode_extraction_and_note(env):
    content = "Chose provider A because latency won. Full rationale lives here."
    out = _remember(content=content)

    assert out["status"] == "ok" and out["outcome"] == "created"
    assert out["note_id"] and out["episode_id"] and out["session_id"]

    # Episode archive still fed, exactly as before the rework.
    ep = env.conn.execute(
        "SELECT id, content, source, project FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    assert ep is not None and ep[1] == content and ep[2] == "manual"

    # Extraction enqueued so the KG stays fed.
    q = env.conn.execute("SELECT episode_id, content_type, status FROM extraction_queue").fetchall()
    assert (out["episode_id"], "manual", "pending") in q

    # Note created with the derived hook (first sentence, <=120), type 'project'.
    hook, body, type_, project, source_ref, superseded_by = _note_row(env, out["note_id"])
    assert hook == "Chose provider A because latency won."
    assert body == content
    assert type_ == "project" and project is None
    assert source_ref == f"ep:{out['episode_id']}"
    assert superseded_by is None


def test_legacy_hook_derivation_truncates_to_120(env):
    content = "x" * 300  # no sentence boundary — falls back to the whole first line
    out = _remember(content=content)
    hook = _note_row(env, out["note_id"])[0]
    assert hook == "x" * 120


# ---------------------------------------------------------------------------
# Structured form — reconcile outcomes
# ---------------------------------------------------------------------------


def test_structured_create(env):
    out = _remember(hook="User prefers dark mode", body="Full body.", type="user")
    assert out["status"] == "ok" and out["outcome"] == "created"
    hook, body, type_, _, source_ref, _ = _note_row(env, out["note_id"])
    assert hook == "User prefers dark mode" and body == "Full body." and type_ == "user"
    assert source_ref == f"ep:{out['episode_id']}"
    # The structured form still archives an episode + enqueues extraction.
    ep = env.conn.execute(
        "SELECT content, source FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    assert ep is not None and ep[1] == "manual" and "Full body." in ep[0]
    assert env.conn.execute("SELECT count(*) FROM extraction_queue").fetchone()[0] == 1
    assert env.llm_calls == []  # empty live set -> no confirm call


def test_high_sim_same_type_confirm_same_updates(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1  # sim 1.0 to the seeded note
    env.relation["value"] = "same"

    out = _remember(hook="User prefers light mode", body="New body.", type="user")
    assert out["outcome"] == "updated"
    assert out["note_id"] == old  # same row, refreshed in place
    hook, body, *_ = _note_row(env, old)
    assert hook == "User prefers light mode" and body == "New body."
    assert len(env.llm_calls) == 1


def test_high_sim_contradicts_supersedes(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User now prefers light mode"] = 1
    env.relation["value"] = "contradicts"

    out = _remember(hook="User now prefers light mode", body="Flipped.", type="user")
    assert out["outcome"] == "superseded"
    assert out["note_id"] != old
    assert _note_row(env, old)[5] == out["note_id"]  # old row's superseded_by -> new id
    assert _note_row(env, out["note_id"])[5] is None


def test_below_threshold_creates(env):
    _seed_note(env, slot=1)
    env.emb.mapping["Unrelated fact about the deploy"] = 2  # sim 0.0

    out = _remember(hook="Unrelated fact about the deploy", body="B.", type="user")
    assert out["outcome"] == "created"
    assert env.llm_calls == []


def test_different_type_high_sim_creates(env):
    _seed_note(env, slot=1, type="user")
    env.emb.mapping["Never suggest dark mode"] = 1  # sim 1.0 but different type

    out = _remember(hook="Never suggest dark mode", body="B.", type="feedback")
    assert out["outcome"] == "created"
    assert env.llm_calls == []  # type mismatch skips the confirm entirely


def test_confirm_kill_switch_updates_without_llm(env, monkeypatch):
    monkeypatch.setenv("SYNAPSE_NOTES_CONFIRM", "0")
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1
    env.relation["value"] = "contradicts"  # would supersede — but the switch wins

    out = _remember(hook="User prefers light mode", body="B.", type="user")
    assert out["outcome"] == "updated" and out["note_id"] == old
    assert env.llm_calls == []


def test_llm_failure_fails_open_to_update(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1
    env.relation["raise"] = True

    out = _remember(hook="User prefers light mode", body="B.", type="user")
    assert out["outcome"] == "updated" and out["note_id"] == old
    assert len(env.llm_calls) == 1


# ---------------------------------------------------------------------------
# Telemetry, caps, validation
# ---------------------------------------------------------------------------


def test_telemetry_row_with_outcome_envelope(env):
    out = _remember(hook="User prefers dark mode", body="B.", type="user")
    # The metrics write is fire-and-forget on a single-worker executor: a no-op
    # barrier submitted after it completes only once the row is in.
    env.engine._async_executor.submit(lambda: None).result(timeout=10)
    row = env.conn.execute(
        "SELECT source, ms_total, served_ids FROM recall_metrics WHERE kind = 'remember'"
    ).fetchone()
    assert row is not None
    assert row[0] == "mcp-tool" and row[1] is not None
    assert row[2] == {"note": out["note_id"], "outcome": "created", "type": "user"}


def test_hook_over_200_chars_is_capped(env):
    out = _remember(hook="H" * 300, body="B.", type="user")
    assert out["status"] == "ok" and out["outcome"] == "created"
    assert _note_row(env, out["note_id"])[0] == "H" * 200


def test_invalid_type_returns_error_dict(env):
    out = _remember(hook="h", body="b", type="bogus")
    assert out["status"] == "error" and "invalid type" in out["detail"]
    # Validation runs before any write — no episode, no note.
    assert env.conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0
    assert env.conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


def test_neither_form_returns_error_dict(env):
    for kw in ({}, {"hook": "only a hook"}, {"body": "only a body"}):
        out = _remember(**kw)
        assert out["status"] == "error" and "hook + body" in out["detail"]
    assert env.conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0
