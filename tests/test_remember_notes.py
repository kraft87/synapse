"""DB-backed tests for the reworked remember() MCP tool (notes form, PR-1b).

Covers both call forms against the real Postgres test DB with the LLM and the
embedder stubbed out: the embedder maps hook text to deterministic one-hot
vectors (so KNN similarity is exact), and ``parse_with_retry`` inside
``ingestion.notes`` is monkeypatched so no model call ever leaves the process.
Asserted here: the episode archive and when its extraction is queued, the note
reconcile outcomes (created / updated / superseded), insert-when-unsure
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
from ingestion.notes import _OWNER  # noqa: E402
from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402
from tests.helpers.embed import GROUP, onehot  # noqa: E402

# GROUP ("technical") is where _group_for(None) routes project-less notes.

_DUP = {"verdict": "DUPLICATE", "keep": None}
_DISTINCT = {"verdict": "DISTINCT", "keep": None}
_NEW_SURVIVES = {"verdict": "CORRECTION", "keep": 2}


class _SlotEmb:
    """Deterministic embedder: hook text -> one-hot vector via ``mapping`` (default slot 0)."""

    model_name = "test-embed"

    def __init__(self) -> None:
        self.mapping: dict[str, int] = {}

    def embed(self, texts, task):
        return [onehot(self.mapping.get(t, 0)) for t in texts]


def _remember(**kw):
    """Drive the async tool the way FastMCP would — on an event loop."""
    return asyncio.run(server.remember(**kw))


@pytest.fixture()
def env(monkeypatch, conn, db_url):
    """Clean tables + a fully stubbed server: test-DB DSN, fresh recall engine,
    slot embedder, recorded/controllable LLM judge. No network anywhere."""
    conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    conn.execute("DELETE FROM notes")

    engine = Recall(db_url, "")
    emb = _SlotEmb()
    llm_calls: list[dict] = []
    relation = {"value": _DUP, "raise": False}

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
        embedding=onehot(slot),
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


def test_credentials_are_redacted_in_note_episode_and_extraction(env):
    key = "sk-" + "proj-" + "aB3dE5fG7hJ9kL2mN4pQ6rS8tU0vW1xY"
    out = _remember(
        hook=f"OpenRouter key is {key}", body=f"Set OPENROUTER_API_KEY={key} in .env.", type="user"
    )
    assert out["status"] == "ok"
    hook, body, *_ = _note_row(env, out["note_id"])
    (ep_content,) = env.conn.execute(
        "SELECT content FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    (queued,) = env.conn.execute("SELECT content FROM extraction_queue").fetchone()
    for text in (hook, body, ep_content, queued):
        assert key not in text and "[REDACTED:openai]" in text


def test_high_sim_same_type_duplicate_updates(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1  # sim 1.0 to the seeded note
    env.relation["value"] = _DUP

    out = _remember(hook="User prefers light mode", body="New body.", type="user")
    assert out["outcome"] == "updated"
    assert out["note_id"] == old  # same row, refreshed in place
    hook, body, *_ = _note_row(env, old)
    assert hook == "User prefers light mode" and body == "New body."
    assert len(env.llm_calls) == 1
    # The wording it replaced is kept (the seed's body), and provenance moved to the new episode.
    assert env.conn.execute(
        "SELECT hook, body FROM notes_history WHERE note_id = %s", (old,)
    ).fetchall() == [("User prefers dark mode", "Existing body.")]
    assert _note_row(env, old)[4] == f"ep:{out['episode_id']}"


def test_high_sim_correction_supersedes(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User now prefers light mode"] = 1
    env.relation["value"] = _NEW_SURVIVES

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


def test_distinct_verdict_inserts_beside_the_near_note(env):
    """Two feedback rules on one topic stay two notes."""
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1
    env.relation["value"] = _DISTINCT

    out = _remember(hook="User prefers light mode", body="B.", type="user")
    assert out["outcome"] == "created" and out["note_id"] != old
    assert _note_row(env, old)[0] == "User prefers dark mode"  # untouched
    assert _note_row(env, old)[5] is None  # still live


def test_llm_failure_inserts_instead_of_overwriting(env):
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1
    env.relation["raise"] = True

    out = _remember(hook="User prefers light mode", body="B.", type="user")
    assert out["outcome"] == "created" and out["note_id"] != old
    assert _note_row(env, old)[0] == "User prefers dark mode"
    assert len(env.llm_calls) == 1


# ---------------------------------------------------------------------------
# Telemetry, caps, validation
# ---------------------------------------------------------------------------


def test_telemetry_row_with_outcome_envelope(env):
    out = _remember(hook="User prefers dark mode", body="B.", type="user")
    # The metrics write is fire-and-forget on a single-worker executor: a no-op
    # barrier submitted after it completes only once the row is in.
    env.engine._async_executor.submit(lambda: None).result(timeout=10)
    # Newest row only: a stale fire-and-forget write from a prior test's engine
    # can land after this test's TRUNCATE — never let it win the assertion.
    row = env.conn.execute(
        "SELECT source, ms_total, served_ids FROM recall_metrics"
        " WHERE kind = 'remember' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "mcp-tool" and row[1] is not None
    assert row[2] == {
        "note": out["note_id"],
        "outcome": "created",
        "type": "user",
        # No surface and no restricted-project match -> the fail-closed default tier.
        "audience": "personal",
    }


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


def test_blank_inputs_return_error_dicts(env):
    # Degenerate values must never create junk notes — error dict, zero writes.
    for kw in (
        {"hook": "", "body": ""},
        {"hook": "   ", "body": "b"},
        {"hook": "h", "body": " \n\t "},
        {"content": ""},
        {"content": "   \n "},
    ):
        out = _remember(**kw)
        assert out["status"] == "error", kw
    assert env.conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0
    assert env.conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


def test_hook_with_content_but_no_body_errors_not_legacy(env):
    # hook + content (no body) is the STRUCTURED form with body missing — it
    # must error, never silently take the legacy path and discard the hook.
    out = _remember(hook="User prefers dark mode", content="Some content.")
    assert out["status"] == "error" and "body" in out["detail"]
    assert env.conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0
    assert env.conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


def test_multiline_hook_stored_single_line(env):
    out = _remember(hook="line one\n\tline   two\r\nline three", body="B.", type="user")
    assert out["status"] == "ok"
    assert _note_row(env, out["note_id"])[0] == "line one line two line three"


def test_structured_values_stored_stripped(env):
    out = _remember(hook="  User prefers dark mode  ", body="  Full body.\n", type="user")
    hook, body, *_ = _note_row(env, out["note_id"])
    assert hook == "User prefers dark mode" and body == "Full body."


def test_content_alongside_full_pair_is_episode_text(env):
    out = _remember(hook="Hook line", body="Body text.", content="Episode text.", type="user")
    assert out["status"] == "ok"
    hook, body, *_ = _note_row(env, out["note_id"])
    assert hook == "Hook line" and body == "Body text."  # note keeps hook + body
    ep = env.conn.execute(
        "SELECT content FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    assert ep[0] == "Episode text."


def test_notes_deps_llm_construction_failure_degrades_to_none(monkeypatch):
    # A misconfigured provider must degrade to llm=None, never raise out of
    # _work after the episode write (fail-open confirm design).
    import ingestion.embedding as emb_mod

    monkeypatch.setattr(emb_mod, "create_embedder", lambda **kw: None)
    monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "bogus")
    _, llm = server._notes_deps()
    assert llm is None


def test_llm_none_inserts(env, monkeypatch):
    # End-to-end with the REAL parse_with_retry and llm=None: the judge call raises
    # inside _write_action's blanket except and the write inserts.
    from ingestion.llm_client import parse_with_retry as _real_parse

    monkeypatch.setattr(notes_mod, "parse_with_retry", _real_parse)
    monkeypatch.setattr(server, "_notes_deps", lambda: (env.emb, None))
    old = _seed_note(env, slot=1)
    env.emb.mapping["User prefers light mode"] = 1

    out = _remember(hook="User prefers light mode", body="B.", type="user")
    assert out["status"] == "ok"
    assert out["outcome"] == "created" and out["note_id"] != old


# ---------------------------------------------------------------------------
# The archive episode: sequence, extraction, replay
# ---------------------------------------------------------------------------

_DEVICE = SurfaceTrust(surface_id="laptop", trust="full", known=True, credential_bound=True)
_OAUTH = SurfaceTrust(surface_id="oauth:kyle", trust="full", known=True, credential_bound=True)


def _remember_as(trust, **kw):
    return asyncio.run(server._remember_as(trust=trust, hook="Hook", body="Body.", **kw))


def _queued(env):
    return env.conn.execute("SELECT episode_id FROM extraction_queue").fetchall()


def test_archive_joins_the_session_at_the_next_sequence(env):
    for seq in (1, 2, 5):
        env.conn.execute(
            "INSERT INTO episodes (session_id, sequence, content, source) VALUES ('s1', %s, 'turn', 'live')",
            (seq,),
        )
    out = _remember(hook="Hook", body="Body.", session_id="s1")
    row = env.conn.execute(
        "SELECT session_id, sequence, source FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone()
    assert row == ("s1", 6, "manual")
    assert (
        env.conn.execute("SELECT count(*) FROM episodes WHERE session_id = 's1'").fetchone()[0] == 4
    )


def test_parallel_remember_calls_keep_every_archive_episode(env):
    """Parallel tool calls in one turn used to read the same maximum sequence and then
    replace one another's archived episode."""
    from concurrent.futures import ThreadPoolExecutor

    def _call(i):
        return _remember(hook=f"Hook {i}", body=f"Body {i}.", session_id="s7", type="reference")

    with ThreadPoolExecutor(6) as pool:
        outs = list(pool.map(_call, range(6)))
    assert len({o["episode_id"] for o in outs}) == 6
    rows = env.conn.execute(
        "SELECT sequence FROM episodes WHERE session_id = 's7' ORDER BY sequence"
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3, 4, 5, 6]


def test_append_episode_serializes_a_burst_on_one_session(env):
    """Both read the same maximum; the later insert must take the next sequence, not
    overwrite (or collide with) the earlier archived episode."""
    from concurrent.futures import ThreadPoolExecutor

    from ingestion.models import Episode

    def _append(i):
        db = Database(env.db._url)
        try:
            return db.append_episode(
                Episode(session_id="s2", sequence=0, content=f"memory {i}", source="manual")
            )
        finally:
            db.close()

    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(_append, range(24)))
    rows = env.conn.execute(
        "SELECT sequence, content FROM episodes WHERE session_id = 's2' ORDER BY sequence"
    ).fetchall()
    assert len(set(ids)) == 24
    assert [r[0] for r in rows] == list(range(1, 25))
    assert {r[1] for r in rows} == {f"memory {i}" for i in range(24)}


def test_device_caller_with_a_session_is_archived_but_not_queued(env):
    """Its Stop hook ingests the session, so the statement is extracted from the turn."""
    out = _remember_as(_DEVICE, session_id="s3")
    assert out["status"] == "ok" and out["episode_id"]
    assert env.conn.execute(
        "SELECT source FROM episodes WHERE id = %s", (out["episode_id"],)
    ).fetchone() == ("manual",)
    assert _queued(env) == []


@pytest.mark.parametrize(
    ("trust", "session_id"),
    [
        (_DEVICE, None),  # no session to ingest: the archive is the only record
        (_OAUTH, "s4"),  # the claude.ai connector has no Stop hook
        (UNKNOWN_SURFACE, "s4"),  # the root token neither
    ],
    ids=["device-no-session", "oauth", "root-token"],
)
def test_hookless_callers_are_archived_and_queued(env, trust, session_id):
    out = _remember_as(trust, session_id=session_id)
    assert _queued(env) == [(out["episode_id"],)]


def test_archived_episode_id_skips_archive_and_extraction(env):
    first = _remember_as(_OAUTH, session_id="s5")
    env.conn.execute("TRUNCATE extraction_queue")
    again = _remember_as(_OAUTH, session_id="s5", archived_episode_id=first["episode_id"])
    assert again["episode_id"] == first["episode_id"]
    assert env.conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 1
    assert _queued(env) == []
    assert again["note_id"] == first["note_id"]  # same text: converged on the note


def test_on_archived_reports_the_episode_once_stored(env):
    seen = []
    out = _remember_as(_DEVICE, session_id="s6", on_archived=seen.append)
    assert seen == [out["episode_id"]]
