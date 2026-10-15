"""The episode writer: one set of guards, one transaction, for /ingest and every backfill.

DB-backed (runs against the scratch test database). Covers what the parsers never see: a
failure mid-call leaves nothing stored (the episode and its queue row land together), a
replay is dropped whichever way it arrives, and a turn deleted from memory stays deleted.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from ingestion.backfill import backfill
from ingestion.db import Database
from ingestion.db_episodes import DELETED_TURNS_PROBE
from ingestion.episode_writer import store_episodes
from ingestion.models import Episode
from mcp_server.dashboard_routes import _delete_episode


@pytest.fixture(scope="module")
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean(clean_tables, conn):
    conn.execute("TRUNCATE private_sessions RESTART IDENTITY")
    DELETED_TURNS_PROBE.reset()
    yield
    DELETED_TURNS_PROBE.reset()


def _tombstones(conn) -> int | None:
    """Rows in deleted_turns, or None on a database without it (schema 123 is optional)."""
    if not conn.execute("SELECT to_regclass('deleted_turns')").fetchone()[0]:
        return None
    return conn.execute("SELECT count(*) FROM deleted_turns").fetchone()[0]


@pytest.fixture()
def tombstones(conn):
    """For the tests whose point is the tombstone: skipped until schema 123 is applied."""
    if _tombstones(conn) is None:
        pytest.skip("schema 123 (deleted_turns) is not applied")


def _ep(session: str, seq: int, tag: str, project: str | None = "proj", **kw) -> Episode:
    """A turn whose content, and so its md5, is unique to ``tag``."""
    kw.setdefault("content", f"[user] question {tag}\n\n[assistant] answer {tag}")
    return Episode(session_id=session, sequence=seq, project=project, span_id=f"jsonl:{tag}", **kw)


def _episodes(conn, session: str | None = None) -> list[tuple]:
    where, args = ("WHERE session_id = %s", (session,)) if session else ("", ())
    return conn.execute(
        f"SELECT session_id, sequence, span_id FROM episodes {where} ORDER BY id", args
    ).fetchall()


def _queued(conn) -> list[int]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT episode_id FROM extraction_queue WHERE content_type = 'episode' ORDER BY id"
        ).fetchall()
    ]


def test_new_turns_append_past_the_high_water_mark_and_are_queued(db, conn):
    for seq in (1, 2, 3):
        db.upsert_episode(_ep("w-1", seq, f"old-{seq}"))
    # A tail parse numbers from 1; the writer renumbers from the stored high-water mark.
    stats = store_episodes(db, [_ep("w-1", 1, "new-a"), _ep("w-1", 2, "new-b")])
    assert stats.written == 2
    assert [r[1] for r in _episodes(conn, "w-1")] == [1, 2, 3, 4, 5]
    ids = [r[0] for r in conn.execute("SELECT id FROM episodes WHERE sequence > 3 ORDER BY id")]
    assert _queued(conn) == ids


def test_a_reshipped_turn_is_a_no_op(db, conn):
    eps = [_ep("w-2", 1, "a"), _ep("w-2", 2, "b")]
    assert store_episodes(db, eps).written == 2
    again = store_episodes(db, eps)
    assert again.written == 0
    assert len(_episodes(conn)) == 2 and len(_queued(conn)) == 2


def test_a_missing_private_sessions_table_does_not_undo_the_calls_earlier_writes(db, db_url, conn):
    """is_private_session tolerates a database without schema 050; inside an atomic block
    that tolerance must not roll back what the block already wrote."""
    conn.execute("ALTER TABLE private_sessions RENAME TO private_sessions_away")
    try:
        with db.atomic():
            db.upsert_episode(_ep("w-050", 1, "a"))
            assert db.is_private_session("w-050") is False
            db.upsert_episode(_ep("w-050", 2, "b"))
    finally:
        conn.execute("ALTER TABLE private_sessions_away RENAME TO private_sessions")
    assert [r[1] for r in _episodes(conn, "w-050")] == [1, 2]


def test_057_rolled_back_mid_call_keeps_the_calls_earlier_writes(db_url, conn):
    """The probe said the provenance column exists and then it did not: the upsert falls back
    to the pre-057 statement for that turn without undoing the turns written before it."""
    from ingestion.surfaces import OWN_SURFACE_PROBE
    from tests.helpers.pre057 import pre057_schema

    with pre057_schema(conn, db_url) as (url, schema):
        pre = Database(url)
        try:
            OWN_SURFACE_PROBE._ok[url] = True
            stats = store_episodes(pre, [_ep("w-057", 1, "a"), _ep("w-057", 2, "b")])
        finally:
            pre.close()
        assert stats.written == 2
        assert conn.execute(f"SELECT count(*) FROM {schema}.episodes").fetchone()[0] == 2
        assert conn.execute(f"SELECT count(*) FROM {schema}.extraction_queue").fetchone()[0] == 2


def test_a_pooled_call_borrows_one_connection_for_all_its_turns(db_url, conn):
    """A POST used to borrow ~5 times per new turn, each paying the pool's check, BEGIN and
    COMMIT; the whole call now holds one borrow and commits once."""
    from unittest.mock import patch

    from ingestion import pg_pool

    url = f"{db_url}{'&' if '?' in db_url else '?'}application_name=episode_writer_borrows"
    pg_pool.open_pool(url)
    borrows: list[int] = []
    real = pg_pool.connection

    def counting(*args, **kwargs):
        borrows.append(1)
        return real(*args, **kwargs)

    try:
        pooled = Database(url)
        with patch.object(pg_pool, "connection", counting):
            stats = store_episodes(pooled, [_ep("w-pool", n, f"p{n}") for n in (1, 2, 3)])
    finally:
        pg_pool.close_pool(url)
    assert stats.written == 3
    assert len(borrows) == 1
    assert len(_episodes(conn)) == 3 and len(_queued(conn)) == 3


def test_a_failed_enqueue_leaves_nothing_stored_and_the_retry_lands_both(db, conn, monkeypatch):
    """The episode used to commit before its queue row: a failure in between left a turn
    the retry skipped by span_id, so it never reached the timeline and preferences gates."""
    real = db.enqueue_extraction

    def boom(item):
        raise RuntimeError("postgres blipped")

    monkeypatch.setattr(db, "enqueue_extraction", boom)
    with pytest.raises(RuntimeError):
        store_episodes(db, [_ep("w-3", 1, "a")])
    assert _episodes(conn) == []

    monkeypatch.setattr(db, "enqueue_extraction", real)
    assert store_episodes(db, [_ep("w-3", 1, "a")]).written == 1
    assert len(_queued(conn)) == 1


def test_a_failure_on_a_later_turn_rolls_back_the_whole_call(db, conn, monkeypatch):
    real, calls = db.enqueue_extraction, []

    def second_fails(item):
        calls.append(item)
        if len(calls) == 2:
            raise RuntimeError("postgres blipped")
        real(item)

    monkeypatch.setattr(db, "enqueue_extraction", second_fails)
    with pytest.raises(RuntimeError):
        store_episodes(db, [_ep("w-4", 1, "a"), _ep("w-4", 2, "b")])
    assert _episodes(conn) == [] and _queued(conn) == []
    # the connection is usable afterwards, and a retry stores both
    monkeypatch.setattr(db, "enqueue_extraction", real)
    assert store_episodes(db, [_ep("w-4", 1, "a"), _ep("w-4", 2, "b")]).written == 2


def test_identical_content_under_a_fresh_session_and_span_is_a_replay(db, conn):
    first = _ep("w-5a", 1, "same")
    assert store_episodes(db, [first]).written == 1
    replay = first.model_copy(update={"session_id": "w-5b", "span_id": "jsonl:fresh"})
    stats = store_episodes(db, [replay])
    assert (stats.written, stats.replays) == (0, 1)
    # the guard is project-scoped: the same words in another silo are a new turn
    other = replay.model_copy(update={"project": "elsewhere"})
    assert store_episodes(db, [other]).written == 1


def test_a_span_id_stored_under_another_session_is_a_replay_not_a_crash(db, conn):
    assert store_episodes(db, [_ep("w-6a", 1, "shared")]).written == 1
    # same span_id (a resumed session re-dumps the old uuids), different words and project
    moved = Episode(
        session_id="w-6b", sequence=1, project="other", span_id="jsonl:shared", content="[user] x"
    )
    stats = store_episodes(db, [moved])
    assert (stats.written, stats.replays) == (0, 1)


def test_turns_without_a_span_id_are_counted_not_stored(db, conn):
    keyless = Episode(session_id="w-7", sequence=1, content="[user] no key")
    stats = store_episodes(db, [keyless])
    assert (stats.written, stats.unkeyed) == (0, 1)
    assert _episodes(conn) == []


def test_contamination_scratch_and_private_turns_are_dropped(db, conn):
    conn.execute("INSERT INTO private_sessions (session_id) VALUES ('w-8-private')")
    eps = [
        _ep("w-8", 1, "a", content="[user] you are a certified court reporter doing verbatim x"),
        _ep("w-8", 2, "b", project="axon-test"),
        _ep("w-8-private", 1, "c"),
        _ep("w-8", 3, "d"),
    ]
    stats = store_episodes(db, eps)
    assert (stats.written, stats.dropped, stats.private) == (1, 2, 1)


def test_project_default_labels_only_the_unlabelled(db, conn):
    eps = [_ep("w-9", 1, "a", project=None), _ep("w-9", 2, "b", project="kept")]
    store_episodes(db, eps, project_default="fallback")
    rows = conn.execute("SELECT project FROM episodes ORDER BY sequence").fetchall()
    assert [r[0] for r in rows] == ["fallback", "kept"]


def test_a_nul_byte_in_a_turn_does_not_reach_the_episode_and_the_queue_row_holds_no_text(db, conn):
    ep = Episode(
        session_id="w-10", sequence=1, span_id="jsonl:nul", content="[user] a\x00b", project="p"
    )
    assert store_episodes(db, [ep]).written == 1
    assert conn.execute("SELECT content FROM episodes").fetchone()[0] == "[user] ab"
    # The worker reads the turn from the episode: the queue keeps no copy of it.
    assert conn.execute("SELECT content FROM extraction_queue").fetchone()[0] == ""


# ---------------------------------------------------------------------------
# Deleted turns stay deleted
# ---------------------------------------------------------------------------


def _delete(db_url, db, session: str, seq: int) -> dict:
    eid = db.get_session_episodes(session)[seq - 1]["id"]
    summary = _delete_episode(db_url, eid)
    assert summary is not None
    return summary


def test_a_deleted_turn_does_not_come_back_on_a_reship(tombstones, db, db_url, conn):
    eps = [_ep("t-1", 1, "keep"), _ep("t-1", 2, "doomed")]
    store_episodes(db, eps)
    assert _delete(db_url, db, "t-1", 2)["tombstoned"] == 1

    # The Stop hook re-ships the last turn; a sweep or backfill re-reads the whole file.
    stats = store_episodes(db, eps)
    assert (stats.written, stats.tombstoned) == (0, 1)
    assert [r[2] for r in _episodes(conn)] == ["jsonl:keep"]


def test_a_deleted_turn_does_not_come_back_in_a_replayed_session(tombstones, db, db_url, conn):
    doomed = _ep("t-2", 1, "doomed", metadata={"cwd": "/work/app"})
    store_episodes(db, [doomed])
    _delete(db_url, db, "t-2", 1)

    replay = doomed.model_copy(update={"session_id": "t-2-retry", "span_id": "jsonl:fresh"})
    stats = store_episodes(db, [replay])
    assert (stats.written, stats.tombstoned) == (0, 1)
    # the same words from another project and folder are not the deleted turn
    other = replay.model_copy(update={"project": "other", "metadata": {"cwd": "/work/other"}})
    assert store_episodes(db, [other]).written == 1


def test_the_backfill_honours_a_tombstone(tombstones, db, db_url, conn, tmp_path):
    root = _transcript_dir(tmp_path, "t-3", ["one", "two"])
    assert backfill("claude-code", root, db)["episodes_written"] == 2
    eid = conn.execute("SELECT id FROM episodes ORDER BY sequence DESC LIMIT 1").fetchone()[0]
    assert _delete_episode(db_url, eid)["tombstoned"] == 1

    assert backfill("claude-code", root, db)["episodes_written"] == 0
    assert len(_episodes(conn)) == 1


def test_without_the_tombstone_table_a_delete_still_works_and_nothing_is_checked(db, db_url, conn):
    """The migration is optional: an image running ahead of it deletes as before."""
    DELETED_TURNS_PROBE.missing("")
    DELETED_TURNS_PROBE.missing(db_url)
    eps = [_ep("t-4", 1, "a")]
    store_episodes(db, eps)
    assert _delete(db_url, db, "t-4", 1)["tombstoned"] == 0
    assert not _tombstones(conn)  # no table, or nothing recorded in it
    assert store_episodes(db, eps).written == 1  # nothing recorded, so the turn returns


# ---------------------------------------------------------------------------
# The backfill driver
# ---------------------------------------------------------------------------


def _transcript_dir(tmp_path: Path, session: str, tags: list[str], cwd: str = "/work/app") -> Path:
    """A Claude Code transcript: one user/assistant exchange per tag."""
    root = tmp_path / session
    root.mkdir()
    recs = []
    for tag in tags:
        base = {"sessionId": session, "cwd": cwd, "timestamp": "2026-09-01T12:00:00Z"}
        recs.append({**base, "type": "user", "uuid": f"u-{session}-{tag}",
                     "message": {"role": "user", "content": f"q {tag}"}})  # fmt: skip
        recs.append({**base, "type": "assistant", "uuid": f"a-{session}-{tag}",
                     "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {tag}"}]}})  # fmt: skip
    (root / f"{session}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    return root


def test_backfill_stores_once_and_skips_replayed_sessions(db, conn, tmp_path):
    root = _transcript_dir(tmp_path, "b-1", ["one", "two"])
    assert backfill("claude-code", root, db)["episodes_written"] == 2
    assert backfill("claude-code", root, db)["episodes_written"] == 0

    # A retried session: the same turns, a new session id and new uuids. /ingest drops these
    # as replays; the disk sweep used to store every one and pay to extract it again.
    retry = _transcript_dir(tmp_path, "b-2", ["one", "two"])
    for f in retry.glob("*.jsonl"):
        f.write_text(f.read_text().replace("b-2-", "fresh-"))
    stats = backfill("claude-code", retry, db)
    assert (stats["episodes_parsed"], stats["episodes_written"]) == (2, 0)


def test_backfill_dry_run_writes_nothing(db, conn, tmp_path):
    root = _transcript_dir(tmp_path, "b-3", ["one"])
    stats = backfill("claude-code", root, db, dry_run=True)
    assert (stats["episodes_parsed"], stats["episodes_written"]) == (1, 0)
    assert _episodes(conn) == []


def test_backfill_labels_unlabelled_turns_with_the_project_flag(db, conn, tmp_path):
    export = tmp_path / "conversations.json"
    export.write_text(
        json.dumps(
            [
                {
                    "uuid": "ai-1",
                    "name": "Chat",
                    "chat_messages": [
                        {"uuid": "m1", "sender": "human", "text": "hi", "created_at": "2026-09-01"},
                        {"uuid": "m2", "sender": "assistant", "text": "yo", "created_at": "2026-09-02"},
                    ],
                }
            ]
        )
    )  # fmt: skip
    stats = backfill("claude-ai", export, db, project="chats")
    assert stats["episodes_written"] == 1
    assert conn.execute("SELECT project, platform FROM episodes").fetchone() == (
        "chats",
        "claude_ai",
    )


# ---------------------------------------------------------------------------
# POST /ingest: the route stores through the same writer
# ---------------------------------------------------------------------------


def _ingest_client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.ingest_route import register

    m = FastMCP("test-episode-writer-ingest")
    register(m, lambda: db_url, lambda r: True)
    return TestClient(m.http_app())


def _records(session: str, tag: str) -> list[dict]:
    base = {"sessionId": session, "cwd": "/work/app", "timestamp": "2026-09-01T12:00:00Z"}
    return [
        {**base, "type": "user", "uuid": f"u-{tag}", "message": {"role": "user", "content": f"q {tag}"}},
        {**base, "type": "assistant", "uuid": f"a-{tag}",
         "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {tag}"}]}},
    ]  # fmt: skip


def test_ingest_retry_after_a_failed_enqueue_stores_the_turn_and_its_queue_row(
    db_url, conn, monkeypatch
):
    tag, sid = uuid.uuid4().hex, f"r-{uuid.uuid4().hex}"
    body = {"records": _records(sid, tag), "project": "app"}
    client = _ingest_client(db_url)

    def boom(self, item):
        raise RuntimeError("postgres blipped")

    with monkeypatch.context() as m:
        m.setattr(Database, "enqueue_extraction", boom)
        assert client.post("/ingest", json=body).status_code == 500
    assert _episodes(conn, sid) == []  # not the episode without its queue row

    r = client.post("/ingest", json=body)
    assert r.status_code == 200 and r.json()["ingested"] == 1
    assert len(_episodes(conn, sid)) == 1 and len(_queued(conn)) == 1
    assert client.post("/ingest", json=body).json()["ingested"] == 0
