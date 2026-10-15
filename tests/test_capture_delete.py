"""Deleting a turn, or purging a private session, deletes the research captured from it.

The artifact carries its turn (``parent_episode_id``) and session; the dashboard's
single-turn delete and private mode's purge share ``delete_episode_content``, which now
takes the captured pages, their chunks, the cold archive's rows and the facts the web lane
read out of them along with the turn. The dashboard tombstone that keeps a deleted turn from
coming back also keeps its sub-agents' research from being captured again.
"""

from __future__ import annotations

import pytest
from psycopg.types.json import Json

from ingestion import capture
from ingestion.db import Database
from ingestion.db_episodes import DELETED_TURNS_PROBE
from ingestion.episode_writer import store_episodes
from ingestion.surfaces import SurfaceTrust
from mcp_server.dashboard_routes import _delete_episode
from mcp_server.private_session_routes import _purge_session
from tests.test_capture_ingest import parse, records

pytestmark = pytest.mark.db

#: Each call fetches its own page: the same page twice is one capture.
_URL = {t: f"https://example.com/{t}" for t in ("b", "keep", "side", "s-5", "s-6")}

_FULL = SurfaceTrust(surface_id="dev-full", trust="full", known=True, credential_bound=True)


@pytest.fixture(scope="module")
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean(clean_tables, conn, monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")
    tables = "web_artifacts, research_archive, private_sessions, kg_relationships, kg_entities"
    conn.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")  # nosec B608 - fixed names
    DELETED_TURNS_PROBE.reset()
    yield
    conn.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")  # nosec B608 - fixed names
    DELETED_TURNS_PROBE.reset()


def _episode(conn, session: str, tag: str) -> int:
    return conn.execute(
        "SELECT id FROM episodes WHERE session_id = %s AND content LIKE %s",
        (session, f"%question {tag}%"),
    ).fetchone()[0]


def _titles(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT tool_use_id FROM web_artifacts ORDER BY id")]


def _archive(conn, session: str, parent: int | None) -> None:
    conn.execute(
        "INSERT INTO research_archive (kind, source_id, url, content, session_id, "
        "parent_episode_id) VALUES ('x_thread', %s, 'https://x.com/a/status/1', 'verbatim', %s, %s)",
        (f"thread:{session}:{parent}", session, parent),
    )


def test_deleting_a_turn_deletes_what_was_captured_from_it(db, db_url, conn):
    store_episodes(db, parse(records("a", "s-1") + records("b", "s-1", url=_URL["b"])))
    assert _titles(conn) == ["toolu_a", "toolu_b"]
    a, b = _episode(conn, "s-1", "a"), _episode(conn, "s-1", "b")
    art_a = conn.execute("SELECT id FROM web_artifacts WHERE tool_use_id = 'toolu_a'").fetchone()[0]
    conn.execute(
        "INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end) "
        "VALUES (%s, 0, 'chunk of a', 0, 1)",
        (art_a,),
    )
    _archive(conn, "s-1", a)
    _archive(conn, "s-1", b)

    summary = _delete_episode(db_url, a)

    assert summary is not None and summary["web_deleted"] == 2  # the page and the archive row
    assert _titles(conn) == ["toolu_b"]
    assert conn.execute("SELECT count(*) FROM web_chunks").fetchone()[0] == 0
    assert [r[0] for r in conn.execute("SELECT parent_episode_id FROM research_archive")] == [b]


def test_facts_read_out_of_a_deleted_page_go_unless_a_surviving_turn_backs_them(db, db_url, conn):
    store_episodes(db, parse(records("a", "s-2") + records("b", "s-2", url=_URL["b"])))
    a, b = _episode(conn, "s-2", "a"), _episode(conn, "s-2", "b")
    art = conn.execute("SELECT id FROM web_artifacts WHERE tool_use_id = 'toolu_a'").fetchone()[0]
    conn.execute("INSERT INTO kg_entities (uuid, group_id, name) VALUES ('w-e', 'technical', 'w')")
    for uuid, episodes in (("w-sole", []), ("w-backed", [b]), ("w-unrelated", [])):
        conn.execute(
            "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, name, fact, "
            "episodes, web_artifact_id) VALUES (%s, 'technical', 'w-e', 'w-e', 'rel', %s, %s, %s)",
            (uuid, uuid, Json(episodes), None if uuid == "w-unrelated" else art),
        )

    summary = _delete_episode(db_url, a)

    assert summary is not None and summary["deleted_fact_ids"] == ["w-sole"]
    facts = dict(conn.execute("SELECT uuid, web_artifact_id FROM kg_relationships ORDER BY uuid"))
    assert facts == {"w-backed": None, "w-unrelated": None}


def test_a_session_purge_also_takes_the_sub_agents_captures(db, db_url, conn):
    store_episodes(db, parse(records("a", "s-3")))
    store_episodes(db, parse(records("keep", "s-keep", url=_URL["keep"])))
    side = [{**r, "isSidechain": True} for r in records("side", "s-3", url=_URL["side"])[1:3]]
    store_episodes(db, [], extra_calls=capture.sidechain_tool_calls(side, lambda cwd: "app", None))
    assert _titles(conn) == ["toolu_a", "toolu_keep", "toolu_side"]
    _archive(conn, "s-3", None)  # a sub-agent's: no turn
    _archive(conn, "s-keep", None)

    summary = _purge_session(db_url, "s-3", _FULL)

    assert summary["web_deleted"] == 3
    assert _titles(conn) == ["toolu_keep"]
    assert [r[0] for r in conn.execute("SELECT session_id FROM research_archive")] == ["s-keep"]


def test_a_single_turn_delete_leaves_the_sub_agents_captures_of_its_session(db, db_url, conn):
    store_episodes(db, parse(records("a", "s-4")))
    side = [{**r, "isSidechain": True} for r in records("side", "s-4", url=_URL["side"])[1:3]]
    store_episodes(db, [], extra_calls=capture.sidechain_tool_calls(side, lambda cwd: "app", None))

    _delete_episode(db_url, _episode(conn, "s-4", "a"))

    assert _titles(conn) == ["toolu_side"]  # it belongs to no turn; a purge is the way to take it


def test_a_deleted_turns_sub_agent_research_is_not_captured_when_shipped_again(db, db_url, conn):
    store_episodes(db, parse(records("a", "s-5")))
    _delete_episode(db_url, _episode(conn, "s-5", "a"))
    assert _titles(conn) == []

    def side_calls(session: str):
        side = [
            {**r, "isSidechain": True} for r in records("side", session, url=_URL[session])[1:3]
        ]
        return capture.sidechain_tool_calls(side, lambda cwd: "app", None)

    # the transcript on disk still carries the sub-agent: its session has a tombstone now
    assert store_episodes(db, [], extra_calls=side_calls("s-5")).captured == 0
    assert _titles(conn) == []
    # another session's sub-agent is captured as ever
    assert store_episodes(db, [], extra_calls=side_calls("s-6")).captured == 1
    assert _titles(conn) == ["toolu_side"]
