"""scripts/rename_owner_in_facts.py against a real database.

The rules for which words change are pinned in tests/test_rename_owner_in_facts.py. What
only a database shows is that the script reads the right facts, writes what the plan says
and can take it back:

* candidates are the owner's live, non-web facts that hold a "user" word;
* the dry run writes nothing (no row changes, no backup table);
* ``--apply`` copies each row to the dated backup table, then writes the new text with an
  embedding from the configured embedder (``task="document"``, as ingest embeds a fact);
  a second apply changes nothing;
* a row whose text changed after the plan was written is skipped, not overwritten;
* ``--restore`` puts the text and embedding back, except on a row edited since.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mcp_server import kg_pg
from tests.helpers.embed import onehot

_SPEC = importlib.util.spec_from_file_location(
    "rename_owner_in_facts",
    Path(__file__).resolve().parent.parent / "scripts" / "rename_owner_in_facts.py",
)
ro = importlib.util.module_from_spec(_SPEC)
sys.modules["rename_owner_in_facts"] = ro
_SPEC.loader.exec_module(ro)

DAY = "20990101"
TABLE = f"kg_owner_rename_{DAY}"


def _drop_backups(conn) -> None:
    for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_owner_rename_%%'"
    ).fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


@pytest.fixture()
def kg(conn, db_url):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        _drop_backups(conn)

    _wipe()
    rows = ro._connect(db_url, read_only=False)
    yield rows
    rows.close()
    _wipe()


def _edge(
    conn, uid: str, fact: str, *, owner: str = "default", retired: bool = False, web: bool = False
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, src_uuid, tgt_uuid, name, fact, "
        "  fact_embedding, episodes, created_at, t_valid, t_invalid, web_artifact_id) "
        "VALUES (%s, %s, 'a', 'b', 'RELATES', %s, %s::vector, '[]'::jsonb, '2026-01-02', "
        "        '2026-01-02', %s, %s)",
        (
            uid,
            owner,
            fact,
            kg_pg._vec_literal(onehot(2)),
            "2026-02-01" if retired else None,
            1 if web else None,
        ),
    )


def _row(conn, uid: str) -> dict:
    return conn.execute(
        "SELECT fact, fact_embedding::text AS emb FROM kg_relationships WHERE uuid = %s", (uid,)
    ).fetchone()


def _slot(emb: str) -> int:
    return json.loads(emb).index(1.0)


def _embed(calls: list[list[str]]):
    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [onehot(7) for _ in texts]

    return embed


def test_candidates_are_the_owners_live_non_web_facts_with_a_user_word(kg) -> None:
    _edge(kg, "live", "User owns a car")
    _edge(kg, "plural", "the panel lists what users are working on")
    _edge(kg, "retired", "User owned a bike", retired=True)
    _edge(kg, "web", "the user of the library calls init()", web=True)
    _edge(kg, "other", "User owns a boat", owner="someone-else")
    assert ro.candidates(kg, "default") == [{"uuid": "live", "fact": "User owns a car"}]


def test_dry_run_reports_and_writes_nothing(kg) -> None:
    _edge(kg, "e1", "User owns a car")
    calls: list[list[str]] = []
    plan = [{"uuid": "e1", "old": "User owns a car", "new": "Sam owns a car"}]
    counts = ro.apply(kg, plan, _embed(calls), DAY, dry_run=True)
    assert counts == {"planned": 1, "matching": 1, "written": 0, "skipped": 0}
    assert _row(kg, "e1")["fact"] == "User owns a car"
    assert calls == []
    assert kg.execute("SELECT to_regclass(%s) AS t", (TABLE,)).fetchone()["t"] is None


def test_apply_backs_up_then_writes_text_and_embedding(kg) -> None:
    _edge(kg, "e1", "User owns a car")
    _edge(kg, "e2", "User's sister visited")
    _edge(kg, "e3", "User plays piano")
    kg.execute("UPDATE kg_relationships SET fact = 'Sam plays piano daily' WHERE uuid = 'e3'")
    plan = [
        {"uuid": "e1", "old": "User owns a car", "new": "Sam owns a car"},
        {"uuid": "e2", "old": "User's sister visited", "new": "Sam's sister visited"},
        # Edited after the plan was written: skipped, not overwritten.
        {"uuid": "e3", "old": "User plays piano", "new": "Sam plays piano"},
        # Gone since the plan was written.
        {"uuid": "missing", "old": "User left", "new": "Sam left"},
    ]
    calls: list[list[str]] = []
    counts = ro.apply(kg, plan, _embed(calls), DAY)
    assert counts == {"planned": 4, "matching": 2, "written": 2, "skipped": 2}
    assert calls == [["Sam owns a car", "Sam's sister visited"]]
    assert _row(kg, "e1")["fact"] == "Sam owns a car"
    assert _slot(_row(kg, "e1")["emb"]) == 7
    assert _row(kg, "e3")["fact"] == "Sam plays piano daily"
    assert _slot(_row(kg, "e3")["emb"]) == 2
    backed = kg.execute(f'SELECT uuid, fact, written FROM "{TABLE}" ORDER BY uuid').fetchall()
    assert [(b["uuid"], b["fact"], b["written"]) for b in backed] == [
        ("e1", "User owns a car", "Sam owns a car"),
        ("e2", "User's sister visited", "Sam's sister visited"),
    ]

    # A second apply of the same plan finds nothing left to write.
    again = ro.apply(kg, plan, _embed(calls), DAY)
    assert again == {"planned": 4, "matching": 0, "written": 0, "skipped": 4}
    assert len(calls) == 1


def test_restore_puts_back_text_and_embedding_but_not_over_a_later_edit(kg) -> None:
    _edge(kg, "e1", "User owns a car")
    _edge(kg, "e2", "User's sister visited")
    plan = [
        {"uuid": "e1", "old": "User owns a car", "new": "Sam owns a car"},
        {"uuid": "e2", "old": "User's sister visited", "new": "Sam's sister visited"},
    ]
    ro.apply(kg, plan, _embed([]), DAY)
    kg.execute("UPDATE kg_relationships SET fact = 'Sam''s sister moved' WHERE uuid = 'e2'")
    assert ro.restore(kg, DAY) == {"backed_up": 2, "restored": 1}
    assert _row(kg, "e1")["fact"] == "User owns a car"
    assert _slot(_row(kg, "e1")["emb"]) == 2
    assert _row(kg, "e2")["fact"] == "Sam's sister moved"
