"""schema/067_kg_one_memory.sql and scripts/pile_merge.py against a real database.

067 folds the 'personal' KG pile into 'technical'. What only a database shows:

  * the twin rule: 1:1, 1:2 (the first row by SURVIVOR_ORDER), 2:1, 2:0, no twin, the empty
    stored name, differing supertypes, the owner pair by flag, the flag that moves, two owners
    that never cross, and the owner's exclusion list
  * facts: re-pointed, regrouped, never deleted or retired, labels and provenance untouched
    (the 056 trigger does not fire); notes and preferences flip with their links intact
  * the file is two transactions: A (the merge) holds no ACCESS EXCLUSIVE lock and a clean
    re-run takes none; B (the tail) takes it only where the catalog says it is needed, retries
    a lock timeout, and a failure there leaves the merged data and the 066 stamp
  * a re-run folds what old code wrote in between, and changes nothing otherwise
  * ``pile_merge.py``: --preview counts equal the applied run's, --health fails on a
    straggler, --undo restores the exact rows (and reports drift instead of overwriting it),
    --exclude lists twins, and none of them prints fact or entity text

Fixtures put the shared test database back to the 066 state, and leave it merged (the state
the rest of the suite expects) afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from ingestion import schema_check
from ingestion.kg_pg_read import SURVIVOR_ORDER, KGPostgresReader
from mcp_server.recall import Recall
from tests.helpers.kg import _axis_list

_ROOT = Path(__file__).resolve().parent.parent
_SQL_066 = _ROOT / "schema" / "066_kg_fact_audience.sql"
_SQL_067 = _ROOT / "schema" / "067_kg_one_memory.sql"

_SPEC = importlib.util.spec_from_file_location("pile_merge", _ROOT / "scripts" / "pile_merge.py")
pm = importlib.util.module_from_spec(_SPEC)
sys.modules["pile_merge"] = pm
_SPEC.loader.exec_module(pm)

OWNER = "default"  # what the readers under test scope to
KG_TABLES = ("kg_entities", "kg_relationships", "notes", "preferences")
INDEXES = ("kg_rel_pair_any", "kg_rel_tgt_any", "kg_entities_norm_any")

pytestmark = pytest.mark.skipif(shutil.which("psql") is None, reason="psql not installed")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _psql(db_url: str, path: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    """Pipe a file to psql on stdin, the way the migration is applied to a real database."""
    with path.open() as f:
        return subprocess.run(
            ["psql", db_url, "-q", "-v", "ON_ERROR_STOP=1"],
            stdin=f,
            check=check,
            capture_output=True,
            text=True,
        )


def _wipe(conn) -> None:
    conn.execute(
        "TRUNCATE kg_relationships, kg_entities, notes, preferences RESTART IDENTITY CASCADE"
    )
    conn.execute("TRUNCATE episodes, recall_feedback RESTART IDENTITY CASCADE")
    for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'"
    ).fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


def _to_066(conn, db_url: str) -> None:
    """The database as it was before 067: the insert trigger, no column defaults, no owner-only
    indexes, stamped 066."""
    _wipe(conn)
    _psql(db_url, _SQL_066)
    for t in KG_TABLES:
        conn.execute(f"ALTER TABLE {t} ALTER COLUMN group_id DROP DEFAULT")
    for ix in INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {ix}")
    conn.execute("UPDATE synapse_meta SET value = '066' WHERE key = 'schema_version'")


@pytest.fixture()
def piles(conn, db_url):
    """A database at 066, empty. Afterwards: wiped and merged (067 applied), as the suite expects."""
    _to_066(conn, db_url)
    yield conn
    _wipe(conn)
    _psql(db_url, _SQL_067)


def _vec(slot: int) -> str:
    return "[" + ",".join(map(str, _axis_list(slot))) + "]"


def ent(
    conn,
    uid: str,
    name: str,
    *,
    group: str = "technical",
    owner: str = OWNER,
    norm: str | None = "<name>",
    supertype: str | None = None,
    summary: str | None = None,
    degree: int = 0,
    flag: bool = False,
    emb: int | None = None,
) -> None:
    """``norm="<name>"`` stores lower(trim(name)); pass ``""`` or ``None`` to store those."""
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, entity_supertype, "
        "  summary, degree, is_self, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
        (
            uid,
            owner,
            group,
            name,
            name.strip().lower() if norm == "<name>" else norm,
            supertype,
            summary,
            degree,
            flag,
            None if emb is None else _vec(emb),
        ),
    )


def fact(
    conn,
    uid: str,
    group: str,
    src: str,
    tgt: str,
    text: str = "a fact",
    *,
    owner: str = OWNER,
    audience: str | None = None,
    label_version: int | None = None,
    retired: bool = False,
    invalidated_by: str | None = None,
    episodes: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  audience, label_version, t_invalid, invalidated_by, episodes) "
        "VALUES (%s, %s, %s, %s, %s, 'RELATES', %s, %s, %s, %s, %s, %s::jsonb)",
        (
            uid,
            owner,
            group,
            src,
            tgt,
            text,
            audience,
            label_version,
            "2026-01-01T00:00:00+00:00" if retired else None,
            invalidated_by,
            episodes,
        ),
    )


def _note(conn, hook: str, group: str, *, superseded_by: int | None = None) -> int:
    return conn.execute(
        "INSERT INTO notes (owner_id, group_id, type, hook, body, superseded_by) "
        "VALUES (%s, %s, 'user', %s, 'body', %s) RETURNING id",
        (OWNER, group, hook, superseded_by),
    ).fetchone()[0]


def _pref(conn, pref: str, group: str) -> int:
    return conn.execute(
        "INSERT INTO preferences (owner_id, group_id, pref, polarity) VALUES (%s, %s, %s, 'like') "
        "RETURNING id",
        (OWNER, group, pref),
    ).fetchone()[0]


def world(c) -> None:
    """Both piles, with every shape of twin the rule distinguishes. Names are all distinct
    words so a printed report can be searched for them."""
    ent(c, "t-alpha", "Alpha", supertype="Tool", summary="short", degree=2)
    ent(c, "t-owner", "Overlord", supertype="Person", summary="", degree=5, flag=True)
    ent(c, "t-gamma1", "Gamma", supertype="Tool", summary="g1", degree=1)
    ent(c, "t-gamma2", "Gamma", supertype="Tool", summary="g2", degree=1)
    ent(c, "t-delta", "Delta", supertype="Tool")
    ent(c, "t-omega", "Omega", norm="")  # the stored name is empty: the key is the name
    ent(c, "t-sigma", "Sigma", supertype="Tool")
    ent(c, "t-o2-alpha", "Alpha", owner="o2")
    ent(
        c,
        "p-alpha",
        "Alpha",
        group="personal",
        supertype="Concept",
        summary="a much longer personal summary",
        degree=3,
        emb=3,
    )
    ent(c, "p-alpha2", "Alpha", group="personal")
    ent(c, "p-gamma", "Gamma", group="personal", supertype="Tool", summary="p", degree=1)
    ent(c, "p-eps", "Epsilon", group="personal", summary="e")
    ent(c, "p-zeta1", "Zeta", group="personal")
    ent(c, "p-zeta2", "Zeta", group="personal")
    ent(
        c,
        "p-me",
        "Narrator",
        group="personal",
        supertype="Person",
        summary="me",
        degree=7,
        flag=True,
    )
    ent(c, "p-delta", "Delta", group="personal", supertype="Person")
    ent(c, "p-omega", "Omega", group="personal")
    ent(c, "p-sigma", "  Sigma ", group="personal", norm="")  # the key falls back to the name
    ent(c, "p-o2-gamma", "Gamma", group="personal", owner="o2")
    fact(
        c,
        "f-tech1",
        "technical",
        "t-alpha",
        "t-owner",
        "Alpha is owned",
        audience="work-safe",
        label_version=1,
    )
    fact(c, "f-tech-nl", "technical", "t-gamma1", "t-delta", "tech unlabelled")
    fact(c, "f-tech-on-p", "technical", "t-delta", "p-gamma", "Delta uses Gamma")
    fact(c, "f-p1", "personal", "p-alpha", "p-eps", "p1")
    fact(c, "f-p2", "personal", "p-eps", "t-alpha", "mixed")
    fact(c, "f-p3", "personal", "p-me", "p-alpha", "me to alpha")
    fact(c, "f-p4", "personal", "p-zeta1", "p-zeta2", "zetas")
    fact(c, "f-dup-p", "personal", "p-alpha", "t-owner", "alpha  is OWNED")
    fact(c, "f-loop-p", "personal", "p-alpha", "t-alpha", "loop")
    fact(c, "f-new-p", "personal", "p-eps", "p-zeta1", "new")
    fact(
        c, "f-old-p", "personal", "p-eps", "p-zeta1", "old", retired=True, invalidated_by="f-new-p"
    )
    fact(
        c,
        "f-p-ws",
        "personal",
        "p-sigma",
        "p-omega",
        "explicit label",
        audience="work-safe",
        label_version=1,
    )
    pn1 = _note(c, "pn1", "personal")
    _note(c, "pn2", "personal", superseded_by=pn1)
    _note(c, "tn1", "technical")
    pp1 = _pref(c, "pp1", "personal")
    _pref(c, "tp1", "technical")
    c.execute("UPDATE preferences SET superseded_by = %s WHERE pref = 'tp1'", (pp1,))


@pytest.fixture()
def seeded(piles):
    world(piles)
    return piles


def merge(db_url: str) -> None:
    _psql(db_url, _SQL_067)


@pytest.fixture()
def merged(seeded, db_url):
    merge(db_url)
    return seeded


def rows(conn, sql: str, *params: Any) -> list[tuple]:
    return conn.execute(sql, params or None).fetchall()


def one(conn, sql: str, *params: Any) -> Any:
    r = conn.execute(sql, params or None).fetchone()
    return None if r is None else r[0]


def ent_row(conn, uid: str) -> dict[str, Any] | None:
    return one(conn, "SELECT to_jsonb(e) FROM kg_entities e WHERE uuid = %s", uid)


def fact_row(conn, uid: str) -> dict[str, Any] | None:
    return one(
        conn, "SELECT to_jsonb(r) - 'fact_embedding' FROM kg_relationships r WHERE uuid = %s", uid
    )


def run_counts(conn, run: int = 1) -> dict[str, Any]:
    return one(conn, "SELECT counts FROM kg_pile_merge_067_runs WHERE run_id = %s", run)


def fold_map(conn, run: int = 1) -> dict[str, tuple[str, str]]:
    return {
        loser: (survivor, rule)
        for loser, survivor, rule in rows(
            conn,
            "SELECT loser_uuid, survivor_uuid, rule FROM kg_pile_merge_067_entity_map WHERE run_id = %s",
            run,
        )
    }


def table_md5(conn) -> dict[str, str]:
    return {
        t: one(conn, f"SELECT md5(COALESCE(string_agg(x::text, '|' ORDER BY id), '')) FROM {t} x")
        for t in KG_TABLES
    }


# ---------------------------------------------------------------------------
# The twin rule (cases 1-11)
# ---------------------------------------------------------------------------


def test_a_one_to_one_twin_is_folded_into_its_survivor(merged):
    assert ent_row(merged, "p-alpha") is None
    survivor = ent_row(merged, "t-alpha")
    assert survivor["group_id"] == "technical"
    assert fold_map(merged)["p-alpha"] == ("t-alpha", "name")
    # the survivor keeps its identity and takes the longest summary, the first embedding, the summed degree
    assert survivor["name"] == "Alpha" and survivor["entity_supertype"] == "Tool"
    assert survivor["summary"] == "a much longer personal summary"
    assert survivor["embedding"] is not None
    assert survivor["degree"] == 2 + 3 + 0  # p-alpha and p-alpha2
    # its fact now hangs on the survivor
    f = fact_row(merged, "f-p1")
    assert (f["src_uuid"], f["tgt_uuid"], f["group_id"]) == ("t-alpha", "p-eps", "technical")


def test_among_several_technical_twins_the_first_by_survivor_order_wins(merged):
    assert fold_map(merged)["p-gamma"] == ("t-gamma1", "name")  # the lower id of the two
    other = ent_row(merged, "t-gamma2")
    assert other["summary"] == "g2" and other["degree"] == 1 and other["group_id"] == "technical"
    assert (
        rows(
            merged,
            "SELECT 1 FROM kg_relationships WHERE src_uuid = 't-gamma2' OR tgt_uuid = 't-gamma2'",
        )
        == []
    )
    assert ent_row(merged, "t-gamma1")["degree"] == 2  # 1 + p-gamma's 1


def test_a_flagged_technical_twin_beats_an_older_one(piles, db_url):
    """SURVIVOR_ORDER is is_self DESC first: the owner's own row wins over an older twin."""
    ent(piles, "t-first", "Twin", owner="o3")
    ent(piles, "t-flagged", "Twin", owner="o3", flag=True)
    ent(piles, "p-twin", "Twin", group="personal", owner="o3")
    merge(db_url)
    assert fold_map(piles)["p-twin"] == ("t-flagged", "name")


def test_two_personal_rows_fold_into_one_technical_survivor(merged):
    m = fold_map(merged)
    assert m["p-alpha"] == m["p-alpha2"] == ("t-alpha", "name")
    assert ent_row(merged, "p-alpha2") is None


def test_two_personal_rows_without_a_technical_twin_are_both_regrouped(merged):
    for uid in ("p-zeta1", "p-zeta2"):
        assert ent_row(merged, uid)["group_id"] == "technical"
        assert uid not in fold_map(merged)
    f = fact_row(merged, "f-p4")
    assert (f["src_uuid"], f["tgt_uuid"]) == ("p-zeta1", "p-zeta2")


def test_an_entity_without_a_twin_is_regrouped_with_every_other_column_unchanged(seeded, db_url):
    before = {u: ent_row(seeded, u) for u in ("p-eps", "p-zeta1", "p-o2-gamma")}
    merge(db_url)
    for uid, old in before.items():
        assert ent_row(seeded, uid) == {**old, "group_id": "technical"}


def test_an_empty_stored_name_falls_back_to_the_lowercased_name_on_either_side(merged):
    m = fold_map(merged)
    assert m["p-omega"] == ("t-omega", "name")  # the technical row has the empty name
    assert m["p-sigma"] == ("t-sigma", "name")  # the personal row does


def test_twins_with_different_supertypes_still_merge(merged):
    m = fold_map(merged)
    assert (
        m["p-alpha"][0] == "t-alpha" and m["p-delta"][0] == "t-delta"
    )  # Concept/Tool, Person/Tool
    c = run_counts(merged)
    assert c["supertype_differ"] == 2 and c["supertype_differ_person"] == 1
    assert c["supertype_pairs"] == {"Concept | Tool": 1, "Person | Tool": 1}


def test_the_owner_pair_merges_by_flag_whatever_the_names_and_one_flag_remains(merged):
    assert fold_map(merged)["p-me"] == ("t-owner", "owner-flag")
    assert ent_row(merged, "p-me") is None
    assert rows(merged, "SELECT uuid FROM kg_entities WHERE is_self") == [("t-owner",)]
    assert ent_row(merged, "t-owner")["degree"] == 12
    f = fact_row(merged, "f-p3")
    assert (f["src_uuid"], f["tgt_uuid"]) == ("t-owner", "t-alpha")


def test_the_flag_moves_to_the_survivor_when_the_owner_has_no_technical_flag(piles, db_url):
    ent(piles, "t-boss", "Boss", owner="o3")
    ent(piles, "p-boss", "Boss", group="personal", owner="o3", flag=True)
    merge(db_url)
    assert fold_map(piles)["p-boss"] == ("t-boss", "name")
    assert rows(piles, "SELECT uuid FROM kg_entities WHERE is_self") == [("t-boss",)]
    assert (
        one(
            piles, "SELECT flag_moved FROM kg_pile_merge_067_entity_map WHERE loser_uuid = 'p-boss'"
        )
        is True
    )


def test_a_flagged_personal_row_without_a_twin_is_regrouped_still_flagged(piles, db_url):
    ent(piles, "p-solo", "Solo", group="personal", owner="o4", flag=True)
    merge(db_url)
    assert fold_map(piles) == {}
    assert rows(piles, "SELECT group_id, is_self FROM kg_entities WHERE uuid = 'p-solo'") == [
        ("technical", True)
    ]


def test_two_owners_never_cross(merged):
    m = fold_map(merged)
    assert "p-o2-gamma" not in m  # o1 has a technical Gamma, o2 does not
    assert ent_row(merged, "p-o2-gamma")["group_id"] == "technical"
    assert m["p-alpha"][0] == "t-alpha"  # not o2's Alpha
    assert rows(merged, "SELECT DISTINCT owner_id FROM kg_pile_merge_067_entity_map") == [(OWNER,)]


# ---------------------------------------------------------------------------
# Facts (cases 12-17)
# ---------------------------------------------------------------------------


def test_a_personal_fact_on_a_technical_endpoint_is_only_regrouped(merged):
    f = fact_row(merged, "f-p2")
    assert (f["src_uuid"], f["tgt_uuid"], f["group_id"]) == ("p-eps", "t-alpha", "technical")


def test_a_technical_fact_on_a_folded_entity_is_repointed_and_stays_technical(merged):
    f = fact_row(merged, "f-tech-on-p")
    assert (f["src_uuid"], f["tgt_uuid"], f["group_id"]) == ("t-delta", "t-gamma1", "technical")
    assert run_counts(merged)["technical_facts_repointed"] == 1


def test_a_self_loop_the_merge_makes_is_kept_and_counted(merged):
    f = fact_row(merged, "f-loop-p")
    assert f["src_uuid"] == f["tgt_uuid"] == "t-alpha"
    c = run_counts(merged)
    assert c["self_loops_created"] == 1 and c["self_loops_created_live"] == 1
    assert c["missing_endpoints_after"] == 0


def test_a_duplicate_pair_keeps_both_uuids_and_a_feedback_reference_still_resolves(seeded, db_url):
    seeded.execute(
        "INSERT INTO recall_feedback (query, helpful) VALUES ('q', %s::jsonb)",
        (json.dumps(["f:f-dup-p"]),),
    )
    merge(db_url)
    assert (
        fact_row(seeded, "f-dup-p")["t_invalid"] is None
        and fact_row(seeded, "f-tech1")["t_invalid"] is None
    )
    assert run_counts(seeded)["live_duplicate_pairs"] == 1
    assert (
        one(
            seeded,
            "SELECT count(*) FROM kg_relationships r, recall_feedback f "
            "WHERE f.helpful ? ('f:' || r.uuid) AND r.uuid = 'f-dup-p'",
        )
        == 1
    )


def test_an_invalidated_by_chain_survives_and_the_superseded_pair_is_still_returned(merged, db_url):
    assert fact_row(merged, "f-old-p")["invalidated_by"] == "f-new-p"
    assert fact_row(merged, "f-old-p")["t_invalid"] is not None
    pairs = Recall(db_url, "")._fetch_superseded_pairs_pg(["f-new-p"], 10)
    assert [p["id"] for p in pairs] == ["f:f-old-p"]


def test_labels_provenance_and_the_source_projects_trigger_are_left_alone(piles, db_url):
    ep = piles.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) VALUES (%s, 1, 'proj', 'turn') RETURNING id",
        (f"m067-{uuid.uuid4().hex[:8]}",),
    ).fetchone()[0]
    ent(piles, "p-a", "Aaa", group="personal")
    ent(piles, "p-b", "Bbb", group="personal")
    # a sentinel the 056 trigger would overwrite from the episode if the merge fired it
    piles.execute("ALTER TABLE kg_relationships DISABLE TRIGGER kg_rel_source_projects")
    fact(piles, "f-sp", "personal", "p-a", "p-b", "provenance", episodes=f"[{ep}]")
    piles.execute(
        "UPDATE kg_relationships SET source_projects = ARRAY['sentinel'] WHERE uuid = 'f-sp'"
    )
    piles.execute("ALTER TABLE kg_relationships ENABLE TRIGGER kg_rel_source_projects")
    fact(
        piles, "f-ws", "personal", "p-a", "p-b", "kept label", audience="work-safe", label_version=1
    )
    fact(
        piles,
        "f-tech",
        "technical",
        "p-a",
        "p-b",
        "technical label",
        audience="work-safe",
        label_version=1,
    )
    before = {u: fact_row(piles, u) for u in ("f-sp", "f-ws", "f-tech")}
    merge(db_url)
    for uid, old in before.items():
        assert {**fact_row(piles, uid), "group_id": None} == {
            **old,
            "group_id": None,
        }  # only group_id may differ
    assert fact_row(piles, "f-sp")["source_projects"] == ["sentinel"]
    assert (fact_row(piles, "f-ws")["audience"], fact_row(piles, "f-ws")["label_version"]) == (
        "work-safe",
        1,
    )
    assert (
        fact_row(piles, "f-sp")["audience"] == "personal"
        and fact_row(piles, "f-sp")["label_version"] == 0
    )


def test_a_null_audience_fact_is_stamped_personal_v0_before_it_is_regrouped(seeded, db_url):
    seeded.execute(
        "UPDATE kg_relationships SET audience = NULL, label_version = NULL WHERE uuid = 'f-p1'"
    )
    assert run_counts_preview(db_url)["stragglers_to_stamp"] == 1
    merge(db_url)
    f = fact_row(seeded, "f-p1")
    assert (f["audience"], f["label_version"], f["group_id"]) == ("personal", 0, "technical")
    assert rows(
        seeded,
        "SELECT audience FROM kg_pile_merge_067_relationships WHERE uuid = 'f-p1' AND image = 'before'",
    ) == [(None,)]  # the before image keeps what was there


def run_counts_preview(db_url: str) -> dict[str, Any]:
    return pm.preview(pm._connect(db_url, read_only=True))["counts"]


# ---------------------------------------------------------------------------
# Notes, preferences, the trigger, idempotence (cases 18-22)
# ---------------------------------------------------------------------------


def test_notes_and_preferences_flip_and_their_links_stay_intact(merged):
    assert rows(merged, "SELECT DISTINCT group_id FROM notes") == [("technical",)]
    assert rows(merged, "SELECT DISTINCT group_id FROM preferences") == [("technical",)]
    pn2 = one(merged, "SELECT superseded_by FROM notes WHERE hook = 'pn2'")
    assert pn2 == one(merged, "SELECT id FROM notes WHERE hook = 'pn1'")
    assert one(merged, "SELECT superseded_by FROM preferences WHERE pref = 'tp1'") == one(
        merged, "SELECT id FROM preferences WHERE pref = 'pp1'"
    )
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_notes WHERE image = 'before'") == 2
    assert (
        one(merged, "SELECT count(*) FROM kg_pile_merge_067_preferences WHERE image = 'before'")
        == 1
    )
    assert run_counts(merged)["applied"] == {"notes_flipped": 2, "preferences_flipped": 1}


def test_the_insert_trigger_and_its_function_are_gone_and_a_personal_insert_stays_unlabelled(
    merged,
):
    assert (
        one(merged, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'")
        == 0
    )
    assert (
        one(
            merged,
            "SELECT count(*) FROM pg_proc WHERE proname = 'kg_relationships_personal_audience_trg'",
        )
        == 0
    )
    fact(merged, "late", "personal", "x", "y")
    assert one(merged, "SELECT audience FROM kg_relationships WHERE uuid = 'late'") is None


def test_a_second_apply_changes_nothing_and_adds_no_run_or_backup_rows(merged, db_url):
    before = table_md5(merged)
    ledger = {
        t: one(merged, f"SELECT count(*) FROM {t}")
        for t in (
            "kg_pile_merge_067_runs",
            "kg_pile_merge_067_entities",
            "kg_pile_merge_067_relationships",
        )
    }
    merge(db_url)
    assert table_md5(merged) == before
    assert {t: one(merged, f"SELECT count(*) FROM {t}") for t in ledger} == ledger


def test_a_rerun_folds_and_regroups_what_old_code_wrote_after_the_merge(merged, db_url):
    # old code, still running between the apply and the new image: no trigger, so no label
    ent(merged, "p-alpha", "Alpha", group="personal")  # a deleted loser's uuid, minted again
    ent(merged, "p-newbie", "Newbie", group="personal")
    fact(merged, "f-ghost", "personal", "p-alpha", "p-newbie", "from the ghost")
    _note(merged, "late-note", "personal")
    merge(db_url)
    assert rows(merged, "SELECT run_id FROM kg_pile_merge_067_runs ORDER BY run_id") == [(1,), (2,)]
    assert fold_map(merged, 2) == {"p-alpha": ("t-alpha", "ghost")}
    f = fact_row(merged, "f-ghost")
    assert (f["src_uuid"], f["tgt_uuid"], f["group_id"]) == ("t-alpha", "p-newbie", "technical")
    assert (f["audience"], f["label_version"]) == ("personal", 0)
    assert (
        ent_row(merged, "p-newbie")["group_id"] == "technical"
        and ent_row(merged, "p-alpha") is None
    )
    assert one(merged, "SELECT group_id FROM notes WHERE hook = 'late-note'") == "technical"
    assert pm.health(pm._connect(db_url, read_only=True))["ok"] is True


def test_a_folded_uuid_minted_again_under_another_name_still_folds_into_its_survivor(
    merged, db_url
):
    """A poller's name index kept the deleted uuid; a mention of a different name resolved onto
    it and the writer inserted the row again. No technical row has its key, so the name rule
    would regroup it and leave it for good."""
    ent(merged, "p-alpha", "Reborn", group="personal")
    fact(merged, "f-ghost", "personal", "p-alpha", "t-delta", "from the ghost")
    merge(db_url)
    assert fold_map(merged, 2) == {"p-alpha": ("t-alpha", "ghost")}
    f = fact_row(merged, "f-ghost")
    assert (f["src_uuid"], f["group_id"]) == ("t-alpha", "technical")
    assert ent_row(merged, "p-alpha") is None
    assert pm.health(pm._connect(db_url, read_only=True))["ok"] is True


def test_a_folded_uuid_written_again_as_technical_is_folded_by_a_rerun(merged, db_url):
    """A writer that names no group writes it as technical: nothing is personal, and the
    re-run must still look at it."""
    ent(merged, "p-alpha", "Alpha Again")
    fact(merged, "f-ghost", "technical", "p-alpha", "t-delta", "from a group-free writer")
    bad = pm.health(pm._connect(db_url, read_only=True))
    gone = bad["checks"]["every folded entity is gone"]
    assert bad["ok"] is False and gone["still_present"]["uuids"] == ["p-alpha"]
    assert "apply 067 again" in gone["remedy"]
    merge(db_url)
    assert fold_map(merged, 2) == {"p-alpha": ("t-alpha", "ghost")}
    f = fact_row(merged, "f-ghost")
    assert (f["src_uuid"], f["group_id"]) == ("t-alpha", "technical")
    assert ent_row(merged, "p-alpha") is None
    assert pm.health(pm._connect(db_url, read_only=True))["ok"] is True


def test_a_ghost_whose_survivor_is_gone_is_left_alone_and_makes_no_run(merged, db_url):
    merged.execute("DELETE FROM kg_entities WHERE uuid = 't-alpha'")
    ent(merged, "p-alpha", "Alpha Again")
    # nothing for the merge to fold, so the re-run returns before it takes any table lock
    assert {m for _, m in _locks_while_open(db_url, _txn("a"))} <= {"AccessShareLock"}
    merge(db_url)
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs") == 1
    assert ent_row(merged, "p-alpha")["group_id"] == "technical"


def test_the_preview_counts_a_ghost_the_way_the_rerun_records_it(merged, db_url):
    ent(merged, "p-alpha", "Reborn", group="personal")
    ent(merged, "p-newbie", "Newbie", group="personal")
    c = run_counts_preview(db_url)
    assert (c["losers"], c["losers_ghost"], c["losers_by_name"], c["regrouped"]) == (1, 1, 0, 1)
    merge(db_url)
    applied = run_counts(merged, 2)
    applied.pop("applied")
    assert applied == c


def test_undoing_a_ghost_run_puts_the_ghost_back_in_the_group_it_was_written_in(merged, db_url):
    ent(merged, "p-alpha", "Alpha Again")  # technical
    fact(merged, "f-ghost", "technical", "p-alpha", "t-delta", "from a group-free writer")
    merge(db_url)
    out = _undo(db_url, run="2")
    assert out["drift"]["folded entity uuid already exists"]["count"] == 0
    row = ent_row(merged, "p-alpha")
    assert (row["group_id"], row["name"]) == ("technical", "Alpha Again")
    f = fact_row(merged, "f-ghost")
    assert (f["src_uuid"], f["group_id"]) == ("p-alpha", "technical")
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs WHERE undone_at IS NULL") == 1


def test_a_rerun_with_only_a_personal_note_flips_it_and_leaves_the_graph_tables_alone(
    merged, db_url
):
    """The old server's remember() writes personal notes after the apply. Flipping them takes
    row locks; the SHARE ROW EXCLUSIVE lock and the full-graph counts are for graph rows."""
    _note(merged, "late-note", "personal")
    held = _locks_while_open(db_url, _txn("a"))
    assert {m for t, m in held if t in ("kg_entities", "kg_relationships")} <= {"AccessShareLock"}
    graph = (
        one(merged, "SELECT md5(string_agg(x::text, '|' ORDER BY id)) FROM kg_entities x"),
        one(merged, "SELECT md5(string_agg(x::text, '|' ORDER BY id)) FROM kg_relationships x"),
    )
    merge(db_url)
    run = one(
        merged, "SELECT max(run_id) FROM kg_pile_merge_067_runs"
    )  # the rolled-back one used 2
    assert one(merged, "SELECT group_id FROM notes WHERE hook = 'late-note'") == "technical"
    assert run_counts(merged, run) == {
        "notes_personal": 1,
        "preferences_personal": 0,
        "applied": {"notes_flipped": 1, "preferences_flipped": 0},
    }
    assert graph == (
        one(merged, "SELECT md5(string_agg(x::text, '|' ORDER BY id)) FROM kg_entities x"),
        one(merged, "SELECT md5(string_agg(x::text, '|' ORDER BY id)) FROM kg_relationships x"),
    )
    assert pm.health(pm._connect(db_url, read_only=True))["ok"] is True
    out = _undo(db_url, run=str(run))
    assert out["notes restored"] == 1
    assert one(merged, "SELECT group_id FROM notes WHERE hook = 'late-note'") == "personal"


def test_health_skips_the_graph_baselines_when_the_first_run_only_flipped_notes(piles, db_url):
    """Such a run records no graph counts; a graph that already had an edge without an endpoint
    must not read as damage the merge did."""
    fact(piles, "f-dangling", "technical", "nowhere-a", "nowhere-b")
    _note(piles, "late-note", "personal")
    merge(db_url)
    assert "facts_total" not in run_counts(piles, 1)
    report = pm.health(pm._connect(db_url, read_only=True))
    assert report["ok"] is True
    assert "no new edges with a missing endpoint" not in report["checks"]


def test_an_empty_database_gets_no_ledger_and_is_stamped(piles, db_url):
    merge(db_url)
    assert (
        rows(piles, "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'")
        == []
    )
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert pm.health(pm._connect(db_url, read_only=True))["ok"] is True


# ---------------------------------------------------------------------------
# pile_merge.py: preview, health, undo (cases 23-28)
# ---------------------------------------------------------------------------


def test_preview_counts_equal_the_applied_runs_counts_and_it_writes_nothing(seeded, db_url):
    before = table_md5(seeded)
    out = pm.preview(pm._connect(db_url, read_only=True))
    assert table_md5(seeded) == before
    assert (
        rows(seeded, "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'")
        == []
    )
    counts = out["counts"]
    assert counts["losers"] == 7 and counts["survivors"] == 6 and counts["regrouped"] == 4
    assert out["gates"]["self_loops_created is 0"] is False  # the world has one
    merge(db_url)
    applied = run_counts(seeded)
    applied.pop("applied")
    assert applied == counts


def test_the_preview_connection_refuses_writes(db_url):
    conn = pm._connect(db_url, read_only=True)
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("CREATE TEMP TABLE nope (a int)")
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("UPDATE kg_entities SET degree = degree")


def test_the_preview_reports_twin_supertypes_row_10b(seeded, db_url):
    c = run_counts_preview(db_url)
    assert c["supertype_pairs"] == {"Concept | Tool": 1, "Person | Tool": 1}
    assert c["supertype_differ_person"] == 1
    assert c["twin_shapes"] == {"1:1": 3, "1:n": 1, "n:1": 1, "n:0": 1}


def test_health_passes_after_the_merge_and_fails_on_a_straggler(merged, db_url):
    ok = pm.health(pm._connect(db_url, read_only=True))
    assert ok["ok"] is True and ok["runs_not_undone"] == [1]
    assert set(ok["retired_restores (do not run unless the merge is undone first)"]) == set(
        pm.RETIRED_RESTORES
    )
    ent(merged, "p-late", "Late", group="personal")
    bad = pm.health(pm._connect(db_url, read_only=True))
    assert bad["ok"] is False and bad["checks"]["no personal rows"]["ok"] is False
    assert pm.main(["--health", "--db-url", db_url]) == 1


def test_health_fails_on_a_missing_index_and_on_a_folded_entity_that_came_back(merged, db_url):
    merged.execute("DROP INDEX kg_rel_tgt_any")
    assert (
        pm.health(pm._connect(db_url, read_only=True))["checks"][
            "owner-only indexes exist and are valid"
        ]["ok"]
        is False
    )
    merged.execute("CREATE INDEX kg_rel_tgt_any ON kg_relationships (owner_id, tgt_uuid)")
    ent(merged, "p-alpha", "Alpha")  # a folded loser's uuid, back in technical
    assert (
        pm.health(pm._connect(db_url, read_only=True))["checks"]["every folded entity is gone"][
            "ok"
        ]
        is False
    )


def test_health_stays_green_when_a_fact_is_deleted_after_the_merge_and_shows_the_counts(
    merged, db_url
):
    """An episode delete removes facts for good: the count falls below the one recorded before
    the merge, and that is the graph moving on, not damage."""
    before = run_counts(merged)["facts_total"]
    merged.execute("DELETE FROM kg_relationships WHERE uuid = 'f-old-p'")
    report = pm.health(pm._connect(db_url, read_only=True))
    assert report["ok"] is True
    assert report["reported_not_checked"]["fact_count"] == {
        "now": before - 1,
        "before_merge": before,
    }


def test_health_stays_green_when_a_survivor_is_swept_after_the_merge_and_lists_it(merged, db_url):
    """A survivor whose edges all went with deleted episodes is removed by the orphan sweep."""
    assert fold_map(merged)["p-omega"][0] == "t-omega"
    merged.execute(
        "DELETE FROM kg_relationships WHERE src_uuid = 't-omega' OR tgt_uuid = 't-omega'"
    )
    merged.execute("DELETE FROM kg_entities WHERE uuid = 't-omega'")
    report = pm.health(pm._connect(db_url, read_only=True))
    assert report["ok"] is True
    assert report["reported_not_checked"]["survivors_no_longer_in_technical"] == {
        "count": 1,
        "uuids": ["t-omega"],
    }


def test_health_names_undo_not_a_rerun_for_a_returned_uuid_whose_survivor_is_gone(merged, db_url):
    merged.execute(
        "DELETE FROM kg_relationships WHERE src_uuid = 't-delta' OR tgt_uuid = 't-delta'"
    )
    merged.execute("DELETE FROM kg_entities WHERE uuid = 't-delta'")
    ent(merged, "p-delta", "Delta Again")  # a stale name index wrote the folded uuid again
    gone = pm.health(pm._connect(db_url, read_only=True))["checks"]["every folded entity is gone"]
    assert gone["ok"] is False and gone["still_present"]["uuids"] == ["p-delta"]
    assert gone["apply_again_leaves_alone"]["uuids"] == ["p-delta"]
    assert "apply_again_folds" not in gone
    assert "apply 067 again" not in gone["remedy"] and "--undo" in gone["remedy"]
    merge(db_url)  # and a re-run indeed leaves it alone
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs") == 1
    assert ent_row(merged, "p-delta")["group_id"] == "technical"
    # one whose survivor is alive is still the re-run's work, and keeps its remedy
    ent(merged, "p-alpha", "Alpha Again")
    gone = pm.health(pm._connect(db_url, read_only=True))["checks"]["every folded entity is gone"]
    assert gone["apply_again_folds"]["uuids"] == ["p-alpha"]
    assert gone["apply_again_leaves_alone"]["uuids"] == ["p-delta"]
    assert "apply 067 again, then restart the pollers" in gone["remedy"]


def _undo(db_url: str, *, run: str = "all", write: bool = True) -> dict[str, Any]:
    return pm.undo(pm._connect(db_url, read_only=False), run=run, write=write)


def _in_thread(fn) -> tuple[threading.Thread, dict[str, Any]]:
    """Run ``fn`` on a thread; its result or exception lands in the returned dict."""
    box: dict[str, Any] = {}

    def go() -> None:
        try:
            box["result"] = fn()
        except BaseException as e:  # handed to the test thread
            box["error"] = e

    t = threading.Thread(target=go)
    t.start()
    return t, box


def _wait_until_blocked_by(probe, pid: int, *, seconds: float = 20) -> None:
    """Wait until some backend is waiting for a lock that backend ``pid`` holds."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        n = one(
            probe,
            "SELECT count(*) FROM pg_stat_activity WHERE %s = ANY(pg_blocking_pids(pid))",
            pid,
        )
        if n:
            return
        time.sleep(0.02)
    pytest.fail(f"nothing waits for a lock held by backend {pid}")


def test_undo_restores_the_exact_rows_the_trigger_and_the_066_stamp(seeded, db_url):
    seeded.execute(
        "UPDATE kg_relationships SET audience = NULL, label_version = NULL WHERE uuid = 'f-p1'"
    )  # a straggler
    before = table_md5(seeded)
    trigger = one(
        seeded,
        "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'",
    )
    merge(db_url)
    assert table_md5(seeded) != before
    dry = _undo(db_url, write=False)  # a dry run rolls back
    assert dry["stamped 066"] is True and dry["written"] is False
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert (
        one(seeded, "SELECT count(*) FROM kg_pile_merge_067_runs WHERE undone_at IS NOT NULL") == 0
    )
    out = _undo(db_url)
    assert all(not d["count"] for d in out["drift"].values() if "count" in d)
    assert table_md5(seeded) == before
    assert (
        one(
            seeded,
            "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'",
        )
        == trigger
    )
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"
    assert (
        one(seeded, "SELECT count(*) FROM kg_pile_merge_067_runs WHERE undone_at IS NOT NULL") == 1
    )
    # the indexes and the column defaults are additive and stay
    assert (
        pm.health(pm._connect(db_url, read_only=True))["checks"][
            "owner-only indexes exist and are valid"
        ]["ok"]
        is True
    )
    with pytest.raises(SystemExit):
        _undo(db_url)  # nothing left to undo
    merge(db_url)  # and the merge can be applied again, under a new run
    assert rows(seeded, "SELECT run_id FROM kg_pile_merge_067_runs ORDER BY run_id")[-1] == (2,)


def test_undo_reports_drift_and_keeps_what_was_written_since(merged, db_url):
    merged.execute("UPDATE kg_entities SET summary = 'edited since' WHERE uuid = 't-gamma1'")
    fact(merged, "f-after", "technical", "t-alpha", "t-delta", "written after the merge")
    ent(
        merged, "p-delta", "Reborn", group="technical"
    )  # a folded loser's uuid, taken by something else
    merged.execute("UPDATE kg_relationships SET src_uuid = 't-eps-moved' WHERE uuid = 'f-p4'")
    out = _undo(db_url)
    drift = out["drift"]
    assert drift["survivor edited since the merge"]["uuids"] == ["t-gamma1"]
    assert drift["folded entity uuid already exists"]["uuids"] == ["p-delta"]
    assert drift["fact changed since the merge, or gone"]["uuids"] == ["f-p4"]
    assert ent_row(merged, "t-gamma1")["summary"] == "edited since"  # not overwritten
    assert ent_row(merged, "p-delta")["name"] == "Reborn"
    assert fact_row(merged, "f-after")["group_id"] == "technical"
    assert fact_row(merged, "f-p4")["src_uuid"] == "t-eps-moved"
    # everything else went back
    assert (
        ent_row(merged, "p-alpha")["group_id"] == "personal"
        and ent_row(merged, "p-gamma") is not None
    )
    assert (
        fact_row(merged, "f-p1")["src_uuid"] == "p-alpha"
        and fact_row(merged, "f-p1")["group_id"] == "personal"
    )
    assert one(merged, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"


def test_undo_of_two_runs_goes_latest_first_and_stamps_066_only_when_all_are_undone(merged, db_url):
    ent(merged, "p-alpha", "Alpha", group="personal")  # old code's ghost of a folded twin
    ent(merged, "p-newbie", "Newbie", group="personal")
    fact(merged, "f-ghost", "personal", "p-alpha", "p-newbie", "from the ghost")
    merge(db_url)
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs") == 2
    with pytest.raises(SystemExit, match="later runs"):
        _undo(db_url, run="1")
    out = _undo(db_url, run="2")
    assert out["stamped 066"] is False
    assert one(merged, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert ent_row(merged, "p-newbie")["group_id"] == "personal"
    ghost = fact_row(merged, "f-ghost")
    assert (ghost["group_id"], ghost["src_uuid"], ghost["audience"]) == (
        "personal",
        "p-alpha",
        None,
    )  # unstamped again
    out = _undo(db_url, run="1")
    assert out["stamped 066"] is True
    # one by one, the ghost row run 2 gave back is in the way of run 1's original: reported, not overwritten
    assert out["drift"]["folded entity uuid already exists"]["uuids"] == ["p-alpha"]
    assert one(merged, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"
    assert one(merged, "SELECT count(*) FROM kg_relationships WHERE group_id = 'personal'") == 10
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs WHERE undone_at IS NULL") == 0


def test_undo_all_across_two_runs_restores_the_original_twin_rows(merged, db_url):
    original_id = one(
        merged,
        "SELECT id FROM kg_pile_merge_067_entities WHERE uuid = 'p-alpha' AND image = 'before'",
    )
    ent(merged, "p-alpha", "Alpha", group="personal")
    merge(db_url)
    out = _undo(db_url)
    assert out["runs"] == [1, 2] and out["stamped 066"] is True
    row = ent_row(merged, "p-alpha")
    assert (
        row["group_id"] == "personal" and row["id"] == original_id
    )  # run 1's image, not the ghost's


def test_undo_all_judges_each_run_by_its_own_map(merged, db_url):
    """Run 1 folded p-alpha into t-alpha. That survivor is gone, and the uuid comes back: run 2
    folds it into the other technical Alpha. An undo that reads one map for both runs expects
    the ghost's fact on t-alpha, does not find it there and leaves it on the wrong entity."""
    original_id = one(
        merged,
        "SELECT id FROM kg_pile_merge_067_entities WHERE uuid = 'p-alpha' AND image = 'before'",
    )
    ent(merged, "t-alpha-b", "Alpha")
    merged.execute("DELETE FROM kg_entities WHERE uuid = 't-alpha'")
    ent(merged, "p-alpha", "Alpha", group="personal")
    fact(merged, "f-ghost", "personal", "p-alpha", "t-delta", "from the ghost")
    merge(db_url)
    assert fold_map(merged, 2) == {"p-alpha": ("t-alpha-b", "name")}
    assert fact_row(merged, "f-ghost")["src_uuid"] == "t-alpha-b"
    out = _undo(db_url)
    assert out["runs"] == [1, 2]
    ghost = fact_row(merged, "f-ghost")
    assert (ghost["group_id"], ghost["src_uuid"]) == ("personal", "p-alpha")
    assert "f-ghost" not in out["drift"]["fact changed since the merge, or gone"]["uuids"]
    assert ent_row(merged, "p-alpha")["id"] == original_id  # run 1's row, not the ghost's
    assert fact_row(merged, "f-p1")["src_uuid"] == "p-alpha"  # run 1's own facts, as before


def test_undo_all_keeps_what_was_written_to_a_survivor_between_two_runs(merged, db_url):
    """A mention of t-alpha at the new image rewrites its summary, then the re-run folds a
    straggler into it again. Undoing both runs must not take that edit back to run 1's image."""
    merged.execute(
        "UPDATE kg_entities SET summary = 'written by the new image' WHERE uuid = 't-alpha'"
    )
    ent(merged, "p-alpha", "Alpha", group="personal")
    merge(db_url)
    out = _undo(db_url)
    assert ent_row(merged, "t-alpha")["summary"] == "written by the new image"
    assert out["drift"]["survivor edited since the merge"]["uuids"] == ["t-alpha"]
    assert ent_row(merged, "p-alpha")["group_id"] == "personal"  # everything else went back


def test_undo_refuses_when_a_later_migration_has_raised_the_stamp_and_changes_nothing(
    merged, db_url
):
    """A full undo stamps 066; with 068 applied that would put the version below 068's objects."""
    merged.execute("UPDATE synapse_meta SET value = '068' WHERE key = 'schema_version'")
    try:
        before = table_md5(merged)
        for write in (False, True):
            with pytest.raises(SystemExit, match="above 067"):
                _undo(db_url, write=write)
        assert table_md5(merged) == before
        assert one(merged, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "068"
        assert (
            one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs WHERE undone_at IS NOT NULL")
            == 0
        )
        assert (
            one(merged, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'")
            == 0
        )  # the 066 trigger was not put back either
    finally:
        # 067's own stamp never moves down, so the fixture's re-apply would leave 068 behind
        merged.execute("UPDATE synapse_meta SET value = '067' WHERE key = 'schema_version'")


def test_undo_reads_the_open_runs_only_after_it_has_the_lock(merged, db_url):
    """An apply that holds the lock when undo starts has committed its run by the time undo
    reads the runs, so undo takes that run too instead of stamping 066 over it."""
    apply_ = psycopg.connect(db_url)
    probe = psycopg.connect(db_url, autocommit=True)
    try:
        apply_.execute("LOCK TABLE kg_entities, kg_relationships IN SHARE ROW EXCLUSIVE MODE")
        apply_.execute(
            "INSERT INTO kg_pile_merge_067_runs (schema_version_before, counts) VALUES ('066', '{}')"
        )
        t, box = _in_thread(lambda: _undo(db_url, write=False))
        _wait_until_blocked_by(probe, apply_.info.backend_pid)
        apply_.commit()
        t.join(30)
    finally:
        apply_.close()
        probe.close()
    assert "error" not in box, box
    assert box["result"]["runs"] == [1, 2]


def test_the_undo_help_says_a_dry_run_holds_writers_off_too(capsys):
    with pytest.raises(SystemExit):
        pm.main(["--help"])
    assert "stop the pollers first" in " ".join(capsys.readouterr().out.split())
    assert "with the pollers stopped too" in " ".join(pm.__doc__.split())


def test_undo_reports_a_regrouped_entity_that_got_edges_since_and_still_moves_it_back(
    merged, db_url
):
    fact(merged, "f-since", "technical", "p-eps", "t-delta", "written after the merge")
    out = _undo(db_url)
    key = "regrouped entity moved back has edges the merge never touched"
    assert out["drift"][key] == {"count": 1, "uuids": ["p-eps"]}
    assert ent_row(merged, "p-eps")["group_id"] == "personal"  # moved back all the same
    assert fact_row(merged, "f-since")["group_id"] == "technical"  # and the edge is left alone


def test_undo_re_creates_an_entity_into_a_column_a_later_migration_made_not_null(seeded, db_url):
    """Run 2 syncs a column added after run 1 into the ledger. Run 1's rows must hold the live
    default there, not NULL, or putting a folded twin back violates the live NOT NULL."""
    merge(db_url)
    seeded.execute("ALTER TABLE kg_entities ADD COLUMN pm_req text NOT NULL DEFAULT 'dflt'")
    try:
        ent(seeded, "p-late", "Late", group="personal")
        merge(db_url)
        out = _undo(db_url)
        assert out["folded entities re-created"] == 7  # run 1's seven folded twins
        assert ent_row(seeded, "p-alpha")["pm_req"] == "dflt"
    finally:
        seeded.execute("ALTER TABLE kg_entities DROP COLUMN IF EXISTS pm_req")


def test_no_report_prints_fact_or_entity_text(seeded, db_url, capsys):
    secret = [
        "Alpha",
        "Gamma",
        "Overlord",
        "Narrator",
        "Epsilon",
        "Zeta",
        "alpha is owned",
        "Delta uses Gamma",
    ]
    assert pm.main(["--preview", "--db-url", db_url]) == 0
    merge(db_url)
    assert pm.main(["--health", "--db-url", db_url]) == 0
    assert pm.main(["--undo", "--db-url", db_url]) == 0  # a dry run
    assert pm.main(["--undo", "--apply", "--db-url", db_url]) == 0
    out = capsys.readouterr().out
    assert len(out) > 500
    assert not [s for s in secret if s.lower() in out.lower()]


# ---------------------------------------------------------------------------
# Write-time parity and the required migration (cases 27, 29, 32)
# ---------------------------------------------------------------------------


def test_write_time_resolution_returns_the_row_the_merge_kept(merged, db_url, monkeypatch):
    from ingestion import dedup

    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    reader = KGPostgresReader()
    assert reader.entity_uuid_by_normalized_name("gamma") == "t-gamma1"
    deduper = dedup.NodeDeduper(reader)
    idx = deduper._build_index()
    assert idx.uuid_by_normalized_name["gamma"] == "t-gamma1"
    assert idx.uuid_by_normalized_name["alpha"] == "t-alpha"


def test_the_name_lookup_ranks_an_old_row_with_no_stored_name_among_the_stored_ones(
    piles, db_url, monkeypatch
):
    """The merge keeps the oldest technical row of a key. A lookup that tried the stored name
    first and the lower(trim(name)) fallback only when that found nothing would send a new
    mention to the newer row, and the split the merge removed would grow back."""
    from ingestion import dedup

    ent(piles, "t-old", "Omega", norm="")  # written before normalized_name existed
    ent(piles, "t-new", "Omega")
    ent(piles, "p-omega", "Omega", group="personal")
    merge(db_url)
    assert fold_map(piles) == {"p-omega": ("t-old", "name")}
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    reader = KGPostgresReader()
    assert reader.entity_uuid_by_normalized_name("omega") == "t-old"
    assert dedup.NodeDeduper(reader)._build_index().uuid_by_normalized_name["omega"] == "t-old"
    assert reader.entity_uuid_by_normalized_name("") is None


def test_the_name_lookup_does_not_take_a_row_stored_under_another_key_for_a_name_match(
    piles, db_url, monkeypatch
):
    ent(piles, "t-legacy", "Omega", norm="omega-legacy")  # stored under another key
    ent(piles, "t-blank", "omega", norm="")
    ent(piles, "p-omega", "Omega", group="personal")
    merge(db_url)
    assert fold_map(piles) == {"p-omega": ("t-blank", "name")}
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    assert KGPostgresReader().entity_uuid_by_normalized_name("omega") == "t-blank"


def test_the_name_lookup_is_answered_from_the_normalized_name_index(piles):
    """Run for every mention: a plan that walked the owner's entities would cost milliseconds
    each. The planner prefers a sequential scan on a table this small, so it is told not to."""
    from ingestion import kg_pg_read

    ent(piles, "t-a", "Alpha")
    piles.execute("SET enable_seqscan = off")
    try:
        plan = "\n".join(
            r[0]
            for r in piles.execute(
                "EXPLAIN (COSTS OFF) " + kg_pg_read._ENTITY_BY_NAME_SQL,
                (OWNER, "alpha", "alpha"),
            )
        )
    finally:
        piles.execute("RESET enable_seqscan")
    assert "Seq Scan" not in plan and "Bitmap Index Scan" in plan, plan


def test_the_survivor_order_the_lookups_use_is_the_one_the_merge_uses():
    text = _SQL_067.read_text()
    assert f"ORDER BY owner_id, k, {SURVIVOR_ORDER}" in text
    assert f"ORDER BY owner_id, k, {SURVIVOR_ORDER}" in pm.shared_sql()["map"]
    from ingestion import kg_pg_read

    src = Path(kg_pg_read.__file__).read_text()
    assert src.count('"ORDER BY " + SURVIVOR_ORDER') == 2  # the name lookup and load_entities


def test_the_shared_text_is_read_from_067_and_the_preview_runs_it():
    parts = pm.shared_sql()
    assert parts["map"].startswith("WITH pk AS") and "_pm_excl" in parts["map"]
    assert "_pm_map" in parts["fx"] and parts["counts"].startswith("jsonb_build_object(")
    sql = pm.preview_sql(with_exclude_table=False)
    assert parts["map"] in sql and parts["fx"] in sql and parts["counts"] in sql


def test_067_is_required_not_optional_and_registered_after_066():
    assert not schema_check.is_optional_migration(_SQL_067)
    assert schema_check.required_schema_version() == "067"
    assert schema_check.expected_schema_version() == "067"
    files = re.findall(
        r"^\s+(\d{3}_\w+\.sql)$", (_ROOT / "scripts" / "apply_schema.sh").read_text(), re.M
    )
    assert files.index("067_kg_one_memory.sql") == files.index("066_kg_fact_audience.sql") + 1
    assert files[-1] == "067_kg_one_memory.sql"


def test_a_build_that_needs_067_refuses_a_database_stamped_066_and_runs_on_067(seeded, db_url):
    with pytest.raises(SystemExit):
        schema_check.check_schema_version(db_url)
    merge(db_url)
    schema_check.check_schema_version(db_url)  # returns


# ---------------------------------------------------------------------------
# The two transactions (cases 30, 31, 34)
# ---------------------------------------------------------------------------


def _txn(name: str) -> str:
    m = re.search(
        rf"^-- txn-{name}:begin\n(.*?)^-- txn-{name}:end$", _SQL_067.read_text(), re.S | re.M
    )
    assert m, name
    return m.group(1)


def _locks_while_open(
    db_url: str, body: str, tables: tuple[str, ...] = KG_TABLES
) -> list[tuple[str, str]]:
    """Run ``body`` in an open transaction on one connection, read that backend's table locks
    from a second, then roll back."""
    with (
        psycopg.connect(db_url, autocommit=True) as a,
        psycopg.connect(db_url, autocommit=True) as b,
    ):
        a.execute("BEGIN")
        try:
            a.execute(body)
            return b.execute(
                "SELECT c.relname, l.mode FROM pg_locks l JOIN pg_class c ON c.oid = l.relation "
                "WHERE l.pid = %s AND l.granted AND c.relname = ANY(%s)",
                (a.info.backend_pid, list(tables)),
            ).fetchall()
        finally:
            a.execute("ROLLBACK")


def test_the_merge_transaction_holds_no_access_exclusive_lock(seeded, db_url):
    held = _locks_while_open(db_url, _txn("a"))
    modes = {(t, m) for t, m in held}
    assert ("kg_entities", "ShareRowExclusiveLock") in modes
    assert ("kg_relationships", "ShareRowExclusiveLock") in modes
    assert not [x for x in modes if x[1] in ("AccessExclusiveLock", "ExclusiveLock")]
    # nothing was committed: the merge is still to do
    assert one(seeded, "SELECT count(*) FROM kg_entities WHERE group_id = 'personal'") == 11


def test_a_rerun_on_a_clean_database_takes_no_lock_on_the_kg_tables_and_writes_no_run(
    merged, db_url
):
    before = one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs")
    held = _locks_while_open(db_url, _txn("a"))
    assert {m for _, m in held} <= {"AccessShareLock"}  # reading "is there anything personal?"
    assert one(merged, "SELECT count(*) FROM kg_pile_merge_067_runs") == before


def test_a_rerun_on_a_database_with_no_ledger_takes_no_lock_either(piles, db_url):
    merge(db_url)
    assert {m for _, m in _locks_while_open(db_url, _txn("a"))} <= {"AccessShareLock"}


def _commit_a(db_url: str) -> None:
    """The merge transaction alone, committed: the state the tail runs in."""
    with psycopg.connect(db_url, autocommit=True) as c:
        c.execute("BEGIN")
        c.execute(_txn("a"))
        c.execute("COMMIT")


def test_the_tail_transaction_asks_for_access_exclusive_only_where_the_catalog_needs_it(
    piles, db_url
):
    _commit_a(db_url)
    needs = {t for t, m in _locks_while_open(db_url, _txn("b")) if m == "AccessExclusiveLock"}
    assert needs == {
        "kg_entities",
        "kg_relationships",
        "notes",
        "preferences",
    }  # the 066 state: all four
    merge(db_url)
    assert [x for x in _locks_while_open(db_url, _txn("b")) if x[1] == "AccessExclusiveLock"] == []


def test_the_tail_drops_the_trigger_sets_the_defaults_and_stamps_last(piles, db_url):
    _commit_a(db_url)
    with psycopg.connect(db_url, autocommit=True) as a:
        a.execute("BEGIN")
        a.execute(_txn("b"))
        assert (
            one(a, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'") == 0
        )
        defaults = rows(
            a,
            "SELECT c.relname, pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d "
            "JOIN pg_class c ON c.oid = d.adrelid JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
            "WHERE a.attname = 'group_id' AND c.relname = ANY(%s) ORDER BY 1",
            list(KG_TABLES),
        )
        a.execute("ROLLBACK")
    assert defaults == [(t, "'technical'::text") for t in sorted(KG_TABLES)]
    text = _txn("b").rstrip()
    assert text.endswith(
        "WHERE synapse_meta.value < EXCLUDED.value;"
    )  # the stamp is the last statement
    assert "kg_relationships_personal_audience_trg" in text.split("INSERT INTO synapse_meta")[0]


def _wait_for_wake(probe, app: str, *, events: tuple[str, ...], seconds: float = 40) -> str:
    """The wait event of backend ``app`` once it is one of ``events`` (psql runs the file)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ev = probe.execute(
            "SELECT wait_event FROM pg_stat_activity WHERE application_name = %s", (app,)
        ).fetchone()
        if ev and ev[0] in events:
            return ev[0]
        time.sleep(0.05)
    pytest.fail(f"{app} never reached one of {events}")


def test_the_tail_retries_a_refused_lock_without_holding_one_or_blocking_a_reader(piles, db_url):
    """A reader holds ACCESS SHARE on kg_relationships. The tail must not queue for it with
    ACCESS EXCLUSIVE already held on kg_entities (every other reader of kg_entities would wait
    behind that, and so would a reader that then asks for it: a deadlock the tail would lose);
    it sleeps between attempts, and the next attempt after the reader is gone completes.
    The session's statement_timeout (300 ms) is shorter than that sleep: the tail runs its
    retry loop as one statement and must be immune to it."""
    blocker = psycopg.connect(db_url)
    blocker.execute("LOCK TABLE kg_relationships IN ACCESS SHARE MODE")
    probe = psycopg.connect(db_url, autocommit=True)
    with _SQL_067.open() as f:
        proc = subprocess.Popen(
            ["psql", db_url, "-q", "-v", "ON_ERROR_STOP=1"],
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "PGAPPNAME": "pm067-retry",
                "PGOPTIONS": "-c statement_timeout=300ms",
            },
            text=True,
        )
        try:
            # asleep between attempts, never waiting on a lock
            assert _wait_for_wake(probe, "pm067-retry", events=("PgSleep", "relation")) == "PgSleep"
            # the reader that holds kg_relationships asks for kg_entities now: not blocked
            blocker.execute("SET statement_timeout = 2000")
            blocker.execute("SELECT count(*) FROM kg_entities")
            blocker.execute("SELECT count(*) FROM kg_relationships")
            blocker.rollback()
            _, err = proc.communicate(timeout=30)
        finally:
            blocker.close()
            probe.close()
            if proc.poll() is None:
                proc.kill()
    assert proc.returncode == 0, err
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert (
        one(piles, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'") == 0
    )


def _start_file(db_url: str, app: str, *settings: str) -> subprocess.Popen:
    """The whole file through psql, as an apply runs it, with ``settings`` (name=value) given to
    the session; ``synapse.pm067_lock_attempts`` and ``..._lock_pause`` shorten the tail's loop."""
    with _SQL_067.open() as f:
        return subprocess.Popen(
            ["psql", db_url, "-q", "-v", "ON_ERROR_STOP=1"],
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "PGAPPNAME": app,
                "PGOPTIONS": " ".join(f"-c {x}" for x in settings),
            },
            text=True,
        )


def _wait_for_table_lock(probe, app: str, table: str, mode: str, *, seconds: float = 20) -> int:
    """Wait until backend ``app`` holds ``mode`` on ``table``; return its pid."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pid = one(
            probe,
            "SELECT a.pid FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
            "WHERE a.application_name = %s AND l.relation = %s::regclass AND l.mode = %s AND l.granted",
            app,
            table,
            mode,
        )
        if pid:
            return pid
        time.sleep(0.02)
    pytest.fail(f"{app} never held {mode} on {table}")


def test_the_tail_gives_up_loudly_on_a_writer_that_holds_a_lock_and_waits_for_the_tails(
    piles, db_url
):
    """A transaction that has locked a row with SELECT ... FOR UPDATE (ROW SHARE on the table,
    as the dashboard's fact and episode deletes do) and then writes waits for the SHARE ROW
    EXCLUSIVE the tail holds; its ROW SHARE refuses every ACCESS EXCLUSIVE attempt. The tail
    stops after its attempts and rolls back, the stamp stays 066 and the trigger stays, the
    writer goes on, and a plain re-run finishes."""
    fact(piles, "f-w", "technical", "a", "b")
    writer = psycopg.connect(db_url)
    probe = psycopg.connect(db_url, autocommit=True)
    proc = None
    try:
        writer.execute("SELECT 1 FROM kg_relationships WHERE uuid = 'f-w' FOR UPDATE")
        proc = _start_file(
            db_url,
            "pm067-exhaust",
            "synapse.pm067_lock_attempts=60",
            "synapse.pm067_lock_pause=0.05",
        )
        tail = _wait_for_table_lock(
            probe, "pm067-exhaust", "kg_relationships", "ShareRowExclusiveLock"
        )
        t, box = _in_thread(
            lambda: writer.execute(
                "UPDATE kg_relationships SET retrieval_count = retrieval_count + 1 WHERE uuid = 'f-w'"
            )
        )
        _wait_until_blocked_by(probe, tail)  # the writer waits for the tail's lock
        _, err = proc.communicate(timeout=60)
        t.join(30)
        assert "error" not in box, box
        writer.commit()
    finally:
        writer.close()
        probe.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
    assert proc.returncode != 0 and "could not obtain lock" in err
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"
    assert (
        one(piles, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'") == 1
    )
    assert one(piles, "SELECT retrieval_count FROM kg_relationships WHERE uuid = 'f-w'") == 1
    merge(db_url)
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert (
        one(piles, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'") == 0
    )


def test_a_writer_that_only_queues_behind_the_tail_does_not_refuse_its_attempts(piles, db_url):
    """The other writer: one that holds nothing yet, such as the recall fact-count bump, and
    queues for ROW EXCLUSIVE behind the tail's SHARE ROW EXCLUSIVE. Being queued does not make
    an ACCESS EXCLUSIVE NOWAIT attempt fail: once the reader is gone the next attempt
    completes, and the writer then goes through."""
    fact(piles, "f-w", "technical", "a", "b")
    blocker = psycopg.connect(db_url)
    blocker.execute("LOCK TABLE kg_relationships IN ACCESS SHARE MODE")
    writer = psycopg.connect(db_url)
    probe = psycopg.connect(db_url, autocommit=True)
    proc = None
    try:
        proc = _start_file(
            db_url,
            "pm067-queued",
            "synapse.pm067_lock_attempts=60",
            "synapse.pm067_lock_pause=0.05",
        )
        tail = _wait_for_table_lock(
            probe, "pm067-queued", "kg_relationships", "ShareRowExclusiveLock"
        )
        t, box = _in_thread(
            lambda: writer.execute(
                "UPDATE kg_relationships SET retrieval_count = retrieval_count + 1 WHERE uuid = 'f-w'"
            )
        )
        _wait_until_blocked_by(probe, tail)  # queued, while the tail is refused by the reader
        blocker.rollback()
        _, err = proc.communicate(timeout=60)
        t.join(30)
        assert "error" not in box, box
        writer.commit()
    finally:
        blocker.close()
        writer.close()
        probe.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert one(piles, "SELECT retrieval_count FROM kg_relationships WHERE uuid = 'f-w'") == 1


def test_the_tail_refuses_to_stamp_a_database_whose_merge_transaction_did_not_finish(piles, db_url):
    """No personal row, so the data guard passes, but the indexes the merge transaction builds
    are missing: a client that ignored ON_ERROR_STOP must not get a 067 stamp from this."""
    with psycopg.connect(db_url, autocommit=True) as a:
        a.execute("BEGIN")
        with pytest.raises(psycopg.errors.RaiseException, match="missing or INVALID"):
            a.execute(_txn("b"))
        a.execute("ROLLBACK")
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"
    assert (
        one(piles, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'") == 1
    )  # nothing was changed
    _commit_a(db_url)
    with psycopg.connect(db_url, autocommit=True) as a:
        a.execute("BEGIN")
        a.execute(_txn("b"))
        a.execute("COMMIT")
    assert one(piles, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"


def test_the_tail_refuses_to_stamp_a_graph_that_still_has_a_personal_row(seeded, db_url):
    """A client that ignored ON_ERROR_STOP would run the tail after a failed merge."""
    with psycopg.connect(db_url, autocommit=True) as a:
        a.execute("BEGIN")
        with pytest.raises(psycopg.errors.RaiseException, match="did not complete"):
            a.execute(_txn("b"))
        a.execute("ROLLBACK")
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"


def test_the_ledger_follows_a_live_table_that_gains_or_loses_a_column(seeded, db_url):
    """A later migration may add or drop a column of a merged table before a straggler re-run."""
    seeded.execute("ALTER TABLE kg_entities ADD COLUMN pm_old text NOT NULL DEFAULT 'x'")
    try:
        merge(db_url)  # the ledger copies pm_old, NOT NULL
        seeded.execute("ALTER TABLE kg_entities DROP COLUMN pm_old")
        seeded.execute("ALTER TABLE kg_entities ADD COLUMN pm_new text")
        ent(seeded, "p-later", "Later", group="personal")
        seeded.execute("UPDATE kg_entities SET pm_new = 'kept' WHERE uuid = 'p-later'")
        merge(db_url)
        assert (
            one(
                seeded,
                "SELECT pm_new FROM kg_pile_merge_067_entities "
                "WHERE uuid = 'p-later' AND image = 'before'",
            )
            == "kept"
        )
        out = _undo(db_url, run="2")  # undo copes with the columns that differ too
        assert out["drift"]["folded entity uuid already exists"]["count"] == 0
    finally:
        seeded.execute("ALTER TABLE kg_entities DROP COLUMN IF EXISTS pm_old")
        seeded.execute("ALTER TABLE kg_entities DROP COLUMN IF EXISTS pm_new")


def test_a_failure_in_the_tail_leaves_the_merged_data_and_the_066_stamp(seeded, db_url):
    # a second trigger on the same function: DROP FUNCTION cannot go through
    seeded.execute(
        "CREATE TRIGGER dep_on_fn BEFORE INSERT ON kg_relationships FOR EACH ROW "
        "EXECUTE FUNCTION kg_relationships_personal_audience_trg()"
    )
    try:
        res = _psql(db_url, _SQL_067, check=False)
        assert res.returncode != 0 and "depend" in res.stderr
        assert (
            one(seeded, "SELECT count(*) FROM kg_entities WHERE group_id = 'personal'") == 0
        )  # A committed
        assert (
            one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"
        )  # B did not
        assert (
            one(seeded, "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'")
            == 1
        )
    finally:
        seeded.execute("DROP TRIGGER dep_on_fn ON kg_relationships")
    merge(db_url)  # the re-run finishes the tail
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
    assert one(seeded, "SELECT count(*) FROM kg_pile_merge_067_runs") == 1


def test_a_failed_self_check_rolls_the_whole_merge_back(seeded, db_url):
    # two flagged rows for one owner after the merge (a third group is outside 067's reach)
    ent(seeded, "x-flag", "Other", group="elsewhere", flag=True)
    res = _psql(db_url, _SQL_067, check=False)
    assert res.returncode != 0 and "more than one flagged entity" in res.stderr
    assert one(seeded, "SELECT count(*) FROM kg_entities WHERE group_id = 'personal'") == 11
    assert (
        rows(seeded, "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'")
        == []
    )
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "066"


def test_an_invalid_owner_only_index_stops_the_file_before_anything_is_written(seeded, db_url):
    # a cancelled or failed CREATE INDEX CONCURRENTLY leaves an INVALID index behind
    with pytest.raises(psycopg.errors.UniqueViolation):
        seeded.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY kg_rel_tgt_any ON kg_relationships (owner_id, tgt_uuid)"
        )
    try:
        assert (
            one(
                seeded,
                "SELECT indisvalid FROM pg_index WHERE indexrelid = 'kg_rel_tgt_any'::regclass",
            )
            is False
        )
        res = _psql(db_url, _SQL_067, check=False)
        assert res.returncode != 0 and "INVALID" in res.stderr
        assert one(seeded, "SELECT count(*) FROM kg_entities WHERE group_id = 'personal'") == 11
        health = pm.health(pm._connect(db_url, read_only=True))
        assert (
            "kg_rel_tgt_any"
            in health["checks"]["owner-only indexes exist and are valid"]["missing_or_invalid"]
        )
    finally:
        seeded.execute("DROP INDEX kg_rel_tgt_any")


def test_the_owner_only_indexes_exist_and_are_valid_after_the_file(piles, db_url):
    merge(db_url)
    valid = dict(
        rows(
            piles,
            "SELECT c.relname, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = ANY(%s)",
            list(INDEXES),
        )
    )
    assert valid == {ix: True for ix in INDEXES}


def test_the_column_defaults_are_technical_after_the_file(piles, db_url):
    merge(db_url)
    piles.execute("INSERT INTO kg_entities (uuid, name) VALUES ('dflt', 'x')")
    piles.execute(
        "INSERT INTO kg_relationships (uuid, src_uuid, tgt_uuid) VALUES ('dflt', 'a', 'b')"
    )
    piles.execute("INSERT INTO notes (owner_id, type, hook, body) VALUES ('o', 'user', 'h', 'b')")
    piles.execute("INSERT INTO preferences (owner_id, pref, polarity) VALUES ('o', 'p', 'like')")
    for t in KG_TABLES:
        assert one(piles, f"SELECT DISTINCT group_id FROM {t}") == "technical"


# ---------------------------------------------------------------------------
# The exclusion list (case 33)
# ---------------------------------------------------------------------------


def _exclusion_world(c) -> None:
    ent(c, "t-xeno", "Xeno", supertype="Tool")
    ent(c, "p-xeno", "Xeno", group="personal", supertype="Person")  # the owner keeps these apart
    ent(c, "t-yarn", "Yarn")
    ent(c, "p-yarn", "Yarn", group="personal")
    ent(c, "t-root", "Root", flag=True)
    ent(c, "p-root", "Different", group="personal", flag=True)
    fact(c, "f-x", "personal", "p-xeno", "p-yarn", "xeno and yarn")
    fact(c, "f-x2", "personal", "p-xeno", "t-xeno", "kept apart")


def _exclude_file(tmp_path: Path, *lines: str) -> str:
    p = tmp_path / "exclude.tsv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_an_excluded_twin_is_regrouped_not_folded_and_its_edges_are_untouched(
    piles, db_url, tmp_path, capsys
):
    _exclusion_world(piles)
    path = _exclude_file(
        tmp_path,
        "# reviewed by hand",
        f"{OWNER}\tp-xeno",
        f"{OWNER}\tp-root",
        f"{OWNER}\tnot-there",
        "",
    )
    # a dry run reads the file and writes nothing
    assert pm.main(["--exclude", path, "--db-url", db_url]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert (
        dry["pairs in file"] == 3
        and dry["exist as personal entities"] == 2
        and dry["written"] is False
    )
    assert (
        dry["of those, personal twins (would have been folded)"] == 1
    )  # p-root has no same-name twin
    # but it is flagged, and the owner already has a flagged technical entity: it folds anyway
    folded = dry["of those, still folded: the owner-flag rule ignores the listing"]
    assert folded["uuids"] == ["p-root"] and "rule a" in dry["warnings"][0]
    assert one(piles, "SELECT to_regclass('public.kg_pile_merge_067_exclude')") is None
    assert pm.main(["--exclude", path, "--apply", "--db-url", db_url]) == 0
    assert json.loads(capsys.readouterr().out)["newly listed"] == 3
    assert pm.main(["--exclude", path, "--apply", "--db-url", db_url]) == 0  # idempotent
    assert json.loads(capsys.readouterr().out)["newly listed"] == 0
    c = run_counts_preview(db_url)
    assert (c["exclude_listed"], c["exclude_personal"], c["exclude_twins"]) == (3, 2, 1)
    assert (c["exclude_owner_flag"], c["exclude_outranks_twin"]) == (1, 0)
    before, kept = fact_row(piles, "f-x"), fact_row(piles, "f-x2")
    merge(db_url)
    m = fold_map(piles)
    assert "p-xeno" not in m  # regrouped as it is
    assert (
        ent_row(piles, "p-xeno")["group_id"] == "technical"
        and ent_row(piles, "p-xeno")["entity_supertype"] == "Person"
    )
    assert m["p-yarn"] == ("t-yarn", "name")  # the other twin folded
    f = fact_row(piles, "f-x")
    assert (f["src_uuid"], f["tgt_uuid"]) == ("p-xeno", "t-yarn") and before["src_uuid"] == "p-xeno"
    assert {**fact_row(piles, "f-x2"), "group_id": None} == {
        **kept,
        "group_id": None,
    }  # its edges are untouched
    # the owner-flag rule is not an exclusion's to override: two flagged rows cannot coexist
    assert m["p-root"] == ("t-root", "owner-flag")
    applied = run_counts(piles)
    assert (applied["exclude_listed"], applied["exclude_personal"], applied["exclude_twins"]) == (
        3,
        2,
        1,
    )


def test_a_listed_entity_with_a_lower_id_than_its_twin_is_reported_as_first_for_its_name(
    piles, db_url, tmp_path, capsys, monkeypatch
):
    """Kept apart, it is still a technical row of that name, and the oldest: every later mention
    resolves to it, the Person where the mentions were about the tool."""
    ent(piles, "p-first", "Jordan", group="personal", supertype="Person")
    ent(piles, "t-jordan", "Jordan", supertype="Tool")
    path = _exclude_file(tmp_path, f"{OWNER}\tp-first")
    assert pm.main(["--exclude", path, "--db-url", db_url]) == 0
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    first = out["of those, kept apart but first for their name by SURVIVOR_ORDER"]
    assert first["uuids"] == ["p-first"] and "outrank" in out["warnings"][0]
    assert "warning:" in cap.err and "Jordan" not in cap.out + cap.err
    assert pm.main(["--exclude", path, "--apply", "--db-url", db_url]) == 0
    capsys.readouterr()
    c = run_counts_preview(db_url)
    assert (c["exclude_outranks_twin"], c["exclude_twins"]) == (1, 1)
    gates = pm.preview(pm._connect(db_url, read_only=True))["gates"]
    assert (
        gates["exclude_outranks_twin is 0 (no listed entity becomes first for its name)"] is False
    )
    merge(db_url)
    assert fold_map(piles) == {}
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    assert KGPostgresReader().entity_uuid_by_normalized_name("jordan") == "p-first"


def test_a_listed_twin_that_is_newer_than_its_twin_is_not_reported_as_first(
    piles, db_url, tmp_path, capsys
):
    ent(piles, "t-jordan", "Jordan", supertype="Tool")
    ent(piles, "p-second", "Jordan", group="personal", supertype="Person")
    assert (
        pm.main(["--exclude", _exclude_file(tmp_path, f"{OWNER}\tp-second"), "--db-url", db_url])
        == 0
    )
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert out["of those, kept apart but first for their name by SURVIVOR_ORDER"]["count"] == 0
    assert "warnings" not in out and cap.err == ""


def test_the_exclusion_table_is_optional_and_067_never_creates_it(seeded, db_url):
    merge(db_url)
    assert one(seeded, "SELECT to_regclass('public.kg_pile_merge_067_exclude')") is None
    assert run_counts(seeded)["exclude_listed"] == 0


def test_the_exclusion_file_must_be_owner_tab_uuid(tmp_path):
    with pytest.raises(SystemExit, match=r":2:"):
        pm.read_exclude_file(_exclude_file(tmp_path, "a\tb", "only-one-field"))
    assert pm.read_exclude_file(_exclude_file(tmp_path, "a\tb", "", "# c", "a\tb", "c\td")) == [
        ("a", "b"),
        ("c", "d"),
    ]


def test_two_applies_racing_record_one_run(seeded, db_url):
    """The second waits on the first's lock, finds nothing personal left and writes no run."""
    errs: list[str] = []

    def go() -> None:
        res = _psql(db_url, _SQL_067, check=False)
        if res.returncode:
            errs.append(res.stderr)

    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    assert one(seeded, "SELECT count(*) FROM kg_pile_merge_067_runs") == 1
    assert one(seeded, "SELECT count(*) FROM kg_entities WHERE group_id = 'personal'") == 0
    assert one(seeded, "SELECT value FROM synapse_meta WHERE key = 'schema_version'") == "067"
