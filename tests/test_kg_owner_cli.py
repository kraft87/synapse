"""``synapse-admin owner``: flag the install owner's own KG entity, by uuid (schema 066).

The flag changes what recall does (the entity is never a hop seed), so the command is a dry
run until ``--apply``, picks its entity by uuid alone, and refuses what would leave two
flagged entities for one owner. Against a database without 066 it exits 2 and says so.
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext

import psycopg
import pytest
from psycopg.rows import dict_row

from ingestion import kg_owner, surface_admin


@pytest.fixture()
def kg(conn):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")

    _wipe()
    yield conn
    _wipe()


def _entity(conn, uid: str, name: str, owner: str = "default") -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, name, normalized_name) VALUES (%s, %s, %s, %s)",
        (uid, owner, name, name.lower()),
    )


def _edge(
    conn,
    src: str,
    tgt: str,
    *,
    owner: str = "default",
    live=True,
    web: int | None = None,
):
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, src_uuid, tgt_uuid, fact, "
        "  t_invalid, web_artifact_id, audience) "
        "VALUES (%s, %s, %s, %s, 'f', %s, %s, 'work-safe')",
        (
            uuid.uuid4().hex,
            owner,
            src,
            tgt,
            None if live else "2026-02-01T00:00:00+00:00",
            web,
        ),
    )


def _flags(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT uuid FROM kg_entities WHERE is_self")}


def _run(db_url: str, *args: str) -> int:
    return surface_admin.main(["--db-url", db_url, "owner", *args])


# ---------------------------------------------------------------------------
# set and clear
# ---------------------------------------------------------------------------


def test_set_is_a_dry_run_until_apply(kg, db_url, capsys):
    _entity(kg, "me", "The Owner")
    _edge(kg, "me", "x")
    _edge(kg, "y", "me")
    _edge(kg, "me", "gone", live=False)
    _edge(kg, "me", "page", web=7)
    assert _run(db_url, "set", "me") == 0
    out = capsys.readouterr().out
    assert "would flag me (The Owner, 2 live edges)" in out
    assert "--apply" in out
    assert _flags(kg) == set()
    assert _run(db_url, "set", "me", "--apply") == 0
    assert capsys.readouterr().out.startswith("flagged me")
    assert _flags(kg) == {"me"}
    assert _run(db_url, "set", "me", "--apply") == 0
    assert "already flagged" in capsys.readouterr().out


def test_the_live_degree_counts_the_entitys_own_owners_edges(kg, db_url, capsys):
    """The count names the entity's owner, as recall's degree gate does, so each half ranges
    over an index that leads with the owner (after schema 069 there is no other kind on the
    edge table). An edge filed under another owner is not this entity's edge."""
    _entity(kg, "me", "The Owner")
    _edge(kg, "me", "x")
    _edge(kg, "y", "me")
    _edge(kg, "me", "z", owner="someone-else")
    _edge(kg, "w", "me", owner="someone-else")
    assert _run(db_url, "set", "me") == 0
    assert "would flag me (The Owner, 2 live edges)" in capsys.readouterr().out
    _entity(kg, "them", "Another Owner", owner="someone-else")
    kg.execute("UPDATE kg_entities SET is_self = true WHERE uuid IN ('me', 'them')")
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        listed = {r["uuid"]: r["live_degree"] for r in kg_owner.flagged(c)}
    assert listed == {"me": 2, "them": 0}


def test_an_owner_holds_one_flagged_entity(kg, db_url, capsys):
    _entity(kg, "me", "Owner")
    _entity(kg, "twin", "Owner twin")
    _entity(kg, "me-2", "Owner")
    assert _run(db_url, "set", "me", "--apply") == 0
    assert _run(db_url, "set", "twin", "--apply") == 1
    assert "me is already flagged: clear it first" in capsys.readouterr().err
    assert _run(db_url, "set", "me-2", "--apply") == 1
    assert "me is already flagged: clear it first" in capsys.readouterr().err
    assert _flags(kg) == {"me"}


def test_another_owner_may_flag_its_own_entity(kg, db_url):
    _entity(kg, "me", "Owner")
    _entity(kg, "them", "Owner", owner="someone-else")
    assert _run(db_url, "set", "me", "--apply") == 0
    assert _run(db_url, "set", "them", "--apply") == 0
    assert _flags(kg) == {"me", "them"}


def test_clear_is_a_dry_run_until_apply_and_frees_the_owner(kg, db_url, capsys):
    _entity(kg, "me", "Owner")
    _entity(kg, "twin", "Owner twin")
    assert _run(db_url, "set", "me", "--apply") == 0
    capsys.readouterr()
    assert _run(db_url, "clear", "me") == 0
    assert "would clear me" in capsys.readouterr().out
    assert _flags(kg) == {"me"}
    assert _run(db_url, "clear", "me", "--apply") == 0
    assert capsys.readouterr().out.startswith("cleared me")
    assert _flags(kg) == set()
    assert _run(db_url, "clear", "me", "--apply") == 0
    assert "not flagged" in capsys.readouterr().out
    assert _run(db_url, "set", "twin", "--apply") == 0


def test_an_unknown_uuid_exits_one_and_changes_nothing(kg, db_url, capsys):
    _entity(kg, "me", "Owner")
    for verb in ("set", "clear"):
        assert _run(db_url, verb, "nope", "--apply") == 1
        assert "no entity nope" in capsys.readouterr().err
    assert _flags(kg) == set()


def test_a_name_is_not_a_uuid(kg, db_url, capsys):
    """The command never looks an entity up by name: a name given as the argument is an
    unknown uuid, even when exactly one entity bears it."""
    _entity(kg, "u-1", "Ada")
    assert _run(db_url, "set", "Ada", "--apply") == 1
    assert "no entity Ada" in capsys.readouterr().err
    assert _flags(kg) == set()


def test_the_parser_has_no_name_or_search_argument(db_url, capsys):
    for argv in (
        ["set", "x", "--name", "Ada"],
        ["set", "--search", "Ada"],
        ["set", "x", "y"],
        ["find", "Ada"],
        ["set"],
    ):
        with pytest.raises(SystemExit) as e:
            _run(db_url, *argv)
        assert e.value.code == 2, argv
    capsys.readouterr()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_shows_the_flagged_rows_and_the_highest_degree_entities(kg, db_url, capsys):
    _entity(kg, "hub", "Hub")
    _entity(kg, "minor", "Minor")
    _entity(kg, "hub-2", "Hub")
    for i in range(3):
        _edge(kg, "hub", f"leaf{i}")
    _edge(kg, "minor", "hub")
    _edge(kg, "hub-2", "leaf")
    assert _run(db_url, "list") == 0
    out = capsys.readouterr().out
    assert "no entity is flagged" in out
    lines = [ln.split() for ln in out.splitlines() if ln.startswith("  ")]
    # by degree: hub (4: three out, one in) before minor (1) and the second Hub (1), ties by age
    assert [ln[0] for ln in lines] == ["hub", "minor", "hub-2"]
    assert lines[0][1] == "edges=4"
    assert _run(db_url, "set", "hub", "--apply") == 0
    capsys.readouterr()
    assert _run(db_url, "list") == 0
    out = capsys.readouterr().out
    assert "flagged (never a hop seed):" in out
    assert "hub  edges=4  Hub" in out
    assert "[flagged]" in out


def test_candidates_are_per_owner_and_capped(kg, db_url):
    for i in range(7):
        _entity(kg, f"e{i}", f"E{i}")
        _edge(kg, f"e{i}", "x")
    _entity(kg, "other-owner", "Elsewhere", owner="someone-else")
    _edge(kg, "other-owner", "x", owner="someone-else")
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        assert len(kg_owner.candidates(c, "default", limit=5)) == 5
        assert [r["uuid"] for r in kg_owner.candidates(c, "someone-else")] == ["other-owner"]


# ---------------------------------------------------------------------------
# Before schema 066
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_066(kg, db_url):
    """A connection on which 066's columns do not exist (dropped in a transaction that is
    rolled back afterwards)."""
    _entity(kg, "me", "Owner")
    tx = psycopg.connect(db_url, row_factory=dict_row)
    try:
        for col in ("audience", "label_version", "label_model"):
            tx.execute(f"ALTER TABLE kg_relationships DROP COLUMN {col} CASCADE")
        tx.execute("ALTER TABLE kg_entities DROP COLUMN is_self CASCADE")
        yield tx
    finally:
        tx.rollback()
        tx.close()


def test_every_owner_command_exits_two_before_066(no_066, db_url, monkeypatch, capsys):
    monkeypatch.setattr(surface_admin.psycopg, "connect", lambda *a, **k: nullcontext(no_066))
    for argv in (["list"], ["set", "me", "--apply"], ["clear", "me", "--apply"]):
        assert _run(db_url, *argv) == 2, argv
        assert "schema 066 not applied" in capsys.readouterr().err
    # and the functions say the same, without touching a column that is not there
    assert kg_owner.set_owner(no_066, "me", apply=True, key="k")[0] == 2
    assert kg_owner.clear_owner(no_066, "me", apply=True, key="k")[0] == 2
