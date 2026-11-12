"""scripts/repair_numbered_entity_magnets.py end to end against a real database.

The rules are pinned in tests/test_repair_numbered_entity_magnets.py. What only a
database shows is that the script writes what it planned and can take it back:

* the dry run writes nothing (no row changes, no backup tables);
* ``--apply`` re-points the planned endpoints, creates the missing entity (embedding from
  the configured embedder with ``task='entity'``, type and project copied from the
  magnet), clears the magnet's summary, un-retires only allow-listed candidates, copies
  every touched row to the dated backup tables first, and leaves ``source_projects``
  (schema 056) exactly as it was; a second ``--apply`` changes nothing;
* a row that changed between the read and the write is skipped, not overwritten;
* ``--restore`` puts every endpoint, retirement and summary back and deletes the entities
  the run created;
* ``--metrics`` reports the audit's two health numbers before and after.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from mcp_server import kg_pg
from tests.helpers.embed import onehot

_SPEC = importlib.util.spec_from_file_location(
    "repair_numbered_entity_magnets",
    Path(__file__).resolve().parent.parent / "scripts" / "repair_numbered_entity_magnets.py",
)
rm = importlib.util.module_from_spec(_SPEC)
sys.modules["repair_numbered_entity_magnets"] = rm
_SPEC.loader.exec_module(rm)

DAY = "20990101"


def _drop_backups(conn) -> None:
    for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_magnet_repair_%%'"
    ).fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


@pytest.fixture()
def kg(conn):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        _drop_backups(conn)

    _wipe()
    yield conn
    _wipe()


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        self.calls.append((list(texts), task))
        return [onehot(7) for _ in texts]


@pytest.fixture()
def embedder(monkeypatch):
    fake = FakeEmbedder()
    import ingestion.embedding as emb

    monkeypatch.setattr(emb, "create_embedder", lambda *a, **k: fake)
    return fake


def _ep(conn, project: str) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"mag-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _ent(conn, uid: str, name: str, *, summary: str = "", etype: str = "pull_request") -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, project, name, normalized_name, "
        "  entity_type, summary, embedding, created_at, t_created, valid_at, t_valid, "
        "  entity_supertype, entity_subtype) "
        "VALUES (%s, 'default', 'alpha', %s, lower(%s), %s, %s, %s::vector, "
        "        '2026-01-01', '2026-01-01', '2026-01-01', '2026-01-01', 'artifact', %s)",
        (uid, name, name, etype, summary, kg_pg._vec_literal(onehot(1)), etype),
    )


def _edge(
    conn,
    uid: str,
    src: str,
    tgt: str,
    fact: str,
    ep: int,
    *,
    retired_by: str | None = None,
    created: str = "2026-01-02",
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, created_at, t_valid, t_invalid, invalid_at, "
        "  invalidated_by) "
        "VALUES (%s, 'default', %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, %s, "
        "        %s, %s, %s, %s)",
        (
            uid,
            src,
            tgt,
            fact,
            kg_pg._vec_literal(onehot(2)),
            json.dumps([ep]),
            created,
            created,
            "2026-02-01" if retired_by else None,
            "2026-02-01" if retired_by else None,
            retired_by,
        ),
    )


def _seed(conn) -> dict[str, Any]:
    """One magnet ('PR #290') holding facts about three PRs, an existing 'PR #259', a
    ticket-key magnet, and two supersessions: one wrong (disjoint PRs), one real."""
    ep = _ep(conn, "alpha")
    _ent(conn, "mag", "PR #290", summary="describes some other PR entirely")
    _ent(conn, "p259", "PR #259", summary="the retry PR")
    _ent(conn, "key", "ABC-100", summary="an epic", etype="ticket")
    _ent(conn, "svc", "widget service", etype="service")
    _ent(conn, "br", "release branch", etype="branch")
    _edge(conn, "own", "mag", "svc", "PR #290 changed the widget service", ep)
    _edge(conn, "to218", "mag", "svc", "PR #218 was merged into the widget service", ep)
    _edge(conn, "to259", "mag", "br", "PR #259 was rebased onto the release branch", ep)
    _edge(conn, "none", "mag", "br", "the release branch was cut", ep)
    _edge(conn, "two", "mag", "br", "PR #211 was closed because PR #212 does it", ep)
    _edge(conn, "tkt", "key", "svc", "ABC-123 tracks the widget service outage", ep)
    # Wrong supersession: a fact about PR #205 retired by one about PR #218.
    _edge(conn, "new218", "mag", "br", "PR #218 was deployed from the release branch", ep)
    _edge(
        conn,
        "old205",
        "mag",
        "br",
        "PR #205 is live on the release branch",
        ep,
        retired_by="new218",
    )
    # Real correction about one PR: moves with its successor, stays retired.
    _edge(conn, "new77", "mag", "svc", "PR #77 was approved for the widget service", ep)
    _edge(
        conn,
        "old77",
        "mag",
        "svc",
        "PR #77 is waiting on review for the widget service",
        ep,
        retired_by="new77",
    )
    return {"ep": ep}


def _state(conn) -> list[tuple]:
    rels = conn.execute(
        "SELECT uuid, src_uuid, tgt_uuid, t_invalid, invalid_at, invalidated_by, "
        "  source_projects FROM kg_relationships ORDER BY uuid"
    ).fetchall()
    ents = conn.execute("SELECT uuid, name, summary FROM kg_entities ORDER BY uuid").fetchall()
    return rels + ents


def _ends(conn, uid: str) -> tuple[str, str]:
    return conn.execute(
        "SELECT src_uuid, tgt_uuid FROM kg_relationships WHERE uuid = %s", (uid,)
    ).fetchone()


def _uuid_named(conn, name: str) -> str:
    rows = conn.execute("SELECT uuid FROM kg_entities WHERE name = %s", (name,)).fetchall()
    assert len(rows) == 1, (name, rows)
    return rows[0][0]


def _backups(conn) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_magnet_repair_%%'"
        ).fetchall()
    }


def _allow(tmp_path: Path, *uuids: str) -> str:
    p = tmp_path / "allow.txt"
    p.write_text("# reviewed\n" + "".join(f"{u}  why\n" for u in uuids))
    return str(p)


def _run(db_url: str, *args: str) -> None:
    assert rm.main(["--db-url", db_url, *args]) == 0


def test_an_existing_landing_entity_is_reused(kg, db_url, embedder):
    ep = _ep(kg, "alpha")
    _ent(kg, "mag", "PR #290")
    _ent(kg, "p12", "PR #12")
    _ent(kg, "svc", "widget service", etype="service")
    _edge(kg, "e1", "mag", "svc", "PR #12 changed the widget service", ep)

    _run(db_url, "--apply", "--date", DAY)

    assert _ends(kg, "e1") == ("p12", "svc")
    assert embedder.calls == []  # nothing was created: the old entity took the edge


def test_dry_run_writes_nothing(kg, db_url, embedder, capsys):
    _seed(kg)
    before = _state(kg)
    _run(db_url, "--samples", "3")
    assert _state(kg) == before
    assert _backups(kg) == set()
    assert embedder.calls == []
    out = capsys.readouterr().out
    assert '"dry_run": true' in out and "health after this plan" in out


def test_apply_moves_creates_clears_and_backs_up(kg, db_url, embedder, tmp_path):
    _seed(kg)
    sp_before = dict(kg.execute("SELECT uuid, source_projects FROM kg_relationships").fetchall())
    _run(db_url, "--apply", "--date", DAY, "--unretire-allow", _allow(tmp_path, "old205"))

    p218 = _uuid_named(kg, "PR #218")
    p205 = _uuid_named(kg, "PR #205")
    p77 = _uuid_named(kg, "PR #77")
    a123 = _uuid_named(kg, "ABC-123")
    assert _ends(kg, "to218") == (p218, "svc")
    assert _ends(kg, "new218") == (p218, "br")
    assert _ends(kg, "to259") == ("p259", "br")  # the existing entity is reused
    assert _ends(kg, "tkt") == (a123, "svc")
    assert _ends(kg, "old205") == (p205, "br")
    # A real correction moves with its successor: still one pair, still retired.
    assert _ends(kg, "old77") == _ends(kg, "new77") == (p77, "svc")
    assert kg.execute(
        "SELECT t_invalid IS NOT NULL, invalidated_by FROM kg_relationships WHERE uuid = 'old77'"
    ).fetchone() == (True, "new77")
    # Own number, no identifier, several identifiers: left where they were.
    for uid in ("own", "none", "two"):
        assert _ends(kg, uid)[0] == "mag"

    # The allow-listed wrong supersession is live again, as invalidate_edges found it.
    assert kg.execute(
        "SELECT t_invalid, invalid_at, invalidated_by FROM kg_relationships WHERE uuid = 'old205'"
    ).fetchone() == (None, None, None)

    # New entities: embedded as entities, type/project from the magnet, no summary.
    (names, task), *_ = embedder.calls
    assert task == "entity" and set(names) == {"PR #218", "PR #205", "PR #77", "ABC-123"}
    row = kg.execute(
        "SELECT entity_type, entity_supertype, entity_subtype, project, summary, "
        "  normalized_name, embedding IS NOT NULL FROM kg_entities WHERE uuid = %s",
        (p218,),
    ).fetchone()
    assert row == ("pull_request", "artifact", "pull_request", "alpha", "", "pr #218", True)
    assert kg.execute(
        "SELECT entity_type FROM kg_entities WHERE uuid = %s", (a123,)
    ).fetchone() == ("ticket",)

    # Losers' summaries are cleared; an entity that only gained keeps its own.
    summaries = dict(kg.execute("SELECT uuid, summary FROM kg_entities").fetchall())
    assert summaries["mag"] == "" and summaries["key"] == ""
    assert summaries["p259"] == "the retry PR"

    # Provenance is untouched: endpoints and retirement are not in its trigger's columns.
    assert (
        dict(kg.execute("SELECT uuid, source_projects FROM kg_relationships").fetchall())
        == sp_before
    )

    # Backups: every touched edge and entity row, as it was, plus the created uuids.
    rel = f"kg_magnet_repair_{DAY}_relationships"
    assert _backups(kg) == {
        rel,
        f"kg_magnet_repair_{DAY}_entities",
        f"kg_magnet_repair_{DAY}_created",
    }
    backed = dict(kg.execute(f"SELECT uuid, src_uuid FROM {rel}").fetchall())
    assert set(backed) == {"to218", "to259", "tkt", "new218", "old205", "new77", "old77"}
    assert set(backed.values()) == {"mag", "key"}
    assert {r[0] for r in kg.execute(f"SELECT uuid FROM kg_magnet_repair_{DAY}_entities")} == {
        "mag",
        "key",
    }
    assert {r[0] for r in kg.execute(f"SELECT uuid FROM kg_magnet_repair_{DAY}_created")} == {
        p218,
        p205,
        p77,
        a123,
    }

    # A second run has nothing left to do.
    after = _state(kg)
    _run(db_url, "--apply", "--date", DAY, "--unretire-allow", _allow(tmp_path, "old205"))
    assert _state(kg) == after


def test_candidates_not_allow_listed_stay_retired(kg, db_url, embedder):
    _seed(kg)
    _run(db_url, "--apply", "--date", DAY)
    assert kg.execute(
        "SELECT t_invalid IS NOT NULL, invalidated_by FROM kg_relationships WHERE uuid = 'old205'"
    ).fetchone() == (True, "new218")


def test_a_row_changed_after_the_read_is_skipped(kg, db_url, embedder):
    _seed(kg)
    conn = rm._connect(db_url, read_only=False)
    ents, edges = rm.load(conn)
    plan = rm.plan_repair(ents, edges)
    # Someone re-points the edge between the read and the write.
    kg.execute("UPDATE kg_relationships SET src_uuid = 'p259' WHERE uuid = 'to218'")
    conn.autocommit = True
    done = rm.apply(conn, plan, embedder, DAY, set())
    conn.close()
    assert _ends(kg, "to218") == ("p259", "svc")
    assert done["endpoints changed since read (skipped)"] == 1


def test_restore_puts_everything_back(kg, db_url, embedder, tmp_path, capsys):
    _seed(kg)
    before = _state(kg)
    _run(db_url, "--apply", "--date", DAY, "--unretire-allow", _allow(tmp_path, "old205"))
    assert _state(kg) != before

    _run(db_url, "--restore", DAY)  # dry run: rolled back
    assert '"written": false' in capsys.readouterr().out
    assert _state(kg) != before

    _run(db_url, "--restore", DAY, "--apply")
    assert _state(kg) == before


def test_restore_keeps_a_created_entity_that_new_facts_use(kg, db_url, embedder, capsys):
    ep = _seed(kg)["ep"]
    _run(db_url, "--apply", "--date", DAY)
    p218 = _uuid_named(kg, "PR #218")
    _edge(kg, "later", p218, "svc", "PR #218 was reverted", ep, created="2026-03-01")
    _run(db_url, "--restore", DAY, "--apply")
    out = capsys.readouterr().out
    assert '"created entities kept (new edges reference them)": 1' in out
    assert _ends(kg, "to218") == ("mag", "svc")
    assert _ends(kg, "later") == (p218, "svc")


def test_restore_keeps_what_was_written_after_the_repair(kg, db_url, embedder, tmp_path):
    """--restore reverts what the repair changed and nothing written since: a moved fact
    that extraction later superseded goes back to its old pair still retired, an
    un-retired fact that was retired again keeps the newer retirement, and a summary the
    next mention wrote over the cleared one is kept."""
    ep = _seed(kg)["ep"]
    _run(db_url, "--apply", "--date", DAY, "--unretire-allow", _allow(tmp_path, "old205"))
    p218 = _uuid_named(kg, "PR #218")
    _edge(kg, "later", p218, "svc", "PR #218 was reverted", ep, created="2026-03-01")
    kg.execute(
        "UPDATE kg_relationships SET t_invalid = '2026-03-01', invalid_at = '2026-03-01', "
        "  invalidated_by = 'later' WHERE uuid IN ('to218', 'old205')"
    )
    kg.execute("UPDATE kg_entities SET summary = 'PR #290 changed the widget' WHERE uuid = 'mag'")

    _run(db_url, "--restore", DAY, "--apply")
    assert _ends(kg, "to218") == ("mag", "svc")
    assert _ends(kg, "old205") == ("mag", "br")
    for uid in ("to218", "old205"):
        assert kg.execute(
            "SELECT t_invalid IS NOT NULL, invalidated_by FROM kg_relationships WHERE uuid = %s",
            (uid,),
        ).fetchone() == (True, "later")
    summaries = dict(kg.execute("SELECT uuid, summary FROM kg_entities").fetchall())
    assert summaries["mag"] == "PR #290 changed the widget"
    assert summaries["key"] == "an epic"  # still cleared, so the backup's comes back


@pytest.mark.parametrize("summary", ["", "describes some other PR"])
def test_restore_brings_back_a_magnet_the_orphan_sweep_deleted(
    kg, db_url, embedder, monkeypatch, summary
):
    """A magnet whose every edge moved has no edge left, and the poller's daily orphan
    sweep deletes it. --restore re-creates it from the backup (every entity that loses an
    edge is backed up, summary or not) before pointing the edges back at it."""
    from ingestion.kg_pg_write import KGPostgresWriter

    ep = _ep(kg, "alpha")
    _ent(kg, "drained", "PR #95", summary=summary)
    _ent(kg, "svc", "widget service", etype="service")
    _edge(kg, "only", "drained", "svc", "PR #94 was deployed to the widget service", ep)
    before = _state(kg)
    _run(db_url, "--apply", "--date", DAY)
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    assert KGPostgresWriter().delete_orphan_entities(0) == 1

    _run(db_url, "--restore", DAY, "--apply")
    assert _state(kg) == before
    assert kg.execute(
        "SELECT entity_type, project, embedding IS NOT NULL FROM kg_entities WHERE uuid = 'drained'"
    ).fetchone() == ("pull_request", "alpha", True)


def test_the_dry_run_and_metrics_read_through_a_read_only_connection(
    kg, db_url, embedder, monkeypatch
):
    seen: list[tuple[bool, str]] = []
    real = rm._connect

    def spy(url: str, *, read_only: bool) -> Any:
        c = real(url, read_only=read_only)
        seen.append((read_only, c.execute("SHOW transaction_read_only").fetchone()[0]))
        c.rollback()
        return c

    monkeypatch.setattr(rm, "_connect", spy)
    _seed(kg)
    _run(db_url)
    _run(db_url, "--metrics")
    assert seen == [(True, "on"), (True, "on")]


def test_metrics_report_the_audit_numbers_before_and_after(kg, db_url, embedder, tmp_path, capsys):
    _seed(kg)
    _run(db_url, "--metrics")
    before = json.loads(capsys.readouterr().out)
    # Live edges on 'PR #290': own, to218, to259, none, two, new218, new77 -> 6 lack '#290';
    # on 'PR #259': none yet.
    assert before["name is PR #N"] == {
        "entities": 1,
        "live_edges": 7,
        "missing_own_number": 6,
        "ratio": 0.857,
    }
    assert before["linked supersessions on ID-named entities"] == {
        "linked": 2,
        "both_name_ids": 2,
        "disjoint": 1,
    }
    _run(db_url, "--apply", "--date", DAY, "--unretire-allow", _allow(tmp_path, "old205"))
    capsys.readouterr()
    _run(db_url, "--metrics")
    after = json.loads(capsys.readouterr().out)
    # Left on "PR #290": own, none, two -> 2 lack "#290"; every moved edge names its own.
    assert after["name is PR #N"]["missing_own_number"] == 2
    assert after["name is PR #N"]["live_edges"] == 8  # old205 is live again
    assert after["linked supersessions on ID-named entities"]["disjoint"] == 0
