"""DB-backed tests for the notes-curation store (schema 051) and the dream→notes
lane's end-to-end behaviour against real Postgres.

Covers the things that can only be checked against a database: the candidate
queries (cosine floor, ordering, the memo join and its updated_at staleness rule),
the upsert semantics of notes_curation, and the applied writes — supersession is
idempotent (the merged pair is gone from the live self-join on the next run) and a
retype lands only for a project slug the judge was offered; plus the operator's
undo path (list / revert) over the rows the lane wrote.

The LLM is stubbed throughout (no live calls), the same seam the NOTES_CONFIRM
tests use. The pure half lives in tests/test_notes_lane.py.
"""

from __future__ import annotations

import math

import pytest

import dream.notes.nightly as lane
from dream.notes.nightly import list_applied, revert_curation, run_lane
from ingestion.db import Database
from ingestion.embedding import embed_dims
from ingestion.settings import KG_OWNER_ID
from tests.helpers.embed import GROUP

pytestmark = pytest.mark.db

_DIMS = embed_dims()

_DUP = {"verdict": "DUPLICATE", "keep": None}
_DIS = {"verdict": "DISTINCT", "keep": None}


def _angled(degrees: float) -> list[float]:
    """A unit vector in one fixed 2D plane of the embedding space: two vectors at
    ``d1`` and ``d2`` degrees have cosine similarity ``cos(d1 - d2)``, which is how
    these tests place candidates precisely above and below the 0.60 floor."""
    v = [0.0] * _DIMS
    v[1] = math.cos(math.radians(degrees))
    v[2] = math.sin(math.radians(degrees))
    return v


def _wipe(conn):
    conn.execute("DELETE FROM notes_curation")
    conn.execute("DELETE FROM notes")


def _insert(
    db, *, hook, type="user", project=None, degrees=0.0, body="Example body.", audience=None
):
    return db.insert_note(
        owner_id=KG_OWNER_ID,
        group_id=GROUP,
        project=project,
        type=type,
        hook=hook,
        body=body,
        embedding=_angled(degrees),
        embed_model="test",
        source_ref=None,
        audience=audience,
    )


def _stub_llm(monkeypatch, pair_verdicts=(), scope_verdicts=()):
    pairs, scopes = list(pair_verdicts), list(scope_verdicts)

    def _fake_pair(llm, first, second):
        return pairs.pop(0)

    def _fake_scope(llm, *, base_prompt, parser, model, max_tokens):
        return scopes.pop(0)

    monkeypatch.setattr(lane, "judge_pair", _fake_pair)
    monkeypatch.setattr(lane, "parse_with_retry", _fake_scope)


# ---------------------------------------------------------------------------
# Candidate queries
# ---------------------------------------------------------------------------


def test_pair_candidates_respect_floor_and_order(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Deploys run from the release branch", degrees=0)
    b = _insert(db, hook="Releases are cut from the release branch", degrees=20)  # cos20 = .94
    c = _insert(db, hook="Release cuts happen on the release branch", degrees=35)  # cos15 = .97
    _insert(db, hook="Unrelated: the office coffee machine", degrees=100)

    got = db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10)
    assert [(p["a"]["id"], p["b"]["id"]) for p in got] == [(b, c), (a, b), (a, c)]
    assert got[0]["sim"] > got[1]["sim"] > got[2]["sim"]  # ordered by similarity, descending
    assert got[0]["sim"] == pytest.approx(math.cos(math.radians(15)), abs=1e-3)
    # Each side carries everything the judge prompt needs.
    assert set(got[0]["a"]) == {
        "id",
        "hook",
        "body",
        "type",
        "project",
        "audience",
        "updated_at",
    }
    # The far-away note pairs with nothing above the floor.
    assert all(0.60 <= p["sim"] for p in got)

    assert db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.95, limit=10) == [
        p for p in got if p["sim"] >= 0.95
    ]
    db.close()
    _wipe(conn)


def test_pair_candidates_skip_superseded_and_null_embedding(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    db.insert_note(
        owner_id=KG_OWNER_ID,
        group_id=GROUP,
        project=None,
        type="user",
        hook="Keyless note",
        body="b",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )
    assert len(db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10)) == 1
    db.supersede_note(a, b)
    assert db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10) == []
    db.close()
    _wipe(conn)


def test_memo_hides_a_judged_pair_until_a_note_changes(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    assert len(db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10)) == 1

    db.record_curation(op="pair", note_a=a, note_b=b, verdict="DISTINCT", applied=False)
    assert db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10) == []

    # An edit to EITHER note makes the verdict stale — the pair comes back.
    conn.execute("UPDATE notes SET updated_at = now() + interval '1 hour' WHERE id = %s", (b,))
    assert len(db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10)) == 1
    db.close()
    _wipe(conn)


def test_memo_is_order_insensitive(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    # Recorded with the ids reversed — still the same judgement.
    db.record_curation(op="pair", note_a=b, note_b=a, verdict="DISTINCT", applied=False)
    assert db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10) == []
    db.close()
    _wipe(conn)


def test_pair_candidates_never_pair_notes_of_different_projects(conn, db_url):
    """A project note is served only on its own project's board, so alpha's note and
    beta's cannot stand in for one another. A project note and a global one still pair."""
    _wipe(conn)
    db = Database(db_url)
    alpha = _insert(
        db, hook="Deploys go through the release branch", type="project", project="alpha", degrees=0
    )
    beta = _insert(
        db,
        hook="Releases are cut from the release branch",
        type="project",
        project="beta",
        degrees=10,
    )
    alpha2 = _insert(
        db,
        hook="Release branch is where alpha deploys from",
        type="project",
        project="alpha",
        degrees=15,
    )
    glob = _insert(db, hook="Always deploy from the release branch", type="feedback", degrees=20)

    pairs = {
        frozenset((p["a"]["id"], p["b"]["id"]))
        for p in db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=20)
    }
    assert frozenset((alpha, beta)) not in pairs
    assert frozenset((beta, alpha2)) not in pairs
    assert frozenset((alpha, alpha2)) in pairs
    assert {frozenset((alpha, glob)), frozenset((beta, glob))} <= pairs
    db.close()
    _wipe(conn)


def test_retype_candidates_are_global_live_and_unjudged(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    u = _insert(db, hook="User prefers numbered steps", type="user")
    f = _insert(db, hook="Never open with praise", type="feedback")
    _insert(db, hook="Service X caches tokens for an hour", type="project", project="demo-api")
    _insert(db, hook="See the deployment runbook", type="reference")

    assert [n["id"] for n in db.find_retype_candidates(KG_OWNER_ID, limit=10)] == [u, f]

    db.record_curation(op="retype", note_a=u, note_b=None, verdict="GLOBAL", applied=False)
    assert [n["id"] for n in db.find_retype_candidates(KG_OWNER_ID, limit=10)] == [f]

    conn.execute("UPDATE notes SET updated_at = now() + interval '1 hour' WHERE id = %s", (u,))
    assert [n["id"] for n in db.find_retype_candidates(KG_OWNER_ID, limit=10)] == [u, f]
    db.close()
    _wipe(conn)


# ---------------------------------------------------------------------------
# Audit table
# ---------------------------------------------------------------------------


def test_record_curation_upserts_in_place(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    first = db.record_curation(
        op="pair", note_a=a, note_b=b, verdict="DISTINCT", applied=False, detail={"reason": "x"}
    )
    second = db.record_curation(
        op="pair", note_a=b, note_b=a, verdict="DUPLICATE", applied=True, detail={"reason": "y"}
    )
    assert first == second  # same row, re-judged
    rows = conn.execute("SELECT verdict, applied, detail FROM notes_curation").fetchall()
    assert len(rows) == 1 and rows[0][0] == "DUPLICATE" and rows[0][1] is True
    assert rows[0][2] == {"reason": "y"}

    # retype rows key on note_a alone and live alongside the pair row.
    db.record_curation(op="retype", note_a=a, note_b=None, verdict="GLOBAL", applied=False)
    db.record_curation(op="retype", note_a=a, note_b=None, verdict="PROJECT", applied=False)
    assert conn.execute("SELECT count(*) FROM notes_curation").fetchone()[0] == 2
    db.close()
    _wipe(conn)


def test_curation_rows_follow_their_notes(conn, db_url):
    """ON DELETE CASCADE: the audit row is about the notes, so it goes when they do
    (and a note delete is never blocked by an FK from the lane's bookkeeping)."""
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    db.record_curation(op="pair", note_a=a, note_b=b, verdict="DISTINCT", applied=False)
    conn.execute("DELETE FROM notes WHERE id = %s", (b,))
    assert conn.execute("SELECT count(*) FROM notes_curation").fetchone()[0] == 0
    db.close()
    _wipe(conn)


def test_known_project_slugs_rank_note_projects_first(conn, db_url):
    _wipe(conn)
    conn.execute("DELETE FROM episodes WHERE session_id = 'notes-curation-test'")
    db = Database(db_url)
    _insert(db, hook="Service X caches tokens", type="project", project="demo-api")
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES ('notes-curation-test', 1, 'demo-web', 'example turn')"
    )
    slugs = db.known_project_slugs()
    assert slugs[0] == "demo-api" and "demo-web" in slugs  # note projects rank first
    db.close()
    conn.execute("DELETE FROM episodes WHERE session_id = 'notes-curation-test'")
    _wipe(conn)


# ---------------------------------------------------------------------------
# End-to-end lane runs
# ---------------------------------------------------------------------------


def test_lane_supersedes_and_is_idempotent(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0)
    b = _insert(db, hook="The nightly backup job fires at 02:00", degrees=20)

    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    res = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert res["counts"]["applied_supersede"] == 1
    assert res["samples"] == [f"n:{a} -> superseded by n:{b} (duplicate)"]
    assert conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (a,)).fetchone()[0] == b
    row = conn.execute("SELECT op, verdict, applied FROM notes_curation").fetchone()
    assert row == ("pair", "DUPLICATE", True)

    # Second run: the merged note is out of the live set, so the pair is gone from the
    # self-join entirely — no second judgement, no second write.
    _stub_llm(monkeypatch, pair_verdicts=[])  # any judge call would IndexError
    again = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert again["counts"] == {
        "pairs_judged": 0,
        "retype_judged": 0,
        "applied_supersede": 0,
        "applied_retype": 0,
    }
    assert conn.execute("SELECT count(*) FROM notes_curation").fetchone()[0] == 1
    db.close()
    _wipe(conn)


def test_lane_distinct_verdict_leaves_both_notes_live(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0)
    b = _insert(db, hook="Backups are verified every Sunday", degrees=20)
    _stub_llm(monkeypatch, pair_verdicts=[_DIS])  # a DISTINCT first order ends the pair
    run_lane(db=db, llm=object(), max_retype_judge=0)
    live = conn.execute(
        "SELECT count(*) FROM notes WHERE superseded_by IS NULL AND id = ANY(%s)", ([a, b],)
    ).fetchone()[0]
    assert live == 2
    # ...but the pair is memoized, so it is not re-judged tomorrow.
    assert db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=0.60, limit=10) == []
    db.close()
    _wipe(conn)


def test_lane_retype_applies_only_for_an_existing_slug(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    known = _insert(db, hook="Service X caches tokens", type="project", project="demo-api")
    real = _insert(db, hook="demo-api migrates its schema on boot", type="feedback", degrees=40)
    ghost = _insert(db, hook="Some other system rotates keys weekly", type="user", degrees=100)

    _stub_llm(
        monkeypatch,
        scope_verdicts=[
            {"scope": "PROJECT", "project": "demo-api"},
            {"scope": "PROJECT", "project": "ghost-service"},
        ],
    )
    res = run_lane(db=db, llm=object(), max_judge=0)
    assert res["counts"] == {
        "pairs_judged": 0,
        "retype_judged": 2,
        "applied_supersede": 0,
        "applied_retype": 1,
    }
    assert conn.execute("SELECT type, project FROM notes WHERE id = %s", (real,)).fetchone() == (
        "project",
        "demo-api",
    )
    assert conn.execute("SELECT type, project FROM notes WHERE id = %s", (ghost,)).fetchone() == (
        "user",
        None,
    )
    rows = dict(
        conn.execute("SELECT note_a, applied FROM notes_curation WHERE op = 'retype'").fetchall()
    )
    assert rows == {real: True, ghost: False}
    # The audit row keeps what a revert needs to put the note back exactly.
    (detail,) = conn.execute(
        "SELECT detail FROM notes_curation WHERE note_a = %s", (real,)
    ).fetchone()
    assert detail == {
        "from_type": "feedback",
        "from_project": None,
        "from_audience": "personal",
        "project": "demo-api",
        "audience": "personal",
    }
    # The retype did not bump updated_at, so its own memo row is not instantly stale.
    assert [n["id"] for n in db.find_retype_candidates(KG_OWNER_ID, limit=10)] == []
    assert known  # the pre-existing project note is untouched
    db.close()
    _wipe(conn)


def test_lane_retype_matches_the_slug_case_folded_and_files_the_stored_casing(
    conn, db_url, monkeypatch
):
    _wipe(conn)
    db = Database(db_url)
    _insert(db, hook="Service X caches tokens", type="project", project="Demo-API")
    note = _insert(db, hook="demo-api migrates its schema on boot", type="feedback", degrees=40)
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "demo-api"}])
    run_lane(db=db, llm=object(), max_judge=0)
    assert conn.execute("SELECT type, project FROM notes WHERE id = %s", (note,)).fetchone() == (
        "project",
        "Demo-API",
    )
    db.close()
    _wipe(conn)


def test_lane_leaves_two_projects_notes_alone_without_asking_the_judge(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    _insert(db, hook="Deploys go through the release branch", type="project", project="alpha")
    _insert(
        db,
        hook="Releases are cut from the release branch",
        type="project",
        project="beta",
        degrees=10,
    )
    _stub_llm(monkeypatch)  # any judge call would IndexError
    res = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert res["counts"]["pairs_judged"] == 0
    db.close()
    _wipe(conn)


# ---------------------------------------------------------------------------
# Undo: list and revert
# ---------------------------------------------------------------------------


def _merged_pair(conn, db, monkeypatch):
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0)
    b = _insert(db, hook="The nightly backup job fires at 02:00", degrees=20)
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    run_lane(db=db, llm=object(), max_retype_judge=0)
    (cid,) = conn.execute("SELECT id FROM notes_curation").fetchone()
    return a, b, cid


def test_revert_brings_a_merged_note_back_and_the_lane_leaves_it_be(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    a, b, cid = _merged_pair(conn, db, monkeypatch)
    (line,) = list_applied(db)
    assert line.startswith(f"#{cid} ") and f"n:{a} superseded by n:{b}" in line
    assert "retired: Backups run nightly at 02:00" in line

    assert revert_curation(db, cid) == f"reverted #{cid}: n:{a} is live again"
    assert conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (a,)).fetchone()[0] is None
    assert conn.execute(
        "SELECT verdict, applied FROM notes_curation WHERE id = %s", (cid,)
    ).fetchone() == ("REVERTED", False)
    assert list_applied(db) == []
    # The memo still covers the pair: tomorrow's run does not merge it again.
    _stub_llm(monkeypatch)
    assert run_lane(db=db, llm=object(), max_retype_judge=0)["counts"]["pairs_judged"] == 0
    # ...and reverting twice is an explicit error, not a silent no-op.
    with pytest.raises(ValueError, match="was not applied"):
        revert_curation(db, cid)
    db.close()
    _wipe(conn)


def test_revert_refuses_when_the_merge_has_since_changed(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    a, _b, cid = _merged_pair(conn, db, monkeypatch)
    conn.execute("UPDATE notes SET superseded_by = NULL WHERE id = %s", (a,))  # someone fixed it
    with pytest.raises(ValueError, match="no longer superseded"):
        revert_curation(db, cid)
    assert conn.execute("SELECT applied FROM notes_curation WHERE id = %s", (cid,)).fetchone()[0]
    db.close()
    _wipe(conn)


def test_revert_puts_a_retyped_note_back_exactly(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    _insert(db, hook="Service X caches tokens", type="project", project="demo-api")
    note = _insert(
        db,
        hook="demo-api migrates its schema on boot",
        type="feedback",
        project="old-shelf",
        audience="work-safe",
        degrees=40,
    )
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "demo-api"}])
    run_lane(db=db, llm=object(), max_judge=0)
    assert conn.execute(
        "SELECT type, project, audience FROM notes WHERE id = %s", (note,)
    ).fetchone() == ("project", "demo-api", "personal")
    (cid,) = conn.execute("SELECT id FROM notes_curation WHERE applied").fetchone()
    assert "retype" in list_applied(db)[0]

    revert_curation(db, cid)
    assert conn.execute(
        "SELECT type, project, audience FROM notes WHERE id = %s", (note,)
    ).fetchone() == ("feedback", "old-shelf", "work-safe")
    assert [
        n["id"] for n in db.find_retype_candidates(KG_OWNER_ID, limit=10)
    ] == []  # still memoized
    db.close()
    _wipe(conn)


def test_revert_refuses_a_retype_it_cannot_undo_exactly(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    note = _insert(db, hook="Moved long ago", type="project", project="demo-api")
    old = db.record_curation(
        op="retype",
        note_a=note,
        note_b=None,
        verdict="PROJECT",
        applied=True,
        detail={"from_type": "feedback", "project": "demo-api", "audience": "personal"},
    )
    with pytest.raises(ValueError, match="predates revert support"):
        revert_curation(db, old)
    with pytest.raises(ValueError, match="no curation row"):
        revert_curation(db, 999999)

    conn.execute("DELETE FROM notes_curation")
    cid = db.record_curation(
        op="retype",
        note_a=note,
        note_b=None,
        verdict="PROJECT",
        applied=True,
        detail={
            "from_type": "feedback",
            "from_project": None,
            "from_audience": "personal",
            "project": "somewhere-else",
            "audience": "personal",
        },
    )
    with pytest.raises(ValueError, match="nothing to undo"):
        revert_curation(db, cid)  # the note is not where the lane left it
    assert conn.execute("SELECT type FROM notes WHERE id = %s", (note,)).fetchone()[0] == "project"
    db.close()
    _wipe(conn)
