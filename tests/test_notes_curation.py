"""DB-backed tests for the notes-curation store (schema 051) and the dream→notes
lane's end-to-end behaviour against real Postgres.

Covers the three things that can only be checked against a database: the candidate
queries (cosine floor, ordering, the memo join and its updated_at staleness rule),
the upsert semantics of notes_curation, and the applied writes — supersession is
idempotent (the merged pair is gone from the live self-join on the next run) and a
retype lands only for a project slug that already exists.

The LLM is stubbed throughout (no live calls), the same seam the NOTES_CONFIRM
tests use. The pure half lives in tests/test_notes_lane.py.
"""

from __future__ import annotations

import math
import os

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

import dream.notes.nightly as lane  # noqa: E402
from dream.notes.nightly import run_lane  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.embedding import embed_dims  # noqa: E402
from ingestion.notes import _OWNER  # noqa: E402

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
        owner_id=_OWNER,
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
    """Stub the judge; returns the list of judge calls it has served."""
    pairs, scopes, calls = list(pair_verdicts), list(scope_verdicts), []

    def _fake(llm, *, base_prompt, parser, model, max_tokens):
        calls.append(parser.__name__)
        if parser is lane.parse_pair_verdict:
            return pairs.pop(0)
        return scopes.pop(0)

    monkeypatch.setattr(lane, "parse_with_retry", _fake)
    return calls


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

    got = db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10)
    assert [(p["a"]["id"], p["b"]["id"]) for p in got] == [(b, c), (a, b), (a, c)]
    assert got[0]["sim"] > got[1]["sim"] > got[2]["sim"]  # ordered by similarity, descending
    assert got[0]["sim"] == pytest.approx(math.cos(math.radians(15)), abs=1e-3)
    # Each side carries everything the judge prompt needs.
    assert set(got[0]["a"]) == {"id", "hook", "body", "type", "project", "audience", "updated_at"}
    # The far-away note pairs with nothing above the floor.
    assert all(0.60 <= p["sim"] for p in got)

    assert db.find_note_pair_candidates(_OWNER, sim_floor=0.95, limit=10) == [
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
        owner_id=_OWNER,
        project=None,
        type="user",
        hook="Keyless note",
        body="b",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )
    assert len(db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10)) == 1
    db.supersede_note(a, b)
    assert db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10) == []
    db.close()
    _wipe(conn)


def test_pair_candidates_carry_each_sides_audience_and_cross_tiers(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0, audience="work-safe")
    b = _insert(db, hook="Alpha restated", degrees=10, audience="personal")
    (pair,) = db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10)
    assert (pair["a"]["id"], pair["a"]["audience"]) == (a, "work-safe")
    assert (pair["b"]["id"], pair["b"]["audience"]) == (b, "personal")
    db.close()
    _wipe(conn)


def test_memo_hides_a_judged_pair_until_a_note_changes(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    assert len(db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10)) == 1

    db.record_curation(op="pair", note_a=a, note_b=b, verdict="DISTINCT", applied=False)
    assert db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10) == []

    # An edit to EITHER note makes the verdict stale — the pair comes back.
    conn.execute("UPDATE notes SET updated_at = now() + interval '1 hour' WHERE id = %s", (b,))
    assert len(db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10)) == 1
    db.close()
    _wipe(conn)


def test_memo_is_order_insensitive(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Alpha", degrees=0)
    b = _insert(db, hook="Alpha restated", degrees=10)
    # Recorded with the ids reversed — still the same judgement.
    db.record_curation(op="pair", note_a=b, note_b=a, verdict="DISTINCT", applied=False)
    assert db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10) == []
    db.close()
    _wipe(conn)


def test_retype_candidates_are_global_live_and_unjudged(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    u = _insert(db, hook="User prefers numbered steps", type="user")
    f = _insert(db, hook="Never open with praise", type="feedback")
    _insert(db, hook="Service X caches tokens for an hour", type="project", project="demo-api")
    _insert(db, hook="See the deployment runbook", type="reference")

    assert [n["id"] for n in db.find_retype_candidates(_OWNER, limit=10)] == [u, f]

    db.record_curation(op="retype", note_a=u, note_b=None, verdict="GLOBAL", applied=False)
    assert [n["id"] for n in db.find_retype_candidates(_OWNER, limit=10)] == [f]

    conn.execute("UPDATE notes SET updated_at = now() + interval '1 hour' WHERE id = %s", (u,))
    assert [n["id"] for n in db.find_retype_candidates(_OWNER, limit=10)] == [u, f]
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


def test_project_slug_lookups(conn, db_url):
    _wipe(conn)
    conn.execute("DELETE FROM episodes WHERE session_id = 'notes-curation-test'")
    db = Database(db_url)
    _insert(db, hook="Service X caches tokens", type="project", project="demo-api")
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES ('notes-curation-test', 1, 'demo-web', 'example turn')"
    )
    assert db.project_slug_exists("demo-api") is True  # from notes
    assert db.project_slug_exists("demo-web") is True  # from episodes
    assert db.project_slug_exists("ghost-service") is False
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


@pytest.mark.parametrize(
    ("older_tier", "newer_tier"),
    [("work-safe", "personal"), ("personal", "work-safe")],
    ids=["work-safe-then-personal", "personal-then-work-safe"],
)
def test_lane_keeps_a_duplicate_pair_that_spans_two_tiers(
    conn, db_url, monkeypatch, older_tier, newer_tier
):
    """A restatement in the other tier is stored as a twin on purpose. Retiring the
    older of the two would take that tier's line away, so a DUPLICATE across tiers
    retires neither, and is memoised so it is not judged again every night."""
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0, audience=older_tier)
    b = _insert(db, hook="The nightly backup job fires at 02:00", degrees=20, audience=newer_tier)

    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    res = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert res["counts"]["pairs_judged"] == 1 and res["counts"]["applied_supersede"] == 0
    assert res["samples"] == []
    live = conn.execute(
        "SELECT id FROM notes WHERE superseded_by IS NULL AND id = ANY(%s) ORDER BY id", ([a, b],)
    ).fetchall()
    assert live == [(a,), (b,)]
    rows = conn.execute("SELECT op, verdict, applied, detail FROM notes_curation").fetchall()
    assert len(rows) == 1
    assert rows[0][:3] == ("pair", "DUPLICATE", False)
    assert rows[0][3]["kept"] == "other tier"

    # The memo holds: tomorrow's run neither sees the pair nor calls the judge.
    calls = _stub_llm(monkeypatch, pair_verdicts=[])
    again = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert calls == [] and again["counts"]["pairs_judged"] == 0 and again["errors"] == []
    db.close()
    _wipe(conn)


@pytest.mark.parametrize("tier", ["personal", "work-safe"])
def test_lane_retires_the_older_note_of_a_duplicate_pair_within_one_tier(
    conn, db_url, monkeypatch, tier
):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0, audience=tier)
    b = _insert(db, hook="The nightly backup job fires at 02:00", degrees=20, audience=tier)
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    res = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert res["counts"]["applied_supersede"] == 1
    assert conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (a,)).fetchone()[0] == b
    assert conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (b,)).fetchone()[0] is None
    row = conn.execute("SELECT verdict, applied, detail FROM notes_curation").fetchone()
    assert row[:2] == ("DUPLICATE", True) and "kept" not in row[2]
    db.close()
    _wipe(conn)


@pytest.mark.parametrize(
    ("a_tier", "b_tier", "keep_slot"),
    [
        ("work-safe", "personal", 2),  # the work-safe note is the one corrected
        ("personal", "work-safe", 2),  # the personal note is the one corrected
        ("work-safe", "personal", 1),  # the personal note is the one corrected
        ("personal", "work-safe", 1),  # the work-safe note is the one corrected
    ],
)
def test_lane_applies_a_correction_across_tiers(
    conn, db_url, monkeypatch, a_tier, b_tier, keep_slot
):
    """A correction names the note that is wrong; no text moves, so the tiers do not
    matter (the write path's "contradicts" behaves the same way)."""
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0, audience=a_tier)
    b = _insert(db, hook="Backups now run weekly on Sunday", degrees=20, audience=b_tier)
    keep, drop = (a, b) if keep_slot == 1 else (b, a)
    # The second judgement has the notes swapped, so it names the other slot.
    corr = [{"verdict": "CORRECTION", "keep": k} for k in (keep_slot, 3 - keep_slot)]
    _stub_llm(monkeypatch, pair_verdicts=corr)
    res = run_lane(db=db, llm=object(), max_retype_judge=0)
    assert res["counts"]["applied_supersede"] == 1
    assert conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (drop,)).fetchone()[0] == (
        keep
    )
    assert (
        conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (keep,)).fetchone()[0] is None
    )
    row = conn.execute("SELECT verdict, applied, detail FROM notes_curation").fetchone()
    assert row[:2] == ("CORRECTION", True) and "kept" not in row[2]
    db.close()
    _wipe(conn)


def test_lane_distinct_verdict_leaves_both_notes_live(conn, db_url, monkeypatch):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="Backups run nightly at 02:00", degrees=0)
    b = _insert(db, hook="Backups are verified every Sunday", degrees=20)
    _stub_llm(monkeypatch, pair_verdicts=[_DIS, _DIS])
    run_lane(db=db, llm=object(), max_retype_judge=0)
    live = conn.execute(
        "SELECT count(*) FROM notes WHERE superseded_by IS NULL AND id = ANY(%s)", ([a, b],)
    ).fetchone()[0]
    assert live == 2
    # ...but the pair is memoized, so it is not re-judged tomorrow.
    assert db.find_note_pair_candidates(_OWNER, sim_floor=0.60, limit=10) == []
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
    # The retype did not bump updated_at, so its own memo row is not instantly stale.
    assert [n["id"] for n in db.find_retype_candidates(_OWNER, limit=10)] == []
    assert known  # the pre-existing project note is untouched
    db.close()
    _wipe(conn)
