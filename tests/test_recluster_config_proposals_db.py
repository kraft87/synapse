"""scripts/recluster_config_proposals.py end to end against a real database.

The identity rule itself is pinned in tests/test_config_proposer*.py; what only a database
shows is that the script writes what it planned and can take it back:

* the dry run writes nothing (no row changes, no backup table);
* ``--apply`` folds each cluster into its lowest id with the union of the evidence, deletes
  the folded rows, runs the gate (proposed, or dropped as covered), copies every row it
  touches to the dated backup table first, and writes the embedding and proposed_at when
  schema 065 is applied; a second ``--apply`` changes nothing;
* it refuses to write when the rows changed between the plan and the write;
* ``--restore`` puts the backed-up rows back and leaves every other row alone.

All rules are synthetic; embeddings are unit vectors with a chosen cosine.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from dream.config import nightly as N

_SPEC = importlib.util.spec_from_file_location(
    "recluster_config_proposals",
    Path(__file__).resolve().parent.parent / "scripts" / "recluster_config_proposals.py",
)
R = importlib.util.module_from_spec(_SPEC)
sys.modules["recluster_config_proposals"] = R
_SPEC.loader.exec_module(R)

DAY = "20990101"
BAK = f"config_lane.config_proposals_bak_{DAY}"
DIMS = 2048


def vec(i, cos=1.0, k=1500):
    v = [0.0] * DIMS
    v[i] = cos
    if cos < 1.0:
        v[k] = math.sqrt(1.0 - cos * cos)
    return v


# (summary, session, embedding): A1/A2/A3 are one rule (A3 a re-scan paraphrase of A1's session),
# C1/C2 another, B stands alone, D1 and D2 are two sessions of a rule already in the rule files.
ROWS = [
    ("rule a one", "s1", vec(0)),
    ("rule a two", "s2", vec(0, 0.95)),
    ("rule a three", "s1", vec(0, 0.78, k=1501)),
    ("rule b", "s3", vec(7)),
    ("rule c one", "s4", vec(20)),
    ("rule c two", "s5", vec(20, 0.9)),
    ("rule d one", "s6", vec(30)),
    ("rule d two", "s7", vec(30, 0.92)),
]
VECS = {text: v for text, _, v in ROWS}


@pytest.fixture()
def table(conn, monkeypatch):
    def wipe():
        conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
        conn.execute("TRUNCATE config_lane.config_registry")
        conn.execute(f"DROP TABLE IF EXISTS {BAK}")

    wipe()
    ids = []
    for text, session, _ in ROWS:
        ids.append(
            conn.execute(
                "INSERT INTO config_lane.config_proposals "
                "(kind, file_key, diff, summary, evidence, status) "
                "VALUES ('add','rules/learned.md',%s,%s,%s::jsonb,'observe') RETURNING id",
                (text, text, json.dumps([{"session_id": session, "quote": f"said {text}"}])),
            ).fetchone()[0]
        )
    monkeypatch.setattr(N, "embed", lambda texts: [VECS[t] for t in texts])
    # rule d is already in the user's files: the judge says covered for it only
    conn.execute(
        "INSERT INTO config_lane.config_registry "
        "(surface_id, scope, file_key, abs_path, content, content_hash) "
        "VALUES ('m1','global','rules/a.md','/x','Rule d, already written.','h1')"
    )
    monkeypatch.setattr(
        N, "_judge_call", lambda p: json.dumps({"covered": "rule d" in p, "by": "rules/a.md"})
    )
    yield ids
    wipe()


def snapshot(conn):
    return conn.execute(
        "SELECT id, summary, status, reject_reason, evidence FROM config_lane.config_proposals "
        "ORDER BY id"
    ).fetchall()


def test_cluster_folds_by_the_lane_rule(table):
    rows = [
        {"id": i + 1, "summary": t, "evidence": [{"session_id": s}], "status": "observe"}
        for i, (t, s, _) in enumerate(ROWS)
    ]
    vecs = {i + 1: v for i, (_, _, v) in enumerate(ROWS)}
    clusters = R.cluster(rows, vecs)
    assert R.signature(clusters) == [[1, 2, 3], [4], [5, 6], [7, 8]]
    first = clusters[0]
    assert first["id"] == 1 and N._distinct_sessions(first["evidence"]) == 2  # s1, s2


def test_dry_run_writes_nothing(conn, db_url, table):
    before = snapshot(conn)
    with R.psycopg.connect(db_url) as c:
        c.read_only = True
        p, _, _ = R.plan(c, use_llm=True)
        c.rollback()
    assert snapshot(conn) == before
    assert conn.execute(f"SELECT to_regclass('{BAK}')").fetchone()[0] is None
    s = p["summary"]
    assert s["rows_before"] == 8 and s["clusters"] == 4 and s["rows_merged_away"] == 4
    assert s["clusters_with_2plus_sessions"] == 3
    assert s["proposed"] == 2 and s["dropped_as_covered"] == 1
    # ids and counts only: no rule text anywhere in the plan
    assert "rule " not in json.dumps(p)


def test_apply_folds_gates_and_backs_up(conn, db_url, table):
    ids = table
    assert R.apply(db_url, use_llm=True, day=DAY, plan_out=None) == 0
    rows = {r[0]: r for r in snapshot(conn)}
    assert sorted(rows) == [ids[0], ids[3], ids[4], ids[6]]  # survivors: the lowest ids
    a, b, c, d = (rows[ids[0]], rows[ids[3]], rows[ids[4]], rows[ids[6]])
    assert a[2] == "proposed" and {e["session_id"] for e in a[4]} == {"s1", "s2"}
    assert a[1] == "rule a one"  # the founding wording stays
    assert b[2] == "observe" and len(b[4]) == 1
    assert c[2] == "proposed" and len(c[4]) == 2
    assert (d[2], d[3]) == ("rejected", "auto:covered")
    extra = conn.execute(
        "SELECT id, proposed_at IS NOT NULL, embedding IS NOT NULL, coverage->>'covered' "
        "FROM config_lane.config_proposals ORDER BY id"
    ).fetchall()
    assert extra == [
        (ids[0], True, True, "false"),
        (ids[3], False, False, None),  # untouched singleton: the nightly run anchors it
        (ids[4], True, True, "false"),
        (ids[6], False, True, "true"),
    ]
    # every touched row is in the backup, as it was
    assert conn.execute(f"SELECT count(*) FROM {BAK}").fetchone()[0] == 7
    assert (
        conn.execute(f"SELECT status FROM {BAK} WHERE id = %s", (ids[1],)).fetchone()[0]
        == "observe"
    )


def test_a_second_apply_has_nothing_to_do(conn, db_url, table, capsys):
    R.apply(db_url, use_llm=True, day=DAY, plan_out=None)
    after = snapshot(conn)
    assert R.apply(db_url, use_llm=True, day="20990102", plan_out=None) == 0
    assert "nothing to do." in capsys.readouterr().out
    assert snapshot(conn) == after
    assert (
        conn.execute("SELECT to_regclass('config_lane.config_proposals_bak_20990102')").fetchone()[
            0
        ]
        is None
    )


def test_no_llm_leaves_the_graduates_at_observe(conn, db_url, table):
    R.apply(db_url, use_llm=False, day=DAY, plan_out=None)
    statuses = [r[2] for r in snapshot(conn)]
    assert statuses == ["observe"] * 4  # merged, evidence unioned, gate left to the nightly run
    assert (
        conn.execute(
            "SELECT max(jsonb_array_length(evidence)) FROM config_lane.config_proposals"
        ).fetchone()[0]
        == 2
    )


def test_apply_refuses_when_a_row_appears_after_the_plan(conn, db_url, table, monkeypatch):
    real = N.embed

    def embed_then_a_row_arrives(texts):
        out = real(texts)
        conn.execute(
            "INSERT INTO config_lane.config_proposals (kind, file_key, diff, summary, evidence) "
            "VALUES ('add','rules/learned.md','late','late','[{\"session_id\":\"s9\"}]')"
        )
        return out

    monkeypatch.setattr(N, "embed", embed_then_a_row_arrives)
    before = snapshot(conn)
    with pytest.raises(SystemExit, match="appeared since the plan"):
        R.apply(db_url, use_llm=True, day=DAY, plan_out=None)
    assert [r for r in snapshot(conn) if r[1] != "late"] == [r for r in before if r[1] != "late"]
    assert conn.execute(f"SELECT to_regclass('{BAK}')").fetchone()[0] is None


def test_without_schema_065_only_existing_columns_are_written(conn, db_url, table, monkeypatch):
    monkeypatch.setattr(R, "identity_ready", lambda c: False)
    ids = table
    R.apply(db_url, use_llm=True, day=DAY, plan_out=None)
    row = conn.execute(
        "SELECT status, embedding IS NULL, proposed_at IS NULL, coverage IS NULL "
        "FROM config_lane.config_proposals WHERE id=%s",
        (ids[0],),
    ).fetchone()
    assert row == ("proposed", True, True, True)  # 065's own UPDATE stamps proposed_at later


def test_restore_puts_the_rows_back_and_leaves_others_alone(conn, db_url, table, capsys):
    ids = table
    original = snapshot(conn)
    R.apply(db_url, use_llm=True, day=DAY, plan_out=None)
    conn.execute(
        "INSERT INTO config_lane.config_proposals (kind, file_key, diff, summary, evidence) "
        "VALUES ('add','rules/learned.md','new','new rule','[{\"session_id\":\"s9\"}]')"
    )
    folded = snapshot(conn)
    # dry run: reports, writes nothing
    assert R.restore(db_url, DAY, do_apply=False) == 0
    assert snapshot(conn) == folded and "dry run" in capsys.readouterr().out
    assert R.restore(db_url, DAY, do_apply=True) == 0
    after = snapshot(conn)
    assert [r for r in after if r[0] in ids] == original  # status, evidence, ids as they were
    assert [r[1] for r in after if r[0] not in ids] == ["new rule"]  # the newcomer is untouched


def test_restore_without_a_backup_is_an_error(db_url, table):
    with pytest.raises(SystemExit, match="does not exist"):
        R.restore(db_url, "20980101", do_apply=True)


def test_plan_out_must_live_outside_the_repo():
    with pytest.raises(SystemExit, match="outside the repository"):
        R._check_path(str(Path(R.__file__).resolve().parent / "plan.json"))
    R._check_path("/tmp/plan.json")
    R._check_path(None)
