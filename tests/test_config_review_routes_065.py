"""DB-backed tests for what the config review routes add with schema 065: the nightly judge's
recommendation and the days a proposal has waited (mcp_server/config_sync_routes), and the
route before 065 is applied.
"""

from __future__ import annotations

from mcp_server.config_sync_routes import _proposal_detail, _proposals_list


def _seed(conn):
    return conn.execute(
        "INSERT INTO config_lane.config_proposals "
        "  (kind, file_key, scope, diff, summary, evidence, status) "
        "VALUES ('add','rules/learned.md','general','Write short answers','Write short answers',"
        '        \'[{"session_id":"s1","quote":"keep it short"}]\'::jsonb,\'proposed\') '
        "RETURNING id"
    ).fetchone()[0]


def _recommend(conn, pid, days_ago):
    conn.execute(
        "UPDATE config_lane.config_proposals SET proposed_at = now() - make_interval(days => %s), "
        'recommendation = \'{"verdict":"reject","reason":"Two quotes conflict."}\'::jsonb '
        "WHERE id=%s",
        (days_ago, pid),
    )


def test_list_and_detail_carry_the_recommendation_and_days_pending(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    _recommend(conn, pid, 9)
    row = _proposals_list(db_url)["proposals"][0]
    assert row["days_pending"] == 9
    assert row["recommendation"] == {"verdict": "reject", "reason": "Two quotes conflict."}
    assert _proposal_detail(db_url, pid)["recommendation"]["verdict"] == "reject"


def test_a_row_with_no_recommendation_yet_lists_none(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    _seed(conn)
    row = _proposals_list(db_url)["proposals"][0]
    assert row["recommendation"] is None


def test_before_schema_065_the_list_is_what_it_was(conn, db_url, monkeypatch):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    _recommend(conn, pid, 9)
    monkeypatch.setattr("mcp_server.config_sync_routes.identity_ready", lambda c: False)
    row = _proposals_list(db_url)["proposals"][0]
    assert "recommendation" not in row and "days_pending" not in row
    assert row["sessions"] == 1 and row["summary"]
    assert "recommendation" not in _proposal_detail(db_url, pid)
