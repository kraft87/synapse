"""DB-backed tests for the config-lane review routes (mcp_server/config_sync_routes proposals*).

Covers the accept/reject/apply lifecycle the review CLI drives: list shows only 'proposed' rows,
accept flips to 'accepted' and can override the blast radius to local, apply requires accepted
first (so a failed disk write stays recoverable), and reject records a reason.
"""

from __future__ import annotations

from mcp_server.config_sync_routes import _proposal_act, _proposal_detail, _proposals_list


def _seed(conn, rule="Do not use em-dashes", scope="general", status="proposed"):
    return conn.execute(
        "INSERT INTO config_lane.config_proposals "
        "  (kind, file_key, scope, diff, summary, evidence, status) "
        "VALUES ('add','rules/learned.md',%s,%s,%s,"
        '        \'[{"session_id":"s1","quote":"no dashes"}]\'::jsonb,%s) '
        "RETURNING id",
        (scope, rule, rule, status),
    ).fetchone()[0]


def test_list_shows_only_proposed(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    _seed(conn, rule="observe-only rule", status="observe")  # not yet graduated -> hidden
    rows = _proposals_list(db_url)["proposals"]
    assert [r["id"] for r in rows] == [pid]
    assert rows[0]["sessions"] == 1 and rows[0]["file_key"] == "rules/learned.md"


def test_accept_then_apply(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    acc = _proposal_act(db_url, pid, "accept", None, None)
    assert acc["status"] == "accepted" and acc["scope"] == "general"  # kept the stored scope
    assert acc["rule"] == "Do not use em-dashes" and acc["file_key"] == "rules/learned.md"
    applied = _proposal_act(db_url, pid, "apply", None, None)
    assert applied["status"] == "applied"
    assert _proposal_detail(db_url, pid)["status"] == "applied"


def test_apply_requires_accepted(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)  # still 'proposed'
    r = _proposal_act(db_url, pid, "apply", None, None)
    assert r["status"] == "refused"
    assert _proposal_detail(db_url, pid)["status"] == "proposed"  # unchanged


def test_accept_local_override(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn, scope="general")
    acc = _proposal_act(db_url, pid, "accept", None, "local")
    assert acc["scope"] == "local"
    assert _proposal_detail(db_url, pid)["scope"] == "local"


def test_reject_records_reason(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    r = _proposal_act(db_url, pid, "reject", "too vague", None)
    assert r["status"] == "rejected" and r["reason"] == "too vague"
    assert _proposal_detail(db_url, pid)["status"] == "rejected"


def test_act_missing(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    assert _proposal_act(db_url, 999999, "accept", None, None) == {"found": False}
