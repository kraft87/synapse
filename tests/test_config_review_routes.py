"""DB-backed tests for the config-lane review routes (mcp_server/config_sync_routes proposals*).

Covers the accept/reject/apply lifecycle the review CLI and the dashboard drive: list shows the
rows awaiting a decision ('proposed') or a disk write ('accepted', e.g. approved on the
dashboard), accept flips to 'accepted' and can override the blast radius to local, apply requires
accepted first (so a failed disk write stays recoverable), reject records a reason, and a
decision once made (applied, rejected) is final.
"""

from __future__ import annotations

import pytest

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


def test_list_shows_rows_awaiting_a_decision_or_a_write(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    proposed = _seed(conn)
    approved = _seed(conn, rule="approved on the dashboard", status="accepted")
    for status in ("observe", "applied", "rejected"):  # nothing left to do for these
        _seed(conn, rule=f"{status} rule", status=status)
    rows = {r["id"]: r for r in _proposals_list(db_url)["proposals"]}
    assert set(rows) == {proposed, approved}
    assert rows[proposed]["sessions"] == 1 and rows[proposed]["file_key"] == "rules/learned.md"
    assert (rows[proposed]["status"], rows[approved]["status"]) == ("proposed", "accepted")
    assert rows[approved]["rule"] == "approved on the dashboard"  # what `apply` writes


def test_accept_then_apply(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn)
    acc = _proposal_act(db_url, pid, "accept", None, None)
    assert acc["status"] == "accepted" and acc["scope"] == "general"  # kept the stored scope
    assert acc["rule"] == "Do not use em-dashes" and acc["file_key"] == "rules/learned.md"
    applied = _proposal_act(db_url, pid, "apply", None, None)
    assert applied["status"] == "applied"
    assert _proposal_detail(db_url, pid)["status"] == "applied"


@pytest.mark.parametrize("final", ["applied", "rejected"])
@pytest.mark.parametrize("action", ["accept", "reject"])
def test_a_final_decision_cannot_be_redone(conn, db_url, final, action):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn, status=final)
    stamp = conn.execute(
        "SELECT updated_at FROM config_lane.config_proposals WHERE id=%s", (pid,)
    ).fetchone()[0]
    r = _proposal_act(db_url, pid, action, "late click", None)
    assert r["status"] == "refused" and final in r["detail"]
    row = conn.execute(
        "SELECT status, updated_at, reject_reason FROM config_lane.config_proposals WHERE id=%s",
        (pid,),
    ).fetchone()
    assert row == (final, stamp, None)  # untouched: a rejection's date still anchors its cooling


def test_an_approved_proposal_can_be_accepted_again_to_finish_it(conn, db_url):
    # the dashboard approved it; `config_review accept <id>` then writes it and marks it applied
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    pid = _seed(conn, status="accepted")
    again = _proposal_act(db_url, pid, "accept", None, None)
    assert again["status"] == "accepted" and again["rule"] == "Do not use em-dashes"
    assert _proposal_act(db_url, pid, "apply", None, None)["status"] == "applied"


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
