"""DB-backed tests for the recall_metrics write path (mcp_server/recall.py::_do_record).

Covers issue #10 (served_ids JSONB round-trips per bucket) and the schema-039 repair:
n_timeline / ms_timeline / n_prefs / ms_prefs were passed by the writer since the
033/035 legs shipped but silently dropped because _METRIC_COLS never gained them.
"""

from __future__ import annotations

from mcp_server.recall import Recall


def test_served_ids_and_leg_columns_round_trip(conn, db_url):
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    served = {
        "episodes": ["e:12", "e:7"],
        "facts": ["uuid-a", "uuid-b"],
        "timeline": [3, 9],
        "prefs": [1],
    }
    r._do_record(
        {
            "kind": "recall",
            "source": "test",
            "query": "q",
            "group_id": "technical",
            "n_timeline": 2,
            "ms_timeline": 12.5,
            "n_prefs": 1,
            "ms_prefs": 3.0,
            "served_ids": served,
        }
    )
    row = conn.execute(
        "SELECT served_ids, n_timeline, ms_timeline, n_prefs, ms_prefs "
        "FROM recall_metrics WHERE source = 'test'"
    ).fetchone()
    assert row is not None, "metrics row was not inserted (writer swallowed an error)"
    assert row[0] == served
    assert row[1] == 2 and row[2] == 12.5
    assert row[3] == 1 and row[4] == 3.0


def test_missing_served_ids_inserts_null(conn, db_url):
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    r._do_record({"kind": "episodes", "source": "test", "query": "q"})
    row = conn.execute("SELECT served_ids FROM recall_metrics WHERE source = 'test'").fetchone()
    assert row is not None
    assert row[0] is None


def test_served_ids_carries_echo_suppression_count(conn, db_url):
    # Query-echo suppression records n_echo_suppressed inside the served_ids envelope (no DDL).
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    served = {"episodes": ["e:5"], "facts": [], "n_echo_suppressed": 3}
    r._do_record({"kind": "recall", "source": "test", "query": "q", "served_ids": served})
    row = conn.execute("SELECT served_ids FROM recall_metrics WHERE source = 'test'").fetchone()
    assert row is not None
    assert row[0] == served
    assert row[0]["n_echo_suppressed"] == 3
