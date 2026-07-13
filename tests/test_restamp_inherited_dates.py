"""Tests for scripts/restamp_inherited_dates.py — the issue #44 one-time repair.

Covers both lanes' selection rules: facts restamp only non-midnight (not
evidence-dated) edges disagreeing >1 day with the recomputed episode max,
syncing t_valid AND valid_at; timeline restamps only non-noon (not
LLM-resolved) chat events; dry-run mutates nothing; re-runs are idempotent.
Skips cleanly when no test DB is reachable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

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

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "restamp_script", _REPO / "scripts" / "restamp_inherited_dates.py"
)
assert _spec and _spec.loader
restamp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(restamp)

# Corrected (true) episode time vs the import-day stamp the edge inherited.
_TRUE_TS = "2026-03-02T15:41:07+00:00"
_TRUE_TS_2 = "2026-03-02T16:03:55+00:00"  # later turn in the same segment
_IMPORT_TS = "2026-06-05T09:30:12+00:00"  # non-midnight: inherited default shape
_EVIDENCE_TS = "2026-04-10T00:00:00+00:00"  # midnight: EdgeDateExtractor shape
_NOON_TS = "2026-04-10T12:00:00+00:00"  # noon: timeline LLM-resolved shape


@pytest.fixture()
def db(conn):
    """Rows created through this fixture's helpers are deleted on teardown."""
    made: dict[str, list] = {"episodes": [], "edges": [], "events": []}

    def episode(created_at: str) -> int:
        eid = conn.execute(
            "INSERT INTO episodes (session_id, sequence, content, created_at) "
            "VALUES (%s, 1, 'restamp test turn', %s) RETURNING id",
            (f"restamp-{uuid.uuid4().hex[:12]}", created_at),
        ).fetchone()[0]
        made["episodes"].append(eid)
        return eid

    def edge(t_valid: str, episode_ids: list[int] | None, mention_count: int = 1) -> int:
        rid = conn.execute(
            "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, fact, "
            "episodes, t_valid, valid_at, mention_count) "
            "VALUES (%s, 'technical', 'restamp-src', 'restamp-tgt', 'restamp test fact', "
            "%s, %s, %s, %s) RETURNING id",
            (
                f"restamp-{uuid.uuid4().hex}",
                json.dumps(episode_ids) if episode_ids is not None else None,
                t_valid,
                t_valid,
                mention_count,
            ),
        ).fetchone()[0]
        made["edges"].append(rid)
        return rid

    def event(t_valid: str, source_ref: str, source: str = "chat") -> int:
        tid = conn.execute(
            "INSERT INTO timeline_events (t_valid, fact, source, source_ref) "
            "VALUES (%s, 'restamp test event', %s, %s) RETURNING id",
            (t_valid, source, source_ref),
        ).fetchone()[0]
        made["events"].append(tid)
        return tid

    yield conn, episode, edge, event

    conn.execute("DELETE FROM timeline_events WHERE id = ANY(%s)", (made["events"],))
    conn.execute("DELETE FROM kg_relationships WHERE id = ANY(%s)", (made["edges"],))
    conn.execute("DELETE FROM episodes WHERE id = ANY(%s)", (made["episodes"],))


def _edge_times(conn, rid: int):
    return conn.execute(
        "SELECT t_valid, valid_at FROM kg_relationships WHERE id = %s", (rid,)
    ).fetchone()


def _event_time(conn, tid: int):
    return conn.execute("SELECT t_valid FROM timeline_events WHERE id = %s", (tid,)).fetchone()[0]


# ---------------------------------------------------------------------------
# Facts lane
# ---------------------------------------------------------------------------


def test_facts_inherited_edge_restamped_to_episode_max_and_valid_at_synced(db):
    conn, episode, edge, _ = db
    e1, e2 = episode(_TRUE_TS), episode(_TRUE_TS_2)
    rid = edge(_IMPORT_TS, [e1, e2])

    restamp.repair_facts(conn, apply=True)

    t_valid, valid_at = _edge_times(conn, rid)
    assert t_valid.isoformat() == "2026-03-02T16:03:55+00:00"  # max of the segment
    assert valid_at == t_valid  # insert-path invariant kept


def test_facts_midnight_evidence_edge_untouched(db):
    conn, episode, edge, _ = db
    rid = edge(_EVIDENCE_TS, [episode(_TRUE_TS)])

    restamp.repair_facts(conn, apply=True)

    t_valid, _ = _edge_times(conn, rid)
    assert t_valid.isoformat() == _EVIDENCE_TS.replace("+00:00", "+00:00")


def test_facts_agreeing_and_backlinkless_edges_untouched(db):
    conn, episode, edge, _ = db
    agreeing = edge(_TRUE_TS_2, [episode(_TRUE_TS), episode(_TRUE_TS_2)])
    no_links = edge(_IMPORT_TS, None)

    restamp.repair_facts(conn, apply=True)

    assert _edge_times(conn, agreeing)[0].isoformat() == "2026-03-02T16:03:55+00:00"
    assert _edge_times(conn, no_links)[0].isoformat() == "2026-06-05T09:30:12+00:00"


def test_facts_dry_run_mutates_nothing_and_apply_is_idempotent(db):
    conn, episode, edge, _ = db
    rid = edge(_IMPORT_TS, [episode(_TRUE_TS)])

    restamp.repair_facts(conn, apply=False)
    assert _edge_times(conn, rid)[0].isoformat() == "2026-06-05T09:30:12+00:00"

    restamp.repair_facts(conn, apply=True)
    first = _edge_times(conn, rid)
    restamp.repair_facts(conn, apply=True)
    assert _edge_times(conn, rid) == first


# ---------------------------------------------------------------------------
# Timeline lane
# ---------------------------------------------------------------------------


def test_timeline_inherited_event_restamped_noon_event_untouched(db):
    conn, episode, _, event = db
    # (source, source_ref) is unique — one event per episode ref.
    inherited = event(_IMPORT_TS, f"ep:{episode(_TRUE_TS)}")
    resolved = event(_NOON_TS, f"ep:{episode(_TRUE_TS)}")

    restamp.repair_timeline(conn, apply=True)

    assert _event_time(conn, inherited).isoformat() == "2026-03-02T15:41:07+00:00"
    assert _event_time(conn, resolved).isoformat() == "2026-04-10T12:00:00+00:00"


def test_timeline_within_a_day_and_dry_run_untouched(db):
    conn, episode, _, event = db
    # <1 day off: a plausible turn ts, left alone
    close = event("2026-03-02T18:00:00+00:00", f"ep:{episode(_TRUE_TS)}")
    inherited = event(_IMPORT_TS, f"ep:{episode(_TRUE_TS)}")

    restamp.repair_timeline(conn, apply=False)

    assert _event_time(conn, close).isoformat() == "2026-03-02T18:00:00+00:00"
    assert _event_time(conn, inherited).isoformat() == "2026-06-05T09:30:12+00:00"
