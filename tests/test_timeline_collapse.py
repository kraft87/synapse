"""Read-time identifier-collapse for the topical timeline view (mcp_server/timeline.py)."""

from __future__ import annotations

from mcp_server.timeline import _collapse_idents, _filters


def _e(fact, date="2026-07-01"):
    return {
        "kind": "event",
        "t_valid": date,
        "fact": fact,
        "salience": 1,
        "project": "synapse",
        "source": "chat",
        "source_ref": "x",
        "event_type": None,
    }


def test_same_ident_folds_keeping_richest():
    items = [
        _e("merged PR #103 to main"),
        _e("shipped PR #103 (history query optimization + parallel recall legs) to production"),
        _e("decided episodic memory lives in its own store"),
    ]
    out = _collapse_idents(items)
    assert len(out) == 2
    assert out[0]["fact"].startswith("shipped PR #103")  # richest survives
    assert out[0]["folded"] == 2
    assert out[1]["fact"].startswith("decided")


def test_distinct_idents_stay():
    out = _collapse_idents(
        [_e("committed to synapse: fix #99"), _e("committed to synapse: fix #100")]
    )
    assert len(out) == 2


# ---- domain scoping (schema 038, issue #17) ----


def test_personal_scope_adds_domain_clause():
    clauses, params = _filters(None, None, None, 0, group_id="personal")
    assert clauses == ["(domain = 'personal' OR domain IS NULL)"]  # NULL fails open
    assert params == []


def test_default_and_technical_scope_stay_unfiltered():
    # Most callers never set group_id; filtering the default would hide personal
    # events from them — only an EXPLICIT personal scope filters.
    assert _filters(None, None, None, 0, group_id=None)[0] == []
    assert _filters(None, None, None, 0, group_id="technical")[0] == []


def test_personal_scope_kill_switch(monkeypatch):
    monkeypatch.setenv("SYNAPSE_TIMELINE_GROUP_SCOPE", "0")
    assert _filters(None, None, None, 0, group_id="personal")[0] == []


def test_domain_clause_composes_with_other_filters():
    clauses, params = _filters("2026-01-01", None, "neuron", 1, group_id="personal")
    assert clauses == [
        "t_valid >= %s",
        "project = %s",
        "salience >= %s",
        "(domain = 'personal' OR domain IS NULL)",
    ]
    assert params == ["2026-01-01", "neuron", 1]
