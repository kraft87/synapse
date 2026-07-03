"""Read-time identifier-collapse for the topical timeline view (mcp_server/timeline.py)."""

from __future__ import annotations

from mcp_server.timeline import _collapse_idents


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
