"""Pure-logic tests for the dream->config proposer ledger (dream/config/nightly).

No DB, no LLM. Guards the identity + evidence-accumulation rules that decide when a recurring
behavioral correction graduates observe -> proposed: same-rule matching by embedding cosine
(a higher bar across sessions than inside one), how a stored row answers a re-sighting by its
status, evidence deduped by session, and the distinct-session propose gate.
"""

from __future__ import annotations

from dream.config.nightly import (
    COOLDOWN_DAYS,
    EXPIRED,
    IDENTITY_COSINE,
    IDENTITY_COSINE_SAME_SESSION,
    PROPOSE_SESSIONS,
    REOPEN_EXPIRED_DAYS,
    _distinct_sessions,
    _evidence_key,
    _union_evidence,
    pick_match,
    row_class,
)


def _row(rid, cos, sessions=("s1",), status="observe", reason=None, age=0.0):
    return {
        "id": rid,
        "cos": cos,
        "evidence": [{"session_id": s} for s in sessions],
        "status": status,
        "reject_reason": reason,
        "age_days": age,
    }


def test_thresholds_order():
    # a session the row already counts is held to the lower bar (re-scan paraphrases)
    assert 0.5 < IDENTITY_COSINE_SAME_SESSION < IDENTITY_COSINE < 1.0


def test_union_evidence_dedups_by_session():
    old = [{"session_id": "s1", "quote": "x"}]
    new = [{"session_id": "s1", "quote": "y"}, {"session_id": "s2", "quote": "z"}]
    merged = _union_evidence(old, new)
    assert _distinct_sessions(merged) == 2  # s1 not double-counted
    assert len(merged) == 2


def test_cross_session_needs_the_high_bar():
    rows = [_row(1, IDENTITY_COSINE + 0.01), _row(2, IDENTITY_COSINE - 0.02)]
    hit = pick_match(rows, "new-session")
    assert hit is not None and hit[0]["id"] == 1
    assert pick_match([rows[1]], "new-session") is None  # just under the bar: its own rule


def test_same_session_paraphrase_clears_the_lower_bar():
    mid = (IDENTITY_COSINE + IDENTITY_COSINE_SAME_SESSION) / 2
    row = _row(1, mid, sessions=("s1",))
    assert pick_match([row], "s1") is not None  # the row already counts s1
    assert pick_match([row], "s2") is None  # the same cosine from another session does not
    below = _row(2, IDENTITY_COSINE_SAME_SESSION - 0.02, sessions=("s1",))
    assert pick_match([below], "s1") is None  # a different rule from the same session


def test_best_cosine_wins_among_matches():
    rows = [_row(1, 0.86), _row(2, 0.93), _row(3, 0.84)]
    assert pick_match(rows, "s9")[0]["id"] == 2


def test_rows_without_an_anchor_never_match():
    assert pick_match([_row(1, None)], "s9") is None


def test_row_class_by_status():
    assert row_class(_row(1, 1, status="observe")) == "open"
    assert row_class(_row(1, 1, status="proposed")) == "open"
    assert row_class(_row(1, 1, status="accepted")) == "held"
    assert row_class(_row(1, 1, status="applied")) == "held"
    # a rejection (a person's or auto:covered) is a cooldown that absorbs sightings
    fresh = _row(1, 1, status="rejected", reason="user_rejected", age=COOLDOWN_DAYS - 1)
    old = _row(1, 1, status="rejected", reason="user_rejected", age=COOLDOWN_DAYS + 1)
    assert row_class(fresh) == "held" and row_class(old) is None
    covered = _row(1, 1, status="rejected", reason="auto:covered", age=3)
    assert row_class(covered) == "held"
    # an expired proposal reopens for a while
    lapsed = _row(1, 1, status="rejected", reason=EXPIRED, age=REOPEN_EXPIRED_DAYS - 1)
    gone = _row(1, 1, status="rejected", reason=EXPIRED, age=REOPEN_EXPIRED_DAYS + 1)
    assert row_class(lapsed) == "reopen" and row_class(gone) is None


def test_old_rejection_is_not_matched():
    rows = [_row(1, 0.95, status="rejected", reason="not for me", age=COOLDOWN_DAYS + 5)]
    assert pick_match(rows, "s9") is None  # a fresh row starts


def test_evidence_key_tracks_rule_and_sessions():
    a = _evidence_key("rule", [{"session_id": "s1"}, {"session_id": "s2"}])
    assert a == _evidence_key("rule", [{"session_id": "s2"}, {"session_id": "s1"}])
    assert a != _evidence_key("rule", [{"session_id": "s1"}])  # a new session: ask again
    assert a != _evidence_key("other rule", [{"session_id": "s1"}, {"session_id": "s2"}])


def test_propose_gate_is_distinct_sessions():
    # the gate regate applies: status flips to proposed at PROPOSE_SESSIONS distinct sessions
    ev = [{"session_id": f"s{i}"} for i in range(PROPOSE_SESSIONS)]
    assert _distinct_sessions(ev) >= PROPOSE_SESSIONS
    assert _distinct_sessions(ev[: PROPOSE_SESSIONS - 1]) < PROPOSE_SESSIONS
