"""Pure-logic tests for the dream->config proposer ledger (dream/config/nightly).

No DB, no LLM. Guards the identity + evidence-accumulation rules that decide when a recurring
behavioral correction graduates observe -> proposed: same-rule matching by token jaccard, evidence
deduped by session, and the distinct-session propose gate.
"""

from __future__ import annotations

from dream.config.nightly import (
    PROPOSE_SESSIONS,
    _best_match,
    _distinct_sessions,
    _jaccard,
    _sig_tokens,
    _union_evidence,
)


def test_sig_tokens_drops_stopwords():
    toks = _sig_tokens("Do not use em-dashes in prose")
    assert "em" in toks and "dashes" in toks and "prose" in toks
    assert "do" not in toks and "not" not in toks and "in" not in toks  # stopwords gone


def test_jaccard_same_rule_high():
    a = _sig_tokens("Do not use em-dashes in drafts")
    b = _sig_tokens("Avoid em-dashes when writing drafts")
    assert _jaccard(a, b) >= 0.4  # shared em/dashes/drafts
    c = _sig_tokens("Filter out contract job postings")
    assert _jaccard(a, c) == 0.0  # unrelated rules don't collide


def test_union_evidence_dedups_by_session():
    old = [{"session_id": "s1", "quote": "x"}]
    new = [{"session_id": "s1", "quote": "y"}, {"session_id": "s2", "quote": "z"}]
    merged = _union_evidence(old, new)
    assert _distinct_sessions(merged) == 2  # s1 not double-counted
    assert len(merged) == 2


def test_best_match_finds_same_rule_only():
    cands = [
        {"summary": "Do not use em-dashes in prose"},
        {"summary": "Filter out RBC job postings"},
    ]
    # a re-phrasing of the same correction matches (shared em/dashes/prose); a distinct rule doesn't
    m = _best_match(_sig_tokens("Avoid em-dashes in prose"), cands)
    assert m is not None and "em-dashes" in m["summary"]
    assert _best_match(_sig_tokens("always run the test suite first"), cands) is None


def test_propose_gate_is_distinct_sessions():
    # the gate the orchestrator applies: status flips to proposed at PROPOSE_SESSIONS distinct sessions
    ev = [{"session_id": f"s{i}"} for i in range(PROPOSE_SESSIONS)]
    assert _distinct_sessions(ev) >= PROPOSE_SESSIONS
    assert _distinct_sessions(ev[: PROPOSE_SESSIONS - 1]) < PROPOSE_SESSIONS
