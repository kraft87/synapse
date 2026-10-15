"""Pure-logic tests for the dream->config proposer ledger (dream/config/nightly).

No DB, no LLM. Guards the identity + evidence-accumulation rules that decide when a recurring
behavioral correction graduates observe -> proposed: same-rule matching by token jaccard, evidence
deduped by session, and the distinct-session propose gate.
"""

from __future__ import annotations

import pytest

from dream.config.nightly import (
    PROPOSE_SESSIONS,
    _best_match,
    _distinct_sessions,
    _rule,
    _same_rule,
    _sig_tokens,
)
from dream.evidence import EVIDENCE_CAP, jaccard, union_evidence


def test_sig_tokens_drops_stopwords():
    toks = _sig_tokens("Do not use em-dashes in prose")
    assert "em" in toks and "dashes" in toks and "prose" in toks
    assert "do" not in toks and "not" not in toks and "in" not in toks  # stopwords gone


def test_jaccard_same_rule_high():
    a = _sig_tokens("Do not use em-dashes in drafts")
    b = _sig_tokens("Avoid em-dashes when writing drafts")
    assert jaccard(a, b) >= 0.4  # shared em/dashes/drafts
    c = _sig_tokens("Filter out contract job postings")
    assert jaccard(a, c) == 0.0  # unrelated rules don't collide


def test_union_evidence_dedups_by_session():
    old = [{"session_id": "s1", "quote": "x"}]
    new = [{"session_id": "s1", "quote": "y"}, {"session_id": "s2", "quote": "z"}]
    merged = union_evidence(old, new)
    assert _distinct_sessions(merged) == 2  # s1 not double-counted
    assert len(merged) == 2


def test_union_evidence_is_capped_like_the_skills_ledgers():
    old = [{"session_id": f"s{i}", "quote": "x"} for i in range(EVIDENCE_CAP)]
    merged = union_evidence(old, [{"session_id": "newest", "quote": "y"}])
    assert len(merged) == EVIDENCE_CAP
    assert merged[0]["session_id"] == "s1" and merged[-1]["session_id"] == "newest"  # oldest went


def test_a_rule_of_only_stop_words_is_not_the_same_as_another():
    assert not _same_rule(_rule("do not"), _rule("never ever"))
    assert _best_match(_rule("do not"), [{"summary": "never ever"}]) is None


def test_best_match_finds_same_rule_only():
    cands = [
        {"summary": "Do not use em-dashes in prose"},
        {"summary": "Filter out RBC job postings"},
    ]
    # a re-phrasing of the same correction matches (shared em/dashes/prose); a distinct rule doesn't
    m = _best_match(_rule("Avoid em-dashes in prose"), cands)
    assert m is not None and "em-dashes" in m["summary"]
    assert _best_match(_rule("always run the test suite first"), cands) is None


def test_a_rule_and_its_reversal_are_not_the_same_rule():
    # not/never/dont are stop words, so the tokens alone say these are identical
    assert (
        jaccard(
            _sig_tokens("Run the tests before committing"),
            _sig_tokens("Do not run the tests before committing"),
        )
        == 1.0
    )
    assert not _same_rule(
        _rule("Run the tests before committing"), _rule("Do not run the tests before committing")
    )
    cands = [{"summary": "Run the tests before committing"}]
    assert _best_match(_rule("Don't run the tests before committing"), cands) is None
    assert _best_match(_rule("Run the tests before you commit"), cands) is not None


@pytest.mark.parametrize(
    "a, b",
    [
        ("Never use em-dashes in prose", "Avoid em-dashes in prose"),
        ("Do not use em-dashes in prose", "Don\u2019t use em-dashes in prose"),
        ("Do not push without asking", "Never push without asking"),
        ("Keep answers short", "Keep your answers short"),
    ],
)
def test_phrasings_of_one_polarity_are_the_same_rule(a, b):
    assert _same_rule(_rule(a), _rule(b))


def test_propose_gate_is_distinct_sessions():
    # the gate the orchestrator applies: status flips to proposed at PROPOSE_SESSIONS distinct sessions
    ev = [{"session_id": f"s{i}"} for i in range(PROPOSE_SESSIONS)]
    assert _distinct_sessions(ev) >= PROPOSE_SESSIONS
    assert _distinct_sessions(ev[: PROPOSE_SESSIONS - 1]) < PROPOSE_SESSIONS
