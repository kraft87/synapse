"""Session-diversity cap on the served passage bucket (recall._compact_to_passages).

Guards the fix for self/recency domination: when the live session's freshly-ingested turns
crowd the top of the ranking, at most SYNAPSE_RECALL_SESSION_CAP of the n served passages may
share a session_id. Pure-logic tests — no DB, no Voyage (fake reranker returns a fixed order).

Contract:
  - cap trims a session that would otherwise monopolise the served bucket (the bug),
  - it backfills from lower-ranked passages so a genuinely single-session pool still serves n
    (diversity-trimming must never cost recall),
  - =0 disables it (old top-n behaviour),
  - episodes lacking a session_id are never capped (unattributable -> don't penalise).
"""

from __future__ import annotations

from mcp_server.recall import Recall


class _FakeEmb:
    """rerank_scored that returns a caller-specified passage order (most->least relevant)."""

    def __init__(self, order: list[int]) -> None:
        self._order = order

    def rerank_scored(self, query, documents, top_k=None):
        idx = [i for i in self._order if i < len(documents)]
        idx += [i for i in range(len(documents)) if i not in idx]
        scored = [(i, 1.0 - 0.01 * r) for r, i in enumerate(idx)]
        return scored[: (top_k or len(documents))]


def _ep(content: str, sid: str | None, eid: str, project: str = "synapse") -> dict:
    # short content -> exactly one markdown chunk -> one passage, so owner<->session is 1:1
    return {
        "content": content,
        "session_id": sid,
        "id": eid,
        "created_at": "2026-06-01T12:00:00+00:00",
        "project": project,
    }


def _ids(out: list[dict]) -> set[str]:
    return {it["id"] for it in out}


# Three passages from session S1 rank highest, one from S2 last. Uncapped -> all three served
# are S1 (the domination bug). Four passages > n=3 so the rerank/cap path actually runs.
_EPS = [
    _ep("alpha one", "S1", "e:1"),
    _ep("alpha two", "S1", "e:2"),
    _ep("alpha three", "S1", "e:3"),
    _ep("bravo one", "S2", "e:4"),
]


def test_cap_breaks_single_session_monopoly(monkeypatch):
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])  # S1,S1,S1,S2
    out = r._compact_to_passages("q", _EPS, n=3)
    assert len(out) == 3
    served = _ids(out)
    assert "e:4" in served  # the other session was pulled into the served bucket
    assert "e:3" not in served  # the 3rd S1 passage was capped out
    assert len({"e:1", "e:2", "e:3"} & served) == 2  # at most cap=2 from S1


def test_cap_off_serves_top_n(monkeypatch):
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "0")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])
    out = r._compact_to_passages("q", _EPS, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:3"}  # pure top-3, one session, no cap


def test_backfill_when_pool_is_one_session(monkeypatch):
    # Every episode is S1: the cap must NOT reduce the served count below n — it backfills.
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    eps = [_ep(f"alpha {w}", "S1", f"e:{i}") for i, w in enumerate(["one", "two", "three", "four"])]
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])
    out = r._compact_to_passages("q", eps, n=3)
    assert len(out) == 3  # backfilled the 3rd S1 passage rather than serving 2


def test_null_session_never_capped(monkeypatch):
    # Episodes with no session_id are unattributable -> the cap ignores them, serves top-n.
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    eps = [_ep(f"alpha {w}", None, f"e:{i}") for i, w in enumerate(["one", "two", "three", "four"])]
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])
    out = r._compact_to_passages("q", eps, n=3)
    assert _ids(out) == {"e:0", "e:1", "e:2"}  # top-3, no cap applied on null sessions


class _FakeScoredEmb:
    """rerank_scored with caller-specified absolute scores (for slack-gate tests)."""

    def __init__(self, scored: list[tuple[int, float]]) -> None:
        self._scored = scored

    def rerank_scored(self, query, documents, top_k=None):
        return self._scored[: (top_k or len(self._scored))]


def test_slack_keeps_dominant_session_when_alternative_is_weak(monkeypatch):
    # Third S1 passage outscores the best S2 alternative by 0.43 > slack=0.1:
    # forcing diversity would cost real relevance, so S1 keeps the slot.
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP_SLACK", "0.1")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    r = Recall("", "")
    r._reranker = _FakeScoredEmb([(0, 0.95), (1, 0.94), (2, 0.93), (3, 0.50)])
    out = r._compact_to_passages("q", _EPS, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:3"}


def test_slack_defers_to_diversity_on_near_tie(monkeypatch):
    # Best S2 alternative is within slack of the third S1 passage: diversity wins.
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP_SLACK", "0.1")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    r = Recall("", "")
    r._reranker = _FakeScoredEmb([(0, 0.95), (1, 0.94), (2, 0.93), (3, 0.90)])
    out = r._compact_to_passages("q", _EPS, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:4"}


def test_slack_zero_is_hard_cap(monkeypatch):
    # Default slack=0 preserves the shipped hard-cap behavior even with a huge score gap.
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP_SLACK", "0")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    r = Recall("", "")
    r._reranker = _FakeScoredEmb([(0, 0.95), (1, 0.94), (2, 0.93), (3, 0.10)])
    out = r._compact_to_passages("q", _EPS, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:4"}


def test_fresh_scope_exempts_stale_sessions(monkeypatch):
    # FRESH_H=48: _EPS timestamps (2026-06-01) are stale -> the cap never fires,
    # a stale session may monopolize the bucket (LME multi-session evidence case).
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP_FRESH_H", "48")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    monkeypatch.delenv("SYNAPSE_RECALL_SESSION_CAP_SLACK", raising=False)
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])
    out = r._compact_to_passages("q", _EPS, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:3"}  # all S1, uncapped because stale


def test_fresh_scope_still_caps_live_session(monkeypatch):
    # FRESH_H=48 with a just-ingested session: the live-session domination fix stays.
    from datetime import UTC, datetime

    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP", "2")
    monkeypatch.setenv("SYNAPSE_RECALL_SESSION_CAP_FRESH_H", "48")
    monkeypatch.delenv("SYNAPSE_PASSAGE_QUOTA", raising=False)
    monkeypatch.delenv("SYNAPSE_RECALL_SESSION_CAP_SLACK", raising=False)
    now = datetime.now(UTC).isoformat()
    eps = [
        dict(_ep("alpha one", "S1", "e:1"), created_at=now),
        dict(_ep("alpha two", "S1", "e:2"), created_at=now),
        dict(_ep("alpha three", "S1", "e:3"), created_at=now),
        _ep("bravo one", "S2", "e:4"),
    ]
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2, 3])
    out = r._compact_to_passages("q", eps, n=3)
    assert _ids(out) == {"e:1", "e:2", "e:4"}  # live S1 capped at 2, S2 pulled in
