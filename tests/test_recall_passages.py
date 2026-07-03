"""Passage compaction (recall._compact_to_passages) — Stage 2 opt-in serving path.

Pure-logic tests — no DB, no Voyage (fake embedder). Guard the contract that makes passage mode
safe to flip on: it returns at most n passages, each carrying its PARENT episode's project/date so
the caller can still attribute and drill down; it skips the rerank entirely when a few chunks already
fit; and it degrades to [] (caller falls back to full episodes) when the reranker errors.
"""

from __future__ import annotations

from mcp_server.recall import (
    Recall,
    _apply_supersessions,
    _parse_episode_ids,
    _to_recall_item,
)

# A markdown doc that chunk_markdown splits into several passages (> CHUNK_TARGET=1500 chars,
# with \n## section boundaries so the recursive splitter has somewhere clean to cut).
_LONG = "".join(f"\n## Section {i}\nlorem ipsum dolor sit amet " * 6 for i in range(24))


def _ep(content: str, project: str = "synapse", date: str = "2026-06-01") -> dict:
    return {"content": content, "project": project, "created_at": f"{date}T12:00:00+00:00"}


class _FakeEmb:
    """rerank_scored that returns a caller-specified passage order (most→least relevant)."""

    def __init__(self, order: list[int]) -> None:
        self._order = order

    def rerank_scored(self, query, documents, top_k=None):
        idx = [i for i in self._order if i < len(documents)]
        idx += [i for i in range(len(documents)) if i not in idx]
        scored = [(i, 1.0 - 0.01 * r) for r, i in enumerate(idx)]
        return scored[: (top_k or len(documents))]


class _Boom:
    def rerank_scored(self, query, documents, top_k=None):
        raise RuntimeError("rate limit")


def test_returns_n_passages_with_parent_meta():
    r = Recall("", "")
    r._reranker = _FakeEmb([2, 0, 5])
    eps = [_ep(_LONG, project="synapse", date="2026-06-02")]
    out = r._compact_to_passages("q", eps, n=2)
    assert len(out) == 2
    for item in out:
        assert item["content"].strip()
        assert item["content"] in _LONG  # a real slice of the parent, not the whole turn
        assert len(item["content"]) < len(_LONG)  # genuinely compacted
        assert item["project"] == "synapse"
        assert item["date"] == "2026-06-02"


def test_few_chunks_skip_rerank():
    # A short episode is a single chunk; total chunks <= n, so the reranker is never called.
    r = Recall("", "")
    r._reranker = _Boom()  # would raise if used
    out = r._compact_to_passages("q", [_ep("a short single-chunk turn")], n=3)
    assert len(out) == 1
    assert out[0]["content"] == "a short single-chunk turn"


def test_rerank_failure_returns_empty():
    # Many chunks + a failing reranker -> [] so the caller falls back to full episodes.
    r = Recall("", "")
    r._reranker = _Boom()
    assert r._compact_to_passages("q", [_ep(_LONG)], n=2) == []


def test_empty_episodes_returns_empty():
    r = Recall("", "")
    r._reranker = _FakeEmb([0])
    assert r._compact_to_passages("q", [], n=3) == []
    assert r._compact_to_passages("q", [_ep("   ")], n=3) == []


def test_passage_meta_omits_null_project():
    r = Recall("", "")
    r._reranker = _FakeEmb([0])
    out = r._compact_to_passages("q", [{"content": "lone turn", "created_at": None}], n=3)
    assert out == [{"content": "lone turn"}]  # no project, no date keys when source had none


def test_passage_carries_parent_episode_id():
    # The passage must carry its parent episode id so fetch_episode() can expand the full turn.
    r = Recall("", "")
    r._reranker = _FakeEmb([0])
    ep = {**_ep("a short turn"), "id": "e:42"}
    out = r._compact_to_passages("q", [ep], n=3)
    assert out[0]["id"] == "e:42"


def test_to_recall_item_keeps_id_first():
    item = _to_recall_item({"id": "e:7", "content": "hi", "project": "synapse"})
    assert item["id"] == "e:7"
    assert item["content"] == "hi"
    # id is dropped only when absent
    assert "id" not in _to_recall_item({"content": "hi"})


def test_parse_episode_ids():
    assert _parse_episode_ids(["e:227168", "13", 9, "e:13"]) == [227168, 13, 9]  # dedup, order-kept
    assert _parse_episode_ids(["garbage", "e:", None, True, "e:5"]) == [5]  # skip unparseable/bool
    assert _parse_episode_ids([str(i) for i in range(50)]) == list(
        range(20)
    )  # capped at _FETCH_MAX


def test_apply_supersessions_attaches_and_dedups():
    items = [
        {"id": "e:42", "content": "old claim about X"},
        {"id": "e:7", "content": "still current"},
        {"content": "no id"},  # unkeyed item is untouched
    ]
    sup = {42: ["X is now Y", "already-served fact"], 7: []}
    _apply_supersessions(items, sup, served_facts={"already-served fact"})
    assert items[0]["superseded_by"] == ["X is now Y"]  # dedups the one already in the facts bucket
    assert "superseded_by" not in items[1]  # empty supersession list -> no key
    assert "superseded_by" not in items[2]  # no id -> skipped
