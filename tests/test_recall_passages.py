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
    _passage_role,
    _role_spans,
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
    r._reranker = _FakeEmb([2, 5, 0])
    eps = [
        _ep(_LONG, project="synapse", date="2026-06-02"),
        _ep(_LONG.replace("lorem", "other"), project="neuron", date="2026-06-03"),
    ]
    out = r._compact_to_passages("q", eps, n=2)
    assert [it["project"] for it in out] == ["synapse", "neuron"]  # chunk 5 = 2nd doc's 1st
    assert [it["date"] for it in out] == ["2026-06-02", "2026-06-03"]
    for item, parent in zip(out, eps, strict=True):
        assert item["content"].strip()
        assert item["content"] in parent["content"]  # a real slice of the parent
        assert len(item["content"]) < len(parent["content"])  # genuinely compacted


def test_chunks_of_one_episode_serve_as_one_item():
    # Chunks 0, 1 and 3 of one turn win: ONE item, adjacent chunks joined without repeating
    # their overlap, the gap before chunk 3 marked, and n still counts chunks.
    from ingestion.web_chunker import chunk_markdown

    chunks = chunk_markdown(_LONG)
    r = Recall("", "")
    r._reranker = _FakeEmb([3, 0, 1])
    stats: dict = {}
    out = r._compact_to_passages("q", [{**_ep(_LONG), "id": "e:9"}], n=3, stats=stats)
    assert len(out) == 1 and out[0]["id"] == "e:9"
    run = _LONG[chunks[0].char_start : chunks[1].char_end]
    assert out[0]["content"] == run + "\n[…]\n" + chunks[3].content
    assert stats == {"n_dup_passages": 0}


def test_repeat_of_another_episode_is_skipped_and_backfilled():
    # Same turn stored twice (two capture sources): the copy's chunk is skipped and the slot
    # goes to the next-ranked passage, not to a repeat.
    short = (
        "[user] which port does the dashboard use\n\n[assistant] it listens on 8081 behind caddy"
    )
    other = "[user] what about backups\n\n[assistant] restic runs nightly to the nas"
    eps = [
        {**_ep(short), "id": "e:1"},
        {**_ep(short), "id": "e:2"},
        {**_ep(other), "id": "e:3"},
    ]
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1, 2])
    stats: dict = {}
    out = r._compact_to_passages("q", eps, n=2, stats=stats)
    assert [it["id"] for it in out] == ["e:1", "e:3"]
    assert stats == {"n_dup_passages": 1}


def test_repeat_skipped_when_pool_fits_without_rerank():
    eps = [{**_ep("the same short turn text here"), "id": f"e:{i}"} for i in (1, 2)]
    r = Recall("", "")
    r._reranker = _Boom()  # few chunks: never called
    assert [it["id"] for it in r._compact_to_passages("q", eps, n=3)] == ["e:1"]


def test_near_copy_with_a_different_number_is_kept():
    # A changed dose/version/PR number is a correction, not a repeat: both serve.
    a = "the nightly backup job keeps 30 daily snapshots and 12 monthly ones on the nas share"
    b = a.replace("30 daily", "14 daily")
    eps = [{**_ep(a), "id": "e:1"}, {**_ep(b), "id": "e:2"}]
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 1])
    stats: dict = {}
    out = r._compact_to_passages("q", eps, n=2, stats=stats)
    assert [it["id"] for it in out] == ["e:1", "e:2"]
    assert stats == {"n_dup_passages": 0}


def test_join_overlap_rebuilds_the_parent_text():
    from ingestion.web_chunker import chunk_markdown
    from mcp_server.recall_passages import _join_overlap

    chunks = chunk_markdown(_LONG)
    text = chunks[0].content
    for c in chunks[1:]:
        text = _join_overlap(text, c.content)
    assert text == _LONG
    assert _join_overlap("abc", "xyz") == "abc\nxyz"  # no overlap: plain line join


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
    # The passage must carry its parent episode id so fetch() can expand the full turn.
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


def test_role_spans_and_passage_role():
    c = "[title] My chat\n\n[user] the real event happened\n\n[assistant] maybe we could...\n\n[tool:Bash] ls\n\n[result] ok"
    spans = _role_spans(c)
    # title is recognized but neutral; user/assistant/tool/result all attributed
    assert [s for _, s in spans] == ["", "user", "assistant", "assistant", "assistant"]
    u0 = c.index("[user]")
    a0 = c.index("[assistant]")
    assert _passage_role(spans, u0, a0) == "user"
    assert _passage_role(spans, a0, len(c)) == "assistant"  # tool/result fold into assistant
    assert _passage_role(spans, u0, len(c)) == "mixed"
    assert _passage_role(spans, 0, 5) is None  # inside the neutral [title] region
    assert _passage_role([], 0, 10) is None  # marker-free episode
    assert _passage_role(spans, 10, 10) is None  # empty span
    # region before the first marker has no attribution
    assert _passage_role(_role_spans("preamble\n[user] hi"), 0, 4) is None


def test_role_marker_requires_line_start():
    # A bracketed token mid-line (e.g. quoted inside a result) is not a marker.
    spans = _role_spans("[assistant] said [user] once upon a time")
    assert [s for _, s in spans] == ["assistant"]


def test_single_chunk_passage_gets_mixed_role():
    # Short episode = one chunk spanning the whole turn -> both sides -> "mixed".
    r = Recall("", "")
    r._reranker = _FakeEmb([0])
    out = r._compact_to_passages("q", [_ep("[user] question\n\n[assistant] answer")], n=3)
    assert out[0]["role"] == "mixed"


def test_no_markers_omits_role():
    r = Recall("", "")
    r._reranker = _FakeEmb([0])
    out = r._compact_to_passages("q", [_ep("plain content, no role markers")], n=3)
    assert "role" not in out[0]


_USER_HALF = "[user] " + "".join(f"\n## U{i}\nuser fact text " * 6 for i in range(12))
_ASST_HALF = "\n\n[assistant] " + "".join(
    f"\n## A{i}\nspeculative plan text " * 6 for i in range(12)
)


def test_multichunk_passages_carry_side_specific_roles():
    # User half and assistant half each long enough to yield whole chunks on one side.
    r = Recall("", "")
    r._reranker = _FakeEmb([2])
    out = r._compact_to_passages("q", [_ep(_USER_HALF + _ASST_HALF)], n=1)
    assert out[0]["role"] == "assistant"
    r._reranker = _FakeEmb([0])
    out = r._compact_to_passages("q", [_ep(_USER_HALF + _ASST_HALF)], n=1)
    assert out[0]["role"] == "user"


def test_merged_runs_keep_their_speaker_markers():
    # A user-side chunk and a non-adjacent assistant-side chunk merge into one "mixed" item;
    # the run that starts mid-turn gets its speaker's marker back.
    from ingestion.web_chunker import chunk_markdown

    chunks = chunk_markdown(_USER_HALF + _ASST_HALF)
    r = Recall("", "")
    r._reranker = _FakeEmb([0, 2])
    out = r._compact_to_passages("q", [_ep(_USER_HALF + _ASST_HALF)], n=2)
    assert len(out) == 1 and out[0]["role"] == "mixed"
    assert chunks[0].content.startswith("[user]")  # already marked: left as is
    assert out[0]["content"] == chunks[0].content + "\n[…]\n[assistant] " + chunks[2].content


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
