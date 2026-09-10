"""Degradation warnings on the recall response (`warnings` list).

Every retrieval leg is fail-soft, which used to make a broken backend indistinguishable
from empty memory: a blank VOYAGE_API_KEY produced HTTP 200 with `{"query": ..., "facts": []}`
and the only evidence lived in the mcp-server log. recall() and recall_episodes() now attach a
`warnings` list naming each leg that degraded.

The key is ABSENT on a healthy call — that is the contract the plugin, the dashboard, and the
rest of the suite depend on, so it gets its own test.

Pure-logic tests: no DB, no Voyage. Legs are stubbed the same way the other recall tests stub
them, and the embedder/reranker are replaced by objects that raise.
"""

from __future__ import annotations

import ingestion.db as db_mod
import mcp_server.recall as recall_mod
from ingestion.surfaces import SurfaceTrust
from mcp_server.recall import Recall, _config_hint, _err_brief, _warn, _warn_sink

# Poison DSN (same rationale as test_recall_floor): an unstubbed leg must fail loudly
# instead of quietly connecting via libpq's PG* env vars.
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


class _Unauthorized(RuntimeError):
    """Stand-in for the real symptom: Voyage rejecting a blank or wrong key."""


class _FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def embed(self, texts, task="query"):
        if self._fail:
            raise _Unauthorized("Unauthorized")
        return [[0.0] * 4 for _ in texts]


class _FakeReranker:
    """rerank_scored either raises or returns descending scores in input order."""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def rerank_scored(self, query, documents, top_k=None):
        if self._fail:
            raise _Unauthorized("Unauthorized")
        scored = [(i, 1.0 - 0.01 * i) for i in range(len(documents))]
        return scored[: (top_k or len(documents))]


def _pool(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"e:{i}",
            "content": f"pool doc {i}",
            "doc_type": "episode",
            "project": None,
            "created_at": None,
            "retrieval_count": 0,
        }
        for i in range(n)
    ]


def _wired(*, embed_fail: bool = False, rerank_fail: bool = False, pool: list[dict] | None = None):
    """A Recall with every network/DB leg stubbed. Embedder and reranker stay REAL objects
    (fakes that raise), because the warnings under test come from their failure branches."""
    r = Recall(_NO_DB, "")
    p = _pool() if pool is None else pool
    r._ensure_embedder = lambda: _FakeEmbedder(fail=embed_fail)
    r._reranker = _FakeReranker(fail=rerank_fail)
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None: list(p)
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None: []
    r._search_vector_web = lambda emb, n: []
    r._search_bm25_web = lambda q, n: []
    r._search_kg = lambda *a, **k: ([], [])
    r._search_notes = lambda q, emb, proj, audience=None: []
    r._fetch_superseded_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._compact_to_passages = lambda q, eps, n: [
        {"id": e["id"], "content": f"passage {e['id']}"} for e in eps[:n]
    ]
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._episode_pool = lambda q, emb, proj, session_id=None, allowed_projects=None: list(p)
    r._select_episodes = lambda q, pl, limit: (pl[:limit], 0, 0.9)
    r._record_metrics = lambda m: None
    return r


def _joined(out: dict) -> str:
    return " || ".join(out.get("warnings", []))


# --- the healthy path keeps its exact shape ---------------------------------------


def test_healthy_recall_has_no_warnings_key():
    out = _wired().recall("what did we decide about the reranker")
    assert "warnings" not in out
    assert out["query"] == "what did we decide about the reranker"


def test_healthy_recall_episodes_has_no_warnings_key():
    out = _wired().recall_episodes("what did we decide about the reranker")
    assert "warnings" not in out


# --- embedding failure: the audit symptom -----------------------------------------


def test_embedding_failure_warns_and_names_the_env_var():
    out = _wired(embed_fail=True).recall("anything")
    msg = _joined(out)
    assert "embedding failed (voyage: Unauthorized)" in msg
    assert "vector legs skipped" in msg
    assert "BM25-only" in msg
    assert "VOYAGE_API_KEY" in msg


def test_embedding_failure_also_warns_the_kg_leg_was_skipped():
    # Without a query embedding the KG leg cannot run, so `facts: []` has TWO causes to
    # explain, not one. Both must be named or the empty facts bucket still reads as
    # "memory is empty".
    msg = _joined(_wired(embed_fail=True).recall("anything"))
    assert "KG facts leg skipped" in msg
    assert "no query embedding" in msg


def test_recall_episodes_reports_embedding_failure():
    out = _wired(embed_fail=True).recall_episodes("anything")
    msg = _joined(out)
    assert "embedding failed (voyage: Unauthorized)" in msg
    assert "VOYAGE_API_KEY" in msg


def test_empty_recall_with_a_warning_is_still_a_200_shaped_body():
    # The regression this whole field exists for: the body still looks normal, so the
    # warning is the ONLY thing distinguishing broken retrieval from empty memory.
    r = _wired(embed_fail=True)
    r._compact_to_passages = lambda q, eps, n: []
    out = r.recall("anything")
    assert out["facts"] == []
    assert "episodes" not in out
    assert out["warnings"]  # non-empty


# --- rerank failures ---------------------------------------------------------------


def test_scored_rerank_failure_warns_rrf_order():
    # This warning is raised inside a _leg_executor worker: it also guards that the
    # ContextVar sink survives the hop into the thread pool (see _submit_ctx).
    msg = _joined(_wired(rerank_fail=True).recall("anything"))
    assert "rerank failed (voyage: Unauthorized)" in msg
    assert "RRF fusion order" in msg
    assert "SYNAPSE_RERANK" in msg or "VOYAGE_API_KEY" in msg


def test_rerank_warning_is_deduped_across_legs():
    # The episode pool and the web pool both call _rerank_pool_scored. One failure
    # cause should read as one line, not one line per leg.
    r = _wired(rerank_fail=True)
    r._search_bm25_web = lambda q, n: [
        {"id": f"w:{i}", "content": f"web {i}", "web_artifact_id": i} for i in range(3)
    ]
    out = r.recall("anything")
    hits = [w for w in out["warnings"] if w.startswith("rerank failed")]
    assert len(hits) == 1


def test_passage_rerank_failure_warns_passages_dropped():
    long_doc = "".join(f"\n## Section {i}\nlorem ipsum dolor sit amet " * 6 for i in range(24))
    r = Recall(_NO_DB, "")
    r._reranker = _FakeReranker(fail=True)
    sink: list[str] = []
    with _warn_sink(sink):
        assert r._compact_to_passages("q", [{"content": long_doc, "created_at": None}], n=2) == []
    msg = " || ".join(sink)
    assert "passage rerank failed (voyage: Unauthorized)" in msg
    assert "episode passages dropped" in msg


def test_relevance_floor_rerank_failure_warns_unfiltered():
    r = Recall(_NO_DB, "")
    r._reranker = _FakeReranker(fail=True)
    items = [{"fact": "a"}, {"fact": "b"}]
    sink: list[str] = []
    with _warn_sink(sink):
        assert r._floor_by_rerank("q", items, 0.5) == items
    assert "relevance-floor rerank failed" in " || ".join(sink)
    assert "serving all items unfiltered" in " || ".join(sink)


# --- other fail-soft legs ----------------------------------------------------------


class _DeadPG:
    """A live-looking psycopg handle whose every query raises — the shape of a schema or
    permission problem, which is what the leg-level try/except blocks actually catch."""

    closed = False

    def execute(self, *a, **k):
        raise RuntimeError('relation "episodes" does not exist')

    def cursor(self, *a, **k):
        raise RuntimeError('relation "kg_entities" does not exist')

    def transaction(self):
        raise RuntimeError('relation "kg_entities" does not exist')


def test_kg_leg_failure_warns():
    r = _wired()
    del r._search_kg  # use the real method
    r._ensure_pg = lambda: _DeadPG()
    # Full trust, or the KG leg is skipped by the schema-053 posture before it can fail.
    msg = _joined(r.recall("anything", trust=SurfaceTrust(trust="full", known=True)))
    assert "KG facts leg failed" in msg
    assert "no facts served" in msg


def test_notes_leg_failure_warns(monkeypatch):
    class _DeadDB:
        def __init__(self, url):
            raise RuntimeError("db down")

    monkeypatch.setattr(db_mod, "Database", _DeadDB)
    monkeypatch.setattr(recall_mod, "_NOTES_IN_RECALL", True)
    r = _wired()
    del r._search_notes  # real notes leg, dead Database
    msg = _joined(r.recall("anything"))
    assert "notes leg failed" in msg
    assert "no notes served" in msg


def test_bm25_and_vector_episode_leg_failures_warn():
    r = _wired()
    del r._search_bm25_episodes  # real legs
    del r._search_vector_episodes
    r._ensure_pg = lambda: _DeadPG()
    msg = _joined(r.recall("anything"))
    assert "BM25 episodes search failed" in msg
    assert "vector episodes search failed" in msg


def test_web_leg_failures_warn():
    r = _wired()
    del r._search_bm25_web
    del r._search_vector_web
    r._ensure_pg = lambda: _DeadPG()
    msg = _joined(r.recall("anything"))
    assert "BM25 web search failed" in msg
    assert "vector web search failed" in msg
    assert "web bucket is degraded" in msg


# --- helpers -----------------------------------------------------------------------


def test_err_brief_redacts_credentials():
    e = RuntimeError("401 calling https://api.voyageai.com with api_key=pa-supersecret123")
    brief = _err_brief(e)
    assert "pa-supersecret123" not in brief
    assert "api_key=***" in brief


def test_err_brief_falls_back_to_class_name_and_is_one_line():
    assert _err_brief(_Unauthorized()) == "_Unauthorized"
    assert "\n" not in _err_brief(RuntimeError("line one\nline two"))
    assert len(_err_brief(RuntimeError("x" * 500))) <= 120


def test_config_hint_only_fires_for_config_shaped_errors():
    assert "VOYAGE_API_KEY" in _config_hint(
        "Unauthorized", backend="voyage", env_prefix="SYNAPSE_EMBED"
    )
    assert _config_hint("rate limit exceeded", backend="voyage", env_prefix="SYNAPSE_EMBED") == ""
    assert "SYNAPSE_EMBED_BASE_URL" in _config_hint(
        "Connection refused", backend="openai", env_prefix="SYNAPSE_EMBED"
    )


def test_warn_outside_a_recall_is_a_noop():
    _warn("nobody is listening")  # must not raise


def test_warn_dedupes_within_one_sink():
    sink: list[str] = []
    with _warn_sink(sink):
        _warn("same")
        _warn("same")
        _warn("other")
    assert sink == ["same", "other"]


def test_sink_does_not_leak_between_calls():
    r = _wired(embed_fail=True)
    assert r.recall("first")["warnings"]
    healthy = _wired()
    assert "warnings" not in healthy.recall("second")


def test_no_em_dashes_in_warning_text():
    # House style for user-facing prose; the messages are read by humans in a terminal.
    msg = _joined(_wired(embed_fail=True, rerank_fail=True).recall("anything"))
    assert "—" not in msg
