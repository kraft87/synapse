"""The pgvector text literal: one format everywhere, formatted once per recall."""

from __future__ import annotations

from unittest.mock import patch

from ingestion import vector
from ingestion.vector import QueryVector, vec_literal
from mcp_server.recall import Recall

_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


def test_literal_is_six_decimals_per_component():
    assert vec_literal([1.0, 2.5, -0.1234567, 1e-05]) == "[1.000000,2.500000,-0.123457,0.000010]"


def test_none_stays_none():
    assert vec_literal(None) is None


def test_query_vector_formats_once_and_stays_a_list():
    emb = [0.25] * 8
    with patch.object(vector, "_format", wraps=vector._format) as fmt:
        q = QueryVector(emb)
        first, second = vec_literal(q), vec_literal(q)
    assert fmt.call_count == 1
    assert first is second is q.literal
    assert first == vec_literal(emb)
    assert q == emb and isinstance(q, list)


def test_recall_embeds_the_query_into_one_shared_literal():
    class _Emb:
        def embed(self, texts, task="query"):
            return [[0.5] * 4 for _ in texts]

    r = Recall(_NO_DB, "")
    r._ensure_embedder = lambda: _Emb()
    emb, _ms = r._embed_query("anything")
    assert isinstance(emb, QueryVector)
    assert vec_literal(emb) == "[0.500000,0.500000,0.500000,0.500000]"
