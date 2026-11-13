"""Tests for the pluggable embedding + rerank backends (ingestion/embedding.py).

Covers:

* ``create_embedder`` provider selection from ``SYNAPSE_EMBED_PROVIDER``
  (voyage default, openai, unknown → EmbeddingConfigError) and the loud
  half-configured-openai failures (missing base URL / model / dims).
* ``OpenAIEmbeddingModel``: request mapping (path, auth, payload, the
  ``dimensions`` param heuristic + SYNAPSE_EMBED_SEND_DIMS override),
  response mapping (index-ordered), LOUD dims-mismatch validation, retry
  semantics (429/5xx transient, 4xx not), key never echoed.
* ``VoyageEmbeddingModel``: request and response mapping for embed and rerank,
  the two retry budgets (patient on 429, brief on 5xx/transport), key and
  endpoint resolution, and that every instance and thread shares one kept-open
  connection.
* ``create_reranker`` provider selection (voyage default, http, none → None).
* ``HTTPReranker``: request mapping, both response shapes (Cohere/Infinity
  ``{"results": [...]}`` and TEI bare-list), top_k slicing.
* ``check_embedding_meta``: config↔database mismatch is loud.
* Recall's rerank=none path: fusion-only order, facts kept, passages skipped.

No network: every HTTP test uses ``httpx.MockTransport``, except the connection
reuse test, which counts real connections to a server on loopback.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from tenacity import stop_after_attempt

import ingestion.embedding as embedding_mod
import mcp_server.recall as recall_mod
from ingestion.embedding import (
    EmbeddingConfigError,
    EmbeddingHTTPError,
    HTTPReranker,
    OpenAIEmbeddingModel,
    TransientEmbeddingHTTPError,
    VoyageEmbeddingModel,
    _pack_batches,
    check_embedding_meta,
    create_embedder,
    create_reranker,
    embed_dims,
)
from mcp_server.recall import Recall

_API_KEY = "sk-embed-test-SECRET-key"

_EMBED_ENV = (
    "SYNAPSE_EMBED_PROVIDER",
    "SYNAPSE_EMBED_BASE_URL",
    "SYNAPSE_EMBED_API_KEY",
    "SYNAPSE_EMBED_MODEL",
    "SYNAPSE_EMBED_DIMS",
    "SYNAPSE_EMBED_SEND_DIMS",
    "SYNAPSE_RERANK_PROVIDER",
    "SYNAPSE_RERANK_BASE_URL",
    "SYNAPSE_RERANK_API_KEY",
    "SYNAPSE_RERANK_MODEL",
)


class TestPackBatches:
    """Shared greedy batch-packer used by both embed backends."""

    def test_empty_yields_nothing(self):
        assert list(_pack_batches([], [], 128, 1000)) == []

    def test_splits_on_count(self):
        items = [f"i{n}" for n in range(300)]
        batches = list(_pack_batches(items, [1] * 300, 128, 10_000))
        assert [len(b) for b in batches] == [128, 128, 44]

    def test_splits_on_size_before_overflow(self):
        # sizes 40,40,40 with cap 100 → [40+40], then [40]
        batches = list(_pack_batches(["a", "b", "c"], [40, 40, 40], 128, 100))
        assert [len(b) for b in batches] == [2, 1]

    def test_oversized_single_item_admitted(self):
        # An item alone exceeding max_size is never dropped: it lands in its own
        # batch (the "always admit at least one" guard).
        batches = list(_pack_batches(["big", "x"], [500, 10], 128, 100))
        assert batches == [["big"], ["x"]]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _EMBED_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-test-key")
    yield


@pytest.fixture(autouse=True)
def _no_retry_sleep(no_retry_sleep):
    """Auto-apply the shared conftest no_retry_sleep patch to every test here."""
    yield


def _embeddings_body(vectors: list[list[float]], shuffle: bool = False) -> dict[str, Any]:
    items = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if shuffle:
        items = items[::-1]
    return {"object": "list", "data": items, "model": "m"}


def _openai_model(
    handler,
    *,
    model: str = "BAAI/bge-base-en-v1.5",
    dims: int = 4,
    api_key: str = _API_KEY,
    send_dimensions: bool | None = None,
) -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(
        base_url="http://local-inference:7997",
        api_key=api_key,
        model=model,
        dims=dims,
        send_dimensions=send_dimensions,
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------------------
# Provider selection — embeddings
# ---------------------------------------------------------------------------


class TestEmbedProviderSelection:
    def test_default_is_voyage(self):
        emb = create_embedder(voyage_api_key="k")
        assert isinstance(emb, VoyageEmbeddingModel)
        assert emb.model_name == "voyage-4-large"
        assert emb.dimensions == 2048

    def test_blank_provider_is_voyage(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "")
        assert isinstance(create_embedder(voyage_api_key="k"), VoyageEmbeddingModel)

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "Voyage")
        assert isinstance(create_embedder(voyage_api_key="k"), VoyageEmbeddingModel)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "bedrock")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_EMBED_PROVIDER"):
            create_embedder(voyage_api_key="k")

    def test_openai_requires_base_url(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_MODEL", "m")
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "768")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_EMBED_BASE_URL"):
            create_embedder()

    def test_openai_requires_model(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_BASE_URL", "http://x:7997")
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "768")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_EMBED_MODEL"):
            create_embedder()

    def test_openai_requires_dims(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_BASE_URL", "http://x:7997")
        monkeypatch.setenv("SYNAPSE_EMBED_MODEL", "m")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_EMBED_DIMS"):
            create_embedder()

    def test_openai_happy_path(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_BASE_URL", "http://local-inference:7997")
        monkeypatch.setenv("SYNAPSE_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "768")
        emb = create_embedder()
        assert isinstance(emb, OpenAIEmbeddingModel)
        assert emb.model_name == "BAAI/bge-base-en-v1.5"
        assert emb.dimensions == 768

    def test_voyage_dims_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "1024")
        emb = create_embedder(voyage_api_key="k")
        assert isinstance(emb, VoyageEmbeddingModel)
        assert emb.dimensions == 1024

    def test_bad_dims_is_loud(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "not-a-number")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_EMBED_DIMS"):
            embed_dims()

    def test_default_dims(self):
        assert embed_dims() == 2048


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings — request/response mapping
# ---------------------------------------------------------------------------


class TestOpenAIEmbedMapping:
    def test_request_payload_and_headers(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content)
            n = len(json.loads(request.content)["input"])
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4] * n))

        emb = _openai_model(handler)
        out = emb.embed(["alpha", "beta"])
        assert seen["path"].endswith("/embeddings")
        assert seen["auth"] == f"Bearer {_API_KEY}"
        assert seen["payload"]["model"] == "BAAI/bge-base-en-v1.5"
        assert seen["payload"]["input"] == ["alpha", "beta"]
        assert len(out) == 2

    def test_no_auth_header_when_key_blank(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler, api_key="")
        emb.embed(["x"])
        assert seen["auth"] is None

    def test_dimensions_param_sent_for_text_embedding_3(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler, model="text-embedding-3-small", dims=4)
        emb.embed(["x"])
        assert seen["payload"]["dimensions"] == 4

    def test_dimensions_param_omitted_for_other_models(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler, model="BAAI/bge-base-en-v1.5", dims=4)
        emb.embed(["x"])
        assert "dimensions" not in seen["payload"]

    def test_dimensions_param_forced_on(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler, model="BAAI/bge-base-en-v1.5", dims=4, send_dimensions=True)
        emb.embed(["x"])
        assert seen["payload"]["dimensions"] == 4

    def test_send_dims_env_wires_through_factory(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_BASE_URL", "http://x:7997")
        monkeypatch.setenv("SYNAPSE_EMBED_MODEL", "nomic-embed-text")
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "768")
        monkeypatch.setenv("SYNAPSE_EMBED_SEND_DIMS", "1")
        emb = create_embedder()
        assert isinstance(emb, OpenAIEmbeddingModel)
        assert emb._send_dimensions is True

    def test_response_order_restored_by_index(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([[1.0] * 4, [2.0] * 4], shuffle=True))

        emb = _openai_model(handler)
        out = emb.embed(["a", "b"])
        assert out == [[1.0] * 4, [2.0] * 4]

    def test_empty_input_no_http_call(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no HTTP call expected")

        emb = _openai_model(handler)
        assert emb.embed([]) == []

    def test_batching_splits_large_input(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            batch = json.loads(request.content)["input"]
            calls.append(len(batch))
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4] * len(batch)))

        emb = _openai_model(handler)
        out = emb.embed(["t"] * 300)  # _MAX_BATCH = 128
        assert len(out) == 300
        assert calls == [128, 128, 44]


class TestOpenAIEmbedErrors:
    def test_dims_mismatch_is_loud(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([[0.1] * 768]))

        emb = _openai_model(handler, dims=1024)
        with pytest.raises(EmbeddingConfigError, match=r"768.*1024"):
            emb.embed(["x"])

    def test_count_mismatch_is_loud(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler)
        with pytest.raises(EmbeddingHTTPError, match=r"2 inputs, got 1"):
            emb.embed(["a", "b"])

    def test_400_not_retried_key_never_echoed(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, json={"error": {"message": "bad model"}})

        emb = _openai_model(handler)
        with pytest.raises(EmbeddingHTTPError) as exc_info:
            emb.embed(["x"])
        assert len(calls) == 1
        assert "400" in str(exc_info.value)
        assert _API_KEY not in str(exc_info.value)
        assert _API_KEY not in repr(exc_info.value)

    def test_429_retried_then_raised(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        emb = _openai_model(handler)
        with pytest.raises(TransientEmbeddingHTTPError, match="429"):
            emb.embed(["x"])
        assert len(calls) == 3  # tenacity: 3 attempts, then reraise

    def test_500_recovers_on_retry(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        emb = _openai_model(handler)
        assert emb.embed(["x"]) == [[0.1] * 4]
        assert len(calls) == 2

    def test_non_json_2xx_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>login page</html>")

        emb = _openai_model(handler)
        with pytest.raises(EmbeddingHTTPError, match="non-JSON"):
            emb.embed(["x"])


# ---------------------------------------------------------------------------
# Voyage — REST backend
# ---------------------------------------------------------------------------


def _voyage_model(handler, *, api_key: str = _API_KEY, dims: int = 4) -> VoyageEmbeddingModel:
    return VoyageEmbeddingModel(api_key=api_key, dims=dims, transport=httpx.MockTransport(handler))


def _rerank_body(scored: list[tuple[int, float]]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"index": i, "relevance_score": s} for i, s in scored],
        "model": "rerank-3-lite",
        "usage": {"total_tokens": 7},
    }


class _CharTokenizer:
    """One token per character, so a test can size batches by string length."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


class TestVoyageEmbedMapping:
    def test_query_request_and_response(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_body([[0.1, 0.2, 0.3, 0.4]]))

        emb = _voyage_model(handler)
        assert emb.embed(["where is munich?"], task="query") == [[0.1, 0.2, 0.3, 0.4]]
        assert seen["url"] == "https://api.voyageai.com/v1/embeddings"
        assert seen["auth"] == f"Bearer {_API_KEY}"
        assert seen["payload"] == {
            "input": ["where is munich?"],
            "model": "voyage-4-large",
            "input_type": "query",
            "truncation": True,
            "output_dimension": 4,
        }

    @pytest.mark.parametrize("task", ["document", "entity"])
    def test_other_tasks_embed_as_documents(self, task):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_body([[0.0] * 4]))

        _voyage_model(handler).embed(["x"], task=task)
        assert seen["payload"]["input_type"] == "document"

    def test_response_is_put_back_in_input_order(self, monkeypatch):
        monkeypatch.setattr(embedding_mod, "_voyage_tokenizer", lambda model: _CharTokenizer())

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([[1.0] * 4, [2.0] * 4], shuffle=True))

        assert _voyage_model(handler).embed(["a", "b"]) == [[1.0] * 4, [2.0] * 4]

    def test_one_item_never_loads_the_tokenizer(self, monkeypatch):
        def no_tokenizer(model: str) -> Any:
            raise AssertionError("a single item needs no token count")

        monkeypatch.setattr(embedding_mod, "_voyage_tokenizer", no_tokenizer)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([[0.5] * 4]))

        assert _voyage_model(handler).embed(["just a query"], task="query") == [[0.5] * 4]

    def test_batches_are_packed_by_token_count(self, monkeypatch):
        monkeypatch.setattr(embedding_mod, "_voyage_tokenizer", lambda model: _CharTokenizer())
        monkeypatch.setattr(VoyageEmbeddingModel, "_TARGET_BATCH_TOKENS", 10)
        batches: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            batch = json.loads(request.content)["input"]
            batches.append(batch)
            return httpx.Response(200, json=_embeddings_body([[float(len(t))] * 4 for t in batch]))

        texts = ["aaaa", "bbbb", "cccccc", "d"]  # 4 + 4 fit under 10; 6 does not join them
        out = _voyage_model(handler).embed(texts)
        assert batches == [["aaaa", "bbbb"], ["cccccc", "d"]]
        assert [v[0] for v in out] == [4.0, 4.0, 6.0, 1.0]

    def test_empty_input_makes_no_call(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        assert _voyage_model(handler).embed([]) == []

    def test_vector_count_mismatch_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_body([]))

        with pytest.raises(EmbeddingHTTPError, match="sent 1 inputs, got 0"):
            _voyage_model(handler).embed(["x"])


class TestVoyageRerankMapping:
    def test_request_and_scored_response(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_rerank_body([(1, 0.9), (0, 0.2)]))

        rr = _voyage_model(handler)
        assert rr.rerank_scored("where is munich?", ["doc a", "doc b"]) == [(1, 0.9), (0, 0.2)]
        assert seen["url"] == "https://api.voyageai.com/v1/rerank"
        assert seen["auth"] == f"Bearer {_API_KEY}"
        assert seen["payload"] == {
            "query": "where is munich?",
            "documents": ["doc a", "doc b"],
            "model": embedding_mod._RERANK_MODEL,
            "top_k": 2,
            "truncation": True,
        }

    def test_rerank_returns_indices_and_passes_top_k(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_rerank_body([(2, 0.8)]))

        assert _voyage_model(handler).rerank("q", ["a", "b", "c"], top_k=1) == [2]
        assert seen["payload"]["top_k"] == 1

    def test_no_documents_makes_no_call(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        rr = _voyage_model(handler)
        assert rr.rerank_scored("q", []) == []
        assert rr.rerank("q", []) == []

    def test_malformed_response_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

        with pytest.raises(EmbeddingHTTPError, match="malformed response shape"):
            _voyage_model(handler).rerank_scored("q", ["a"])


class TestVoyageRetry:
    """Two budgets: a 429 clears on its own, so it is waited out for minutes; anything
    else transient may be an outage, so it gets a few quick attempts and then raises."""

    def test_429_recovers(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return httpx.Response(429, json={"detail": "rate limited"})
            return httpx.Response(200, json=_rerank_body([(0, 0.5)]))

        assert _voyage_model(handler).rerank_scored("q", ["a"]) == [(0, 0.5)]
        assert len(calls) == 3

    def test_429_outlasts_the_budget_for_other_failures(self, monkeypatch):
        # With sleeps patched out the real 180s budget would spin, so give the 429 lane an
        # attempt count of its own, above the six every other transient failure gets.
        monkeypatch.setattr(embedding_mod, "_RATE_LIMIT_STOP", stop_after_attempt(9))
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429, json={"detail": "rate limited"})

        with pytest.raises(TransientEmbeddingHTTPError, match="429"):
            _voyage_model(handler).embed(["x"])
        assert len(calls) == 9

    def test_5xx_is_retried_six_times_then_raised(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, text="unavailable")

        with pytest.raises(TransientEmbeddingHTTPError, match="503"):
            _voyage_model(handler).rerank_scored("q", ["a"])
        assert len(calls) == 6

    def test_dropped_connection_is_retried(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(200, json=_embeddings_body([[0.1] * 4]))

        assert _voyage_model(handler).embed(["x"]) == [[0.1] * 4]
        assert len(calls) == 2

    @pytest.mark.parametrize("status", [400, 401, 422])
    def test_permanent_errors_fail_fast_without_the_key(self, status):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(status, json={"detail": "Provided API key is invalid."})

        with pytest.raises(EmbeddingHTTPError) as exc_info:
            _voyage_model(handler).embed(["x"])
        assert len(calls) == 1
        assert not isinstance(exc_info.value, TransientEmbeddingHTTPError)
        assert str(status) in str(exc_info.value)
        assert _API_KEY not in str(exc_info.value)
        assert _API_KEY not in repr(exc_info.value)


class TestVoyageKeyAndEndpoint:
    def _auth_seen(self, api_key: str) -> tuple[str | None, str]:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            seen["host"] = request.url.host
            return httpx.Response(200, json=_embeddings_body([[0.0] * 4]))

        _voyage_model(handler, api_key=api_key).embed(["x"])
        return seen["auth"], seen["host"]

    def test_blank_key_falls_back_to_the_environment(self):
        assert self._auth_seen("") == ("Bearer voyage-test-key", "api.voyageai.com")

    def test_key_file_wins_over_the_environment_variable(self, monkeypatch, tmp_path):
        key_file = tmp_path / "voyage.key"
        key_file.write_text("pa-from-file\n")
        monkeypatch.setenv("VOYAGE_API_KEY_PATH", str(key_file))
        assert self._auth_seen("")[0] == "Bearer pa-from-file"

    def test_no_key_anywhere_is_a_config_error(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY")
        with pytest.raises(EmbeddingConfigError, match="VOYAGE_API_KEY"):
            VoyageEmbeddingModel(api_key="")

    def test_atlas_keys_use_the_atlas_endpoint(self):
        assert self._auth_seen("al-atlas-key") == ("Bearer al-atlas-key", "ai.mongodb.com")


class _CountingVoyageServer(ThreadingHTTPServer):
    """A loopback stand-in for Voyage that counts the TCP connections it accepts."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CountingVoyageHandler)
        self.connections = 0
        self.requests = 0
        self.lock = threading.Lock()


class _CountingVoyageHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, as Voyage serves
    server: _CountingVoyageServer

    def setup(self) -> None:
        super().setup()
        with self.server.lock:
            self.server.connections += 1

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.requests += 1
        if self.path.endswith("/rerank"):
            body = _rerank_body([(i, 0.5) for i in range(len(payload["documents"]))])
        else:
            body = _embeddings_body([[0.0] * 4 for _ in payload["input"]])
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture()
def voyage_server(monkeypatch):
    """Point the Voyage backend at a loopback server, with a client registry of its own."""
    server = _CountingVoyageServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clients: dict[str, httpx.Client] = {}
    monkeypatch.setattr(embedding_mod, "_VOYAGE_CLIENTS", clients)
    monkeypatch.setattr(
        embedding_mod, "_VOYAGE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1"
    )
    yield server
    for client in clients.values():
        client.close()
    server.shutdown()
    server.server_close()


class TestVoyageSharedConnection:
    def test_instances_with_one_key_share_a_client(self, voyage_server):
        a = VoyageEmbeddingModel(api_key="k1")
        assert VoyageEmbeddingModel(api_key="k1")._http is a._http
        assert VoyageEmbeddingModel(api_key="k2")._http is not a._http

    def test_every_instance_and_thread_reuses_one_connection(self, voyage_server):
        # The shape of one recall: the embedder on the calling thread, then the reranker
        # (a second instance) on a worker thread, then the caller again.
        embedder = VoyageEmbeddingModel(api_key="k", dims=4)
        reranker = VoyageEmbeddingModel(api_key="k")
        embedder.embed(["q"], task="query")
        with ThreadPoolExecutor(max_workers=1) as leg:
            assert leg.submit(reranker.rerank_scored, "q", ["a", "b"]).result() == [
                (0, 0.5),
                (1, 0.5),
            ]
        reranker.rerank_scored("q", ["a"])
        assert voyage_server.requests == 3
        assert voyage_server.connections == 1


# ---------------------------------------------------------------------------
# check_embedding_meta — config↔database validation
# ---------------------------------------------------------------------------


class TestEmbeddingMetaCheck:
    def test_matching_meta_passes(self):
        check_embedding_meta(
            {"embed_dims": "2048", "embed_model": "voyage-4-large"}, 2048, "voyage-4-large"
        )

    def test_empty_meta_passes(self):
        # pre-034 database: nothing recorded, nothing to validate against
        check_embedding_meta({}, 2048, "voyage-4-large")

    def test_dims_mismatch_is_loud(self):
        with pytest.raises(EmbeddingConfigError, match=r"768.*2048|2048.*768"):
            check_embedding_meta({"embed_dims": "768"}, 2048, "voyage-4-large")

    def test_model_mismatch_is_loud(self):
        with pytest.raises(EmbeddingConfigError, match=r"voyage-4-large.*bge|bge.*voyage-4-large"):
            check_embedding_meta(
                {"embed_dims": "2048", "embed_model": "voyage-4-large"},
                2048,
                "BAAI/bge-base-en-v1.5",
            )

    def test_blank_recorded_model_passes(self):
        check_embedding_meta({"embed_dims": "2048", "embed_model": ""}, 2048, "anything")


# ---------------------------------------------------------------------------
# Provider selection — rerank
# ---------------------------------------------------------------------------


class TestRerankProviderSelection:
    def test_default_is_voyage(self):
        rr = create_reranker(voyage_api_key="k")
        assert isinstance(rr, VoyageEmbeddingModel)

    def test_none_returns_none(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "none")
        assert create_reranker(voyage_api_key="k") is None

    def test_http_requires_base_url(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "http")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_RERANK_BASE_URL"):
            create_reranker()

    def test_http_happy_path(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "http")
        monkeypatch.setenv("SYNAPSE_RERANK_BASE_URL", "http://local-inference:7997")
        monkeypatch.setenv("SYNAPSE_RERANK_MODEL", "BAAI/bge-reranker-base")
        rr = create_reranker()
        assert isinstance(rr, HTTPReranker)
        assert rr.model_name == "BAAI/bge-reranker-base"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "cohere-native")
        with pytest.raises(EmbeddingConfigError, match="SYNAPSE_RERANK_PROVIDER"):
            create_reranker()


# ---------------------------------------------------------------------------
# HTTPReranker — request/response mapping
# ---------------------------------------------------------------------------


def _http_reranker(handler, *, model: str = "BAAI/bge-reranker-base") -> HTTPReranker:
    return HTTPReranker(
        base_url="http://local-inference:7997",
        api_key=_API_KEY,
        model=model,
        transport=httpx.MockTransport(handler),
    )


class TestHTTPReranker:
    def test_request_payload(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.2},
                        {"index": 1, "relevance_score": 0.9},
                    ]
                },
            )

        rr = _http_reranker(handler)
        scored = rr.rerank_scored("where is munich?", ["doc a", "doc b"])
        assert seen["path"].endswith("/rerank")
        assert seen["auth"] == f"Bearer {_API_KEY}"
        assert seen["payload"]["query"] == "where is munich?"
        assert seen["payload"]["documents"] == ["doc a", "doc b"]  # Cohere/Infinity field
        assert seen["payload"]["texts"] == ["doc a", "doc b"]  # TEI field
        assert seen["payload"]["model"] == "BAAI/bge-reranker-base"
        assert scored == [(1, 0.9), (0, 0.2)]  # most→least relevant

    def test_tei_bare_list_response_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=[{"index": 1, "score": 0.8}, {"index": 0, "score": 0.3}]
            )

        rr = _http_reranker(handler)
        assert rr.rerank_scored("q", ["a", "b"]) == [(1, 0.8), (0, 0.3)]

    def test_rerank_returns_indices_only(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 2, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.5},
                        {"index": 1, "relevance_score": 0.1},
                    ]
                },
            )

        rr = _http_reranker(handler)
        assert rr.rerank("q", ["a", "b", "c"]) == [2, 0, 1]

    def test_top_k_slices(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 2, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.5},
                        {"index": 1, "relevance_score": 0.1},
                    ]
                },
            )

        rr = _http_reranker(handler)
        assert rr.rerank_scored("q", ["a", "b", "c"], top_k=2) == [(2, 0.9), (0, 0.5)]

    def test_blank_model_omitted_from_payload(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

        rr = _http_reranker(handler, model="")
        rr.rerank_scored("q", ["a"])
        assert "model" not in seen["payload"]

    def test_empty_documents_no_http_call(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no HTTP call expected")

        rr = _http_reranker(handler)
        assert rr.rerank_scored("q", []) == []

    def test_error_raises_for_caller_fallback(self):
        """recall._rerank_pool_scored owns the degrade-to-RRF path — the client raises."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "unknown model"})

        rr = _http_reranker(handler)
        with pytest.raises(EmbeddingHTTPError, match="400"):
            rr.rerank_scored("q", ["a"])

    def test_malformed_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"weird": "shape"})

        rr = _http_reranker(handler)
        with pytest.raises(EmbeddingHTTPError, match="malformed"):
            rr.rerank_scored("q", ["a"])


# ---------------------------------------------------------------------------
# rerank=none — recall serves the fusion (RRF) order, no rerank calls
# ---------------------------------------------------------------------------


def _pool(n: int) -> list[dict[str, Any]]:
    return [{"content": f"episode {i}", "id": f"e:{i}"} for i in range(n)]


class TestRerankNoneRecallPath:
    def test_factory_wires_none_through_recall(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "none")
        r = Recall(db_url="postgresql://unused/db", voyage_api_key="")
        assert r._ensure_reranker() is None

    def test_pool_scored_serves_fusion_order(self):
        r = Recall("", "")
        r._reranker = None
        pool = _pool(4)
        scored = r._rerank_pool_scored("q", pool)
        # RRF (incoming) order preserved; 0.0 = fixed-k signal downstream
        assert scored == [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]

    def test_rerank_pool_preserves_pool_order(self):
        r = Recall("", "")
        r._reranker = None
        pool = _pool(3)
        assert r._rerank_pool("q", pool) == pool

    def test_floor_facts_keeps_all(self, monkeypatch):
        monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.40)
        r = Recall("", "")
        r._reranker = None
        facts = [{"fact": f"f{i}"} for i in range(3)]
        assert r._floor_facts("q", facts) == facts

    def test_passages_fall_back_to_full_episodes(self):
        # many chunks + no reranker -> [] so the caller serves full episodes.
        # Same multi-chunk markdown shape as test_recall_passages._LONG.
        r = Recall("", "")
        r._reranker = None
        long = "".join(f"\n## Section {i}\nlorem ipsum dolor sit amet " * 6 for i in range(24))
        out = r._compact_to_passages("q", [{"content": long, "id": "e:1"}], n=2)
        assert out == []

    def test_single_doc_pool_needs_no_reranker(self):
        r = Recall("", "")
        r._reranker = None
        assert r._rerank_pool_scored("q", _pool(1)) == [(0, 1.0)]
