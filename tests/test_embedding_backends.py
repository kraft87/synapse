"""Tests for the pluggable embedding + rerank backends (ingestion/embedding.py).

Covers:

* ``create_embedder`` provider selection from ``SYNAPSE_EMBED_PROVIDER``
  (voyage default, openai, unknown → EmbeddingConfigError) and the loud
  half-configured-openai failures (missing base URL / model / dims).
* ``OpenAIEmbeddingModel``: request mapping (path, auth, payload, the
  ``dimensions`` param heuristic + SYNAPSE_EMBED_SEND_DIMS override),
  response mapping (index-ordered), LOUD dims-mismatch validation, retry
  semantics (429/5xx transient, 4xx not), key never echoed.
* ``create_reranker`` provider selection (voyage default, http, none → None).
* ``HTTPReranker``: request mapping, both response shapes (Cohere/Infinity
  ``{"results": [...]}`` and TEI bare-list), top_k slicing.
* ``check_embedding_meta``: config↔database mismatch is loud.
* Recall's rerank=none path: fusion-only order, facts kept, passages skipped.

No network: every HTTP test uses ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
import pytest

import mcp_server.recall as recall_mod
from ingestion.embedding import (
    EmbeddingConfigError,
    EmbeddingHTTPError,
    HTTPReranker,
    OpenAIEmbeddingModel,
    TransientEmbeddingHTTPError,
    VoyageEmbeddingModel,
    _pack_batches,
    _voyage_stop,
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
# Voyage backend — request timeouts and retry budget
# ---------------------------------------------------------------------------


class _FakeVoyageClient:
    """Records the timeout each client was built with and the calls made on it."""

    built: ClassVar[list[_FakeVoyageClient]] = []

    def __init__(self, api_key: str = "", timeout: float | None = None, **_: Any) -> None:
        self.timeout = timeout
        self.calls: list[str] = []
        self.tokenized: list[list[str]] = []
        _FakeVoyageClient.built.append(self)

    def tokenize(self, texts: list[str], model: str = "") -> list[list[int]]:
        self.tokenized.append(list(texts))
        return [[0] * len(t.split()) for t in texts]

    def embed(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls.append(f"embed:{kwargs['input_type']}")
        return type("R", (), {"embeddings": [[float(len(t))] for t in texts]})()

    def rerank(self, query: str, documents: list[str], **kwargs: Any) -> Any:
        self.calls.append("rerank")
        return type("R", (), {"results": []})()


class _VoyageError(Exception):
    pass


def _named_error(name: str) -> Exception:
    return type(name, (_VoyageError,), {})()


class TestVoyageTimeouts:
    @pytest.fixture
    def emb(self, monkeypatch):
        import voyageai

        _FakeVoyageClient.built = []
        monkeypatch.setattr(voyageai, "Client", _FakeVoyageClient)
        return VoyageEmbeddingModel(api_key="k")

    def test_interactive_calls_time_out_in_seconds_not_ten_minutes(self, emb):
        """voyageai's default request timeout is 600 s. A stalled query embed or rerank
        inside recall() must fail over to the RRF fallback, not hang the recall."""
        emb.embed(["what did I decide"], task="query")
        emb.rerank_scored("q", ["d1", "d2"])
        (interactive,) = [c for c in _FakeVoyageClient.built if c.calls]
        assert interactive.timeout == 30
        assert interactive.calls == ["embed:query", "rerank"]

    def test_bulk_document_embeds_get_a_longer_timeout(self, emb):
        """A 100K-token document batch legitimately takes longer than a query embed."""
        emb.embed(["a turn", "another turn"], task="document")
        emb.embed(["an entity"], task="entity")
        (bulk,) = [c for c in _FakeVoyageClient.built if c.calls]
        assert bulk.timeout == 120
        assert bulk.calls == ["embed:document", "embed:document"]

    @staticmethod
    def _state(exc: Exception, attempt: int, elapsed: float) -> Any:
        from unittest.mock import MagicMock

        state = MagicMock()
        state.outcome.exception.return_value = exc
        state.attempt_number = attempt
        state.seconds_since_start = elapsed
        return state

    def test_non_rate_limit_retries_stop_after_a_minute(self):
        """Six 30-s timeouts plus backoff would hold a caller for minutes."""
        timeout = _named_error("Timeout")
        assert not _voyage_stop(self._state(timeout, attempt=2, elapsed=31))
        assert _voyage_stop(self._state(timeout, attempt=3, elapsed=61))
        assert _voyage_stop(self._state(timeout, attempt=6, elapsed=5))

    def test_rate_limit_keeps_its_three_minute_patience(self):
        rate_limit = _named_error("RateLimitError")
        assert not _voyage_stop(self._state(rate_limit, attempt=8, elapsed=61))
        assert _voyage_stop(self._state(rate_limit, attempt=9, elapsed=181))


@pytest.fixture
def fake_voyage(monkeypatch):
    """Fake voyageai.Client and an empty backend registry, so factory calls build fresh
    instances over the fake instead of returning one an earlier test built."""
    import voyageai

    import ingestion.embedding as emb_mod

    _FakeVoyageClient.built = []
    monkeypatch.setattr(voyageai, "Client", _FakeVoyageClient)
    monkeypatch.setattr(emb_mod, "_SHARED", {})
    return _FakeVoyageClient


class _FakeClock:
    """Stands in for tenacity's clock and its backoff sleep: sleeping advances time."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestVoyageRetryPolicy:
    """Recall waits on its query embed and reranks; batch work can wait out a 429."""

    @pytest.fixture
    def clock(self, monkeypatch, fake_voyage):
        import tenacity

        clock = _FakeClock()
        monkeypatch.setattr(tenacity, "time", clock)
        monkeypatch.setattr(tenacity.nap, "time", clock)
        return clock

    @staticmethod
    def _rate_limited(emb: VoyageEmbeddingModel, clock: _FakeClock) -> list[float]:
        """Make every Voyage call on ``emb`` a 429; return the times of the attempts."""
        attempts: list[float] = []

        def _429(*_a: Any, **_k: Any) -> Any:
            attempts.append(clock.now)
            raise _named_error("RateLimitError")

        emb._client.embed = _429
        emb._client.rerank = _429
        return attempts

    def test_interactive_429_gives_up_within_seconds(self, clock):
        emb = create_embedder(voyage_api_key="k", interactive=True)
        attempts = self._rate_limited(emb, clock)
        with pytest.raises(_VoyageError):
            emb.embed(["what did I decide"], task="query")
        assert len(attempts) > 1  # a blip still gets retried
        assert clock.now < 5

    def test_interactive_rerank_429_gives_up_within_seconds(self, clock):
        rr = create_reranker(voyage_api_key="k", interactive=True)
        assert isinstance(rr, VoyageEmbeddingModel)
        attempts = self._rate_limited(rr, clock)
        with pytest.raises(_VoyageError):
            rr.rerank_scored("q", ["d1", "d2"])
        assert len(attempts) > 1
        assert clock.now < 5

    def test_batch_429_keeps_three_minute_patience(self, clock):
        emb = create_embedder(voyage_api_key="k")
        attempts = self._rate_limited(emb, clock)
        with pytest.raises(_VoyageError):
            emb.embed(["what did I decide"], task="query")
        assert attempts[-1] >= 120

    def test_interactive_and_batch_are_separate_instances(self, fake_voyage):
        batch = create_embedder(voyage_api_key="k")
        interactive = create_embedder(voyage_api_key="k", interactive=True)
        assert batch is not interactive
        assert interactive._interactive and not batch._interactive


class TestSharedBackends:
    """A voyageai Client pins its ~43 MB tokenizer for the life of the process, so the
    factories hand every caller with the same configuration the same instance."""

    def test_same_config_returns_one_embedder(self, fake_voyage):
        first = create_embedder(voyage_api_key="k")
        assert create_embedder(voyage_api_key="k") is first
        assert len(fake_voyage.built) == 2  # one model: its query and bulk clients

    def test_changed_config_builds_a_new_embedder(self, fake_voyage, monkeypatch):
        first = create_embedder(voyage_api_key="k")
        assert create_embedder(voyage_api_key="other") is not first
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "1024")
        resized = create_embedder(voyage_api_key="k")
        assert resized is not first
        assert resized.dimensions == 1024

    def test_openai_embedder_is_shared(self, monkeypatch):
        import ingestion.embedding as emb_mod

        monkeypatch.setattr(emb_mod, "_SHARED", {})
        monkeypatch.setenv("SYNAPSE_EMBED_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_EMBED_BASE_URL", "http://x:7997")
        monkeypatch.setenv("SYNAPSE_EMBED_MODEL", "m")
        monkeypatch.setenv("SYNAPSE_EMBED_DIMS", "768")
        assert create_embedder() is create_embedder()

    def test_same_config_returns_one_reranker(self, fake_voyage, monkeypatch):
        assert create_reranker(voyage_api_key="k") is create_reranker(voyage_api_key="k")
        monkeypatch.setenv("SYNAPSE_RERANK_PROVIDER", "http")
        monkeypatch.setenv("SYNAPSE_RERANK_BASE_URL", "http://local-inference:7997")
        assert create_reranker() is create_reranker()


class TestVoyageTokenCounting:
    def test_a_batch_is_tokenized_in_one_call(self, fake_voyage):
        emb = VoyageEmbeddingModel(api_key="k")
        emb.embed(["one turn", "another turn here", "a third"], task="document")
        (counter,) = [c for c in fake_voyage.built if c.tokenized]
        assert counter.tokenized == [["one turn", "another turn here", "a third"]]

    def test_a_single_text_needs_no_tokenizer(self, fake_voyage):
        emb = VoyageEmbeddingModel(api_key="k")
        emb.embed(["what did I decide"], task="query")
        emb.embed(["a lone document"], task="document")
        assert not any(c.tokenized for c in fake_voyage.built)


class TestQueryCache:
    def test_repeated_query_is_embedded_once(self, fake_voyage):
        emb = VoyageEmbeddingModel(api_key="k")
        first = emb.embed(["what did I decide"], task="query")
        first[0].append(99.0)  # callers own what they get back
        again = emb.embed(["what did I decide"], task="query")
        assert again == [[17.0]]
        calls = [call for c in fake_voyage.built for call in c.calls]
        assert calls == ["embed:query"]

    def test_documents_and_batches_are_not_cached(self, fake_voyage):
        emb = VoyageEmbeddingModel(api_key="k")
        emb.embed(["same text"], task="document")
        emb.embed(["same text"], task="document")
        emb.embed(["q1", "q2"], task="query")
        emb.embed(["q1", "q2"], task="query")
        calls = sorted(call for c in fake_voyage.built for call in c.calls)
        assert calls == ["embed:document", "embed:document", "embed:query", "embed:query"]

    def test_least_recent_query_is_evicted(self, fake_voyage, monkeypatch):
        import ingestion.embedding as emb_mod

        monkeypatch.setattr(emb_mod, "_QUERY_CACHE_SIZE", 2)
        emb = VoyageEmbeddingModel(api_key="k")
        for q in ("a", "b", "a", "c", "a", "b"):
            emb.embed([q], task="query")
        # a, b, (a hit), c evicts b, (a hit), b again
        calls = [call for c in fake_voyage.built for call in c.calls]
        assert len(calls) == 4

    def test_openai_backend_caches_queries(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=_embeddings_body([[0.5] * 4]))

        emb = _openai_model(handler)
        assert emb.embed(["q"], task="query") == emb.embed(["q"], task="query") == [[0.5] * 4]
        assert len(calls) == 1


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

    def test_passages_are_picked_without_a_reranker(self):
        # many chunks + no reranker -> passages chosen by keyword overlap, never an empty bucket.
        # Same multi-chunk markdown shape as test_recall_passages._LONG.
        r = Recall("", "")
        r._reranker = None
        long = "".join(f"\n## Section {i}\nlorem ipsum dolor sit amet " * 6 for i in range(24))
        out = r._compact_to_passages("q", [{"content": long, "id": "e:1"}], n=2)
        assert [it["id"] for it in out] == ["e:1"]

    def test_single_doc_pool_needs_no_reranker(self):
        r = Recall("", "")
        r._reranker = None
        assert r._rerank_pool_scored("q", _pool(1)) == [(0, 1.0)]
