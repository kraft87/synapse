from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Protocol, cast

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

# Default (and production) embedding geometry: Voyage voyage-4-large at 2048 dims.
# The schema's vector/halfvec declarations are provisioned at these dims on first
# boot (scripts/apply_schema.sh substitutes SYNAPSE_EMBED_DIMS); a populated
# database keeps whatever it was provisioned with — see create_embedder().
DEFAULT_EMBED_DIMS = 2048
DEFAULT_VOYAGE_EMBED_MODEL = "voyage-4-large"


class EmbeddingConfigError(RuntimeError):
    """Misconfigured embedding/rerank backend, or a config↔database mismatch.

    Raised LOUDLY (never swallowed into a degraded path) because a dims or
    model mismatch doesn't error at query time — it silently breaks retrieval
    (cosine distances between vectors from different models/widths are noise).
    """


class EmbeddingHTTPError(Exception):
    """Non-transient HTTP failure from an embeddings/rerank endpoint.

    Mirrors ``ingestion.llm_client.LLMHTTPError``: carries the status code and
    a short body snippet, never request headers, so the API key can't leak
    into logs or exception chains.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TransientEmbeddingHTTPError(EmbeddingHTTPError):
    """Retryable HTTP failure (429 rate limit, 5xx) — tenacity retries these,
    then re-raises (reraise=True) so failures are never silently swallowed."""


# Voyage's client defaults to max_retries=0, so a transient rate-limit (429) or 5xx
# RAISES immediately. In recall() that means _rerank_pool silently degrades to RRF
# order — worse retrieval with no alarm. Retry the transient Voyage errors with
# exponential backoff + jitter so the API and client negotiate the rate DYNAMICALLY
# (no hardcoded TPM/worker guesses); permanent errors (auth / invalid-request /
# malformed) are NOT retried — they fail fast. Matched by class NAME so we don't import
# voyageai.error at module load (Client is imported lazily in __init__).
_VOYAGE_RETRYABLE_NAMES = frozenset(
    {"RateLimitError", "ServiceUnavailableError", "ServerError", "Timeout", "APIConnectionError"}
)


def _is_voyage_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _VOYAGE_RETRYABLE_NAMES


# Reranker model. rerank-2.5-lite is the default after a 2-sample A/B (2026-06-20):
# equal-or-better quality vs rerank-2.5 on Synapse's golden sets (EXACT 0.929->0.976
# deterministic; BROAD/REL tied within judge noise) at ~5-9% lower rerank latency and
# lower per-call cost. Override (rollback) with SYNAPSE_RERANK_MODEL=rerank-2.5.
_RERANK_MODEL = os.environ.get("SYNAPSE_RERANK_MODEL", "rerank-2.5-lite")


# Retry policy differs by failure type. A 429 (TPM rate-limit) is GUARANTEED transient —
# the rolling-minute token window clears on its own — so be patient: bigger backoff and
# ride it out for up to 3 min (this is what eliminates the rare _rerank_pool->RRF fallback
# under heavy parallel load). Other transient errors (5xx / timeout / connection) might be a
# real outage, so retry briefly then re-raise and let the caller degrade gracefully rather
# than hang a live recall() for minutes.
_RATE_LIMIT_WAIT = wait_random_exponential(multiplier=2, max=90)
_OTHER_WAIT = wait_random_exponential(multiplier=1, max=20)
_RATE_LIMIT_STOP = stop_after_delay(180)
_OTHER_STOP = stop_after_attempt(6)


def _is_rate_limit(retry_state: Any) -> bool:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return type(exc).__name__ == "RateLimitError"


def _voyage_wait(retry_state: Any) -> float:
    return (_RATE_LIMIT_WAIT if _is_rate_limit(retry_state) else _OTHER_WAIT)(retry_state)


def _voyage_stop(retry_state: Any) -> bool:
    return (_RATE_LIMIT_STOP if _is_rate_limit(retry_state) else _OTHER_STOP)(retry_state)


_voyage_retry = retry(
    retry=retry_if_exception(_is_voyage_retryable),
    wait=_voyage_wait,
    stop=_voyage_stop,
    # cast: tenacity's before_sleep_log wants a LoggerProtocol; stdlib Logger.log has a
    # stricter (compatible) signature the stubs reject. Not a redundant cast under strict.
    before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    reraise=True,
)


def _pack_batches(
    items: list[str], sizes: list[int], max_count: int, max_size: int
) -> Iterator[list[str]]:
    """Greedy batch packing shared by the embedding backends: fill a batch up to
    `max_count` items or `max_size` total weight, whichever comes first. Always
    admits at least one item per batch even if it alone exceeds `max_size` (the
    ``if batch and ...`` guard), so an oversized single item is never dropped."""
    i = 0
    while i < len(items):
        batch: list[str] = []
        total = 0
        while i < len(items) and len(batch) < max_count:
            s = sizes[i]
            if batch and total + s > max_size:
                break
            batch.append(items[i])
            total += s
            i += 1
        yield batch


class EmbeddingModel(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        texts: list[str],
        task: Literal["query", "document", "entity"] = "document",
    ) -> list[list[float]]: ...


class VoyageEmbeddingModel:
    """Voyage AI embeddings + rerank (the default backend): voyage-4-large at
    2048 dims unless overridden via ``create_embedder()``."""

    _client: Any
    model_name: str

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_VOYAGE_EMBED_MODEL,
        dims: int = DEFAULT_EMBED_DIMS,
    ) -> None:
        from voyageai import Client  # type: ignore[attr-defined]

        self._client = Client(api_key=api_key)
        self.model_name = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    _PER_ITEM_CHAR_CAP = 24_000
    _MAX_BATCH = 1000
    # Voyage hard cap is 120K tokens per request; pack to 100K to leave headroom
    # for tokenizer drift between our local count and the server's.
    _TARGET_BATCH_TOKENS = 100_000

    def embed(
        self,
        texts: list[str],
        task: Literal["query", "document", "entity"] = "document",
    ) -> list[list[float]]:
        input_type = "query" if task == "query" else "document"
        capped = [
            t[: self._PER_ITEM_CHAR_CAP] if len(t) > self._PER_ITEM_CHAR_CAP else t for t in texts
        ]
        if not capped:
            return []
        item_tokens = [self._client.count_tokens([t], model=self.model_name) for t in capped]
        out: list[list[float]] = []
        for batch in _pack_batches(
            capped, item_tokens, self._MAX_BATCH, self._TARGET_BATCH_TOKENS
        ):
            result = self._embed_batch(batch, input_type)
            out.extend(result.embeddings)
        return out

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[int]:
        """Voyage rerank-2.5 cross-encoder: return document indices ordered
        most→least relevant to the query. Used by recall() to pick which pooled
        summary/chunk candidates to surface. Transient failures are retried with
        backoff (_voyage_retry); a permanent failure still raises so the caller
        decides the fallback."""
        if not documents:
            return []
        res = self._rerank_call(query, documents, top_k or len(documents))
        return [r.index for r in res.results]

    def rerank_scored(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Same Voyage rerank-2.5 call as ``rerank()`` but returns
        ``(doc_index, relevance_score)`` pairs in most→least relevant order.

        ``rerank()`` discards the scores; callers that want a RELATIVE score
        cutoff (variable-k serving — keep docs scoring >= tau*top) need them.
        Transient failures retry via ``_voyage_retry``; permanent ones raise so
        the caller decides the fallback (recall._rerank_pool_scored degrades to
        RRF order)."""
        if not documents:
            return []
        res = self._rerank_call(query, documents, top_k or len(documents))
        return [(r.index, float(r.relevance_score)) for r in res.results]

    # Raw Voyage API calls, wrapped in tenacity backoff so transient 429/5xx/timeout
    # are retried (dynamic rate negotiation) instead of bubbling up as a hard failure.
    @_voyage_retry
    def _embed_batch(self, batch: list[str], input_type: str) -> Any:
        return self._client.embed(
            batch, model=self.model_name, input_type=input_type, output_dimension=self._dims
        )

    @_voyage_retry
    def _rerank_call(self, query: str, documents: list[str], top_k: int) -> Any:
        return self._client.rerank(query, documents, model=_RERANK_MODEL, top_k=top_k)


# ---------------------------------------------------------------------------
# Configured embedding geometry (dims) — the single source for SQL casts
# ---------------------------------------------------------------------------
#
# Every vector/halfvec SQL cast in the codebase interpolates embed_dims() so the
# query-time cast matches the width the schema was provisioned with (the halfvec
# HNSW indexes are built on the expression `embedding::halfvec(N)`; the ORDER BY
# cast must match it verbatim to be index-served). Default 2048 keeps the
# production (Voyage) SQL byte-identical.


def embed_dims() -> int:
    """Configured embedding width. SYNAPSE_EMBED_DIMS, default 2048 (Voyage)."""
    raw = os.environ.get("SYNAPSE_EMBED_DIMS", "").strip()
    if not raw:
        return DEFAULT_EMBED_DIMS
    try:
        dims = int(raw)
    except ValueError:
        raise EmbeddingConfigError(
            f"SYNAPSE_EMBED_DIMS={raw!r} is not an integer — set it to the embedding "
            "model's output width (e.g. 2048 for voyage-4-large, 768 for bge-base)."
        ) from None
    if dims <= 0:
        raise EmbeddingConfigError(f"SYNAPSE_EMBED_DIMS={dims} must be a positive integer.")
    return dims


def embed_provider() -> str:
    """Normalized SYNAPSE_EMBED_PROVIDER (default 'voyage')."""
    return os.environ.get("SYNAPSE_EMBED_PROVIDER", "voyage").strip().lower() or "voyage"


def rerank_provider() -> str:
    """Normalized SYNAPSE_RERANK_PROVIDER (default 'voyage')."""
    return os.environ.get("SYNAPSE_RERANK_PROVIDER", "voyage").strip().lower() or "voyage"


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings backend — POST {base_url}/embeddings
# ---------------------------------------------------------------------------

_EMBED_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_BODY_SNIPPET_LEN = 300

# Tenacity retry mirrors ingestion.llm_client's OpenAI backend (PR #206):
# 3 attempts, exponential 2s -> 4s -> 8s capped at 30s, on wire-level
# transients only (connect/read/timeout + 429/5xx). Non-transient 4xx raise
# immediately; after retries exhaust the error still raises (reraise=True) —
# an embedding failure must never silently produce a wrong-width vector.
_HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.TransportError, TransientEmbeddingHTTPError)),
    before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    reraise=True,
)


def _body_snippet(response: httpx.Response) -> str:
    """First ``_BODY_SNIPPET_LEN`` chars of the response BODY (never request
    headers), so the Authorization bearer cannot appear in error messages."""
    try:
        return response.text[:_BODY_SNIPPET_LEN]
    except Exception:  # pragma: no cover - undecodable body
        return "<undecodable body>"


def _raise_for_http_status(response: httpx.Response, what: str) -> None:
    """Map non-2xx embeddings/rerank responses onto the module's error types."""
    status = response.status_code
    if 200 <= status < 300:
        return
    message = f"{what} HTTP {status}: {_body_snippet(response)}"
    if status == 429 or status >= 500:
        raise TransientEmbeddingHTTPError(message, status_code=status)
    raise EmbeddingHTTPError(message, status_code=status)


def _json_body(response: httpx.Response, what: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise EmbeddingHTTPError(
            f"{what}: non-JSON 2xx body: {_body_snippet(response)}",
            status_code=response.status_code,
        ) from exc


class OpenAIEmbeddingModel:
    """Any OpenAI-compatible embeddings server: POST ``{base_url}/embeddings``.

    Works against OpenAI (base_url ``https://api.openai.com/v1``), Ollama
    (``http://host:11434/v1``), TEI (``http://host:8080/v1``), Infinity
    (``http://host:7997``), vLLM, etc. Auth is a bearer token, omitted when the
    key is blank (local servers need none); the key is never logged and never
    appears in raised errors.

    ``dims`` MUST match both the model's output width and the width the
    database schema was provisioned with — every returned vector is validated
    against it and a mismatch raises ``EmbeddingConfigError`` LOUDLY (a wrong
    width would otherwise only surface as silently broken retrieval).

    The OpenAI ``dimensions`` request param (matryoshka truncation on
    text-embedding-3-*) is sent when the model name starts with
    ``text-embedding-3`` or when ``SYNAPSE_EMBED_SEND_DIMS=1``; servers that
    reject unknown params can force it off with ``SYNAPSE_EMBED_SEND_DIMS=0``.
    """

    model_name: str

    _PER_ITEM_CHAR_CAP = 24_000  # same per-item cap as the Voyage backend
    _MAX_BATCH = 128
    # No local tokenizer for arbitrary models — pack batches by chars instead
    # (~4 chars/token heuristic keeps requests far under typical server caps).
    _TARGET_BATCH_CHARS = 200_000

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        dims: int = DEFAULT_EMBED_DIMS,
        send_dimensions: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self._dims = dims
        if send_dimensions is None:
            send_dimensions = model.startswith("text-embedding-3")
        self._send_dimensions = send_dimensions
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_EMBED_HTTP_TIMEOUT,
            transport=transport,
        )

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(
        self,
        texts: list[str],
        task: Literal["query", "document", "entity"] = "document",
    ) -> list[list[float]]:
        # The OpenAI /embeddings protocol has no input_type / task field —
        # symmetric-embedding models (bge, gte, e5-base usage without prefixes)
        # are the fit here; ``task`` is accepted for interface parity.
        del task
        capped = [
            t[: self._PER_ITEM_CHAR_CAP] if len(t) > self._PER_ITEM_CHAR_CAP else t for t in texts
        ]
        out: list[list[float]] = []
        for batch in _pack_batches(
            capped, [len(t) for t in capped], self._MAX_BATCH, self._TARGET_BATCH_CHARS
        ):
            out.extend(self._embed_batch(batch))
        return out

    @_HTTP_RETRY
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model_name, "input": batch}
        if self._send_dimensions:
            payload["dimensions"] = self._dims
        response = self._http.post("/embeddings", json=payload)
        _raise_for_http_status(response, "embeddings")
        data = _json_body(response, "embeddings")
        try:
            items = sorted(data["data"], key=lambda d: int(d["index"]))
            vectors = [[float(x) for x in d["embedding"]] for d in items]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingHTTPError(
                f"embeddings: malformed response shape: {_body_snippet(response)}",
                status_code=response.status_code,
            ) from exc
        if len(vectors) != len(batch):
            raise EmbeddingHTTPError(
                f"embeddings: sent {len(batch)} inputs, got {len(vectors)} vectors back",
                status_code=response.status_code,
            )
        for v in vectors:
            if len(v) != self._dims:
                raise EmbeddingConfigError(
                    f"Embedding endpoint returned {len(v)}-dim vectors but "
                    f"SYNAPSE_EMBED_DIMS={self._dims} (model {self.model_name!r}). "
                    "A width mismatch silently breaks retrieval — set "
                    "SYNAPSE_EMBED_DIMS to the model's true output width and make "
                    "sure it matches the dims the database schema was provisioned with."
                )
        return vectors


# ---------------------------------------------------------------------------
# HTTP rerank backend — POST {base_url}/rerank (TEI / Infinity / Cohere shape)
# ---------------------------------------------------------------------------


class HTTPReranker:
    """Standard rerank server: POST ``{base_url}/rerank``.

    Sends the Cohere-compatible body (``query`` + ``documents``, plus
    ``texts`` for TEI — both servers ignore the extra key) and understands
    both response shapes:

    * Cohere / Infinity / Jina: ``{"results": [{"index": i, "relevance_score": s}]}``
    * TEI: ``[{"index": i, "score": s}]``

    Transient failures (429/5xx/transport) retry with backoff, then raise —
    the caller (recall's rerank leg) owns the degrade-to-RRF fallback, exactly
    as with the Voyage backend.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_EMBED_HTTP_TIMEOUT,
            transport=transport,
        )

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[int]:
        """Document indices ordered most→least relevant (Voyage-interface parity)."""
        return [i for i, _ in self.rerank_scored(query, documents, top_k)]

    def rerank_scored(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """``(doc_index, relevance_score)`` pairs in most→least relevant order."""
        if not documents:
            return []
        k = top_k or len(documents)
        scored = self._rerank_call(query, documents, k)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @_HTTP_RETRY
    def _rerank_call(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,  # Cohere / Infinity field name
            "texts": documents,  # TEI field name (others ignore it)
            "top_n": top_k,
            "return_documents": False,
        }
        if self.model_name:
            payload["model"] = self.model_name
        response = self._http.post("/rerank", json=payload)
        _raise_for_http_status(response, "rerank")
        data = _json_body(response, "rerank")
        results = data.get("results") if isinstance(data, dict) else data
        try:
            if not isinstance(results, list):
                raise TypeError("results is not a list")
            return [
                (int(r["index"]), float(r.get("relevance_score", r.get("score")))) for r in results
            ]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingHTTPError(
                f"rerank: malformed response shape: {_body_snippet(response)}",
                status_code=response.status_code,
            ) from exc


# ---------------------------------------------------------------------------
# Config ↔ database validation (synapse_meta, schema 034)
# ---------------------------------------------------------------------------

_META_VALIDATED: set[str] = set()


def check_embedding_meta(meta: Mapping[str, str], dims: int, model: str) -> None:
    """Validate configured dims/model against the values recorded at provision time.

    ``meta`` is the synapse_meta key→value map. Raises ``EmbeddingConfigError``
    on mismatch; missing keys pass (pre-034 databases record nothing).
    """
    db_dims_raw = meta.get("embed_dims")
    db_model = meta.get("embed_model")
    if db_dims_raw is not None:
        try:
            db_dims = int(db_dims_raw)
        except ValueError:
            raise EmbeddingConfigError(
                f"synapse_meta.embed_dims={db_dims_raw!r} is not an integer — "
                "the meta row is corrupt; fix it to the width the schema's "
                "vector/halfvec columns were provisioned with."
            ) from None
        if db_dims != dims:
            raise EmbeddingConfigError(
                f"Embedding width mismatch: this database was provisioned with "
                f"embed_dims={db_dims} but the configured backend produces {dims} dims. "
                "Mixed-width embeddings silently break retrieval. Point SYNAPSE_EMBED_* "
                "back at the provisioned stack, or provision a FRESH database "
                "(first boot) with the new settings."
            )
    if db_model and model and db_model != model:
        raise EmbeddingConfigError(
            f"Embedding model mismatch: this database's vectors were embedded with "
            f"{db_model!r} but the configured model is {model!r}. Vectors from "
            "different models are not comparable — retrieval would silently break. "
            "Set SYNAPSE_EMBED_MODEL back to the provisioned model, or provision a "
            "FRESH database (first boot) with the new settings."
        )


def _validate_embedding_meta(db_url: str, dims: int, model: str) -> None:
    """Read synapse_meta and fail LOUD on a recorded mismatch.

    Fail-open on infrastructure problems only: a missing table (schema 034 not
    applied yet) or an unreachable database logs a warning and proceeds —
    recall/ingestion will surface those failures on their own. A *recorded*
    mismatch always raises.
    """
    if db_url in _META_VALIDATED:
        return
    import psycopg

    try:
        with psycopg.connect(db_url, connect_timeout=10) as conn:
            rows = conn.execute(
                "SELECT key, value FROM synapse_meta WHERE key IN ('embed_dims', 'embed_model')"
            ).fetchall()
    except psycopg.errors.UndefinedTable:
        logger.warning(
            "synapse_meta table missing — embedding dims/model not validated against "
            "the database (apply schema/034_embedding_meta.sql to record them)."
        )
        _META_VALIDATED.add(db_url)
        return
    except Exception as e:
        logger.warning("embedding meta validation skipped (database unavailable: %s)", e)
        return  # not cached — re-validate once the DB is reachable
    check_embedding_meta({str(k): str(v) for k, v in rows}, dims, model)
    _META_VALIDATED.add(db_url)


# ---------------------------------------------------------------------------
# Factories — backend selection via SYNAPSE_EMBED_PROVIDER / SYNAPSE_RERANK_PROVIDER
# ---------------------------------------------------------------------------


def create_embedder(
    voyage_api_key: str | None = None, db_url: str | None = None
) -> VoyageEmbeddingModel | OpenAIEmbeddingModel:
    """Build the embedding backend from env. All construction sites route here.

    ``SYNAPSE_EMBED_PROVIDER``:

    * ``voyage`` (default, also blank) — the existing Voyage client;
      ``voyage_api_key`` argument wins over ``VOYAGE_API_KEY``. Model/dims
      overridable via ``SYNAPSE_EMBED_MODEL`` / ``SYNAPSE_EMBED_DIMS``
      (defaults voyage-4-large / 2048 — production behavior unchanged).
    * ``openai`` — any OpenAI-compatible POST /embeddings endpoint
      (``SYNAPSE_EMBED_BASE_URL``, optional ``SYNAPSE_EMBED_API_KEY``,
      ``SYNAPSE_EMBED_MODEL``, ``SYNAPSE_EMBED_DIMS`` — all but the key
      required, so a half-configured backend fails at startup, not mid-ingest).

    When ``db_url`` is given, the configured dims/model are validated against
    the values recorded in synapse_meta at provision time (schema 034) and a
    mismatch raises ``EmbeddingConfigError`` — see ``_validate_embedding_meta``.
    """
    provider = embed_provider()
    dims = embed_dims()
    emb: VoyageEmbeddingModel | OpenAIEmbeddingModel
    if provider == "voyage":
        key = voyage_api_key if voyage_api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        model = os.environ.get("SYNAPSE_EMBED_MODEL", "").strip() or DEFAULT_VOYAGE_EMBED_MODEL
        emb = VoyageEmbeddingModel(api_key=key, model=model, dims=dims)
    elif provider == "openai":
        base_url = os.environ.get("SYNAPSE_EMBED_BASE_URL", "").strip()
        model = os.environ.get("SYNAPSE_EMBED_MODEL", "").strip()
        if not base_url:
            raise EmbeddingConfigError(
                "SYNAPSE_EMBED_PROVIDER=openai requires SYNAPSE_EMBED_BASE_URL "
                "(e.g. https://api.openai.com/v1, http://localhost:11434/v1, "
                "or http://local-inference:7997 for the bundled compose profile)."
            )
        if not model:
            raise EmbeddingConfigError(
                "SYNAPSE_EMBED_PROVIDER=openai requires SYNAPSE_EMBED_MODEL "
                "(the model id the endpoint serves, e.g. text-embedding-3-small "
                "or BAAI/bge-base-en-v1.5)."
            )
        if not os.environ.get("SYNAPSE_EMBED_DIMS", "").strip():
            raise EmbeddingConfigError(
                "SYNAPSE_EMBED_PROVIDER=openai requires SYNAPSE_EMBED_DIMS — it must "
                "equal the model's output width AND the width the database schema was "
                "provisioned with (SYNAPSE_EMBED_DIMS at first boot; default 2048)."
            )
        send_dims_env = os.environ.get("SYNAPSE_EMBED_SEND_DIMS", "").strip().lower()
        send_dimensions = {"1": True, "true": True, "0": False, "false": False}.get(send_dims_env)
        emb = OpenAIEmbeddingModel(
            base_url=base_url,
            api_key=os.environ.get("SYNAPSE_EMBED_API_KEY", ""),
            model=model,
            dims=dims,
            send_dimensions=send_dimensions,
        )
    else:
        raise EmbeddingConfigError(
            f"Unknown SYNAPSE_EMBED_PROVIDER={provider!r} — expected 'voyage' or 'openai'."
        )
    if db_url:
        _validate_embedding_meta(db_url, dims=dims, model=emb.model_name)
    return emb


def create_reranker(
    voyage_api_key: str | None = None,
) -> VoyageEmbeddingModel | HTTPReranker | None:
    """Build the rerank backend from env, or ``None`` when rerank is disabled.

    ``SYNAPSE_RERANK_PROVIDER``:

    * ``voyage`` (default, also blank) — the existing Voyage cross-encoder
      (model via ``SYNAPSE_RERANK_MODEL``, default rerank-2.5-lite; unchanged).
    * ``http`` — a TEI/Infinity/Cohere-compatible POST /rerank server
      (``SYNAPSE_RERANK_BASE_URL``, optional ``SYNAPSE_RERANK_API_KEY``,
      ``SYNAPSE_RERANK_MODEL`` = the served model id).
    * ``none`` — no rerank calls at all; recall serves the pre-rerank fusion
      (RRF) candidate order. Logged once here (factory runs once per process),
      not per-query.
    """
    provider = rerank_provider()
    if provider == "voyage":
        key = voyage_api_key if voyage_api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        return VoyageEmbeddingModel(api_key=key)
    if provider == "http":
        base_url = os.environ.get("SYNAPSE_RERANK_BASE_URL", "").strip()
        if not base_url:
            raise EmbeddingConfigError(
                "SYNAPSE_RERANK_PROVIDER=http requires SYNAPSE_RERANK_BASE_URL "
                "(e.g. http://local-inference:7997 for the bundled compose profile)."
            )
        return HTTPReranker(
            base_url=base_url,
            api_key=os.environ.get("SYNAPSE_RERANK_API_KEY", ""),
            model=os.environ.get("SYNAPSE_RERANK_MODEL", "").strip(),
        )
    if provider == "none":
        logger.info(
            "SYNAPSE_RERANK_PROVIDER=none — cross-encoder rerank disabled; "
            "recall serves the pre-rerank fusion (RRF) candidate order."
        )
        return None
    raise EmbeddingConfigError(
        f"Unknown SYNAPSE_RERANK_PROVIDER={provider!r} — expected 'voyage', 'http', or 'none'."
    )
