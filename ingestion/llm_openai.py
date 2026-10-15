"""The OpenAI-compatible backend: pydantic-ai over any ``/chat/completions`` endpoint
(OpenRouter, Ollama, vLLM), with ``OpenAIChatClient`` as the anthropic-shaped wrapper and
``_run_agent_sync`` as the one place a pydantic-ai agent runs (retry policy and error
taxonomy included), shared with ``llm_client.structured_call``.

Selected with ``SYNAPSE_LLM_PROVIDER=openai`` (see ``llm_client.create_llm_client``). The
error taxonomy is ``ingestion.llm_common``'s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, cast

import httpx
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError as PydanticAIModelAPIError
from pydantic_ai.exceptions import ModelHTTPError as PydanticAIModelHTTPError
from tenacity import before_sleep_log, retry, retry_if_exception_type

from ingestion.llm_common import (
    _RETRY_STOP,
    _RETRY_WAIT,
    DEFAULT_MODEL,
    LLMHTTPError,
    LLMUnavailableError,
    MalformedResponseError,
    TransientLLMHTTPError,
    UsageLimitError,
    _flatten_chat,
    _Response,
    _strip_json_fence,
    _with_json_schema,
)

logger = logging.getLogger(__name__)

# --- SYNAPSE_OPENROUTER_PROVIDERS=auto: cheapest-first provider ranking -------
# OpenRouter sale pricing moves daily, and (measured 2026-08-27) neither
# default routing nor ``sort: "price"`` reaches the discounted endpoints — an
# unpinned glm-5.2 call billed at full list ($1.40/M) while 70%-off endpoints
# sat healthy. An explicit ``provider.order`` preference is the only way to
# capture sale pricing, so ``auto`` rebuilds that order from the live endpoint
# listing instead of a hardcoded list that rots when a promo ends.
# Blend weights ~4:1 input:output (extraction-shaped traffic). Fail-soft: any
# fetch problem falls back to no preference (default routing) and retries
# sooner than the normal TTL.
_PROVIDER_RANK_TTL_S = 6 * 3600.0
_PROVIDER_RANK_RETRY_S = 600.0
_PROVIDER_RANK_TOP_N = 8
_provider_rank_cache: dict[str, tuple[float, list[str]]] = {}
_provider_rank_lock = threading.Lock()


def _fetch_model_endpoints(model: str) -> list[dict[str, Any]]:
    resp = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=10.0)
    resp.raise_for_status()
    return list((resp.json().get("data") or {}).get("endpoints") or [])


def _ranked_openrouter_providers(model: str) -> list[str]:
    """Provider slugs for ``model``, cheapest blended price first (cached)."""
    now = time.monotonic()
    with _provider_rank_lock:
        hit = _provider_rank_cache.get(model)
        if hit and now < hit[0]:
            return hit[1]
    try:
        cheapest: dict[str, float] = {}
        for ep in _fetch_model_endpoints(model):
            slug = str(ep.get("tag") or "").split("/")[0]
            status = ep.get("status")
            if not slug or (isinstance(status, int | float) and status < 0):
                continue  # deranked/broken endpoint
            pricing = ep.get("pricing") or {}
            blend = (
                4 * float(pricing.get("prompt") or 0) + float(pricing.get("completion") or 0)
            ) / 5
            if blend <= 0:  # free/unpriced rows are stale listings, not wins
                continue
            if slug not in cheapest or blend < cheapest[slug]:
                cheapest[slug] = blend
        order = sorted(cheapest, key=cheapest.__getitem__)[:_PROVIDER_RANK_TOP_N]
        expires = now + _PROVIDER_RANK_TTL_S
    except Exception as e:
        logger.warning("OpenRouter provider ranking failed for %s: %s", model, e)
        order, expires = [], now + _PROVIDER_RANK_RETRY_S
    with _provider_rank_lock:
        _provider_rank_cache[model] = (expires, order)
    return order


DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "anthropic/claude-haiku-4.5"  # OpenRouter id for Haiku 4.5

_HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_BODY_SNIPPET_LEN = 300


# Profile overrides for every model driven through this client:
# * ``max_tokens`` must go on the wire as ``max_tokens`` (OpenRouter and most
#   OpenAI-compatible servers), not OpenAI-proper's ``max_completion_tokens``.
# * When a call uses native structured output, pydantic-ai ALSO injects the
#   JSON schema into the instructions — providers that ignore
#   ``response_format`` still see the contract in-prompt (belt and braces,
#   preserving the pre-pydantic-ai schema-in-prompt behavior).
def _openai_compat_profile() -> Any:
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    return OpenAIModelProfile(
        openai_chat_supports_max_completion_tokens=False,
        native_output_requires_schema_in_instructions=True,
    )


class _ErrorBodyStatusTransport(httpx.AsyncBaseTransport):
    """OpenRouter quirk shim: some failures arrive as HTTP 200 carrying an
    ``{"error": {...}}`` body and no ``choices``. Rewrite those onto their
    real status code so the OpenAI SDK raises ``APIStatusError`` and the
    normal status-based taxonomy applies — a 200-wrapped 402 must still STOP
    the cycle (``UsageLimitError``), never turn into empty text.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if response.status_code != 200:
            return response
        body = await response.aread()
        await response.aclose()
        status = 200
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            err = data.get("error") or {}
            code = err.get("code") if isinstance(err, dict) else None
            # Real HTTP-ish codes keep their semantics (402 → usage limit,
            # 429/5xx → transient); anything else becomes a non-transient 400.
            status = code if isinstance(code, int) and 400 <= code < 600 else 400
        # ``aread()`` above returned the DECODED body. If the origin response
        # was compressed, re-attaching its headers verbatim makes httpx decode
        # the already-decompressed bytes a second time — every gzipped
        # 200-wrapped error surfaced as ``DecodingError: incorrect header
        # check`` → ``APIConnectionError`` instead of its real status (the
        # 2026-07-18 "Connection error" storms during upstream 429 windows).
        # Drop the stale encoding/length headers; httpx recomputes length.
        headers = [
            (k, v)
            for k, v in response.headers.raw
            if k.lower() not in (b"content-encoding", b"content-length")
        ]
        return httpx.Response(status, headers=headers, content=body, request=request)


def _map_model_http_error(exc: PydanticAIModelHTTPError) -> Exception:
    """Map pydantic-ai's ``ModelHTTPError`` onto the module's error taxonomy.

    * 402 (payment required / out of credits) → ``UsageLimitError`` — the
      caller must STOP the cycle, exactly like a Claude Max window
      exhaustion. Never swallowed as empty text.
    * 429 / 5xx → ``TransientLLMHTTPError`` — retried by tenacity, raised
      after retries exhaust.
    * Other 4xx → ``LLMHTTPError`` — non-transient, raised immediately.

    Only the response BODY is quoted — never request headers — so the
    Authorization bearer cannot appear in the message.
    """
    status = exc.status_code
    snippet = str(exc.body)[:_BODY_SNIPPET_LEN]
    message = f"chat completions HTTP {status}: {snippet}"
    if status == 402:
        return UsageLimitError(message)
    if status == 429 or status >= 500:
        return TransientLLMHTTPError(message, status_code=status)
    return LLMHTTPError(message, status_code=status)


def _last_response_text(messages: list[Any]) -> str:
    """Last assistant text in a captured pydantic-ai run, for
    ``MalformedResponseError.raw_response``."""
    from pydantic_ai.messages import ModelResponse as _MR
    from pydantic_ai.messages import TextPart as _TP

    for message in reversed(messages):
        if isinstance(message, _MR):
            for part in message.parts:
                if isinstance(part, _TP) and part.content:
                    return str(part.content)
    return ""


_LOOP_TLS = threading.local()


def _thread_loop() -> asyncio.AbstractEventLoop:
    """One persistent event loop per worker thread.

    The 2026-07-18 "Connection error." storm was a process-wide httpx
    pool shared across worker threads whose event loops differ — fixed by
    the thread-local provider (``OpenAIChatClient._provider``). This loop
    pin complements it: one loop per thread for the pool to live on, plus
    a guard against a CLOSED loop left on the thread by other sync-async
    bridges (``asyncio.run`` in the SDK path), which ``get_event_loop``
    wouldn't recover from.
    """
    loop = getattr(_LOOP_TLS, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _LOOP_TLS.loop = loop
    return loop


def _run_agent_sync(
    agent: Any,
    user_prompt: str,
    *,
    instructions: str | None,
    structured: bool,
) -> Any:
    """Run a pydantic-ai agent with the module's retry + error taxonomy.

    tenacity mirrors the pre-pydantic-ai config (3 attempts, exp backoff
    2s → 4s → 8s, cap 30s) and keys on transients only: mapped 429/5xx
    (``TransientLLMHTTPError``), connection-level ``ModelAPIError``, and raw
    ``httpx.TransportError``. ``UsageLimitError`` (402) and plain
    ``LLMHTTPError`` (other 4xx) are never retried.

    ``UnexpectedModelBehavior`` — pydantic-ai's content-level failure — maps
    by mode:
    * structured runs whose output validation retries are exhausted →
      ``MalformedResponseError`` (the parse-failure signal call sites already
      degrade on), with the last raw response attached;
    * everything else (empty completion, token-limit-before-output, malformed
      200 body) → ``LLMHTTPError``. Empty completions are an error, never a
      result — the OpenRouter-credits incident started with empties being
      treated as "the model found nothing".

    Transport failures still standing after the last attempt (a connection-level
    ``ModelAPIError``, a raw ``httpx.TransportError``) raise as
    ``TransientLLMHTTPError``: pydantic-ai's ``ModelAPIError`` is a ``RuntimeError``,
    outside ``LLM_TRANSPORT_ERRORS``, and a stage that swallows everything else
    would read an outage as "found nothing" and let the row be marked done.

    No ``_is_usage_limit(text)`` sniffing on this path: real quota errors
    arrive as status 402/429 (mapped above). Scanning completion CONTENT
    misfires when the extraction subject itself discusses usage limits (a KG
    entity list containing "usage limit" raised a false UsageLimitError and a
    5-minute backoff, seen live 2026-07-17). The Claude-CLI backend sniffs only
    the CLI's own quota notices (``_is_usage_limit_result``) — text is the only
    quota signal there.
    """
    from pydantic_ai import capture_run_messages

    @retry(
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(
            (TransientLLMHTTPError, PydanticAIModelAPIError, httpx.TransportError)
        ),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
        reraise=True,
    )
    def _attempt() -> Any:
        with capture_run_messages() as captured:
            try:
                # One persistent loop per worker thread (see _thread_loop)
                # so the thread-local connection pool always runs on the
                # loop that created it, even if another sync-async bridge
                # left a closed loop on this thread.
                return _thread_loop().run_until_complete(
                    agent.run(user_prompt, instructions=instructions)
                )
            except PydanticAIModelHTTPError as exc:
                raise _map_model_http_error(exc) from exc
            except UnexpectedModelBehavior as exc:
                if structured and "output retries" in str(exc).lower():
                    raw = _last_response_text(list(captured))
                    if not raw.strip():
                        # Every attempt came back EMPTY. That is not the model deciding
                        # there was nothing to extract — it is the model never speaking
                        # (unrecognised id, dead auth). Retryable, not a content result.
                        raise LLMUnavailableError(
                            f"model returned no output on any attempt: {exc}"
                        ) from exc
                    raise MalformedResponseError(str(exc), raw_response=raw) from exc
                # Empty/malformed completion on the text path keeps its existing class;
                # LLM_TRANSPORT_ERRORS covers both, so the queue treats them alike.
                raise LLMHTTPError(f"empty or invalid completion: {exc}") from exc

    try:
        return _attempt()
    except (PydanticAIModelAPIError, httpx.TransportError) as exc:
        raise TransientLLMHTTPError(f"chat completions unreachable after retries: {exc}") from exc


class _OpenAIMessagesProxy:
    """``OpenAIChatClient.messages`` — same call shape as ``_MessagesProxy``.

    Each ``create`` runs one plain-text pydantic-ai Agent against the
    client's chat-completions model, preserving the legacy text surface for
    callers that haven't moved to ``structured_call`` (dream/*, notes,
    preferences gate, ``parse_with_retry``). Legacy ``response_format``
    dicts still travel in-prompt as schema text (``_with_json_schema``, as on
    the CLI path) because arbitrary hand-written dicts can't be trusted to
    satisfy a provider's strict json_schema mode.
    """

    def __init__(self, client: OpenAIChatClient) -> None:
        self._client = client

    def create(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> _Response:
        from pydantic_ai import Agent

        c = self._client
        instructions, prompt = _flatten_chat(messages, system=system)
        if response_format is not None:
            prompt = _with_json_schema(prompt, response_format.get("schema", {}))

        # retries=0: this surface has no output validation, so pydantic-ai
        # retries could only re-fire empty completions — the legacy client
        # raised on the first empty, keep that.
        agent = Agent(
            c.pydantic_ai_model(model),
            output_type=str,
            model_settings=c.model_settings(max_tokens, model),
            retries=0,
        )
        result = _run_agent_sync(agent, prompt, instructions=instructions, structured=False)
        text = str(result.output)
        if response_format is not None:
            return _Response(_strip_json_fence(text))
        return _Response(text)


class OpenAIChatClient:
    """OpenAI-compatible chat-completions backend, driven by pydantic-ai.

    ``messages.create(model=..., max_tokens=..., messages=..., system=...,
    response_format=...)`` returns an object with ``.content[0].text`` —
    so ``parse_with_retry`` and every legacy call site work unchanged.
    Structured call sites go through :func:`structured_call`, which builds a
    pydantic-ai Agent with ``NativeOutput`` on this client's model.

    Auth: bearer token via the OpenAI SDK (an ``"EMPTY"`` placeholder is
    substituted when the key is blank — the SDK requires one; keyless
    endpoints like a local Ollama ignore it). The key is never logged and
    never appears in raised errors.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_OPENAI_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:

        self.model = model
        self.base_url = base_url.rstrip("/")
        # OpenRouter accepts vendor-specific ``reasoning`` / ``provider``
        # request fields; other OpenAI-compatible servers (OpenAI proper,
        # Ollama, vLLM) may reject unknown params, so gate on the URL.
        self.is_openrouter = "openrouter" in base_url.lower()
        # SYNAPSE_OPENROUTER_PROVIDERS: comma-separated provider slugs to
        # prefer (e.g. "fireworks,deepinfra"), or "auto" to rank providers
        # cheapest-first from the live endpoint listing (see
        # _ranked_openrouter_providers). OpenRouter-only.
        raw_providers = os.environ.get("SYNAPSE_OPENROUTER_PROVIDERS", "").strip()
        self.openrouter_providers_auto = raw_providers.lower() == "auto"
        self.openrouter_providers = (
            []
            if self.openrouter_providers_auto
            else [p.strip() for p in raw_providers.split(",") if p.strip()]
        )
        self._api_key = api_key
        self._transport = transport
        # Connection state is THREAD-LOCAL, built lazily per worker thread.
        # A process-wide httpx.AsyncClient looked fine under MockTransport
        # tests but broke in prod (2026-07-18): each _run_agent_sync call
        # runs in its own short-lived event loop, and pooled keepalive
        # connections created under one loop fail with APIConnectionError
        # ("Connection error.") when reused from the next — ~570 failures
        # in 10 min. One loop + one pool per long-lived worker thread
        # (see _thread_loop) keeps connections and loops married for life.
        self._tls = threading.local()
        self.messages = _OpenAIMessagesProxy(self)

    @property
    def _provider(self) -> Any:
        from openai import AsyncOpenAI
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = getattr(self._tls, "provider", None)
        if provider is None:
            inner = self._transport if self._transport is not None else httpx.AsyncHTTPTransport()
            http_client = httpx.AsyncClient(
                transport=_ErrorBodyStatusTransport(inner), timeout=_HTTP_TIMEOUT
            )
            # max_retries=0: retry policy lives in tenacity (_run_agent_sync),
            # not the OpenAI SDK, so attempt counts stay deterministic.
            openai_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self._api_key or "EMPTY",
                http_client=http_client,
                max_retries=0,
                timeout=_HTTP_TIMEOUT,
            )
            provider = OpenAIProvider(openai_client=openai_client)
            self._tls.provider = provider
        return provider

    def resolve_model(self, model: str | None) -> str:
        """Per-call model: honored when the call site chose one explicitly —
        ``stage_model()`` resolutions are provider-valid by construction. The
        bare Claude code default is the "unspecified" sentinel (call sites
        that never chose, e.g. ``parse_with_retry``'s default) and maps to
        the client's single configured model, which this provider recognises.
        """
        if not model or model == DEFAULT_MODEL:
            return self.model
        return model

    def pydantic_ai_model(self, model: str | None = None) -> Any:
        """A pydantic-ai ``OpenAIChatModel`` over this client's connection."""
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(
            self.resolve_model(model),
            provider=self._provider,
            profile=_openai_compat_profile(),
        )

    def model_settings(self, max_tokens: int, model: str | None = None) -> Any:
        """Per-call model settings, carrying the OpenRouter must-haves. ``model`` is
        the call's model as passed to ``pydantic_ai_model``: the provider order is
        ranked for the model that actually goes on the wire."""
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        settings = OpenAIChatModelSettings(max_tokens=max_tokens)
        if self.is_openrouter:
            extra: dict[str, Any] = {}
            # Every call through this client is an extraction-shaped task
            # that wants the completion, not chain-of-thought. Reasoning
            # models (DeepSeek V4, o-series) can spend the entire
            # ``max_tokens`` budget on reasoning tokens and return an EMPTY
            # completion with ``finish_reason='length'`` — 196 queue items
            # failed exactly that way on deepseek-v4-pro (2026-07-17).
            extra["reasoning"] = {"enabled": False}
            providers = (
                _ranked_openrouter_providers(self.resolve_model(model))
                if self.openrouter_providers_auto
                else self.openrouter_providers
            )
            if providers:
                # Prefer these providers in order; OpenRouter still falls
                # back to others if none are available, so a provider
                # outage degrades to slower/pricier routing, not failure.
                extra["provider"] = {"order": providers}
            settings["extra_body"] = extra
        return settings
