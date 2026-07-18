"""LLM client backed by claude-agent-sdk, with an OpenAI-compatible fallback.

Single module providing two interfaces over the Claude Code Agent SDK
(no subprocess to ``claude --print`` — the SDK manages the CLI handle):

  * ``ClaudeCLIClient`` — sync, drop-in replacement that mimics
    ``anthropic.Anthropic()``'s ``messages.create()`` shape. Existing
    callers in ``ingestion/extractor.py``, ``dream/*``, etc. work
    unchanged.

  * ``agent_call`` / ``agent_call_batch`` — native async API with an
    asyncio.Semaphore for parallel concurrency control. Use these from
    any async-aware caller that wants to fan out N independent calls.

Plus an alternative backend for users without a Claude subscription:

  * ``OpenAIChatClient`` — same ``messages.create()`` shape, but speaks
    the plain OpenAI ``/chat/completions`` protocol over HTTP (OpenRouter,
    Ollama, vLLM, or any compatible endpoint). Selected via env:

        SYNAPSE_LLM_PROVIDER=openai         # default: claude-code
        SYNAPSE_LLM_BASE_URL=...            # default: https://openrouter.ai/api/v1
        SYNAPSE_LLM_API_KEY=...             # blank OK for keyless endpoints (Ollama)
        SYNAPSE_LLM_MODEL=...               # default: anthropic/claude-haiku-4.5

    Model resolution: every pipeline stage resolves its model through
    ``stage_model()`` — ``SYNAPSE_<STAGE>_MODEL`` beats ``SYNAPSE_LLM_MODEL``
    beats the stage's code default. In openai mode a per-call model is
    honored only when the call site chose one (stage envs produce
    provider-valid ids); the bare Claude code default falls back to the
    client's configured model, so legacy call sites can't send an id the
    provider doesn't recognise.

  * ``create_llm_client`` — the factory every construction site goes
    through; picks the backend from ``SYNAPSE_LLM_PROVIDER``.

Follows the same Agent-SDK client pattern used across the Synapse stack
for Claude Code interaction at scale.

Auth: a Claude subscription OAuth token or ``ANTHROPIC_API_KEY``, consumed
by the spawned ``claude`` CLI (or an OpenAI-compatible endpoint via
``SYNAPSE_LLM_PROVIDER=openai``).

Resilience (Phase 5)
--------------------
Every ``messages.create`` call is wrapped in a ``tenacity`` retry that
mirrors Graphiti's client-layer pattern (``graphiti_core/llm_client/
client.py``): exponential backoff on transient SDK errors (rate limits,
network blips, timeouts) so EVERY caller — dedup, contradiction, edge
dates, extractor, dream/* — inherits resilience for free without
repeating the boilerplate at each call site.

What's retried, and what isn't:
  * Retried — ``anthropic.RateLimitError``, ``APIConnectionError``,
    ``APITimeoutError``, and the SDK's untyped transient ``Exception``s
    that bubble out of the lower-level ``agent_call`` retry loop.
  * NOT retried — ``BadRequestError`` (prompt is malformed; retrying
    burns budget), ``AuthenticationError``, and the custom
    ``UsageLimitError`` (Max-subscription window exhausted; the caller
    needs to STOP the cycle, not retry).

For structured-output JSON validation, ``parse_with_retry`` re-fires the
call with the malformed response appended to the prompt as feedback —
mirrors Graphiti's ``generate_response`` retry-with-validation-error loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fence(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences from a model response."""
    return _FENCE_RE.sub("", text.strip()).strip()


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
_MAX_RETRIES = 2

# Cap on extended-thinking reasoning tokens per call (maps to the CLI's
# ``--max-thinking-tokens``). The extraction pipeline's calls are
# structured-classification tasks (extract-into-schema, dedup, contradiction)
# — they do NOT need deep chain-of-thought. Left unset, Haiku 4.5 spent
# ~12-16K reasoning tokens *per call* (measured 2026-06-05 in Logfire:
# 12.4K reasoning tokens to answer one yes/no dedup), which dominated both
# cost and Max-quota burn. A modest cap keeps enough headroom for the hard
# entity-equivalence cases while killing the runaway reasoning. Tune here.
_MAX_THINKING_TOKENS = 2048

# Tenacity retry config — direct port of Graphiti's defaults
# (``client.py::_generate_response_with_retry``). 3 attempts total,
# exponential 2s → 4s → 8s capped at 30s. Tight enough to recover from
# a transient rate-limit blip; bounded enough not to stall the pipeline.
_RETRY_STOP = stop_after_attempt(3)
_RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=30)

# Substrings that indicate the Max-subscription session is over its window.
# Matched case-insensitively against the raw text the CLI sends back.
#
# NB: this is the *content-based* detector — it triggers when the CLI
# returns a SUCCESS response whose text body looks like a rate-limit
# message (the false-positive case where the model paraphrases the
# limit notice). The tenacity retry above catches the *structured*
# ``anthropic.RateLimitError`` exception path — a different code path
# from a different SDK error mode. Keep both.
_USAGE_LIMIT_PATTERNS = (
    "claude usage limit reached",
    "your limit will reset at",
    "rate_limit_error",
    "credit balance is too low",
    "out of usage credits",
    "quota exceeded",
    "you've hit your limit",
)


class UsageLimitError(Exception):
    """Raised when the Claude CLI reports usage / rate-limit exhaustion.

    Callers (e.g. the extraction queue drainer) should treat this as a signal
    to STOP the current cycle and leave items in ``pending`` for later retry,
    rather than marking them failed and burning the rest of the queue.

    Distinct from ``anthropic.RateLimitError``: that's a transient
    structured exception worth retrying; this signals the whole
    Max-subscription window is gone — retrying within the window just
    burns more time, the caller should pause until reset.
    """


class MalformedResponseError(Exception):
    """Raised by ``parse_with_retry`` when the LLM response can't be parsed.

    The retry-with-feedback loop catches this internally and re-fires the
    call with the original failure quoted back to the model. After the
    final attempt the same exception propagates so the caller can decide
    whether to drop the item or fall through to a default value.
    """

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class LLMHTTPError(Exception):
    """Non-transient HTTP failure from an OpenAI-compatible endpoint.

    Carries the status code and a short body snippet so operators can see
    WHY a call failed (e.g. a 400 schema complaint) without grepping
    provider logs. Never includes request headers, so the API key can't
    leak into logs or exception chains.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TransientLLMHTTPError(LLMHTTPError):
    """Retryable HTTP failure (429 rate limit, 5xx) — tenacity retries these.

    After retries exhaust, the error still RAISES (reraise=True). A 429
    must never degrade into an empty-text result: a past incident had
    OpenRouter 402s swallowed as empty extraction output, silently
    producing zero facts. See project_openrouter_credits_exhausted.
    """


def _is_usage_limit(text: str | None) -> bool:
    """Content-based usage-limit detector.

    Pairs with ``UsageLimitError`` above — see that class's docstring for
    why this lives separately from the tenacity retry on
    ``anthropic.RateLimitError``.
    """
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in _USAGE_LIMIT_PATTERNS)


# ---------------------------------------------------------------------------
# Transient exception detection for tenacity
# ---------------------------------------------------------------------------
#
# The Claude Code Agent SDK does not export typed transient errors; it
# wraps everything in plain ``Exception``s. We still want tenacity to
# retry the structured ``anthropic.*`` exceptions when callers wire those
# in directly (e.g. tests that pass a real ``anthropic.Anthropic()`` to
# a deduper). Import them lazily so the SDK is optional at runtime — if
# the import fails, the tuple just falls back to ``UsageLimitError``
# (which is excluded from retry below) so no spurious retries happen.

try:
    from anthropic import (
        APIConnectionError as _AnthropicAPIConnectionError,
    )
    from anthropic import (
        APITimeoutError as _AnthropicAPITimeoutError,
    )
    from anthropic import (
        RateLimitError as _AnthropicRateLimitError,
    )

    _TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
        _AnthropicRateLimitError,
        _AnthropicAPIConnectionError,
        _AnthropicAPITimeoutError,
    )
except ImportError:  # pragma: no cover - anthropic is a direct dep
    _TRANSIENT_ERRORS = ()


def _is_transient(exc: BaseException) -> bool:
    """Return True for SDK-level transient errors worth retrying.

    Explicitly excludes:
      * ``UsageLimitError`` — whole window is exhausted; retrying within
        the window just burns time.
      * ``MalformedResponseError`` — that's a content / parse failure;
        the structured-feedback retry in ``parse_with_retry`` handles
        it, not the wire-level retry here.
      * ``BadRequestError`` / ``AuthenticationError`` — non-transient,
        retrying loops on the same failure.
    """
    if isinstance(exc, UsageLimitError | MalformedResponseError):
        return False
    return isinstance(exc, _TRANSIENT_ERRORS)


def _find_cli_path() -> str:
    """Locate the latest installed Claude Code CLI version."""
    versions_dir = Path.home() / ".local/share/claude/versions"
    if versions_dir.exists():
        versions = sorted(versions_dir.iterdir(), key=lambda p: p.name, reverse=True)
        if versions:
            return str(versions[0])
    return "claude"  # fallback to PATH


_CLI_PATH = _find_cli_path()


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


async def agent_call(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    semaphore: asyncio.Semaphore | None = None,
    log_name: str | None = None,
    output_format: dict[str, Any] | None = None,
) -> str:
    """Single async call to the Agent SDK. Returns the response text.

    ``semaphore`` (optional) gates concurrency when several ``agent_call``s
    run inside the same event loop. Pass the same instance to all callers
    that should share a quota.

    ``output_format`` (optional) constrains the response to a JSON schema.
    Shape: ``{"type": "json", "schema": {...}}`` — the SDK's structured-output
    pattern. When set, ``ResultMessage.structured_output`` carries the
    parsed dict; we serialize it back to a JSON string for the caller.
    Requires ``max_turns >= 3`` so the SDK's StructuredOutput tool can
    resolve before the stop hook fires.
    """
    full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
    if output_format is not None:
        schema_str = json.dumps(output_format.get("schema", {}))
        full_prompt = (
            f"{full_prompt}\n\n"
            f"Respond with ONLY valid JSON matching this schema — "
            f"no markdown, no explanation:\n{schema_str}"
        )

    for attempt in range(_MAX_RETRIES + 1):
        opts = ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=max_turns,
            model=model,
            cli_path=_CLI_PATH,
            output_format=output_format,
            max_thinking_tokens=_MAX_THINKING_TOKENS,
            # Never load filesystem settings into the spawned CLI. On a host
            # whose user settings carry the Stop ingest hook, every
            # agent_call otherwise ships its own transcript into /ingest and
            # the corpus eats the harness's prompts — 1,895 judge/extraction
            # episodes purged 2026-06-12. Hooks, CLAUDE.md, and output styles
            # are all interactive-session concerns; extraction wants none.
            setting_sources=[],
        )
        ctx_sem = semaphore if semaphore is not None else _NULL_CTX
        async with ctx_sem:
            try:
                async with ClaudeSDKClient(options=opts) as client:
                    await client.query(full_prompt)
                    result_text: str | None = None
                    structured: dict[str, Any] | None = None
                    async for msg in client.receive_response():
                        if isinstance(msg, ResultMessage):
                            result_text = msg.result
                            structured = getattr(msg, "structured_output", None)
                if structured is not None:
                    return json.dumps(structured)
                if result_text is None:
                    raise RuntimeError("agent_call: no ResultMessage")
                if _is_usage_limit(result_text):
                    raise UsageLimitError(result_text.strip()[:300])
                # When output_format is requested but the SDK delivered
                # plain text, strip ```json fences so callers can json.loads
                # the result directly.
                if output_format is not None:
                    return _strip_json_fence(result_text)
                return result_text
            except UsageLimitError:
                raise  # do not retry usage limits — bubble up immediately
            except Exception as e:
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "agent_call(%s) attempt %d/%d failed: %s",
                        log_name,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        str(e)[:120],
                    )
                    continue
                raise RuntimeError(f"agent_call({log_name}): failed after retries: {e}") from e

    raise RuntimeError(f"agent_call({log_name}): unreachable")


async def agent_call_batch(
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 1,
    concurrency: int = 6,
    log_prefix: str | None = None,
) -> list[str]:
    """Run N prompts concurrently, gated by an asyncio.Semaphore."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, p: str) -> str:
        return await agent_call(
            p,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            semaphore=sem,
            log_name=f"{log_prefix}_{i}" if log_prefix else None,
        )

    return await asyncio.gather(*[_one(i, p) for i, p in enumerate(prompts)])


class _NullCtx:
    """No-op async context manager — used when no semaphore is provided."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> None:
        return None


_NULL_CTX = _NullCtx()


# ---------------------------------------------------------------------------
# Sync compat shim — drop-in for callers expecting anthropic SDK shape
# ---------------------------------------------------------------------------


class _Content:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Content(text)]


class _MessagesProxy:
    """``ClaudeCLIClient.messages`` — wraps every ``create`` in tenacity retry.

    The retry decorator lives at this layer (not inside ``agent_call``) so
    that callers using the ``anthropic.Anthropic()`` shape — every dedup,
    contradiction, edge-dates, and dream-pipeline call site — get
    exponential backoff on transient errors for free, without each call
    site repeating its own ``@retry`` boilerplate.
    """

    def __init__(self, default_model: str) -> None:
        self._default_model = default_model

    @retry(
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        # `logger` is a stdlib `Logger`; tenacity's `LoggerProtocol` is
        # structurally compatible (the `log` method differs only in
        # parameter naming), but mypy's strict mode doesn't recognise
        # that. Cast keeps the runtime call identical.
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
        reraise=True,
    )
    def create(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> _Response:
        m = model or self._default_model
        content = ""
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    content = str(msg["content"])

        text = asyncio.run(
            agent_call(
                content,
                system_prompt=system,
                model=m,
                output_format=response_format,
            )
        )
        return _Response(text)


class ClaudeCLIClient:
    """Sync drop-in for ``anthropic.Anthropic()``.

    Internally routes to the async ``agent_call`` via ``asyncio.run``. Each
    call has its own event loop — appropriate for sync callers that don't
    have one already and don't need parallelism. Loop-bound callers should
    use ``agent_call`` / ``agent_call_batch`` directly.

    Resilience: ``messages.create()`` retries transient SDK errors
    (rate-limit, network, timeout) with exponential backoff. Non-transient
    errors (BadRequest, Auth, UsageLimit) propagate immediately.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.messages = _MessagesProxy(model)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend — chat completions over plain HTTP
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "anthropic/claude-haiku-4.5"  # OpenRouter id for Haiku 4.5

_HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_BODY_SNIPPET_LEN = 300


def _body_snippet(response: httpx.Response) -> str:
    """First ``_BODY_SNIPPET_LEN`` chars of the response body, for errors.

    Only ever reads the response BODY — never request headers — so the
    Authorization bearer cannot appear in the message.
    """
    try:
        return response.text[:_BODY_SNIPPET_LEN]
    except Exception:  # pragma: no cover - undecodable body
        return "<undecodable body>"


def _raise_for_openai_status(response: httpx.Response) -> None:
    """Map non-2xx chat-completions responses onto the module's error types.

    * 402 (payment required / out of credits) → ``UsageLimitError`` — the
      caller must STOP the cycle, exactly like a Claude Max window
      exhaustion. Never swallowed as empty text.
    * 429 / 5xx → ``TransientLLMHTTPError`` — retried by tenacity, raised
      after retries exhaust.
    * Other 4xx → ``LLMHTTPError`` — non-transient, raised immediately.
    """
    status = response.status_code
    if 200 <= status < 300:
        return
    snippet = _body_snippet(response)
    message = f"chat completions HTTP {status}: {snippet}"
    if status == 402:
        raise UsageLimitError(message)
    if status == 429 or status >= 500:
        raise TransientLLMHTTPError(message, status_code=status)
    raise LLMHTTPError(message, status_code=status)


class _OpenAIMessagesProxy:
    """``OpenAIChatClient.messages`` — same shape as ``_MessagesProxy``.

    One plain (non-streaming) POST to ``{base_url}/chat/completions`` per
    ``create``. The tenacity retry mirrors ``_MessagesProxy.create``'s
    config (3 attempts, exp backoff 2s → 4s → 8s, cap 30s) but keys on
    HTTP-level transients: ``httpx.TransportError`` (connect/read/timeout)
    and ``TransientLLMHTTPError`` (429, 5xx). ``UsageLimitError`` (402)
    and plain ``LLMHTTPError`` (other 4xx) are never retried.
    """

    def __init__(self, client: OpenAIChatClient) -> None:
        self._client = client

    @retry(
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type((httpx.TransportError, TransientLLMHTTPError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
        reraise=True,
    )
    def create(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> _Response:
        c = self._client
        # Per-call model: honored when the call site chose one explicitly —
        # stage_model() resolutions are provider-valid by construction. The
        # bare Claude code default is the "unspecified" sentinel (call sites
        # that never chose, e.g. parse_with_retry's default) and maps to the
        # client's single configured model, which this provider recognises.
        if not model or model == DEFAULT_MODEL:
            model = c.model

        chat_messages: list[dict[str, str]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for msg in messages or []:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if response_format is not None and role == "user":
                # Providers disagree on structured-output support, so the
                # schema constraint travels in-prompt — same wording as
                # ``agent_call`` uses for the SDK path.
                schema_str = json.dumps(response_format.get("schema", {}))
                content = (
                    f"{content}\n\n"
                    f"Respond with ONLY valid JSON matching this schema — "
                    f"no markdown, no explanation:\n{schema_str}"
                )
            chat_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if c.is_openrouter:
            # Every call through this client is an extraction-shaped task
            # that wants the completion, not chain-of-thought. Reasoning
            # models (DeepSeek V4, o-series) can spend the entire
            # ``max_tokens`` budget on reasoning tokens and return an EMPTY
            # completion with ``finish_reason='length'`` — 196 queue items
            # failed exactly that way on deepseek-v4-pro (2026-07-17).
            payload["reasoning"] = {"enabled": False}

        response = c.http.post("/chat/completions", json=payload)
        _raise_for_openai_status(response)

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMHTTPError(
                f"chat completions: non-JSON 2xx body: {_body_snippet(response)}",
                status_code=response.status_code,
            ) from exc

        # OpenRouter quirk: some failures arrive as HTTP 200 with an
        # ``{"error": {...}}`` body. Map those onto the same error types
        # as their status-code twins — a 200-wrapped 402 must still STOP
        # the cycle, not turn into empty text.
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            err = data.get("error") or {}
            code = err.get("code") if isinstance(err, dict) else None
            err_msg = f"chat completions error body: {json.dumps(err)[:_BODY_SNIPPET_LEN]}"
            if code == 402:
                raise UsageLimitError(err_msg)
            if code == 429 or (isinstance(code, int) and code >= 500):
                raise TransientLLMHTTPError(err_msg, status_code=code)
            raise LLMHTTPError(err_msg, status_code=code if isinstance(code, int) else None)

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMHTTPError(
                f"chat completions: malformed response shape: {_body_snippet(response)}",
                status_code=response.status_code,
            ) from exc

        if not text or not str(text).strip():
            # Empty completions are an error, never a result — the
            # OpenRouter-credits incident started with empties being
            # treated as "the model found nothing".
            raise LLMHTTPError(
                "chat completions: empty completion text "
                f"(finish_reason={data['choices'][0].get('finish_reason')!r})",
                status_code=response.status_code,
            )

        text = str(text)
        # No _is_usage_limit(text) sniffing here: on the HTTP path real quota
        # errors arrive as status 402/429 (mapped above). Scanning completion
        # CONTENT misfires when the extraction subject itself discusses usage
        # limits — a KG entity list containing "usage limit" raised a false
        # UsageLimitError and a 5-minute backoff (seen live 2026-07-17).
        if response_format is not None:
            return _Response(_strip_json_fence(text))
        return _Response(text)


class OpenAIChatClient:
    """OpenAI-compatible chat-completions backend, same shape as ``ClaudeCLIClient``.

    ``messages.create(model=..., max_tokens=..., messages=..., system=...,
    response_format=...)`` returns an object with ``.content[0].text`` —
    so ``parse_with_retry`` and every existing call site work unchanged.

    Auth: bearer token in the ``Authorization`` header (omitted when the
    key is blank, e.g. a local Ollama). The key is never logged and never
    appears in raised errors.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_OPENAI_MODEL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        # OpenRouter accepts a vendor-specific ``reasoning`` request field;
        # other OpenAI-compatible servers (OpenAI proper, Ollama, vLLM) may
        # reject unknown params, so the flag gates on the URL.
        self.is_openrouter = "openrouter" in base_url.lower()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_HTTP_TIMEOUT,
            transport=transport,
        )
        self.messages = _OpenAIMessagesProxy(self)


# ---------------------------------------------------------------------------
# Factory — backend selection via SYNAPSE_LLM_PROVIDER
# ---------------------------------------------------------------------------


def stage_model(stage: str, default: str = DEFAULT_MODEL) -> str:
    """Resolve the LLM model for a named pipeline stage (issue #8).

    Precedence: ``SYNAPSE_<STAGE>_MODEL`` → ``SYNAPSE_LLM_MODEL`` → *default*
    (in openai-provider mode the fallback is ``DEFAULT_OPENAI_MODEL``, never a
    bare Claude id the provider wouldn't recognise).

    Stages: EXTRACTOR, TIMELINE, PREFERENCES, DEDUP, CONTRADICTION,
    EDGE_DATES, DREAM, QUERY_GRAPH, NOTES_CONFIRM. The A/B work behind this (Flash-vs-Haiku,
    DeepSeek-vs-Haiku) showed model choice matters per stage — a cheap model
    can be fine for binary confirms while extraction wants a stronger one.
    """
    v = os.environ.get(f"SYNAPSE_{stage.upper()}_MODEL", "").strip()
    if v:
        return v
    v = os.environ.get("SYNAPSE_LLM_MODEL", "").strip()
    if v:
        return v
    provider = os.environ.get("SYNAPSE_LLM_PROVIDER", "claude-code").strip().lower()
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return default


def create_llm_client(model: str = DEFAULT_MODEL) -> ClaudeCLIClient | OpenAIChatClient:
    """Build the extraction LLM client from env. All construction sites route here.

    ``SYNAPSE_LLM_PROVIDER``:

    * ``claude-code`` (default, also blank) — ``ClaudeCLIClient`` over the
      Agent SDK; auth via the local ``claude`` CLI login,
      ``CLAUDE_CODE_OAUTH_TOKEN``, or ``ANTHROPIC_API_KEY``. ``model`` is
      the per-client default Claude model name.
    * ``openai`` — ``OpenAIChatClient`` against
      ``SYNAPSE_LLM_BASE_URL`` (default OpenRouter) with
      ``SYNAPSE_LLM_API_KEY``; ``SYNAPSE_LLM_MODEL`` is the configured
      default model. Per-call models resolved via ``stage_model()``
      (``SYNAPSE_<STAGE>_MODEL``) are honored; unresolved per-call Claude
      ids fall back to the configured model.
    """
    provider = os.environ.get("SYNAPSE_LLM_PROVIDER", "claude-code").strip().lower()
    if provider in ("", "claude-code"):
        return ClaudeCLIClient(model=model)
    if provider == "openai":
        return OpenAIChatClient(
            base_url=os.environ.get("SYNAPSE_LLM_BASE_URL", DEFAULT_OPENAI_BASE_URL)
            or DEFAULT_OPENAI_BASE_URL,
            api_key=os.environ.get("SYNAPSE_LLM_API_KEY", ""),
            model=os.environ.get("SYNAPSE_LLM_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL,
        )
    raise ValueError(
        f"Unknown SYNAPSE_LLM_PROVIDER={provider!r} — expected 'claude-code' or 'openai'"
    )


# ---------------------------------------------------------------------------
# parse_with_retry — structured JSON parsing with retry-on-feedback
# ---------------------------------------------------------------------------


def parse_with_retry[T](
    llm_client: Any,
    *,
    base_prompt: str,
    parser: Callable[[str], T],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    response_format: dict[str, Any] | None = None,
    max_attempts: int = 3,
    system: str | None = None,
) -> T:
    """Call the LLM and parse the response, retrying with feedback on parse failure.

    Mirrors Graphiti's ``generate_response`` retry-with-feedback loop
    (``graphiti_core/llm_client/anthropic_client.py::generate_response``).
    Behaviour:

    1. Send ``base_prompt`` to the LLM.
    2. Run ``parser(response_text)``. If it succeeds, return the result.
    3. If the parser raises ``MalformedResponseError`` (or any
       ``ValueError`` / ``json.JSONDecodeError`` / ``ValidationError``),
       append the failure to the next prompt as feedback and try again.
    4. After ``max_attempts``, re-raise the last failure.

    The parser MUST raise an exception (preferred:
    ``MalformedResponseError``) on bad input; returning ``None`` is
    treated as a successful parse to a ``None`` value, NOT a retry signal.

    Note: this does NOT replicate the wire-level retry inside
    ``_MessagesProxy.create`` — the two layers compose. A transient
    ``RateLimitError`` is handled by tenacity inside ``create``; a parse
    failure is handled by this loop on top.
    """
    feedback = ""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + feedback
        response = llm_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system,
            response_format=response_format,
        )
        raw = str(response.content[0].text)
        try:
            return parser(raw)
        except MalformedResponseError as exc:
            last_error = exc
            feedback = (
                "\n\nYour last response failed to parse: "
                f"{exc}. Output ONLY valid JSON matching the schema above."
            )
            if attempt < max_attempts:
                logger.warning(
                    "parse_with_retry attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    str(exc)[:120],
                )
                continue
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            # Wrap raw parser exceptions in MalformedResponseError so the
            # caller catches a uniform type after retries exhaust.
            wrapped = MalformedResponseError(str(exc), raw_response=raw)
            last_error = wrapped
            feedback = (
                "\n\nYour last response failed to parse: "
                f"{exc}. Output ONLY valid JSON matching the schema above."
            )
            if attempt < max_attempts:
                logger.warning(
                    "parse_with_retry attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    str(exc)[:120],
                )
                continue
            raise wrapped from exc

    # Defensive — the for-loop returns or raises on each iteration.
    if last_error is not None:  # pragma: no cover - unreachable
        raise last_error
    raise RuntimeError("parse_with_retry: unreachable")  # pragma: no cover
