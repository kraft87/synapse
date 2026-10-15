"""LLM client backed by claude-agent-sdk, with an OpenAI-compatible fallback.

Single module providing two interfaces over the Claude Code Agent SDK
(no subprocess to ``claude --print`` — the SDK manages the CLI handle):

  * ``ClaudeAgentClient`` — sync, drop-in replacement that mimics
    ``anthropic.Anthropic()``'s ``messages.create()`` shape. Existing
    callers in ``ingestion/extractor.py``, ``dream/*``, etc. work
    unchanged.

  * ``agent_call`` — the native async call underneath it: one prompt, one
    CLI process, the response text back.

Plus an alternative backend for users without a Claude subscription:

  * ``OpenAIChatClient`` — same ``messages.create()`` shape, driven by
    pydantic-ai (``OpenAIChatModel`` over the OpenAI SDK) against any
    ``/chat/completions``-compatible endpoint (OpenRouter, Ollama, vLLM).
    Selected via env:

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

  * ``structured_call`` — the ONE structured-output surface: builds a
    pydantic-ai Agent (``NativeOutput`` + a stage model from
    ``ingestion.llm_schemas``) on whichever backend the client selects —
    ``OpenAIChatModel`` for the HTTP path, ``ClaudeAgentSDKModel``
    (``ingestion.pydantic_ai_claude_sdk``) for the subscription CLI, and the
    same text-call adapter over ``messages.create`` for anything else — and
    returns a validated model instance.

Auth: a Claude subscription OAuth token or ``ANTHROPIC_API_KEY``, consumed
by the spawned ``claude`` CLI (or an OpenAI-compatible endpoint via
``SYNAPSE_LLM_PROVIDER=openai``).

Resilience
----------
Retries are layered by failure mode, at the client so every caller —
dedup, contradiction, edge dates, extractor, dream/* — inherits them:

  * Transport failures — one ``tenacity`` policy (3 attempts, exponential
    backoff) around the wire call of each backend: the CLI exchange inside
    ``agent_call`` (crash, hang past ``_CLI_TIMEOUT_S``, no result), and the
    HTTP call inside ``_run_agent_sync`` (429/5xx, connection errors,
    timeouts). Once retries are exhausted the failure is always a member of
    ``LLM_TRANSPORT_ERRORS``, so no stage can mistake an outage for "found
    nothing".
  * Content failures (schema-invalid output) — pydantic-ai's output
    validation retries on the structured path; ``parse_with_retry``'s
    retry-with-feedback loop on the legacy text path.
  * NOT retried — malformed-prompt 4xx (retrying burns budget), auth
    errors, the CLI's own error results (it already retried API errors), and
    ``UsageLimitError`` (quota window exhausted; the caller needs to STOP the
    cycle, not retry).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from pydantic import BaseModel
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError as PydanticAIModelAPIError
from pydantic_ai.exceptions import ModelHTTPError as PydanticAIModelHTTPError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.llm_schemas import first_json_object

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fence(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences from a model response."""
    return _FENCE_RE.sub("", text.strip()).strip()


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


DEFAULT_MODEL = "claude-haiku-4-5"

# Cap on extended-thinking reasoning tokens per call (maps to the CLI's
# ``--max-thinking-tokens``). The extraction pipeline's calls are
# structured-classification tasks (extract-into-schema, dedup, contradiction)
# — they do NOT need deep chain-of-thought. Left unset, Haiku 4.5 spent
# ~12-16K reasoning tokens *per call* (measured 2026-06-05 in Logfire:
# 12.4K reasoning tokens to answer one yes/no dedup), which dominated both
# cost and Max-quota burn. A modest cap keeps enough headroom for the hard
# entity-equivalence cases while killing the runaway reasoning. Tune here.
_MAX_THINKING_TOKENS = 2048

# Transport retry policy shared by both backends (agent_call's CLI exchange,
# _run_agent_sync's HTTP call): 3 attempts total, exponential 2s → 4s → 8s
# capped at 30s. Tight enough to recover from a transient blip; bounded
# enough not to stall the pipeline.
_RETRY_STOP = stop_after_attempt(3)
_RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=30)

# One CLI exchange (spawn, prompt, result) may take at most this long — the
# same budget as the HTTP backend's read timeout (_HTTP_TIMEOUT). Without it a
# hung CLI pins a drain thread, and the batch waiting on it, for good.
_CLI_TIMEOUT_S = 300.0

# Substrings that indicate the Max-subscription session is over its window.
# Matched case-insensitively — but only against what can be a CLI quota
# notice, never against a completion (see ``_is_usage_limit_result``).
_USAGE_LIMIT_PATTERNS = (
    "usage limit reached",
    "your limit will reset at",
    "rate_limit_error",
    "credit balance is too low",
    "out of usage credits",
    "quota exceeded",
    "you've hit your limit",
)

# A quota notice is one short line ("You've hit your limit · resets 5pm");
# a reply longer than this is a completion, whatever it mentions.
_USAGE_NOTICE_MAX_CHARS = 300


class UsageLimitError(Exception):
    """Raised when the Claude CLI reports usage / rate-limit exhaustion.

    Callers (e.g. the extraction queue drainer) should treat this as a signal
    to STOP the current cycle and leave items in ``pending`` for later retry,
    rather than marking them failed and burning the rest of the queue.

    Distinct from a transient 429 (``TransientLLMHTTPError``, retried):
    this signals the whole Max-subscription window is gone — retrying within
    the window just burns more time, the caller should pause until reset.
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


class LLMUnavailableError(Exception):
    """The LLM produced no usable output at all — a transport/config failure, not content.

    The distinction that matters to the queue: ``MalformedResponseError`` means the model
    ANSWERED and the answer did not parse (a content decision — a conservative no-op is a
    fair reading of it), while this means the call never really happened. Every attempt
    came back blank: a model id the backend does not recognise (``claude -p --model
    anthropic/claude-haiku-4.5`` exits 0 with an empty completion), a missing or expired
    token, an endpoint that is not there. Or every attempt failed in transport: the CLI
    crashed, hung, or ended without a result.

    Nothing about the item was decided, so a stage must not swallow this into "found
    nothing" and let the row be marked done. It raises, the row is marked failed with the
    reason, and the queue retries it once the config is fixed.
    """


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


#: "The model never really answered" — the failures a pipeline stage must NOT absorb
#: into a no-op decision. A stage that catches broadly (the timeline / preferences gates,
#: the dedup confirms) re-raises these so ``process_item`` fails, the queue row is marked
#: failed with the reason, and the item is retried later. ``UsageLimitError`` rides along
#: because its handling is stricter still, not looser: the poller releases the batch back
#: to ``pending`` and backs off, which a swallow inside a gate silently prevented.
LLM_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    LLMUnavailableError,
    LLMHTTPError,
    UsageLimitError,
)


def _is_usage_limit(text: str | None) -> bool:
    """Does ``text`` contain one of the known quota phrases?"""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in _USAGE_LIMIT_PATTERNS)


def _is_usage_limit_result(text: str, *, is_error: bool) -> bool:
    """Is this CLI result a quota notice rather than a completion?

    The CLI has no status codes, so its result text is the only quota signal on
    this path. But a completion can quote a quota message: Synapse ingests its
    own development sessions, so an extraction about this module legitimately
    contains "quota exceeded" (the HTTP path hit exactly that false positive on
    2026-07-17). A false UsageLimitError makes the drain release the batch, back
    off up to 30 minutes and re-claim the same item first, forever.

    The CLI reports quota notices as error results, so an ``is_error`` result is
    checked. Anything else counts only in the shape of a notice: short, with no
    JSON object in it — never the shape of an extraction.
    """
    if not _is_usage_limit(text):
        return False
    if is_error:
        return True
    if len(text.strip()) > _USAGE_NOTICE_MAX_CHARS:
        return False
    try:
        first_json_object(text)
    except ValueError:
        return True
    return False


def _with_json_schema(prompt: str, schema: dict[str, Any]) -> str:
    """Append a JSON-schema response contract to ``prompt``.

    The text surfaces' only enforcement: the CLI gets no ``--json-schema`` (the
    SDK forwards one only for ``output_format`` type ``json_schema``, which adds a
    StructuredOutput tool turn), and hand-written ``response_format`` dicts can't
    be trusted to satisfy a provider's strict json_schema mode. Callers validate
    the reply themselves.
    """
    return (
        f"{prompt}\n\n"
        f"Respond with ONLY valid JSON matching this schema — "
        f"no markdown, no explanation:\n{json.dumps(schema)}"
    )


def _flatten_chat(
    messages: list[dict[str, Any]] | None,
    *,
    system: str | None = None,
    prompt: str | None = None,
) -> tuple[str | None, str]:
    """Collapse anthropic-shaped chat messages into ``(instructions, prompt)``.

    ``system`` and every system-role message join the instructions; ``prompt``
    and every other message's content join the prompt, in order. One flattening
    for every text surface, so the backends can't diverge on what reaches the
    model.
    """
    instruction_parts: list[str] = [system] if system else []
    user_parts: list[str] = [prompt] if prompt else []
    for msg in messages or []:
        content = str(msg.get("content", ""))
        if str(msg.get("role", "user")) == "system":
            instruction_parts.append(content)
        else:
            user_parts.append(content)
    return "\n\n".join(instruction_parts) or None, "\n\n".join(user_parts)


# ---------------------------------------------------------------------------
# Claude CLI backend (Agent SDK)
# ---------------------------------------------------------------------------

# The SDK runs ``claude -v`` before every session (one extra process spawn,
# ~120 ms) only to warn about a CLI older than it supports. agent_call runs the
# SDK's own bundled CLI, pinned with it by uv.lock, so the warning can't fire.
os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


def _cli_options(model: str) -> ClaudeAgentOptions:
    """The options every ``agent_call`` spawns the CLI with.

    A one-shot extraction call wants none of an interactive session's features,
    and each one costs tokens, time or data hygiene. No ``cli_path``: the SDK
    runs its bundled CLI, pinned by uv.lock, rather than whatever is installed.
    """
    return ClaudeAgentOptions(
        # No built-in tools. allowed_tools=[] only removes auto-approval: without
        # this the CLI still ships its ~23 tool definitions (about 10K prompt
        # tokens on every call, measured) and Read/Grep stay callable while the
        # model reads untrusted transcripts and scraped pages.
        tools=[],
        # Nothing can use a second turn once there are no tools.
        max_turns=1,
        model=model,
        max_thinking_tokens=_MAX_THINKING_TOKENS,
        # Never load filesystem settings into the spawned CLI. On a host
        # whose user settings carry the Stop ingest hook, every
        # agent_call otherwise ships its own transcript into /ingest and
        # the corpus eats the harness's prompts — 1,895 judge/extraction
        # episodes purged 2026-06-12. Hooks, CLAUDE.md, and output styles
        # are all interactive-session concerns; extraction wants none.
        setting_sources=[],
        # Each call is a new process with a unique prompt, so the CLI's
        # prompt-cache write (1.25-2x the input price) is never read back.
        env={"DISABLE_PROMPT_CACHING": "1"},
        extra_args={
            # The CLI otherwise saves every prompt, turn text included, as a
            # session transcript under ~/.claude/projects: disk churn, and a
            # copy of conversation text outside Postgres that deletes can't reach.
            "no-session-persistence": None,
            # No MCP servers from user or project config.
            "strict-mcp-config": None,
        },
    )


async def _cli_exchange(prompt: str, model: str) -> tuple[str, bool]:
    """One CLI session: send ``prompt``, return the result text and its ``is_error`` flag.

    Every failure in here is a transport failure (crash, hang, no result) and
    ``agent_call`` retries it.
    """
    result: ResultMessage | None = None
    async with asyncio.timeout(_CLI_TIMEOUT_S):
        async with ClaudeSDKClient(options=_cli_options(model)) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    result = msg
    if result is None or result.result is None:
        raise RuntimeError("claude CLI ended the session without a result")
    return result.result, bool(result.is_error)


async def agent_call(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    json_schema: dict[str, Any] | None = None,
) -> str:
    """One prompt through the Claude CLI; returns the response text.

    ``json_schema`` (optional) is appended to the prompt as the response
    contract (``_with_json_schema``) and the reply comes back with any ```json
    fence stripped. The caller validates it.

    Raises:
      * ``UsageLimitError`` — the CLI reported the subscription window or the
        credit balance exhausted. Not retried.
      * ``LLMUnavailableError`` — the CLI answered with its own error (no login,
        an unrecognised model, an API error it already retried), or the exchange
        failed (crash, hang past ``_CLI_TIMEOUT_S``, no result) on every attempt
        of the transport retry policy (``_RETRY_STOP``, backoff ``_RETRY_WAIT``).
    """
    full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
    if json_schema is not None:
        full_prompt = _with_json_schema(full_prompt, json_schema)

    retrying = AsyncRetrying(
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT,
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
        reraise=True,
    )
    try:
        exchange: tuple[str, bool] = await retrying(_cli_exchange, full_prompt, model)
    except Exception as e:
        raise LLMUnavailableError(
            f"claude CLI failed on every attempt: {type(e).__name__}: {e}"
        ) from e

    text, is_error = exchange
    if _is_usage_limit_result(text, is_error=is_error):
        raise UsageLimitError(text.strip()[:300])
    if is_error:
        # The CLI answered, but the "answer" is its own error text — no
        # login ("Not logged in · Please run /login"), an unrecognised
        # model id, a spawn failure. Callers must not parse that as a
        # model response: it looks exactly like a content failure
        # (unparseable text) while being a config failure that will
        # repeat identically until someone fixes the environment.
        raise LLMUnavailableError(f"claude CLI returned an error result: {text.strip()[:300]}")
    if json_schema is not None:
        return _strip_json_fence(text)
    return text


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
    """``ClaudeAgentClient.messages`` — the anthropic-shaped text surface over
    ``agent_call``, which owns the retry policy."""

    def __init__(self, default_model: str) -> None:
        self._default_model = default_model

    def create(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> _Response:
        del max_tokens  # the CLI has no output-token flag
        instructions, prompt = _flatten_chat(messages, system=system)
        text = asyncio.run(
            agent_call(
                prompt,
                system_prompt=instructions,
                model=model or self._default_model,
                json_schema=None if response_format is None else response_format.get("schema", {}),
            )
        )
        return _Response(text)


class ClaudeAgentClient:
    """Sync drop-in for ``anthropic.Anthropic()``.

    Internally routes to the async ``agent_call`` via ``asyncio.run``. Each
    call has its own event loop — appropriate for sync callers that don't
    have one already and don't need parallelism. Loop-bound callers should
    await ``agent_call`` directly.

    Resilience: ``agent_call`` retries a failed CLI exchange with backoff and
    then raises ``LLMUnavailableError``; the CLI's own error results and
    ``UsageLimitError`` propagate immediately.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.messages = _MessagesProxy(model)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend — pydantic-ai over chat completions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Factory — backend selection via SYNAPSE_LLM_PROVIDER
# ---------------------------------------------------------------------------


def _provider() -> str:
    return os.environ.get("SYNAPSE_LLM_PROVIDER", "claude-code").strip().lower()


def check_model_id(model: str, var: str = "SYNAPSE_LLM_MODEL") -> str:
    """Reject an OpenRouter-style model id on the claude-code backend. Returns *model*.

    The two backends spell the same model differently, and only one of them says so when
    it is handed the wrong spelling. ``claude -p --model anthropic/claude-haiku-4.5``
    exits 0 with ``[claude-code:unrecognized_model]`` and an EMPTY completion, so the
    failure surfaces four layers away as "no JSON object in response" on every extraction
    stage — a wrong config that reads as a broken pipeline. (.env.example shipped exactly
    that id, uncommented, under the default provider.)

    Cheap and total: a Claude CLI model name never contains a slash, and a provider-
    routed id always does.
    """
    if model and "/" in model and _provider() in ("", "claude-code"):
        raise ValueError(
            f"{var}={model!r} is an OpenRouter-style model id, which the Claude Code CLI "
            "rejects: it returns [claude-code:unrecognized_model] with empty output, and "
            "every extraction stage then fails with 'no JSON object in response'.\n"
            "With SYNAPSE_LLM_PROVIDER=claude-code use the CLI's own spelling, e.g. "
            f"{DEFAULT_MODEL!r} (dashes, no vendor prefix). Slashed ids like "
            f"{DEFAULT_OPENAI_MODEL!r} belong to SYNAPSE_LLM_PROVIDER=openai.\n"
            "Leaving the variable unset is also fine — each stage has a working default."
        )
    return model


def stage_model(stage: str, default: str = DEFAULT_MODEL) -> str:
    """Resolve the LLM model for a named pipeline stage (issue #8).

    Precedence: ``SYNAPSE_<STAGE>_MODEL`` → ``SYNAPSE_LLM_MODEL`` → *default*
    (in openai-provider mode the fallback is ``DEFAULT_OPENAI_MODEL``, never a
    bare Claude id the provider wouldn't recognise).

    Stages: EXTRACTOR, TIMELINE, PREFERENCES, DEDUP, CONTRADICTION, EDGE_DATES,
    NOTES_CONFIRM, NOTES_CURATE, SKILLS_JUDGE, SKILLS_DRAFT, CONTEXTUALIZE. The A/B
    work behind this (Flash-vs-Haiku, DeepSeek-vs-Haiku) showed model choice matters
    per stage — a cheap model can be fine for binary confirms while extraction wants
    a stronger one.
    """
    stage_var = f"SYNAPSE_{stage.upper()}_MODEL"
    v = os.environ.get(stage_var, "").strip()
    if v:
        return check_model_id(v, stage_var)
    v = os.environ.get("SYNAPSE_LLM_MODEL", "").strip()
    if v:
        return check_model_id(v)
    if _provider() == "openai":
        return DEFAULT_OPENAI_MODEL
    return default


def create_llm_client(model: str = DEFAULT_MODEL) -> ClaudeAgentClient | OpenAIChatClient:
    """Build the extraction LLM client from env. All construction sites route here.

    ``SYNAPSE_LLM_PROVIDER``:

    * ``claude-code`` (default, also blank) — ``ClaudeAgentClient`` over the
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
    provider = _provider()
    if provider in ("", "claude-code"):
        # Startup validation: every construction site routes here, so a mis-spelled
        # model id fails loudly at client creation instead of as empty completions.
        return ClaudeAgentClient(model=check_model_id(model, "SYNAPSE_LLM_MODEL"))
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
# structured_call — backend-agnostic structured outputs via pydantic-ai
# ---------------------------------------------------------------------------


def _snake_name(name: str) -> str:
    """CamelCase → snake_case for the json_schema ``name`` field."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name.lstrip("_")).lower()


def _resolve_pydantic_ai_model(llm_client: Any, model: str) -> Any:
    """Pick the pydantic-ai Model behind a Synapse LLM client.

    The same env selection that picks the backend today
    (``SYNAPSE_LLM_PROVIDER`` → ``create_llm_client``) transitively picks the
    pydantic-ai Model here: ``OpenAIChatClient`` → ``OpenAIChatModel`` over
    its OpenRouter/OpenAI-compatible connection, ``ClaudeAgentClient`` →
    ``ClaudeAgentSDKModel`` over the subscription CLI. Anything else (test
    doubles, third-party clients with the ``messages.create(model=...,
    max_tokens=..., messages=..., system=..., response_format=...) ->
    .content[0].text`` duck type) gets the same text-call adapter over its
    ``messages.create``, with native output delivered as the legacy
    ``{"type": "json", "schema": ...}`` response_format.
    """
    from ingestion.pydantic_ai_claude_sdk import ClaudeAgentSDKModel, TextCallModel

    if isinstance(llm_client, OpenAIChatClient):
        return llm_client.pydantic_ai_model(model)
    if isinstance(llm_client, ClaudeAgentClient):
        return ClaudeAgentSDKModel(model)

    async def messages_create(
        prompt: str, system: str | None, json_schema: dict[str, Any] | None, max_tokens: int
    ) -> str:
        response = llm_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system,
            response_format=None
            if json_schema is None
            else {"type": "json", "schema": json_schema},
        )
        return str(response.content[0].text)

    return TextCallModel(model, messages_create, system="legacy-client")


def structured_call[M: BaseModel](
    llm_client: Any,
    *,
    output_model: type[M],
    base_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    max_attempts: int = 1,
) -> M:
    """One structured-output LLM call → a validated ``output_model`` instance.

    The single structured path for every pipeline stage: builds a pydantic-ai
    ``Agent`` with ``NativeOutput(output_model, strict=True)`` on whichever
    backend ``llm_client`` selects (see ``_resolve_pydantic_ai_model``). On
    the OpenRouter path this sends the provider-native
    ``response_format={"type": "json_schema", ...}`` (strict json_schema —
    Fireworks supports it) AND the schema in-prompt for providers that ignore
    ``response_format``; on the Claude-CLI path the schema travels in-prompt
    (``agent_call``'s ``json_schema``).

    ``max_attempts`` bounds output-validation attempts (``max_attempts - 1``
    pydantic-ai retries with the validation error quoted back — the same
    retry-with-feedback contract ``parse_with_retry`` provides for text).

    Raises:
      * ``MalformedResponseError`` — response never validated (call sites map
        this onto their stage's conservative no-op fallback).
      * ``UsageLimitError`` / ``TransientLLMHTTPError`` / ``LLMHTTPError`` —
        same wire-level taxonomy as ``messages.create``.
    """
    from pydantic_ai import Agent, NativeOutput
    from pydantic_ai.settings import ModelSettings

    instructions, prompt = _flatten_chat(messages, system=system, prompt=base_prompt)

    if isinstance(llm_client, OpenAIChatClient):
        settings: Any = llm_client.model_settings(max_tokens, model)
    else:
        settings = ModelSettings(max_tokens=max_tokens)

    stage = _snake_name(output_model.__name__)
    agent = Agent(
        _resolve_pydantic_ai_model(llm_client, model),
        output_type=NativeOutput(output_model, name=stage, strict=True),
        model_settings=settings,
        retries=max(0, max_attempts - 1),
        # Names the run's span ("invoke_agent <stage>") where pydantic-ai is
        # instrumented, so token use and cost can be read per stage.
        name=stage,
    )
    result = _run_agent_sync(agent, prompt, instructions=instructions, structured=True)
    return cast(M, result.output)


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

    Note: this does NOT replicate the backends' transport retries — the two
    layers compose. A transport failure is retried inside ``messages.create``
    (``agent_call`` / ``_run_agent_sync``); a parse failure is handled by this
    loop on top.
    """
    feedback = ""
    last_error: Exception | None = None
    #: Was there EVER text to parse? All-blank across every attempt means the backend
    #: never answered (wrong model id, dead auth), which is a retryable transport
    #: failure — not the parse failure the caller is entitled to degrade on.
    saw_text = False

    def _exhausted(exc: Exception) -> Exception:
        if saw_text:
            return exc
        return LLMUnavailableError(
            f"model returned no output on any of {max_attempts} attempt(s) (model={model!r}): {exc}"
        )

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
        saw_text = saw_text or bool(raw.strip())
        try:
            return parser(raw)
        except (MalformedResponseError, ValueError) as exc:  # JSONDecodeError is a ValueError
            # Wrap raw parser exceptions so the caller catches a uniform type
            # after retries exhaust.
            failure = (
                exc
                if isinstance(exc, MalformedResponseError)
                else MalformedResponseError(str(exc), raw_response=raw)
            )
            last_error = failure
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
            raise _exhausted(failure) from exc

    # Defensive — the for-loop returns or raises on each iteration.
    if last_error is not None:  # pragma: no cover - unreachable
        raise last_error
    raise RuntimeError("parse_with_retry: unreachable")  # pragma: no cover
