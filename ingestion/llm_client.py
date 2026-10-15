"""The LLM client surface: the factory, ``stage_model``, ``structured_call`` and
``parse_with_retry``, over two backends and one shared error taxonomy.

The modules, bottom up:

  * ``ingestion.llm_common`` — the exception classes (re-exported here), the transport retry
    policy and the helpers every text surface shares.

  * ``ingestion.llm_claude_cli`` — ``ClaudeAgentClient``, a sync drop-in that mimics
    ``anthropic.Anthropic()``'s ``messages.create()`` shape (no subprocess to
    ``claude --print``: the Claude Code Agent SDK manages the CLI handle), and
    ``agent_call``, the native async call underneath it: one prompt, one CLI process, the
    response text back.

  * ``ingestion.llm_openai`` — the alternative backend for users without a Claude
    subscription. ``OpenAIChatClient`` has the same ``messages.create()`` shape, driven by
    pydantic-ai (``OpenAIChatModel`` over the OpenAI SDK) against any ``/chat/completions``-
    compatible endpoint (OpenRouter, Ollama, vLLM). Selected via env:

        SYNAPSE_LLM_PROVIDER=openai         # default: claude-code
        SYNAPSE_LLM_BASE_URL=...            # default: https://openrouter.ai/api/v1
        SYNAPSE_LLM_API_KEY=...             # blank OK for keyless endpoints (Ollama)
        SYNAPSE_LLM_MODEL=...               # default: anthropic/claude-haiku-4.5

Here:

  * ``create_llm_client`` — the factory every construction site goes through; picks the
    backend from ``SYNAPSE_LLM_PROVIDER``.

  * Model resolution: every pipeline stage resolves its model through ``stage_model()`` —
    ``SYNAPSE_<STAGE>_MODEL`` beats ``SYNAPSE_LLM_MODEL`` beats the stage's code default. In
    openai mode a per-call model is honored only when the call site chose one (stage envs
    produce provider-valid ids); the bare Claude code default falls back to the client's
    configured model, so legacy call sites can't send an id the provider doesn't recognise.

  * ``structured_call`` — the ONE structured-output surface: builds a pydantic-ai Agent
    (``NativeOutput`` + a stage model from ``ingestion.llm_schemas``) on whichever backend
    the client selects — ``OpenAIChatModel`` for the HTTP path, ``ClaudeAgentSDKModel``
    (``ingestion.pydantic_ai_claude_sdk``) for the subscription CLI, and the same text-call
    adapter over ``messages.create`` for anything else — and returns a validated model
    instance.

Auth: a Claude subscription OAuth token or ``ANTHROPIC_API_KEY``, consumed by the spawned
``claude`` CLI (or an OpenAI-compatible endpoint via ``SYNAPSE_LLM_PROVIDER=openai``).

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

import logging
import os
import re
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from ingestion.llm_claude_cli import ClaudeAgentClient, agent_call
from ingestion.llm_common import (
    DEFAULT_MODEL,
    LLM_TRANSPORT_ERRORS,
    LLMHTTPError,
    LLMUnavailableError,
    MalformedResponseError,
    TransientLLMHTTPError,
    UsageLimitError,
    _flatten_chat,
    extract_json_object,
)
from ingestion.llm_openai import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    OpenAIChatClient,
    _run_agent_sync,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "LLM_TRANSPORT_ERRORS",
    "ClaudeAgentClient",
    "LLMHTTPError",
    "LLMUnavailableError",
    "MalformedResponseError",
    "OpenAIChatClient",
    "TransientLLMHTTPError",
    "UsageLimitError",
    "agent_call",
    "check_model_id",
    "create_llm_client",
    "extract_json_object",
    "parse_with_retry",
    "stage_model",
    "structured_call",
]

logger = logging.getLogger(__name__)

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
