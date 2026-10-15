"""The Claude Code backend: ``agent_call`` runs one prompt through the Agent SDK's bundled
CLI, and ``ClaudeAgentClient`` is its sync, anthropic-shaped wrapper (``messages.create``).

Auth is a Claude subscription OAuth token or ``ANTHROPIC_API_KEY``, consumed by the spawned
``claude`` CLI. The transport retry policy and the error taxonomy are
``ingestion.llm_common``'s; ``ingestion.llm_client`` is the module callers import from.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from tenacity import AsyncRetrying, before_sleep_log

from ingestion.llm_common import (
    _RETRY_STOP,
    _RETRY_WAIT,
    DEFAULT_MODEL,
    LLMUnavailableError,
    UsageLimitError,
    _flatten_chat,
    _Response,
    _strip_json_fence,
    _with_json_schema,
)
from ingestion.llm_schemas import first_json_object

logger = logging.getLogger(__name__)

# Cap on extended-thinking reasoning tokens per call (maps to the CLI's
# ``--max-thinking-tokens``). The extraction pipeline's calls are
# structured-classification tasks (extract-into-schema, dedup, contradiction)
# — they do NOT need deep chain-of-thought. Left unset, Haiku 4.5 spent
# ~12-16K reasoning tokens *per call* (measured 2026-06-05 in Logfire:
# 12.4K reasoning tokens to answer one yes/no dedup), which dominated both
# cost and Max-quota burn. A modest cap keeps enough headroom for the hard
# entity-equivalence cases while killing the runaway reasoning. Tune here.
_MAX_THINKING_TOKENS = 2048

# One CLI exchange (spawn, prompt, result) may take at most this long — the
# same budget as the HTTP backend's read timeout (_HTTP_TIMEOUT). Without it a
# hung CLI pins a drain thread, and the batch waiting on it, for good.
_CLI_TIMEOUT_S = 300.0

# Substrings that indicate the Max-subscription session is over its window.
# Matched case-insensitively — but only against what can be a CLI quota
# notice, never against a completion (see ``_is_usage_limit_result``). The CLI
# builds its notice as "You've hit your <session|weekly|Opus|Sonnet|...> limit
# · resets 5pm" (also "You've hit your limit" when overage is rejected),
# "You're out of extra usage · resets 5pm" (an individual account whose extra-usage
# credits ran out) and "Your org is out of usage · add funds to continue"; its
# warnings ("You've used 90% ...", "You're close to ...") are not exhaustion.
_USAGE_LIMIT_PATTERNS = (
    "usage limit reached",
    "your limit will reset at",
    "rate_limit_error",
    "credit balance is too low",
    "out of usage",
    "out of extra usage",
    "quota exceeded",
    "you've hit your",
)

# A quota notice is one short line ("You've hit your session limit · resets 5pm");
# a reply longer than this is a completion, whatever it mentions.
_USAGE_NOTICE_MAX_CHARS = 300


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
        # model reads untrusted transcripts and scraped pages. (The feature-flagged
        # Monitor / PushNotification / RemoteTrigger survive it; the env below
        # removes those.)
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
        env={
            # Each call is a new process with a unique prompt, so the CLI's
            # prompt-cache write (1.25-2x the input price) is never read back.
            "DISABLE_PROMPT_CACHING": "1",
            # No session-title request to Haiku (it carries the whole prompt, so
            # every call would bill its input twice) and no feature-flagged
            # tools (~2K tokens): background traffic a one-shot call never needs.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
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
