"""What the LLM backends share: the error taxonomy, the transport retry policy and the text
helpers every anthropic-shaped surface uses.

``ingestion.llm_client`` is the module callers import from; it re-exports the exception
classes below. They live here so the Claude CLI backend (``llm_claude_cli``) and the
OpenAI-compatible one (``llm_openai``) can raise them without importing the factory that
imports both.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tenacity import stop_after_attempt, wait_exponential

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fence(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences from a model response."""
    return _FENCE_RE.sub("", text.strip()).strip()


DEFAULT_MODEL = "claude-haiku-4-5"

# Transport retry policy shared by both backends (agent_call's CLI exchange,
# _run_agent_sync's HTTP call): 3 attempts total, exponential 2s → 4s → 8s
# capped at 30s. Tight enough to recover from a transient blip; bounded
# enough not to stall the pipeline.
_RETRY_STOP = stop_after_attempt(3)
_RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=30)


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


def extract_json_object(text: str) -> dict[str, Any]:
    """The outermost JSON object in an LLM reply (first ``{`` to last ``}``), as the first
    step of a ``parse_with_retry`` parser. Prose or a code fence around it is ignored;
    anything else raises ``MalformedResponseError`` so the retry quotes the failure back."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"JSON decode error: {exc}", text[:200]) from exc
    if not isinstance(obj, dict):
        raise MalformedResponseError("expected a JSON object", text[:200])
    return obj


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


class _Content:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Content(text)]
