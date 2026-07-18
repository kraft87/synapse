"""pydantic-ai Model connector for the Claude Agent SDK (subscription CLI).

Lets extraction stages run as backend-agnostic pydantic-ai Agents: the same
``Agent(output_type=...)`` call sites drive either the OpenAI-compatible HTTP
backend (OpenRouter etc.) or — through this connector — the local ``claude``
CLI via ``claude_agent_sdk``, which pydantic-ai cannot speak natively.

Design: ``request()`` flattens the pydantic-ai message history into a single
prompt string (system parts + user parts, exactly the flattening
``agent_call`` applies today) and delegates the actual SDK round-trip to
``ingestion.llm_client.agent_call``. Delegation — not reimplementation — is
deliberate: ``agent_call`` carries the hard-won guards this path must never
lose:

* ``setting_sources=[]`` — a host whose user settings carry the Stop ingest
  hook otherwise ships every extraction transcript into /ingest (1,895 judge
  episodes purged 2026-06-12).
* ``allowed_tools=[]`` and ``max_turns=3`` (>= 3 so the SDK's
  StructuredOutput tool can resolve before the stop hook fires).
* ``_MAX_THINKING_TOKENS`` cap (Haiku spent 12-16K reasoning tokens per
  yes/no dedup call when uncapped).
* ``_is_usage_limit(text)`` content sniffing → ``UsageLimitError``. The CLI
  has no HTTP status codes, so response text is the ONLY quota signal on
  this path. (PR #74 removed sniffing on the HTTP path only, where real
  quota errors arrive as status 402/429 — keep it here.)
* CLI path resolution and the SDK-level retry loop.

Structured output: pydantic-ai's ``native`` output mode is translated to the
SDK's ``output_format={"type": "json", "schema": ...}``; ``agent_call``
returns ``ResultMessage.structured_output`` re-serialized as JSON text (or
fence-stripped text when the SDK falls back to plain text), which pydantic-ai
then validates against the output model.

Not supported (both raise, loudly, at request time):

* Function tools — the CLI backend cannot do tool-call rounds; extraction
  never registers tools.
* Streaming — extraction is batch-shaped; nothing streams.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from ingestion.llm_client import DEFAULT_MODEL, agent_call


def _part_text(content: Any) -> str:
    """Flatten a prompt part's content to text. Extraction only ever sends
    strings; sequence content (multi-modal) keeps its string members."""
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return "\n".join(c for c in content if isinstance(c, str))
    return str(content)


def flatten_messages(messages: list[ModelMessage]) -> tuple[str | None, str]:
    """Collapse a pydantic-ai message history into (system_prompt, prompt).

    Mirrors ``agent_call``'s flattening contract: system text is prepended by
    ``agent_call`` itself (``f"{system}\\n\\n{prompt}"``). Retry feedback
    (validation errors quoted back by pydantic-ai) and prior assistant text
    are appended in order so the model sees the correction context, matching
    the retry-with-feedback shape ``parse_with_retry`` uses on this backend.
    """
    system_parts: list[str] = []
    prompt_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    system_parts.append(_part_text(part.content))
                elif isinstance(part, UserPromptPart):
                    prompt_parts.append(_part_text(part.content))
                elif isinstance(part, RetryPromptPart):
                    prompt_parts.append(part.model_response())
        elif isinstance(message, ModelResponse):
            for rp in message.parts:
                if isinstance(rp, TextPart) and rp.content:
                    prompt_parts.append(f"(your previous response)\n{rp.content}")
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, "\n\n".join(p for p in prompt_parts if p)


class ClaudeAgentSDKModel(Model):
    """pydantic-ai ``Model`` over the Claude Agent SDK / subscription CLI."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__(
            profile=ModelProfile(
                # The SDK's output_format handles JSON-schema output natively.
                supports_json_schema_output=True,
                default_structured_output_mode="native",
                # No function-tool support (see request()).
                supports_tools=False,
            )
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "claude-agent-sdk"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if model_request_parameters.function_tools:
            raise UserError(
                "ClaudeAgentSDKModel does not support function tools — the CLI "
                "backend cannot do tool-call rounds. Use the OpenAI-compatible "
                "backend (SYNAPSE_LLM_PROVIDER=openai) for tool-using agents."
            )

        system, prompt = flatten_messages(messages)
        # pydantic-ai delivers Agent/run instructions separately from the
        # message history; fold them into the system text.
        instructions = "\n\n".join(
            p.content for p in (model_request_parameters.instruction_parts or []) if p.content
        )
        if instructions:
            system = f"{system}\n\n{instructions}" if system else instructions

        output_format: dict[str, Any] | None = None
        output_object = model_request_parameters.output_object
        if model_request_parameters.output_mode == "native" and output_object is not None:
            output_format = {"type": "json", "schema": output_object.json_schema}

        # agent_call carries every CLI guard (setting_sources=[], allowed_tools=[],
        # max_turns=3, thinking cap, usage-limit sniffing) — see module docstring.
        text = await agent_call(
            prompt,
            system_prompt=system,
            model=self._model_name,
            output_format=output_format,
        )

        if output_format is not None:
            # Tolerate prose around the JSON object (the SDK's plain-text
            # fallback can ramble) so pydantic-ai's output validation sees
            # clean JSON — same posture as the legacy raw_decode call sites.
            from ingestion.llm_schemas import first_json_object

            try:
                text = first_json_object(text)
            except ValueError:
                pass  # leave as-is; output validation will retry/fail

        return ModelResponse(parts=[TextPart(text)], model_name=self._model_name)

    async def request_stream(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError(
            "ClaudeAgentSDKModel does not support streaming — extraction calls "
            "are batch-shaped; use request()/run_sync."
        )


__all__ = ["ClaudeAgentSDKModel", "flatten_messages"]
