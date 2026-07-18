"""Tests for the Claude Agent SDK pydantic-ai connector.

The SDK boundary (``agent_call``) is monkeypatched — no real ``claude`` CLI is
ever spawned. Covers message flattening, structured-output translation to the
SDK's ``output_format``, tool-definition rejection, streaming rejection, and
usage-limit propagation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

import ingestion.pydantic_ai_claude_sdk as sdk_mod
from ingestion.llm_client import (
    ClaudeCLIClient,
    MalformedResponseError,
    UsageLimitError,
    structured_call,
)
from ingestion.pydantic_ai_claude_sdk import ClaudeAgentSDKModel, flatten_messages


class _Verdict(BaseModel):
    contradicted_facts: list[int]


def _fake_agent_call(seen: dict[str, Any], response_text: str):
    async def fake(
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str = "m",
        max_turns: int = 3,
        semaphore: Any = None,
        log_name: str | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> str:
        seen.update(
            prompt=prompt,
            system=system_prompt,
            model=model,
            output_format=output_format,
        )
        return response_text

    return fake


class TestFlattenMessages:
    def test_system_and_user_parts_split(self):
        system, prompt = flatten_messages(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(content="be a judge"),
                        UserPromptPart(content="judge this"),
                    ]
                )
            ]
        )
        assert system == "be a judge"
        assert prompt == "judge this"

    def test_retry_history_appended_in_order(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="original")]),
            ModelResponse(parts=[TextPart("bad json")]),
            ModelRequest(parts=[RetryPromptPart(content="fix it")]),
        ]
        system, prompt = flatten_messages(messages)
        assert system is None
        assert prompt.index("original") < prompt.index("bad json") < prompt.index("fix it")


class TestRequest:
    def test_structured_output_translated_to_sdk_output_format(self, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            sdk_mod,
            "agent_call",
            _fake_agent_call(seen, '{"contradicted_facts": [3]}'),
        )
        result = structured_call(
            ClaudeCLIClient(),
            output_model=_Verdict,
            base_prompt="judge this",
            system="be a judge",
            model="claude-haiku-4-5",
        )
        assert result.contradicted_facts == [3]
        assert seen["model"] == "claude-haiku-4-5"
        # System text travels via agent_call's system_prompt flattening.
        assert "be a judge" in seen["system"]
        assert seen["prompt"] == "judge this"
        # Native output mode → the SDK's {"type": "json", "schema": ...}.
        assert seen["output_format"]["type"] == "json"
        assert "contradicted_facts" in seen["output_format"]["schema"]["properties"]

    def test_trailing_prose_is_tolerated(self, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            sdk_mod,
            "agent_call",
            _fake_agent_call(seen, '{"contradicted_facts": [1]} — hope that helps!'),
        )
        result = structured_call(ClaudeCLIClient(), output_model=_Verdict, base_prompt="x")
        assert result.contradicted_facts == [1]

    def test_usage_limit_propagates(self, monkeypatch):
        async def limit_call(prompt: str, **kwargs: Any) -> str:
            raise UsageLimitError("claude usage limit reached")

        monkeypatch.setattr(sdk_mod, "agent_call", limit_call)
        with pytest.raises(UsageLimitError):
            structured_call(ClaudeCLIClient(), output_model=_Verdict, base_prompt="x")

    def test_unparseable_output_maps_to_malformed(self, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr(sdk_mod, "agent_call", _fake_agent_call(seen, "not json"))
        with pytest.raises(MalformedResponseError):
            structured_call(
                ClaudeCLIClient(), output_model=_Verdict, base_prompt="x", max_attempts=1
            )

    def test_function_tools_rejected(self):
        model = ClaudeAgentSDKModel("claude-haiku-4-5")
        params = ModelRequestParameters(
            function_tools=[ToolDefinition(name="t", parameters_json_schema={"type": "object"})]
        )
        with pytest.raises(UserError, match="does not support function tools"):
            asyncio.run(model.request([], None, params))

    def test_streaming_rejected(self):
        model = ClaudeAgentSDKModel("claude-haiku-4-5")
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            asyncio.run(model.request_stream())

    def test_plain_text_request_passes_through(self, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr(sdk_mod, "agent_call", _fake_agent_call(seen, "plain answer"))
        model = ClaudeAgentSDKModel("claude-haiku-4-5")
        response = asyncio.run(
            model.request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])],
                None,
                ModelRequestParameters(),
            )
        )
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "plain answer"
        assert seen["output_format"] is None
