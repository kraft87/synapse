"""Tests for the OpenAI-compatible extraction-LLM backend.

Covers:

* ``create_llm_client`` provider selection from ``SYNAPSE_LLM_PROVIDER``
  (claude-code default, openai, unknown → ValueError).
* ``OpenAIChatClient.messages.create`` happy path through
  ``parse_with_retry`` — request shape (path, auth header, payload),
  fence stripping, and the anthropic-shaped ``.content[0].text`` result.
* HTTP error surfacing: status + body snippet in the message, the API
  key NEVER echoed.
* 402 → ``UsageLimitError`` and 429 → raised after retries — neither is
  ever swallowed as empty text (the OpenRouter-credits incident).
* Single-model mode: ``SYNAPSE_LLM_MODEL`` / the configured model
  overrides per-call Claude model names.

No network: every test uses ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from ingestion.llm_client import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    ClaudeCLIClient,
    LLMHTTPError,
    MalformedResponseError,
    OpenAIChatClient,
    TransientLLMHTTPError,
    UsageLimitError,
    create_llm_client,
    parse_with_retry,
    stage_model,
    structured_call,
)

_API_KEY = "sk-or-test-SECRET-key"


def _completion_body(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "gen-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _client_with(
    handler,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    api_key: str = _API_KEY,
) -> OpenAIChatClient:
    return OpenAIChatClient(
        base_url=DEFAULT_OPENAI_BASE_URL,
        api_key=api_key,
        model=model,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def _no_retry_sleep():
    """Short-circuit tenacity's backoff sleep so retry tests don't stall."""
    with patch("tenacity.nap.time.sleep"):
        yield


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_default_is_claude_code(self, monkeypatch):
        monkeypatch.delenv("SYNAPSE_LLM_PROVIDER", raising=False)
        assert isinstance(create_llm_client(), ClaudeCLIClient)

    def test_explicit_claude_code(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "claude-code")
        assert isinstance(create_llm_client(), ClaudeCLIClient)

    def test_blank_provider_is_claude_code(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "")
        assert isinstance(create_llm_client(), ClaudeCLIClient)

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "openai")
        monkeypatch.delenv("SYNAPSE_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("SYNAPSE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("SYNAPSE_LLM_MODEL", raising=False)
        client = create_llm_client()
        assert isinstance(client, OpenAIChatClient)
        assert client.model == DEFAULT_OPENAI_MODEL
        assert client.base_url == DEFAULT_OPENAI_BASE_URL

    def test_openai_env_overrides(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("SYNAPSE_LLM_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("SYNAPSE_LLM_MODEL", "qwen2.5:14b")
        client = create_llm_client()
        assert isinstance(client, OpenAIChatClient)
        assert client.model == "qwen2.5:14b"
        assert "localhost:11434" in client.base_url

    def test_stage_env_beats_global_env(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EXTRACTOR_MODEL", "per-stage/model")
        monkeypatch.setenv("SYNAPSE_LLM_MODEL", "global/model")
        assert stage_model("EXTRACTOR") == "per-stage/model"
        assert stage_model("extractor") == "per-stage/model"  # case-insensitive stage

    def test_global_env_beats_default(self, monkeypatch):
        monkeypatch.delenv("SYNAPSE_EXTRACTOR_MODEL", raising=False)
        monkeypatch.setenv("SYNAPSE_LLM_MODEL", "global/model")
        assert stage_model("EXTRACTOR", "claude-haiku-4-5") == "global/model"

    def test_default_when_no_env_claude_code(self, monkeypatch):
        monkeypatch.delenv("SYNAPSE_EXTRACTOR_MODEL", raising=False)
        monkeypatch.delenv("SYNAPSE_LLM_MODEL", raising=False)
        monkeypatch.delenv("SYNAPSE_LLM_PROVIDER", raising=False)
        assert stage_model("EXTRACTOR", "claude-opus-4-8") == "claude-opus-4-8"

    def test_default_when_no_env_openai_is_provider_valid(self, monkeypatch):
        # In openai mode an unset stage must never resolve to a bare Claude
        # id the provider wouldn't recognise.
        monkeypatch.delenv("SYNAPSE_EXTRACTOR_MODEL", raising=False)
        monkeypatch.delenv("SYNAPSE_LLM_MODEL", raising=False)
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "openai")
        assert stage_model("EXTRACTOR", "claude-opus-4-8") == DEFAULT_OPENAI_MODEL

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "bedrock")
        with pytest.raises(ValueError, match="SYNAPSE_LLM_PROVIDER"):
            create_llm_client()

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "OpenAI")
        assert isinstance(create_llm_client(), OpenAIChatClient)


# ---------------------------------------------------------------------------
# Happy path — request shape and parse_with_retry integration
# ---------------------------------------------------------------------------


class TestSuccessfulCompletion:
    def test_create_returns_anthropic_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_completion_body("hello"))

        client = _client_with(handler)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.content[0].text == "hello"

    def test_request_payload_and_headers(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=777,
            messages=[{"role": "user", "content": "extract facts"}],
            system="you are an extractor",
        )
        assert seen["path"].endswith("/chat/completions")
        assert seen["auth"] == f"Bearer {_API_KEY}"
        assert seen["payload"]["max_tokens"] == 777
        assert seen["payload"]["stream"] is False
        assert seen["payload"]["messages"][0] == {
            "role": "system",
            "content": "you are an extractor",
        }
        assert seen["payload"]["messages"][1]["content"] == "extract facts"

    def test_openrouter_payload_disables_reasoning(self):
        """Reasoning models can burn the whole max_tokens budget on
        chain-of-thought and return an empty completion with
        finish_reason='length'; extraction wants the completion only."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert seen["payload"]["reasoning"] == {"enabled": False}

    def test_openrouter_provider_order_from_env(self, monkeypatch):
        """SYNAPSE_OPENROUTER_PROVIDERS pins preferred providers in order."""
        monkeypatch.setenv("SYNAPSE_OPENROUTER_PROVIDERS", "fireworks, deepinfra")
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert seen["payload"]["provider"] == {"order": ["fireworks", "deepinfra"]}

    def test_no_provider_field_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SYNAPSE_OPENROUTER_PROVIDERS", raising=False)
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert "provider" not in seen["payload"]

    def test_non_openrouter_payload_has_no_reasoning_field(self):
        """Other OpenAI-compatible servers may reject unknown params."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("ok"))

        client = OpenAIChatClient(
            base_url="http://localhost:11434/v1",
            api_key="",
            model=DEFAULT_OPENAI_MODEL,
            transport=httpx.MockTransport(handler),
        )
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning" not in seen["payload"]

    def test_blank_key_sends_placeholder_bearer(self):
        """Keyless endpoints (local Ollama) still work: the OpenAI SDK
        requires *a* key, so a blank env key becomes the ``EMPTY``
        placeholder (vLLM convention) — never an empty bearer, and the real
        env key is never fabricated."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler, api_key="")
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert seen["auth"] == "Bearer EMPTY"

    def test_parse_with_retry_happy_path(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_completion_body('```json\n{"facts": [1, 2]}\n```'))

        client = _client_with(handler)
        result = parse_with_retry(
            client,
            base_prompt="extract",
            parser=json.loads,
            model="claude-haiku-4-5",
            max_tokens=128,
            response_format={"type": "json", "schema": {"type": "object"}},
        )
        assert result == {"facts": [1, 2]}

    def test_parse_with_retry_refires_on_malformed(self):
        """First response unparseable → feedback retry → second parses."""
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["messages"][-1]["content"])
            text = "not json at all" if len(calls) == 1 else '{"ok": true}'
            return httpx.Response(200, json=_completion_body(text))

        client = _client_with(handler)
        result = parse_with_retry(
            client,
            base_prompt="extract",
            parser=json.loads,
            model="claude-haiku-4-5",
        )
        assert result == {"ok": True}
        assert len(calls) == 2
        assert "failed to parse" in calls[1]

    def test_response_format_schema_travels_in_prompt(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body("{}"))

        client = _client_with(handler)
        client.messages.create(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json", "schema": {"required": ["facts"]}},
        )
        user_msg = seen["payload"]["messages"][-1]["content"]
        assert "ONLY valid JSON" in user_msg
        assert '"required": ["facts"]' in user_msg


# ---------------------------------------------------------------------------
# Model resolution — per-call models honored, Claude code-default remapped
# ---------------------------------------------------------------------------


class TestModelOverride:
    def test_configured_model_wins_over_default_sentinel(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler, model="meta-llama/llama-3.3-70b-instruct")
        client.messages.create(
            model="claude-haiku-4-5",  # the bare Claude code default — must NOT hit the wire
            messages=[{"role": "user", "content": "hi"}],
        )
        assert seen["model"] == "meta-llama/llama-3.3-70b-instruct"

    def test_explicit_per_call_model_is_honored(self):
        # stage_model() resolutions (SYNAPSE_<STAGE>_MODEL) arrive as explicit
        # non-default ids and must reach the wire as-is (issue #8).
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler, model="meta-llama/llama-3.3-70b-instruct")
        client.messages.create(
            model="qwen/qwen-2.5-72b-instruct",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert seen["model"] == "qwen/qwen-2.5-72b-instruct"

    def test_omitted_model_uses_configured(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler, model="meta-llama/llama-3.3-70b-instruct")
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert seen["model"] == "meta-llama/llama-3.3-70b-instruct"

    def test_default_model_is_openrouter_haiku_id(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert seen["model"] == "anthropic/claude-haiku-4.5"


# ---------------------------------------------------------------------------
# HTTP errors — status + snippet surfaced, key never echoed, no empties
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    def test_400_raises_with_status_and_snippet(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad schema xyz"}})

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError) as exc_info:
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert "400" in str(exc_info.value)
        assert "bad schema xyz" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_error_message_never_contains_api_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid key"}})

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError) as exc_info:
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert _API_KEY not in str(exc_info.value)
        assert _API_KEY not in repr(exc_info.value)

    def test_400_is_not_retried(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, json={"error": {"message": "nope"}})

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert len(calls) == 1

    def test_402_raises_usage_limit_not_empty(self):
        """OpenRouter out-of-credits must STOP the cycle, never return ''."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                402, json={"error": {"code": 402, "message": "Insufficient credits"}}
            )

        client = _client_with(handler)
        with pytest.raises(UsageLimitError, match="402"):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert len(calls) == 1  # non-transient: no retry burn

    def test_429_retried_then_raised_not_swallowed(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        client = _client_with(handler)
        with pytest.raises(TransientLLMHTTPError, match="429"):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert len(calls) == 3  # tenacity: 3 attempts, then reraise

    def test_429_recovers_on_retry(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=_completion_body("recovered"))

        client = _client_with(handler)
        resp = client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert resp.content[0].text == "recovered"
        assert len(calls) == 3

    def test_500_is_transient(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        resp = client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert resp.content[0].text == "ok"
        assert len(calls) == 2

    def test_connect_error_is_transient(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, json=_completion_body("ok"))

        client = _client_with(handler)
        resp = client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert resp.content[0].text == "ok"
        assert len(calls) == 2

    def test_200_wrapped_402_error_body_raises_usage_limit(self):
        """OpenRouter quirk: HTTP 200 carrying {"error": {"code": 402}}."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"error": {"code": 402, "message": "Insufficient credits"}}
            )

        client = _client_with(handler)
        with pytest.raises(UsageLimitError, match="Insufficient credits"):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])

    def test_empty_completion_raises_not_returns(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_completion_body("", finish_reason="length"))

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError, match="empty or invalid completion"):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])

    def test_non_json_2xx_body_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>login page</html>")

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError, match="empty or invalid completion"):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])

    def test_200_wrapped_unknown_error_code_is_not_retried(self):
        """Error bodies without a usable code map to a non-transient 400."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json={"error": {"message": "weird failure"}})

        client = _client_with(handler)
        with pytest.raises(LLMHTTPError):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert len(calls) == 1

    def test_usage_limit_text_in_completion_is_returned(self):
        """No content sniffing on the HTTP path: quota errors are status
        402/429 here, and completion TEXT legitimately contains phrases like
        "credit balance" when the extraction subject discusses usage limits
        (a real KG-entity completion misfired as UsageLimitError, 2026-07-17)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=_completion_body("Your credit balance is too low to run this.")
            )

        client = _client_with(handler)
        resp = client.messages.create(messages=[{"role": "user", "content": "hi"}])
        assert "credit balance" in resp.content[0].text


# ---------------------------------------------------------------------------
# structured_call — native structured outputs over the HTTP backend
# ---------------------------------------------------------------------------


class _StrictVerdict(BaseModel):
    """Local strict model: no defaults, no coercion — used to exercise the
    validation-failure paths (the tolerant production models in
    ``ingestion.llm_schemas`` rarely fail validation by design)."""

    contradicted_facts: list[int]


class TestStructuredCall:
    def test_native_response_format_and_extra_body_sent(self, monkeypatch):
        """The core of the migration: OpenRouter gets the provider-native
        strict json_schema response_format, plus the incident-hardened
        extra_body fields (reasoning off, provider pinning)."""
        monkeypatch.setenv("SYNAPSE_OPENROUTER_PROVIDERS", "fireworks")
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body('{"contradicted_facts": [1]}'))

        client = _client_with(handler, model="deepseek/deepseek-v4-flash")
        result = structured_call(
            client, output_model=_StrictVerdict, base_prompt="judge", max_tokens=333
        )
        assert result.contradicted_facts == [1]

        payload = seen["payload"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "strict_verdict"
        assert rf["json_schema"]["strict"] is True
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["contradicted_facts"]
        # Incident must-haves survive on the structured path too.
        assert payload["reasoning"] == {"enabled": False}
        assert payload["provider"] == {"order": ["fireworks"]}
        assert payload["max_tokens"] == 333
        # Belt and braces: the schema ALSO travels in-prompt for providers
        # that ignore response_format.
        system_text = " ".join(m["content"] for m in payload["messages"] if m["role"] == "system")
        assert "contradicted_facts" in system_text

    def test_wrapped_indices_coerced_over_http(self):
        from ingestion.llm_schemas import ContradictionVerdict

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_completion_body('{"contradicted_facts": [{"index": 2}, "3", null]}'),
            )

        client = _client_with(handler)
        result = structured_call(client, output_model=ContradictionVerdict, base_prompt="judge")
        assert result.contradicted_facts == [2, 3]

    def test_default_model_sentinel_resolves_to_configured(self):
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return httpx.Response(200, json=_completion_body('{"contradicted_facts": []}'))

        client = _client_with(handler, model="meta-llama/llama-3.3-70b-instruct")
        structured_call(
            client, output_model=_StrictVerdict, base_prompt="x", model="claude-haiku-4-5"
        )
        assert seen["model"] == "meta-llama/llama-3.3-70b-instruct"

    def test_402_maps_to_usage_limit(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402, json={"error": {"code": 402, "message": "Insufficient credits"}}
            )

        client = _client_with(handler)
        with pytest.raises(UsageLimitError):
            structured_call(client, output_model=_StrictVerdict, base_prompt="x")

    def test_validation_failure_degrades_to_malformed_with_raw(self):
        """A response that never validates raises MalformedResponseError —
        the signal every call site maps onto its conservative no-op — with
        the raw text attached (the dedup yes/no legacy path reads it)."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=_completion_body('{"events": "wrong shape"}'))

        client = _client_with(handler)
        with pytest.raises(MalformedResponseError) as exc_info:
            structured_call(client, output_model=_StrictVerdict, base_prompt="x", max_attempts=2)
        assert len(calls) == 2  # one validation-feedback retry, then raise
        assert "wrong shape" in exc_info.value.raw_response

    def test_validation_retry_recovers(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            text = "garbage" if len(calls) == 1 else '{"contradicted_facts": [0]}'
            return httpx.Response(200, json=_completion_body(text))

        client = _client_with(handler)
        result = structured_call(
            client, output_model=_StrictVerdict, base_prompt="x", max_attempts=3
        )
        assert result.contradicted_facts == [0]
        assert len(calls) == 2


class TestThreadLocalConnectionState:
    """One event loop + one connection pool per worker thread — a process-wide
    AsyncClient driven by per-call event loops broke in prod with an
    APIConnectionError storm (2026-07-18): pooled keepalive connections die
    with the loop that created them."""

    def test_provider_is_thread_local(self):
        import threading as _threading

        from ingestion.llm_client import OpenAIChatClient

        client = OpenAIChatClient(api_key="k")
        main_provider = client._provider
        assert client._provider is main_provider  # stable within a thread

        seen: dict[str, Any] = {}

        def worker():
            seen["provider"] = client._provider

        t = _threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["provider"] is not main_provider

    def test_thread_loop_is_persistent_per_thread(self):
        import threading as _threading

        from ingestion.llm_client import _thread_loop

        loop = _thread_loop()
        assert _thread_loop() is loop  # reused, not per-call

        seen: dict[str, Any] = {}

        def worker():
            seen["loop"] = _thread_loop()

        t = _threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["loop"] is not loop
