"""Tests for ``ingestion.llm_client`` — Phase 5 resilience layer.

Covers:

* The tenacity-wrapped ``_MessagesProxy.create`` retries on transient
  errors (``RateLimitError``, ``APIConnectionError``, ``APITimeoutError``)
  and gives up immediately on non-transient errors
  (``BadRequestError``, ``AuthenticationError``).
* The ``parse_with_retry`` helper that re-fires the LLM call with
  feedback on parse failure (mirrors Graphiti's
  ``generate_response`` retry loop).

All tests run without any external services — the SDK is mocked at the
``messages.create`` boundary or, where the retry decorator itself is
under test, at the ``asyncio.run(agent_call(...))`` boundary by
patching ``agent_call`` directly.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from tenacity import wait_exponential

from ingestion.llm_client import (
    _MAX_THINKING_TOKENS,
    ClaudeCLIClient,
    MalformedResponseError,
    _is_usage_limit,
    _MessagesProxy,
    agent_call,
    parse_with_retry,
)

# ---------------------------------------------------------------------------
# Helpers for constructing the structured ``anthropic`` exceptions.
# Their constructors all require a request/response object so we fabricate
# a minimal ``httpx.Request`` once and reuse it.
# ---------------------------------------------------------------------------

_FAKE_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=_FAKE_REQUEST)
    return RateLimitError("rate limit", response=response, body=None)


def _bad_request_error() -> BadRequestError:
    response = httpx.Response(400, request=_FAKE_REQUEST)
    return BadRequestError("bad request", response=response, body=None)


def _auth_error() -> AuthenticationError:
    response = httpx.Response(401, request=_FAKE_REQUEST)
    return AuthenticationError("auth", response=response, body=None)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=_FAKE_REQUEST)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_FAKE_REQUEST)


def _mock_response(text: str) -> Any:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ---------------------------------------------------------------------------
# _MessagesProxy.create — retry on transient errors
# ---------------------------------------------------------------------------


class TestMessagesProxyRetry:
    """``_MessagesProxy.create`` wraps ``agent_call`` with tenacity.

    The retry config matches Graphiti's defaults: 3 attempts, exponential
    backoff (2s → 4s → 8s, capped at 30s), retry only on the structured
    transient exceptions.

    All tests patch ``agent_call`` (the inner SDK boundary) so we don't
    spin up a real CLI.
    """

    def setup_method(self) -> None:
        # Short-circuit the tenacity wait so retries don't actually
        # sleep — we still want to verify the call_count, not the wait.
        # ``_MessagesProxy.create`` references ``_RETRY_WAIT`` at decorator
        # creation time, so we have to patch ``time.sleep`` inside
        # tenacity. Easier: patch sleep on the tenacity nap helper.
        self._sleep_patcher = patch("tenacity.nap.time.sleep")
        self._sleep_patcher.start()

    def teardown_method(self) -> None:
        self._sleep_patcher.stop()

    def test_retries_on_rate_limit_error(self):
        """RateLimitError raised twice, success on third → 3 calls total."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")

        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            if len(call_log) < 3:
                raise _rate_limit_error()
            return "ok"

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            response = proxy.create(messages=[{"role": "user", "content": "hi"}])

        assert response.content[0].text == "ok"
        assert len(call_log) == 3

    def test_retries_on_api_connection_error(self):
        """APIConnectionError is also transient → retried."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")

        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            if len(call_log) < 2:
                raise _connection_error()
            return "ok"

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            response = proxy.create(messages=[{"role": "user", "content": "hi"}])

        assert response.content[0].text == "ok"
        assert len(call_log) == 2

    def test_retries_on_api_timeout_error(self):
        """APITimeoutError is transient → retried."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")

        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            if len(call_log) < 2:
                raise _timeout_error()
            return "ok"

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            response = proxy.create(messages=[{"role": "user", "content": "hi"}])

        assert response.content[0].text == "ok"
        assert len(call_log) == 2

    def test_no_retry_on_bad_request(self):
        """BadRequestError is non-transient → 1 call, exception re-raised."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")
        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            raise _bad_request_error()

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            with pytest.raises(BadRequestError):
                proxy.create(messages=[{"role": "user", "content": "hi"}])

        assert len(call_log) == 1

    def test_no_retry_on_auth_error(self):
        """AuthenticationError is non-transient → 1 call, re-raised."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")
        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            raise _auth_error()

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            with pytest.raises(AuthenticationError):
                proxy.create(messages=[{"role": "user", "content": "hi"}])

        assert len(call_log) == 1

    def test_gives_up_after_three_attempts(self):
        """RateLimitError on every attempt → 3 calls then raise."""
        proxy = _MessagesProxy(default_model="claude-haiku-4-5")
        call_log: list[int] = []

        async def fake_agent_call(*args: Any, **kwargs: Any) -> str:
            call_log.append(1)
            raise _rate_limit_error()

        with patch("ingestion.llm_client.agent_call", fake_agent_call):
            with pytest.raises(RateLimitError):
                proxy.create(messages=[{"role": "user", "content": "hi"}])

        # 3 total attempts (Graphiti default: stop_after_attempt(3)).
        assert len(call_log) == 3


# ---------------------------------------------------------------------------
# Exponential backoff bounds
# ---------------------------------------------------------------------------


def test_exponential_backoff_respects_min_max():
    """``wait_exponential(multiplier=1, min=2, max=30)`` produces 2s → 4s → 8s
    on the first three retries, all within the 2s..30s envelope.

    Uses tenacity's wait helper directly (instead of timing real sleeps)
    so the test is deterministic and fast. Mirrors Graphiti's
    ``client.py`` configuration shape — keeps the spec-mandated bounds
    enforced by the type system instead of by side effect.
    """
    wait = wait_exponential(multiplier=1, min=2, max=30)

    # tenacity passes a RetryCallState; we only need the attempt number.
    state = MagicMock()
    state.outcome = MagicMock()
    state.outcome.failed = True

    waits: list[float] = []
    for attempt in range(1, 8):
        state.attempt_number = attempt
        waits.append(wait(state))

    # Every wait must be in [2, 30] — the spec's bounds.
    assert all(2 <= w <= 30 for w in waits), waits
    # First wait should be 2s (the min, not the raw 2**0 == 1).
    assert waits[0] == 2
    # Higher attempts saturate at the 30s cap.
    assert waits[-1] == 30


# ---------------------------------------------------------------------------
# parse_with_retry — JSON validation retry-with-feedback loop
# ---------------------------------------------------------------------------


class TestParseWithRetry:
    """``parse_with_retry`` re-fires the LLM call with the malformed
    response appended to the prompt as feedback when the parser raises.

    The behaviour mirrors Graphiti's ``generate_response`` loop in
    ``anthropic_client.py``: up to N attempts, each with the prior error
    quoted back at the model so it can correct itself.
    """

    @staticmethod
    def _parser(raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(str(exc), raw_response=raw) from exc

    def test_recovers_from_bad_json(self):
        """First call returns malformed JSON, second returns valid → result."""
        client = MagicMock()
        client.messages.create.side_effect = [
            _mock_response("not json"),
            _mock_response('{"ok": true}'),
        ]

        result = parse_with_retry(
            client,
            base_prompt="extract this",
            parser=self._parser,
            max_attempts=3,
        )

        assert result == {"ok": True}
        assert client.messages.create.call_count == 2
        # The second call's prompt must include the failure feedback.
        second_call = client.messages.create.call_args_list[1]
        second_prompt = second_call.kwargs["messages"][0]["content"]
        assert "Your last response failed to parse" in second_prompt
        # Base prompt is preserved on the retry.
        assert "extract this" in second_prompt

    def test_gives_up_after_max_attempts(self):
        """Bad JSON every attempt → raises MalformedResponseError, N calls made."""
        client = MagicMock()
        client.messages.create.return_value = _mock_response("not json ever")

        with pytest.raises(MalformedResponseError):
            parse_with_retry(
                client,
                base_prompt="extract this",
                parser=self._parser,
                max_attempts=3,
            )

        assert client.messages.create.call_count == 3

    def test_no_retry_on_first_success(self):
        """Valid JSON first time → 1 call, no feedback appended."""
        client = MagicMock()
        client.messages.create.return_value = _mock_response('{"ok": true}')

        result = parse_with_retry(
            client,
            base_prompt="extract this",
            parser=self._parser,
            max_attempts=3,
        )

        assert result == {"ok": True}
        assert client.messages.create.call_count == 1
        first_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Your last response failed to parse" not in first_prompt

    def test_wraps_raw_value_error(self):
        """A parser that raises plain ValueError/JSONDecodeError still drives
        the retry loop and ultimately surfaces a ``MalformedResponseError``.
        """
        client = MagicMock()
        client.messages.create.return_value = _mock_response("garbage")

        def strict_parser(raw: str) -> dict[str, Any]:
            # Raise the raw library exception instead of MalformedResponseError.
            return json.loads(raw)  # raises JSONDecodeError

        with pytest.raises(MalformedResponseError):
            parse_with_retry(
                client,
                base_prompt="extract this",
                parser=strict_parser,
                max_attempts=2,
            )

        assert client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# _is_usage_limit — content-based detector (kept distinct from RateLimitError)
# ---------------------------------------------------------------------------


class TestUsageLimitDetector:
    """The substring detector stays scoped to known phrases — pairs with the
    structured tenacity retry above on different error modes."""

    def test_detects_known_phrases(self):
        assert _is_usage_limit("Claude usage limit reached")
        assert _is_usage_limit("Your limit will reset at 5pm")
        assert _is_usage_limit("rate_limit_error from API")
        assert _is_usage_limit("Out of usage credits")
        assert _is_usage_limit("You've hit your limit")

    def test_does_not_false_positive_on_unrelated_text(self):
        assert not _is_usage_limit("here is your extraction result")
        assert not _is_usage_limit("entity: rate")  # 'rate' alone, not the phrase
        assert not _is_usage_limit("")
        assert not _is_usage_limit(None)


# ---------------------------------------------------------------------------
# ClaudeCLIClient smoke test — verify the wired-up sync shim still works
# ---------------------------------------------------------------------------


def test_claude_cli_client_routes_through_messages_proxy():
    """ClaudeCLIClient.messages is a _MessagesProxy with the configured model."""
    client = ClaudeCLIClient(model="claude-sonnet-4-5")
    assert isinstance(client.messages, _MessagesProxy)
    assert client.messages._default_model == "claude-sonnet-4-5"


# Suppress an unused-import warning while keeping ``time`` available
# for any future timing-based tests.
_ = time


# ---------------------------------------------------------------------------
# Extended-thinking budget cap
# ---------------------------------------------------------------------------


def test_agent_call_caps_thinking_budget():
    """``agent_call`` must construct ``ClaudeAgentOptions`` with the thinking
    budget capped at ``_MAX_THINKING_TOKENS``. Left unset, Haiku 4.5 burned
    ~12-16K reasoning tokens per call — the dominant cost/quota driver. This
    pins the cap so a future refactor can't silently drop it back to unbounded.
    """
    import ingestion.llm_client as mod

    captured: dict[str, Any] = {}

    def spy_options(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    class _FakeResult:
        def __init__(self) -> None:
            self.result = "hello"
            self.structured_output = None

    class _FakeClient:
        def __init__(self, options: Any = None) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):  # type: ignore[no-untyped-def]
            yield _FakeResult()

    with (
        patch.object(mod, "ClaudeAgentOptions", spy_options),
        patch.object(mod, "ClaudeSDKClient", _FakeClient),
        patch.object(mod, "ResultMessage", _FakeResult),
    ):
        out = asyncio.run(agent_call("hi"))

    assert out == "hello"
    assert captured.get("max_thinking_tokens") == _MAX_THINKING_TOKENS
    assert _MAX_THINKING_TOKENS <= 4096  # guard the intent: a real reduction
