"""Tests for ``ingestion.llm_client`` — the Claude CLI backend and ``parse_with_retry``.

Covers:

* ``agent_call``: the options every CLI spawn carries (no tools, one turn, no prompt
  cache, no session transcript, the thinking cap), its one retry policy (transport
  failures only, then ``LLMUnavailableError``), the per-exchange timeout, and when a
  result counts as a quota notice.
* ``ClaudeAgentClient.messages.create`` flattening chat messages the same way the
  OpenAI-compatible proxy does.
* The ``parse_with_retry`` helper that re-fires the LLM call with feedback on parse
  failure (mirrors Graphiti's ``generate_response`` retry loop).

No ``claude`` CLI is ever spawned: the SDK is replaced at the ``ClaudeSDKClient``
boundary, or ``agent_call`` itself is patched.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tenacity import wait_none

import ingestion.llm_client as llm_mod
from ingestion.llm_client import (
    _MAX_THINKING_TOKENS,
    LLM_TRANSPORT_ERRORS,
    ClaudeAgentClient,
    LLMUnavailableError,
    MalformedResponseError,
    UsageLimitError,
    _is_usage_limit,
    agent_call,
    parse_with_retry,
)
from tests.helpers.llm import mock_llm_message


class _FakeResult:
    """Stands in for the SDK's ``ResultMessage`` (patched over it below)."""

    def __init__(self, result: str | None, is_error: bool = False) -> None:
        self.result = result
        self.is_error = is_error


def _fake_sdk(sessions: list[Any], log: list[str] | None = None) -> type:
    """A ``ClaudeSDKClient`` stand-in. Each session pops the next entry of ``sessions``:
    an exception is raised from the exchange, ``"hang"`` never answers, ``None`` ends
    the stream with no result, anything else is yielded as the result message. Prompts
    sent are appended to ``log``."""

    class _Client:
        def __init__(self, options: Any = None) -> None:
            self._entry = sessions.pop(0)

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(self, prompt: str) -> None:
            if log is not None:
                log.append(prompt)
            if isinstance(self._entry, BaseException):
                raise self._entry
            if self._entry == "hang":
                await asyncio.sleep(3600)

        async def receive_response(self):  # type: ignore[no-untyped-def]
            if self._entry is not None:
                yield self._entry

    return _Client


@pytest.fixture
def cli(monkeypatch):
    """Install a fake SDK: ``cli(sessions)`` returns the list of prompts the CLI received.
    Retry backoff is zeroed so exhausted-retry tests don't sleep."""
    monkeypatch.setattr(llm_mod, "ResultMessage", _FakeResult)
    monkeypatch.setattr(llm_mod, "_RETRY_WAIT", wait_none())

    def install(sessions: list[Any]) -> list[str]:
        log: list[str] = []
        monkeypatch.setattr(llm_mod, "ClaudeSDKClient", _fake_sdk(sessions, log))
        return log

    return install


# ---------------------------------------------------------------------------
# The CLI command every agent_call spawns
# ---------------------------------------------------------------------------


def _cli_command() -> list[str]:
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    transport = SubprocessCLITransport(prompt="", options=llm_mod._cli_options("claude-haiku-4-5"))
    transport._cli_path = "claude"
    return transport._build_command()


def test_cli_spawns_with_no_tools_and_one_turn():
    """``allowed_tools=[]`` only removes auto-approval: without ``--tools ""`` the CLI
    loads its whole built-in toolset (23 definitions, ~10K prompt tokens per call) and
    Read/Grep stay callable while the model reads untrusted transcripts. Nothing needs
    a second turn once there are no tools."""
    cmd = _cli_command()
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert "--allowedTools" not in cmd


def test_cli_spawns_without_mcp_servers_or_session_transcripts():
    """No user-scoped MCP servers load, and no prompt is saved under
    ~/.claude/projects — a second copy of conversation text outside Postgres that the
    dashboard delete cannot reach."""
    cmd = _cli_command()
    assert "--strict-mcp-config" in cmd
    assert "--no-session-persistence" in cmd
    assert "--setting-sources=" in cmd


def test_cli_spawns_with_prompt_caching_off_and_no_background_requests():
    """Every call is a fresh process with a unique prompt, so a cache write is never read.
    Nonessential traffic is off too: without it the CLI sends a second request (a
    session-title call to Haiku carrying the whole prompt, doubling input tokens) and
    still ships the feature-flagged Monitor / PushNotification / RemoteTrigger tools
    that ``--tools ""`` does not remove (measured against CLI 2.1.126)."""
    assert llm_mod._cli_options("claude-haiku-4-5").env == {
        "DISABLE_PROMPT_CACHING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def test_cli_spawns_with_the_thinking_cap():
    """Left unset, Haiku 4.5 burned ~12-16K reasoning tokens per call — the dominant
    cost/quota driver. Pins the cap so a refactor can't drop it back to unbounded."""
    cmd = _cli_command()
    assert cmd[cmd.index("--max-thinking-tokens") + 1] == str(_MAX_THINKING_TOKENS)
    assert _MAX_THINKING_TOKENS <= 4096  # guard the intent: a real reduction


def test_cli_sends_no_json_schema_flag():
    """The schema travels in-prompt only. The SDK forwards ``--json-schema`` solely for
    ``output_format`` type ``json_schema`` (a StructuredOutput tool turn); the old
    ``{"type": "json"}`` value was silently dropped."""
    assert "--json-schema" not in _cli_command()


def test_cli_runs_the_sdk_bundled_binary_without_a_version_probe():
    """No ``cli_path``: the SDK runs its own lock-pinned CLI. And no ``claude -v`` spawn
    before every session (the probe only warns about a CLI too old for the SDK)."""
    assert llm_mod._cli_options("claude-haiku-4-5").cli_path is None
    assert llm_mod.os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK") == "1"


def test_agent_call_appends_the_schema_and_strips_the_fence(cli):
    prompts = cli([_FakeResult('```json\n{"ok": true}\n```')])
    out = asyncio.run(agent_call("extract", system_prompt="be terse", json_schema={"a": 1}))
    assert out == '{"ok": true}'
    assert prompts[0].startswith("be terse\n\nextract")
    assert prompts[0].endswith(
        'Respond with ONLY valid JSON matching this schema — no markdown, no explanation:\n{"a": 1}'
    )


# ---------------------------------------------------------------------------
# agent_call — one retry policy: transport failures only, then LLMUnavailableError
# ---------------------------------------------------------------------------


def test_transport_failure_recovers_on_retry(cli):
    prompts = cli([RuntimeError("CLI crashed"), _FakeResult("hello")])
    assert asyncio.run(agent_call("hi")) == "hello"
    assert len(prompts) == 2


def test_exhausted_transport_failure_is_llm_unavailable(cli):
    """A CLI crash on every attempt must end in a member of LLM_TRANSPORT_ERRORS. A bare
    RuntimeError is swallowed as "found nothing" by every stage that catches broadly,
    and the queue row is marked done with nothing extracted."""
    prompts = cli([RuntimeError("Control request timeout: initialize")] * 3)
    with pytest.raises(LLMUnavailableError, match="Control request timeout") as e:
        asyncio.run(agent_call("hi"))
    assert isinstance(e.value, LLM_TRANSPORT_ERRORS)
    assert len(prompts) == 3


def test_missing_result_is_a_transport_failure(cli):
    cli([None, None, None])
    with pytest.raises(LLMUnavailableError, match="without a result"):
        asyncio.run(agent_call("hi"))


def test_hung_cli_times_out(cli, monkeypatch):
    """A CLI that never answers used to pin a drain thread for good."""
    monkeypatch.setattr(llm_mod, "_CLI_TIMEOUT_S", 0.05)
    prompts = cli(["hang", "hang", "hang"])
    with pytest.raises(LLMUnavailableError, match="TimeoutError"):
        asyncio.run(agent_call("hi"))
    assert len(prompts) == 3


def test_error_result_is_not_retried(cli):
    """The CLI reports its own failures as a normal result with ``is_error=True`` and
    error TEXT in the result field — "Not logged in · Please run /login" for a stack
    with no token, an unrecognised-model notice for a bad id. Parsed as a model answer
    that is exactly a content failure (unparseable text), which is how a whole queue of
    turns got marked done and extracted nothing. It has to be its own signal, and it
    repeats identically, so it is not retried (the CLI already retried API errors)."""
    prompts = cli([_FakeResult("Not logged in · Please run /login", is_error=True)])
    with pytest.raises(LLMUnavailableError, match="Not logged in"):
        asyncio.run(agent_call("hi"))
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# Usage-limit detection — CLI quota notices, never the model's own completions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "notice",
    [
        # The CLI builds "You've hit your ${scope}limit" (session / weekly / Opus / Sonnet / ...).
        "You've hit your session limit · resets 5pm (UTC)",
        "You've hit your weekly limit · resets Mon 9am (UTC)",
        "You've hit your Opus limit · resets 5pm (UTC)",
        "Your org is out of usage · add funds to continue",
    ],
)
def test_quota_notice_error_result_raises_usage_limit(cli, notice):
    prompts = cli([_FakeResult(notice, is_error=True)])
    with pytest.raises(UsageLimitError):
        asyncio.run(agent_call("hi"))
    assert len(prompts) == 1


def test_short_quota_notice_without_error_flag_raises_usage_limit(cli):
    cli([_FakeResult("Claude AI usage limit reached|1760000000")])
    with pytest.raises(UsageLimitError):
        asyncio.run(agent_call("hi"))


def test_completion_quoting_a_quota_message_is_returned(cli):
    """Synapse ingests its own development sessions, so an extraction can legitimately
    contain 'quota exceeded'. Raising UsageLimitError on it made the drain release the
    batch, back off up to 30 minutes and re-claim the same item first, forever."""
    completion = json.dumps(
        {"facts": [{"fact": "agent_call raises UsageLimitError when the CLI says quota exceeded"}]}
    )
    cli([_FakeResult(completion)])
    assert asyncio.run(agent_call("extract", json_schema={"type": "object"})) == completion


def test_long_plain_text_mentioning_a_rate_limit_is_returned(cli):
    text = "The rate_limit_error branch backs off. " * 20
    cli([_FakeResult(text)])
    assert asyncio.run(agent_call("summarise")) == text


class TestUsageLimitDetector:
    """The substring detector stays scoped to known phrases."""

    def test_detects_known_phrases(self):
        assert _is_usage_limit("Claude usage limit reached")
        assert _is_usage_limit("Your limit will reset at 5pm")
        assert _is_usage_limit("rate_limit_error from API")
        assert _is_usage_limit("Out of usage credits")
        assert _is_usage_limit("You've hit your session limit · resets 5pm (UTC)")
        assert _is_usage_limit("You've hit your weekly limit")
        assert _is_usage_limit("Your org is out of usage · add funds to continue")

    def test_does_not_false_positive_on_unrelated_text(self):
        assert not _is_usage_limit("here is your extraction result")
        assert not _is_usage_limit("entity: rate")  # 'rate' alone, not the phrase
        # The CLI's warning prefixes are not exhaustion.
        assert not _is_usage_limit("You've used 90% of your weekly limit")
        assert not _is_usage_limit("You're close to your usage limit")
        assert not _is_usage_limit("")
        assert not _is_usage_limit(None)


# ---------------------------------------------------------------------------
# ClaudeAgentClient.messages.create — same message flattening as the OpenAI proxy
# ---------------------------------------------------------------------------


def test_messages_create_folds_system_messages_and_keeps_every_turn():
    """System-role messages join the system prompt and every other message joins the
    prompt in order — the OpenAI proxy's flattening. The CLI proxy used to keep only the
    last user message and drop system-role messages."""
    seen: dict[str, Any] = {}

    async def fake_agent_call(prompt: str, **kwargs: Any) -> str:
        seen.update(prompt=prompt, **kwargs)
        return '{"ok": true}'

    with patch.object(llm_mod, "agent_call", fake_agent_call):
        resp = ClaudeAgentClient(model="claude-sonnet-4-5").messages.create(
            messages=[
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
            system="persona",
            response_format={"type": "json", "schema": {"type": "object"}},
        )

    assert resp.content[0].text == '{"ok": true}'
    assert seen["system_prompt"] == "persona\n\nrules"
    assert seen["prompt"] == "first\n\nsecond"
    assert seen["model"] == "claude-sonnet-4-5"
    assert seen["json_schema"] == {"type": "object"}


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
            mock_llm_message("not json"),
            mock_llm_message('{"ok": true}'),
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
        client.messages.create.return_value = mock_llm_message("not json ever")

        with pytest.raises(MalformedResponseError):
            parse_with_retry(
                client,
                base_prompt="extract this",
                parser=self._parser,
                max_attempts=3,
            )

        assert client.messages.create.call_count == 3

    def test_all_blank_responses_are_a_transport_failure(self):
        """An EMPTY body on every attempt is not a parse failure — the model never
        answered. Call sites are entitled to degrade on MalformedResponseError (the
        model spoke, badly); degrading here would record a decision that was never
        made and let the queue mark the item done. This is exactly what a wrong model
        id looks like: `claude -p --model anthropic/claude-haiku-4.5` exits 0 with
        [claude-code:unrecognized_model] and no output."""
        client = MagicMock()
        client.messages.create.return_value = mock_llm_message("")

        with pytest.raises(LLMUnavailableError) as e:
            parse_with_retry(
                client, base_prompt="extract this", parser=self._parser, max_attempts=3
            )
        assert client.messages.create.call_count == 3
        assert "no output" in str(e.value)

    def test_one_non_blank_answer_stays_a_parse_failure(self):
        """The model DID speak on some attempt, so the failure is about content and the
        caller's conservative fallback is a fair reading of it."""
        client = MagicMock()
        client.messages.create.side_effect = [
            mock_llm_message(""),
            mock_llm_message("not json"),
            mock_llm_message("   "),
        ]

        with pytest.raises(MalformedResponseError):
            parse_with_retry(
                client, base_prompt="extract this", parser=self._parser, max_attempts=3
            )

    def test_no_retry_on_first_success(self):
        """Valid JSON first time → 1 call, no feedback appended."""
        client = MagicMock()
        client.messages.create.return_value = mock_llm_message('{"ok": true}')

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
        client.messages.create.return_value = mock_llm_message("garbage")

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
