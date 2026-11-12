"""Tests for the messages-format parser and its /ingest wiring (no database).

The parser tests are pure. The route tests drive the real ``/ingest`` handler through a
Starlette TestClient: every request either fails validation (400, before any work) or
parses to no episodes, and the handler returns before it opens a database. The DSN
callback raises, so a request that did reach the database would surface as a 500.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

import pytest

from ingestion.messages_client import (
    CLIENT_PATTERN,
    RESERVED_CLIENTS,
    MessagesParser,
    valid_client,
)
from ingestion.models import Episode

SID = "hermes-20261112-0001"
CLIENT = "hermes"


def _user(text, msg_id=None, **extra) -> dict:
    return {"role": "user", "content": text, **({"id": msg_id} if msg_id else {}), **extra}


def _asst(text=None, msg_id=None, tool_calls=None, **extra) -> dict:
    rec = {"role": "assistant", "content": text, **extra}
    if msg_id:
        rec["id"] = msg_id
    if tool_calls is not None:
        rec["tool_calls"] = tool_calls
    return rec


def _tool(text, msg_id=None, **extra) -> dict:
    return {"role": "tool", "content": text, **({"id": msg_id} if msg_id else {}), **extra}


def _call(name, arguments) -> dict:
    """The OpenAI chat shape: the name and arguments nested under "function"."""
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse(records, project_override=None, **kw) -> list[Episode]:
    kw.setdefault("session_id", SID)
    kw.setdefault("client", CLIENT)
    return MessagesParser().parse_records(records, "test", project_override, **kw)


def _basic_session() -> list[dict]:
    return [
        {"role": "system", "content": "You are Hermes."},  # skip
        _user("hi", "m1", ts="2026-11-12T10:00:00Z"),
        _asst("Hello! How can I help?", "m2"),
        _user("run the tests", "m3", ts="2026-11-12T10:01:00Z"),
        _asst(
            "Running them now.", "m4", tool_calls=[_call("terminal", '{"command": "pytest -q"}')]
        ),
        _tool("47 passed in 3.21s — all green", "m5", name="terminal"),
        _asst("All 47 tests pass.", "m6"),
    ]


# ---------------------------------------------------------------------------
# client names
# ---------------------------------------------------------------------------


class TestValidClient:
    @pytest.mark.parametrize("name", ["hermes", "a", "my_agent2", "a" + "b" * 31])
    def test_accepts(self, name):
        assert valid_client(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "Hermes",
            "1agent",
            "_agent",
            "her-mes",
            "her mes",
            "a" * 33,
            "hermes\n",
            None,
            42,
            ["hermes"],
        ],
    )
    def test_rejects_bad_names(self, name):
        assert not valid_client(name)

    @pytest.mark.parametrize("name", sorted(RESERVED_CLIENTS))
    def test_rejects_reserved_names_that_match_the_pattern(self, name):
        assert re.fullmatch(CLIENT_PATTERN, name)
        assert not valid_client(name)

    def test_reserved_set(self):
        # The platforms the other parsers stamp. Their span-id prefixes need no entry:
        # every span id this parser writes starts with its own source, "messages:".
        assert RESERVED_CLIENTS == {
            "claude_code",
            "codex",
            "cursor",
            "chatgpt",
            "claude_ai",
        }

    @pytest.mark.parametrize("client", ["hermes", "jsonl", "openai", "x"])
    def test_span_id_starts_with_the_source_whatever_the_client_is_called(self, client):
        # "jsonl:" and "openai:" are other parsers' span-id prefixes; a client by that
        # name still writes under "messages:", so it can never claim one of their ids.
        withid, noid = MessagesParser().parse_records(
            [
                {"role": "user", "content": "first question", "id": "m1"},
                {"role": "user", "content": "second question"},
            ],
            "t",
            session_id="s1",
            client=client,
        )
        assert withid.source == noid.source == "messages"
        assert withid.span_id == f"messages:{client}:m1"
        assert noid.span_id.startswith(f"messages:{client}:s1:")

    def test_parser_refuses_a_reserved_client(self):
        with pytest.raises(ValueError):
            _parse(_basic_session(), client="codex")


# ---------------------------------------------------------------------------
# grouping and rendering
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_two_turns(self):
        first, second = _parse(_basic_session())

        assert first.session_id == SID
        assert first.sequence == 1
        assert first.human_turn == "hi"
        assert first.assistant_turn == "Hello! How can I help?"
        assert first.content == "[user] hi\n\n[assistant] Hello! How can I help?"
        assert first.span_id == "messages:hermes:m2"
        assert first.platform == "hermes"
        assert first.source == "messages"

        assert second.sequence == 2
        assert second.human_turn == "run the tests"
        assert second.assistant_turn == "All 47 tests pass."
        assert second.span_id == "messages:hermes:m6"
        assert second.content.split("\n\n") == [
            "[context] Hello! How can I help?",
            "[user] run the tests",
            "[assistant] Running them now.",
            '[tool:terminal] {"command": "pytest -q"}',
            "[result] 47 passed in 3.21s — all green",
            "[assistant] All 47 tests pass.",
        ]

    def test_batch_starting_mid_turn_yields_a_leading_fragment(self):
        tail = [*_basic_session()[5:], _user("next", "m7"), _asst("ok", "m8")]
        frag, turn = _parse(tail)

        assert frag.human_turn is None
        assert (
            frag.content
            == "[result] 47 passed in 3.21s — all green\n\n[assistant] All 47 tests pass."
        )
        assert frag.span_id == "messages:hermes:m6"  # the full turn's span id: the route skips it
        assert turn.human_turn == "next"
        assert turn.content.startswith("[context] All 47 tests pass.\n\n[user] next")

    def test_consecutive_user_rows_each_open_a_turn(self):
        eps = _parse([_user("one"), _user("two"), _asst("reply")])
        assert [e.human_turn for e in eps] == ["one", "two"]
        assert eps[0].content == "[user] one"
        assert eps[1].assistant_turn == "reply"


class TestRendering:
    def test_both_tool_call_shapes(self):
        nested = _call("terminal", '{"command": "ls"}')
        flat = {"name": "read_file", "arguments": {"path": "/tmp/x"}}
        (ep,) = _parse([_user("look"), _asst(None, tool_calls=[nested, flat])])
        assert ep.content.split("\n\n")[1:] == [
            '[tool:terminal] {"command": "ls"}',
            '[tool:read_file] {"path": "/tmp/x"}',
        ]
        assert ep.assistant_turn is None  # tool calls alone are not assistant text

    def test_object_arguments_render_like_the_same_json_string(self):
        as_str = _parse([_user("q"), _asst(None, tool_calls=[_call("t", '{"a": 1, "b": "x"}')])])
        as_obj = _parse([_user("q"), _asst(None, tool_calls=[_call("t", {"b": "x", "a": 1})])])
        assert as_str[0].content == as_obj[0].content == '[user] q\n\n[tool:t] {"a": 1, "b": "x"}'

    def test_tool_calls_without_arguments_or_name(self):
        calls = [
            _call("empty_str", "  "),
            _call("none", None),
            _call("empty_obj", {}),
            {"arguments": "x=1"},  # flat, no name
            {"function": {"arguments": "y=2"}},  # nested, no name
            "not a dict",
        ]
        (ep,) = _parse([_user("q"), _asst(None, tool_calls=calls)])
        assert ep.content == "[user] q\n\n[tool:tool] x=1\n\n[tool:tool] y=2"

    def test_tool_arguments_are_truncated(self):
        (ep,) = _parse([_user("q"), _asst(None, tool_calls=[_call("t", "z" * 400)])])
        assert ep.content.endswith("[tool:t] " + "z" * 300)

    def test_list_of_parts_keeps_text_parts_only(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "input_text", "text": "second"},
            "bare string",
        ]
        reply = [{"type": "output_text", "text": "seen"}, {"type": "refusal", "refusal": "no"}]
        (ep,) = _parse([_user(content), _asst(reply)])
        assert ep.human_turn == "first\nsecond"
        assert ep.assistant_turn == "seen"
        assert "AAAA" not in ep.content and "refusal" not in ep.content

    def test_image_only_user_row_is_dropped(self):
        image = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
        (ep,) = _parse([_user("q1"), _asst("a1"), _user(image), _asst("a2")])
        assert ep.assistant_turn == "a2"

    @pytest.mark.parametrize("role", ["system", "developer", "function", None, 7])
    def test_other_roles_are_ignored(self, role):
        records = [
            _user("q1"),
            {"role": role, "content": "injected instructions"},
            _asst("a1"),
            {"role": role, "content": "more instructions"},
            _user("q2"),
            _asst("a2"),
        ]
        eps = _parse(records)
        assert [e.human_turn for e in eps] == ["q1", "q2"]
        assert "instructions" not in "\n".join(e.content for e in eps)

    def test_non_dict_records_are_dropped(self):
        eps = _parse(["hi", None, ["user", "x"], 3, _user("q"), _asst("a")])
        assert len(eps) == 1
        assert eps[0].content == "[user] q\n\n[assistant] a"

    def test_synthetic_rows_neither_open_a_turn_nor_render(self):
        records = [
            _user("q1", "m1"),
            _asst("working", "m2", tool_calls=[_call("terminal", "ls")]),
            _user("[harness] tool budget exceeded, continue", "m3", synthetic=True),
            _asst("done", "m4"),
        ]
        (ep,) = _parse(records)
        assert ep.human_turn == "q1"
        assert ep.assistant_turn == "done"
        assert ep.span_id == "messages:hermes:m4"
        assert "[harness]" not in ep.content

    def test_short_tool_output_is_dropped_and_long_truncated(self):
        (ep,) = _parse([_user("q"), _tool("ok"), _tool("y" * 600), _asst("a")])
        assert ep.content.split("\n\n") == ["[user] q", "[result] " + "y" * 500, "[assistant] a"]

    def test_tool_name_is_accepted_but_not_rendered(self):
        (ep,) = _parse([_user("q"), _tool("a long enough tool result", tool_name="grep")])
        assert ep.content == "[user] q\n\n[result] a long enough tool result"

    def test_user_and_assistant_truncation(self):
        (ep,) = _parse([_user("u" * 3500), _asst("a" * 3500)])
        assert ep.human_turn == "u" * 3000
        assert ep.assistant_turn == "a" * 3500  # stored whole, as in the Codex parser
        assert ep.content == f"[user] {'u' * 3000}\n\n[assistant] {'a' * 3000}"

    def test_nul_bytes_are_stripped(self):
        (ep,) = _parse([_user("a\x00b"), _asst("c\x00d")])
        assert ep.human_turn == "ab"
        assert ep.assistant_turn == "cd"
        assert "\x00" not in ep.content


class TestMemoryContext:
    def test_blocks_are_removed_from_user_text(self):
        text = (
            "<memory-context>\nThe user runs Proxmox.\nRecalled: e:12\n</memory-context>\n\n"
            "what's next?\n<memory-context>second block</memory-context>"
        )
        (ep,) = _parse([_user(text), _asst("Next is the router.")])
        assert ep.human_turn == "what's next?"
        assert "Proxmox" not in ep.content and "memory-context" not in ep.content

    def test_a_block_split_across_text_parts_is_removed(self):
        parts = [
            {"type": "text", "text": "<memory-context>recalled"},
            {"type": "text", "text": "facts</memory-context>"},
            {"type": "text", "text": "the question"},
        ]
        (ep,) = _parse([_user(parts)])
        assert ep.human_turn == "the question"

    def test_the_tag_is_case_sensitive(self):
        (ep,) = _parse([_user("<MEMORY-CONTEXT>kept</MEMORY-CONTEXT> q")])
        assert ep.human_turn == "<MEMORY-CONTEXT>kept</MEMORY-CONTEXT> q"

    def test_assistant_text_is_not_touched(self):
        text = "<memory-context>quoted back</memory-context>"
        (ep,) = _parse([_user("q"), _asst(text)])
        assert ep.assistant_turn == text

    def test_a_user_row_that_is_only_context_is_dropped(self):
        records = [
            _user("q1"),
            _asst("a1"),
            _user("<memory-context>\nrecalled\n</memory-context>\n"),
            _asst("a2"),
        ]
        (ep,) = _parse(records)
        # No turn opens at the emptied row, so its reply stays in the previous turn.
        assert ep.human_turn == "q1"
        assert ep.assistant_turn == "a2"
        assert "recalled" not in ep.content


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


class TestSpanId:
    def test_last_id_in_the_group(self):
        records = [_user("q", "u1"), _asst("a", "a1"), _tool("x" * 30), _asst("b", 99)]
        (ep,) = _parse(records)
        assert ep.span_id == "messages:hermes:a1"  # the trailing rows carry no string id

    def test_hash_fallback_without_ids(self):
        (ep,) = _parse([_user("q"), _asst("a")])
        digest = hashlib.sha256(b"[user] q\n\n[assistant] a").hexdigest()[:24]
        assert ep.span_id == f"messages:hermes:{SID}:{digest}"

    def test_hash_ignores_the_context_line(self):
        turn = [_user("second question"), _asst("second answer")]
        with_prev = _parse([_user("first"), _asst("first answer"), *turn])[1]
        alone = _parse(turn)[0]
        assert with_prev.content.startswith("[context] first answer")
        assert not alone.content.startswith("[context]")
        assert with_prev.span_id == alone.span_id

    def test_hash_differs_between_turns_and_sessions(self):
        a, b = _parse([_user("q1"), _asst("a1"), _user("q2"), _asst("a2")])
        assert a.span_id != b.span_id
        (other,) = _parse([_user("q1"), _asst("a1")], session_id="another-session")
        assert other.span_id != a.span_id

    def test_a_replayed_turn_is_emitted_once(self):
        replay = [_user("hi", "m1"), _asst("hello", "m2")]
        eps = _parse([*replay, *replay, _user("next", "m3"), _asst("ok", "m4")])
        assert [e.span_id for e in eps] == ["messages:hermes:m2", "messages:hermes:m4"]
        assert [e.sequence for e in eps] == [1, 3]  # the skipped group still counts
        assert eps[1].content.startswith("[context] hello")

    def test_identical_turns_without_ids_collapse(self):
        eps = _parse([_user("ok"), _asst("Done."), _user("ok"), _asst("Done.")])
        assert len(eps) == 1


# ---------------------------------------------------------------------------
# timestamps, model, project
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_epoch_seconds(self):
        (ep,) = _parse([_user("q", ts=1762960000.12), _asst("a")])
        assert ep.created_at == datetime.fromtimestamp(1762960000.12, tz=UTC)
        assert ep.created_at.tzinfo is not None
        assert ep.metadata["ts"] == ep.created_at.isoformat()

    def test_epoch_int(self):
        (ep,) = _parse([_user("q", ts=1762960000)])
        assert ep.created_at == datetime(2025, 11, 12, 15, 6, 40, tzinfo=UTC)

    def test_iso_string(self):
        (ep,) = _parse([_user("q", ts="2026-11-12T10:00:00.5Z")])
        assert ep.created_at == datetime(2026, 11, 12, 10, 0, 0, 500000, tzinfo=UTC)
        assert ep.metadata["ts"] == "2026-11-12T10:00:00.500000+00:00"

    @pytest.mark.parametrize(
        "bad", ["yesterday", "", True, float("nan"), float("inf"), 1e20, {"t": 1}, [1]]
    )
    def test_unusable_values_are_skipped(self, bad):
        records = [_user("q", ts=bad), _asst("a", ts="2026-11-12T10:05:00Z")]
        (ep,) = _parse(records)
        assert ep.created_at == datetime(2026, 11, 12, 10, 5, tzinfo=UTC)

    def test_no_usable_ts(self):
        (ep,) = _parse([_user("q", ts="garbage"), _asst("a")])
        assert ep.created_at is None
        assert ep.metadata["ts"] is None

    def test_first_usable_ts_in_the_group_wins(self):
        records = [_user("q"), _asst("a", ts=1762960000), _asst("b", ts=1762960999)]
        (ep,) = _parse(records)
        assert ep.created_at == datetime.fromtimestamp(1762960000, tz=UTC)


class TestModelAndProject:
    def test_per_message_model_overrides_the_batch_default(self):
        records = [
            _user("q1"),
            _asst("a1", model="model-a"),
            _asst("a1b", model="model-b"),
            _user("q2"),
            _asst("a2"),
        ]
        first, second = _parse(records, model="batch-default")
        assert first.model == "model-b"  # the last one named in the turn
        assert second.model == "batch-default"

    def test_project_override_wins_over_cwd(self):
        (ep,) = _parse([_user("q")], "Alpha", cwd="/work/Other")
        assert ep.project == "Alpha"
        assert ep.metadata["cwd"] == "/work/Other"

    @pytest.mark.parametrize("cwd", ["/work/Alpha", "/work/Alpha/", "C:\\Users\\pat\\Alpha"])
    def test_project_from_cwd(self, cwd):
        (ep,) = _parse([_user("q")], cwd=cwd)
        assert ep.project == "Alpha"

    @pytest.mark.parametrize("cwd", [None, "", 42])
    def test_no_project_without_a_usable_cwd(self, cwd):
        (ep,) = _parse([_user("q")], cwd=cwd)
        assert ep.project is None
        assert ep.metadata["cwd"] is None

    def test_metadata(self):
        (ep,) = MessagesParser().parse_records(
            [_user("q", ts=1762960000)], "hermes-hook", session_id=SID, client="hermes"
        )
        assert ep.metadata == {
            "source_label": "hermes-hook",
            "ts": "2025-11-12T15:06:40+00:00",
            "cwd": None,
            "client": "hermes",
        }


class TestEmpty:
    @pytest.mark.parametrize("session_id", [None, ""])
    def test_no_session_id(self, session_id):
        assert _parse(_basic_session(), session_id=session_id) == []

    def test_nothing_renders(self):
        records = [{"role": "system", "content": "x"}, _tool("short"), _asst(None, "m1")]
        assert _parse(records) == []

    def test_empty_batch(self):
        assert _parse([]) == []


# ---------------------------------------------------------------------------
# POST /ingest: validation and dispatch, without a database
# ---------------------------------------------------------------------------


def _no_db() -> str:
    raise AssertionError("the request reached the database")


@pytest.fixture()
def ingest():
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.ingest_route import register

    m = FastMCP("test-messages-ingest")
    register(m, _no_db, lambda r: True)
    client = TestClient(m.http_app())
    return lambda body: client.post("/ingest", json=body)


def _spy(monkeypatch, cls) -> list[tuple[tuple, dict]]:
    """Replace ``cls.parse_records`` with a recorder that parses to nothing."""
    calls: list[tuple[tuple, dict]] = []

    def parse_records(self, *args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(cls, "parse_records", parse_records)
    return calls


_DROP = object()  # a parametrize value meaning "leave this key out of the body"


def _messages_body(**overrides) -> dict:
    body = {
        "format": "messages",
        "client": CLIENT,
        "session_id": SID,
        "records": [{"role": "system", "content": "renders nothing"}],
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not _DROP}


class TestIngestRoute:
    @pytest.mark.parametrize("client", [_DROP, None, "", "Hermes", "her-mes", "hermes\n", 42])
    def test_bad_client_is_a_400(self, ingest, client):
        r = ingest(_messages_body(client=client))
        assert r.status_code == 400
        assert r.json()["detail"] == f"format 'messages' needs a 'client' matching {CLIENT_PATTERN}"

    @pytest.mark.parametrize("client", sorted(RESERVED_CLIENTS))
    def test_reserved_client_is_a_400(self, ingest, client):
        r = ingest(_messages_body(client=client))
        assert r.status_code == 400
        assert r.json()["detail"] == f"client {client!r} is reserved for a built-in parser"

    @pytest.mark.parametrize("session_id", [_DROP, None, "", "   ", 42])
    def test_missing_session_id_is_a_400(self, ingest, session_id):
        r = ingest(_messages_body(session_id=session_id))
        assert r.status_code == 400
        assert r.json()["detail"] == "format 'messages' needs a non-empty 'session_id' string"

    def test_unknown_format_is_still_a_400(self, ingest):
        r = ingest({"records": [], "format": "hermes", "client": CLIENT, "session_id": SID})
        assert r.status_code == 400
        assert r.json()["detail"] == "unknown format 'hermes'"

    def test_a_valid_messages_body_parses(self, ingest):
        r = ingest(_messages_body())
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok", "ingested": 0, "span_collisions": 0}

    def test_messages_body_fields_reach_the_parser(self, ingest, monkeypatch):
        calls = _spy(monkeypatch, MessagesParser)
        records = [_user("q", "m1")]
        body = _messages_body(
            records=records,
            project="C:\\Users\\pat\\Alpha\\",
            cwd="/work/Other",
            model="m-default",
            source="hermes-hook",
        )
        assert ingest(body).status_code == 200
        assert calls == [
            (
                (records, "hermes-hook", "Alpha"),
                {
                    "session_id": SID,
                    "client": CLIENT,
                    "cwd": "/work/Other",
                    "model": "m-default",
                },
            )
        ]

    @pytest.mark.parametrize("fmt", [_DROP, None, "", "claude_code"])
    def test_default_format_is_still_claude_code(self, ingest, monkeypatch, fmt):
        from ingestion.codex_client import CodexRolloutParser
        from ingestion.jsonl_client import JSONLParser

        jsonl = _spy(monkeypatch, JSONLParser)
        codex = _spy(monkeypatch, CodexRolloutParser)
        messages = _spy(monkeypatch, MessagesParser)
        # The messages-only fields mean nothing here, valid or not.
        body = {"records": [{"type": "user"}], "client": "codex", "format": fmt}
        r = ingest({k: v for k, v in body.items() if v is not _DROP})
        assert r.status_code == 200, r.text
        assert len(jsonl) == 1 and not codex and not messages

    def test_codex_still_needs_neither_client_nor_session_id(self, ingest, monkeypatch):
        from ingestion.codex_client import CodexRolloutParser

        codex = _spy(monkeypatch, CodexRolloutParser)
        r = ingest({"records": [{"type": "session_meta"}], "format": "codex"})
        assert r.status_code == 200, r.text
        assert codex == [(([{"type": "session_meta"}], "hook", None), {"session_id_hint": None})]
