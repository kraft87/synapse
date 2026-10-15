"""Turn assembly shared by the five transcript parsers, and the parsers that had no tests.

Pure unit tests (tmp files only): the shared layout and the empty-turn rule in
``ingestion.turns``, the Episode validator that owns NUL stripping, the claude.ai and
Cursor-SQLite parsers, the sources that had drifted from the rest, and the client-side
copies of the machinery prefixes, which must agree with the server's.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from ingestion import codex_client, jsonl_client
from ingestion.claude_ai_client import ClaudeAIParser, parse_export
from ingestion.cursor_sqlite_client import CursorSQLiteParser, _project_from_uris
from ingestion.models import Episode
from ingestion.turns import TurnSequence, parse_iso

_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# ingestion.turns
# ---------------------------------------------------------------------------


def _turn(seq: TurnSequence, user: str | None = None, reply: str | None = None, title=None):
    turn = seq.begin(title)
    turn.user(user)
    turn.assistant_text(reply)
    return turn


def test_a_turn_is_tagged_lines_and_the_first_user_text_wins():
    turn = TurnSequence().begin()
    turn.user("first question")
    turn.line("[tool:Bash] ls")
    turn.user("second user record, which is tool output")
    turn.result("x" * 600)
    turn.result("too short")  # at or under 20 characters carries no information
    turn.assistant_text("one")
    turn.assistant_text("two")
    parts = turn.content.split("\n\n")
    assert parts[0] == "[user] first question"
    assert parts[1] == "[tool:Bash] ls"
    assert parts[2] == "[result] " + "x" * 500
    assert parts[3:] == ["[assistant] one", "[assistant] two"]
    assert (turn.human, turn.assistant) == ("first question", "two")


def test_text_is_truncated_for_the_line_but_the_reply_is_kept_whole():
    turn = TurnSequence().begin()
    turn.user("q" * 4000)
    turn.assistant_text("a" * 4000)
    assert turn.human == "q" * 3000
    assert turn.assistant == "a" * 4000
    assert turn.content == f"[user] {'q' * 3000}\n\n[assistant] {'a' * 3000}"
    wide = TurnSequence(text_chars=5000).begin()
    wide.user("q" * 6000)
    assert wide.human == "q" * 5000


def test_the_next_turn_leads_with_the_last_reply_and_the_first_with_the_title():
    seq = TurnSequence()
    first = _turn(seq, "q1", "a1", title="Chat name")
    assert first.content.startswith("[title] Chat name\n\n[user] q1")
    assert seq.accept(first, "s1")
    second = _turn(seq, "q2", "a2", title="Chat name")
    assert second.content.startswith("[context] a1\n\n[user] q2")
    assert second.sequence == 2


def test_a_turn_with_only_its_lead_is_empty_and_is_not_accepted():
    seq = TurnSequence()
    assert seq.accept(_turn(seq, "q1", "a1"), "s1")
    bounce = seq.begin()  # a hook bounce: nothing but the [context] line
    assert bounce.empty and not seq.accept(bounce, "s2")
    # it consumed a sequence number and did not replace the carried reply
    after = _turn(seq, "q3", "a3")
    assert after.sequence == 3 and after.content.startswith("[context] a1")


def test_a_repeated_span_id_is_emitted_once():
    seq = TurnSequence()
    assert seq.accept(_turn(seq, "q1", "a1"), "s1")
    assert not seq.accept(_turn(seq, "q1 again", "a1 again"), "s1")
    assert seq.accept(_turn(seq, "q2", "a2"), None)  # no span_id: nothing to dedup on
    assert seq.accept(_turn(seq, "q3", "a3"), None)


def test_parse_iso_takes_a_z_suffix_and_shrugs_off_junk():
    assert parse_iso("2026-09-01T12:00:00Z").isoformat() == "2026-09-01T12:00:00+00:00"
    assert parse_iso("2026-09-01").year == 2026
    assert parse_iso("not a date") is None
    assert parse_iso(None) is None and parse_iso("") is None


# ---------------------------------------------------------------------------
# Episode: NUL bytes go before redaction, in one place
# ---------------------------------------------------------------------------


def test_episode_strips_nul_bytes_from_every_text_field():
    ep = Episode(
        session_id="s",
        sequence=1,
        content="a\x00b",
        human_turn="c\x00d",
        assistant_turn="e\x00f",
    )
    assert (ep.content, ep.human_turn, ep.assistant_turn) == ("ab", "cd", "ef")


def test_a_nul_cannot_hide_a_credential_from_redaction():
    token = "sk-" + "ant-api03-" + "aB3dE5fG7hJ9kL2mN4pQ6rS8tU0vW1xY" * 3
    ep = Episode(session_id="s", sequence=1, content=f"key {token[:20]}\x00{token[20:]} end")
    assert token[:20] not in ep.content and "\x00" not in ep.content


# ---------------------------------------------------------------------------
# claude.ai export
# ---------------------------------------------------------------------------


def _convo(messages: list[dict], name: str = "Planning") -> dict:
    return {"uuid": "convo-1", "name": name, "chat_messages": messages}


def _msg(uuid: str, sender: str, text: str, at: str, **extra) -> dict:
    return {"uuid": uuid, "sender": sender, "text": text, "created_at": at, **extra}


def test_claude_ai_pairs_human_and_assistant_turns():
    eps = ClaudeAIParser.parse_conversation(
        _convo(
            [
                _msg("m1", "human", "what is a gate", "2026-09-01T10:00:00Z"),
                _msg("m2", "assistant", "a filter", "2026-09-01T10:00:05Z"),
                _msg("m3", "human", "and a fence", "2026-09-01T10:01:00Z"),
                _msg("m4", "assistant", "a barrier", "2026-09-01T10:01:05Z"),
            ]
        )
    )
    assert [e.sequence for e in eps] == [1, 2]
    first, second = eps
    assert first.content == "[title] Planning\n\n[user] what is a gate\n\n[assistant] a filter"
    assert second.content == "[context] a filter\n\n[user] and a fence\n\n[assistant] a barrier"
    assert (first.span_id, second.span_id) == ("claude_ai:convo-1:m2", "claude_ai:convo-1:m4")
    assert first.platform == "claude_ai" and first.project is None
    assert first.created_at.isoformat() == "2026-09-01T10:00:00+00:00"


def test_claude_ai_keeps_up_to_five_thousand_characters_a_turn():
    eps = ClaudeAIParser.parse_conversation(
        _convo([_msg("m1", "human", "q" * 6000, "2026-09-01"), _msg("m2", "assistant", "a" * 6000, "2026-09-02")])
    )  # fmt: skip
    assert eps[0].human_turn == "q" * 5000
    assert eps[0].assistant_turn == "a" * 6000
    assert eps[0].content.endswith("[assistant] " + "a" * 5000)


def test_claude_ai_notes_attachments_and_sorts_out_of_order_messages():
    eps = ClaudeAIParser.parse_conversation(
        _convo(
            [
                _msg("m2", "assistant", "read it", "2026-09-01T10:00:05Z"),
                _msg(
                    "m1",
                    "human",
                    "see file",
                    "2026-09-01T10:00:00Z",
                    attachments=[{"file_name": "a.pdf"}],
                ),
            ]
        )
    )
    assert len(eps) == 1
    assert "[attachments] file:a.pdf" in eps[0].content
    assert eps[0].content.index("[user]") < eps[0].content.index("[assistant]")


def test_claude_ai_has_nothing_to_store_without_text_or_an_id():
    assert (
        ClaudeAIParser.parse_conversation({"chat_messages": [_msg("m", "human", "x", "t")]}) == []
    )
    assert ClaudeAIParser.parse_conversation(_convo([])) == []
    assert ClaudeAIParser.parse_conversation(_convo([_msg("m", "human", "", "t")])) == []


def test_claude_ai_drops_a_group_that_is_only_attachments():
    """A file dropped in before the first prompt is not a turn (it never was stored, and each
    stored turn costs an extraction call); the group still spends its sequence number."""
    eps = ClaudeAIParser.parse_conversation(
        _convo(
            [
                _msg("m0", "human", "", "2026-09-01T09:59:00Z", attachments=[{"file_name": "a.pdf"}]),
                _msg("m1", "human", "summarise it", "2026-09-01T10:00:00Z"),
                _msg("m2", "assistant", "done", "2026-09-01T10:00:05Z"),
            ]
        )
    )  # fmt: skip
    assert [(e.sequence, e.content) for e in eps] == [
        (2, "[user] summarise it\n\n[assistant] done")
    ]


def test_claude_ai_export_loads_every_conversation_and_rejects_a_non_list(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([_convo([_msg("m1", "human", "hi", "2026-09-01")]), "junk"]))
    assert len(parse_export(path)) == 1
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        parse_export(path)


# ---------------------------------------------------------------------------
# Cursor SQLite
# ---------------------------------------------------------------------------

_COMPOSER = "11111111-1111-1111-1111-111111111111"


def _bubble_id(n: int) -> str:
    return f"{n:08d}-0000-0000-0000-000000000000"


def _cursor_db(tmp_path: Path, bubbles: list[dict]) -> Path:
    path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)")
    for n, b in enumerate(bubbles, start=1):
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (f"bubbleId:{_COMPOSER}:{_bubble_id(n)}", json.dumps(b)),
        )
    conn.commit()
    conn.close()
    return path


def test_cursor_groups_bubbles_into_turns_with_tool_results_and_a_project(tmp_path):
    uri = "file:///c%3A/Users/me/src/Acme"
    db = _cursor_db(
        tmp_path,
        [
            {"type": 1, "text": "add a test", "createdAt": "2026-09-01T10:00:00Z", "workspaceUris": [uri]},
            {"type": 2, "text": "done", "createdAt": "2026-09-01T10:00:05Z",
             "modelInfo": {"model": "gpt-x"}, "toolResults": [{"name": "run", "result": "12 passed in 0.4s total"}]},
            {"type": 1, "text": "thanks", "createdAt": "2026-09-01T10:01:00Z"},
            {"type": 2, "text": "welcome", "createdAt": "2026-09-01T10:01:05Z"},
        ],
    )  # fmt: skip
    eps = CursorSQLiteParser.parse_file(db)
    assert [e.sequence for e in eps] == [1, 2]
    first, second = eps
    assert (
        first.content
        == "[user] add a test\n\n[assistant] done\n\n[tool:run] 12 passed in 0.4s total"
    )
    assert second.content == "[context] done\n\n[user] thanks\n\n[assistant] welcome"
    assert (first.project, first.model, first.platform) == ("Acme", "gpt-x", "cursor")
    assert first.span_id == f"cursor-sqlite:{_COMPOSER}:{_bubble_id(2)}"
    assert first.created_at.isoformat() == "2026-09-01T10:00:00+00:00"


def test_cursor_drops_a_group_that_is_only_tool_output(tmp_path):
    """Tool results ahead of the first prompt are not a turn; the group still spends its sequence."""
    db = _cursor_db(
        tmp_path,
        [
            {"type": 2, "text": "", "createdAt": "2026-09-01T09:59:00Z",
             "toolResults": [{"name": "run", "result": "12 passed in 0.4s total"}]},
            {"type": 1, "text": "hi", "createdAt": "2026-09-01T10:00:00Z"},
            {"type": 2, "text": "yo", "createdAt": "2026-09-01T10:00:05Z"},
        ],
    )  # fmt: skip
    eps = CursorSQLiteParser.parse_file(db)
    assert [(e.sequence, e.content) for e in eps] == [(2, "[user] hi\n\n[assistant] yo")]


def test_cursor_without_the_kv_table_or_a_readable_file_yields_nothing(tmp_path):
    empty = tmp_path / "empty.vscdb"
    sqlite3.connect(empty).close()
    assert CursorSQLiteParser.parse_file(empty) == []
    assert CursorSQLiteParser.parse_file(tmp_path / "missing" / "nope.vscdb") == []


@pytest.mark.parametrize(
    ("uris", "label"),
    [
        (["file:///home/me/app/"], "app"),
        (["file:///c%3A/Users/me/src/Acme"], "Acme"),
        (["file:///home/me/projects", "file:///home/me/dev/real"], "real"),
        (["file:///c%3A/"], None),
        ([], None),
        (None, None),
    ],
)
def test_cursor_labels_a_workspace_the_way_every_other_source_does(uris, label):
    assert _project_from_uris(uris) == label


# ---------------------------------------------------------------------------
# The client hooks keep dependency-free copies of the machinery prefixes
# ---------------------------------------------------------------------------


def _prefixes_in(path: Path) -> tuple[str, ...]:
    """The literal ``_MACHINERY_PREFIXES`` tuple a hook defines, read without importing it
    (a hook runs under the CLI's bare Python and loads its own config on import)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_MACHINERY_PREFIXES" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{path} defines no _MACHINERY_PREFIXES")


def test_claude_code_hook_mirrors_the_server_machinery_prefixes():
    hook = _REPO / "plugin" / "scripts" / "ingest_hook.py"
    assert _prefixes_in(hook) == jsonl_client._MACHINERY_PREFIXES


def test_codex_hook_mirrors_the_server_machinery_prefixes():
    hook = _REPO / "plugin-codex" / "hooks" / "synapse_stop_hook.py"
    assert _prefixes_in(hook) == codex_client._MACHINERY_PREFIXES
