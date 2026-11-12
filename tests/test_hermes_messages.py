"""The Hermes message normaliser and turn slicing (plugin/synapse_client/harnesses/hermes/messages.py).

The inputs are shaped like the live dicts Hermes hands a memory provider: OpenAI chat
messages plus Hermes's own bookkeeping (``message_uid``, ``timestamp``, ``api_content``,
``display_kind``, producer flags, compaction rows). The records are then fed to the
server's own ``MessagesParser`` to show what the episodes would be.
"""

from __future__ import annotations

import copy
import json

import pytest

from ingestion.messages_client import MessagesParser
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402

SID = "20261112_101500_ab12cd"
_KEYS = {"role", "content", "tool_calls", "name", "ts", "id", "synthetic"}
_MEMORY = "<memory-context>\n[Synapse] Kyle prefers terse answers.\n</memory-context>"


def _call(name: str, arguments: object, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _conversation() -> list[dict]:
    """Two finished turns, the second with a nudge Hermes injected mid-turn, a compaction
    summary before them and an unfinished third turn after."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier the user set up a repo.",
            "_compressed_summary": True,
            "message_uid": "u-summary",
        },
        {
            "role": "user",
            "content": "Which branch is the release on?",
            "api_content": "Which branch is the release on?\n\n" + _MEMORY,
            "timestamp": 1_794_000_000.25,
            "message_uid": "u-1",
            "display_metadata": {"typed_ask": "Which branch is the release on?"},
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Let me check git.",
            "tool_calls": [_call("terminal", {"command": "git branch --show-current"})],
            "timestamp": 1_794_000_001.0,
            "message_uid": "u-2",
            "codex_reasoning_items": [{"opaque": "blob"}],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
            "content": "release/0.24\n" + "x" * 3000,
            "timestamp": 1_794_000_002.0,
            "message_uid": "u-3",
            "_tool_output_risk": {"level": "low"},
        },
        {
            "role": "assistant",
            "content": "The release is on release/0.24.",
            "timestamp": 1_794_000_003.0,
            "message_uid": "u-4",
            "finish_reason": "stop",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now tag it. " + _MEMORY},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "timestamp": 1_794_000_100.0,
            "message_uid": "u-5",
        },
        {
            "role": "assistant",
            "content": "Tagging.",
            "tool_calls": [_call("terminal", '{"command": "git tag v0.24.0"}', "call_2")],
            "message_uid": "u-6",
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_2",
            "content": "",
            "message_uid": "u-7",
        },
        {
            "role": "user",
            "content": "[System: Your previous response was truncated. Continue.]",
            "message_uid": "u-8",
        },
        {
            "role": "user",
            "content": "Remember to push the tag.",
            "_todo_snapshot_synthetic": True,
            "message_uid": "u-9",
        },
        {
            "role": "user",
            "content": "\n\n[OUT-OF-BAND USER MESSAGE — a direct message from the user]\nand push\n",
            "display_kind": "steer",
            "message_uid": "u-10",
        },
        {"role": "assistant", "content": "Tagged v0.24.0 and pushed it.", "message_uid": "u-11"},
        {"role": "user", "content": "Thanks, now the changelog.", "message_uid": "u-12"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("read_file", {"path": "CHANGELOG.md"}, "call_3")],
            "message_uid": "u-13",
        },
    ]


def test_records_keep_only_what_the_server_reads_and_never_touch_the_input():
    messages = _conversation()
    before = copy.deepcopy(messages)
    ids = [id(m) for m in messages]

    records = hm.to_records(messages)

    assert messages == before
    assert [id(m) for m in messages] == ids
    assert all(set(r) <= _KEYS for r in records)
    assert all(r is not m for r in records for m in messages)
    assert records[:4] == [
        {
            "role": "user",
            "content": "Which branch is the release on?",
            "ts": 1_794_000_000.25,
            "id": "u-1",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "terminal", "arguments": '{"command": "git branch --show-current"}'}
            ],
            "ts": 1_794_000_001.0,
            "id": "u-2",
        },
        {
            "role": "tool",
            "content": ("release/0.24\n" + "x" * 3000)[: hm.TOOL_RESULT_CAP],
            "name": "terminal",
            "ts": 1_794_000_002.0,
            "id": "u-3",
        },
        {
            "role": "assistant",
            "content": "The release is on release/0.24.",
            "ts": 1_794_000_003.0,
            "id": "u-4",
        },
    ]
    blob = json.dumps(records)
    for leaked in (
        "memory-context",
        "Synapse]",
        "reasoning",
        "blob",
        "typed_ask",
        "Earlier the user",
    ):
        assert leaked not in blob


def test_text_parts_are_joined_and_memory_context_removed():
    records = hm.to_records(_conversation())
    assert records[4] == {
        "role": "user",
        "content": "Now tag it.",
        "ts": 1_794_000_100.0,
        "id": "u-5",
    }


def test_rows_hermes_injected_are_marked_synthetic():
    by_id = {r["id"]: r for r in hm.to_records(_conversation())}
    assert by_id["u-8"]["synthetic"] is True  # a truncation nudge, by its text
    assert by_id["u-9"]["synthetic"] is True  # a producer flag
    assert by_id["u-10"]["synthetic"] is True  # a /steer row: mid-turn, must not split it
    assert "synthetic" not in by_id["u-5"]
    assert "synthetic" not in by_id["u-12"]


@pytest.mark.parametrize(
    "text",
    [
        "[System note: the model was switched]",
        "[Runtime note: background job finished]",
        "[CONTEXT SUMMARY]: earlier work",
        "[IMPORTANT: Background process 41 exited]",
        "[Your active task list was preserved across context compression]",
        "  Cronjob Response: nightly report",
    ],
)
def test_hermes_control_frames_are_synthetic_by_their_opening_text(text):
    (record,) = hm.to_records([{"role": "user", "content": text}])
    assert record["synthetic"] is True


def test_compaction_rows_of_any_role_and_other_roles_are_dropped():
    records = hm.to_records(
        [
            {"role": "assistant", "content": "[micro compact]", "_compressed_summary": True},
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": _MEMORY},  # nothing left once the block is gone
            "not a dict",
            None,
        ]
    )
    assert records == []
    assert hm.to_records(None) == []


def test_tool_arguments_are_capped_and_user_and_assistant_text_are_not():
    long = "y" * 5000
    records = hm.to_records(
        [
            {"role": "user", "content": long},
            {"role": "assistant", "content": long, "tool_calls": [_call("write", {"body": long})]},
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [{"name": "flat", "arguments": long}],
            },
        ]
    )
    assert records[0]["content"] == long
    assert records[1]["content"] == long
    assert len(records[1]["tool_calls"][0]["arguments"]) == hm.TOOL_ARGUMENTS_CAP
    assert records[2]["tool_calls"] == [
        {"name": "flat", "arguments": long[: hm.TOOL_ARGUMENTS_CAP]}
    ]


def test_turns_start_at_real_user_rows_and_only_finished_ones_count():
    turns = hm.turns(hm.to_records(_conversation()))
    assert [[r["id"] for r in t] for t in turns] == [
        ["u-1", "u-2", "u-3", "u-4"],
        ["u-5", "u-6", "u-7", "u-8", "u-9", "u-10", "u-11"],
        ["u-12", "u-13"],
    ]
    assert [hm.complete(t) for t in turns] == [True, True, False]
    # An assistant row that calls tools is followed by their results: not finished yet.
    assert not hm.complete(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "On it.",
                "tool_calls": [{"name": "t", "arguments": ""}],
            },
        ]
    )
    assert not hm.complete([])


def test_records_before_the_first_user_row_belong_to_no_turn():
    records = [
        {"role": "assistant", "content": "left over"},
        {"role": "user", "content": "[System: resumed]", "synthetic": True},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert hm.turns(records) == [records[2:]]


def test_turn_key_is_the_last_id_else_a_stable_hash():
    with_id = [
        {"role": "user", "content": "q", "id": "a"},
        {"role": "assistant", "content": "r", "id": "b"},
    ]
    assert hm.turn_key(with_id) == "b"
    without = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]
    key = hm.turn_key(without)
    assert key.startswith("sha256:")
    assert key == hm.turn_key(copy.deepcopy(without))
    assert key != hm.turn_key(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "s"}]
    )


def test_a_synthetic_row_after_a_finished_turn_neither_reopens_it_nor_changes_its_key():
    """Hermes appends a task-list snapshot (a synthetic user row) after compression; it
    lands on the end of the turn before it."""
    finished = [
        {"role": "user", "content": "q", "id": "q"},
        {"role": "assistant", "content": "r", "id": "r"},
    ]
    snapshot = {"role": "user", "content": "[Your active task list", "synthetic": True}
    later = [*finished, {**snapshot, "id": "s"}]
    assert hm.complete(later)
    assert hm.turn_key(later) == hm.turn_key(finished) == "r"
    bare = [{k: v for k, v in r.items() if k != "id"} for r in finished]
    assert hm.turn_key([*bare, snapshot]) == hm.turn_key(bare)
    assert not hm.complete([{"role": "user", "content": "q"}, snapshot])


def _numbered(n: int) -> list[list[dict]]:
    return [
        [
            {"role": "user", "content": f"q{i}", "id": f"q{i}"},
            {"role": "assistant", "content": f"a{i}", "id": f"a{i}"},
        ]
        for i in range(n)
    ]


def test_pending_pairs_each_unsent_turn_with_the_finished_turn_before_it():
    turns = _numbered(4)
    unfinished = [{"role": "user", "content": "q-open", "id": "q-open"}]
    pairs = hm.pending([*turns, unfinished], {"a0", "a1"})
    assert [(c and c[-1]["id"], t[-1]["id"]) for c, t in pairs] == [("a1", "a2"), ("a2", "a3")]
    assert hm.pending(turns, set())[0] == (None, turns[0])


def test_pending_with_a_lookback_considers_only_the_last_turns():
    turns = _numbered(15)
    pairs = hm.pending(turns, set(), lookback=10)
    assert [t[-1]["id"] for _, t in pairs] == [f"a{i}" for i in range(5, 15)]
    assert pairs[0][0] == turns[4]  # its context is still the turn before


def test_batches_split_by_count_and_lead_each_with_its_context():
    turns = _numbered(45)
    out = hm.batches(hm.pending(turns, set()))
    assert [len(new) for _, new in out] == [20, 20, 5]
    first, second, third = out
    assert first[0][0]["id"] == "q0"  # no context before the very first turn
    assert second[0][:2] == turns[19]  # the turn before its first new one leads
    assert second[0][2:4] == turns[20]
    assert third[0][:2] == turns[39]
    assert len(second[0]) == 2 * 21


def test_batches_split_by_size_and_an_oversized_turn_goes_alone():
    big = [
        [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "z" * 100_000}]
        for i in range(5)
    ]
    out = hm.batches(hm.pending(big, set()), max_bytes=256_000)
    # The leading context turn is part of the body, so it counts toward the size.
    assert [len(new) for _, new in out] == [2, 1, 1, 1]
    assert all(len(json.dumps(records)) < 256_000 for records, _ in out)
    huge = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "z" * 400_000}]]
    assert [len(new) for _, new in hm.batches(hm.pending(huge, set()))] == [1]


def _episodes(records: list[dict]):
    return MessagesParser().parse_records(
        records, "hermes-sync", None, session_id=SID, client="hermes", cwd="/work/repo", model="m-1"
    )


def test_the_servers_parser_reads_the_records_as_the_two_finished_turns():
    turns = hm.turns(hm.to_records(_conversation()))
    records = [r for t in turns if hm.complete(t) for r in t]
    first, second = _episodes(records)

    assert first.span_id == "hermes:u-4"
    assert first.human_turn == "Which branch is the release on?"
    assert first.assistant_turn == "The release is on release/0.24."
    assert first.platform == "hermes" and first.source == "messages"
    assert first.project == "repo" and first.model == "m-1"
    assert '[tool:terminal] {"command": "git branch --show-current"}' in first.content
    assert "[result] release/0.24" in first.content

    assert second.span_id == "hermes:u-11"
    assert second.human_turn == "Now tag it."
    assert second.assistant_turn == "Tagged v0.24.0 and pushed it."
    assert second.content.startswith("[context] The release is on release/0.24.")
    for absent in ("truncated", "Remember to push", "OUT-OF-BAND", "memory-context", "Synapse]"):
        assert absent not in second.content


def test_a_turn_resent_with_its_context_converges_on_the_same_span_id():
    turns = hm.turns(hm.to_records(_conversation()))
    alone = _episodes(turns[1])
    with_context = _episodes([*turns[0], *turns[1]])
    assert alone[0].span_id == with_context[1].span_id == "hermes:u-11"
