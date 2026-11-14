"""Failed turns, and the other assistant rows that are not a reply.

When a turn fails Hermes closes it with an assistant row typed ``display_kind``
"failed_turn" whose text is a notice ("This turn did not complete..."), on the agent
path (agent/conversation_loop.py) and the gateway's (gateway/run_turn_persistence.py,
gateway/session_results.py). Rows persisted before Hermes typed them carry the bare
notice (agent/turn_failure_copy.py ``untyped_failed_turn_display_kind``). A held turn's
interim rows and the interrupt placeholder are typed "hidden"
(agent/stream_delivery.py, agent/agent_runtime_helpers_placeholders.py). The rows below
copy those shapes. None of that text may be stored as something the assistant said.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ingestion.messages_client import MessagesParser
from tests import test_hermes_provider as base
from tests.helpers.client_core import use_core

use_core()
from synapse_client import messages as cm  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402

hermes_stub = base.hermes_stub
user = base.user
plugin = base.plugin
server = base.server

FAILED = "Your request was not processed. Send it again if you still want me to carry it out."
PARTIAL = (
    "This turn did not complete. Some actions may already have run; verify their effects "
    "before resending."
)
AUTOMATION = (
    "An automated delivery was interrupted and was not run again. Some "
    "actions may already have run; verify their effects."
)
_SID = "20261112_150000_f00d01"


def _user(uid: str, text: str) -> dict[str, Any]:
    return {"role": "user", "content": text, "message_uid": uid, "timestamp": 1_794_000_000.0}


def _reply(uid: str, text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "message_uid": uid, "finish_reason": "stop"}


def _calls(uid: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command": "make deploy"}'},
            }
        ],
        "message_uid": uid,
    }


def _result(uid: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": "deploy started",
        "tool_call_id": "call_1",
        "tool_name": "terminal",
        "message_uid": uid,
    }


def _boundary(text: str = PARTIAL, **extra: Any) -> dict[str, Any]:
    """The agent path's boundary row (conversation_loop.py), without a message_uid."""
    return {
        "role": "assistant",
        "content": text,
        "display_kind": "failed_turn",
        "display_metadata": {"failure": {"kind": "provider_error"}},
        **extra,
    }


def _conversation() -> list[dict[str, Any]]:
    """A finished turn, a turn that failed after running a tool, and the user's resend."""
    return [
        {"role": "system", "content": "You are Hermes."},
        _user("u-1", "What is on the release branch?"),
        _reply("a-1", "Two commits since the last tag."),
        _user("u-2", "Deploy it to staging."),
        _calls("c-2"),
        _result("t-2"),
        _boundary(),
        _user("u-3", "Try the deploy again."),
        _reply("a-3", "Staging is on the new build."),
    ]


def _episodes(records: list[dict[str, Any]]) -> list[Any]:
    return MessagesParser().parse_records(
        records, "hermes-sync", None, session_id=_SID, client="hermes", cwd="/work/repo"
    )


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        _boundary(FAILED),
        _boundary(PARTIAL),
        _boundary(AUTOMATION),
        # The gateway's own boundary (run_turn_persistence.py): its own notice, a timestamp.
        _boundary("The gateway restarted before this turn finished.", timestamp=1_794_000_009.0),
        _boundary(PARTIAL, message_uid="m-9"),
        # Persisted before the closers typed it: the bare notice.
        _reply("a-9", FAILED),
        _reply("a-9", "  " + PARTIAL + "\n"),
        {"role": "assistant", "content": [{"type": "text", "text": AUTOMATION}]},
    ],
)
def test_a_failure_boundary_is_never_a_reply(row):
    assert hm.to_records([row]) == []


def test_a_reply_that_quotes_a_notice_stays_a_reply():
    text = f'Hermes printed "{PARTIAL}" because the deploy timed out; it did run once.'
    (record,) = hm.to_records([_reply("a-1", text)])
    assert record["content"] == text


def test_hidden_rows_keep_only_their_tool_calls():
    placeholder = {
        "role": "assistant",
        "content": "",
        "display_kind": "hidden",
        "api_content": "[interrupt: no assistant output for this turn]",
    }
    rejected = {**_reply("a-5", "Done, all tests pass."), "display_kind": "hidden"}
    commentary = {**_calls("c-5"), "content": "Checking the build first.", "display_kind": "hidden"}
    unknown = {**_reply("a-6", "A notice of some new kind."), "display_kind": "credits_notice"}
    assert hm.to_records([placeholder, rejected, unknown]) == []
    (record,) = hm.to_records([commentary])
    assert record == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "terminal", "arguments": '{"command": "make deploy"}'}],
        "id": "c-5",
    }


def test_a_turn_whose_only_reply_was_held_back_is_not_complete():
    rows = [_user("u-5", "Fix the tests."), {**_reply("a-5", "Done."), "display_kind": "hidden"}]
    (turn,) = cm.turns(hm.to_records(rows))
    assert not cm.complete(turn)


@pytest.mark.parametrize(
    "kind",
    [
        "steer",
        "personality_switch",
        "process_complete",
        "async_delegation_complete",
        "internal_notification",
        "hidden",
        "content_refused",
        "background_result",
    ],
)
def test_every_kind_hermes_puts_on_a_user_row_is_synthetic(kind):
    (record,) = hm.to_records([{**_user("u-7", "a row Hermes wrote"), "display_kind": kind}])
    assert record["synthetic"] is True


# ---------------------------------------------------------------------------
# The turns around a failure
# ---------------------------------------------------------------------------


def _text(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False)


def test_a_failed_turn_never_completes_and_its_message_is_not_sent():
    turns = cm.turns(hm.to_records(_conversation()))
    assert [cm.complete(t) for t in turns] == [True, False, True]
    (only,) = cm.batches(cm.pending(turns, set()))
    records, new = only
    assert [cm.turn_key(t) for t in new] == ["a-1", "a-3"]
    for absent in ("Deploy it to staging", "make deploy", "did not complete", "provider_error"):
        assert absent not in _text(records)
    first, second = _episodes(records)
    assert (first.human_turn, second.human_turn) == (
        "What is on the release branch?",
        "Try the deploy again.",
    )
    assert second.content.startswith("[context] Two commits since the last tag.")
    assert second.span_id == "hermes:a-3"


def test_the_turn_after_a_failure_leads_with_the_last_complete_turn():
    turns = cm.turns(hm.to_records(_conversation()))
    ((records, new),) = cm.batches(cm.pending(turns, {"a-1"}))
    assert [cm.turn_key(t) for t in new] == ["a-3"]
    assert [r.get("id") for r in records] == ["u-1", "a-1", "u-3", "a-3"]
    (episode,) = _episodes(records)[1:]
    assert episode.content.startswith("[context] Two commits since the last tag.")


def test_a_failure_at_the_end_sends_nothing():
    rows = _conversation()[:7]
    turns = cm.turns(hm.to_records(rows))
    assert cm.pending(turns, {"a-1"}) == []


# ---------------------------------------------------------------------------
# Through the provider
# ---------------------------------------------------------------------------


def test_the_provider_uploads_no_failed_turn(plugin, server, tmp_path):
    provider = base._start(base._new(plugin), tmp_path / "hermes-a", _SID)
    rows = _conversation()
    provider.on_turn_start(1, "q")
    provider.sync_turn("q", "a", messages=rows[:3])
    # The gateway does not call sync_turn for a failed turn; the boundary reaches the
    # provider in on_session_end's (or on_pre_compress's) list.
    provider.on_turn_start(2, "q", previous_message="q")
    provider.on_session_end(rows[:7])
    provider.on_pre_compress(rows[:7])
    assert len(server.posts()) == 1
    provider.on_turn_start(3, "q", previous_message="q")
    provider.sync_turn("q", "a", messages=rows)
    _, second = server.posts()
    assert [r.get("id") for r in second["records"]] == ["u-1", "a-1", "u-3", "a-3"]
    sent = json.dumps(server.posts(), ensure_ascii=False)
    for absent in (FAILED, PARTIAL, AUTOMATION, "Deploy it to staging", "provider_error"):
        assert absent not in sent
