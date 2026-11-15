"""Hermes uploads read from the session store: the review findings, one test each.

Each test builds ``state.db`` with Hermes's own tables and row shapes
(tests/helpers/hermes_store.py), drives the provider as Hermes does (``sync_turn``
with the list Hermes hands over, after the run's rows are in the store), and reads what
the server would keep: the posted bodies through the server's own ``MessagesParser``,
one episode per span id and none whose content is already stored in the project, as
the ingest route keeps them (mcp_server/ingest_route.py).

The findings are those of the two reviews of the list-based live path
(followups/hermes-turn-cut/review-round1 and review-round2): each test names its
script. Where a reviewer's fixture and a store Hermes v0.23.1 really wrote disagree,
the test follows the real store and says so: a ``/model`` marker or a note is a row of
its own, and the next message another (probe session ``20261115_021935_147097``, rows
75 and 76); only the live list merges them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ingestion.messages_client import MessagesParser
from tests import test_hermes_provider as hp
from tests.helpers.client_core import PLUGIN
from tests.helpers.hermes_store import HermesStore, call

hermes_stub = hp.hermes_stub
user = hp.user
plugin = hp.plugin
server = hp.server
_new, _start = hp._new, hp._start

SID = "20261115_101500_aa11bb"
MODEL = (
    "[System: The active model for this chat has changed to gpt-x via provider y. From this "
    "point forward, use this runtime metadata when answering questions about what "
    "model/provider is active.]"
)
PERSONA = (
    "[System: The user has changed the assistant's personality. From this point forward, "
    "adopt the following persona and respond accordingly: You are a pirate.\nRules:\n"
    "[always say arr]\n\nPERSONA-TAIL Never break character.]"
)
SKILL = (
    '[IMPORTANT: The user has invoked the "research" skill, indicating they want you to '
    "follow its instructions. The full skill content is loaded below.]\n\nSKILLBODY Search "
    "widely, then summarise.\n\nThe user has provided the following instruction alongside "
    "the skill invocation: USERTOKEN-skill instruction"
)


# -- what the server keeps -----------------------------------------------------------


def stored(server: Any) -> list[Any]:
    """The episodes the ingest route would keep from every body posted, in order."""
    kept: list[Any] = []
    spans: set[str] = set()
    contents: set[tuple[Any, str]] = set()
    for body in server.posts():
        for ep in MessagesParser().parse_records(
            body["records"],
            body.get("source") or "t",
            body.get("project"),
            session_id=body["session_id"],
            client=body["client"],
            cwd=body.get("cwd"),
            model=body.get("model"),
        ):
            digest = hashlib.md5(ep.content.encode("utf-8"), usedforsecurity=False).hexdigest()
            if not ep.span_id or ep.span_id in spans or (ep.project, digest) in contents:
                continue
            spans.add(ep.span_id)
            contents.add((ep.project, digest))
            kept.append(ep)
    return kept


def count(episodes: list[Any], token: str) -> int:
    """In how many episodes ``token`` appears outside a ``[context]`` line."""
    total = 0
    for ep in episodes:
        lines = [line for line in ep.content.split("\n") if not line.startswith("[context]")]
        total += any(token in line for line in lines)
    return total


# -- a session as Hermes runs it -------------------------------------------------------


class Session:
    """A store, a provider bound to one in-process session, and the live list."""

    def __init__(self, plugin: ModuleType, home: Path, sid: str = SID) -> None:
        self.home, self.sid, self.plugin = home, sid, plugin
        self.store = HermesStore(home)
        self.store.session(sid)
        self.live: list[dict[str, Any]] = [{"role": "system", "content": "You are Hermes."}]
        self.provider = self.agent()

    def agent(self) -> Any:
        """A provider Hermes built for the session (a rebuild builds another)."""
        return _start(_new(self.plugin), self.home, self.sid, platform="tui")

    def turn(self, user: str, reply: str, **kwargs: Any) -> None:
        self.live += self.store.turn(self.sid, user, reply, **kwargs)
        self.sync()

    def report(self, text: str, reply: str) -> None:
        self.live += self.store.report(self.sid, text, reply)
        self.sync()

    def sync(self, live: list[dict[str, Any]] | None = None, provider: Any = None) -> None:
        """The end of a run: Hermes has written its rows, then calls ``sync_turn``."""
        held = self.live if live is None else live
        (provider or self.provider).sync_turn("q", "a", session_id=self.sid, messages=held)


@pytest.fixture()
def chat(plugin, server, tmp_path) -> Session:
    return Session(plugin, tmp_path / "hermes-a")


def _import(home: Path) -> int:
    path = PLUGIN / "scripts" / "import_hermes.py"
    spec = importlib.util.spec_from_file_location("import_hermes_store_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(["--hermes-home", str(home), "--yes"])


# -- the original double upload ------------------------------------------------------


def test_a_report_after_a_finished_turn_ships_each_row_once(chat, server):
    """review-round2 t5, t9 (f_b, f_f), h_store ``report_after_turn``: Hermes appended a
    background report and the reply to it after the turn was uploaded; the turn grew,
    its key changed and it was uploaded again, its user text stored twice. Now the
    second upload sends only the rows not shipped: the reply, as an episode of its own.
    The report's text is never sent."""
    chat.turn("USERTOKEN-first question", "FINALTOKEN-first answer")
    chat.report("SUBAGENT-OUTPUT the subagent found 391", "FINALTOKEN-report reply")
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-first question") == 1
    assert count(episodes, "FINALTOKEN-first answer") == 1
    assert count(episodes, "FINALTOKEN-report reply") == 1
    assert count(episodes, "SUBAGENT-OUTPUT") == 0
    assert len(episodes) == 2


def test_an_agent_rebuilt_mid_session_ships_nothing_again(chat, server):
    """Probe, gateway ``/model``: Hermes rebuilt the agent with no ``on_session_end``,
    and the new agent's list had lost the report row's ``display_kind``. The store kept
    it, and what was shipped is on disk: the new agent sends only its own turn."""
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    chat.report("SUBAGENT-OUTPUT done", "FINALTOKEN-report")
    rebuilt = chat.agent()
    lost = [{k: v for k, v in row.items() if k != "display_kind"} for row in chat.live]
    chat.live += chat.store.turn(chat.sid, "USERTOKEN-two", "FINALTOKEN-two")
    chat.sync([*lost, *chat.live[len(lost) :]], provider=rebuilt)
    episodes = stored(server)
    for token in ("USERTOKEN-one", "FINALTOKEN-one", "FINALTOKEN-report", "USERTOKEN-two"):
        assert count(episodes, token) == 1, token
    assert len(server.posts()) == 3


def test_two_reports_answered_alike_are_both_stored(chat, server):
    """review-round1/2 t4: two reports whose replies read the same made two episodes
    with the same content, and the server keeps one per content. Each part that goes on
    from a turn is led by a label with its time, so both are stored."""
    chat.turn("USERTOKEN-start the build", "FINALTOKEN-started")
    chat.report("PROCESS-OUTPUT build 1 ok", "The build finished successfully.")
    chat.turn("USERTOKEN-run it again", "FINALTOKEN-started again")
    chat.report("PROCESS-OUTPUT build 2 ok", "The build finished successfully.")
    episodes = stored(server)
    assert count(episodes, "The build finished successfully.") == 2
    assert count(episodes, "PROCESS-OUTPUT") == 0


# -- rows Hermes writes inside a run ---------------------------------------------------


@pytest.mark.parametrize(
    "nudge",
    [
        "[System: Continue now. Execute the required tool calls and only send your final "
        "answer after completing the task.]",
        "[System: Your previous response was truncated by the output length limit. Continue "
        "exactly where you left off.]",
        "[System note: the language server reported problems in files you edited this "
        "turn.]\n\nx.py:3 undefined name",
    ],
    ids=["codex ack", "length continuation", "late LSP note"],
)
def test_nudges_inside_a_run_stay_in_its_turn(chat, server, nudge):
    """review-round1/2 t1: a nudge Hermes writes after an interim reply, inside the run.
    The run is uploaded once it has ended, from the store: one episode, the user's text
    once, the final reply once."""
    chat.turn("USERTOKEN-earlier", "FINALTOKEN-earlier")
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "USERTOKEN-fix the failing test"),
        s.add(sid, "assistant", "INTERIM I will look into it."),
        s.add(sid, "user", nudge),
        s.add(sid, "assistant", "", tool_calls=[call()]),
        s.add(sid, "tool", "1 passed after the fix", tool_name="terminal"),
        s.add(sid, "assistant", "FINALTOKEN-fixed"),
    ]
    chat.sync()
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-fix the failing test") == 1
    assert count(episodes, "FINALTOKEN-fixed") == 1
    assert len(episodes) == 2


def test_a_steer_inside_a_run_stays_in_its_turn(chat, server):
    """h_store ``steer``: a ``/steer`` message lands mid-run with ``display_kind`` steer."""
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "USERTOKEN-do x"),
        s.add(sid, "assistant", "", tool_calls=[call()]),
        s.add(sid, "tool", "RESULT some long tool output text here", tool_name="terminal"),
        s.add(
            sid, "user", "[OUT-OF-BAND USER MESSAGE]\nSTEER-TEXT go faster", display_kind="steer"
        ),
        s.add(sid, "assistant", "FINALTOKEN-done"),
    ]
    chat.sync()
    (episode,) = stored(server)
    assert episode.human_turn == "USERTOKEN-do x" and "FINALTOKEN-done" in episode.content


def test_a_flush_inside_a_run_stores_the_users_text_once(chat, server):
    """review-round2 t12: Hermes flushes before a compression inside a run, right after
    an interim reply. The flush reads the store up to the row it was handed; the rest of
    the run goes up once the run ends, as the rest of that turn."""
    chat.turn("USERTOKEN-0 question", "FINALTOKEN-0 answer")
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "USERTOKEN-1 write the long plan"),
        s.add(sid, "assistant", "INTERIM first half of the pl"),
        s.add(sid, "user", "[System: Your previous response was truncated. Continue.]"),
    ]
    chat.provider.on_pre_compress(list(chat.live))
    chat.live.append(s.add(sid, "assistant", "FINALTOKEN-1 an, second half."))
    chat.sync()
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-1 write the long plan") == 1
    assert count(episodes, "FINALTOKEN-1") == 1
    assert count(episodes, "USERTOKEN-0 question") == 1


def test_the_next_run_is_not_read(chat, server):
    """A hook reads the store up to the last row of the list it was handed: the next
    run's user row, written as it is submitted, and its tool rows wait for that run."""
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    sid, s = chat.sid, chat.store
    upto = list(chat.live)
    s.add(sid, "user", "USERTOKEN-two")
    s.add(sid, "assistant", "", tool_calls=[call()])
    s.add(sid, "assistant", "INTERIM-TWO looking")
    chat.sync(upto)
    assert count(stored(server), "USERTOKEN-two") == 0
    assert count(stored(server), "INTERIM-TWO") == 0


def test_a_trailing_message_with_no_reply_waits(chat, server):
    sid, s = chat.sid, chat.store
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    chat.live.append(s.add(sid, "user", "USERTOKEN-unanswered"))
    chat.provider.on_session_end(list(chat.live))
    assert count(stored(server), "USERTOKEN-unanswered") == 0
    chat.live.append(s.add(sid, "assistant", "FINALTOKEN-late"))
    chat.sync()
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-unanswered") == 1 and count(episodes, "FINALTOKEN-late") == 1


# -- markers and notes: the store keeps them as rows of their own ----------------------


def _merged(marker: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """The live list's merge of a row that got no reply with the next message
    (agent/agent_runtime_helpers.py ``_merge_consecutive_users``): the marker's uid, both
    texts, no ``display_kind``. Probe: the TUI's list after ``/model``."""
    out = {k: v for k, v in marker.items() if k != "display_kind"}
    out["content"] = marker["content"] + "\n\n" + message["content"]
    out["_merged_turn_prefix"] = marker["content"]
    out["_absorbed_message_uids"] = [message["message_uid"]]
    return out


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("model_switch", MODEL),
        ("personality_switch", PERSONA),
        ("async_delegation_complete", "[ASYNC DELEGATION COMPLETE]\nSUBAGENT-OUTPUT 391"),
        ("process_complete", "[IMPORTANT: Background process exited.]\nPROCESS-OUTPUT ok"),
        ("internal_notification", "CRON-PLAIN nightly report: all good"),
        ("hidden", "HIDDEN-TEXT goal control"),
    ],
)
def test_a_marker_or_note_and_the_next_message_are_two_rows(chat, server, kind, text):
    """review-round1/2 t2, round2 t8, h_note, h_store ``model_marker``,
    ``persona_terminator``, ``note_*``. The reviewers' fixtures stored the merged row
    (marker and message in one ``content``); a store Hermes v0.23.1 wrote keeps the
    marker as its own row and the message as another. The list Hermes hands over still
    merges them. The user's message is stored once, as theirs; the marker, the persona
    and the note's text never."""
    chat.turn("USERTOKEN-first question", "FINALTOKEN-first answer")
    sid, s = chat.sid, chat.store
    marker = s.add(sid, "user", text, display_kind=kind)
    message = s.add(sid, "user", "USERTOKEN-second real message")
    reply = s.add(sid, "assistant", "FINALTOKEN-second reply")
    chat.sync([*chat.live, _merged(marker, message), reply])
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-second real message") == 1
    assert count(episodes, "FINALTOKEN-second reply") == 1
    assert count(episodes, "USERTOKEN-first question") == 1
    for secret in (
        "SUBAGENT-OUTPUT",
        "PROCESS-OUTPUT",
        "CRON-PLAIN",
        "HIDDEN-TEXT",
        "PERSONA-TAIL",
    ):
        assert count(episodes, secret) == 0
    assert all("active model" not in ep.content for ep in episodes)


def test_two_markers_then_a_message(chat, server):
    """h_store ``two_markers``."""
    chat.turn("USERTOKEN-first", "FINALTOKEN-first")
    sid, s = chat.sid, chat.store
    a = s.add(sid, "user", MODEL, display_kind="model_switch")
    b = s.add(sid, "user", PERSONA, display_kind="personality_switch")
    message = s.add(sid, "user", "USERTOKEN-after switch")
    reply = s.add(sid, "assistant", "FINALTOKEN-after switch")
    merged = _merged(_merged(a, b), message)
    chat.sync([*chat.live, merged, reply])
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-after switch") == 1
    assert count(episodes, "PERSONA-TAIL") == 0


def test_a_marker_with_no_message_after_it_sends_nothing(chat, server):
    """h_store ``marker_only_tail``: a marker nobody answered changes no turn."""
    chat.turn("USERTOKEN-first", "FINALTOKEN-first")
    chat.live.append(chat.store.add(chat.sid, "user", MODEL, display_kind="model_switch"))
    chat.provider.on_session_end(list(chat.live))
    assert len(server.posts()) == 1


# -- pictures and rows Hermes never stores ------------------------------------------------


def test_a_picture_alone_is_stored_once_live_and_imported(chat, server):
    """review-round1/2 t3 (a), h_store ``picture_only``: the live list holds the image
    part, the store ``[screenshot]``. The store is read either way, so importing after
    the live upload adds nothing."""
    chat.turn("USERTOKEN-q0", "FINALTOKEN-a0")
    picture = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "[screenshot]", live=picture),
        s.add(sid, "assistant", "FINALTOKEN-That is a receipt for 42 dollars."),
    ]
    chat.sync()
    posted = len(server.posts())
    assert _import(chat.home) == 0
    assert len(server.posts()) == posted
    episodes = stored(server)
    assert count(episodes, "FINALTOKEN-That is a receipt") == 1
    assert count(episodes, "FINALTOKEN-a0") == 1


def test_the_dropped_tool_call_pair_never_reaches_the_server(chat, server):
    """review-round1/2 t3 (b), h_store ``ephemeral_pair``: the re-prompt pair after a
    dropped tool call is in the live list only (no uid, never stored)."""
    sid, s = chat.sid, chat.store
    user_row = s.add(sid, "user", "USERTOKEN-run the tests")
    pair = [
        {"role": "assistant", "content": "Running the tests now.", "_dropped_toolcall_nudge": True},
        {
            "role": "user",
            "content": "[System: Your tool call was dropped.]",
            "_dropped_toolcall_nudge": True,
        },
    ]
    rest = [
        s.add(sid, "assistant", "", tool_calls=[call()]),
        s.add(sid, "tool", "12 passed in 3.2 seconds, no failures", tool_name="terminal"),
        s.add(sid, "assistant", "FINALTOKEN-All 12 tests pass."),
    ]
    chat.sync([*chat.live, user_row, *pair, *rest])
    assert all("Running the tests now" not in ep.content for ep in stored(server))
    posted = len(server.posts())
    assert _import(chat.home) == 0
    assert len(server.posts()) == posted


# -- runs Hermes starts itself ---------------------------------------------------------


def test_a_session_that_opens_with_a_summary_and_a_report(chat, server):
    """review-round1/2 t6, h_store ``report_first``: the first rows hold no real user
    row. The report's reply is a turn of its own, stored once, and so is the next."""
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "[CONTEXT COMPACTION] summary of earlier work", summary=True),
        *s.report(sid, "SUBAGENT-OUTPUT found the bug", "FINALTOKEN-The subagent found the bug."),
    ]
    chat.sync()
    chat.turn("USERTOKEN-thanks, now fix it", "FINALTOKEN-Fixed.")
    episodes = stored(server)
    assert count(episodes, "FINALTOKEN-The subagent found the bug.") == 1
    assert count(episodes, "USERTOKEN-thanks, now fix it") == 1
    assert count(episodes, "summary of earlier work") == 0


def test_a_report_after_a_failed_run_completes_that_run(chat, server):
    """h_store ``failed_then_report``: a run that ended with Hermes's failure boundary
    and no reply, then a report. The failed message goes up with the reply that came
    after it, once; the boundary text never."""
    chat.turn("USERTOKEN-first", "FINALTOKEN-first")
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "USERTOKEN-failed ask"),
        s.add(
            sid,
            "assistant",
            "This turn did not complete. Some actions may already have run; verify their "
            "effects before resending.",
            display_kind="failed_turn",
        ),
    ]
    chat.sync()
    chat.report("SUBAGENT-OUTPUT x", "FINALTOKEN-report reply")
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-failed ask") == 1
    assert count(episodes, "FINALTOKEN-report reply") == 1
    assert count(episodes, "did not complete") == 0


def test_an_auto_continue_after_an_interim_reply(chat, server):
    """h_store ``auto_continue_after_interim``."""
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", "USERTOKEN-long task"),
        s.add(sid, "assistant", "INTERIM I will start now."),
    ]
    chat.sync()
    chat.live += [
        s.add(
            sid,
            "user",
            "[System note: the app stopped before the turn could finish.]\n\nUSERTOKEN-long task",
            display_kind="auto_continue",
        ),
        s.add(sid, "assistant", "FINALTOKEN-done at last"),
    ]
    chat.sync()
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-long task") == 1
    assert count(episodes, "FINALTOKEN-done at last") == 1


def test_a_skill_run_after_a_finished_turn(chat, server):
    """review-round2 t13, h_skill: the skill row as the CLI stores it (the whole
    scaffold, no ``display_kind``). Nothing is stored twice. The instruction inside the
    scaffold is not sent, as the released client did not send it."""
    chat.turn("USERTOKEN-first question", "FINALTOKEN-first answer")
    sid, s = chat.sid, chat.store
    chat.live += [
        s.add(sid, "user", SKILL),
        s.add(sid, "assistant", "FINALTOKEN-skill reply: vendor A is cheaper"),
    ]
    chat.sync()
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-first question") == 1
    assert count(episodes, "FINALTOKEN-skill reply") == 1
    assert count(episodes, "SKILLBODY") == 0


# -- restarts, look-back, compaction, branches ------------------------------------------


def test_unsent_turns_across_a_restart_all_go(chat, server):
    """review-round2 t9 (f_c, UNSENT=1) and t11: more than ten turns ended while no
    upload ran (the agent was gone), then a new agent. What was shipped is on disk, so
    the new agent sends every turn not shipped, not only the last ten."""
    chat.turn("USERTOKEN-0", "FINALTOKEN-0")
    sid, s = chat.sid, chat.store
    for n in range(1, 14):
        chat.live += s.turn(sid, f"USERTOKEN-{n} q", f"FINALTOKEN-{n} a")
        chat.live += s.report(sid, f"SUBAGENT-OUTPUT {n}", f"FINALTOKEN-report-{n} r")
    later = chat.agent()
    chat.sync(provider=later)
    episodes = stored(server)
    for n in range(1, 14):
        assert count(episodes, f"USERTOKEN-{n} q") == 1, n
        assert count(episodes, f"FINALTOKEN-report-{n} r") == 1, n


def test_a_compaction_ships_nothing_again(chat, server):
    """An in-place compaction archives every row and copies the kept ones with their
    uids (probe rows 27-40). The copies and the summary go nowhere; the next turn goes
    once; the flush with the list from before the compaction sends nothing."""
    for n in range(1, 5):
        chat.turn(f"USERTOKEN-{n}", f"FINALTOKEN-{n}", tools=n == 2)
    before = list(chat.live)
    chat.provider.on_pre_compress(before)
    chat.live = chat.store.compact(chat.sid, keep=4)
    chat.provider.on_session_end(before)
    chat.turn("USERTOKEN-5", "FINALTOKEN-5")
    episodes = stored(server)
    for n in range(1, 6):
        assert count(episodes, f"USERTOKEN-{n}") == 1, n
    assert count(episodes, "PRIOR CONTEXT") == 0
    assert len(server.posts()) == 5


def test_a_branch_ships_none_of_its_parents_rows(plugin, server, tmp_path):
    """A ``/branch`` child holds copies of the parent's rows with their uids: none goes
    up under the child's id, the child's own turn does."""
    chat = Session(plugin, tmp_path / "hermes-a")
    chat.turn("USERTOKEN-parent one", "FINALTOKEN-parent one")
    chat.turn("USERTOKEN-parent two", "FINALTOKEN-parent two")
    child = "20261115_111111_cc22dd"
    copied = chat.store.branch(chat.sid, child)
    chat.provider.on_session_switch(child, parent_session_id=chat.sid, reason="branch")
    own = chat.store.turn(child, "USERTOKEN-child", "FINALTOKEN-child")
    chat.provider.sync_turn("q", "a", session_id=child, messages=[*copied, *own])
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-parent one") == 1
    assert count(episodes, "USERTOKEN-parent two") == 1
    assert count(episodes, "USERTOKEN-child") == 1
    child_posts = [b for b in server.posts() if b["session_id"] == child]
    assert all("USERTOKEN-parent" not in json.dumps(b["records"]) for b in child_posts[:-1])


def test_a_turn_no_upload_ran_for_goes_up_before_a_branch_copies_it(plugin, server, tmp_path):
    """A run ended while no upload ran (the agent was gone), then ``/branch``. The child
    holds a copy of that turn, which never goes up under the child's id; the session
    left ships it under its own id as the switch happens. A run the child goes on with
    after copied rows sends only what follows them."""
    chat = Session(plugin, tmp_path / "hermes-a")
    chat.turn("USERTOKEN-shipped", "FINALTOKEN-shipped")
    chat.live += chat.store.turn(chat.sid, "USERTOKEN-unsent", "FINALTOKEN-unsent")
    child = "20261115_111111_ee44ff"
    copied = chat.store.branch(chat.sid, child)
    chat.provider.on_session_switch(child, parent_session_id=chat.sid, reason="branch")
    own = chat.store.turn(child, "USERTOKEN-child", "FINALTOKEN-child")
    chat.provider.sync_turn("q", "a", session_id=child, messages=[*copied, *own])
    episodes = stored(server)
    for token in ("USERTOKEN-shipped", "USERTOKEN-unsent", "USERTOKEN-child"):
        assert count(episodes, token) == 1, token
    unsent = [b for b in server.posts() if "USERTOKEN-unsent" in json.dumps(b["records"])]
    assert unsent and all(b["session_id"] == chat.sid for b in unsent)


def test_on_session_end_twice_sends_nothing_again(chat, server):
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    chat.provider.on_session_end(list(chat.live))
    chat.provider.on_session_end(list(chat.live))
    assert len(server.posts()) == 1


def test_a_failed_upload_waits_in_the_outbox_and_goes_once(chat, server):
    """The batch goes to the outbox before its request; what was shipped is recorded
    once it is there. The next upload drains it, and nothing goes twice."""
    server.ingest_status = 503
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    server.ingest_status = 200
    chat.turn("USERTOKEN-two", "FINALTOKEN-two")
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-one") == 1 and count(episodes, "USERTOKEN-two") == 1


# -- the importer and the live path share one ledger -----------------------------------


def test_live_then_import_and_import_then_live_add_nothing(plugin, server, tmp_path):
    """review-round2 t10 and eq.py: the importer reads through the provider's reader
    and records what it sends where the provider does."""
    chat = Session(plugin, tmp_path / "hermes-a")
    chat.turn("USERTOKEN-one", "FINALTOKEN-one", tools=True)
    chat.report("SUBAGENT-OUTPUT", "FINALTOKEN-report")
    chat.turn("USERTOKEN-two", "FINALTOKEN-two")
    posted = len(server.posts())
    assert _import(chat.home) == 0
    assert len(server.posts()) == posted

    other = Session(plugin, tmp_path / "hermes-b", "20261115_120000_dd33ee")
    other.provider.shutdown()
    other.live += other.store.turn(other.sid, "USERTOKEN-three", "FINALTOKEN-three")
    other.live += other.store.report(other.sid, "SUBAGENT-OUTPUT", "FINALTOKEN-report-b")
    assert _import(other.home) == 0
    posted = len(server.posts())
    later = other.agent()
    other.live += other.store.turn(other.sid, "USERTOKEN-four", "FINALTOKEN-four")
    other.sync(provider=later)
    episodes = stored(server)
    for token in ("USERTOKEN-three", "FINALTOKEN-report-b", "USERTOKEN-four"):
        assert count(episodes, token) == 1, token
    assert len(server.posts()) == posted + 1


# -- the fallback ------------------------------------------------------------------------


def test_a_store_of_an_unknown_schema_falls_back_to_the_list(plugin, server, tmp_path, caplog):
    """A newer Hermes: the list Hermes hands over is uploaded instead, the reason is
    logged once, and ``synapse status`` can say so."""
    home = tmp_path / "hermes-a"
    store = HermesStore(home, version=34)
    store.session(SID)
    provider = _start(_new(plugin), home, SID, platform="tui")
    live = store.turn(SID, "USERTOKEN-one", "FINALTOKEN-one")
    with caplog.at_level(logging.WARNING):
        provider.sync_turn("q", "a", session_id=SID, messages=live)
        live += store.turn(SID, "USERTOKEN-two", "FINALTOKEN-two")
        provider.sync_turn("q", "a", session_id=SID, messages=live)
    episodes = stored(server)
    assert count(episodes, "USERTOKEN-one") == 1 and count(episodes, "USERTOKEN-two") == 1
    said = [r.getMessage() for r in caplog.records if "live list" in r.getMessage()]
    assert len(said) == 1 and "schema version 34" in said[0]
    status = json.loads((home / "synapse" / "store_fallback.json").read_text("utf-8"))
    assert "schema version 34" in status["reason"]


def test_a_turn_missing_from_the_store_falls_back_to_the_list(chat, server):
    """Hermes failed to write the turn: the row the hook names is not in the store."""
    unstored = [
        {"role": "user", "content": "USERTOKEN-lost write", "message_uid": "not-stored-1"},
        {"role": "assistant", "content": "FINALTOKEN-lost write", "message_uid": "not-stored-2"},
    ]
    chat.sync([*chat.live, *unstored])
    assert count(stored(server), "USERTOKEN-lost write") == 1
    assert (chat.home / "synapse" / "store_fallback.json").exists()
    chat.turn("USERTOKEN-stored", "FINALTOKEN-stored")
    assert not (chat.home / "synapse" / "store_fallback.json").exists()


# -- privacy -----------------------------------------------------------------------------


def test_a_private_session_sends_nothing_from_the_store(chat, server, tmp_path):
    marker = tmp_path / "private" / chat.sid
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    chat.turn("USERTOKEN-secret", "FINALTOKEN-secret")
    assert server.posts() == []


def _owner() -> str | None:
    from synapse_client.harnesses.hermes import sessions as hs

    return hs.owner_subject()


def _gateway_chat(store: HermesStore, home: Path, sid: str) -> dict[str, Any]:
    """A local chat the gateway daemon runs: its row and creation receipt, as Hermes
    v0.23.1 wrote them (probe store), and the keywords its agent gets."""
    from synapse_client.harnesses.hermes import sessions as hs

    owner = hs.owner_subject()
    route = f"agent:main:local:dm:{sid}"
    store.conn.execute(
        "INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, started_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (sid, "local", owner, route, sid, "dm", store.tick()),
    )
    receipt = {
        "profile_id": str(home),
        "principal_id": owner,
        "request_id": "req-1",
        "session_id": sid,
        "route": route,
        "policy": {
            "source": "cli",
            "platform": "cli",
            "cwd": "/srv/hermes-daemon",
            "request_json": json.dumps({"cwd": "/home/u/work", "source": "cli"}),
        },
        "entry": {"origin": {"chat_id": sid}},
    }
    store.conn.execute(
        "INSERT INTO state_meta VALUES (?,?)",
        ("gateway.local_policy.v1:" + sid, json.dumps(receipt)),
    )
    store.conn.commit()
    return {
        "platform": "cli",
        "agent_context": "primary",
        "user_id": owner,
        "chat_id": sid,
        "chat_type": "dm",
        "gateway_session_key": route,
    }


@pytest.mark.skipif(_owner() is None, reason="this machine's owner subject cannot be read")
def test_a_gateway_turn_admitted_from_someone_else_turns_the_session_off(plugin, server, tmp_path):
    """The store says who started each turn the gateway ran. The owner's turns and the
    session's own automation (a report) go up; a turn admitted from anyone else turns
    the provider off for good, though ``sync_turn`` named the owner as its author."""
    home = tmp_path / "hermes-a"
    store = HermesStore(home)
    sid = "local-" + "ab" * 32
    keywords = _gateway_chat(store, home, sid)
    provider = _new(plugin)
    provider.initialize(sid, hermes_home=str(home), **keywords)
    owner = keywords["user_id"]
    store.admit(sid, owner)
    live = store.turn(sid, "USERTOKEN-mine", "FINALTOKEN-mine")
    provider.sync_turn("q", "a", session_id=sid, messages=live, turn_author={"id": owner})
    store.admit(sid, "automation:" + keywords["gateway_session_key"])
    live += store.report(sid, "SUBAGENT-OUTPUT", "FINALTOKEN-report")
    provider.sync_turn("q", "a", session_id=sid, messages=live, turn_author={"id": owner})
    assert len(server.posts()) == 2
    store.admit(sid, "api-peer")
    live += store.turn(sid, "USERTOKEN-someone else", "FINALTOKEN-theirs")
    provider.sync_turn("q", "a", session_id=sid, messages=live, turn_author={"id": owner})
    assert len(server.posts()) == 2
    assert provider._mode == "inert"
    assert count(stored(server), "USERTOKEN-someone else") == 0
