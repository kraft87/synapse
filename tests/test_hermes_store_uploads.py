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


def stored(server: Any, refused: Any = ()) -> list[Any]:
    """The episodes the ingest route would keep from every body posted, in order, but
    the posts it answered with an error (``refused``: their indexes)."""
    kept: list[Any] = []
    spans: set[str] = set()
    contents: set[tuple[Any, str]] = set()
    for i, body in enumerate(server.posts()):
        if i in refused:
            continue
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
    # Hermes's own scrub stands in as the identity (test_hermes_scrub.py runs the real one).
    module.scrub.interpreter = lambda: "hermes-python"
    module.scrub.rows = lambda sessions, python, home=None: sessions
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
    posted = len(server.posts())
    assert _import(chat.home) == 0  # the store's two rows are the same turn: nothing added
    assert len(server.posts()) == posted
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
    refused = set(range(len(server.posts())))
    server.ingest_status = 200
    chat.turn("USERTOKEN-two", "FINALTOKEN-two")
    episodes = stored(server, refused)
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


def test_a_store_of_an_unknown_schema_costs_only_the_kinds(plugin, server, tmp_path, caplog):
    """A newer Hermes: the list Hermes hands over is uploaded all the same; the kinds the
    store would add are not read, and that is logged once."""
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
    said = [r.getMessage() for r in caplog.records if "display kinds" in r.getMessage()]
    assert len(said) == 1 and "schema version 34" in said[0]


def test_rows_hermes_could_not_write_go_up_from_the_list(chat, server):
    """review-round3 probe2 ``test_rows_hermes_could_not_write``: the final flush
    failed, so the run's rows are in the list with no ``message_uid`` (Hermes mints it
    at insert) and not in the store. They go up, keyed by their text, once."""
    chat.turn("USERTOKEN-stored", "FINALTOKEN-stored")
    unstored = [
        {"role": "user", "content": "USERTOKEN-lost write"},
        {"role": "assistant", "content": "FINALTOKEN-lost write"},
    ]
    chat.sync([*chat.live, *unstored])
    more = [
        {"role": "user", "content": "USERTOKEN-lost again"},
        {"role": "assistant", "content": "FINALTOKEN-again"},
    ]
    chat.sync([*chat.live, *unstored, *more])
    chat.sync([*chat.live, *unstored, *more])
    episodes = stored(server)
    assert (
        count(episodes, "USERTOKEN-lost write") == 1
        and count(episodes, "USERTOKEN-lost again") == 1
    )
    assert len(server.posts()) == 3


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


# -- review round 3 -------------------------------------------------------------------------


def _queued(chat: Session, *, read_after_drain: bool, compacted: bool = False) -> None:
    """A prompt queued in the TUI while a tool-calling run is going, as the stdio TUI of
    Hermes v0.23.1 did it (probe session ``20261115_050146_dfd17b``, rows 114-120): the
    prompt is written as it is accepted (``display_metadata._queued_prompt``), before the
    run's assistant rows; the list of that run never holds it; when its own run starts the
    early row is deactivated and the prompt written again, with a new uid. At the first
    ``sync_turn`` the early row was still active."""
    sid, s = chat.sid, chat.store
    chat.turn("USERTOKEN-zero", "FINALTOKEN-zero")
    one = call()
    run_a = [s.add(sid, "user", "USERTOKEN-A first request")]
    early = s.add(
        sid, "user", "USERTOKEN-B queued follow-up", display_metadata={"_queued_prompt": True}
    )
    if compacted:  # a compaction in place during the run archives the early row and clones it
        s.conn.execute(
            "UPDATE messages SET active=0, compacted=1 WHERE message_uid=?", (early["message_uid"],)
        )
        s.add(
            sid,
            "user",
            "USERTOKEN-B queued follow-up",
            uid=early["message_uid"],
            display_metadata={"_queued_prompt": True},
        )
    run_a += [
        s.add(sid, "assistant", "", tool_calls=[one]),
        s.add(
            sid,
            "tool",
            "RESULT-A tool output of the first request",
            tool_name="terminal",
            tool_call_id=one["id"],
        ),
        s.add(sid, "assistant", "FINALTOKEN-A answer to the first request"),
    ]
    list_a = [*chat.live, *run_a]

    def drain() -> dict[str, Any]:
        s.conn.execute(
            "UPDATE messages SET active=0 WHERE message_uid=? AND active=1", (early["message_uid"],)
        )
        s.conn.commit()
        return s.add(sid, "user", "USERTOKEN-B queued follow-up")

    if read_after_drain:
        late = drain()
        chat.sync(list_a)
    else:
        chat.sync(list_a)
        late = drain()
    chat.sync([*list_a, late, s.add(sid, "assistant", "FINALTOKEN-B answer to the follow-up")])


@pytest.mark.parametrize(
    ("after", "compacted"),
    [(False, False), (True, False), (True, True)],
    ids=["read before the drain", "read after the drain", "compacted during the run"],
)
def test_a_prompt_queued_during_a_run_keeps_its_place(chat, server, after, compacted):
    """review-round3 finding 3 (probe1 ``test_queued_prompt``): the store branch read the
    accept-time row inside run A, so A's request and tool work were never stored and B
    twice. The rows now come from the list, which is the run as Hermes ran it; the
    importer, which reads the store, leaves the accept-time row out."""
    _queued(chat, read_after_drain=after, compacted=compacted)
    episodes = stored(server)
    for token in ("USERTOKEN-A", "RESULT-A", "FINALTOKEN-A", "USERTOKEN-B", "FINALTOKEN-B"):
        assert count(episodes, token) == 1, token
    posted = len(server.posts())
    assert _import(chat.home) == 0
    assert len(server.posts()) == posted


def test_the_importer_reads_a_queued_prompt_where_it_was_answered(plugin, server, tmp_path):
    """The same store, imported with no live upload before: one turn per request."""
    chat = Session(plugin, tmp_path / "hermes-a")
    chat.provider.shutdown()
    chat.provider._mode = "inert"  # nothing goes up live: the import alone reads the store
    _queued(chat, read_after_drain=True, compacted=True)
    assert server.posts() == []
    assert _import(chat.home) == 0
    episodes = stored(server)
    for token in ("USERTOKEN-A", "RESULT-A", "FINALTOKEN-A", "USERTOKEN-B", "FINALTOKEN-B"):
        assert count(episodes, token) == 1, token
    a = next(ep for ep in episodes if "FINALTOKEN-A" in ep.content)
    assert a.human_turn == "USERTOKEN-A first request"


def test_a_rotation_inside_a_run_stores_the_whole_turn(chat, server):
    """review-round3 finding 4 (probe1 ``test_rotation_mid_run``): a compaction that moves
    to a new session id inside a run (``compression.in_place`` off). The child's list
    holds the run's rows under their uids and the reply; the turn goes up whole under the
    child, as the released client sent it."""
    sid, s = chat.sid, chat.store
    chat.turn("USERTOKEN-one", "FINALTOKEN-one")
    one = call()
    run = [
        s.add(sid, "user", "USERTOKEN-two long task"),
        s.add(sid, "assistant", "", tool_calls=[one]),
        s.add(
            sid,
            "tool",
            "RESULT-two tool output before the compaction",
            tool_name="terminal",
            tool_call_id=one["id"],
        ),
    ]
    child = "20261115_111111_cc22dd"
    s.session(child, parent=sid, model_config={"_compressed_from": sid})
    summary = s.add(
        child, "user", "[CONTEXT COMPACTION] summary", summary=True, display_kind="hidden"
    )
    clones = [
        s.add(
            child,
            r["role"],
            r["content"],
            uid=r["message_uid"],
            tool_calls=r.get("tool_calls"),
            tool_name=r.get("tool_name"),
            tool_call_id=r.get("tool_call_id"),
        )
        for r in run
    ]
    chat.provider.on_session_switch(child, parent_session_id=sid, reason="compression")
    final = s.add(child, "assistant", "FINALTOKEN-two done")
    chat.provider.sync_turn("q", "a", session_id=child, messages=[summary, *clones, final])
    more = s.turn(child, "USERTOKEN-three", "FINALTOKEN-three")
    chat.provider.sync_turn("q", "a", session_id=child, messages=[summary, *clones, final, *more])
    episodes = stored(server)
    for token in (
        "USERTOKEN-one",
        "USERTOKEN-two",
        "RESULT-two",
        "FINALTOKEN-two",
        "USERTOKEN-three",
    ):
        assert count(episodes, token) == 1, token


def test_turns_the_server_dropped_as_private_go_again_after_forget(chat, server, tmp_path):
    """review-round3 finding 5 (probe1 ``test_private_off_then_forget``): the marker went
    (plain ``off``, or its twelve hours) while the server row stayed, so a turn went and
    the server dropped it with a 200; it was recorded as shipped. ``synapse private off
    --forget`` clears that record (private_toggle.py, shipped.forget_everywhere), so after
    a restart the look-back sends it, as the released client did."""
    from synapse_client.settings import Settings
    from synapse_client.shipped import forget_everywhere

    marker = tmp_path / "private" / chat.sid
    chat.turn("USERTOKEN-1", "FINALTOKEN-1")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    server.private_rows.add(chat.sid)
    chat.turn("USERTOKEN-2 said in private", "FINALTOKEN-2")
    before = len(server.posts())
    marker.unlink()
    chat.turn("USERTOKEN-3 after the marker went", "FINALTOKEN-3")
    dropped = set(range(before, len(server.posts())))  # the server's row drops these
    server.private_rows.discard(chat.sid)
    forget_everywhere(Settings.load(), chat.sid)  # what `off --forget` does on this machine
    chat.turn("USERTOKEN-4 back on the record", "FINALTOKEN-4")
    rebuilt = chat.agent()  # a restart
    chat.live += chat.store.turn(chat.sid, "USERTOKEN-5", "FINALTOKEN-5")
    chat.sync(provider=rebuilt)
    kept = [b for i, b in enumerate(server.posts()) if i not in dropped]
    episodes = stored(type("S", (), {"posts": lambda self: kept})())
    for n in range(1, 6):
        assert count(episodes, f"USERTOKEN-{n}") == 1, n


@pytest.mark.parametrize("reason", ["resume", "branch"])
def test_a_recorded_copy_left_by_a_switch_posts_nothing(plugin, server, tmp_path, reason):
    """review-round3 finding 2 (probe3): a home a client 0.24.10 to 0.24.15 recorded as
    holding copies, a new agent for that session, then a switch away before any turn
    ended there. Nothing is posted for the session left (no upload is queued for it)."""
    home = tmp_path / "hermes-a"
    chat = Session(plugin, home)
    copied = chat.store.turn(chat.sid, "USERTOKEN-copied one", "FINALTOKEN-copied one")
    copied += chat.store.turn(chat.sid, "USERTOKEN-copied two", "FINALTOKEN-copied two")
    (home / "synapse" / "copied").mkdir(parents=True, exist_ok=True)
    from synapse_client.harnesses.hermes import carried

    carried.note_copied(home, chat.sid)
    agent = chat.agent()
    agent.on_session_switch("20261115_222222_ff00ff", parent_session_id=chat.sid, reason=reason)
    agent.shutdown()
    assert server.posts() == []


def test_a_part_waiting_in_the_outbox_is_not_sent_again_as_context(chat, server):
    """review-round3 probe2 ``test_part_in_the_outbox_then_the_whole_turn_as_context``: a
    flush inside a run sent the first part; the rest failed and waits in the outbox; the
    next turn led with the whole turn as context, which the server stored whole. A turn
    that went in parts is not used as context."""
    sid, s = chat.sid, chat.store
    chat.turn("USERTOKEN-1", "FINALTOKEN-1")
    chat.live += [
        s.add(sid, "user", "USERTOKEN-2 long plan"),
        s.add(sid, "assistant", "INTERIM-2 I will start"),
        s.add(sid, "user", "[System: Continue now.]"),
    ]
    chat.provider.on_pre_compress(list(chat.live))
    server.ingest_status = 503
    chat.live.append(s.add(sid, "assistant", "FINALTOKEN-2 done"))
    n = len(server.posts())
    chat.sync()
    refused = set(range(n, len(server.posts())))
    server.ingest_status = 200
    chat.turn("USERTOKEN-3", "FINALTOKEN-3")
    episodes = stored(server, refused)
    for token in ("USERTOKEN-2", "INTERIM-2", "FINALTOKEN-2", "USERTOKEN-3"):
        assert count(episodes, token) == 1, token


def test_a_list_row_the_store_holds_otherwise_goes_as_the_list_has_it(chat, server, caplog):
    """The store is a check too: a list row whose uid the store holds under another
    role is said once in the log and sent as the list has it (the released client's
    behaviour); a row the store does not hold is no mismatch."""
    sid, s = chat.sid, chat.store
    row = s.add(sid, "assistant", "stored as the assistant's")
    odd = {**row, "role": "user", "content": "USERTOKEN-as the list has it"}
    reply = s.add(sid, "assistant", "FINALTOKEN-odd")
    with caplog.at_level(logging.WARNING):
        chat.sync([*chat.live, odd, reply])
        chat.sync([*chat.live, odd, reply])
    assert count(stored(server), "USERTOKEN-as the list has it") == 1
    said = [r.getMessage() for r in caplog.records if "disagree" in r.getMessage()]
    assert len(said) == 1
