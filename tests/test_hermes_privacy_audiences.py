"""The audiences a privacy review found the Hermes provider serving, and must not.

Each shape below is built from what Hermes writes (checkout bca51750), cited per test:
a Desktop Group Chat member, a Bot Chat the Desktop minted and then reset, a cron job,
a turn another agent wrote, a gateway session whose row is gone, a side agent the
gateway daemon builds, a process started with ``--source``, a session renamed after
its system prompt was built, and a chat others wrote into after its title is gone, its
agent is rebuilt or it is branched. ``tests/test_hermes_gateway_sessions.py`` has the store
fixture (the tables and receipt ``commit_local_session`` writes) and the owner's shapes.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_gateway_sessions as gs

hs = gs.hs
hermes_stub = gs.hermes_stub
user = gs.user
plugin = gs.plugin
server = gs.server
_new = gs._new
_conversation = gs._conversation
_store = gs._store
_gateway = gs._gateway
_in_process = gs._in_process
_start_gateway = gs._start_gateway
_LOCAL = gs._LOCAL
_WORK = gs._WORK

pytestmark = gs.pytestmark

#: A Desktop request id: canonical-protocol.ts prepareCreate adds a random one.
_DESKTOP_REQUEST = "6f1c0d2e9a7b4c3d8e5f"
#: apps/desktop/src/plugins/hermes-bots/group-turns.ts:297-300 composes the title.
_GROUP_TITLE = "Group: room-7c1e · legacy"
_RESET_CHILD = "20261112_170501_9b8c7d"
_SIDE_TASK = "bg_1a2b3c"  # gateway/session_tui_side.py ``_admit``: f"{prefix}_{hex[:6]}"


@pytest.fixture(autouse=True)
def _process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No ``--source`` label and a Hermes home of the test's own, so the daemon check
    never reads a real home."""
    monkeypatch.delenv(hs.SOURCE_ENV, raising=False)
    monkeypatch.delenv(hs.AUTHOR_ENV, raising=False)
    home = tmp_path / "process-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _row(
    home: Path,
    sid: str,
    chat: str,
    *,
    title: str | None = None,
    model_config: dict[str, Any] | None = None,
    source: str = "gui",
) -> None:
    """One more ``sessions`` row, as hermes_state_compression.py
    ``_publish_child_session_row`` (:351-366) writes a reset or compressed segment: the
    parent's chat_id, user_id and session_key, no title column."""
    if not (home / "state.db").exists():
        _store(home, "20261101_000000_000000", receipt=False, chat_id="")  # the tables
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, source, created_source, user_id, chat_id, "
            "chat_type, session_key, title, model_config) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sid,
                source,
                source,
                f"uid:{os.getuid()}",
                chat,
                "dm" if chat else None,
                f"agent:main:local:dm:{chat}" if chat else None,
                title,
                json.dumps(model_config) if model_config is not None else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _policy(home: Path, chat: str, **fields: Any) -> None:
    """Set fields of the receipt's frozen policy (``config_json`` and the like)."""
    conn = sqlite3.connect(home / "state.db")
    try:
        key = "gateway.local_policy.v1:" + chat
        (value,) = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        receipt = json.loads(value)
        receipt["policy"].update(fields)
        conn.execute("UPDATE state_meta SET value=? WHERE key=?", (json.dumps(receipt), key))
        conn.commit()
    finally:
        conn.close()


def _drop_row(home: Path, sid: str) -> None:
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        conn.commit()
    finally:
        conn.close()


def _assert_nothing_served(provider: Any, server: Any) -> None:
    assert provider.system_prompt_block() == ""
    assert provider.prefetch("x") == ""
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(2))
    provider.on_pre_compress(_conversation(2))
    provider.on_session_end(_conversation(2))
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    assert server.posts() == [] and server.posts("/mcp") == []


# ---------------------------------------------------------------------------
# A Desktop Group Chat member
# ---------------------------------------------------------------------------

#: group-turns.ts:395-409 sends title, ``hidden``, ``room_plumbing`` and
#: ``follow_profile_config``; canonical-protocol.ts:533-562 turns ``room_plumbing`` into
#: ``follow_profile_config``, drops it, and adds ``request_id`` and ``source: 'gui'``;
#: gateway/session_policy.py ``build_policy`` stores the rest minus ``request_id``.
_GROUP_REQUEST = {"title": _GROUP_TITLE, "hidden": True, "follow_profile_config": True}


def test_a_group_chat_member_is_inert_by_its_receipt(tmp_path):
    _store(tmp_path, source="gui", request_id=_DESKTOP_REQUEST, request=_GROUP_REQUEST)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway(platform="desktop"))
    assert verdict.mode == hs.INERT and "follows its profile" in verdict.reason


def test_a_group_chat_member_is_inert_by_its_title(tmp_path):
    # The title alone, on the keywords or on the row (session_controls.py:543 writes it
    # after the commit), with a receipt that says nothing else.
    assert "Group Chat" in hs.from_keywords(_gateway(session_title=_GROUP_TITLE)).reason
    assert hs.from_keywords(_in_process(session_title="  group: room-1")).mode == hs.INERT
    _store(tmp_path, source="gui", request_id=_DESKTOP_REQUEST, title=_GROUP_TITLE)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway(platform="desktop"))
    assert verdict.mode == hs.INERT and "Group Chat" in verdict.reason
    assert hs.recheck(tmp_path, _LOCAL, gateway=True).mode == hs.INERT


def test_a_group_chat_member_gets_nothing(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home, source="gui", request_id=_DESKTOP_REQUEST, request=_GROUP_REQUEST)
    provider = _start_gateway(_new(plugin), home, platform="desktop")
    _assert_nothing_served(provider, server)


def test_a_title_that_only_mentions_a_group_is_the_owners(tmp_path):
    _store(tmp_path, title="Notes on the Group: meeting")
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE


# ---------------------------------------------------------------------------
# The Bot Chat the Desktop mints, after a reset
# ---------------------------------------------------------------------------


def _client_bot_chat(home: Path, request: dict[str, Any]) -> None:
    """canonical-chat.ts:538-557 mints the Bot Chat itself: a random request id (no
    ``bot-chat:`` prefix), source gui, title "Bot Chat". ``/new`` (or the compression
    auto-reset) then publishes an untitled child on the same chat
    (hermes_state_local_lineage.py:173-199, ``model_config={'_reset_from': parent}``),
    and delivery resolves "Bot Chat" to it (gateway/session_bot.py:184-212)."""
    _store(home, source="gui", request_id=_DESKTOP_REQUEST, title="Bot Chat", request=request)
    _row(home, _RESET_CHILD, _LOCAL, model_config={"_reset_from": _LOCAL})


def test_a_reset_bot_chat_is_inert_by_its_receipt(tmp_path):
    _client_bot_chat(tmp_path, {"title": "Bot Chat", "hidden": True, "follow_profile_config": True})
    # No title on any row: session_controls.py:542-543 writes it after the commit, and a
    # failed title write is swallowed (agent/conversation_compression.py:3201-3205).
    gs._retitle(tmp_path, _LOCAL, None)
    verdict = hs.check_session(tmp_path, _RESET_CHILD, _gateway(platform="desktop"))
    assert verdict.mode == hs.INERT and "follows its profile" in verdict.reason


def test_a_reset_bot_chat_is_inert_by_its_parents_title(tmp_path):
    # A receipt without the flag: an older Desktop, or a create an older build staged
    # (gateway/session_local.py ``staged_creation_follows``). The untitled child is held
    # to the title of the chat's other row.
    _client_bot_chat(tmp_path, {"title": "Bot Chat", "hidden": True})
    verdict = hs.check_session(tmp_path, _RESET_CHILD, _gateway(platform="desktop"))
    assert verdict.mode == hs.INERT and "Bot Chat" in verdict.reason
    assert hs.recheck(tmp_path, _RESET_CHILD, gateway=True).mode == hs.INERT
    assert hs.recheck(tmp_path, _RESET_CHILD).mode == hs.INERT


def test_a_reset_bot_chat_gets_nothing(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _client_bot_chat(home, {"title": "Bot Chat", "hidden": True})
    provider = _new(plugin)
    provider.initialize(_RESET_CHILD, hermes_home=str(home), **_gateway(platform="desktop"))
    _assert_nothing_served(provider, server)


def test_an_owner_chat_continued_in_an_untitled_row_stays_active(tmp_path):
    _store(tmp_path, title="Cache design")
    _row(tmp_path, _RESET_CHILD, _LOCAL, model_config={"_reset_from": _LOCAL}, source="oneshot")
    assert hs.check_session(tmp_path, _RESET_CHILD, _gateway()) == hs.Verdict(hs.ACTIVE, "", _WORK)


# ---------------------------------------------------------------------------
# The follow-profile markers
# ---------------------------------------------------------------------------


def test_follow_profile_config_in_the_request_is_inert(tmp_path):
    # gateway/session_controls.py:508-514 sets it on a titled "Bot Chat" create;
    # session_bot.py:369-374 on the gateway's own Bot Chat. Neither is an owner's chat.
    for flag in ("follow_profile_config", "room_plumbing"):
        home = tmp_path / flag  # a home each: the first verdict marks the chat for good
        _store(home, request={"cwd": _WORK, flag: True})
        assert hs.check_session(home, _LOCAL, _gateway()).mode == hs.INERT
    _store(tmp_path, request={"cwd": _WORK, "follow_profile_config": False})
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE


def test_follow_adopted_alone_is_not_read(tmp_path):
    """gateway/session_local.py:314-321 (``stage_local_session``) marks
    ``follow_adopted`` on an ordinary chat whose model is the welcome trial's
    (hermes_cli/anon_auth.py ``GUEST_MODEL``): the owner's own chat, so the mark alone
    must not turn it off. The Bot Chats ``follow_bot_chat`` marks
    (session_local_migration.py:163-198) are caught by their titles."""
    _store(tmp_path, source="cli")
    _policy(
        tmp_path, _LOCAL, model="nous/welcome", config_json=json.dumps({"follow_adopted": True})
    )
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE


def test_an_unreadable_request_is_inert(tmp_path):
    _store(tmp_path)
    _policy(tmp_path, _LOCAL, request_json="{not json")
    assert "could not be read" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason


@pytest.mark.parametrize(
    "model_config",
    [{"follow_profile_config": True}, {"room_plumbing": True, "follow_profile_config": True}],
)
def test_an_in_process_row_that_follows_its_profile_is_inert(tmp_path, model_config):
    """tui_gateway/session_workdir.py:576-580 stores both flags in the row's
    ``model_config``; tui_gateway/agent_factory.py:112-119 reads the first as the Bot
    Chat."""
    sid = "20261112_171000_ab12cd"
    _row(
        tmp_path, sid, "", model_config={"model": "x", "follow_profile_config": False}, source="tui"
    )
    assert hs.check_session(tmp_path, sid, _in_process()).mode == hs.ACTIVE
    _row(tmp_path, sid, "", model_config=None, source="tui")
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("UPDATE sessions SET model_config='{broken' WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    assert hs.check_session(tmp_path, sid, _in_process()).mode == hs.ACTIVE
    _row(tmp_path, sid, "", title=None, model_config=model_config, source="tui")
    verdict = hs.check_session(tmp_path, sid, _in_process())
    assert verdict.mode == hs.INERT and "follows its profile" in verdict.reason


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------


def test_a_cron_job_gets_nothing_not_even_the_tools(plugin, server, tmp_path):
    """cron/scheduler.py:2478-2512 builds the job's agent with ``platform="cron"`` and
    ``skip_memory=False``; agent/agent_init_memory_provider.py:114-115 makes that the
    ``agent_context``."""
    provider = _new(plugin)
    provider.initialize(
        "cron_job1_20261112",
        hermes_home=str(tmp_path / "hermes-a"),
        platform="cron",
        agent_context="cron",
    )
    _assert_nothing_served(provider, server)


# ---------------------------------------------------------------------------
# A turn another agent wrote
# ---------------------------------------------------------------------------

#: agent/turn_author.py ``parse_turn_author``: ``{"id", "name", "is_bot"}``; a bot on
#: this install is ``bot:<profile>`` (``bot_author_id``); tools/bot_relay.py
#: ``delivery_turn_author`` and ``relaying_principal_author`` always mark a relay a bot.
_BOT = {"id": "bot:research", "name": "research", "is_bot": True}


def _owner_author() -> dict[str, Any]:
    """gateway/run_turn_runner.py:1059-1064: "Sent on every transport"; with no admission
    author the gateway names ``ctx.source``, the source of the ``user_id`` keyword too
    (gateway/session_execution.py:174)."""
    return {"id": gs._owner(), "name": None, "is_bot": False}


def test_the_owners_own_turn_author_is_uploaded(plugin, server, tmp_path):
    """The owner's turn in the daemon carries the owner as its author, on
    ``on_turn_start`` (agent/turn_context.py:1061-1080) and ``sync_turn``
    (run_agent.py:919-925)."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    author = _owner_author()
    provider.on_turn_start(
        1, "q", model="m", author_id=author["id"], author_name=None, author_is_bot=False
    )
    assert "Call remember FIRST" in provider.prefetch("x")
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1), turn_author=author)
    assert len(server.posts()) == 1
    assert "BOARD-TEXT" in provider.system_prompt_block()


@pytest.mark.parametrize(
    "author",
    [
        _BOT,
        {"id": "bot:relay/1f2e3d", "name": "relayed teammate", "is_bot": True},
        {"id": "uid:0", "name": "root", "is_bot": False},
        {"id": None, "name": "someone", "is_bot": False},
        "not a mapping",
    ],
)
def test_a_turn_author_not_the_owner_is_never_uploaded(plugin, server, tmp_path, caplog, author):
    """tui_gateway/methods_bot_relay.py:88-96: "an unattributed turn is the HUMAN's to the
    recipient's memory, so a bot DM must stay bot-authored"."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1), turn_author=None)
    assert len(server.posts()) == 1
    with caplog.at_level(logging.INFO):
        provider.sync_turn(
            "q", "a", session_id=_LOCAL, messages=_conversation(2), turn_author=author
        )
        provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(3))
        provider.on_session_end(_conversation(3))
    assert len(server.posts()) == 1
    assert provider.prefetch("x") == "" and provider.system_prompt_block() == ""
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    said = [r.getMessage() for r in caplog.records if "off for session" in r.getMessage()]
    assert said == [f"synapse: off for session {_LOCAL}: a turn another agent or person wrote"]


def test_a_bot_named_as_the_owner_is_still_a_bot(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.sync_turn(
        "q",
        "a",
        session_id=_LOCAL,
        messages=_conversation(1),
        turn_author={**_owner_author(), "is_bot": True},
    )
    assert server.posts() == [] and provider.prefetch("x") == ""


def test_an_authored_turn_is_off_before_its_nudge(plugin, server, tmp_path):
    """agent/turn_context.py:1061-1080 calls ``on_turn_start`` with the author trio, then
    ``prefetch``, before the model sees the turn."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.on_turn_start(
        2, "hi", model="m", author_id=None, author_name=None, author_is_bot=False
    )
    assert provider.prefetch("x")
    provider.on_turn_start(
        3, "hi", model="m", author_id="bot:research", author_name="research", author_is_bot=True
    )
    assert provider.prefetch("x") == ""
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(3))
    assert server.posts() == []


@pytest.mark.parametrize("author", [_BOT, {"id": "1", "name": "kyle", "is_bot": False}])
def test_any_author_in_process_is_someone_else(plugin, server, tmp_path, author):
    """In-process a human turn has no author (agent/turn_context.py:1216-1217 resets it
    each turn; a client's own ``_turn_author`` is refused, tui_gateway/contracts/
    prompt_voice.py:49-51): the only authors are relayed bots and dispatched deliveries
    (``HERMES_TURN_AUTHOR``, tools/bot_relay.py ``delivery_env``)."""
    provider = _new(plugin)
    provider.initialize(_RESET_CHILD, hermes_home=str(tmp_path / "hermes-a"), **_in_process())
    provider.sync_turn("q", "a", session_id=_RESET_CHILD, messages=_conversation(1))
    assert len(server.posts()) == 1
    provider.sync_turn(
        "q", "a", session_id=_RESET_CHILD, messages=_conversation(2), turn_author=author
    )
    assert len(server.posts()) == 1 and provider.prefetch("x") == ""


def test_a_bot_with_no_id_or_name_is_still_a_bot(plugin, server, tmp_path):
    """gateway/run_turn_runner.py builds the author from ``ctx.source``, with None for
    an empty id or name: the bot flag alone names someone else."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.on_turn_start(2, "hi", model="m", author_id=None, author_name=None, author_is_bot=True)
    assert provider.prefetch("x") == ""
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert server.posts() == []


def test_a_process_that_carries_a_turn_author_is_inert(plugin, server, tmp_path, monkeypatch):
    """A backstop. tools/bot_relay.py ``delivery_env`` sets HERMES_TURN_AUTHOR on the
    child that runs a delivered message. No Hermes path builds an agent in such a child
    today: ``hermes chat`` and the one-shots run their turn in the daemon
    (hermes_cli/cli_agent_setup_mixin.py), whose environment has the variable stripped.
    A process that did build one would get nothing, from before its first prompt."""
    monkeypatch.setenv(hs.AUTHOR_ENV, json.dumps(_BOT))
    verdict = hs.from_keywords(_in_process(platform="cli"))
    assert verdict.mode == hs.INERT and "another agent" in verdict.reason
    provider = _new(plugin)
    provider.initialize(_RESET_CHILD, hermes_home=str(tmp_path / "hermes-a"), **_in_process())
    assert provider.system_prompt_block() == "" and provider.prefetch("x") == ""
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    assert server.seen == []
    # The daemon strips it from its own environment (gateway_runtime_start.py:41); a
    # gateway session is judged by its receipt and each turn's author.
    _store(tmp_path / "hermes-b")
    assert hs.check_session(tmp_path / "hermes-b", _LOCAL, _gateway()).mode == hs.ACTIVE
    assert hs.from_keywords(_gateway()).mode == hs.ACTIVE


# ---------------------------------------------------------------------------
# A gateway session whose row is gone
# ---------------------------------------------------------------------------


def test_a_gateway_session_without_its_row_is_inert(plugin, server, tmp_path):
    """hermes_state_local.py:69-77 (``commit_local_session``) writes the row with the
    receipt, so a gateway session the store has no row for is not one it vouches for."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    assert provider.prefetch("x")
    _drop_row(home, _LOCAL)
    assert provider.prefetch("x") == ""
    _store(home)  # back again: off for good all the same
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert server.posts() == []
    _drop_row(home, _LOCAL)
    verdict = hs.check_session(home, _LOCAL, _gateway())
    assert verdict.mode == hs.INERT and "no longer has" in verdict.reason
    assert hs.recheck(home, _LOCAL, gateway=True).mode == hs.INERT
    assert hs.recheck(None, _LOCAL, gateway=True).mode == hs.INERT
    assert hs.recheck(tmp_path / "nowhere", _LOCAL, gateway=True).mode == hs.INERT
    assert hs.recheck(home, _LOCAL).mode == hs.ACTIVE  # in-process: not written yet


# ---------------------------------------------------------------------------
# A side agent of the gateway daemon
# ---------------------------------------------------------------------------


def _daemon_record(home: Path, name: str = "gateway.lock", **fields: Any) -> None:
    """gateway/status.py:789-797 ``_build_pid_record``, as the rig's gateway.lock holds it."""
    record = {
        "pid": os.getpid(),
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--quiet"],
        "start_time": 53174688,
        "hermes_home": str(home),
        "pidns": "4026531836",
    }
    record.update(fields)
    home.mkdir(parents=True, exist_ok=True)
    (home / name).write_text(json.dumps(record), encoding="utf-8")


def _side_agent() -> dict[str, Any]:
    """gateway/session_tui_side.py:89-116: ``session_id=task_id``,
    ``platform=policy.platform``, no identity and no parent; the keywords
    agent_init_memory_provider.py:105-146 builds from that."""
    return {
        "platform": "cli",
        "agent_context": "primary",
        "agent_identity": "default",
        "agent_workspace": "hermes",
    }


@pytest.mark.parametrize("name", ["gateway.lock", "gateway.pid"])
def test_a_side_agent_in_the_daemon_is_inert(tmp_path, name):
    home = tmp_path / "hermes-a"
    assert hs.check_session(home, _SIDE_TASK, _side_agent()).mode == hs.ACTIVE
    _daemon_record(home, name)
    verdict = hs.check_session(home, _SIDE_TASK, _side_agent())
    assert verdict.mode == hs.INERT and "gateway daemon" in verdict.reason
    # The daemon's own sessions carry their chat and are judged by the full rule.
    _store(home)
    assert hs.check_session(home, _LOCAL, _gateway()).mode == hs.ACTIVE


def test_the_daemon_is_found_by_the_process_home_too(tmp_path, _process_env):
    """A side agent's ``owner_scope`` may bind a served profile's home; the record sits
    in the process's own (``HERMES_HOME``, gateway/status.py ``_get_pid_path``)."""
    _daemon_record(_process_env)
    assert hs.gateway_daemon(tmp_path / "served-profile")
    assert hs.check_session(tmp_path / "served-profile", _SIDE_TASK, _side_agent()).mode == hs.INERT
    assert hs.check_session(None, _SIDE_TASK, _side_agent()).mode == hs.INERT


@pytest.mark.parametrize(
    "fields",
    [{"pid": 1}, {"kind": "hermes-dashboard"}, {"pid": str(os.getpid())}],
)
def test_another_process_record_is_not_the_daemon(tmp_path, fields):
    # A stale record, another process, or a record that is not the gateway's.
    _daemon_record(tmp_path, **fields)
    assert not hs.gateway_daemon(tmp_path)
    assert hs.check_session(tmp_path, _SIDE_TASK, _side_agent()).mode == hs.ACTIVE


def test_a_record_too_large_or_broken_is_not_the_daemon(tmp_path):
    _daemon_record(tmp_path, argv=["x" * hs.RECORD_LIMIT])
    assert not hs.gateway_daemon(tmp_path)
    (tmp_path / "gateway.lock").write_bytes(b"\xff\xfe not json")
    assert not hs.gateway_daemon(tmp_path)


def test_a_side_agent_in_the_daemon_gets_nothing(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _daemon_record(home)
    provider = _new(plugin)
    provider.initialize(_SIDE_TASK, hermes_home=str(home), **_side_agent())
    _assert_nothing_served(provider, server)


# ---------------------------------------------------------------------------
# A process started with --source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["tool", "kanban", "a2a", "webhook"])
def test_an_in_process_session_of_another_source_is_inert(tmp_path, monkeypatch, source):
    """hermes_cli/main.py:1636-1640 exports ``--source`` as HERMES_SESSION_SOURCE;
    hermes_cli/kanban_db_dispatch.py:2796 sets "kanban" for a worker. The row says so
    only after the first write; the process says so at once."""
    monkeypatch.setenv(hs.SOURCE_ENV, source)
    verdict = hs.from_keywords(_in_process(platform="cli"))
    assert verdict.mode == hs.INERT and repr(source) in verdict.reason
    # The daemon strips it from its own environment (gateway_runtime_start.py:37); a
    # gateway session is judged by its receipt.
    _store(tmp_path)
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE


@pytest.mark.parametrize("source", ["cli", "tui", "desktop", "acp", "oneshot", " TUI "])
def test_an_in_process_session_of_an_owner_source_stays_active(monkeypatch, source):
    monkeypatch.setenv(hs.SOURCE_ENV, source)
    assert hs.from_keywords(_in_process()).mode == hs.ACTIVE


# ---------------------------------------------------------------------------
# The system prompt block after a rename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", ["Bot Chat", _GROUP_TITLE])
def test_the_block_is_withheld_once_the_session_is_renamed(plugin, server, tmp_path, title):
    """conversation_loop.py:802-868 builds the prompt only when no stored one matches;
    a rebuild after a rename asks for the block again, and gets nothing."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    assert "BOARD-TEXT" in provider.system_prompt_block()
    gs._retitle(home, _LOCAL, title)
    assert provider.system_prompt_block() == ""
    gs._retitle(home, _LOCAL, "Cache design")
    assert provider.system_prompt_block() == ""  # off for good


# ---------------------------------------------------------------------------
# A chat others wrote into stays off: its branches, and the marks
# ---------------------------------------------------------------------------

_BRANCH = gs._OTHER
_FOREIGN_TURN = {"author_id": "bot:research", "author_name": "research", "author_is_bot": True}


def _branch(
    home: Path, parent: str = _LOCAL, child: str = _BRANCH, *, title: str = "Research notes"
) -> None:
    """hermes_state_mutation_branch.py ``branch_in_transaction``: a new chat whose id is
    its session's, the parent's policy under a ``branch:`` request id, a copy of the
    parent's turns, ``parent_session_id``, and a title of its own."""
    _store(home, child, source="gui", request_id="branch:" + "4d" * 16, title=title)
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute(
            "UPDATE sessions SET parent_session_id=?, model_config=? WHERE id=?",
            (parent, json.dumps({"_branched_from": parent}), child),
        )
        conn.commit()
    finally:
        conn.close()


def _old_bot_chat(home: Path, **fields: Any) -> None:
    """A Bot Chat from before Hermes flagged them: its title is all that says so."""
    _store(home, **{"source": "gui", "request_id": _DESKTOP_REQUEST, "title": "Bot Chat", **fields})


def test_a_branch_of_a_bot_chat_under_its_own_title_is_inert(tmp_path):
    _old_bot_chat(tmp_path)
    _branch(tmp_path)
    kwargs = _gateway(_BRANCH, platform="desktop")
    verdict = hs.check_session(tmp_path, _BRANCH, kwargs)
    assert verdict.mode == hs.INERT and "Bot Chat" in verdict.reason
    assert hs.recheck(tmp_path, _BRANCH, gateway=True).mode == hs.INERT
    owner = tmp_path / "owner"
    _store(owner, title="Cache design")
    _branch(owner)
    assert hs.check_session(owner, _BRANCH, kwargs).mode == hs.ACTIVE


def test_a_branch_is_held_to_every_row_of_its_parents_chat(tmp_path):
    # Branched from the untitled row a reset left; the chat's first row has the title.
    _client_bot_chat(tmp_path, {"title": "Bot Chat", "hidden": True})
    _branch(tmp_path, parent=_RESET_CHILD)
    verdict = hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH))
    assert verdict.mode == hs.INERT and "Bot Chat" in verdict.reason


def test_a_branch_is_held_to_its_parents_receipt(tmp_path):
    # The gateway's own Bot Chat, archived: no title left, the ``bot-chat:`` id still is.
    _store(tmp_path, request_id="bot-chat:/home/kyle/.hermes")
    _branch(tmp_path)
    verdict = hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH))
    assert verdict.mode == hs.INERT and "Bot Chat" in verdict.reason
    # A parent's receipt is read for that alone: one that is merely missing says nothing.
    owner = tmp_path / "owner"
    _store(owner, receipt=False)
    _branch(owner)
    assert hs.check_session(owner, _BRANCH, _gateway(_BRANCH)).mode == hs.ACTIVE


def test_a_bot_chat_that_loses_its_title_stays_off(tmp_path):
    """Archive and rename take the title away (hermes_state_titles.py); the peers' turns
    stay in the chat, and the lookback would upload them as the owner's."""
    _old_bot_chat(tmp_path)
    assert "Bot Chat" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason
    gs._retitle(tmp_path, _LOCAL, "Old notes")
    for verdict in (
        hs.check_session(tmp_path, _LOCAL, _gateway()),
        hs.check_session(tmp_path, _LOCAL, _in_process()),
        hs.recheck(tmp_path, _LOCAL, gateway=True),
        hs.recheck(tmp_path, _LOCAL),
    ):
        assert verdict.mode == hs.INERT and "marked" in verdict.reason
    # So does a continuation of the chat, and a branch made after the rename.
    _row(tmp_path, _RESET_CHILD, _LOCAL, model_config={"_reset_from": _LOCAL})
    assert hs.check_session(tmp_path, _RESET_CHILD, _gateway()).mode == hs.INERT
    _branch(tmp_path)
    assert hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)).mode == hs.INERT


def test_a_chat_never_seen_as_a_bot_chat_has_no_mark(tmp_path):
    """The limit of the marks: a title gone before the provider first judged the chat
    leaves nothing to go on."""
    _old_bot_chat(tmp_path, title="Old notes")
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE
    assert not (tmp_path / "synapse").exists()


def test_a_mark_is_an_owner_only_file_that_names_its_chat(tmp_path):
    session = "20261112_135415_30e776"
    hs.mark(tmp_path, [_LOCAL, "", None, session, _LOCAL], "why")
    folder = tmp_path / "synapse" / "inert"
    files = sorted(folder.iterdir())
    assert len(files) == 2
    for path in (folder, folder.parent):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert len(path.name) == 64 and not set(path.name) - set("0123456789abcdef")
    assert sorted(path.read_text() for path in files) == sorted(
        f"{key}\nwhy\n" for key in (_LOCAL, session)
    )
    assert hs.marked(tmp_path, [_BRANCH, _LOCAL]) and hs.marked(tmp_path, [session])
    assert not hs.marked(tmp_path, [_BRANCH, "", None])
    hs.mark(tmp_path, [_LOCAL], "said again")  # the first reason is kept
    assert sorted(path.read_text() for path in folder.iterdir())[1].endswith("why\n")


def test_marking_never_raises_and_says_when_it_could_not(tmp_path, caplog):
    hs.mark(None, [_LOCAL], "why")
    assert not hs.marked(None, [_LOCAL])
    missing = tmp_path / "no" / "such" / "home"
    with caplog.at_level(logging.WARNING):
        hs.mark(missing, [_LOCAL], "why")
    assert "could not note a chat others write into (FileNotFoundError)" in caplog.text
    assert not hs.marked(missing, [_LOCAL])  # nothing there: not marked
    (tmp_path / "synapse").write_text("a file where the folder goes")
    hs.mark(tmp_path, [_LOCAL], "why")
    with pytest.raises(NotADirectoryError):  # not "nothing there": the marks are broken
        hs.marked(tmp_path, [_LOCAL])


def test_a_marks_folder_made_between_two_looks_is_not_a_broken_link(tmp_path, monkeypatch):
    """Another session of the home makes ``synapse/`` (its first upload, its first mark)
    after this one found nothing there and before it asks whether a link stands there.
    That is a folder, and no reason to skip the call."""
    folder = tmp_path / "synapse"
    real, raced = os.stat, []

    def stat(path, *args, **kwargs):
        if not raced and Path(path) == folder:
            raced.append(path)
            folder.mkdir()  # made by the other session, just after this look
            raise FileNotFoundError(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    assert not hs.marked(tmp_path, [_LOCAL])
    assert raced and hs.recheck(tmp_path, _RESET_CHILD).mode == hs.ACTIVE


def _broken_marks(home: Path, target: str = "inert") -> None:
    """A marks folder no lookup gets through: a link to itself, or (``target`` "gone")
    to a folder that is not there, as when the volume it lives on is not mounted."""
    (home / "synapse").mkdir(parents=True, exist_ok=True)
    (home / "synapse" / "inert").symlink_to(target)


@pytest.mark.parametrize("target", ["inert", "gone"])
def test_marks_that_cannot_be_read_turn_the_session_off(tmp_path, target):
    """Not being able to look is not "not marked": the marks may have kept this chat
    off."""
    _store(tmp_path)
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE
    _broken_marks(tmp_path, target)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway())
    assert verdict.mode == hs.INERT and "the marks could not be read" in verdict.reason
    with pytest.raises(hs.StoreError):
        hs.recheck(tmp_path, _LOCAL, gateway=True)
    bare = tmp_path / "bare"  # an in-process session, no state.db
    _broken_marks(bare, target)
    verdict = hs.check_session(bare, _RESET_CHILD, _in_process())
    assert verdict.mode == hs.INERT and "the marks could not be read" in verdict.reason
    with pytest.raises(hs.StoreError):
        hs.recheck(bare, _RESET_CHILD)


def test_a_call_is_skipped_while_the_marks_cannot_be_read(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    assert "BOARD-TEXT" in provider.system_prompt_block()
    _broken_marks(home)
    assert provider.system_prompt_block() == "" and provider.prefetch("x") == ""
    (home / "synapse" / "inert").unlink()  # skipped, not turned off: it comes back
    assert "BOARD-TEXT" in provider.system_prompt_block()


def test_an_agent_rebuilt_after_a_foreign_turn_stays_off(plugin, server, tmp_path):
    """Off "for good" has to outlast the instance: the gateway builds a new agent for
    the same session (a restart, an evicted cache), and a resumed session uploads up to
    ten earlier turns, the foreign one among them, as the owner's."""
    home = tmp_path / "hermes-a"
    _store(home)
    _store(home, _BRANCH)
    provider = _start_gateway(_new(plugin), home)
    provider.on_turn_start(1, "hi", model="m", **_FOREIGN_TURN)
    assert provider.prefetch("x") == ""
    rebuilt = _start_gateway(_new(plugin), home, resumed=True, message_count=2)
    _assert_nothing_served(rebuilt, server)
    assert "marked" in hs.check_session(home, _LOCAL, _gateway()).reason
    # Another chat of the same home is the owner's as before.
    other = _start_gateway(_new(plugin), home, _BRANCH)
    assert "BOARD-TEXT" in other.system_prompt_block()


def test_an_in_process_session_rebuilt_after_a_foreign_turn_stays_off(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"  # the installed plugin's home; no state.db in it
    provider = _new(plugin)
    provider.initialize(_RESET_CHILD, hermes_home=str(home), **_in_process())
    provider.sync_turn(
        "q", "a", session_id=_RESET_CHILD, messages=_conversation(1), turn_author=_BOT
    )
    assert server.posts() == []
    rebuilt = _new(plugin)
    rebuilt.initialize(_RESET_CHILD, hermes_home=str(home), **_in_process(resumed=True))
    _assert_nothing_served(rebuilt, server)


def test_a_title_the_keywords_carried_marks_the_session(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    provider = _new(plugin)
    provider.initialize(
        _RESET_CHILD, hermes_home=str(home), **_in_process(session_title="Bot Chat")
    )
    renamed = _new(plugin)
    renamed.initialize(_RESET_CHILD, hermes_home=str(home), **_in_process(session_title="Notes"))
    _assert_nothing_served(renamed, server)


def test_a_foreign_turn_in_a_branch_leaves_its_parent_on(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home, title="Cache design")
    _branch(home)
    branch = _start_gateway(_new(plugin), home, _BRANCH)
    branch.on_turn_start(1, "hi", model="m", **_FOREIGN_TURN)
    assert hs.check_session(home, _BRANCH, _gateway(_BRANCH)).mode == hs.INERT
    assert hs.check_session(home, _LOCAL, _gateway()).mode == hs.ACTIVE
    # A branch of the marked branch carries its turns: off.
    third = "local-" + "7a" * 32
    _branch(home, parent=_BRANCH, child=third, title="More notes")
    assert "marked" in hs.check_session(home, third, _gateway(third)).reason


def test_a_bot_chat_resumed_onto_the_owners_route_does_not_mark_that_chat(tmp_path):
    """The mark names the chat the evidence is about: the Bot Chat, not the owner's
    chat whose route ran it for a while."""
    _store(tmp_path)
    _store(tmp_path, _BRANCH, request_id="bot-chat:/home/kyle/.hermes:2")
    assert "Bot Chat" in hs.check_session(tmp_path, _BRANCH, _gateway()).reason
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE
    assert hs.marked(tmp_path, [_BRANCH]) and not hs.marked(tmp_path, [_LOCAL])


def _delete(home: Path, sid: str, *, keep_origin: bool = True) -> None:
    """A deleted or pruned session (hermes_state_sessions.py :1768-1771,
    hermes_state_mutation_retirement.py, hermes_state_maintenance.py): its row and
    receipt go, and its children lose their ``parent_session_id``. Their
    ``model_config`` keeps naming it unless ``keep_origin`` is off."""
    conn = sqlite3.connect(home / "state.db")
    try:
        if not keep_origin:
            conn.execute("UPDATE sessions SET model_config=NULL WHERE parent_session_id=?", (sid,))
        conn.execute("UPDATE sessions SET parent_session_id=NULL WHERE parent_session_id=?", (sid,))
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        conn.execute("DELETE FROM state_meta WHERE key=?", ("gateway.local_policy.v1:" + sid,))
        conn.commit()
    finally:
        conn.close()


def test_a_branch_stays_off_after_its_marked_parent_is_deleted(tmp_path):
    """The branch holds a copy of the peers' turns, and a resume uploads up to ten of
    them. Its parent's mark is copied down when it is first judged, so it does not
    depend on a row Hermes may delete."""
    _old_bot_chat(tmp_path)
    assert "Bot Chat" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason
    gs._retitle(tmp_path, _LOCAL, "Old notes")
    _branch(tmp_path)
    assert "marked" in hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)).reason
    _delete(tmp_path, _LOCAL, keep_origin=False)  # nothing in the store names the parent
    for verdict in (
        hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)),
        hs.recheck(tmp_path, _BRANCH, gateway=True, chat_id=_BRANCH),
    ):
        assert verdict.mode == hs.INERT and "marked" in verdict.reason


def test_a_branch_never_opened_is_known_by_the_origin_hermes_keeps(tmp_path):
    """Branched after the rename and never opened before the Bot Chat was deleted:
    ``_branched_from`` still names the marked session."""
    _old_bot_chat(tmp_path)
    assert "Bot Chat" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason
    gs._retitle(tmp_path, _LOCAL, "Old notes")
    _branch(tmp_path)
    _delete(tmp_path, _LOCAL)
    verdict = hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH))
    assert verdict.mode == hs.INERT and "marked" in verdict.reason
    # A config that names no session says nothing.
    conn = sqlite3.connect(":memory:")
    try:
        assert hs.lineage(conn, None, "not json") == []
        assert hs.lineage(conn, None, json.dumps({"_branched_from": 7})) == []
    finally:
        conn.close()


@pytest.mark.parametrize("key", ["_reset_from", "_compressed_from"])
def test_a_continuation_is_known_by_its_origin_too(tmp_path, key):
    """``origin`` reads the other two keys as it reads ``_branched_from``: a row with
    no parent link that names a marked session is its continuation, and the mark is
    copied down to it. Hermes writes these keys for gateway chats
    (hermes_state_mutation_reset.py, hermes_state_mutation_compress.py); the row here is
    the barest one that carries one."""
    hs.mark(tmp_path, [_RESET_CHILD], "a turn another agent or person wrote")
    later = "20261112_181500_0a1b2c"
    _row(tmp_path, later, "", model_config={key: _RESET_CHILD}, source="cli")
    assert "marked" in hs.check_session(tmp_path, later, _in_process()).reason
    assert hs.marked(tmp_path, [later])


def test_evidence_read_from_a_parent_marks_the_parent(tmp_path):
    """The Bot Chat was never opened here, its branch was: the mark names the Bot Chat
    too, so it stays off once its title is gone."""
    _old_bot_chat(tmp_path)
    _branch(tmp_path)
    assert "Bot Chat" in hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)).reason
    assert hs.marked(tmp_path, [_LOCAL]) and hs.marked(tmp_path, [_BRANCH])
    gs._retitle(tmp_path, _LOCAL, "Old notes")
    assert "marked" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason


def test_a_session_run_on_a_marked_chats_route_is_not_marked_itself(tmp_path):
    """The mark is about that chat's history. A session of the owner's that a route of
    it runs for a while is off there, and the owner's again on its own route."""
    _old_bot_chat(tmp_path)
    assert "Bot Chat" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason
    gs._retitle(tmp_path, _LOCAL, "Old notes")
    _store(tmp_path, _BRANCH, source="gui")
    assert "marked" in hs.check_session(tmp_path, _BRANCH, _gateway()).reason
    assert hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)).mode == hs.ACTIVE


def _marks(home: Path) -> set[str]:
    """The ids the marks of ``home`` name: the first line of each file."""
    folder = home / "synapse" / "inert"
    if not folder.is_dir():
        return set()
    return {path.read_text().split("\n", 1)[0] for path in folder.iterdir()}


#: A chat and two sessions whose ids are not their chat's, so a session's mark and a
#: chat's can be told apart (``_store`` gives a first session its chat's id, as Hermes).
_COPY_CHAT = "local-" + "9c" * 32
_COPY = "20261112_190000_c0ffee"
_SEGMENT = "20261112_183000_5e6f70"


def _copy_of(home: Path, parent: str) -> None:
    """A session of its own chat that names ``parent`` as where it was branched from."""
    _store(home, _COPY_CHAT, source="gui")
    _row(home, _COPY, _COPY_CHAT, model_config={"_branched_from": parent})


def test_a_parents_title_marks_the_copy_its_chat_the_parent_and_the_parents_chat(tmp_path):
    _store(tmp_path, source="gui")
    _row(tmp_path, _SEGMENT, _LOCAL, title="Bot Chat")
    _copy_of(tmp_path, _SEGMENT)
    assert "Bot Chat" in hs.check_session(tmp_path, _COPY, _gateway(_COPY_CHAT)).reason
    assert _marks(tmp_path) == {_COPY, _COPY_CHAT, _SEGMENT, _LOCAL}


@pytest.mark.parametrize("evidence", ["a row of the parent's chat", "the parent chat's receipt"])
def test_evidence_in_a_parents_chat_marks_that_chat_and_the_parents_down_to_it(tmp_path, evidence):
    if evidence == "the parent chat's receipt":
        _store(tmp_path, request_id="bot-chat:/home/kyle/.hermes")
    else:
        _old_bot_chat(tmp_path)
    _row(tmp_path, _SEGMENT, _LOCAL)  # an untitled later segment; it names no origin
    _copy_of(tmp_path, _SEGMENT)
    assert "Bot Chat" in hs.check_session(tmp_path, _COPY, _gateway(_COPY_CHAT)).reason
    assert _marks(tmp_path) == {_COPY, _COPY_CHAT, _SEGMENT, _LOCAL}


def test_an_inherited_mark_is_copied_to_the_session_its_chat_and_the_parents_between(tmp_path):
    """A never-opened sibling of the copy, cut from the same segment, is then found by
    that segment's own mark once the whole chat is deleted."""
    _store(tmp_path, source="gui")
    _row(tmp_path, _SEGMENT, _LOCAL, model_config={"_reset_from": _LOCAL})
    hs.mark(tmp_path, [_LOCAL], "a turn another agent or person wrote")
    _copy_of(tmp_path, _SEGMENT)
    assert "marked" in hs.check_session(tmp_path, _COPY, _gateway(_COPY_CHAT)).reason
    assert _marks(tmp_path) == {_LOCAL, _COPY, _COPY_CHAT, _SEGMENT}
    _branch(tmp_path, parent=_SEGMENT)  # the sibling
    _delete(tmp_path, _SEGMENT)
    _delete(tmp_path, _LOCAL)
    assert "marked" in hs.check_session(tmp_path, _BRANCH, _gateway(_BRANCH)).reason


def test_a_mark_on_a_parents_id_alone_leaves_the_parents_chat_unmarked(tmp_path):
    """The parent was marked for what it was handed; its chat's other sessions were not."""
    _store(tmp_path, source="gui")
    _row(tmp_path, _SEGMENT, _LOCAL)
    hs.mark(tmp_path, [_SEGMENT], "a turn another agent or person wrote")
    _copy_of(tmp_path, _SEGMENT)
    assert "marked" in hs.check_session(tmp_path, _COPY, _gateway(_COPY_CHAT)).reason
    assert _marks(tmp_path) == {_SEGMENT, _COPY, _COPY_CHAT}
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE
