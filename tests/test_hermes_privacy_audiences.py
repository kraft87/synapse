"""The audiences a privacy review found the Hermes provider serving, and must not.

Each shape below is built from what Hermes writes (checkout bca51750), cited per test:
a Desktop Group Chat member, a Bot Chat the Desktop minted and then reset, a cron job,
a turn another agent wrote, a gateway session whose row is gone, a side agent the
gateway daemon builds, a process started with ``--source``, and a session renamed after
its system prompt was built. ``tests/test_hermes_gateway_sessions.py`` has the store
fixture (the tables and receipt ``commit_local_session`` writes) and the owner's shapes.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
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
    _store(tmp_path, request={"cwd": _WORK, "follow_profile_config": True})
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.INERT
    _store(tmp_path, request={"cwd": _WORK, "room_plumbing": True})
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.INERT
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
    _row(tmp_path, sid, "", title=None, model_config=model_config, source="tui")
    verdict = hs.check_session(tmp_path, sid, _in_process())
    assert verdict.mode == hs.INERT and "follows its profile" in verdict.reason
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
#: this install is ``bot:<profile>`` (``bot_author_id``).
_BOT = {"id": "bot:research", "name": "research", "is_bot": True}


def test_a_turn_author_is_never_uploaded_and_turns_the_session_off(
    plugin, server, tmp_path, caplog
):
    """run_agent.py:919-925 passes ``turn_author`` to ``sync_turn``, None on a human
    turn; tui_gateway/methods_bot_relay.py:88-96: "an unattributed turn is the HUMAN's
    to the recipient's memory, so a bot DM must stay bot-authored"."""
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1), turn_author=None)
    assert len(server.posts()) == 1
    with caplog.at_level(logging.INFO):
        provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(2), turn_author=_BOT)
        provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(3))
        provider.on_session_end(_conversation(3))
    assert len(server.posts()) == 1
    assert provider.prefetch("x") == "" and provider.system_prompt_block() == ""
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    said = [r.getMessage() for r in caplog.records if "off for session" in r.getMessage()]
    assert said == [f"synapse: off for session {_LOCAL}: a turn another agent or person wrote"]


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
