"""Where the Hermes provider works: the gateway daemon's local sessions, and everyone else.

``hermes chat`` and the TUI run their turns in Hermes's gateway daemon, which hands the
provider a chat identity even for the owner's own session. The keyword shapes below are
the ones a recording provider captured from real Hermes (the checkout's
``_memory_provider_init_kwargs``) on 2026-11-12: a plain ``hermes chat -q``, ``hermes
chat --source tool -q``, ``hermes chat -c "Bot Chat" --create-if-missing -q``, the TUI
through the daemon, and the in-process stdio TUI. The recorded ``user_id`` was
``uid:1000``; here it is this test process's own uid. The session store is a minimal
``state.db`` with the two tables and the receipt shape Hermes writes
(hermes_state_local.py ``commit_local_session``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_provider as base
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402

# The provider tests' loader, fake server and conversations.
hermes_stub = base.hermes_stub
user = base.user
plugin = base.plugin
server = base.server
_new = base._new
_conversation = base._conversation
_users = base._users

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="the owner subject is POSIX")

_HEX = "3e92f8ba0647c963b66d514db0cb594ffd73a1931bcff8f210f4f544d435ca66"
_LOCAL = f"local-{_HEX}"
_OTHER = "local-" + "51138ffd97c4c3c890dac040908effd31f309c2a0120811ca3c984da313ae6dc"
_WORK = "/home/kyle/work/hermes-scratch"


def _owner() -> str:
    return f"uid:{os.getuid()}"


def _gateway(sid: str = _LOCAL, **extra: Any) -> dict[str, Any]:
    """A local session's keywords in the gateway daemon (recorded: plain chat, --source
    tool and the Bot Chat all look like this; the Bot Chat adds its title)."""
    kwargs: dict[str, Any] = {
        "platform": "cli",
        "agent_context": "primary",
        "resumed": False,
        "message_count": 0,
        "user_id": _owner(),
        "chat_id": sid,
        "chat_type": "dm",
        "gateway_session_key": f"agent:main:local:dm:{sid}",
        "agent_identity": "default",
        "agent_workspace": "hermes",
    }
    kwargs.update(extra)
    return kwargs


def _in_process(**extra: Any) -> dict[str, Any]:
    """The stdio TUI's own agent (tui_gateway/agent_factory.py), recorded."""
    kwargs: dict[str, Any] = {
        "platform": "tui",
        "agent_context": "primary",
        "resumed": False,
        "message_count": 1,
        "cwd": _WORK,
        "agent_identity": "default",
        "agent_workspace": "hermes",
    }
    kwargs.update(extra)
    return kwargs


def _store(
    home: Path,
    sid: str = _LOCAL,
    *,
    source: str = "oneshot",
    principal: str | None = None,
    request_id: str = "0983bef1ab994f6e8a1c",
    title: str | None = None,
    request: dict[str, Any] | None = None,
    route: str | None = None,
    receipt: bool = True,
    api_bound: bool = False,
    chat_id: str | None = None,
) -> Path:
    """A session row and, for a local chat, its creation receipt, as Hermes stores them."""
    home.mkdir(parents=True, exist_ok=True)
    db = home / "state.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, "
            "created_source TEXT, user_id TEXT, chat_id TEXT, chat_type TEXT, "
            "session_key TEXT, cwd TEXT, title TEXT, model_config TEXT)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
        chat = sid if chat_id is None else chat_id
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, source, created_source, user_id, chat_id, "
            "chat_type, session_key, cwd, title) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sid,
                source,
                source,
                principal or _owner(),
                chat,
                "dm",
                f"agent:main:local:dm:{chat}",
                None,
                title,
            ),
        )
        if receipt:
            sent = (
                {"cwd": _WORK, "row_cwd": _WORK, "source": source} if request is None else request
            )
            value = {
                "profile_id": str(home),
                "principal_id": principal or _owner(),
                "request_id": request_id,
                "session_id": chat,
                "route": route or f"agent:main:local:dm:{chat}",
                "policy": {
                    "source": source,
                    "platform": "cli",
                    "cwd": "/srv/hermes-daemon",
                    "request_json": json.dumps(sent),
                },
                "entry": {"origin": {"chat_id": chat}},
            }
            conn.execute(
                "INSERT OR REPLACE INTO state_meta VALUES (?,?)",
                ("gateway.local_policy.v1:" + chat, json.dumps(value)),
            )
        if api_bound:
            conn.execute(
                "INSERT OR REPLACE INTO state_meta VALUES (?,?)",
                ("gateway.api.binding.v1." + sid, "{}"),
            )
        conn.commit()
    finally:
        conn.close()
    return db


def _retitle(home: Path, sid: str, title: str | None) -> None:
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))
        conn.commit()
    finally:
        conn.close()


def _provider_module(plugin: Any) -> Any:
    return sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.provider"]


# ---------------------------------------------------------------------------
# The keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        _gateway(),
        _gateway(platform="tui"),
        _gateway(platform="desktop"),
        _gateway(platform="acp"),
        _gateway(session_title="Bot Chat ideas"),  # a title, not the Bot Chat
        _in_process(),
        _in_process(user_id="uid:1000"),
        {"session_id": "s", "platform": "cli", "agent_context": "primary"},
    ],
)
def test_the_owners_local_shapes_pass_the_keywords(kwargs):
    assert hs.from_keywords(kwargs) == hs.Verdict(hs.ACTIVE)


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        (_gateway(session_title="Bot Chat", session_title_source="user"), "Bot Chat"),
        (_gateway(session_title="  bot chat "), "Bot Chat"),
        (_gateway(session_title="Bot Chat #2"), "Bot Chat"),
        (_gateway(session_title="Bot Chat (old)"), "Bot Chat"),
        (_in_process(session_title="Bot Chat"), "Bot Chat"),
        (_gateway(user_id="api-peer"), "owner"),
        (_gateway(user_id="uid:99999"), "owner"),
        (_gateway(user_id=None), "owner"),
        (_gateway(user_id="sid:S-1-5-21-1"), "owner"),
        (_gateway(chat_type="group"), "not a local session"),
        (_gateway(chat_id="local-abc"), "not a local session"),
        (_gateway(chat_id="123456789"), "not a local session"),
        (_gateway(gateway_session_key=f"agent:main:telegram:dm:{_LOCAL}"), "route"),
        (_gateway(gateway_session_key=f"agent:main:local:dm:{_OTHER}"), "route"),
        (_gateway(gateway_session_key=None), "route"),
        (_gateway(user_name="someone"), "messaging identity"),
        (_gateway(user_id_alt="+15550000"), "messaging identity"),
        (_gateway(chat_name="family"), "messaging identity"),
        (_gateway(thread_id="42"), "messaging identity"),
        (_gateway(platform="telegram"), "platform"),
        (_gateway(platform="api_server"), "platform"),
        (_gateway(platform="bot_room"), "platform"),
        (_gateway(platform="a2a"), "platform"),
        (_gateway(platform="kanban"), "platform"),
        (_gateway(platform="webhook"), "platform"),
        (_gateway(platform="cron", agent_context="cron"), "cron job"),
        (_gateway(agent_context="subagent"), "agent_context"),
        (_gateway(agent_context="flush"), "agent_context"),
        ({"platform": "api_server", "user_id": "api", "chat_id": "c1"}, "platform"),
        ({"platform": "discord", "chat_id": "1", "chat_type": "group", "user_id": "9"}, "platform"),
        ({"platform": "cli", "chat_id": "123"}, "not a local session"),
    ],
)
def test_every_other_shape_is_inert_and_says_why(kwargs, why):
    verdict = hs.from_keywords(kwargs)
    assert verdict.mode == hs.INERT
    assert why in verdict.reason


def test_a_cron_job_is_inert_with_or_without_a_chat():
    """cron/scheduler.py ``_construct_cron_agent`` builds the job's agent with
    ``platform="cron"``; agent_init_memory_provider.py makes that ``agent_context``."""
    for kwargs in (
        {"platform": "cron", "agent_context": "cron"},
        {"platform": "cli", "agent_context": "cron"},
        _gateway(platform="cron"),
    ):
        verdict = hs.from_keywords(kwargs)
        assert verdict.mode == hs.INERT and "cron job" in verdict.reason


def test_off_posix_a_gateway_session_is_inert(monkeypatch):
    shape = _gateway()
    monkeypatch.delattr(os, "getuid")
    assert hs.owner_subject() is None
    assert hs.from_keywords(shape).mode == hs.INERT
    assert hs.from_keywords(_in_process()).mode == hs.ACTIVE


# ---------------------------------------------------------------------------
# The session store
# ---------------------------------------------------------------------------


def test_the_owners_gateway_session_is_active_with_the_clients_cwd(tmp_path):
    _store(tmp_path)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway())
    assert verdict == hs.Verdict(hs.ACTIVE, "", _WORK)
    # A client that sent no cwd: none, never the policy's (the daemon's) own.
    _store(tmp_path, request={"source": "tui"}, source="tui")
    assert hs.check_session(tmp_path, _LOCAL, _gateway()) == hs.Verdict(hs.ACTIVE)


@pytest.mark.parametrize("source", ["cli", "tui", "gui", "acp", "oneshot"])
def test_each_owner_source_is_active(tmp_path, source):
    _store(tmp_path, source=source)
    assert hs.check_session(tmp_path, _LOCAL, _gateway()).mode == hs.ACTIVE


@pytest.mark.parametrize(
    ("store", "why"),
    [
        ({"source": "tool"}, "'tool'"),
        ({"source": "bot_room"}, "'bot_room'"),
        ({"source": "a2a"}, "'a2a'"),
        ({"request_id": "bot-chat:/home/kyle/.hermes"}, "Bot Chat"),
        ({"request_id": "bot-chat:/home/kyle/.hermes:3", "title": None}, "Bot Chat"),
        ({"source": "local", "title": "Bot Chat"}, "Bot Chat"),
        ({"principal": "api-peer"}, "someone other"),
        ({"principal": "uid:99999"}, "someone other"),
        ({"receipt": False}, "no creation record"),
        ({"route": f"agent:other:local:dm:{_LOCAL}"}, "another session"),
        ({"api_bound": True}, "API server"),
    ],
)
def test_a_gateway_session_the_store_does_not_vouch_for_is_inert(tmp_path, store, why):
    _store(tmp_path, **store)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway())
    assert verdict.mode == hs.INERT
    assert why in verdict.reason


def test_no_store_or_a_broken_one_is_inert_for_the_gateway(tmp_path):
    assert "no session store" in hs.check_session(tmp_path, _LOCAL, _gateway()).reason
    assert hs.check_session(None, _LOCAL, _gateway()).mode == hs.INERT
    (tmp_path / "state.db").write_bytes(b"this is not a database" * 100)
    verdict = hs.check_session(tmp_path, _LOCAL, _gateway())
    assert verdict.mode == hs.INERT and "could not be read" in verdict.reason


def test_the_store_is_read_only(tmp_path):
    db = _store(tmp_path)
    before = db.stat().st_mtime_ns, db.read_bytes()
    hs.check_session(tmp_path, _LOCAL, _gateway())
    hs.recheck(tmp_path, _LOCAL)
    assert (db.stat().st_mtime_ns, db.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.db"]


def test_an_in_process_session_is_checked_but_needs_no_store(tmp_path):
    sid = "20261112_135415_30e776"
    assert hs.check_session(tmp_path, sid, _in_process()).mode == hs.ACTIVE  # no state.db
    _store(tmp_path, sid, source="tui", receipt=False, chat_id="")
    assert hs.check_session(tmp_path, sid, _in_process()) == hs.Verdict(hs.ACTIVE)
    _retitle(tmp_path, sid, "Bot Chat")
    assert hs.check_session(tmp_path, sid, _in_process()).mode == hs.INERT
    _store(tmp_path, sid, source="tool", receipt=False, chat_id="")
    assert hs.check_session(tmp_path, sid, _in_process()).mode == hs.INERT
    # A gateway session resumed in-process is held to its receipt.
    _store(tmp_path, _LOCAL, request_id="bot-chat:/home/kyle/.hermes")
    assert hs.check_session(tmp_path, _LOCAL, _in_process()).mode == hs.INERT
    _store(tmp_path, _LOCAL)
    assert hs.check_session(tmp_path, _LOCAL, _in_process()) == hs.Verdict(hs.ACTIVE)


_TELEGRAM = "20261112_090000_aa11bb"


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ({"source": "telegram", "chat_id": "123456789"}, "'telegram'"),
        ({"source": "api_server", "chat_id": _TELEGRAM}, "'api_server'"),
        ({"source": "bot_room", "chat_id": _TELEGRAM}, "'bot_room'"),
        ({"source": "cron", "chat_id": ""}, "'cron'"),
        ({"source": None, "chat_id": ""}, "None"),
        ({"source": "local", "chat_id": "123456789"}, "not a local one"),
    ],
)
def test_a_session_resumed_from_elsewhere_is_inert(tmp_path, row, why):
    """``/resume`` puts another session on the owner's route (the gateway rebuilds the
    agent with the route's keywords), or under an in-process agent: the row decides."""
    _store(tmp_path)
    _store(tmp_path, _TELEGRAM, receipt=False, **row)
    for kwargs in (_gateway(), _in_process()):
        verdict = hs.check_session(tmp_path, _TELEGRAM, kwargs)
        assert verdict.mode == hs.INERT
        assert why in verdict.reason
    assert hs.recheck(tmp_path, _TELEGRAM).mode == hs.INERT


def test_a_session_of_another_local_chat_needs_its_own_receipt(tmp_path):
    _store(tmp_path)
    _store(tmp_path, _OTHER, request={"cwd": "/home/kyle/elsewhere", "source": "tui"})
    assert hs.check_session(tmp_path, _OTHER, _gateway()) == hs.Verdict(hs.ACTIVE, "", _WORK)
    _store(tmp_path, _OTHER, principal="api-peer")
    assert "someone other" in hs.check_session(tmp_path, _OTHER, _gateway()).reason
    _store(tmp_path, _OTHER, request_id="bot-chat:/home/kyle/.hermes:2")
    assert "Bot Chat" in hs.check_session(tmp_path, _OTHER, _gateway()).reason
    _store(tmp_path, _OTHER, receipt=False, source="local")
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("DELETE FROM state_meta WHERE key=?", ("gateway.local_policy.v1:" + _OTHER,))
    conn.commit()
    conn.close()
    assert "no creation record" in hs.check_session(tmp_path, _OTHER, _gateway()).reason


def test_recheck_reads_the_row_without_the_keywords(tmp_path):
    assert hs.recheck(tmp_path, _LOCAL) == hs.Verdict(hs.ACTIVE)  # no store
    assert hs.recheck(None, _LOCAL) == hs.Verdict(hs.ACTIVE)
    _store(tmp_path)
    assert hs.recheck(tmp_path, _LOCAL) == hs.Verdict(hs.ACTIVE)  # no cwd from it
    assert hs.recheck(tmp_path, "20261112_101600_cd34ef") == hs.Verdict(hs.ACTIVE)  # no row
    _retitle(tmp_path, _LOCAL, "Bot Chat")
    assert hs.recheck(tmp_path, _LOCAL).mode == hs.INERT
    (tmp_path / "state.db").write_bytes(b"not a database" * 200)
    with pytest.raises(hs.StoreError):
        hs.recheck(tmp_path, _LOCAL)


# ---------------------------------------------------------------------------
# The provider in the daemon
# ---------------------------------------------------------------------------


def _start_gateway(provider: Any, home: Path, sid: str = _LOCAL, **extra: Any) -> Any:
    provider.initialize(sid, hermes_home=str(home), **_gateway(sid, **extra))
    return provider


def test_the_owners_gateway_session_gets_everything(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    assert "BOARD-TEXT" in provider.system_prompt_block()
    (ctx,) = [r for r in server.seen if r["path"].startswith("/context")]
    assert "project=hermes-scratch" in ctx["path"]
    assert "Call remember FIRST" in provider.prefetch("x")
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    (payload,) = server.posts()
    assert payload["session_id"] == _LOCAL
    assert payload["project"] == "hermes-scratch" and payload["cwd"] == _WORK
    assert _users(payload["records"]) == ["question 1"]
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(2))
    assert _users(server.posts()[1]["records"]) == ["question 1", "question 2"]  # 1 is context
    out = json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert out == {"query": "x", "self_session": _LOCAL}


@pytest.mark.parametrize(
    ("kwargs", "store"),
    [
        ({"session_title": "Bot Chat"}, {"source": "cli", "title": "Bot Chat"}),
        ({}, {"source": "tool"}),
        ({}, {"request_id": "bot-chat:x"}),
        ({"user_id": "api-peer"}, {"principal": "api-peer"}),
        ({}, {"principal": "api-peer"}),
        ({}, {"receipt": False}),
        ({"platform": "telegram"}, {}),
        ({"user_name": "someone", "chat_type": "group"}, {}),
    ],
)
def test_a_session_that_is_not_the_owners_gets_nothing(plugin, server, tmp_path, kwargs, store):
    home = tmp_path / "hermes-a"
    _store(home, **store)
    provider = _start_gateway(_new(plugin), home, **kwargs)
    assert provider.system_prompt_block() == ""
    assert provider.prefetch("x") == ""
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(2))
    provider.on_pre_compress(_conversation(2))
    provider.on_session_end(_conversation(2))
    out = json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert "local sessions only" in out["error"]
    assert server.seen == []


def test_the_reason_is_logged_once(plugin, server, tmp_path, caplog):
    home = tmp_path / "hermes-a"
    _store(home, source="tool")
    with caplog.at_level(logging.INFO):
        provider = _start_gateway(_new(plugin), home)
        provider.handle_tool_call("recall", {"query": "x"})
        provider.sync_turn("q", "a", messages=_conversation(1))
    lines = [r.getMessage() for r in caplog.records if "off for session" in r.getMessage()]
    assert lines == [
        f"synapse: off for session {_LOCAL}: created as 'tool', not one of the owner's own chats"
    ]


def test_a_session_renamed_to_the_bot_chat_turns_off_for_good(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert len(server.posts()) == 1
    _retitle(home, _LOCAL, "Bot Chat")
    assert provider.prefetch("x") == ""
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(2))
    provider.on_session_end(_conversation(3))
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    _retitle(home, _LOCAL, "My chat")
    assert provider.prefetch("x") == ""
    assert len(server.posts()) == 1 and server.posts("/mcp") == []


def test_a_switch_to_a_session_that_is_not_the_owners_turns_off(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    _store(home, _TELEGRAM, source="telegram", chat_id="123456789", receipt=False)
    provider = _start_gateway(_new(plugin), home)
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert len(server.posts()) == 1
    provider.on_session_switch(_TELEGRAM, parent_session_id=_LOCAL, reset=False)
    provider.sync_turn("q", "a", session_id=_TELEGRAM, messages=_conversation(3))
    assert provider.prefetch("x") == ""
    assert "local sessions only" in json.loads(provider.handle_tool_call("recall", {}))["error"]
    provider.on_session_end(_conversation(3))
    assert len(server.posts()) == 1 and server.posts("/mcp") == []


def test_a_switch_to_another_of_the_owners_sessions_keeps_working(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    first, second = "20261112_101500_ab12cd", "20261112_101600_cd34ef"
    _store(home, first, source="cli", chat_id="", receipt=False)
    provider = _new(plugin)
    provider.initialize(first, hermes_home=str(home), **_in_process(platform="cli"))
    provider.sync_turn("q", "a", messages=_conversation(1))
    _store(home, second, source="cli", chat_id="", receipt=False)
    provider.on_session_switch(second, parent_session_id=first, reset=True)
    provider.sync_turn("q", "a", messages=_conversation(2))
    assert [p["session_id"] for p in server.posts()] == [first, second]


def test_an_unreadable_store_skips_a_call_without_turning_off(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    db = _store(home)
    provider = _start_gateway(_new(plugin), home)
    good = db.read_bytes()
    db.write_bytes(b"not a database" * 200)
    assert provider.prefetch("x") == ""
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert "error" in json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert server.posts() == []
    db.write_bytes(good)
    assert provider.prefetch("x")
    provider.sync_turn("q", "a", session_id=_LOCAL, messages=_conversation(1))
    assert len(server.posts()) == 1


def test_a_failed_initialize_leaves_the_tools_off(plugin, server, tmp_path, monkeypatch):
    home = tmp_path / "hermes-a"
    _store(home)
    module = _provider_module(plugin)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("config unreadable")

    monkeypatch.setattr(module.Settings, "load", broken)
    provider = _start_gateway(_new(plugin), home)
    monkeypatch.undo()
    assert "error" in json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert server.posts("/mcp") == []


def test_a_later_login_reaches_the_next_session(plugin, server, user, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    config = user / ".synapse" / "config.json"
    saved = config.read_text(encoding="utf-8")
    config.unlink()
    first = _start_gateway(_new(plugin), home)
    assert not first.is_available()
    assert first.system_prompt_block() == ""
    config.write_text(saved, encoding="utf-8")  # `synapse login` in a terminal
    second = _start_gateway(_new(plugin), home)
    assert second.is_available()
    assert "BOARD-TEXT" in second.system_prompt_block()


def _new_turns(payloads: list[Any]) -> list[str]:
    """The user rows each upload carried as new turns, in order. Only an upload's first
    turn may repeat an earlier one (the context it leads with); no turn is new twice."""
    seen: list[str] = []
    for payload in payloads:
        ids = [r["id"] for r in payload["records"] if r["role"] == "user"]
        if ids and ids[0] in seen:
            ids = ids[1:]
        assert not set(ids) & set(seen), "a turn uploaded twice"
        seen.extend(ids)
    return seen


def test_concurrent_daemon_sessions_keep_their_own_turns(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home, _LOCAL)
    _store(home, _OTHER, request={"cwd": "/home/kyle/elsewhere", "source": "tui"}, source="tui")
    a = _start_gateway(_new(plugin), home, _LOCAL)
    b = _start_gateway(_new(plugin), home, _OTHER, platform="tui")

    def conversation(tag: str, count: int) -> list[dict[str, Any]]:
        rows = _conversation(count)
        for row in rows:
            if row.get("message_uid"):
                row["message_uid"] = f"{tag}-{row['message_uid']}"
            if row["role"] in ("user", "assistant"):
                row["content"] = f"{tag} {row['content']}"
        return rows

    def run(provider: Any, tag: str) -> None:
        for count in range(1, 6):
            provider.sync_turn("q", "a", messages=conversation(tag, count))
        provider.on_session_end(conversation(tag, 5))

    threads = [threading.Thread(target=run, args=(p, t)) for p, t in ((a, "A"), (b, "B"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    by_session: dict[str, list[Any]] = {}
    for payload in server.posts():
        tags = {text.split()[0] for text in _users(payload["records"])}
        assert len(tags) == 1  # never one session's turns in another's upload
        tag = tags.pop()
        assert payload["session_id"] == {"A": _LOCAL, "B": _OTHER}[tag]
        assert payload["project"] == {"A": "hermes-scratch", "B": "elsewhere"}[tag]
        by_session.setdefault(tag, []).append(payload)
    for tag in ("A", "B"):
        assert _new_turns(by_session[tag]) == [f"{tag}-u{n}" for n in range(1, 6)]


def test_one_session_uploads_each_turn_once_under_concurrent_hooks(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    provider = _start_gateway(_new(plugin), home)
    conversation = _conversation(4)
    threads = [
        threading.Thread(
            target=provider.sync_turn, args=("q", "a"), kwargs={"messages": conversation}
        ),
        threading.Thread(target=provider.on_session_end, args=(conversation,)),
        threading.Thread(target=provider.on_pre_compress, args=(conversation,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(_new_turns(server.posts())) == [f"u{n}" for n in range(1, 5)]
