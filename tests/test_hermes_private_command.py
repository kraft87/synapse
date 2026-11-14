"""``/private`` inside Hermes (harnesses/hermes/command.py).

The plugin registers one slash command beside its provider. Hermes hands the handler
the id of the session the command was typed in, so the user needs no id: ``/private on``
takes that session off the record, with the toggle every harness has.

What these tests hold it to: it answers "ON" only when the marker and the server's row
are both in place; nothing is uploaded after it, an upload that was waiting included;
it makes no request and writes no file for anyone but the owner in a local session;
and a Hermes that takes no plugin command still gets the provider.

What Hermes passes is copied from runs at bca51750 (the TUI backend and the gateway
daemon, with a recorder plugin). The fakes are the ones tests/test_hermes_provider.py
builds.
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
import types
from typing import Any

import pytest

from tests import test_hermes_provider as base

hermes_stub = base.hermes_stub
user = base.user
plugin = base.plugin
server = base.server

_SID = base._SID
_CHAT = "local-" + "ab" * 32
_ROUTE = f"agent:main:local:dm:{_CHAT}"


def _command(plugin: types.ModuleType) -> types.ModuleType:
    return sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.command"]


def _handler(plugin: types.ModuleType) -> Any:
    """The handler as Hermes gets it: from ``register``."""
    ctx = base._Ctx()
    plugin.register(ctx)
    return ctx.commands["private"]["handler"]


def _local(plugin: types.ModuleType, raw_args: str, session_id: str = _SID) -> str:
    """The command as the TUI backend calls it for a session of its own."""
    return _handler(plugin)(
        raw_args, source=None, session_id=session_id, session_key=session_id, platform="tui"
    )


def _source(**changes: Any) -> types.SimpleNamespace:
    """What the gateway passes for a local session (gateway/session.py SessionSource)."""
    fields: dict[str, Any] = {
        "platform": types.SimpleNamespace(value="local"),
        "chat_id": _CHAT,
        "chat_type": "dm",
        "user_id": f"uid:{os.getuid()}",
        "user_name": None,
        "chat_name": None,
        "thread_id": None,
        "user_id_alt": None,
        "is_bot": False,
    }
    fields.update(changes)
    return types.SimpleNamespace(**fields)


def _requests(server: Any) -> list[str]:
    """Every request but a session start's."""
    return [
        f"{r['method']} {r['path']}" for r in server.seen if not r["path"].startswith("/context")
    ]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_gives_hermes_the_provider_and_the_command(plugin):
    ctx = base._Ctx()
    plugin.register(ctx)
    (provider,) = ctx.providers
    assert provider.name == "synapse"
    entry = ctx.commands["private"]
    assert entry["handler"] is _command(plugin).private_command
    assert "off the record" in entry["description"] and entry["hint"] == "[on|status]"
    # Hermes passes a handler the session-context keywords it names
    # (hermes_cli/plugin_command_dispatch.py): the first parameter gets the arguments.
    names = list(inspect.signature(entry["handler"]).parameters)
    assert names[0] == "raw_args"
    assert set(names[1:]) == {"session_id", "session_key", "platform", "source"}


class _NoCommands:
    def __init__(self) -> None:
        self.providers: list[Any] = []

    def register_memory_provider(self, provider: Any) -> None:
        self.providers.append(provider)


class _RefusesCommands(_NoCommands):
    def register_command(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no commands here")


@pytest.mark.parametrize("ctx_type", [_NoCommands, _RefusesCommands])
def test_a_hermes_that_takes_no_command_still_gets_the_provider(plugin, ctx_type, caplog):
    ctx = ctx_type()
    plugin.register(ctx)
    (provider,) = ctx.providers
    assert provider.name == "synapse"
    assert "/private was not registered" in caplog.text


# ---------------------------------------------------------------------------
# on
# ---------------------------------------------------------------------------


def test_on_takes_the_session_off_the_record(plugin, server, tmp_path):
    provider = base._start(base._new(plugin), tmp_path / "hermes-a", platform="tui")
    base._say(provider, base._conversation(1), _SID)
    assert len(server.posts()) == 1

    reply = _local(plugin, "on")
    assert f"private mode is ON for this session ({_SID})" in reply
    assert "uploaded before stays stored" in reply and "new session" in reply
    assert (tmp_path / "private" / _SID).is_file()
    # Told, then asked: "ON" is said of the server's own answer.
    assert _requests(server)[1:] == [
        f"PUT /private-sessions/{_SID}",
        f"GET /private-sessions/{_SID}",
    ]

    base._say(provider, base._conversation(2), _SID)
    provider.on_pre_compress(base._conversation(2))
    provider.on_session_end(base._conversation(2))
    assert len(server.posts()) == 1


def test_on_is_said_only_when_the_server_has_the_row(plugin, server, tmp_path):
    server.private_status = 503
    reply = _local(plugin, "on")
    assert "private mode FAILED" in reply and "NOT off the record" in reply
    assert "is ON" not in reply
    # The half that needs no network is in place all the same.
    assert (tmp_path / "private" / _SID).is_file()


def test_on_before_the_sessions_first_turn(plugin, server, tmp_path):
    """Hermes names the stored id for a session that has had no turn (run in the TUI
    backend and the gateway): the provider built for it later uploads nothing."""
    assert "is ON" in _local(plugin, "on")
    provider = base._start(base._new(plugin), tmp_path / "hermes-a", platform="tui")
    base._say(provider, base._conversation(1), _SID)
    provider.on_session_end(base._conversation(1))
    assert server.posts() == []


class _Held:
    """A client whose ``ingest`` waits until released: an upload in flight."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.entered = threading.Event()
        self.go = threading.Event()

    def ingest(self, payload: Any, timeout: float) -> Any:
        self.entered.set()
        assert self.go.wait(10)
        return self._client.ingest(payload, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def test_an_upload_that_waits_when_on_runs_is_not_sent(plugin, server, tmp_path):
    """Turn uploads run after ``sync_turn`` has returned (background.py). When the
    session goes private, one that has not made its request is stopped where it would
    leave the machine (the client's own check), and one that has not begun reads the
    marker when it runs. Neither is kept in the outbox."""
    home = tmp_path / "hermes-a"
    provider = base._start(base._new(plugin, waits=False), home, platform="tui")
    held = provider._client = provider._uploader.client = _Held(provider._client)
    base._say(provider, base._conversation(1), _SID)
    assert held.entered.wait(10)  # turn 1's upload runs, its request not yet made
    base._say(provider, base._conversation(2), _SID)  # waits for it

    assert "is ON" in _local(plugin, "on")
    held.go.set()
    assert base._uploaded(provider)
    assert server.posts() == []
    assert base.Outbox(home / "synapse" / "outbox").pending() == 0


# ---------------------------------------------------------------------------
# Who may run it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "acp", "TUI"])
def test_the_owners_own_process_may_run_it(plugin, server, platform):
    reply = _handler(plugin)(
        "status", source=None, session_id=_SID, session_key=_SID, platform=platform
    )
    assert reply == f"Synapse private mode, {_SID}: marker=no server=no"


@pytest.mark.parametrize("as_mapping", [False, True])
def test_a_local_session_of_the_gateway_may_run_it(plugin, server, tmp_path, as_mapping):
    source: Any = _source()
    if as_mapping:
        source = {**vars(source), "platform": "local"}
    reply = _handler(plugin)(
        "on", source=source, session_id=_SID, session_key=_ROUTE, platform="local"
    )
    assert "is ON" in reply and (tmp_path / "private" / _SID).is_file()


_REFUSED = {
    "a messaging platform, in process": {"platform": "telegram", "source": None},
    "no surface named": {"platform": "", "source": None},
    "a cron job": {"platform": "cron", "source": None},
    "a messaging chat": {"source": _source(platform=types.SimpleNamespace(value="telegram"))},
    "a platform with no name": {"source": _source(platform=None)},
    "a group chat": {"source": _source(chat_type="group")},
    "a chat that is not a local one": {"source": _source(chat_id="12345")},
    "someone else": {"source": _source(user_id="uid:4242424")},
    "no user": {"source": _source(user_id=None)},
    "a bot": {"source": _source(is_bot=True)},
    "a named sender": {"source": _source(user_name="mallory")},
    "a thread": {"source": _source(thread_id="7")},
    "the route of another chat": {"session_key": "agent:main:local:dm:local-" + "cd" * 32},
    "no route": {"session_key": ""},
    "a source of an unknown shape": {"source": object()},
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
@pytest.mark.parametrize("action", ["on", "status", "off", ""])
def test_anyone_else_changes_nothing_and_causes_no_request(plugin, server, tmp_path, case, action):
    context: dict[str, Any] = {
        "source": _source(),
        "session_id": _SID,
        "session_key": _ROUTE,
        "platform": "local",
    }
    context.update(_REFUSED[case])
    reply = _handler(plugin)(action, **context)
    assert reply.startswith("Synapse: /private works only in the owner's own local sessions.")
    assert "Nothing was changed" in reply
    assert str(tmp_path) not in reply and _SID not in reply  # no path, no id
    assert server.seen == [] and not (tmp_path / "private").exists()


def test_a_process_started_for_another_agents_message_may_not_run_it(
    plugin, server, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_TURN_AUTHOR", "agent:other")
    assert "works only in the owner's own local sessions" in _local(plugin, "on")
    assert server.seen == [] and not (tmp_path / "private").exists()


# ---------------------------------------------------------------------------
# The rest of the command
# ---------------------------------------------------------------------------


def test_status_says_what_the_marker_and_the_server_say(plugin, server, tmp_path):
    assert _local(plugin, "status") == f"Synapse private mode, {_SID}: marker=no server=no"
    assert "is ON" in _local(plugin, "on")
    assert _local(plugin, "status") == f"Synapse private mode, {_SID}: marker=yes server=yes"
    # With no argument it is the status, and how to use the command.
    bare = _local(plugin, "")
    assert bare.splitlines() == [
        f"Synapse private mode, {_SID}: marker=yes server=yes",
        "Usage: /private on, /private status.",
    ]
    server.private_status = 503
    assert "server=unknown" in _local(plugin, "status")


def test_off_is_not_done_here(plugin, server, tmp_path):
    assert "is ON" in _local(plugin, "on")
    before = list(server.seen)
    reply = _local(plugin, "off")
    assert "does not turn it off" in reply and "start a new session" in reply
    assert "Nothing was changed" in reply
    assert server.seen == before and (tmp_path / "private" / _SID).is_file()
    assert _local(plugin, "off --forget") == reply


@pytest.mark.parametrize("session_id", ["", "   ", None])
def test_no_session_id_is_refused_before_any_request(plugin, server, tmp_path, session_id):
    reply = _handler(plugin)(
        "on", source=None, session_id=session_id, session_key="", platform="tui"
    )
    assert "named no session" in reply and "NOT off the record" in reply
    assert server.seen == [] and not (tmp_path / "private").exists()


def test_a_session_id_that_is_no_file_name_is_refused(plugin, server, tmp_path):
    reply = _local(plugin, "on", "../../etc/passwd")
    assert "private mode FAILED" in reply and "NOT off the record" in reply
    assert server.seen == []


@pytest.mark.parametrize("raw_args", ["on now", "On extra words", "forget", "--help", "private"])
def test_anything_else_gets_the_usage(plugin, server, tmp_path, raw_args):
    assert _local(plugin, raw_args) == "Synapse: Usage: /private on, /private status."
    assert server.seen == [] and not (tmp_path / "private").exists()


def test_on_is_read_whatever_its_case(plugin, server, tmp_path):
    assert "is ON" in _local(plugin, "  ON  ")
    assert (tmp_path / "private" / _SID).is_file()


def test_the_handler_never_raises(plugin, server, tmp_path, monkeypatch, caplog):
    def broken() -> Any:
        raise RuntimeError("no settings")

    monkeypatch.setattr(_command(plugin).Settings, "load", staticmethod(broken))
    reply = _local(plugin, "on")
    assert "/private failed" in reply and "may NOT be off the record" in reply
    assert "synapse: /private failed" in caplog.text
