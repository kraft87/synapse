"""``/private`` inside Hermes: take the session it is typed in off the record.

Outside Hermes the toggle is ``synapse private on <session id>``, and the user has to
find the id. Hermes hands a plugin's slash command the id of the session it was typed in
(hermes_cli/plugin_command_dispatch.py), so here the command needs none:

    /private            whether this session is off the record (as ``status``)
    /private on         take it off the record
    /private status

It is the toggle every harness has (private_toggle.Toggle): a marker on this machine,
which every upload reads, and a row on the server. ``on`` answers "ON" only when both
are in place, the server's flag read back. What was uploaded before stays stored.

There is no ``off``. Private mode covers whole sessions, and a session taken off the
record stays off: the way back on the record is a new session. ``/private off`` says so.

Who may run it
--------------
Hermes runs a plugin command on every surface, a messaging chat included, where anyone
in the chat can type it. The command would then make requests with the owner's token.
So it goes by where it was typed, and refuses, with no request and no file written,
anywhere but the owner's own local session. That is the rule ``initialize`` applies to
the same facts (sessions.from_keywords):

- a ``source`` (the gateway daemon, where the CLI, the TUI and Desktop send a command with
  ``slash.exec``): its platform must be "local", its chat a local dm, its route
  (``session_key``) the route of that chat, and its ``user_id`` this machine's owner. The
  gateway names a local session's surface ("desktop") to ``initialize`` and not to a
  command;
- no ``source`` (a surface that runs the command in its own process: the classic CLI off
  the gateway, and the stand-alone TUI backend, which only test harnesses start): the
  ``platform`` Hermes names must be a local one.

A session the store says someone else writes into is not looked up here: the provider
uploads nothing from it whatever the marker says, and the toggle can only reach the one
session Hermes names.

What Hermes passes, run at bca51750: the stored session id, the one ``initialize`` and
``sync_turn`` name, also for a session that has had no turn yet (gateway, in both plugin
isolation modes, and the stand-alone TUI backend). Hermes registers the command when it
loads the provider. Its plugin discovery skips a memory provider (an "exclusive" plugin,
hermes_cli/plugins_discovery.py) and the provider is loaded as the first agent of the
process is built. Before that Hermes answers that the command is unknown: in a gateway
started just now, the first message of the first session is uploaded before the command
can be typed. ``synapse private on <session id>`` in a terminal has no such gap.

One handler serves every session of the process: a later registration replaces the
earlier one. So the handler keeps nothing and goes by the id it is given.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from typing import Any

from ... import __version__
from ...api import Client
from ...private_toggle import Toggle
from ...settings import Settings
from . import sessions as hs

NAME = "private"
DESCRIPTION = "Synapse: take this session off the record (/private on, /private status)"
ARGS_HINT = "[on|status]"

_USAGE = "Usage: /private on, /private status."
_NOT_LOCAL = "Synapse: /private works only in the owner's own local sessions. Nothing was changed."
_NO_SESSION = (
    "Synapse: Hermes named no session for this command. Nothing was changed, and this "
    "session is NOT off the record."
)
_OFF = (
    "Synapse: private mode covers whole sessions, and this command does not turn it off. "
    "To go back on the record, start a new session. Nothing was changed."
)
_ON = (
    "Synapse: private mode is ON for this session ({session_id}). Nothing more is "
    "uploaded from it. What was uploaded before stays stored. To go back on the record, "
    "start a new session."
)
_FAILED = "Synapse: /private failed (see the Hermes log). This session may NOT be off the record."


def _field(source: Any, name: str) -> Any:
    """``source.name``, for the gateway's SessionSource or a mapping of its fields."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _word(value: Any) -> str:
    value = getattr(value, "value", value)  # an enum member, as SessionSource.platform is
    return value.strip().lower() if isinstance(value, str) else ""


def refusal(platform: Any, session_key: Any, source: Any) -> str:
    """Why the command is not for whoever typed it, and "" when it is: the owner, in a
    local session (see the module docstring). Anything not known to be that is refused."""
    if source is None:
        place = _word(platform)
        if not place:
            return "no surface named"
        verdict = hs.from_keywords({"platform": place})
    else:
        place = _word(_field(source, "platform"))
        if place != "local":
            return f"a chat on {place or 'an unnamed platform'}"
        if _field(source, "is_bot"):
            return "a bot's message"
        keys = ("chat_id", "chat_type", "user_id", "user_id_alt", "user_name", "chat_name")
        facts = {key: _field(source, key) for key in (*keys, "thread_id")}
        # "desktop" stands for the local surface the gateway does not name here.
        verdict = hs.from_keywords(
            {**facts, "platform": "desktop", "gateway_session_key": session_key}
        )
    return "" if verdict.mode == hs.ACTIVE else verdict.reason


def _toggle(action: str, session_id: str) -> tuple[int, str]:
    """Run the shared toggle's ``action`` for ``session_id``: its exit code and what it
    printed, errors after the rest."""
    settings = Settings.load()
    out, err = io.StringIO(), io.StringIO()
    toggle = Toggle(
        Client(settings, f"synapse-client/{__version__} hermes"),
        program="synapse private",
        out=out,
        err=err,
    )
    code = toggle.on(session_id) if action == "on" else toggle.status(session_id)
    return code, (out.getvalue() + err.getvalue()).strip()


def private_command(
    raw_args: str = "",
    *,
    session_id: str = "",
    session_key: str = "",
    platform: str = "",
    source: Any = None,
) -> str:
    """The handler Hermes calls for ``/private``, with the session-context keywords it
    passes a handler that names them. Always returns the reply; never raises."""
    try:
        words = raw_args.split() if isinstance(raw_args, str) else []
        action = words[0].lower() if words else "status"
        if refusal(platform, session_key, source):
            return _NOT_LOCAL
        if action == "off":
            return _OFF
        if action not in ("on", "status") or len(words) > 1:
            return f"Synapse: {_USAGE}"
        sid = session_id if isinstance(session_id, str) else ""
        if not sid.strip():
            return _NO_SESSION
        code, said = _toggle(action, sid)
        if action == "status":
            return (
                f"Synapse private mode, {said}\n{_USAGE}"
                if not words
                else f"Synapse private mode, {said}"
            )
        if code == 0:
            return _ON.format(session_id=sid)
        reason = said.splitlines()[0] if said else "private mode FAILED"
        return f"Synapse: {reason}\nThis session is NOT off the record."
    except Exception:
        logging.getLogger(__name__).warning("synapse: /private failed", exc_info=True)
        return _FAILED
