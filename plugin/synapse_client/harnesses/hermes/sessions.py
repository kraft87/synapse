"""Whose Hermes session this is: the rule that decides where the provider works.

Synapse is one person's memory. The provider may show it to the model, and store a
conversation in it, only when the person at this machine's keyboard is the one talking:
never a Telegram/Discord/Slack/WhatsApp chat, a hosted room, an API client, a peer agent
or anyone else. When the evidence is missing or unfamiliar the answer is INERT.

The keywords (``from_keywords``)
--------------------------------
Hermes builds them in agent/agent_init_memory_provider.py (``_memory_provider_init_kwargs``):
``platform`` (default "cli"), ``agent_context`` (``memory_agent_context``, else "cron" or
"subagent" by platform, else "primary"), ``session_title`` when the session has one, and
the gateway identity of ``_GATEWAY_IDENTITY_PARAMS`` (agent/agent_init_params.py) when set.

- Any of ``user_id_alt``, ``user_name``, ``chat_name``, ``thread_id``: a messaging chat.
  INERT.
- No gateway identity and ``agent_context`` or ``platform`` "cron": CRON (tools only).
- ``agent_context`` other than "primary" (a subagent, a memory flush or other fork):
  INERT. A ``platform`` other than "cli", "tui", "desktop", "acp" (every messaging
  platform, "api_server", "bot_room", "a2a", "kanban", "webhook", ...): INERT.
- ``session_title`` "Bot Chat" (also "Bot Chat #2", "Bot Chat (...)"): INERT. The Bot
  Chat is a local session other agents and API-key peers write into
  (gateway/session_bot.py, gateway/platforms/api_server_bot_chat.py).
- No gateway identity at all: the agent runs in the process that created it (the stdio
  TUI and Desktop backend, tui_gateway/agent_factory.py; ACP; the classic CLI runner).
  ACTIVE, as before: that process belongs to the person who started it.
- Gateway identity: the turn runs in the gateway daemon, which serves every local
  client and every messaging platform. ACTIVE only for the shape of a local session
  (gateway/session_execution.py ``_build_fresh_agent``; gateway/run_turn_runner.py maps
  ``Platform.LOCAL`` to "cli"/"tui"/"desktop"/"acp"): ``chat_type`` "dm", ``chat_id``
  ``local-<64 hex>`` (hermes_state_keys.py ``local_identity``), ``gateway_session_key``
  ``agent:<profile>:local:dm:<chat_id>``, and ``user_id`` this machine's owner,
  ``uid:<os.getuid()>``: the subject gateway/control_socket.py mints only for a peer
  of the same OS user (SO_PEERCRED). POSIX only: on Windows the subject is a SID and
  such a session is INERT.

The session store (``check_session``)
-------------------------------------
The keywords of a local session do not say who created it or for what: ``hermes chat
--source tool`` (a third-party integration) and a session named "Bot Chat" arrive with
the same keywords as the owner's own chat. Hermes records that once, at creation, in
``<hermes_home>/state.db``: ``state_meta['gateway.local_policy.v1:<chat_id>']`` holds the
creation receipt (hermes_state_local.py ``commit_local_session``), with ``principal_id``
(who created it), ``request_id`` (``bot-chat:<profile>[:<n>]`` for a Bot Chat the
gateway minted, gateway/session_bot.py ``_bot_chat_creation_id``) and the frozen
``policy`` (``source``: "cli", "tui", "gui", "acp", "oneshot", "tool", ...; the
client's own request, with the ``cwd`` it ran in when it sent one). It is read
read-only, with a short timeout, and never written. A gateway-shaped session is ACTIVE
only when that receipt is there and says: created by the owner, not a Bot Chat, a
source in OWNER_SOURCES, and the same route. ``tool`` stays INERT: Hermes keeps it out
of the owner's own chats too (gateway/session_policy.py ``DAYBREAK_LABELS``). A session
with an API binding (``gateway.api.binding.v1.<id>``) is INERT. A session the store
does not know, or a store that cannot be read, is INERT.

The session's own row is checked too, for every session: the session a chat runs need
not be the one it was created with (``/resume`` switches a route to another session,
gateway/slash_commands_session.py; the classic CLI's ``/resume`` and compression switch
the agent's session in place). A row titled as a Bot Chat, stored under a ``source``
other than the local ones in ROW_SOURCES (the gateway stores its local chats as
"local", "oneshot" or "tool", gateway/session_policy.py ``STORED_AS_LOCAL``), or on a
chat that is not a local one makes the session INERT; a row on another local chat needs
that chat's receipt to pass as well. For an in-process session no store, or no row (a
session not written yet), leaves it ACTIVE.

What the store says can change after the agent is built: a session may be renamed "Bot
Chat" (hermes_state_titles.py; Hermes then routes the Bot Chat's traffic to it without
rebuilding the agent), or switched to another session in place. ``recheck`` reads the
row again before each nudge, upload and tool call. An API binding cannot be added to a
local session later: gateway/session_api.py ``bind_api_steps`` refuses a session whose
row or live origin is not the API server's.

The working directory of a gateway session is the client's: the receipt's request
``cwd``, only when the client sent one. The policy's own ``cwd`` falls back to the
daemon's (gateway/session_policy.py ``resolve_agent_cwd``) and is never used.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

ACTIVE = "active"
CRON = "cron"
INERT = "inert"

LOCAL_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp"})
#: Creation labels of the owner's own chats (gateway/session_policy.py ``SURFACES``).
OWNER_SOURCES = frozenset({"cli", "tui", "gui", "acp", "oneshot"})
#: The ``sessions.source`` of the owner's chats: an in-process surface's own label, or
#: what the gateway stores for a local chat ("local", or a kept creation label).
ROW_SOURCES = frozenset({"cli", "tui", "desktop", "gui", "acp", "oneshot", "local"})
#: Seconds to wait for a locked session store.
DB_TIMEOUT = 1.0

_MESSAGING_IDENTITY = ("user_id_alt", "user_name", "chat_name", "thread_id")
_GATEWAY_IDENTITY = ("gateway_session_key", "chat_id", "chat_type")
_LOCAL_CHAT = re.compile(r"local-[0-9a-f]{64}")
_LOCAL_ROUTE = re.compile(r"agent:[^:\s]+:local:dm:(local-[0-9a-f]{64})")
_BOT_CHAT = re.compile(r"bot chat(\s*#\s*\d+|\s*\(.*\))?", re.IGNORECASE | re.DOTALL)
_POLICY_KEY = "gateway.local_policy.v1:"
_API_KEY = "gateway.api.binding.v1."
_BOT_CHAT_REQUEST = "bot-chat:"


class Verdict(NamedTuple):
    """``mode`` (ACTIVE, CRON or INERT), why it is INERT, and a gateway session's cwd."""

    mode: str
    reason: str = ""
    cwd: str | None = None


class StoreError(Exception):
    """The session store is there but could not be read."""


def _word(value: Any) -> str:
    return str(value or "").strip().lower()


def owner_subject() -> str | None:
    """The gateway's subject for this machine's owner, ``uid:<n>``; None off POSIX."""
    getuid = getattr(os, "getuid", None)
    return f"uid:{getuid()}" if getuid is not None else None


def bot_chat(title: Any) -> bool:
    """Whether ``title`` names a Bot Chat."""
    return isinstance(title, str) and _BOT_CHAT.fullmatch(title.strip()) is not None


def gateway_session(kwargs: Mapping[str, Any]) -> bool:
    """Whether the keywords carry the gateway's chat identity."""
    return any(kwargs.get(key) for key in _GATEWAY_IDENTITY)


def from_keywords(kwargs: Mapping[str, Any]) -> Verdict:
    """The verdict ``initialize``'s keywords alone allow (see the module docstring)."""
    for key in _MESSAGING_IDENTITY:
        if kwargs.get(key):
            return Verdict(INERT, f"a messaging identity ({key})")
    agent_context = _word(kwargs.get("agent_context")) or "primary"
    platform = _word(kwargs.get("platform")) or "cli"
    gateway = gateway_session(kwargs)
    if not gateway and CRON in (agent_context, platform):
        return Verdict(CRON)
    if agent_context != "primary":
        return Verdict(INERT, f"agent_context {agent_context!r}, not a primary session")
    if platform not in LOCAL_PLATFORMS:
        return Verdict(INERT, f"platform {platform!r}, not a local surface")
    if bot_chat(kwargs.get("session_title")):
        return Verdict(INERT, "the Bot Chat, which other agents and API peers write into")
    if not gateway:
        return Verdict(ACTIVE)
    chat_id = kwargs.get("chat_id")
    if kwargs.get("chat_type") != "dm" or not (
        isinstance(chat_id, str) and _LOCAL_CHAT.fullmatch(chat_id)
    ):
        return Verdict(INERT, "a gateway chat that is not a local session")
    route = kwargs.get("gateway_session_key")
    match = _LOCAL_ROUTE.fullmatch(route) if isinstance(route, str) else None
    if match is None or match.group(1) != chat_id:
        return Verdict(INERT, "a gateway route that is not a local session")
    owner = owner_subject()
    if owner is None or kwargs.get("user_id") != owner:
        return Verdict(INERT, "a user_id that is not this machine's owner")
    return Verdict(ACTIVE)


# -- the session store ------------------------------------------------------------------


def _connect(home: str) -> sqlite3.Connection | None:
    """A read-only connection to ``<home>/state.db``; None when there is no store."""
    path = Path(home) / "state.db"
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=DB_TIMEOUT)
    except (sqlite3.Error, OSError, ValueError) as err:
        raise StoreError(type(err).__name__) from err


def _meta(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row and isinstance(row[0], str) else None


def _receipt_verdict(receipt: Any, chat_id: str, route: Any) -> Verdict:
    """What a local session's creation receipt allows."""
    if not isinstance(receipt, dict):
        return Verdict(INERT, "no creation record for this local session")
    owner = owner_subject()
    if owner is None or receipt.get("principal_id") != owner:
        return Verdict(INERT, "created by someone other than this machine's owner")
    if receipt.get("session_id") != chat_id or (route and receipt.get("route") != route):
        return Verdict(INERT, "a creation record for another session")
    request_id = receipt.get("request_id")
    if not isinstance(request_id, str) or request_id.startswith(_BOT_CHAT_REQUEST):
        return Verdict(INERT, "the Bot Chat, which other agents and API peers write into")
    policy = receipt.get("policy")
    source = policy.get("source") if isinstance(policy, dict) else None
    if source not in OWNER_SOURCES:
        return Verdict(INERT, f"created as {source!r}, not one of the owner's own chats")
    try:
        request = json.loads(policy.get("request_json") or "{}")
    except (TypeError, ValueError):
        request = {}
    cwd = request.get("cwd") if isinstance(request, dict) else None
    return Verdict(ACTIVE, cwd=cwd if isinstance(cwd, str) and cwd.strip() else None)


def _verdict(conn: sqlite3.Connection, session_id: str, kwargs: Mapping[str, Any]) -> Verdict:
    """The store's verdict for a session. RAISES sqlite3.Error or ValueError."""
    gateway = gateway_session(kwargs)
    row = conn.execute(
        "SELECT title, source, chat_id FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    chats = [str(kwargs.get("chat_id"))] if gateway else []
    row_chat = str(row[2] or "") if row is not None else ""
    if row_chat and row_chat not in chats:
        chats.append(row_chat)
    for key in [session_id, *chats]:
        if conn.execute("SELECT 1 FROM state_meta WHERE key=?", (_API_KEY + key,)).fetchone():
            return Verdict(INERT, "an API server conversation")
    if row is not None:
        if bot_chat(row[0]):
            return Verdict(INERT, "the Bot Chat, which other agents and API peers write into")
        if row[1] == "tool":
            return Verdict(INERT, "created as 'tool', not one of the owner's own chats")
        if row[1] not in ROW_SOURCES:
            return Verdict(INERT, f"a session stored as {row[1]!r}, not a local chat")
        if row_chat and _LOCAL_CHAT.fullmatch(row_chat) is None:
            return Verdict(INERT, "a session of a chat that is not a local one")
    verdict = Verdict(ACTIVE)
    for index, chat in enumerate(chats):
        route = kwargs.get("gateway_session_key") if gateway and index == 0 else None
        found = _receipt_verdict(_meta(conn, _POLICY_KEY + chat), chat, route)
        if found.mode != ACTIVE:
            return found
        if gateway and index == 0:
            verdict = found
    return verdict


def check_session(home: Any, session_id: str, kwargs: Mapping[str, Any]) -> Verdict:
    """The verdict for a session the keywords allow, checked against the session store
    of ``home``. Never raises."""
    gateway = gateway_session(kwargs)
    if not home:
        return (
            Verdict(INERT, "no Hermes home to check the session in") if gateway else Verdict(ACTIVE)
        )
    try:
        conn = _connect(str(home))
    except StoreError as err:
        return Verdict(INERT, f"the session store could not be read ({err})")
    if conn is None:
        return Verdict(INERT, "no session store (state.db)") if gateway else Verdict(ACTIVE)
    try:
        return _verdict(conn, session_id, kwargs)
    except (sqlite3.Error, ValueError) as err:
        return Verdict(INERT, f"the session store could not be read ({type(err).__name__})")
    finally:
        conn.close()


def recheck(home: Any, session_id: str) -> Verdict:
    """The store's verdict for ``session_id`` now, without the keywords (what the row and
    its chat's receipt say); ACTIVE without a store or a row. RAISES StoreError when the
    store is there but could not be read."""
    if not home or not session_id:
        return Verdict(ACTIVE)
    conn = _connect(str(home))
    if conn is None:
        return Verdict(ACTIVE)
    try:
        return _verdict(conn, session_id, {})
    except (sqlite3.Error, ValueError) as err:
        raise StoreError(type(err).__name__) from err
    finally:
        conn.close()
