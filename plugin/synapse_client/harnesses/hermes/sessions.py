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
- ``agent_context`` or ``platform`` "cron": a scheduled job (cron/scheduler.py
  ``_construct_cron_agent``). INERT, with no way to turn it on: whoever can have the bot
  schedule a job, from any chat, chooses its prompt and where its output goes
  (tools/cronjob_tools.py).
- ``agent_context`` other than "primary" (a subagent, a memory flush or other fork):
  INERT. A ``platform`` other than "cli", "tui", "desktop", "acp" (every messaging
  platform, "api_server", "bot_room", "a2a", "kanban", "webhook", ...): INERT.
- ``session_title`` "Bot Chat" (also "Bot Chat #2", "Bot Chat (...)"): INERT. The Bot
  Chat is a local session other agents and API-key peers write into
  (gateway/session_bot.py, gateway/platforms/api_server_bot_chat.py). A title that
  starts "Group:": INERT. The Desktop titles a Group Chat member's session
  "Group: <room> · <thread>" (apps/desktop/src/plugins/hermes-bots/group-turns.ts), and
  other bots' messages are submitted into it; Hermes reads the same prefix
  (tui_gateway/agent_factory.py).
- No gateway identity at all: the agent runs in the process that built it (the stdio
  TUI and Desktop backend, tui_gateway/agent_factory.py; ACP; the classic CLI runner).
  ACTIVE, as before, that process belonging to the person who started it, unless the
  process says otherwise: ``HERMES_SESSION_SOURCE`` (``hermes chat --source``,
  hermes_cli/main.py; "kanban" for a Kanban worker, hermes_cli/kanban_db_dispatch.py)
  naming a source outside ROW_SOURCES is INERT; so is a process whose environment
  carries ``HERMES_TURN_AUTHOR``, which a bot-to-bot dispatcher sets on the child that
  runs a delivered message (tools/bot_relay.py ``delivery_env``). That check is a
  backstop no Hermes path reaches today: ``hermes chat`` and both one-shot paths run
  their turn in the gateway daemon (hermes_cli/cli_agent_setup_mixin.py), whose
  environment has the variable stripped (hermes_cli/gateway_runtime_start.py). There a
  dispatched run is known only by its turn's author (``foreign``),
  which Hermes reports after it has built the first turn's system prompt. And so is
  any agent the gateway daemon itself builds without a chat: a ``/background`` or
  ``/btw`` side agent
  (gateway/session_tui_side.py ``_side_agent``) carries no identity and no parent, so
  nothing says whose session it serves. The daemon is recognised the way Hermes
  recognises itself (tools/process_registry.py ``_is_supervised_gateway_process``): its
  ``gateway.lock`` or ``gateway.pid`` record (gateway/status.py ``_build_pid_record``)
  names this process.
- Gateway identity: the turn runs in the gateway daemon, which serves every local
  client and every messaging platform. ACTIVE only for the shape of a local session
  (gateway/session_execution.py ``_build_fresh_agent``; gateway/run_turn_runner.py maps
  ``Platform.LOCAL`` to "cli"/"tui"/"desktop"/"acp"): ``chat_type`` "dm", ``chat_id``
  ``local-<64 hex>`` (hermes_state_keys.py ``local_identity``), ``gateway_session_key``
  ``agent:<profile>:local:dm:<chat_id>``, and ``user_id`` this machine's owner,
  ``uid:<os.getuid()>``. That is the subject gateway/control_socket.py mints for a peer
  of the same OS user (SO_PEERCRED), and also the one every dashboard or Webapp login
  acts as (gateway/owner_identity.py ``principal_subject``): Hermes treats whoever
  signs in to its dashboard as the machine's owner, and so does this rule. POSIX only:
  on Windows the subject is a SID and such a session is INERT.

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
source in OWNER_SOURCES, the same route, and a request that does not follow its
profile. ``tool`` stays INERT: Hermes keeps it out of the owner's own chats too
(gateway/session_policy.py ``DAYBREAK_LABELS``). ``oneshot`` (``hermes chat -q``,
``hermes -z``) stays ACTIVE: it is the owner's own one-shot question, and a script of
the same OS user that sends one could read the Synapse token anyway. A request with
``follow_profile_config`` is a chat the Desktop or the gateway makes for other agents:
the Bot Chat the Desktop mints with a random ``request_id``
(apps/desktop/src/plugins/hermes-bots/canonical-chat.ts, gateway/session_controls.py
``_create``) and every Group Chat member (group-turns.ts; the Desktop turns
``room_plumbing`` into this flag, apps/desktop/src/api/canonical-protocol.ts, so
``room_plumbing`` itself is never stored and is checked only in case it ever is).
INERT. The policy's ``follow_adopted`` mark is NOT read: Hermes also sets it on an
ordinary chat created on the welcome trial's model (gateway/session_local.py
``stage_local_session``). A session with an API binding (``gateway.api.binding.v1.<id>``)
is INERT. A session the store does not know, or a store that cannot be read, is INERT.

The session's own row is checked too, for every session: the session a chat runs need
not be the one it was created with (``/resume`` switches a route to another session,
gateway/slash_commands_session.py; the classic CLI's ``/resume`` and compression switch
the agent's session in place). A row stored under a ``source`` other than the local
ones in ROW_SOURCES (the gateway stores its local chats as "local", "oneshot" or
"tool", gateway/session_policy.py ``STORED_AS_LOCAL``), or on a chat that is not a
local one makes the session INERT; a row on another local chat needs that chat's
receipt to pass as well. So does a row titled as a Bot Chat or "Group:", or whose
``model_config`` carries ``follow_profile_config`` or ``room_plumbing`` (what the
in-process backend stores for those chats, tui_gateway/session_workdir.py; Hermes's
own Bot Chat rule, gateway/session_local_migration.py ``bot_chat_row``), and that holds
for EVERY row of the same chat: a reset (``/new``) or compression continues a chat in
an untitled row with the same ``chat_id`` (hermes_state_local_lineage.py
``reset_local_target``, hermes_state_compression.py ``_publish_child_session_row``),
and Hermes delivers the Bot Chat's traffic to that row (gateway/session_bot.py
``_target``). For an in-process session no store, or no row (a session not written
yet), leaves it ACTIVE; a gateway session always has its row, written with its receipt
(hermes_state_local.py ``commit_local_session``), so one without is INERT.

What the store says can change after the agent is built: a session may be renamed "Bot
Chat" (hermes_state_titles.py; Hermes then routes the Bot Chat's traffic to it without
rebuilding the agent), or switched to another session in place. ``recheck`` reads the
rows again before the system prompt block, each nudge, upload and tool call. An API
binding cannot be added to a local session later: gateway/session_api.py
``bind_api_steps`` refuses a session whose row or live origin is not the API server's.

The sessions a session descends from (lineage.py) are read too. A branch is a new chat
with its own title and a ``branch:`` request id, holding a copy of its parent's turns.
A parent row, any row of a parent's chat, or a parent chat's receipt that names a Bot
Chat, a Group Chat member or a chat that follows its profile makes the session INERT.

A title can also go away (archive, rename), and Hermes builds a new agent for a session
when it likes. So a session judged a Bot Chat, a Group Chat member or a chat that
follows its profile is marked, with the session or chat that says so (marks.py). A mark
on a session, its chats or any session it descends from makes it INERT, and is copied
down to it, so it stays INERT when the marked parent's row is deleted. Marks that
cannot be read make it INERT too.

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

from .lineage import Parent, chain, lineage
from .marks import mark, marked

ACTIVE = "active"
#: A mode no session gets any more: a cron job is INERT. Kept so older imports resolve.
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
#: The process's session source (hermes_cli/main.py ``--source``).
SOURCE_ENV = "HERMES_SESSION_SOURCE"
#: Who wrote a dispatched run's prompt, when it is not the person at the keyboard
#: (agent/turn_author.py ``TURN_AUTHOR_ENV``).
AUTHOR_ENV = "HERMES_TURN_AUTHOR"
#: Largest gateway pid record read, in bytes.
RECORD_LIMIT = 65536

_MESSAGING_IDENTITY = ("user_id_alt", "user_name", "chat_name", "thread_id")
_GATEWAY_IDENTITY = ("gateway_session_key", "chat_id", "chat_type")
_LOCAL_CHAT = re.compile(r"local-[0-9a-f]{64}")
_LOCAL_ROUTE = re.compile(r"agent:[^:\s]+:local:dm:(local-[0-9a-f]{64})")
_BOT_CHAT = re.compile(r"bot chat(\s*#\s*\d+|\s*\(.*\))?", re.IGNORECASE | re.DOTALL)
_GROUP_CHAT = "group:"
_FOLLOWS = ("follow_profile_config", "room_plumbing")
_POLICY_KEY = "gateway.local_policy.v1:"
_API_KEY = "gateway.api.binding.v1."
_BOT_CHAT_REQUEST = "bot-chat:"
_DAEMON_RECORDS = ("gateway.lock", "gateway.pid")
_DAEMON_KIND = "hermes-gateway"
_IS_BOT_CHAT = "the Bot Chat, which other agents and API peers write into"
_IS_GROUP = "a Group Chat member, which other agents write into"
_FOLLOWS_PROFILE = "a chat that follows its profile (the Bot Chat or a Group Chat member)"
_MARKED = "a chat other agents or people have written into (marked when that was seen)"


class Verdict(NamedTuple):
    """``mode`` (ACTIVE or INERT), why it is INERT, a gateway session's cwd, whether the
    reason is that others write into the chat (``foreign``; such a chat is marked), and,
    for an ACTIVE session read from the store, the sessions whose turns it holds, nearest
    first (``continues``, lineage.chain)."""

    mode: str
    reason: str = ""
    cwd: str | None = None
    foreign: bool = False
    continues: tuple[str, ...] = ()


class StoreError(Exception):
    """The session store is there but could not be read."""


def _word(value: Any) -> str:
    return str(value or "").strip().lower()


def owner_subject() -> str | None:
    """The gateway's subject for this machine's owner, ``uid:<n>``; None off POSIX."""
    getuid = getattr(os, "getuid", None)
    return f"uid:{getuid()}" if getuid is not None else None


def foreign(owner_id: str | None, author_id: Any, author_name: Any, is_bot: Any) -> bool:
    """Whether a turn's author is someone other than the owner; ``owner_id`` is the
    ``user_id`` the gateway bound the session to, None in-process. No author: the owner
    at the keyboard (in-process). The gateway names one on every turn, the owner's own
    as ``{"id": <the session's user_id>, "is_bot": False}`` (gateway/run_turn_runner.py,
    ``ctx.source``, the source of ``user_id`` too); a bot, another id, or any author
    where no user was bound is someone else."""
    if is_bot:
        return True
    if author_id is None and author_name is None:
        return False
    return owner_id is None or author_id != owner_id


def bot_chat(title: Any) -> bool:
    """Whether ``title`` names a Bot Chat."""
    return isinstance(title, str) and _BOT_CHAT.fullmatch(title.strip()) is not None


def group_chat(title: Any) -> bool:
    """Whether ``title`` names a Group Chat member's session ("Group: ...")."""
    return isinstance(title, str) and title.strip().lower().startswith(_GROUP_CHAT)


def _title_verdict(title: Any) -> Verdict | None:
    if bot_chat(title):
        return Verdict(INERT, _IS_BOT_CHAT, foreign=True)
    if group_chat(title):
        return Verdict(INERT, _IS_GROUP, foreign=True)
    return None


def gateway_session(kwargs: Mapping[str, Any]) -> bool:
    """Whether the keywords carry the gateway's chat identity."""
    return any(kwargs.get(key) for key in _GATEWAY_IDENTITY)


def _record(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            data = handle.read(RECORD_LIMIT + 1)
        return json.loads(data) if len(data) <= RECORD_LIMIT else None
    except (OSError, ValueError):
        return None


def gateway_daemon(home: Any) -> bool:
    """Whether this process is the Hermes gateway daemon: a ``gateway.lock`` or
    ``gateway.pid`` record under ``home`` or the process's own Hermes home names it."""
    homes = [str(home)] if home else []
    try:
        own = os.environ.get("HERMES_HOME", "").strip() or str(Path.home() / ".hermes")
        homes.append(os.path.expanduser(os.path.expandvars(own)))
    except RuntimeError:  # no home directory to fall back on
        pass
    pid = os.getpid()
    for base in homes:
        for name in _DAEMON_RECORDS:
            record = _record(Path(base) / name)
            if (
                isinstance(record, dict)
                and record.get("kind") == _DAEMON_KIND
                and record.get("pid") == pid
            ):
                return True
    return False


def from_keywords(kwargs: Mapping[str, Any]) -> Verdict:
    """The verdict ``initialize``'s keywords alone allow (see the module docstring)."""
    for key in _MESSAGING_IDENTITY:
        if kwargs.get(key):
            return Verdict(INERT, f"a messaging identity ({key})")
    agent_context = _word(kwargs.get("agent_context")) or "primary"
    platform = _word(kwargs.get("platform")) or "cli"
    gateway = gateway_session(kwargs)
    if CRON in (agent_context, platform):
        return Verdict(INERT, "a scheduled cron job, not a conversation the owner is having")
    if agent_context != "primary":
        return Verdict(INERT, f"agent_context {agent_context!r}, not a primary session")
    if platform not in LOCAL_PLATFORMS:
        return Verdict(INERT, f"platform {platform!r}, not a local surface")
    titled = _title_verdict(kwargs.get("session_title"))
    if titled is not None:
        return titled
    if not gateway:
        source = _word(os.environ.get(SOURCE_ENV))
        if source and source not in ROW_SOURCES:
            return Verdict(INERT, f"started as {source!r}, not one of the owner's own chats")
        if os.environ.get(AUTHOR_ENV, "").strip():
            return Verdict(INERT, "a run that delivers another agent's message")
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
        return Verdict(INERT, _IS_BOT_CHAT, foreign=True)
    policy = receipt.get("policy")
    source = policy.get("source") if isinstance(policy, dict) else None
    if source not in OWNER_SOURCES:
        return Verdict(INERT, f"created as {source!r}, not one of the owner's own chats")
    try:
        request = json.loads(policy.get("request_json") or "{}")
    except (TypeError, ValueError):
        request = None
    if not isinstance(request, dict):
        return Verdict(INERT, "a creation record that could not be read")
    if any(request.get(flag) is True for flag in _FOLLOWS):
        return Verdict(INERT, _FOLLOWS_PROFILE, foreign=True)
    titled = _title_verdict(request.get("title"))
    if titled is not None:
        return titled
    cwd = request.get("cwd")
    return Verdict(ACTIVE, cwd=cwd if isinstance(cwd, str) and cwd.strip() else None)


def _row_verdict(title: Any, model_config: Any) -> Verdict | None:
    """What a session row's title and ``model_config`` rule out, or None."""
    titled = _title_verdict(title)
    if titled is not None:
        return titled
    try:
        config = json.loads(model_config) if isinstance(model_config, str) else None
    except ValueError:
        config = None  # Hermes reads an unparseable model_config as empty too
    if isinstance(config, dict) and any(config.get(flag) for flag in _FOLLOWS):
        return Verdict(INERT, _FOLLOWS_PROFILE, foreign=True)
    return None


def _chat_hits(conn: sqlite3.Connection, chats: list[str]) -> dict[str, Verdict]:
    """What a row of each of ``chats`` rules out, by chat, in one read: every segment of
    a chat (a reset or compressed continuation keeps its ``chat_id``) is held to the
    others' titles."""
    hits: dict[str, Verdict] = {}
    if chats:
        marks = ",".join("?" * len(chats))
        query = f"SELECT chat_id, title, model_config FROM sessions WHERE chat_id IN ({marks})"
        for chat, title, model_config in conn.execute(query, chats):
            found = _row_verdict(title, model_config)
            if found is not None:
                hits.setdefault(str(chat), found)
    return hits


def _parents_verdict(
    conn: sqlite3.Connection, parents: list[Parent], hits: Mapping[str, Verdict]
) -> tuple[Verdict, list[str]] | None:
    """What the sessions a session descends from rule out: a parent row, any row of a
    parent's chat (``hits``), or a parent chat's receipt that names a chat others write
    into. With it come the chat that says so and the parents down to it. A parent's
    receipt is read for that alone; the session's own says who created it."""
    for index, parent in enumerate(parents):
        found = _row_verdict(parent.title, parent.model_config)
        if found is not None:
            return found, [*_ids(parents, index), parent.chat]
    last = {parent.chat: index for index, parent in enumerate(parents) if parent.chat}
    for chat, index in last.items():
        if chat in hits:
            return hits[chat], [*_ids(parents, index), chat]
    for chat, index in last.items():
        receipt = _meta(conn, _POLICY_KEY + chat)
        if isinstance(receipt, dict):
            found = _receipt_verdict(receipt, chat, None)
            if found.foreign:
                return found, [*_ids(parents, index), chat]
    return None


def _ids(parents: list[Parent], index: int) -> list[str]:
    """The parents from the nearest to ``parents[index]``: each holds or continues the
    next one's turns, so what marks that one marks them all."""
    return [parent.id for parent in parents[: index + 1]]


def _judge(
    conn: sqlite3.Connection,
    session_id: str,
    kwargs: Mapping[str, Any],
    row: Any,
    chats: list[str],
    parents: list[Parent],
) -> tuple[Verdict, list[str]]:
    """What the rows and receipts say of a session: its own row, every row of its
    chats, the sessions it descends from, and each chat's creation receipt. With a
    ``foreign`` verdict come the chats and parents it was read from: the session's own
    chat, the one its route answers on, or a parent and the session's own chat."""
    gateway = gateway_session(kwargs)
    own = str(row[2] or "") if row is not None else ""
    for key in [session_id, *chats]:
        if conn.execute("SELECT 1 FROM state_meta WHERE key=?", (_API_KEY + key,)).fetchone():
            return Verdict(INERT, "an API server conversation"), []
    if row is not None:
        found = _row_verdict(row[0], row[3])
        if found is not None:
            return found, [own]
        if row[1] == "tool":
            return Verdict(INERT, "created as 'tool', not one of the owner's own chats"), []
        if row[1] not in ROW_SOURCES:
            return Verdict(INERT, f"a session stored as {row[1]!r}, not a local chat"), []
        if own and _LOCAL_CHAT.fullmatch(own) is None:
            return Verdict(INERT, "a session of a chat that is not a local one"), []
    theirs = [parent.chat for parent in parents if parent.chat]
    hits = _chat_hits(conn, list(dict.fromkeys([*chats, *theirs])))
    for chat in chats:
        if chat in hits:
            return hits[chat], [chat]
    inherited = _parents_verdict(conn, parents, hits)
    if inherited is not None:
        return inherited[0], [own, *inherited[1]]
    verdict = Verdict(ACTIVE)
    for index, chat in enumerate(chats):
        route = kwargs.get("gateway_session_key") if gateway and index == 0 else None
        found = _receipt_verdict(_meta(conn, _POLICY_KEY + chat), chat, route)
        if found.mode != ACTIVE:
            return found, [chat]
        if gateway and index == 0:
            verdict = found
    return verdict, []


def _verdict(
    conn: sqlite3.Connection,
    session_id: str,
    kwargs: Mapping[str, Any],
    *,
    need_row: bool = False,
    home: Any = None,
) -> Verdict:
    """The store's verdict for a session; INERT without its row when ``need_row``. A
    session found to be one others write into is marked under ``home`` with the chat
    that says so and the parents down to it. A mark on the session, its chats or any
    session it descends from makes it INERT, and a parent's mark is copied down to it
    and the parents between. RAISES sqlite3.Error or ValueError, and OSError when the
    marks cannot be read."""
    row = conn.execute(
        "SELECT title, source, chat_id, model_config, parent_session_id FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if row is None and need_row:
        return Verdict(INERT, "a gateway session the store no longer has")
    chats = [str(kwargs.get("chat_id"))] if gateway_session(kwargs) else []
    row_chat = str(row[2] or "") if row is not None else ""
    if row_chat and row_chat not in chats:
        chats.append(row_chat)
    parents = lineage(conn, row[4], row[3]) if row is not None else []
    found, keys = _judge(conn, session_id, kwargs, row, chats, parents)
    if found.foreign:
        mark(home, [session_id, *keys], found.reason)
    elif found.mode == ACTIVE:
        for index, parent in enumerate(parents):
            if marked(home, [parent.id, parent.chat]):
                # Copied down: Hermes forgets a parent when its row is deleted.
                mark(home, [session_id, row_chat, *_ids(parents, index)], _MARKED)
                return Verdict(INERT, _MARKED, foreign=True)
        if marked(home, [session_id, *chats]):
            return Verdict(INERT, _MARKED, foreign=True)
        if row is not None:
            return found._replace(continues=chain(conn, row[4], row[3]))
    return found


def check_session(home: Any, session_id: str, kwargs: Mapping[str, Any]) -> Verdict:
    """The verdict for a session the keywords allow, checked against the session store
    and the marks of ``home``. Never raises."""
    gateway = gateway_session(kwargs)
    if not gateway and gateway_daemon(home):
        return Verdict(INERT, "an agent the gateway daemon built without a chat (a side task)")
    if not home:
        return (
            Verdict(INERT, "no Hermes home to check the session in") if gateway else Verdict(ACTIVE)
        )
    try:
        conn = _connect(str(home))
    except StoreError as err:
        return Verdict(INERT, f"the session store could not be read ({err})")
    try:
        if conn is not None:
            return _verdict(conn, session_id, kwargs, need_row=gateway, home=home)
        if gateway:
            return Verdict(INERT, "no session store (state.db)")
        return _unstored(home, session_id)
    except (sqlite3.Error, ValueError) as err:
        return Verdict(INERT, f"the session store could not be read ({type(err).__name__})")
    except OSError as err:
        return Verdict(INERT, f"the marks could not be read ({type(err).__name__})")
    finally:
        if conn is not None:
            conn.close()


def _unstored(home: Any, session_id: str) -> Verdict:
    """An in-process session with no store to read: its mark is all there is. RAISES
    OSError when that cannot be read."""
    return Verdict(INERT, _MARKED, foreign=True) if marked(home, [session_id]) else Verdict(ACTIVE)


def recheck(
    home: Any,
    session_id: str,
    *,
    gateway: bool = False,
    chat_id: str | None = None,
    route: str | None = None,
) -> Verdict:
    """The store's verdict for ``session_id`` now: what its rows, its chat's rows and
    receipt, its parents and the marks say. A ``gateway`` session passes the ``chat_id``
    and ``route`` it was built for, so the chat it answers on is read again as well as
    the chat of the row it runs. Without a store or a row: ACTIVE for an in-process
    session (unless marked), INERT for a ``gateway`` one. RAISES StoreError when the
    store is there, or the marks are, but could not be read."""
    if not home or not session_id:
        return Verdict(INERT, "no session to check") if gateway else Verdict(ACTIVE)
    conn = _connect(str(home))
    kwargs = {"chat_id": chat_id, "gateway_session_key": route} if gateway and chat_id else {}
    try:
        if conn is not None:
            return _verdict(conn, session_id, kwargs, need_row=gateway, home=home)
        if gateway:
            return Verdict(INERT, "no session store (state.db)")
        return _unstored(home, session_id)
    except (sqlite3.Error, ValueError, OSError) as err:
        raise StoreError(type(err).__name__) from err
    finally:
        if conn is not None:
            conn.close()
