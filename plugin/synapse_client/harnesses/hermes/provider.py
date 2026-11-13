"""Synapse as a Hermes memory provider.

Hermes loads one external memory provider per agent from ``$HERMES_HOME/plugins/``
(``synapse install hermes`` puts this one there, see install.py). The provider shows the
model the session-start board, a short nudge each turn and the Synapse tools, and
uploads every finished turn to ``/ingest`` in the ``messages`` format.

Where it is active
------------------
Synapse is one person's memory, so the provider only works for a session that person
runs on this machine. ``initialize`` decides, from Hermes's keywords and, for a session
they allow, Hermes's own record of who created it (sessions.py has the rule and the
Hermes lines it rests on):

- ACTIVE: a primary session on a local surface ("cli", "tui", "desktop", "acp") of the
  machine's owner. ``hermes chat`` and the TUI run their turns in the gateway daemon,
  which passes a chat identity even for them: such a session is ACTIVE only with the
  shape of a local session (``local-<hex>`` chat, its ``agent:<profile>:local:dm:``
  route, ``user_id`` ``uid:<this OS user>``) and a creation receipt in
  ``<hermes_home>/state.db`` that names the owner as its creator, an owner's chat as its
  source (not ``--source tool``), no Bot Chat and no chat that follows its profile. A
  session with no chat identity runs in the process that built it (the stdio TUI and
  Desktop backend, ACP, the classic CLI) and is ACTIVE as before, unless that process
  is the gateway daemon or was started with a ``--source`` that is not the owner's own.
  The board, the nudge, uploads and the tools all work.
- INERT, everything else: every messaging platform, the API server, hosted rooms and
  peer agents, the Bot Chat (by title, by its receipt, or by any row of its chat), a
  Group Chat member, a cron job, a subagent, a memory flush or other fork, a side agent
  the daemon builds (``/background``, ``/btw``), a session created by someone else or
  by a third-party integration, one the store does not know, and any shape not listed
  above. Nothing is injected or uploaded and a tool call returns a JSON error: someone
  else must not read the owner's memory through the agent, nor have their messages
  stored as the owner's. The reason is logged once per session.

The session's rows in the store (title, source, chat, and every row of that chat) are
read again before the system prompt block, each nudge, upload and tool call: a session
renamed "Bot Chat" receives other agents' messages, and ``/resume`` or compression
switches the agent to another session, without a new instance. A turn another agent or
person wrote is never uploaded either: Hermes names a turn's author on ``on_turn_start``
and in ``sync_turn``'s ``turn_author``. The gateway names one on every turn, the owner's
own as the session's ``user_id`` and not a bot; in-process, a human turn has none. An
author that is a bot, or not the session's user, means someone else wrote it. One that
no longer passes turns the instance INERT for good.
What is already in the system prompt (the board) cannot be taken back: Hermes stores
the prompt and restores it verbatim for a continuing session, a renamed, branched or
handed-off one included.

The settings (URL, token) are read at each ``initialize``, so a later ``synapse login``
reaches the next session. Environment variables are the process's own: in the gateway
daemon, those it was started with.

Uploads
-------
Hermes calls ``sync_turn`` on its background worker after each turn with the whole
conversation. messages.py turns it into records and the core uploads them
(upload.TurnUploader): every complete turn not sent yet goes up, with the turn before
it as context; a resumed session looks back at most ten turns the first time. Each
batch is written to the outbox under ``<hermes_home>/synapse/outbox`` before its request
and removed once delivered, so a batch in flight when the process exits waits there;
one that fails stays and counts as handled. The next successful upload drains the
outbox, which every session of the home shares; no drain takes a batch in flight until
its request has had three times its timeout. Turns are remembered in memory only: the
server dedups on span id, so a restart that re-sends a turn costs a request, not a
duplicate. A private session sends nothing and remembers nothing, as the Claude Code
hook does: the core's uploader checks, here and again when the outbox drains.

Compression
-----------
Recall leaves out the session's own turns, which the model still holds. When Hermes
compresses a session in place it keeps the id and replaces the turns before that point
with a summary, so from then on they are memory like any other: at the commit
(``on_session_switch`` with ``reason="compression"`` and the same id) the provider takes
the time, and sends it as ``self_since`` on recall and recall_full_turns, so only the turns
since are left out. The time is the client's wall clock, the one Hermes stamps each
message's ``timestamp`` with and so the time the server stores the turn under. A
compression that moves to a new id needs no cut (the new id holds only what follows); a
new id, a reset or a restart forgets the cut, and the session is left out whole again.
Hermes signals an in-place compression even when its commit failed; that costs only
some recall of turns still in the context.

Every hook catches what it raises and logs it; nothing here raises into Hermes. Hermes
builds one instance per agent (plugins/memory ``register`` runs per load), so each
session of the gateway daemon has its own: an instance holds its own settings, client,
outbox and sent turns, its own state is guarded by one lock, and every thread it starts
goes through Hermes's ``spawn_context_thread``. Sessions share only the outbox folder,
which is safe across threads and processes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider, spawn_context_thread

from ... import __version__, context, project, tools
from ...api import Client
from ...outbox import Outbox, failure_warning
from ...settings import Settings
from ...upload import TurnUploader, Uploader
from . import messages as hm
from . import sessions as hs

ACTIVE = "active"
#: No session is CRON any more (a cron job is INERT); kept so older imports resolve.
CRON = "cron"
INERT = "inert"

#: The session-start requests run side by side and share this budget, in seconds.
START_BUDGET = 5.0
#: Per-request timeouts: a turn upload on Hermes's background worker (which counts a
#: call past 30 s as stuck and skips the provider's next syncs), and the last uploads
#: before Hermes compresses or ends a session.
SYNC_TIMEOUT = 20.0
FLUSH_TIMEOUT = 10.0
#: How many complete turns a resumed session sends at most on its first upload.
LOOKBACK = 10

_LOCAL_ONLY = "Synapse is enabled for local sessions only (the Hermes CLI, TUI, Desktop or ACP)."
_LOGIN = "Synapse is not signed in on this machine: run `synapse login` in a terminal."
_ENV_LOCATION = "Hermes's .env file or your shell profile"


def _log() -> logging.Logger:
    return logging.getLogger(__name__)


def _say(line: str) -> None:
    _log().info("%s", line)


def _clock() -> str:
    """Now on the client's wall clock (time.time, as Hermes stamps a message), ISO-8601."""
    return datetime.fromtimestamp(time.time(), tz=UTC).isoformat()


def scope(kwargs: Mapping[str, Any]) -> str:
    """ACTIVE or INERT for ``initialize``'s keywords alone (sessions.from_keywords);
    ``initialize`` also checks an ACTIVE session against Hermes's session store. CRON is
    no longer one of the answers: a cron job is INERT."""
    return hs.from_keywords(kwargs).mode


def _json_result(text: str) -> str:
    """Hermes wants a JSON string from a tool: a JSON object passes through, any other
    text is wrapped as ``{"result": text}``, the shape Hermes's own providers use."""
    try:
        if isinstance(json.loads(text), dict):
            return text
    except (TypeError, ValueError):
        pass
    return json.dumps({"result": text}, ensure_ascii=False)


def _error(message: str, tool: str = "") -> str:
    body = {"error": message, "tool": tool} if tool else {"error": message}
    return json.dumps(body, ensure_ascii=False)


class SynapseMemoryProvider(MemoryProvider):
    """Synapse for one Hermes agent. Hermes builds a new instance for every agent it
    loads, then calls ``initialize`` once."""

    pre_compress_checkpoint_api_version = 1
    #: Whether ``initialize`` got the gateway's chat identity; such a session must keep
    #: its row in the store.
    _gateway = False
    #: The ``user_id`` the gateway bound the session to: the owner's own turns carry it
    #: as their author. None in-process, where a human turn has no author.
    _owner_id: str | None = None
    #: The chat and route the gateway built the session for; read again at each recheck.
    _chat_id: str | None = None
    _route: str | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings: Settings | None = None
        self._client: Client | None = None
        self._outbox: Outbox | None = None
        self._mode: str | None = None  # None until initialize
        self._session_id = ""
        self._project: str | None = None
        self._cwd: str | None = None
        self._model = ""
        self._block = ""
        self._uploader: Uploader | None = None
        self._turns: TurnUploader | None = None
        self._home: str | None = None
        self._said: set[str] = set()
        self._cut: str | None = None  # the last in-place compression, for self_since

    # -- identity and availability -------------------------------------------------

    @property
    def name(self) -> str:
        return "synapse"

    def is_available(self) -> bool:
        """Whether a server URL and a token are configured. Reads config, no request."""
        try:
            settings = Settings.load()
            return bool(settings.base_url and settings.token)
        except Exception:
            _log().warning("synapse: could not read the settings", exc_info=True)
            return False

    def unavailable_reason(self) -> str:
        return _LOGIN

    def get_config_schema(self) -> list[dict[str, Any]]:
        return []

    # -- lifecycle -----------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Bind this instance to its session, profile and folder; for an active session,
        fetch the session-start block (at most START_BUDGET seconds) and start draining
        the outbox in the background. Never raises."""
        try:
            self._initialize(session_id, kwargs)
        except Exception:
            self._mode = INERT  # undecided is not active: no tools for an unknown session
            _log().warning("synapse: initialize failed", exc_info=True)

    def _initialize(self, session_id: str, kwargs: Mapping[str, Any]) -> None:
        verdict = hs.from_keywords(kwargs)
        home = kwargs.get("hermes_home")
        if verdict.mode == ACTIVE:
            verdict = hs.check_session(home, str(session_id or ""), kwargs)
        elif verdict.foreign:
            # The keywords carried the title, so the store check, which marks, never ran.
            hs.mark(home, [session_id, kwargs.get("chat_id")], verdict.reason)
        mode = verdict.mode
        settings = Settings.load()
        client = Client(settings, f"synapse-client/{__version__} hermes")
        outbox = Outbox(Path(str(home)) / "synapse" / "outbox", log=_say) if home else None
        uploader = Uploader(client, outbox, log=_say)
        lookback = LOOKBACK if kwargs.get("resumed") else None
        turns = TurnUploader(uploader, client="hermes", source="hermes-sync", lookback=lookback)
        cwd = kwargs.get("cwd")
        cwd = cwd if isinstance(cwd, str) and cwd else verdict.cwd
        self._gateway = hs.gateway_session(kwargs)
        user_id, chat_id, route = (
            kwargs.get(key) if self._gateway and isinstance(kwargs.get(key), str) else None
            for key in ("user_id", "chat_id", "gateway_session_key")
        )
        self._owner_id, self._chat_id, self._route = user_id, chat_id, route
        with self._lock:
            self._mode = mode
            self._settings, self._client, self._outbox = settings, client, outbox
            self._uploader, self._turns = uploader, turns
            self._session_id = str(session_id or "")
            self._cwd = cwd
            self._project = project.project_label(cwd) if cwd else None
            self._block = ""
            self._home = str(home) if home else None
            self._cut = None
        if mode == INERT:
            self._say_once(f"synapse: off for session {self._session_id}: {verdict.reason}")
        if mode != ACTIVE:
            return
        self._block = self._session_start(settings, client, outbox)
        if outbox is not None:
            spawn_context_thread(self._drain, name="synapse-outbox", args=(SYNC_TIMEOUT,)).start()

    def _session_start(self, settings: Settings, client: Client, outbox: Outbox | None) -> str:
        """The credential notice and upload warning (local), then the board and the
        preferences, fetched side by side; a piece not back within START_BUDGET is
        left out."""
        label = self._project
        jobs: dict[str, Callable[[], str]] = {
            "board": lambda: context.board_part(client, label, settings, timeout=START_BUDGET),
            "preferences": lambda: context.preferences_part(client, settings, timeout=START_BUDGET),
        }
        got: dict[str, str] = {}
        done = threading.Lock()

        def run(key: str) -> None:
            text = jobs[key]()
            with done:
                got[key] = text

        threads = [spawn_context_thread(run, name=f"synapse-{key}", args=(key,)) for key in jobs]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + START_BUDGET
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        with done:
            board, prefs = got.get("board", ""), got.get("preferences", "")
        warning = failure_warning(outbox) if outbox is not None else None
        return context.assemble(
            (
                context.notice_part(settings, env_location=_ENV_LOCATION),
                warning,
                board,
                prefs,
            )
        )

    def system_prompt_block(self) -> str:
        """The session-start block, for a session still the owner's (``_still_owner``).
        Hermes stores the prompt and restores it verbatim for a continuing session, so a
        block it already has stays there whatever this returns later."""
        if self._mode != ACTIVE or not self._still_owner():
            return ""
        return self._block

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """The per-turn nudge (a timestamp and the recall/remember directive). No request."""
        if self._mode != ACTIVE or self._settings is None:
            return ""
        if not self._still_owner():
            return ""
        return context.prompt_nudge(self._settings, recall="recall", remember="remember")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def recall_status(self) -> None:
        return None

    def _say_once(self, line: str) -> None:
        with self._lock:
            if line in self._said:
                return
            self._said.add(line)
        _say(line)

    def _still_owner(self, session_id: str = "") -> bool:
        """For an ACTIVE session, whether it is still the owner's: the session store is
        read again (sessions.recheck), since a session renamed "Bot Chat" starts receiving
        other agents' messages, and a switch moves the agent to another session, without
        a new instance. A session that fails turns this instance INERT for good; a store
        that cannot be read skips this one call. Never raises."""
        if self._mode != ACTIVE:
            return False
        sid = str(session_id or self._session_id)
        try:
            verdict = hs.recheck(
                self._home, sid, gateway=self._gateway, chat_id=self._chat_id, route=self._route
            )
        except Exception:
            self._say_once(
                "synapse: the session store or its marks could not be read; skipped this call"
            )
            return False
        if verdict.mode == ACTIVE:
            return True
        with self._lock:
            self._mode = INERT
        self._say_once(f"synapse: off for session {sid}: {verdict.reason}")
        return False

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Note the turn's model. A turn someone else wrote (``_foreign``, from the
        ``author_*`` keywords agent/turn_context.py passes) turns the instance INERT
        before its nudge and tools, as ``sync_turn`` does with ``turn_author``."""
        model = kwargs.get("model")
        if isinstance(model, str):
            self._model = model
        author = kwargs.get("author_id"), kwargs.get("author_name"), kwargs.get("author_is_bot")
        if self._foreign(*author):
            self._authored()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Rebind to ``new_session_id``. A new id or a reset forgets which turns were
        sent; the next upload then looks back at most LOOKBACK turns, as for a resume.
        It forgets the compression cut too; a compression that keeps the id sets it."""
        try:
            with self._lock:
                new = str(new_session_id or "")
                if reset or (new and new != self._session_id):
                    if self._turns is not None:
                        self._turns.forget(LOOKBACK)
                    self._cut = None
                elif kwargs.get("reason") == "compression":
                    self._cut = _clock()
                if new:
                    self._session_id = new
        except Exception:
            _log().warning("synapse: on_session_switch failed", exc_info=True)

    # -- uploads -------------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        turn_author: dict[str, Any] | None = None,
    ) -> None:
        """Upload every complete turn not sent yet. Without ``messages``, the two strings
        are the turn. A ``turn_author`` naming someone other than the owner (``_foreign``;
        a bot or peer delivery names its sender, tools/bot_relay.py) means nothing is
        uploaded and the instance turns INERT for good. Never raises."""
        try:
            author = turn_author if isinstance(turn_author, Mapping) else {"id": turn_author}
            if self._foreign(author.get("id"), author.get("name"), author.get("is_bot")):
                self._authored()
                return
            if messages is None:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            if self._upload(messages, session_id, SYNC_TIMEOUT):
                self._drain(SYNC_TIMEOUT)
        except Exception:
            _log().warning("synapse: sync_turn failed", exc_info=True)

    def _foreign(self, author_id: Any, author_name: Any, is_bot: Any) -> bool:
        """Whether a turn's author is someone other than the owner. No author: the owner
        at the keyboard (in-process). The gateway names one on every turn, the owner's own
        as ``{"id": <the session's user_id>, "is_bot": False}``
        (gateway/run_turn_runner.py, ``ctx.source``, the source of ``user_id`` too); a bot,
        another id, or any author where no user was bound is someone else."""
        if is_bot:
            return True
        if author_id is None and author_name is None:
            return False
        return self._owner_id is None or author_id != self._owner_id

    def _authored(self) -> None:
        """A turn someone else wrote reached this agent: off for good, said once, and the
        session and its chat are marked so an agent built for them later is off too."""
        reason = "a turn another agent or person wrote"
        with self._lock:
            self._mode = INERT
            sid = self._session_id
        hs.mark(self._home, [sid, self._chat_id], reason)
        self._say_once(f"synapse: off for session {sid}: {reason}")

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        try:
            self._upload(messages, "", FLUSH_TIMEOUT)
        except Exception:
            _log().warning("synapse: on_session_end failed", exc_info=True)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Upload what is about to be compressed away; adds nothing to the summary."""
        try:
            self._upload(messages, "", FLUSH_TIMEOUT)
        except Exception:
            _log().warning("synapse: on_pre_compress failed", exc_info=True)
        return ""

    def _upload(self, messages: Any, session_id: str, timeout: float) -> bool:
        """Send the complete turns of ``messages`` not sent yet (upload.TurnUploader).
        True when at least one batch reached the server."""
        turns = self._turns
        if self._mode != ACTIVE or turns is None:
            return False
        if not self._still_owner(session_id):
            return False
        return turns.upload(
            hm.to_records(messages),
            str(session_id or self._session_id),
            timeout=timeout,
            project=self._project,
            cwd=self._cwd,
            model=self._model or None,
        )

    def _drain(self, timeout: float) -> None:
        uploader = self._uploader
        if uploader is None:
            return
        try:
            uploader.drain(timeout)
        except Exception:
            _log().warning("synapse: outbox drain failed", exc_info=True)

    # -- tools ---------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        try:
            return tools.schemas()
        except Exception:
            _log().warning("synapse: the tool schemas could not be read", exc_info=True)
            return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        """Call ``tool_name`` on the server with this session's id, and on recall and
        recall_full_turns the compression cut; always a JSON string. Only for an ACTIVE
        session still the owner's: an instance not initialized yet has no session to
        judge, and Hermes itself refuses a tool call before ``initialize``
        (agent/memory_manager.py ``handle_tool_call``, ``caller_scope``)."""
        try:
            client = self._client
            if self._mode != ACTIVE or client is None or not self._still_owner():
                return _error(_LOCAL_ONLY, tool_name)
            with self._lock:
                session_id, cut = self._session_id or None, self._cut
            text = tools.call(client, tool_name, args, session_id=session_id, since=cut)
            return _json_result(text)
        except Exception as err:
            _log().warning("synapse: tool call %s failed", tool_name, exc_info=True)
            return _error(f"Synapse could not answer: {type(err).__name__}", tool_name)
