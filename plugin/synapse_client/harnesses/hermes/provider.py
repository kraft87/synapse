"""Synapse as a Hermes memory provider.

Hermes loads one external memory provider per agent from ``$HERMES_HOME/plugins/``
(``synapse install hermes`` puts this one there, see install.py). The provider shows the
model the session-start board, a short nudge each turn and the Synapse tools, and
uploads every finished turn to ``/ingest`` in the ``messages`` format.

Where it is active
------------------
Synapse is one person's memory, so the provider only works for a session that person
runs on this machine. ``initialize`` decides once, from Hermes's keywords:

- ACTIVE: ``agent_context`` "primary" (Hermes's default when it is blank) on a local
  surface, ``platform`` "cli" (Hermes's default), "tui", "desktop" or "acp", with no
  chat identity. The board, the nudge, uploads and the tools all work.
- CRON (``agent_context`` or ``platform`` "cron"): the tools work, nothing is injected
  and nothing is uploaded. A scheduled job may need the memory; its prompts are not a
  conversation the user had.
- INERT, everything else: a subagent (its parent's turns carry the work), a memory
  flush or other fork, the API server, and every messaging gateway. A chat identity
  (``gateway_session_key``, ``chat_id``, ``chat_name``, ``chat_type``, ``thread_id``,
  ``user_id_alt``, ``user_name``) marks a gateway session even on a platform this list
  does not know. Nothing is injected or uploaded and a tool call returns a JSON error:
  someone else in a group chat must not read the owner's memory through the agent, nor
  have their messages stored as the owner's. ``user_id`` alone does not count: the TUI
  and Desktop pass the owner's own login there.

The Hermes lines this rests on: ``_memory_provider_init_kwargs`` in
agent/agent_init_memory_provider.py (platform defaults to "cli"; agent_context is
``memory_agent_context``, else "cron"/"subagent" by platform, else "primary"; the chat
identity comes from ``_GATEWAY_IDENTITY_PARAMS`` in agent/agent_init_params.py);
gateway/session_execution.py (a gateway passes its platform key and the chat identity);
gateway/platforms/api_server.py ("api_server"); tui_gateway/agent_factory.py ("tui",
"desktop"); acp_adapter/session.py ("acp"); cron/scheduler.py ("cron");
tools/delegate_tool.py (``memory_agent_context="subagent"``).

Uploads
-------
Hermes calls ``sync_turn`` on its background worker after each turn with the whole
conversation. Every complete turn not sent yet goes up (messages.pending), with the
turn before it as context; a resumed session looks back at most ten turns the first
time. A batch that fails goes to the outbox under ``<hermes_home>/synapse/outbox`` and
counts as handled; the next successful upload drains it. Turns are remembered by
``messages.turn_key`` in memory only: the server dedups on span id, so a restart that
re-sends a turn costs a request, not a duplicate. A private session (private.is_private)
sends nothing and remembers nothing, as the Claude Code hook does.

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

Every hook catches what it raises and logs it; nothing here raises into Hermes. Each
instance holds its own settings, client and outbox, and every thread it starts goes
through Hermes's ``spawn_context_thread``.
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

from ... import __version__, context, private, project, tools
from ...api import Client
from ...outbox import Outbox, failure_warning, short_reason
from ...settings import Settings
from . import messages as hm

ACTIVE = "active"
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

_LOCAL_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp"})
_CHAT_IDENTITY = (
    "gateway_session_key",
    "chat_id",
    "chat_name",
    "chat_type",
    "thread_id",
    "user_id_alt",
    "user_name",
)
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
    """ACTIVE, CRON or INERT for ``initialize``'s keywords (see the module docstring)."""
    if any(kwargs.get(key) for key in _CHAT_IDENTITY):
        return INERT
    agent_context = str(kwargs.get("agent_context") or "").strip().lower() or "primary"
    platform = str(kwargs.get("platform") or "").strip().lower() or "cli"
    if agent_context == CRON or platform == CRON:
        return CRON
    if agent_context == "primary" and platform in _LOCAL_PLATFORMS:
        return ACTIVE
    return INERT


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
        self._sent: set[str] = set()
        self._lookback: int | None = None
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
            _log().warning("synapse: initialize failed", exc_info=True)

    def _initialize(self, session_id: str, kwargs: Mapping[str, Any]) -> None:
        mode = scope(kwargs)
        settings = Settings.load()
        client = Client(settings, f"synapse-client/{__version__} hermes")
        home = kwargs.get("hermes_home")
        outbox = Outbox(Path(str(home)) / "synapse" / "outbox", log=_say) if home else None
        cwd = kwargs.get("cwd")
        cwd = cwd if isinstance(cwd, str) and cwd else None
        with self._lock:
            self._mode = mode
            self._settings, self._client, self._outbox = settings, client, outbox
            self._session_id = str(session_id or "")
            self._cwd = cwd
            self._project = project.project_label(cwd) if cwd else None
            self._lookback = LOOKBACK if kwargs.get("resumed") else None
            self._sent.clear()
            self._block = ""
            self._cut = None
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
        return self._block if self._mode == ACTIVE else ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """The per-turn nudge (a timestamp and the recall/remember directive). No request."""
        if self._mode != ACTIVE or self._settings is None:
            return ""
        return context.prompt_nudge(self._settings, recall="recall", remember="remember")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def recall_status(self) -> None:
        return None

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if isinstance(model, str):
            self._model = model

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
                    self._sent.clear()
                    self._lookback = LOOKBACK
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
        are the turn. Never raises."""
        try:
            if messages is None:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            if self._upload(messages, session_id, SYNC_TIMEOUT):
                self._drain(SYNC_TIMEOUT)
        except Exception:
            _log().warning("synapse: sync_turn failed", exc_info=True)

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
        """Send the complete turns of ``messages`` not sent yet. True when at least one
        batch reached the server."""
        client, outbox, settings = self._client, self._outbox, self._settings
        if self._mode != ACTIVE or client is None or settings is None:
            return False
        sid = str(session_id or self._session_id)
        if not sid or private.is_private(settings, sid, _say):
            return False
        all_turns = hm.turns(hm.to_records(messages))
        with self._lock:
            if self._lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if hm.complete(turn)]
                older = done[: max(0, len(done) - self._lookback)]
                self._sent |= {hm.turn_key(turn) for turn in older}
                self._lookback = None
            selected = hm.pending(all_turns, self._sent)
            # Claimed before the request: a concurrent hook must not send them again.
            claimed = {hm.turn_key(turn) for _, turn in selected}
            self._sent |= claimed
        sent_any = failed = False
        reason = ""
        for records, new in hm.batches(selected):
            payload = self._payload(sid, records)
            if not failed:
                try:
                    client.ingest(payload, timeout=timeout)
                    sent_any = True
                    continue
                except Exception as err:
                    failed = True
                    reason = short_reason(err)
                    _say(f"synapse: upload failed ({reason}); queued for retry")
            if not self._queue(outbox, payload, reason):
                with self._lock:
                    self._sent -= {hm.turn_key(turn) for turn in new}
        return sent_any

    def _payload(self, session_id: str, records: list[hm.Record]) -> dict[str, Any]:
        return {
            "format": "messages",
            "client": "hermes",
            "session_id": session_id,
            "project": self._project,
            "cwd": self._cwd,
            "model": self._model or None,
            "source": "hermes-sync",
            "records": records,
        }

    @staticmethod
    def _queue(outbox: Outbox | None, payload: Mapping[str, Any], reason: str) -> bool:
        if outbox is None:
            return False
        try:
            outbox.put(payload, reason=reason)
            return True
        except (OSError, TypeError, ValueError):
            _log().warning("synapse: could not queue a failed upload", exc_info=True)
            return False

    def _drain(self, timeout: float) -> None:
        client, outbox = self._client, self._outbox
        if client is None or outbox is None:
            return
        try:
            outbox.drain(lambda payload: client.ingest(payload, timeout=timeout))
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
        recall_full_turns the compression cut; always a JSON string."""
        try:
            if self._mode == INERT:
                return _error(_LOCAL_ONLY, tool_name)
            client = self._client
            if client is None:
                with self._lock:
                    if self._client is None:
                        settings = Settings.load()
                        self._settings = settings
                        self._client = Client(settings, f"synapse-client/{__version__} hermes")
                    client = self._client
            with self._lock:
                session_id, cut = self._session_id or None, self._cut
            text = tools.call(client, tool_name, args, session_id=session_id, since=cut)
            return _json_result(text)
        except Exception as err:
            _log().warning("synapse: tool call %s failed", tool_name, exc_info=True)
            return _error(f"Synapse could not answer: {type(err).__name__}", tool_name)
