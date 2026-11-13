"""What the model is shown: the session-start block and the per-prompt nudge.

The wording lives here once, for every harness; an adapter only picks the envelope (a
hook's JSON, a provider's return value) and the few words that are its own: how a user
signs in from it (``login_hint``), where an environment token lives for it, and the
names its model sees the memory tools under.

The session-start block is assembled from pieces:

- ``notice_part``: the credential notice (credentials.override_notice), local;
- the upload warning, passed in by the caller (outbox.failure_warning), local;
- ``server_parts``: what the server renders, from ONE request: the board (or the reason
  this machine is served none) and the preferences block.

``session_block`` is all of it in order. A harness with a deadline of its own calls
``server_parts`` on a thread and assembles the pieces that came back in time.

``notice_part``, ``server_parts``, ``session_block`` and ``prompt_nudge`` are FAIL-OPEN:
an unreachable server or a malformed reply yields "" for that piece, never an
exception, because a broken board must not break a session start. The ``*_block``
renderers are plain functions that raise on bad input.
"""

from __future__ import annotations

import time
import types
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from . import credentials

if TYPE_CHECKING:
    # Only named in signatures. A hook that prints the prompt nudge reads flags and
    # sends nothing, so it does not load the request stack (urllib, http, ssl) for it.
    from .api import Client
    from .settings import Settings

DEFAULT_LOGIN_HINT = "run `synapse login` in a terminal"
#: The command that mints a device token on a server with no identity provider (a local
#: `docker compose` install), and where such a token goes on this machine.
BOOTSTRAP_CMD = 'docker compose exec mcp-server synapse-admin bootstrap "<label>"'
PASTE_CMD = "synapse login --token"

_PREFS_MAX_LINES = 7
_MARK: Mapping[Any, str] = types.MappingProxyType(
    {"like": "likes", "dislike": "dislikes", "rule": "rule"}
)
_NUDGE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume it has "
    "context on any past work, project, device, or person they name. BEFORE answering "
    "anything that references the past: {recall} (skip only if the answer is already in "
    "this conversation). About to say 'noted' or 'I'll remember'? Call {remember} FIRST, "
    "then reply."
)
_TIMESTAMP = "[%a %Y-%m-%d %H:%M:%S %Z]"
_FEEDBACK_NUDGE = (
    "[Synapse] recall() returned. If you used any of the served ids (e:/n:/f:/t:/w:) "
    "in your answer, call {recall_feedback}(query=<verbatim query>, helpful=[...], "
    'noise=[...], comment="...") ONCE to close the retrieval-quality loop: '
    "helpful=load-bearing ids, noise=irrelevant ids, comment=free text for anything "
    "else (too much served, buried hit, bad ordering). ONLY IF you needed specific "
    "content memory likely holds and it was not served, add missing=what it was + "
    "found_via=where you got it instead (full_turns/filesystem/user/nowhere...) — "
    "omit both otherwise. Skip the call only if you used none of the results."
)


def _sentence(hint: str) -> str:
    return hint[:1].upper() + hint[1:]


# ---------------------------------------------------------------------------
# Renderers: pure text
# ---------------------------------------------------------------------------


def preferences_block(items: Iterable[Mapping[str, Any]]) -> str:
    """The preferences block for the items of ``/preferences/top``: a header and up to
    seven ``  - (<likes|dislikes|rule>) <pref>`` lines, items without a ``pref``
    skipped; "" when no line remains. Raises on items that are not mappings.

    Only for a server from before ``/context`` carried the block, which renders the same
    lines itself (mcp_server/preferences_routes.render_block)."""
    lines = ["[Synapse preferences]"]
    for item in list(items)[:_PREFS_MAX_LINES]:
        pref = item.get("pref")
        if not pref:
            continue
        polarity = item.get("polarity")
        tag = _MARK.get(polarity, polarity or "")
        lines.append(f"  - ({tag}) {pref}")
    return "\n".join(lines) if len(lines) > 1 else ""


def not_enrolled_block(login_hint: str = DEFAULT_LOGIN_HINT) -> str:
    """Why a machine with no device credential is served nothing (the server said 401),
    and the one thing that fixes it.

    ``login_hint`` is how a user signs in from this harness, as a lowercase imperative
    clause ("run `synapse login` in a terminal"): it opens a sentence here (capitalised)
    and follows a colon in ``restricted_block``."""
    return (
        "[Synapse — this machine is not enrolled]\n"
        "No memory is being served here: since schema 054 every machine authenticates "
        "with its own device token.\n"
        f"{_sentence(login_hint)} and approve the sign-in on any device."
    )


def restricted_block(login_hint: str = DEFAULT_LOGIN_HINT, *, paste_aside: str = "") -> str:
    """Why the server answered but served nothing: this caller is ``restricted`` with no
    device credential of its own (an open server, or the shared root token). The quiet
    twin of ``not_enrolled_block``: same outcome, no error to notice.

    ``paste_aside`` follows "here" in the last line, before its full stop: another place
    this harness takes a pasted token, worded as the harness shows it ("": none)."""
    return (
        "[Synapse — this machine is served nothing]\n"
        "The server answered, but this caller is `restricted` with no projects: it holds no "
        "device credential of its own (schema 054), so the board and recall come back empty.\n"
        "Fix it either way:\n"
        f"  - Server with a sign-in configured: {login_hint}.\n"
        "  - Local server, no sign-in: run\n"
        f"        {BOOTSTRAP_CMD}\n"
        f"    on the server, then run `{PASTE_CMD}` here{paste_aside}."
    )


# ---------------------------------------------------------------------------
# The pieces, each fail-open
# ---------------------------------------------------------------------------


def notice_part(
    settings: Settings,
    *,
    env_location: str = "your shell profile",
    hint: str = "",
    config_path: str = "~/.synapse/config.json",
    source_lines: Mapping[str, str] | None = None,
) -> str:
    """The credential notice for ``settings.token_note`` (see credentials.override_notice
    for ``env_location``, ``hint``, ``config_path`` and ``source_lines``); "" when
    nothing was overridden. Local, no request. Never raises."""
    try:
        return credentials.override_notice(
            settings.token_note,
            env_location=env_location,
            config_path=config_path,
            hint=hint,
            source_lines=source_lines,
        )
    except Exception:
        return ""


def _board(
    reply: Mapping[str, Any], settings: Settings, login_hint: str, restricted: str | None
) -> str:
    """The board part of a ``/context`` reply: its text, with ``restricted_block`` ahead
    of it when the reply says ``restricted`` to an unenrolled machine."""
    if reply.get("status") != "ok":
        return ""
    parts = []
    if reply.get("trust") == "restricted" and not settings.enrolled:
        parts.append(restricted_block(login_hint) if restricted is None else restricted)
    if reply.get("text"):
        parts.append(str(reply["text"]))
    return "\n\n".join(parts)


def _preferences(client: Client, reply: Mapping[str, Any], left: float) -> str:
    """The preferences block of a ``/context`` reply. A server from before the block
    came with the board leaves it out: its top preferences are then read from
    ``/preferences/top`` and rendered here, in the ``left`` seconds of the caller's
    timeout (kept for one release). Raises what that request raises."""
    if "preferences" in reply:
        text = reply["preferences"]
        return text if isinstance(text, str) else ""
    if left <= 0:
        return ""
    return preferences_block(client.preferences(limit=_PREFS_MAX_LINES, timeout=left))


def server_parts(
    client: Client,
    project: str | None,
    settings: Settings,
    *,
    timeout: float = 10,
    login_hint: str = DEFAULT_LOGIN_HINT,
    not_enrolled: str | None = None,
    restricted: str | None = None,
) -> tuple[str, str]:
    """What the server renders for a session start, from ONE request: the board part
    and the preferences block.

    The board part is the board for ``project``, or the reason this machine is served
    none: a 401 for an unenrolled machine becomes ``not_enrolled_block``, and a reply
    that says ``restricted`` to an unenrolled machine gets ``restricted_block`` ahead of
    the board. ``not_enrolled`` and ``restricted``, when given, are those explainers
    already worded by the harness, used in place of the ones rendered from
    ``login_hint``.

    A piece switched off (``SYNAPSE_BOARD``, ``SYNAPSE_PREFS_BLOCK``: "0") is not asked
    for and is ""; with the board off no project is sent, and with both off there is no
    request. ``timeout`` bounds each socket
    wait, and the whole call when a second request is needed (``_preferences``). Never
    raises: an unreachable server, an error status or a malformed reply is "" for both
    pieces."""
    try:
        want_board = settings.flag("SYNAPSE_BOARD", "1") != "0"
        want_prefs = settings.flag("SYNAPSE_PREFS_BLOCK", "1") != "0"
        parts = [name for name, on in (("board", want_board), ("preferences", want_prefs)) if on]
        if not parts:
            return "", ""
        import urllib.error  # with the request stack, which the client has loaded

        started = time.monotonic()
        try:
            # The project is the board's: the preferences are not scoped by one.
            reply = client.context(project if want_board else None, timeout=timeout, parts=parts)
        except urllib.error.HTTPError as err:
            if want_board and err.code == 401 and not settings.enrolled:
                return not_enrolled_block(login_hint) if not_enrolled is None else not_enrolled, ""
            return "", ""
        board = _board(reply, settings, login_hint, restricted) if want_board else ""
    except Exception:
        return "", ""
    prefs = ""
    if want_prefs:
        try:
            prefs = _preferences(client, reply, timeout - (time.monotonic() - started))
        except Exception:
            pass
    return board, prefs


def assemble(parts: Iterable[str | None]) -> str:
    """The non-empty parts, in order, separated by blank lines."""
    return "\n\n".join(part for part in parts if part)


def session_block(
    client: Client,
    project: str | None,
    settings: Settings,
    *,
    timeout: float = 10,
    upload_warning: str = "",
    login_hint: str = DEFAULT_LOGIN_HINT,
    env_location: str = "your shell profile",
    notice_hint: str = "",
) -> str:
    """Everything a session starts with, in order: the credential notice, the upload
    warning, the board, the preferences. "" when there is nothing to say.

    One request (``server_parts``, ``timeout`` as there). The local lines lead, so they
    show even with the server down. Never raises."""
    board, prefs = server_parts(client, project, settings, timeout=timeout, login_hint=login_hint)
    return assemble(
        (
            notice_part(settings, env_location=env_location, hint=notice_hint),
            upload_warning,
            board,
            prefs,
        )
    )


# ---------------------------------------------------------------------------
# Per prompt
# ---------------------------------------------------------------------------


def prompt_nudge(
    settings: Settings,
    *,
    recall: str = "synapse:recall",
    remember: str = "synapse:remember",
    now: float | None = None,
) -> str:
    """The per-prompt context: a wall-clock line (``[Wed 2026-08-26 15:55:01 EDT]``,
    host-local time, so remember() and timeline references know the moment) and the
    directive to recall before answering about the past and to remember before saying
    "noted". ``recall`` and ``remember`` are the tool names as this harness's model sees
    them. ``now`` is epoch seconds (default: the current time).

    Kill switches, each "0" to disable: ``SYNAPSE_PROMPT_TIMESTAMP``,
    ``SYNAPSE_RECALL_NUDGE``. "" when both are off. Makes no request; never raises."""
    lines = []
    try:
        if settings.flag("SYNAPSE_PROMPT_TIMESTAMP", "1") != "0":
            lines.append(time.strftime(_TIMESTAMP, time.localtime(now)))
        if settings.flag("SYNAPSE_RECALL_NUDGE", "1") != "0":
            lines.append(_NUDGE.format(recall=recall, remember=remember))
    except Exception:
        pass
    return "\n".join(lines)


def recall_feedback_nudge(*, recall_feedback: str = "recall_feedback") -> str:
    """The reminder shown after a recall that served rateable ids: close the
    retrieval-quality loop with ``recall_feedback`` (the tool's name as this harness's
    model sees it).

    A developer setting, OFF by default in every harness: the labels only serve whoever
    tunes the retrieval stack, and for everyone else the line is noise after each recall
    plus an extra tool call. A harness shows it only when ``SYNAPSE_RECALL_FEEDBACK_NUDGE``
    is exactly "1"; the hook applies that rule (it reads its own settings). Text only,
    no request; never raises."""
    return _FEEDBACK_NUDGE.format(recall_feedback=recall_feedback)
