"""What the model is shown: the session-start block and the per-prompt nudge.

The wording lives here once, for every harness; an adapter only picks the envelope (a
hook's JSON, a provider's return value) and the few words that are its own: how a user
signs in from it (``login_hint``), where an environment token lives for it, and the
names its model sees the memory tools under.

The session-start block is assembled from pieces, each exposed on its own:

- ``notice_part``: the credential notice (credentials.override_notice), local;
- the upload warning, passed in by the caller (outbox.failure_warning), local;
- ``board_part``: the server-rendered board, or the reason this machine is served none;
- ``preferences_part``: the top preferences.

Every ``*_part``, ``session_block`` and ``prompt_nudge`` is FAIL-OPEN: an unreachable
server or a malformed reply yields "" for that piece, never an exception, because a
broken board must not break a session start. ``board_parts`` and the ``*_block``
renderers are plain functions that raise on bad input; the parts catch around them.
"""

from __future__ import annotations

import time
import types
import urllib.error
from collections.abc import Iterable, Mapping
from typing import Any

from . import credentials
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
    """The preferences block for the server's top items: a header and up to seven
    ``  - (<likes|dislikes|rule>) <pref>`` lines, items without a ``pref`` skipped; ""
    when no line remains. Raises on items that are not mappings."""
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
# The board
# ---------------------------------------------------------------------------


def board_parts(
    client: Client,
    project: str | None,
    settings: Settings,
    timeout: float = 10,
    *,
    login_hint: str = DEFAULT_LOGIN_HINT,
    not_enrolled: str | None = None,
    restricted: str | None = None,
) -> list[str]:
    """The board for ``project``, plus the enrollment explainer when one applies.

    A 401 for an unenrolled machine becomes ``not_enrolled_block``; any other HTTP error
    status yields nothing. A 200 that says ``restricted`` to an unenrolled machine gets
    ``restricted_block`` ahead of the board text. ``not_enrolled`` and ``restricted``,
    when given, are those explainers already worded by the harness, used as they are in
    place of the ones rendered from ``login_hint``. RAISES what the client raises for an
    unreachable server or a malformed reply (``board_part`` catches)."""
    parts: list[str] = []
    try:
        reply = client.context(project, timeout=timeout)
    except urllib.error.HTTPError as err:
        if err.code == 401 and not settings.enrolled:
            parts.append(not_enrolled_block(login_hint) if not_enrolled is None else not_enrolled)
        return parts
    ok = reply.get("status") == "ok"
    if ok and reply.get("trust") == "restricted" and not settings.enrolled:
        parts.append(restricted_block(login_hint) if restricted is None else restricted)
    text = reply.get("text") if ok else None
    if text:
        parts.append(str(text))
    return parts


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


def board_part(
    client: Client,
    project: str | None,
    settings: Settings,
    *,
    timeout: float = 10,
    login_hint: str = DEFAULT_LOGIN_HINT,
    not_enrolled: str | None = None,
    restricted: str | None = None,
) -> str:
    """``board_parts`` joined by blank lines (``login_hint``, ``not_enrolled`` and
    ``restricted`` as there); "" when ``SYNAPSE_BOARD`` is "0" or anything fails. Never
    raises."""
    if settings.flag("SYNAPSE_BOARD", "1") == "0":
        return ""
    try:
        parts = board_parts(
            client,
            project,
            settings,
            timeout,
            login_hint=login_hint,
            not_enrolled=not_enrolled,
            restricted=restricted,
        )
        return "\n\n".join(parts)
    except Exception:
        return ""


def preferences_part(client: Client, settings: Settings, *, timeout: float = 10) -> str:
    """``preferences_block`` for the server's top eight items; "" when
    ``SYNAPSE_PREFS_BLOCK`` is "0" or anything fails. Never raises."""
    if settings.flag("SYNAPSE_PREFS_BLOCK", "1") == "0":
        return ""
    try:
        return preferences_block(client.preferences(limit=8, timeout=timeout))
    except Exception:
        return ""


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

    ``timeout`` bounds each of the two requests. The local lines lead, so they show even
    with the server down. Never raises."""
    return assemble(
        (
            notice_part(settings, env_location=env_location, hint=notice_hint),
            upload_warning,
            board_part(client, project, settings, timeout=timeout, login_hint=login_hint),
            preferences_part(client, settings, timeout=timeout),
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
