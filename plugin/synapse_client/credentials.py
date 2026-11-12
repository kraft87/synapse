"""Which token is sent where: the credential rules as pure functions.

A machine keeps one credential slot whose meaning changes over its life. Before
enrollment it carries whatever token the user put there, wherever they put it. Once
`synapse login` enrolls the machine it holds a token minted for this device, which the
enrollment record recognises by fingerprint (``token_sha256``) and which only ever goes
to the server that minted it (the record's ``server``).

Everything here takes its inputs as arguments: the token candidates, the enrollment
record, the URLs requests go to, the saved URL and, for the Claude plugin's legacy
config, the origins it points at. Nothing reads the environment or a file.

Candidates are ``(source, value)`` pairs in precedence order. The source names the rules
know are ``env`` (SYNAPSE_INGEST_TOKEN), ``config_file`` (~/.synapse/config.json) and the
Claude plugin's legacy ``plugin_option`` (CLAUDE_PLUGIN_OPTION_*) and ``plugin_config``
(settings.json). Only source names ever leave this module; values stay in the bearer.

A port of plugin/scripts/config.py (``_resolve_ingest_token`` and its helpers); the
equivalence test in tests/test_client_core_credentials.py keeps the two in step until
the hooks move onto this module.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Origin = tuple[str, str]
Candidate = tuple[str, str]

#: Sources that belong to the Claude plugin's own config: a device token held there is
#: also sent by that config's MCP header, wherever it points.
PLUGIN_SOURCES = ("plugin_option", "plugin_config")


def token_sha256(token: str) -> str:
    """Fingerprint a credential for the enrollment record. The record has to recognise
    the device token without becoming a second copy of it, and the hash of a random
    bearer gives nothing back."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def origin(url: str) -> Origin:
    """``(scheme, host[:port])`` of ``url``, lowercased; ``("", "")`` for an empty one."""
    parts = urllib.parse.urlsplit(url or "")
    return parts.scheme.lower(), parts.netloc.lower()


def base(url: str) -> str:
    """``scheme://host[:port]`` of ``url``, as written."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def device_index(record: Mapping[str, Any], candidates: Sequence[Candidate]) -> int:
    """Which candidate is the device token; -1 when none is.

    A record with ``token_sha256`` names whichever configured value hashes to it; none
    does when the token was replaced by hand after enrolling. A record written before
    the hash existed (only ever a legacy device.json) takes the ``plugin_config`` value:
    enrollment wrote it there and nothing in the plugin replaces it while the record
    exists."""
    pinned = record.get("token_sha256")
    if pinned:
        return next(
            (
                i
                for i, (_, value) in enumerate(candidates)
                if value and token_sha256(value) == pinned
            ),
            -1,
        )
    return next(
        (i for i, (name, value) in enumerate(candidates) if name == "plugin_config" and value), -1
    )


def device_origins(
    record: Mapping[str, Any],
    device: str,
    candidates: Sequence[Candidate],
    *,
    saved_url: str = "",
    plugin_origins: Iterable[Origin] = (),
) -> set[Origin]:
    """Where the device token may go.

    The server that minted it (the record's ``server``), plus the origins the Claude
    plugin config points at when that config holds this very token (its own MCP header
    sends it there whatever the hooks do). A record too old to name its server, with no
    such plugin config, goes by the saved URL (config.json's)."""
    out = {origin(str(record["server"]))} if record.get("server") else set()
    if device and any(name in PLUGIN_SOURCES and value == device for name, value in candidates):
        out |= set(plugin_origins)
    if not out and saved_url:
        out.add(origin(saved_url))
    return out


def enrolled_origins(
    record: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    saved_url: str = "",
    plugin_origins: Iterable[Origin] = (),
) -> set[Origin]:
    """The origins this machine's device token may be sent to; empty when the machine is
    not enrolled. The CLI refuses to point anything elsewhere."""
    if not record.get("surface_id"):
        return set()
    idx = device_index(record, candidates)
    device = candidates[idx][1] if idx >= 0 else ""
    return device_origins(
        record, device, candidates, saved_url=saved_url, plugin_origins=plugin_origins
    )


def resolve(
    candidates: Sequence[Candidate],
    record: Mapping[str, Any],
    urls: Sequence[str],
    *,
    saved_url: str = "",
    plugin_origins: Iterable[Origin] = (),
    legacy_token: str = "",
) -> tuple[str, dict[str, Any]]:
    """The bearer to send, plus a note of what it overrode.

    ``urls`` are the URLs requests go to, the base URL first (a withheld note names it).
    Empty entries are skipped; with none left there is nowhere to send a token, and the
    origin rule has nothing to check. ``legacy_token`` is the pre-consolidation
    credentials file's value, the last fallback.

    An unenrolled machine (no ``surface_id``) keeps plain precedence: the first
    non-empty candidate, else ``legacy_token``. An enrolled machine resolves to its
    device token even when an earlier source says otherwise, but only while every URL
    is at an origin the token may go to (``device_origins``). Pointed elsewhere, a token
    configured for that server (any other value) is sent instead, and the device token
    never is.

    The note is ``{}`` unless something needs saying: ``{"surface_id", "sources"}`` when
    configured values were overridden (source names, never values), or ``{"surface_id",
    "sources": [], "withheld": <base>, "server": <record server>}`` when the device token
    was withheld and nothing else could be sent. Never raises."""
    plain = next((value for _, value in candidates if value), "") or legacy_token
    surface = record.get("surface_id")
    if not surface:
        return plain, {}
    idx = device_index(record, candidates)
    if idx < 0:
        return plain, {}
    device = candidates[idx][1]
    allowed = device_origins(
        record, device, candidates, saved_url=saved_url, plugin_origins=plugin_origins
    )
    targets = [url for url in urls if url]
    if any(origin(url) not in allowed for url in targets):
        other = next((value for _, value in candidates if value and value != device), "")
        other = other or (legacy_token if legacy_token != device else "")
        if other:
            return other, {}
        server = str(record.get("server") or "")
        return "", {
            "surface_id": surface,
            "sources": [],
            "withheld": base(targets[0]),
            "server": server,
        }
    overridden = [name for name, value in candidates[:idx] if value and value != device]
    return device, ({"surface_id": surface, "sources": overridden} if overridden else {})


def withheld_notice(url: str, server: str) -> str:
    """Why this machine sends no token to ``url``, and the two ways out."""
    enrolled = (
        f"the server this machine enrolled with ({server})"
        if server
        else "where this machine's device token was enrolled"
    )
    return (
        f"[Synapse] {url} is not {enrolled}, so its device token is not sent there. "
        f"The same server at a new address: `synapse config url {url} --same-server`. "
        f"Another server: `synapse login --url {url} --reenroll`."
    )


def override_notice(
    note: Mapping[str, Any],
    *,
    env_location: str = "your shell profile",
    config_path: str = "~/.synapse/config.json",
    hint: str = "",
) -> str:
    """One line per token source the note says was overridden, or the withheld line, or
    "" when the note is empty. Names the source and the enrolled surface, never a token
    value or any part of one.

    The harness supplies what only it knows: ``env_location``, where an environment
    token most likely lives for it (Hermes: its .env file); ``hint``, appended to the
    line about a source the harness itself loaded (Claude Code: reload the plugins).
    """
    if not note:
        return ""
    if note.get("withheld"):
        return withheld_notice(str(note["withheld"]), str(note.get("server") or ""))
    surface = note.get("surface_id")
    lines = []
    for source in note.get("sources") or ():
        if source == "env":
            lines.append(
                f"[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this "
                f"machine is enrolled as {surface}. Remove it from {env_location}."
            )
        elif source == "config_file":
            lines.append(
                f"[Synapse] The token in {config_path} is ignored: this machine is enrolled "
                f"as {surface} with a different one. Run `synapse login --reenroll` to "
                "replace the enrollment, or `synapse config token --unset`."
            )
        else:
            line = (
                f"[Synapse] The Synapse token from {source} is ignored: this machine is "
                f"enrolled as {surface} and uses its device token."
            )
            lines.append(f"{line} {hint}" if hint else line)
    return "\n".join(lines)
