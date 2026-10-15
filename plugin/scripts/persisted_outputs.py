"""Inline the files Claude Code saves large tool results to, for the records a hook ships.

A tool result too large for the context is written to a file and the transcript keeps only a
redirect ("Output too large (N KB). Full output saved to: <path>"). The server cannot read
client files, so web capture would skip the call (`persisted_output`). Before a batch of
records is posted (Stop hook, catch-up sweep, history import, `synapse capture`), the
redirect of a result the server could capture is replaced by the file's text.

Only results of calls that look like web research are inlined (the input has a URL or a
query, and the tool is not a browser, a personal-data connector, a shell or Synapse itself),
only from files under the Claude projects directory, never larger than MAX_RESULT_BYTES, and
no more than MAX_POST_BYTES in one POST. Whatever cannot be inlined keeps its redirect. An
old server ignores the difference: it stores the record's text as it always did.

The Stop hook, the sweep and the history import call it only for a server whose last reply said
web capture is on (ingest_hook): a server that ignores the files would otherwise be sent
megabytes it drops on every POST. `synapse capture` always does, having asked the server.

Stdlib only, like every hook. Codex needs nothing like this: its rollouts carry tool output
inline, truncated by Codex itself.
"""

from __future__ import annotations

import copy
import json
import os
import re
import stat
from typing import Any

#: Constants, not settings. The server skips a result over 2,000,000 characters anyway.
MAX_RESULT_BYTES = 1_000_000
MAX_POST_BYTES = 2_000_000  # the hook's 30 s POST timeout must hold on a slow uplink

# The server's reading of a redirect (ingestion.capture.detect._PERSISTED_REDIRECT) and the
# path it names. The path may contain spaces (a Windows profile name): it runs to the
# extension.
_REDIRECT = re.compile(r"<persisted-output>|Output too large|Error: result \(", re.I)
_PATH = re.compile(
    r"saved to:?\s*((?:[A-Za-z]:[\\/]|/)[^\n\"'<>]*?\.(?:json|txt))(?=[\s.)]|$)", re.I
)

# Mirrors of what the server never captures (ingestion/capture/shapes.py and guards.py,
# rules/deny.toml). The server applies its own, possibly edited, lists; these only keep
# output it would refuse off the wire. tests/test_persisted_outputs.py keeps them in step.
_RESEARCH_KEYS = (
    "url", "urls", "uri", "link", "links", "href",
    "query", "queries", "q", "objective", "search_query", "search_term", "keywords",
)  # fmt: skip
_NEVER_WORDS = frozenset(
    {
        "synapse", "playwright", "puppeteer", "selenium", "chrome", "chromium", "browser",
        "gmail", "mail", "email", "outlook", "calendar", "drive", "docs", "sheets", "slack",
        "discord", "teams", "linear", "jira", "github", "gitlab", "bitbucket", "notion",
        "confluence", "asana", "trello", "dropbox", "onedrive", "sharepoint", "telegram",
        "whatsapp",
    }
)  # fmt: skip
_SHELL_KEYS = ("command", "cmd")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _CAMEL_BOUNDARY.sub(" ", name).lower()))


def _looks_like_research(name: str, tool_input: Any) -> bool:
    if not isinstance(tool_input, dict) or _words(name) & _NEVER_WORDS:
        return False
    if any(k in tool_input for k in _SHELL_KEYS):
        return False
    return any(tool_input.get(k) for k in _RESEARCH_KEYS)


def _blocks(rec: dict[str, Any]) -> list[Any]:
    content = (rec.get("message") or {}).get("content")
    return content if isinstance(content, list) else []


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return _text(block.get("text") or block.get("content"))
    return block if isinstance(block, str) else ""


def _read(path: str, root: str) -> str | None:
    """The text of the persisted file, or None: it is missing, outside ``root``, not a
    regular file, or too large."""
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
        if os.path.commonpath([real, base]) != base:
            return None
        info = os.stat(real)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESULT_BYTES:
            return None
        with open(real, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except (OSError, ValueError):  # ValueError: paths on different drives
        return None
    if real.lower().endswith(".json"):
        try:
            data = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(data, list):  # an array of text blocks
            return "\n".join(t for item in data if (t := _block_text(item)))
    return raw


def _inline(records: list[dict[str, Any]], root: str) -> list[dict[str, Any]]:
    uses: dict[str, dict[str, Any]] = {}
    for rec in records:
        for block in _blocks(rec):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses[str(block.get("id"))] = block
    budget = MAX_POST_BYTES
    out = []
    for rec in records:
        new = rec
        for i, block in enumerate(_blocks(rec)):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            use = uses.get(str(block.get("tool_use_id")))
            text = _text(block.get("content"))
            if block.get("is_error") or use is None or not _REDIRECT.search(text[:500]):
                continue
            match = _PATH.search(text[:2000])
            if not match or not _looks_like_research(str(use.get("name") or ""), use.get("input")):
                continue
            body = _read(match.group(1), root)
            if body is None or len(body.encode()) > budget:
                continue
            budget -= len(body.encode())
            if new is rec:
                new = copy.deepcopy(rec)  # the caller's records stay as they were read
            _blocks(new)[i]["content"] = body
        out.append(new)
    return out


def inline(records: list[dict[str, Any]], projects_dir: Any) -> list[dict[str, Any]]:
    """``records`` with the persisted outputs of capturable calls read into their results.
    Never raises: a record that cannot be handled ships as it is."""
    try:
        return _inline(records, os.fspath(projects_dir))
    except Exception:
        return records
