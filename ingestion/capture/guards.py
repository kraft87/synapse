"""What server-side capture never stores, whatever the tool returned.

Two guards, both applied before any shape is looked at:

* the tool: Synapse's own tools and sub-agent results are never captured. Browser
  automation (a page may be logged in), connectors for personal data (the editable
  ``rules/deny.toml``) and shell commands (a command's output is whatever it printed) are
  never captured by the generic detector; a capture rule (``ingestion.capture.ruleset``) is
  a deliberate choice and is not held to those;
* the URL: every source URL must be a public http(s) address. Localhost, private,
  loopback and link-local IPs (v4 and v6), single-label hosts and the names a private
  network uses (``.local``, ``.lan``, ``.home``, ``.internal``, ``.ts.net`` ...) are dropped,
  because an unauthenticated crawler never saw those pages: whoever fetched them was
  inside the network, or logged in.

A hostname that merely resolves to a private address cannot be told from a public one
without a lookup, which a guard on stored text must not do.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit

from ingestion.capture.ruleset import active_rules
from ingestion.web_extractors import tool_id

# Names a private network uses. A bare suffix also matches the host that IS the suffix.
_PRIVATE_SUFFIXES = (
    ".local",
    ".localhost",
    ".localdomain",
    ".lan",
    ".home",
    ".home.arpa",
    ".internal",
    ".intranet",
    ".corp",
    ".private",
    ".ts.net",
    ".onion",
    ".test",
    ".invalid",
    ".example",
)

# A public top-level domain starts with a letter ("xn--" included). This also turns away
# numeric hosts that are not IP literals ("0x7f.1", "017700000001").
_TLD = re.compile(r"[a-z][a-z0-9-]+")


def is_public_url(url: str) -> bool:
    """True if ``url`` is an http(s) address on the public internet."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or parts.username or parts.password or not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return _is_public_hostname(host)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def _is_public_hostname(host: str) -> bool:
    if "." not in host or not all(c.isalnum() or c in "-._" for c in host):
        return False
    if any(host.endswith(suffix) or host == suffix[1:] for suffix in _PRIVATE_SUFFIXES):
        return False
    return bool(_TLD.fullmatch(host.rsplit(".", 1)[-1]))


# ---------- Tools ----------

# This server's own tools. The server segment of the name is matched too, so a renamed
# connector for the same server is still caught; the ids below catch a server named
# something unrelated ("recall" and friends are not names other tools use).
_OWN_TOOL_IDS = frozenset({"recall", "recall_full_turns", "recall_feedback", "remember"})
_SUBAGENT_TOOLS = frozenset({"Task", "Agent"})
_BROWSER_WORDS = frozenset({"playwright", "puppeteer", "selenium", "chrome", "chromium", "browser"})
_SHELL_TOOLS = frozenset(
    {
        "bash",
        "bashoutput",
        "powershell",
        "sh",
        "shell",
        "shell_command",
        "local_shell",
        "exec",
        "exec_command",
        "run_command",
        "terminal",
    }
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _words(name: str) -> set[str]:
    """``claude_ai_Gmail`` -> {claude, ai, gmail}; ``WebFetch`` -> {web, fetch}."""
    return set(re.findall(r"[a-z0-9]+", _CAMEL_BOUNDARY.sub(" ", name).lower()))


def own_exclusion(tool_name: str) -> str | None:
    """The skip reason when this tool is never captured, by any rule: ``own_tool`` or
    ``subagent``."""
    tid = tool_id(tool_name)
    server = tool_name.rsplit("__", 1)[0] if "__" in tool_name else ""
    if tid in _OWN_TOOL_IDS or "synapse" in _words(server):
        return "own_tool"
    if tid in _SUBAGENT_TOOLS:
        return "subagent"
    return None


def generic_exclusion(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """The skip reason when the generic detector must not look at this tool's output:
    ``browser``, ``denied_family`` or ``shell``."""
    words = _words(tool_name)
    if words & _BROWSER_WORDS:
        return "browser"
    if words & active_rules().deny_words:
        return "denied_family"
    if tool_id(tool_name).lower() in _SHELL_TOOLS or "command" in tool_input or "cmd" in tool_input:
        return "shell"
    return None
