"""Shared config + HTTP layer for the Codex-side Synapse hooks.

Mirror of plugin/scripts/config.py, adapted for Codex: hooks inherit plain
env only, so resolution is env var first, then the Synapse *Claude Code*
plugin's saved options in ~/.claude/settings.json (a machine running both
plugins configures once), then the default. One exception: on a machine the Claude
plugin has enrolled, its saved device token beats an env SYNAPSE_INGEST_TOKEN (see
_resolve_token). Dependency-free (urllib).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _claude_plugin_options() -> dict[str, str]:
    try:
        data = json.loads(
            Path(os.path.expanduser("~/.claude/settings.json")).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    merged: dict[str, str] = {}
    for cfg_key, cfg in (data.get("pluginConfigs") or {}).items():
        if str(cfg_key).split("@", 1)[0] == "synapse":
            opts = cfg.get("options") or {}
            merged.update({k: str(v) for k, v in opts.items() if v not in (None, "")})
    return merged


_FALLBACK = _claude_plugin_options()


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _FALLBACK.get(key) or default


def _base_url() -> str:
    base = _cfg("SYNAPSE_URL") or _cfg("SYNAPSE_INGEST_URL") or "http://localhost:8765"
    base = base.rstrip("/")
    for suffix in ("/ingest", "/recall", "/mcp", "/skills"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


BASE_URL = _base_url()
INGEST_URL = _cfg("SYNAPSE_INGEST_URL") or BASE_URL + "/ingest"

# The Claude Code plugin's enrollment record (plugin/scripts/config.py DEVICE_FILE). It
# holds the device token's sha256 and the server that minted it, never the token.
_DEVICE_FILE = (
    Path(os.path.expanduser(os.environ.get("SYNAPSE_DATA_DIR") or "~/.local/share/synapse-skills"))
    / "device.json"
)


def _device_state() -> dict[str, Any]:
    try:
        data = json.loads(_DEVICE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def _resolve_token() -> tuple[str, str]:
    """Mirror of plugin/scripts/config.py ``_resolve_ingest_token`` (read that for the
    reasoning), minus the CLAUDE_PLUGIN_OPTION_* source Codex never sets.

    Plain precedence is env first, which let a stale SYNAPSE_INGEST_TOKEN in the shell
    shadow the device token the Claude plugin enrolled and saved: Codex client configs
    needed an ``env -u SYNAPSE_INGEST_TOKEN`` wrapper to make the saved token win. Once
    this machine is enrolled, the device token wins on its own, but only toward a server
    that already gets it. Returns (token, enrolled surface_id when an env value was
    overridden, else "").
    """
    env = os.environ.get("SYNAPSE_INGEST_TOKEN") or ""
    saved = _FALLBACK.get("SYNAPSE_INGEST_TOKEN") or ""
    plain = env or saved
    state = _device_state()
    surface = str(state.get("surface_id") or "")
    if not surface:
        return plain, ""
    pinned = state.get("token_sha256")
    if pinned:
        matches = [
            v for v in (env, saved) if v and hashlib.sha256(v.encode()).hexdigest() == pinned
        ]
        device = matches[0] if matches else ""
    else:
        device = saved  # legacy record: the plugin config is where enrollment wrote it
    if not device:
        return plain, ""
    allowed = {_origin(v) for k in ("SYNAPSE_URL", "SYNAPSE_INGEST_URL") if (v := _FALLBACK.get(k))}
    if state.get("server"):
        allowed.add(_origin(str(state["server"])))
    if any(_origin(url) not in allowed for url in (BASE_URL, INGEST_URL)):
        return plain, ""
    return device, (surface if env and env != device else "")


TOKEN, _OVERRIDDEN_FOR = _resolve_token()


def token_override_notice() -> str:
    """One line when an env token was overridden by the device token, else "".
    Names the surface, never a token value or any part of one."""
    if not _OVERRIDDEN_FOR:
        return ""
    return (
        f"[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
        f"enrolled as {_OVERRIDDEN_FOR}. Remove it from your shell profile or wherever "
        "Codex inherits it."
    )


PRIVATE_DIR = Path(os.path.expanduser(_cfg("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")))
# This machine's display NAME. Since schema 054 it grants nothing and identifies
# nothing — trust rides on the device token in SYNAPSE_INGEST_TOKEN. Same env-override +
# hostname fallback as plugin/scripts/config.py so the label matches across both
# plugins; enrollment itself happens in the Claude Code plugin, and this mirror reuses
# the token it writes.
SURFACE = _cfg("SYNAPSE_SURFACE") or socket.gethostname() or "default"

_UA = "synapse-codex-plugin/0.1"


def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    params: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    url = path if path.startswith("http") else BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": _UA}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out: dict[str, Any] = json.loads(r.read() or b"{}")
        return out


def get_json(
    path: str, params: dict[str, Any] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    return _request("GET", path, None, params, timeout)


def post_json(path: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    return _request("POST", path, payload, None, timeout)


def request_json(
    method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    return _request(method, path, payload, None, timeout)
