"""Shared config + HTTP layer for the Codex-side Synapse hooks.

Mirror of plugin/scripts/config.py, adapted for Codex: hooks inherit plain env only, so
resolution is env var, then ~/.synapse/config.json (what `synapse login` writes; the
same file the Claude plugin reads), then the Synapse *Claude Code* plugin's legacy saved
options in ~/.claude/settings.json, then the default. A Codex-only machine therefore
configures with `synapse login` alone. One exception: on an enrolled machine the saved
device token beats an env SYNAPSE_INGEST_TOKEN (see _resolve_token). Dependency-free
(urllib).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# ~/.synapse/config.json is read through the Claude plugin's module (plugin-codex has no
# copy), loaded by path from the sibling checkout like skills_sync's engine.
_HOME_PY = Path(__file__).resolve().parents[2] / "plugin" / "scripts" / "synapse_home.py"


def _load_home() -> Any:
    mod = sys.modules.get("synapse_home")
    if mod is None:
        spec = importlib.util.spec_from_file_location("synapse_home", _HOME_PY)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["synapse_home"] = mod
        spec.loader.exec_module(mod)
    return mod


synapse_home = _load_home()


def _claude_plugin_options() -> dict[str, str]:
    opts = synapse_home.plugin_options([Path(os.path.expanduser("~/.claude/settings.json"))])
    return {k: str(v) for k, v in opts.items()}


_FALLBACK = _claude_plugin_options()
_HOME: dict[str, Any] = synapse_home.load()


def _cfg(key: str, default: str = "") -> str:
    if os.environ.get(key):
        return os.environ[key]
    if synapse_home.has(_HOME, key):  # set in config.json, even if to nothing
        return synapse_home.option(_HOME, key) or default
    return _FALLBACK.get(key) or default


def _base_url() -> str:
    base = _cfg("SYNAPSE_URL") or _cfg("SYNAPSE_INGEST_URL") or "http://localhost:8765"
    base = base.rstrip("/")
    for suffix in ("/ingest", "/recall", "/mcp", "/skills"):
        # a path segment only: a host that happens to be named "mcp" stays
        if base.endswith(suffix) and "://" in base[: -len(suffix)]:
            base = base[: -len(suffix)]
    return base.rstrip("/")


BASE_URL = _base_url()
INGEST_URL = _cfg("SYNAPSE_INGEST_URL") or BASE_URL + "/ingest"

# The enrollment record: "device" in ~/.synapse/config.json, or the pre-0.20 location the
# Claude plugin wrote (plugin/scripts/config.py DEVICE_FILE). It holds the device token's
# sha256 and the server that minted it, never the token.
_DEVICE_FILE = (
    Path(os.path.expanduser(os.environ.get("SYNAPSE_DATA_DIR") or "~/.local/share/synapse-skills"))
    / "device.json"
)


def _device_state() -> dict[str, Any]:
    record = _HOME.get("device")
    if isinstance(record, dict) and record:
        return record
    try:
        data = json.loads(_DEVICE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def _resolve_token() -> tuple[str, str, str]:
    """Mirror of plugin/scripts/config.py ``_resolve_ingest_token`` (read that for the
    reasoning), minus the CLAUDE_PLUGIN_OPTION_* source Codex never sets.

    Plain precedence is env, then config.json, then the Claude plugin's saved token,
    which let a stale SYNAPSE_INGEST_TOKEN in the shell shadow the device token: Codex
    client configs needed an ``env -u SYNAPSE_INGEST_TOKEN`` wrapper to make the saved
    token win. Once this machine is enrolled, the device token wins on its own, but only
    toward a server that already gets it. Returns (token, enrolled surface_id when an
    env value was overridden, else "", the base URL the device token was withheld from
    when nothing else could be sent, else "").
    """
    env = os.environ.get("SYNAPSE_INGEST_TOKEN") or ""
    home = synapse_home.option(_HOME, "SYNAPSE_INGEST_TOKEN")
    saved = _FALLBACK.get("SYNAPSE_INGEST_TOKEN") or ""
    plain = env or home or saved
    state = _device_state()
    surface = str(state.get("surface_id") or "")
    if not surface:
        return plain, "", ""
    pinned = state.get("token_sha256")
    if pinned:
        matches = [
            v for v in (env, home, saved) if v and hashlib.sha256(v.encode()).hexdigest() == pinned
        ]
        device = matches[0] if matches else ""
    else:
        device = saved  # legacy record: the plugin config is where enrollment wrote it
    if not device:
        return plain, "", ""
    # The server that minted it, plus where the Claude plugin config holding this very
    # token points; a record too old to name its server, with neither, goes by
    # config.json's URL (config.py _device_origins).
    server = state.get("server")
    allowed = {_origin(str(server))} if server else set()
    if saved == device:
        allowed |= {
            _origin(v) for k in ("SYNAPSE_URL", "SYNAPSE_INGEST_URL") if (v := _FALLBACK.get(k))
        }
    if not allowed and (url := synapse_home.option(_HOME, "SYNAPSE_URL")):
        allowed.add(_origin(url))
    if any(_origin(url) not in allowed for url in (BASE_URL, INGEST_URL)):
        # Never the device token toward a server that didn't mint it.
        other = next((v for v in (env, home, saved) if v and v != device), "")
        parts = urllib.parse.urlsplit(BASE_URL)
        return other, "", "" if other else f"{parts.scheme}://{parts.netloc}"
    return device, (surface if env and env != device else ""), ""


TOKEN, _OVERRIDDEN_FOR, _WITHHELD_FROM = _resolve_token()


def token_override_notice() -> str:
    """One line when an env token was overridden by the device token, or when the device
    token is not sent at all, else "". Names the surface, never a token value or any
    part of one."""
    if _WITHHELD_FROM:
        server = str(_device_state().get("server") or "")
        enrolled = (
            f"the server this machine enrolled with ({server})"
            if server
            else "where this machine's device token was enrolled"
        )
        return (
            f"[Synapse] {_WITHHELD_FROM} is not {enrolled}, so its device token is not sent "
            f"there. The same server at a new address: `synapse config url {_WITHHELD_FROM} "
            f"--same-server`. Another server: `synapse login --url {_WITHHELD_FROM} --reenroll`."
        )
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
# plugins; enrollment itself is `synapse login`, and this mirror reuses the token it
# writes.
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
