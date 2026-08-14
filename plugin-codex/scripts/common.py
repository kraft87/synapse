"""Shared config + HTTP layer for the Codex-side Synapse hooks.

Mirror of plugin/scripts/config.py, adapted for Codex: hooks inherit plain
env only, so resolution is env var first, then the Synapse *Claude Code*
plugin's saved options in ~/.claude/settings.json (a machine running both
plugins configures once), then the default. Dependency-free (urllib).
"""

from __future__ import annotations

import json
import os
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
TOKEN = _cfg("SYNAPSE_INGEST_TOKEN")
PRIVATE_DIR = Path(os.path.expanduser(_cfg("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")))

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
