"""Config layer for the Synapse Claude Code plugin — the host-independence seam.
# mypy: ignore-errors

Every host-specific value resolves from an env var with a sane default, so the same
plugin runs for anyone: clone Synapse, `docker compose up`, install this plugin, set a
couple of env vars, done. Nothing hardcodes a username or a path.

The plugin is a THIN CLIENT: it talks to Synapse over HTTP only (ingest + the /skills
sync/review routes) and wires the recall/remember MCP tools. It needs NO database access —
the dream→skills lane and all Postgres work live server-side. One base URL + an optional
bearer token is the whole surface.

Env vars (all optional):
  CLAUDE_SKILLS_DIR     skills library to maintain   (default ~/.claude/skills)
  CLAUDE_PROJECTS_DIR   transcript root              (default ~/.claude/projects)
  SYNAPSE_DATA_DIR      local state / proposal drafts (default ~/.local/share/synapse-skills)
  SYNAPSE_URL           base URL of the server       (default http://localhost:8765)
  SYNAPSE_INGEST_TOKEN  bearer token (auth-gated server; else `synapse-login` fetches it)
  SYNAPSE_INGEST_URL    legacy override for /ingest  (else derived from SYNAPSE_URL)
  SYNAPSE_RECALL_URL    legacy override for /recall  (else derived)
  SYNAPSE_MCP_URL       legacy override for /mcp     (else derived)
  SYNAPSE_SKILLS_SYNC   "0" disables the SessionStart skills sync (default on)
  SYNAPSE_CONFIG_SYNC   "1" enables config-file mirroring (default OFF — opt-in)
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _path(env: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(env, default)))


SKILLS_DIR = _path("CLAUDE_SKILLS_DIR", "~/.claude/skills")
PROJECTS_DIR = _path("CLAUDE_PROJECTS_DIR", "~/.claude/projects")
DATA_DIR = _path("SYNAPSE_DATA_DIR", "~/.local/share/synapse-skills")
PROPOSALS_DIR = DATA_DIR / "proposals"

# Config lane: the root the mirrored config files live under (file_key = path relative to it), and
# the opt-in manifest of globs to mirror (default none -> the lane is off until the user opts in).
CONFIG_DIR = _path("CLAUDE_CONFIG_DIR", "~/.claude")


def _settings_files() -> list[Path]:
    """Claude Code settings.json locations, least-specific first (later wins on merge): the user
    dir (~/.claude) then the current project's .claude. Pure paths — same on Windows/macOS/Linux."""
    out = [CONFIG_DIR / "settings.json", CONFIG_DIR / "settings.local.json"]
    proj = Path.cwd() / ".claude"
    out += [proj / "settings.json", proj / "settings.local.json"]
    return out


def _load_plugin_options() -> dict:
    """The `/plugin install` prompt stores answers in settings.json under
    pluginConfigs['synapse@<marketplace>'].options. Read them here so the scripts work straight
    from the install config — no environment variables required. The plugin's hooks get these
    injected as CLAUDE_PLUGIN_OPTION_*, but slash-command scripts don't; this closes that gap."""
    merged: dict = {}
    for path in _settings_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # missing / unreadable / malformed -> just skip, fail-soft
        for cfg_key, cfg in (data.get("pluginConfigs") or {}).items():
            if cfg_key.split("@", 1)[0] == "synapse":  # synapse@<any-marketplace>
                opts = cfg.get("options") or {}
                merged.update({k: v for k, v in opts.items() if v not in (None, "")})
    return merged


# Read the install-time options once at import (cheap; settings.json is small).
_FILE_OPTIONS = _load_plugin_options()


def _cfg(key: str, default: str = "") -> str:
    # Resolution order, env optional everywhere: explicit env var, then the plugin userConfig form
    # (CLAUDE_PLUGIN_OPTION_<KEY>, injected for hooks), then the install prompt's value persisted in
    # settings.json, then the default. A new user fills the install prompt and needs no env vars.
    val = os.environ.get(key) or os.environ.get(f"CLAUDE_PLUGIN_OPTION_{key}")
    if val:
        return val
    file_val = _FILE_OPTIONS.get(key)
    return str(file_val) if file_val not in (None, "") else default


def write_user_config(key: str, value: str) -> None:
    """Persist a plugin userConfig value into the user settings.json — the ONE place both
    consumers read: the MCP server interpolates `${user_config.<key>}` from here, and the hooks
    read it via _load_plugin_options(). `synapse login` calls this, so the recall/remember MCP
    server authenticates with no manual paste. Touches only
    pluginConfigs['synapse@<marketplace>'].options[key]; every other setting is preserved."""
    settings = CONFIG_DIR / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    plugin_configs = data.setdefault("pluginConfigs", {})
    # Reuse an existing synapse@<marketplace> entry if present, else default to synapse@synapse.
    pc_key = next(
        (k for k in plugin_configs if str(k).split("@", 1)[0] == "synapse"), "synapse@synapse"
    )
    plugin_configs.setdefault(pc_key, {}).setdefault("options", {})[key] = value
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Refresh the in-process cache so a same-run read (and any later import) sees the new value.
    _FILE_OPTIONS[key] = value


# Legacy: pre-consolidation `synapse login` stashed the token here. Read-only fallback so a
# machine that hasn't re-logged-in since the consolidation keeps working; nothing writes it now.
CREDENTIALS_FILE = DATA_DIR / "credentials.json"


def _cred(key: str) -> str:
    try:
        return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8")).get(key, "") or ""
    except Exception:
        return ""


def _base_url() -> str:
    """The single Synapse base URL (scheme://host:port, no path). New `SYNAPSE_URL` wins; falls
    back to the legacy `SYNAPSE_INGEST_URL` with its endpoint suffix stripped (deprecated)."""
    base = _cfg("SYNAPSE_URL") or _cfg("SYNAPSE_INGEST_URL") or "http://localhost:8765"
    base = base.rstrip("/")
    for suffix in ("/ingest", "/recall", "/mcp", "/skills"):  # tolerate a full endpoint pasted in
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


BASE_URL = _base_url()
# Legacy per-endpoint keys still win if set (existing installs); else derive from the base.
INGEST_URL = _cfg("SYNAPSE_INGEST_URL") or BASE_URL + "/ingest"
RECALL_URL = _cfg("SYNAPSE_RECALL_URL") or BASE_URL + "/recall"
MCP_URL = _cfg("SYNAPSE_MCP_URL") or BASE_URL + "/mcp"
SKILLS_URL = BASE_URL + "/skills"
# env / userConfig wins; else a token fetched by `synapse login`.
INGEST_TOKEN = _cfg("SYNAPSE_INGEST_TOKEN") or _cred("SYNAPSE_INGEST_TOKEN")

# Skills sync: ON by default (two-way skill sync is a stated plugin feature); set
# SYNAPSE_SKILLS_SYNC=0 to disable.
SKILLS_SYNC = _cfg("SYNAPSE_SKILLS_SYNC", "1") != "0"

# Config lane: OFF by default — mirroring your personal CLAUDE.md + rules/*.md to the server is
# opt-in (set SYNAPSE_CONFIG_SYNC=1). When enabled, config_sync auto-discovers the well-known
# config files under ~/.claude and the current project's .claude. CONFIG_PATHS adds EXTRA globs
# (relative to CONFIG_DIR) beyond the auto set. Surface = this machine.
CONFIG_SYNC = _cfg("SYNAPSE_CONFIG_SYNC", "0") != "0"
CONFIG_PATHS = [g for g in re.split(r"[,\s]+", _cfg("SYNAPSE_CONFIG_PATHS", "")) if g]
SURFACE = _cfg("SYNAPSE_SURFACE") or socket.gethostname() or "default"

_UA = "synapse-plugin/0.8"


def post_json(path: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST JSON to a Synapse endpoint under BASE_URL and return the parsed JSON reply.

    `path` is endpoint-relative ("/skills/list") or absolute ("http://..."). Sends the bearer
    when one is configured. Raises on transport / HTTP error — callers decide whether to
    fail-open (hooks) or surface it (the review CLI)."""
    url = path if path.startswith("http") else BASE_URL + path
    headers = {"User-Agent": _UA, "Content-Type": "application/json"}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST", headers=headers
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def get_json(path: str, params: dict | None = None, timeout: float = 30.0) -> dict:
    """GET JSON from a Synapse endpoint under BASE_URL and return the parsed reply.

    `path` is endpoint-relative ("/preferences/top") or absolute. `params` are urlencoded
    onto the query string. Sends the bearer when one is configured. Raises on transport /
    HTTP error — callers decide whether to fail-open (hooks) or surface it."""
    url = path if path.startswith("http") else BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": _UA}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
