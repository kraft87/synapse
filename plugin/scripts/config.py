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
  SYNAPSE_INGEST_TOKEN  bearer token. Prefer the plugin config (`synapse-login` writes it
                        there); once this machine is enrolled its device token wins over
                        an env value — see _resolve_ingest_token
  SYNAPSE_INGEST_URL    legacy override for /ingest  (else derived from SYNAPSE_URL)
  SYNAPSE_RECALL_URL    legacy override for /recall  (else derived)
  SYNAPSE_MCP_URL       legacy override for /mcp     (else derived)
  SYNAPSE_SKILLS_SYNC   "1" enables the SessionStart skills sync (default OFF — opt-in)
  SYNAPSE_CONFIG_SYNC   "1" enables config-file mirroring (default OFF — opt-in)
  SYNAPSE_MACHINE_ROLE  "personal" (default) or "work" — the trust this device enrolls at
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _force_utf8_stdio() -> None:
    """Hook output is UTF-8 — the server renders arrows and box-drawing into the board.
    Windows Python defaults stdout to the ANSI codepage (cp1252), where a bare `→` raises
    UnicodeEncodeError and the hook dies with a traceback instead of printing its block.
    Every hook script imports this module, so reconfiguring here covers all of them, and
    errors="replace" keeps a still-exotic character from turning into a crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # not a TextIOWrapper (redirected / detached) — nothing to fix


_force_utf8_stdio()


def _path(env: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(env, default)))


SKILLS_DIR = _path("CLAUDE_SKILLS_DIR", "~/.claude/skills")
PROJECTS_DIR = _path("CLAUDE_PROJECTS_DIR", "~/.claude/projects")
DATA_DIR = _path("SYNAPSE_DATA_DIR", "~/.local/share/synapse-skills")
PROPOSALS_DIR = DATA_DIR / "proposals"

# Private mode: one marker file per off-the-record session (name = session id). The Stop
# hook stats this directory before every POST, so it lives outside DATA_DIR on a short,
# obvious path the user can inspect (and `rm`) without knowing plugin internals.
PRIVATE_DIR = _path("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")

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
# Enrollment state for this device (schema 054). Kept in DATA_DIR rather than
# settings.json because it is local bookkeeping, not configuration: which surface row
# this machine got and what it was granted. settings.json holds the credential; this
# holds the story around it, and its presence is how the hooks know this machine has a
# credential of its OWN rather than one someone pasted in.
#
# It never holds the token. Records written now also hold `token_sha256`, the fingerprint
# of the token enrollment wrote into the config slot, and `server`, the origin that
# minted it, so the resolution below can recognise the device token wherever it sits.
DEVICE_FILE = DATA_DIR / "device.json"


def read_device_state() -> dict:
    """This device's enrollment record, or ``{}``. Never raises."""
    try:
        data = json.loads(DEVICE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_device_state(state: dict) -> None:
    """Persist the enrollment record. Fail-soft: a hook must not die over bookkeeping."""
    try:
        DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def token_sha256(token: str) -> str:
    """Fingerprint a credential for the enrollment record. The record has to recognise
    the device token without becoming a second copy of it: data dirs get copied by
    backups and sync tools, and the hash of a random bearer gives nothing back."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url or "")
    return parts.scheme.lower(), parts.netloc.lower()


_TOKEN_KEY = "SYNAPSE_INGEST_TOKEN"


def _token_candidates() -> list[tuple[str, str]]:
    """Every place the token can come from, in plain precedence order, as (source, value).
    Only the source names ever leave this module; the values stay in the bearer header."""
    return [
        ("env", os.environ.get(_TOKEN_KEY) or ""),
        ("plugin_option", os.environ.get(f"CLAUDE_PLUGIN_OPTION_{_TOKEN_KEY}") or ""),
        ("plugin_config", str(_FILE_OPTIONS.get(_TOKEN_KEY) or "")),
    ]


def _plugin_config_origins() -> set:
    """Origins the plugin config itself points at: where the MCP server already sends
    whatever token that config holds."""
    out = set()
    for key in ("SYNAPSE_URL", "SYNAPSE_INGEST_URL"):
        for val in (os.environ.get(f"CLAUDE_PLUGIN_OPTION_{key}"), _FILE_OPTIONS.get(key)):
            if val:
                out.add(_origin(str(val)))
    return out


def _resolve_ingest_token() -> tuple[str, dict]:
    """The bearer every hook and script sends, plus a note of what it overrode.

    Unenrolled machines keep plain precedence (env var, CLAUDE_PLUGIN_OPTION_*, the
    settings.json plugin config, then the legacy credentials file): before enrollment
    the slot carries the root token, wherever the user put it, and enrolling needs it.

    An ENROLLED machine resolves to its device token even when the environment says
    otherwise. Under plain precedence a token left in a shell profile, or in
    settings.json's "env" block (which Claude Code exports to every hook), shadowed the
    device token for ingest, the board, private mode and the sync lanes, while the MCP
    server, which reads only the plugin config, used the device token. Uploads were
    attributed to one credential and recall served by another, with no signal.

    How the device token is recognised:

    * The record has ``token_sha256``: it is whichever configured value hashes to it.
      If none does, the configured token was replaced by hand after enrolling, the
      record no longer describes it, and nothing is overridden.
    * A record written before the hash existed: the plugin-config value is taken
      as the device token. Enrollment wrote it there, and nothing in the plugin replaces
      it while the record exists (login's root-token write deletes the record first), so
      only a hand edit changes it, and that value is what the MCP header sends anyway.
      Following it can only bring the hooks in line with recall. The record gains its
      hash at the next ``synapse-login --reenroll``; hooks never rewrite it.

    Either way, only toward a server that already gets that token: the one that minted
    it, or the one the plugin config points the MCP server at. If SYNAPSE_URL or
    SYNAPSE_INGEST_URL send the hooks somewhere else, the env token is presumably that
    server's, and a device token must not go to an origin that never had it.

    The second value is ``{}`` unless a configured value was overridden; then it names
    the surface and the overridden SOURCES, never values, for the session-start notice.
    """
    candidates = _token_candidates()
    plain = next((v for _, v in candidates if v), "") or _cred(_TOKEN_KEY)
    state = read_device_state()
    surface = state.get("surface_id")
    if not surface:
        return plain, {}
    pinned = state.get("token_sha256")
    if pinned:
        idx = next(
            (i for i, (_, v) in enumerate(candidates) if v and token_sha256(v) == pinned), -1
        )
    else:
        idx = 2 if candidates[2][1] else -1
    if idx < 0:
        return plain, {}
    allowed = _plugin_config_origins()
    if state.get("server"):
        allowed.add(_origin(str(state["server"])))
    if any(_origin(url) not in allowed for url in (BASE_URL, INGEST_URL)):
        return plain, {}
    device = candidates[idx][1]
    overridden = [name for name, v in candidates[:idx] if v and v != device]
    return device, ({"surface_id": surface, "sources": overridden} if overridden else {})


# Schema 054 changed what this value MEANS over a machine's lifetime without changing
# where it lives. At install it is the ENROLLMENT credential — the shared root token,
# pasted or fetched by `synapse login`. On first session, enroll.py trades it for a
# token minted for THIS device and overwrites it here, so plugin.json's
# `Authorization: Bearer ${user_config.SYNAPSE_INGEST_TOKEN}` header keeps working with
# no change on either side. One slot, two lifecycle stages: the client never has to
# manage two credentials, and nothing downstream had to learn a new config key. The
# stages only move forward: `synapse login` refuses to run on an enrolled machine, since
# its root token would overwrite the device token (`--reenroll` replaces it on purpose).
# Once enrolled, the device token also wins over an env value (_resolve_ingest_token).
INGEST_TOKEN, TOKEN_OVERRIDE = _resolve_ingest_token()


def _env_block_files() -> list[Path]:
    """settings.json files whose "env" block sets the token. Claude Code exports those
    values to every hook, so that is where a stale token usually lives."""
    found = []
    for path in _settings_files():
        try:
            env = json.loads(path.read_text(encoding="utf-8")).get("env")
        except Exception:
            continue
        if isinstance(env, dict) and env.get(_TOKEN_KEY):
            found.append(path)
    return found


def _display_path(path: Path) -> str:
    home, text = os.path.expanduser("~"), str(path)
    return "~" + text[len(home) :] if home and text.startswith(home + os.sep) else text


def token_override_notice() -> str:
    """One line per token source this process overrode, or "" when none was.

    Printed at session start so the override is never silent. Names the source and the
    enrolled surface; never a token value or any part of one.
    """
    if not TOKEN_OVERRIDE:
        return ""
    surface = TOKEN_OVERRIDE["surface_id"]
    lines = []
    for source in TOKEN_OVERRIDE["sources"]:
        if source == "env":
            files = _env_block_files()
            where = (
                " and ".join(f'the "env" block of {_display_path(p)}' for p in files)
                or "your shell profile"
            )
            lines.append(
                f"[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this "
                f"machine is enrolled as {surface}. Remove it from {where}."
            )
        elif source == "plugin_option":
            lines.append(
                f"[Synapse] The Synapse token Claude Code loaded for this session is ignored: "
                f"this machine is enrolled as {surface}, and hooks use its device token. "
                "Run /reload-plugins (or restart) so recall uses it too."
            )
    return "\n".join(lines)


# Skills sync: OFF by default — a hook that writes files into ~/.claude/skills on
# every session start should be opt-in for a public plugin (issue #9). Set
# SYNAPSE_SKILLS_SYNC=1 to enable two-way skill sync.
SKILLS_SYNC = _cfg("SYNAPSE_SKILLS_SYNC", "0") not in ("", "0")

# Config lane: OFF by default — mirroring your personal CLAUDE.md + rules/*.md to the server is
# opt-in (set SYNAPSE_CONFIG_SYNC=1). When enabled, config_sync auto-discovers the well-known
# config files under ~/.claude and the current project's .claude. CONFIG_PATHS adds EXTRA globs
# (relative to CONFIG_DIR) beyond the auto set. Surface = this machine.
CONFIG_SYNC = _cfg("SYNAPSE_CONFIG_SYNC", "0") != "0"
CONFIG_PATHS = [g for g in re.split(r"[,\s]+", _cfg("SYNAPSE_CONFIG_PATHS", "")) if g]
# This machine's NAME. Since schema 054 it is a display label sent once at enrollment —
# it identifies nothing and grants nothing. The config lane still keys mirrored files on
# it (those are per-machine files, not a trust boundary). Serving trust comes from the
# device token; a hostname is no longer accepted as evidence of anything.
SURFACE = _cfg("SYNAPSE_SURFACE") or socket.gethostname() or "default"

# What this machine is, from the install prompt: "personal" (default) or "work". It
# travels once, at enrollment, and it is authoritative there — the person answering it
# has just authenticated to the IdP, so they are the authority for what their own
# machine is.
#
# The prompt defaults to "personal" because the single-user common case is a machine
# that should see everything, and a default that makes the normal path silently useless
# gets worked around rather than understood. The narrow default lives one layer down
# instead: the SERVER treats an unstated role as restricted, so a client that never
# asked the question cannot resolve it to full access. Human says nothing -> personal;
# software says nothing -> restricted.
MACHINE_ROLE = (_cfg("SYNAPSE_MACHINE_ROLE", "personal") or "personal").strip().lower()

#: role -> trust level. Anything unrecognised states nothing, and the server's own
#: narrow default (restricted, inherited work projects) applies.
_ROLE_TRUST = {"personal": "full", "work": "restricted"}


def requested_trust() -> str | None:
    """The trust level this machine declares at enrollment, or None if unstated."""
    return _ROLE_TRUST.get(MACHINE_ROLE)


_UA = "synapse-plugin/0.9"


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


def request_json(
    method: str, path: str, payload: dict | None = None, timeout: float = 30.0
) -> dict:
    """Send an arbitrary-verb JSON request to a Synapse endpoint under BASE_URL.

    The sibling helpers cover the POST/GET lanes; private mode's routes are PUT/DELETE
    (the session id is the resource, not a body field), which urllib only reaches through
    an explicit method. Same bearer + error contract: raises on transport / HTTP error."""
    url = path if path.startswith("http") else BASE_URL + path
    headers = {"User-Agent": _UA}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def act(endpoint: str, cid: int, action: str, *, expect: str, **extra) -> dict | None:
    """POST an accept/reject/promote to `<endpoint>/proposals/act` and gate on the expected
    status. Returns the response dict when status == expect; otherwise prints (detail or
    "not found") and returns None. `extra` carries per-action fields (reason, scope, ...)."""
    r = post_json(f"{endpoint}/proposals/act", {"id": cid, "action": action, **extra})
    if r.get("status") != expect:
        print(r.get("detail") or "not found")
        return None
    return r


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
