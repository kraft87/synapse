"""Config layer for the Synapse Claude Code plugin — the host-independence seam.
# mypy: ignore-errors

Every host-specific value resolves from an env var with a sane default, so the same
plugin runs for anyone: clone Synapse, `docker compose up`, install this plugin, set a
couple of env vars, done. Nothing hardcodes a username or a path.

The plugin is a THIN CLIENT: it talks to Synapse over HTTP only (ingest + the /skills
sync/review routes) and wires the recall/remember MCP tools. It needs NO database access —
the dream→skills lane and all Postgres work live server-side. One base URL + an optional
bearer token is the whole surface.

Every setting resolves env var -> ~/.synapse/config.json (synapse_home.py, written by
`synapse login` / `synapse config`) -> the legacy Claude plugin options
(CLAUDE_PLUGIN_OPTION_*, then settings.json) -> default. The resolution is the client
core's (plugin/synapse_client: settings.py, credentials.py); this module hands it the
Claude plugin's own sources and keeps the names the scripts import. Env vars (all
optional):
  CLAUDE_SKILLS_DIR     skills library to maintain   (default ~/.claude/skills)
  CLAUDE_PROJECTS_DIR   transcript root              (default ~/.claude/projects)
  SYNAPSE_DATA_DIR      local state                  (default ~/.local/share/synapse-skills)
  SYNAPSE_HOME          config.json + the installed client (default ~/.synapse)
  SYNAPSE_URL           base URL of the server       (default http://localhost:8765)
  SYNAPSE_INGEST_TOKEN  bearer token. Prefer config.json (`synapse login` writes it
                        there); once this machine is enrolled its device token wins over
                        an env value (synapse_client.credentials)
  SYNAPSE_INGEST_URL    legacy override for /ingest  (else derived from SYNAPSE_URL)
  SYNAPSE_MCP_URL       legacy override for /mcp     (else derived)
  SYNAPSE_SKILLS_SYNC   "1" enables the SessionStart skills sync (default OFF — opt-in)
  SYNAPSE_CONFIG_SYNC   "1" enables config-file mirroring (default OFF — opt-in)
  SYNAPSE_MACHINE_ROLE  "personal" (default) or "work" — the trust this device enrolls at
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import sys
import urllib.request
from pathlib import Path


def _load_home():
    """synapse_home.py from this directory, by path: callers load this module by path
    too (tests, plugin-codex), so this directory need not be on sys.path."""
    mod = sys.modules.get("synapse_home")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "synapse_home", Path(__file__).resolve().with_name("synapse_home.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["synapse_home"] = mod
        spec.loader.exec_module(mod)
    return mod


synapse_home = _load_home()

# The client core, reached the way synapse_project.py reaches it: plugin/ first on
# sys.path, so a different synapse_client installed on the host cannot stand in for the
# one this plugin ships. Every setting below resolves on it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse_client import __version__, context, credentials  # noqa: E402
from synapse_client.http import Transport  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402


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
    if os.environ.get("SYNAPSE_USER_SETTINGS_ONLY"):  # the MCP headers helper
        return out
    proj = Path.cwd() / ".claude"
    out += [proj / "settings.json", proj / "settings.local.json"]
    return out


def _load_plugin_options() -> dict:
    """The legacy install answers: `/plugin install` stored them in settings.json under
    pluginConfigs['synapse@<marketplace>'].options. Still read, below config.json, so an
    install that predates config.json works with no action. Hooks also get them as
    CLAUDE_PLUGIN_OPTION_*, but slash-command scripts and the CLI don't; this closes that
    gap."""
    return synapse_home.plugin_options(_settings_files())


# Read both config sources once at import (cheap; both files are small).
_FILE_OPTIONS = _load_plugin_options()
_HOME = synapse_home.load()


_OPTION_ENV = "CLAUDE_PLUGIN_OPTION_"


def _plugin_sources(*, resolving: bool = False) -> list[tuple[str, dict]]:
    """The legacy Claude plugin options as the core's extra sources, in their order:
    CLAUDE_PLUGIN_OPTION_<KEY> (Claude Code exports them to hooks), then settings.json.
    The source names are the ones the core's token rules know them by.

    ``resolving``: for the URL and the token. A saved value that is falsy JSON (0, false,
    [], {}) has never been a token candidate, nor an address the device token may go
    to, while a URL lookup reads it as its string, as a flag does (a saved 0 turns a
    switch off). Such values go in a source of their own, last, that the token rules
    don't know; the token's is dropped."""
    exported = {
        name[len(_OPTION_ENV) :]: value
        for name, value in os.environ.items()
        if name.startswith(_OPTION_ENV)
    }
    if not resolving:
        return [("plugin_option", exported), ("plugin_config", _FILE_OPTIONS)]
    saved = {k: v for k, v in _FILE_OPTIONS.items() if v or isinstance(v, str)}
    falsy = {k: v for k, v in _FILE_OPTIONS.items() if k not in saved and k != _TOKEN_KEY}
    return [("plugin_option", exported), ("plugin_config", saved), ("settings_falsy", falsy)]


def live_settings() -> Settings:
    """``SETTINGS`` with the sources its flags read as they are now (see ``_cfg``); the
    URL, the token and the enrollment stay as resolved at import. What a hook hands the
    client core."""
    return SETTINGS.with_sources(
        environ=os.environ, config=_HOME, extra_candidates=_plugin_sources()
    )


def _cfg(key: str, default: str = "") -> str:
    # Resolution order, env optional everywhere (Settings.flag): explicit env var, then
    # ~/.synapse/config.json (what `synapse login` / `synapse config` write, read by every
    # harness), then the legacy Claude plugin options: CLAUDE_PLUGIN_OPTION_<KEY> (injected
    # for hooks) and settings.json. The plugin options rank below config.json because
    # Claude Code keeps exporting answers to options the plugin no longer declares, so a
    # stale install-prompt value must not beat what the user set since with `synapse
    # config`. Read live: the environment as it is now, and this process's view of
    # config.json, which set_home_values keeps in step with its own writes.
    return live_settings().flag(key, default)


def set_home_values(changes: dict) -> dict:
    """Write ``changes`` (config.json keys; ``None`` deletes) to ~/.synapse/config.json and
    refresh this process's view, so a value set and then read in the same run (login asks
    the role, then enrolls with it) agrees. Raises OSError."""
    data = synapse_home.update(changes)
    _HOME.clear()
    _HOME.update(data)
    return data


def write_user_config(key: str, value: str) -> None:
    """Persist a plugin userConfig value into the user settings.json, the Claude plugin's own
    store: the MCP entry interpolates `${user_config.<key>}` from here. Used for the server
    URL (the one value the Claude harness must hold itself) and to keep a legacy plaintext
    token slot in step with config.json. Touches only
    pluginConfigs['synapse@<marketplace>'].options[key]; every other setting is preserved."""
    settings = CONFIG_DIR / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    plugin_configs = data.setdefault("pluginConfigs", {})
    pc_key = _plugin_block(data) or "synapse@synapse"
    plugin_configs.setdefault(pc_key, {}).setdefault("options", {})[key] = value
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Refresh the in-process cache so a same-run read (and any later import) sees the new value.
    _FILE_OPTIONS[key] = value


def _plugin_block(data: dict) -> str:
    """Which synapse@<marketplace> options block of a settings.json is the plugin's: the
    enabled one, else the only or first one; "" when there is none. A fork's leftover
    block next to the installed plugin's must not be the one written."""
    configs = data.get("pluginConfigs") if isinstance(data, dict) else None
    keys = [k for k in (configs or {}) if str(k).split("@", 1)[0] == "synapse"]
    enabled = data.get("enabledPlugins") if isinstance(data, dict) else None
    on = [k for k in keys if isinstance(enabled, dict) and enabled.get(k) is True]
    return (on or keys or [""])[0]


def plugin_block_options() -> dict:
    """The options of the plugin's own block in the user settings.json (see
    _plugin_block): where its static header's URL and token come from."""
    try:
        data = json.loads((CONFIG_DIR / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    key = _plugin_block(data) if isinstance(data, dict) else ""
    opts = ((data.get("pluginConfigs") or {}).get(key) or {}).get("options") if key else None
    return dict(opts) if isinstance(opts, dict) else {}


def _drop_user_option(key: str) -> None:
    """Remove ``key`` from every synapse@* options block in the user settings.json; an
    unreadable file is left alone."""
    settings = CONFIG_DIR / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    touched = False
    for cfg_key, cfg in (data.get("pluginConfigs") or {}).items() if isinstance(data, dict) else ():
        opts = cfg.get("options") if isinstance(cfg, dict) else None
        if str(cfg_key).split("@", 1)[0] == "synapse" and isinstance(opts, dict) and key in opts:
            del opts[key]
            touched = True
    if touched:
        settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _FILE_OPTIONS.pop(key, None)


# Legacy: pre-consolidation `synapse login` stashed the token here. Read-only fallback so a
# machine that hasn't re-logged-in since the consolidation keeps working; nothing writes it now.
CREDENTIALS_FILE = DATA_DIR / "credentials.json"


def _cred(key: str) -> str:
    try:
        return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8")).get(key, "") or ""
    except Exception:
        return ""


# Enrollment state for this device (schema 054): which surface row this machine got and
# what it was granted. It lives under "device" in ~/.synapse/config.json, beside the
# credential it describes; its presence is how the hooks know this machine has a
# credential of its OWN rather than one someone pasted in.
#
# The record never holds the token itself. It holds `token_sha256`, the fingerprint of the
# token enrollment stored, and `server`, the origin that minted it, so the resolution
# below can recognise the device token wherever it sits.
#
# Installs before 0.20 kept it in DATA_DIR/device.json. That file is still read when
# config.json has no record, and removed whenever a record is written or cleared, so an
# old record can never resurface over a newer one.
DEVICE_FILE = DATA_DIR / "device.json"


def _legacy_device() -> dict:
    """The pre-0.20 record (DEVICE_FILE), or ``{}``. Never raises."""
    try:
        data = json.loads(DEVICE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_device_state() -> dict:
    """This device's enrollment record, or ``{}``. Never raises."""
    record = synapse_home.load().get("device")
    if isinstance(record, dict) and record:
        return record
    return _legacy_device()


def write_device_state(state: dict) -> None:
    """Persist the enrollment record. Fail-soft: a hook must not die over bookkeeping."""
    try:
        set_home_values({"device": state})
        DEVICE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def clear_device_state() -> None:
    """Forget the enrollment record, wherever it is. Raises OSError, because the callers
    (`--reenroll`, the root-token login) must not go on to store a new token under a
    record that describes the old one."""
    if "device" in synapse_home.load():
        set_home_values({"device": None})
    DEVICE_FILE.unlink(missing_ok=True)


def store_token(token: str, record: dict | None = None, server: str = "") -> None:
    """Store this machine's credential in config.json, with its enrollment record when
    there is one (one atomic write, so a crash can't separate a token from its record).

    A legacy install whose settings.json still carries a plaintext token gets the same
    value there too: that slot feeds the Claude plugin's static Authorization header,
    the fallback when the headers helper can't run, and a stale copy left behind would
    be sent instead. A machine with no such slot never gets one: the token lives in
    config.json only. ``server`` is the token's server, when neither ``record`` nor the
    configured URL says it yet. Raises OSError."""
    changes: dict = {"token": token}
    if record is not None:
        changes["device"] = record
    set_home_values(changes)
    if record is not None:
        DEVICE_FILE.unlink(missing_ok=True)
    opts = plugin_block_options()
    if opts.get(_TOKEN_KEY):
        # Only while the plugin points at this token's server: its static header goes to
        # the plugin's URL, and a move made with --no-install leaves that at the old one.
        server = server or str((record or {}).get("server") or BASE_URL)
        plugin_url = str(opts.get("SYNAPSE_URL") or opts.get("SYNAPSE_INGEST_URL") or "")
        if plugin_url and credentials.origin(plugin_url) == credentials.origin(server):
            write_user_config(_TOKEN_KEY, token)
        else:
            _drop_user_option(_TOKEN_KEY)


#: The fingerprint the enrollment record keeps of the device token (never the token).
token_sha256 = credentials.token_sha256

_TOKEN_KEY = "SYNAPSE_INGEST_TOKEN"
#: Where requests go when no URL is configured anywhere. The core sends nothing then;
#: the Claude hooks have always tried a local server.
DEFAULT_URL = "http://localhost:8765"


def _load_settings(environ) -> Settings:
    return Settings.load(
        environ,
        home=synapse_home.home(),
        extra_candidates=_plugin_sources(resolving=True),
        legacy_device=_legacy_device(),
        legacy_token=_cred(_TOKEN_KEY),
    )


def load_settings() -> Settings:
    """This machine's settings on the client core, from the Claude plugin's sources:
    the environment, config.json, the legacy plugin options (``_plugin_sources``), the
    pre-0.20 device.json and credentials.json.

    The URL is the one base URL (``SYNAPSE_URL``, else the legacy ``SYNAPSE_INGEST_URL``
    with a pasted endpoint suffix stripped), and http://localhost:8765 when neither is
    configured anywhere: resolved as if configured, so the device token's origin rule
    judges the server requests actually go to. The token is the device token on an
    enrolled machine, only toward the server that minted it (synapse_client.credentials).
    Flags read through ``_cfg`` follow the live environment, not this snapshot."""
    settings = _load_settings(os.environ)
    if not (settings.flag("SYNAPSE_URL") or settings.flag("SYNAPSE_INGEST_URL")):
        settings = _load_settings({**os.environ, "SYNAPSE_URL": DEFAULT_URL})
    elif not settings.ingest_url:
        # A URL of only slashes: the core derives no ingest URL from the empty base it
        # leaves, the hooks have always derived "/ingest", and judged the token by it.
        settings = _load_settings({**os.environ, "SYNAPSE_INGEST_URL": "/ingest"})
    return settings.with_sources(environ=os.environ, extra_candidates=_plugin_sources())


SETTINGS = load_settings()
BASE_URL = SETTINGS.base_url
# The legacy per-endpoint keys still win if set (existing installs); else from the base.
INGEST_URL = SETTINGS.ingest_url
MCP_URL = _cfg("SYNAPSE_MCP_URL") or BASE_URL + "/mcp"
SKILLS_URL = BASE_URL + "/skills"


def device_origins() -> set:
    """The (scheme, host) origins this machine's device token may be sent to; empty when
    the machine is not enrolled. The CLI refuses to point anything elsewhere. Read now,
    from the files as they are (the CLI asks after writing them)."""
    return set(_load_settings(os.environ).device_origins)


# Schema 054 changed what this value MEANS over a machine's lifetime without changing
# where it lives. Before enrollment it is whatever credential was pasted or fetched; once
# `synapse login` enrolls, it is a token minted for THIS device, stored in config.json,
# where the MCP headers helper (mcp_headers.py) reads it too. One slot, two lifecycle
# stages: the client never has to manage two credentials. The stages only move forward:
# `synapse login` does not sign in again on an enrolled machine, since that would
# replace the device token (`--reenroll` replaces it on purpose). Once enrolled, the
# device token also wins over an env value, and is never sent to a server that did not
# mint it (synapse_client.credentials.resolve). TOKEN_OVERRIDE is the core's note of
# what it overrode or withheld (source names, never values), for the session-start notice.
INGEST_TOKEN = SETTINGS.token
TOKEN_OVERRIDE = dict(SETTINGS.token_note)


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


#: Why this machine sends no token to a URL, and the two ways out (the core's wording).
withheld_notice = credentials.withheld_notice

# The notice line for a token Claude Code itself loaded: only this harness can say how to
# make recall (its MCP connection) pick up the device token too.
_PLUGIN_OPTION_LINE = (
    "[Synapse] The Synapse token Claude Code loaded for this session is ignored: "
    "this machine is enrolled as {surface}, and hooks use its device token. "
    "Run /reload-plugins (or restart) so recall uses it too."
)
# Claude's own token sources, worded here; "" says nothing (the settings.json value is the
# last candidate, so nothing after it can override it).
_SOURCE_LINES = {"plugin_option": _PLUGIN_OPTION_LINE, "plugin_config": ""}


def token_override_notice() -> str:
    """One line per token source this process overrode, or "" when none was.

    Printed at session start so the override is never silent. Names the source and the
    enrolled surface; never a token value or any part of one. The wording is the core's
    (synapse_client.context.notice_part); this adds where Claude Code keeps an
    environment token (a settings.json "env" block) and its own token line."""
    where = "your shell profile"
    if "env" in (TOKEN_OVERRIDE.get("sources") or ()):
        files = _env_block_files()
        where = " and ".join(f'the "env" block of {_display_path(p)}' for p in files) or where
    return context.notice_part(
        SETTINGS,
        env_location=where,
        config_path=_display_path(synapse_home.config_file()),
        source_lines=_SOURCE_LINES,
    )


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


# What this machine is, as `synapse login` asked it (or a pre-0.20 install prompt did):
# "personal" (default) or "work". It travels once, at enrollment, and it is authoritative
# there — the person answering it has just authenticated to the IdP, so they are the
# authority for what their own machine is.
#
# The question defaults to "personal" because the single-user common case is a machine
# that should see everything, and a default that makes the normal path silently useless
# gets worked around rather than understood. The narrow default lives one layer down
# instead: the SERVER treats an unstated role as restricted, so a client that never
# asked the question cannot resolve it to full access. Human says nothing -> personal;
# software says nothing -> restricted.
def machine_role() -> str:
    """This machine's declared role, read at call time: `synapse login` asks for it and
    enrolls in the same process."""
    return (_cfg("SYNAPSE_MACHINE_ROLE", "personal") or "personal").strip().lower()


MACHINE_ROLE = machine_role()

#: role -> trust level. Anything unrecognised states nothing, and the server's own
#: narrow default (restricted, inherited work projects) applies.
_ROLE_TRUST = {"personal": "full", "work": "restricted"}


def requested_trust() -> str | None:
    """The trust level this machine declares at enrollment, or None if unstated."""
    return _ROLE_TRUST.get(machine_role())


#: Names the client and the harness; Cloudflare in front of a hosted server refuses
#: urllib's default.
USER_AGENT = f"synapse-client/{__version__} claude-code"
_UA = USER_AGENT


def _urlopen(req: urllib.request.Request, timeout: float):
    """Open ``req`` through the client core (synapse_client.http.urlopen), whose redirect
    handler keeps the bearer off a redirect to another origin; urllib's stock one copies
    it wherever the server points. For the scripts that still build their own request
    (synapse_login's sign-in calls); the helpers below send through the core's Transport,
    which opens through the same function."""
    from synapse_client.http import urlopen

    return urlopen(req, timeout=timeout)


def _transport() -> Transport:
    # Built per request from this module's names, so a caller that repoints them (enroll
    # sends without a bearer; a test points BASE_URL at its server) is obeyed. The bearer
    # goes to the origins this machine's URLs name, BASE_URL's, INGEST_URL's and
    # MCP_URL's, and nowhere else, redirects included.
    return Transport(BASE_URL, INGEST_TOKEN, USER_AGENT, token_origins=(INGEST_URL, MCP_URL))


def post_json(path: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST JSON to a Synapse endpoint under BASE_URL and return the parsed JSON reply.

    `path` is endpoint-relative ("/skills/list") or absolute ("http://..."). Sends the bearer
    when one is configured, to this machine's server only. Raises on transport / HTTP error
    — callers decide whether to fail-open (hooks) or surface it (the review CLI)."""
    return _transport().post_json(path, payload, timeout=timeout)


def get_json(path: str, params: dict | None = None, timeout: float = 30.0) -> dict:
    """GET JSON from a Synapse endpoint under BASE_URL and return the parsed reply.

    `path` is endpoint-relative ("/preferences/top") or absolute. `params` are urlencoded
    onto the query string. Sends the bearer when one is configured, to this machine's
    server only. Raises on transport / HTTP error — callers decide whether to fail-open
    (hooks) or surface it."""
    return _transport().get_json(path, params, timeout=timeout)


def request_json(
    method: str, path: str, payload: dict | None = None, timeout: float = 30.0
) -> dict:
    """Send an arbitrary-verb JSON request to a Synapse endpoint under BASE_URL.

    The sibling helpers cover the POST/GET lanes; private mode's routes are PUT/DELETE
    (the session id is the resource, not a body field), which urllib only reaches through
    an explicit method. Same bearer + error contract: raises on transport / HTTP error."""
    return _transport().request_json(method, path, payload, timeout=timeout)


def act(endpoint: str, cid: int, action: str, *, expect: str, **extra) -> dict | None:
    """POST an accept/reject/promote to `<endpoint>/proposals/act` and gate on the expected
    status. Returns the response dict when status == expect; otherwise prints (detail or
    "not found") and returns None. `extra` carries per-action fields (reason, scope, ...)."""
    r = post_json(f"{endpoint}/proposals/act", {"id": cid, "action": action, **extra})
    if r.get("status") != expect:
        print(r.get("detail") or "not found")
        return None
    return r
