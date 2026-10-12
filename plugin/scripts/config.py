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
(CLAUDE_PLUGIN_OPTION_*, then settings.json) -> default. Env vars (all optional):
  CLAUDE_SKILLS_DIR     skills library to maintain   (default ~/.claude/skills)
  CLAUDE_PROJECTS_DIR   transcript root              (default ~/.claude/projects)
  SYNAPSE_DATA_DIR      local state / proposal drafts (default ~/.local/share/synapse-skills)
  SYNAPSE_HOME          config.json + the installed client (default ~/.synapse)
  SYNAPSE_URL           base URL of the server       (default http://localhost:8765)
  SYNAPSE_INGEST_TOKEN  bearer token. Prefer config.json (`synapse login` writes it
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
import importlib.util
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
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


def _cfg(key: str, default: str = "") -> str:
    # Resolution order, env optional everywhere: explicit env var, then ~/.synapse/config.json
    # (what `synapse login` / `synapse config` write, read by every harness), then the legacy
    # Claude plugin options: CLAUDE_PLUGIN_OPTION_<KEY> (injected for hooks) and settings.json.
    # The plugin options rank below config.json because Claude Code keeps exporting answers
    # to options the plugin no longer declares, so a stale install-prompt value must not beat
    # what the user set since with `synapse config`.
    val = os.environ.get(key)
    if val:
        return val
    if synapse_home.has(_HOME, key):
        return synapse_home.option(_HOME, key) or default
    val = os.environ.get(f"CLAUDE_PLUGIN_OPTION_{key}")
    if val:
        return val
    file_val = _FILE_OPTIONS.get(key)
    return str(file_val) if file_val not in (None, "") else default


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


def _base_url() -> str:
    """The single Synapse base URL (scheme://host:port, no path). New `SYNAPSE_URL` wins; falls
    back to the legacy `SYNAPSE_INGEST_URL` with its endpoint suffix stripped (deprecated)."""
    base = _cfg("SYNAPSE_URL") or _cfg("SYNAPSE_INGEST_URL") or "http://localhost:8765"
    base = base.rstrip("/")
    for suffix in ("/ingest", "/recall", "/mcp", "/skills"):  # tolerate a full endpoint pasted in
        # a path segment only: a host that happens to be named "mcp" stays
        if base.endswith(suffix) and "://" in base[: -len(suffix)]:
            base = base[: -len(suffix)]
    return base.rstrip("/")


BASE_URL = _base_url()
# Legacy per-endpoint keys still win if set (existing installs); else derive from the base.
INGEST_URL = _cfg("SYNAPSE_INGEST_URL") or BASE_URL + "/ingest"
RECALL_URL = _cfg("SYNAPSE_RECALL_URL") or BASE_URL + "/recall"
MCP_URL = _cfg("SYNAPSE_MCP_URL") or BASE_URL + "/mcp"
SKILLS_URL = BASE_URL + "/skills"
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


def read_device_state() -> dict:
    """This device's enrollment record, or ``{}``. Never raises."""
    record = synapse_home.load().get("device")
    if isinstance(record, dict) and record:
        return record
    try:
        data = json.loads(DEVICE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
        if plugin_url and _origin(plugin_url) == _origin(server):
            write_user_config(_TOKEN_KEY, token)
        else:
            _drop_user_option(_TOKEN_KEY)


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
        ("config_file", synapse_home.option(_HOME, _TOKEN_KEY)),
        ("plugin_option", os.environ.get(f"CLAUDE_PLUGIN_OPTION_{_TOKEN_KEY}") or ""),
        ("plugin_config", str(_FILE_OPTIONS.get(_TOKEN_KEY) or "")),
    ]


def _plugin_config_origins() -> set:
    """Origins the Claude plugin config points at. Not config.json, whose URL `synapse
    config url` changes."""
    out = set()
    for key in ("SYNAPSE_URL", "SYNAPSE_INGEST_URL"):
        for val in (os.environ.get(f"CLAUDE_PLUGIN_OPTION_{key}"), _FILE_OPTIONS.get(key)):
            if val:
                out.add(_origin(str(val)))
    return out


def _resolve_ingest_token() -> tuple[str, dict]:
    """The bearer every hook and script sends, plus a note of what it overrode.

    Unenrolled machines keep plain precedence (env var, config.json,
    CLAUDE_PLUGIN_OPTION_*, the settings.json plugin config, then the legacy credentials
    file): before enrollment the slot carries whatever token the user put there,
    wherever they put it.

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
    * A record written before the hash existed (only ever a legacy device.json): the
      settings.json plugin-config value is taken as the device token. Enrollment wrote it there, and nothing in the plugin replaces
      it while the record exists (login's root-token write deletes the record first), so
      only a hand edit changes it, and that value is what the MCP header sends anyway.
      Following it can only bring the hooks in line with recall. The record gains its
      hash at the next ``synapse-login --reenroll``; hooks never rewrite it.

    Either way, only toward the server that minted it (the record's ``server``; for a
    record too old to have one, where the plugin config points). If the URL sends the
    hooks somewhere else, an env token is presumably that server's and is sent; the
    device token never goes to an origin that never had it, even when nothing else is
    configured.

    The second value is ``{}`` unless a configured value was overridden; then it names
    the surface and the overridden SOURCES, never values, for the session-start notice.
    """
    candidates = _token_candidates()
    plain = next((v for _, v in candidates if v), "") or _cred(_TOKEN_KEY)
    state = read_device_state()
    surface = state.get("surface_id")
    if not surface:
        return plain, {}
    idx = _device_index(state, candidates)
    if idx < 0:
        return plain, {}
    device = candidates[idx][1]
    if any(
        _origin(url) not in _device_origins(state, device, candidates)
        for url in (BASE_URL, INGEST_URL)
    ):
        # Pointed elsewhere: a token configured for that server (an env value) is still
        # sent; the device token never is. With nothing else to send, say so at session
        # start rather than fail every request in silence.
        other = next((v for _, v in candidates if v and v != device), "")
        cred = _cred(_TOKEN_KEY)
        other = other or (cred if cred != device else "")
        withheld = {"surface_id": surface, "sources": [], "withheld": _base(BASE_URL)}
        return other, ({} if other else {**withheld, "server": str(state.get("server") or "")})
    overridden = [name for name, v in candidates[:idx] if v and v != device]
    return device, ({"surface_id": surface, "sources": overridden} if overridden else {})


def _device_index(state: dict, candidates: list[tuple[str, str]]) -> int:
    """Which candidate is the device token, -1 when none is (see _resolve_ingest_token)."""
    pinned = state.get("token_sha256")
    if pinned:
        return next(
            (i for i, (_, v) in enumerate(candidates) if v and token_sha256(v) == pinned), -1
        )
    return next((i for i, (name, v) in enumerate(candidates) if name == "plugin_config" and v), -1)


def _device_origins(state: dict, device: str, candidates: list[tuple[str, str]]) -> set:
    """Where the device token may go: the server that minted it (the record's ``server``),
    and where the Claude plugin config that holds this very token points, since that
    config's own MCP header sends it there whatever the hooks do (a 0.19 plugin URL
    changed to the same server's new address keeps working). A record too old to name
    its server, with no such plugin config, goes by config.json's URL."""
    out = {_origin(str(state["server"]))} if state.get("server") else set()
    if device and any(
        name in ("plugin_option", "plugin_config") and v == device for name, v in candidates
    ):
        out |= _plugin_config_origins()
    saved = synapse_home.option(_HOME, "SYNAPSE_URL")
    if not out and saved:
        out.add(_origin(str(saved)))
    return out


def device_origins() -> set:
    """The (scheme, host) origins this machine's device token may be sent to; empty when
    the machine is not enrolled. The CLI refuses to point anything elsewhere."""
    state = read_device_state()
    if not state.get("surface_id"):
        return set()
    candidates = _token_candidates()
    idx = _device_index(state, candidates)
    return _device_origins(state, candidates[idx][1] if idx >= 0 else "", candidates)


def _base(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


# Schema 054 changed what this value MEANS over a machine's lifetime without changing
# where it lives. Before enrollment it is whatever credential was pasted or fetched; once
# `synapse login` enrolls, it is a token minted for THIS device, stored in config.json,
# where the MCP headers helper (mcp_headers.py) reads it too. One slot, two lifecycle
# stages: the client never has to manage two credentials. The stages only move forward:
# `synapse login` does not sign in again on an enrolled machine, since that would
# replace the device token (`--reenroll` replaces it on purpose). Once enrolled, the
# device token also wins over an env value (_resolve_ingest_token).
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


def withheld_notice(url: str, server: str) -> str:
    """Why this machine sends no token to ``url``, and the two ways out."""
    enrolled = (
        f"the server this machine enrolled with ({server})"
        if server
        else ("where this machine's device token was enrolled")
    )
    return (
        f"[Synapse] {url} is not {enrolled}, so its device token is not sent there. "
        f"The same server at a new address: `synapse config url {url} --same-server`. "
        f"Another server: `synapse login --url {url} --reenroll`."
    )


def token_override_notice() -> str:
    """One line per token source this process overrode, or "" when none was.

    Printed at session start so the override is never silent. Names the source and the
    enrolled surface; never a token value or any part of one.
    """
    if not TOKEN_OVERRIDE:
        return ""
    surface = TOKEN_OVERRIDE["surface_id"]
    if TOKEN_OVERRIDE.get("withheld"):
        return withheld_notice(TOKEN_OVERRIDE["withheld"], TOKEN_OVERRIDE.get("server") or "")
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
        elif source == "config_file":
            lines.append(
                f"[Synapse] The token in {_display_path(synapse_home.config_file())} is "
                f"ignored: this machine is enrolled as {surface} with a different one. Run "
                "`synapse login --reenroll` to replace the enrollment, or "
                "`synapse config token --unset`."
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
