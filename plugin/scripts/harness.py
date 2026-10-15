#!/usr/bin/env python3
# mypy: ignore-errors
"""Wire Synapse into the coding harnesses on this machine: Claude Code and Codex.

`synapse login` and `synapse install` call this; nothing here asks questions or signs in.
Each harness keeps only the server URL in its own config, because it opens the MCP
connection itself (a headers helper can supply headers, not the URL). Everything else,
the credential included, is read from ~/.synapse/config.json at run time.

Claude Code: the plugin, through its own CLI (`claude plugin marketplace add` +
`claude plugin install … --config SYNAPSE_URL=<url>`), so it is installed and loaded
before the next session starts. Without the CLI, the same intent goes into settings.json
(`extraKnownMarketplaces` + `enabledPlugins`), which Claude Code acts on only after an
interactive session has started, so the first session runs without it.

Codex: [mcp_servers.synapse] and the hook blocks in ~/.codex/config.toml, through
plugin-codex/install.py. `codex mcp add` can't set a headers helper, and on a server with
OAuth it would start a browser sign-in whose token outranks the device credential.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

synapse_home = config.synapse_home

MARKETPLACE_REPO = "kraft87/synapse"
PLUGIN_ID = "synapse@synapse"
#: The four install-prompt options 0.20 dropped from plugin.json; their answers move to
#: config.json (adopt_legacy). The URL and the optional token stay declared.
DROPPED_OPTIONS = (
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
)
_CLI_TIMEOUT = 180  # `marketplace add` clones the repo

# A base URL every shell and both harness config formats carry without quoting: a
# scheme, a host[:port] and an optional plain path.
_URL_RE = re.compile(
    r"^https?://(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(?::\d{1,5})?(?:/[A-Za-z0-9._~\-/]*)?$"
)


def normalize_url(url: str) -> str:
    """The base URL to store: no trailing slash, no endpoint suffix pasted along with it.
    Raises ValueError when it isn't a plain http(s) URL."""
    base = (url or "").strip().rstrip("/")
    for suffix in ("/ingest", "/recall", "/mcp", "/skills", "/install"):
        # a path segment only: a host that happens to be named "mcp" stays
        if base.endswith(suffix) and "://" in base[: -len(suffix)]:
            base = base[: -len(suffix)]
    base = base.rstrip("/")
    if not _URL_RE.match(base):
        raise ValueError(f"not a plain http(s) base URL: {url!r}")
    return base


def detect() -> dict[str, bool]:
    """Which harnesses this machine has: the CLI on PATH, or its config dir."""
    home = Path(os.path.expanduser("~"))
    return {
        "claude": bool(shutil.which("claude")) or config.CONFIG_DIR.is_dir(),
        "codex": bool(shutil.which("codex")) or (home / ".codex").is_dir(),
    }


# --------------------------------------------------------------------------- legacy


def _user_settings() -> Path:
    return config.CONFIG_DIR / "settings.json"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _plugin_secret(name: str) -> str:
    """A value the old install dialog stored as sensitive: Claude Code keeps those in
    ~/.claude/.credentials.json (Linux, Windows) under pluginSecrets, or in the macOS
    keychain, which only Claude Code itself can read."""
    secrets = _read_json(config.CONFIG_DIR / ".credentials.json").get("pluginSecrets")
    for key, values in (secrets or {}).items() if isinstance(secrets, dict) else ():
        if str(key).split("@", 1)[0] == "synapse" and isinstance(values, dict):
            if isinstance(values.get(name), str) and values[name].strip():
                return values[name].strip()
    return ""


def adopt_legacy() -> list[str]:
    """Copy what the Claude plugin's install prompt stored (and the old enrollment record)
    into config.json, for every key config.json doesn't have yet. A copy, not a move: a
    Claude plugin older than 0.20 still reads the old places until it updates, and
    config.json outranks them for everything newer. Returns the config.json keys
    adopted. Raises OSError."""
    home = synapse_home.load()
    opts = dict(synapse_home.plugin_options([_user_settings()]))
    if not opts.get("SYNAPSE_INGEST_TOKEN") and _plugin_secret("SYNAPSE_INGEST_TOKEN"):
        opts["SYNAPSE_INGEST_TOKEN"] = _plugin_secret("SYNAPSE_INGEST_TOKEN")
    changes: dict = {}
    for key, name in synapse_home.KEYS.items():
        value = opts.get(name)
        if key in home or value in (None, ""):
            continue
        text = str(value).strip()
        if key in synapse_home.BOOL_KEYS:
            changes[key] = text != "0"  # the plugin's own reading: anything but "0" is on
        elif key == "config_paths":
            changes[key] = [g for g in re.split(r"[,\s]+", text) if g]
        elif key == "role":
            changes[key] = text.lower()
        else:
            changes[key] = text
    if "device" not in home:
        record = config.read_device_state()
        if record:
            token = changes.get("token") or home.get("token") or ""
            if not record.get("token_sha256") and token:
                record = {**record, "token_sha256": config.token_sha256(token)}
            changes["device"] = record
    if changes:
        config.set_home_values(changes)
    return sorted(changes)


def drop_options(names) -> None:
    """Remove ``names`` from every synapse@* options block in the user settings.json.
    Leaves an unreadable file alone."""
    synapse_home.drop_plugin_options(_user_settings(), names)


# --------------------------------------------------------------------------- Claude Code


def _run(cmd: list[str], runner=subprocess.run):
    try:
        # The CLI writes UTF-8 whatever the console code page is (Windows: cp1252, cp936...).
        return runner(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CLI_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def _json_out(done) -> list | dict | None:
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout or "null")
    except ValueError:
        return None


def claude_plugin_id(runner=subprocess.run) -> str | None:
    """The installed Synapse plugin's id (``synapse@<marketplace>``), or None."""
    exe = shutil.which("claude")
    if exe:
        plugins = _json_out(_run([exe, "plugin", "list", "--json"], runner))
        for p in plugins if isinstance(plugins, list) else []:
            if str(p.get("id", "")).split("@", 1)[0] == "synapse":
                return str(p["id"])
    enabled = _read_json(_user_settings()).get("enabledPlugins") or {}
    return next((k for k in enabled if str(k).split("@", 1)[0] == "synapse"), None)


def install_claude(url: str, runner=subprocess.run) -> str:
    """Install (or update the URL of) the Claude Code plugin. Returns a one-line report.
    Raises RuntimeError when nothing could be done."""
    exe = shutil.which("claude")
    if exe:
        why = _claude_cli(exe, url, runner)
        if not why:
            return f"Claude Code: plugin installed, SYNAPSE_URL={url}"
        print(f"Claude Code: `claude plugin` failed ({why}); writing settings.json instead")
    _settings_bootstrap(url)
    return (
        "Claude Code: plugin enabled in settings.json; it installs when an interactive "
        "session starts, so restart Claude Code twice (or run /reload-plugins in the first)"
    )


def _claude_cli(exe: str, url: str, runner) -> str:
    """'' on success, else what failed."""
    marketplaces = _json_out(_run([exe, "plugin", "marketplace", "list", "--json"], runner))
    if marketplaces is None:
        return "marketplace list"
    if not any(m.get("name") == "synapse" for m in marketplaces if isinstance(m, dict)):
        done = _run(
            [exe, "plugin", "marketplace", "add", MARKETPLACE_REPO, "--scope", "user"], runner
        )
        if done.returncode != 0:
            return "marketplace add: " + (done.stderr or done.stdout or "").strip()[:200]
    plugins = _json_out(_run([exe, "plugin", "list", "--json"], runner)) or []
    mine = [
        p
        for p in plugins
        if isinstance(p, dict) and str(p.get("id", "")).split("@", 1)[0] == "synapse"
    ]
    # The list includes local and project installs made in other directories; installing
    # at their scope here would enable the plugin in this directory only (and edit its
    # .claude/). Reuse such a scope only inside its own project; else user scope.
    here = os.path.realpath(os.getcwd())
    pick = next((p for p in mine if p.get("scope") == "user"), None) or next(
        (
            p
            for p in mine
            if p.get("projectPath") and os.path.realpath(str(p["projectPath"])) == here
        ),
        None,
    )
    pid = str((pick or (mine[0] if mine else {})).get("id") or PLUGIN_ID)
    scope = str((pick or {}).get("scope") or "user")
    if mine:
        # Installing again only applies the URL; an older cached plugin would keep running
        # without the headers helper, reading only its own old settings.
        _run([exe, "plugin", "marketplace", "update", pid.split("@", 1)[1]], runner)
        _run([exe, "plugin", "update", pid], runner)
    done = _run(
        [exe, "plugin", "install", pid, "--scope", scope, "--config", f"SYNAPSE_URL={url}"], runner
    )
    if done.returncode != 0:
        return "plugin install: " + (done.stderr or done.stdout or "").strip()[:200]
    plugins = _json_out(_run([exe, "plugin", "list", "--json"], runner)) or []
    version = next(
        (
            str(p.get("version") or "")
            for p in plugins
            if isinstance(p, dict) and p.get("id") == pid
        ),
        "",
    )
    if _version_tuple(version) >= (0, 20):
        # A plugin that no longer declares these would keep exporting their stale answers
        # to its hooks, and `claude plugin configure` refuses to edit them. Copied into
        # config.json first (`synapse install` and `config url` get here without a login):
        # dropped without that, they would be gone.
        try:
            adopt_legacy()
        except OSError:
            return ""
        drop_options(DROPPED_OPTIONS)
    return ""


def _version_tuple(version: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.+-]", version or "0"))


def _settings_bootstrap(url: str) -> None:
    path = _user_settings()
    data = synapse_home.settings_for_update(path)
    data.setdefault("extraKnownMarketplaces", {}).setdefault(
        "synapse", {"source": {"source": "github", "repo": MARKETPLACE_REPO}}
    )
    enabled = data.setdefault("enabledPlugins", {})
    key = next((k for k in enabled if str(k).split("@", 1)[0] == "synapse"), PLUGIN_ID)
    enabled[key] = True
    configs = data.setdefault("pluginConfigs", {})
    configs.setdefault(key, {}).setdefault("options", {})["SYNAPSE_URL"] = url
    synapse_home.atomic_write_json(path, data)


def update_claude(runner=subprocess.run) -> str | None:
    """Refresh the marketplace and update the plugin, when the CLI is there."""
    exe = shutil.which("claude")
    pid = claude_plugin_id(runner)
    if not exe or not pid:
        return None
    _run([exe, "plugin", "marketplace", "update", pid.split("@", 1)[1]], runner)
    done = _run([exe, "plugin", "update", pid], runner)
    return "Claude Code: plugin " + ("updated" if done.returncode == 0 else "update failed")


# --------------------------------------------------------------------------- Codex


def codex_installer():
    """plugin-codex/install.py of the client to wire Codex to: the managed client
    (~/.synapse/lib, through the fixed launcher) when it is installed, else the
    plugin-codex next to this plugin (a git checkout). None when neither exists: the
    Claude plugin's own copy carries only plugin/."""
    managed = managed_root()
    sibling = Path(__file__).resolve().parents[2]
    root = managed or (sibling if (sibling / "plugin-codex" / "install.py").is_file() else None)
    if root is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "synapse_codex_install", root / "plugin-codex" / "install.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Spelled as synapse_run.py spells it (resolved): Codex trusts a hook by its command
    # text, and a symlinked ~ would otherwise give each entry point its own spelling.
    runner = synapse_home.home().resolve() / "bin" / "synapse_run.py" if managed else None
    token = mod.python_token()
    mod.LAUNCHER = mod.Launcher(
        token, root=root, runner=runner, helper_python=mod.helper_interpreter(token)
    )
    return mod


def managed_root() -> Path | None:
    """The managed client's current version directory, when it is installed (the
    launcher in ~/.synapse/bin decides which version that is)."""
    home = synapse_home.home().resolve()
    runner = home / "bin" / "synapse_run.py"
    if not runner.is_file():
        return None
    spec = importlib.util.spec_from_file_location("synapse_run_probe", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = mod.client_root(home)
    return root if root and (root / "plugin-codex" / "install.py").is_file() else None


def install_codex(url: str, dry_run: bool = False) -> str:
    """Write the Codex MCP entry and hooks. Returns a one-line report. Raises
    RuntimeError (no client to point at) or ValueError (config.toml left unchanged)."""
    installer = codex_installer()
    if installer is None:
        raise RuntimeError(
            "no Codex client files here: install the client with "
            f"`curl -sSL {url}/install | python3 -`"
        )
    changed = installer.install(url, dry_run=dry_run)
    if not changed:
        return "Codex: already wired"
    return "Codex: wired; start codex, run /hooks and trust the Synapse hooks"


def codex_wired() -> bool:
    """Whether ~/.codex/config.toml already has the Synapse MCP entry."""
    path = Path(os.path.expanduser("~/.codex/config.toml"))
    try:
        return "[mcp_servers.synapse]" in path.read_text(encoding="utf-8")
    except OSError:
        return False
