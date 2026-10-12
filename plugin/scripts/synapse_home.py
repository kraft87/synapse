#!/usr/bin/env python3
# mypy: ignore-errors
"""~/.synapse/config.json — this machine's Synapse settings, owned by no harness.

Until 0.20 every setting lived in Claude Code's settings.json under
pluginConfigs."synapse@<marketplace>".options, and the enrollment record in the Claude
plugin's data dir. A machine without Claude Code (Codex only, or just the CLI) therefore
had no server URL, no credential and no way to log in. This file is the one place both
plugins and the `synapse` CLI read and write instead. The legacy locations stay readable
as fallbacks, so an existing install keeps working with no action.

    {
      "url": "https://synapse.example.net",
      "token": "<this machine's device token>",
      "role": "personal" | "work",
      "skills_sync": true,
      "config_sync": false,
      "config_paths": ["output-styles/*.md"],
      "device": {"surface_id": ..., "trust": ..., "token_sha256": ..., "server": ...}
    }

It holds a bearer credential, so it is written 0600 on POSIX (a fresh temp file renamed
over the old one: a crash leaves the previous file, never half of one). On Windows the
profile folder's ACLs apply, as they do to settings.json.

Stdlib only, and imported by path from both plugins (plugin-codex has no copy), so it
must not import anything from either plugin.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

#: config.json key -> the env / plugin-option name the rest of the client resolves by.
KEYS = {
    "url": "SYNAPSE_URL",
    "token": "SYNAPSE_INGEST_TOKEN",
    "role": "SYNAPSE_MACHINE_ROLE",
    "skills_sync": "SYNAPSE_SKILLS_SYNC",
    "config_sync": "SYNAPSE_CONFIG_SYNC",
    "config_paths": "SYNAPSE_CONFIG_PATHS",
}
_BY_OPTION = {opt: key for key, opt in KEYS.items()}
BOOL_KEYS = frozenset({"skills_sync", "config_sync"})

_TRUE = frozenset({"1", "true", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "no", "n", "off"})


def home() -> Path:
    """The Synapse home: config.json, the installed client (lib/) and its shims (bin/).
    SYNAPSE_HOME overrides it (tests, a second profile); harness helpers that clear the
    environment always land on the default, which is the one that matters."""
    return Path(os.path.expanduser(os.environ.get("SYNAPSE_HOME") or "~/.synapse"))


def config_file() -> Path:
    return home() / "config.json"


def load(path: Path | None = None) -> dict:
    """The parsed config, or ``{}`` when it is missing, unreadable or not an object.
    Hooks call this at import, so it never raises."""
    try:
        data = json.loads((path or config_file()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict, path: Path | None = None) -> None:
    """Write ``data`` atomically, owner-only. Raises OSError: callers decide whether a
    failed write is fatal (the CLI) or ignorable (hook bookkeeping)."""
    path = path or config_file()
    created = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if created and os.name != "nt":
        os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update(changes: dict, path: Path | None = None) -> dict:
    """Merge ``changes`` into the file (a ``None`` value deletes that key) and return the
    new contents."""
    data = load(path)
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    save(data, path)
    return data


def option(data: dict, name: str) -> str:
    """``data``'s value for an env-style name (``SYNAPSE_URL``) as the string the env
    var would carry: booleans as "1"/"0", lists comma-joined, "" when unset."""
    key = _BY_OPTION.get(name)
    value = data.get(key) if key else None
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def has(data: dict, name: str) -> bool:
    """Whether ``data`` sets the env-style name at all. An explicitly empty value (a
    cleared list of globs) still counts: it must not fall through to a legacy one."""
    key = _BY_OPTION.get(name)
    return bool(key) and data.get(key) is not None


def parse_bool(text: str) -> bool | None:
    """yes/no, on/off, 1/0, true/false -> bool; anything else -> None."""
    low = str(text).strip().lower()
    return True if low in _TRUE else False if low in _FALSE else None


def plugin_options(settings_files) -> dict:
    """The Claude plugin's saved install answers: every pluginConfigs."synapse@<any>"
    options block across ``settings_files``, later files winning, empty values dropped.
    The legacy source both plugins fall back to."""
    merged: dict = {}
    for path in settings_files:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        configs = data.get("pluginConfigs") if isinstance(data, dict) else None
        if not isinstance(configs, dict):
            continue
        for cfg_key, cfg in configs.items():
            opts = cfg.get("options") if isinstance(cfg, dict) else None
            if str(cfg_key).split("@", 1)[0] == "synapse" and isinstance(opts, dict):
                merged.update({k: v for k, v in opts.items() if v not in (None, "")})
    return merged
