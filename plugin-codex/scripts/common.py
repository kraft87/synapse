"""Configuration and HTTP for the Codex-side Synapse hooks: a thin layer over the client
core (plugin/synapse_client).

Every setting resolves the way the core resolves it (synapse_client.settings): env var,
then ~/.synapse/config.json (what `synapse login` writes; the same file the Claude
plugin reads), then the Synapse *Claude Code* plugin's legacy saved options in
~/.claude/settings.json, then the default. A Codex-only machine therefore configures
with `synapse login` alone. On an enrolled machine the device token beats an env
SYNAPSE_INGEST_TOKEN, and it never goes to a server that did not mint it
(synapse_client.credentials). Requests go through the core's Transport, which sends the
bearer to this server's origin only and drops it on a redirect elsewhere.

What this layer adds is Codex's own: the two legacy stores the core does not read itself
(the Claude plugin's saved options and its pre-0.20 device.json), the
http://localhost:8765 default the hooks have always had, the User-Agent, and the names
the hooks import. Settings are read when a hook starts (at import), as before.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The client core is the sibling plugin/ package, reached the way
# plugin/scripts/synapse_project.py reaches it: plugin/ first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugin"))
from synapse_client import __version__, context
from synapse_client.settings import Settings

if TYPE_CHECKING:
    # What sends a request (urllib, http, ssl, socket) is imported where one is sent:
    # every hook imports this module, and the ones that run at each prompt only read
    # flags.
    from synapse_client.api import Client
    from synapse_client.http import Transport

#: Where requests go when no URL is configured anywhere. The core sends nothing then;
#: the Codex hooks have always tried a local server.
DEFAULT_URL = "http://localhost:8765"
USER_AGENT = f"synapse-client/{__version__} codex"
#: Where an environment token most likely comes from, for the token notice.
ENV_LOCATION = "your shell profile or wherever Codex inherits it"

# The Claude plugin's saved options are parsed by its own module (plugin-codex has no
# copy), loaded by path from the sibling checkout like skills_sync's engine.
_HOME_PY = Path(__file__).resolve().parents[2] / "plugin" / "scripts" / "synapse_home.py"


def _load_home() -> Any:
    mod = sys.modules.get("synapse_home")
    if mod is None:
        # Its directory leads sys.path for the one import, as in plugin/scripts/config.py:
        # importlib.util would load the file by path, and brings threading with it.
        there = str(_HOME_PY.parent)
        sys.path.insert(0, there)
        try:
            import synapse_home as mod
        finally:
            sys.path.remove(there)
    return mod


synapse_home = _load_home()


def _claude_plugin_options() -> dict[str, str]:
    opts = synapse_home.plugin_options([Path(os.path.expanduser("~/.claude/settings.json"))])
    return {k: str(v) for k, v in opts.items()}


# The enrollment record's pre-0.20 location, where the Claude plugin wrote it
# (plugin/scripts/config.py DEVICE_FILE). config.json's "device" wins over it. It holds
# the device token's sha256 and the server that minted it, never the token.
_DEVICE_FILE = (
    Path(os.path.expanduser(os.environ.get("SYNAPSE_DATA_DIR") or "~/.local/share/synapse-skills"))
    / "device.json"
)


def _legacy_device() -> dict[str, Any]:
    try:
        data = json.loads(_DEVICE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """``Settings.load`` with Codex's legacy sources: the Claude plugin's saved options
    rank below config.json (as ``plugin_config``, the source the token rules know them
    by), and its old device.json stands in for a config.json without a record. With no
    URL configured anywhere, the default is resolved as if configured, so the device
    token's origin rule judges the server requests actually go to."""
    environ = os.environ if env is None else env
    extras = [("plugin_config", _claude_plugin_options())]
    device = _legacy_device()
    settings = Settings.load(environ, extra_candidates=extras, legacy_device=device)
    if not settings.base_url:
        settings = Settings.load(
            {**environ, "SYNAPSE_URL": DEFAULT_URL}, extra_candidates=extras, legacy_device=device
        )
    return settings


SETTINGS = load_settings()
BASE_URL = SETTINGS.base_url
INGEST_URL = SETTINGS.ingest_url
TOKEN = SETTINGS.token


def live_settings() -> Settings:
    """``SETTINGS`` with the environment as it is now. The URL and token stay as resolved
    at start; flags and kill switches follow a variable set or removed since (the hooks'
    ``_cfg`` always read the live environment)."""
    snapshot = {k: v for k, v in os.environ.items() if k.startswith("SYNAPSE_")}
    return SETTINGS.replace(environ=types.MappingProxyType(snapshot))


def _cfg(key: str, default: str = "") -> str:
    """A setting by its env-style name: env, config.json, the Claude plugin's saved
    options, ``default`` (Settings.flag)."""
    return live_settings().flag(key, default)


def token_override_notice() -> str:
    """The credential notice (synapse_client.context.notice_part): one line per token
    source the device token overrode, or why the device token is not sent at all, else
    "". Names the source and the surface, never a token value or any part of one."""
    return context.notice_part(SETTINGS, env_location=ENV_LOCATION)


PRIVATE_DIR = Path(os.path.expanduser(_cfg("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")))


def client() -> Client:
    """The core's API client for this machine's server and credential."""
    from synapse_client.api import Client

    return Client(SETTINGS, USER_AGENT)


def _transport() -> Transport:
    # Built per request from the module's names, so a caller that repoints them (a test)
    # is obeyed. The bearer goes to BASE_URL's origin and to INGEST_URL's.
    from synapse_client.http import Transport

    return Transport(BASE_URL, TOKEN, USER_AGENT, token_origins=(INGEST_URL,))


def get_json(
    path: str, params: dict[str, Any] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    out: dict[str, Any] = _transport().get_json(path, params, timeout=timeout)
    return out


def post_json(path: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    out: dict[str, Any] = _transport().post_json(path, payload, timeout=timeout)
    return out


def request_json(
    method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    out: dict[str, Any] = _transport().request_json(method, path, payload, timeout=timeout)
    return out
