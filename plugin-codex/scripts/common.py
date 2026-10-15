"""The Codex hooks' view of the Synapse client config: plugin/scripts/config.py.

One implementation resolves the server URL, the bearer (an enrolled machine's device
token beats a stale env SYNAPSE_INGEST_TOKEN, and goes only to the server that minted
it) and the HTTP helpers for both harnesses; this module loads it by path from the
sibling checkout, as the skills sync loads its engine, and re-exports what the Codex
scripts use. A Codex-only machine configures with `synapse login` alone: the settings
come from env, then ~/.synapse/config.json, then the legacy options the Synapse Claude
Code plugin saved in the USER's settings.json (never a project's .claude/, which is not
Codex's).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

_CONFIG_PY = Path(__file__).resolve().parents[2] / "plugin" / "scripts" / "config.py"


def _load_config() -> Any:
    mod = sys.modules.get("config")
    if mod is None:
        os.environ.setdefault("SYNAPSE_USER_SETTINGS_ONLY", "1")
        spec = importlib.util.spec_from_file_location("config", _CONFIG_PY)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["config"] = mod
        spec.loader.exec_module(mod)
        # A process that had the module before this one (the `synapse` CLI installing the
        # Codex hooks) keeps its own name.
        mod.USER_AGENT = f"synapse-codex-plugin/{mod.synapse_home.client_version()}"
    return mod


config = _load_config()
synapse_home = config.synapse_home
USER_AGENT = config.USER_AGENT

_cfg = config._cfg
BASE_URL = config.BASE_URL
INGEST_URL = config.INGEST_URL
TOKEN = config.INGEST_TOKEN
PRIVATE_DIR = config.PRIVATE_DIR
token_override_notice = config.token_override_notice
get_json = config.get_json
post_json = config.post_json
request_json = config.request_json
