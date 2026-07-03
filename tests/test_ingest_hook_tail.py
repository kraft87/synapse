"""The Stop hook ships a bounded TAIL, not the whole transcript.

``_select_tail`` must (a) start the slice at a real turn boundary so a leading
fragment of an already-ingested turn is never POSTed, and (b) fall back to the
full file when the tail window holds no boundary (a single mega-turn longer than
the window) so that turn still lands complete. Both copies of the hook are
covered: the repo-root ``scripts/synapse_ingest_hook.py`` and the plugin's
``plugin/scripts/ingest_hook.py``.

The plugin copy also resolves its endpoint through the plugin config layer
(``plugin/scripts/config.py``) — regression-tested here so it can never again
silently fall back to localhost when the user configured ``SYNAPSE_URL``
through the ``/plugin install`` prompt (persisted in settings.json).

The hooks live outside the package and run under the CLI's bare Python, so
they're loaded by path here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_ROOT_HOOK = _REPO / "scripts" / "synapse_ingest_hook.py"
_PLUGIN_HOOK = _REPO / "plugin" / "scripts" / "ingest_hook.py"

# Every env var the plugin config layer consults for the endpoint/token, so the
# regression tests are hermetic on any machine (dev box or CI).
_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


def _load_hook(path: Path) -> ModuleType:
    """Load a hook copy fresh. `config` is popped around the load so the plugin
    copy re-resolves its endpoint from the current env/settings rather than a
    module cached by an earlier test."""
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


def _isolated_plugin_env(monkeypatch, tmp_path, options: dict | None = None) -> None:
    """Point the plugin config layer at a scratch config dir: a controlled
    settings.json, no real env vars, no credentials file, no project .claude."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if options is not None:
        (cfg_dir / "settings.json").write_text(
            json.dumps({"pluginConfigs": {"synapse@synapse": {"options": options}}}),
            encoding="utf-8",
        )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # keep Path.cwd()/.claude from finding real project settings
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _records() -> list[dict]:
    sid = "s-tail-1"

    def u(uid: str, text: str) -> dict:
        return {"type": "user", "uuid": uid, "sessionId": sid, "message": {"content": text}}

    def a(uid: str, text: str) -> dict:
        return {
            "type": "assistant",
            "uuid": uid,
            "sessionId": sid,
            "message": {"content": [{"type": "text", "text": text}]},
        }

    # 3 turns, 6 records: u1 a1 | u2 a2 | u3 a3
    return [
        u("u1", "q1"),
        a("a1", "r1"),
        u("u2", "q2"),
        a("a2", "r2"),
        u("u3", "q3"),
        a("a3", "r3"),
    ]


def _lines(recs: list[dict]) -> list[bytes]:
    return [json.dumps(r).encode() for r in recs]


@pytest.mark.parametrize("hook_path", [_ROOT_HOOK, _PLUGIN_HOOK], ids=["root", "plugin"])
def test_tail_starts_at_turn_boundary(hook_path, monkeypatch):
    mod = _load_hook(hook_path)
    monkeypatch.setattr(mod, "TAIL_RECORDS", 3)
    records, mode = mod._select_tail(_lines(_records()))
    # last 3 raw records are [a2, u3, a3]; trimmed to the first turn-start (u3)
    assert mode == "tail"
    assert [r["uuid"] for r in records] == ["u3", "a3"]


@pytest.mark.parametrize("hook_path", [_ROOT_HOOK, _PLUGIN_HOOK], ids=["root", "plugin"])
def test_tail_drops_leading_fragment(hook_path, monkeypatch):
    """When the window cuts mid-turn, the dangling tail of the prior turn is dropped
    (it was already ingested); the slice begins at the next user turn."""
    mod = _load_hook(hook_path)
    monkeypatch.setattr(mod, "TAIL_RECORDS", 5)
    records, mode = mod._select_tail(_lines(_records()))
    # last 5 = [a1, u2, a2, u3, a3]; first turn-start is u2 → drop a1
    assert mode == "tail"
    assert [r["uuid"] for r in records] == ["u2", "a2", "u3", "a3"]


@pytest.mark.parametrize("hook_path", [_ROOT_HOOK, _PLUGIN_HOOK], ids=["root", "plugin"])
def test_full_fallback_when_no_boundary_in_window(hook_path, monkeypatch):
    """A window with no turn boundary (one mega-turn) falls back to the full file."""
    mod = _load_hook(hook_path)
    monkeypatch.setattr(mod, "TAIL_RECORDS", 1)
    records, mode = mod._select_tail(_lines(_records()))
    # last 1 record = a3 (an assistant record, not a turn-start) → ship everything
    assert mode == "full-fallback"
    assert len(records) == 6


@pytest.mark.parametrize("hook_path", [_ROOT_HOOK, _PLUGIN_HOOK], ids=["root", "plugin"])
def test_machinery_user_record_is_not_a_boundary(hook_path):
    """Slash-command / system-reminder user records must not count as turn starts —
    starting a slice there would split a turn at machinery noise."""
    mod = _load_hook(hook_path)
    assert mod._is_turn_start({"type": "user", "message": {"content": "real question"}})
    assert not mod._is_turn_start(
        {"type": "user", "message": {"content": "<command-name>/compact</command-name>"}}
    )
    assert not mod._is_turn_start(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}}
    )


# ---------------------------------------------------------------------------
# Plugin-hook endpoint resolution (the pre-0.8 launch-blocker regression).
# ---------------------------------------------------------------------------


def test_plugin_hook_targets_settings_url_not_localhost(monkeypatch, tmp_path):
    """A user who answers the /plugin install prompt (SYNAPSE_URL lands in
    settings.json pluginConfigs) must have transcripts POSTed there — not
    silently to the localhost default. Pre-0.8 the hook read only raw env vars
    (SYNAPSE_INGEST_URL), which the install flow never sets, so every transcript
    vanished into localhost."""
    _isolated_plugin_env(
        monkeypatch, tmp_path, options={"SYNAPSE_URL": "https://synapse.example.net"}
    )
    mod = _load_hook(_PLUGIN_HOOK)
    assert mod.INGEST_URL == "https://synapse.example.net/ingest"
    assert "localhost" not in mod.INGEST_URL


def test_plugin_hook_legacy_env_override_still_wins(monkeypatch, tmp_path):
    """config.py precedence: an explicit SYNAPSE_INGEST_URL env var outranks the
    settings.json install value (existing installs keep working)."""
    _isolated_plugin_env(
        monkeypatch, tmp_path, options={"SYNAPSE_URL": "https://synapse.example.net"}
    )
    monkeypatch.setenv("SYNAPSE_INGEST_URL", "http://10.9.9.9:8765/ingest")
    mod = _load_hook(_PLUGIN_HOOK)
    assert mod.INGEST_URL == "http://10.9.9.9:8765/ingest"


def test_plugin_hook_token_from_settings(monkeypatch, tmp_path):
    """The bearer token from the install prompt reaches the hook too (pre-0.8 it
    was env-only, so an auth-gated server rejected every hook POST)."""
    _isolated_plugin_env(
        monkeypatch,
        tmp_path,
        options={
            "SYNAPSE_URL": "https://synapse.example.net",
            "SYNAPSE_INGEST_TOKEN": "tok-test-123",
        },
    )
    mod = _load_hook(_PLUGIN_HOOK)
    assert mod.INGEST_TOKEN == "tok-test-123"


def test_plugin_hook_defaults_to_localhost_when_unconfigured(monkeypatch, tmp_path):
    """No env, no settings → the documented local-quickstart default."""
    _isolated_plugin_env(monkeypatch, tmp_path, options=None)
    mod = _load_hook(_PLUGIN_HOOK)
    assert mod.INGEST_URL == "http://localhost:8765/ingest"
    assert mod.INGEST_TOKEN == ""
