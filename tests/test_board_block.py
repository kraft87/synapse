"""SessionStart board block (plugin/scripts/board_block.py) — thin-client hook.

The hook must (a) print the server-rendered board text verbatim (stdout lands in the
session's context), (b) scope it to the project derived from the hook payload's cwd
the same way the ingest path labels episodes (basename of cwd; process cwd fallback),
(c) honor the SYNAPSE_BOARD=0 kill switch without touching the network, and (d) be
fail-open: server down, timeout, or a non-ok payload prints nothing and exits 0 — a
broken board must never break a session start.

Stdlib-only script loaded by path (it lives outside the package). No live server:
config.get_json is monkeypatched on the loaded module.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "plugin" / "scripts" / "board_block.py"

# Every env var the plugin config layer (or this hook) consults — cleared so the
# tests are hermetic on any machine (dev box or CI).
_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_BOARD",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_BOARD",
)

_BOARD_TEXT = (
    "[Synapse board — project: synapse]\n"
    "42 episodes across 2 projects (most recent: synapse, scripts).\n"
    "\n"
    "## Rules & feedback\n"
    "- verify before shipping (n:3, upd 07-01)"
)


def _isolated_env(monkeypatch, tmp_path) -> None:
    """Point the plugin config layer at a scratch config dir: no real env vars,
    no settings.json options, no credentials file, no project .claude."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _load_hook() -> ModuleType:
    """Load the hook fresh. `config` is popped around the load so it re-resolves
    from the current env rather than a module cached by an earlier test."""
    for name in ("config", "board_block"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("board_block", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("config", "board_block"):
        sys.modules.pop(name, None)
    return mod


def _run(monkeypatch, mod: ModuleType, stdin: str, reply) -> list[tuple[str, dict]]:
    """Run main() with the hook payload on stdin and get_json mocked; return the
    (path, params) calls made. `reply` is a payload dict or an exception to raise."""
    calls: list[tuple[str, dict]] = []

    def fake_get_json(path, params=None, timeout=30.0):
        calls.append((path, dict(params or {})))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(mod, "get_json", fake_get_json)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    mod.main()
    return calls


def test_board_text_printed_verbatim(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    payload = {"status": "ok", "text": _BOARD_TEXT, "n_notes": 1, "overflow": 0}
    calls = _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), payload)
    assert capsys.readouterr().out == _BOARD_TEXT + "\n"
    assert calls == [("/context", {"project": "synapse"})]


def test_kill_switch_no_output_no_http(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SYNAPSE_BOARD", "0")
    mod = _load_hook()
    payload = {"status": "ok", "text": _BOARD_TEXT}
    calls = _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), payload)
    assert capsys.readouterr().out == ""
    assert calls == []


@pytest.mark.parametrize(
    "reply",
    [
        urllib.error.URLError("server down"),
        TimeoutError("timed out"),
        urllib.error.HTTPError("http://x/context", 404, "not found", None, None),  # older server
    ],
    ids=["conn-refused", "timeout", "http-404"],
)
def test_server_errors_are_silent(monkeypatch, tmp_path, capsys, reply):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), reply)
    assert capsys.readouterr().out == ""  # fail-open: no block, no noise, exit 0


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "error", "detail": "board build failed"},
        {"status": "ok"},  # ok but no text
        {"status": "ok", "text": ""},
        {},
    ],
    ids=["error-status", "missing-text", "empty-text", "empty-payload"],
)
def test_non_ok_payload_is_silent(monkeypatch, tmp_path, capsys, reply):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), reply)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/home/user/services/synapse", "synapse"),
        ("/home/user/scripts/", "scripts"),  # trailing slash — same as the ingest labeler
    ],
)
def test_project_derived_from_hook_cwd(monkeypatch, tmp_path, capsys, cwd, expected):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    calls = _run(monkeypatch, mod, json.dumps({"cwd": cwd}), {"status": "ok", "text": "b"})
    assert calls == [("/context", {"project": expected})]
    assert capsys.readouterr().out == "b\n"


@pytest.mark.parametrize(
    "stdin",
    ["", "not json {", json.dumps({}), json.dumps({"cwd": "/"})],
    ids=["empty-stdin", "bad-json", "no-cwd", "root-cwd"],
)
def test_project_falls_back_to_process_cwd(monkeypatch, tmp_path, stdin):
    _isolated_env(monkeypatch, tmp_path)  # chdirs to tmp_path
    mod = _load_hook()
    calls = _run(monkeypatch, mod, stdin, {"status": "ok", "text": "b"})
    assert calls == [("/context", {"project": tmp_path.name})]
