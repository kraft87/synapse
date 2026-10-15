"""Codex SessionStart (plugin-codex/hooks/session_start.py) under a slow or dead server.

The hook used to fetch the preferences, then the board, each with a 10 s timeout, after a
12 s skills sync: a blackholed server used the whole 20 s the hook is given, Codex killed
it, and nothing printed, not even the local lines (the token notice, the upload warning)
that exist for exactly that situation. The fetches now run side by side under one budget.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

_SESSION_START = Path(__file__).resolve().parents[1] / "plugin-codex" / "hooks" / "session_start.py"


@pytest.fixture()
def hook(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("common", "config"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("codex_session_start_budget_test", _SESSION_START)
    assert spec and spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("common", "config"):
        sys.modules.pop(name, None)
    monkeypatch.setattr(mod, "_upload_warning", lambda: None)
    monkeypatch.setattr(mod, "token_override_notice", lambda: "")
    monkeypatch.setattr(mod, "_spawn_catchup", lambda skip="": None)
    return mod


def _run(hook: ModuleType, monkeypatch, capsys, payload: dict | None = None) -> dict:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload or {})))
    hook.main()
    out = capsys.readouterr().out
    return json.loads(out) if out else {}


def test_board_preferences_and_skills_sync_run_side_by_side(hook, monkeypatch, capsys):
    """A barrier only opens when all three are running at once: one after another would
    time it out instead."""
    barrier = threading.Barrier(3, timeout=5)

    def wait(value):
        def job(*_a):
            barrier.wait()
            return value

        return job

    monkeypatch.setattr(hook, "_prefs_text", wait("[prefs]"))
    monkeypatch.setattr(hook, "_board_text", wait("[board]"))
    monkeypatch.setattr(hook, "_sync_skills", wait(None))
    out = _run(hook, monkeypatch, capsys)
    assert out["hookSpecificOutput"]["additionalContext"] == "[prefs]\n\n[board]"


def test_a_hung_server_costs_the_budget_not_the_local_lines(hook, monkeypatch, capsys):
    release = threading.Event()
    monkeypatch.setattr(hook, "_BUDGET", 0.3)
    monkeypatch.setattr(hook, "token_override_notice", lambda: "[notice]")
    monkeypatch.setattr(hook, "_upload_warning", lambda: "uploads failing")
    monkeypatch.setattr(hook, "_prefs_text", lambda: release.wait(10) and "[prefs]")
    monkeypatch.setattr(hook, "_board_text", lambda project: "[board]")
    monkeypatch.setattr(hook, "_sync_skills", lambda cwd=None: release.wait(10) and None)
    started = time.monotonic()
    try:
        out = _run(hook, monkeypatch, capsys)
        assert time.monotonic() - started < 3
    finally:
        release.set()
    # what arrived is printed, what did not is left out, the local lines always print
    assert out["hookSpecificOutput"]["additionalContext"] == (
        "[notice]\n\nuploads failing\n\n[board]"
    )
    assert out["systemMessage"] == "uploads failing"


def test_every_fetch_hung_still_prints_the_local_lines(hook, monkeypatch, capsys):
    release = threading.Event()
    monkeypatch.setattr(hook, "_BUDGET", 0.2)
    monkeypatch.setattr(hook, "token_override_notice", lambda: "[notice]")
    monkeypatch.setattr(hook, "_prefs_text", lambda: release.wait(10) and None)
    monkeypatch.setattr(hook, "_board_text", lambda project: release.wait(10) and None)
    monkeypatch.setattr(hook, "_sync_skills", lambda cwd=None: None)
    try:
        out = _run(hook, monkeypatch, capsys)
    finally:
        release.set()
    assert out["hookSpecificOutput"]["additionalContext"] == "[notice]"


def test_a_job_that_raises_does_not_take_the_hook_down(hook, monkeypatch, capsys):
    def boom(*_a):
        raise RuntimeError("fetch blew up")

    monkeypatch.setattr(hook, "_prefs_text", boom)
    monkeypatch.setattr(hook, "_board_text", lambda project: "[board]")
    monkeypatch.setattr(hook, "_sync_skills", lambda cwd=None: None)
    out = _run(hook, monkeypatch, capsys)
    assert out["hookSpecificOutput"]["additionalContext"] == "[board]"


@pytest.mark.parametrize(
    ("source", "reruns"),
    [("startup", True), ("resume", True), ("clear", False), ("compact", False)],
)
def test_a_clear_or_compaction_reinjects_context_without_the_sync_and_the_sweep(
    hook, monkeypatch, capsys, source, reruns
):
    calls: list[str] = []
    monkeypatch.setattr(hook, "_spawn_catchup", lambda skip="": calls.append("sweep"))
    monkeypatch.setattr(hook, "_sync_skills", lambda cwd=None: calls.append("sync"))
    monkeypatch.setattr(hook, "_prefs_text", lambda: "[prefs]")
    monkeypatch.setattr(hook, "_board_text", lambda project: "[board]")
    out = _run(hook, monkeypatch, capsys, {"source": source})
    assert out["hookSpecificOutput"]["additionalContext"] == "[prefs]\n\n[board]"
    assert sorted(calls) == (["sweep", "sync"] if reruns else [])
