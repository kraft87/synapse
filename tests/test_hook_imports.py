"""What the hooks a user waits for import.

Each hook is a fresh python3, so what it imports is time the user waits. Four of them
run at every prompt or after every recall and only read flags. Two more run after every
turn and only start a detached worker. None of the six sends a request or writes
config.json, so none of them loads the modules that do: those cost more to import than
everything else these processes run.

Each script is run as its harness runs it, under ``-X importtime``, and the modules it
imported are read from the log. Modules the bare interpreter loads by itself (a
distribution's sitecustomize, a virtualenv's .pth files) are left out of the check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.client_core import scratch_user

_ROOT = Path(__file__).resolve().parents[1]

#: The request stack, and the modules it and config.json's writer bring in.
_REQUESTS = (
    "urllib.request",
    "urllib.error",
    "http.client",
    "email.parser",
    "ssl",
    "socket",
    "tempfile",
)
#: `dataclasses` is the single dearest import the settings had (it brings `inspect`).
_SETTINGS = ("dataclasses", "inspect")

_FLAG_HOOKS = (
    "plugin/scripts/recall_nudge.py",
    "plugin/scripts/recall_feedback_nudge.py",
    "plugin-codex/hooks/user_prompt_submit.py",
    "plugin-codex/hooks/post_tool_use.py",
)
_SPAWNING_HOOKS = (
    "plugin/scripts/ingest_hook.py",
    "plugin-codex/hooks/synapse_stop_hook.py",
)


def _imported(args: list[str], stdin: str = "") -> set[str]:
    """The modules a python3 run with ``args`` imported, from its ``-X importtime`` log."""
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", *args],
        input=stdin,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    names = set()
    for line in proc.stderr.splitlines():
        if line.startswith("import time:") and "|" in line:
            names.add(line.rsplit("|", 1)[1].strip())
    assert "os" in names or "json" in names, "no import log to read"
    return names


@pytest.fixture()
def bare(monkeypatch, tmp_path) -> set[str]:
    """A scratch user, and what the interpreter loads before it runs anything."""
    scratch_user(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home" / ".codex"))
    monkeypatch.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp_path / "codex-hook.log"))
    return _imported(["-c", "pass"])


@pytest.mark.parametrize("script", _FLAG_HOOKS)
def test_a_hook_that_only_reads_flags_loads_no_request_stack(bare, script):
    loaded = _imported([str(_ROOT / script)], '{"prompt": "hi"}') - bare
    assert "json" in loaded or "json" in bare
    assert sorted(loaded & {*_REQUESTS, *_SETTINGS, "subprocess"}) == []


@pytest.mark.parametrize("script", _SPAWNING_HOOKS)
def test_a_hook_that_only_starts_its_worker_loads_no_request_stack(bare, script):
    """The payload names no session and no transcript, so nothing is sent: Claude Code's
    hook starts no worker for it, and the worker Codex's hook starts has nothing to
    ship. The imports read here are the waiting process's own."""
    loaded = _imported([str(_ROOT / script)], "{}") - bare
    assert "subprocess" in loaded or "subprocess" in bare
    assert sorted(loaded & {*_REQUESTS, *_SETTINGS}) == []


def test_the_check_sees_a_request_stack_when_one_is_loaded(bare):
    """The log is read correctly: a process that does import the stack is caught."""
    loaded = _imported(["-c", "import urllib.request, dataclasses"]) - bare
    assert {"urllib.request", "http.client", "dataclasses"} <= loaded | bare
    assert loaded & set(_REQUESTS) or set(_REQUESTS) <= bare
