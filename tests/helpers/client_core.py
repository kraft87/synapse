"""Helpers for the client core's tests (plugin/synapse_client/).

The core is imported as a package from ``plugin/`` (appended to sys.path, the way a
harness host would). Its equivalence tests also load the hook scripts it ports
(plugin/scripts/config.py and friends) by path, and those read the user's files at
import: HOME, ~/.synapse, the Claude settings.json, the plugin data dir and the cwd's
``.claude``. ``scratch_user`` points every one of them under a temp dir first, so no
real configuration or credential is ever read.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin"
SCRIPTS = PLUGIN / "scripts"
#: Hook modules the loaders below may leave in sys.modules.
SCRIPT_MODULES = (
    "config",
    "synapse_home",
    "enroll",
    "harness",
    "board_block",
    "recall_nudge",
    "ingest_hook",
    "synapse_ingest_retry",
    "synapse_project",
    "synapse_filelock",
)


def use_core() -> None:
    """Make ``import synapse_client`` resolve to plugin/synapse_client."""
    if str(PLUGIN) not in sys.path:
        sys.path.append(str(PLUGIN))


def core_http() -> ModuleType:
    """The core's ``synapse_client.http``. Every hook that sends the bearer opens its
    request through its ``urlopen`` (config._urlopen, and the core's Transport, which
    Codex's common.py sends through), so a test that fakes the network patches that one
    name."""
    use_core()
    from synapse_client import http

    return http


def fake_urlopen(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[urllib.request.Request]:
    """Replace the core's ``urlopen`` (``core_http``) for one test: each request a hook
    opens is recorded in the returned list and answered with ``answer`` as a JSON body,
    or raised, when ``answer`` is an exception."""
    seen: list[urllib.request.Request] = []

    def fake(req: urllib.request.Request, timeout: float | None = None) -> io.BytesIO:
        seen.append(req)
        if isinstance(answer, BaseException):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(core_http(), "urlopen", fake)
    return seen


def requested(req: urllib.request.Request) -> tuple[str, dict[str, str]]:
    """``(path, query parameters)`` of a request ``fake_urlopen`` recorded."""
    parts = urllib.parse.urlsplit(req.full_url)
    return parts.path, dict(urllib.parse.parse_qsl(parts.query))


def scratch_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point everything a hook reads at import under ``tmp_path``; returns the scratch
    HOME. Every SYNAPSE_* and CLAUDE_PLUGIN_OPTION_* variable from the real environment
    is removed, so a test sets exactly what it means to."""
    home = tmp_path / "home"
    for name in list(os.environ):
        if name.startswith(("SYNAPSE_", "CLAUDE_PLUGIN_OPTION_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("SYNAPSE_HOME", str(home / ".synapse"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_PRIVATE_DIR", str(tmp_path / "private"))
    monkeypatch.setenv("SYNAPSE_INGEST_LOG", str(tmp_path / "ingest.log"))
    monkeypatch.chdir(tmp_path)
    return home


def drop_scripts() -> None:
    """Forget every hook module a loader imported, and the scripts dir on sys.path."""
    for name in SCRIPT_MODULES:
        sys.modules.pop(name, None)
    while str(SCRIPTS) in sys.path:
        sys.path.remove(str(SCRIPTS))


def load_script(name: str) -> ModuleType:
    """plugin/scripts/<name>.py, freshly imported as a hook process would (scripts dir
    first on sys.path, so its own ``import config`` resolves). Call ``scratch_user``
    first and ``drop_scripts`` after; a test module wraps the two in a fixture:

        @pytest.fixture()
        def scratch(monkeypatch, tmp_path):
            yield scratch_user(monkeypatch, tmp_path)
            drop_scripts()
    """
    drop_scripts()
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The head uuid a built Claude Code transcript's boundary preserves.
HEAD_UUID = "0b7c2a4e-1111-4c22-9d33-000000000002"


def transcript(folder: Path, *, codex: bool, at: str | None) -> Path:
    """A transcript as each PreToolUse hook is handed one: a Codex rollout file whose
    last ``compacted`` line is stamped ``at``, or a Claude Code transcript whose last
    boundary preserves a head stamped ``at``. ``at=None``: a session that never
    compacted."""
    if codex:
        rows: list[dict[str, object]] = [
            {"timestamp": "2026-11-12T13:00:00Z", "type": "session_meta", "payload": {}},
            {"timestamp": "2026-11-12T13:05:00Z", "type": "response_item", "payload": {}},
        ]
        if at is not None:
            rows.append({"timestamp": at, "type": "compacted", "payload": {"message": "x"}})
    else:
        rows = [{"type": "user", "uuid": HEAD_UUID, "timestamp": at or "2026-11-12T13:05:00Z"}]
        if at is not None:
            rows.append(
                {
                    "type": "system",
                    "subtype": "compact_boundary",
                    "timestamp": "2026-11-12T14:30:00Z",
                    "compactMetadata": {"preservedSegment": {"headUuid": HEAD_UUID}},
                }
            )
    path = folder / ("rollout.jsonl" if codex else "transcript.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path
