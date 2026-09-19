#!/usr/bin/env python3
# mypy: ignore-errors
"""Two-way sync of Synapse's global skills with Codex's user-skills folder.

Reuses the Claude plugin's sync engine (``plugin/scripts/skills_sync.py``) with this
plugin's HTTP client, so the merge rules are identical on both hosts: newest edit wins,
deletes never propagate, a push never shrinks the server's file set.

Only ``SYNAPSE_CODEX_SKILLS_DIR`` (default ``~/.agents/skills``, Codex's user-skills
directory) is scanned. Codex's bundled skills under ``~/.codex/skills/.system`` and any
plugin caches live outside that folder and never enter the sync. Global scope only:
project skills stay Claude-side because ``<repo>/.claude/skills`` is committed to git.

Opt-in through ``SYNAPSE_SKILLS_SYNC=1``, the same option the Claude plugin reads, so a
machine running both plugins configures once. A non-blocking lock serialises overlapping
session starts; a second caller skips instead of double-syncing. Fail-open: any error
exits 0 with nothing printed, and session start is never blocked.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import _cfg

_CLAUDE_SCRIPTS = Path(__file__).resolve().parents[2] / "plugin" / "scripts"

SKILLS_SYNC = _cfg("SYNAPSE_SKILLS_SYNC", "0") not in ("", "0")
SKILLS_DIR = Path(os.path.expanduser(_cfg("SYNAPSE_CODEX_SKILLS_DIR", "~/.agents/skills")))
LOCK_PATH = Path(os.path.expanduser("~/.synapse/codex_skills_sync.lock"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    # Loaded under a private name: this file shares its basename with the engine, so a plain
    # ``import skills_sync`` would resolve back to us. The engine imports ``config`` from its
    # own directory, hence the path insert.
    if str(_CLAUDE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_CLAUDE_SCRIPTS))
    return _load("synapse_claude_skills_sync", _CLAUDE_SCRIPTS / "skills_sync.py")


def run(client, target_dir: Path, lock_path: Path) -> tuple[int, int] | None:
    """Sync ``target_dir`` against the global scope. None when another sync holds the lock."""
    lock = _load("synapse_filelock", _CLAUDE_SCRIPTS / "synapse_filelock.py")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        try:
            lock.lock_exclusive(lf, blocking=False)
        except OSError:
            return None
        return _engine()._sync("global", target_dir, client=client)


def main() -> int:
    if not SKILLS_SYNC:
        return 0
    try:
        import common

        result = run(common, SKILLS_DIR, LOCK_PATH)
    except Exception:
        return 0
    if result and result != (0, 0):
        print(f"skills: pulled {result[0]}, pushed {result[1]} -> {SKILLS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
