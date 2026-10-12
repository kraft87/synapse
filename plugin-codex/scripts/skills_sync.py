#!/usr/bin/env python3
# mypy: ignore-errors
"""Two-way sync of Synapse's skills with Codex's skill folders.

Reuses the Claude plugin's sync engine (``plugin/scripts/skills_sync.py``) with this
plugin's HTTP client, so the merge rules are identical on both hosts: newest edit wins,
a local delete never propagates, a skill retired, moved to another scope or no longer
served to this device leaves its folder when the copy there is one Synapse delivered, and
a push never shrinks the server's file set.

Two folders, the ones Codex itself reads:

* global scope <-> ``SYNAPSE_CODEX_SKILLS_DIR`` (default ``~/.agents/skills``, Codex's
  user-skills directory);
* ``project:<label>`` <-> ``<project root>/.agents/skills`` for the session's project
  (``--cwd``, labelled by the shared resolver ``plugin/scripts/synapse_project.py``; the
  root is the marker's folder or the git toplevel). Codex reads ``.agents/skills`` from
  the working directory up to the repository root, so a session in any subfolder sees
  them. Each skill folder pulled there is listed in the repository's .git/info/exclude.

Codex's bundled skills under ``~/.codex/skills/.system`` and any plugin caches live
outside both folders and never enter the sync.

Opt-in through ``skills_sync`` in ~/.synapse/config.json (``synapse login`` asks;
``SYNAPSE_SKILLS_SYNC=1`` in the environment works too), the same setting the Claude
plugin reads, so a machine running both plugins configures once. A non-blocking lock
serialises overlapping session starts; a second caller skips instead of double-syncing. Fail-open: any error
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


def project_pass(cwd: str | None, user_dir: Path) -> tuple[str, Path] | None:
    """``(scope, folder)`` for the session's project, or None: no cwd, a folder that is not
    on this disk, or a project whose folder is the user-skills folder itself (a session in
    the home directory; syncing one folder under two scopes flips every skill's scope)."""
    if not cwd:
        return None
    resolver = _load("synapse_project", _CLAUDE_SCRIPTS / "synapse_project.py")
    label, root = resolver.resolve(cwd)
    if not label or root is None:
        return None
    folder = root / ".agents" / "skills"
    try:
        if folder.resolve() == user_dir.resolve():
            return None
    except OSError:
        return None
    return f"project:{label}", folder


def run(
    client, target_dir: Path, lock_path: Path, project: tuple[str, Path] | None = None
) -> tuple[int, int, int] | None:
    """Sync ``target_dir`` against the global scope, then the project folder when one is
    given (``project_pass``). None when another sync holds the lock."""
    lock = _load("synapse_filelock", _CLAUDE_SCRIPTS / "synapse_filelock.py")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        try:
            lock.lock_exclusive(lf, blocking=False)
        except OSError:
            return None
        engine = _engine()
        result = engine._sync("global", target_dir, client=client)
        if project is not None:
            scope, folder = project
            more = engine._sync(scope, folder, client=client, git_exclude=True)
            result = tuple(a + b for a, b in zip(result, more, strict=True))
        return result


def _cwd_arg(argv: list[str]) -> str | None:
    return argv[argv.index("--cwd") + 1] if "--cwd" in argv[:-1] else None


def main() -> int:
    if not SKILLS_SYNC:
        return 0
    try:
        import common

        result = run(common, SKILLS_DIR, LOCK_PATH, project_pass(_cwd_arg(sys.argv), SKILLS_DIR))
    except Exception:
        return 0
    if result and any(result):
        pulled, pushed, removed = result
        print(f"skills: pulled {pulled}, pushed {pushed}, removed {removed} -> {SKILLS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
