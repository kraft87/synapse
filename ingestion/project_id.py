from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def _read_dot_file(cwd: Path) -> str | None:
    p = cwd / ".memory-project"
    try:
        val = p.read_text().strip()
        return val if val else None
    except FileNotFoundError:
        return None


def _first_commit_hash(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def resolve_project_id(cwd: Path | str | None = None) -> str:
    """Resolve a stable project ID for the given working directory.

    Priority:
    1. .memory-project file in cwd
    2. First commit hash of git repo (immutable, survives renames)
    3. MEMORY_PROJECT env var
    4. Deterministic hash of hostname + cwd
    """
    path = Path(cwd) if cwd else Path.cwd()

    if (dot := _read_dot_file(path)) is not None:
        return dot

    if (git := _first_commit_hash(path)) is not None:
        return git

    if env := os.environ.get("MEMORY_PROJECT", "").strip():
        return env

    hostname = os.uname().nodename
    cwd_str = str(path.resolve())
    digest = hashlib.sha256(f"{hostname}:{cwd_str}".encode()).hexdigest()[:12]
    return digest
