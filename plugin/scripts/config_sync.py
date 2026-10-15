#!/usr/bin/env python3
# mypy: ignore-errors
"""SessionStart hook — mirror this machine's config files into Synapse (config lane).

PUSH-ONLY (V1 scaffolding). Auto-discovers the well-known config files and publishes each to the
server's /config/publish route (machine-token gated), tagged with this surface + a scope:
  * 'global'         — CLAUDE.md + rules/**/*.md under ~/.claude (CONFIG_DIR)
  * 'project:<name>' — CLAUDE.md, .claude/CLAUDE.md, .claude/rules/**/*.md under $CLAUDE_PROJECT_DIR
SYNAPSE_CONFIG_PATHS adds extra globs (under CONFIG_DIR, global scope). file_key = path relative to
that scope's root. The server stores it as the mirror the dashboard lists; the nightly config lane
mines transcripts for corrections and does not read it.

OFF by default — these files often carry personal instructions, so mirroring them off-box is
opt-in: set SYNAPSE_CONFIG_SYNC=1 to enable. DSN-free: HTTP only. Fail-open: never blocks or
fails session start; an unreachable/auth-less server is a no-op.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import synapse_project

_GLOBAL_GLOBS = ["CLAUDE.md", "rules/**/*.md"]
_PROJECT_GLOBS = ["CLAUDE.md", ".claude/CLAUDE.md", ".claude/rules/**/*.md"]


def _scan(root: Path, globs: list[str], scope: str) -> list[tuple[str, str, Path]]:
    """(scope, file_key, path) for every file matched under root, deduped by file_key."""
    seen: dict[str, Path] = {}
    for pat in globs:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                seen.setdefault(p.relative_to(root).as_posix(), p)
    return [(scope, key, p) for key, p in seen.items()]


def _targets() -> list[tuple[str, str, Path]]:
    out = _scan(config.CONFIG_DIR, _GLOBAL_GLOBS + config.CONFIG_PATHS, "global")
    proj = os.environ.get("CLAUDE_PROJECT_DIR")
    # The project ROOT's files (marker folder or git toplevel), whichever subfolder the
    # session started in: one (surface, scope, file_key) row per file, never two files.
    label, root = synapse_project.resolve(proj) if proj else (None, None)
    if proj and label:
        out += _scan(root or Path(proj), _PROJECT_GLOBS, f"project:{label}")
    return out


def _publish(scope: str, file_key: str, p: Path) -> None:
    config.post_json(
        "/config/publish",
        {
            "surface": config.SURFACE,
            "scope": scope,
            "file_key": file_key,
            "abs_path": str(p),
            "content": p.read_text(encoding="utf-8", errors="ignore"),
            "modified_at": datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat(),
        },
    )


def main() -> None:
    if not config.CONFIG_SYNC:
        return
    try:
        for scope, file_key, p in _targets():
            _publish(scope, file_key, p)
    except Exception:
        return  # fail-open: server/token/permission issues never break session start


if __name__ == "__main__":
    main()
