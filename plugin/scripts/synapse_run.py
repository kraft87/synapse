#!/usr/bin/env python3
# mypy: ignore-errors
"""Run a script of the installed Synapse client: ``synapse_run.py <path> [args...]``.

The installer copies this file to ~/.synapse/bin/synapse_run.py. Harness configs and the
`synapse` shims call it, never a versioned path: Codex trusts a hook by a hash of its
command text, so a command naming ~/.synapse/lib/<version>/ would switch itself off at
every update until the user re-trusted it in /hooks. This resolves the version when it
runs instead: ~/.synapse/lib/CURRENT names it, and the newest unpacked version stands in
when that file is missing.

``<path>`` is relative to the client root (``plugin-codex/hooks/session_start.py``,
``plugin/scripts/synapse_cli.py``) and must stay inside it. The script runs in this
process (runpy), so a hook still costs one interpreter start.

Its home is the directory above the one it sits in. Codex runs the headers helper with
an emptied environment, so the client scripts are pointed at that home through
SYNAPSE_HOME unless the caller set one. Stdlib only.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent


def _version_key(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.+-]", name))


def client_root(home: Path = HOME) -> Path | None:
    """The installed client directory (holding plugin/ and plugin-codex/), or None."""
    lib = home / "lib"
    try:
        current = (lib / "CURRENT").read_text(encoding="utf-8").strip()
    except OSError:
        current = ""
    if (
        current
        and "/" not in current
        and "\\" not in current
        and (lib / current / "plugin").is_dir()
    ):
        return lib / current
    try:
        versions = [d for d in lib.iterdir() if (d / "plugin").is_dir()]
    except OSError:
        return None
    return max(versions, key=lambda d: _version_key(d.name)) if versions else None


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: synapse_run.py <client-relative script> [args...]", file=sys.stderr)
        return 1
    root = client_root()
    if root is None:
        print(
            f"Synapse client not installed under {HOME / 'lib'}: run "
            "`curl -sSL https://<your-synapse>/install | python3 -`",
            file=sys.stderr,
        )
        return 1
    target = (root / argv[0]).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        print(f"synapse_run: no such client script: {argv[0]}", file=sys.stderr)
        return 1
    os.environ.setdefault("SYNAPSE_HOME", str(HOME))
    sys.argv = [str(target), *argv[1:]]
    sys.path[0] = str(target.parent)  # what `python <script>` would have put there
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
