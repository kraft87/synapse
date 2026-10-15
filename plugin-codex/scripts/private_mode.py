#!/usr/bin/env python3
"""Private mode toggle for Codex sessions.

Runs plugin/scripts/private_mode.py (usage, the two-write contract and --purge are
documented there) against the same marker dir and server routes: Codex and Claude
session ids share the private_sessions table. Codex's SessionEnd hook names this file
(`private_mode.py --session-end`), so it stays in place; common.py points the shared
script at the user-level settings only and gives it the Codex User-Agent.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: F401  (loads the shared config the script below imports)

_SHARED = Path(__file__).resolve().parents[2] / "plugin" / "scripts" / "private_mode.py"

if __name__ == "__main__":
    sys.argv[0] = str(_SHARED)
    runpy.run_path(str(_SHARED), run_name="__main__")
