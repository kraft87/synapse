#!/usr/bin/env python3
"""Kept for Codex configs written before 0.20, whose http_headers_helper names this file.

The helper now lives in plugin/scripts/mcp_headers.py, shared with the Claude plugin;
this runs it with the same arguments. `synapse install codex` points config.toml at the
shared one.

Stdout is a credential channel consumed by Codex, not a diagnostic log.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[2] / "plugin" / "scripts" / "mcp_headers.py"

if __name__ == "__main__":
    sys.argv[0] = str(_SHARED)
    runpy.run_path(str(_SHARED), run_name="__main__")
