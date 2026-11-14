#!/usr/bin/env python3
# mypy: ignore-errors
"""Private mode toggle for Codex sessions.

    private_mode.py on  <session_id>          # nothing from this session ever ingests
    private_mode.py off <session_id>          # stop skipping locally (row stays)
    private_mode.py off <session_id> --forget # ALSO delete the server row
    private_mode.py status <session_id>
    private_mode.py --session-end             # SessionEnd hook: payload on stdin

The toggle itself is the client core's, the one every harness runs
(synapse_client/private_toggle.py), against the same marker folder and server routes as
the Claude plugin's: Codex and Claude session ids share the private_sessions table. This
script hands it Codex's server, credential and marker folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from synapse_client.api import Client
from synapse_client.private_toggle import Toggle

PRIVATE_DIR = common.PRIVATE_DIR


def _toggle() -> Toggle:
    # The marker folder the Stop hook reads.
    settings = common.SETTINGS.replace(private_dir=Path(PRIVATE_DIR))
    return Toggle(Client(settings, common.USER_AGENT), program="private_mode.py")


def main(argv: list[str]) -> int:
    return _toggle().main(argv, sys.stdin)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
