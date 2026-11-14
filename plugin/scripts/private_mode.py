#!/usr/bin/env python3
# mypy: ignore-errors
"""Private mode toggle — take a Claude Code session off the record.

    private_mode.py on  <session_id>          # nothing from this session ever ingests
    private_mode.py off <session_id>          # stop skipping locally (row stays)
    private_mode.py off <session_id> --forget # ALSO delete the server row
    private_mode.py status <session_id>
    private_mode.py --session-end             # SessionEnd hook: reads the payload on stdin

The toggle itself is the client core's, the one every harness runs
(synapse_client/private_toggle.py: the two writes, why both are required, and what `off`
and `--forget` leave behind). This script hands it this plugin's server, credential and
marker folder. Only exit 0 from `on` means "off the record" is true.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from synapse_client.api import Client
from synapse_client.private_toggle import Toggle

PRIVATE_DIR = config.PRIVATE_DIR


def _toggle(program: str) -> Toggle:
    # The marker folder the Stop hook reads (ingest_hook._private_settings).
    settings = config.SETTINGS.replace(private_dir=Path(PRIVATE_DIR))
    return Toggle(Client(settings, config.USER_AGENT), program=program)


def main(argv: list[str], program: str = "private_mode.py") -> int:
    """``program`` is the name this was run under (`synapse private` passes its own), for
    the lines that name a command to run next."""
    return _toggle(program).main(argv, sys.stdin)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
