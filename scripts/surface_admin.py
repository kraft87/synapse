#!/usr/bin/env python3
"""Host-side shim for ``ingestion.surface_admin`` (the ``synapse-admin`` console script).

The logic moved into the package so it ships inside the runtime image, where the
first-device bootstrap actually runs::

    docker compose exec mcp-server synapse-admin bootstrap "<label>"

This file stays for host checkouts with no venv on PATH — same arguments, same output.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.surface_admin import main

if __name__ == "__main__":
    raise SystemExit(main())
