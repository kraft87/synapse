#!/usr/bin/env python3
"""Supply MCP auth from the same credential source as the Synapse hooks.

Stdout is a credential channel consumed by Codex, not a diagnostic log.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlsplit

from common import BASE_URL, TOKEN

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _allowed(url: str, gateway: bool) -> bool:
    """The saved credential goes only to its own Synapse — or, with ``--gateway``, to an
    MCP gateway on that same host (any port) or on loopback. The gateway relays the
    credential to that Synapse and nowhere else (docs/gateway.md)."""
    target, configured = urlsplit(url), urlsplit(BASE_URL)
    if (target.scheme, target.netloc) == (configured.scheme, configured.netloc):
        return True
    if not gateway or target.scheme not in ("http", "https"):
        return False
    return target.hostname in _LOOPBACK or (
        (target.scheme, target.hostname) == (configured.scheme, configured.hostname)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Synapse MCP credential helper")
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="URL is a Synapse MCP gateway on the Synapse host or loopback",
    )
    args = parser.parse_args()
    if not _allowed(args.url, args.gateway):
        print("Synapse MCP URL differs from the saved credential's server", file=sys.stderr)
        return 1
    if not TOKEN:
        print(
            "Synapse device credential missing: configure the Synapse plugin first", file=sys.stderr
        )
        return 1
    print(json.dumps({"Authorization": f"Bearer {TOKEN}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
