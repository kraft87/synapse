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


def main() -> int:
    parser = argparse.ArgumentParser(description="Synapse MCP credential helper")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    target, configured = urlsplit(args.url), urlsplit(BASE_URL)
    if (target.scheme, target.netloc) != (configured.scheme, configured.netloc):
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
