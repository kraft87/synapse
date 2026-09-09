#!/usr/bin/env python3
"""Surface admin — the first device's bootstrap, and the break-glass recovery path.

Installed as the ``synapse-admin`` console script, so it is on PATH inside the image::

    docker compose exec mcp-server synapse-admin bootstrap "<label>"
    docker compose exec mcp-server synapse-admin list
    docker compose exec mcp-server synapse-admin mint <label> [--full] [--projects a,b]
    docker compose exec mcp-server synapse-admin revoke <surface_id>

It lives in the package (not ``scripts/``) precisely so it ships in the runtime image:
the Dockerfile copies ingestion/, mcp_server/, dream/ and schema/, and nothing else.
``scripts/surface_admin.py`` is a thin shim over this module for host-side use.

**Bootstrap** is the ordinary first step of a fresh install. Every other path to a
device credential needs something already working: enrollment needs an identity provider
(``POST /surfaces/enroll`` polls it and enforces the login allowlist), and the HTTP admin
routes need a full-trust device token, which the root machine token deliberately cannot
substitute for. A brand-new local instance has neither, so it starts here.

The same command is the break-glass path for the situations where nothing works:

  * **No IdP** — a bearer-only deployment with no GitHub/OIDC configured, so there is no
    identity to anchor an enrollment to.
  * **IdP down or locked out** — Authelia is unreachable, or the account is locked.
  * **No trusted device left** — the only full-trust device was revoked, lost or wiped.

That path is deliberately NOT a token: it is a Postgres connection, which means shell
access on the database host (or ``docker compose exec``, which is the same authority).
That is a strictly higher bar than holding a bearer, and it is the right shape for a
recovery tool — hard to reach casually, always available to whoever owns the box.

DSN from ``--db-url``, else ``SYNAPSE_DB_URL`` (already set in every container's env).
``mint`` defaults to restricted, inheriting the project scope other approved restricted
devices already have; a recovery tool that over-grants by omission is worse than no
tool. ``bootstrap`` is the one exception and says so.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from ingestion.surfaces import (
    list_surfaces,
    mint_surface,
    revoke_surface,
)


def _grant(args: argparse.Namespace) -> tuple[str, list[str] | None]:
    """(trust, projects). ``None`` projects means "unstated", which lets a restricted
    surface inherit the scope other restricted surfaces already have — the usual intent
    for a second work machine, and never wider than what already exists."""
    projects = [p.strip() for p in (args.projects or "").split(",") if p.strip()]
    return ("full" if args.full else "restricted"), (projects or None)


def _print_token(out: dict[str, Any], *, bootstrap: bool = False) -> None:
    s = out["surface"]
    print(f"minted {s['surface_id']} trust={s['trust']} label={s.get('label') or '-'}")
    print(f"token: {out['token']}")
    print("Shown once — only the hash is stored.")
    if bootstrap:
        # The whole point of the command: a token nobody knows where to put is the same
        # as no token. Name the exact prompt, because that is where a new user is stuck.
        print()
        print("Paste it into the plugin's Synapse token prompt:")
        print("  /plugin > synapse > Synapse token   (SYNAPSE_INGEST_TOKEN)")
        print("Then start a new session — the board should stop coming back empty.")
    else:
        print("Set it as SYNAPSE_INGEST_TOKEN there.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="synapse-admin",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    ap_mint = sub.add_parser("mint")
    ap_mint.add_argument("label")
    ap_mint.add_argument("--full", action="store_true")
    ap_mint.add_argument("--projects", default="")

    ap_boot = sub.add_parser("bootstrap")
    ap_boot.add_argument("label")

    ap_rev = sub.add_parser("revoke")
    ap_rev.add_argument("surface_id")

    args = ap.parse_args(argv)
    if not args.db_url:
        print("no DSN — pass --db-url or set SYNAPSE_DB_URL", file=sys.stderr)
        return 2

    if args.cmd == "list":
        rows = list_surfaces(args.db_url)
        if not rows:
            print("no surfaces registered")
            return 0
        for r in rows:
            bits = [r["surface_id"], r["status"], r["trust"]]
            if r.get("label"):
                bits.append(f"label={r['label']}")
            if r["allowed_projects"]:
                bits.append("projects=" + ",".join(r["allowed_projects"]))
            if r.get("last_seen_at"):
                bits.append(f"seen={r['last_seen_at'][:19]}")
            print("  ".join(bits))
        return 0

    if args.cmd == "mint":
        trust, projects = _grant(args)
        _print_token(mint_surface(args.db_url, args.label, trust, projects))
        return 0

    if args.cmd == "bootstrap":
        # The deliberate exception to "default narrow": bootstrap exists to make a fresh
        # install usable (and to end a lockout), and a restricted token cannot reach the
        # admin routes, so a restricted bootstrap would leave you exactly as stuck.
        _print_token(mint_surface(args.db_url, args.label, "full", []), bootstrap=True)
        return 0

    if args.cmd == "revoke":
        n = revoke_surface(args.db_url, args.surface_id)
        print(f"revoked {args.surface_id} ({n} row(s))")
        return 0

    return 2  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":
    raise SystemExit(main())
