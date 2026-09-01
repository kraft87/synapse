#!/usr/bin/env python3
"""Break-glass surface admin — list, mint and revoke devices over a direct DSN.

Every other path to a device credential needs something working: enrollment needs the
IdP (`POST /surfaces/enroll` polls it and enforces the login allowlist), and the HTTP
admin routes need a full-trust device token, which the root machine token deliberately
cannot substitute for. That leaves the situations where nothing works:

  * **No IdP** — a bearer-only deployment with no GitHub/OIDC configured, so there is no
    identity to anchor an enrollment to.
  * **IdP down or locked out** — Authelia is unreachable, or the account is locked.
  * **No trusted device left** — the only full-trust device was revoked, lost or wiped.

So the break-glass path is deliberately NOT a token: it is a Postgres connection, which
means shell access on the database host. That is a strictly higher bar than holding a
bearer, and it is the right shape for a recovery tool — hard to reach casually, always
available to whoever actually owns the box.

Usage (run on the DB host, or anywhere with the DSN):
    scripts/surface_admin.py list
    scripts/surface_admin.py mint <label> [--full] [--projects a,b]
    scripts/surface_admin.py revoke <surface_id>
    scripts/surface_admin.py bootstrap <label>     # mint a full-trust device, unlock

DSN from --db-url, else SYNAPSE_DB_URL. ``mint`` defaults to restricted, inheriting the
project scope other approved restricted devices already have; a recovery tool that
over-grants by omission is worse than no tool. ``bootstrap`` is the one exception and
says so.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _print_token(out: dict[str, Any]) -> None:
    s = out["surface"]
    print(f"minted {s['surface_id']} trust={s['trust']} label={s.get('label') or '-'}")
    print(f"token: {out['token']}")
    print("Shown once — only the hash is stored. Set it as SYNAPSE_INGEST_TOKEN there.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        # The deliberate exception to "default narrow": bootstrap exists to end a
        # lockout, and a restricted token cannot reach the admin routes, so a restricted
        # bootstrap would leave you exactly as locked out as before.
        _print_token(mint_surface(args.db_url, args.label, "full", []))
        return 0

    if args.cmd == "revoke":
        n = revoke_surface(args.db_url, args.surface_id)
        print(f"revoked {args.surface_id} ({n} row(s))")
        return 0

    return 2  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":
    raise SystemExit(main())
