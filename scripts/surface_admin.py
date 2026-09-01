#!/usr/bin/env python3
"""Break-glass surface admin — approve, mint, list and revoke devices over a direct DSN.

The HTTP admin routes require a FULL-TRUST DEVICE token, and the root/enrollment token
is refused there on purpose (every machine holds it, so if it could approve, a device
could approve itself). That leaves two situations with no way in over HTTP:

  * **First deploy** — no device exists yet, so no device token exists to approve with.
    (Enrollment's bootstrap branch usually covers this: the first device in is approved
    automatically. This is the fallback when it doesn't, e.g. rows were pre-created.)
  * **Locked out** — the only full-trust device was revoked, lost, or wiped.

So the break-glass path is deliberately NOT a token: it is a Postgres connection, which
means shell access on the database host. That is a strictly higher bar than holding a
bearer, and it is the right shape for a recovery tool — hard to reach casually, always
available to whoever actually owns the box.

Usage (run on the DB host, or anywhere with the DSN):
    scripts/surface_admin.py list
    scripts/surface_admin.py approve <PAIR_CODE> [--full] [--projects a,b]
    scripts/surface_admin.py mint <label> [--full] [--projects a,b]
    scripts/surface_admin.py revoke <surface_id>
    scripts/surface_admin.py bootstrap <label>     # mint a full-trust device, unlock

DSN from --db-url, else SYNAPSE_DB_URL. ``approve`` with no flags grants what the device
requested at enrollment (shown by ``list``); ``--full``/``--restricted``/``--projects``
override. A device that requested nothing falls back to restricted with an empty
allowlist — a recovery tool that over-grants by omission is worse than no tool.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.surfaces import (
    NoSuchPairCode,
    approve_surface,
    list_surfaces,
    mint_surface,
    revoke_surface,
)


def _grant(args: argparse.Namespace) -> tuple[str, list[str]]:
    projects = [p.strip() for p in (args.projects or "").split(",") if p.strip()]
    return ("full" if args.full else "restricted"), projects


def _approve_args(args: argparse.Namespace) -> tuple[str | None, list[str] | None]:
    """None means "not stated" — which is what lets approve_surface fall back to the
    device's own request. Sending a default here would silently override it."""
    trust = "full" if args.full else ("restricted" if args.restricted else None)
    projects = [p.strip() for p in (args.projects or "").split(",") if p.strip()]
    return trust, (projects or None)


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

    ap_ok = sub.add_parser("approve")
    ap_ok.add_argument("pair_code")
    ap_ok.add_argument("--full", action="store_true")
    ap_ok.add_argument("--restricted", action="store_true")
    ap_ok.add_argument("--projects", default="")

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
            if r.get("pair_code"):
                bits.append(f"code={r['pair_code']}")
            if r["status"] == "pending" and r.get("requested_trust"):
                bits.append(f"requests={r['requested_trust']}")
            if r["allowed_projects"]:
                bits.append("projects=" + ",".join(r["allowed_projects"]))
            if r.get("last_seen_at"):
                bits.append(f"seen={r['last_seen_at'][:19]}")
            print("  ".join(bits))
        return 0

    if args.cmd == "approve":
        trust, projects = _approve_args(args)
        try:
            s = approve_surface(args.db_url, args.pair_code, trust, projects)
        except NoSuchPairCode as e:
            print(str(e), file=sys.stderr)
            return 1
        scope = ",".join(s["allowed_projects"]) or "(no projects)"
        print(
            f"approved {s['surface_id']} trust={s['trust']} {scope} label={s.get('label') or '-'}"
        )
        return 0

    if args.cmd == "mint":
        trust, projects = _grant(args)
        _print_token(mint_surface(args.db_url, args.label, trust, projects))
        return 0

    if args.cmd == "bootstrap":
        # The deliberate exception to "default narrow": bootstrap exists to end a
        # lockout, and a restricted token cannot reach the admin routes, so a
        # restricted bootstrap would leave you exactly as locked out as before.
        _print_token(mint_surface(args.db_url, args.label, "full", []))
        return 0

    if args.cmd == "revoke":
        n = revoke_surface(args.db_url, args.surface_id)
        print(f"revoked {args.surface_id} ({n} row(s))")
        return 0

    return 2  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":
    raise SystemExit(main())
