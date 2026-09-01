#!/usr/bin/env python3
# mypy: ignore-errors
"""Device admin — approve, list, mint and revoke Synapse device credentials.

    device_admin.py list
    device_admin.py mint <label> [--full] [--projects a,b]
    device_admin.py revoke <surface_id>

Runs from a machine that ALREADY has full Synapse trust, because that is the only
credential these routes accept (schema 054). The root machine token is deliberately
refused: every machine that ever ran the plugin has held it, so a credential that
widespread must not be able to create new credentials. A 401 here is the design
working — use ``scripts/surface_admin.py`` on the database host if you have no trusted
machine yet.

There is no ``approve``. A machine gets its own token by ENROLLING, which needs an
OAuth/OIDC sign-in the allowlist admits (``synapse-login`` on that machine), and an
owner who just authenticated has already answered everything a second approval would
ask. ``mint`` covers the case with no browser anywhere: create the token here and carry
it over. It defaults to RESTRICTED, inheriting whatever project scope other restricted
devices already have; say ``--full`` or ``--projects`` to change that.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_USAGE = __doc__.split("\n\n")[1].strip()


def _flag(args: list[str], name: str) -> bool:
    return name in args


def _opt(args: list[str], name: str, default: str = "") -> str:
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def _grant(args: list[str]) -> dict:
    """The trust/allowed_projects fields to send. Omission is meaningful: an unstated
    allowlist lets a restricted surface inherit the scope other restricted surfaces
    already have, so don't send an empty list unless the operator asked for one."""
    body: dict = {"trust": "full" if _flag(args, "--full") else "restricted"}
    projects = [p.strip() for p in _opt(args, "--projects").split(",") if p.strip()]
    if projects:
        body["allowed_projects"] = projects
    return body


def _die(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def cmd_list() -> int:
    rows = config.get_json("/surfaces").get("surfaces") or []
    if not rows:
        print("no surfaces registered")
        return 0
    for r in rows:
        bits = [r["surface_id"], r["status"], r["trust"]]
        if r.get("label"):
            bits.append(f"label={r['label']}")
        if r.get("pair_code"):
            bits.append(f"code={r['pair_code']}")
        if r.get("allowed_projects"):
            bits.append("projects=" + ",".join(r["allowed_projects"]))
        if r.get("last_seen_at"):
            bits.append(f"seen={r['last_seen_at'][:19]}")
        print("  ".join(bits))
    return 0


def cmd_mint(args: list[str]) -> int:
    if not args or args[0].startswith("-"):
        return _die("usage: device_admin.py mint <label> [--full] [--projects a,b]")
    r = config.post_json("/surfaces/mint", {"label": args[0], **_grant(args)})
    if r.get("status") != "ok":
        return _die(r.get("detail") or "mint failed")
    s = r["surface"]
    scope = ",".join(s["allowed_projects"]) or "(no projects)"
    print(f"minted {s['surface_id']} trust={s['trust']} {scope}")
    # Only chance to see it — the server keeps the hash, not the token.
    print(f"token: {r['token']}")
    print("Set it as SYNAPSE_INGEST_TOKEN on the target machine. It is shown only once.")
    return 0


def cmd_revoke(args: list[str]) -> int:
    if not args:
        return _die("usage: device_admin.py revoke <surface_id>")
    r = config.request_json("DELETE", f"/surfaces/{args[0]}")
    if r.get("status") != "ok":
        return _die(r.get("detail") or "revoke failed")
    n = r.get("revoked", r.get("deleted", 0))
    print(f"revoked {args[0]} ({n} row(s)) — its token stops working on the next request")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    rest = argv[1:]
    try:
        if cmd == "list":
            return cmd_list()
        if cmd == "mint":
            return cmd_mint(rest)
        if cmd == "revoke":
            return cmd_revoke(rest)
    except Exception as e:
        if "401" in str(e):
            return _die(
                "unauthorized — these routes need a FULL-TRUST device token. The shared "
                "machine token cannot create credentials by design. Run this from an "
                "already-trusted machine, or use scripts/surface_admin.py on the DB host."
            )
        return _die(f"{cmd} failed: {e}")
    print(_USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
