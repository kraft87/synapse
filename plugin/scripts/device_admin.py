#!/usr/bin/env python3
# mypy: ignore-errors
"""Device admin — approve, list, mint and revoke Synapse device credentials.

    device_admin.py list
    device_admin.py approve <PAIR_CODE> [--full|--restricted] [--projects a,b]
    device_admin.py mint <label> [--full] [--projects a,b]
    device_admin.py revoke <surface_id>

Runs from a machine that ALREADY has full Synapse trust, because that is the only
credential these routes accept (schema 054). The root/enrollment token is deliberately
refused: every machine holds it at install time, so if it could approve, a device could
approve itself and the whole trust-on-first-use gate would be decoration. A 401 from
this CLI on a fresh install is the design working, not a bug — bootstrap the first
device via ``synapse login`` + enrollment, or use ``scripts/surface_admin.py`` on the
database host.

``approve`` with no flags grants what the device REQUESTED at enrollment (its install
prompt asked "personal or work?"), which is printed next to the pair code on both ends —
so the common path is confirming a role you can already read, not retyping one. ``--full``
/ ``--restricted`` / ``--projects`` override it. A device that stated no role falls back
to restricted with an empty allowlist, which serves nothing: an unstated role must never
become a grant by omission.

``mint`` has no request to read, so it defaults to RESTRICTED with an empty allowlist.
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


def _grant(args: list[str], *, default_to_request: bool = False) -> dict:
    """The trust/allowed_projects fields to send, omitting whichever were not stated.

    Omission is meaningful on approve: the server fills the gap from what the device
    requested. So an unflagged approve must send NEITHER field rather than sending its
    own idea of a default and silently overriding the request.
    """
    body: dict = {}
    if _flag(args, "--full"):
        body["trust"] = "full"
    elif _flag(args, "--restricted") or not default_to_request:
        body["trust"] = "restricted"
    projects = [p.strip() for p in _opt(args, "--projects").split(",") if p.strip()]
    if projects or not default_to_request:
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
        if r.get("status") == "pending" and r.get("requested_trust"):
            bits.append(f"requests={r['requested_trust']}")
        if r.get("allowed_projects"):
            bits.append("projects=" + ",".join(r["allowed_projects"]))
        if r.get("last_seen_at"):
            bits.append(f"seen={r['last_seen_at'][:19]}")
        print("  ".join(bits))
    return 0


def cmd_approve(args: list[str]) -> int:
    if not args or args[0].startswith("-"):
        return _die(
            "usage: device_admin.py approve <PAIR_CODE> "
            "[--full|--restricted] [--projects a,b]   (default: what the device asked for)"
        )
    body = {"pair_code": args[0].strip().upper(), **_grant(args, default_to_request=True)}
    r = config.post_json("/surfaces/approve", body)
    if r.get("status") != "ok":
        return _die(r.get("detail") or "approve failed")
    s = r["surface"]
    scope = ",".join(s["allowed_projects"]) or "(no projects)"
    print(f"approved {s['surface_id']} ({s.get('label') or 'unnamed'}) trust={s['trust']} {scope}")
    return 0


def cmd_mint(args: list[str]) -> int:
    if not args or args[0].startswith("-"):
        return _die("usage: device_admin.py mint <label> [--full] [--projects a,b]")
    r = config.post_json("/surfaces/mint", {"label": args[0], **_grant(args)})
    if r.get("status") != "ok":
        return _die(r.get("detail") or "mint failed")
    s = r["surface"]
    print(f"minted {s['surface_id']} trust={s['trust']}")
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
        if cmd == "approve":
            return cmd_approve(rest)
        if cmd == "mint":
            return cmd_mint(rest)
        if cmd == "revoke":
            return cmd_revoke(rest)
    except Exception as e:
        if "401" in str(e):
            return _die(
                "unauthorized — these routes need a FULL-TRUST device token. The shared "
                "enrollment token cannot approve devices by design. Run this from an "
                "already-trusted machine, or use scripts/surface_admin.py on the DB host."
            )
        return _die(f"{cmd} failed: {e}")
    print(_USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
