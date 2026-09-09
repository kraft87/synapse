#!/usr/bin/env python3
# mypy: ignore-errors
"""Device enrollment — get THIS machine its own Synapse credential.

Schema 054 made trust credential-bound: what a session is served depends on WHICH token
it presents, not on a hostname it claims. So every machine needs its own token, and the
only way to get one is to prove you are the owner.

Enrollment is the IdP's device flow (RFC 8628), the same lane `synapse login` uses:

  1. `POST /device/code` — the server asks the IdP for a short user code.
  2. We print the verification URL and the code. The human approves in a browser on ANY
     device, phone included; no browser is needed on this machine.
  3. `POST /surfaces/enroll {device_code, label, trust}` — the server polls the IdP,
     reads the identity, checks it against the allowlist, and mints a device token
     scoped to the role this install declared. Approved immediately: the person who just
     authenticated IS the authority for what this machine is, so there is nothing left
     to confirm from somewhere else.
  4. The minted token is written to SYNAPSE_INGEST_TOKEN via the same
     ``write_user_config`` path `synapse login` uses, so the MCP server's Authorization
     header picks it up with no other change anywhere.

Interactive by nature — it prints a code and waits for a human — so the SessionStart
hook never runs it. The hook only reports that this machine is not enrolled; `synapse
login` (or `enroll.py` run directly) does the flow.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

#: Poll ceiling. The IdP's own device_code expiry (`expires_in`, typically 15 min) wins
#: when it is shorter; this is the backstop for a server that reports neither.
_MAX_WAIT_S = 900


def is_enrolled() -> bool:
    """True once this machine holds a token minted for it (as opposed to a pasted one)."""
    return bool(config.read_device_state().get("surface_id"))


#: The command that mints a device token on a server with no identity provider — a
#: local `docker compose` install, which is the common case for a fresh clone. Kept as
#: one string so the wording can't drift between the hook, `synapse-login` and enroll.
BOOTSTRAP_CMD = 'docker compose exec mcp-server synapse-admin bootstrap "<label>"'


def bootstrap_hint(prefix: str = "") -> str:
    """What to do when this server has no IdP: there is nothing to sign in to.

    Printed instead of "enrollment unavailable: unknown", which reads like a broken
    server when it is in fact a correctly-configured local one that never had a login.
    """
    return (
        f"{prefix}No sign-in is configured on this Synapse, so there is nothing to approve.\n"
        "Mint this machine a token on the server instead:\n"
        f"    {BOOTSTRAP_CMD}\n"
        "then paste it into /plugin > synapse > Synapse token (SYNAPSE_INGEST_TOKEN)."
    )


def restricted_block() -> str:
    """What the SessionStart hook prints when the server answers but serves nothing.

    The 401 case has an explainer (below); this is the QUIET one: an unenrolled caller
    on an open server, or one holding the shared root token, gets a perfectly successful
    200 with an empty board. Same outcome as a 401 — no memory — with none of the signal,
    which is how a new install ends up looking like "Synapse just has nothing in it".
    """
    return (
        "[Synapse — this machine is served nothing]\n"
        "The server answered, but this caller is `restricted` with no projects: it holds no "
        "device credential of its own (schema 054), so the board and recall come back empty.\n"
        "Fix it either way:\n"
        "  - Server with a sign-in configured: run `! synapse-login` in this session.\n"
        "  - Local server, no sign-in: run\n"
        f"        {BOOTSTRAP_CMD}\n"
        "    on the server, then paste the token into /plugin > synapse > Synapse token."
    )


def not_enrolled_block() -> str:
    """What the SessionStart hook prints when this machine has no device credential.

    A machine with no credential of its own is served nothing, and an unexplained empty
    session start is worse than a restricted one — so say the reason and the one command
    that fixes it.
    """
    return (
        "[Synapse — this machine is not enrolled]\n"
        "No memory is being served here: since schema 054 every machine authenticates "
        "with its own device token.\n"
        "Run `! synapse-login` in this session (or `synapse-login` in a terminal) and "
        "approve the sign-in on any device."
    )


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST that returns the parsed body for BOTH success and 4xx.

    The enroll route answers 202 while the human has not approved yet and 403 once the
    IdP says no; both carry the reason in the body, so an exception on non-2xx would
    throw away exactly the information the poll loop needs.
    """
    try:
        return config.post_json(path, payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except Exception:
            body = {}
        # An empty body (a bare 404 from a server that never mounted the route) parses
        # to {}, which would read downstream as "no reason given". Keep the status.
        return body if isinstance(body, dict) and body else {"error": f"http_{e.code}"}


def enroll(interactive: bool = True) -> dict:
    """Run the device flow and enroll this machine. Returns the local record, or ``{}``.

    Never raises: it is called from `synapse login`, where a failure should print a
    reason and leave the machine exactly as it was, not traceback.
    """
    state = config.read_device_state()
    if state.get("surface_id"):
        return state

    try:
        start = _post("/device/code", {})
    except Exception as e:
        print(f"enrollment unavailable: {e}", file=sys.stderr)
        return {}
    if "user_code" not in start:
        reason = start.get("error", "unknown")
        if reason in ("no_idp", "no_machine_token", "http_404"):
            # Nothing to sign in to — not a failure of this machine's enrollment, and
            # the user needs the OTHER remedy, not a retry.
            print(bootstrap_hint(), file=sys.stderr)
            return {}
        print(f"enrollment unavailable: {reason}", file=sys.stderr)
        if reason == "device_flow_disabled":
            print("Turn on 'Enable Device Flow' in the OAuth App settings.", file=sys.stderr)
        return {}

    verify = start.get("verification_uri") or ""
    complete = start.get("verification_uri_complete")
    interval = max(1, int(start.get("interval") or 5))
    deadline = time.time() + min(int(start.get("expires_in") or _MAX_WAIT_S), _MAX_WAIT_S)

    if interactive:
        print("\nEnrolling this machine. Approve on ANY device:\n", file=sys.stderr)
        print(f"    {verify}", file=sys.stderr)
        print(f"    code:  {start['user_code']}\n", file=sys.stderr)
        if complete:
            print(f"(or open the direct link: {complete})\n", file=sys.stderr)
        print(f"This machine will enroll as: {config.MACHINE_ROLE}", file=sys.stderr)
        print("Waiting for approval...", file=sys.stderr)

    body = {
        "device_code": start["device_code"],
        # DISPLAY ONLY server-side — it names nothing and grants nothing.
        "label": config.SURFACE,
        # The install prompt's answer. Authoritative because the person about to
        # authenticate is the one who gave it.
        "trust": config.requested_trust() or "restricted",
    }
    while time.time() < deadline:
        time.sleep(interval)
        r = _post("/surfaces/enroll", body)
        if r.get("status") == "ok" and r.get("token"):
            return _persist(r)
        reason = r.get("error") or r.get("detail") or ""
        if reason == "slow_down":
            interval += 5
            continue
        if reason == "authorization_pending":
            continue
        print(f"enrollment failed: {reason or 'no token returned'}", file=sys.stderr)
        return {}

    print("enrollment failed: timed out waiting for approval.", file=sys.stderr)
    return {}


def _persist(r: dict) -> dict:
    """Store the minted credential, then the local record. Order matters on a crash: a
    machine with a token and no record re-enrolls harmlessly, while a machine with a
    record and no token is locked out of a credential it can never obtain again."""
    surface = r.get("surface") or {}
    config.write_user_config("SYNAPSE_INGEST_TOKEN", r["token"])
    state = {
        "surface_id": surface.get("surface_id", ""),
        "trust": surface.get("trust", ""),
        "allowed_projects": surface.get("allowed_projects") or [],
        "label": config.SURFACE,
        "login": r.get("login", ""),
    }
    config.write_device_state(state)
    scope = ", ".join(state["allowed_projects"]) or "no projects"
    detail = "the full corpus" if state["trust"] == "full" else f"work-safe notes + {scope}"
    print(f"\nEnrolled as {state['surface_id']} ({state['trust']}) — this machine sees {detail}.")
    return state


def main() -> int:
    if "--status" in sys.argv:
        state = config.read_device_state()
        if not state.get("surface_id"):
            print("not enrolled — run `synapse-login`")
            return 1
        print(f"surface_id={state['surface_id']} trust={state.get('trust')}")
        return 0
    return 0 if enroll() else 1


if __name__ == "__main__":
    raise SystemExit(main())
