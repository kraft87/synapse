#!/usr/bin/env python3
# mypy: ignore-errors
"""Device enrollment — trade the shared enrollment credential for this machine's own.

Schema 054 made trust credential-bound: what a session is served depends on WHICH token
it presents, not on a hostname it claims. So every machine needs its own token, and
this is the ~one call that gets it.

Lifecycle, all of it idempotent and all of it fail-soft:

  1. First run on a machine — POST /surfaces/enroll with the hostname as a display
     label and the install prompt's role (``SYNAPSE_MACHINE_ROLE``: personal ⇒ asks for
     full trust, work ⇒ asks for restricted) as a REQUEST. The request is recorded and
     shown next to the pairing code so approving is a one-word confirmation; it grants
     nothing on its own. The server mints a token and answers one of two ways:
       * ``approved`` — this Synapse had no approved device yet, so the first one in
         bootstraps at full trust. Reachable exactly once per deployment.
       * ``pending``  — a 6-character pair code comes back. The token is real but
         authenticates as nothing until an already-trusted machine approves that code.
  2. The minted token is written to SYNAPSE_INGEST_TOKEN via the same
     ``write_user_config`` path ``synapse login`` uses, so the MCP server's
     Authorization header picks it up with no other change anywhere.
  3. Later runs read the local record and do nothing.

Never raises, never prints on the happy path: it runs from SessionStart, ahead of the
board fetch, and a machine that cannot reach Synapse must still get a session.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def ensure_enrolled(timeout: float = 10.0) -> dict:
    """Enroll this device if it hasn't been, and return the local enrollment record.

    Shape: ``{"surface_id", "status", "pair_code"?, "label"}``, or ``{}`` when
    enrollment has not happened and could not happen (no token configured, server
    unreachable, a server too old to have the route).
    """
    state = config.read_device_state()
    if state.get("surface_id"):
        return state
    if not config.INGEST_TOKEN:
        # Nothing to enroll WITH. An open/local server needs no credential at all, and
        # a gated one needs `synapse login` first — either way, silence is correct.
        return {}
    try:
        payload = {"label": config.SURFACE}
        asked = config.requested_trust()
        if asked:
            payload["requested_trust"] = asked
        r = config.post_json("/surfaces/enroll", payload, timeout=timeout)
    except Exception:
        # Server down, pre-054, or the credential was already revoked. Retry next
        # session; do NOT record a half-state that would suppress the retry.
        return {}
    token = r.get("token")
    surface = r.get("surface") or {}
    if not token or not surface.get("surface_id"):
        return {}

    # Order matters: persist the credential FIRST. If the process dies between the two
    # writes, a machine with a stored token and no record re-enrolls (one stray pending
    # row, harmless); a machine with a record and no token is locked out.
    config.write_user_config("SYNAPSE_INGEST_TOKEN", token)
    state = {
        "surface_id": surface["surface_id"],
        "status": r.get("status") or surface.get("status") or "pending",
        "pair_code": r.get("pair_code") or surface.get("pair_code"),
        "label": config.SURFACE,
        "role": config.MACHINE_ROLE,
    }
    config.write_device_state(state)
    return state


def mark_approved() -> None:
    """Record that this device is serving — i.e. an authenticated call just succeeded.

    The client is never TOLD it was approved; it finds out by being served. Clearing
    the pair code here is what stops the pairing block from printing forever.
    """
    state = config.read_device_state()
    if state.get("status") == "approved":
        return
    if not state.get("surface_id"):
        return
    state["status"] = "approved"
    state.pop("pair_code", None)
    config.write_device_state(state)


def pairing_block(state: dict) -> str:
    """The block a PENDING device prints instead of a board.

    An unapproved device is served nothing, so without this the user sees an empty
    session start and no reason for it. The pair code appears here AND on the trusted
    machine's board — an approval where only one side shows the code cannot be checked.
    """
    code = state.get("pair_code") or "?"
    label = state.get("label") or "this machine"
    role = state.get("role") or "personal"
    return (
        "[Synapse — device pending approval]\n"
        f"This machine ({label}) enrolled as a {role} machine but is not approved yet, "
        "so no memory is being served here.\n"
        f"Pairing code: {code}\n"
        "From a machine that already has full Synapse trust, run:\n"
        f"  /synapse-devices approve {code}\n"
        "The same code is listed on that machine's Synapse board. Until then recall, "
        "the board, and remember all return nothing on this host."
    )


def main() -> None:
    """CLI: `enroll.py` enrolls quietly; `enroll.py --status` prints the record."""
    state = ensure_enrolled()
    if "--status" in sys.argv:
        if not state:
            print("not enrolled (no credential configured, or Synapse unreachable)")
        else:
            print(f"surface_id={state['surface_id']} status={state.get('status')}")
            if state.get("pair_code"):
                print(f"pair_code={state['pair_code']}")


if __name__ == "__main__":
    main()
