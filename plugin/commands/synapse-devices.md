---
description: Approve, list, mint or revoke the per-device credentials that decide what each machine is served from Synapse memory.
---

Manage Synapse device trust (schema 054). Every machine authenticates with its OWN token; what it is served depends on that token's row, not on any hostname it reports.

Steps:
1. With no argument, or "list"/"pending": `python3 "$CLAUDE_PLUGIN_ROOT/scripts/device_admin.py" list`. Report pending devices first — those are the ones waiting on a human. Each shows the trust level it REQUESTED at install (`requests=full` for a personal machine, `requests=restricted` for a work one).
2. Approve a pending device by its pairing code (the enrolling machine printed the same code): `… approve <CODE>`.
   - With no flags this grants what the device requested, and a restricted device with no stated projects inherits the project scope other approved restricted devices already have. Say the resulting grant back to the user in plain words.
   - Override when the request looks wrong: `… approve <CODE> --restricted --projects work-thing,other-thing` or `… approve <CODE> --full`.
   - A device that requested nothing (older plugin) lands restricted with no projects, which serves nothing.
3. Mint a token for a machine you do NOT want holding the enrollment credential: `… mint "<label>" [--full|--projects a,b]`. The token prints once — tell the user to set it as `SYNAPSE_INGEST_TOKEN` on that machine, and do not repeat it back later.
4. Revoke: `… revoke <surface_id>`. Effective on that device's very next request.

Never approve, mint, or widen a grant without the user's explicit say-so — this is the control that keeps personal memory off machines that shouldn't have it. A device's own request is a suggestion from the machine being trusted, so confirm it out loud rather than treating it as settled. If the CLI reports 401, say so plainly: these routes require a full-trust device token, and the shared enrollment token is refused by design.

Pass any argument the user gave straight through.
