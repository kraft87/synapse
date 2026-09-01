---
description: List, mint or revoke the per-device credentials that decide what each machine is served from Synapse memory.
---

Manage Synapse device trust (schema 054). Every machine authenticates with its OWN token; what it is served depends on that token's row, not on any hostname it reports.

Steps:
1. With no argument, or "list": `python3 "$CLAUDE_PLUGIN_ROOT/scripts/device_admin.py" list`. Report each surface's id, trust, label, project scope and last-seen.
2. To give a NEW machine access, the answer is usually not this command: that machine runs `! synapse-login`, signs in, and enrolls itself. Say that rather than minting on its behalf.
3. Mint only for a machine that can't run a browser sign-in anywhere (a headless box, a service): `… mint "<label>" [--full] [--projects a,b]`. Defaults to restricted, inheriting the project scope other restricted devices already have. The token prints once — tell the user to set it as `SYNAPSE_INGEST_TOKEN` there, and do not repeat it back later.
4. Revoke: `… revoke <surface_id>`. Effective on that device's very next request.

Never mint or widen a grant without the user's explicit say-so — this is the control that keeps personal memory off machines that shouldn't have it. If the CLI reports 401, say so plainly: these routes require a full-trust device token, and the shared machine token is refused by design; recovery without one is `scripts/surface_admin.py` on the database host.

Pass any argument the user gave straight through.
