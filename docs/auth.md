# Auth and device trust

Who can talk to your Synapse, and what each machine is served once it can. The install steps
themselves are in [docs/install.md](install.md); the design reasoning is
[ARCHITECTURE.md §7.4](../ARCHITECTURE.md#74-auth) and
[§7.5](../ARCHITECTURE.md#75-audience-scoping--credential-bound-surface-trust-schema-053--054).

## Two kinds of bearer token

- **Machine token** (`SYNAPSE_MACHINE_TOKEN`): one shared root bearer per deployment. It is
  the services' credential for `/ingest` and the internal write lanes, and it is verified by
  constant-time compare with no database read, so a Postgres blip never becomes an auth
  outage on the lane that repairs things. It identifies a *deployment*, not a machine, so it
  resolves to no surface and is served restricted. It also cannot enroll a device, mint one,
  list surfaces, or reach the dashboard API.
- **Device token**: one per machine, stored as a `sha256` hash in the `surfaces` table with
  a trust level and a project allowlist. This is what decides what a session is served.
  Revoking one clears its hash, so it matches no row at all and the row survives as the
  audit record.

The plugin puts whichever token you give it in `SYNAPSE_INGEST_TOKEN`, and one token covers
ingest, recall, skill sync, and MCP. In practice that value should be a device token.

## Getting a device token

**Self-enrollment (needs an IdP configured).** On the new machine:

```
! synapse-login        # in the Claude prompt, runs in-session, no LLM, streams live
```

It signs you in with the device flow (RFC 8628): it prints a short code, you approve at
`github.com/login/device` from any device, and it polls until done. No same-host browser, so
it works on servers and headless boxes (`--browser` keeps the legacy loopback flow). It then
runs a second device-flow approval that **enrolls** this machine: the server polls the IdP,
reads the identity, checks the same allowlist every other login clears, and mints a token
belonging to this machine, written back into the same config slot.

**Bootstrap (no IdP, or nothing trusted left).** On the server host:

```bash
docker compose exec mcp-server synapse-admin bootstrap "<label>"
```

Prints a full-trust device token once. This is the local quickstart's normal path, since a
purely local stack usually has no IdP at all.

**Mint from an already-trusted machine.** For a box that will never run a browser flow (a
service, a container):

```
/synapse-devices mint "<label>" [--full] [--projects a,b]
```

Defaults to restricted, inheriting the project scope other restricted devices already have.
The token prints once; set it as `SYNAPSE_INGEST_TOKEN` on the target machine.

The `label` is display only and is deliberately never matched against an existing row:
keying an enrollment on a self-reported name would reintroduce hostname spoofing through the
back door.

## Personal or work

The plugin's install prompt asks for this machine's role (`SYNAPSE_MACHINE_ROLE`), and the
answer becomes the enrolled device's trust level. It is authoritative because the person who
answered it is the person who just authenticated.

- **personal** (default): full trust. The single-user common case is a machine that should
  see everything, and a default that makes the normal path silently useless gets worked
  around rather than understood.
- **work**: restricted. Work-safe notes plus an allowlist of projects, so personal memory is
  never served on employer-owned hardware.

The narrow default lives one layer down: the *server* treats an unstated role as restricted.
Human says nothing means personal; software says nothing means restricted.

## What a restricted machine is served

| Path | Notes | Episodes / timeline | KG facts |
| --- | --- | --- | --- |
| `GET /context` (board) | `audience='work-safe'` only | filtered to the project allowlist, NULL project excluded | n/a |
| `recall` / `POST /recall` | `audience='work-safe'` only | both legs filtered to the allowlist | **leg skipped entirely** |
| `recall_full_turns` | n/a | same allowlist | n/a |
| `fetch(ids)` | `audience='work-safe'` only | same allowlist | n/a |
| `fetch_session` | n/a | allowlist on the probe and both row reads | n/a |

The KG leg is skipped rather than filtered because `kg_relationships` has no `project`
column, so serving zero facts is the only fail-closed answer available. `fetch` and
`fetch_session` are filtered because ids are sequential integers: an unfiltered drill-down
would let a restricted caller enumerate exactly what the board withholds. A session outside
the allowlist reports the same "not indexed" answer an unknown id does, deliberately
indistinguishable.

Resolution never fails open. No credential, no row, a non-approved row, a missing table, an
unreachable database, a malformed row: all resolve to restricted with an empty allowlist. An
unknown caller serves nothing rather than everything.

Note-writing follows the same line: a write from a live, approved restricted surface defaults
to `work-safe`, symmetric with what it may read, so notes written at work do not vanish from
the work board next session.

## Managing devices

```
/synapse-devices                       # list: id, trust, label, project scope, last seen
/synapse-devices mint "<label>"        # for a machine that cannot run a browser flow
/synapse-devices revoke <surface_id>   # effective on that device's very next request
```

These routes require a full-trust **device** token. The shared machine token is refused by
design, so if the CLI reports 401 that is the reason.

**Break-glass is not a token.** `synapse-admin` (`list` / `mint` / `revoke` / `bootstrap`)
talks to Postgres over a direct DSN, so recovery needs shell access on the database host, a
strictly higher bar than holding a bearer. It covers the three cases where nothing else
works: no IdP configured at all, the IdP down or the account locked, and no full-trust device
left.

```bash
docker compose exec mcp-server synapse-admin list
docker compose exec mcp-server synapse-admin mint "<label>" [--full] [--projects a,b]
docker compose exec mcp-server synapse-admin revoke <surface_id>
docker compose exec mcp-server synapse-admin bootstrap "<label>"
```

## GitHub OAuth and the claude.ai connector

The MCP server supports two auth modes at once, so machines and the claude.ai web connector
can share one server:

- **Machine bearer** — the static tokens above, sent by the plugin's hooks to `/ingest`,
  `/recall`, and `/mcp`. Headless boxes set it directly.
- **GitHub OAuth** — for the claude.ai web connector and the plugin's `synapse-login`. Login
  defaults to the device flow (RFC 8628): approve a short code at
  `github.com/login/device` from any device — no same-host browser — and the token is stored
  for you (`--browser` keeps the legacy loopback flow). Access is gated to an allowlist of
  GitHub users (`ALLOWED_GITHUB_USERS`); the GitHub OAuth App needs "Enable Device Flow" on.

Set `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `ALLOWED_GITHUB_USERS`, and
`SYNAPSE_PUBLIC_URL` (the public base URL advertised in OAuth discovery metadata).
`SYNAPSE_OAUTH_SIGNING_KEY` keeps issued tokens valid across server restarts.

A central instance can be exposed to claude.ai over a Cloudflare tunnel; the MCP server
handles auth itself, so no separate proxy is needed.

## OIDC instead of GitHub

Instead of GitHub, any OIDC-compliant IdP (Authelia, Keycloak, Pocket ID, ...) can back the
same three interactive flows — set `OIDC_CONFIG_URL` (the provider's
`/.well-known/openid-configuration`), `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, and
`ALLOWED_OIDC_USERS` (matched against `preferred_username`, then `email`). MCP discovery can
only advertise one authorization server, so this replaces GitHub for that deployment; leave
the OIDC vars unset to keep the GitHub default. Register both `{base_url}/auth/callback` and
`{base_url}/auth/callback/dash` as redirect URIs on the OIDC client, grant
`openid profile email offline_access`, and (for the MCP leg's allowlist) configure the IdP to
put `preferred_username`/`email` in the id_token — e.g. an Authelia claims policy. The device
flow needs the IdP to support the device authorization grant. Two knobs adapt to IdP
differences: `OIDC_SCOPES` (default `openid profile email offline_access`; trim
`offline_access` for IdPs that reject the scope, like Google, at the cost of short-lived
connector sessions) and `OIDC_USER_CLAIMS` (default `preferred_username,email`; the ordered
claims read for the user's identity).

## Private mode

Auth decides what a machine is *served*. Private mode decides what gets *captured*: it takes
one session off the record, so nothing from it ever becomes memory. It is a plugin-side
command, documented in
[plugin/README.md](../plugin/README.md#private-mode).

## Memory-write spool

A `remember()` that cannot reach the server is queued locally and replayed when the server is
back, so a memory write is never silently lost. Details:
[plugin/README.md](../plugin/README.md#memory-write-spool).
