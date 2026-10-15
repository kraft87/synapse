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

The client keeps whichever token you give it in one slot, `token` in
`~/.synapse/config.json` (env name `SYNAPSE_INGEST_TOKEN`), and one token covers ingest,
recall, skill sync, and MCP. In practice that value should be a device token.

That slot is the config file, not an environment variable. The MCP connection reads it
through a headers helper (`plugin/scripts/mcp_headers.py`) that Claude Code and Codex both
run with credential-named variables stripped from its environment, so a
`SYNAPSE_INGEST_TOKEN` exported from a shell profile, or from the `env` block of
`settings.json`, reaches only the hooks, which would then upload as one credential while
recall is served as another. To prevent that, an enrolled machine's hooks resolve to its
device token whatever the environment says. The enrollment record (`device` in
`config.json`; `device.json` in the Claude plugin's data dir before 0.20) keeps the token's
sha256 and the server that minted it, never the token. Session start prints one line when
an env value was overridden, naming where to remove it. Two limits:

- The device token only goes to the server that minted it (the record's `server`), and to
  wherever a Claude plugin config that holds that very token points, since that config's
  own MCP header sends it there anyway (a pre-0.20 plugin URL changed to the same server's
  new address keeps working). A record too old to name its server, with no such plugin
  config, goes by `config.json`'s URL. If the URL points anywhere else
  (`SYNAPSE_URL`/`SYNAPSE_INGEST_URL` in the environment, or a hand-edited `config.json`),
  an env token is used for that server, as before; with nothing else to send, no token is
  sent and session start says so in one line, as does the headers helper's log.
- Nothing in the CLI points a harness at another origin while the machine holds a
  credential (an enrollment, a pasted token, or a token Claude Code keeps for the plugin,
  including one in the macOS keychain): `synapse config url`, `synapse install` and
  `synapse update` refuse, naming the two ways out. `--same-server` says the new address
  is the same server (the enrollment record, even one too old to have named its server,
  moves with it); `synapse login --url <url> --reenroll` moves the machine, and saves the
  new URL only once that server has enrolled it.
- A token copy left in the Claude plugin's `settings.json` options by a pre-0.20 install
  feeds the plugin's static header, which goes to the plugin's URL. It is kept in step
  with a new token only while that URL is the new token's server: a move takes it out
  until `synapse login` has re-pointed the plugin, and a move made with `--no-install`
  leaves it out.
- A machine enrolled before the record carried a hash has none. Its plugin-config token is
  taken as the device token, because enrollment wrote it there and only a hand edit
  replaces it. `synapse login --reenroll` records the hash.

Installs from before 0.20 kept the token in the Claude plugin's config (`/plugin` > synapse >
Synapse token). That still works as a fallback, and the first `synapse login` copies it into
`config.json`: from `settings.json`, or from `~/.claude/.credentials.json` where Claude Code
kept a token typed into its install dialog. On macOS Claude Code keeps such a token in the
keychain, which nothing else can read; `synapse login` then says so, and
`synapse login --token` (or `--reenroll`) gives `config.json` its own copy.

## Getting a device token

**Self-enrollment (needs an IdP configured).** On the new machine, after installing the
client (`curl -sSL https://<your-synapse>/install | python3 -`):

```
synapse login          # or `! synapse-login` in the Claude prompt
```

It runs one device flow (RFC 8628): it prints a short code, you approve at
`github.com/login/device` (or your OIDC provider) from any device, and it polls until done. No
same-host browser, so it works on servers and headless boxes. That one approval both signs
you in and **enrolls** this machine: the server polls the IdP, reads the identity, checks the
same allowlist every other login clears, and mints a token belonging to this machine,
written into `~/.synapse/config.json`. The shared root token is never fetched.

`--browser` keeps the legacy loopback flow. Its OAuth token cannot enroll a machine, so it
is followed by the device-flow enrollment: two approvals. Prefer the default.

**Bootstrap (no IdP, or nothing trusted left).** On the server host:

```bash
docker compose exec mcp-server synapse-admin bootstrap "<label>"
```

Prints a full-trust device token once, because only its hash is stored. This is the local
quickstart's normal path, since a purely local stack usually has no IdP at all. Store it on
the machine with `synapse login --token`.

**Mint from an already-trusted machine.** For a box that will never run a browser flow (a
service, a container):

```
/synapse-devices mint "<label>" [--full] [--projects a,b]
```

Defaults to restricted, inheriting the project scope other restricted devices already have.
The token prints once. On the target machine, run `synapse login --token`. Use a
`SYNAPSE_INGEST_TOKEN` environment variable only where there is no config file to hold it,
such as a service or container.

The `label` is display only and is deliberately never matched against an existing row:
keying an enrollment on a self-reported name would reintroduce hostname spoofing through the
back door.

## Personal or work

`synapse login` asks for this machine's role (`role` in `~/.synapse/config.json`,
`SYNAPSE_MACHINE_ROLE`; before 0.20 the plugin's install prompt asked), and the answer
becomes the enrolled device's trust level. It is authoritative because the person who
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
| `GET /context` (board) | `audience='work-safe'` only | filtered to the project allowlist plus the machine's own uploads, NULL project excluded | n/a |
| `recall` / `POST /recall` | `audience='work-safe'` only | both legs filtered to the allowlist plus own uploads | only facts whose source episodes all sit in allowlisted projects (schema 056) |
| `recall_full_turns` | n/a | same as recall | n/a |
| `fetch(ids)` | `audience='work-safe'` only | same as recall | n/a |
| `fetch_session` | n/a | same as recall, on the probe and both row reads | n/a |
| `GET /preferences/top` | n/a | preferences from allowlisted projects only | n/a |
| `/config/*` | n/a | its own mirrored config only; proposals mined only from its own sessions | n/a |
| `/skills/*`, `skill://` resources | n/a | only `project:<p>` skills for allowlisted projects (any case); no global skill; publish only there; no retire, scope change or review | n/a |

`fetch` and `fetch_session` are filtered because ids are sequential integers: an unfiltered
drill-down would let a restricted caller enumerate exactly what the board withholds. A session
outside the allowlist reports the same "not indexed" answer an unknown id does, deliberately
indistinguishable.

An enrolled restricted machine always reads back the conversations it ingested itself,
even with an empty project allowlist: each ingested turn is stamped with the surface of the
device token that uploaded it (schema 057). Turns uploaded under the root token, and every
turn ingested before 057, carry no stamp and stay reachable only through the allowlist. A
turn rewritten by a different credential loses its stamp. Until 057 is applied the server
behaves exactly as before it: nothing is stamped, and the allowlist alone applies.

The session-start route `/preferences/top` and the config lane (`/config/*`) resolve the
machine from its token as well, never from a `surface` field in the request. The root token
is not a machine: it gets nothing from them, and a 403 on config writes. `/timeline/events`
(the git feeder's push) stamps each event with the pushing device's surface (schema 057,
probed; NULL for the root token and on a database without 057), so a restricted device's own
git events are served back to it on the board's digest. `POST /timeline/recent` no longer
exists: the board reads the digest directly.

Resolution never fails open. No credential, no row, a non-approved row, a missing table, an
unreachable database, a malformed row: all resolve to restricted with an empty allowlist. An
unknown caller serves nothing rather than everything.

Serving nothing is not silent. When an authenticated caller (root token, device token, or
OAuth sign-in) resolves to unknown, or to a restricted surface with no projects, the server
says so: a `WARNING:` line right under the board header, and the same text first in the
`warnings` list of `recall`, `recall_full_turns`, `fetch`, `fetch_session` and
`POST /recall`. The text names the cause and the fix (enroll with `synapse-login`; mint a
device a new credential with projects; grant an `oauth:<login>` identity with `PUT`), and
no ids. Full trust and restricted-with-projects get no notice, and neither does an open
server, where no credential would change the answer. Once schema 057 is applied, neither
does a restricted device with no projects: it reads back its own uploads, which is the
intended work-laptop setup, and minting it a new credential would leave those uploads
behind under the old one. Old clients show the notice too, since it is part of the
server's response rather than the plugin.

Note-writing follows the same line: a write from a live, approved restricted surface defaults
to `work-safe`, symmetric with what it may read, so notes written at work do not vanish from
the work board next session.

## Managing devices

```
/synapse-devices                       # list: id, trust, label, project scope, last seen
/synapse-devices mint "<label>"        # for a machine that cannot run a browser flow
/synapse-devices revoke <surface_id>   # effective on that device's very next request
```

These routes (mint, list, `PUT`, revoke) take the admin gate with no fallback: a full-trust
**device** token. The shared machine token is refused by design, so if the CLI reports 401
that is the reason. A revoke answers `{status, surface_id, revoked}` (the `deleted` alias of
0.16.x is gone).

**Break-glass is not a token.** `synapse-admin` (`list` / `mint` / `revoke` / `bootstrap`)
talks to Postgres over a direct DSN, so recovery needs shell access on the host, a strictly
higher bar than holding a bearer. It is a console script inside the image;
`scripts/surface_admin.py` remains as a host-side shim for a database you reach without the
container. It covers the three cases where nothing else
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
one session off the record, so nothing from it becomes memory from then on. It is a
plugin-side command, documented in
[plugin/README.md](../plugin/README.md#private-mode). Turns stored before the toggle stay
unless asked for: the server reports how many (`stored_turns`), and
`PUT /private-sessions/{id}?purge=1` (`private_mode.py on <id> --purge`) deletes them,
together with the preferences derived from them and the web pages captured in the session,
in one audited transaction. A purge is allowed to a full-trust device, or to a restricted
device purging a session whose every turn carries its own stamp (schema 057); anyone else,
the root token included, gets 403 with the flag still set and nothing deleted. A restricted
device can therefore purge only after 057 is applied.

## Memory-write spool

A `remember()` that cannot reach the server is queued locally and replayed when the server is
back, so a memory write is never silently lost. Details:
[plugin/README.md](../plugin/README.md#memory-write-spool).
