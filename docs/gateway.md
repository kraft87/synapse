# MCP gateway (pilot)

One MCP connection that gives Claude Code and Codex the same Synapse memory, the same
`skill://` skills, and the same Exa/Firecrawl web research. The gateway (`mcp_gateway/`) is a
separate FastMCP 3.4.2 service that runs beside Synapse; Synapse itself is unchanged except for
one small route, `GET /auth/whoami`.

```
Claude Code ─┐                    ┌─ Synapse /mcp      (caller's OWN device bearer)
             ├─ gateway /mcp ─────┼─ Exa MCP           (gateway's Exa key)
Codex ───────┘   (device bearer)  └─ Firecrawl MCP     (gateway's Firecrawl key)
```

## What a client sees

| Name | From | Who sees it |
| --- | --- | --- |
| `synapse_<tool>` (`synapse_recall`, `synapse_fetch`, `synapse_remember`, …) | Synapse | every approved device; Synapse filters the data per device as always |
| `exa_<tool>`, `firecrawl_<tool>` | research upstreams, names as they publish them | full-trust devices (see `RESEARCH_TRUST`) |
| `list_resources`, `read_resource` | FastMCP `ResourcesAsTools` | every approved device |
| `skill://<name>/SKILL.md`, `/_manifest`, `/<file>` | Synapse's PG skills, URIs unchanged | whatever Synapse serves that device |
| `skill://gateway-research/SKILL.md` | bundled with the gateway | same devices as the research tools |

Names are stable: each upstream's tools get a fixed prefix and keep their upstream name,
description, input schema and output schema unchanged. Upstream tool errors (`isError`) and
JSON-RPC errors pass through verbatim. Transport failures become a generic
`"<upstream> upstream request failed (ConnectError)"` because the raw text could quote a
keyed URL. Resource URIs are not prefixed, so stock `sync_skills` and the plugins' skill
contract are unaffected.

## Identity and security model

- **The gateway has no Synapse credential and no database DSN.** A caller authenticates with
  its Synapse device token. The gateway presents that same token to Synapse's
  `GET /auth/whoami` and admits the caller only if Synapse answers
  `{"kind": "device", "trust": …}`. Pending, revoked and unknown tokens are rejected (401),
  and so is the root/enrollment token, which names a deployment, not a device. Anything
  unexpected, including Synapse being unreachable, also fails closed.
- **Every Synapse request carries the caller's own bearer**, set per request. Synapse applies
  its normal device rules to it: restricted scoping, project allowlists, the KG leg skipped,
  hidden tools. The gateway never widens that. A tool Synapse leaves out of a device's
  `tools/list` is not callable through the gateway either.
- **Credentials only go to the upstream they belong to.** The upstream clients are built with
  FastMCP's inbound-header forwarding turned off. That matters because the stock
  `ProxyClient`/`create_proxy` forwards the caller's `Authorization` header to every upstream,
  Exa and Firecrawl included. Exa and Firecrawl see only their own configured key. Synapse
  sees only the caller's bearer.
- **No shared state across identities.** The stock `ProxyProvider` keeps one component cache
  for all sessions, so the gateway doesn't use it. Synapse listings are cached per identity,
  keyed by `sha256(token)`, and research listings are shared because the credential behind
  them is the gateway's own. Every upstream operation opens a fresh client, so no HTTP pool,
  cookie jar or MCP session is reused between callers, and the gateway runs stateless HTTP.
- **Revocation.** Memory access ends on a revoked device's next call, because Synapse checks
  the bearer again on every request. Research access can outlive revocation by at most
  `AUTH_CACHE_TTL` (default 30s).
- **Research policy.** By default only full-trust devices see research tools and the research
  skill. A restricted, work-profile device therefore can't send queries through the owner's
  paid keys or pick up tools it wasn't granted. Set `RESEARCH_TRUST=restricted` to allow any
  approved device.
- **No credentials in logs.** Keys come from env or secret files and never from the repo.
  Logs show upstreams as `scheme://host/…`. A log-record scrubber removes configured keys from
  every logger, tracebacks included, and httpx request logging is capped at WARNING. Don't
  run with DEBUG logging in production: the `mcp` library then logs full request and response
  payloads, which include memory content.
- The gateway adds no raw HTTP forwarding routes. Its only custom route is an unauthenticated
  `/health` that reports liveness and makes no upstream call.

## Degradation

Each upstream listing runs under a discovery timeout (research defaults to 5s, Synapse to 10s).
If a listing fails, the gateway serves the last good snapshot. If there isn't one, it logs a
redacted warning and leaves that upstream out. A failed research upstream then sits out for
`FAILURE_BACKOFF` seconds, so a dead Exa or Firecrawl can't slow down every request. Memory
tools and skills keep working. Synapse is never backed off, because memory is the core service.

## Skills: publishing is not activating

MCP resources are just readable documents. A client doesn't turn `skill://` resources into
skills by itself. The pilot bridges that gap in three small layers:

1. **Server instructions** (under 2 KB, the size Claude Code keeps) name the tool prefixes and
   point research at `skill://gateway-research/SKILL.md`.
2. **Resource access for every client.** Claude Code reads MCP resources natively. Any client
   can also use the `read_resource`/`list_resources` tools. These go through the server's
   normal resource path, so a bridged read is authorized exactly like `resources/read`: a
   restricted device can't read the research skill either way.
3. **An optional local pointer skill**, `synapse-gateway-research`, installed per client by
   `python -m mcp_gateway.client_setup install-bootstrap --client claude|codex`. Its only job
   is to fetch and follow the gateway-served skill, so both clients run the single copy. It
   uses a different name from the served skill, so the two can never collide. If opt-in
   two-way skills sync is on, the pointer gets published like any other local skill. It does
   nothing on machines that aren't connected to the gateway.

Canonical skills stay in Postgres and reach clients through the existing sync. The gateway
doesn't fork them. It only adds the one pilot skill that is tied to its own tool names.
Disable that skill with `SYNAPSE_GATEWAY_LOCAL_SKILLS=0`. Don't publish a PG skill named
`gateway-research`.

**Scripts:** reading `skill://x/scripts/foo.py` returns text and runs nothing. A skill's
scripts only work after the skill is materialized into a local skills folder (plugin skills
sync, or FastMCP's `sync_skills`). `gateway-research` has no scripts by design.

## Configuration

All variables are prefixed `SYNAPSE_GATEWAY_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYNAPSE_URL` | `http://127.0.0.1:8765` | Synapse base URL (`/mcp`, `/auth/whoami` are appended) |
| `HOST` / `PORT` | `127.0.0.1` / `8766` | listener; MCP at `/mcp` |
| `PUBLIC_URL` | unset | public base URL, only for auth metadata |
| `RESEARCH_TRUST` | `full` | `full` or `restricted` (= any approved device) |
| `EXA_URL`, `FIRECRAWL_URL` | unset | setting one enables that upstream |
| `<UP>_API_KEY` or `<UP>_API_KEY_FILE` | unset | the upstream's key (prefer `_FILE`: a docker/systemd secret) |
| `<UP>_AUTH_HEADER` | unset | send the key in this header (`Authorization` becomes `Bearer <key>`)… |
| `{api_key}` in `<UP>_URL` | — | …or put it in the URL (URL-quoted). Exactly one of the two. |
| `DISCOVERY_TIMEOUT`, `<UP>_DISCOVERY_TIMEOUT` | `5` | research listing bound (s) |
| `<UP>_CALL_TIMEOUT` | `120` | research tool-call bound (s) |
| `SYNAPSE_DISCOVERY_TIMEOUT` / `SYNAPSE_CALL_TIMEOUT` | `10` / `120` | Synapse bounds (s) |
| `AUTH_CACHE_TTL` | `30` | how long a verified caller is reused (s); `0` disables |
| `SYNAPSE_CACHE_TTL` / `RESEARCH_CACHE_TTL` | `30` / `300` | listing caches (s) |
| `FAILURE_BACKOFF` | `30` | research upstream sit-out after a failed listing (s) |
| `LOCAL_SKILLS` | `1` | serve the bundled `gateway-research` skill |

Invalid combinations are refused at startup: a key with nowhere to go, both placements, a
placeholder without a key, or a non-http URL. Check your provider's documentation for where
its hosted MCP endpoint expects the key. The gateway doesn't assume either placement.
`examples/gateway/` has an env template and a compose override.

## Run locally

Synapse must include `GET /auth/whoami` (this branch). Then:

```bash
export SYNAPSE_GATEWAY_SYNAPSE_URL=http://127.0.0.1:8765
export SYNAPSE_GATEWAY_EXA_URL=...            # see examples/gateway/gateway.env.example
export SYNAPSE_GATEWAY_EXA_API_KEY_FILE=...   # never commit keys
uv run python -m mcp_gateway                  # or: uv run synapse-gateway
```

## Connect clients

`python -m mcp_gateway.client_setup snippets --gateway-url http://127.0.0.1:8766` prints the
add and remove commands for both clients. It edits nothing. Both clients reuse the device token
this machine already has, through `plugin-codex/scripts/mcp_headers.py --gateway`. That
helper sends the token only to a gateway on the saved Synapse host (same scheme, any port) or
on loopback. No token is written into client config.

- **Claude Code**: `claude mcp add-json --scope user synapse-gateway '{"type":"http",…,"headersHelper":…}'`.
  Undo with `claude mcp remove --scope user synapse-gateway`.
- **Codex**: add a `[mcp_servers.synapse-gateway]` table with `url` and `http_headers_helper`.
  Undo by deleting the table.

## Verify

```bash
# 1. Synapse knows the device (prints kind/trust/surface_id, never the token)
curl -s -H "Authorization: Bearer $SYNAPSE_INGEST_TOKEN" http://127.0.0.1:8765/auth/whoami
# 2. The gateway, as this device sees it (read-only; no research call)
SYNAPSE_INGEST_TOKEN=... uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp
# 3. Optional: one real research call (spends credits; the query goes to Exa)
uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp --search "fastmcp release notes"
```

Then, in each client, ask for web research. The client should read
`skill://gateway-research/SKILL.md`, call `synapse_recall` first, then use the `exa_*` and
`firecrawl_*` tools.

## Limitations (pilot)

- Only device-bearer callers are supported. The claude.ai OAuth connector keeps talking to
  Synapse directly.
- Per-call overhead: each upstream operation opens a fresh MCP client (initialize plus the
  request). This is the price of zero shared state, and it's acceptable at pilot scale.
- Exa and Firecrawl are exercised in tests only through local stubs. Live compatibility,
  including key placement and tool names, still has to be verified against the real services.
- Upstream sampling and elicitation requests are not relayed to the caller, because the plain
  `Client` has no handlers for them.
- Rollback: stop the gateway service and remove the client entries. Synapse is unaffected
  except for the additive `/auth/whoami` route.
