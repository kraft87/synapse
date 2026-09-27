# MCP gateway (pilot)

One MCP connection that gives Claude Code and Codex the same Synapse memory, the same
`skill://` skills, and the same Exa/Firecrawl web research. The gateway (`mcp_gateway/`) is a
separate FastMCP 3.4.2 service that runs beside Synapse; Synapse itself is unchanged except for
one small route, `GET /auth/whoami`.

```
Claude Code ─┐                    ┌─ Synapse /mcp      (caller's OWN device bearer)
             ├─ gateway /mcp ─────┼─ Exa MCP           (gateway's Exa config, key optional)
Codex ───────┘   (device bearer)  └─ Firecrawl MCP     (gateway's Firecrawl config, key optional)
```

## What a client sees

| Name | From | Who sees it |
| --- | --- | --- |
| `recall`, `recall_full_turns`, `fetch`, `fetch_session`, `remember`, `recall_feedback` | Synapse, **names unchanged** | every approved device; Synapse filters the data per device as always |
| `exa_<tool>`, `firecrawl_<tool>` | research upstreams, upstream name behind a fixed prefix | full-trust devices (see `RESEARCH_TRUST`) |
| `list_resources`, `read_resource` | FastMCP `ResourcesAsTools` | every approved device (reads are gated like `resources/read`) |
| `skill://<name>/SKILL.md`, `/_manifest`, `/<file>` | Synapse's PG skills, URIs unchanged | full-trust devices (see `SKILLS_TRUST`) |
| `skill://gateway-research/SKILL.md` | bundled with the gateway | same devices as the research tools |

Memory tools keep their Synapse names so that every existing plugin hook keeps working
through the gateway (see [Hooks](#hooks)). Research tools keep their upstream name behind
the prefix, which can double it: `firecrawl_firecrawl_scrape`, `exa_web_search_exa`. Tool
descriptions, input schemas and output schemas pass through unchanged. Upstream tool errors (`isError`) and
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
- **Restricted devices get no Synapse skills.** Synapse's skills provider isn't scoped to
  the caller: it serves every active skill, personal ones included, to any valid bearer, and
  skills carry no audience label to filter on. So the gateway withholds Synapse's whole
  `skill://` class (listing, templates, reads, and the `read_resource` bridge) from any
  device below `SKILLS_TRUST` (default `full`), however permissive the upstream is.
  Restricted devices keep their memory tools. Direct connections to Synapse's own `/mcp`
  are outside the gateway and unchanged; they still list skills to restricted devices.
- **Research policy.** By default only full-trust devices see research tools and the research
  skill. A restricted, work-profile device therefore can't send queries through the owner's
  research configuration or pick up tools it wasn't granted. Set `RESEARCH_TRUST=restricted` to allow any
  approved device.
- **No credentials in logs.** Keys come from env or secret files and never from the repo.
  Logs show upstreams as `scheme://host/…`. A log-record scrubber removes configured keys from
  every logger, tracebacks included, and httpx request logging is capped at WARNING. Don't
  run with DEBUG logging in production: the `mcp` library then logs full request and response
  payloads, which include memory content.
- The gateway adds no raw HTTP forwarding routes. Its only custom route is an unauthenticated
  `/health` that reports liveness and makes no upstream call.

## Hooks

The plugins' hooks match memory tools by name under any server:
`mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$` for
self-session injection, `mcp__.*__remember$` for the remember spool, and
`mcp__.*__(recall|recall_full_turns)$` for the feedback nudge. Because the gateway keeps the
memory names unchanged, `mcp__synapse-gateway__recall` triggers the same hooks as
`mcp__plugin_synapse_synapse__recall`. A test pins that equivalence.

One matcher had to change for this. The Claude plugin's feedback-nudge matcher was pinned
to the plugin's own server (`mcp__plugin_synapse_synapse__…`). It now uses the same
server-agnostic pattern as the Codex hook already did. Installed plugins pick up the change
only after a plugin version bump and release.

SessionStart board/preferences, ingest, and skills sync use Synapse's HTTP routes directly,
so the gateway doesn't affect them. If a client connects to both Synapse and the gateway,
it sees two sets of memory tools. Pilot one connection per client.

## Degradation

Each upstream listing runs under a discovery timeout (research defaults to 5s, Synapse to 10s).
If a listing fails, the gateway serves the last good snapshot. If there isn't one, it logs a
redacted warning and leaves that upstream out. A failed research upstream then sits out for
`FAILURE_BACKOFF` seconds, so a dead Exa or Firecrawl can't slow down every request. Memory
tools and skills keep working. Synapse is never backed off, because memory is the core service.

## Skills: publishing is not activating

MCP resources are just readable documents. A client doesn't turn `skill://` resources into
skills by itself. The pilot bridges that gap in three small layers:

1. **Server instructions** (under 2 KB, the size Claude Code keeps) name the memory tools and
   research prefixes, and point research at `skill://gateway-research/SKILL.md`.
2. **Resource access for every client.** Claude Code reads MCP resources natively. Any client
   can also use the `read_resource`/`list_resources` tools. These go through the server's
   normal resource path, so a bridged read is authorized exactly like `resources/read`: a
   restricted device can't read the research skill or any Synapse skill either way.
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

The pilot skill itself is short and gateway-specific. It picks tools from the live list and
fills every required argument from the input schema. It uses `recall` first only when the
question touches the user's own history, and reads as many sources as the question needs. If
one research provider is down, it says so and falls back to the other. It passes the
skill-creator `quick_validate.py` check.

## Configuration

All variables are prefixed `SYNAPSE_GATEWAY_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYNAPSE_URL` | `http://127.0.0.1:8765` | Synapse base URL (`/mcp`, `/auth/whoami` are appended) |
| `HOST` / `PORT` | `127.0.0.1` / `8766` | listener; MCP at `/mcp` |
| `PUBLIC_URL` | unset | public base URL, only for auth metadata |
| `RESEARCH_TRUST` | `full` | `full` or `restricted` (= any approved device) |
| `SKILLS_TRUST` | `full` | who gets Synapse's `skill://` resources; `restricted` = any approved device (they are not caller-scoped upstream) |
| `EXA_URL`, `FIRECRAWL_URL` | unset | setting one enables that upstream |
| `<UP>_API_KEY` or `<UP>_API_KEY_FILE` | unset | optional key (prefer `_FILE`: a docker/systemd secret); omit for keyless endpoints |
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
placeholder without a key, or a non-http URL. If a provider's hosted MCP endpoint works
without a key, set only its URL. Otherwise check the provider's documentation for where the
endpoint expects the key. The gateway doesn't assume either placement.
`examples/gateway/` has an env template and a compose override.

## Run locally

Synapse must include `GET /auth/whoami` (this branch). Then:

```bash
export SYNAPSE_GATEWAY_SYNAPSE_URL=http://127.0.0.1:8765
export SYNAPSE_GATEWAY_EXA_URL=...            # see examples/gateway/gateway.env.example
export SYNAPSE_GATEWAY_EXA_API_KEY_FILE=...   # only if the endpoint needs one; never commit keys
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
  Undo by deleting the table. `codex exec` can't prompt for approval, so tools that aren't
  marked read-only (for example `recall` or a Firecrawl scrape) fail there unless
  pre-approved with `[mcp_servers.synapse-gateway.tools.<tool>] approval_mode = "approve"`,
  the same per-tool mechanism a direct `synapse` entry uses.

If a shell exports `SYNAPSE_INGEST_TOKEN` with some other credential, such as the root
enrollment token kept for admin tooling, that value overrides the saved device token. The
gateway then refuses it, correctly. Generate the snippets with `--saved-credential`: the
helper then runs under `env -u SYNAPSE_INGEST_TOKEN -u CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN`,
so the saved device token wins for the gateway only, and other env consumers keep theirs.
The probe explains a refusal like this with a clear message instead of a bare 401.

Personal hooks keyed on the plugin's server name, such as a Stop hook that counts
`mcp__plugin_synapse_synapse__recall`, need the `mcp__synapse-gateway__` form added. So does
the plugin's feedback-nudge matcher until an installed plugin carries the any-server version.
A narrow user-level PostToolUse hook on `mcp__synapse-gateway__(recall|recall_full_turns)$`
that runs the plugin's `recall_feedback_nudge.py` covers the gap. Remove it after the plugin
update, or gateway recalls get the nudge twice.

When scripting a client check from inside another Claude Code session, start the child
without the parent's inherited `CLAUDE_*` session variables. A nested session marker such as
`CLAUDE_CODE_SAFE_MODE` stops the child from loading any MCP server.

## Verify

```bash
# 1. Synapse knows the device (prints kind/trust/surface_id, never the token)
curl -s -H "Authorization: Bearer $SYNAPSE_INGEST_TOKEN" http://127.0.0.1:8765/auth/whoami
# 2. The gateway, as this device sees it (read-only; no research call)
SYNAPSE_INGEST_TOKEN=... uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp
# 3. Optional: one real research call. The query goes to the provider and may spend credits.
#    Every required string argument (e.g. Exa's `objective`) is filled from the text.
uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp --search "fastmcp release notes"
```

Then, in each client, ask for web research. The client should read
`skill://gateway-research/SKILL.md` and use the `exa_*` and `firecrawl_*` tools. It should
call `recall` first only when the question involves the user's own history. On a restricted
device, expect the memory tools only: no research tools and no `skill://` resources.

## Limitations (pilot)

- Only device-bearer callers are supported. The claude.ai OAuth connector keeps talking to
  Synapse directly.
- Per-call overhead: each upstream operation opens a fresh MCP client (initialize plus the
  request). This is the price of zero shared state, and it's acceptable at pilot scale.
- Automated tests use local Exa and Firecrawl stubs. The pilot also passed live search,
  scraping, skill reads, and memory recall through both Claude Code and Codex. Hosted
  keyless endpoints have provider quotas; existing Claude connector subscriptions do not
  configure or fund the gateway.
- Skills for restricted devices are all or nothing. Allowing some skills would need an
  audience or scope field that Synapse's skills don't have yet.
- Upstream sampling and elicitation requests are not relayed to the caller, because the plain
  `Client` has no handlers for them.
- Rollback: stop the gateway service and remove the client entries. Synapse is unaffected
  except for the additive `/auth/whoami` route.
