# MCP gateway (pilot)

One MCP connection that gives an agent client (Claude Code, Codex, …) the user's Synapse
memory, their published `skill://` skills, and any other MCP services **the deployment chooses
to put behind it**. The gateway (`mcp_gateway/`) is a separate FastMCP 3.4.2 service that runs
beside Synapse. The only change to Synapse itself is one small route, `GET /auth/whoami`.

```
Claude Code ─┐                    ┌─ Synapse /mcp              (the caller's OWN device bearer)
             ├─ gateway /mcp ─────┤
Codex ───────┘   (device bearer)  └─ 0..n configured MCP services  (each with its own credential)
```

Synapse memory and skills are built in. Everything else is deployment configuration: the
gateway bundles no third-party services, research workflow, or tool assumptions. With no
registry file, it serves Synapse memory and skills, and nothing else.

## What a client sees

| Name | From | Who sees it |
| --- | --- | --- |
| `recall`, `recall_full_turns`, `fetch`, `fetch_session`, `remember`, `recall_feedback` | Synapse, **names unchanged** | every approved device; Synapse filters the data per device as always |
| `<namespace>_<tool>` | each configured upstream, its tool name behind the namespace you chose | devices meeting that upstream's `min_trust` |
| `list_resources`, `read_resource` | FastMCP `ResourcesAsTools` | every approved device (reads are gated like `resources/read`) |
| `skill://<name>/SKILL.md`, `/_manifest`, `/<file>` | Synapse's skills, URIs unchanged | devices meeting `SKILLS_TRUST` (default: full trust) |
| `skill://<name>/…` from configured `skills_dirs` | FastMCP's native `SkillsDirectoryProvider` | devices meeting that directory's `min_trust` |
| `<scheme>://<namespace>/…` resources | configured upstreams, URIs prefixed with the namespace | devices meeting that upstream's `min_trust` |

Memory tools keep their Synapse names so every existing plugin hook keeps working through
the gateway (see [Hooks](#hooks)). Upstream tools keep their own names behind the prefix, so
an upstream tool called `tracker_search` under namespace `tracker` becomes
`tracker_tracker_search`. Descriptions, input schemas and output schemas pass through
unchanged, as do upstream tool errors (`isError`) and JSON-RPC errors. A transport failure
becomes a generic `"<namespace> upstream request failed (ConnectError)"`, because the raw text
could quote a keyed URL.

## Configure upstreams

Name a JSON file with `SYNAPSE_GATEWAY_CONFIG_FILE`. Relative paths inside it resolve against
the file's own directory.

```json
{
  "upstreams": [
    {
      "namespace": "tracker",
      "description": "Project tracker: issues and milestones",
      "url": "https://tracker.example.com/mcp",
      "auth": {"type": "bearer", "secret_env": "TRACKER_MCP_TOKEN"},
      "min_trust": "full",
      "discovery_timeout": 5,
      "call_timeout": 120,
      "cache_ttl": 300,
      "failure_backoff": 30
    },
    {
      "namespace": "wiki",
      "url": "https://wiki.example.com/mcp",
      "auth": {"type": "header", "header": "X-Api-Key", "secret_file": "secrets/wiki.key"},
      "min_trust": "restricted"
    }
  ],
  "skills_dirs": [{"path": "team-skills", "min_trust": "full"}]
}
```

| Key | Meaning |
| --- | --- |
| `namespace` | required. 1–32 chars: lowercase letters, digits, single hyphens, starting with a letter. It becomes the `<namespace>_` tool prefix. It must be unique, and can't be one of the reserved names (`synapse`, `gateway`, `skill(s)`, `mcp`, `recall`, `fetch`, `remember`, `list`, `read`, `issue`), which would be confusable with the gateway's own tools. |
| `url` | required. A streamable-HTTP MCP endpoint (`http`/`https`), with no `user:pass@` in it. |
| `auth.type` | `none` (default), `bearer` (`Authorization: Bearer <secret>`), `header` (`auth.header: <secret>`; `Host`, `Cookie`, `Content-*`, MCP session headers and the like are refused), or `url` (the secret replaces `{secret}` in `url`, URL-quoted). |
| `auth.secret_env` / `auth.secret_file` | exactly one for any type except `none`: the NAME of an environment variable, or a path to a file holding the secret (a docker/systemd secret, for example). Inline secrets aren't accepted. |
| `min_trust` | `full` (default) or `restricted`, the lowest Synapse device trust that sees this upstream at all |
| `discovery_timeout` / `call_timeout` | seconds; defaults 5 / 120 |
| `cache_ttl` / `failure_backoff` | seconds; defaults 300 / 30 (see [Degradation](#degradation)) |
| `description` | optional operator label, up to 120 chars. Kept in configuration; not exposed in shared server instructions. |
| `skills_dirs[].path` / `.min_trust` | extra skill folders served through FastMCP's native provider, behind the same trust gate. Pick names that don't clash with your Synapse skills. |

The file is validated strictly at startup, and the gateway refuses to start on any problem:
an unknown key (so a typo can't silently drop a setting, and a pasted credential can't hide in
a stray field), a duplicate, reserved or unsafe namespace, a non-HTTP URL, credentials in the
URL, a `{secret}` placeholder without `type: url` (or the reverse), a forbidden header, an
unset or empty secret reference, a missing directory, or malformed JSON. Error messages name
keys, variables and paths, never values.

**Supported:** streamable-HTTP MCP servers, with no auth or a static secret. **Not
supported:** stdio servers, OAuth-enrolled servers (no interactive consent flow runs on the
gateway), and per-caller upstream credentials. Every caller who passes an upstream's
`min_trust` uses the deployment's one credential for it.

Integrations belong to each deployment. The repository ships generic examples only
(`examples/gateway/`). Your own services and their credentials live in your private
deployment config.

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
- **Credentials only go to the upstream they belong to.** Upstream clients are built with
  FastMCP's inbound-header forwarding turned off. That matters because the stock
  `ProxyClient`/`create_proxy` forwards the caller's `Authorization` header to every upstream.
  Each configured upstream sees only its own configured credential, and Synapse sees only the
  caller's bearer.
- **No shared state across identities.** The stock `ProxyProvider` keeps one component cache
  for all sessions, so the gateway doesn't use it. Synapse listings are cached per identity,
  keyed by `sha256(token)`. Upstream listings are shared, because the credential behind them
  is the deployment's own, and access is still decided per caller before any upstream I/O.
  Every upstream operation opens a fresh client, so no HTTP pool, cookie jar or MCP session
  is reused between callers, and the gateway runs stateless HTTP.
- **Revocation.** Memory access ends on a revoked device's next call, because Synapse checks
  the bearer again on every request. Access to configured upstreams can outlive revocation by
  at most `AUTH_CACHE_TTL` (default 30s).
- **Restricted devices get no Synapse skills by default.** Synapse's skills provider isn't
  scoped to the caller: it serves every active skill, personal ones included, to any valid
  bearer, and skills carry no audience label to filter on. So the gateway withholds Synapse's
  whole `skill://` class (listing, templates, reads, and the `read_resource` bridge) from any
  device below `SKILLS_TRUST`, however permissive the upstream is. Restricted devices keep
  their memory tools. Direct connections to Synapse's own `/mcp` are outside the gateway and
  unchanged; they still list skills to restricted devices.
- **Per-upstream trust.** `min_trust` defaults to `full`, so a restricted work device can't
  reach a service, or spend its credential, unless the deployment explicitly allows it.
- **No credentials in logs.** Upstreams are logged as `scheme://host/…`. A log-record scrubber
  removes every configured secret from every logger, tracebacks included, and httpx request
  logging is capped at WARNING. Don't run with DEBUG logging in production: the `mcp` library
  then logs full request and response payloads, which include memory content.
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
it sees two sets of memory tools.

## Degradation

Every upstream listing runs under its `discovery_timeout` (Synapse: 10s). If a listing
fails, the gateway serves the last good snapshot, up to an hour old. If there isn't one, it
logs a redacted warning and leaves that upstream out, and the upstream then sits out for its
`failure_backoff`, so a dead service can't slow every request. Memory tools and skills keep
working. Synapse is never backed off, because memory is the core service.

## Skills: publishing is not activating

MCP resources are just readable documents. A client doesn't turn `skill://` resources into
skills by itself. The gateway bridges that gap without shipping any workflow of its own:

1. **Server instructions** (under 2 KB, the size Claude Code keeps) describe the memory tools,
   how to discover authorized tools without exposing restricted service names, and skills:
   list resources, read the `SKILL.md` whose description fits the task, and follow it.
2. **Resource access for every client.** Claude Code reads MCP resources natively. Any client
   can also use the `read_resource`/`list_resources` tools. These go through the server's
   normal resource path, so a bridged read is authorized exactly like `resources/read`.
3. **An optional local pointer skill**, `synapse-gateway`, installed per client by
   `python -m mcp_gateway.client_setup install-bootstrap --client claude|codex`. It names no
   workflow or service. It tells the client to look for a matching skill among the user's
   published ones on the gateway and load it. Installing it also retires this tool's earlier
   pilot pointer (`synapse-gateway-research`), but only if that copy is byte-identical to
   what the tool generated; edited copies are left alone. If opt-in two-way skills sync is on,
   the pointer gets published like any other local skill.

Canonical skills stay in Synapse and reach clients through the existing sync. The gateway
serves them as they are.

**Scripts:** reading `skill://x/scripts/foo.py` returns text and runs nothing. A skill's
scripts only work after the skill is materialized into a local skills folder (plugin skills
sync, or FastMCP's `sync_skills`).

**Portability boundary.** The gateway makes the same memory, skills and services reachable
from every client. It does not translate one client's personal instructions (CLAUDE.md,
AGENTS.md, client-specific tool names in a skill's text) for another. A skill that names
another client's built-in tools still needs editing to be portable.

## Core settings

All variables are prefixed `SYNAPSE_GATEWAY_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYNAPSE_URL` | `http://127.0.0.1:8765` | Synapse base URL (`/mcp`, `/auth/whoami` are appended) |
| `HOST` / `PORT` | `127.0.0.1` / `8766` | listener; MCP at `/mcp` |
| `PUBLIC_URL` | unset | public base URL, only for auth metadata |
| `CONFIG_FILE` | unset | the upstream/skills registry above; unset = Synapse memory + skills only |
| `SKILLS_TRUST` | `full` | who gets Synapse's `skill://` resources; `restricted` = any approved device |
| `AUTH_CACHE_TTL` | `30` | how long a verified caller is reused (s); `0` disables |
| `SYNAPSE_CACHE_TTL` | `30` | per-caller Synapse listing cache (s) |
| `SYNAPSE_DISCOVERY_TIMEOUT` / `SYNAPSE_CALL_TIMEOUT` | `10` / `120` | Synapse bounds (s) |
| `WHOAMI_TIMEOUT` | `5` | caller verification bound (s) |

## Run locally

Synapse must include `GET /auth/whoami` (this branch). Then:

```bash
export SYNAPSE_GATEWAY_SYNAPSE_URL=http://127.0.0.1:8765
export SYNAPSE_GATEWAY_CONFIG_FILE=/path/to/gateway.json   # optional; see examples/gateway/
uv run python -m mcp_gateway                               # or: uv run synapse-gateway
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
  marked read-only fail there unless pre-approved with
  `[mcp_servers.synapse-gateway.tools.<tool>] approval_mode = "approve"`, the same per-tool
  mechanism a direct `synapse` entry uses.

If a shell exports `SYNAPSE_INGEST_TOKEN` with some other credential, such as the root
enrollment token kept for admin tooling, that value overrides the saved device token. The
gateway then refuses it, correctly. Generate the snippets with `--saved-credential`: the
helper then runs under `env -u SYNAPSE_INGEST_TOKEN -u CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN`,
so the saved device token wins for the gateway only, and other env consumers keep theirs.

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
# 2. The gateway's structure, as this device sees it (read-only: listings and one skill read)
SYNAPSE_INGEST_TOKEN=... uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp \
    --expect-namespace tracker
# 3. Optional: exactly one call you specify. The probe prints only ok/error and the size.
uv run python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp \
    --call tracker_list_issues --args '{"project": "web"}'
```

The probe exits non-zero if the core memory tools are missing, an expected namespace isn't
listed for this device, a native skill read disagrees with the bridged one, or the explicit
call errors. On a restricted device, expect the memory tools plus whichever namespaces allow
`restricted`, and no Synapse `skill://` resources unless `SKILLS_TRUST=restricted`.

## Limitations (pilot)

- Only device-bearer callers are supported. The claude.ai OAuth connector keeps talking to
  Synapse directly.
- Upstreams are HTTP MCP with static credentials only (see [Configure upstreams](#configure-upstreams)).
- Per-call overhead: each upstream operation opens a fresh MCP client (initialize plus the
  request). This is the price of zero shared state, and it's acceptable at pilot scale.
- Skills for restricted devices are all or nothing. Allowing some skills would need an
  audience or scope field that Synapse's skills don't have yet.
- Upstream sampling and elicitation requests are not relayed to the caller, because the plain
  `Client` has no handlers for them.
- Rollback: stop the gateway service and remove the client entries. Synapse is unaffected
  except for the additive `/auth/whoami` route.
