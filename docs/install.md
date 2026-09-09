# Installing Synapse

The full path from an empty directory to a Claude Code session that recalls your history.
The short version lives in the [README](../README.md#quick-start).

Two halves, and both are needed: the **server** (Docker, one box) and the **plugin** (one
install per Claude Code machine).

## 1. Clone and configure

```bash
git clone https://github.com/kraft87/synapse.git synapse && cd synapse
cp .env.example .env
```

Fill in these values in `.env`. Everything else has a working default.

| Variable | What to put there |
|----------|-------------------|
| `SYNAPSE_DB_PASSWORD` | Password for the bundled Postgres. Must match the password inside `SYNAPSE_DB_URL`. |
| `SYNAPSE_DB_URL` | Postgres DSN. The `.env.example` default just needs the same password filled in. |
| `SYNAPSE_MACHINE_TOKEN` | The shared root bearer. `.env.example` ships it as `CHANGEME`; generate a real one with `openssl rand -hex 32`. |
| `VOYAGE_API_KEY` | Voyage AI key, used for embeddings and rerank. Skip it only if you configure an alternative backend (see [local inference](#local-inference-no-external-accounts)). |
| `CLAUDE_CODE_OAUTH_TOKEN` **or** `ANTHROPIC_API_KEY` | Auth for the extraction LLM. The subscription token wins if both are set. |

`SYNAPSE_MACHINE_TOKEN` is required: the server refuses to boot while it is blank or still
the `CHANGEME` placeholder, and prints what to do about it. The one exception is
`SYNAPSE_ALLOW_OPEN=1`, a dev-and-stdio-only mode that starts without a token and then
serves **every** caller restricted, which means an empty board and empty recalls. It is for
poking at a throwaway stack, not for running one.

Optional knobs (`SYNAPSE_INGEST_TAIL`, `POLL_INTERVAL_SECONDS`, recall-serving tuning such
as the shadow-phase `SYNAPSE_RECALL_FLOOR` / `SYNAPSE_RECALL_FLOOR_ENFORCE` abstention
floor, and more) are listed in
[ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration). Set `LOGFIRE_TOKEN` to stream
traces to [Pydantic Logfire](https://logfire.pydantic.dev); leave it blank and telemetry is
fully off.

## 2. Start the stack

```bash
docker compose up -d --build
```

This builds the image from public bases (no registry auth), starts Postgres, the poller, the
MCP server, and the `dream` container, and applies the schema automatically on Postgres's
first boot.

If the default ports are taken, set `MCP_PORT` and/or `POSTGRES_HOST_PORT` in `.env` before
starting. See [Ports and a second instance](#ports-and-a-second-instance).

## 3. Mint this machine's device token

```bash
docker compose exec mcp-server synapse-admin bootstrap "this laptop"
```

```
minted dev-<surface-id> trust=full label=this laptop
token: <one-time token>
Shown once — only the hash is stored.
```

Copy the token: you paste it as the "Synapse token" in the plugin install prompt in the
next step. It is shown once because only its hash is stored, so if you lose it, run
`bootstrap` again for a new one and revoke the old row.

Why this step exists: since schema 054, what a machine is served depends on **its own**
device token, not on the shared machine token and not on any name it reports. The shared
machine token identifies the deployment rather than a machine, so a caller presenting it is
served restricted, which looks like an empty board and empty recalls. There is no fail-open
path here: an unrecognised caller is served nothing rather than everything.

On a server that has GitHub OAuth or OIDC configured, a machine can enroll itself instead
with `! synapse-login`, and the bootstrap command is the break-glass path. See
[docs/auth.md](auth.md).

## 4. Install the plugin

On each Claude Code machine. The repo ships its own marketplace manifest, so there is
nothing to publish:

```
/plugin marketplace add kraft87/synapse
/plugin install synapse@synapse
```

The install prompts for:

- **Synapse URL**: base URL, no path. `http://localhost:8765` for the local quickstart.
- **Synapse token**: the device token from step 3. On an OAuth-backed server you can leave
  it blank and run `! synapse-login` instead.
- **This machine's role**: `personal` (default, sees everything) or `work` (restricted:
  work-safe notes plus an allowlist of projects, so personal memory is never served there).
- **Opt-in syncs**: two-way skill sync and config mirroring, both off by default.

Then `/reload-plugins` (or restart Claude Code) to activate the hooks and the MCP server.
Full plugin detail: [plugin/README.md](../plugin/README.md).

## 5. Verify

```bash
# Server is up:
curl -fsS localhost:8765/health

# 1. Ship a test transcript (>= 4 turns) to the server:
curl -sX POST localhost:8765/ingest -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SYNAPSE_MACHINE_TOKEN" \
  -d @docs/example-transcript.json
# 2. Wait one poll cycle (up to ~5 minutes), then ask for it back:
curl -sX POST localhost:8765/recall -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SYNAPSE_DEVICE_TOKEN" \
  -d '{"query": "what caching layer did we pick for the demo app search service"}'
```

Recall is served according to the bearer, so use the device token from step 3 for the recall
call, not the shared machine token.

Two timing expectations that look like bugs but aren't: **episodes** appear within seconds
of ingest, but **knowledge-graph facts** are extracted from 4-turn sliding windows — a
session with fewer than 4 human turns produces episodes and an *empty graph*. And extraction
runs on a poll cycle (`POLL_INTERVAL_SECONDS`, default 300), so first facts land a few
minutes after ingest, not instantly.

From inside Claude Code: start a fresh session. The SessionStart board block's banner reports
the episode count and the most recently active projects, so yours should appear with a rising
count.

## 6. Import your existing history

A fresh install doesn't have to start cold. If you've been using Claude Code, months of
transcripts already sit in `~/.claude/projects` — import them once and `recall()` knows your
history on day one:

```
! synapse-import        # inside a Claude Code session (the plugin puts it on PATH)
```

It offers an optional date range (by each file's last-activity date), prints a summary
(file count, size, estimated turns), and asks for confirmation before sending anything —
importing runs KG extraction on your configured LLM for every new turn, which consumes
subscription usage or API credits roughly in proportion to the turn count, so bounding a
first import by date bounds the spend. The server dedups turns by `span_id`, so Ctrl-C
and re-running are always safe: an interrupted import resumes where it left off.

Cursor history can be imported too, but only as a server-side dev path for now
(`python -m ingestion.cursor_sqlite_backfill`).

## Model ids: one thing that bites

`SYNAPSE_LLM_MODEL` is a **provider-specific** id. A value with an `anthropic/` prefix, such
as `anthropic/claude-haiku-4.5`, is the OpenRouter spelling and is only valid with
`SYNAPSE_LLM_PROVIDER=openai`. The default `claude-code` provider passes the id straight to
the Claude CLI, which rejects it, and every extraction stage then fails with empty output.
Two guards now catch that: `.env.example` ships the line commented out, and the
`claude-code` provider refuses a prefixed id at startup, naming the spelling it wants
(`claude-haiku-4-5`) instead of failing once per queue item. On the default provider, either
leave `SYNAPSE_LLM_MODEL` unset or use a plain Claude model name.

A missing extraction credential is loud now too. With no `CLAUDE_CODE_OAUTH_TOKEN` the Claude
CLI answers with its own error text (`Not logged in · Please run /login`) rather than empty
output, which used to be parsed as a model response and swallowed. Those rows are now marked
`failed` carrying that exact string, and retried, instead of being recorded as a silent
`done`. Episodes are unaffected either way: they are stored and recallable even when fact
extraction fails.

## Local inference (no external accounts)

The bundled `local-inference` compose profile runs embeddings and rerank locally, so
`VOYAGE_API_KEY` can stay blank. The recipe (provider, base URL, model ids, and the
`SYNAPSE_EMBED_DIMS` value you must set **before** the database's first boot) is in
`.env.example` under "Fully local, zero-signup recipe". Honest caveat: published retrieval
quality was measured on the Voyage stack, and the local models are a convenience floor.

Embedding width is fixed at first-boot schema provisioning and recorded in `synapse_meta`;
the server fails loudly on a later mismatch rather than serving garbage.

## Ports and a second instance

`MCP_PORT` sets both the server's listener and the published host port. `POSTGRES_HOST_PORT`
does the same for Postgres. To run a second stack on the same host (a test instance
alongside a live one), set both plus `COMPOSE_PROJECT_NAME`, so the two get separate
containers and volumes.

## Upgrading

First-boot init never re-runs on an existing data volume. To bring an existing database up
to date, run [`scripts/apply_schema.sh`](../scripts/apply_schema.sh) (the single source of
truth for migration order) against it — see the script's header for caveats.

Every service verifies at boot that the database schema matches the code (the script
stamps the applied version; a mismatch refuses to start with instructions rather than
failing mid-request). So the upgrade order is: pull, run `apply_schema.sh`, restart.
`SYNAPSE_SCHEMA_CHECK=0` skips the guard.

Releases are tagged (`v0.8.1`, ...). `main` is kept releasable, but for a known-good
build check out the latest tag; a change that needs a migration or renames an env var
gets a release note saying so.

## Troubleshooting

- **The board is empty but the URL is right, and recall returns nothing.** The machine is
  being served restricted: either it never enrolled, or it is presenting the shared machine
  token instead of a device token. Mint one with
  `docker compose exec mcp-server synapse-admin bootstrap "<label>"`, or run
  `! synapse-login`, then paste it into the plugin config. The SessionStart block says which
  case you are in.
- **`POST /recall` returns `{"facts": []}` with no `episodes` key.** The embedding call
  failed, almost always a bad or blank `VOYAGE_API_KEY`. The HTTP status is still 200, so
  the only signal is the log: run `docker compose logs mcp-server` and look for
  `Embedding query failed`.
- **Episodes appear but the graph stays empty.** KG facts are extracted from 4-turn-or-longer
  windows on a poll cycle (default 5 min); a short session or a fresh import needs a few
  minutes. If it stays empty for longer, check the extraction LLM: a bad credential or model
  id now leaves `failed` extraction rows carrying the reason. See
  [Model ids](#model-ids-one-thing-that-bites).
- **`401 Unauthorized`.** Token missing, stale, or revoked. Re-run `! synapse-login`, or
  paste a fresh device token into the `/plugin` config.
- **Recall returns nothing and the hooks seem dead.** Hooks are fail-soft, so an unreachable
  server is a silent no-op. Check `curl -fsS $SYNAPSE_URL/health`, then that `SYNAPSE_URL`
  and the token are set (`/plugin` shows the stored values).
- **`synapse-login` / `synapse-import` not found.** The plugin's `bin/` is only on PATH
  inside Claude Code sessions, hence the `!` prefix. From a plain terminal, run the script
  by full path.
- **`python3: not found`.** The hooks and scripts are stdlib Python 3, so put it on PATH.
- **Device-flow login fails on an older server.** It needs `/device/code` and "Enable Device
  Flow" on the GitHub OAuth App; fall back to `! synapse-login --browser`.
- **A service refuses to start, complaining about the schema version.** Run
  `scripts/apply_schema.sh` against the database, then restart. See
  [Upgrading](#upgrading).

## Next

- [docs/auth.md](auth.md): device tokens, enrollment, exposing the server to claude.ai.
- [docs/tools.md](tools.md): what the MCP tools and hooks actually do.
- [ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration): every configuration variable.
