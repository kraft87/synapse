# Installing Synapse

The full path from an empty directory to a Claude Code session that recalls your history.
The short version lives in the [README](../README.md#quick-start).

Two halves, and both are needed: the **server** (Docker, one box) and the **plugin** (one
install per Claude Code machine).

## 1. Clone and configure

```bash
git clone https://github.com/kraft87/synapse.git synapse && cd synapse
```

**Pick a starting file.** `.env.example` is the annotated reference for every variable that
exists. [`examples/env/`](../examples/env/) holds three complete `.env` files, one per
deployment shape, so a non-default setup is one copy instead of seven hand-edited variables:

- [`voyage.env`](../examples/env/voyage.env): today's default. Needs a Voyage AI key for
  embeddings and rerank, plus a Claude subscription token or an Anthropic API key.
- [`local.env`](../examples/env/local.env): no retrieval signups. The bundled
  `local-inference` container serves embeddings and rerank; extraction still needs the
  Claude token or an Anthropic key. See [Local inference](#local-inference-no-external-accounts).
- [`openrouter.env`](../examples/env/openrouter.env): no Claude subscription. Voyage for
  retrieval, OpenRouter for extraction, so one key each.

```bash
cp examples/env/voyage.env .env      # or local.env, or openrouter.env, or .env.example
```

Whichever you copied, fill in these values in `.env`. Everything else has a working default.

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

Optional knobs (`POLL_INTERVAL_SECONDS`, `SYNAPSE_DRAIN_CONCURRENCY`, recall-serving tuning such
as the `SYNAPSE_RECALL_FLOOR` abstention floor, which every `examples/env` preset pins to
the value calibrated for its reranker, and more) are listed in
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
starting. See [Ports and a second instance](#ports-and-a-second-instance). Postgres (and the
opt-in local-inference port) publish on `127.0.0.1` only; to reach Postgres from another
machine, uncomment the override line in `compose.yml`'s `postgres` ports block, or tunnel
over ssh.

## 3. Mint this machine's device token

```bash
docker compose exec mcp-server synapse-admin bootstrap "this laptop"
```

```
minted dev-<surface-id> trust=full label=this laptop
token: <one-time token>
Shown once — only the hash is stored.
```

Copy the token: `synapse login --token` asks for it and stores it on this machine in the next
step.
It is shown once because only its hash is stored, so if you lose it, run `bootstrap` again
for a new one and revoke the old row.

Why this step exists: since schema 054, what a machine is served depends on **its own**
device token, not on the shared machine token and not on any name it reports. The shared
machine token identifies the deployment rather than a machine, so a caller presenting it is
served restricted, which looks like an empty board and empty recalls. There is no fail-open
path here: an unrecognised caller is served nothing rather than everything.

On a server that has GitHub OAuth or OIDC configured, a machine can enroll itself instead
with plain `synapse login`, and the bootstrap command is the break-glass path. See
[docs/auth.md](auth.md).

## 4. Install the client and log in

On each machine you code on. The server serves its own client, so the version always
matches:

```bash
curl -sSL http://localhost:8765/install | python3 -
# Windows PowerShell:
irm http://localhost:8765/install | python -
```

That unpacks the client into `~/.synapse/lib/<version>/`, puts a `synapse` command in
`~/.synapse/bin` (it prints the line to add to your shell profile if that isn't on `PATH`;
on Windows it adds it to your user `PATH`), and records the server URL. Python 3.11+ is
the only requirement. Then:

```bash
synapse login --token     # server without sign-in: paste the step 3 token when asked
synapse login                               # server with GitHub OAuth / OIDC: one approval
```

`synapse login` is the one setup step. It asks this machine's questions once and saves the
answers in `~/.synapse/config.json`:

- **personal or work**: `personal` (default, sees everything) or `work` (restricted:
  work-safe notes plus an allowlist of projects, so personal memory is never served there).
  It is declared when the machine enrolls.
- **Sync skills to this machine?**: two-way sync of your skills library at session start.

Then it wires every harness it finds. **Claude Code**: it installs the plugin
(`claude plugin marketplace add kraft87/synapse`, `claude plugin install synapse@synapse`)
with the server URL; restart Claude Code or run `/reload-plugins`. **Codex**: it writes the
hooks and the `synapse` MCP server into `~/.codex/config.toml`; start `codex`, run `/hooks`
and trust them. Re-running `synapse login` wires a harness you install later and never asks
for a new sign-in on an enrolled machine (`--reenroll` does that on purpose).

Settings live in `~/.synapse/config.json`, owned by no harness: `synapse config` lists
them, `synapse config <key> <value>` changes one (`url`, `role`, `skills_sync`,
`config_sync`, `config_paths`). Full client detail: [plugin/README.md](../plugin/README.md)
and [plugin-codex/README.md](../plugin-codex/README.md).

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
synapse import         # or `! synapse-import` inside a Claude Code session
```

It offers an optional date range (by each file's last-activity date), prints a summary
(file count, size, estimated turns), and asks for confirmation before sending anything —
importing runs KG extraction on your configured LLM for every new turn, which consumes
subscription usage or API credits roughly in proportion to the turn count, so bounding a
first import by date bounds the spend. The server dedups turns by `span_id`, so Ctrl-C
and re-running are always safe: an interrupted import resumes where it left off.

Cursor history can be imported too, but only as a server-side dev path for now
(`python -m ingestion.backfill --source cursor <path>`; the same driver takes
`--source codex|claude-ai|chatgpt`).

Web research is a separate, opt-in capture (`SYNAPSE_WEB_CAPTURE=on`): run
`synapse capture --dry-run` first to see what it would store from your history, then turn it
on, and `synapse capture --backfill --since 30d` imports past research. Rules for sources of
your own go in a directory you mount into the `mcp-server` container
(`SYNAPSE_CAPTURE_DIR`); they run as code on the server, so keep it read-only and trusted.
See [ARCHITECTURE.md §8](../ARCHITECTURE.md#8-web-research-subsystem).

## Model ids: one thing that bites

`SYNAPSE_LLM_MODEL` is a **provider-specific** id. A value with an `anthropic/` prefix, such
as `anthropic/claude-haiku-4.5`, is the OpenRouter spelling and is only valid with
`SYNAPSE_LLM_PROVIDER=openai`. The default `claude-code` provider passes the id straight to
the Claude CLI, which rejects it with empty output, and every extraction stage would then
fail with "no JSON object in response". Two guards stop that: `.env.example` ships the line
commented out, and a slashed id is refused at startup with an error naming the spelling it
wants (`claude-haiku-4-5`). The refusal happens while the extraction pipeline is being built,
before any work is claimed, so the poller container does not start at all: it crash-loops
with that message, the same shape as the missing-machine-token refusal. The same applies to a
slashed per-stage override (`SYNAPSE_TIMELINE_MODEL` and the rest), which is resolved at the
same moment. On the default provider, either leave `SYNAPSE_LLM_MODEL` unset or use a plain
Claude model name.

A failure the guard cannot see statically shows up differently, as a `failed` row in the
extraction queue carrying its reason instead of a silent `done`. Two cases reach that state:
a missing or expired credential (`claude CLI returned an error result: Not logged in · Please
run /login`), and a model name that is well formed and unslashed but that the CLI does not
recognise, a typo like `claude-haiku-45`, which comes back as the CLI's error result or as
`model returned no output on any of N attempt(s)`. The queue re-claims such a row after 30
minutes, up to 5 attempts, so fixing the configuration is enough: those turns pick up their
facts on a later cycle rather than staying graph-less. Episodes are unaffected throughout,
stored and recallable even when extraction fails.

## Personal scope (optional)

Synapse keeps two knowledge graphs: `technical` (the default, and what `recall()` searches
unless told otherwise) and `personal` (family, health, job search, anything about the
owner's life rather than their work). Extraction routes each entity and timeline event to
one of them by name patterns and by the `SYNAPSE_PERSONAL_PROJECTS` list.

If nobody will ever talk to this server about their life, turn the split off:

```
SYNAPSE_PERSONAL_SCOPE=0
```

Everything then lands in the single technical graph, and a recall that asks for the
personal scope is answered from it. Work-only deployments should set this: with the split
on, a technical entity that happens to match a personal word ("sleep" in a retry loop,
"book" in a domain model) lands in the graph the default search never reads. The switch
changes routing from now on; rows already written keep their group.

This is a different "personal" from the machine role in step 4. The role says what a
machine may be served. The scope says which graph a fact is written to.

## Local inference (no external accounts)

Everything below is one file. Copy [`examples/env/local.env`](../examples/env/local.env)
instead of editing variables by hand:

```bash
cp examples/env/local.env .env
```

It ships the whole recipe already applied (provider, base URLs, model ids,
`SYNAPSE_EMBED_DIMS=768`, and `COMPOSE_PROFILES=local-inference`, so a plain
`docker compose up -d --build` starts the inference container), and `VOYAGE_API_KEY` stays
blank. Honest caveat: retrieval quality was measured on the Voyage stack; the local models
are a convenience floor and have not been benchmarked.

One thing you cannot fix later: embedding width is fixed at first-boot schema provisioning
and recorded in `synapse_meta`, so `SYNAPSE_EMBED_DIMS=768` has to be in place **before** the
first `docker compose up`. The server fails loudly on a later mismatch rather than serving
garbage, and switching stacks after the fact means an empty data volume.

## Ports and a second instance

`MCP_PORT` sets both the server's listener and the published host port. `POSTGRES_HOST_PORT`
does the same for Postgres, on `127.0.0.1` only (the compose file has a commented line to
publish it on every interface; an existing deployment must re-apply `compose.yml` for a
change to the binding to take effect). To run a second stack on the same host (a test instance
alongside a live one), set both plus `COMPOSE_PROJECT_NAME`, so the two get separate
containers and volumes.

## Upgrading

First-boot init never re-runs on an existing data volume. To bring an existing database up
to date, run [`scripts/apply_schema.sh`](../scripts/apply_schema.sh) (the single source of
truth for migration order) against it. It applies only the files newer than the database's
`schema_version` stamp (a fresh database gets all of them), so "pull, run `apply_schema.sh`,
restart" is the whole upgrade. `--all` re-applies every file (each is safe to repeat, but it
repeats their backfills too, so it is slow on a big database) and `--from NNN` starts at a
number; a database that has tables but no stamp (it predates migration 034) is refused until
you pass one of them. The `postgres` service now has `shm_size: 1g`; re-apply `compose.yml`
(`docker compose up -d --remove-orphans`) to pick that and any other compose change up. This
recreates the postgres container, so Postgres restarts: run `docker compose stop -t 120
postgres` first so it shuts down cleanly (Docker's default is 10 s, and an unclean stop can
corrupt the BM25 index). If BM25 recall degrades afterwards, run
`REINDEX INDEX CONCURRENTLY episodes_search_idx` (ARCHITECTURE.md §15).

Every service verifies at boot that the database schema matches the code (the script
stamps the applied version; a mismatch refuses to start with instructions rather than
failing mid-request). `SYNAPSE_SCHEMA_CHECK=0` skips the guard. A database that is behind
only by migrations marked `-- schema-check: optional` boots with a warning: the code shipped
with them runs without them and keeps its old behaviour until they are applied, so the image
can land first. A migration that builds an index `CONCURRENTLY` cannot run in a
transaction: do not use `psql -1`, and if a build fails drop the INVALID index it left and
run the file again.

**The optional migrations** of the current release, all safe to apply after the image is
running (HANDOFF.md has the order and what each needs by hand). `apply_schema.sh` runs each
file whole, so on a populated database do the two hand steps below (`062`, `072`) BEFORE
running it; it then finds their statements already applied.

- `062` `extraction_queue.chunk_id` and its indexes (on a populated queue build the
  `episode_id` index `CONCURRENTLY` first; the command is in the file header);
  `063` the partial index of unembedded episodes (`CONCURRENTLY`).
- `064` `timeline_events.reported_refs`; `068` the dream lane cursors; `069` / `070` the
  skills ledger's open-identity index and draft-attempt columns (apply `069` soon after the
  image, and never roll the image back past it); `071` `notes_history`; `073` `deleted_turns`
  (delete tombstones); `076` `research_archive`.
- `065` the live-edge hop indexes, `066` the retrieval-count index drop and fillfactor,
  `067` the chunks vector index (only needed to turn on `SYNAPSE_RECALL_CHUNK_CANDIDATES`),
  `074` the GIN on `kg_relationships.episodes`, `075` the redundant-index drop: all
  `CONCURRENTLY`. Apply `074` before any project relabel.
- `072` backfills `t_created` / `entity_subtype` for the dashboard. On a populated table run
  its three `UPDATE`s, then build `kg_rel_t_created_idx` `CONCURRENTLY` by hand with the
  command in the file header, instead of running the file whole. Do this, and the `062`
  index above, before `apply_schema.sh`.

Two one-off scripts accompany this release, both idempotent and counts-only (run each with
`--dry-run` first): `python scripts/backfill_normalized_name.py` (entities with a NULL
`normalized_name` are invisible to dedup now that the unindexed fallback lookup is gone, with
no migration involved) and `python scripts/restamp_date_only_dates.py` (fact dates stored at
UTC midnight move to noon UTC, so the user's calendar day survives).

Anything outside the repo that runs `python -m ingestion.cursor_sqlite_backfill`,
`ingestion.codex_backfill`, `ingestion.claude_ai_backfill` or `ingestion.openai_backfill`
must switch to `python -m ingestion.backfill --source {cursor,codex,claude-ai,chatgpt} <path>`
before the image lands; those modules are gone. A client-side web-capture cron is retired
the same way once `SYNAPSE_WEB_CAPTURE=on` (ARCHITECTURE.md §8).

Clients follow the server: `synapse update` on each machine fetches the client version the
server now serves (the same `/install`), updates the Claude Code plugin, and keeps Codex's
hook commands as they were, so their trust survives. An install from before 0.20 keeps
working with no action, reading its old Claude plugin settings; the first `synapse login`
copies those answers into `~/.synapse/config.json`.

Releases are tagged (`v0.8.1`, ...). `main` is kept releasable, but for a known-good
build check out the latest tag; a change that needs a migration or renames an env var
gets a release note saying so.

## Troubleshooting

- **The board is empty but the URL is right, and recall returns nothing.** The machine is
  being served restricted: either it never enrolled, or it is presenting the shared machine
  token instead of a device token. Mint one with
  `docker compose exec mcp-server synapse-admin bootstrap "<label>"` and store it with
  `synapse login --token`, or run `synapse login` on a server with sign-in. The
  SessionStart block says which
  case you are in, and so does the server itself: a `WARNING:` line under the board header
  and the first entry of recall's `warnings` list, which reach even an old plugin.
- **`POST /recall` returns `{"facts": []}` with no `episodes` key.** The embedding call
  failed, almost always a bad or blank `VOYAGE_API_KEY`. The HTTP status is still 200, but
  the response now carries a `warnings` list naming the leg that degraded, for example
  `embedding failed (voyage: Unauthorized): vector legs skipped, results are BM25-only.
  Check VOYAGE_API_KEY.` The same list appears on the `recall` and `recall_full_turns` MCP
  tools. A `warnings` key means broken retrieval, not empty memory. For the full error, run
  `docker compose logs mcp-server` and look for `Embedding query failed`.
- **Episodes appear but the graph stays empty.** KG facts are extracted from 4-turn-or-longer
  windows on a poll cycle (default 5 min); a short session or a fresh import needs a few
  minutes. If it stays empty for longer, check the extraction LLM: a bad credential or an
  unrecognised model name leaves `failed` extraction rows carrying the reason, and the queue
  retries them once the configuration is fixed. A slashed model id is a different symptom,
  a poller that never starts. See [Model ids](#model-ids-one-thing-that-bites).
- **`401 Unauthorized`.** Token missing, stale, or revoked. Run `synapse login --reenroll`,
  or store a fresh device token with `synapse login --token`.
- **Session start says "Synapse uploads have been failing since …".** The last several
  transcript uploads failed, with no success since; the reason is in brackets. `HTTP 401` or
  `HTTP 403` usually means a revoked or stale token, or a proxy or WAF in front of the server.
  `connection refused` or `timed out` means the server is down or unreachable. Nothing is
  lost: every session start retries the failed transcripts, at any age, until they upload.
  The line goes away after the first successful upload. The hook log
  (`~/.synapse/logs/ingest-hook.log`) has each failure. Codex CLI
  sessions show the same line from the Codex hooks, which log to
  `~/.synapse/logs/codex-hook.log`.
- **"SYNAPSE_INGEST_TOKEN from your environment is ignored" at session start.** An old token
  is still exported from your shell profile or from the `env` block of `settings.json`. This
  machine is enrolled, so the hooks already use its device token. Delete the export from the
  place the line names; nothing else needs to change.
- **Recall returns nothing and the hooks seem dead.** Hooks are fail-soft, so an unreachable
  server is a silent no-op. Check `curl -fsS $SYNAPSE_URL/health`, then `synapse config`:
  the URL should be right and the token set.
- **Recall answers 401 but the hooks work.** The MCP connection authenticates through a
  headers helper that reads `~/.synapse/config.json`; when it can't run it falls back to the
  plugin's static token. The helper's failures (wrong server, no token) are logged to
  `~/.synapse/logs/mcp_headers.log`. `synapse install` puts the harnesses' own copy of the
  URL back in step with `config.json`.
- **`synapse` not found.** Install the client (step 4) and add `~/.synapse/bin` to `PATH` as
  the installer printed. Inside Claude Code sessions the plugin's own `bin/` provides
  `synapse` and `synapse-login` too.
- **Codex hooks don't run.** Codex skips hooks nobody trusted: run `/hooks` in `codex` and
  trust the Synapse ones. `synapse update` never changes their commands, so trust survives
  updates.
- **`python3: not found`.** The hooks and scripts are stdlib Python 3, so put it on PATH.
- **Device-flow login fails on an older server.** It needs `/device/code` and "Enable Device
  Flow" on the GitHub OAuth App; fall back to `synapse login --browser`.
- **A service refuses to start, complaining about the schema version.** Run
  `scripts/apply_schema.sh` against the database, then restart (`scripts/apply_schema.sh
  --all` when the database has no stamp). See [Upgrading](#upgrading).

## Next

- [docs/auth.md](auth.md): device tokens, enrollment, exposing the server to claude.ai.
- [docs/tools.md](tools.md): what the MCP tools and hooks actually do.
- [ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration): every configuration variable.
