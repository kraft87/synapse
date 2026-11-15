# Installing Synapse

The full path from an empty directory to Claude Code, Codex or Hermes sessions that recall
your history. The short version lives in the [README](../README.md#quick-start).

Two halves, and both are needed: the **server** (Docker, one box) and the **client** (one
install per machine you code on; it wires Claude Code, Codex and Hermes).

**You need:**

- On the server box: Docker with the Compose v2 plugin (`docker compose version` must work)
  and git. The first build takes about 3 minutes on Linux.
- On each machine you code on: Python 3.11 or newer, and curl (PowerShell on Windows). On
  Linux and macOS it must run as `python3`: check with `python3 --version`.
- Accounts, depending on the `.env` you pick below: a Voyage AI key, and a Claude
  subscription token (`claude setup-token`) or an Anthropic API key.

One server holds one person's memory. Two people need two servers.

**On Windows** (native, not WSL), the same steps work from PowerShell. You need:

- Docker Desktop with the WSL 2 backend, for the server. Start it once and wait until it
  says the engine is running. Its sign-in screen can be skipped.
- Git for Windows. Claude Code runs the Synapse hooks with its Git Bash.
- Python 3.11 or newer from python.org. Tick "Add python.exe to PATH" in its installer.
  Without that tick, use `py -3` wherever this guide says `python`.

`python3` in a Windows terminal often prints "Python was not found" and offers the
Microsoft Store. That is fine: the client never needs it. Each step below gives the
PowerShell form where it differs.

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
| `SYNAPSE_DB_PASSWORD` | Password for the bundled Postgres. Must match the password inside `SYNAPSE_DB_URL`. Generate it with `openssl rand -hex 24` (Windows: `python -c "import secrets; print(secrets.token_hex(24))"`): characters such as `@`, `:` or `/` break the URL. |
| `SYNAPSE_DB_URL` | Postgres DSN. The `.env.example` default just needs the same password filled in. |
| `SYNAPSE_MACHINE_TOKEN` | The shared root bearer. `.env.example` ships it as `CHANGEME`; generate a real one with `openssl rand -hex 32` (Windows: `python -c "import secrets; print(secrets.token_hex(32))"`). |
| `VOYAGE_API_KEY` | Voyage AI key, used for embeddings and rerank. Skip it only if you configure an alternative backend (see [local inference](#local-inference-no-external-accounts)). |
| `CLAUDE_CODE_OAUTH_TOKEN` **or** `ANTHROPIC_API_KEY` | Auth for the extraction LLM. The subscription token wins if both are set. |
| `SYNAPSE_LLM_API_KEY` | Only with `openrouter.env`: your OpenRouter key, in place of the Claude token. |
| `SYNAPSE_TIMEZONE` | Your IANA zone, for example `Europe/London`. It decides which day "today" is. The default is `America/Toronto`. |
| `SYNAPSE_OWNER_NAME` | Your first name, the way memory should refer to you (for example `Sam`). Commented out in the presets; uncomment it. Set it before your first session: facts stored while it is unset say "User" and need `scripts/rename_owner_in_facts.py` to fix later. |

On Windows, edit `.env` in Notepad (`notepad .env`). Do not write it with PowerShell's `>`
or `Out-File`: Windows PowerShell writes those files as UTF-16, which Docker Compose cannot
read.

`SYNAPSE_MACHINE_TOKEN` is required: the server refuses to boot while it is blank or still
the `CHANGEME` placeholder, and prints what to do about it. The one exception is
`SYNAPSE_ALLOW_OPEN=1`, a dev-and-stdio-only mode that starts without a token and then
serves **every** caller restricted, which means an empty board and empty recalls. It is for
poking at a throwaway stack, not for running one.

Optional knobs (`SYNAPSE_INGEST_TAIL`, `POLL_INTERVAL_SECONDS`, recall-serving tuning such
as the `SYNAPSE_RECALL_FLOOR` abstention floor, which every `examples/env` preset pins to
the value calibrated for its reranker, and more) are listed in
[ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration). Set `LOGFIRE_TOKEN` to stream
traces to [Pydantic Logfire](https://logfire.pydantic.dev); leave it blank and telemetry is
fully off.

## 2. Start the stack

```bash
docker compose up -d --build
```

This builds the image from public bases (no registry auth), starts Postgres, the two pollers,
the MCP server and the `dream` container, and applies the schema automatically on Postgres's
first boot. On Windows with Docker Desktop the first build takes 10 to 20 minutes.

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

Copy the token: `synapse login --token` asks for it and stores it on this machine in the next
step.
It is shown once because only its hash is stored, so if you lose it, run `bootstrap` again
for a new one and revoke the old row.

Why this step exists: what a machine is served depends on **its own**
device token, not on the shared machine token and not on any name it reports. The shared
machine token identifies the deployment rather than a machine, so a caller presenting it is
served restricted, which looks like an empty board and empty recalls. There is no fail-open
path here: an unrecognised caller is served nothing rather than everything.

On a server that has GitHub OAuth or OIDC configured, a machine can enroll itself instead
with plain `synapse login`, and the bootstrap command is the break-glass path. See
[docs/auth.md](auth.md).

## 4. Install the client and log in

On each machine you code on. Use the address that machine reaches the server at:
`localhost` works only on the server box itself, and the port is your `MCP_PORT` if you
changed it. The installer records that address as this machine's server. The server serves
its own client, so the `synapse` command and the Codex and Hermes wiring match the server's
version. The Claude Code plugin comes from the GitHub marketplace instead, so keep the
server on a recent release.

```bash
curl -sSL http://localhost:8765/install | python3 -
# Windows PowerShell:
irm http://localhost:8765/install | python -
```

That unpacks the client into `~/.synapse/lib/<version>/`, puts a `synapse` command in
`~/.synapse/bin`, and records the server URL. To make the command run from any shell it
then does the least it can: where `~/.local/bin` is already on `PATH` it links `synapse`
into it (never over a file of that name that is not its own link); otherwise it prints the
line to add to your shell profile; on Windows it adds `~/.synapse/bin` to your user `PATH`.
On Linux and macOS it needs Python 3.11 or newer as `python3` on `PATH`, because the hooks
run `python3`, not the interpreter you install with. Check with `python3 --version`: macOS's
built-in one is 3.9 and Ubuntu 22.04's is 3.10, and with those the install succeeds but
nothing is captured. On Windows the Claude Code hooks try `python3`, `python` and `py -3`,
in that order, and use the first that is Python 3.11 or newer, so the Microsoft Store
stand-in is skipped. They run in Git Bash, so Git for Windows must be installed. After
`synapse login`, `synapse status` says on Windows when the hooks will find no Python.
Then:

```bash
synapse login --token     # server without sign-in: paste the step 3 token when asked
synapse login                               # server with GitHub OAuth / OIDC: one approval
```

`synapse login` is the one setup step. It asks this machine's questions once and saves the
answers in `~/.synapse/config.json`:

- **personal or work**: `personal` (default, sees everything) or `work` (restricted:
  work-safe notes plus an allowlist of projects, so personal memory is never served there).
  It is declared when the machine enrolls. With `--token` it is not asked: the token's trust
  decides, and a `bootstrap` token is full trust.
- **Sync skills to this machine?**: two-way sync of your skills library at session start.

Then it wires every harness it finds, and each needs one step from you:

- **Claude Code**: it installs the plugin (`claude plugin marketplace add kraft87/synapse`,
  `claude plugin install synapse@synapse`) with the server URL. Restart Claude Code or run
  `/reload-plugins`.
- **Codex**: it writes the hooks and the `synapse` MCP server into `~/.codex/config.toml`.
  Start `codex`: after the folder-trust prompt it shows "Hooks need review"; choose "Trust
  all and continue" (dismissed it? run `/hooks` and press `t`). Untrusted hooks are skipped
  with no message, so that session gets no memory and uploads nothing. The first time Codex
  calls each Synapse tool (`recall`, `remember`, `fetch`, `recall_feedback`) it asks for
  approval: choose "Always allow", or a stray keypress can cancel a memory write.
- **Hermes**: it installs a memory provider in `$HERMES_HOME/plugins/synapse/` and selects
  it (`synapse install hermes` does that step alone). Start a new `hermes` session; one
  already open stays without Synapse. `hermes memory status` should show
  `synapse ← active`. Hermes runs one external memory provider at a time: if another one
  (Honcho, Hindsight, mem0) is selected, the output says `left memory.provider at '<name>'`
  and Synapse is not active. `synapse install hermes --replace` switches to Synapse, and
  `hermes config set memory.provider <name>` switches back. Local sessions only: see
  [plugin/README.md](../plugin/README.md#hermes).

`synapse status` shows which harnesses are wired. Re-running `synapse login` wires a harness
you install later and never asks for a new sign-in on an enrolled machine (`--reenroll` does
that on purpose).

Settings live in `~/.synapse/config.json`, owned by no harness: `synapse config` lists
them, `synapse config <key> <value>` changes one (`url`, `role`, `skills_sync`,
`config_sync`, `config_paths`). Full client detail: [plugin/README.md](../plugin/README.md)
and [plugin-codex/README.md](../plugin-codex/README.md).

### Another machine

Each machine gets its own token. The `bootstrap` from step 3 is for the first one; mint the
rest, then install there with the server's LAN or Tailscale address instead of `localhost`:

```bash
# On the server box (or `/synapse-devices mint "<label>" --full` in Claude Code on a set-up machine):
docker compose exec mcp-server synapse-admin mint "work laptop" --full

# On the new machine:
curl -sSL http://<server-address>:8765/install | python3 -
synapse login --token
```

`--full` serves that machine everything, like the first one. Without it the machine is
restricted: it sees only work-safe notes and the projects you allow with `--projects`. See
[docs/auth.md](auth.md).

## 5. Verify

Run these from the repo directory on the server box. The first line loads
`SYNAPSE_MACHINE_TOKEN` from `.env`; the second reads the device token that `synapse login`
stored in step 4.

```bash
set -a; . ./.env; set +a
SYNAPSE_DEVICE_TOKEN=$(python3 -c 'import json,pathlib;print(json.loads((pathlib.Path.home()/".synapse/config.json").read_text())["token"])')

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

In Windows PowerShell, use this block instead. `curl` there is a name for `Invoke-WebRequest`,
not curl, and it rejects curl's flags. `irm` (`Invoke-RestMethod`) does the same job, and
`curl.exe` is the real curl.

```powershell
$machine = (Select-String -Path .env -Pattern '^SYNAPSE_MACHINE_TOKEN=(.*)$').Matches[0].Groups[1].Value.Trim()
$device = (Get-Content $HOME\.synapse\config.json -Raw | ConvertFrom-Json).token

# Server is up:
irm http://localhost:8765/health

# 1. Ship a test transcript (>= 4 turns) to the server:
irm -Method Post http://localhost:8765/ingest -ContentType 'application/json' -Headers @{Authorization = "Bearer $machine"} -InFile docs\example-transcript.json
# 2. Wait one poll cycle (up to ~5 minutes), then ask for it back:
irm -Method Post http://localhost:8765/recall -ContentType 'application/json' -Headers @{Authorization = "Bearer $device"} -Body '{"query": "what caching layer did we pick for the demo app search service"}'
```

Recall is served according to the bearer, so use the device token from step 3 for the recall
call, not the shared machine token. The test transcript stays in your memory as the project
`synapse-install-test`. To keep memory clean, skip the two calls and check with a real
session instead (below).

Two timing expectations that look like bugs but aren't: **episodes** appear within seconds
of ingest, but **knowledge-graph facts** are extracted from 4-turn sliding windows — a
session with fewer than 4 human turns produces episodes and an *empty graph*. And extraction
runs on a poll cycle (`POLL_INTERVAL_SECONDS`, default 300), so first facts land a few
minutes after ingest, not instantly. Embeddings wait for that cycle too, so for the first
few minutes recall matches on words only.

From inside Claude Code: start a fresh session. The SessionStart board block's banner reports
the episode count and the most recently active projects, so yours should appear with a rising
count.

## 6. Import your existing history

A fresh install doesn't have to start cold. If you've been using Claude Code, months of
transcripts already sit in `~/.claude/projects` — import them once and `recall()` knows your
history on day one:

```
synapse import
```

Run it in a terminal: the date range and the confirmation need a keyboard (`--yes` skips
the confirmation in scripts). It imports Claude Code history. Past Hermes sessions have
their own command, with the same prompts:

```
synapse import hermes          # --hermes-home PATH for another Hermes home
```

It imports the local sessions the Hermes provider would have uploaded, and skips the
rest: see [plugin/README.md](../plugin/README.md#hermes).

Past Codex sessions go through a server-side backfill. On the server box, from the repo
directory (`--dry-run` first shows the count; `--since-days N` bounds the spend):

```bash
docker compose run --rm -v ~/.codex/sessions:/codex:ro mcp-server \
  python -m ingestion.codex_backfill /codex --dry-run
```

Drop `--dry-run` to import. Turns already uploaded by the hooks are skipped.

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

## One memory per owner

Synapse keeps one knowledge graph per owner: there is no separate "personal" graph and no
setting that splits it, so a work-only install and a mixed one are configured the same way.
What a machine is served is decided by its role from step 4 (`personal` is full trust,
`work` is restricted) together with two facts about each row: its audience label and where it
came from. A restricted machine is served only facts whose source sessions sit inside its
project allowlist and that pass the audience-label rule, and only `work-safe` notes
([docs/auth.md](auth.md#what-a-restricted-machine-is-served)).

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
does the same for Postgres. To run a second stack on the same host (a test instance
alongside a live one), set both plus `COMPOSE_PROJECT_NAME`, so the two get separate
containers and volumes.

Compose publishes Postgres on every interface of the host. On a laptop, or anywhere other
machines can reach, keep it local with `POSTGRES_HOST_PORT=127.0.0.1:5432`. The MCP port has
to stay reachable from your other machines, and plain http sends the device token
unencrypted, so between machines use a VPN such as Tailscale or WireGuard, or put a TLS
reverse proxy in front.

## Upgrading

First-boot init never re-runs on an existing data volume. To bring an existing database up
to date, run [`scripts/apply_schema.sh`](../scripts/apply_schema.sh) against it, from a
checkout of the version you are upgrading to: `scripts/apply_schema.sh "$SYNAPSE_DB_URL"`.
The script keeps a ledger, the `schema_migrations` table, with one row per schema file the
database has had. It applies the files that have no row, in order, and records each, so a
second run does nothing. `scripts/apply_schema.sh --status "$SYNAPSE_DB_URL"` prints what is
applied and what is pending, and changes nothing.

- **A database from before the ledger** carries only its stamp
  (`SELECT value FROM synapse_meta WHERE key = 'schema_version'`). The first run takes every
  file up to that number as applied, records them without running them, and applies the
  rest. If you skipped an optional file below the stamp, delete its row afterwards
  (`DELETE FROM schema_migrations WHERE version = 'NNN'`) and run the script again.
- **A file marked `-- schema-apply: manual`** needs you first on a database that holds
  episodes: 067 wants the pollers stopped, 069 has no undo. An upgrade does not run such a
  file. It leaves an optional one alone and names it, and it stops before a required one
  (exit status 2). Follow the file's section below, which ends in
  `scripts/apply_schema.sh --only NNN "$SYNAPSE_DB_URL"`, then run the script once more for
  the files after it. A database with no episodes runs these files in turn.
- **A file applied some other way** (piped to `psql` by hand) has no row, so the script
  would apply it again. Record it instead:
  `INSERT INTO schema_migrations (version, filename) VALUES ('NNN', 'NNN_name.sql')`.

Every service verifies at boot that the database schema matches the code (the script keeps
`synapse_meta.schema_version` at the highest number it has recorded; a database behind the
code refuses to start with instructions rather than failing mid-request). So the upgrade
order is: `git pull`, run `apply_schema.sh`, then `docker compose up -d --build` (a plain
restart keeps the old image). Run the script from the host with a DSN that points at
`localhost:<POSTGRES_HOST_PORT>`: the one in `.env` names the compose service `postgres`,
which only resolves inside the stack. The script needs `psql` and bash 4.4 or newer.
`SYNAPSE_SCHEMA_CHECK=0` skips the guard. A database that is behind only by migrations marked
`-- schema-check: optional` boots with a warning: the code shipped with them runs without
them and keeps its old behaviour until they are applied.

**Schema 067, one memory.** The release that makes the graph one memory per owner requires
schema 067, which merges the old `personal` group into `technical` and folds a personal
entity into its same-name technical twin. Its services refuse to start on a database that
has not applied it, so on a database that holds data apply 067 first and deploy the release
after (an older image on a database that already has 067 only logs a warning). The file runs
two transactions of its own, so it is applied by itself and never wrapped in `BEGIN` or
run with `-1`. It needs 066 applied first, and an upgrade applies the files up to 066 and
then stops before 067, which is marked manual.

1. Stop the pollers (`docker compose stop poller poller-drain`), so nothing writes graph
   rows during the merge.
2. Apply: `scripts/apply_schema.sh --only 067 "$SYNAPSE_DB_URL"`.
   It takes minutes on a large graph (graph writers wait while it runs; the merge itself
   does not block readers) and does nothing when there is nothing left to merge, so a
   re-run is safe. It stamps 067 only once the graph holds no personal row and no
   owner-only index is missing or the 066 insert trigger left.
3. Run `scripts/apply_schema.sh "$SYNAPSE_DB_URL"` once more for the files after 067.
4. Deploy the release and start the pollers.

The merge's preview, health check and undo (`scripts/pile_merge.py`) are retired, and so is
the ledger they read. A fresh install needs none of this: its first boot applies 067 with
nothing to merge.

**Schema 069, the cleanup.** Optional: no service needs it, and it is marked manual because
it has no undo, so an upgrade leaves it alone. Apply it to the running system, nothing
stopped, once 068 is in place and you have a dump:
`scripts/apply_schema.sh --only 069 "$SYNAPSE_DB_URL"` (after a failed step, run the same
command again). It drops `group_id` from the knowledge graph, notes and
preferences, the indexes that lead with it, and the `kg_pile_merge_067_*` tables. A
preflight refuses, changing nothing, below 068, on a row whose `group_id` is not
`technical`, on a missing owner-only index, and on anything else that depends on the column;
each refusal names what to look at. Each step waits at most 5 s for its lock and the file is
safe to run again.

Clients follow the server: `synapse update` on each machine fetches the client version the
server now serves (the same `/install`), updates the Claude Code plugin, keeps Codex's
hook commands as they were, so their trust survives, and rebuilds the Hermes provider folder
when Hermes is wired. An install from before 0.20 keeps
working with no action, reading its old Claude plugin settings; the first `synapse login`
copies those answers into `~/.synapse/config.json`.

Releases are tagged with the client version (`v0.24.4`, ...) and made automatically when
a new client version lands on main; their notes are that version's HANDOFF.md entries.
Server-only changes ship from `main` without a tag. `main` is kept releasable, but for a
known-good build check out the latest tag; a change that needs a migration or renames an
env var says so in its HANDOFF.md entry.

## Troubleshooting

- **Services can't log in to Postgres, or report an embedding width mismatch, after you
  edited `.env`.** Postgres keeps the password and the embedding width from its first start.
  On a fresh install with nothing to keep, `docker compose down -v` (this deletes the
  database) and then `docker compose up -d --build`.
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
  (`/tmp/synapse-ingest-hook.log`, `SYNAPSE_INGEST_LOG`) has each failure. Codex CLI
  sessions show the same line from the Codex hooks, which log to
  `/tmp/synapse-codex-hook.log` (`SYNAPSE_CODEX_HOOK_LOG`). On Windows both logs are in
  your temp folder: `%TEMP%\synapse-ingest-hook.log` and `%TEMP%\synapse-codex-hook.log`.
- **"SYNAPSE_INGEST_TOKEN from your environment is ignored" at session start.** An old token
  is still exported from your shell profile or from the `env` block of `settings.json`. This
  machine is enrolled, so the hooks already use its device token. Delete the export from the
  place the line names; nothing else needs to change.
- **Recall returns nothing and the hooks seem dead.** Hooks are fail-soft, so an unreachable
  server is a silent no-op. Check `curl -fsS $SYNAPSE_URL/health` (PowerShell:
  `irm <your server>/health`), then `synapse config`:
  the URL should be right and the token set.
- **Recall answers 401 but the hooks work.** The MCP connection authenticates through a
  headers helper that reads `~/.synapse/config.json`; when it can't run it falls back to the
  plugin's static token. The helper's failures (wrong server, no token) are logged to
  `~/.synapse/logs/mcp_headers.log`. `synapse install` puts the harnesses' own copy of the
  URL back in step with `config.json`.
- **`synapse` not found.** Install the client (step 4). On Linux and macOS the installer
  links the command into `~/.local/bin` when that directory is on `PATH`; when it is not,
  or a different `synapse` already sits there, it prints the line that adds `~/.synapse/bin`
  to `PATH` instead: add that line and open a new shell. Inside Claude Code sessions the
  plugin's own `bin/` provides `synapse` and `synapse-login` too.
- **Codex hooks don't run** (no board, no `hook:` lines, nothing in
  `/tmp/synapse-codex-hook.log`, on Windows `%TEMP%\synapse-codex-hook.log`). Codex skips
  hooks nobody trusted, silently: run `/hooks` in
  `codex` and press `t` to trust the Synapse ones. `synapse update` never changes their commands, so trust survives
  updates.
- **`python3: not found`.** The hooks and scripts are stdlib Python 3, so put it on PATH.
- **"Synapse: no Python 3.11 or newer found" at session start (Windows).** The Claude Code
  hooks tried `python3`, `python` and `py -3` and found no Python 3.11 or newer. Install
  Python from python.org, then start a new session. `synapse status` runs the same check.
- **"Git for Windows not found" from `synapse status` (Windows).** Claude Code runs plugin
  hooks with Git Bash, and runs them in PowerShell without it, where the Synapse hooks do
  not work. Install Git for Windows, then start a new Claude Code session.
- **Device-flow login fails on an older server.** It needs `/device/code` and "Enable Device
  Flow" on the GitHub OAuth App; fall back to `synapse login --browser`.
- **A service refuses to start, complaining about the schema version.** Run
  `scripts/apply_schema.sh` against the database, then restart. See
  [Upgrading](#upgrading).

## Next

- [docs/auth.md](auth.md): device tokens, enrollment, exposing the server to claude.ai.
- [docs/tools.md](tools.md): what the MCP tools and hooks actually do.
- [ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration): every configuration variable.
