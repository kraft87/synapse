# Synapse — Claude Code plugin

Connects Claude Code to a [Synapse](../) memory instance — either local (Docker on your
machine) or central (one hosted server shared by all your machines).

The plugin is a thin client: stdlib-only scripts that talk to the server over HTTP with a
base URL and a bearer token — never a database connection. Extraction, recall, and the
dream→skills lane all run server-side.

## What it does

Thirteen hooks (`hooks/hooks.json`) plus MCP wiring. In short: a `Stop` hook ships each
session's transcript tail to `/ingest`, a set of `SessionStart` hooks print the board and
your standing preferences into context (and optionally sync skills and config), a
`UserPromptSubmit` hook keeps recall/remember present, and `PreToolUse`/`PostToolUse` hooks
wire session identity and the memory-write spool. Every hook is fail-soft: an unreachable
server is a silent no-op.

The per-hook table, the MCP tool list, every configuration variable, and exactly what leaves
your machine: **[docs/tools.md](../docs/tools.md)**.

The dream→skills lane (mines your transcripts → proposes new skills, retunes triggers,
nominates merges) runs server-side in the `dream` container — you don't run a cron. You
just review what it proposes with `/synapse:skill-review`.

## Setup

Prerequisites: a running Synapse server, Claude Code, and Python 3 on PATH — everything is
stdlib, no `pip install`.

### 1. Stand up the server

Not covered here: [docs/install.md](../docs/install.md) walks the whole server side, from
`git clone` to a verified `/health`. For a central deployment, expose the one server URL
behind auth and point every machine's plugin at it.

### 2. Get this machine a device token

What a machine is served depends on its **own** device token, so the shared machine token is
not enough: presented on its own it is served restricted, which looks like an empty board.
Two ways to get one:

- On a server with GitHub OAuth or OIDC configured, skip ahead and run `! synapse-login`
  after installing (below). It signs you in and enrolls this machine, writing the minted
  token into the config slot for you.
- Otherwise, on the server host:
  `docker compose exec mcp-server synapse-admin bootstrap "<label>"`. It prints a full-trust
  device token once (only the hash is stored). Copy it for the next step.

Details, roles, and revocation: [docs/auth.md](../docs/auth.md).

### 3. Install the plugin on each Claude Code machine

The repo is its own marketplace (it ships `.claude-plugin/marketplace.json`), so there's
nothing to publish:

```
/plugin marketplace add kraft87/synapse
/plugin install synapse@synapse
```

Claude Code prompts for the configuration and stores secrets in the OS keychain — no
hand-editing `settings.json` or `.mcp.json`. Paste the device token from step 2 as the
**Synapse token**, and answer **personal** or **work** for this machine's role. Then
`/reload-plugins` (or restart) to activate the hooks + MCP server.

To enroll instead of pasting, leave the token blank and run:

```
! synapse-login        # in the Claude prompt — runs in-session, no LLM, streams live
```

Default is the device flow (RFC 8628): it prints a short code, you approve at
`github.com/login/device` from any device (phone, another laptop), and it polls until done —
no same-host browser, so it works on servers and headless boxes. `--browser` falls back to
the legacy loopback flow. Either way the token is stored for the hooks and MCP server.

> `synapse-login` and `synapse-import` ship in the plugin's `bin/`, which Claude Code puts
> on PATH **inside sessions only** — hence the `!` prefix. From an outside terminal, run the
> script by full path, e.g.
> `python ~/.claude/plugins/cache/synapse/synapse/<version>/scripts/synapse_login.py`.

### 4. Import your existing history (recommended)

The `Stop` hook only sees sessions from install onward, but months of transcripts likely
already sit under `~/.claude/projects`. Import them once and recall works on day one:

```
! synapse-import        # in the Claude prompt — or by full path from any terminal
```

It discovers every transcript (oldest-first; `--projects-dir` to override), offers an
optional date range to import (by each file's last-activity date — bound a first import,
bound the spend), prints a summary — file count, total size, estimated turns — and **asks
for confirmation before sending anything**: importing runs KG extraction on the server's
LLM for every new turn, which consumes subscription usage or API credits (`--yes` skips
all prompts and imports everything, for scripted runs).
Files ship full-length in turn-aligned batches (`--batch-size`, default 500 records per
POST). Safe to Ctrl-C and re-run: the server dedups turns by `span_id`, so a re-run resumes
where it left off, and one bad file never stops the rest.

Cursor history is importable too, but only as a server-side dev path for now
(`python -m ingestion.cursor_sqlite_backfill` on the Synapse host).

### 5. Verify it's working

Run a few turns and end one — the `Stop` hook ships the transcript. Then start a fresh
session; the SessionStart board block's banner reports the episode count and the most
recently active projects, so yours should appear with a rising count. Knowledge-graph facts
land a few minutes later, on the poll cycle. Seeing nothing? Hooks fail silently by design —
work through [the troubleshooting list](../docs/install.md#troubleshooting).

## Commands

Slash commands:

- **`/synapse:skill-review`** — triage dream→skills proposals (new skills, trigger retunes,
  merges): accept / reject / promote. Nothing touches your live skills without an explicit
  accept.
- **`/synapse:config-review`** — triage dream→config proposals. Only relevant with config
  mirroring on.
- **`/synapse-devices`** lists, mints, or revokes the per-device credentials that decide
  what each machine is served ([docs/auth.md](../docs/auth.md)).

Bundled commands (`!` prefix in a session; full path from an outside terminal):

- **`! synapse-login`** — sign in and enroll this machine (or `--browser`).
- **`! synapse-import`** — backfill your existing history (step 4).
- **`! synapse-private on|off|status <session-id>`** — private mode (below).

Scripts (run by full path; see [Memory-write spool](#memory-write-spool)):

- **`scripts/remember_spool.py add|list|flush`** — queue a memory write while the server (or
  the MCP transport) is down, and replay it when it's back.

## Private mode

Take one session off the record — nothing from it becomes memory, ever:

```
! synapse-private on <session-id>
```

Two writes, both required, both verified — the command exits nonzero and says so if either
fails, so "off the record" is never claimed on a half-write:

- a marker file at `~/.synapse/private/<session-id>` — the `Stop` hook stats it before every
  POST, so the turns never leave the machine even with the server down. It expires after 12h
  and the `SessionEnd` hook removes it;
- a row in `private_sessions` on the server — the durable half. Your transcript stays on disk
  after the marker is gone, so this row is what stops a later catch-up sweep or
  `synapse-import` from ingesting the very turns the hook skipped.

`off` removes the marker and **keeps** the row: private turns already spoken stay
uningestable, which is the only honest reading of "we were off the record". `--forget`
deletes the row too, deliberately re-exposing that session to future imports.

Server requirement: schema 050 (`private_sessions`). Against an older server the toggle
fails loudly with `503 apply schema/050` rather than half-enabling.

## Memory-write spool

A `remember()` that can't reach the server used to be a lost memory — the model said "noted"
and nothing was written. Now the intent is queued to `~/.local/share/synapse-skills/remember_spool.jsonl`
(`SYNAPSE_DATA_DIR`) and replayed automatically:

- **`PostToolUse`** — when `remember()` comes back without a confirmed write, the intent is
  spooled and the model is told it is *queued*, not saved.
- **`SessionStart`** — probes the write lane. Server up and the spool non-empty → it flushes
  and prints `[Synapse] flushed N spooled memory write(s): …`. Server down → it prints one
  line telling the model to route memory writes through the CLI below until it recovers
  (this covers the case where the MCP server never connected at all, so no tool call — and
  no `PostToolUse` hook — ever fires).

```
python3 <plugin>/scripts/remember_spool.py add --hook "<one-line hook>" --body "<full note>" --type user
python3 <plugin>/scripts/remember_spool.py list      # what's queued
python3 <plugin>/scripts/remember_spool.py flush     # replay now
```

`add` also accepts the same fields as JSON on stdin, and tries an immediate write-through
before queueing. Nothing leaves the spool until the server confirms, and each intent carries
a client-generated id the server dedups on — so a flush interrupted mid-way neither loses an
intent nor writes one twice. Log: `/tmp/synapse-remember-spool.log` (`SYNAPSE_SPOOL_LOG`).

Server requirement: schema 052 (`remember_intents`) and the `/remember/spool` route. Against
an older server the flush fails loudly into the log and the spool simply keeps its intents.

## More

- [docs/tools.md](../docs/tools.md): hooks, MCP tools, configuration variables, and what
  gets sent to your server.
- [docs/auth.md](../docs/auth.md): tokens, enrollment, personal vs work.
- [docs/install.md](../docs/install.md): the server side, and troubleshooting.
