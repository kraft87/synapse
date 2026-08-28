# Synapse — Claude Code plugin

Connects Claude Code to a [Synapse](../) memory instance — either local (Docker on your
machine) or central (one hosted server shared by all your machines).

The plugin is a thin client: stdlib-only scripts that talk to the server over HTTP with a
base URL and an optional bearer token — never a database connection. Extraction, recall,
and the dream→skills lane all run server-side.

## What it does

Seven hooks (`hooks/hooks.json`) plus MCP wiring:

1. **Transcript ingest** (`Stop`) — after every turn, pushes a bounded tail of the session
   transcript to `/ingest`. This is how sessions become memory, and it works over the
   network, so a work laptop can feed a central Synapse. It only captures sessions from
   install onward — run `synapse-import` once to backfill
   ([step 4](#4-import-your-existing-history-recommended)).
2. **Skill sync** (`SessionStart`) — two-way sync between the server and `~/.claude/skills`
   (plus the project's `.claude/skills`). Newest edit wins per skill, with an append-only
   server-side history; deletes never auto-propagate.
3. **Config mirroring** (`SessionStart`, off by default) — mirrors your `CLAUDE.md` +
   `rules/*.md` to the server so the dream lane can propose config edits.
4. **Timeline git feeder** (`SessionStart`, off until configured) — pushes commit subjects
   from repos listed in `SYNAPSE_TIMELINE_REPOS` to the server's timeline.
5. **Board block** (`SessionStart`) — prints the board (`GET /context`) into the session's
   context: curated note hooks, the last week's milestones, and what memory exists at all.
   Server-rendered and hard-capped (~80 lines / ~2K tokens), scoped to the session's
   project.
6. **Private-mode cleanup** (`SessionEnd`) — removes the local private-mode marker for the
   session that just ended (see [Private mode](#private-mode)). The server-side flag is
   left in place on purpose.
7. **Memory-write spool** (`PostToolUse` + `SessionStart`) — a `remember()` that fails is
   queued to local disk and replayed when the server is back, so a memory write is never
   silently lost (see [Memory-write spool](#memory-write-spool)).

MCP tools (`recall`, `fetch`, `remember`, …) are registered automatically —
no hand-written `.mcp.json`.

The dream→skills lane (mines your transcripts → proposes new skills, retunes triggers,
nominates merges) runs server-side in the `dream` container — you don't run a cron. You
just review what it proposes with `/synapse:skill-review`.

> Recall-injection (a `UserPromptSubmit` hook that pushed memory into every prompt) was
> removed: unconditional top-k injection added noise and anchored the model on stale
> priors. Use the `recall` MCP tool to pull on demand instead.

## What gets sent to your server

Everything below goes only to the Synapse URL **you** configure. All hooks are fail-soft:
an unreachable server is a silent no-op.

- **Transcript ingest** — **on** (the core function). Sends the raw JSONL tail of each
  session transcript — your prompts, Claude's replies, tool calls. Per-session opt-out:
  [private mode](#private-mode). No global toggle: if transcripts shouldn't leave the
  machine at all, don't install the plugin.
- **Skill sync** — **off** (opt-in). When enabled, sends skill bodies + bundled files and
  pulls server versions back into `~/.claude/skills` at session start. On:
  `SYNAPSE_SKILLS_SYNC=1`.
- **Config mirroring** — **off** (opt-in). When enabled, sends your `~/.claude/CLAUDE.md` +
  `rules/*.md` and the project's equivalents — these often carry personal instructions,
  which is why it ships off. On: `SYNAPSE_CONFIG_SYNC=1`.
- **Timeline git feeder** — **off** (opt-in). When `SYNAPSE_TIMELINE_REPOS` is set, sends
  commit subjects, dates, and a coarse salience score from those repos. Unset = nothing runs.
- **Board block** — **on**, but it *sends* nothing: it reads `GET /context` and prints one
  bounded index block into your context. Off: `SYNAPSE_BOARD=0`.
- **Memory-write spool** — **on**. Sends nothing extra: it replays a `remember()` you already
  asked for but that failed to reach the server. Until it succeeds the note sits in a local
  jsonl file, and you can inspect or drop it (`remember_spool.py list`, or delete the file).

## Configuration

Settings resolve in order: env var → `CLAUDE_PLUGIN_OPTION_*` → your `/plugin install`
answers (stored in `settings.json`) → built-in default. A fresh install just answers the
install prompt; env vars are optional overrides (e.g. CI).

Prompted at install:

- **`SYNAPSE_URL`** (required) — base URL of your server, no path (`http://localhost:8765`
  or `https://synapse.example.net`). The plugin derives `/ingest`, `/recall`, `/skills`,
  `/timeline`, and `/mcp`.
- **`SYNAPSE_INGEST_TOKEN`** — bearer token for an auth-gated server; blank for a
  local/open one. One token covers ingest, recall, skill sync, and MCP. Fetch it with
  `! synapse-login`.
- **`SYNAPSE_CONFIG_SYNC`** — `1` to mirror config for the dream lane. Off by default.
- **`SYNAPSE_CONFIG_PATHS`** — optional extra globs (relative to `~/.claude`) to mirror.

Env / `settings.json` only:

- **`SYNAPSE_SKILLS_SYNC`** — `1` enables two-way skill sync (default off).
- **`SYNAPSE_TIMELINE_REPOS`** — comma/space-separated repo paths for the timeline feeder.
- **`SYNAPSE_BOARD`** — `0` disables the session-start board block.
- **`SYNAPSE_RECALL_NUDGE`** — `0` disables the per-prompt recall/remember reminder.
- **`SYNAPSE_RECALL_FEEDBACK_NUDGE`** — `1` enables a post-recall reminder to label
  results via `recall_feedback` (default off — the labels are offline tuning data,
  only useful if you're tuning the retrieval stack).
- **`SYNAPSE_PROMPT_TIMESTAMP`** — `0` disables the per-prompt timestamp line
  (`[Wed 2026-08-26 15:55:01 EDT]`, host-local time; on by default so remember()
  and timeline references know the current moment, weekday included).
- **`SYNAPSE_INGEST_URL`** — legacy full-endpoint override, still honored.

## Setup

Prerequisites: a running Synapse server (below), Claude Code, and Python 3 on PATH —
everything is stdlib, no `pip install`.

### 1. Stand up Synapse (the server)

```bash
cd <repo> && docker compose up -d
```

This provisions Postgres, applies the schema, and starts the poller, MCP server, and the
`dream` container. For a central deployment, expose the one server URL behind auth and
point every machine's plugin at it.

### 2. Install the plugin (the client) on each Claude Code machine

The repo is its own marketplace (it ships `.claude-plugin/marketplace.json`), so there's
nothing to publish:

```
/plugin marketplace add kraft87/synapse
/plugin install synapse@synapse
```

Claude Code prompts for the configuration above and stores secrets in the OS keychain — no
hand-editing `settings.json` or `.mcp.json`. Then `/reload-plugins` (or restart) to activate
the hooks + MCP server.

### 3. Auth (only for auth-gated servers)

The local quickstart (`docker compose up` + `http://localhost:8765`) needs no auth — leave
the token blank and skip this section.

For a central/hosted server, paste a machine token into the install prompt, or run the
bundled login once:

```
! synapse-login        # in the Claude prompt — runs in-session, no LLM, streams live
```

Default is the GitHub **device flow** (RFC 8628): it prints a short code, you approve at
`github.com/login/device` from any device (phone, another laptop), and it polls until done —
no same-host browser, so it works on servers and headless boxes. It needs a server exposing
`/device/code` with "Enable Device Flow" on the GitHub OAuth App; on older servers, use
`synapse-login --browser` (legacy loopback flow). Either way the token is stored for the
hooks and MCP server.

Truly headless with no second device? Set `SYNAPSE_INGEST_TOKEN` directly instead.

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
recently active projects, so yours should appear with a rising count. Knowledge-graph facts land
a few minutes later, on the poll cycle. Seeing nothing? Hooks fail silently by design —
check that `curl -fsS $SYNAPSE_URL/health` returns ok and that `SYNAPSE_URL` and any token
are set (`/plugin` shows the stored values).

## Commands and MCP tools

Slash commands:

- **`/synapse:skill-review`** — triage dream→skills proposals (new skills, trigger retunes,
  merges): accept / reject / promote. Nothing touches your live skills without an explicit
  accept.
- **`/synapse:config-review`** — triage dream→config proposals. Only relevant with config
  mirroring on.

Bundled commands (`!` prefix in a session; full path from an outside terminal):

- **`! synapse-login`** — fetch a machine token via GitHub device flow (or `--browser`).
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

MCP tools (registered automatically; Claude calls them during a session):

- **`recall`** — primary retrieval: reranked episodes + KG facts + web + history.
- **`recall_full_turns`** — raw-episode drill-down: complete unabridged turns; the
  retry when an overview recall comes back thin (formerly `recall_episodes`,
  then `recall(mode="turns")`).
- **`fetch`** — expand ids into full records: `e:N` episode ids from recall results and
  `n:N` note ids from the session-start board block (mixed lists fine).
- **`remember`** — write a curated memory (note + episode + graph extraction).
- **`recall_feedback`** — after using a recall's results, report which served ids helped,
  which were noise, and what was missing (offline labeled data; never changes ranking).

## Troubleshooting

- **Recall returns nothing / hooks seem dead.** Hooks are fail-soft — an unreachable server
  is a silent no-op. Check `curl -fsS $SYNAPSE_URL/health`, then that `SYNAPSE_URL` and any
  token are set (`/plugin`).
- **`401 Unauthorized`.** Token missing or stale. Re-run `! synapse-login`, or paste a fresh
  token into the `/plugin` config.
- **`synapse-login` / `synapse-import` not found.** The plugin's `bin/` is only on PATH
  inside Claude Code sessions (hence the `!` prefix). From a plain terminal, run the script
  by full path.
- **`python3: not found`.** The hooks and scripts are stdlib Python 3 — put it on PATH.
- **Episodes appear but the graph stays empty.** KG facts are extracted from ≥4-turn windows
  on a poll cycle (default 5 min); a short session or a fresh import needs a few minutes.
- **Device-flow login fails on an older server.** It needs `/device/code` + "Enable Device
  Flow" on the GitHub OAuth App; fall back to `! synapse-login --browser`.

## How it talks to Synapse

HTTP only: `/ingest`, `/skills/*`, `/config/publish`, `/timeline/*`, `/context`,
`/remember/spool`, and `/mcp`,
all under the one `SYNAPSE_URL`, gated by one machine token. The client holds no Postgres credentials,
and skill/config proposals are only ever applied through your explicit review commands.
