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
wire session identity and the memory-write spool. Session identity lets recall leave out the
turns the model still holds: the whole session, or after a compaction only the turns since
it. Every hook is fail-soft: an unreachable
server is a silent no-op.

The per-hook table, the MCP tool list, every configuration variable, and exactly what leaves
your machine: **[docs/tools.md](../docs/tools.md)**.

The dream→skills lane (mines your transcripts → proposes new skills, retunes triggers,
nominates merges) runs server-side in the `dream` container — you don't run a cron. You
just review what it proposes with `/synapse:skill-review`.

## Setup

Prerequisites: a running Synapse server, Claude Code (and/or Codex), and Python 3.11+ on
PATH — everything is stdlib, no `pip install`.

### 1. Stand up the server

Not covered here: [docs/install.md](../docs/install.md) walks the whole server side, from
`git clone` to a verified `/health`. For a central deployment, expose the one server URL
behind auth and point every machine at it.

### 2. Get this machine a device token

What a machine is served depends on its **own** device token, so the shared machine token is
not enough: presented on its own it is served restricted, which looks like an empty board.
Two ways to get one:

- On a server with GitHub OAuth or OIDC configured, `synapse login` (next step) does it:
  one approval signs you in and enrolls this machine.
- Otherwise, on the server host:
  `docker compose exec mcp-server synapse-admin bootstrap "<label>"`. It prints a full-trust
  device token once (only the hash is stored). Copy it for the next step.

Details, roles, and revocation: [docs/auth.md](../docs/auth.md).

### 3. Install the client and run `synapse login`

On each machine. The server serves the client matching its own version:

```
curl -sSL https://<your-synapse>/install | python3 -     # PowerShell: irm .../install | python -
synapse login                                            # or: synapse login --token  (asks for the step 2 token)
```

`synapse login` is the one setup step. It signs in and enrolls (the device flow, RFC 8628:
ONE short code you approve at `github.com/login/device` or your OIDC provider, from any
device, so it works on servers and headless boxes; `--browser` is the legacy loopback flow),
asks this machine's questions once (**personal** or **work**; sync skills here?), saves the
answers and the device token in `~/.synapse/config.json`, then wires every harness it finds.
For Claude Code that is this plugin, installed with your server's URL through Claude Code's
own CLI:

```
claude plugin marketplace add kraft87/synapse
claude plugin install synapse@synapse --config SYNAPSE_URL=<your-synapse>
```

Then `/reload-plugins` (or restart) to activate the hooks + MCP server. Re-running
`synapse login` on an enrolled machine keeps its device token and only re-wires;
`--reenroll` replaces the token.

Already in a session with the plugin installed the old way (`/plugin install`)? `! synapse-login`
runs the same command from the plugin's own `bin/`, and copies the answers the old install
prompt stored in `settings.json` into `~/.synapse/config.json`.

**How the plugin authenticates.** Hooks and scripts read the token from
`~/.synapse/config.json` (then, for installs from before 0.20, the plugin's saved options).
The recall/remember MCP server gets it from a headers helper, `scripts/mcp_headers.py`,
which Claude Code runs before each connection and which reads the same file; the only value
the plugin config itself must hold is the server URL, because Claude Code opens that
connection. The plugin's static `Authorization` header (`Synapse token` in `/plugin`) is the
fallback for when the helper can't run, and is best left blank. Don't export
`SYNAPSE_INGEST_TOKEN` from your shell profile or the `env` block of `settings.json`: Claude
Code never passes it to the helper, and on an enrolled machine the hooks ignore it too,
printing one line at session start saying where to remove it
([docs/auth.md](../docs/auth.md)).

Settings live in `~/.synapse/config.json`, owned by no harness:

```
synapse config                         # list
synapse config skills_sync on          # two-way skill sync at session start
synapse config config_sync on          # mirror CLAUDE.md + rules/*.md for the dream lane
synapse config url https://...         # same server, new address; also rewires the harnesses
```

> Inside a Claude Code session the plugin's `bin/` puts `synapse` and `synapse-login` on
> PATH (hence the `!` prefix). Everywhere else the client's shim in `~/.synapse/bin` does.

### 4. Import your existing history (recommended)

The `Stop` hook only sees sessions from install onward, but months of transcripts likely
already sit under `~/.claude/projects`. Import them once and recall works on day one:

```
synapse import          # or `! synapse-import` in the Claude prompt
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

A failed upload is not lost. The hook records it, and every session start's catch-up sweep
retries it, however long the outage lasted: a down server, a revoked token, a proxy answering
`403`. The sweep picks up any lagging transcript modified in the last
`SYNAPSE_INGEST_CATCHUP_DAYS` (default 3) plus every transcript whose upload failed, at any
age, `SYNAPSE_INGEST_CATCHUP_MAX` (default 20) files per session start. When uploads keep
failing, session start shows one line saying so:

```
Synapse uploads have been failing since 2026-01-31 09:12 (HTTP 403 Forbidden); transcripts will be retried automatically.
```

The sweep never picks up a transcript the hook didn't try to upload, so history from before
you installed the plugin still needs `synapse-import` (step 4).

## Commands

Slash commands:

- **`/synapse:skill-review`** — triage dream→skills proposals (new skills, trigger retunes,
  merges): list → show (a diff against the current skill) → accept or reject. Accept applies
  the change: the server writes it into its skill registry and the two-way skills sync
  delivers it to the machine that owns the skill at its next session start. Nothing touches
  your live skills without an explicit accept.
- **`/synapse:config-review`** — triage dream→config proposals. Only relevant with config
  mirroring on.
- **`/synapse-devices`** lists, mints, or revokes the per-device credentials that decide
  what each machine is served ([docs/auth.md](../docs/auth.md)).

CLI (`synapse …` from any terminal once the client is installed; `!` prefix in a session):

- **`synapse login`** (alias `synapse-login`) — sign in, enroll this machine with one
  approval, and wire Claude Code / Codex / Hermes. On an enrolled machine it keeps the existing
  device token; `--reenroll` replaces it, `--token` stores a minted one (it asks for it, hidden).
- **`synapse config [key [value]]`** — read or change this machine's settings.
- **`synapse install claude|codex|hermes`** — the wiring step alone (Hermes: below).
- **`synapse update`** — install the client version the server now serves, and update
  this plugin and every other harness already wired. The harnesses are updated by the
  client just installed, in a process of its own.
- **`synapse status`** — one line per harness: wired, found and not wired, or not found.
- **`synapse skill retire <name>`** — retire a skill everywhere (below).
- **`synapse skill scope <name> global|project:<label>`** — move a skill to another scope
  (below).
- **`synapse import`** / **`! synapse-import`** — backfill your existing history (step 4).
- **`synapse private on|off|status <session-id>`** — private mode (below).

## Retiring a skill

`synapse skill retire <name>` marks the skill retired on the server. At their next session
start, machines with skills sync on delete their copy if it is still exactly a version
Synapse delivered to them; a copy edited after the retirement is pushed back instead and
the skill is active again (newest edit wins, as everywhere in the sync). A folder Synapse
doesn't recognise is never touched, and deleting a skill locally never retires it.

## Moving a skill, and who gets which skills

A skill lives in one scope: `global` (every machine's `~/.claude/skills`) or
`project:<label>` (`.claude/skills` at that project's root, on machines that work in it;
the label is the one every hook uses: a `.synapse-project` marker, else the git toplevel's
name). `synapse skill scope <name> project:<label>` moves it, from a full-trust machine. A
push from a folder of another scope never moves it.

A work-role (restricted) machine is served only the project skills of the projects in its
allowlist, never global ones. Whenever a machine is no longer served a skill it holds
(moved away, or not its to read), it deletes its copy at the next session start if the copy
is still exactly a version Synapse delivered, the same rule as a retirement; an edited copy
stays and is not pushed back. A skill folder git tracks is never deleted or overwritten
(git distributes it), and a project skill Synapse pulls into a repository is listed in its
`.git/info/exclude`, not committed.

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
  POST, so the turns never leave the machine even with the server down. The check sits in
  the sender itself, so the catch-up sweep and `synapse-import` skip the session too. It
  expires after 12h and the `SessionEnd` hook removes it;
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

## Hermes

The same client wires [Hermes](https://github.com/NousResearch/hermes-agent) as a memory
provider. `synapse login` does it when it finds Hermes; to do it alone:

```
synapse install hermes                         # the home is $HERMES_HOME, else ~/.hermes
synapse install hermes --hermes-home PATH      # another home (a Hermes profile)
synapse install hermes --replace               # take over from another external provider
```

It writes `$HERMES_HOME/plugins/synapse/` (a copy of the client core and a `plugin.yaml`)
and sets `memory.provider: synapse` in that home's `config.yaml`: through `hermes config
set` when Hermes already has a config there, by writing a new two-line file when there is
none. Hermes runs one external memory provider at a time, so a different one already
selected is left alone unless you pass `--replace`; Hermes's built-in memory keeps working
either way. `synapse update` rebuilds the folder for a new client version, and `synapse
status` says whether it is wired. Restart Hermes to load it: `hermes chat` and the TUI run
their turns in Hermes's gateway daemon, so stop it with `hermes gateway stop` (the next
`hermes chat` starts it again).

**What it does.** In a session you run on this machine (the Hermes CLI, TUI, Desktop or an
ACP editor): the board and your preferences in the system prompt at session start, the
recall nudge each turn, the memory tools (`recall`, `recall_full_turns`, `fetch`,
`fetch_session`, `remember`, `recall_feedback`), and each finished turn uploaded.
Chat-platform sessions (Telegram, Discord, Slack and the rest, through Hermes's gateway),
the API server, the Bot Chat and Desktop Group Chats, cron jobs, side agents and subagents
get nothing, not even the tools, and a session stops for good at the first turn another
agent or person wrote: there, others talk to the agent, and they must not read your memory
or be stored as you. Such a chat is noted in `$HERMES_HOME/synapse/inert/`, so it and
its branches stay off after a rename or a restart. To turn one back on, delete every file
there that names it (each holds the id it is for), once nothing in the chat says Bot Chat
or Group any more. The reasons, the limits and the Hermes source behind them:
[docs/client-harnesses.md](../docs/client-harnesses.md).

**What leaves the machine.** Requests go only to the server in `~/.synapse/config.json`,
with this machine's device token. At session start: `GET /context` and
`GET /preferences/top`. Each tool call: one MCP `tools/call`. Each finished turn, not
one that failed: a `POST /ingest` (format `messages`, client `hermes`) holding the
session id, the project label and working directory, the model name, and per message its
role, text, tool-call names and arguments (first 1000 characters), tool results (first
2000) and timestamp.
Hermes scrubs secrets from what it hands a memory provider before the client sees it.
Not sent: the system prompt, Hermes's compaction summaries, the memory context it injects
into prompts, reasoning, and its own bookkeeping fields. Uploads that fail wait in
`$HERMES_HOME/synapse/outbox/` and go with the next successful one. A session marked
private (`synapse private on <session-id>`, in a terminal; `hermes sessions list` shows
the id) uploads nothing. Hermes gives a conversation a new session id when it branches
it, and when a compression starts a new session: the new id is private when a session it
holds turns of is, and gets the marker and the server row, so what was said off the
record stays there. Which session that is comes from the ids Hermes names when it moves
the agent, and from the session's row in Hermes's store, read before the first upload
under any id. A marker lasts twelve hours and `off` removes it, so when none says, the
server is asked (`GET /private-sessions/<earlier id>`) before the new session's first
upload, once per session; until it answers, nothing is uploaded for that session, and
the turns that finish are kept in memory.
`/new` and `/resume` open another conversation: nothing is carried to it from the one
you left, and it is on the record unless you mark it or its own row says it continues a
private session. `synapse private off --forget <id>` puts a session back on the record
with the turns spoken in it. For a session that was private because of the one it
continues, the agent running it uploads only the turns that end from then on; an agent
started for it later looks back ten turns, copied ones included.

**Not ported yet.** The remember spool (a failed `remember` is reported to the model, not
queued), the recall-feedback nudge, a Hermes slash command for private mode (use
`synapse private` in a terminal), skills and config sync, and importing past Hermes
sessions from its `state.db`.

## More

- [docs/tools.md](../docs/tools.md): hooks, MCP tools, configuration variables, and what
  gets sent to your server.
- [docs/auth.md](../docs/auth.md): tokens, enrollment, personal vs work.
- [docs/install.md](../docs/install.md): the server side, and troubleshooting.
