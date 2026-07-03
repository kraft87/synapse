# Synapse — Claude Code plugin

Connects Claude Code to a [Synapse](../) memory instance. Single-user; works against a
**local** Synapse (Docker on your machine) or a **central** one (hosted, shared by all your
machines — laptop, work box, server).

The plugin is a **thin client**: stdlib-only scripts that talk to the server over HTTP
(one base URL + an optional bearer token — never a database connection). All the heavy
lifting (memory extraction, recall, the dream→skills lane) runs server-side in the Synapse
stack.

## What it does

Five hooks (see `hooks/hooks.json`), plus the MCP wiring:

1. **Transcript ingest** (`Stop` hook) — after every turn, pushes a bounded tail of the
   session transcript to your Synapse `/ingest`. This is the point of the plugin — it's how
   your sessions become memory. Works over the network, so a work laptop can feed a central
   Synapse. The hook only captures sessions from install onward; run the bundled
   `synapse-import` once to backfill everything that came before (see
   [Import your existing history](#4-import-your-existing-history-optional-recommended)).
2. **Skill sync** (`SessionStart` hook) — two-way sync of your skills between the server and
   `~/.claude/skills` (plus the current project's `.claude/skills`). Newest edit wins per
   skill, with an append-only server-side history; deletes never auto-propagate. The
   server-side dream→skills lane can improve a skill and have it flow back to every machine.
3. **Config mirroring** (`SessionStart` hook, **opt-in — off by default**) — mirrors your
   `CLAUDE.md` + `rules/*.md` to the server so the dream lane can propose config edits
   (reviewed with `/synapse:config-review`).
4. **Timeline git feeder** (`SessionStart` hook, **opt-in — off until configured**) — pushes
   commit subjects from repos you list in `SYNAPSE_TIMELINE_REPOS` to the server's timeline.
5. **Timeline milestones** (`SessionStart` hook) — fetches the last 7 days' high-salience
   timeline events from the server and prints a bounded block (max 5 lines) into the
   session's context.

Plus **MCP tools** — the plugin registers the Synapse MCP server, so `recall` / `remember` /
`query_graph` are wired up automatically (no hand-written `.mcp.json`).

The **dream→skills lane** (mines your transcripts → proposes new skills, retunes triggers,
nominates merges) runs **server-side** in the Synapse `dream` container — you don't run a cron.
You just review what it proposes with `/synapse:skill-review`.

> Recall-injection (a `UserPromptSubmit` hook that pushed memory into every prompt) was removed:
> it injected top-k unconditionally with no relevance gate, which added noise and anchored the
> model on stale priors. Use the `recall`/`remember` MCP tools for pull-on-demand recall instead.

## What gets sent to your server (and how to turn each off)

Everything below goes only to the Synapse URL **you** configure. All hooks are fail-soft: an
unreachable server is a silent no-op and never breaks or spams your session.

- **Transcript ingest** — **on** (the plugin's core function). Sends the raw JSONL tail of
  each session transcript — your prompts, Claude's replies, tool calls — to `/ingest`. There
  is no separate toggle; if you don't want transcripts leaving the machine, don't install the
  plugin (or disable it with `/plugin`).
- **Skill sync** — **on by default**. Sends skill bodies + bundled files from your skills
  dirs to `/skills/*`, and pulls server-side versions back. Turn off with
  `SYNAPSE_SKILLS_SYNC=0`.
- **Config mirroring** — **off by default** (opt-in). When enabled, sends your global
  `~/.claude/CLAUDE.md` + `rules/*.md` and the current project's equivalents to
  `/config/publish`. These files often carry personal instructions, which is why it ships
  off. Enable with `SYNAPSE_CONFIG_SYNC=1`; add extra globs with `SYNAPSE_CONFIG_PATHS`.
- **Timeline git feeder** — **off by default** (opt-in). When you set
  `SYNAPSE_TIMELINE_REPOS` (comma/space-separated checkout paths), sends commit subjects,
  dates, and a coarse salience score from those repos to `/timeline/events`. Unset = nothing
  runs.
- **Timeline milestones** — **on by default**, but it *sends* nothing: it reads
  `/timeline/recent` and prints up to 5 lines into your context. Turn off with
  `SYNAPSE_TIMELINE_MILESTONES=0`.

Each toggle can be set as an environment variable or a plugin option in `settings.json`
(`pluginConfigs."synapse@<marketplace>".options`).

## Setup

### 1. Stand up Synapse (the server)

```bash
cd <repo> && docker compose up -d
```
This provisions Postgres, runs migrations, and starts the poller, MCP server, and the
`dream` container (which runs the dream→skills lane on its daily schedule). Central deployments
expose the one server URL behind auth and point every machine's plugin at it.

### 2. Install the plugin (the client) on each Claude Code machine

This repo *is* a marketplace (it ships `.claude-plugin/marketplace.json`), so there's nothing
to publish or get approved:

```
/plugin marketplace add kraft87/synapse
/plugin install synapse@synapse
```

On install, Claude Code prompts for the `userConfig` and stores secrets in the OS keychain — no
hand-editing `settings.json` or `.mcp.json`:

- `SYNAPSE_URL` — the **base** URL of your Synapse server, e.g. `https://synapse.example.net`
  or `http://localhost:8765` (no path). The plugin derives `/ingest`, `/recall`, `/skills`, and `/mcp`.
- `SYNAPSE_INGEST_TOKEN` — bearer for an auth-gated server. This one token covers ingest,
  recall, skill sync, and the MCP tools — the client never needs a database DSN.

Then `/reload-plugins` (or restart) to activate the hooks + MCP server in the current session.

**No environment variables required.** `scripts/config.py` resolves settings in this order:
explicit env var → `CLAUDE_PLUGIN_OPTION_*` (injected for hooks/MCP) → the install-prompt value
in `settings.json` (`pluginConfigs."synapse@<marketplace>".options`) → built-in default. A new
user just answers the `/plugin install` prompt; every hook and command reads straight from it.
Env vars stay available as optional overrides (e.g. CI). The legacy `SYNAPSE_INGEST_URL` is
still honored.

### 3. Auth (only for auth-gated servers)

The default local quickstart (`docker compose up` + `http://localhost:8765`) needs **no auth
at all** — leave the token blank and skip this section.

For an auth-gated (central/hosted) server, either paste a machine token into the install
prompt, or run the bundled login once:

```
! synapse-login        # in the Claude prompt — runs directly in-session, no LLM, streams live
```

By default this is the **device flow** (RFC 8628): it prints a short code, you approve at
`github.com/login/device` on **any device** (phone, another laptop), and it polls until done —
no same-host browser, no loopback, so it works on servers and headless boxes. It needs a server
new enough to expose `/device/code` with the GitHub OAuth App's "Enable Device Flow" turned on.
Older servers (or if you prefer): `synapse-login --browser` runs the legacy loopback
authorization-code flow that opens a browser on the same machine. Either way the token is
stashed for the hooks and MCP server.

The `synapse-login` command (+ `synapse-login.cmd` for Windows) ships in the plugin's `bin/`,
which Claude Code adds to PATH **inside its own session** — so `! synapse-login` runs the
script directly. That `bin/` is *not* on a separate OS terminal's PATH; to run it from one,
call the script by full path: `python "<plugin>/scripts/synapse_login.py"` (the path prints in
the plugin cache, e.g. `~/.claude/plugins/cache/synapse/synapse/<version>/scripts/`).

Truly headless with no second device? Set `SYNAPSE_INGEST_TOKEN` directly (env / plugin
config) instead of running login.

### 4. Import your existing history (optional, recommended)

The `Stop` hook only sees sessions from install onward, but your machine likely already
holds months of transcripts under `~/.claude/projects`. Import them once and recall works
on day one:

```
! synapse-import        # in the Claude prompt — or run it from any terminal
```

What it does:

- Discovers every session transcript under `~/.claude/projects` (`--projects-dir` to
  override), oldest-first.
- Prints a summary — file count, total size, estimated turns — and **asks for confirmation
  before sending anything**: importing runs KG extraction on the server's configured LLM
  for every new turn, which consumes subscription usage or API credits (`--yes` skips the
  prompt for scripted runs).
- Ships each file full-length (not the hook's bounded tail) in turn-aligned batches
  (`--batch-size`, default 500 records per POST), with a per-file progress line.
- Is safe to interrupt and re-run: the server dedups turns by `span_id`, so a re-run skips
  everything already stored and resumes where it left off. Failures are per-file — one bad
  file never stops the rest, and the exit code is non-zero only if *every* file failed.

It resolves the endpoint and token exactly like the ingest hook (env var → plugin
userConfig → `settings.json` install answers → localhost default), and ships as
`synapse-import` / `synapse-import.cmd` in the plugin's `bin/`, which is on PATH inside
Claude Code sessions — from an outside terminal, call the script by full path:
`python "<plugin>/scripts/import_history.py"`.

Cursor history is importable too, but only as a server-side dev path for now
(`python -m ingestion.cursor_sqlite_backfill` on the Synapse host).

## Using it

- **Review skill proposals:** `/synapse:skill-review` (or `python3 scripts/skill_review.py list`).
- **Accept / reject / promote:** `… accept <id>` → edit/place the draft into your skills
  dir → `… promote <id>`. Nothing touches your live skills without a grounded accept.
- **Review config proposals** (if you opted into config mirroring): `/synapse:config-review`.

## How it talks to Synapse

HTTP only, machine-token gated: `/ingest`, `/skills/*`, `/config/publish`, `/timeline/*`, and
`/mcp`, all under the one `SYNAPSE_URL`. The client holds no Postgres credentials — the
database stays behind the server, and skill/config proposals are only ever applied through
your explicit review commands.
