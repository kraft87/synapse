# The client: one core, one adapter per harness

Synapse runs inside other people's agents ("harnesses"): Claude Code, Codex, Hermes. Each
has its own way of running extension code, but what Synapse does in all of them is the
same: put the board in front of the model at session start, nudge it to recall, upload
each finished turn, and give it the memory tools. This page describes how the client code
is split so that the shared part exists once and a new harness is a small adapter.

## Layout

```
plugin/                          what every install method ships
  synapse_client/                THE CORE: a real Python package, stdlib only
    settings.py                  ~/.synapse/config.json + env -> Settings (no globals)
    credentials.py               which token is sent where (pure functions)
    http.py                      bearer, User-Agent, timeouts; JSON and MCP tools/call
    api.py                       Client: one method per server endpoint
    context.py                   session-start block, per-prompt nudge (the wording)
    project.py                   project label for a working directory
    outbox.py                    durable queue for uploads that failed
    tools.py + tool_schemas.json the memory tools, as static schemas + a tools/call proxy
    harnesses/
      __init__.py                the Harness protocol and the registry
      hermes/                    provider, message normaliser, installer, plugin template
  scripts/                       Claude Code hook scripts and the `synapse` CLI
  hooks/, commands/, bin/        Claude Code plugin surface
plugin-codex/                    Codex hook scripts and installer
```

`plugin/` keeps its name because it is the directory the Claude marketplace, the managed
client (`~/.synapse/lib/<version>/`) and the server's `/install` bundle all copy.

## Rules for the core

1. **Standard library only**, Python 3.11+. It runs inside other programs' interpreters.
2. **Relative imports only.** The package is copied into a Hermes plugin directory and
   imported there under a name Hermes chooses, so `import synapse_client` from inside the
   package would not resolve.
3. **Nothing is resolved at import.** URL, token and flags live on a `Settings` value that
   a caller builds and passes down. A long-running host (Hermes) may hold several
   instances; module-level state would mix them.
4. **The core never imports a harness, and a harness imports only the core.** The Hermes
   installer copies the core plus `harnesses/hermes/` and nothing else.
5. **Wording lives in the core.** The nudge, the preferences block and the enrollment
   notices are the same text in every harness; an adapter only chooses the envelope.

## What an adapter provides

| Piece | Question it answers |
|---|---|
| Wiring | Is the harness on this machine, how is Synapse installed into it, how is it updated, is it wired already? (`Harness.detect / install / update / wired`) |
| Lifecycle mapping | Which harness event calls which core function, and in what envelope does the result go back? |
| Transcript source | Where do finished turns come from, and which `/ingest` format carries them? |
| Skills folder | Where does this harness read skills from? |

`synapse login`, `synapse install`, `synapse update` and `synapse status` iterate the
registry; none of them names a harness.

## Lifecycle mapping

| Synapse behaviour | Claude Code | Codex | Hermes |
|---|---|---|---|
| Board + preferences at session start | `SessionStart` hooks | `SessionStart` hook | `system_prompt_block()`, fetched once in `initialize()` |
| Per-prompt recall nudge + timestamp | `UserPromptSubmit` hook | `UserPromptSubmit` hook | `prefetch()` (returns the nudge; no recall is run) |
| Upload a finished turn | `Stop` hook reads the session JSONL | `Stop` hook reads the rollout | `sync_turn(messages=...)` |
| Catch up after failed uploads | `SessionStart --catchup` sweep | session-start sweep | outbox, flushed in `initialize()` and after each upload |
| Memory tools | MCP entry in the plugin manifest | MCP entry in `config.toml` | `get_tool_schemas()` / `handle_tool_call()`, proxied to MCP `tools/call` |
| Caller's session id on tool calls | `PreToolUse` hook | `PreToolUse` hook | set by the provider before the proxy call |
| `/ingest` format | `claude_code` | `codex` | `messages` (client `hermes`) |
| Install | `claude plugin install` | hook blocks in `config.toml` | folder in `$HERMES_HOME/plugins/synapse/` + `memory.provider: synapse` |

## Hermes

Hermes loads a memory provider as a class inside its own process (`agent/memory_provider.py`,
`MemoryProvider`). One external provider can be active at a time; selecting Synapse takes
that slot and leaves Hermes's built-in MEMORY.md / USER.md alone.

**Installed layout.** `synapse install hermes` (and `synapse login`, when it finds Hermes)
writes a self-contained folder and selects it:

```
$HERMES_HOME/plugins/synapse/
  plugin.yaml
  __init__.py                    register(ctx) -> SynapseMemoryProvider()
  synapse_client/                copy of the core + harnesses/hermes
```

`__init__.py` is the only top-level module: Hermes executes every top-level `*.py` beside
it before the package exists. The folder is replaced as a whole on `synapse update`.
Settings and the credential are not copied: the provider reads `~/.synapse/config.json`
like every other harness, so there is no Hermes setup wizard step.

**What each provider method does.**

| Method | Behaviour |
|---|---|
| `is_available()` | A URL and a token resolve. No network. |
| `initialize(session_id, **kw)` | Binds the session, `hermes_home`, the project label for `cwd`, and `agent_context`. For a primary session: fetches the board and preferences (bounded, fail-open) and flushes the outbox. |
| `system_prompt_block()` | The text fetched in `initialize()`. Hermes caches it for the session and asks again after a compression. |
| `prefetch(query)` | The timestamp line and the recall nudge. No request is made. |
| `sync_turn(user, assistant, messages=...)` | Uploads the turn that just finished. A failed upload goes to the outbox. Skipped for private sessions and for `cron` / `subagent` contexts. |
| `on_session_end(messages)`, `on_pre_compress(messages)` | Upload any turn not yet uploaded. Hermes calls the first at every compression as well as at exit, so both rely on the server's span-id dedup. `on_pre_compress` returns `""`. |
| `on_session_switch(new_id, ...)` | Rebinds the session id (`/new`, `/resume`, `/branch`, rewind). |
| `get_tool_schemas()` | `recall`, `recall_full_turns`, `fetch`, `fetch_session`, `remember`, `recall_feedback`, from `tool_schemas.json`, in the server's order. Parameters a hook fills in elsewhere (`self_session`, `surface`, `session_id`) are not shown to the model. Static: Hermes asks before `initialize()`. |
| `handle_tool_call(name, args)` | Adds the session id the way the `PreToolUse` hook does, calls the server's MCP tool, returns its JSON. |

**What is uploaded.** Hermes passes the whole conversation on every turn, as live message
dicts. The normaliser keeps only `role`, the text content, tool-call names and arguments,
tool results, `timestamp`, `message_uid` (as the record `id`) and the model. It drops
everything else, in particular `api_content` (which holds the memory context Hermes
injected, our own nudge included), `display_metadata`, reasoning blobs and compaction
summaries (`_compressed_summary`). The dicts are never modified: Hermes hands out
references to its own transcript.

**Not there yet.** These work in Claude Code and are not ported to Hermes: the remember
spool (a `remember` call that fails is reported to the model, not queued), the
recall-feedback nudge, a private-mode toggle (markers are honoured; there is no Hermes
command that writes one), skills sync into `$HERMES_HOME/skills`, and a history importer
for `state.db`.

## Where the migration stands

The core is new. Claude Code's hooks (`plugin/scripts/`) and Codex's (`plugin-codex/`)
still carry their own configuration and HTTP code (`config.py`, `common.py`); moving them
onto the core is the next two changes, Codex first.

A module moves like this: the code goes into `synapse_client/`, the old path keeps a
few-line stub that imports it (hook commands and Codex's trusted command text never name
a new path), and the tests that loaded the old file by path are pointed at the package.
`project.py` is the first module moved this way.

## Adding a harness

1. Decide the transcript source. If the harness can hand over chat messages, use the
   `messages` ingest format with a new `client` name: no server change. Only a harness
   whose transcript exists solely as a file in its own format needs a server parser.
2. Add `synapse_client/harnesses/<name>/` with the wiring (`detect / install / update /
   wired`) and register it. `synapse login` picks it up.
3. Map the harness's events onto the core: session start -> `context.session_block`,
   prompt -> `context.prompt_nudge`, turn end -> `Client.ingest` (+ outbox), tool calls ->
   `tools.call`.
4. Add a column to the table above and say what is not ported.
