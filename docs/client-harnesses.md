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
    private.py                   is this session private (the marker file and its TTL)
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

`synapse login`, `synapse install`, `synapse update` and `synapse status` walk one list,
`harnesses()` in `plugin/scripts/harness.py`: Claude Code and Codex, as small objects over
the functions that already wired them, then the core's registry. A harness added to the
registry is picked up by all four without a change to the CLI.

## Lifecycle mapping

| Synapse behaviour | Claude Code | Codex | Hermes |
|---|---|---|---|
| Board + preferences at session start | `SessionStart` hooks | `SessionStart` hook | `system_prompt_block()`, fetched once in `initialize()` |
| Which sessions | every session | every session | local sessions only (below) |
| Per-prompt recall nudge + timestamp | `UserPromptSubmit` hook | `UserPromptSubmit` hook | `prefetch()` (returns both; no recall is run) |
| Upload a finished turn | `Stop` hook reads the session JSONL | `Stop` hook reads the rollout | `sync_turn(messages=...)` |
| Catch up after failed uploads | `SessionStart --catchup` sweep | session-start sweep | outbox, flushed in `initialize()` and after each upload |
| Memory tools | MCP entry in the plugin manifest | MCP entry in `config.toml` | `get_tool_schemas()` / `handle_tool_call()`, proxied to MCP `tools/call` |
| Caller's session id on tool calls | `PreToolUse` hook | `PreToolUse` hook | set by the provider before the proxy call |
| `/ingest` format | `claude_code` | `codex` | `messages` (client `hermes`) |
| Install | `claude plugin install` | hook blocks in `config.toml` | `synapse install hermes`: folder in `$HERMES_HOME/plugins/synapse/` + `memory.provider: synapse` |

## Hermes

Hermes loads a memory provider as a class (`agent/memory_provider.py`, `MemoryProvider`),
inside its own process or, with `plugins.isolation: host`, in its plugin host, where the
agent holds a proxy; the provider works the same in both. One external provider can be
active at a time; selecting Synapse takes that slot and leaves Hermes's built-in
MEMORY.md / USER.md alone.

**Local sessions only.** Synapse is one person's memory, and Hermes also runs as a chat
bot (Telegram, Discord, Slack and the rest, through its gateway) and an API server, where
other people talk to it. The provider decides in `initialize()`, from the keywords Hermes
passes and, for a session they allow, Hermes's own record of who created it
(`harnesses/hermes/sessions.py` has the rule):

- **Active**: `agent_context` "primary" (Hermes's default) on a local surface, `platform`
  "cli" (Hermes's default), "tui", "desktop" or "acp", run by this machine's owner. With
  no chat identity the agent runs in the process that created it (the stdio TUI and
  Desktop backend, ACP, the classic CLI). `hermes chat` and the TUI run their turns in
  the gateway daemon, which passes a chat identity even for them: such a session is
  active only with the shape of a local session (`chat_id` `local-<hex>`, `chat_type`
  "dm", its `agent:<profile>:local:dm:` route, `user_id` `uid:<this OS user>`) and a
  creation receipt in `<hermes_home>/state.db`, read read-only, naming the owner as its
  creator, an owner's chat as its source (not `--source tool`) and no Bot Chat. The
  board, the nudge, uploads and the tools all work.
- **Cron** (`agent_context` or `platform` "cron", no chat identity): the tools work;
  nothing is injected and nothing is uploaded. A scheduled job may need the memory, but
  its prompts are not a conversation the user had.
- **Inert**, everything else: subagents and other forks, the API server, every messaging
  gateway, hosted rooms and peer agents, the Bot Chat, a session created by someone else
  or by a third-party integration, one the store does not know, and any shape not listed
  above. Nothing is injected or uploaded, and a tool call returns an error: someone else
  must not read the owner's memory through the agent, nor have their messages stored as
  the owner's. The reason is logged once.

The session's row in `state.db` (title, stored source, chat) is read again before each
nudge, upload and tool call, since a session can be renamed "Bot Chat" or switched to
another (`/resume`, compression) without a new provider; one that no longer passes turns
the instance inert for good. A gateway session's project comes from the `cwd` its client
sent, never the daemon's.

The Hermes source this rests on: `_memory_provider_init_kwargs` in
`agent/agent_init_memory_provider.py` (platform defaults to "cli"; `agent_context` is the
agent's `memory_agent_context`, else "cron" or "subagent" by platform, else "primary";
the chat identity is `_GATEWAY_IDENTITY_PARAMS` in `agent/agent_init_params.py`),
`gateway/session_execution.py` (a gateway passes its platform and the chat identity),
`hermes_state_keys.py` and `hermes_state_local.py` (the local chat id and its creation
receipt), `gateway/control_socket.py` (the `uid:` subject, minted for a peer of the same
OS user), `gateway/platforms/api_server.py` ("api_server"), `tui_gateway/agent_factory.py`
("tui", "desktop"), `acp_adapter/session.py` ("acp"), `cron/scheduler.py` ("cron") and
`tools/delegate_tool.py` (`memory_agent_context="subagent"`).

**Install.**

```
synapse install hermes [--hermes-home PATH] [--replace] [--dry-run]
```

`synapse login` does the same when it finds Hermes (a `hermes` program on PATH, or a
Hermes home on disk), and `synapse update` rebuilds the folder when it was built by
another client version. `synapse status` (new in 0.24.0) prints a line per harness:
wired, found and not wired, or not found. The Hermes home is
`--hermes-home`, else `$HERMES_HOME`, else Hermes's default (`~/.hermes`). Selecting the
provider:

- With the `hermes` program on PATH and a `config.yaml` already in the home, through
  `hermes config get --json` / `hermes config set memory.provider synapse`. An empty
  value or one of Hermes's built-in names is replaced; another external provider is left
  alone and the report says how to switch, unless `--replace` is given.
- A home with no `config.yaml` gets one with the two lines `memory:` / `provider:
  synapse`. The program is never run there: a first launch sets up the home, which
  downloads Hermes's toolchain.
- A `config.yaml` without the program on PATH is left as it is, and the report names
  the `hermes config set` command to run, or says it is already selected.

**Installed layout.** The folder is self-contained:

```
$HERMES_HOME/plugins/synapse/
  plugin.yaml                    name, version, kind: exclusive
  __init__.py                    register(ctx) -> SynapseMemoryProvider()
  synapse_client/                copy of the core + harnesses/hermes
```

`__init__.py` is the only top-level module: Hermes executes every top-level `*.py` beside
it before the package exists. A new folder is built beside the old one under a dot-name,
which Hermes skips, and swapped in with two renames, so Hermes never loads half of one.
Settings and the credential are not copied: the provider reads `~/.synapse/config.json`
like every other harness, so there is no Hermes setup wizard step.

**What each provider method does.**

| Method | Behaviour |
|---|---|
| `is_available()` | A URL and a token resolve. No network. |
| `initialize(session_id, **kw)` | Decides the scope (above), binds the session, the project label for `cwd` and the outbox under `<hermes_home>/synapse/outbox`. In an active session: fetches the board and preferences side by side within 5 seconds, fail-open, then drains the outbox on a thread of its own. |
| `system_prompt_block()` | The text fetched in `initialize()`, in an active session. Hermes caches it for the session and asks again after a compression. |
| `prefetch(query)` | The timestamp line and the recall nudge, in an active session. No request is made. |
| `on_turn_start(n, message, model=...)` | Keeps the model name for the uploads. |
| `sync_turn(user, assistant, messages=...)` | Uploads every complete turn not yet uploaded, each batch led by the turn before it as context. A batch that fails goes to the outbox; the next success drains it. Active sessions only, and nothing for a private session. Hermes runs it on its background worker, whose stall gate skips a provider stuck over 30 seconds, so a request is given 20. |
| `on_session_end(messages)`, `on_pre_compress(messages)` | Upload any turn not yet uploaded (10-second requests). Hermes calls the first at every compression and `/new` as well as at exit, so both rely on the server's span-id dedup. `on_pre_compress` returns `""`. |
| `on_session_switch(new_id, ...)` | Rebinds the session id (`/new`, `/resume`, `/branch`, rewind). A reset or a new id starts the uploaded-turn memory afresh, looking back at most ten turns. |
| `get_tool_schemas()` | `recall`, `recall_full_turns`, `fetch`, `fetch_session`, `remember`, `recall_feedback`, from `tool_schemas.json`, in the server's order (`tools.schemas()`). Not shown to the model: `surface` anywhere, `self_session` on the recall family and `fetch_session`, `session_id` on `remember` and `recall_feedback`. The `session_id` of `recall_full_turns` and `fetch_session` names the conversation to read and stays. Static: Hermes asks before `initialize()`. |
| `handle_tool_call(name, args)` | `tools.call()`: adds the session id the way the `PreToolUse` hook does (never over a value already set), calls the server's MCP tool, returns its text: a JSON object as it is, other text as `{"result": ...}`, the shape Hermes's own providers return. Never raises: a failure, or a call in an inert session, comes back as a JSON object with an `error` field. |

**What is uploaded.** Hermes passes the whole conversation on every turn, as live message
dicts, after scrubbing secrets from them. The normaliser keeps only `role`, the text
content, tool-call names and arguments (capped at 1000 characters), tool results (capped
at 2000), `timestamp` and `message_uid` (as the record `id`). The model is not read from
the messages: it is the batch's `model`, from `on_turn_start`. Everything else is
dropped, in particular `api_content` (which holds the memory context Hermes injected,
our own nudge included), `display_metadata`, reasoning blobs and compaction summaries
(`_compressed_summary`). A user-role row Hermes wrote itself (a nudge, a task-list
snapshot, a `/steer` message, a control frame) is sent marked `synthetic`, which the
server skips. The dicts are never modified: Hermes hands out references to its own
transcript. A turn is uploaded once it is complete: it ends in an assistant reply with
no tool calls. Hermes's notice for a failed turn (an assistant row typed `failed_turn`,
or one holding only the bare notice) is not a reply, so a failed turn is never uploaded,
its user message included; an assistant row of any other `display_kind` (a held turn's
interim text, the interrupt placeholder) keeps only its tool calls.

**Not there yet.** These work in Claude Code and are not ported to Hermes: the remember
spool (a `remember` call that fails is reported to the model, not queued), the
recall-feedback nudge, a private-mode toggle (markers are honoured; there is no Hermes
command that writes one), skills sync into `$HERMES_HOME/skills`, config sync, a history
importer for `state.db`, and Synapse in messaging-gateway and API-server sessions
(above). Skill invocations are uploaded as Hermes hands them over: the manager strips
the skill body from the user text it passes, not from `messages`.

**Tests.** `tests/test_hermes_messages.py`, `test_hermes_provider.py` (against a stub of
Hermes's base class) and `test_hermes_install.py` run everywhere, with
`test_hermes_gateway_sessions.py` (the rule, over a `state.db` built like Hermes's),
`test_hermes_failed_turns.py`, `test_hermes_inflight_upload.py` and
`test_hermes_install_selection.py`. `tests/test_hermes_integration.py` runs only with
`SYNAPSE_HERMES_CHECKOUT` set to a Hermes checkout with its `.venv`: it installs into a
temp home, loads the provider with Hermes's own loader and drives Hermes's
`MemoryManager`, in-process and with `plugins.isolation: host`, against a local fake
server. `tests/test_hermes_checkout_names.py`, under the same variable, checks the
failed-turn notices and local-session names the client copies against the checkout.

## Where the migration stands

The core and its first adapter exist: every module in the layout above, the `Harness`
protocol, and a registry holding Hermes (plugin 0.24.0). The CLI walks one list for
login, install, update and status, with Claude Code and Codex wrapped as list entries
over their old functions.

Codex's hooks (`plugin-codex/`) run on the core since plugin 0.24.4: `common.py` resolves
with `Settings.load` and sends through `Transport`, and session start, the prompt nudge,
the recall-feedback nudge and the Stop hook's private check come from the core. Codex
keeps its own Stop-hook shipper (rollout cursors, catch-up sweep and retry), the
session-id injection hook (`pre_tool_use.py`), the private-mode toggle
(`scripts/private_mode.py`) and the skills-sync glue. Claude Code's hooks
(`plugin/scripts/`) still carry their own configuration and HTTP code (`config.py`);
moving them onto the core is the next change, and the two harnesses' wiring then moves
into `harnesses/` beside Hermes's.

A module moves like this: the code goes into `synapse_client/`, the old path keeps a
few-line stub that imports it (hook commands and Codex's trusted command text never name
a new path), and the tests that loaded the old file by path are pointed at the package.
`project.py` is the first module moved this way.

`tool_schemas.json` is a snapshot of the server's tools/list, written by
`scripts/gen_client_tool_schemas.py` (no database needed); a test regenerates it and fails
when the server's tools change without it.

### Where the core differs from the hooks

The core ports the hooks' behaviour and wording, with these differences:

- **MCP without a handshake.** The server runs stateless streamable HTTP, so `tools/call`
  goes out alone, with `MCP-Protocol-Version: 2025-06-18`; no `initialize`. The reply is
  read as JSON or as server-sent events, up to the answer to this call.
- **The bearer stays on its origin.** `http.Transport` sends it only to the configured
  server (and the ingest URL's origin), and Codex's hooks send through it; Claude Code's
  hook helpers send it to any absolute URL they are given. A redirect carries it only
  within its origin, or from http to https on the same host and the default ports (80 to
  443, the rule the `requests` library uses). Claude Code's hooks open their requests
  through the same handler (`http.urlopen`, reached through `config._urlopen`): urllib's
  stock one forwards the header to wherever a redirect points.
- **Legacy credentials are passed in, not read.** `Settings.load` takes the old
  `device.json` record and `credentials.json` token as arguments (`legacy_device`,
  `legacy_token`); the harness that has those files reads them.
- **Session-start order** is notice, upload warning, board, preferences in every harness.
- **Enrollment notices** take the login hint as an argument; the restricted-device
  notice has no Claude-only "paste it into /plugin" aside.
- **Failed uploads** go to `Outbox`, one file per payload, claimed by rename so several
  processes can drain one directory. Up to 2000 wait; past that the oldest are dropped. A
  payload the server refuses with 400, 413 or 422 is dropped; any other failure, 404
  included, stops the drain and waits. Payloads dropped for the cap or as refused are
  counted in `.dropped` beside the queue, and the count is cleared, with the recorded
  reason, when a drain leaves the queue empty: it covers the current backlog. The
  session-start warning is the retry module's sentence, dated from the oldest payload,
  followed by "N uploads were dropped and will not be retried." while the count is not
  zero; the count alone does not raise the warning. A harness whose process may exit
  mid-request (Hermes) queues each payload before its own send, `put(..., sending=s)`,
  and removes it once delivered or `keep`s it as waiting when the send fails. No drain
  takes such a payload, nor does the cap count it, until `s` seconds have passed; after
  that its sender is taken for dead and a drain sends it, at most once more than needed,
  which the server's span-id dedup absorbs.

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
