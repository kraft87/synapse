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
    private_toggle.py            the private-mode toggle: marker + server row, on/off/status
    project.py                   project label for a working directory
    outbox.py                    durable queue for uploads that failed
    messages.py                  a conversation in the `messages` format, cut into turns
    upload.py                    the one way out for a turn: private check, send, queue
    turn_upload.py               a conversation's finished turns, uploaded once each
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

## Versions and releases

The client version is written twice and the two must agree: `version` in
`plugin/.claude-plugin/plugin.json` and `__version__` in `plugin/synapse_client/__init__.py`.
Machines update by that number, so any change to a file a machine receives ships under a
new version: every tracked file under `plugin/` and `plugin-codex/`, Markdown included
(`plugin/commands/*.md` are slash commands, and the READMEs are in the bundle). A
server-only change leaves the version alone. The rules live in `scripts/release_notes.py`.

- **The Version check** on a pull request: when the PR changes a client file, the merged
  version must be above the base's and have no `v<version>` tag yet. A PR that changes no
  client file passes. It is not a required check yet, and it does not re-run when main
  moves: if two open PRs bump to the same number, update the second after the first lands.
- **Releases are automatic.** When the client changes on main, `release.yml` tags
  `v<version>` at the commit that brought it and publishes a GitHub release whose notes
  are every HANDOFF.md entry whose heading names `plugin <version>`, as in
  `## 2026-11-12 — client: what changed (plugin 0.24.4)` (other items may share the
  parentheses). A client change merged under an already released version fails that run:
  bump the version in a follow-up PR.
- **A missed release** (a failed run, or a version from before the workflow): run the
  Release workflow on `main` by hand (Actions > Release > Run workflow, or
  `gh workflow run release.yml --ref main`). It releases the version main carries now and
  exits green when that one is already out. A version main no longer carries is released
  by hand: `gh release create vX.Y.Z --target <commit> --title vX.Y.Z --notes-file
  <(python3 scripts/release_notes.py notes X.Y.Z)`.

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
6. **A turn leaves through one door, and the private rule stands at it.** `Client.ingest`
   is the only place the core posts to the ingest route. It refuses a body that names a
   session in private mode before any request (`private.check_payload`: the body's own
   `session_id` and each record's `sessionId`). An adapter hands its records to
   `turn_upload.TurnUploader`, or a ready body to `upload.Uploader`, and never posts to the
   ingest route itself, so no adapter can forget the check. The two older shippers that
   still post on their own (Claude Code's and Codex's Stop hooks) run the same check
   inside their senders.
7. **One private-mode toggle.** `private_toggle.Toggle` is the toggle of every harness:
   the marker (`private.mark`), the server row (`Client.private_put`), the read-back,
   and what `off` and `--forget` leave behind. `synapse private`, and the
   `private_mode.py` of the Claude Code and Codex plugins, only build the client and
   hand it the arguments. A harness that gives a continuing conversation a new session
   id calls `Uploader.carry_private`, or the turns spoken off the record go up under
   the new id.

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

A process runs one client version: the list is built from the core beside the CLI that
is running. `synapse update` is the one command that changes the client under itself, so
it is two processes. The first fetches `<server>/install` and runs it. The second is the
CLI of the client that installer made current, started through the launcher as `synapse
update --rewire`: it refreshes every wired harness with its own code, and an adapter that
copies the core (Hermes) copies the new one. Its exit code is the update's. A client
whose CLI has no such step (0.24.8 and older) is asked for a whole `synapse update`
instead, which every managed client has. That is the usual case for a CLI newer than
its server, such as the Claude plugin's `synapse` against a server that is behind. The
first process tells the two apart by reading the installed CLI for the flag, not by its
version number, and a process started by a hand-over never hands over again, so the
update cannot become a chain. Every later client keeps `update --rewire`: an older CLI
still on some machine asks for it.

## Lifecycle mapping

| Synapse behaviour | Claude Code | Codex | Hermes |
|---|---|---|---|
| Board + preferences at session start (one request) | `SessionStart` hook | `SessionStart` hook | `system_prompt_block()`, fetched once in `initialize()` |
| Which sessions | every session | every session | local sessions only (below) |
| Per-prompt recall nudge + timestamp | `UserPromptSubmit` hook | `UserPromptSubmit` hook | `prefetch()` (returns both; no recall is run) |
| Upload a finished turn | `Stop` hook reads the session JSONL | `Stop` hook reads the rollout | `sync_turn(messages=...)` |
| Catch up after failed uploads | `SessionStart --catchup` sweep | session-start sweep | outbox, flushed in `initialize()` and after each upload |
| Memory tools | MCP entry in the plugin manifest | MCP entry in `config.toml` | `get_tool_schemas()` / `handle_tool_call()`, proxied to MCP `tools/call` |
| Caller's session id on tool calls | `PreToolUse` hook | `PreToolUse` hook | set by the provider before the proxy call |
| Last compaction time on recall (`self_since`) | `PreToolUse` hook, from the transcript's last `compact_boundary` (the first turn it kept, if it names one) | `PreToolUse` hook, from the rollout's last `compacted` line | the provider's clock at an in-place compression |
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
  no chat identity the agent runs in the process that built it (the stdio TUI and
  Desktop backend, ACP, the classic CLI), unless that process is the gateway daemon
  (its `gateway.lock` or `gateway.pid` record names it) or was started with a
  `--source` that is not the owner's own (`HERMES_SESSION_SOURCE` "tool", "kanban", ...).
  `hermes chat` and the TUI run their turns in the gateway daemon, which passes a chat
  identity even for them: such a session is active only with the shape of a local
  session (`chat_id` `local-<hex>`, `chat_type` "dm", its `agent:<profile>:local:dm:`
  route, `user_id` `uid:<this OS user>`), its row in `<hermes_home>/state.db`, and a
  creation receipt there, read read-only, naming the owner as its creator, an owner's
  chat as its source (not `--source tool`; `-q`/`-z` one-shot runs count as the owner's),
  no Bot Chat, and no `follow_profile_config` in its request. The board, the nudge,
  uploads and the tools all work.
- **Inert**, everything else: subagents and other forks, cron jobs (no tools either:
  whoever can have the bot schedule a job chooses its prompt and where its output goes),
  the API server, every messaging gateway, hosted rooms and peer agents, the Bot Chat
  (by its title, its receipt, or the title of any row of its chat, so a `/new` in it
  stays inert), a Desktop Group Chat member (a "Group:" title, or the
  `follow_profile_config` the Desktop sends for it), a `/background` or `/btw` side agent
  of the daemon, a session created by someone else or by a third-party integration, one
  the store does not know, and any shape not listed above. Nothing is injected or
  uploaded, and a tool call returns an error: someone else must not read the owner's
  memory through the agent, nor have their messages stored as the owner's. The reason is
  logged once.

The session's rows in `state.db` are read again before the system prompt block, each
nudge, upload and tool call: its title, stored source and chat, every row of that chat
and of the chat its route answers on, and the sessions it was branched or continued from
(`parent_session_id`, or the origin its `model_config` names once Hermes has cleared
that link: a parent row, any row of a parent's chat, or a parent chat's receipt that
says Bot Chat, Group Chat member or `follow_profile_config`). A session can
be renamed "Bot Chat" or switched to another (`/resume`, compression) without a new
provider; one that no longer passes turns the instance inert for good, and so does a
gateway session whose row is gone. A turn another agent or person wrote is not uploaded
and turns the instance inert for good. Hermes names a turn's author on `on_turn_start`
and in `sync_turn`'s `turn_author`: the daemon names one on every turn, the owner's own
as the session's `user_id` (`uid:<this OS user>`, not a bot); in-process, a human turn
has none. An author that is a bot, or not the session's user, is someone else. The
owner's id means only that nobody stamped the turn. A gateway session's project comes
from the `cwd` its client sent, never the daemon's.

**Marks.** A session judged a Bot Chat, a Group Chat member or a chat that follows its
profile, or handed a turn someone else wrote, is marked together with the chat that said
so and the parent sessions down to it: one owner-only file each under
`<hermes_home>/synapse/inert/`, named by the SHA-256 of the id and holding the id and
the reason. A marked session and
every session of a marked chat stay inert after the title is removed (archive, rename)
and after Hermes builds a new agent for the session. So does a session branched or
continued from a marked one. Its parent's mark is copied down to it, its chat and the
parents between the first time it is judged, because Hermes clears a child's
`parent_session_id` when the parent is deleted or pruned; before that first judgment,
the `_branched_from`, `_reset_from` or `_compressed_from` the gateway wrote into its
`model_config` still names the parent. Marks are only
added, and a mark on a branch never marks its parent. A session comes back on only when
every file that names it, its chat and the sessions it descends from is deleted (each
file holds the id it is for), and the rows no longer say what was marked. Marks that
cannot be read (a folder that cannot be searched, a file in its place, a link that leads
nowhere) turn the session inert at `initialize` and skip the call afterwards; only
"nothing there" reads as not marked. A mark that cannot be written is logged.

What this does not cover:

- The board already in a system prompt stays there. Hermes stores the prompt and
  restores it verbatim for a continuing session, so a session renamed "Bot Chat",
  branched or handed off after its first turn keeps the board it was built with; only
  later rebuilds go without it. Two moves take such a prompt, with the transcript, to a
  messaging chat: the owner's own `/handoff`, and on Matrix `/resume <id or title>
  --cross-room` from anyone allowed to message the bot (other platforms need a
  configured admin). The provider cannot stop either.
- Hermes builds the first turn's prompt before it names that turn's author. A
  bot-to-bot delivery run as a one-shot through the daemon would get the board on that
  one turn; no Hermes path starts such a run today. A process whose own environment
  carries `HERMES_TURN_AUTHOR` is inert from `initialize`, a backstop nothing reaches.
- Hermes treats whoever can sign in to its dashboard or Webapp as the machine's owner
  (`gateway/owner_identity.py`): their sessions carry the owner's `uid:` subject, so they
  are the owner's here too. That covers a remote Desktop, a phone and a standalone
  dashboard.
- The welcome-trial mark (`follow_adopted`) is not read: Hermes puts it on an ordinary
  chat created on the trial model as well as on Bot Chats, which are caught by title.
- A chat that stopped looking like a Bot Chat before the provider first judged it has
  no mark: an older Bot Chat archived or renamed before then reads as an owner's chat.
- A copy that names no origin reads as the owner's chat: ACP `fork_session`
  (`acp_adapter/session.py`) and a seeded `session.create` without a parent
  (`tui_gateway/methods_session.py`) copy a session's turns into a new row with no
  `parent_session_id` and no `_branched_from`. Only the owner's own client can make one.
  Nor is a branch found that was never judged before its parent was deleted, unless the
  session it was branched from carries a mark itself; nor the child of an in-process
  compression whose parent row alone is deleted, which names its parent by the link only.
- A side agent built in the stdio TUI or Desktop backend's own process (not the daemon)
  carries no parent and no row, so it is active even when the session that started it
  is not (a Bot Chat open there, say). In that same process a model switch on a draft
  builds the prompt before the session has a row.
- Automation turns in an owner's chat (wake, goals, notifications) run with the nudge
  and the tools: Hermes gives them the owner as author.
- Wrongly inert: a session from before Hermes's creation receipts resumed through the
  daemon, the owner's own `/background` and `/btw` there, every daemon session on
  Windows, and an agent built while `state.db` could not be read. For good: a chat the
  owner titled exactly "Bot Chat", "Bot Chat #N", "Bot Chat (...)" or "Group: ...", even
  briefly, once it is judged under that title, with every branch of it and every later
  segment (`/new`) of such a branch; and a branch cut before its parent was handed a
  turn someone else wrote. And every session of a home whose marks folder cannot be
  read.
- An automation turn inside an owner's chat (a cron delivery or notice, which Hermes
  marks with a `display_kind`) is skipped as synthetic, so the assistant's reply to it
  is uploaded with the previous real turn.

The Hermes source this rests on: `_memory_provider_init_kwargs` in
`agent/agent_init_memory_provider.py` (platform defaults to "cli"; `agent_context` is the
agent's `memory_agent_context`, else "cron" or "subagent" by platform, else "primary";
the chat identity is `_GATEWAY_IDENTITY_PARAMS` in `agent/agent_init_params.py`),
`gateway/session_execution.py` (a gateway passes its platform and the chat identity),
`hermes_state_keys.py` and `hermes_state_local.py` (the local chat id and its creation
receipt), `gateway/control_socket.py` and `gateway/owner_identity.py` (the `uid:` subject,
minted for a peer of the same OS user and for every dashboard login),
`gateway/session_policy.py` (`follow_profile_config`), `gateway/session_bot.py` and
`hermes_state_local_lineage.py` (the Bot Chat and its reset),
`hermes_state_mutation_branch.py`, `hermes_state_mutation_reset.py` and
`hermes_state_mutation_compress.py` (a branch or continuation, its `parent_session_id`
and the origin in its `model_config`), `hermes_state_sessions.py` (a deleted session's
children lose the link), the Desktop's
`plugins/hermes-bots/group-turns.ts` and `canonical-chat.ts` (Group Chat members and the
Bot Chat it mints), `gateway/session_tui_side.py` and `gateway/status.py` (side agents and
the daemon's pid record), `run_agent.py` (`turn_author`), `gateway/platforms/api_server.py`
("api_server"), `tui_gateway/agent_factory.py` ("tui", "desktop"),
`acp_adapter/session.py` ("acp"), `cron/scheduler.py` ("cron") and
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
| `initialize(session_id, **kw)` | Decides the scope (above), binds the session, the project label for `cwd` and the outbox under `<hermes_home>/synapse/outbox`. In an active session: fetches the board and preferences (one request) within 5 seconds, fail-open, then drains the outbox on a thread of its own. A session whose row in the store says it holds another's turns (a branch, a compressed continuation: `lineage.chain`) is made private when one of those sessions is, as at a switch: Hermes's TUI, desktop app and gateway build a new agent for a branch and announce no switch. With a marker to go by, the server is told here, beside that request and inside the same 5 seconds; with none, it is asked at the first upload. Once per session (`carried.py`). |
| `system_prompt_block()` | The text fetched in `initialize()`, in an active session that the store still says is the owner's. Hermes caches it for the session, stores it with the session and restores it for a continuing one, and asks again after a compression. |
| `prefetch(query, session_id=...)` | The timestamp line and the recall nudge, in an active session. No request is made. Hermes calls it inline as a turn starts, with the agent's session id: the provider notes it as the session the agent is in (`live.py`), for the hooks that name none. |
| `on_turn_start(n, message, model=...)` | Keeps the model name for the uploads. A turn whose author is a bot or not the session's user turns the session inert before its nudge. |
| `sync_turn(user, assistant, messages=...)` | Uploads every complete turn not yet uploaded, each batch led by the turn before it as context. A batch that fails goes to the outbox; the next success drains it. Active sessions only, nothing for a private session, and nothing once a turn's `turn_author` is a bot or not the session's user. The session id is the one Hermes passes, which need not be one the instance was moved to (a session resumed in place, an agent Hermes pointed at a compression's new id, a row written after `initialize`): before the first upload under an id, its row in the store is read, and it is made private when a session it holds turns of is (`_follow`, once per id). After `synapse private off --forget` on a session that was private because of the one it continues, the next turn the user begins goes up when it ends, alone: no turn from before goes with it, as a turn or as its context, and a reply Hermes began itself (a background-process notice in the user's place) lands on the turn before it and stays off the record with it. A session marked by hand sends every turn once it is back on the record. Hermes runs `sync_turn` on its background worker, whose stall gate skips a provider stuck over 30 seconds, so a request is given 20. |
| `on_session_end(messages)`, `on_pre_compress(messages)` | Upload any turn not yet uploaded (10-second requests). Neither hook carries a session id, and the one the instance was built for or switched to need not be the agent's, so they go under the id Hermes named last (`live.py`: `initialize`, `on_session_switch`, `prefetch` and `sync_turn` name one; a `sync_turn` that runs late does not take the agent back to a session it was moved away from). Otherwise the rules are those of `sync_turn`, except that a flush never takes a turn for the one that has just ended. For a session held until the server answers, nothing is sent or queued: its turns are kept in memory (200 at most, per session id) and go up at the first upload for that id after the answer, also those a compression in place has taken out of the conversation. A compression that starts a new session leaves them with the id left: they are not uploaded. Hermes calls the first at every compression and `/new` as well as at exit, so both rely on the server's span-id dedup. `on_pre_compress` returns `""`. |
| `on_session_switch(new_id, ...)` | Rebinds the session id. At Hermes bca51750 a compression and the TUI's rewind reach it; the classic CLI's `/new`, `/resume` and `/branch` make the call, but its agent holds no memory manager. A reset or a new id starts the uploaded-turn memory afresh, looking back at most ten turns. A conversation that goes on under a new id (a compression, `/branch`: any new id that is neither a reset nor `reason="resume"`) stays private when the id it leaves (the one the instance is bound to, and the one Hermes named last), or the parent the switch names, is: the new id gets the marker and the server row (`Uploader.carry_private`), and no turn before the switch is looked back at. Nor is any when the id it leaves was off the record earlier in this agent and is back on it (`off --forget`): the new id is on the record, and holds those turns. With no marker to say so the server is asked at the session's next upload, and nothing is uploaded for it until the server answers. `/new` and `/resume` open another conversation: nothing is carried to it from the one left, and a resumed session answers for its own row at its first upload. If the check raises, the instance turns inert. A compression that keeps the id (`reason="compression"`) records the time, on the clock Hermes stamps its messages with; recall and recall_full_turns then carry it as `self_since`, so the turns before it serve again. A new id, a reset or a restart forgets it. |
| `get_tool_schemas()` | `recall`, `recall_full_turns`, `fetch`, `fetch_session`, `remember`, `recall_feedback`, from `tool_schemas.json`, in the server's order (`tools.schemas()`). Not shown to the model: `surface` anywhere, `self_session` on the recall family and `fetch_session`, `self_since` on `recall` and `recall_full_turns`, `session_id` on `remember` and `recall_feedback`. The `session_id` of `recall_full_turns` and `fetch_session` names the conversation to read and stays. Static: Hermes asks before `initialize()`. |
| `handle_tool_call(name, args)` | `tools.call()`: adds the session id, and the compression time where there is one, the way the `PreToolUse` hook does (never over a value already set), calls the server's MCP tool, returns its text: a JSON object as it is, other text as `{"result": ...}`, the shape Hermes's own providers return. Never raises: a failure, a call in an inert session, or a call before `initialize()` comes back as a JSON object with an `error` field. |

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
recall-feedback nudge, a Hermes command for private mode (the toggle is
`synapse private on|off|status <session-id>` in a terminal, the one Claude Code and Codex
use; there is no slash command or tool for it), skills sync into `$HERMES_HOME/skills`,
config sync, a history
importer for `state.db`, and Synapse in messaging-gateway and API-server sessions
(above). Skill invocations are uploaded as Hermes hands them over: the manager strips
the skill body from the user text it passes, not from `messages`.

**Tests.** The uploader, the private rule at the request and the carry-over are in
`tests/test_client_core_upload.py`, what the turn uploader remembers per session id in
`tests/test_client_core_turn_upload.py`; the toggle in `tests/test_client_core_private_toggle.py`
and, against a server, `tests/test_private_mode.py`. `tests/test_hermes_messages.py`,
`test_hermes_provider.py` (against a stub of Hermes's base class) and
`test_hermes_install.py` run everywhere, with
`test_hermes_gateway_sessions.py` (the rule, over a `state.db` built like Hermes's),
`test_hermes_private_lineage.py` (which sessions a row holds turns of, and private mode
for a new agent bound to such a session), `test_hermes_private_moves.py` (a flush after a
move Hermes did not announce, and what `off --forget` leaves behind),
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
the recall-feedback nudge and the Stop hook's private check come from the core; its
sender (`_post`) runs `private.check_payload` on the body it is about to send. Codex
keeps its own Stop-hook shipper (rollout cursors, catch-up sweep and retry), the
session-id injection hook (`pre_tool_use.py`) and the skills-sync glue. Its
`scripts/private_mode.py` is the core's toggle with Codex's settings (plugin 0.24.10).

Claude Code's hooks (`plugin/scripts/`) run on the core since plugin 0.24.5: `config.py`
resolves with `Settings.load` and sends through `Transport`; the board hook (which prints
the preferences too) and the prompt-nudge hook print what `context` builds, with Claude
Code's enrollment wording (`enroll.py`) passed in; the Stop hook uploads through
`Transport` and its private check
is `private.is_private`, with `private.check_payload` run again inside the sender
(`_post_records`), which the catch-up sweep and `synapse-import` share. Claude Code keeps
its own Stop-hook shipper (byte cursor,
chunking, the detached `--ship`, retry bookkeeping, the catch-up sweep), the session-id
injection hook (`self_session_inject.py`), the remember spool, the git feeder and the
skills and config sync, which send through `config.py`'s helpers. Its
`scripts/private_mode.py` is the core's toggle with the plugin's settings (plugin
0.24.10). The two harnesses' wiring moving into
`harnesses/` beside Hermes's is the next change.

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
  server (and the ingest URL's origin; for Claude Code's `config.py`, the MCP URL's too),
  and both Codex's and Claude Code's hooks send through it. A redirect carries it only
  within its origin, or from http to https on the same host and the default ports (80 to
  443, the rule the `requests` library uses). The sign-in calls that build their own
  request (`synapse_login.py`) open it through the same handler (`http.urlopen`, reached
  through `config._urlopen`): urllib's stock one forwards the header to wherever a
  redirect points.
- **Legacy credentials are passed in, not read.** `Settings.load` takes the old
  `device.json` record and `credentials.json` token as arguments (`legacy_device`,
  `legacy_token`); the harness that has those files reads them.
- **Session-start order** is notice, upload warning, board, preferences in every harness.
  The board and the preferences are one request (`context.server_parts`:
  `GET /context?parts=board,preferences`) and both are rendered by the server. A piece
  switched off is not asked for. A server from before the `parts` parameter answers
  with the board alone; its preferences are then read from `/preferences/top` and
  rendered by the client, in what is left of the same timeout.
- **Enrollment notices** take the login hint as an argument, and the restricted-device
  notice an optional aside after "here"; Claude Code passes its "paste it into /plugin"
  one, the other harnesses none.
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
  takes such a payload, and neither the cap nor the warning counts it, until `s` seconds
  have passed; after that its sender is taken for dead and a drain sends it, at most
  once more than needed, which the server's span-id dedup absorbs.
- **Uploads go through `upload`.** `Uploader.send` is that sequence (queue ahead, send,
  remove or keep) behind one call that answers sent, queued, lost or private, and
  `Uploader.drain` sends what waits. A payload whose session went private after it was
  queued is dropped by the drain; one whose marker cannot be read stays, unsent; the
  drain goes on to the rest either way. `turn_upload.TurnUploader` is the `messages`
  uploader Hermes started with: finished turns once each, the turn before as context, a
  resumed conversation looking back only so far. A harness names the session at every
  upload and may name several to one uploader, so what it remembers (turns sent, turns
  kept, turns spoken off the record, the look-back) is kept per session id.
- **A conversation that changes its session id.** Private mode is kept per session id.
  `Uploader.carry_private(earlier, new)` gives `new` the marker and the server row when
  one of `earlier` is private here (a marker, or one that cannot be read), and answers
  whether it did. A marker lasts twelve hours and `off` removes it, while the server's
  row stays, so with no marker the answer is not "no" yet: the uploader holds `new` and
  asks the server about `earlier` at the session's next upload (`settle_private`). A
  session carried a second time keeps what it was asking, and an earlier id that turned
  private here meanwhile is the answer without a request. An uploader makes a session
  private this way once: carried again it is left as it is, so `off --forget` stands.
  Nothing is sent or queued for a held session; `TurnUploader` keeps the turns finished
  meanwhile in memory (`turn_upload.KEPT_TURNS`, per session id) and sends them once the
  server says the earlier sessions are on the record, also those the conversation no
  longer holds. `Uploader.settle` says where a session stands for an upload: open,
  waiting for the answer, private, or made private by the server's answer just now. Where a write
  fails the uploader refuses the session itself: for as long as it lives when the marker
  could not be written, and until the server has the row when the request failed. The
  server is told again each time that session has a turn, and after any delivered
  upload for every session still owed (`settle_owed`, which stops at the first request
  that fails); each failed try is said to `warn`. With `timeout=None` the carry makes no request, and the caller settles.
  `settled` is the caller's hook for a question that was closed: the server said no
  earlier session is private, or it has the session's own row. Hermes records it so
  that a session is asked about once. It is not called when there was nothing to ask.
  `TurnUploader.upload(..., finished=True)` says a turn has just ended: after time off
  the record (`lookback` 0) that turn is sent, alone. Not when it only goes on from a
  turn spoken off the record: a reply with no real user row before it lands on the turn
  before (`messages.stems`). A look-back of 0 stays with the session, and with every id
  started as going on from it (`start(..., continues=...)`): a later start asks for a
  look-back in vain.

## Adding a harness

1. Decide the transcript source. If the harness can hand over chat messages, use the
   `messages` ingest format with a new `client` name: no server change. Only a harness
   whose transcript exists solely as a file in its own format needs a server parser.
2. Add `synapse_client/harnesses/<name>/` with the wiring (`detect / install / update /
   wired`) and register it. `synapse login` picks it up.
3. Map the harness's events onto the core: session start -> `context.session_block`,
   prompt -> `context.prompt_nudge`, turn end -> `turn_upload.TurnUploader.upload` with the
   conversation as `messages` records (`messages.py` says what a record holds; the
   adapter only converts its harness's shapes), tool calls -> `tools.call`. Do not post
   to the ingest route from the adapter: the uploader is where a private session's turns
   are stopped and a failed upload is queued. If the harness can give a conversation a
   new session id while it goes on, call `Uploader.carry_private` when it does.
4. Add a column to the table above and say what is not ported.
