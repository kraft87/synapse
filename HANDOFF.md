# Handoff log

Newest first. One entry per change that alters how a machine is set up or operated: what
changed, what to do on each machine, what was proven, and what is still open.

## 2026-11-12 — cli: `synapse update` lets the client it installed update the harnesses (plugin 0.24.9)

**What changed**

- **Before, one process ran two client versions.** `synapse update` ran the installer,
  then went on in the same process: the old CLI dropped the old `synapse_client` from the
  interpreter (`_forget_other_core`), imported the new client's, and drove its harness
  list. Old CLI code called new core code, and anything that had already imported the old
  package kept it.
- **Now the update is two processes.** The first fetches `<server>/install` and runs it,
  as before. It then starts the CLI of the client that is now current, through the
  launcher (`~/.synapse/bin/synapse_run.py plugin/scripts/synapse_cli.py update
  --rewire`), and exits with its exit code. That second process updates the Claude Code
  plugin, re-writes Codex's wiring when it is wired, and rebuilds the Hermes provider
  folder, all with the new client's own code. `--rewire` is internal and not in `--help`.
- **A server that serves an older client** (0.24.8 or earlier, which has no `--rewire`):
  that client is asked for a whole `synapse update`. It installs itself once more, which
  changes nothing, and updates the harnesses with its own code. Every managed client has
  `update`: it arrived with the launcher in 0.20.0.
- **No managed client to hand over to** (no launcher, or a current version whose manifest
  cannot be read): the harnesses are updated from the running process, as before.
- **The harness list always comes from the core beside the running CLI.** It used to
  prefer the managed client's core. So `synapse install hermes` typed inside a Claude Code
  session (the plugin's own `synapse`) builds the Hermes folder from the plugin's core;
  the next `synapse update` rebuilds it if the managed client is another version.
- **One URL rule.** `harness.py` had its own copy of `normalize_url`, kept equal to the
  core's by a test, because importing the core there would have pinned the old package
  during an update. It now uses the core's.

**On each machine**

1. `synapse update` for plugin 0.24.9. This run is still done the old way (the client
   being replaced is the one running it); updates after it use the hand-over. Nothing to
   migrate, no setting to change, no server change.

**Proven**

- `tests/test_update_handover.py` runs the update as real processes for a scratch user
  against a local server: this checkout's CLI installs a client calling itself 9.9.9,
  and the Hermes folder is rebuilt as 9.9.9 from that client's core, after the installer
  made it current. With an installed client from before the step, that client's `update`
  runs, its exit code comes back, and the caller builds nothing.
- `tests/test_synapse_cli.py`: the installer runs first and the hand-over second with
  `SYNAPSE_HOME` set; the caller touches no harness; `update --rewire` fetches nothing
  and runs no installer; no managed client falls back to the running process; the flag is
  not in the help.
- `tests/test_claude_on_core.py`: a managed client of another version on the machine
  does not put a second `synapse_client` into the process.

**Still open**

- Not run on Windows. The hand-over is a child process, not an `exec`, so it does not
  depend on one.

## 2026-11-12 — client: the private-mode check sits where a turn leaves the machine (plugin 0.24.8)

**What changed**

- **Before, every uploader had to remember the check.** A session in private mode is kept
  off the server by a check each uploader made before it called the sender. The senders
  themselves sent whatever they were handed. `synapse-import` made no check at all: it
  relied on the server's `private_sessions` row alone, so a session whose row had not
  landed (the toggle then exits nonzero and leaves the marker in place) was imported.
- **Now the sender refuses.** `private.check_payload` reads the sessions a body names
  (its own `session_id`, and each record's `sessionId`) and raises `PrivateSession`
  before any request when one is in private mode, or has a marker that cannot be read.
  It runs in:
  - `Client.ingest`, the only place the client core posts to the ingest route;
  - `_post_records` in the Claude Code Stop hook, which the catch-up sweep and
    `synapse-import` share;
  - `_post` in the Codex Stop hook.
  The uploaders keep their own earlier check (it saves reading the transcript). A session
  marked between that check and the request is skipped, and is not counted as a failed
  upload: no cursor moves, nothing is marked pending, no warning streak starts.
- **`synapse-import` skips private sessions**, by the transcript's file name and by its
  records, prints `skipped (private session)` for each, and no longer exits nonzero when
  the only files it did not import were private ones. It reads the whole file before the
  first batch goes, so a file is imported whole or not at all. A file skipped because a
  marker could not be read is counted apart (`not imported: a private marker could not be
  read`), and an import that sent nothing for that reason exits 1.
- **One uploader in the core** (`plugin/synapse_client/upload.py`). `Uploader.send` queues
  a body ahead of its request, sends it, and removes it or keeps it as waiting; it answers
  sent, queued, lost or private. `TurnUploader` uploads a conversation's finished turns
  once each in the `messages` format. Both were inside the Hermes provider; the turn
  rules moved with them (`plugin/synapse_client/messages.py`), and the Hermes adapter
  keeps only the conversion from Hermes's message dicts. A fourth harness hands its
  records to `TurnUploader` and gets the check, the queue and the retry with them.
- **A drain no longer sends what went private while it waited.** A queued upload whose
  session was taken off the record afterwards is dropped by the next drain (the server
  would refuse it anyway). One whose marker cannot be read stays queued, unsent, with a
  log line, and does not count toward the 20 uploads a drain handles. The drain goes on
  to the rest in both cases.
- **Hermes: an upload that is lost is a warning again.** A batch that failed and could
  not be queued either is the one thing nothing sends again; its line is logged at
  WARNING (`could not queue a failed upload (...); it is lost`).
- **Hermes: a session is no longer skipped because another one started at the same
  moment.** The marks lookup (`harnesses/hermes/marks.py`) asked "is anything there?"
  and then "is it a broken link?" as two steps. When another session of the same home
  created `<hermes_home>/synapse/` between them, the lookup took the new folder for a
  broken link, and that call was skipped with `the session store or its marks could not
  be read`. It now looks once more. This was the cause of a test that failed about one
  run in twelve.

**On each machine**

1. `synapse update` for plugin 0.24.8. Nothing to migrate, no setting to change, no
   server change.

**Proven**

- `tests/test_client_core_upload.py` (24 tests): `Client.ingest` sends nothing for a
  private session however the body names it; an unreadable marker refuses without
  claiming the session is private; a body is on disk before its request and gone after;
  a failed send waits with its reason; a private session is neither sent nor queued,
  including one marked between the uploader's check and the request; a drain drops what
  went private, keeps what it cannot tell about, and sends the rest; turns go once each,
  with the lookback and the retry the Hermes provider had.
- Claude Code hook, Codex hook and import: each sender refuses a private batch with no
  request made; a session marked after the first check is skipped with nothing recorded
  as failed; the import skips by file name, by record and for an unreadable marker, and
  still imports the rest.
- The Hermes provider's own tests pass unchanged apart from where the moved functions are
  imported from, the in-flight and failed-turn suites included.
- The marks race: a test that creates the folder between the two looks fails on the old
  code and passes on the new. The concurrency test that failed 1 run in 12 on main passed
  40 of 40 with the fix.
- An independent review of the change found no path by which a private session's turns
  leave. Its follow-ups are in: the whole-file check in the import, the import's exit
  code, the drain's count, and tests that fail when each of these is taken out (a
  transcript marked between two of its requests keeps its cursor and resumes; a turn
  refused at the request is sent once the session is back on the record; the Hermes
  provider's drain goes through the check).

**Still open**

- Claude Code's and Codex's Stop hooks still have their own shippers (cursor files, the
  catch-up sweep). They run the shared check; they do not use `Uploader` yet.
- The private-mode toggle is still one script per harness, and Hermes has none.
- Hermes: a conversation that goes on under a new session id (a compression, `/branch`)
  is not private because the id it left was. No Hermes session can be private without a
  hand-made marker today; this has to be closed before Hermes gets a toggle.
- Hermes: while a session's marker cannot be read, its turns are neither sent nor
  queued, and are lost if the process exits first. One log line per attempt says so.

## 2026-11-12 — hermes: the provider works on `hermes chat`, for the owner only; failed turns and exits no longer corrupt or lose uploads (plugin 0.24.7)

**What changed**

- **Before this change the provider did nothing on `hermes chat`.** `hermes chat` and
  the TUI run their turns in Hermes's gateway daemon, which passes a chat identity even
  for the owner's own terminal. The provider read any chat identity as a messaging
  gateway and went inert: no board, no nudge, no upload, and every tool call an error.
  Only the in-process surfaces (stdio TUI backend, Desktop, ACP, the classic CLI) worked.
- **A daemon session is active only when it is provably the owner's local session**
  (`harnesses/hermes/sessions.py`):
  - the local shape: `chat_id` `local-<hex>`, `chat_type` "dm", the
    `agent:<profile>:local:dm:` route and `user_id` `uid:<this OS user>`;
  - its row in `<hermes_home>/state.db`, opened read-only;
  - and a creation receipt there that names the owner as creator, an owner's chat as
    the source (not `--source tool`), no Bot Chat title, and no `follow_profile_config`
    in the request (the Desktop's mark for its Bot Chat and Group Chat members).
  - No row of the same chat may be titled "Bot Chat" or "Group: ...": a `/new` or an
    auto-reset in the Bot Chat publishes an untitled child that peers still reach.
  - Everything else stays inert: messaging platforms, the API server, hosted rooms,
    peer agents, the Bot Chat, Desktop Group Chat members, subagents, sessions someone
    else created, and any shape not listed. The reason is logged once.
  - The session's rows, and those of the chat its route answers on, are read again
    before the system prompt block and each nudge, upload and tool call. A session that
    stops passing, or a gateway session whose row is gone, turns the instance inert for
    good.
  - The project comes from the `cwd` the client sent, never from the daemon's.
- **In-process sessions** (no chat identity) are active unless the process is the
  gateway daemon (its `gateway.lock` or `gateway.pid` record holds this pid: a
  `/background` or `/btw` side agent), it was started with a `--source` that is not an
  owner's chat (`HERMES_SESSION_SOURCE` "tool", "kanban", ...), or its row is titled
  "Bot Chat" / "Group: ..." or carries `follow_profile_config` / `room_plumbing` in
  `model_config`.
- **Cron jobs are inert, the tools included.** They used to get the tools: whoever can
  have the bot schedule a job chooses its prompt and where its output is delivered.
  `CRON` stays importable; no session gets it.
- **A turn another agent or person wrote turns the session inert for good**, logged once,
  and is not uploaded. Hermes names a turn's author on `on_turn_start` (before the
  nudge) and in `sync_turn`'s `turn_author`. The daemon names one on every turn, the
  owner's own as the session's `user_id` and not a bot (`gateway/run_turn_runner.py`);
  in-process, a human turn has none. A bot (the flag alone, with or without an id),
  another id, or any author in-process is someone else. The owner's id as author means
  only that nobody stamped the turn: an unstamped delivery into a Bot Chat is stopped
  by the chat's own marks, not by this rule.
- **A chat others wrote into stays off after its title is gone, its agent is rebuilt or
  it is branched.**
  - The first time a session is judged a Bot Chat, a Group Chat member or a chat that
    follows its profile, or is handed a turn someone else wrote, it is marked, together
    with the chat that said so and the parent sessions down to it: one owner-only file
    each under
    `<hermes_home>/synapse/inert/`, named by the SHA-256 of the id and holding the id
    and the reason (`harnesses/hermes/marks.py`).
  - A mark makes that session and every session of that chat inert. Before, "for good"
    lasted only as long as the provider instance: a rebuilt agent started active again
    and, resumed, uploaded up to ten earlier turns as the owner's, the foreign ones
    among them. A Bot Chat archived or renamed read as an owner's chat.
  - A session is held to the sessions it descends from (`harnesses/hermes/lineage.py`):
    a parent row, any row of a parent's chat, a parent chat's receipt that says Bot
    Chat, Group Chat member or `follow_profile_config`, or a parent's mark. A branch of
    a Bot Chat under a title of its own used to pass.
  - A parent's mark is copied down to the session that inherits it, its chat and the
    parents between, the first time that session is judged. Hermes clears a child's
    `parent_session_id` when the parent is
    deleted or pruned, so the copy is what keeps a branch off after that. A branch not
    judged before then is still found by the `_branched_from` (or `_reset_from`,
    `_compressed_from`) its `model_config` keeps, when the session it names was marked.
  - Marks are only added, and a mark on a branch never marks its parent. A session
    comes back on only when every file that names it, its chat and the sessions it
    descends from is deleted (each file holds the id it is for), and the rows no longer
    say what was marked.
  - Marks that cannot be read make the session inert at `initialize` and skip the call
    afterwards: a folder that cannot be searched, a file in its place, or a link that
    leads nowhere (an unmounted volume). Only "nothing there" reads as not marked. A
    mark that cannot be written is logged; the session is still off for that agent.
- **A process whose environment carries `HERMES_TURN_AUTHOR` is inert from
  `initialize`.** A backstop only: no Hermes path builds an agent in such a process
  today (see Open).
- **A tool call before `initialize` is refused.** It used to reach the server. Hermes
  refuses such a call itself (`agent/memory_manager.py`, `caller_scope`), so nothing
  that worked stops working. With cron gone, a tool call works only in an active
  session.
- **A failed turn is never uploaded.** Hermes's notice for a failed turn (an assistant
  row typed `display_kind: failed_turn`, or the bare notice) was uploaded as the
  assistant's reply. The failed turn is now dropped, its user message included. An
  assistant row of any other `display_kind` keeps only its tool calls.
- **An upload in flight when the process exits waits in the outbox.**
  - Each batch is written to the outbox before its request, unsynced, as
    `<name>.json.<deadline>.sending`. It is removed once delivered; a failed send renames
    it to waiting.
  - No drain takes it until three times the request timeout has passed (60 s for a turn
    upload). After that a drain sends it, at most once more than needed, which the
    server's span-id dedup absorbs.
  - This holds for every session and process sharing a home's outbox. It adds about
    0.2 ms per batch on Hermes's background worker.
  - The session-start upload warning does not count a batch still within its send.
- **The outbox's directories are owner-only**, every parent it creates included (POSIX).
- **`synapse install hermes`** reads `memory.provider` with `hermes config get --json`.
  - A provider named "null" now counts as another external provider, not as Hermes's
    built-in memory.
  - A config.yaml that already selects synapse is reported "already selected": without
    the program, or when the program cannot answer. It no longer prints a command to
    run.

**On each machine**

1. `synapse update` once the server serves 0.24.7 (or `synapse install hermes`).
2. `hermes gateway stop`, so that the daemon loads the new provider; the next
   `hermes chat` starts it again. Until then a running daemon keeps the old provider,
   which stays inert on `hermes chat`.
3. Nothing on the server.

**Proven**

- New tests in seven new files: `test_hermes_gateway_sessions.py` (the rule, over a
  `state.db` built like Hermes's), `test_hermes_privacy_audiences.py` (73: each
  audience above, its shape built from what Hermes writes and cited),
  `test_hermes_failed_turns.py`, `test_hermes_inflight_upload.py` (a child process
  killed mid-request, sessions sharing an outbox), `test_client_core_outbox_writeahead.py`,
  `test_client_core_outbox_private_dirs.py`, `test_hermes_install_selection.py`.
  - Non-db suite: 4463 passed, 38 skipped, 2 xfailed.
  - With `SYNAPSE_HERMES_CHECKOUT`, the integration test and
    `test_hermes_checkout_names.py` pass against Hermes `bca51750`, the columns the
    rule reads from its store and the `model_config` keys that name a copy's origin
    among the names checked. Ruff is clean.
  - The rule as merged, run over a copy of the `state.db` those real runs left (30
    sessions): the 25 owner sessions active, the three `--source tool` ones inert, the
    Bot Chat and the `follow_profile_config` session inert and marked. That store has
    no branch, so the parent check ran on built stores only.
- Real `hermes chat` (Hermes `bca51750`, a scripted stub model, the production server):
  ten messages, nine complete turns and one failed; seven turns uploaded.
  - The board and preferences reached the system prompt and the nudge reached each
    user message. A `recall` call round-tripped to the model.
  - Each complete turn became one row, platform `hermes`, project `hermes-scratch`
    (from the client's cwd). A resumed session's second turn added exactly one row.
  - A turn whose model call failed was not uploaded. Neither its notice nor its user
    message is in the next turn's upload.
  - A turn whose upload hung (a black-hole proxy) was still in the outbox after
    `hermes gateway stop`. The next session, past the deadline, sent it once (one POST
    in the server log) and left the outbox empty.
  - Two concurrent `hermes chat` sessions in one daemon got one row each, each holding
    only its own turn.
  - Sessions started with `--source tool` stayed inert: no board, no nudge, no upload,
    the reason logged. The model's `recall` there got the local-only JSON error and
    nothing reached the server.
- After the privacy fixes, three more turns (tests 11 to 13):
  - An ordinary `hermes chat` stays active: board and preferences in the system prompt,
    the nudge on the message, `recall` returned real results, one row uploaded (245709).
    (Test 11, before the turn-author fix, showed the daemon names the owner as author on
    every turn: the session went off and uploaded nothing.)
  - A session created over the control channel with `follow_profile_config` (Hermes's
    own client, no title, the owner's receipt) stayed inert: no board, no nudge,
    `recall` got the local-only error, nothing uploaded, reason "a chat that follows
    its profile".

**Open**

- Messaging-gateway and API-server sessions still get nothing: they need a way to say
  whose chat it is.
- Not driven live in this change: `hermes --tui` over the daemon (it reaches the
  provider as `hermes chat` does), Desktop, the Webapp and ACP. Whether a Desktop or
  Webapp submit stamps a turn author was not determined; if one does, those chats turn
  inert at their first turn.
- The marks, the parent check, a deleted parent and a rebuilt agent are covered by unit
  tests only.
- Hermes keeps the nudge in its stored transcript (`api_content`). The upload strips
  it; the transcript is Hermes's.
- An automation turn in an owner's chat (a cron delivery or notice, a `display_kind`
  row) is skipped as synthetic, so the assistant's reply to it is filed with the
  previous real turn.
- The welcome-trial mark `follow_adopted` is not read: Hermes sets it on an ordinary
  chat created on the trial model (`gateway/session_local.py`) as well as on Bot Chats.
- The board already in a stored system prompt stays: Hermes restores it verbatim for a
  continuing session (renamed, branched, handed off). Only later rebuilds go without.
  Two moves then take it, with the whole transcript, to a messaging chat: the owner's
  own `/handoff`, and on Matrix `/resume <id or title> --cross-room` from anyone allowed
  to message the bot (other platforms need a configured admin for it). Read in Hermes's
  source, not run; not reported upstream yet. The provider cannot stop either.
- The first turn's prompt is built before `on_turn_start` names an author. A bot-to-bot
  delivery run as a one-shot through the daemon would get the board on that one turn
  before the session turns off. Nothing in Hermes `bca51750` starts such a run:
  `hermes chat` and both one-shot paths run in the daemon, whose environment has
  `HERMES_TURN_AUTHOR` stripped, so the environment check never fires there.
- Whoever can sign in to the Hermes dashboard or Webapp is the owner here: Hermes maps
  every login to the owner's `uid:` subject. That covers a remote Desktop, a phone and
  a standalone dashboard.
- A side agent built in the stdio TUI or Desktop backend's own process has no parent
  and no row, so it is active even when its parent is a Bot Chat.
- A chat that stopped looking like a Bot Chat before this provider first judged it has
  no mark. An older Bot Chat (no `follow_profile_config`, a random request id) that was
  archived or renamed before this version ran reads as an owner's chat, and resuming it
  uploads up to ten earlier turns.
- A copy that names no origin reads as the owner's chat. Two in-process paths copy a
  session's turns into a new row with no `parent_session_id` and no `_branched_from`:
  ACP `fork_session` (`acp_adapter/session.py`) and a seeded `session.create` without a
  parent (`tui_gateway/methods_session.py` `_seed_row`). Only the owner's own client can
  make one. A copy of a Bot Chat made that way uploads up to ten copied turns when
  resumed.
- A branch never judged before its parent was deleted is found only when the session it
  was branched from carries a mark itself: it was judged, or it lies between a judged
  session and what marked it. One cut from a segment of a marked chat that no judged
  session descends from reads as the owner's chat. So does the child of an in-process
  compression whose parent row alone is deleted: it names its parent by the link only
  (`agent/conversation_compression.py`).
- In the stdio TUI or Desktop backend a model switch on a draft builds the prompt
  before the session has a row, so a draft that becomes a Bot Chat or Group Chat member
  can keep the board in its first prompt.
- Automation turns in an owner's chat (wake, goals, notifications) run with the nudge
  and the tools: Hermes gives them the owner as author.
- Wrongly inert, known:
  - a session from before Hermes's creation receipts, resumed through the daemon (its
    chat id is the old row id, not `local-<hex>`);
  - the owner's own `/background` and `/btw` in the daemon;
  - every daemon session on Windows (the owner subject there is a SID);
  - an agent built while `state.db` could not be read, until Hermes rebuilds it;
  - a chat the owner titled exactly "Bot Chat", "Bot Chat #N", "Bot Chat (...)" or
    "Group: ...", even briefly: once judged under that title it is marked for good, and
    so is each branch of it judged from then on;
  - every later segment (`/new`) of a chat branched from a marked one: the branch's
    chat is marked with it;
  - a branch cut from a session before that session was handed a turn someone else
    wrote: it holds none of it, and is marked with its chat once judged;
  - every session of a home whose marks folder cannot be read.

## 2026-11-12 — one memory, the cleanup: the group column and the merge ledger go (schema 069, optional)

**What changed**

- **Schema 069, optional.** Since 067 no service reads or writes `group_id`, and every row
  says `technical`. 069 drops the column from `kg_entities`, `kg_relationships`, `notes` and
  `preferences`, the nine indexes that lead with it, and the seven `kg_pile_merge_067_*`
  tables (the merge's undo ledger). `kg_entities_one_self`, `notes_live_idx` and
  `preferences_live_idx` are built again on `owner_id` alone before the column goes.
  `recall_metrics.group_id` stays: it is telemetry history.
- **No service needs it.** The boot guard still asks for 068, and the same image runs with
  or without 069. A fresh install gets it at first boot.
- **It refuses rather than guesses.** A preflight changes nothing and names what to look at
  when the stamp is below 068, a row's `group_id` is not `technical`, an owner-only index
  from 067 is missing, or anything else depends on the column (a view, a trigger, a
  generated column, a constraint, an index 069 does not know). Every later step refuses
  unless that preflight passed in the same session, takes its table lock first, and waits
  at most 5 s for it. The stamp moves to 069 only once the file has checked the end state.
- **Reads of one entity's facts name its owner.** After 069 every index on the fact table
  leads with `owner_id`; `kg_rel_src`, which 069 drops, was the last narrow one a read by
  source alone could walk. The dashboard's entity page and graph, and `synapse-admin owner`,
  now read an entity's facts under that entity's own owner (a fact is filed under its
  endpoints' owner, so nothing is hidden) and range over an index with or without 069.
- **The merge's undo is over.** `scripts/pile_merge.py` is deleted, with its preview, health
  check and exclude list (its undo already refused on a database stamped above 067), and
  069 drops the before-images it read. 067 is its file alone, and stamps only once no
  personal row is left.
- **Older schema files leave the column alone once it is gone.** 017, 035, 041, 066 and 067
  run their `group_id` statements only while the column exists, so applying one again
  after 069 brings back no column, index, trigger or function.
- **`repair_numbered_entity_magnets.py --restore` refuses a backup taken before 069** on a
  merged database (stamp 067 or later): such a backup may predate the merge, and restoring
  it could bring back entities the merge folded away. New backups name their columns.
- **The upgrade instructions are right now.** The boot hint and `docs/install.md` no longer
  send an existing database to `scripts/apply_schema.sh`, which is for an empty one and
  stops at the first file that cannot be applied twice. They say: read the stamp, apply the
  files above it one at a time, in order.

**On each machine**

1. The release itself needs nothing: no migration, no setting, no client update.
2. The cleanup, whenever you want it, on a database at 068 or later, with nothing stopped:
   `psql -X "$SYNAPSE_DB_URL" -v ON_ERROR_STOP=1 -f schema/069_one_memory_cleanup.sql`.
   Never `-1`. Take a dump first: after 069 a dump is the only way back. If a step gives up
   on its lock (a long reader was in the way), nothing is half done: run the whole file
   again, with no older schema file applied in between.

**Rolling back**

- The release: revert as usual. Any build from the one-memory release on runs with or
  without the column.
- 069: there is no undo. A build older than the one-memory release cannot write these
  tables afterwards (its statements name `group_id`); going back that far means restoring
  a dump.

**Proven**

- `tests/test_one_memory_cleanup_069_db.py`: on a database at 068 the file drops what it
  says and keeps every row; each preflight refusal changes nothing, also when the client
  does not stop on the error; a second run changes nothing and waits for no lock; a run that
  stopped after its first step completes when run again; a reader that then writes is never
  a deadlock victim; a column dropped by hand, or a one-self index of another shape, is not
  stamped; every schema file applied again after 069 brings nothing back.
- `tests/test_no_group_sql.py`: no statement in `mcp_server/`, `ingestion/` or `dream/`
  names `group_id`, and none reads the four tables with `*` (a prepared `SELECT *` fails
  when a column is dropped under it).

- **Rehearsed on a copy of a live database** (53k entities, 105k facts, PostgreSQL 17)
  brought to 068 the way that database got there. The file applied in 0.35 s while pooled
  connections of the services' own code held prepared statements on the four tables: no
  read or write path failed after the drop, and the longest read waited 9 ms. Every row of
  the four tables was identical before and after, all 21 remaining indexes valid. A reader
  holding a transaction open made step 1 give up after 5 s with nothing changed; a row not
  `technical` was refused by the preflight; a second run under an open reader took 0.3 s
  and no lock; 066, 067 and 068 applied again brought nothing back; the boot check passed
  on this build and on the one before it.
- The recall bench on that copy found every gold answer before and after (five gathers).
  481 of 490 contexts were identical across the cleanup, against 487 of 490 between two
  gathers before it; about half of the differences are the same ids in another order. The
  rehearsal wrote and removed its own marker rows between those gathers, and the file
  touches neither search index, but one changed context was not traced to a cause.
- The reads that now name the owner, on that copy at 069, without and with the owner: an
  entity page's facts 27 to 1.5 ms (the owner's own node 29 to 13.5), its counts 30 to 0.3
  ms, a graph level 23 to 4.9 ms, the graph's edges 24 to 0.8 ms, the owner node's degree
  46 to 5 ms.
- The rehearsal found that the preflight accepted a database at 067 and would have stamped
  069 over a 068 that never ran; the floor is 068 now. The file rehearsed differs from the
  one shipped in that floor and in comments.
- CI applies every schema file on PostgreSQL 18, the gate on 17.

**Still open**: `recall_metrics.group_id` (history) and the `group_id` argument that
`recall` still accepts and ignores. 067 applied again after 069 still runs its two
`ANALYZE`s (about 2.6 s on that database).


## 2026-11-12 — install: `synapse` is linked into `~/.local/bin` when that is on PATH

**What changed**

- **The command works without a profile edit where the usual user bin directory exists.**
  The installer keeps the client in `~/.synapse` and its shim in `~/.synapse/bin`, as
  before. On Linux and macOS, when `~/.synapse/bin` is not on `PATH` but `~/.local/bin` is
  an existing directory on `PATH`, it now also creates `~/.local/bin/synapse` as a
  symlink to the shim, and says so. Before, it only printed the line to add to a shell
  profile, so on a machine where nobody added it the command was not found outside a
  Claude Code session.
- **It takes the name only when the name is free.** A file, or a link to anything else,
  named `synapse` in `~/.local/bin` is left as it is; the installer says so and prints the
  profile line as before. It never creates `~/.local/bin` and never edits a profile.
- **Only the default home is linked.** An install under a `SYNAPSE_HOME` other than
  `~/.synapse` is a side install and gets the printed line, never the link.
- **Windows is unchanged** (the installer already adds `~/.synapse/bin` to the user
  `PATH` there).

**On each machine**

1. Nothing until the server runs this image. Then `synapse update` (or
   `~/.synapse/bin/synapse update` where the command is not found yet): the update runs the
   installer again, which creates the link. No plugin version change; the installer is
   served by the server.
2. A line that adds `~/.synapse/bin` to `PATH` in a shell profile can stay or go; with it
   the installer finds the directory on `PATH` and creates no link.
3. To remove the client by hand, remove `~/.local/bin/synapse` along with `~/.synapse`.

**Proven**

- `tests/test_install_route.py` runs the rendered installer in a subprocess with a scratch
  `HOME`: the default home with `~/.local/bin` on `PATH` gets the link, `synapse` resolves
  through it and runs the installed client, no profile line is printed, and a second run
  finds the link and says nothing; with `~/.local/bin` missing or off `PATH` there is no
  link, the directory is not created and the line is printed; a regular file, a link
  elsewhere and a dangling link of that name are each left untouched; a `SYNAPSE_HOME`
  elsewhere is never linked.
- Each rule was broken once (side home linked, off-`PATH` directory linked, foreign file
  replaced, own link not recognized, link never made) and a test failed each time.

**Still open**

- No uninstall command exists; removal is by hand (step 3).

## 2026-11-12 — recall: after a compaction, a session can find its own earlier turns again (plugin 0.24.6)

**What changed**

- **Self-exclusion stops at the last compaction.** `recall` and `recall_full_turns` leave
  out the calling session's own turns because the model still holds them. That stops being
  true when the harness compacts the conversation: the turns before that point are gone
  from the context and were still hidden, so a long session could not look up its own
  earlier work. Both tools now take `self_since`, the time of the session's last
  compaction. With it the server hides only the session's turns from that time on, and a
  fact counts as the session's own only when every source is one of those turns. Without
  it nothing changes.
- **Earlier own turns are added, never swapped in.** With the cut, the other sessions'
  turns served are exactly those served without it. At most two of the session's earlier
  turns are then added (`SYNAPSE_RECALL_SELF_PRIOR_LIMIT`, `0` = none), each only when it
  scores at least as well as the weakest other turn served, so a list can be two longer.
- **The client reads the compaction time from the transcript.** The Claude Code and Codex
  PreToolUse hooks add `self_since` next to `self_session`
  (`plugin/synapse_client/compaction.py`: the last `compact_boundary` line of a Claude Code
  transcript, at the first message the harness kept verbatim when the line names one; the
  last `compacted` line of a Codex rollout). The Hermes provider takes the time when Hermes
  commits an in-place compression. A transcript that cannot be read, or one with no
  compaction, sends nothing, and the whole session stays hidden as before.
- **Telemetry.** A call that carried a cut records `self_since` and `n_self_served` (how
  many of the served episodes are the caller's own) in `recall_metrics.served_ids`.

**On each machine**

1. Nothing to migrate and no setting to change. The server part takes effect with the image.
2. `synapse update` for plugin 0.24.6; until then a client sends no `self_since` and
   behaves as before. A server older than this release rejects the argument, so the client
   must not get ahead of its server; `synapse update` cannot, because the installer serves
   the client that matches the server it runs on.

**Proven**

- With no cut the SQL and its parameters are what they were. On the standard recall bench
  (491 questions, no cut sent) the result is identical to main's in 475, where main run
  twice twenty minutes apart agrees with itself in 465, and every gold answer one finds
  the other finds.
- Replayed on 48 real recalls made after a compaction: the old path served none of the
  session's own turns; with the cut, 20 calls served one or two, and 21 gained facts from
  earlier in the session. A pairwise judge preferred the new results in 20 of the 30 calls
  whose results changed and the old in 3, with 7 ties. (Before the earlier turns had a lane
  of their own, when they competed for the same places, it was 17 and 12.)
- In 45 of those 48 calls the other sessions' turns were identical with and without the
  cut. The 3 that differed are the reranker's own variation, which repeats of the old path
  alone show too: a score moves by about 0.001 between two identical calls, and two
  passages with the same text come back in either order.
- With the cut, both tools serve the same other-session turns, in the same order and with
  the same scores, as without it: with the reranker on, degraded, and on the score-cutoff
  path, including a pool where more own turns than the passage window holds rank first.
- Each rule has a test that fails when the rule is broken: the comparison at the cut; an
  own turn with no usable time stays hidden; the session-scoped drill-down ignores the cut;
  the hooks add it only to the two tools, only for their own session, and never overwrite
  one; a marker that appears inside tool output is skipped.

**Still open**

- Hermes keeps a verbatim tail across an in-place compression. Those few turns are stamped
  before the cut, so they can be served while still in the context.
- Subagent calls are unchanged: they carry the parent's session id, and now its cut.
- Not run on Windows: the scan maps the transcript read-only while the harness appends to it.

## 2026-11-12 — plugin: the Claude Code hooks run on the client core (plugin 0.24.5)

**What changed**

- **`plugin/scripts/config.py` resolves and sends through the core.**
  - URL, ingest URL, MCP URL, flags and token come from `Settings.load`, fed the plugin's
    own sources in the old order: the `CLAUDE_PLUGIN_OPTION_*` environment, the
    `settings.json` plugin options, and the pre-0.20 `credentials.json` and `device.json`.
    Every name another script, test or doc uses is kept.
  - `get_json`, `post_json` and `request_json` send through `Transport`. The bearer goes
    only to the origins of the server, ingest and MCP URLs; before, the helpers sent it to
    any absolute URL they were given. No shipped script asks for one.
  - Requests carry `User-Agent: synapse-client/<version> claude-code` (was
    `synapse-plugin/0.9`).
- **Session start prints the core's blocks.** `board_block.py` prints the token notice and
  `context.board_part`, with Claude Code's enrollment explainers passed in;
  `preferences_block.py` prints `context.preferences_part`. Still two hooks with their
  own kill switches (`SYNAPSE_BOARD`, `SYNAPSE_PREFS_BLOCK`).
- **The prompt hook prints `context.prompt_nudge`**, naming `synapse:recall` and
  `synapse:remember`.
- **The Stop hook uploads through `Transport`** to the ingest URL exactly as configured
  (trailing slash and query kept), and asks `private.is_private` about private mode. The
  marker TTL is still read from the environment only. The shipper is unchanged: byte
  cursor, chunking, the detached `--ship`, retry bookkeeping, `--catchup`.
- **`enroll.py`'s explainers are the core's**, with Claude Code's sign-in hints. The core
  gains `restricted_block(paste_aside=)` for the "/plugin" aside.
- Core additions, each tested: `Settings.with_sources`, the `notice_part` and
  `override_notice` source wording, `board_part(not_enrolled=, restricted=)`,
  `restricted_block(paste_aside=)` and `Transport.open`.

**On each machine**

1. `synapse update` once the server serves 0.24.5. `hooks.json` is byte for byte the
   same (pinned), so no hook changes.
2. What a Claude Code user sees: no change in any hook's output for any reply the server
   sends. The server's logs show the new User-Agent.
3. Changes only a broken setup or a malformed reply can reach:
   - `SYNAPSE_URL` of only slashes, an ingest URL from another source, and an enrolled
     machine: the device-token decision now looks at the ingest URL alone, since the
     empty base URL is never requested. The notice and token differ in 3 of 4200
     generated homes.
   - A board reply whose `text` is not a string now prints, with the notice; before, the
     hook printed nothing at all. A preferences reply with a non-dict item prints the
     valid ones; before, nothing.
   - A clock the platform cannot format: the prompt hook prints nothing and exits 0;
     before, a traceback and exit 1.

**Proven**

- Full non-db suite: 3866 passed, 34 skipped, 2 xfailed (3781 before). Ruff check and
  format are clean.
- New `tests/test_claude_on_core.py` (73 tests), against local fake servers:
  - the resolution ranking and the token rules, notices byte for byte;
  - the bearer kept off another origin, for `config.py` and for the Stop hook's upload,
    redirects included;
  - each session-start hook's bytes for a normal reply, a 401 enrolled or not, a
    restricted caller with and without a device token, a 500, the server down and its
    kill switch;
  - the prompt hook for each kill switch and source;
  - the Stop hook's POST (URL, User-Agent, bearer, body), its 401 and 500 log lines and
    recorded reasons, and the private skip, stale marker and path-like record id;
  - `hooks.json` unchanged.
- Old vs new, one fresh process per case, temp homes, made-up tokens:
  - `config.py` and `mcp_headers.py`, 4200 generated homes: before vs before 0; before
    vs after 3, the slashes-only URL case above. `mcp_headers.py` identical in all.
  - The hooks against a fake server, 1112 cases (normal, 401, restricted, 500, hung,
    not ok, kill switches, malformed replies): only the User-Agent and the two
    malformed-reply cases above.
  - `ingest_hook.py --ship`, 318 cases (URL forms, token sources, replies, private
    markers, TTL values): identical apart from the User-Agent.
- `recall_nudge.py` takes 26 ms more (median of 30). All of it is `config.py` now
  importing the core (`typing`, and `dataclasses` pulling in `inspect`), which
  `recall_feedback_nudge.py` already paid.
- Read-only smoke on this machine, against the real server: `board_block.py`,
  `preferences_block.py`, `recall_nudge.py` (timestamp masked) and `mcp_headers.py`
  print the same bytes from the installed 0.24.1, from 0.24.4 and from this change
  (compared by length and hash).

**Open**

- Not run on Windows. The hook paths use no POSIX-only calls, and `~` expands as before.
- The two harnesses' wiring moving into `harnesses/` beside Hermes's.

## 2026-11-12 — ingest: the transcript line rules live in one module

**What changed**

- **New `ingestion/render.py`.** It holds what every transcript parser rendered by hand:
  - the line shapes `[context]`, `[user]`, `[assistant]`, `[tool:name]`, `[result]` and
    `[title]`, with their caps (300, 3000, 3000, 300, 500, 200);
  - a `[result]` line only for text longer than 20 characters;
  - the NUL strip and the blank-line join;
  - the loop that opens a new group at each user turn;
  - the check that emits a repeated span id once.
- **The six parsers use it** (`jsonl`, `codex`, `messages`, `openai`, `cursor_sqlite`,
  `claude_ai`). Their private `_strip_nulls` copies and grouping loops are gone. Each
  parser still decides which records become which line, in what order, and when a group
  is too empty to keep.
- `claude_ai_client.py` keeps its own `[user]` and `[assistant]` lines: it caps them at
  5000, not 3000.
- **New `tests/test_ingest_render.py`:**
  - unit tests for each rule at its boundary (a 20-character result gets no line, 21
    does);
  - a golden render for each of jsonl, codex and messages, pinning the full content
    string, so their differences stay visible: jsonl puts a record's tool lines before
    its `[assistant]` line and messages after; all three join a message's text parts
    with a newline; jsonl drops `Stop hook feedback:` text; messages hashes
    a turn's own lines before `[context]` is added;
  - a source guard that fails when an `ingestion/*_client.py` file spells a line prefix
    in a string or f-string, or defines `_strip_nulls`. claude.ai's two lines are its
    one named exception.

**On each machine**

1. Nothing. Server-only refactor; no schema, plugin or config change. Stored span ids
   and content do not move.

**Proven**

- Output is byte-identical. A throwaway differential script ran the parsers from
  origin/main and from this branch over the same inputs and compared every Episode
  field: 17,507 local Claude Code transcripts (plus 3,502 each re-parsed whole and as a
  tail through the `/ingest` seam), 28 Codex rollouts (whole, pushed, tail), synthetic
  messages, ChatGPT, claude.ai and Cursor inputs built from 599 real turns, and the 174
  parser calls the old and new tests make. 25,543 cases, 33,663 episodes, 0 differences.
  The same run with origin/main on both sides also found 0, and a deliberately changed
  rule was caught.
- Full non-db suite: 3782 passed, 34 skipped, 2 xfailed. Ruff check and format are
  clean; mypy reports nothing in `ingestion/`; bandit finds no issues.

**Open**

- Three defects this change left alone were fixed the same day, in the parsers:
  - jsonl kept only the last text block of a record. It now joins every text block in
    order, and in a user record leaves out a block of Claude Code machinery;
  - a null tool name rendered as `[tool:None]` in jsonl and codex. It now falls back to
    `tool` (codex: `function` for a function call), as messages does;
  - jsonl rendered a whitespace-only tool detail. It now gets no line.
  - None of the three changes what any transcript on cortex renders to: Claude Code
    writes one block to a record.
- Left as it is: Cursor renders tool results as `[tool:name]` lines, never `[result]`.
- codex's "a group with only `[context]` is kept" cannot happen: every group after the
  first opens with a user message, which always renders a `[user]` line.

## 2026-11-12 — plugin-codex: the Codex hooks run on the client core (plugin 0.24.4)

**What changed**

- **`plugin-codex/scripts/common.py` is a thin layer over the core.**
  - It resolves with `Settings.load`, adding what only Codex has: the Claude plugin's
    saved options (ranked below `config.json`, as before), its pre-0.20 `device.json`,
    and the `http://localhost:8765` default.
  - `get_json`, `post_json` and `request_json` send through `Transport`, and so does the
    Stop hook's upload. The bearer goes only to the configured server's origin and the
    ingest URL's; before, the helpers sent it to any absolute URL.
  - Requests carry `User-Agent: synapse-client/<version> codex` (was
    `synapse-codex-plugin/0.1`, and `synapse-codex-hook/0.1` for uploads).
- **Session start prints the core's block**: token notice, upload warning, board,
  preferences, in that order (Codex printed preferences before the board).
  - A machine that is not enrolled and gets a 401 sees the core's "not enrolled"
    explainer; one served as restricted without a device token sees the restricted one.
  - The board and preferences are fetched side by side, each request at most 10 s, while
    the skills sync runs (at most `SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT`, default 12 s).
    15 s after the hook starts, it prints whatever has arrived. Codex kills it at 20 s.
- **The prompt hook prints the core's nudge**, and gains the timestamp line Claude Code
  already had (`SYNAPSE_PROMPT_TIMESTAMP=0` turns it off).
- **One recall-feedback nudge text, in the core** (`context.recall_feedback_nudge`).
  - Both hooks print it. It is a developer setting, off unless
    `SYNAPSE_RECALL_FEEDBACK_NUDGE=1`; Codex's hook used to be on unless `0`.
  - The Claude hook's output is unchanged, byte for byte (pinned). Codex keeps its own
    check on the tool name.
- **The Stop hook asks the core whether a session is private** (`private.is_private`).
  - A marker older than `SYNAPSE_PRIVATE_TTL_HOURS` (default 12) is removed and the hook
    uploads again; the server still drops sessions its private-mode row names.
  - A marker that cannot be read counts as private.
- `http.Transport` opens through the module's `urlopen`, so a test that fakes the network
  patches one name for the core and the hooks.

**On each machine**

1. `synapse update` once the server serves 0.24.4. No hook needs re-trusting in Codex:
   every hook path and command text is unchanged, and `install.py` writes the same blocks
   byte for byte.
2. What a Codex user sees change:
   - session-start context in the order above, with the enrollment explainers when they
     apply;
   - a timestamp line with each prompt;
   - session start never waits past 15 s;
   - a private marker older than 12 hours no longer holds uploads back on the machine;
   - one more notice case: a token in `~/.synapse/config.json` that the device token
     overrides;
   - the new User-Agent, in the server's logs.
3. The recall-feedback nudge is now off in Codex. To keep it, put
   `SYNAPSE_RECALL_FEEDBACK_NUDGE=1` in Codex's environment.

**Proven**

- Full non-db suite: 3781 passed, 34 skipped, 2 xfailed (3728 before, plus the 53 new
  tests). Ruff check and format are clean.
- New `tests/test_codex_on_core.py` (53 tests), against local fake servers:
  - common.py keeps no HTTP or token logic of its own;
  - the localhost default, and a device token kept from that default;
  - the bearer kept off another origin;
  - session start's order, explainers, kill switches, partial output on a 500, the 15 s
    deadline, and a hook against a hung server exiting on time;
  - the prompt nudge with and without the timestamp;
  - the feedback nudge's rule and bytes in both hooks;
  - the Stop hook's private check: fresh, stale, unreadable and non-plain ids.
- `install.py` writes byte-identical hook and MCP blocks for the checkout and the managed
  client (pinned).
- The old and new common.py were run over 630 combinations of environment, `config.json`,
  Claude settings and `device.json`. The URL, ingest URL, token, private dir and the two
  settings sampled (`SYNAPSE_SKILLS_SYNC`, `SYNAPSE_BOARD`) agree in all of them. The
  notice differs in 26, each by the core's added `config.json` line.

**Open**

- Not tried under a real Codex: hook trust after the update, how Codex shows the
  `systemMessage` warning, the 20 s kill, the skills sync's time on a large folder, and
  payloads Codex might send that the tests do not.
- Claude Code's hooks are next onto the core.


## 2026-11-12 — client: dead code out, and the hooks' copies of server rules pinned (plugin 0.24.3)

**What changed**

- **`scripts/synapse_ingest_hook.py` is gone.** It was the stale Stop hook outside the
  plugin, still carrying the defects listed in the 0.24.2 entry below. Nothing shipped
  or ran it: no Dockerfile, compose file, Makefile, CI job or `/install` bundle names
  it. ARCHITECTURE.md now describes the live hook, `plugin/scripts/ingest_hook.py`, in
  the overview, the ingest diagram, §4.2 and the repo tree, and names the Codex `Stop`
  hook where the same holds for it.
- **Code that only tests called is removed:**
  - `ingest_hook._select_tail`. The Stop path seeds its cursor inline in `_ship`.
  - `config.RECALL_URL` (and its `SYNAPSE_RECALL_URL` docstring line),
    `config.PROPOSALS_DIR` and `config.ensure_dirs`. A pasted `/recall` suffix is still
    stripped from the base URL.
- **`enroll.py` takes `BOOTSTRAP_CMD` and `PASTE_CMD` from the core**
  (`synapse_client.context`), reaching it the way `synapse_project.py` does. The two
  notices that use them are unchanged byte for byte.
- **New `tests/test_client_server_rule_parity.py`.** It pins the hooks' copies of server
  rules:
  - the machinery prefixes, for Claude Code and for Codex;
  - whether a record starts a turn, on a table of records for each;
  - the Hermes `<memory-context>` pattern.

**On each machine**

1. Nothing beyond the usual update: `synapse update` once the server serves 0.24.3. No
   behaviour changes on the client. On the server, a Claude Code transcript record with
   `"message": null` no longer fails its whole `/ingest` batch (below).

**Proven**

- Full non-db suite: 3728 passed, 34 skipped, 2 xfailed. Ruff check and format are clean.
- The new byte-for-byte tests of `bootstrap_hint()` and `restricted_block()` passed on
  the old definitions in `enroll.py`, and still pass after the switch.
- `harness.normalize_url` and the core's `settings.normalize_url` agree on every case in
  the parity table, and their patterns and flags are equal.
- With `--runxfail`, each of the two known differences fails at the hook's answer, not
  in the test's setup.
- The parity table found one server defect, fixed here: `JSONLParser` read a record's
  message with `record.get("message", {})`, which is `None` for an explicit null, and
  raised AttributeError. That returned 500 for the whole batch, and the hook never
  advances its cursor past a failed POST. It now reads a null message as an empty one,
  as the hook always did; the case is an ordinary test in the parity table.

**Open**

- The hooks and the server differ on two records. Each is an xfail in the parity
  tests; no code changed for them:
  - A sidechain user record with real text: `ingest_hook._is_turn_start` treats it as a
    turn start, but the server drops sidechain records before grouping.
  - A Codex user message whose only part is `output_text`: the server counts it as a
    turn, but the hook's `_is_user_boundary` reads only `input_text` and `text`.
- `harness.normalize_url` keeps its own copy of the core's rule. `synapse update` calls
  it before the installer runs. Importing the core at that point would leave the old
  client's `synapse_client` in `sys.modules`, and the update would build its harness
  registry from that one. Instead, a test pins the two copies equal.
- No test covers `_ship`'s stop-mode fallback: when the tail window holds no turn
  boundary, it seeds the cursor at offset 0.

## 2026-11-12 — client: six defect fixes from the client parity review (plugin 0.24.2)

**What changed**

- **The bearer stays on its own origin.** Every hook that sends a credential now opens
  through the core's `synapse_client.http.urlopen`: Claude Code's config helpers, the
  ingest hook, the login's machine-token call, and Codex's helpers and Stop hook. A
  redirect to another scheme, host or port drops the Authorization header; a redirect
  within the server keeps it. Before, urllib copied the header wherever the server
  pointed.
- **A session id must be a plain file name before a hook puts it in a path.** An id
  holding `/`, `\` or NUL, or one that is `.` or `..`, is not a plain file name. The
  ingest hook and Codex's Stop hook build no private-marker path from such an id (it
  reads as not private). Both private-mode toggles refuse it, and quote the id in the
  URL they send to the server. `config_review accept` refuses a file key that resolves
  outside the config directory, and writes and applies nothing.
- **A mistyped number in the environment no longer stops a hook.** The ingest hook,
  Codex's Stop hook, the remember spool and git_feeder fall back to their defaults.
  Before, `SYNAPSE_INGEST_TIMEOUT=thirty` killed the hook at import.
- **The remember spool keeps an intent refused with 401, 403, 408 or 429** and retries it
  at the next flush. Any other 4xx still drops it.
- **The core outbox (Hermes) drops a payload only on 400, 413 or 422.** Every other
  status, 404 included, keeps it.
  - The cap rose from 500 to 2000.
  - Payloads dropped for the cap or as refused are counted in `.dropped` beside the
    queue, and the session-start warning names the count.
  - A drain that leaves the queue empty clears the count, along with the recorded
    reason.
- **The `/install` bundle leaves out symlinks**, and anything under `__pycache__`,
  `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `.venv`, `venv`, `node_modules` or
  `.git`. This is the one server change. Other dot-directories, such as
  `.claude-plugin`, stay in.

**On each machine**

1. Nothing beyond the usual update: `synapse update` once the server serves 0.24.2
   (merge, then the image).

**Proven**

- Full non-db suite: 3640 passed, 34 skipped. Ruff check and format are clean.
- With each fix's source reverted at the final commit, its own tests fail:
  - redirect: 27 (`test_client_redirect_bearer.py`, `test_client_core_http.py`);
  - session id: 32 (`test_session_id_paths.py`);
  - env numbers: 55 (`test_hook_env_numbers.py`);
  - spool: 3 (`test_remember_spool.py`; 401 was already kept);
  - outbox: 18 (`test_client_core_outbox.py`);
  - bundle: 8 (`test_install_route.py`).
- The redirect tests drive all eight credential senders at two real stdlib servers on
  127.0.0.1, one redirecting to the other's port.

**Open**

- `scripts/synapse_ingest_hook.py`, the stale copy outside the plugin, still has the
  redirect, session-id and env-number defects. Deleting it is a separate change.
- Codex's Stop hook, with no `transcript_path` in the payload, still puts the session id
  into an `rglob` pattern (`_find_rollout`). It reads only what the payload could already
  name through `transcript_path`.
- On Windows, the plain-name rule (in the core and in the hook copies) allows a
  drive-relative `C:x`.
- The spool and the outbox keep different 4xx rules:
  - the spool drops every 4xx except 401, 403, 408 and 429;
  - the outbox drops only 400, 413 and 422.

## 2026-11-12 — span ids start with the platform; `/ingest` takes a neutral `messages` format (schema 068)

**What changed**

- **One name per client.** A turn's span id now starts with the platform that produced it.
  Three parsers used the transcript format instead, so one row said `claude_code` in
  `platform` and `jsonl:` in `span_id`. Renamed: `jsonl:` → `claude_code:`, `openai:` →
  `chatgpt:`, `cursor-sqlite:` → `cursor:`. `codex:` and `claude_ai:` already matched.
  `source` still names the transcript format.
- **Schema 068** renames the stored span ids and leaves a trigger
  (`episodes_canonical_span_id`) that rewrites the three old spellings on every write.
- **`/ingest` format `messages`**: a client that holds its conversation as chat messages
  posts them with its own `client` name (ARCHITECTURE §4.1). Nothing uses it yet; the
  Hermes provider will. A client may not take a name another parser stamps.

**Order of rollout: the database first, then the image**

1. Merge order: after schema 067. 068 stamps the database past 067, and the boot guard
   only compares the highest number, so 067 must already be applied.
2. Apply `schema/068_span_id_prefix_is_platform.sql` while the previous image is still
   serving. That image keeps writing `jsonl:<uuid>`; the trigger stores `claude_code:<uuid>`,
   and a turn that is already stored hits `episodes_span_id_idx`, which the route counts as
   a span collision and skips. The boot guard lets a build start on a database ahead of it.
3. Then merge, so watchtower ships the image. Not the other way round: the new image on a
   database without 068 would miss every stored `jsonl:` span and store re-sent turns a
   second time. 068 is therefore not marked optional, and the new image refuses to start
   without it.
4. `VACUUM (ANALYZE) episodes` afterwards. The UPDATE rewrites every renamed row (32k of
   54k on prod: 27,554 `jsonl:`, 3,944 `cursor-sqlite:`, 490 `openai:`), and each new row
   version is entered into the BM25 and HNSW indexes too.

If 068 stops with "stored under both the old and the new span id", a newer build already
ran against the database. Nothing was changed; delete the newer copy of each pair and run
the file again.

**What was proven**

- On a scratch ParadeDB built to 066 and seeded with every spelling a parser has written:
  068 renamed exactly the three old prefixes, left `codex:`, `claude_ai:`, the older
  `cursor:<hash>:<n>` rows, bare Logfire ids and NULLs alone, and a second run changed
  nothing.
- `tests/test_span_prefix_migration_db.py`: the trigger on insert and update; the
  both-spellings refusal; and the route end to end with the previous build's parser
  re-sending a stored turn (`ingested: 1, span_collisions: 1`, no second row).
- Prod, read-only: no `cursor-sqlite:` span collides with an existing `cursor:` one, and
  `episodes` and `episodes_quarantine` are the only tables with a span-id column.

**Still open**

- Not rehearsed on a copy of prod, so the UPDATE's run time there is an estimate.
- `ingestion/backfill.py` from a checkout older than this change raises on a turn that is
  already stored (it has no span-collision handler). It is a manual tool; run it from a
  current checkout.

## 2026-11-12 — guards: only a full-trust device takes a session back on the record; appends to one session take turns (plugin 0.24.1)

**What changed**

- **`DELETE /private-sessions/{session_id}` needs a full-trust device.** It removes the row
  that keeps an off-the-record session from being ingested, and it sat behind the ordinary
  client gate, so a restricted device or the root token could put back on the record a
  session another device had taken off it. It now sits behind the admin gate: a caller that
  clears the client gate only gets a 403 with a plain reason. `PUT` and `GET` are unchanged.
- **`private_mode.py off --forget` says what happened** (the Claude plugin and its Codex
  copy). A 403 says the delete must be done from a full-trust device. Any other failed
  delete reads the flag back first: still there after an HTTP error means the session stays
  off the record; still there after a transport error means the delete may still be
  applying, check `status`; gone means the forget worked; unreadable means unknown. Every
  failure also says the local marker was removed, which leaves the session where a plain
  `off` leaves it.
- **Appends to one session take turns.** `remember`, `/ingest` and the backfills each read
  the session's highest sequence number and then upserted at the next one, so two writers
  could pick the same number and the second replaced the first one's row.
  `EpisodeStore.append_episode` takes a per-session advisory lock and takes the number
  inside the `INSERT`; a row is never rewritten by another writer's append.
  `upsert_episode` stays for a deliberate rewrite of a known row.
- **A remembered item reads session context only from turns of its own project.** The
  names handed to the extraction model came from every turn of the session, so a name from
  another project could end up in a fact that cites only this one's episode.

**On each machine**

1. Nothing to migrate and no setting to change.
2. `synapse update` for the 0.24.1 `private_mode.py` wording (older clients keep working;
   they print the old lines). The server change takes effect with the image.
3. To take a session back on the record, run `private_mode.py off --forget` on a full-trust
   device.

**Proven**

- A restricted device's `DELETE` through the routes the server itself registers gets a 403
  and its `PUT` lands; a full-trust device's `DELETE` removes the row.
- A writer that loses its number appends after the winner, and many appends at once get
  distinct numbers and keep their own rows; the tests fail
  if the session lock is released before the `INSERT` runs or a writer goes back to
  upserting.

## 2026-11-12 — client: Hermes gets Synapse as its memory provider; the CLI walks one harness list (plugin 0.24.0)

**What changed**

- **A Hermes adapter on the client core** (`plugin/synapse_client/harnesses/hermes/`): a
  `MemoryProvider` with the board and preferences in the system prompt, the recall nudge
  each turn, the six memory tools proxied to MCP `tools/call`, and every finished turn
  uploaded to `/ingest` in the `messages` format (client `hermes`, source `hermes-sync`).
  Failed uploads wait in `$HERMES_HOME/synapse/outbox/`. No server change: the `messages`
  format landed in #251.
- **Local sessions only.** The provider is active for a primary session on the Hermes
  CLI, TUI, Desktop or ACP with no chat identity. Cron jobs get the tools only; gateway
  (chat-platform) sessions, the API server and subagents get nothing, and a tool call
  there returns an error. The rule and the Hermes source lines it rests on:
  `docs/client-harnesses.md`.
- **`synapse install hermes [--hermes-home PATH] [--replace]`** builds
  `$HERMES_HOME/plugins/synapse/` (a copy of the core plus `harnesses/hermes/`, swapped in
  whole) and selects `memory.provider: synapse`. Another external provider already selected
  is left alone unless `--replace`. The `hermes` program is only run on a home that already
  has a `config.yaml`; a home without one gets a new two-line file.
- **The CLI walks one list** (`harness.harnesses()`): Claude Code and Codex wrapped over
  their old functions, then the core's registry. `synapse login` and `synapse install`
  wire Hermes when they find it, `synapse update` rebuilds the Hermes folder when it is
  wired, and the new `synapse status` prints a line per harness. Claude Code and Codex
  behave and print exactly as before.

**On each machine**

1. `synapse update` (the server must serve 0.24.0 first: merge, then the image).
2. On a machine with Hermes: `synapse login` (re-wires, no new sign-in) or
   `synapse install hermes`; for a Hermes profile, add `--hermes-home <profile home>`.
   Restart Hermes. `synapse status` should print `Hermes: wired`.
3. If the report says another provider was left selected, decide: `--replace`, or keep it
   (Hermes runs one external provider at a time).

**Proven**

- Full non-db suite: 3457 passed, 32 skipped.
- `tests/test_hermes_integration.py` against a real Hermes checkout
  (`SYNAPSE_HERMES_CHECKOUT=~/workspace/hermes-agent`): Hermes's own loader and
  `MemoryManager` through initialize, system prompt, prefetch, a tool call, two synced
  turns, pre-compress, session end and shutdown, with plugins in-process and with
  `plugins.isolation: host`. The fake server's bodies, parsed by the route's own
  `MessagesParser`, are four turns each stored once; no connection left the loopback
  address; the temp Hermes home stayed at about 145 KB.

**Open**

- Not ported to Hermes: the remember spool, the recall-feedback nudge, a private-mode
  command, skills and config sync, a `state.db` history importer, and any gateway session.
- Claude Code's hooks and Codex's still carry their own config and HTTP code; moving them
  onto the core is next, Codex first.

## 2026-11-11 — one memory per owner: the personal/technical split is gone (schema 067)

**What changed**

- **One knowledge graph per owner.** Extraction no longer sorts entities and facts into a
  `personal` and a `technical` group, so a fact between an entity of each kind is written
  instead of dropped, and matching and contradiction search see the whole graph. What a
  restricted device may be served is decided by each fact's audience label (schema 066) and
  the provenance of its source episodes, never by a group. `domain` survives only as a tag on
  timeline events.
- **`recall` takes no `group_id`.** A client that still sends one is accepted and the
  argument is ignored (the server writes one log line per call that carries it); `tools/list`
  no longer shows it. `POST /recall` ignores the body field too, without a log line.
- **`SYNAPSE_PERSONAL_SCOPE` and `SYNAPSE_PERSONAL_PROJECTS` are gone.** Remove them from your
  `.env`; nothing reads them. `SYNAPSE_OWNER_NAME` and `SYNAPSE_OWNER_ALIASES` stay.
- **A restricted read needs the label.** Without schema 066 a restricted device is served no
  facts; the provenance-only fallback is gone. The boot guard keeps a service from starting
  on a database without 066 unless it is switched off (`SYNAPSE_SCHEMA_CHECK=0`) or could not
  reach the database at that moment (it logs a warning and does not look again). A new fact
  is written unlabelled, so a restricted device is not served it until the labeller has
  marked it `work-safe`.
- **Schema 067** is required: it merges the existing `personal` rows into `technical`, folds a
  personal entity into its same-name technical twin, builds three owner-only indexes, gives
  `group_id` a `'technical'` default on the four tables that carry it, drops the 066 insert
  trigger and stamps 067. `scripts/pile_merge.py` previews, health-checks and undoes it.

**Order of rollout**

1. Stop the pollers (`docker compose stop poller poller-drain`).
2. `python scripts/pile_merge.py --preview` (read-only; counts and uuids, no text). Read the
   numbers.
3. Apply 067 by itself, before the release: `psql "$SYNAPSE_DB_URL" -v ON_ERROR_STOP=1 <
   schema/067_kg_one_memory.sql`. It runs two transactions of its own, so never inside
   `BEGIN` and never with `-1`. A re-run is safe and does nothing when nothing is left.
4. `python scripts/pile_merge.py --health --just-merged` must exit 0 (pollers still stopped,
   no dashboard deletes since step 3). The flag makes two more checks fail the run: no
   survivor has left `technical`, and the fact count equals the one before the merge. Without
   the flag (later, once the graph has moved on) those two sit in the `reported_not_checked`
   block, for reading; they never fail the run.
5. Deploy the release, then start the pollers. If a poller ran the old image after step 3,
   it may have written a personal row, or a folded uuid again: apply 067 once more, check
   with `--health`, and restart the pollers (a poller keeps a deleted uuid in its name index
   for up to 15 minutes).

**Rolling back**

- Code only (behaviour, latency, a crash): revert the release. The data stays merged and the
  older code runs on it (a database ahead of a build only logs a warning), but that code
  writes the personal group again, and without the insert trigger those facts arrive
  unlabelled, so a restricted device is not served them. Re-deploying the new release and
  re-running 067 folds them back.
- Data (a bad fold): revert the code first, stop the pollers, then `python
  scripts/pile_merge.py --undo` (a dry run) and `--undo --apply`. It re-creates the folded
  entities from their before-images, restores the groups and the 066 trigger, and stamps 066
  once every run is undone; a row changed since the merge is reported as drift and left alone.
- Restores that write `group_id`, `src_uuid` or `tgt_uuid` from an older backup must not run
  after the merge unless it is undone first: `repair_numbered_entity_magnets.py --restore`
  of a run made before it, and a restore from the tz-fix backup table.

**Still open**: a later migration drops the `group_id` columns and the `kg_pile_merge_067_*`
ledger, once nobody needs the undo.

## 2026-11-07 — dream→config lane: identity by meaning, a coverage gate, a queue that tends itself (plugin 0.23.1, schema 065)

**What changed**

- **The lane never produced a proposal** because two corrections were "the same rule" only
  at 0.5 token overlap, and the best cross-session pair scored 0.294: 208 rows, all
  `observe`, none with two sessions, and a long session re-scanned whole each night added
  near-duplicates. Identity is now the cosine of the rules' embeddings against the row's
  founding embedding (0.82 across sessions, 0.75 for a session the row already counts;
  `CONFIG_IDENTITY_COSINE`, `CONFIG_IDENTITY_COSINE_SAME_SESSION`).
- **Coverage gate**: before a row is proposed, one judge call asks whether the user's own
  global rule files (`config_registry`, scope `global`) already cover it. Covered rows go to
  `rejected` / `auto:covered`. Proposed rows are re-checked when the files change.
- **A queue that tends itself** (the skills lane's treatment): a stored `recommendation`
  per proposal (it rejects a rule whose quotes ask for opposite things, which embeddings
  cannot see), a 21-day expiry (`rejected` / `expired`, reopened by a new session's
  sighting), and a plain-text report stored in
  `skills_lane.skill_scan_cursor.config.config_last_report` (`{at, text}`) next to the
  skills lane's own `last_report`, also posted to `SKILLS_DISCORD_WEBHOOK` when set.
  Proposals stay ask-only: nothing is applied without `/synapse:config-review accept`.
- A rule rejected in the last 30 days, or accepted or applied, absorbs re-sightings instead
  of opening a new proposal (closes the config-lane cooldown gap in
  `docs/dashboard-contract.md`).
- `/config/proposals` and `config_review.py list|show` carry the recommendation and the
  days pending (plugin 0.23.1).

**Order of rollout**

1. Merge → the dream image. Until 065 is applied the lane probes, logs one line
   (`schema 065 not applied; the lane idles`) and does nothing; its scan watermark does not
   move, so the first run after 065 starts where the lane stopped. That run still scans at
   most `CONFIG_LANE_LIMIT` x 4 sessions (120) before it moves the watermark, as the lane
   always has, so apply 065 soon after the image to keep the backlog under that.
2. Apply `schema/065_config_lane_identity.sql` (marked optional; four nullable columns,
   guarded, one UPDATE that stamps `proposed_at` for rows already proposed). Apply 063 and
   064 first if they have merged, so the version stamp stays in order.
3. Re-cluster the existing rows (needs no schema change, so it can run before or after 2):
   `scripts/recluster_config_proposals.py` is a read-only dry run;
   `--apply` folds the clusters, runs the gate and copies every touched row to
   `config_lane.config_proposals_bak_<YYYYMMDD>` first. `--restore YYYYMMDD --apply` undoes
   it. Credentials via the environment (`SYNAPSE_DB_URL`, `VOYAGE_API_KEY`); the judge is the
   local `claude` CLI, called stripped.
4. Morning briefing: a separate patch to the operator's briefing script adds a "Config
   proposals" block (it reads `config_proposals` directly, with a pre-065 fallback); it is
   not part of this change.
5. `synapse update` on each machine for the 0.23.1 review CLI (older clients keep working;
   they just don't print the recommendation).

**Note for restricted devices**: merged evidence spans sessions, so a restricted caller
(`_OWN_PROPOSAL` needs every evidence session to be its own) sees fewer proposals; a full-trust
caller is unaffected.

## 2026-11-06 — dream→skills autopilot: the lane decides the routine proposals itself (plugin 0.23.0, schema 064)

**What changed**

- **Applied without a person** when a reviewer (the `claude` CLI, Opus) accepts and an
  independent verifier (a non-Claude model, evidence in both orders) passes it: a new skill
  that met the recurrence gate, or a description-only change to an unpinned skill. At most 3
  a night, 1 new skill a night, 3 a week. **Auto-rejected** (30-day cooldown, no grounded -3)
  when the reviewer rejects and the verifier agrees. Body edits, merges, pinned skills and
  any disagreement stay proposed, with the reviewer's recommendation.
- One write path: `skill_sync_routes.apply_candidate(actor)` serves a person's accept and the
  autopilot; `skill_history` keeps the replaced body, two-way sync delivers the change. The
  constraint is now "a grounded weight, or `applied_by='auto'` with a recorded review"; the
  auto path never writes a grounded entry. Every decision is a row in
  `skills_lane.skill_auto_reviews` (audit trail and judge-once memo).
- **Undo**: `skill_review.py revert N` (retires an auto-created skill; puts a description back
  and keeps later hand edits). Automatic for 14 days: a dismissed or deviating fire after the
  change, the owner editing the description away or retiring the new skill. A local edit a sync
  pulled over the change is kept: the description is merged onto it, or the local body is
  restored and the row returns to the queue. Two rollbacks in 14 days pause the autopilot
  (`skill_review.py autopilot status|resume`).
- **Report**: auto-applied lines first, each ending `undo: revert N`; every built report is
  also stored in `skill_scan_cursor.config.last_report` (`{at, text}`) for whoever delivers it
  (`skill_review.py report`).
- **The reviewer call is stripped** (`dream/skills/claude_cli.py`): no tools, settings, MCP,
  hooks, memory files or side calls, so a call carries the prompt plus about 1.6k tokens of
  preamble instead of 16k to 76k. The calibration script makes the same call, and so does
  every judge and drafting call of the lane (`claude_cli.call`, behind `skill_measure.run_judge`
  and `skill_derive._draft_call`).

**Order of rollout**

1. Merge → image. Until 064 is applied the stage logs one line and skips; a person's accept
   works as before.
2. Apply `schema/064_skills_autopilot.sql` (marked optional; `scripts/apply_schema.sh`).
3. The reviewer needs a logged-in `claude` CLI. If the dream container has none, run
   `scripts/skills_autopilot.py` on a host that does, after the nightly dream (drafting
   happens there), through `scripts/skills_autopilot_cron.sh`, which reads the database URL
   and the verifier's key from env files so no password sits in the crontab:
   `<min> <hour> * * * <checkout>/scripts/skills_autopilot_cron.sh <file with SYNAPSE_DB_URL>
   <file with OPENROUTER_API_KEY> [--notify-cmd '<command that posts its last argument>']
   >> <log> 2>&1`. The nightly lane then reconciles and reports only. Set
   `SKILLS_AUTOREVIEW_REVIEW=1` instead when the container has a login.
4. `SKILLS_AUTOREVIEW` defaults to `live`; `shadow` records what it would do, `off` skips.
5. `synapse update` on each machine for the 0.23.0 review CLI (`revert`, `report`, `autopilot`).

**Still open**: see the PR's Not done list.

## 2026-11-06 — repair numbered-entity magnets (script only, no schema or serving change)

**What changed**

- **`scripts/repair_numbered_entity_magnets.py`** repairs graphs written before the
  numbered-name guard (#230). For every edge, live or retired, with an endpoint whose whole
  name is an identifier (`PR #N`, `draft PR #N`, `pull request #N`, `issue #N`, `task #N`,
  a ticket key `ABC-123`), it reads the fact for identifiers of the same kind. A fact that
  names exactly one, other than the endpoint's own number, moves that endpoint to the
  entity named for it for the same owner (an existing one spelled `PR #N`, `PR N` or
  `pull request N` is reused; created if missing, embedded as an entity, type and project
  copied). Facts naming their own number, none, several or a range stay; so
  does a move that would join an edge to itself. Entities that lose an edge get an empty
  summary, refilled by the next mention. Fact text, embeddings, `episodes` and
  `source_projects` are never touched.
- **Wrong supersessions are listed, not auto-fixed.** A retired edge whose successor shares
  an identifier entity with it but names a disjoint set of identifiers is a candidate. Many
  candidates are real corrections whose identifier is the changed value, so only uuids an
  operator reviewed and listed in `--unretire-allow FILE` are un-retired.

**How to run** (DSN from `--db-url` or `$SYNAPSE_DB_URL`; the plan, the allow file and the
samples hold private text: keep them outside the repository)

1. Dry run (read-only connection): `scripts/repair_numbered_entity_magnets.py --samples 20
   --plan-out <outside>/plan.json`. It prints counts per family, the health numbers now and
   as they will be after the plan, and sample moves.
2. Review `unretire_candidates` in the plan; write the uuids to un-retire, one per line, to
   an allow file. Leave out a candidate whose old edge does not move (it would come back
   live on an entity named for another number).
3. Benchmark before, then `--apply --unretire-allow <file> --date <YYYYMMDD>`. The plan is
   recomputed at apply time and every write is conditional on the row still holding the
   planned value. Every touched row (each moved or un-retired edge, each entity that loses
   an edge) is first copied to `kg_magnet_repair_<YYYYMMDD>_{relationships,entities,created}`.
4. `--metrics` prints the audit's two numbers to verify.
5. Undo: `--restore <YYYYMMDD>` (dry run) then `--restore <YYYYMMDD> --apply`. It reverts
   what the run changed and keeps what was written since: an entity drained of every edge
   and then deleted by the daily orphan sweep is re-created from its backup, endpoints go
   back, an un-retired edge is retired again unless something retired it since, a cleared
   summary comes back unless a mention rewrote it, and created entities that new facts
   have attached to are kept and counted.

No restart is needed: the extractor's dedup index re-reads entities within its TTL
(`SYNAPSE_DEDUP_CACHE_TTL_SECONDS`, 900 s), and its exact-name lookup falls through to
the database.

**Proven**

- Rules: 46 unit tests; 11 database tests on a scratch database (dry run and metrics read
  through a read-only connection and write nothing, apply moves/creates/clears/backs up,
  provenance unchanged, a second apply is a no-op, a row changed after the read is
  skipped, restore returns every row, keeps a retirement or summary written after the run,
  re-creates a magnet the orphan sweep deleted, metrics before/after).
- Read-only dry run on prod, 2026-11-06: 2,961 identifier endpoints examined; 1,175
  endpoint moves (966 live, 209 retired) out of 83 entities, 10 of which lose every edge;
  62 existing targets reused, 213 created; 10 supersession candidates, 1 un-retired after
  review (the other 9 are real updates, stale on their own, or stay on the magnet). The
  audit's numbers as planned: live edges on `PR #N` entities lacking their own number
  72.4% -> 13.8% (whole name) and 75.5% -> 26.4% (name contains `PR #N`); disjoint-ID
  supersessions 29 -> 28.
- Independent review read 51 sampled moves (all families, live and retired) and 33 sampled
  stays against their fact text: no wrong move, no wrong stay.

**Open**

- The prod `--apply` and the after-benchmark: run by an operator (this session's
  permission policy refused the production write).
- Left alone by design: facts naming no or several identifiers (the residual 13.8%),
  entities whose name decorates the identifier ("widget refactor (PR #12)": 56 edges there
  would match the rule), bare `#N` entities, and the same number in two repositories (one
  target, as the guard itself cannot tell them apart: of 20 targets fed by facts carrying
  a repository URL, 1 is fed by two repositories).
- A plan's "PR 1" / "PR 3" (a step in a planned series, not a pull request number) moves
  onto the real early `PR #1` / `PR #3` entities: 2 edges on prod, no rule can tell them
  apart.
- Between the plan and the write, live extraction can create the same target the run
  creates (seconds wide): the result is a duplicate entity, not a wrong edge.
- For up to the dedup TTL after the apply, a mention of a magnet can write its old summary
  back from the extractor's in-memory index (`NodeDeduper.summary_of`, longer wins).
  Recall no longer serves entity summaries, so this is cosmetic.

## 2026-11-06 — dream→skills lane: live detectors, a real recurrence gate, a queue that tends itself (plugin 0.22.1, schema 062)

**What changed**

- **post_fire and dismissals see the user again.** A Skill fire is stored as the call
  (seq N) and the loader's "Base directory for this skill: .../<skill>" banner + body
  (seq N+1). Both detectors read N+1 as the next fire / the user's reply, so since 07-18
  every post-fire window was empty and nothing was ever marked dismissed.
  `skill_db_source.is_loader_episode` skips it; post_fire counts `empty_window` on its own.
- **Derive proposes on real recurrence**: 2 distinct sessions started on 2 distinct days
  (salience and score paths unchanged). Scan nights no longer count, and a judge signal
  votes once per session. Session days come from episodes (min created_at) when an entry
  has none, and a nightly re-gate applies this to open rows.
- **Derive identity** also matches rejected rows (cooldown and -3 kept; while a person's
  reject stands, only the score path can re-propose one) and rows retired in the last 60
  days (they reopen instead of starting over).
- **Under-trigger judges each session against the skills it was served** (global +
  `project:<label>` by `_evidence_scope`; a multi-project session is global).
- **The review queue looks after itself** (`dream/skills/review_queue.py`, after drafting):
  undeliverable retunes are auto-rejected as `auto:<code>` (no_target, retired, renamed,
  scope_mismatch, no_change, covered) with the 30-day cooldown and **no** -3; proposals
  unanswered for 21 days retire as `expired`; each draft gets one stored recommendation
  (accept/reject + a sentence) from the lane's judge. Nothing is auto-applied.
- **Report** (replaces the new-ids-only digest): plain text, verdict first, under 1500
  chars, sent when tonight added a proposal, auto-rejected or expired one, else once a week.
  Still stdout when `SKILLS_DISCORD_WEBHOOK` is unset.
- **dream_runs**: the lane's counts land in `counts` as `skills_*`; `proposals_raised`
  counts `proposed_at` (a row that went proposed after sitting in observe now counts).
- **Review CLI** (plugin 0.22.1): `skill_review.py list` shows the recommendation and days
  pending; `skill_review.py revive <id>` puts a rejected/expired/retired proposal back.

**Order of rollout**

1. Merge → image (watchtower). The lane runs without 062: expiry, recommendations and
   proposed_at stamping skip with one log line, proposals_raised keeps counting new rows.
2. Apply `schema/062_skill_review_queue.sql` (marked optional; `scripts/apply_schema.sh`).
   Its backfill sets proposed_at = updated_at on rows already proposed, capped at 7 days
   ago, so **#142** (bug-hunt extend, proposed 10-11) gets a recommendation and two weeks
   in the queue instead of expiring on the first night.
3. Optional: set `SKILLS_DISCORD_WEBHOOK` in the docker VM's synapse `.env` so the report
   reaches Discord instead of the container log.
4. Optional: apply `/home/neuron/workspace/synapse-salvage/morning-briefing-skills.patch`
   to `~/scripts/morning-briefing.py` (verdict + days pending per item; falls back before 062).
5. `synapse update` on each machine for the 0.22.1 review CLI (older clients keep working;
   they just don't show the new fields or have `revive`).

**Proven** (read-only replays of prod, 2026-11-06)

- post_fire over the 61 registered-skill fires of the last 60 days: 61/61 empty windows
  before; after, 54 windows, 5 session-tail fires, 2 genuinely empty (the loader itself
  chains into another skill). Dismissal over all 102 fires: the check read the loader
  banner 94 times and a skill body 8 times; after, 67 short real turns, 13 long ones, 9
  next-skill banners (0 dismissals in the window: no reply was a short negation).
- The gate over the 84 gap_scan derive rows: 9 all-time, 2 opening in the last 60 days
  (#153 job-application-tailoring 10-23 + 10-28, #157 android-tv-adb-debug 10-29 + 11-02),
  matching the audit verifier; none of the six single-session re-scans passes.
- First night on a scratch copy of prod's open rows: re-gate raises #153 and #157; #76
  (docx widen) auto-rejects as scope_mismatch, keeping its 2 transcribe_ai sessions; #142
  would have expired under an uncapped backfill (the shipped backfill gives it two weeks).

**Open**

- Embedding identity for derive (the audit's 0.78 cosine leg) is not in this change; the
  signature check still misses the cited repeat pairs (sigJ 0.07-0.35).
- Whether the lane should keep excluding transcribe_ai (it does not by default today) is
  Kyle's call.
- Auto-apply of any skill change is a separate policy decision; the 023 CHECK is unchanged.

## 2026-10-12 — skills scoped by trust and project (plugin 0.22.0, no schema change)

**What changed**

- **Who is served which skills** (`mcp_server/skill_sync_routes.py::SkillAccess`, the board's
  trust rule applied to skills): full trust reads and writes every scope; a restricted
  device is served only `project:<p>` skills whose project is in its allowlist (any case)
  and may publish only there; a caller with no device credential (the root token included)
  is served nothing. This covers `/skills/list`, `/skills/fetch`, `/skills/publish`,
  `/skills/overwrite` and the `skill://` MCP resources (`PgSkillsProvider`). Retire,
  rescope and the review lane are full-trust only (403). Before this, every approved device
  was served every global skill (and could fetch any skill by name, list any project's
  skills, and publish global skills that fanned out to full-trust machines).
- **One scope per skill, moved explicitly**: a publish from another scope's folder is refused
  (`scoped`) instead of moving the skill, so an old copy can no longer undo a move.
  `synapse skill scope <name> global|project:<label>` (`POST /skills/rescope`, full trust)
  moves it, logged in `skill_history` as `rescoped`.
- **Withdrawn skills leave like retired ones, through one path**: the client sends the names
  it holds (`held`); the listing answers `withdrawn` for those still active but no longer
  served there (moved, or not this device's to read). The client's `_settle_unserved` (the
  retire path from #222/#223, generalised) deletes a copy that is exactly a version Synapse
  delivered (current or recent past bodies, `content_hashes`, or a version a pull wrote into
  that folder: the new `pulled-<folder>.json` record), never pushes it back, and never
  deletes a folder git tracks or one linked in. `publish: false` keeps a restricted
  device's own global-folder skills local instead of being refused every session.
- **Codex project skills**: the Codex session hook passes `--cwd`; `project:<label>` syncs
  into `<project root>/.agents/skills` (Codex reads it from any subfolder).
- **`.git/info/exclude`**: every project skill folder a sync pulls into a git repository is
  listed in that repository's `info/exclude` (a worktree's go to the main repo's), never
  `.gitignore`. A project skill folder the repository tracks is never pulled over. Empty
  skill folders are no longer created.
- **A pull cut short is safe**: SKILL.md is written last (whole, already carrying the
  server's time), so a sync killed mid-pull (Codex's session-start budget) pulls again
  next time instead of pushing the half-pulled copy over the server's.

**Order of rollout (matters)**

1. Merge → image (server first; old clients keep working: they ignore `publish`/`withdrawn`,
   their refused pushes write nothing).
2. From a full-trust machine, move the work skills the work laptop needs into its project
   (Kyle to confirm which project each belongs to), e.g.
   `synapse skill scope <skill> project:<label>`. A skill whose SKILL.md hard-codes
   its global install path needs that path made project-relative first.
3. Update the work laptop's client (`synapse update`). At its next session start its
   exact delivered copies of global skills are removed, and the moved skills arrive in
   the project folder (and are excluded from that repository).

**Proven**

- Reproduced first on `main` with the real routes, real device credentials and the real
  sync engine (local ParadeDB): a restricted device was served all global skills, could
  fetch any skill and list every `skill://` resource, published its own skills as global,
  and a re-scope was undone at the next sync of a full-trust machine holding the old copy.
  After: none of that.
- On a local replica of prod's registry (names/scopes/status, read-only) with the work
  laptop's real 13-entry allowlist: `main` serves it 30 global skills, a `transcribe_ai`
  project skill, 37 `skill://` resources and any skill by name; this branch serves exactly
  the 6 skills of its allowlisted project (in either casing) and returns its 30 held global
  skills as `withdrawn`.

**Open**

- Kyle: confirm the project for each of the three work skills the work laptop needs, then
  run the three `synapse skill scope` commands (not run here).
- Copies of global skills an older client delivered whose bundled files changed on the
  server since are not recognised (no file history); delete those by hand if any remain.
- The same before/after against prod itself needs a read-only run Kyle approves.

## 2026-10-12 — one project label everywhere (plugin 0.21.0, no schema change)

**What changed**

- **One resolver** (`plugin/scripts/synapse_project.py`, loaded by the Codex plugin too):
  a `.synapse-project` marker file (first non-comment line; searched from the working
  directory up to the git toplevel, and in a linked worktree also at the main checkout's
  root; UTF-8 with or without BOM, or UTF-16 from PowerShell 5.1), else the git toplevel's
  name (a linked worktree, Claude Code's `.claude/worktrees/<id>` included, is its main
  checkout; a bare repo's worktree is the repo's name), else the folder's name. Both
  separators, case kept. A folder that is gone (a removed worktree swept by catch-up) takes
  the label of the repository its nearest surviving parent is in. No subprocess.
- **Every client uses it**: the Claude ingest hook (Stop, catch-up, `synapse-import`) and
  the Codex Stop hook now send `"project"` with every `/ingest` body; the board hook and
  Codex session start ask `/context` for it; skills sync, skill review, config sync and the
  git feeder label with it. Claude project skills now sync into `.claude/skills` at the
  project ROOT (Claude Code reads it from any subfolder), not into each subfolder.
- **Server**: `ingestion/project_label.py` is the parsers' fallback for bodies without a
  label (both parsers used to split on "/" only, so a Windows cwd became the whole path);
  `/ingest` cuts a client-sent label to a bare name (a non-string `project` used to 500).
  A restricted allowlist matches case-insensitively (`resolve_caller` widens it to every
  stored casing via a skip scan on the `project` index, cached 60 s); skill scopes match
  case-insensitively. `ingestion/project_id.py` (dead) and `MEMORY_PROJECT` are gone.
- **`scripts/relabel_projects.py`**: maps stored labels to canonical names and rewrites
  `episodes.project` (the 056 trigger keeps KG `source_projects` in step), every other
  `project` column, `surfaces.allowed_projects`, and `project:` scopes in the skill
  registry and config lane. `--dry-run` (or `--inventory-sql` + `--dry-run --inventory`,
  which needs no DSN) writes nothing; `--plan-out` saves its plan and `--apply --plan`
  executes exactly that plan, never a fresh one. An allowlist entry whose new name would
  reach another folder's sessions is kept and flagged. Keep the alias file and the plan
  outside the repo.

**On each machine**: `synapse update` (or the plugin update) to 0.21.0. Optional: put a
`.synapse-project` file in a repository whose folder name differs between machines.

**Server**: before deploying, read the dry run's "case-insensitive matching" section: the
server matches a restricted allowlist case-insensitively from its deploy on, and an old
subfolder entry (`dev`, `src`) can match another project's `Dev`. If it lists one, run the
relabel first or drop that entry. Then merge → image, and only after Kyle approves the
dry-run table: `scripts/relabel_projects.py --apply --plan <the dry run's plan file>`.
Until the relabel runs, old labels stay as they are (new sessions arrive under canonical
names; a replay of an old session is still recognised by its folder).

**Proven**

- Real failures reproduced first, then fixed: the server parsers labelled
  a work project's `C:\Users\…\<repo>` (and a lowercase-drive variant) with the whole path, from
  both harnesses; main's real hooks, run as subprocesses against a local server in a real
  repo, labelled a session in `MyRepo/tickets/lab` as `lab`, a worktree as `agent-x`, sent no
  `project` from either ingest hook, and synced `project:lab`. After: all five hooks send
  `MyRepo`.
- Prod (read-only, 2026-10-12): 189 label/source rows; 1,858 episodes under Windows paths;
  the work laptop's allowlist has 13 entries for its few projects. The mapping table names
  private projects and paths, so it goes to Kyle directly and is not committed.
- Tests: resolver on real git layouts (worktrees, bare, submodule, markers), hook wiring with
  real subprocesses, case-insensitive allowlists end to end on ParadeDB, relabel planner +
  apply/idempotency on ParadeDB.

**Open**

- Kyle approves the relabel mapping; then `--apply` on prod.
- Windows not run on a real host (paths exercised as strings; the resolver's filesystem
  walk is OS-neutral).

## 2026-10-12 — `synapse login` is the single setup step (#222, plugin 0.20.0, schema 060)

**What changed**

- **Harness-independent config: `~/.synapse/config.json`** (`plugin/scripts/synapse_home.py`):
  server URL, token, machine role, `skills_sync`, `config_sync`, `config_paths`, and the
  enrollment record (formerly `~/.local/share/synapse-skills/device.json`). Written 0600,
  atomically. Both `plugin/scripts/config.py` and `plugin-codex/scripts/common.py` resolve
  env → `config.json` → the Claude plugin's legacy options (`CLAUDE_PLUGIN_OPTION_*`, then
  `settings.json`) → default; the Codex Stop hook's private copy of that resolution is gone.
  `SYNAPSE_HOME` overrides `~/.synapse`.
- **Claude plugin MCP auth via a headers helper**: `plugin.json` gains
  `"headersHelper": "python3 scripts/mcp_headers.py || python … || py -3 …"` (Windows
  often has no real `python3`); Codex uses the same script
  (`plugin/scripts/mcp_headers.py`; `plugin-codex/scripts/mcp_headers.py` is now a shim
  for old `config.toml` lines). The static `Bearer ${user_config.SYNAPSE_INGEST_TOKEN}`
  header stays as the fallback. `userConfig` now asks only for `SYNAPSE_URL` and the
  optional token. Correction to the issue's premise, verified on Claude Code 2.1.296: a
  *plugin's* helper runs with no trust prompt (also in `claude -p`); only project/local
  `.mcp.json` helpers are trust-gated. So the helper is the primary path everywhere and the
  static header covers helper failure.
- **`synapse` CLI** (`plugin/scripts/synapse_cli.py`, `harness.py`): `login` (sign in /
  enroll, ask role + "Sync skills to this machine?" once, save to `config.json`, wire every
  harness found), `config`, `install claude|codex`, `update`, `skill retire`, plus
  `devices`/`private`/`import` passthroughs. `synapse-login` is now an alias of
  `synapse login`. Claude Code is wired through `claude plugin marketplace add` +
  `claude plugin install synapse@synapse --config SYNAPSE_URL=<url>` (settings-key
  bootstrap when there is no `claude` CLI); Codex through `plugin-codex/install.py`.
  First `synapse login` on an old install copies the old install-prompt answers (and a
  token Claude kept in `.credentials.json`) into `config.json`, updates a cached older
  plugin, and drops the four options the plugin no longer declares once 0.20 is
  installed. Moving to another server needs `--reenroll` (or a yes), also on a machine
  set up with a pasted token; the old server's token is never sent to the new one, and the
  new URL is saved only after it enrolls. `synapse config url`, `install` and `update`
  refuse another origin while the machine holds a credential, including one Claude Code
  keeps for the plugin (`.credentials.json`, the macOS keychain); `--same-server` is for
  a server's new address. The hooks and the headers helper send a device token only to
  the server that minted it (the record's `server`) and to wherever a plugin config that
  holds that token points (its MCP header sends it there anyway), even when `config.json`
  is edited by hand; when they withhold it, session start says so in one line. A pre-0.20
  plaintext token copy in the plugin's options follows a new token only while the plugin
  points at that token's server (a move re-points the plugin first; with `--no-install`
  the copy is removed).
- **`GET /install`** (`mcp_server/install_routes.py`, `install_script.py`): an
  unauthenticated stdlib installer with this server's own client inside. It unpacks into
  `~/.synapse/lib/<version>/`, points `lib/CURRENT` at it, writes `~/.synapse/bin/synapse`
  (+ `.cmd`) and the fixed launcher `~/.synapse/bin/synapse_run.py`, records the server URL,
  and adds `~/.synapse/bin` to the user PATH on Windows (prints the line on POSIX;
  the login-bash startup file on macOS). `synapse.cmd` is ASCII only (cmd reads batch
  files in the console code page): the runner is `%~dp0synapse_run.py`, a profile- or
  Program Files-relative interpreter is written as `%LOCALAPPDATA%\…`/`%ProgramW6432%\…`,
  and a missing one falls back to the `python` on PATH with the current directory
  excluded. The call jumps to the exit on its own line, so `synapse update` rewriting the
  running shim is harmless. All `.cmd` shims now return Python's exit code. The
  Dockerfile now copies `plugin/` and `plugin-codex/` into the image.
- **Codex installer is Windows-safe** (`plugin-codex/install.py`): every command is a bare
  interpreter token plus double-quoted forward-slash paths (no `shlex.join`, no raw
  backslashes in TOML); the whole file is validated with `tomllib` before one write; old
  broken Windows lines are repaired; Synapse hooks that point elsewhere are re-pointed in
  place. Under the managed client, commands call `~/.synapse/bin/synapse_run.py <relative
  script>`, never `lib/<version>/`, because Codex trusts a hook by a hash of its command:
  updates no longer switch the hooks off.
- **Skill retire**: schema 060 (`skill_registry.retired_at`, optional), `POST
  /skills/retire`, `/skills/list` returns retired entries under a separate `retired` key
  (old clients ignore it), and `/skills/publish` refuses to reactivate a retired skill
  unless the pushed edit is newer than the retirement (old clients can no longer resurrect
  it), nor with the retired content itself under a newer timestamp (a clone or a copy
  renews an mtime). `/skills/list` lists every retirement whatever the scope asked for. The
  client sync removes a local copy that is exactly a version Synapse delivered (ledger
  under `~/.synapse/skills/`, kept after the removal so a restored copy goes again)
  whatever its timestamp, pushes an unrecognised one edited after the retirement, and
  leaves everything else alone (symlinks and junctions included). In another scope's
  folder only the exact retired content counts: a different skill of the same name there
  is never touched.
  Accepting a retune of a retired skill now needs `--force`.

**On each machine**

1. Existing installs keep working with no action (legacy `settings.json` and `device.json`
   are still read).
2. To move over: `curl -sSL https://<server>/install | python3 -` (PowerShell:
   `irm https://<server>/install | python -`), then `synapse login`. On an enrolled machine
   it signs in again only with `--reenroll`.
3. Codex machines: after that first `synapse login`, trust the re-pointed hooks once in
   `/hooks`. The work laptop's hand-patched helper line is rewritten by it too.
4. A token pasted into Claude Code's install dialog lives in Claude's secure storage, which
   nothing else can read; `synapse login` says so and `--reenroll` fixes it.

**Server**: merge → image → apply `schema/060_skill_retired_at.sql`
(`scripts/apply_schema.sh`). Until 060 is applied the server boots and behaves as before;
only `synapse skill retire` is refused.

**Proven** (scratch homes against a local server on this branch)

- Codex-only machine (no `~/.claude`): `curl …/install | python3 -`, `synapse login
  --token …`, then a real `codex exec` (Codex 0.162.1): SessionStart/UserPromptSubmit/Stop
  hooks ran, the board text reached the model, the `mcp__synapse` tools were listed through
  the headers helper, and the turn was ingested.
- Claude Code 2.1.296: after `synapse login`, `claude mcp list` connects through the helper
  with no token in Claude's config; with the helper refusing it fails (401) — and connects
  again from the static fallback token alone.
- Retire across a Claude-side and a Codex-side machine: delivered copies removed on both;
  a copy edited after the retirement pushed back and active again.
- A legacy install (token in `settings.json`, `device.json`, no `config.json`) works
  untouched; `! synapse-login` then migrates it.

**Open**

- Windows was not run on a real host: command strings are proven by `tomllib` and by
  executing them through `sh`/`bash`, and by the documented cmd/PowerShell quoting rules;
  the `.cmd` shims (goto-based exit codes, `NoDefaultCurrentDirectoryInExePath`) by their
  text only.
- Pre-existing, not changed here: the Claude hooks' `commandWindows` in
  `plugin/hooks/hooks.json` still run `python3`, which a python.org-only Windows machine
  lacks (the MCP helper now falls back to `python`/`py -3`).
- `docs/media/setup-demo.gif` still shows the old `/plugin install` flow.
- Out of scope (separate issues): Go port of the client, SEP-2640 serving, gateway-pilot
  teardown, board cap-loop/connection-pool perf.
