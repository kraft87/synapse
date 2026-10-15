# Tools, hooks, and plugin configuration

What Synapse exposes to a model, what the Claude Code plugin runs on your machine, and every
knob the plugin reads.

## MCP tools

Six tools, listed in the order the server registers them. The order is deliberate: tool-list
position biases which tool a model picks, and `tests/test_tool_surface.py` pins it.

The board is deliberately NOT a tool: the plugin's SessionStart hook injects it via
`GET /context`, and a listed board tool would invite a double-inject of a block the model
already has (the Hermes pattern — when injection covers the read, ship no read tool).

- **`recall(query, project=None, session_focus=None, group_id="technical")`**: the primary
  retrieval tool. Reranked episode passages blended with knowledge-graph facts, captured
  web research, and fact history. Served passages carry a `role` label
  (`user` / `assistant` / `mixed`) and a date, so a caller can weight a human-stated fact
  over the agent's own past output. Every served item carries an id (`e:N` episode, `n:N`
  note, `f:N` fact, `w:N` web, `t:N` timeline event on the board); only `e:` and `n:` are
  fetchable, the rest are feedback-only. A fact id is the `kg_relationships.id`; the older
  `f:<uuid>` form is still accepted wherever ids are (`recall_feedback`). `group_id` scopes the knowledge graph to `technical` (default) or
  `personal`.
- **`recall_full_turns(query, project=None, limit=5, session_id=None)`**: the drill-down and
  retry sibling. `recall()` serves compressed ~1400-char passage slices; this serves whole
  turns and nothing else, ranked by relevance plus recency. It is the right call for exact
  wording, for the retry when an overview recall comes back thin, and when a passage was cut
  off mid-thought. A call serves up to ~40,000 characters of turn text: a turn that would
  pass that comes back as its opening plus the stretch that best matches the query, with
  `full_chars` giving its real length (`fetch` its `e:N` id for the rest). It replaced the former `recall(mode="turns")`, which in turn had absorbed
  the standalone `recall_episodes` tool.
- **`fetch(ids)`** expands ids into full records: `e:N` episode ids from recall results
  (bare `N` also accepted) and `n:N` note ids, so the session-start board block's `n:ID`
  lines resolve to their full note bodies. Mixed lists are fine, unknown ids come back under
  `skipped`, and at most 20 ids expand per call. The first episode comes back whole; later
  ones share a ~40,000-character budget and come back as an opening with `full_chars` once it
  is spent. It only expands ids you already hold; it does not search.
- **`fetch_session(session_id, around=None, radius=3, offset=0, limit=10)`** reads one
  conversation sequentially, like opening the transcript file at a spot instead of searching
  it. Every recall/fetch episode carries a `session` field; pass it here to see what
  surrounded that turn. With `around` set to an `e:N` id, that anchor comes back full and
  its neighbours come back as 500-char heads; without one, page the session with
  `offset`/`limit`. `session_id="self"` reads the current conversation. An unindexed session
  returns an explicit error, which is the signal to read the on-disk transcript instead.
- **`remember(content=None, hook=None, body=None, type="project", project=None, audience=None)`**
  writes a curated memory. The preferred form passes `hook` (a one-line index entry, ~120
  chars) plus `body` (the full self-contained note) and a `type` (`user` / `feedback` /
  `project` / `reference`); the legacy `content`-only form still works and derives the hook
  from the first sentence. Both forms also archive the text as an episode and enqueue
  knowledge-graph extraction. Notes land in a dedicated store that is reconciled on write: a
  new note that restates an existing one updates it in place (the old wording goes to
  `notes_history`), one that corrects it supersedes the old note keeping the lineage, and
  anything else, including a second distinct rule on the same topic, is added beside it
  — when the judge is unsure it adds, and a nightly lane merges what is left. A call from a
  device that names its `session_id` is archived but not queued for extraction, because the
  Stop hook ingests that session; in a private session, whose turns the hook never ships,
  the statement is queued like a hookless caller's.
- **`recall_feedback(query, helpful=None, noise=None, missing=None, found_via=None, comment=None, project=None)`**
  reports retrieval quality after a recall whose results you used: which served ids were
  load-bearing, which were noise, and what was missing. It writes one row of offline labeled data for eval
  goldens and reranker tuning, and is deliberately not wired into live ranking.

Two parameters exist on several of these that a caller should never set. `self_session` is
injected by the plugin's PreToolUse hook so serving can suppress the calling session's own
episodes. `surface` is deprecated and ignored: the server identifies the calling device from
its own credential ([docs/auth.md](auth.md)).

One more tool exists but is hidden from tool listings: **`issue_machine_token()`**, the
auth-gated plumbing `synapse login` calls to fetch the root bearer. It stays callable by name
(`tools/call`), it just never competes for a model's attention. A device token is refused
there: leaves do not fetch the root.

`/ingest` and `POST /recall` are plain HTTP routes rather than MCP tools, so the hook scripts
can reach them with a one-shot POST instead of an MCP handshake. They do not appear in the
tool list.

## Plugin hooks

Thirteen hook entries across six Claude Code events (`plugin/hooks/hooks.json`). All of them
are fail-soft: an unreachable server is a silent no-op.

| Event | Script | What it does |
|-------|--------|--------------|
| `Stop` | `ingest_hook.py` | Pushes the part of the session transcript past a durable per-file cursor to `/ingest` after every turn (the cursor advances only after a successful POST; the last turn is re-shipped once because Claude Code may still be writing it). A first run on a transcript ships its last 400 records. While the server captures web research it also ships the session's sub-agent transcripts (`subagents/**/agent-*.jsonl`) and inlines large saved tool outputs of web-research calls. This is how sessions become memory, and it works over the network, so a work laptop can feed a central Synapse. |
| `SessionStart` (startup, resume) | `skills_sync.py` | Two-way sync between the server and `~/.claude/skills` (plus `.claude/skills` at the session's project root: the marker's folder or the git toplevel, which Claude Code reads from any subfolder). Newest edit wins per skill, with an append-only server-side history. A local delete never retires anything; a skill retired on the server, moved to another scope, or not this machine's to read (a work-role machine gets only its allowlisted projects' skills) is removed from machines whose copy is still exactly a version Synapse delivered (whatever its timestamp); a copy edited after a retirement is pushed back instead. Off by default. |
| `SessionStart` (startup, resume) | `config_sync.py` | Mirrors your `CLAUDE.md` and `rules/*.md` to the server (all files in one request) so the dream lane can propose config edits. Off by default. |
| `SessionStart` (startup, resume) | `git_feeder.py` | Pushes commit subjects from repos listed in `SYNAPSE_TIMELINE_REPOS` to the server's timeline. Off until configured. |
| `SessionStart` | `board_block.py` | Prints the board (`GET /context`) into the session's context: curated note hooks, the last week's milestones, and what memory exists at all. Server-rendered, hard-capped (~80 lines / ~2K tokens), scoped to the session's project (labelled like every hook: `.synapse-project` marker, else git toplevel, else folder name). On a 401 from an unenrolled machine it prints why instead. |
| `SessionStart` | `preferences_block.py` | Prints your top standing preferences (max 8 lines) from `/preferences/top`. Not query-scoped, because preferences are few and apply to every turn. |
| `SessionStart` (every start) | `ingest_hook.py --catchup` | Sweeps up transcripts from sessions that ended without the `Stop` hook shipping them: anything modified in the last 3 days (a constant, at most 20 files per start), plus any transcript whose upload failed, however old. When uploads keep failing, it prints one line saying since when and why; on `/clear` or compaction it only repeats that warning. Transcripts the hook never tried to upload, such as history from before install, are left to `synapse-import`. |
| `SessionStart` | `remember_spool.py session-start` | Probes the write lane. Server up and the spool non-empty means it flushes; server down means it tells the model to route memory writes through the CLI until it recovers. |
| `SessionEnd` | `private_mode.py --session-end` | Removes the local private-mode marker for the session that just ended. The server-side row is left in place on purpose. Private mode covers a session from the toggle onward; `private_mode.py on <id> --purge` also deletes the turns already stored. |
| `UserPromptSubmit` | `recall_nudge.py` | One static ~45-token line per prompt reminding the model to use recall/remember, plus a wall-clock timestamp so `remember()` and timeline references know the current moment. Zero latency, zero API calls. Off with `SYNAPSE_RECALL_NUDGE=0` (or `synapse config recall_nudge off`). |
| `PostToolUse` (recall, recall_full_turns) | `recall_feedback_nudge.py` | One line reminding the model to close the retrieval-quality loop with `recall_feedback`. Off by default, on Codex too: the labels only help whoever is tuning the retrieval stack. |
| `PostToolUse` (remember) | `remember_spool.py post-tool-use` | When `remember()` comes back without a confirmed write, spools the intent locally and tells the model it is queued, not saved. |
| `PreToolUse` (synapse tools) | `self_session_inject.py` | Injects the calling session's id so serving can exclude the session's own episodes (already in the model's context) and so feedback and remembered episodes attach to the right session. The model cannot know its own session id. |

The dream lanes (skills: mines your transcripts, proposes new skills and retunes triggers;
config: proposes behavioral rules you corrected the agent on; notes: merges duplicate notes)
run server-side in the `dream` container, all on by default. You do not run a cron; you
review what the first two propose with `/synapse:skill-review` and `/synapse:config-review`.

> Recall-injection (a `UserPromptSubmit` hook that pushed memory *results* into every prompt)
> was removed: unconditional top-k injection added noise and anchored the model on stale
> priors. Use the `recall` tool to pull on demand instead.

## Plugin configuration

The client's settings live in `~/.synapse/config.json`, which belongs to no harness: both
the Claude plugin and the Codex hooks read it, and `synapse login` / `synapse config` write
it. Each setting resolves in order: env var → `~/.synapse/config.json` → the Claude plugin's
legacy saved options (`CLAUDE_PLUGIN_OPTION_*`, then `pluginConfigs."synapse@<marketplace>".options`
in `settings.json`, for installs from before 0.20) → built-in default. Env vars are optional
overrides (e.g. CI).

The token is the exception. Don't set `SYNAPSE_INGEST_TOKEN` in your shell profile or in
the `env` block of `settings.json`: run `synapse login` (or `synapse login --token` for a
minted one). The MCP connection authenticates through a headers helper that both harnesses
run with your environment stripped of credentials, so a token set anywhere but
`config.json` reaches the hooks but not recall. Once this machine is enrolled, its device
token wins over an env value. Session start then prints one line naming the variable and
where to remove it.

Set by `synapse login`, changed with `synapse config <key> <value>`:

| Key (env var) | Meaning |
|---------------|---------|
| `url` (`SYNAPSE_URL`) | Base URL of your server, no path (`http://localhost:8765` or `https://synapse.example.net`). The client derives `/ingest`, `/recall`, `/skills`, `/timeline`, and `/mcp`. The one value each harness also keeps in its own config, because it opens the MCP connection; `synapse config url` updates those copies too. Moving to another server is `synapse login --url <url> --reenroll`: `config url` refuses to carry this machine's credential to another origin unless `--same-server` says it is the same server under a new address. |
| `token` (`SYNAPSE_INGEST_TOKEN`) | This machine's bearer token. It should be a **device** token, not the shared machine token ([docs/auth.md](auth.md)). One token covers ingest, recall, skill sync, and MCP. Written by `synapse login`. |
| `role` (`SYNAPSE_MACHINE_ROLE`) | `personal` (default) or `work`. `synapse login` asks; declared once when this machine enrolls, and `work` enrolls at restricted access. Change it later with `/synapse-devices`. |
| `skills_sync` (`SYNAPSE_SKILLS_SYNC`) | Two-way skill sync. `synapse login` asks ("Sync skills to this machine?"); off when unanswered. |
| `config_sync` (`SYNAPSE_CONFIG_SYNC`) | Mirrors config for the dream lane. Off by default. |
| `config_paths` (`SYNAPSE_CONFIG_PATHS`) | Optional extra globs (relative to `~/.claude`) to mirror. |
| `timeline_repos` (`SYNAPSE_TIMELINE_REPOS`) | Repo paths for the timeline git feeder. |
| `board` (`SYNAPSE_BOARD`) | The session-start board block. |
| `prefs_block` (`SYNAPSE_PREFS_BLOCK`) | The session-start preferences block. |
| `recall_nudge` (`SYNAPSE_RECALL_NUDGE`) | The per-prompt recall/remember reminder. |
| `prompt_timestamp` (`SYNAPSE_PROMPT_TIMESTAMP`) | The per-prompt timestamp line. |
| `recall_feedback_nudge` (`SYNAPSE_RECALL_FEEDBACK_NUDGE`) | The post-recall `recall_feedback` reminder (off by default). |

The Claude plugin's install prompt asks only for `SYNAPSE_URL` and an optional fallback
token; `synapse login` fills in the URL.

Env or `settings.json` only:

| Variable | Meaning |
|----------|---------|
| `SYNAPSE_TIMELINE_REPOS` | Comma/space-separated repo paths for the timeline git feeder. Unset means the feeder does nothing. |
| `SYNAPSE_BOARD` | `0` disables the session-start board block. |
| `SYNAPSE_PREFS_BLOCK` | `0` disables the session-start preferences block. |
| `SYNAPSE_RECALL_NUDGE` | `0` disables the per-prompt recall/remember reminder. |
| `SYNAPSE_RECALL_FEEDBACK_NUDGE` | `1` enables the post-recall reminder to label results via `recall_feedback` (default off in both harnesses: the labels are offline tuning data). |
| `SYNAPSE_PROMPT_TIMESTAMP` | `0` disables the per-prompt timestamp line (`[Wed 2026-08-26 15:55:01 EDT]`, host-local time; on by default, in both harnesses, so `remember()` and timeline references know the current moment, weekday included). |
| `SYNAPSE_DATA_DIR` | Where the ingest cursors, the retry bookkeeping and the memory-write spool live (default `~/.local/share/synapse-skills`). |
| `CLAUDE_PROJECTS_DIR` | Transcript root for the ingest hook's catch-up sweep and `synapse import` (default `~/.claude/projects`). |
| `SYNAPSE_HOME` | The client's config directory (default `~/.synapse`): `config.json`, the installed client, and the hook logs. |
| `SYNAPSE_INGEST_URL` / `SYNAPSE_MCP_URL` | Legacy full-endpoint overrides for `/ingest` and `/mcp`, still honored (else both derive from `SYNAPSE_URL`). |

The six switches above (`SYNAPSE_TIMELINE_REPOS` through `SYNAPSE_PROMPT_TIMESTAMP`) are also `synapse config` keys, so they can be set without editing the environment. The sweep window (3 days, 20 files per start, 300 s grace for a live session) is a constant shared by the Claude and Codex hooks, not a knob. Every client hook logs to `~/.synapse/logs/<name>.log` (`ingest-hook`, `codex-hook`, `remember-spool`, `mcp_headers`), each capped at 64 KB; the `SYNAPSE_INGEST_*`, `SYNAPSE_SPOOL_*` and `SYNAPSE_CODEX_CATCHUP*` variables of earlier releases are no longer read.

Server-side configuration is a separate list:
[ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration).

## What gets sent to your server

Everything below goes only to the Synapse URL **you** configure. All hooks are fail-soft: an
unreachable server is a silent no-op.

- **Transcript ingest** — **on** (the core function). Sends the raw JSONL of each session
  transcript past the upload cursor: your prompts, Claude's replies, tool calls. Per-session
  opt-out is [private mode](../plugin/README.md#private-mode) (from the toggle onward;
  `--purge` also deletes what was already stored). No global toggle: if transcripts
  shouldn't leave the machine at all, don't install the plugin. Once the server says it
  captures web research (`SYNAPSE_WEB_CAPTURE`, off by default), the hook also sends the
  session's sub-agent transcripts and, for web-research-shaped calls only, the text of the
  large tool outputs Claude Code saved under `~/.claude/projects` (at most 1 MB each, 2 MB
  per upload); a private session sends neither.
- **Skill sync** — **off** (opt-in). When enabled, sends skill bodies and bundled files and
  pulls server versions back into `~/.claude/skills` at session start. On:
  `SYNAPSE_SKILLS_SYNC=1`.
- **Config mirroring** — **off** (opt-in). When enabled, sends your `~/.claude/CLAUDE.md` and
  `rules/*.md` and the project's equivalents. These often carry personal instructions, which
  is why it ships off. On: `SYNAPSE_CONFIG_SYNC=1`.
- **Timeline git feeder** — **off** (opt-in). When `SYNAPSE_TIMELINE_REPOS` is set, sends
  commit subjects, dates, and a coarse salience score from those repos. Unset means nothing
  runs.
- **Board and preferences blocks** — **on**, but they *send* nothing: they read
  `GET /context` and `/preferences/top` and print bounded blocks into your context. Off:
  `SYNAPSE_BOARD=0`, `SYNAPSE_PREFS_BLOCK=0`.
- **Memory-write spool** — **on**. Sends nothing extra: it replays a `remember()` you already
  asked for but that failed to reach the server. Until it succeeds the note sits in a local
  jsonl file, and you can inspect or drop it (`remember_spool.py list`, or delete the file).

The whole client surface is HTTP: `/ingest`, `/recall`, `/skills/*`, `/config/publish`,
`/timeline/*`, `/context`, `/preferences/top`, `/remember/spool`, and `/mcp`, all under the
one `SYNAPSE_URL` and gated by one token. The client holds no Postgres credentials, and skill
or config proposals are only ever applied through your explicit review commands.
