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
  note, `f:<uuid>` fact, `w:N` web); only `e:` and `n:` are fetchable, the
  rest are feedback-only. `group_id` scopes the knowledge graph to `technical` (default) or
  `personal`.
- **`recall_full_turns(query, project=None, limit=5, session_id=None)`**: the drill-down and
  retry sibling. `recall()` serves compressed ~1400-char passage slices; this serves whole
  turns and nothing else, ranked by relevance plus recency. It is the right call for exact
  wording, for the retry when an overview recall comes back thin, and when a passage was cut
  off mid-thought. It replaced the former `recall(mode="turns")`, which in turn had absorbed
  the standalone `recall_episodes` tool.
- **`fetch(ids)`** expands ids into full records: `e:N` episode ids from recall results
  (bare `N` also accepted) and `n:N` note ids, so the session-start board block's `n:ID`
  lines resolve to their full note bodies. Mixed lists are fine, unknown ids come back under
  `skipped`, and at most 20 ids expand per call. It only expands ids you already hold; it
  does not search.
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
  new note that restates an existing one updates it in place, and one that contradicts it
  supersedes the old note while keeping the lineage — so the curated set stays small and
  current instead of accumulating duplicates.
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
| `Stop` | `ingest_hook.py` | Pushes a bounded tail of the session transcript to `/ingest` after every turn. This is how sessions become memory, and it works over the network, so a work laptop can feed a central Synapse. |
| `SessionStart` | `skills_sync.py` | Two-way sync between the server and `~/.claude/skills` (plus the project's `.claude/skills`). Newest edit wins per skill, with an append-only server-side history; deletes never auto-propagate. Off by default. |
| `SessionStart` | `config_sync.py` | Mirrors your `CLAUDE.md` and `rules/*.md` to the server so the dream lane can propose config edits. Off by default. |
| `SessionStart` | `git_feeder.py` | Pushes commit subjects from repos listed in `SYNAPSE_TIMELINE_REPOS` to the server's timeline. Off until configured. |
| `SessionStart` | `board_block.py` | Prints the board (`GET /context`) into the session's context: curated note hooks, the last week's milestones, and what memory exists at all. Server-rendered, hard-capped (~80 lines / ~2K tokens), scoped to the session's project. On a 401 from an unenrolled machine it prints why instead. |
| `SessionStart` | `preferences_block.py` | Prints your top standing preferences (max 8 lines) from `/preferences/top`. Not query-scoped, because preferences are few and apply to every turn. |
| `SessionStart` | `ingest_hook.py --catchup` | Sweeps up transcripts from sessions that ended without the `Stop` hook shipping them. |
| `SessionStart` | `remember_spool.py session-start` | Probes the write lane. Server up and the spool non-empty means it flushes; server down means it tells the model to route memory writes through the CLI until it recovers. |
| `SessionEnd` | `private_mode.py --session-end` | Removes the local private-mode marker for the session that just ended. The server-side row is left in place on purpose. |
| `UserPromptSubmit` | `recall_nudge.py` | One static ~45-token line per prompt reminding the model to use recall/remember, plus a wall-clock timestamp so `remember()` and timeline references know the current moment. Zero latency, zero API calls. Off with `SYNAPSE_RECALL_NUDGE=0`. |
| `PostToolUse` (recall, recall_full_turns) | `recall_feedback_nudge.py` | One line reminding the model to close the retrieval-quality loop with `recall_feedback`. Off by default: the labels only help whoever is tuning the retrieval stack. |
| `PostToolUse` (remember) | `remember_spool.py post-tool-use` | When `remember()` comes back without a confirmed write, spools the intent locally and tells the model it is queued, not saved. |
| `PreToolUse` (synapse tools) | `self_session_inject.py` | Injects the calling session's id so serving can exclude the session's own episodes (already in the model's context) and so feedback and remembered episodes attach to the right session. The model cannot know its own session id. |

The dream to skills lane (mines your transcripts, proposes new skills, retunes triggers,
nominates merges) runs server-side in the `dream` container. You do not run a cron; you
review what it proposes with `/synapse:skill-review`.

> Recall-injection (a `UserPromptSubmit` hook that pushed memory *results* into every prompt)
> was removed: unconditional top-k injection added noise and anchored the model on stale
> priors. Use the `recall` tool to pull on demand instead.

## Plugin configuration

Settings resolve in order: env var → `CLAUDE_PLUGIN_OPTION_*` → your `/plugin install`
answers (stored in `settings.json`) → built-in default. A fresh install just answers the
install prompt; env vars are optional overrides (e.g. CI).

Prompted at install:

| Variable | Meaning |
|----------|---------|
| `SYNAPSE_URL` (required) | Base URL of your server, no path (`http://localhost:8765` or `https://synapse.example.net`). The plugin derives `/ingest`, `/recall`, `/skills`, `/timeline`, and `/mcp`. |
| `SYNAPSE_INGEST_TOKEN` | This machine's bearer token. It should be a **device** token, not the shared machine token ([docs/auth.md](auth.md)). One token covers ingest, recall, skill sync, and MCP. Leave blank and run `! synapse-login` to enroll instead. |
| `SYNAPSE_MACHINE_ROLE` | `personal` (default) or `work`. Declared once when this machine enrolls; `work` enrolls at restricted access. Change it later with `/synapse-devices`. |
| `SYNAPSE_SKILLS_SYNC` | `1` enables two-way skill sync. Off by default. |
| `SYNAPSE_CONFIG_SYNC` | `1` mirrors config for the dream lane. Off by default. |
| `SYNAPSE_CONFIG_PATHS` | Optional extra globs (relative to `~/.claude`) to mirror. |

Env or `settings.json` only:

| Variable | Meaning |
|----------|---------|
| `SYNAPSE_TIMELINE_REPOS` | Comma/space-separated repo paths for the timeline git feeder. Unset means the feeder does nothing. |
| `SYNAPSE_BOARD` | `0` disables the session-start board block. |
| `SYNAPSE_PREFS_BLOCK` | `0` disables the session-start preferences block. |
| `SYNAPSE_RECALL_NUDGE` | `0` disables the per-prompt recall/remember reminder. |
| `SYNAPSE_RECALL_FEEDBACK_NUDGE` | `1` enables the post-recall reminder to label results via `recall_feedback` (default off: the labels are offline tuning data). |
| `SYNAPSE_PROMPT_TIMESTAMP` | `0` disables the per-prompt timestamp line (`[Wed 2026-08-26 15:55:01 EDT]`, host-local time; on by default so `remember()` and timeline references know the current moment, weekday included). |
| `SYNAPSE_DATA_DIR` | Where the memory-write spool and lane state live (default `~/.local/share/synapse-skills`). |
| `SYNAPSE_INGEST_URL` | Legacy full-endpoint override, still honored. |

Server-side configuration is a separate list:
[ARCHITECTURE.md §13](../ARCHITECTURE.md#13-configuration).

## What gets sent to your server

Everything below goes only to the Synapse URL **you** configure. All hooks are fail-soft: an
unreachable server is a silent no-op.

- **Transcript ingest** — **on** (the core function). Sends the raw JSONL tail of each
  session transcript: your prompts, Claude's replies, tool calls. Per-session opt-out is
  [private mode](../plugin/README.md#private-mode). No global toggle: if transcripts
  shouldn't leave the machine at all, don't install the plugin.
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
