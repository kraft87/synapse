# Handoff log

Newest first. One entry per change that alters how a machine is set up or operated: what
changed, what to do on each machine, what was proven, and what is still open.

## 2026-10-15 — the code-review fix pass (plugin 0.24.0, optional schema 062-076)

A verified review of the whole codebase was worked off in one pass. Nothing in it is required
to run the new image: all fifteen migrations are optional (first line `-- schema-check:
optional`, code probes for them), and the old image keeps working against a migrated
database except where a section below says otherwise.

**What changed**

- **Ingest.** One writer, `ingestion/episode_writer.store_episodes`, for `POST /ingest` and
  every backfill: span index, contamination and harness calls, scratch workspaces, private
  sessions, delete tombstones, then replays (the cross-session content guard has no switch
  any more). One transaction, and each session's advisory lock until commit, so a
  concurrent `remember()` can no longer be overwritten by the upsert. A turn with no
  `span_id` is skipped and counted in every path. The four per-source backfill modules are
  gone: `python -m ingestion.backfill --source {claude-code,codex,cursor,claude-ai,chatgpt}`.
  A queue row names its source and stores no text; finished rows lose theirs after 7 days.
- **Poller.** One process, two threads: maintenance (chunks, embeddings, sweeps, web pages)
  and extraction (the queue). The `poller-drain` service is gone; one replica holds the
  maintenance advisory lock, every replica extracts. `SYNAPSE_DRAIN_CONCURRENCY` (default 4)
  is the burst knob; a claim is one row per free thread, so a kill strands at most that.
  An idle session's tail (its last 1-3 turns, or a whole 1-3 turn session) is now chunked and
  extracted, at catch-up priority.
- **Extraction.** `ContradictionDetector` and the single-fact prompts are deleted
  (retirements come only from the 6b verdicts, which now retry once and fail closed with a
  WARNING); entities route to the personal or technical graph by name, and a fact whose ends
  disagree is no longer dropped; a matched entity keeps its name, embedding and subtype; fact
  dates with no time are stored at 12:00 UTC (the readers serve the user's local day); the KG
  is written only through the `t_*` columns and `KGClient`.
- **LLM and embeddings.** `llm_client` is four modules; the CLI is the one bundled with
  `claude-agent-sdk` (pinned by `uv.lock`), spawned with no tools, no prompt cache, no session
  transcript (about 10K fewer prompt tokens per call); every exhausted transport failure
  ends in `LLM_TRANSPORT_ERRORS`, so an outage fails queue items for retry instead of marking
  them done. Voyage requests time out at 30 s / 120 s; interactive recall gives up on a 429
  in about 5 s and falls back to RRF order; the poller and dream keep the 3-minute patience.
  The `anthropic`, `pgvector` (Python) and `tokenizers` dependencies are gone, embedder and
  reranker instances are shared per process, and secret redaction is about twice as fast.
- **Recall.** Notes and web legs behind reranker-calibrated floors
  (`SYNAPSE_RECALL_NOTES_FLOOR`, `SYNAPSE_RECALL_WEB_FLOOR`); facts are served as
  `f:<kg_relationships.id>` (old `f:<uuid>` still resolves everywhere); the abstention floor now
  judges the episodes that survive echo suppression, so it fires slightly more often;
  `recall_full_turns` and `fetch` cap a call at ~40,000 characters and excerpt past it;
  restricted callers are served the web pages of their own projects and credential; chunk
  candidates are an optional episode-pool source (default off, see Open). Many recall flags are
  deleted and `mcp_server/recall_config.py` holds the rest.
- **Dream lanes.** Skills, config and notes share `dream/lane.py`: an episode-id cursor per
  lane (late uploads are mined), oldest-last-touch-first session caps, a memo that re-judges a
  session only after 3 more user turns, stage isolation, one run at a time under an advisory
  lock, interrupted runs closed on the next start, a catch-up run at startup, and a
  `recall_metrics` retention prune at 180 days. `SKILLS_LANE_ENABLED` is a kill switch like
  the other two: unset now means ON. All lane LLM calls go through `llm_client`. The skills
  review is accept or reject only; a retune's identity is unique among open candidates; the
  config lane suppresses a rejected rule for 30 days and skips rules already in a mirrored
  config file or a feedback note; the notes lane has an undo CLI.
- **Notes.** `remember()` asks one three-way judge (DUPLICATE, CORRECTION, DISTINCT) against
  the nearest same-type note and inserts when unsure, so expect more notes and fewer
  overwrites for a few nights while the nightly lane merges leftovers; `scripts/import_notes.py`
  is deleted and the notes lane has an undo (`docker compose exec dream python -m
  dream.notes.nightly --list`, then `--revert ID`).
- **HTTP, auth, dashboard.** Routes share `http_helpers.route`; the dashboard is
  `mcp_server/dash/`. `POST /timeline/recent` and `/dash/api/behavior/linkgraph` are gone (404);
  `/config/list` and `/config/fetch` are gone and `/config/publish` takes `{surface, files}`;
  `DELETE /surfaces/{id}` answers `{status, surface_id, revoked}`; device verification runs in a
  worker thread and custom routes reuse FastMCP's verdict. `PUT /private-sessions/{id}` reports
  `stored_turns` and `?purge=1` deletes them. The dashboard fixes (queue badge and SSE
  `ready`/`reset`, proposals ordering, search totals, months jump, delete summary, keyboard
  access, history) ship in the image's bundle.
- **Web capture.** Server-side, by call shape, off by default (`SYNAPSE_WEB_CAPTURE`); see
  the rollout. `POST /capture/dry-run`, `synapse capture`, capture rules, the cold
  `research_archive`, and a poller that chunks, contextualizes and embeds the pages. Extraction
  of captured pages is opt-in per artifact (`metadata.kg_extract`).
- **Timeline and preferences gates** are kept and measured, not merged: retries no longer
  double count, a [user] text under 10 characters skips the preferences gate, and the board
  logs the `t:` ids it serves (`recall_metrics` `kind='board'`) so `recall_feedback` can rate
  them. A re-told timeline event now survives its owner turn's deletion when another turn
  re-told it (064).
- **Clients (0.23.0 and 0.24.0).** A durable per-file upload cursor, a retry/catch-up sweep with
  a 3-day window, sub-agent transcripts shipped while the server captures, hook logs under
  `~/.synapse/logs/`, one config resolver for Claude and Codex, more switches as `synapse
  config` keys, `synapse private on --purge`, `synapse capture`, a User-Agent that carries the
  client version. The Codex recall_feedback nudge is off by default, like Claude's.
- **Infra.** Postgres and the opt-in local-inference port publish on `127.0.0.1`; `shm_size:
  1g`; a poller healthcheck over both heartbeat files; `apply_schema.sh` applies only the
  files above the stamp (`--all`, `--from NNN`); CI builds the image on every PR, tags `:<sha>` and
  retags `latest` after the checks on `main`, and scans `ingestion/ mcp_server/ dream/` with
  bandit; Dependabot has a `uv` group. One-off scripts that have run are deleted.

**Order of rollout (matters)**

Before the merge (nothing here needs the image):

1. Read prod's `.env` for flags that are gone. They are ignored now, so a non-default value
   silently changes behaviour: `SYNAPSE_CONTENT_DEDUP=0`, `SYNAPSE_DEDUP_TYPE_GATE`,
   `SYNAPSE_TIMELINE_DEDUP`, `SYNAPSE_TIMELINE_GROUP_SCOPE`, `SYNAPSE_BANNED_NAME_FILTER`,
   `SYNAPSE_NOTES_CONFIRM`, `SYNAPSE_NOTES_SIM_MATCH`, `NOTES_LANE_*`, `CONFIG_PROPOSE_SESSIONS`,
   `SKILLS_JUDGE_MODEL`, `SKILLS_JUDGE_BACKEND`, `SKILL_MEASURE_MODEL`, `SKILL_DERIVE_DRAFT_MODEL`,
   `SYNAPSE_SKILLS_DETECTORS`, `STRUGGLE_INCLUDE_PROJECTS`, `SYNAPSE_DREAM_MODEL`,
   `SYNAPSE_QUERY_GRAPH_MODEL`, and the recall flags `SYNAPSE_RERANK_RECENCY=0`,
   `SYNAPSE_SUPPRESS_QUERY_ECHO=0`, `SYNAPSE_RERANK_WINDOW`, `SYNAPSE_RECALL_NOTES=0`,
   `SYNAPSE_RECALL_FLOOR_ENFORCE=0`, `SYNAPSE_RECALL_FLOOR_KEEP_MIN`, `SYNAPSE_RECALL_FACT_FLOOR`,
   `SYNAPSE_EPISODE_CUTOFF_TAU`, `SYNAPSE_RECALL_WEB_FETCH`, `SYNAPSE_RECALL_WEB_POOL`,
   `SYNAPSE_RECALL_NOTES_LIMIT`, `SYNAPSE_RECALL_NOTES_FETCH`. Client-side: `SYNAPSE_INGEST_TAIL`,
   `SYNAPSE_INGEST_CATCHUP_DAYS`, `SYNAPSE_INGEST_LOG`, `SYNAPSE_SPOOL_LOG`,
   `SYNAPSE_CODEX_HOOK_LOG`, `SYNAPSE_PRIVATE_TTL_HOURS`, `SYNAPSE_CODEX_CATCHUP*`,
   `SYNAPSE_RECALL_URL` (a Codex user who set `SYNAPSE_CODEX_CATCHUP=0` now gets the sweep). A replacement custom judge model is `SYNAPSE_SKILLS_JUDGE_MODEL`
   (a deployment that set `SKILL_MEASURE_MODEL=deepseek` uses `SYNAPSE_LLM_PROVIDER=openai`).
   If `SYNAPSE_LLM_MODEL` is set globally, the lane judges and drafters now follow it unless
   the two `SYNAPSE_SKILLS_*_MODEL` variables are set; before they always used Opus.
2. Set `SKILLS_EXCLUDE_PROJECTS=transcribe_ai` in prod's `.env`. That project was hard-coded
   out of two detectors; now the variable is the only exclusion, so without it every
   detector includes `transcribe_ai` sessions. `SKILLS_LANE_ENABLED` unset is now ON: if
   prod's `.env` never set it to 1, the skills lane starts mining (propose-only, review-gated)
   on the first night; set `=0` to keep it off.
3. Find host cron or launchd jobs that run `python -m ingestion.cursor_sqlite_backfill`,
   `codex_backfill`, `claude_ai_backfill` or `openai_backfill` and switch them to
   `python -m ingestion.backfill --source {cursor,codex,claude-ai,chatgpt} <path>`; those modules
   no longer exist and the job would fail with `ModuleNotFoundError`. Leave the client-side
   web-capture cron alone until step 9: it keeps writing `web_artifacts` with its own checkout
   until that checkout is updated, and then fails on its imports.
4. Where Watchtower runs, set `WATCHTOWER_TIMEOUT` to at least `5m`. The pollers now take up to
   3-4 minutes to finish an item on SIGTERM; with the default an auto-deploy kills them and
   their claims wait out the 45-minute stale-claim sweep.
5. Read prod's schema stamp (`SELECT value FROM synapse_meta WHERE key = 'schema_version'`)
   and check that 056, 057, 060 and 061 are applied; the probes for them stay until it is
   confirmed. If the row is missing, `apply_schema.sh` refuses and wants `--from NNN` or `--all`.

Then, in order:

6. **Merge → image.** Watchtower swaps the services one by one. For a few minutes the mcp-server
   may run the new image while the old poller is still up: new episode and manual queue rows
   have empty `content`, so an old poller that claims one extracts nothing and marks it done.
   That turn only skips its timeline and preferences gates. After the first deploy, smoke-test
   one extraction and one dream-lane call with the OAuth token (the bundled CLI is untested
   against prod auth), and look in Logfire for a drop of about 10K input tokens per
   `claude_agent_sdk` call and `cache_creation_input_tokens` at 0. `ls /root/.claude/projects | wc -l`
   in the poller container should stop growing. Rollback is the previous image tag.
7. **Migrations, by hand, outside a transaction** (never `psql -1`; a failed `CONCURRENTLY`
   build leaves an INVALID index: drop it and run the file again). `scripts/apply_schema.sh`
   with no flag applies everything above the stamp in order, and runs each file WHOLE, so two
   files that take a write lock on a populated table must be handled by hand BEFORE it (a, b).
   Apply a file alone with `psql "$DSN" -v ON_ERROR_STOP=1 -f schema/NNN_*.sql`. All are
   idempotent.
   a. `072_kg_dashboard_reads`: run its three `UPDATE`s (`kg_relationships.t_created`,
      `kg_entities.t_created`, `kg_entities.entity_subtype`; safe on the old image), then
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS kg_rel_t_created_idx ON kg_relationships
      (t_created DESC, uuid DESC) WHERE t_created IS NOT NULL;`. Run as a whole file the plain
      `CREATE INDEX` locks writes on `kg_relationships`. Soon after the deploy: until the
      `UPDATE`s run, legacy-only facts are missing from the dashboard feed and legacy-only
      entities show no type.
   b. `062_extraction_queue_chunk_id`: on a populated queue first build
      `extraction_queue_episode_idx` `CONCURRENTLY` (the command is in the file); as a whole
      file the plain form briefly blocks enqueues.
   c. Decide on `067` (below) now: the script builds it too.
   d. Then `scripts/apply_schema.sh`. 062 and 072 are now no-ops for the statements above.
   What the files do:
   - `074_kg_episodes_gin`: `CONCURRENTLY`; apply it before the pending project relabel (it
     removes the per-episode full scan in the 056 trigger).
   - `063_episodes_unembedded_index`, `065_kg_live_hop_indexes`: `CONCURRENTLY`, no lock.
   - `066_drop_episode_retrieval_index`: drops `idx_episodes_retrieval_count` and sets
     `fillfactor 90`. Destructive but nothing reads the index, so the old image is unaffected;
     only newly written pages get the slack.
   - `067_chunks_vector_index`: an HNSW build (minutes; `maintenance_work_mem` 512MB, no parallel
     workers). It is needed only before turning on `SYNAPSE_RECALL_CHUNK_CANDIDATES`; to skip
     it until you plan the A/B, apply 062-066 as single files, run
     `scripts/apply_schema.sh --from 068`, and apply 067 alone later.
   - `068_dream_lane_cursors`: additive. The first run after it seeds each lane's cursor from
     its `last_scan_at`.
   - `069_skill_candidate_open_identity`: only after the new image is running, and soon after
     (until then a re-sighted promoted or retired retune is dropped with a warning). **Do not
     roll the image back past it**: an older image's `ON CONFLICT` cannot infer the new index
     and every candidate INSERT fails (the rollback is in ARCHITECTURE.md §7.6).
   - `064_timeline_reported_refs`, `070_skill_draft_attempts`, `071_notes_history`,
     `073_deleted_turns`, `076_research_archive`: additive tables or columns. Tombstones in
     `073` are permanent; until it is applied a deleted turn can come back on a re-ship.
     `076` is safe whether or not the old cron already created `research_archive`.
   - `075_drop_redundant_indexes`: drops the dead `search_cache` table and five btrees the
     old image no longer names; `episodes_project_idx` goes only if 061's index is valid. Not
     reversible by the script; re-running is a no-op.
   The schema stamp ends at 076. The `legacy` columns, `kg_entities.degree`, `reported_count`
   and the retired `recall_metrics` columns are NOT dropped: the old image still names them.
8. **Compose, once, on the server:** `docker compose up -d --remove-orphans`. Watchtower
   updates images, not `compose.yml`. This RECREATES the postgres container (`shm_size` and
   the port binding changed), so Postgres restarts: do it in a quiet moment, ideally
   `docker compose stop -t 120 postgres` first so it shuts down cleanly (Docker's default is
   10 s, and an unclean stop can corrupt `episodes_search_idx`; if BM25 recall degrades
   afterwards, `REINDEX INDEX CONCURRENTLY episodes_search_idx`, ARCHITECTURE.md §15). It also
   removes `poller-drain`, recreates the poller with the two-heartbeat healthcheck and
   `stop_grace_period: 5m`, binds Postgres and local-inference to `127.0.0.1` and drops
   `CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS` from `dream`. Anything that reaches Postgres through
   the published port from another machine (remote psql, scripts, tunnels) stops working:
   uncomment the override line in the `postgres` ports block first, or tunnel over ssh. Do the
   recreate together with the image (an old image never writes `/tmp/poller.maintenance.alive`,
   so the new healthcheck would show unhealthy; nothing acts on it). Then set
   `SYNAPSE_DRAIN_CONCURRENCY=8` in `.env` to keep today's peak (the poller plus poller-drain
   ran 4 + 4; the default 4 is gentler) and `docker compose up -d --no-deps poller`.
9. **Web capture, in this order:** update a client to plugin 0.24.0, run `synapse capture
   --dry-run --since 30d` and read what it would store, set `SYNAPSE_WEB_CAPTURE=on` in
   `.env` and recreate `mcp-server`, then remove the client-side cron's crontab line (after
   the switch is on and the clients run 0.24.0: before 0.24.0 the Stop hook does not send the
   sub-agent transcripts where most deep research runs). Optionally `synapse capture
   --backfill --since 30d`. Rules of your own go in `SYNAPSE_CAPTURE_DIR` (see `.env.example`).
   The poller will chunk, contextualize (an LLM call per page, 20 pages a cycle) and embed any
   artifact the cron left without them: `SELECT count(*) FROM web_chunks WHERE context_prefix
   IS NULL` is the expected call count. Old unmarked pages are never extracted into the KG.
10. **One-off scripts**, each with `--dry-run` first, counts only:
    `python scripts/backfill_normalized_name.py` (entities with a NULL `normalized_name` are
    invisible to dedup now that the unindexed fallback is gone) and
    `python scripts/restamp_date_only_dates.py` (fact dates at UTC midnight move to noon UTC;
    until it runs, old dates stay at midnight).
11. **Watch the first cycles.** The first poller cycle flags every session idle for 2 hours
    that has turns beyond its last chunk and queues one tail chunk each at priority 10 (3-5
    LLM calls per chunk, a one-time burst behind live work). Size it first with:
    ```sql
    WITH ep AS (SELECT session_id, COUNT(*) AS c,
                       MAX(sequence) FILTER (WHERE source IS DISTINCT FROM 'manual') AS t,
                       MAX(created_at) AS last_at FROM episodes GROUP BY session_id),
         ch AS (SELECT session_id, MAX(end_sequence) AS m FROM chunks GROUP BY session_id)
    SELECT count(*) AS tail_chunks_to_extract,
           count(*) FILTER (WHERE e.c < 4) AS sessions_under_4_turns,
           count(*) FILTER (WHERE e.last_at > now() - interval '30 days') AS idle_under_30_days
    FROM ep e LEFT JOIN ch c USING (session_id)
    WHERE e.last_at < now() - interval '2 hours' AND e.t > COALESCE(c.m, 0);
    ```
    The first daily sweep blanks the text of every finished queue row older than 7 days in
    batches (minutes; a `VACUUM FULL extraction_queue` in a quiet moment returns the space).
    The first dream night deletes every `recall_metrics` row older than 180 days in one
    `DELETE` (glance at the table size) and, if the container was redeployed after 02:03,
    runs once at startup. Spot-check the first Dream report: new `skills_*` counts appear,
    and `skills_judge_unparsed` says how often a judge reply had no JSON (the merged analysis
    prompt has not been run against real calls). A window stage that fails every night pins
    the cursor and shows as a failed run each night: read `dream_runs.errors`.

On each machine: `synapse update` to 0.24.0 (a plugin update works too). Old `/tmp/synapse-*.log`
files and `~/.synapse/codex_skills_sync.lock` are orphaned and harmless. Codex hooks keep their
commands, so nothing needs re-trusting in `/hooks`. Codex prompts now start with the timestamp
line, Codex no longer prints the recall_feedback directive, and skills sync, config mirroring,
the git feeder and the catch-up sweep run on startup and resume only (not after `/clear` or
compaction). Codex now reads `CLAUDE_CONFIG_DIR` and only the user-level `settings.json`.
`synapse private on <id> --purge` against an older server exits 1 with private mode ON and
nothing deleted; a work (restricted) device can purge only sessions it stamped itself, so only
after 057 is applied, and the root token gets 403.

**Behaviour changes to expect**

- Recall: the abstention floor fires slightly more often (watch `would_abstain`); fact ids
  look like `f:41`; a restricted device now sees web pages it fetched itself (once capture
  is on); `est_tokens` includes the notes bucket.
- The single-turn dashboard delete now leaves a re-told timeline event (064) and writes a
  tombstone (073), so a deleted turn does not return on a re-ship.
- Accept or reject on a skill candidate or config proposal already in a final status answers
  `conflict` (the dashboard route answers 409); a skills `reject` of a promoted candidate no
  longer works, and a config approval on the dashboard only marks the rule `accepted` until
  `config_review apply` runs on a machine.
- A deployment missing required migrations 033, 035, 050, 053 or 054 (the boot guard normally
  refuses it) now sees HTTP 500 with the database error on the surfaces, private-session,
  timeline-events and preferences routes, not a canned 503.
- `kg_entities` live-degree is counted per matching entity on every typeahead (50-160 ms
  typical, about a second when a query matches nearly every entity).

**Proven**

- Bugs were reproduced in failing tests on the pre-fix code before they were fixed. The suite
  runs three ways, all green on the final branch: non-DB (3,359 passed), DB on a database provisioned
  with `apply_schema.sh` (1,127 passed, 1 skipped) and DB with the connection pool on (1,128
  passed); ruff, mypy over 131 files and bandit over `ingestion/ mcp_server/ dream/` are clean.
- Real ParadeDB (the prod image) for every DB test and for each migration, including the
  optional ones against a database without them; the probes were run against a queue table
  without `chunk_id`, a database without `deleted_turns`, `notes_history` and the lane tables,
  and a `schema_version` stamp of 076 from a fresh provision.
- The dashboard bundle was driven in Chromium (Playwright) against a local server for the
  queue badge, SSE `ready`/`reset`, deep links, the delete summary and the months jump.
- The bundled CLI was driven under `env -i` against a fake Messages API: one request, no tools,
  no cache write, no session file. No real LLM or Voyage call was made anywhere in this pass.
- `vulture` over the shipped code lists no function without a non-test caller other than
  FastMCP and Starlette handlers, validators and the capture entry points.
- Not proven: the CI build and `latest` retag first run on the next `main` merge (watch it);
  the bundled CLI against prod auth (step 6); the skills/config judge prompts against real
  calls; web capture against a real Codex rollout (none was available).

**Open**

- Decide with data (read-only, on prod):
  - Chunk candidates: apply `067`, then run `scripts/ab_chunk_candidates.py` on your goldens
    (JSONL of `query`, gold `e:` ids, optional `project`; needs `VOYAGE_API_KEY` and
    `SYNAPSE_DB_URL`; it exits 1 if any rerank degraded). The losing side is deleted later:
    either the candidate source and the chunks' vector index, or the chunk embeddings and BM25.
  - The two gates: two weeks of `recall_metrics` rows `kind='board'` (`served_ids.events`) joined to
    `recall_feedback` say whether the timeline gate earns its LLM call; the preferences gate
    now skips [user] text under 10 characters, so check whether a short standing rule
    (`no emoji`) goes missing (`_MIN_USER_CHARS` in `ingestion/preferences_gate.py`).
  - Web leg: `SELECT count(*) FILTER (WHERE jsonb_array_length(served_ids->'web') > 0) AS
    recalls_with_web, count(*) AS recalls, avg(jsonb_array_length(served_ids->'web')) AS avg_web
    FROM recall_metrics WHERE kind='recall' AND created_at > '2026-07-24' AND served_ids ? 'web';`
    and the helpful/noise counts of `w:` ids in `recall_feedback`. Near-zero helpful means
    delete `mcp_server/recall_web.py` and its knobs.
  - Floors: the abstention floor was calibrated before echo suppression (recalibrate from a week
    of `rerank_top_all` against `rerank_top_score`), and the web and notes floors (0.60) were
    calibrated on `rerank-2.5-lite` while the default reranker is `rerank-3-lite`.
  - `FULL_TURN_BUDGET` (40,000 characters) is an estimate: `SELECT
    percentile_cont(0.9) WITHIN GROUP (ORDER BY chars) FROM recall_metrics WHERE kind IN
    ('episodes','fetch')` and the `n_excerpted` counts.
  - Entity routing: read the INFO lines "Moved N entities into the technical group" and
    "Dropped N fact(s)" for a day before tightening the bare-noun list; the measuring query is
    `SELECT normalized_name FROM kg_entities GROUP BY 1 HAVING count(DISTINCT group_id) > 1`.
  - Does a turn that grew after Stop get stored twice? `SELECT count(*) FROM episodes a JOIN
    episodes b ON b.session_id = a.session_id AND b.id <> a.id AND length(b.content) >
    length(a.content) AND left(b.content, length(a.content)) = a.content WHERE a.source =
    'jsonl';` Non-zero means a server-side fix (key the turn on its first record uuid).
  - `EXPLAIN ANALYZE` the restricted and project-filtered episode vector queries; if they
    under-fill, set `hnsw.iterative_scan = relaxed_order` in `ingestion/pg_pool.py`.
  - The `recall_nudge` call-rate comparison (sessions with and without the per-prompt
    directive) was never run; if the directive loses, keep only the timestamp line.
- Decide: whether an accepted config proposal should become a `remember(type='feedback')` note
  instead of a file write on one machine; a cheaper default judge for the config lane;
  whether the notes lane's retype may widen audience (it re-derives rule 3 both ways today, with no
  explicit-versus-derived provenance); whether the Search page snippet gets the keyboard access the
  Feed gist got; whether the delete summary should name the pages removed (`web_deleted`).
- Remove later: `SYNAPSE_DRAIN_ONLY` handling (next release), the legacy login paths (once
  every machine runs 0.23.0 or later), the config sync's per-file fallback, the content-keyed
  chunk queue branches (once `062` is applied everywhere and no pending row lacks `chunk_id`),
  the 056 / 057 / 060 / 061 probes (once prod's version is confirmed and 061 has aged). A later
  migration drops the legacy KG columns, `kg_entities.degree`, `timeline_events.reported_count`,
  the retired `recall_metrics` columns (with `test_metrics_recall_retired_legs_are_not_reported`),
  the unread skills-ledger columns and `synth_documents`, each only when no older image can run.
- Known gaps: two concurrent flushes of one `remember()` intent that both pass the claim before
  either records the archive episode each archive it (the client's flush lock makes it very
  rare); quota detection is wording-based, so a CLI upgrade that rewords the notice needs
  `_USAGE_LIMIT_PATTERNS` updated; `post_fire`, `struggle_arc` and the procedure miner judge
  against the all-scope skill catalog. Left alone by decision: the 1 s SSE status loop,
  `git_feeder`'s re-read window, the notes lane's O(N²) pair SQL, `recall_feedback` as
  offline data only.
- Recoverable from history: `scripts/ab_dehub_prompt.py` (a prompt A/B harness whose control arm
  was the banned-name filter) and the two one-time date repair scripts; restore them before
  running anything if prod's data predates those fixes.

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
  atomically. `plugin/scripts/config.py` resolves
  env → `config.json` → the Claude plugin's legacy options (`CLAUDE_PLUGIN_OPTION_*`, then
  `settings.json`) → default, and the Codex hooks' `plugin-codex/scripts/common.py` loads that
  file by path (so the two cannot resolve differently); the Codex Stop hook's private copy
  of that resolution is gone.
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
