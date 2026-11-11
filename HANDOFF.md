# Handoff log

Newest first. One entry per change that alters how a machine is set up or operated: what
changed, what to do on each machine, what was proven, and what is still open.

## 2026-11-11 — one memory per owner: the personal/technical split is gone (schema 067)

**What changed**

- **One knowledge graph per owner.** Extraction no longer sorts entities and facts into a
  `personal` and a `technical` group, so a fact between an entity of each kind is written
  instead of dropped, and matching and contradiction search see the whole graph. What a
  restricted device may be served is decided by each fact's audience label (schema 066) and
  the provenance of its source episodes, never by a group. `domain` survives only as a tag on
  timeline events.
- **`recall` takes no `group_id`.** A client that still sends one is accepted and the
  argument is ignored (the server logs one line); `tools/list` no longer shows it. `POST
  /recall` ignores the body field too.
- **`SYNAPSE_PERSONAL_SCOPE` and `SYNAPSE_PERSONAL_PROJECTS` are gone.** Remove them from your
  `.env`; nothing reads them. `SYNAPSE_OWNER_NAME` and `SYNAPSE_OWNER_ALIASES` stay.
- **A restricted read needs the label.** Without schema 066 (reachable only with
  `SYNAPSE_SCHEMA_CHECK=0`) a restricted device is served no facts; the provenance-only
  fallback is gone.
- **Schema 067** is required: it merges the existing `personal` rows into `technical`, folds a
  personal entity into its same-name technical twin, builds three owner-only indexes, gives
  `group_id` a `'technical'` default on the four tables that carry it, drops the 066 insert
  trigger and stamps 067. `scripts/pile_merge.py` previews, health-checks and undoes it.

**Order of rollout**

1. Stop the pollers (`docker compose stop poller poller-drain`).
2. `python scripts/pile_merge.py --preview` (read-only; counts only). Read the numbers.
3. Apply 067 by itself, before the release: `psql "$SYNAPSE_DB_URL" -v ON_ERROR_STOP=1 <
   schema/067_kg_one_memory.sql`. It runs two transactions of its own, so never inside
   `BEGIN` and never with `-1`. A re-run is safe and does nothing when nothing is left.
4. `python scripts/pile_merge.py --health` must exit 0.
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
