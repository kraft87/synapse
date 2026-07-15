# Dashboard API contract

The operator dashboard (issue #12) is a React/esbuild single-bundle app served by the
MCP server process. This file pins the wire contract between `mcp_server/dashboard_routes.py`
(server) and `web/` (client) — both sides build to THIS document; change it before changing
either side. Design source of truth: the handoff bundle (prototype + spec), not committed here.

## Namespace & auth

- App shell: `GET /dash` (HTML, inlines the built CSS, links `/dash/app.js`).
  `GET /dash/app.js`, `GET /dash/assets/{file}` (self-hosted IBM Plex woff2; whitelist
  served from the build's `assets/` dir — no path traversal). Static routes are
  UNAUTHENTICATED (the code is public; no data in the bundle).
- Every `/dash/api/*` route requires `Authorization: Bearer <SYNAPSE_MACHINE_TOKEN>`
  via the existing `_machine_authorized` seam. 401 → `{"status":"error","detail":"unauthorized"}`.
  The client keeps the token in `localStorage` (paste-once login screen; also accepts
  `#token=...` in the URL fragment on first load, then strips it). Fragment, not query
  param — fragments never reach server logs.
- Spec §7 names endpoints bare (`/feed`, `/search`); they are namespaced under
  `/dash/api/` here to avoid colliding with existing routes (`/timeline/*`,
  `/preferences/top`, `/recall`, `/context`).
- Errors: non-200 with `{"status":"error","detail":str}`. All list endpoints enforce
  server-side limit caps. All timestamps are ISO-8601 UTC strings.

## Phase 1 endpoints

### GET /dash/api/catalog
Response:
```json
{"projects": [{"name": "synapse", "count": 3403}],
 "sources":  [{"name": "claude-code", "count": 31000}],
 "group_ids": ["technical", "personal"]}
```
Projects/sources aggregated from episodes (NULL project or source → name "untagged").
`group_ids` = distinct scopes across kg_relationships/kg_entities group_id ∪
timeline_events.domain. Cached in-process ~5 min.

### GET /dash/api/feed?cursor&limit=30&project&group_id&source
Reverse-chronological merge of episodes + KG facts + timeline events, keyset-paged.
- `cursor`: opaque string from a previous response (`next_cursor`); absent = newest.
- `limit` ≤ 100. Filters apply where the column exists (episodes: project+source;
  facts: group_id; timeline: project + group_id via its `domain` column, schema 038) —
  a filter a type lacks does not exclude that type.
- Feed timestamps: episode `created_at`, fact `created_at`, timeline `ingested_at`
  (the feed shows when memory was WRITTEN; the event's own date is `t_valid` in data).

```json
{"items": [
  {"type": "episode", "id": "227168", "ts": "...", "project": "synapse",
   "source": "claude-code", "gist": "first meaningful line…", "flagged": false,
   "data": {"session_id": "…", "sequence": 12}},
  {"type": "fact", "id": "<uuid>", "ts": "...", "group_id": "technical",
   "gist": "<fact text>", "flagged": false,
   "data": {"fact": "…", "src_name": "…", "tgt_name": "…", "t_valid": "…",
             "t_invalid": null, "provenance_episode_id": 227168}},
  {"type": "timeline_event", "id": "8841", "ts": "...", "project": "synapse",
   "gist": "<event text>", "flagged": false, "sal": 0.9,
   "data": {"fact": "…", "t_valid": "…", "source": "git:synapse", "episode_id": 227001}}
 ],
 "next_cursor": "…" }
```
`sal` maps salience 0/1/2 → 0.3/0.6/0.9 (timeline items only). Fact/timeline items are
fully inline (no expand fetch); episode expansion fetches `/dash/api/episode/:id`.
`data.episode_id` on timeline items is the resolved `ep:N` source_ref or null.

### GET /dash/api/episode/{id}
```json
{"id": 227168, "session_id": "…", "sequence": 12, "project": "…", "source": "…",
 "platform": "…", "model": "…", "created_at": "…", "flagged": false,
 "human_turn": "…", "assistant_turn": "…", "content": "…"}
```
`content` is the full stored turn (may include tool traces); client derives display.

### GET /dash/api/episode/{id}/derived
```json
{"facts": [{"uuid": "…", "fact": "…", "group_id": "…", "t_valid": "…", "t_invalid": null}],
 "timeline_events": [{"id": 8841, "fact": "…", "t_valid": "…", "salience": 2}]}
```
Facts whose `episodes` jsonb array contains the id; events with `source_ref = 'ep:<id>'`.

### GET /dash/api/session/{id}?highlight=<episode_id>
```json
{"session_id": "…", "project": "…", "source": "…", "highlight": 227168,
 "episodes": [{"id": 1, "sequence": 1, "created_at": "…",
                "human_turn": "…", "assistant_turn": "…", "content": "…"}]}
```
Ordered by sequence, cap 500 rows. `highlight` echoes the query param (client scrolls to it).

### GET /dash/api/entity/{uuid}?mentions_offset=0
```json
{"entity": {"uuid": "…", "name": "…", "entity_type": "…", "summary": "…",
             "degree": 12, "created_at": "…"},
 "stats": {"edges": 12, "served": 340, "facts": 15},
 "facts": [{"uuid": "…", "fact": "…", "name": "<verb>", "t_valid": "…", "t_invalid": null,
             "other": {"uuid": "…", "name": "…"}, "provenance_episode_id": 227001,
             "flagged": false}],
 "mentions": {"items": [{"episode_id": 227001, "created_at": "…", "gist": "…"}],
               "offset": 0, "limit": 20, "total": 63}}
```
Facts = live + superseded edges touching the uuid (superseded have `t_invalid` set).
`served` = sum of edge `retrieval_count`. Mentions = distinct provenance episodes, newest
first, paged. (Timeline/preference cross-links: reserved for a later phase.)

### GET /dash/api/search?q&type=episodes|facts|entities|events&offset=0&limit=20&project&group_id&source
ParadeDB BM25 (`@@@`) for episodes / facts / events; entity names via ILIKE. NOT recall().
```json
{"hits": [{"type": "episodes", "id": "227168", "snippet": "…",
            "meta": {"project": "…", "source": "…", "ts": "…", "session_id": "…"}}],
 "total_by_type": {"episodes": 41, "facts": 7, "entities": 2, "events": 12},
 "offset": 0, "limit": 20}
```
`total_by_type` is always computed for all four types (tab counts); `hits` only for the
requested `type`. `limit` ≤ 50. Entity hits: `meta: {name, entity_type, degree}`.
Fact/event hits carry `meta.episode_id` (fact → first provenance episode, event →
resolved `ep:N` source_ref; null when unresolvable) so the client can deep-link.

### GET /dash/api/flags · POST /dash/api/flag
Flag kinds: `episode | fact | timeline_event | preference | note`.
`item_id`: episode/timeline/preference/note → numeric id as string; fact → edge uuid.
```json
POST body: {"kind": "fact", "id": "<item_id>", "note": "optional"}
POST resp: {"status": "ok", "flagged": true}
GET  resp: {"flags": [{"id": 3, "kind": "fact", "item_id": "…", "note": null,
                        "created_at": "…", "gist": "<resolved item text, best effort>"}]}
```
POST toggles: no active row → insert; active row → set `removed_at`. Every toggle appends
a `dashboard_audit` row. Flag state on feed/entity payloads comes from the active set.

## Phase 2 endpoints (Recall console)

Phase 2 adds the Recall debugging page. It reuses the existing `POST /recall` route
(the only change to an existing endpoint) and adds one dedicated history endpoint.

### POST /recall — `debug` flag

`POST /recall` already accepts `{query, group_id?, write_feedback?, source?}` (machine-token
gated, same bearer as `/dash/api/*`). Phase 2 adds two body fields:

- `project?` (string) — scopes the episode legs, same as the MCP `recall(project=…)` arg.
- `debug?` (bool, default false) — when true the response gains a `debug` envelope.

The dashboard console calls it with `{query, project?, group_id, debug: true, source: "dashboard"}`
and **never** sets `write_feedback` (it defaults false on this route) — a debug recall must not
bump the retrieval-count feedback signal.

The `debug` envelope surfaces the SAME numbers the engine already measures for the
`recall_metrics` telemetry row (no re-instrumentation):

```json
{"query": "...", "facts": [...], "episodes": [...], "...": "normal recall payload",
 "debug": {
   "total_ms": 341.0,
   "legs_ms": {"embed": 12.0, "bm25": 48.0, "vector": 96.0, "kg": 141.0,
               "web": 4.0, "rerank": 168.0, "timeline": 33.0, "prefs": 9.0},
   "pool_sizes": {"bm25": 100, "vector": 100, "fused": 100, "kg_candidates": 12},
   "rerank": {"model": "rerank-2.5-lite", "top_score": 0.91},
   "est_tokens": 1874}}
```

`legs_ms` carries only legs the engine timed. `embed / bm25 / vector / kg / web / rerank`
are always present; `timeline` / `prefs` appear only when their leg is enabled
(`SYNAPSE_RECALL_TIMELINE` / `SYNAPSE_RECALL_PREFS` ≠ 0) — an omitted leg renders as
untimed/skipped in the waterfall. Absent `debug` key ⇒ `debug` was not requested. The waterfall
UI models the parallel band schematically (all parallel legs start at embed-end, rerank at the
max parallel end) from these durations; the payload carries durations, not start offsets.

### GET /dash/api/recall/history?limit=50

Recent `recall()` calls from the `recall_metrics` log (`kind = 'recall'`), newest first.
`limit` ≤ 200 (default 50). Machine-token gated like every `/dash/api/*` route.
Deviation from spec §8 (which routed history through the phase-4 `/metrics/recall` aggregate):
a dedicated slim endpoint ships now and can be superseded by the aggregate later.

```json
{"items": [
  {"id": 88231, "created_at": "...", "query": "postgres connection pooling decisions",
   "source": "dashboard", "ms_total": 341.0, "est_tokens": 1874, "rerank_top_score": 0.91}
]}
```
## Phase 2b endpoints — Review → Proposals

A THIN review console over the two independent proposal lanes
(`skills_lane.skill_gap_candidates`, schema 022/027; `config_lane.config_proposals`,
schema 030). The dashboard REUSES each lane's own `_proposal_act` (state transition +
side effects) and `_proposal_detail` (row read) from `mcp_server/skill_sync_routes.py`
and `mcp_server/config_sync_routes.py`; it never reimplements them and never materializes
an accepted change (promote / apply stay with the lanes). Proposal ids are namespaced
`"skill:<n>" | "config:<n>"`; kinds normalize to `"skill" | "config-edit"`.

### GET /dash/api/proposals?status&kind
Unified, lane-merged list + the nav-badge count.
```json
{"proposals": [
  {"id": "skill:12", "kind": "skill", "name": "latency-triage",
   "gist": "Recurring recall latency debugging…", "status": "proposed",
   "age_days": 0, "created_at": "…"},
  {"id": "config:4", "kind": "config-edit", "name": "CLAUDE.md",
   "gist": "Add the raw-SQL rule…", "status": "proposed", "age_days": 1, "created_at": "…"}],
 "pending_count": 2}
```
- `name`: skill candidate name / config `file_key`. `gist`: the lane `summary` (capped 200).
- Rows capped 200 **per lane**, newest first, then merged and re-sorted by `created_at`.
- Statuses shown: skills `{proposed, accepted, promoted, rejected}`, config
  `{proposed, accepted, applied, rejected}`. `observe` (pre-graduation) and skills'
  `retired` (decayed) are NOT proposals and are excluded.
- `status` filters to one status; `kind` (`skill`|`config-edit`) to one lane. `status=all` == unset.
- `pending_count` = `status='proposed'` across BOTH lanes, **independent** of the view
  filters — the header Review badge reads it (accent pill, dark text; hidden at 0).

### GET /dash/api/proposals/{id}
Normalized detail over the lane's `_proposal_detail`.
```json
{"id": "skill:12", "kind": "skill", "name": "latency-triage", "status": "proposed",
 "evidence": [{"session_id": "…", "class": "grounded", "signal": "explicit_request", "why": "…"}],
 "provenance_episodes": [227168, 227201],
 "payload": {"type": "markdown", "content": "# latency-triage\n…"},
 "audit_log": [{"ts": "…", "action": "proposal_approve", "note": "…"}]}
```
- `evidence` (`str | list`): the lane's raw evidence list (both lanes store a list).
- `provenance_episodes`: best-effort episode ids behind the evidence — explicit
  `episode_id` / `ep:<id>` refs on entries, plus the episodes of each distinct evidence
  `session_id` (bounded 40). `[]` when nothing resolves (lane evidence is keyed by
  session, not episode, so this is genuinely best-effort).
- `payload.type`: `markdown` (skills SKILL.md draft from `proposal_body`; falls back to
  `summary`) or `diff` (config unified `diff`).
- `audit_log`: `dashboard_audit` rows for this id, oldest first; plus the lane's own
  `reject_reason` as a synthetic entry when a decision left no dashboard row (e.g. a CLI reject).
- 400 on a malformed id, 404 when the row is absent.

### POST /dash/api/proposals/{id}/decision
```json
body: {"action": "approve" | "reject", "note": "…"}
```
- `approve` → the lane's `_proposal_act(..., "accept", ...)`, delegating the lane's own accept:
  - **skills**: appends a grounded `accept` signal, recomputes the evidence roll-ups, sets
    `status='accepted'`. Materializing (`promoted` = the `mv` into `~/.claude/skills`) stays
    with the skills CLI. The lane's accept takes an advisory `llm` callable (a routing-eval
    on RETUNE candidates — an Anthropic call that only annotates the result). The dashboard
    passes a **stub** returning a skip note, so a decision is a pure state transition + note;
    the real eval runs in the skills CLI.
  - **config**: sets `status='accepted'`, keeps the stored blast-radius `scope` (the
    dashboard never re-scopes → `scope=None`). Materializing (`applied` = the disk write)
    stays with the config CLI.
- `reject` → the lane's `_proposal_act(..., "reject", note)`. **`note` is REQUIRED**
  (non-empty; 400 otherwise) and becomes the lane's `reject_reason`.
- Every decision appends a `dashboard_audit` row: `action` `proposal_approve|proposal_reject`,
  `kind` (`skill|config-edit`), `item_id` (the namespaced id), `detail` `{note, lane_result}`.
- Returns the lane's own result dict. 404 when the row is absent.

**30-day reject cooldown.** The **skills** lane implements it (sets
`rejected_until = now() + interval '30 days'` on reject; its own list hides cooldowned rows)
— the dashboard reject surfaces it via the lane's act, no extra code. The **config** lane
has **no cooldown** (`config_proposals` has no `rejected_until` column). This is a GAP,
left to the config lane to add rather than bolted onto the dashboard.

**Deviations / notes.**
- The unified list does NOT reuse the lanes' `_proposals_list` (those return only `proposed`
  rows in lane-specific shapes); it queries the two tables directly for the all-status,
  normalized, merged view. The write path (`_proposal_act`) and detail read
  (`_proposal_detail`) DO reuse the lane functions.
- `dashboard_audit.action` gains `proposal_approve` / `proposal_reject` (the column is
  intentionally CHECK-free — schema 042 — so no migration is needed to widen it).

## Phase 4 endpoints (Metrics ops page)

Three read-only aggregate endpoints backing the Metrics page's Recall / Ingestion / Corpus
tabs. Machine-token gated, threadpool, bounded — same posture as every `/dash/api/*` route.

**Honesty notes (read before consuming a series):**
- **Estimated corpus counts.** A table's headline `rows` uses `pg_class.reltuples` (a fast
  planner estimate, refreshed by ANALYZE/autovacuum) once the table exceeds ~50K rows;
  below that it's an exact `count(*)`. `rows_estimated` says which. The sparkline / delta are
  ALWAYS exact (they count real rows in the window off the table's own time column).
- **No ingestion depth history.** `extraction_queue` overwrites a row's `status` in place, so
  historical queue depth is NOT reconstructable and is never fabricated. `queue_depth` is a
  LIVE snapshot only. The only honest throughput series the columns support are
  **enqueued/hour** (`enqueued_at`) and **completed/hour** (`processed_at` — the single
  completion timestamp; set when a row goes done/failed).

### GET /dash/api/metrics/recall?window=7d&bucket=1h
Percentiles over `recall_metrics` where `kind='recall'`, hour-bucketed, within the window.
`window` accepts `Nd`/`Nh`/`Nm` (or bare seconds), floored at 1h and **capped at 30d**. The
`bucket` param is accepted but hourly is the only granularity today. Percentiles are
`percentile_cont` (continuous); NULL leg timings are ignored by the aggregate, so a disabled
leg simply doesn't contribute to that bucket's `legs_p50`.
```json
{"series": [
   {"t": "2026-07-01T12:00:00+00:00", "p50": 218.0, "p95": 389.0, "calls": 12,
    "tokens_p50": 1540,
    "legs_p50": {"embed": 12.0, "bm25": 48.0, "vector": 96.0, "kg": 141.0,
                 "web": 4.0, "rerank": 168.0, "timeline": 33.0, "prefs": 9.0}}],
 "slowest": [{"query": "…", "ms_total": 612.0, "created_at": "…"}],
 "score_hist": [{"lo": 0.0, "hi": 0.1, "n": 0}, {"lo": 0.9, "hi": 1.0, "n": 41}]}
```
- `series` ordered oldest→newest; a leg key is present in `legs_p50` only when that leg was
  timed in the bucket. `p50/p95/tokens_p50` are null for a bucket with no non-null values.
- `slowest`: top 10 recalls by `ms_total` desc within the window.
- `score_hist`: 10 fixed bins over `rerank_top_score` in [0,1] (`width_bucket`); a 1.0 score
  folds into the top bin, sub-0 into the bottom bin.

### GET /dash/api/metrics/ingestion?window=48h
Extraction-queue health. `window` parsing identical to `/metrics/recall` (default 48h).
```json
{"queue_depth": 7,
 "queue": {"pending": 7, "processing": 1, "failed": 3},
 "throughput": {
   "enqueued_per_hour":  [{"t": "…", "n": 40}],
   "completed_per_hour": [{"t": "…", "n": 38}]},
 "failures": [{"id": 91, "episode_id": 227168, "error": "…", "enqueued_at": "…",
               "processed_at": "…", "attempts": 3}],
 "last_dream": {"id": 5, "started_at": "…", "finished_at": "…", "duration_s": 41.0,
                "ok": true, "stages": {"config": {"ran": true, "ok": true}},
                "counts": {"proposals_raised": 2, "config_proposals": 2},
                "samples": {"proposals": [{"id": "config:4", "kind": "config-edit",
                                           "name": "rules/learned.md"}]},
                "errors": []}}
```
- `queue_depth` = live pending count (headline). `queue` breaks out pending/processing/failed.
- `throughput.*_per_hour`: hour-bucketed, oldest→newest, over the window. These are the two
  honest series (see honesty note); there is deliberately no depth history.
- `failures`: up to 20 most-recent `status='failed'` rows.
- `last_dream`: the latest `dream_runs` row (schema 044), or `null` when the table is empty
  ("no runs recorded yet"). `duration_s` = finished−started; `null` while a run is in flight.
  `counts`/`samples`/`stages` carry only what the nightly lanes cheaply report today
  (`proposals_raised` + the config lane's session/correction/proposal counts); the fuller
  set (facts_extracted, superseded, dedup_merges, timeline_events) is honestly absent until a
  lane reports it — the dream lanes propose, they do not extract facts.

### GET /dash/api/metrics/corpus
Per-table row counts + 30-day sparkline, and the episodes-by-project / by-source
proportions. **The whole response is cached in-process for 1h** (row counts over 40K+ rows
are the expensive part).
```json
{"tables": [{"name": "episodes", "rows": 44210, "rows_estimated": true,
             "spark_30d": [12, 8, …30 daily counts…], "delta_30d": 640}],
 "by_project": [{"name": "synapse", "n": 3403}],
 "by_source":  [{"name": "claude-code", "n": 31000}]}
```
- `tables`: episodes, kg_entities, kg_relationships, timeline_events, preferences, notes,
  chunks (each only if the table exists). `spark_30d` = 30 daily counts (oldest→newest) off
  the table's own time column (`created_at`, or `ingested_at` for timeline/preferences);
  `[]` when the table has no time column. `delta_30d` = sum of `spark_30d`.
- `rows_estimated` — see the honesty note. `by_project`/`by_source` are exact, top 12,
  `untagged` for NULL/empty.

## Schema (migrations 042, 044)

`dashboard_flags(id, kind, item_id, note, created_at, removed_at)` — active = `removed_at IS NULL`,
partial-unique on (kind, item_id) where active.
`dashboard_audit(id, ts, action, kind, item_id, detail jsonb)` — append-only; actions this
phase: `flag`, `unflag`. Later phases append proposal decisions.
`dream_runs(id, started_at, finished_at, stages jsonb, counts jsonb, samples jsonb,
errors jsonb, ok bool)` (migration 044) — one row per nightly dream-pipeline run, written
fail-soft by `dream/__main__.py` (insert at start with `ok=NULL`, update at finish). Backs
`/metrics/ingestion`'s `last_dream` and the phase-5 Dream-report page.

## Later phases (reserved paths)

`/dash/api/stream` (SSE), `/dash/api/timeline`, `/dash/api/preferences`,
`/dash/api/behavior/*`, `/dash/api/dream/report`, `/dash/api/graph/*`.
(`POST /recall`'s `debug: true` flag + `/dash/api/recall/history` shipped in phase 2;
`/dash/api/proposals*` shipped in phase 2b; `/dash/api/metrics/{recall,ingestion,corpus}`
shipped in phase 4 — see above.)
