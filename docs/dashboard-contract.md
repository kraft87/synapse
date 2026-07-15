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

## Phase 3 — Live stream (SSE)

Phase 3 replaces the Feed's 30s polling with a Server-Sent-Events live stream. A NOTIFY
trigger (schema 043) fires on every insert into the three feed source tables; a single
server-side LISTEN worker hydrates each notification into a full FeedItem and fans it out
to connected clients over `GET /dash/api/stream`.

### GET /dash/api/stream
Machine-token gated like every `/dash/api/*` route, `Content-Type: text/event-stream`.

**Auth is why the client uses `fetch`, not `EventSource`.** `EventSource` cannot set the
`Authorization` header, and this route requires the same `Bearer` token as the rest of the
API. So the client reads the stream with `fetch` + a `ReadableStream` line parser (honoring
`event:` / `data:` / `id:` fields) rather than the native `EventSource`. Unauthorized → the
usual 401 `{"status":"error","detail":"unauthorized"}` (a plain JSON body, not a stream).

**Events**
- `new_episode` | `new_fact` | `new_timeline_event` — `data` is the FeedItem JSON, the SAME
  shape `/dash/api/feed` items carry (produced by the same hydration code path). Each carries
  an `id:` — a monotonically increasing integer event id.
- `processing_status` — `data` is `{"queue_depth": N, "active": bool}` from `extraction_queue`
  (pending count + whether anything is processing). Emitted on connect and then ~1/s. It has
  **no `id:`** (only feed events advance Last-Event-ID). The refresh is a single shared query
  throttled to ~1/s across ALL connected clients — never a per-connection query.
- Heartbeat — a `: heartbeat` SSE comment every 15s so idle proxies don't kill the stream.

**Resume / reset (Last-Event-ID).** The server keeps the last 512 feed events in an
in-process ring buffer. On connect the client may present its last seen event id via the
`Last-Event-ID` request header (or `?last_event_id=`):
- still inside the buffer → the server **replays** every event after it, then streams live;
- aged out of the buffer, or ahead of the server (buffer was reset, e.g. a server restart) →
  the server sends a `reset` event (`data: {}`, no id) and the client **refetches page 1**
  (`GET /feed`) to resync, then resumes live;
- absent → the client already has page 1 from its initial `/feed` fetch, so the server
  streams live only (from the current buffer head).

**Client reconnect (spec §2).** The reader reconnects manually with 1s→30s exponential
backoff, resending `Last-Event-ID`. A transient loss turns the header live dot amber
("reconnecting") and keeps the stale items on screen. After >5 consecutive failed
reconnects the client gives up on the stream and falls back to the original 30s `/feed`
polling automatically. New items that pass the CURRENT filter cluster prepend with the
slidein animation; items filtered out client-side are dropped (they reappear correctly on
the next full reload). `MOCK` mode has no server and stays on polling.

**Zero cost when idle.** The LISTEN worker (one dedicated psycopg connection) starts lazily
on the first subscriber and stops when the last disconnects; with no client connected there
is no LISTEN connection and NOTIFY is discarded by Postgres.

**Deviations from spec §7.**
- Timeline FeedItems now carry a top-level `group_id` (mirroring the `domain` column, schema
  038) so the live client can apply the group filter uniformly across all three feed types.
  Additive — episodes/facts are unchanged; `/feed` timeline items gain the same field.
- `processing_status` is a plain event with no event id (ids track only the resumable feed
  events); spec §7 lists it among the stream's event names, which it is.
- The NOTIFY payload is `{type, id}` only (NOTIFY's 8KB cap) — the server re-SELECTs the full
  row via the id, so the wire FeedItem is produced by ONE code path (`/feed`'s hydration),
  never duplicated in SQL.

## Schema (migrations 042, 043)

`dashboard_flags(id, kind, item_id, note, created_at, removed_at)` — active = `removed_at IS NULL`,
partial-unique on (kind, item_id) where active.
`dashboard_audit(id, ts, action, kind, item_id, detail jsonb)` — append-only; actions this
phase: `flag`, `unflag`. Later phases append proposal decisions.

Migration 043 (`043_dash_notify.sql`) adds `dash_notify_feed()` + AFTER INSERT triggers on
`episodes` / `kg_relationships` / `timeline_events` that `pg_notify('dash_feed', {type,id})`
— the source of the Phase 3 live stream above. No new tables.

## Later phases (reserved paths)

`/dash/api/metrics/{recall,ingestion,corpus}`,
`/dash/api/timeline`, `/dash/api/preferences`,
`/dash/api/behavior/*`, `/dash/api/dream/report`, `/dash/api/graph/*`.
(`POST /recall`'s `debug: true` flag + `/dash/api/recall/history` shipped in phase 2;
`/dash/api/proposals*` shipped in phase 2b; `/dash/api/stream` (SSE) shipped in phase 3 —
see above.)
