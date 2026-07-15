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

## Schema (migration 042)

`dashboard_flags(id, kind, item_id, note, created_at, removed_at)` — active = `removed_at IS NULL`,
partial-unique on (kind, item_id) where active.
`dashboard_audit(id, ts, action, kind, item_id, detail jsonb)` — append-only; actions this
phase: `flag`, `unflag`. Later phases append proposal decisions.

## Later phases (reserved paths)

`/dash/api/stream` (SSE), `/dash/api/metrics/{recall,ingestion,corpus}`,
`/dash/api/timeline`, `/dash/api/preferences`, `/dash/api/proposals*`,
`/dash/api/behavior/*`, `/dash/api/dream/report`, `/dash/api/graph/*`.
`POST /recall` gains `debug: true` (phase 2) — the only change to an existing endpoint.
