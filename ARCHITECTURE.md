# Synapse Architecture

Synapse is a persistent episodic memory layer for AI assistants. It captures Claude Code (and Cursor) session transcripts, structures them into retrievable memory units, extracts entities and facts into a knowledge graph, and exposes everything through an MCP server that Claude Code and claude.ai call during active sessions.

**The core problem it solves**: Claude Code sessions have no memory across restarts. Every new session starts cold. Synapse bridges that gap by indexing everything that happened in past sessions and making it retrievable via hybrid search that fuses reranked episodes with knowledge-graph facts. (Recall makes two Voyage API round-trips — query-embed and cross-encoder rerank — so it is not instant: roughly 1.5s warm, rising to a few seconds on a cold first call after the server has been idle. Measured on the homelab prod instance.)

**One-paragraph mental model**: A Claude Code `Stop` hook pushes the not-yet-shipped part of each session transcript to an HTTP `/ingest` endpoint. Episodes land in Postgres, get grouped into overlapping chunks, and the chunks are the substrate the knowledge-graph extractor mines for entities and facts (written to a knowledge graph in Postgres). At recall time, the MCP server runs a reranked episode leg (BM25 + vector), a KG fact leg, a notes leg and a web-research leg in parallel (plus the bitemporal history of the facts it serves), and returns a compact structured result. There is **no Logfire polling, no summary layer, and no served chunk bucket** — all three were retired during 2026.

> **Status note (2026-06-05):** This document was reconciled against the live code after a large architectural drift. The headline changes from the previous version: (1) ingestion moved from Logfire SQL-API polling to a Stop-hook `/ingest` push; (2) KG facts are extracted from **chunks**, not summaries; (3) the **summary/synth-document layer is retired**; (4) the **chunk bucket is no longer served** by `recall()` — episodes are served via a deep-fetch + cross-encoder rerank; (5) a **web-research ingestion subsystem** was added; (6) halfvec HNSW vector indexes shipped; (7) edges are **bitemporal** (`t_created`/`t_valid`/`t_invalid`); (8) the Docker migration is done — all services run in Docker on the server host.
>
> **Status note (2026-06-24):** Reconciled again for the FalkorDB→Postgres KG cutover and the public-release plumbing. (1) the **knowledge graph moved into Postgres** (`kg_entities`/`kg_relationships`), FalkorDB decommissioned (#67) — §10 rewritten; (2) the **communities bucket was retired** with that cutover; (3) the MCP server runs **standalone `fastmcp` 3.4.2 with MultiAuth** (machine bearer + GitHub OAuth), which replaced the hand-rolled OAuth proxy (#166); (4) skills are **hosted in Postgres and served as `skill://` resources** with two-way sync (§7.6); (5) **26 migrations** (was 16); (6) a Claude Code **plugin + marketplace** is the distribution path.
>
> **Status note (2026-10-15):** Reconciled after a verified code review and its fix pass ([HANDOFF.md](HANDOFF.md) has the rollout). (1) every turn, from `/ingest` or any backfill, is stored by **one writer** (`ingestion/episode_writer.py`, §4.1); (2) the poller is **one process with two threads** (§4.5); (3) web capture is **server-side**, by call shape, from the records `/ingest` receives (§8); (4) recall serves **notes and web buckets behind reranker-calibrated floors**, facts as `f:<bigint>`, and an optional chunk candidate source (§6); (5) the dream lanes share one **window and cursor** plumbing, and all three are on by default with kill switches (§11); (6) the LLM client is four modules (§2); (7) **15 optional migrations (062-076)** follow 061 (§9).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack](#2-tech-stack)
3. [Document Types](#3-document-types)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [Knowledge Graph Extraction](#5-knowledge-graph-extraction)
6. [Recall Pipeline](#6-recall-pipeline)
7. [MCP Server](#7-mcp-server)
8. [Web Research Subsystem](#8-web-research-subsystem)
9. [Database Schema](#9-database-schema)
10. [Postgres Graph Model](#10-postgres-graph-model)
11. [Nightly Maintenance & Dream Pipeline](#11-nightly-maintenance--dream-pipeline)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Configuration](#13-configuration)
14. [File Structure](#14-file-structure)
15. [Operational Notes](#15-operational-notes)
16. [Known Gaps & Planned Work](#16-known-gaps--planned-work)

---

## 1. System Overview

```mermaid
graph TD
    CC[Claude Code / Cursor\nactive session] -->|Stop hook: records past the cursor| ING[POST /ingest\nFastMCP custom route]

    ING -->|episode_writer guards\nappend at max seq+1| EP[episodes\nPG table]
    ING -->|enqueue content_type=episode| EQ[extraction_queue]

    POL[synapse-poller\nmaintenance + extraction threads] -->|sliding window| CH[chunks\nPG table]
    CH -->|enqueue content_type=chunk| EQ
    POL -->|voyage-4-large| V[embeddings\nhalfvec HNSW]
    POL -->|claim FOR UPDATE SKIP LOCKED| EQ
    EQ -->|LLM extraction\nfrom CHUNKS| FDB[(Postgres KG\nkg_entities / kg_relationships)]

    MCP[synapse MCP server\nstreamable HTTP :8765] -->|BM25 + halfvec ANN\n+ Voyage rerank| EP
    MCP -->|vector + BM25 + 1-hop| FDB
    MCP -->|BM25 + vector| WC[web_chunks]

    CC -->|synapse:recall| MCP
    MCP -->|structured result| CC
```

**Data flow summary**:

1. A Claude Code `Stop` hook (`plugin/scripts/ingest_hook.py`; Codex has its own, `plugin-codex/hooks/synapse_stop_hook.py`) fires on every turn and POSTs the part of the raw session JSONL past a durable per-file cursor to `http://<server-host>:8765/ingest`. It runs detached and never blocks the turn. (Any other agent frontend that drives the real `claude` CLI ingests the same way.)
2. `/ingest` parses the records and hands the turns to `ingestion/episode_writer.store_episodes`, which applies the guards (already-stored `span_id`, contamination and harness calls, scratch workspaces, private sessions, delete tombstones, replays) and appends genuinely-new turns as episodes at `max(sequence)+1`, enqueuing each for KG extraction. With `SYNAPSE_WEB_CAPTURE=on` the same call captures the web research those turns carry ([§8](#8-web-research-subsystem)).
3. The **poller** (one process, two threads; *not* an ingester) has a maintenance thread that rebuilds sliding-window chunks, embeds pending episodes/chunks/web chunks via Voyage AI and runs the sweeps, and an extraction thread that drains the extraction queue.
4. KG extraction runs on **chunks** (3–5 turn windows): Haiku extracts entities + facts, resolves entities against the Postgres KG, dedups/contradicts against existing facts, and writes bitemporal relationship edges (`kg_relationships`).
5. When a new session calls `synapse:recall`, the MCP server runs its legs in parallel (episodes, KG facts, notes, web, with the bitemporal history of the served facts) and returns a compact ranked result.

**Where things run**: All Synapse services (poller, MCP server, dream) run in Docker on the **server host** alongside Postgres. Claude Code runs on the **client machine** and reaches the MCP server over the network at `http://<server-host>:8765/mcp` with a machine **bearer token**. The hosted claude.ai connector reaches the same server at a public hostname (e.g. `synapse.example.net`, via a Cloudflare tunnel) and authenticates via **GitHub OAuth** — handled by the server's own `fastmcp` **MultiAuth**, which replaced the former hand-rolled OAuth proxy (#166). Prod compose + `.env` live at a dedicated deploy path on the server host (e.g. `/opt/synapse`).

---

## 2. Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Language | Python 3.12 | |
| Package manager | uv | `uv run` everywhere, no pip/venv activation |
| Relational store | PostgreSQL + ParadeDB image (server host :5432) | published on `127.0.0.1` only; the app services reach it as `postgres:5432` on the compose network |
| Vector search | pgvector, `vector(2048)` columns + **halfvec(2048) HNSW** | HNSW shipped (#104); `embedding::halfvec(2048)` because pgvector HNSW caps plain `vector` at 2000 dims |
| Full-text search | ParadeDB `pg_search` (BM25) | `id @@@ paradedb.match('content', …)` |
| Graph store | PostgreSQL (`kg_entities`, `kg_relationships`) | FalkorDB **decommissioned** (#67) — the KG is Postgres-only (`ingestion.kg_client`, recall via `mcp_server.kg_pg`); bitemporal edges via `t_valid` / `t_invalid` |
| Embeddings | Voyage AI `voyage-4-large`, 2048 dims | separate `input_type` for query vs document |
| Reranker | Voyage `rerank-3-lite` cross-encoder (default since 2026-10-02; WIN1 measured on `rerank-2.5`) | the active ingredient in episode recall (WIN1) |
| LLM calls | `claude-agent-sdk` via `ClaudeAgentClient` / `agent_call` | `claude-haiku-4-5`; subscription OAuth token or `ANTHROPIC_API_KEY` (or an OpenAI-compatible endpoint via `SYNAPSE_LLM_PROVIDER=openai`) |
| MCP server | **standalone `fastmcp` 3.4.2**, HTTP (stateless), port 8765 | `/mcp` (tools + `skill://` resources), `/ingest` + `/recall` custom routes; **MultiAuth** (machine bearer + GitHub OAuth) |
| Ingestion trigger | Claude Code `Stop` hook → HTTP push | **replaced Logfire polling** (now removed) |
| Data models | Pydantic v2 | |
| Postgres driver | psycopg3 (`dict_row`); the server borrows from one shared pool (`ingestion/pg_pool`) | |
| JSON | orjson | |
| Telemetry | Logfire **emit only** | all Logfire *read/poll* paths removed (#108); spans still emitted to the configured Logfire project. The poller and `dream` also instrument pydantic-ai (`include_content=False`): one span per `structured_call` / OpenAI-path run (model, tokens, cost; no prompts) beside the Claude Agent SDK and httpx spans |
| Auth | `fastmcp` MultiAuth: `StaticTokenVerifier` (bearer) + `GitHubProvider` OAuth | machines send one bearer on every endpoint; claude.ai does GitHub OAuth on `/mcp`; **replaced the hand-rolled OAuth proxy** (#166) |
| Skills serving | machine-token `/skills/*` HTTP routes (DSN-free, two-way) + `PgSkillsProvider` `skill://` resources | skills hosted in Postgres (`skills_lane.skill_registry.body` + `skill_files`), materialized to `~/.claude/skills`; per-skill hash + `content_modified_at` reconcile, append-only `skill_history` |
| Distribution | the server serves its own client (`GET /install`) + Claude Code plugin via `.claude-plugin/marketplace.json` | `curl <server>/install \| python3 -` then `synapse login`: one approval enrolls the machine, settings live in `~/.synapse/config.json`, and login wires the Claude plugin and Codex hooks itself |

### The LLM call path (`claude-agent-sdk`)

Extraction does **not** shell out to `claude --print`. All LLM calls go through `claude-agent-sdk`'s `ClaudeSDKClient` + `ClaudeAgentOptions`. The surface is `ingestion/llm_client.py` (`create_llm_client`, `stage_model`, `structured_call`, `parse_with_retry`, and the re-exported `agent_call` and exception classes); the Claude backend is `ingestion/llm_claude_cli.py` (`ClaudeAgentClient`, `agent_call`), the OpenAI-compatible one `ingestion/llm_openai.py` (`OpenAIChatClient`), and `ingestion/llm_common.py` holds the exception classes, `LLM_TRANSPORT_ERRORS` and the transport retry policy. Two shapes over the same SDK:

- `ClaudeAgentClient.messages.create(...)` — a synchronous, `anthropic.Anthropic()`-shaped interface (what the extractor/dedup/edge-date call sites use); it bridges to async via `asyncio.run` and has no retry of its own.
- `agent_call(prompt, *, system_prompt, model, json_schema)` — native async, the only async entry point, used where concurrency matters (the dream lanes). It retries a failed CLI exchange (a crash, no result, more than 300 s) 3 times with a 2–30 s exponential backoff, then raises `LLMUnavailableError`. A CLI error result and `UsageLimitError` are not retried.

Every exhausted transport failure on either backend ends in a member of `LLM_TRANSPORT_ERRORS` (`LLMUnavailableError` on the CLI path, `TransientLLMHTTPError` on the OpenAI path, `UsageLimitError` for a quota), and the extraction stages re-raise those, so during an outage a queue item fails and is retried instead of being marked done with nothing extracted. On the CLI path `UsageLimitError` comes only from a CLI error result, or a short reply with no JSON that contains a quota phrase: a completion that merely mentions a quota is never a quota error.

The CLI is spawned lean (`_cli_options`): `tools=[]`, `max_turns=1`, `max_thinking_tokens=2048`, `setting_sources=[]`, `env={"DISABLE_PROMPT_CACHING": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}`, `extra_args={"no-session-persistence": None, "strict-mcp-config": None}`, no `cli_path` (the SDK's own bundled CLI, pinned by `uv.lock`, so it changes only with a `claude-agent-sdk` bump; the image symlinks it at `/usr/local/bin/claude` so an operator can run it by name) and no `output_format`. A one-shot call needs no tool definitions (about 10K prompt tokens), no prompt-cache write (never read back), no session transcript under `~/.claude/projects` (a copy of conversation text outside Postgres that deletes cannot reach) and no background session-title request. The SDK forwards only an `output_format` of type `json_schema`, so the JSON schema goes into the prompt and the caller validates the reply.

Embedding and rerank clients (`ingestion/embedding.py`) are built by `create_embedder` / `create_reranker`, which return one shared instance per configuration. `interactive=True` (recall) gives up on a Voyage 429 within about 5 s instead of the 3-minute patience of the poller and dream (the OpenAI-compatible backends ignore it). Voyage requests time out after 30 s (query embeds, rerank) or 120 s (document and entity batches; the library default was 600 s); a non-429 transient error retries at most 6 times or 60 s. Single-text query embeds are cached in a 256-entry LRU per backend, and document batches are tokenized in one call.

Auth is a Claude subscription OAuth token (`CLAUDE_CODE_OAUTH_TOKEN`) or an `ANTHROPIC_API_KEY`, consumed by the spawned `claude` CLI rather than by Synapse itself. Scripting a subscription OAuth token against the raw Anthropic API is ToS-prohibited; the SDK/CLI subprocess is the sanctioned path. Extraction is Haiku-only — ~10x cheaper than Sonnet and sufficient for short structured classification.

---

## 3. Document Types

Synapse keeps three **live** document substrates plus the KG. Episodes are the served retrieval workhorse; chunks are the KG extraction substrate; web chunks are the web-research substrate. The summary/dream layer is **retired**.

### 3.1 Episodes — the served workhorse

**One episode per user-initiated exchange.** Each captures a single human turn: a short orientation prefix (the prior assistant response), the new user message, all tool calls and results, and the final assistant response.

```
[context] truncated previous assistant response — orientation anchor
[user]    the human message text
[tool:Bash] command string
[result] tool output
[assistant] final response text
```

**Why human-turn granularity, not per-API-call**: Claude Code makes 3–8 API calls per user turn (each tool result requires re-submission). A 1:1 span→episode mapping produces ~63% assistant-only fragments with no human context and weak embeddings. Grouping by the human-message boundary gives coherent units. Episodes are served directly by `recall()` (BM25 + vector, reranked) — they own both broad and needle retrieval.

**Stored in**: `episodes`. Key columns: `session_id`, `sequence`, `project`, `platform`, `content`, `embedding vector(2048)`, `human_turn`, `assistant_turn`, `span_id` (dedup key), `retrieval_count`, `source`.

### 3.2 Chunks — the KG extraction substrate

**Sliding window of 4 consecutive episodes, step 3 (1-episode overlap).** Only complete windows are written while a session is live, so a chunk is created once and never superseded. A session idle for 2 hours is over: it gets one last window from the next grid position (its first turn is the overlap turn), written once, so a short session (1-3 turns) gets one chunk and the last turns of a longer one are not left out of extraction. Tail chunks are enqueued at priority 10 (catch-up), live chunks at 0. `remember()` archives stay out of chunk text.

**Role**: Chunks are **never served by `recall()`**. They are the substrate the KG extractor mines for facts — a 4-episode window carries enough context (question → attempts → errors → resolution) to extract quality entities and relationships, where a single thin turn ("ok, continue") does not. Measured ~2.66 quality-weighted facts/episode from chunks vs much lower from episode-level or summary-level extraction (#63). They are still embedded and BM25-indexed, and are an optional **candidate source** for recall: with `SYNAPSE_RECALL_CHUNK_CANDIDATES=1` (default off) the top chunks by keyword and by embedding feed their episodes into the pool the episode rerank scores ([§6.2](#62-episode-leg-the-workhorse)). `scripts/ab_chunk_candidates.py` is the A/B that decides whether that source or the chunks' embeddings and BM25 index get deleted.

**Stored in**: `chunks`. Key columns: `session_id`, `start_sequence`, `end_sequence`, `episode_ids JSONB`, `content` (episodes joined by `---`), `embedding vector(2048)`. Idempotent on `(session_id, start_sequence, end_sequence)`.

### 3.3 Web artifacts & web chunks — the web-research substrate

Web-research tool results (WebFetch, Exa, Firecrawl, WebSearch and anything else that fetches a public page) are captured as first-class rows so past research is recallable: server-side, from the records `/ingest` receives, when `SYNAPSE_WEB_CAPTURE=on`. A `web_artifacts` row is the captured page/search/job; `web_chunks` are its embedded slices (with a contextual-retrieval prefix). The `web` bucket in `recall()` is served from `web_chunks`. See [§8](#8-web-research-subsystem).

### 3.4 Knowledge graph (Postgres)

`kg_entities` rows and bitemporal `kg_relationships` fact edges (migrated out of FalkorDB in #67). The KG is the relational/multi-hop specialist served as a side bucket (`facts`), never the primary retrieval path. See [§10](#10-postgres-graph-model).

### 3.5 Timeline events — the episodic date log

`timeline_events` (schema 033) is the EPISODIC complement to the semantic KG: an append-only, time-ordered log of dated **point-events that happened** ("shipped X", "decided Y", "fixed Z"), distinct from the KG's deduplicated states-with-duration. Two feeders:

- **Chat gate** (`ingestion/timeline_gate.py`): one small LLM call per ingested turn asking "did something actually HAPPEN?" — most turns are discussion and emit nothing. The gate reads the compact turn (the ask, the final assistant text and the last tool lines, with no result bodies), and writes nothing for a turn deleted while it ran (the write takes a `FOR KEY SHARE` lock on the source episode, which the dashboard delete and the private-session purge serialize against). Emitted events are naked past-tense facts (no actor field; the verb carries decides-vs-executes), self-contained, dated to the turn with validated resolution of "last Tuesday"-style mentions.
- **Git feeder** (plugin `git_feeder.py`): commits pushed as `source='git:<project>'` events, author-dated.

Idempotency rides on `UNIQUE(source, source_ref)`. The gate is on by default and `SYNAPSE_TIMELINE_GATE=0` stops it. Three write-time quality mechanisms: the confirm-merge and the quoted-material rule are always on; domain scoping follows `SYNAPSE_PERSONAL_SCOPE`:

- **Confirm-merge dedup** (schema 037): a re-told happening (same real-world event narrated again later, often with a differently-resolved date) merges into its existing row instead of inserting — nearest same-project candidates within ±14d / cosine < 0.20 go to an LLM confirm that reads **both full source turns** and must answer SAME in **both presentation orders**. Non-destructive: a merge appends the re-telling turn's `ep:<id>` to `reported_refs` (schema 064, probed; a retried merge finds its own ref and changes nothing) and moves `t_valid` to the earliest, with no deletes. The old `reported_count` is no longer written. Measured before shipping: snippet-only judging is unreliable (79% on 120 stratified pairs — the discriminating info lives in the source turns); the shipped form leaves ~2% residual false-merge, all recoverable. Distance alone cannot be the merge key in either direction (recurring chores sit at d≈0.02; date-split retellings sit further out).
- **Domain scoping** (schema 038): the gate labels each event `personal` (the user's own life) or `technical` (engineering/work); git events are technical by construction; NULL = unlabeled and fails open. An explicit `group_id="personal"` on the read side filters the timeline to life events. Measured motivation: for a reworded personal query, cross-domain junk sat *closer* in embedding space (0.704) than the true personal events (0.748+), so no relevance floor can do this job — only a domain label can. The default/technical scope stays unfiltered so callers that never set `group_id` lose nothing. With `SYNAPSE_PERSONAL_SCOPE=0` the gate is never asked for a domain and every event is stored `technical`.
- **Quoted-material suppression**: happenings narrated inside pasted third-party content (someone else's email, a meeting/interview/legal transcript, an article with first-person narration) are never emitted as the user's events — the gate distinguishes content the user *asserted* from content the user *quoted*. The user's own work on the material, and third-party statements *about* the user, stay eligible. Probed on labeled cases before shipping: third-party leaks 2/8 → 0/8 with no legit-event loss.

**Serving**: `recall()` does not retrieve timeline events and there is no `recall_timeline` tool (both removed 2026-08-07: 39 helpful against 92 noise citations lifetime; the store, its gate and the board block stayed). The events are read by the board's **Last 7 days** block (`GET /context`; `timeline_routes._recent_events`, a pure time-window read of the newest salient events from every project the caller may see, limit 10, under a per-project-day cap, a day being the user's local day, each line carrying its `t:N` id) and by the dashboard's Timeline page and search. The board is the only reader, so the served `t:` ids are logged in `recall_metrics` (`kind='board'`, `served_ids.events`), which is what `recall_feedback` labels join to when it is time to decide whether the gate earns its LLM call. Sibling gate: `ingestion/preferences_gate.py` turns [user] text into standing preferences, skips a turn whose [user] text is under 10 characters, and does not count a preference twice when the same turn is retried (matched through `source_ref = ep:<id>`); `SYNAPSE_PREFS_GATE=0` stops it.

### 3.6 Retired: summaries & dream documents

`synth_documents` (`doc_type='summary'|'dream'`) is a **frozen legacy table** (~669 stale rows). Nothing reads or writes it any more: `recall()` never served it, the poller no longer embeds it, and the extractor has no `summary` branch (the queue's content types are `episode | manual | chunk | web_chunk`). The table itself awaits a snapshot + drop. Treat the summary layer as **dead**; the episode leg and KG facts absorbed both its broad-recall and relational jobs.

---

## 4. Ingestion Pipeline

Episodes enter Synapse through `/ingest`: a `Stop` hook (Claude Code's or Codex's) POSTs the transcript records it has not shipped yet. **There is no Logfire polling** — `ingestion/logfire_client.py` no longer exists, and Logfire is used only to *emit* telemetry.

```mermaid
flowchart TD
    HK[Stop hook\nplugin/scripts/ingest_hook.py] -->|raw JSONL past the cursor| ING[POST /ingest\nmcp_server/ingest_route.py]
    ING --> PARSE[JSONLParser\nsame parser as disk sweep]
    PARSE --> WR[episode_writer.store_episodes\none transaction, per-session lock]
    WR --> GUARD{turn refused\nby a guard?}
    GUARD -- yes --> SKIP[skip]
    GUARD -- no --> APPEND[sequence = max+1\nupsert_episode]
    APPEND --> ENQ[enqueue_extraction\ncontent_type=episode]
    WR -.->|after commit, if SYNAPSE_WEB_CAPTURE=on| CAP[capture web research]

    subgraph Poller maintenance thread
        REB[rebuild_pending_chunks\nwindow=4 step=3] -->|on_new| ENQC[enqueue content_type=chunk]
        EMB[embed_pending\n60 s budget]
    end
    subgraph Poller extraction thread
        DRAIN[drain_extraction_queue\nclaim FOR UPDATE SKIP LOCKED]
    end
```

### 4.1 The `/ingest` endpoint

A Starlette custom route on the FastMCP server: `@mcp.custom_route("/ingest", methods=["POST"])`. The Stop hook posts `{"records": [...], "source": "hook", "project": "<label>"}`; the body is parsed with the same `JSONLParser` the disk-sweep backstop uses, and the turns go to `ingestion/episode_writer.store_episodes`, the one place a parsed turn becomes a stored episode. `/ingest` and every backfill (`python -m ingestion.backfill --source {claude-code,codex,cursor,claude-ai,chatgpt}`, the single driver over the five parsers) end there, so a turn meets the same guards however it arrives. Per turn, in order:

- **Span index** — a turn's stable last-record UUID is its `span_id`; the session's `(seen span_ids, max_seq)` index is loaded once per session. A turn already stored is skipped, and so is one with no span_id to key it (counted as `unkeyed` on every path). The tail's own positional sequence is discarded (a tail renumbers from 1 and would collide).
- **Contamination and harness calls** (`ingestion/contamination.py`) — drops PII-bearing third-party transcript payloads (produced by another agent project sharing the same hooks) and Synapse's own extraction and judge prompts, so the eval cannot eat itself.
- **Scratch workspaces** (`SYNAPSE_SCRATCH_PROJECTS`) — throwaway test projects.
- **Private-mode filter** (schema 050) — `PrivateSessions.is_private` drops every turn whose `session_id` sits in `private_sessions`. The writer applies it to every source, so a session taken off the record can't be ingested by a path that bypasses the hook (catch-up sweep, disk backfill, a re-import months later). Rows are permanent by design: the transcript survives on disk, so deleting the row would re-expose exactly the turns the user hid. Private mode covers the session **from the toggle onward**: `PUT /private-sessions/{id}` reports `stored_turns` (what is already stored for that session), and `?purge=1` deletes those turns with the dashboard's episode-delete cascade plus the preferences they gave birth to and the web pages captured in the session, in one transaction, audited as `purge_private_session` in `dashboard_audit`. The flag commits before the purge. A purge needs a full-trust device, or a restricted device purging a session whose every turn carries its own 057 stamp; anyone else, the root token included, gets 403 with the flag still set and nothing deleted.
- **Delete tombstones** (`deleted_turns`, schema 073, probed) — the delete cascade records one row per deleted turn (by `span_id`, and by content md5 within the project or working directory) in the delete's own transaction, and the writer drops a matching turn, so a re-shipped transcript, sweep, import or backfill does not bring a deleted turn back. Tombstones are permanent, like `private_sessions`. Without the table a delete works as before and ingest checks nothing.
- **Replays** (schema 036, always on — there is no switch) — a `span_id` already stored under another session, or a turn whose content is byte-identical (`md5(content)` index) to an episode already stored in the same project or from the same working directory under any label (so a replay whose first copy carries an older label for that folder, before `scripts/relabel_projects.py` merges them, is still caught). The span guard is per-session, so a retry session or a re-import replays turns under a fresh session and span id and sails through it; ~4.6% of episodes on a real corpus were exact replays before this guard.

A new turn is appended at `max(sequence)+1`, upserted, and queued for extraction (the queue row names the episode and stores no text, [§4.4](#44-extraction-queue-enqueue-gating)). The whole call is **one transaction on one connection**: a failure rolls back every write of the call, and the client's retry skips by `span_id` what did land. It holds each session's advisory lock `pg_advisory_xact_lock(hashtext(session_id))` (sorted, so overlapping calls cannot deadlock) from before the span-index read to the commit, the same lock `remember()`'s `append_episode` takes, so a `remember()` arriving meanwhile waits for the commit and takes the next sequence instead of being overwritten by the upsert.

After the commit, in a transaction of its own, the writer captures the web research of those turns, and of turns already stored (a re-ship captures what an earlier run missed), when `SYNAPSE_WEB_CAPTURE=on` ([§8](#8-web-research-subsystem)); a capture failure is logged and counted, never raised. The reply is `{"status", "ingested", "captured", "capture"}`, where `capture` says whether the server captures web research (absent on an older server): it is what tells a client whether to ship sub-agent transcripts and saved tool outputs. `POST /capture/dry-run` (client gate, the `/ingest` body) runs the detector over posted records and reports what it would capture without writing anything. `/ingest` runs in a threadpool so the async route never blocks.

**Project labels.** Every client labels a session's project with one resolver,
`plugin/scripts/synapse_project.py` (the Codex plugin loads the same file): a
`.synapse-project` marker file (its first non-comment line; searched from the working
directory up to the git toplevel, and in a linked worktree also at the main checkout's
root), else the git toplevel's name (a linked worktree, including Claude Code's
`.claude/worktrees/<id>`, is its main checkout; a bare repository's worktree is the
repository's name), else the folder's name. Both separators, case kept. The ingest hooks
send it as `"project"`, the board asks for that project, and the skills sync syncs
`project:<label>`, so a session started in a subfolder or a worktree is filed, shown and
served as its repository. A body without a label (a catch-up of a transcript whose folder
is gone, an older plugin) is labelled server-side from the transcript's `cwd` by
`ingestion/project_label.py`: the last component of the path, either separator. A restricted
device's allowlist matches labels case-insensitively (`resolve_caller` widens it to every
stored casing of its entries). `scripts/relabel_projects.py` maps labels stored before this
onto canonical names (`--dry-run` first; the alias file stays outside the repository).

### 4.2 The Stop hook

`plugin/scripts/ingest_hook.py` fires on every turn; the Codex analog, `plugin-codex/hooks/synapse_stop_hook.py`, shares its retry bookkeeping (`synapse_ingest_retry.py`). It ships the transcript records past a **durable per-file cursor** (`DATA_DIR/ingest_cursors.json`): the byte offset of the start of the last shipped turn, never EOF, because Claude Code may write a turn's final message after Stop fires, so the last turn is re-shipped once. The cursor advances only after a successful POST, so a failed POST self-heals: the next Stop hook, or the SessionStart `--catchup` sweep, re-ships from the same offset and the server skips every turn it already stores (a cursor past EOF means the file was truncated and resets to 0). A transcript with no cursor entry yet ships its last 400 records (`TAIL_RECORDS`, a constant; stop mode) or the whole file (catch-up mode), and any POST is cut at turn boundaries, 400 records per chunk, so cost stays proportional to the new bytes (a stat plus a seek from the cursor). The POST runs detached (`start_new_session=True`) and the hook always exits 0 — **it never blocks or fails a turn**.

The catch-up sweep (shared by the Claude and Codex hooks through `synapse_ingest_retry`: a 3-day window, at most 20 files per session start, a 300 s grace for a file a live session is still writing, all constants) ships any transcript modified within the window whose cursor lags its size, plus every file whose last upload failed (marked `pending` in the cursor file) however old, so an outage longer than the window loses nothing; history from before the plugin was installed is `synapse-import`'s job. After repeated failures the SessionStart parent prints one "uploads have been failing since …" line. Enqueue-gating on new `(session, span_id)` fixed an earlier O(turns²) re-extraction loop (#61, #105, #106).

**Private mode** short-circuits the hook before any POST: `private_mode.py on <session_id>` writes a marker file at `~/.synapse/private/<session_id>`, and `_ship` skips the whole transcript while that marker is live — on the Stop path and the catch-up sweep alike. Markers expire after 12h (ignored *and* deleted) and are removed by a `SessionEnd` hook, because a session id is never reused and an orphaned marker would otherwise be permanent. The check is fail-safe in one direction: a marker directory that exists but can't be read counts as private, since silently capturing a session the user believes is off the record is the unacceptable failure and a deferred turn is not (the cursor doesn't advance). The marker is only the fast local half — the durable half is the `private_sessions` row the same CLI writes over `PUT /private-sessions/{session_id}`, which is what §4.1 enforces. Private mode covers the session from the toggle onward: `private_mode.py on <id>` (or `synapse private on <id>`) prints how many turns of it are already stored (`stored_turns`) and never claims a deletion it did not make, and `on <id> --purge` also deletes them through `PUT /private-sessions/{id}?purge=1`. Against a server that predates the purge, or one that refuses it (403), `on --purge` exits 1 with private mode ON and nothing deleted.

### 4.3 Chunk construction

Lives in `ingestion/chunks.py`. `_CHUNK_WINDOW = 4`, `_CHUNK_STEP = 3` → 1-episode overlap. While a session is live only complete windows are emitted (`range(0, n - window + 1, step)`); the trailing partial window waits until it is full. A session idle for 2 hours (`_IDLE_AFTER`) is over, so its tail is chunked as it stands: one last window from the next grid position, whose first turn is the overlap turn, written once (a session of 1-3 turns gets a single chunk). `Database.sessions_with_pending_chunks` flags those sessions in SQL. When the poller rebuilds chunks, an `on_new` callback enqueues each genuinely-new chunk for extraction with `content_type="chunk"` — exactly once, at birth, at priority 0 for a live chunk and 10 for a flushed tail. Idempotent via `ON CONFLICT (session_id, start_sequence, end_sequence) DO NOTHING`.

### 4.4 Extraction-queue enqueue gating

A queue row **names its source and stores no text**: `episode_id` for `episode` and `manual` items, `chunk_id` for chunks (schema 062, probed), `web_chunk_id` for web chunks. The worker reads the text by key, and finishes an item whose source is gone (a turn deleted since it was queued) without work. Content types are `episode | manual | chunk | web_chunk`. Finished rows keep their status but lose their text after 7 days (a daily sweep); before 062 a chunk row keeps its text. `enqueue_extraction` is the idempotency backstop, with branches keyed by item shape (all checking only `status IN ('pending','processing')`):
- **episode/manual** (have `episode_id`) — dedup by `episode_id`.
- **chunk** — `ON CONFLICT` on the unique `chunk_id` index (schema 062); before it, by exact `(session_id, content)`, not `(session_id, content_type)`, which would collapse every chunk of a session into one row. A chunk's queue row follows the chunk (`ON DELETE CASCADE`), and an episode's follows the episode foreign key.
- **web_chunk** — enqueued by the poller's web stage (`enqueue_web_extraction`, [§8](#8-web-research-subsystem)), only for artifacts that opted in to extraction.

### 4.5 Poller: two threads

The poller (`ingestion/poller.py`) owns **post-ingest work only** — it does not ingest episodes. It is one process with two threads, so extraction never waits behind maintenance and maintenance never waits behind a slow LLM call:

- **Maintenance thread** (`Maintenance.run`; its own `(db, pipeline)` stack). Every `POLL_INTERVAL_SECONDS` (300 s) it runs `Maintenance.cycle`, a list of stages each in its own `try` block: `rebuild_pending_chunks`, web chunking (`chunk_web_artifacts`, 200 pages a cycle), `embed_pending` (episodes, then chunks, then web chunks; 500 rows a batch, newest first, 60 s budget a cycle, each batch one `UPDATE … FROM unnest`), web context (`contextualize_web_chunks`, 20 pages or 180 s a cycle; it runs *after* embedding so a web backlog never delays a new turn's vector, which puts a page's chunks in vectors one cycle after their context prefix), web extraction enqueue, and the stale-claim sweep (`release_stale_claims`, 45 min threshold, which now bounds one item rather than a batch). Once a day: the orphan-entity sweep and the finished-queue-text sweep.
- **Extraction thread** (`Poller._extract`). `drain_extraction_queue(batch_limit=2*SYNAPSE_DRAIN_CONCURRENCY)` in a loop. Items run on a pool of `SYNAPSE_DRAIN_CONCURRENCY` threads (default 4; 1 is the serial kill switch, a pool of one), each with its own Postgres connection and LLM-client stack, and a row is **claimed only when a pool thread is free**, so a killed worker strands at most one claim per thread. New ingest (priority 0) drains before catch-up (10), pending before retries. It pauses 3 s after a full batch and 60 s otherwise. A `UsageLimitError` releases the claims and backs this thread off 5, 10, … 30 min (jittered) while maintenance keeps embedding.

Both threads touch a heartbeat file (`/tmp/poller.alive` for extraction, `/tmp/poller.maintenance.alive` for maintenance, which is touched even on a replica that does not hold the maintenance lock), and the compose healthcheck passes only while both are under 120 minutes old. SIGTERM/SIGINT stop both threads (in-flight items finish, claims not started are released, both threads are joined); a thread dying of an unexpected error stops the other and exits the process nonzero, so compose restarts it. The compose pollers have `stop_grace_period: 5m`; Watchtower stops containers with its own timeout, so set `WATCHTOWER_TIMEOUT` where it runs to match, or an auto-deploy kills in-flight items and they wait out the 45 min sweep. `SYNAPSE_DRAIN_ONLY=1` (the retired `poller-drain` service) is honoured for one release: the replica runs only the extraction thread and logs a deprecation warning.

### 4.6 Horizontal scaling: atomic claim

`docker compose up -d --scale poller=N` scales **extraction only**. Every replica runs an extraction thread and claims a disjoint slice of the queue atomically (#37):

```sql
UPDATE extraction_queue
SET status = 'processing', claimed_at = now()
WHERE id IN (
    SELECT id FROM extraction_queue
    WHERE status = 'pending'
       OR (status = 'failed' AND attempts < %s AND last run older than FAILED_RETRY_MINUTES)
    ORDER BY (status <> 'pending'), priority ASC, enqueued_at ASC   -- new work first, then the priority lane
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

Maintenance is single-instance: each replica's maintenance thread tries `pg_try_advisory_lock(hashtext('synapse.poller'), hashtext('maintenance'))` every interval, on its own connection. It is the two-key form, so it can never collide with the per-session single-key locks the episode writer takes. The holder cycles; the others retry and take over when the holder's connection ends.

`priority` (schema 015) keeps a bulk backfill (priority 10) from starving fresh turns (priority 0). `claimed_at` (schema 016) lets `release_stale_claims(older_than=45min)` recover crashed-worker orphans without clobbering a live peer mid-item. A `failed` row is re-claimed after 30 minutes, up to 5 attempts, so a fixed configuration picks its turns up on a later cycle.

### 4.7 Embedding

Voyage `voyage-4-large`, 2048 dims. `input_type="query"` at recall time, `"document"` for all ingestion content. Token-aware batching packs requests under Voyage's 120K-token cap (target 100K, max 1000 items, 24K-char per-item cap); a document batch is tokenized in one call. `embed_pending` writes each batch back with one `UPDATE … FROM unnest`, covers episodes, chunks and web chunks, and has no chunk cap. Retry and timeout policy (batch versus interactive) is in [§2](#the-llm-call-path-claude-agent-sdk).

---

## 5. Knowledge Graph Extraction

KG facts are extracted from **chunks**, not summaries. `ExtractionPipeline.process_item()` orchestrates per queue item; the heavy per-fact stages are **batched into one LLM/SQL call each** across all of an item's facts.

```mermaid
flowchart TD
    Q[extraction_queue\nclaimed item] --> CT{content_type?}
    CT -- episode --> S2E[per-turn gates only\ntimeline + preferences → return]
    CT -- chunk --> S2[Stage 2: DeterministicExtractor\nregex: files, URLs, errors, tools]
    CT -- manual --> S2
    CT -- web_chunk --> S3W[Stage 3: web prompt variant\nattribution firewall]
    S3W --> S4

    S2 --> S3[Stage 3: LLMExtractor\nHaiku: entities + facts JSON\nCombinedExtraction cross-ref valide]
    S3 --> S4[Stage 4: EntityResolver\nnormalized-name → LSH → vector → batched LLM confirm]
    S4 --> S5[Stage 5: write_nodes\ninsert new entities with their embedding\nupdate matched ones: summary + supertype only]

    S3 --> S6A[Stage 6a: embedding_filter\nfact ANN + pair check, no LLM]
    S6A --> S6B[Stage 6b: BATCHED LLM confirm\nduplicate / contradict / none]
    S6B --> S7[Stage 7: write_edges]
    S7 --> ED[EdgeDateExtractor batch\nt_valid only]
    S7 --> WR[batched INSERT + UPDATE\nbitemporal kg_relationships]
```

### 5.1 content_type gating

```python
if content_type == "episode":
    self._timeline_gate.process(item)                     # per-turn gates (§3.5): no KG facts
    self._preferences_gate.process(item)
    return
elif content_type == "chunk":
    det_entities = self._stage2_deterministic([{"content": content, "metadata": {}}])
    llm_result = self._stage3_llm(content, det_entities)  # full extraction on the window
elif content_type == "manual":                            # one remember() note
    det_entities = self._stage2_deterministic([{"content": content, "metadata": {}}])
    llm_result = self._stage3_llm(content, det_entities)
elif content_type == "web_chunk":                         # a slice of a captured page (§8)
    llm_result = self._llm.extract_web(content, det_entities, web_provenance)
else:
    raise ValueError(...)                                 # no `summary` type any more
```

The item's text is read from the row it names (the turn, the chunk, the remember() note), not from the queue.

**Chunks are the live fact source.** An episode item runs only the two per-turn gates (timeline events, standing preferences) and returns without extracting facts. The `manual` branch serves the `remember()` tool; the summary content type is gone.

### 5.2 Stages

- **Stage 2 — `DeterministicExtractor`**: regex/metadata extraction of `File` / `URL` / `Issue` / `Tool` entities. No LLM.
- **Stage 3 — `LLMExtractor`**: one Haiku call → residual entities + facts, validated through a `CombinedExtraction` Pydantic cross-ref validator that drops facts referencing undeclared entities.
- **Stage 4 — `EntityResolver`**: resolve each entity to an existing node or mint a UUID. Strategy ladder: exact `normalized_name` (an indexed lookup; entities with a NULL `normalized_name` are invisible to it until `scripts/backfill_normalized_name.py` has run) → entropy gate → MinHash/LSH → **batched** Haiku "same entity?" confirm. It embeds once, only the names the deduper did not settle exactly, and hands the vectors to Stage 5. Each entity is routed to the `personal` or `technical` graph **by its name only**; a fact whose endpoints disagree has the odd endpoint moved into the item's default group before Stage 4 (logged at INFO, "Moved N entities…"), so facts are no longer dropped as cross-group. See [§5.3](#53-entity-resolution).
- **Stage 5 — write nodes**: INSERT new (non-orphan) entities into `kg_entities` with their embedding. A matched entity is never renamed and its embedding is never rewritten; it only takes a longer summary and a known supertype, and its supertype is re-derived from its own stored subtype through `kg_entity_types` whenever the taxonomy map is loaded (an unknown subtype gives `other`), never from the mention's extracted type.
- **Same-batch twins** (`ingestion/extraction_twins.py`, 2026-10-09): before Stage 6, a fact the batch states twice is written once. Stage 6 only compares against edges already in the graph, and nearly every twin is one sentence fanned out over each entity it names, so it links a different entity pair. Twins need an identical identifier signature (every token with a digit, path, URL, identifier shape, capital letter, or negation), content words that differ on at most one side (a reorder or added detail, never a substitution), and cosine >= 0.90. Cosine alone cannot do this: about 19 in 20 non-identical within-batch pairs at >= 0.95 differ in a number or a name. The most specific twin survives, and is user-attributed if any twin was.
- **Stage 6a — embedding filter**: no-LLM candidate find. A `pair_pool` (the 8 live same-pair edges nearest to the new fact, `_PAIR_POOL_LIMIT = 8`, ranked in SQL; the lookup is skipped when either endpoint is a new entity) + a `semantic_pool` (RRF of vector + BM25 over fact text). `_SEMANTIC_POOL_LIMIT = 8` (#121, down from 20 — the rank 9–20 tail was noise and pushed the dedup prompt to ~30K tokens).
- **Stage 6b — batched LLM confirm**: ONE Haiku call classifies every fact-with-candidates as `duplicate` / `contradict` / `none`. A reply with no usable verdicts gets one validation retry, then fails closed: the facts are written without verdicts and a WARNING says so (`stage 6b: no usable verdicts for N fact(s) after a retry`); the `llm_ok` span attribute counts verdict-less writes.
- **Stage 7 — write edges**: `EdgeDateExtractor.extract_batch` (one call for all facts' `valid_at`; it asks for noon UTC and a date with no time is stored at 12:00 UTC, so the user's local day survives the round trip), then `invalidate_edges_batch` + `create_edges_batch` (one batched SQL statement each) inside one transaction. Retirements come only from Stage-6b verdicts; there is no separate contradiction detector (`ContradictionDetector` and the single-fact resolve, dedup and edge-date prompts were deleted in 2026-10). Every edge a new fact retires is linked to its successor (`invalidated_by`, schema 028): the new edge, or the confirmed existing duplicate when the write is skipped (a drop-in replacement). Until 2026-10 Stage-6b verdicts were written with no link, so a correction found at extraction was invisible to recall's successor legs. Retirement is dated by the correcting fact's evidenced valid-from (in-text date, else the conversation timestamp), clamped by the writer to no earlier than the old edge's own `t_valid`; never the write clock.

Batched stages (one call across the item's facts): Stage-4 confirm, 6b, EdgeDateExtractor, Stage-7 writes/invalidations (#38, #40, #42, PR #83/#94). Per-item single passes: 2, 3, 5, 6a. All KG reads and writes go through `KGClient(db_url)` (`ingestion/kg_client.py`), one class that owns one connection and retries a connection error once on a fresh connection, so a Postgres restart does not cost a failed attempt per worker.

### 5.3 Entity resolution

**Write-path** (`EntityResolver`): pgvector returns cosine *distance* → `similarity = 1 - distance`. `< 0.85` → new node; `≥ 0.95` → trust the embedding, skip the LLM; `0.85–0.95` → batched Haiku confirm. On any error, every pending entity becomes a **new** node — a spurious duplicate is cheap, a wrong merge corrupts the graph.

**Semantic identity merge** (#48) handles lexically-*different* same-entities. Three lanes: exact-name (auto), moderate (cosine ≥ 0.95 **and** Levenshtein ratio ≤ 0.30 → Haiku confirm), and semantic (embedding-only, no name gate — the only lane that can reach `User` ≡ the user's real name). Winner is deterministic (most edges → oldest → lowest uuid); merge re-creates every edge with the full bitemporal quad.

> **Open gap (#49):** the semantic lane is **detection-only — it never auto-merges**. Hub-touching pairs (degree ≥ 50) are surfaced for human review because a wrong hub merge is irreversible. So identity hubs still re-fragment: `User` / the user's first name / the user's full name / `The User` persist as separate nodes (~2065 edges across 6 nodes). External research independently names entity resolution *the* hard problem of KG memory — this is the load-bearing fix to keep cheap. See [§16](#16-known-gaps--planned-work).

### 5.4 Bitemporal edges

`kg_relationships` edges are bitemporal (#47). The canonical quad. The FalkorDB-era single-field mirrors (`created_at`, `valid_at`, `invalid_at`, and `entity_type` on entities) are no longer written by anything and read by nothing; they still exist, and a later migration can drop them once no older image can run (the previous image's INSERTs name them). Rows written by this code have them NULL; migration 072 backfills `t_created` / `entity_subtype` for rows that only ever had the legacy column.

- `t_created` — write time, never updated.
- `t_valid` — when the fact began being true; from `EdgeDateExtractor` or `now()`; a date with no time is stored at noon UTC.
- `t_invalid` — when the fact was superseded by a later fact; **always NULL on create**, set only by Stage-6b. The `EdgeDateExtractor` returns `valid_at` only (no `invalid_at`): a fact recording a dated happening or a bounded past state ("the outage ran Aug–Sep", "User told the assistant on 08-29 to stop X") is a true statement for good, and writing it already-retired dropped it out of the live index with no successor link (616 such edges in prod). Currency for the reader rides `t_valid` and the 058 fields.
- `t_expired` — reserved for future selective forgetting; nothing writes it yet.

Contradictions **invalidate the old edge** (an `UPDATE` setting `t_invalid`) and always write the new one — they never block it. All live-edge read filters use `t_invalid IS NULL`. This gives a queryable history of fact evolution (surfaced by recall's `history` leg).

**Evidence currency (schema 058).** `t_valid` says when a fact *became* true; nothing said when it was last *confirmed*, so a habit extracted once stayed a live present-tense fact until someone contradicted it. Three columns fix the bookkeeping without rewriting any fact text (the pipeline never invents end dates — `extract_edge_dates` forbids it): `ongoing` (the extractor's call: the fact asserts an activity, usage, or state that can lapse silently — a habit, a tool in use, a job — as opposed to a dated happening, a permanent trait, or a *description of an artifact*: "the profile lists rock climbing" is about the profile), `last_supported_at` (the conversation time of the most recent USER-attributed statement asserting the proposition; set on create and moved forward on reinforcement only when the extractor attributes the restating fact to the user and the chunk is new provenance — assistant repeats and replayed chunks never refresh it) and `last_supported_by` (that statement's episode). The extractor emits `ongoing` and `attribution` (`user` / `assistant` / `third_party`) per fact; a missing or unrecognised attribution is `unknown` and never counts as support. NULL on every pre-058 edge: unknown, not false. Recall serves `ongoing` and `supported` beside `date` ([§6.1](#61-return-shape)).

`EdgeDateExtractor` (#46) skips the LLM for any fact with no date signal via a `_TEMPORAL_RE` regex prefilter (years, ISO/slash dates, month/weekday names — "may" only when used as a month —, "yesterday/since/until", durations, "started/stopped/no longer/used to", …) — a non-temporal fact resolves to the `now()` fallback identically, so the call is wasted.

---

## 6. Recall Pipeline

`recall()` fans out over a persistent leg executor (8 workers). Every leg borrows its own connection from the shared `ingestion/pg_pool` pool or makes its own Voyage call, so concurrent psycopg use is safe. The BM25 episode leg starts first so its fetch overlaps the query-embedding call; the embedding then releases the vector episode leg, the KG facts leg, the supersession surface, the notes leg and the web leg, all concurrently. The cross-encoder rerank starts the moment the episode pool is joined and runs while the other legs and the fact bookkeeping are still in flight. Returns a **compact, conditionally-populated** result. Interactive recall builds its Voyage clients with `interactive=True`: any retryable error, a 429 included, gets short backoffs and is given up on within about 5 s, so recall falls back to BM25-only legs and RRF order (each with a `warnings` entry) instead of hanging; the poller and dream keep the 3-minute patience.

```mermaid
flowchart TD
    Q[query] --> BM[episode BM25\nfetch 100, starts first]
    Q --> EMB[embed query\nvoyage-4-large input_type=query]
    EMB --> VEC[episode vector\nhalfvec HNSW, fetch 100]
    EMB --> CHK[chunk sources\nBM25 + vector, default off]
    EMB --> KG[KG facts: vector + BM25 + 1-hop\nRRF k=60, top 12]
    EMB --> SUP[supersession surface]
    EMB --> NOTES[notes: hook KNN\n+ rerank floor]
    EMB --> WEB[web: BM25 + vector RRF\n+ rerank floor]
    BM --> POOL[RRF pool, own session out,\ncut to 100]
    VEC --> POOL
    CHK --> POOL
    POOL --> RR[Voyage rerank-3-lite\n+ recency, echo suppression]
    RR --> PASS[passage mining:\nsecond rerank, top 6 passages]
    KG --> HIST[superseded pairs]
    PASS --> OUT[structured result]
    KG --> OUT
    SUP --> OUT
    HIST --> OUT
    NOTES --> OUT
    WEB --> OUT
```

### 6.1 Return shape

```python
out = {"query": query, "facts": facts}   # always present (facts may be []); each {fact, id, date, ongoing?, supported?}
if ep_items:        out["episodes"] = [...]   # reranked passages, one item per parent episode, cap 6 passages (see role note)
if note_items:      out["notes"]    = [...]   # {id, hook, note, project?}, cap 3
if web_chunks:      out["web"]      = [...]   # {id, context|excerpt, url?, title?, date?}, cap 3
if superseded:      out["superseded_facts"] = [...]  # {fact: old, superseded_by: current}, cap 2: the displaced version of each served fact
if warnings:        out["warnings"] = [...]   # a leg degraded; absent on a healthy call
return out
```

Every served item carries an `id` that `recall_feedback` takes: `e:N` episode, `n:N` note, `f:N` fact, `w:N` web. Facts are served as `f:<kg_relationships.id>` (a bigint, the same id the KG tables key on), and `superseded_facts` entries use the same form; the older `f:<uuid>` form is still accepted everywhere ids are parsed, so feedback on a pre-change serve keeps resolving. Only `e:` and `n:` ids are `fetch()`-able. The retired buckets (`entities`, `timeline`, `chunks`, `summaries`, `communities`) are not returned.

Served episode passages carry a **`role` provenance label** (`"user"` / `"assistant"` / `"mixed"`): episode content is assembled from role-marked parts, but passage compaction slices it into chunks and a mid-block chunk loses its marker — without the label, a reader cannot tell a human-stated fact from the agent's own past output (which may be speculation). The label is recovered deterministically from chunk char-offsets against the parent turn's marker layout; unattributable slices omit the key (fail-open, advisory only). Rank-side provenance *weighting* was deliberately rejected: serve-time rerank score surgery is the historically dead lever here, and most technical needle answers live in assistant halves — a blanket penalty would trade one failure class for a worse one.

Served facts carry `date` = `t_valid` (when the fact became true *as stored*: an in-text date, else the conversation it came from — not the last time anyone confirmed it), `ongoing: true` when the edge is flagged as a claim that can lapse silently, and `supported` = the date the user themself last asserted it (absent = never user-confirmed, or pre-058). Both dates are the user's local day (`SYNAPSE_TIMEZONE`); a value at exactly 00:00:00 UTC is a date-only value and keeps its own date. The reader contract that rides the server instructions and the `recall` description: a fact supports exactly what it states — one mention is not frequency, identity, or current activity; a fact about a profile or document describes the document; an ongoing fact with no recent `supported` is a lead to ask about, not current truth; anything written in the user's voice uses only what they said. `tests/fixtures/reader_contract.json` holds synthetic cases (licensed vs forbidden readings) that double as an eval set.

`recall_metrics.served_ids` records which ids were served per bucket (`{"episodes": ["e:12"], "facts": ["f:41"], "notes": [...], "web": [...]}`, plus `trust`, the echo, duplicate-passage and `n_chunks_cut` counters, and the abstention markers of §6.2); rows written before facts moved to `f:<bigint>` hold the raw edge uuids. `est_tokens` counts everything served including notes, never `warnings`.

### 6.2 Episode leg (the workhorse)

Two sub-legs deep-fetch 100 candidates each, fuse, then a cross-encoder reranks:

- **BM25** (ParadeDB): `WHERE id @@@ paradedb.match('content', %s) ORDER BY paradedb.score(id) DESC`.
- **Vector** (halfvec HNSW): `ORDER BY embedding::halfvec(2048) <=> %s::halfvec(2048)` — the cast must match the index expression *verbatim* to be served by HNSW. `SET hnsw.ef_search = 200` on every pooled connection (default 40 under-recalls a 100-deep fetch). This took episode vector search from ~878ms to ~15–23ms (38x) at recall@10 = 1.000 vs exact scan.
- **Fusion** (`merge_rrf`): RRF `k=60` × recency decay (half-life 30 days) × feedback boost (`min(2.0, 1 + log1p(retrieval_count)·0.3)`). The counter behind the boost is the episode's own, bumped asynchronously by `fetch`, `fetch_session` and `recall_full_turns`; the fact counter feeds only the dashboard.
- **Self exclusion**: the calling session's own turns (`self_session`, injected by the client's PreToolUse hook) are dropped before the pool is cut to 100, so their slots backfill with real history; its self-only facts are dropped from the 12 + 5 candidates the KG leg is asked for, so the bucket stays full.
- **Chunk candidates** (`SYNAPSE_RECALL_CHUNK_CANDIDATES=1`, default off): the top 10 chunks by keyword (BM25 over `chunks`) and by embedding (the optional chunks HNSW index, migration 067) name their episodes, which join the RRF pool through the same project and surface predicates as the episode legs. Chunks are never served, only their episodes; without the index the vector half says so in `warnings` and serves nothing while the keyword half works. `scripts/ab_chunk_candidates.py` runs the on/off A/B over a goldens JSONL (`query`, gold `e:` ids, optional `project`); the losing side gets deleted.
- **Rerank (WIN1)**: the fused 100-pool goes to Voyage `rerank-3-lite` by default (WIN1 measured on `rerank-2.5`). The reranker is the active ingredient — deeper RRF alone can't lift a rank-40 gold; the cross-encoder can. Lifted answerability 90.5%→95.2%, exact-gold hit +7pts, served tokens −20%. The cross-encoder is recency-blind, so its scores are multiplied by a recency decay afterwards (14-day half-life, floored at `SYNAPSE_RERANK_RECENCY_FLOOR`), and an episode that merely quotes the query back (a long verbatim run of the prompt) is dropped, with the next-ranked backfilling. `recall()` and `recall_full_turns` share this ranking contract (`_rank_episodes`).
- **Passages**: `recall()` does not serve whole turns. The top 20 reranked episodes are split into markdown chunks (`ingestion/web_chunker`), the chunks (80 at most; past that budget it is shared across the episodes, BM25-best chunks first, round-robin by episode rank) go through a second rerank, and the top `SYNAPSE_RECALL_PASSAGE_N` = 6 are served, one item per parent episode, skipping a chunk that repeats another episode's chosen one and capping how many may share a session. About 1/4 the tokens of whole turns at near-baseline answer survival.
- **Abstention floor**: `SYNAPSE_RECALL_FLOOR` (0 = off). When the raw (pre-recency) top rerank score among the episodes that survive echo suppression is strictly below it, `recall()` serves no episode bucket instead of the least-bad ones, and `served_ids` gains `would_abstain` / `floor`; `rerank_top_all` records the pool's own top when it differs. Enforcement is always on; `recall_full_turns` records the marker but never withholds (the caller asked for turns).
- **Degradation**: with `SYNAPSE_RERANK_PROVIDER=none`, or a failed rerank, the pool stays in RRF order and passages are picked by BM25 keyword overlap, one best chunk per top episode; a failed rerank adds a `warnings` entry saying so. Recall never hard-fails on the rerank leg.

### 6.3 KG fact leg (`mcp_server/kg_pg.py`)

Three candidate lists over Postgres (`kg_relationships`), all filtering `t_invalid IS NULL`, fused by a separate RRF (`k=60`, `_RRF_K`; no recency or feedback factor):
1. **Fact-embedding vector** over `kg_relationships.fact_embedding` (halfvec HNSW, partial on `t_invalid IS NULL`), over-fetched on the bare partial-index predicate and then filtered to the owner and group.
2. **BM25 fulltext** (ParadeDB) over fact text.
3. **1-hop traversal** from seed entities: one `LATERAL` statement for all seeds takes each seed's newest 8 live edges, either direction (the optional `kg_rel_src_live` / `kg_rel_tgt_live` indexes of migration 065 serve it; the rows are the same without them).

Seed entities come from a pgvector KNN over `kg_entities.embedding` (halfvec HNSW, fetch 25), gated to entities with at least one live edge (an `EXISTS` probe per side), capped at 8, with a `session_focus` bonus (0.3 subtracted from distance when a seed's name/uuid is in the caller-supplied focus set). The whole leg is four statements however many seeds there are. Same-text twin facts serve once. `_FACT_LIMIT = 12` (#100 — doubled exact-fact keyword recall 0.119→0.214 at zero graph bloat). A restricted caller (§7.5) is served only facts whose cached `source_projects` set (schema 056) is non-empty and inside its allowlist, through an exact allowlist KNN; an empty allowlist, or a database without 056, serves no facts. A KG-leg failure degrades to an empty `facts` bucket with a `warnings` entry.

**Supersession.** Two extras ride beside the leg. The *supersession surface* answers a query that matches a now-invalid fact with its current successor (the `invalidated_by` link, accepted only within `SYNAPSE_SUPERSEDE_LINK_MAX_DIST` of the fact it retired), up to 3 facts beyond the 12, gated by `SYNAPSE_SUPERSEDE_MAX_DIST`. The *episode-validity overlay* attaches the current fact to a served passage that states a claim the KG has since superseded. `superseded_facts` carries the `{previously, now}` pair for each served fact that was itself superseded.

### 6.4 Notes, web & history legs

- **Notes**: the hook-embedding KNN over live notes (global types plus the caller's project; `work-safe` notes only for a restricted caller) fetches 24, a cross-encoder floor on `hook — body` text (`SYNAPSE_RECALL_NOTES_FLOOR`) keeps the relevant ones, and the top 3 are served with the body capped at 700 characters (`fetch("n:N")` returns the full note). A note already on the board may be served again. The leg self-gates: an all-subfloor result serves no bucket.
- **Web**: BM25 and vector over `web_chunks` fused by RRF (50 per leg, pool 60), reranked, cut by `SYNAPSE_RECALL_WEB_FLOOR` with no keep-one backstop (the bucket may serve nothing, which gates web on intent), deduped to one chunk per parent page, joined to `web_artifacts` for url and title. Cap 3. A restricted caller is served only pages fetched in an allowlisted project or by its own credential; an unknown or root-token caller gets no web bucket.
- **History**: bitemporal `{previously, now}` pairs for facts whose edges were surfaced — the supersession story flat vector search can't tell. Cap 2.

The floors are properties of the reranker, not of recall: scores are not comparable across models, so every `examples/env` preset pins `SYNAPSE_RECALL_FLOOR`, `SYNAPSE_RECALL_WEB_FLOOR` and `SYNAPSE_RECALL_NOTES_FLOOR` (0.60 for the Voyage and OpenRouter presets, calibrated on `rerank-2.5-lite`, whose scores run 0.02–0.04 below `rerank-3-lite`; 0 for the bundled local reranker).

### 6.5 Parallelism

Legs fan out on the 8-worker leg executor as above; a separate single-worker executor handles fire-and-forget `retrieval_count` and `recall_metrics` writes. Postgres connections come from the shared `ingestion/pg_pool` pool (autocommit, dict rows, `hnsw.ef_search = 200` set on each), borrowed per leg. This took `recall()` from ~2.85s to ~1.3s (#60, #103, #104).

---

## 7. MCP Server

**Standalone `fastmcp` 3.4.2** (migrated off the SDK-bundled `mcp.server.fastmcp` in 2026-06, PR #166), HTTP transport with `stateless_http=True`, bound `0.0.0.0:8765`, MCP at `/mcp`. A `--stdio` flag switches to stdio on localhost. Stateless HTTP means each call is a short-lived request with no state to lose on container restart or client sleep/wake. The standalone package brought first-class **auth** ([§7.4](#74-auth)) and a **skills provider** ([§7.6](#76-skills-serving)) that the bundled one lacked.

### 7.1 Tools

Registration order is deliberate (tool-list position biases model tool choice); `tests/test_tool_surface.py` pins it. The board is push-only — `GET /context` feeds the plugin's SessionStart hook; there is no MCP board tool (the hook already injects the block).

Six visible tools, in registration order, plus one hidden. Per-tool detail (what each parameter does, and when a model should reach for which) lives in [docs/tools.md](docs/tools.md).

- **`recall(query, project=None, session_focus=None, group_id="technical")`**: the primary retrieval tool ([§6](#6-recall-pipeline)). Reranked episode passages fused with KG facts, notes, captured web research, and fact history.
- **`recall_full_turns(query, project=None, limit=5, session_id=None)`**: the drill-down and retry sibling. Whole unabridged turns, relevance + recency ranked, same deep-fetch + rerank machinery and the same ranking contract as `recall()`. A call serves up to 40,000 characters of turn text (`FULL_TURN_BUDGET`): a turn that would pass it comes back as its first 1,000 characters plus the best-matching 2,000, with `full_chars` giving its real length, and `served_ids` records `n_excerpted`. It absorbed the standalone `recall_episodes` tool (item 6 tool-surface audit; telemetry keeps `kind='episodes'`), first as `recall(mode="turns")`, then re-split into its own tool in 0.9.9.
- **`fetch(ids)`**: expand `e:N` episode ids (bare `N` accepted) and `n:N` note ids into full records; mixed lists fine, unknown ids reported under `skipped`, capped at 20 per call. The first id asked for always comes back whole; later episodes share the same 40,000-character budget and come back as an opening plus `full_chars` once it is spent.
- **`fetch_session(session_id, around=None, radius=3, offset=0, limit=10)`**: read one session sequentially instead of searching it. `around` centers a window on an `e:N` anchor (anchor full, neighbours as 500-char heads); anchorless, it pages with `offset`/`limit`. `session_id="self"` is the calling conversation, and an unindexed session returns an explicit error rather than an empty read ([spec](docs/session-drilldown-spec.md)).
- **`remember(content=None, hook=None, body=None, type="project", project=None, session_id=None, audience=None)`**: reconciles a note into the notes store and archives the text as a manual `Episode` (appended with `append_episode`: the next sequence under the session's advisory lock, never an overwrite) with KG extraction. Reconcile compares the new hook with the nearest live note of the same type (same project, for a project note) at cosine ≥ 0.80 and, unless the text is an exact restatement, asks one three-way judgement (the lane's pair prompt, `SYNAPSE_NOTES_CONFIRM_MODEL`): **DUPLICATE** updates the note in place, copying the hook, body and `source_ref` it replaces to `notes_history` (schema 071; without it the rewrite leaves no record and `source_ref` stays on the old episode); **CORRECTION** that keeps the new note supersedes the old one (lineage kept); **DISTINCT**, an unclear verdict or any LLM failure inserts the new note beside the old one — when unsure it inserts, and the nightly lane merges what is left. A device-token caller that names a `session_id` gets the archive stored but not queued for extraction, because the Stop hook ingests that session — except in a private session, whose turns the hook never ships, so the statement is queued like a hookless caller's. Chunk text leaves `source='manual'` episodes out.
- **`recall_feedback(query, helpful=None, noise=None, missing=None, found_via=None, comment=None, session_id=None, project=None)`**: after-the-fact retrieval-quality report. Which served ids (`e:N`/`n:N`/`f:N`/`w:N`/`t:N`, validated; the older `f:<uuid>` still resolves) were load-bearing vs noise, what content was missing (`missing` + `found_via`), and `comment` as the free-text spot for everything else (serving volume, ordering, misleading results). Writes one `recall_feedback` row (schema 046; `comment` lands in the `note` column): offline labeled data for eval goldens and reranker tuning, deliberately not wired into live ranking.
- **`issue_machine_token()`**: returns the shared machine bearer token, auth-gated by MultiAuth + the allowlist; lets `synapse login` fetch it over OAuth instead of a manual copy-paste ([§7.4](#74-auth)). **Hidden from `tools/list`** by an `on_list_tools` middleware; still callable by name via `tools/call`.

Removed from the surface (git history keeps the code): `list_projects` (the board banner carries per-project activity now), `query_graph` (experimental NL→SQL; `recall()` owns retrieval), `recall_episodes` (~4% of recall-family calls, now served by `recall_full_turns`), and `recall_timeline`, dropped on 2026-08-07 together with recall's inline timeline leg (39 helpful vs 92 noise citations lifetime; the `timeline_events` store, its ingestion gate, and the board's "Last 7 days" block are untouched, only retrieval went).

### 7.2 Custom HTTP routes (`/ingest`, `/recall`, `/capture/dry-run`)

**`GET /install`** (`mcp_server/install_routes.py`) is the client's distribution channel: an unauthenticated stdlib Python script (`mcp_server/install_script.py`) carrying this server's own `plugin/` + `plugin-codex/` trees, zipped and base64-encoded, plus the client version (`plugin.json`) and the bundle's sha256. `curl <server>/install | python3 -` unpacks it into `~/.synapse/lib/<version>/`, points `lib/CURRENT` at it, writes the `synapse` shims and the fixed launcher `~/.synapse/bin/synapse_run.py`, and records the request's origin as the server URL. Client version = server version, no package index; `synapse update` fetches it again. Harness configs only ever name the launcher, never `lib/<version>/`: Codex trusts a hook by a hash of its command text, so a versioned path would switch the hooks off at every update.

`/ingest` and `/recall` are Starlette **custom routes**, not MCP tools (they won't appear in the tool list), so the hook scripts can hit them with a one-shot `urllib` POST instead of an MCP handshake. `/ingest` is the Stop-hook push endpoint ([§4.1](#41-the-ingest-endpoint)); `POST /capture/dry-run` takes the same body and reports what web capture would store without writing ([§8](#8-web-research-subsystem)); `/recall` serves non-MCP HTTP callers that want recall results without an MCP handshake (the dashboard's Recall console and scripts; same engine as the `recall` tool, `write_feedback=False`).

The plain-HTTP routes are decorated with `http_helpers.route(gate, body=, not_found=)`: it asks the gate once (401), parses a JSON **object** body (400 `invalid JSON body` / `body must be a JSON object`), runs a plain `def` handler in the threadpool and an `async def` one on the loop, maps a `RouteError` to the error envelope and anything else to a logged 500. The skill and config sync routes use the lenient `guarded_json` instead, because their handlers answer a malformed body themselves (a 200 `status: error` the CLIs print). The dashboard API lives in `mcp_server/dash/`, one module per surface (static, feed, stream, kg, search, metrics, curation, review, records) mounted by `mcp_server.dash.register`; `/skills/proposals*` are in `mcp_server/skill_review_routes.py` and the sync routes in `mcp_server/skill_sync_routes.py`. fastmcp's auth deliberately does **not** gate custom routes (they're intended for unauthenticated ops endpoints — [issue #3704](https://github.com/jlowin/fastmcp/issues/3704)), so each handler resolves the bearer itself ([§7.4](#74-auth)): a constant-time compare against `SYNAPSE_MACHINE_TOKEN`, else a hashed device-token lookup, through the client or admin gate as appropriate ([§7.5](#75-audience-scoping--credential-bound-surface-trust-schema-053--054)). A health helper returns `{"status": "ok"}` for container probes.

**`/remember/spool`** (schema 052) is the durability seam for memory writes. On 2026-08-25 the MCP transport's OAuth failed mid-session: `remember()` could not be called at all, so a user correction was silently lost for two days while the poisoned note it was meant to replace kept re-injecting at every session start — the plain-HTTP lane (`/ingest`, bearer token) worked the whole time. The plugin now spools an unwritable intent to `DATA_DIR/remember_spool.jsonl` and replays it here, and this route calls **`remember` itself** (injected into `mcp_server/remember_routes.register`, not imported — no cycle, and no chance of the replay drifting from the tool it replays). The client generates the intent id once and re-posts it until the server confirms; the route CLAIMS that id in `remember_intents` before writing, so an already-`done` id short-circuits to the recorded note (`outcome: "duplicate"`) instead of minting a second one, while a row still `pending` — a prior attempt that died mid-write — is re-run, because `reconcile_note` converges on the same live note but a skip would lose it entirely. The route records the archive episode's id on the `remember_intents` row, and a re-run of a pending intent reuses that episode instead of archiving and extracting again (the identical-text convergence needs no LLM call). A missing `remember_intents` table (052) surfaces as the database's own 500; the route no longer answers a canned 503. A `probe: true` body answers "reachable AND authorized" without writing: that is what the plugin's SessionStart step calls, since a `/health` ping would report "up" with a dead bearer.

### 7.3 Lazy init & connection resilience

The `Recall` engine and its Voyage clients initialize lazily on first call. The engine holds no Postgres connection of its own: every recall leg, `fetch` and the notes lookups borrow an autocommit dict-row connection from the server's one pool (`ingestion/pg_pool`, 2–16 connections, held open from startup to shutdown by the server lifespan) and hand it back. The pool checks a connection when it hands it out, so a half-open TCP connection (PG restart, network blip) is replaced transparently. Every borrow starts with `hnsw.ef_search = 200`, so nothing that borrows issues a session-level `SET`.

### 7.4 Auth

`fastmcp` **MultiAuth** composes two credentials on the one `/mcp` endpoint (PR #166, which retired the 636-line hand-rolled OAuth proxy):

- **`SynapseTokenVerifier`** (`mcp_server/auth_tokens.py`, replaced `StaticTokenVerifier` with schema 054) — verifies TWO kinds of machine bearer. The **root** token (`SYNAPSE_MACHINE_TOKEN`) matches by constant-time compare with **no database touch**, because the Docker-host services authenticate with it to reach `/ingest` and must survive a PG blip; it identifies a deployment, not a machine, so it resolves to no surface. A **device** token (schema 054) is `sha256`-looked-up in `surfaces` and verifies only when `status='approved'`, carrying that row's trust in its claims. See [§7.5](#75-audience-scoping--credential-bound-surface-trust-schema-053--054) for what each is allowed to do — notably, root is the *enrollment* credential and deliberately cannot clear the admin gate.
- **`GitHubProvider`** (an `OAuthProxy`) — the **claude.ai web-connector** path. fastmcp emits the OAuth 2.1 discovery metadata + the DCR `/register` endpoint claude.ai needs and bridges to GitHub login. An `on_call_tool` middleware enforces `ALLOWED_GITHUB_USERS` (GitHubProvider has no built-in allowlist — without it, any GitHub account could read the memory).

Auth is **env-gated**: with no `SYNAPSE_MACHINE_TOKEN` the server runs open (dev/stdio). The client's `synapse login` runs ONE device-flow approval that both signs in and ENROLLS this machine (`POST /surfaces/enroll` reads the approving identity and checks the allowlist itself), writing the minted device token and its enrollment record into `~/.synapse/config.json` in one atomic write. Only the legacy `--browser` flow fetches the root bearer, via the `issue_machine_token` tool (device tokens are refused there — leaves do not fetch the root), and then enrolls with a device-flow approval of its own. The enrollment record keeps that token's sha256 and minting server, never the token, and the hooks use it to resolve to the device token even when a stale `SYNAPSE_INGEST_TOKEN` is exported in the environment, which used to split uploads and recall across two credentials. Session start names the overridden source in one line. The MCP connection authenticates through a **headers helper** (`plugin/scripts/mcp_headers.py`), the same script for both harnesses: Claude Code runs it from `plugin.json`'s `headersHelper` (from the plugin root, for plugin servers with no trust prompt), Codex from `config.toml`'s `http_headers_helper` (through `cmd /C` on Windows, with an emptied environment). Neither passes credential-named env vars, so the helper reads `config.json` and prints exactly `{"Authorization": "Bearer …"}`; on failure it prints nothing, and Claude Code falls back to the plugin's static `Bearer ${user_config.SYNAPSE_INGEST_TOKEN}` header. Each harness keeps only the server URL in its own MCP entry, because it opens the connection.

**`synapse login` — device flow (default).** The original loopback authorization-code flow needed an interactive browser on the *same host* as the CLI — wrong for servers/headless boxes (and `GitHubProvider.allowed_client_redirect_uris` originally allowed only the claude.ai/claude.com callbacks, so the CLI's `127.0.0.1` redirect was rejected outright until loopback patterns were added). Login now defaults to the **OAuth 2.0 Device Authorization Grant** (RFC 8628), served by `mcp_server/device_routes.py` as two unauthenticated bootstrap routes that **proxy GitHub's native device flow**:

- **`POST /device/code`** → calls GitHub's `login/device/code`, returns `{user_code, verification_uri, device_code, interval}` to the client.
- **`POST /device/token`** → polls GitHub's token endpoint with the `device_code`; on approval it reads the GitHub login and enforces `ALLOWED_GITHUB_USERS` (same gate as the web leg) before returning `{token: SYNAPSE_MACHINE_TOKEN}`. Stateless — the `device_code` lives on the client and is replayed each poll; no server-side state, no redirect URIs.

The user approves at `github.com/login/device` on **any device**; no same-host browser. Requires "Enable Device Flow" on the GitHub OAuth App. The legacy browser flow remains behind `synapse login --browser`. The legacy login paths (`--browser`, `/device/token`, `issue_machine_token`, the `settings.json` / `device.json` / `credentials.json` fallbacks and the Codex `mcp_headers` shim) go in the release after every machine has run `synapse login` on 0.23.0 or later; from 0.23.0 the client's `User-Agent` carries its version, so the server logs show which clients still call them. The dashboard's browser login takes the first entry of a comma-joined `X-Forwarded-Proto` / `-Host` chain, validates its shape, and returns the token only to an origin in `SYNAPSE_PUBLIC_URL` or `SYNAPSE_DASH_ORIGINS`; the login screen asks `GET /dash/oauth/available` and shows "sign in" only when an IdP is configured. **Client config is harness-independent**: `plugin/scripts/config.py` resolves every setting for both harnesses (the Codex hooks load that file by path through `plugin-codex/scripts/common.py`, and read only the USER-level Claude `settings.json`, never a project's) in order — explicit env → `~/.synapse/config.json` (`plugin/scripts/synapse_home.py`; `url`, `token`, `role`, `skills_sync`, `config_sync`, `config_paths`, `timeline_repos`, `board`, `prefs_block`, `recall_nudge`, `prompt_timestamp`, `recall_feedback_nudge`, `device`) → the Claude plugin's legacy saved options (`CLAUDE_PLUGIN_OPTION_*`, then `pluginConfigs."synapse@<marketplace>".options` in `settings.json`) → default. The legacy options rank below `config.json` because Claude Code keeps exporting answers to options the plugin no longer declares; the first `synapse login` copies them into `config.json` and drops the four the 0.20 plugin stopped declaring (`SYNAPSE_MACHINE_ROLE`, `SYNAPSE_SKILLS_SYNC`, `SYNAPSE_CONFIG_SYNC`, `SYNAPSE_CONFIG_PATHS`). `synapse login` then wires each harness it finds (`plugin/scripts/harness.py`): the Claude plugin through `claude plugin marketplace add` / `claude plugin install … --config SYNAPSE_URL=<url>` (a `settings.json` bootstrap when there is no CLI), and Codex through `plugin-codex/install.py`'s `config.toml` writes (`codex mcp add` can't set a headers helper). The plugin's `bin/synapse` and `bin/synapse-login` run the same CLI inside Claude Code sessions.

### 7.5 Audience scoping — credential-bound surface trust (schema 053 + 054)

Private mode ([§4.1](#41-the-ingest-endpoint)) controls what gets **captured**. Nothing controlled what gets **served**: the board injected the whole curated note set into every session on every machine running the plugin, and `recall()` searched the whole corpus regardless of where the query came from. A work laptop ingesting into the same Synapse therefore got personal notes injected into transcripts on employer-owned hardware — and the board's recency weighting meant whatever was most personally urgent floated to the top. **Serving is the leak, so the filter goes at serving** — one enforcement chokepoint, the same call as private mode's drop-at-ingest inverted.

Three pieces:

- **`notes.audience`** — `personal` (default) or `work-safe`. Two tiers, not N: every extra tier is a per-note decision someone has to make, and a rich taxonomy nobody maintains fails silently. The `personal` default is fail-closed *in the schema*, so a note inserted by any path that forgets the column never leaves a trusted host.
- **`surfaces`** — one row per surface, carrying `trust IN ('full','restricted')` and an `allowed_projects` TEXT[]. The column default is `restricted`, so promotion to `full` is always an explicit act rather than something an `INSERT` can do by omission.
- **A device token per surface (054)** — `token_hash` (sha256 hex, unique where non-null), `status IN ('approved','revoked')` defaulting to the fail-closed `revoked`, a display-only `label`, and `last_seen_at`.

#### Why the hostname anchor had to go

Schema 053 keyed trust on the plugin's self-reported `SYNAPSE_SURFACE` constant, accepted under the one shared machine token, and said so in its threat model: "an employer READING a transcript, not attacking the API." But the untrusted machine must hold the machine token to ingest at all, so anything running there could send `surface=<trusted-host>` and be served the whole corpus. The trust boundary was a string chosen by the side being bounded. **Schema 054 makes the credential the identity**: one token per device, `sha256` matched against `surfaces.token_hash`, and nothing a caller *says* can change which row applies to it.

**Only `status='approved'` serves.** Revoking clears the hash too, so a revoked credential matches no row at all and the row survives purely as the audit record. A revoked row resolves to `UNKNOWN_SURFACE` rather than "restricted but known", because `known` is what lets a restricted write default to `work-safe` — a dead device must not be able to widen a note's later audience. The column default is `revoked`, which reads odd and is deliberate: every real creation path sets it explicitly, so the default only catches a hand-written row, and "nobody granted this" should behave exactly like "this grant was pulled".

#### Enrollment is anchored to the owner's identity

A new machine gets its device token by completing the IdP's **device flow** (RFC 8628 — the same lane `synapse login` already uses: a short code approved in a browser on any device, phone included) and passing the resulting `device_code` to `POST /surfaces/enroll {device_code, label, trust, allowed_projects}`. The server polls the IdP, reads the identity, enforces the same allowlist every other login clears, and only then mints. The row lands `approved` immediately.

**Trust-on-first-use was built and then removed.** The first cut used a pair code: a new machine enrolled `pending`, printed a 6-character code, and a second already-trusted machine approved it. That ceremony exists to compensate for an *anonymous* enrollment credential — if all a new machine can prove is "I hold the shared token", something else has to vouch for it. An OAuth-verified owner standing at the new machine has already answered the only question the ceremony was asking, so it bought a second round trip, a code to copy between screens, and a pending state in which a laptop silently has no memory, in exchange for nothing. Accepted in the threat model: theft of the owner's IdP credential, mitigated by the IdP's own 2FA and in any case already sufficient to open the dashboard and read everything directly.

The `label` is display only and deliberately never matched against an existing row: keying an enrollment on a self-reported name would reintroduce hostname spoofing through the back door ("enroll as `trusted-laptop`, inherit its grant"). The row's id is server-generated and opaque.

`trust` is the answer to `synapse login`'s **personal or work** question (`role` in `~/.synapse/config.json`, `SYNAPSE_MACHINE_ROLE`; the plugin's install prompt asked it before 0.20), authoritative because the person who answered it is the person who just authenticated. The question defaults to `personal` — the single-user common case is a machine that should see everything, and a default that makes the normal path silently useless gets worked around rather than understood. The narrow default lives one layer down: the SERVER treats an *unstated* role as `restricted`, so a client that never asked the question cannot resolve it to full access. **Human says nothing ⇒ personal; software says nothing ⇒ restricted.** A restricted enrollment that names no projects inherits the union of what approved restricted surfaces already read (a second work machine should see the same work projects as the first); an explicit `[]` still means empty.

`POST /surfaces/mint` is the same database operation for a machine that will never run a browser flow — a headless box, a service, a container. It is `admin` gated, so it needs a machine that is already trusted, and the token is carried over by hand. `DELETE /surfaces/{id}` revokes (and answers `{status, surface_id, revoked}`; the older `deleted` alias is gone).

#### Two gates, and their asymmetry is the point

| Gate | Admits | Guards |
| --- | --- | --- |
| **client** (`_machine_authorized`) | root token, or any APPROVED device token | `/ingest`, `/capture/dry-run`, `POST /recall`, `GET /context`, skills / config / timeline / preferences / private-session / remember-spool routes |
| **admin** (`_admin_authorized`) | APPROVED **full-trust device** token only — **never the root token** | `/surfaces` list + `PUT`/`DELETE`, `/surfaces/mint`, every `/dash/api/*` route; these take the admin gate with no fallback |

The root machine token is the services' credential for `/ingest` and the internal write paths, and every machine that ever ran the plugin has held it. A credential that widespread must not be able to create a *new* credential, list who is trusted, or read the dashboard's unfiltered view — hence it clears neither the admin gate nor enrollment. `/dash/*` had to move for the same reason (it serves the whole corpus unfiltered and can flag/edit, so leaving it on root would be a bypass around the entire filter); the dashboard login flow now mints a full-trust device token for the OAuth-allowlisted identity (`dash:<login>`, rotated per login, one row per identity) instead of handing the browser the root token. `issue_machine_token` refuses device tokens: leaves do not fetch the root.

**Break-glass is not a token.** `scripts/surface_admin.py` (list / mint / revoke / bootstrap) talks to Postgres over a direct DSN, so recovery needs shell access on the database host — a strictly higher bar than holding a bearer. It covers the three cases where nothing else works: no IdP configured at all, the IdP down or the account locked, and no full-trust device left.

**Root verification never touches the database** (`mcp_server/auth_tokens.SynapseTokenVerifier`): constant-time compare, no I/O. The services on the Docker host authenticate with the root token to reach `/ingest`, so a verifier that had to read PG to admit root would turn every PG hiccup into a total auth outage on the one lane that repairs things. Device tokens do read PG, in a worker thread so the event loop stays free. FastMCP runs the verifier app-wide, and the custom routes' gates and trust reads reuse its verdict through `http_auth.request_trust` (one lookup per device request, kept on `request.state`, never cached across requests); a bearer the verifier rejected is UNKNOWN with no second lookup, which is also the outcome during a Postgres stall (fail-closed, loop free). `last_seen_at` is stamped only when the row's stamp is older than 5 minutes, and the SELECT reports that, so no write is attempted inside the window (best-effort: auth must not depend on a WRITE succeeding).

#### Resolution and enforcement

`ingestion/surfaces.resolve_caller` is the single resolution point and it **never fails open**: no credential, no row, a non-approved row, missing table, unreachable database, malformed row — all return restricted with an empty allowlist. `project = ANY('{}')` is false for every row, NULL project included, so an unknown caller serves *nothing* rather than everything. There is no code path that turns an error into `full`.

`mcp_server/server._caller_trust` (MCP tools) and `http_auth.request_trust` (plain-HTTP routes) apply one precedence, and a caller never picks which lane it gets:

1. **Device token** — the credential names the surface; any `surface` param is ignored outright.
2. **`oauth:<login>`** — MCP clients on the interactive OAuth/OIDC lane (the claude.ai connector) run no hook and hold no device token, but carry an identity this server verified and allowlisted. `_caller_trust` resolves it to `oauth:<login>` (the same lowercased claim the `_UserAllowlist` gate reads: `login` on GitHub, `preferred_username`/`email` on OIDC) and looks that up like any other id. No row still means restricted; nothing auto-creates one.
3. **Anything else is unknown** — restricted, empty allowlist. That covers the ROOT token, an open (tokenless) server, and an OAuth token with no identity claim. The root token's old **legacy `surface` lane is closed**: for one release after 054 a root-token caller could still name a surfaces row by id and inherit its trust, which let anyone holding the shared token (every machine that ever ran the plugin) borrow a `full` row and read personal memory. Neither resolver in `mcp_server/caller_trust.py` takes a `surface` argument any more, so the `?surface=` query arg on `/context`, the `surface` body field on `/recall` and `/remember/spool`, and the `surface` argument on every MCP tool are ignored (logged once per call site so an operator can find stale clients). The write side follows: `/remember/spool` attributes a note to the bearer's surface, never the body's. As defence in depth the id lane (`oauth:<login>`) matches credential-less rows only, so a device row is reachable by its token and nothing else. The 0.17.0 client stopped sending the param, and the PreToolUse surface-injection lane is deleted from both `plugin/scripts/self_session_inject.py` and `plugin-codex/hooks/pre_tool_use.py`.

What a `restricted` surface gets:

| Path | Notes | Episodes / timeline | KG facts |
| --- | --- | --- | --- |
| `GET /context` (board) | `audience='work-safe'` | digest + banner filtered to `allowed_projects` + own uploads, NULL project excluded | n/a |
| `recall` / `POST /recall` | `audience='work-safe'` | BM25 + vector legs `project = ANY(allowed) OR surface_id = <caller>` | only facts whose provenance is wholly inside `allowed` (schema 056) |
| `recall_full_turns` | n/a | same predicate on the pool | n/a |
| `fetch(ids)` | `audience='work-safe'` | same predicate | n/a |
| `fetch_session` | n/a | same predicate on the metadata probe *and* both row reads | n/a |
| `/skills/*` + `skill://` resources | n/a | only `project:<p>` skills with `<p>` in `allowed` (case-insensitive); never `global`; publish/overwrite only there; retire, rescope and the review lane 403 ([§7.6](#76-skills-serving)) | n/a |

**KG facts are filtered by provenance (schema 056).** A fact has no project of its own, so `kg_relationships.source_projects` caches the distinct projects of its source episodes, and a restricted caller is served a fact only when that set is known, non-empty, and a subset of its allowlist. Unknown means NULL: any source episode missing or project-less, or no episode provenance at all (web-artifact facts). Mixed provenance fails the subset test. Triggers maintain the column, so every writer is covered with no application change: inserts and `episodes` updates on `kg_relationships` recompute it, as do a source episode's relabel, delete or truncate. The same rule gates every KG-derived extra recall serves (superseded pairs, where both sides must pass; the supersession surface; the episode-validity overlay), and seed entities are never served restricted because their summaries distil every fact. An empty allowlist skips all of it, and so does a database without 056: the code ships first and probes for the column, so until the migration is applied a restricted caller gets the pre-056 answer of zero facts. **`fetch` and `fetch_session` enforcement is not belt-and-braces** — `e:N`/`n:N` ids are sequential integers, so an unfiltered drill-down would let a restricted caller enumerate exactly what the board withholds. A session outside the allowlist reports the same "not indexed" answer an unknown id does, deliberately indistinguishable: a distinct "exists but forbidden" reply would itself disclose the project map.

**A restricted surface also reads what it ingested itself (schema 057).** Every episode written by `/ingest`, `remember()` or the remember-spool replay is stamped with `episodes.surface_id`, taken from the *credential-resolved* caller (device token or verified `oauth:<login>`), never from the request body, and never from the legacy self-reported `surface` param. Restricted reads widen from `project = ANY(allowed)` to `project = ANY(allowed) OR surface_id = <caller>` (one helper, `ingestion/surfaces.episode_scope_sql`, used by every episode path in the table above). Anything a work laptop uploaded is work by definition, so it can recall its own conversations with an empty allowlist and nobody maintains a project list for it. Root-token writes, unknown callers and every pre-057 row are stamped NULL, and NULL never matches, so legacy rows stay exactly as reachable as before. A stamp means every byte of the row came from that surface: a rewrite of the same turn keeps it only when the same surface writes again, a rewrite by anyone else clears it, and a NULL row can never be claimed. `timeline_events.surface_id` is copied from the source episode by the timeline gate, and `POST /timeline/events` stamps the pushing device's own surface id (probed; NULL for the root token and on a pre-057 database), so the board digest follows the same rule and a restricted device's own git-feeder events are served back to it. (`POST /timeline/recent` no longer exists: the board reads the digest directly.) KG facts do not widen (yet): they still need provenance wholly inside the allowlist. The code ships before 057 is applied and probes for the columns (`ingestion/surfaces.OWN_SURFACE_PROBE`); until both exist every statement is the pre-057 one, so ingest stamps nothing and a restricted caller is served its allowlist, as before.

**The session-start routes and the config lane are scoped by the bearer too.** `/preferences/top` applies the project allowlist only: a preference keeps just its first turn's `source_ref` while `merge_preference_text` folds later turns into its text, so which device the text came from cannot be proven. `/config/publish` takes `{surface, files:[...]}` (one request, one transaction; the single-file body is still accepted, and the plugin falls back to one file per request on a 500 from a server older than the batch form) and pins a restricted device to its own surface, ignoring the body's `surface` (full trust may still name any); `/config/list` and `/config/fetch` no longer exist. `/config/proposals*` list `proposed` and `accepted` rows and show a restricted device only proposals whose every evidence session consists solely of its own turns (none before 057 is applied). Unknown callers, the root token included, get empty reads and 403 on config writes, and a route registered without a trust resolver serves nothing.

**Timeline gets no `audience` column.** `timeline_events` already carries `domain IN ('personal','technical')` (schema 038) which fails *open* (NULL); a second overlapping label with the opposite fail semantics invites drift. The project allowlist is the filter instead.

**Write-side tagging** happens at the single note chokepoint (`ingestion/notes.reconcile_note`), in precedence order: (1) an explicit `audience` argument on `remember()`; (2) a write from a **live, approved** restricted surface defaults `work-safe`, symmetric with what it may read (else notes written at work vanish from the work board next session); (3) the note's `project` is in the union of **approved** restricted surfaces' allowlists ⇒ `work-safe` (one helper, `surfaces.project_audience`, case-folded, shared by `derive_audience` and the notes lane's retype stage); (4) otherwise `personal`. Rule 2 requires an approved surface on purpose — unknown restricts reads, but must never widen a write. A **restatement preserves** the stored tier (`COALESCE(%s, audience)`), so rephrasing is never reclassification; a **contradiction derives** its own. The dream→notes lane re-derives by rule 3 when a retype changes a note's project.

**Rollout is enforcement-first, no feature flag**, and ordered: apply 054 (existing rows are stamped `approved`, so nothing working stops working) → deploy → run `synapse-login` on each machine to sign in and enroll → log into the dashboard again to pick up its minted token → `mint` for anything headless. The legacy `surface` param has since been dropped. **A machine that has not enrolled is served nothing:** a root-token caller resolves to `UNKNOWN_SURFACE` whatever it sends, restricted with an empty allowlist, a 200 with an empty board rather than a 401. Fail-closed is unchanged, but it is no longer silent: the SessionStart board hook prints an explainer whenever `/context` answers ok with `trust=restricted` and the machine holds no device credential, and the server says it too, for clients too old to: an authenticated caller that resolves to unknown, or to a restricted surface with no projects, gets a one-line `WARNING:` under the board header and the same text first in `warnings` on `recall`, `recall_full_turns`, `fetch`, `fetch_session` and `POST /recall` (`mcp_server/recall_warnings.serving_notice`; no ids in the text, nothing for full trust, restricted-with-projects, or an open server, and, once 057 is applied, nothing for a restricted device with no projects, which then reads back its own uploads by design). Enrolling is the fix (`synapse login`); on a deployment with no IdP at all, mint the first device token on the server host with `docker compose exec mcp-server synapse-admin bootstrap "<label>"` (a console script in the image; `scripts/surface_admin.py` stays as the host-side shim) and store it on that machine with `synapse login --token`. `docs/audience-scoping-spec.md` carries the step-by-step, and [docs/install.md](docs/install.md) the fresh-install path. Every pre-053 note is `personal`, so restricted boards run empty but safe until `scripts/audience_backfill.py --propose` writes a review table (to a path **outside the repo** — it lists every note hook, and this repository is public), a human edits the audience column, and `--apply` writes it back.

### 7.6 Skills serving

The server hosts the skill library in Postgres (`skills_lane.skill_registry.body` + `skill_files`). The **Synapse plugin** reaches it over machine-token-gated **HTTP routes** — `mcp_server/skill_sync_routes.py`: `/skills/list`, `/skills/fetch`, `/skills/publish`, `/skills/overwrite` (sync), `/skills/retire` and `/skills/rescope`; the review routes `/skills/proposals[/act]` are in `mcp_server/skill_review_routes.py`. `/skills/list` entries for active skills carry `content_hash` (the whole-skill hash) beside `body`, `description` and `files`, and the client compares it when present (an older server's entries are hashed from the body). This is the **DSN-free seam**: a client needs one base URL + a bearer, never a database connection. The same content is *also* exposed as native fastmcp **`skill://` resources** via `PgSkillsProvider` (`mcp_server/skills_provider.py`) for any generic fastmcp client (`skill://{name}/SKILL.md`, `/_manifest`, `/{path}`).

The plugin's `SessionStart` hook **materializes** skills to `~/.claude/skills` (where they become native Claude Code skills) and **publishes** local edits back: a per-skill whole-skill hash decides *if* a skill changed, `content_modified_at` decides *which side is newer* (pulls `os.utime` the files so "newest" tracks the edit, not the sync). Every superseded body is logged to an append-only `skill_history` table (a PG trigger on server writes + a `/skills/overwrite` snapshot for a local edit that never reached the server). Postgres is the single source of truth; the **dream→skills lane** ([§11](#11-nightly-maintenance--dream-pipeline)) writes it, so the central dream pipeline can improve a skill and have it sync back out.

**Retiring is server-side and timestamped (schema 060).** A local delete never retires anything. `/skills/retire` sets `status='retired'` and stamps `retired_at`, and the sync treats that as one more edit under newest-edit-wins. `/skills/list` keeps `skills` to active rows, so older clients never see a retirement, and adds a `retired` list (name, whole-skill hash, `retired_at`) of every retirement, whatever scope was asked for (a name is one registry row, and a copy can sit in another scope's folder): a client removes its copy only when that copy is still exactly what Synapse delivered, whatever its timestamp (in another scope's folder, only the exact retired content: that folder's history is of whatever skill it holds under the name), and keeps the delivered hashes so a restored copy is removed again. A publish whose `content_modified_at` is not newer than `retired_at` is refused, and so is the retired content itself under any timestamp (a clone or a copy renews an mtime, not the skill), so a machine that still holds the skill cannot bring it back. A newer local edit does, and so does a forced review accept (`retired` is a `--force` issue). Until 060 is applied, `/skills/retire` reports the missing schema and list and publish behave as before (`RETIRED_AT_PROBE`).

**Who is served what (trust, the board's rule applied to skills).** `SkillAccess` (from the bearer's `SurfaceTrust`) gates every route and `PgSkillsProvider`: full trust reads and writes every scope; a restricted device is served only `project:<p>` skills whose `<p>` is in its allowlist (case-insensitively) and may publish only there; anyone else, the root token included, is served nothing. Retire, `/skills/rescope` and the review lane are full-trust only (403). Before this, any approved device was served every global skill — the leak schema 053 closed for notes. A skill this caller may not read is "not found", in `/skills/fetch` and as a `skill://` resource alike.

**Names and paths stay inside a skill's folder.** A publish whose name is not usable as one folder name (a separator, a colon, `.`/`..`, control characters) or whose file paths leave the folder is refused (`safe_skill_name`, `safe_file_path`), and the client refuses to write either from any server. Publishes of one name are serialised (an advisory lock: a row that doesn't exist yet has nothing for `FOR UPDATE` to lock).

**One skill, one scope; moving it is explicit.** An active skill keeps its scope: a publish from another scope's folder is refused (`status: scoped`) instead of moving it, so a machine still holding a copy in the folder a skill was moved out of cannot move it back. `synapse skill scope <name> global|project:<label>` (`/skills/rescope`, full trust) moves it and logs `op='rescoped'` in `skill_history`.

**Withdrawn skills leave like retired ones.** The client sends the names it holds in the folder it syncs (`held`); `/skills/list` answers `withdrawn` for each that is still active but no longer served there (moved to another scope, or not this device's to read), and only for held names (a caller with no device credential gets none). An entry shows its scope only to a caller served that scope (else `""`), and a withdrawn copy is removed only when it is a version Synapse itself put there: one the entry vouches for, or one a pull wrote into that folder (the client's `pulled` record; its `delivered` ledger also holds the folder's own pushes, and an older client's holds pushes a newer server refused, so that one vouches only for a retirement in the folder's own scope). The client settles a withdrawn copy through the same function as a retired one (`_settle_unserved`): removed when it is exactly a version Synapse delivered (the entry carries `content_hashes`, the current and recent superseded registry bodies), never pushed back, and never removed from a folder git tracks or one linked in. `publish: false` in the listing (a restricted device's global folder) keeps that folder's own skills local instead of being refused every session. A project skill folder pulled into a git repository is listed in that repository's `.git/info/exclude`; one the repository tracks is the repository's own, never pulled over (git distributes it) and never listed. A pull writes SKILL.md last, already stamped with the server's time, so a pull cut short (a hook's timeout) is pulled again at the next session rather than pushed back as a newer edit; Codex syncs `project:<label>` into `<project root>/.agents/skills`, which it reads from any subfolder.

**The review lane.** A skill candidate is decided with `accept` (applies the proposal and promotes it) or `reject` (30-day cooldown) and nothing else: the `promote` action, `skill_review.py promote`, the `accepted` skill status, the `consolidate` kind and the accept-time routing eval are gone. A decision acts only on a candidate in `observe`, `proposed` or `accepted`; one in a final status (`promoted`, `rejected`, `retired`) answers `{"status": "conflict", "detail": ...}` (HTTP 200 on these CLI routes, the CLI prints the detail), and the check runs under the candidate's row lock, so two decisions landing together cannot both win. `/skills/proposals` no longer returns `proposal_path`; the detail returns `draft_attempts` and `last_draft_error`. A retune's identity key `(kind, name, direction)` is unique among **open** candidates only (migration 069): a sighting after an accept or a retirement opens a fresh candidate and the promoted row stays as history; without 069 a re-sighting of a closed retune is dropped with a warning. **Do not roll the image back past 069**: an older image's `ON CONFLICT` clause cannot infer the new index, so every candidate INSERT, derive included, fails. A rollback needs `DROP INDEX skills_lane.skill_gap_named_idx`, deleting or renaming all but one row of each duplicated `(kind='retune', name, direction)` key (the closed promoted or retired row where an open one shares its key; a retired row alone is not enough because the 022 index has no status predicate), then recreating the index from `schema/022`. Derive identity: a row or sighting without a `signature_key` (an explicit "make this a skill" request, a legacy key-less row) matches only the row of the same name, and a shared session alone is never identity; an explicit request is stored as its own grounded row (`explicit:<sid8>`) and never merges into a session's gap. A failed draft backs off 1, 2, 4… days (at most 16) once 070 is applied, and its reason shows in `skill_review show`. A widen adds at most 160 characters and may not pass 400 (a longer description may not grow). The judge sees only the skills served in the session's project; the procedure miner reads episodes only and skips clusters already in the ledger or decided within 8 weeks; `post_fire` writes `skill_usage.outcome`, and the nightly effects stage reports each retune accepted 7-30 days ago (fires, dismissals, deviations, before versus since). Evidence dedups on `(session_id, signal, class, work date)` (falling back to the scan night for entries without a date), so re-scanning the same work adds no evidence or weight.

**Config review** is the same shape: `config_review accept` / `reject` act on `proposed` and `accepted` rows, an `applied` or `rejected` proposal answers `conflict`, and `apply` (which writes every accepted rule to the learned file; the dashboard's approve only sets `accepted`) needs `accepted`.

### 7.7 Operator dashboard

`mcp_server/dash/` serves the dashboard API (`/dash/api/*`, admin gate) and the React bundle built from `web/` into `web/dist` (the image build runs `npm run build`; CI's web job runs `npm test` first). [docs/dashboard-contract.md](docs/dashboard-contract.md) is the contract between the two. It reads the KG's own bitemporal columns (`t_created`, `entity_subtype`) and counts live edges from `kg_relationships` (the `kg_entities.degree` column is a one-off backfill nothing maintains). The live feed is SSE: the stream opens with a `ready` frame and carries a `processing_status` (queue depth, active and failed rows) re-read every second, and its `LISTEN dash_feed` worker (schema 043) reconnects with a 1 s to 30 s backoff while any client is subscribed and appends a `reset` event (with an id) to the 512-slot ring after each reconnect, so clients refetch page 1. Deleting a turn runs one cascade shared with the private-session purge (`mcp_server.dash.curation.delete_episode_content`): its chunks and queue rows, the facts and timeline events it backs (a re-told event moves to a surviving re-telling turn when 064 is applied), a tombstone (073), and the web pages captured from it ([§8](#8-web-research-subsystem)); the single-turn delete leaves preferences alone, the purge removes them. `/dash/api/behavior/linkgraph` no longer exists, and the retired recall buckets (entities, communities, timeline, preferences) are gone from the Recall console.

---

## 8. Web Research Subsystem

Web-research tool results are captured so prior research is recallable, mirroring the Anthropic Contextual Retrieval pattern.

- **`web_artifacts`** (schema 011) — one row per captured web tool result, `kind`-discriminated over `web_scrape | search_result_set | research_job_ref`. Dedup key `tool_use_id` (UNIQUE), and a page already stored (same canonical URL and content hash) from another call is not stored twice. Holds url/title/markdown for scrapes, `items JSONB` for search result sets, research-job refs for deep research, and `metadata` (`capture`, `project`, `surface_id`, `kg_extract`). `session_id` and `parent_episode_id` (the turn's episode; NULL for a sub-agent's call) are columns. BM25-indexed on title + markdown + query. No embedding column (embeddings live in chunks).
- **`web_chunks`** (schema 012) — 1:N embedded slices (`voyage-4-large`, 2048), `ON DELETE CASCADE` from the artifact. BM25-indexed; halfvec HNSW (schema 014).
- **`web_chunk_context`** (schema 013) — a `context_prefix` column: a 50–100 token LLM-generated context blurb prepended before embedding (Contextual Retrieval) to lift recall. Filled by the poller (`ingestion/web_context.py`, stage model `CONTEXTUALIZE`); `''` = no context (a single-chunk page, or the model kept answering badly), NULL = not yet.
- **`research_archive`** (schema 076, optional) — the cold archive: the verbatim results a capture rule sends to `destination = 'archive'` (X and Bluesky threads, YouTube transcripts). Never chunked, embedded, extracted or served; `session_id` and `parent_episode_id` are there for deletes.

`recall()` serves the `web` bucket from `web_chunks` ([§6.4](#64-notes-web--history-legs)): BM25 and vector fused, reranked and floored, joined to artifacts, deduped per page. Only pages are served; a search-result list is stored but not chunked.

**How it's fed — server-side, off by default.** Capture runs inside `/ingest`: `episode_writer.store_episodes` hands the tool calls of the turns it stored (and of turns already stored, so a re-ship or a history import captures what an earlier run missed) to `ingestion.capture` after its transaction commits, in a transaction of its own, with a savepoint per call; a failure is logged and counted, never raised. Set `SYNAPSE_WEB_CAPTURE=on` to enable it (unset or `off` is off; any other value stops the server at start with the bad value named). Capture works from the records the Stop hook already ships, so it needs nothing installed on a client beyond the plugin, and it is by **shape, not tool name**: a call is captured when its input carries a URL or a search query and its output is page-shaped (one public URL plus a substantial body) or result-list-shaped (two or more public items). Operation-specific parsers (`web_extractors`, picked by operation name under any MCP server name, so Firecrawl and Exa are recognised wherever they are mounted) only refine title, author and date.

- **Never stored:** private sessions, non-public URLs, Synapse's own tools, a sub-agent launch's result, browser automation, personal-data connectors (`rules/deny.toml`), shell output (except through a rule), error and captcha pages, a body under 400 characters, a call whose session has had a turn deleted (reason `deleted`). Credentials are redacted. Caps: 200 captures per POST and 1,000 per UTC day.
- **Sub-agent research** is captured live too: the Stop hook and the SessionStart sweep ship each session's `subagents/**/agent-*.jsonl` (including a workflow's agents under `subagents/workflows/<wf>/`), every file under its own cursor, from byte 0, and only while the server's last `/ingest` reply said `"capture": true` (unknown means off). For the same gate the hook replaces a "Full output saved to: <path>" redirect in a web-research-shaped `tool_result` with that file's text (under `~/.claude/projects` only, at most 1 MB per result and 2 MB per POST). A private session sends neither. Codex rollouts carry tool output inline (an MCP call's JSON envelope is unwrapped); Codex's native `web_search` records the query and no results, so it cannot be captured.
- **Rules** (`ingestion/capture/ruleset.py`, whose module docstring is the reference) teach capture a source the shapes cannot see. Shipped in `ingestion/capture/rules/`: `x_thread.toml`, `bsky_thread.toml`, `yt_transcript.toml` (destination `archive`) and the editable `deny.toml`. An operator directory, `SYNAPSE_CAPTURE_DIR` (mount it read-only into `mcp-server`: `compose.yml` has the commented `./capture-rules:/capture-rules:ro` volume line), overrides a shipped rule by file name, `enabled = false` turns one off, and an operator `deny.toml` replaces the shipped one. A rule has `[match]` (tool globs and/or a command regex with named groups), `[fields]` templates (`{group}`, `{input.x}`, `{output}`, `{json.a.0.b}`; a list is alternatives), `destination`, `kind`, `kg_extract`, `min_chars`, and optionally a `hook = "file.py:function"`. **Operator rules and hooks run as code on the server**: edit, then restart the MCP server; a bad file stops the server from starting.
- **Dry run first.** `POST /capture/dry-run` (client gate, the `/ingest` body) runs the detector, rules, private-session check, duplicate check and caps over posted records and reports what a live capture would do, by tool (`{"status", "enabled", "calls", "ignored", "errors", "tools": {<tool>: {"captured", "skipped": {reason: n}, "samples"}}}`). It writes nothing; it does not run the episode guards (tombstoned, contaminated, scratch and replay turns), and it counts duplicates and caps per POST. `synapse capture --dry-run|--backfill [--since 30d|12h|2w|YYYY-MM-DD]` walks `~/.claude/projects` (sub-agent files included) and `~/.codex/sessions`, asks the server first (an old server gets "update the server"), and `--backfill` is refused unless the server has capture on and asks before it posts, because turns the server has not stored are ingested and extracted too. Order of adoption: deploy, `synapse capture --dry-run`, read it, set `SYNAPSE_WEB_CAPTURE=on`, optionally `synapse capture --backfill --since 30d`.

**The poller finishes the job.** The maintenance thread chunks new pages (`chunk_web_artifacts`), embeds them, writes their context prefixes, and enqueues extraction ([§4.5](#45-poller-two-threads)); stage order is chunk rebuild, web chunking, embed pending, web context, web extraction enqueue, stale-claim sweep, so a page's chunks reach vectors one cycle after their prefix. Extraction (`web_enqueue`) is **strictly opt-in**: only an artifact whose `metadata.kg_extract` is true is read by the knowledge graph (a rule sets it for a source it trusts; `scripts/ingest_research_brief.py` marks its briefs), and a chunk that is mostly link syntax, under 200 characters, or a near-duplicate page is skipped. A generic capture is searchable and never extracted. To redo every context prefix after a prompt change, `UPDATE web_chunks SET context_prefix = NULL` (the poller refills 20 pages a cycle and re-embeds); to extract an old unmarked page's chunks, set `metadata = metadata || '{"kg_extract": true}'` on its artifact.

**Who is served and who can delete.** A captured page carries the `project` and `surface_id` of the turn that fetched it (`web_artifacts.metadata`). A restricted caller is served the pages of its allowlisted projects plus those its own credential fetched; a page without the stamps (everything stored before server-side capture, which an out-of-repo cron wrote), an empty allowlist with no own surface, an unknown surface and the root token are served nothing. Deleting a turn from the dashboard, or purging a private session, also deletes the pages and archive rows captured from it (the `web_deleted` count), their chunks and queue rows by foreign key, and the facts the web lane extracted from those pages (unlinked instead when a surviving episode also backs them). A sub-agent's captures belong to no turn, so only a session purge removes them.

The earlier arrangement, a client-side cron that walked the session JSONLs and wrote `web_artifacts` itself (with `ingestion_state` mtime checkpoints and `scripts/contextualize_web_chunks.py`), is gone with the modules it imported. Retire its crontab line once `SYNAPSE_WEB_CAPTURE=on` is set and the clients run plugin 0.24.0 or later: before that the Stop hook does not send sub-agent transcripts, and most deep research runs there.

---

## 9. Database Schema

Migrations are numbered SQL files applied with `scripts/apply_schema.sh` (no runner — small project; the script is the single source of truth for order: every `schema/NNN_*.sql` ascending, then it stamps `synapse_meta.schema_version`). 001–076 exist; 010 was removed from the apply list. Migrations marked *optional* carry `-- schema-check: optional`: the code probes for what they add and keeps its previous behaviour until they are applied, so the image can land before the hand-applied file. A file numbered below a database's stamp is applied with `scripts/apply_schema.sh --from NNN`.

- **001–009** — base: `episodes`, `extraction_queue`, `ingestion_state` (001; nothing reads or writes it since the client-side web cron went); `span_id` (002); voyage dims 3072→2048 (003); session embeddings, now deprecated (004); ParadeDB BM25 (005); `retrieval_count` (006); drop old `session_summaries` (007); `chunks` (008); `synth_documents` (009, now legacy).
- **010 `memory_proposals`** — was the dream stage-3 auto-memory-proposal sink; stage 3 was deleted 2026-07-24 and 047 drops the table (010 removed from the apply list).
- **011 `web_artifacts`** — web-research capture ([§8](#8-web-research-subsystem)).
- **012 `web_chunks`** — embedded web slices.
- **013 `web_chunk_context`** — contextual-retrieval prefix column.
- **014 halfvec HNSW** — `episodes.embedding` and `web_chunks.embedding` indexed as `hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WHERE is_embedded = TRUE`. `synth_documents` and `chunks` deliberately left on exact scan (tiny/unused).
- **015 `extraction_queue.priority`** — priority lane (new ingest 0 < backfill 10) + partial claim index.
- **016 `extraction_queue.claimed_at`** — stale-claim sweep support.
- **017 `kg_entities` + `kg_relationships`** — the knowledge graph **in Postgres** (the FalkorDB→PG migration target, #67). `owner_id` multi-tenant seam; halfvec HNSW on entity embeddings and on live-fact embeddings (partial `WHERE t_invalid IS NULL`); ParadeDB BM25 on fact text; the full bitemporal quad on edges.
- **018 web→KG lane** — web-artifact extraction into the KG.
- **019 `kg_relationships.mention_count`** — fact re-assertion signal, bumped at the Stage-7 dedup-skip.
- **020 entity-type taxonomy** — data-driven 2-level supertype/subtype (`kg_supertypes`, `kg_entity_types` tables + `entity_supertype`/`entity_subtype` columns).
- **021 `recall_metrics`** — fire-and-forget recall telemetry, one row per serve (`kind='recall'` for `recall()`, `'episodes'` for `recall_full_turns`, `'fetch'`, and `'board'` for each board serve, whose `served_ids.events` holds the `t:` ids). `est_tokens` includes the notes bucket; `n_entities`, `n_timeline`, `ms_timeline`, `n_prefs` and `ms_prefs` are no longer written (the columns remain, NULL on new rows, until a later migration drops them once no older image can run). The nightly dream prunes rows older than 180 days.
- **022–023 `skills_lane`** — dream→skills candidate ledger, firing log, skill registry, scan cursor (022); ledger hardening with the grounded-signal CHECK constraint (023).
- **024–026 skill serving + sync** — `skill_registry` body/scope/status + `skill_files` (024); `content_modified_at` for two-way sync (025); append-only `skill_history` + trigger (026).
- **027 `skill_registry.proposal_body`** — full proposal text on the registry row.
- **028–029 episode-validity overlay** — `kg_relationships.invalidated_by` provenance link (028) + partial GIN on superseded episode ids (029), powering the `superseded_by` annotation on served episodes.
- **030–032 config lane** — mirrored config-file store, scope column, scan cursor.
- **033 `timeline_events`** — the episodic date log ([§3.5](#35-timeline-events--the-episodic-date-log)).
- **034 embedding meta** — `synapse_meta` records provisioned embed dims/model; app validates against it at boot. The same table carries the `schema_version` stamp `apply_schema.sh` writes after a full run — every service checks it at startup (`ingestion/schema_check.py`) and refuses to boot when the database is behind the code's schema, pointing at the upgrade command instead of failing mid-request (`SYNAPSE_SCHEMA_CHECK=0` skips). Two exceptions only warn: a database *ahead* of the build (migrations are additive), and one behind only by migrations whose header carries `-- schema-check: optional`, the marker for a migration whose code probes for it and keeps its pre-migration behaviour until it is applied (056 was the first; 062–076 all carry it), so the image can land before the hand-applied migration without a crash loop.
- **035 `preferences`** — standing user-preference store. Fed by the ingestion preferences gate and read by `/preferences/top` (the SessionStart block) and the dashboard. It was ALSO a `recall()` bucket until 2026-07-27, when the leg was removed: measured on `recall_feedback` since all six buckets became citable, prefs served 380 rows for 6 helpful citations (1.6% of served, 11.1% of the rated subset) against episodes 39.0% / facts 16.8% / timeline 8.2% — ~2.68 items of context cost on nearly every recall for nothing.
- **036 content-md5 index** — `episodes (md5(content))`, the cross-session replay guard ([§4.1](#4-ingestion-pipeline)).
- **037 `timeline_events.reported_count`** — the original confirm-merge counter ([§3.5](#35-timeline-events--the-episodic-date-log)); no longer written (064 replaces it), kept because the previous image still bumps it.
- **038 `timeline_events.domain`** — `personal`/`technical` scoping label ([§3.5](#35-timeline-events--the-episodic-date-log)).
- **039 `recall_metrics.served_ids`** — which ids each recall served, per bucket (`e:` episodes, `f:` facts, `n:` notes, `w:` web), for the episode-validity overlay and retrieval-quality debugging ([§6.1](#61-return-shape)); also the timeline/preferences telemetry columns the writer used to fill.
- **040 `dedup_gate_shadow`** — was the shadow log for a Stage-6 embedding-similarity gate meant to skip the 6b LLM confirm on clear-cut pairs. Measured over its shadow run (2026-07 to 2026-10), a 0.95 merge threshold would have swallowed ~1.6% of genuine corrections (similarity barely registers digits: a changed dose, version or PR number reads as a duplicate), and it could never skip the batched call (BM25-only candidates carry no score, so they are always gray). The gate was deleted 2026-10-09 and 059 drops the table.
- **041 `notes`** — the explicit-memory store behind the always-injected board: a ~120-char `hook` (the board line, also the embed target for dedup KNN) and a self-contained `body`. The live set is `superseded_by IS NULL`; a contradiction supersedes (lineage), a restatement updates in place.
- **042 `dashboard_flags` / `dashboard_audit`** — the dashboard's only writable state: a soft, reversible "look at this" mark on any memory item, and the append-only history of flag toggles and proposal decisions.
- **043 dashboard notify** — `AFTER INSERT` triggers on `episodes`, `kg_relationships` and `timeline_events` that `pg_notify('dash_feed', …)` for the dashboard's SSE live feed.
- **044 `dream_runs`** — one bookkeeping row per nightly run: per-lane outcome, counts and samples ([§11](#11-nightly-maintenance--dream-pipeline)).
- **045 skills v2 foundation** — richer candidates (a verbatim quote, `scan_night` recurrence), retune directions, and a `skill_usage` data repair.
- **046 `recall_feedback`** — labeled retrieval-quality reports from agents (`helpful` / `noise` / `missing` ids and a free-text note): offline data for eval goldens and reranker tuning.
- **047 drop `memory_proposals`** — see 010.
- **048 `recall_feedback.found_via`** — where the missing information was eventually found, which tells a serving miss from an ingest gap.
- **049 `recall_metrics` notes telemetry** — `n_notes` and `ms_notes` for the notes leg.
- **050 `private_sessions`** — the durable off-the-record list behind private mode, enforced at the `/ingest` chokepoint and by the backfill ([§4.1](#41-the-ingest-endpoint)).
- **051 `notes_curation`** — the audit log and re-judging memo of the notes lane ([§11](#11-nightly-maintenance--dream-pipeline)).
- **052 `remember_intents`** — idempotency ledger for spooled `remember()` writes ([§7.2](#72-custom-http-routes-ingest-recall)).
- **053 audience scoping** — `surfaces` (a machine's trust tier and project allowlist) and `notes.audience` ([§7.5](#75-audience-scoping--credential-bound-surface-trust-schema-053--054)).
- **054 credential-bound trust** — a surface is a credential, not a hostname: per-device tokens bound to a surface row (§7.5).
- **055 `skill_proposal_base_hash`** — the registry body a proposal was drafted against, so an accept refuses to revert an edit made since.
- **056 `kg_relationships.source_projects`** — the cached provenance set that lets a restricted caller be served facts (§7.5). Optional.
- **057 `episodes.surface_id`** (and `timeline_events.surface_id`) — a provenance stamp so a restricted surface can read back what it ingested itself (§7.5). Optional.
- **058 `kg_relationships` evidence currency** — `ongoing`, `last_supported_at`, `last_supported_by` ([§5.4](#54-bitemporal-edges)). No backfill: attribution is not recoverable from stored edges.
- **059 drop `dedup_gate_shadow`** — see 040. Marked `-- schema-check: optional`: nothing reads the table, so the image can land before the drop is applied.
- **060 `skill_registry.retired_at`** — when a skill was retired, so a retirement holds against stale publishes ([§7.6](#76-skills-serving)). Rows already retired are backfilled from `updated_at`. Marked optional: the routes probe for the column and keep their pre-060 behaviour until it exists.
- **061 `episodes (project, created_at)`** — the board banner's project list walks the distinct projects through it. Optional.
- **062 `extraction_queue.chunk_id`** — a chunk's queue row names the chunk and stores no text: the worker reads the chunk by primary key, the enqueue dedup is an `ON CONFLICT` on the unique index, and deleting the chunk deletes the row. Also an `episode_id` index for the delete cascade. Build the indexes `CONCURRENTLY` on a populated queue (steps in the file header). Until applied, a chunk item keeps its text in the row.
- **063 unembedded-episodes index** — a partial index for the poller's "which episodes still lack a vector" question, so the usual answer (nothing) stops costing a scan. `CONCURRENTLY`; the query is the same without it.
- **064 `timeline_events.reported_refs`** — the append-only list of re-telling turns' `ep:<id>` refs for confirm-merge ([§3.5](#35-timeline-events--the-episodic-date-log)): a retried merge finds its own ref and changes nothing, and a delete re-points the event at a surviving re-telling turn instead of deleting it. `reported_count` is no longer written but stays (the previous image still bumps it) until a later migration drops it. Until applied, a merge only moves `t_valid` and a delete takes a re-told event with its owner turn.
- **065 live-edge hop indexes** — `kg_rel_src_live` / `kg_rel_tgt_live`: partial btrees on `(owner_id, group_id, src_uuid | tgt_uuid, t_valid DESC) WHERE t_invalid IS NULL` that hand the KG leg's hop its 8 newest edges per seed without reading a supernode whole ([§6.3](#63-kg-fact-leg-mcp_serverkg_pgpy)). `CONCURRENTLY`; a failed build leaves an INVALID index, so drop it and re-run the file. The recall query returns the same rows without them.
- **066 drop the episode retrieval-count index** — drops `idx_episodes_retrieval_count` (`CONCURRENTLY`; nothing reads it) and sets `episodes` to `fillfactor 90`, so the `retrieval_count` bump `fetch`, `fetch_session` and `recall_full_turns` make can be a HOT update. Safe for the previous image.
- **067 chunks vector index** — `chunks_embedding_hnsw`, a halfvec HNSW over `chunks.embedding`, needed only before turning on `SYNAPSE_RECALL_CHUNK_CANDIDATES` ([§6.2](#62-episode-leg-the-workhorse)). Build it out-of-band and `CONCURRENTLY`, exactly as the file header says: a plain build blocks chunk inserts, and the parallel build needs more `/dev/shm` than the container has. Without it the setting's vector half says so in `warnings`.
- **068 dream lane cursors** — `dream_lane_cursors` (the last `episodes.id` each lane covered) and `dream_session_memo` (sessions a lane judged, at how many user turns) ([§11](#11-nightly-maintenance--dream-pipeline)). Until applied, a window starts from the lane's legacy `last_scan_at` and a long session is re-sent in full every night.
- **069 skill candidate open identity** — a retune's identity key `(kind, name, direction)` is unique among the OPEN candidates only, so a sighting after an accept or a retirement opens a fresh candidate instead of folding into the closed row. Apply it soon after the image: until then a re-sighting of a closed retune's key is dropped with a warning. Do not roll the image back past it (the previous image's `ON CONFLICT` clause cannot infer the new index).
- **070 skill draft attempts** — `draft_attempts`, `last_draft_at` and `last_draft_error` on the candidate ledger, so a failing draft backs off (1, 2, 4… days, at most 16) and says why in `skill_review show`. Until applied every failing draft is retried nightly.
- **071 `notes_history`** — the text a note had before `remember()` rewrote it in place: an operator's undo copy (`SELECT * FROM notes_history WHERE note_id = N`), never read at run time. Until applied the rewrite leaves no record.
- **072 KG dashboard reads** — backfills `t_created` / `entity_subtype` from the FalkorDB-era `created_at` / `entity_type` mirrors where only the legacy column was ever written, and builds `kg_rel_t_created_idx` for the feed. On a populated table don't run the file whole: run its UPDATEs, then build the index `CONCURRENTLY` by hand, as its header says. Apply soon after the image: until the UPDATEs have run, a legacy-only fact is absent from the feed.
- **073 `deleted_turns`** — tombstones (`span_id`, `content_md5`, project/cwd) for turns deleted from memory, checked at the `/ingest` chokepoint so a re-shipped transcript does not bring a deleted turn back. Rows are permanent by design. Until applied, a delete leaves no tombstone and ingest checks nothing.
- **074 GIN on `kg_relationships.episodes`** — for the "facts that cite episode N" lookup (056's relabel trigger, the dashboard drill-down and episode delete). `CONCURRENTLY`; the lookups are correct without it, only slower.
- **075 drop redundant indexes** — the dead `search_cache` table and the btrees another index already covers (`episodes_session_idx`, `episodes_project_idx` — only once 061's index is valid — `kg_rel_src`, `kg_rel_owner`, `kg_entities_owner`). `CONCURRENTLY`; safe for the previous image.
- **076 `research_archive`** — the cold archive of the verbatim results a capture rule sends to `destination = 'archive'` (X and Bluesky threads, YouTube transcripts): never chunked, embedded, extracted or served. Gives a table an out-of-repo cron may already have created the `session_id` / `parent_episode_id` provenance a delete needs. Until applied, archive-rule captures are skipped (`archive_unavailable`) instead of failing.

Key live tables: `episodes`, `chunks`, `extraction_queue` (status `pending|processing|done|failed`, `priority`, `claimed_at`), `kg_entities`, `kg_relationships`, `web_artifacts`, `web_chunks`, `notes`, `deleted_turns` and the `skills_lane.*` schema. `synth_documents` exists but is dormant (`memory_proposals` was dropped by 047).

---

## 10. Postgres Graph Model

The graph is **two Postgres tables** (`kg_entities`, `kg_relationships`, schema 017), not a separate database. FalkorDB was decommissioned in #67 after the cutover measured quality-equal on the relational golden set (delta −0.017, CI spans zero) and removed a moving part. Scope is selected by `group_id` (**`technical`** default, **`personal`**); `owner_id` (default `'default'`) is the forward-compatible multi-tenant seam, so a write can never land unscoped. A work-only deployment can retire the second scope with `SYNAPSE_PERSONAL_SCOPE=0` ([§13](#13-configuration)): writes all route to `technical`, and a read that asks for `personal` is answered from it.

**`kg_entities`**:
```sql
kg_entities (
    uuid, owner_id, group_id, project,
    name, normalized_name,         -- normalized_name = lower/trim/punct-stripped dedup key (NOT unique; #49)
    entity_supertype, entity_subtype,  -- the 2-level taxonomy (#020); legacy entity_type is no longer written
    summary,
    embedding vector(2048),        -- Voyage embedding of the entity NAME
    degree,                        -- a one-off backfill value nothing maintains: recall and the dashboard count live edges from kg_relationships
    t_created, t_valid             -- legacy created_at/valid_at mirrors are no longer written
)
```

**`kg_relationships`** (bitemporal):
```sql
kg_relationships (
    uuid, owner_id, group_id,
    src_uuid, tgt_uuid,            -- entity uuids, matched at read time (not FKs: an edge can be written before its endpoints)
    name,                          -- predicate verb (uses, implements, causes, replaces, depends_on, ...)
    fact,                          -- full searchable statement
    fact_embedding vector(2048),   -- Voyage embedding of the FACT text
    episodes JSONB,                -- provenance: source episode ids
    mention_count,                 -- re-assertion count, bumped at the dedup-skip (#019)
    ongoing,                       -- claim can lapse silently (habit/usage/state); NULL pre-058 (#058)
    last_supported_at,             -- conversation time of the last USER-attributed assertion (#058)
    last_supported_by,             -- that assertion's episode id (#058)
    retrieval_count,
    t_created, t_valid, t_invalid, t_expired   -- bitemporal quad (legacy created_at/valid_at/invalid_at mirrors no longer written); t_invalid NULL = live
)
```

**Indexes**: **halfvec(2048) HNSW** on `kg_entities.embedding` (partial `WHERE embedding IS NOT NULL`) and on `kg_relationships.fact_embedding` (partial `WHERE t_invalid IS NULL` — only live facts are ANN-searchable); **ParadeDB BM25** on `kg_relationships.fact`; btree on the `(owner_id, group_id, normalized_name)` alias lookup and the per-pair `(owner_id, group_id, src_uuid, tgt_uuid)` lookup (`kg_rel_pair`, which also serves the Stage-6a pair-pool lookup and the dashboard's per-node counts together with `kg_rel_tgt`), the optional live-edge hop partials (065), the optional GIN on `episodes` (074) and the optional `t_created` feed index (072).

> **Partial-index gotcha (the Postgres analog of the old FalkorDB `efRuntime` trap):** the live-fact HNSW index is **partial on `t_invalid IS NULL`**, so dead edges are never in the index and there is no prefilter-then-filter under-return (FalkorDB's default `efRuntime=10` returned zero rows on 39% of queries after the dead-edge post-filter). As with episodes (§6.2), the query's halfvec cast (`fact_embedding::halfvec(2048)`) must match the index expression *verbatim* to be served by HNSW.

**Graph shape**: ~41K entities, ~72K relationships (~59K live). Episodes live in Postgres alongside the graph, so there are no `MENTIONS`/episodic edges — the KG is a fact+entity store queried as pgvector-KNN + BM25 + 1-hop join, not a deep-traversal graph (measured: depth dilutes recall for this workload). Community clustering was retired with the cutover ([§3.4](#34-knowledge-graph-postgres), [§6.1](#61-return-shape)).

---

## 11. Nightly Maintenance & Dream Pipeline

> **Status:** the `dream` container's daily `run_once()` (`python -m dream --schedule`, 02:03 UTC) runs four lanes in order: **dream→skills**, **dream→config**, **dream→notes**, then a telemetry retention prune (`recall_metrics` rows older than 180 days). Every lane switch is a **kill switch**: the lanes are ON by default and `=0` turns one off (`SKILLS_LANE_ENABLED`, `CONFIG_LANE_ENABLED`, `SYNAPSE_NOTES_LANE`). A run the container missed (down at 02:03, or redeployed mid-run) starts as a catch-up (`--schedule` runs once at startup when `dream_runs` has history but no run finished since the last 02:03 UTC; a fresh install waits for 02:03), and a session-level advisory lock keeps a manual `python -m dream` from racing the scheduled run (the second one logs a warning and exits 0). A run the process died in is closed by the next start as `ok=false`, `finished_at` NULL, error `interrupted: the process stopped before the run finished`. A lane error cannot crash the loop: each lane's stages are isolated and reported, a lane that cannot reach its model stops with its cursor where it was, and one row per run lands in `dream_runs` (schema 044) with the per-lane counts, samples and errors the dashboard's Dream report reads. The legacy memory-proposal stages are gone entirely (stage 3 and its `--stage` plumbing were deleted 2026-07-24; the substrate — `doc_type='summary'` — and the auto-memory-file sink were both already retired).

**Shared plumbing** (`dream/lane.py`). A run covers the episodes with `after_id < id <= until_id`; `episodes.id` follows insertion order, where `created_at` is the conversation's event time, so a turn uploaded late (a catch-up retry, an offline laptop, an import) is still mined. Each lane keeps its own cursor in `dream_lane_cursors` (migration 068, optional) and advances it only past the sessions it got to, so a capped run's overflow rolls to the next night; `SKILLS_LANE_LIMIT` and `CONFIG_LANE_LIMIT` (30) cap the **sessions a lane judges per run**, oldest last touch first. The judges' unit is the whole session, and a judged session waits until it has 3 more user turns (`dream_session_memo`) before it is judged again. Without 068 a window starts from the lane's legacy `last_scan_at` and nothing is skipped as already judged. Every lane LLM call goes through `ingestion.llm_client` (the `claude-code` or `openai` provider; `SYNAPSE_SKILLS_JUDGE_MODEL` / `SYNAPSE_SKILLS_DRAFT_MODEL`, default Opus): a failure to reach the model raises instead of reading as "found nothing".

**dream→skills lane** (`dream/skills/`) reads its catalog from `skills_lane.skill_registry` (no disk) and runs these stages: usage sync (`[tool:Skill]` markers → `skill_usage`); judge (one analysis call per session: gaps for DERIVE, would-have-helped skills for RETUNE); grounded signals (dismissals, explicit "make a skill" asks); the detectors (struggle arcs, post-fire outcomes, the weekly procedure miner); drafts (a complete `SKILL.md` + `base_body_hash` for every proposed row, backing off 1, 2, 4… days after a failed draft once migration 070 is applied); decay of stale `observe` candidates; effect measurement of accepted retunes; and a throttled Discord digest. Every stage covers the same id range and runs on its own: a failed stage is listed in `dream_runs.errors` and the later stages still run, but the cursor holds when a window stage (usage, judge, grounded, struggle_arc, post_fire) failed, so a persistent failure shows as a failed Dream run every night and the window keeps growing until the cause is fixed. Fire events are read once, `post_fire` judges at most 20 fires per run, newest first (`skills_fires_over_cap` counts the rest), and a derive is proposed on evidence from 2 distinct days (evidence dates, not scan nights), on salience ≥ 4, or on the legacy score. LLM calls run 3 at a time (`LLM_WORKERS`, sharing the subscription with the poller); the lane has no per-call timeout of its own and relies on `llm_client`'s. **Propose-only**: nothing writes `skill_registry` until a review accept does ([§7.6](#76-skills-serving)). The lane moved here from the plugin so the server owns it; the plugin is a thin client that publishes skills to the registry and reviews proposals.

**dream→config lane** (`dream/config/`) mines the same session window for **behavioral corrections** — moments the user corrected how the agent writes, works or behaves, not task content — into `config_lane.config_proposals`. A correction must recur in 2 distinct sessions before it reaches review, proposals target a dedicated `rules/learned.md` (`CONFIG_LEARNED_FILE`) so dream never rewrites a hand-kept file, and a rule the user already has is not proposed: that is a mirrored config line (the plugin's SessionStart hook pushes the opted-in config files to `/config/publish` when `SYNAPSE_CONFIG_SYNC=1`; the dashboard lists the same mirror; `rules/learned.md` itself is exempt) or a live feedback note. "Already known" is token Jaccard ≥ 0.5 **and** the same polarity (negation read from not/never/no/avoid/stop/dont/n't); headings, labels (lines ending `:`) and lines with fewer than 3 significant tokens are not known rules, so a correction that reverses a config line is proposed and an opposite-polarity correction of an applied rule does not count as its recurrence. Every status takes part in a rule's identity: a rejected rule is suppressed for 30 days from the rejection (`updated_at`; sightings inside the window only add evidence), accepted and applied rules only collect evidence, an applied rule corrected again in a new session is counted (`config_applied_recurrences`), and `observe` rows not re-sighted for 28 days are deleted. The review is `accept`/`reject`, then `config_review apply` writes every accepted rule on a machine. Propose-only.

**dream→notes curation lane** (`dream/notes/`, kill switch `SYNAPSE_NOTES_LANE=0`, ON by default) — the notes store's missing outflow. `remember()` was the only thing that ever wrote a note, so paraphrase duplicates, notes a later correction had already replaced, and project-specific content typed as global `feedback` all accumulated on a board that is injected into every session. The lane runs three stages nightly: (A) candidate pairs from a pure-SQL self-join of live notes on the cosine similarity of their **stored hook embeddings** (no new embedding calls; floor 0.60, permissive because a correction often shares little wording with what it corrects); (B) a pair judge (the same prompt `remember()` uses, [§7.1](#71-tools)) that merges DUPLICATEs (older `updated_at` loses) and retires the note a CORRECTION replaces; (C) a scope judge that re-types a global note to `project` when its content is specific to one named codebase.

Unlike the skills and config lanes it **auto-applies** — no proposals, no review queue, no review command, by explicit user decision (curation should happen silently as a self-improvement step, not as a second inbox). The safety model is structural instead: every write is reversible and non-destructive (`superseded_by` keeps the row and the lineage; retype changes only `type`/`project`), every judgement is made in **both orders** with consensus required and disagreement collapsing to the safe verdict (a first-order DISTINCT or a cross-type DUPLICATE ends the pair after one call, memo reasons `first-order-distinct` / `cross-type-duplicate`), both prompts carry an explicit under-report bias, per-night caps bound the blast radius (40 pairs / 10 scopes / 10 applied ops, constants in `dream/notes/nightly.py`), a merge needs a winner visible wherever the loser was (a project note never retires a global or another project's note, `winner-narrower-than-loser`; a personal note never retires a work-safe one, `audience-narrowing`; pairs of two different projects are not candidates), a retype applies only for a project the judge was offered (case-folded, stored casing) and re-derives audience by rule 3, and every verdict — applied or not — is written to `notes_curation` (schema 051), which doubles as the memo that stops re-judging and the staleness cursor (`judged_at` vs `notes.updated_at`); an applied row records `from_type`, `from_project` and `from_audience`. The operator's undo is `docker compose exec dream python -m dream.notes.nightly --list [N]` then `--revert ID` (the row stays as REVERTED with its memo intact; a retype applied before the audit columns existed refuses with a clear error and is fixed by hand). A restated note updated in place is undone from `notes_history` with SQL.

The real KG-hygiene and web-capture work runs as **independent entry points**, not orchestrated by `dream/__main__`:

- **Semantic identity merge** (#48, [§5.3](#53-entity-resolution)) — exact-name and moderate (name-gated) pairs auto-merge on the write path; the lexically-different semantic lane is detection-only, surfaced for human review (the #49 gap). The former standalone `dream/entity_dedup` pass was removed in #67.
- **Communities** — **legacy/retired.** `Community` clusters (Raghavan label propagation) and `HAS_MEMBER` edges lived in FalkorDB; the bucket was retired with the #67 KG cutover (never measured as contributing), so there is no runnable module and no Postgres target.
- **Cursor ingest** — there is no live Cursor path. Cursor chat history is imported by hand: `python -m ingestion.backfill --source cursor <dir>` parses the `*.vscdb` files with `CursorSQLiteParser` (`ingestion/cursor_sqlite_client.py`) into `Episode` rows (`platform="cursor"`) through the same writer `/ingest` uses.

**Logfire**: all *read* paths were removed (#108) when Logfire ingestion was ripped out — episodes arrive via `/ingest`. Logfire **emit** (telemetry spans) is retained: the poller, the MCP server and `dream` each configure it (`dream` opens one span per lane) and send to the project named by `LOGFIRE_TOKEN`; with no token nothing is sent.

---

## 12. Key Design Decisions

### 12.1 Hybrid retrieval, episodes primary, KG as a side specialist

`recall()` serves a reranked **episode** leg as the workhorse and the **KG facts** as a side specialist for relational/multi-hop queries — the graph is never the primary retrieval path. This matches the 2026 field consensus (see [§12.7](#127-why-a-thin-graph-not-graph-maximalist)).

### 12.2 Stop-hook push over Logfire polling

**Decision**: episodes enter via a `Stop`-hook tail POST to `/ingest`, not a Logfire SQL-API poll.
**Rationale**: polling Logfire added a 5-minute latency floor, a read dependency on a telemetry vendor, and a watermark/dedup surface. A push hook ingests within the turn, dedups by stable `span_id`, and removed an entire client (`logfire_client.py`) and its failure modes. Shipping only what is past a durable per-file cursor (not the full transcript each turn) plus enqueue-gating on new `(session, span_id)` fixed an O(turns²) re-extraction loop.

### 12.3 KG facts from chunks, not summaries or episodes

**Decision**: the LLM extraction substrate is the 4-episode **chunk**.
**Rationale**: a measured A/B put chunks at ~2.66 quality-weighted facts/episode — ~3x the best summary approach and ~15x the old production path — at 87% high-value. Single episodes are too thin (shallow/hallucinated facts); summaries diluted and added a generation step. Chunks carry a full mini-task arc and preserve tool-call traces (file paths, commands). This also retired the summary layer entirely.

### 12.4 Human-turn episode granularity

One episode per human turn (grouping the 3–8 API calls between human messages), not 1:1 with transcript records — avoids ~63% context-free assistant-only fragments.

### 12.5 Bitemporal edges + invalidate-don't-delete

Facts are never deleted; contradictions set `t_invalid` and a new edge is written. This is the one KG technique the external field universally endorses, and it powers the `history` recall leg ("what was true then vs now"). `t_expired` is reserved for a future selective-forgetting/decay mechanism.

### 12.6 Deep-fetch + cross-encoder rerank (WIN1)

Fetch 100 candidates per leg, fuse, rerank with Voyage `rerank-3-lite` by default (WIN1 measured on `rerank-2.5`), serve top 5. Exact-fact misses were pure truncation (golds ranked 16–88 but prod fetched 15); the reranker — not deeper RRF — recovers them. Tokens dropped 20% because fewer, better episodes are served.

### 12.7 Why a thin graph, not graph-maximalist

A 2026 multi-source review of production agent-memory systems converged on exactly this shape, and Synapse's choices line up:

- **Graph-as-primary-path is dead.** Mem0 deleted its graph layer in v3 (Apr 2026) after measuring its graph variant beat plain vector by only ~1.5 pts while running 3x slower; GraphRAG-Bench (ICLR 2026) found GraphRAG "frequently underperforms vanilla RAG." Synapse serves the graph as a *side bucket*, never the main leg — it never made that mistake.
- **Community/cluster summarization is widely flagged as overhead.** Synapse retired summaries independently, and the community bucket along with the #67 KG cutover — neither earned a serving slot.
- **The two techniques that survive every critique** are bitemporal fact-evolution and multi-hop/relational traversal. Synapse has the first (12.5); it deliberately does *not* expose deep multi-hop connection-answering because, measured on this workload, depth dilutes recall (1-hop > 2-hop) — that would be a product feature, not a retrieval win.
- **Entity resolution is the named hard problem.** This is Synapse's one real open gap (#49) — see [§16](#16-known-gaps--planned-work).

**What we tried and retired** (don't re-litigate): summary-as-substrate; summary recall bucket; chunk recall bucket (redundant with the episode leg); co-fusion of KG facts into the episode RRF; a Haiku reranker (Voyage `rerank-2.5` won); deep (2–3 hop) traversal; episode-mention-count reranking (neutral). Each was rejected by measurement, not taste.

### 12.8 Haiku-only extraction via the Agent SDK

`claude-haiku-4-5` for all extraction/dedup/contradiction/edge-date/confirm calls, through `claude-agent-sdk` (a Claude subscription OAuth token or an `ANTHROPIC_API_KEY`; scripting a subscription OAuth token at the raw API is ToS-prohibited, hence the SDK/CLI subprocess). Thinking capped at 2048 tokens (#121 — capping *improved* extraction quality by killing filler) and the dedup candidate pool trimmed to 8.

### 12.9 Recency + feedback in episode fusion

`exp(-age_days·ln2/30)` recency multiplier (30-day half-life) and a `log1p(retrieval_count)` feedback boost are applied to the episode RRF — personal memory is strongly recency-biased ("what did we decide about X" means recently). The KG fact leg deliberately omits both (they drag relevance on facts).

### 12.10 Decisions of the 2026-10 review pass

- **Deleted:** `ContradictionDetector` and the single-fact resolve, dedup and edge-date prompts and code (retirements come only from Stage-6b verdicts; the "6b gets 79% of its candidates from BM25" note in `extraction_policy.py` is unchanged), the four per-source backfill modules (one `ingestion/backfill.py` driver replaces them), `kg_pg_read` / `kg_pg_write` (one `KGClient`), `recall_settings.py` (one frozen `RecallConfig`), the `poller-drain` service, and every flag whose winning behaviour was already the default (the cross-session replay guard, the type gate, the timeline confirm-merge, the banned-name filter, the recall floors' enforce switch).
- **Kept on purpose, measure first:** the timeline gate and the preferences gate (their bugs are fixed — retry double counts, a near-empty [user] turn is skipped, the board logs the `t:` ids it serves so `recall_feedback` can rate them — and neither is merged or retired until that data is in), and chunk embeddings and the chunks BM25 index (the optional candidate source and `scripts/ab_chunk_candidates.py` decide which side is deleted).
- **Kept as documented-intentional:** the dashboard's 1 s SSE status loop, `git_feeder`'s re-read window, the notes lane's O(N²) pair SQL, and `recall_feedback` as offline data that never feeds live ranking.
- **Defaults changed:** `SKILLS_LANE_ENABLED` is a kill switch like the other two lanes (on unless `=0`); Postgres and the opt-in local-inference port publish on `127.0.0.1` only; interactive recall gives up on a Voyage 429 in about 5 s; facts are served as `f:<bigint>`; the Codex `recall_feedback` nudge is off by default, as Claude's is.

---

## 13. Configuration

This is the reference list. The install path that uses it is [docs/install.md](docs/install.md), and `.env.example` is the annotated copy.

Every process (poller, MCP server, dream, scripts) resolves config through `ingestion/settings.py`: the process environment first (an empty variable counts as unset), then one `.env` file (`SYNAPSE_ENV_FILE`, default the repo-root `.env`, i.e. `/app/.env` in the image), parsed as compose parses it. The poller needs `SYNAPSE_DB_URL`; `POLL_INTERVAL_SECONDS` and `VOYAGE_API_KEY` are optional there. Five things are required for a working install: `SYNAPSE_DB_PASSWORD` + `SYNAPSE_DB_URL` (storage), an embedding/rerank backend (`VOYAGE_API_KEY`, or the `SYNAPSE_EMBED_*` / `SYNAPSE_RERANK_*` alternatives), an extraction-LLM credential (`CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`, without which nothing reaches the knowledge graph), and `SYNAPSE_MACHINE_TOKEN`. Everything else has a sane default. A variable listed here as removed is ignored if a `.env` still sets it.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_DB_URL` | required | psycopg3 DSN to Postgres on the server host |
| `VOYAGE_API_KEY` | required* | Voyage AI key (embeddings + rerank); *not needed when a non-Voyage embed/rerank backend is configured below |
| `SYNAPSE_DB_PASSWORD` | required (compose) | Postgres password injected into the `postgres` service by `compose.yml`; must match the password in `SYNAPSE_DB_URL` |
| `MCP_PORT` | `8765` | MCP streamable-HTTP / `/ingest` / `/recall` port |
| `POSTGRES_HOST_PORT` | `5432` | host port Postgres publishes on, bound to `127.0.0.1`. `compose.yml` has a commented override line for remote access; an existing deployment must re-apply the compose file for the binding to change |
| `COMPOSE_PROFILES` | unset | read by compose from `.env`; `local-inference` starts the bundled embedding/rerank container, also bound to `127.0.0.1` |
| `POLL_INTERVAL_SECONDS` | `300` | pacing of the poller's maintenance thread |
| `LOGFIRE_TOKEN` | optional | telemetry **emit** only |
| `LOGFIRE_SERVICE_NAME` | `synapse-mcp` / `synapse-poller` / `synapse-dream` | service name on emitted spans (compose sets it per container) |
| `SYNAPSE_ENV_FILE` | `<repo>/.env` | the `.env` fallback path for every process. Real env vars always win over it; tests that reload `mcp_server.server` point this at a nonexistent path so a developer's local `.env` can't satisfy an assertion about *unset* config |
| `SYNAPSE_SCHEMA_CHECK` | `1` | boot-time guard: refuse to start when the database schema is behind the code (`0` = skip) |
| `SYNAPSE_TIMEZONE` | `America/Toronto` | the user's IANA zone. Turns the stored UTC instants into the user's calendar day wherever a conversation date is derived: the extraction prompt's session date, the edge-date reference time, the timeline gate's turn date, the board's "Last 7 days" dates and per-project-day cap, `/preferences/top`'s `since`, and the date and `supported` recall serves on episodes and facts. Unset, a turn after 20:00 EDT would be dated the next day. An unknown name logs a warning and uses UTC. Existing rows are not rewritten |
| `SYNAPSE_KG_OWNER_ID` | `default` | `owner_id` scope for all KG reads/writes, one constant read at import (`ingestion.settings.KG_OWNER_ID`). The multi-tenant seam, not a knob to turn: changing it strands every existing row |
| `SYNAPSE_OWNER_NAME` / `SYNAPSE_OWNER_ALIASES` | `User` / empty | canonical name of your identity hub in the knowledge graph, and extra comma-separated spellings that resolve to it |
| `SYNAPSE_PERSONAL_PROJECTS` | empty | extra project slugs (beyond `jobs`, `personal`, `email-templates`) whose content defaults to the personal graph |
| `SYNAPSE_PERSONAL_SCOPE` | `1` | keep the `personal`/`technical` split in the KG and on timeline events. `0` routes every extracted entity, fact and event to the single `technical` graph and domain, drops the domain question from the timeline gate prompt, and answers `group_id="personal"` from the technical graph. For work-only deployments. Existing rows are not moved |

`CLAUDE_CODE_OAUTH_TOKEN` (Max-subscription auth) or `ANTHROPIC_API_KEY` must be present in the environment for extraction, but neither is read by Synapse code — they are consumed by the spawned `claude` CLI that `claude-agent-sdk` drives ([§2](#the-llm-call-path-claude-agent-sdk)).

### LLM backend (`ingestion/llm_client.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_LLM_PROVIDER` | `claude-code` | `claude-code` (Agent SDK via the token above) or `openai` (any OpenAI-compatible `/chat/completions` endpoint: OpenRouter, Ollama, vLLM) |
| `SYNAPSE_LLM_BASE_URL` / `SYNAPSE_LLM_API_KEY` | OpenRouter / empty | `openai` provider only; the key may be blank for keyless endpoints |
| `SYNAPSE_OPENROUTER_PROVIDERS` | unset | OpenRouter only: provider slugs to prefer, comma-separated, or `auto` to rank the model's providers cheapest-first from OpenRouter's live listing |
| `SYNAPSE_LLM_MODEL` | stage default | global model override, **provider-specific**: the CLI's own name (`claude-haiku-4-5`) under `claude-code`, the routed id (`anthropic/claude-haiku-4.5`) under `openai`. A slashed id under `claude-code` is refused at startup. Leave it unset unless you mean it: it also overrides the Opus defaults of the dream stages |
| `SYNAPSE_<STAGE>_MODEL` | unset | per-stage override, falling back to `SYNAPSE_LLM_MODEL`, then the stage default. Stages: `EXTRACTOR`, `TIMELINE`, `PREFERENCES`, `DEDUP`, `CONTRADICTION`, `EDGE_DATES`, `NOTES_CONFIRM` (the write-time judge in `remember()`), `NOTES_CURATE` (the nightly notes lane's judge), `SKILLS_JUDGE` and `SKILLS_DRAFT` (default `claude-opus-4-8`: the judge of the skills and config lanes, and the `SKILL.md` drafter), `CONTEXTUALIZE` (the poller's web-chunk context prefixes, default Haiku) |

### Embeddings & rerank backends (`ingestion/embedding.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_EMBED_PROVIDER` | `voyage` | embedding backend: `voyage` (default) or `openai` (any OpenAI-compatible POST `/embeddings` — OpenAI, Ollama, TEI, Infinity, vLLM) |
| `SYNAPSE_EMBED_BASE_URL` | — | `openai` provider: endpoint base URL (e.g. `http://local-inference:7997`) |
| `SYNAPSE_EMBED_API_KEY` | empty | `openai` provider: bearer token; blank for keyless local servers |
| `SYNAPSE_EMBED_MODEL` | `voyage-4-large` | embedding model id (required for `openai`) |
| `SYNAPSE_EMBED_DIMS` | `2048` | embedding width; **fixed at first-boot schema provisioning** (apply_schema.sh substitutes it into the vector/halfvec DDL) and recorded in `synapse_meta` — the factory fails loudly on a config↔database mismatch |
| `SYNAPSE_EMBED_SEND_DIMS` | auto | force the OpenAI `dimensions` request param on (`1`) / off (`0`); default sends it only for `text-embedding-3-*` |
| `SYNAPSE_RERANK_PROVIDER` | `voyage` | rerank backend: `voyage` (default), `http` (TEI/Infinity/Cohere-shape POST `/rerank`), or `none` (fusion-only serving — logged once at startup) |
| `SYNAPSE_RERANK_BASE_URL` | — | `http` provider: rerank endpoint base URL |
| `SYNAPSE_RERANK_API_KEY` | empty | `http` provider: bearer token; blank for keyless local servers |

### Auth (§7.4)

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_MACHINE_TOKEN` | **required** | ROOT bearer: the services' credential for `/ingest` + the internal write lanes. Generate with `openssl rand -hex 32`. Deliberately NOT sufficient to enroll or mint a device, list surfaces, or reach `/dash/api`, and it resolves to no surface, so a caller presenting it is served restricted ([§7.5](#75-audience-scoping--credential-bound-surface-trust-schema-053--054)). Blank → the server refuses to start unless `SYNAPSE_ALLOW_OPEN=1` |
| `SYNAPSE_ALLOW_OPEN` | `0` | dev-only escape hatch: `1` starts the server with no machine token. Every caller then resolves to `UNKNOWN_SURFACE` and is served restricted, so the board and recall come back empty. Not a deployment mode |
| `SYNAPSE_PUBLIC_URL` | `https://synapse.example.net` | public base URL advertised in OAuth discovery metadata |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | optional | GitHub OAuth app for the claude.ai web connector; unset → no OAuth leg |
| `ALLOWED_GITHUB_USERS` | empty | comma-separated GitHub logins admitted by the OAuth leg |
| `SYNAPSE_OAUTH_SIGNING_KEY` | optional | stable signing key so issued OAuth tokens survive server restarts |
| `OIDC_CONFIG_URL` | unset | any OIDC IdP instead of GitHub: the provider's `/.well-known/openid-configuration`. Set → replaces the GitHub leg for this deployment (MCP discovery advertises one authorization server) |
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | unset | OIDC client credentials; register `{base_url}/auth/callback` and `{base_url}/auth/callback/dash` as redirect URIs |
| `ALLOWED_OIDC_USERS` | empty | comma-separated identities admitted by the OIDC leg, matched against `preferred_username` then `email` |
| `OIDC_SCOPES` | `openid profile email offline_access` | trim `offline_access` for IdPs that reject it (Google), at the cost of short-lived connector sessions |
| `OIDC_USER_CLAIMS` | `preferred_username,email` | ordered claims read for the user's identity |
| `SYNAPSE_MACHINE_ROLE` | `personal` | **plugin side**, not the server: this machine's declared role at enrollment, `personal` (full trust) or `work` (restricted). An *unstated* role resolves to restricted server-side ([§7.5](#75-audience-scoping--credential-bound-surface-trust-schema-053--054)) |

### Recall tuning (`mcp_server/recall_config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_RECALL_PASSAGE_N` | `6` | passages served per `recall()` from the second-pass passage rerank |
| `SYNAPSE_RERANK_RECENCY_FLOOR` | `0.25` | lower bound of the recency multiplier applied to rerank scores (14-day half-life): a ~3-month-old episode keeps a quarter of its score instead of ~0.01x, so a query that asks for old content still gets it |
| `SYNAPSE_RECALL_FLOOR` | `0` (off) | abstention floor on the raw (pre-recency) rerank top score of the episodes that survive echo suppression: strictly below it, `recall()` serves no episodes for that call instead of the least-bad ones, and `recall_metrics.served_ids` gains `would_abstain`/`floor` markers (`rerank_top_all` records the pool's own top when it differs; `recall_full_turns` records the marker but never withholds). Always enforced when set. The threshold belongs to the reranker (scores are not comparable across models), so each `examples/env` preset sets it: `0.60` for the default `rerank-3-lite` (`0.58` if you roll back to `rerank-2.5-lite`), `0` for the bundled local reranker |
| `SYNAPSE_RECALL_WEB_FLOOR` | `0.60` | cross-encoder floor on the web bucket, with no keep-one backstop: the bucket may serve nothing, which gates web on intent. Reranker-specific like `SYNAPSE_RECALL_FLOOR` (`0` = off; the local preset) |
| `SYNAPSE_RECALL_NOTES_FLOOR` | `0.60` | the same floor for the notes bucket |
| `SYNAPSE_RECALL_CHUNK_CANDIDATES` | `0` | `1` adds the top 10 chunks by keyword and by embedding as a candidate source for the episode pool ([§6.2](#62-episode-leg-the-workhorse)); chunks are never served. The vector half needs the optional chunks index (migration 067). `scripts/ab_chunk_candidates.py` runs the on/off A/B over a goldens file; the losing side gets deleted |
| `SYNAPSE_SUPERSEDE_MAX_DIST` | `0.45` | cosine-distance gate for surfacing successor facts of superseded edges |
| `SYNAPSE_SUPERSEDE_LINK_MAX_DIST` | `0.35` | a successor link is believed only when the successor is within this distance of the fact it retired; recalibrate with the embedder |

### Ingestion & extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_DRAIN_CONCURRENCY` | `4` | queue items one poller replica extracts at once: the one per-deployment burst knob (raise it for a backlog instead of adding replicas). Each worker holds its own Postgres connection and LLM-client stack; `1` is the serial kill switch. A claim is one row per worker thread |
| `SYNAPSE_DRAIN_ONLY` | unset | deprecated, honoured for one release: `1` makes the replica run only the extraction thread and logs a warning. The `poller-drain` compose service it served is gone |
| `SYNAPSE_RERANK_MODEL` | `rerank-3-lite` | rerank model: for `voyage` the Voyage cross-encoder (set `rerank-2.5-lite` to roll back); for `http` the served model id |
| `SYNAPSE_TIMELINE_GATE` | `1` | the per-turn LLM gate that promotes ingested turns to timeline events (`0` = off) |
| `SYNAPSE_PREFS_GATE` | `1` | the per-turn gate that extracts standing preferences (`0` = off) |
| `SYNAPSE_SCRATCH_PROJECTS` | `axon-test,axon_test,test,tmp,scratch` | exact project names dropped at ingest as throwaway workspaces; a value REPLACES the defaults, empty turns the filter off |
| `SYNAPSE_DEDUP_CACHE_TTL_SECONDS` | `900` | how long a worker's entity-dedup index (MinHash/LSH over the graph's entity names) is reused before it is rebuilt |
| `SYNAPSE_WEB_CAPTURE` | off | `on` captures web research server-side out of the records `/ingest` receives ([§8](#8-web-research-subsystem)); unset or `off` is off, anything else stops the server at start. Per deployment: check `synapse capture --dry-run` first |
| `SYNAPSE_CAPTURE_DIR` | unset | a directory of capture rules (`*.toml` plus the Python hooks they name), mounted read-only into `mcp-server`; they run as code on the server |
| `SYNAPSE_DASH_ORIGINS` | empty | extra origins (`scheme://host[:port]`, comma-separated) the dashboard's browser login may return a token to; `SYNAPSE_PUBLIC_URL` is always allowed |

Removed (the winning behaviour is now unconditional): `SYNAPSE_CONTENT_DEDUP`, `SYNAPSE_DEDUP_TYPE_GATE`, `SYNAPSE_TIMELINE_DEDUP`, `SYNAPSE_TIMELINE_GROUP_SCOPE`, `SYNAPSE_BANNED_NAME_FILTER`, `SYNAPSE_NOTES_CONFIRM`, `SYNAPSE_NOTES_SIM_MATCH`, `SYNAPSE_DREAM_MODEL`, `SYNAPSE_QUERY_GRAPH_MODEL`, `SYNAPSE_KG_READ`, and the recall knobs listed in [§6](#6-recall-pipeline).

### Dream, skills lane & config lane

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLS_LANE_ENABLED` | on | kill switch for the nightly dream→skills lane ([§11](#11-nightly-maintenance--dream-pipeline)): on unless `=0` |
| `SKILLS_LANE_LIMIT` | `30` | substantive sessions the skills lane judges per run, oldest last touch first; the rest roll to the next night |
| `CONFIG_LANE_ENABLED` | on | kill switch for the nightly config lane (`=0` = off) |
| `CONFIG_LANE_LIMIT` | `30` | sessions (with 2 or more user turns) the config lane judges per run (same cursor rule) |
| `CONFIG_LEARNED_FILE` | `rules/learned.md` | target file (relative to the config mirror) for learned behavioral rules; the only config-lane override (the reject window and observe TTL are constants in `dream/config/nightly.py`) |
| `SYNAPSE_NOTES_LANE` | on | kill switch for the nightly dream→notes curation lane (`=0` = off); its caps are constants and `run_lane` arguments / CLI flags |
| `SKILLS_DISCORD_WEBHOOK` | empty | optional webhook for the skills lane's digest |
| `SKILLS_EXCLUDE_PROJECTS` | empty | comma-separated project ids every skills detector skips (session judge, struggle arcs, post-fire, procedure miner) and the config lane; there is no built-in default, so a deployment that wants a project left out lists it here |

Removed with the lane rework: `SKILLS_JUDGE_MODEL`, `SKILLS_JUDGE_BACKEND`, `SKILL_MEASURE_MODEL`, `SKILL_DERIVE_DRAFT_MODEL`, `SYNAPSE_SKILLS_DETECTORS`, `STRUGGLE_INCLUDE_PROJECTS`, `CONFIG_PROPOSE_SESSIONS` and the `NOTES_LANE_*` tuning variables. Use `SYNAPSE_SKILLS_JUDGE_MODEL`, or `SYNAPSE_LLM_PROVIDER=openai` for a non-Claude judge.

The client-side variables (`SYNAPSE_URL`, `SYNAPSE_INGEST_TOKEN`, `SYNAPSE_MACHINE_ROLE`, the board, nudge and sync switches) are listed in [docs/tools.md](docs/tools.md#plugin-configuration).

---

## 14. File Structure

Directory-level on purpose: per-file comments are kept to modules whose role is not obvious from the name, because a long annotated tree rots.

```
synapse/
├── ingestion/                 # everything that turns transcripts into memory (shared by the poller and the server)
│   ├── __main__.py            # poller bootstrap
│   ├── poller.py              # two threads: maintenance (chunks, embed, sweeps, web pages) and extraction (drains the queue)
│   ├── episode_writer.py      # the one place a parsed turn becomes a stored episode (guards, dedup, tombstones)
│   ├── jsonl_client.py        # transcript parsers → Episodes: Claude Code (jsonl_client), Codex, Cursor (SQLite),
│   │                          #   claude.ai and ChatGPT exports (codex_client, cursor_sqlite_client, claude_ai_client, chatgpt_export)
│   ├── backfill.py            # manual import CLI over those parsers (python -m ingestion.backfill --source ...)
│   ├── chunks.py              # sliding-window chunk construction (window=4 step=3)
│   ├── db.py + db_*.py        # PG CRUD facade: episodes, queue, notes, preferences, timeline, documents
│   ├── pg_pool.py             # the server's one connection pool; every route and recall leg borrows from it
│   ├── embedding.py           # Voyage / OpenAI-compatible embed + rerank clients (interactive vs batch retry policy)
│   ├── extractor.py           # ExtractionPipeline; extraction_*.py are its stages (entities, nodes, edges, twins, recheck)
│   ├── edge_dates.py          # EdgeDateExtractor + temporal prefilter
│   ├── kg_client.py           # Postgres KG client: nodes/edges, bitemporal writes, candidate-find
│   ├── llm_client.py          # LLM surface (create_llm_client, structured_call, agent_call, stage_model); llm_claude_cli.py,
│   │                          #   llm_openai.py are the two backends, llm_common.py the shared errors and retry policy
│   ├── timeline_gate.py       # per-turn gate → timeline events;  preferences_gate.py: → standing preferences
│   ├── capture/               # server-side web capture out of the records /ingest receives (rules in capture/rules/*.toml)
│   ├── web_artifacts.py       # web-research store (+ web_extractors, web_chunker, web_context, web_enqueue)
│   ├── surfaces.py            # credential-bound surface trust and the audience rules
│   ├── schema_check.py        # boot-time schema-version guard + the ColumnProbe the optional migrations use
│   ├── vector.py              # the one pgvector text literal (vec_literal, QueryVector)
│   └── settings.py            # env-then-.env resolution for every process
│
├── mcp_server/
│   ├── server.py              # fastmcp app: instructions, MultiAuth, tool and route registration
│   ├── retrieval_tools.py     # recall / recall_full_turns / fetch / fetch_session
│   ├── remember_tool.py       # remember;  feedback_tools.py: recall_feedback;  hidden_tools.py: issue_machine_token
│   ├── recall.py              # Recall engine, assembled from the recall_*.py mixins (episode_search, rerank, passages,
│   │                          #   sources, web, overview = recall(), episodes = recall_full_turns, fetch); settings in recall_config.py
│   ├── kg_pg.py               # recall-side KG search (vector + BM25 + 1-hop over kg_*)
│   ├── board.py               # the always-injected note board (GET /context)
│   ├── ingest_route.py        # POST /ingest and POST /capture/dry-run
│   ├── recall_route.py        # POST /recall
│   ├── http_helpers.py        # route(gate, body=...): gate once, JSON-object body, RouteError, logged 500
│   ├── auth_tokens.py         # root + per-device bearer verifier;  caller_trust.py / http_auth.py: the gates
│   ├── device_routes.py       # /device/code + /device/token (RFC 8628 login), allowlist-gated
│   ├── install_routes.py      # GET /install: install_script.py with this server's client inside
│   ├── skill_sync_routes.py   # machine-token-gated /skills/* routes (DSN-free sync);  skill_review_routes.py: proposal review
│   ├── config_sync_routes.py  # /config/* routes: the mirror of a machine's config files the dashboard lists and the config lane reads
│   ├── remember_routes.py     # /remember/spool: replays a write the MCP tool could not make, idempotent on the client's intent id
│   ├── skills_provider.py     # PgSkillsProvider: skill:// resources from skills_lane
│   └── dash/                  # operator dashboard API (/dash/api/*), serving the React bundle built from web/
│
├── dream/
│   ├── __main__.py            # nightly orchestrator: skills, config and notes lanes, then the retention prune
│   ├── lane.py                # shared window/cursor/session plumbing and the lanes' LLM calls
│   ├── skills/  config/  notes/   # the three lanes
│   └── evidence.py            # the evidence union the skills and config ledgers share
│
├── plugin/                    # the Claude Code plugin (hooks, commands) + the `synapse` CLI
│   └── scripts/               #   ingest_hook.py is the Stop hook; synapse_cli.py, synapse_login.py, harness.py, synapse_run.py ...
├── plugin-codex/              # the Codex hooks + installer (reuses plugin/scripts by relative path)
├── web/                       # the dashboard SPA (React, built by `npm run build` into web/dist, served by mcp_server/dash/static.py)
├── schema/                    # numbered SQL migrations, 001-076 (010 was removed); scripts/apply_schema.sh applies them in order
├── scripts/                   # apply_schema.sh and one-off operator tools (backfills, relabel, redaction, chunk-candidate A/B, board bench)
├── examples/env/              # per-backend .env presets (voyage, openrouter, local)
├── tests/
├── docs/                      # install, auth, tools, features, dashboard contract, specs
├── .claude-plugin/marketplace.json  # makes the repo a self-hosted plugin marketplace
├── compose.yml                # postgres / poller / mcp-server / dream (+ opt-in local-inference) services
├── HANDOFF.md                 # operator log: what changed and what to apply, newest first
├── README.md
└── ARCHITECTURE.md
```

---

## 15. Operational Notes

### Deploying

A merge to `main` publishes the image and Watchtower (label `watchtower.scope=synapse`) updates the app services; CI no longer deploys. CI tags every image `:<sha>` and moves `latest` only after every check has passed on `main`. Watchtower updates the image but not `compose.yml`: a change to the compose file (a published port, a healthcheck, a removed service) takes effect only when it is re-applied by hand on the server, with `docker compose up -d --remove-orphans` (the flag removes a service the file no longer names, such as `poller-drain`). When the postgres service's definition changed (`shm_size`, the `127.0.0.1` binding) this recreates Postgres, so run `docker compose stop -t 120 postgres` first (see the `--no-deps` gotcha below). The new poller healthcheck needs both heartbeat files, which only the new image writes, so recreate the poller onto the new image together with the compose change. Set `WATCHTOWER_TIMEOUT` where Watchtower runs to at least the pollers' `stop_grace_period` (5 minutes), or an auto-deploy kills in-flight items.

**Deploy gotcha — `--no-deps` (#62).** By hand, `docker compose up -d <app-svc>` pulls in the service's `depends_on` data services and **restarts Postgres**. An unclean PG restart corrupts the ParadeDB `episodes_search_idx` (assertion `left==right`), silently degrading episode BM25 to vector-only. **Always deploy app services with `--no-deps`** (the apps reach the database as `postgres:5432` on the compose network). If BM25 recall degrades after a deploy: `REINDEX INDEX CONCURRENTLY episodes_search_idx` (~9s). Data volumes survive.

**Migrations** are applied by hand after the image lands (`scripts/apply_schema.sh`, which applies only the files above the database's `schema_version` stamp). The optional ones build indexes with `CREATE`/`DROP INDEX CONCURRENTLY`, which cannot run inside a transaction: never use `psql -1` / `--single-transaction`. A failed `CONCURRENTLY` build leaves an INVALID index; drop it and re-run the file.

### CI

`.github/workflows/ci.yml`: ruff (lint + format), mypy (`uv run mypy`; the typed set is `pyproject.toml`'s `[tool.mypy] files`: `ingestion`, `dream`, the recall modules, `kg_pg`, `board`, the auth modules and `mcp_server/dash/`), bandit over `ingestion/ mcp_server/ dream/` (a SQL string with interpolated fragments carries a trailing `# nosec B608 — <why the fragment is fixed>` on a line inside the string's own lines), the web bundle (`npm test` then the build), pytest on the digest-pinned ParadeDB PG17 image (the non-DB suite in parallel, then the plain and pooled DB passes side by side on separate databases), then the image build. Dependabot has a `uv` ecosystem beside actions and docker, so `claude-agent-sdk` (and with it the bundled CLI) bumps arrive as PRs; the group opens one weekly PR, split it by package if that is too coarse.

### Extraction backfill (#64)

Re-extracting the ~10.7K existing chunks into the KG runs as a low-priority lane (`priority=10`) so it never starves fresh ingest. It is self-completing; a watcher tears down any temporary worker fleet on completion.

### Quota & stale claims

On `UsageLimitError` the extraction thread releases its in-flight claims back to `pending` and backs off exponentially (5 to 30 min, jittered); maintenance keeps running. `release_stale_claims(45min)` recovers orphans from crashed or killed workers without touching live peers (guarded by `claimed_at`); because a worker claims one row per thread, the threshold now bounds one item rather than a batch.

### Connection resilience

The pool (`ingestion/pg_pool`) checks a connection when it hands it out; half-open connections after a PG restart are replaced transparently, and `KGClient` retries a connection error once on a fresh connection. With the graph folded into Postgres (#67), there is one data store to watch, not two.

---

## 16. Known Gaps & Planned Work

### 16.1 Entity resolution / identity hubs (#49) — the load-bearing gap

The semantic-merge lane detects but never auto-applies lexically-different identity merges, so `User` / the user's first name / the user's full name / `The User` persist as separate nodes (~2065 edges, 6 fragments). External research names entity resolution *the* hard problem of KG memory; this is the highest-ROI fix. The intended shape (kept deliberately simple): mark unambiguous aliases, **physically merge** the canonical hub (not pointer-merge, which hurts traversal), keep a side audit log for reversibility, and resolve self-references to the canonical user. Avoid per-deployment config maps — others using Synapse shouldn't have to hand-curate alias lists.

### 16.2 Structured forgetting / decay

The field's most under-benchmarked lever. Synapse has the primitives — recency decay in fusion, bitemporal invalidation, and the reserved `t_expired` field — but no actual forgetting/priority-tier strategy. This is the most promising next research direction once entity resolution is settled.

### 16.3 Retire the summary residue

The code is gone (no poller leg, no extractor branch); what is left is the `synth_documents` table (~669 rows). Snapshot + drop.

### 16.4 FalkorDB → Postgres — DONE (#67)

Shipped. The graph was folded into Postgres (`kg_entities` / `kg_relationships`) and FalkorDB decommissioned. The cutover measured quality-equal on the relational golden set (delta −0.017, CI spans zero) and fixed FalkorDB's silently-dead BM25 leg (AND/phrase semantics returned 0 rows on real multi-word queries). One data store now, not two; the `SYNAPSE_KG_READ` rollback seam is gone, and a KG-leg failure degrades to an empty `facts` bucket rather than failing the recall.

### 16.5 Deferred destructive migrations

Nothing below can be dropped while an older image can still run (its INSERTs and SELECTs name them), so each waits for a release after the one that stops using it: the FalkorDB-era mirrors `kg_relationships.created_at` / `valid_at` / `invalid_at` and `kg_entities.created_at` / `valid_at` / `entity_type`; `kg_entities.degree`; `timeline_events.reported_count`; `recall_metrics.n_entities` / `n_timeline` / `ms_timeline` / `n_prefs` / `ms_prefs`; `skills_lane.skill_gap_candidates.proposal_path` / `summary_embedding` and `skill_registry.description_embedding` / `fire_count` / `last_fired`; the `synth_documents` table. The optional-migration probes for 056 / 057 / 060 / 061 stay until prod's schema version is confirmed.

### 16.6 Smaller items

- `r.name` predicate distribution sampling to confirm causal verbs (#51).
- Logfire span exporter recovery from sustained 429 (#45).
- Docker prune janitor (#65).
```
