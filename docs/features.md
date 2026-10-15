# What's in the box

The parts of Synapse, and the stack they run on. How they fit together is
[ARCHITECTURE.md](../ARCHITECTURE.md); how to install them is [docs/install.md](install.md).

## Components

- **Episodic memory** — one episode per human turn, served by a deep fetch (100 candidates
  per leg) plus a Voyage cross-encoder rerank. The retrieval workhorse for broad and needle
  queries alike.
- **Knowledge graph** — entities and bitemporal `RELATES_TO` fact edges in Postgres; the
  relational/multi-hop specialist. Facts are never deleted: contradictions invalidate the
  old edge and write a new one, which powers a "what changed" history leg.
- **Timeline** — an append-only log of dated point-events ("shipped X", "decided Y") mined
  from turns by an LLM gate and fed by git commits, serving "when / in what order" questions.
  A re-told happening confirm-merges into its existing row (an LLM reads both source turns,
  both presentation orders must agree) instead of duplicating; each event carries a
  `personal`/`technical` domain label so personal-scope queries exclude work noise; and
  happenings narrated inside quoted third-party material (someone else's email, a pasted
  transcript, an article) are never logged as the user's own.
- **Notes** — `remember()` writes curated one-line memories (a hook plus a body) that the
  session-start board shows and `recall()` searches. A restated note updates in place, a
  correction supersedes, and a nightly lane merges what is left.
- **Web-research capture** — off by default (`SYNAPSE_WEB_CAPTURE=on`, after a
  `synapse capture --dry-run`). Public pages and search results your agents fetched, from
  any web tool, are captured server-side out of the transcripts the Stop hook already ships,
  then chunked and embedded so past research is recallable.
- **MCP server** — FastMCP over streamable HTTP. Tool list: [docs/tools.md](tools.md).
- **Claude Code plugin** — ingest + recall wiring, with opt-in two-way skill sync
  (`synapse login` asks). On the server a nightly `dream` container runs three lanes (all on
  by default, each with a kill switch): skills (mines your transcripts to propose and retune
  skills), config (proposes behavioral rules you corrected the agent on) and notes (merges
  and re-files duplicate notes). Skills and config are propose-only, reviewed with
  `/synapse:skill-review` and `/synapse:config-review`. See
  [plugin/README.md](../plugin/README.md).
- **Codex CLI client** brings the same per-session surface to OpenAI Codex CLI. See
  [plugin-codex/README.md](../plugin-codex/README.md).
- **One setup command per machine** — the server serves its own client
  (`curl <server>/install | python3 -`), and `synapse login` enrolls the machine, keeps its
  settings in `~/.synapse/config.json`, and wires Claude Code and Codex.

## Stack

- **Storage:** PostgreSQL (ParadeDB image) for everything — episodes, chunks, the queue, the
  web store, and the knowledge graph.
- **Vector search:** pgvector `halfvec` HNSW (2048 dims by default; width is fixed at
  first-boot schema provisioning via `SYNAPSE_EMBED_DIMS`).
- **Full-text search:** ParadeDB `pg_search` (BM25), fused with vector via reciprocal-rank
  fusion.
- **Embeddings + rerank:** Voyage AI by default (`voyage-4-large`, 2048 dims, and
  `rerank-3-lite`). Pluggable: any OpenAI-compatible `/embeddings` endpoint plus any
  TEI/Infinity/Cohere-shape `/rerank` server, or the bundled `local-inference` compose
  profile (see `.env.example`). Published retrieval quality was measured on the Voyage
  stack, and the rerank leg matters — `SYNAPSE_RERANK_PROVIDER=none` (or a failed rerank)
  degrades recall to fusion-only ordering, with passages picked by keyword overlap.
- **Extraction LLM:** Claude Haiku 4.5 by default, via `claude-agent-sdk` with a Claude
  subscription token or an `ANTHROPIC_API_KEY`. Set `SYNAPSE_LLM_PROVIDER=openai` to point
  at any OpenAI-compatible `/chat/completions` endpoint (OpenRouter, a local model, etc.).
  Model ids are provider-specific, and mixing the two spellings is the one config mistake
  that fails quietly: see [docs/install.md](install.md#model-ids-one-thing-that-bites).
- **MCP:** FastMCP, streamable HTTP, port 8765.
- **Language:** Python 3.12, managed with `uv`.

> An earlier version stored the graph in FalkorDB. The graph now lives in Postgres
> (`ingestion/kg_client.py`); FalkorDB has been decommissioned.
