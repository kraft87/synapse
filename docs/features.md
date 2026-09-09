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
- **Web-research capture** — WebFetch/Exa/Firecrawl/search results are captured and embedded
  so past research is recallable.
- **MCP server** — FastMCP over streamable HTTP. Tool list: [docs/tools.md](tools.md).
- **Claude Code plugin** — ingest + recall wiring, plus a nightly dream→skills lane that
  mines your transcripts to maintain a self-improving skill library, with opt-in two-way
  skill sync (`SYNAPSE_SKILLS_SYNC=1`). See [plugin/README.md](../plugin/README.md).
- **Codex CLI client** brings the same per-session surface to OpenAI Codex CLI. See
  [plugin-codex/README.md](../plugin-codex/README.md).

## Stack

- **Storage:** PostgreSQL (ParadeDB image) for everything — episodes, chunks, the queue, the
  web store, and the knowledge graph.
- **Vector search:** pgvector `halfvec` HNSW (2048 dims by default; width is fixed at
  first-boot schema provisioning via `SYNAPSE_EMBED_DIMS`).
- **Full-text search:** ParadeDB `pg_search` (BM25), fused with vector via reciprocal-rank
  fusion.
- **Embeddings + rerank:** Voyage AI by default (`voyage-4-large`, 2048 dims, and
  `rerank-2.5-lite`). Pluggable: any OpenAI-compatible `/embeddings` endpoint plus any
  TEI/Infinity/Cohere-shape `/rerank` server, or the bundled `local-inference` compose
  profile (see `.env.example`). Published retrieval quality was measured on the Voyage
  stack, and the rerank leg matters — `SYNAPSE_RERANK_PROVIDER=none` degrades recall to
  fusion-only ordering.
- **Extraction LLM:** Claude Haiku 4.5 by default, via `claude-agent-sdk` with a Claude
  subscription token or an `ANTHROPIC_API_KEY`. Set `SYNAPSE_LLM_PROVIDER=openai` to point
  at any OpenAI-compatible `/chat/completions` endpoint (OpenRouter, a local model, etc.).
  Model ids are provider-specific, and mixing the two spellings is the one config mistake
  that fails quietly: see [docs/install.md](install.md#model-ids-one-thing-that-bites).
- **MCP:** FastMCP, streamable HTTP, port 8765.
- **Language:** Python 3.12, managed with `uv`.

> An earlier version stored the graph in FalkorDB. The graph now lives in Postgres
> (`ingestion/kg_client.py`); FalkorDB has been decommissioned.
