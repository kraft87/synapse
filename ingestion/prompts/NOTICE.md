# NOTICE

Files in this directory contain prompt text adapted verbatim from Graphiti
(https://github.com/getzep/graphiti), licensed under Apache 2.0.

Original copyright (c) 2024 Zep Software, Inc.

Synapse modifications: variable name adaptations to match Synapse's
schema. Prompt logic, examples, and structure preserved verbatim.

License text: https://www.apache.org/licenses/LICENSE-2.0

## Upstream source files

The exact Graphiti source files that were ported (paths inside the
Graphiti repo at https://github.com/getzep/graphiti):

| Synapse file                                | Upstream Graphiti file                          |
|---------------------------------------------|--------------------------------------------------|
| `ingestion/prompts/dedupe_nodes.py`         | `graphiti_core/prompts/dedupe_nodes.py`          |
| `ingestion/prompts/dedupe_edges.py`         | `graphiti_core/prompts/dedupe_edges.py`          |
| `ingestion/prompts/invalidate_edges.py`     | `graphiti_core/prompts/dedupe_edges.py` (`resolve_edge` — Graphiti folds invalidation and dedup into one prompt with two output lists; Synapse splits them so the writer-side detector can call invalidation alone) |
| `ingestion/prompts/extract_edge_dates.py`   | `graphiti_core/prompts/extract_edges.py` (`extract_timestamps` — Graphiti's name for the single-fact temporal-bounds extractor) |
| `ingestion/prompts/models.py`               | `graphiti_core/prompts/models.py` (`Message` type) and the per-prompt `BaseModel` classes from each upstream file |
