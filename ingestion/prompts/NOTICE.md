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
| `ingestion/prompts/extract_edge_dates.py`   | `graphiti_core/prompts/extract_edges.py` (`extract_timestamps` — Graphiti's name for the temporal-bounds extractor) |
| `ingestion/prompts/models.py`               | `graphiti_core/prompts/models.py` (`Message` type) |

Additionally, parts of the extraction rule set in `ingestion/extractor.py`
(entity-name/skip-class discipline, direct-edge preference, output-discipline
rules) adapt guidance from `graphiti_core/prompts/extract_nodes_and_edges.py`,
rewritten for Synapse's summary-based extraction input.
