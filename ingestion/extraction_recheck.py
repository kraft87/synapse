"""Stage 6c: a fact restated by an edge written after Stage 6a pooled is still a duplicate.

Stage 6a pools a fact's dedup candidates from the graph as it stands when the batch gets
there, and the 6b judge rules only on those pools. Overlapping chunk windows share a turn
(window 4, step 3), so two batches extract the same claim from it. When a whole session is
enqueued at once (backfill, import, re-extraction), the poller claims its adjacent windows
together and runs them on parallel worker threads: both pool and judge before either
writes, and the claim lands twice. Prod on 2026-10-10: 128 of 393 merged duplicate live
facts came from different batches, 112 of them created under 10 minutes apart and 105 of
those sharing a source episode.

The recheck calls no LLM and runs as late as it can: after 6b and its saturation rounds,
inside Stage 7 once that stage's own LLM pre-pass (edge dates, writer-side contradictions)
is done, right before anything is written. Each fact Stage 7 would create is looked up
against its nearest live edges in the group. An edge whose text is the same after case and
whitespace folding, or that is a twin under the same-batch rule (ingestion.extraction_twins:
identical identifier signature, content words differing on one side only, cosine >=
_TWIN_MIN_SIM), makes the fact a duplicate of that edge. The fact joins ``skip_indices`` and
the edge goes into ``reinforce``, so Stage 7 treats it exactly like a judge-detected
duplicate: the edge's provenance gains the batch's episodes and its mention_count counts the
restatement, and an old edge the fact also retires is linked to the edge it was absorbed
into (``_superseder``).

Like the same-batch rule, endpoints are not compared, and the existing edge survives even
when the new copy is the more detailed one: its text is already in the graph and the
restatement only adds provenance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ingestion.extraction_twins import _TWIN_MIN_SIM, fact_tokens, lexical_twins
from ingestion.models import ExtractedFact

#: Live edges fetched per fact, nearest first. More than one so an edge this batch is
#: about to retire cannot hide the restated one behind it.
_RECHECK_NEIGHBOURS = 5

#: Cosine-distance bound of the lookup: the twin rule's cosine floor, applied in SQL.
_RECHECK_MAX_DIST = 1.0 - _TWIN_MIN_SIM


def _folded(text: str) -> str:
    return " ".join(text.split()).casefold()


def restated_edge(fact: str, neighbours: list[dict[str, Any]]) -> str | None:
    """uuid of the first neighbour (nearest first) whose text restates ``fact``, else None.

    The caller has already bounded the neighbours to cosine >= _TWIN_MIN_SIM."""
    folded = _folded(fact)
    tokens = fact_tokens(fact)
    for edge in neighbours:
        text = edge.get("fact") or ""
        if _folded(text) == folded or lexical_twins(tokens, fact_tokens(text)):
            return str(edge["uuid"])
    return None


def recheck_restated(
    facts: list[ExtractedFact],
    fact_embeddings: list[list[float]],
    uuid_map: dict[str, str],
    group_id: str,
    kg: Any,
    skip_indices: set[int],
    invalidate: dict[int, list[str]],
    reinforce: dict[int, list[str]],
    also_retiring: Iterable[str] = (),
) -> int:
    """Stage 6c. Moves restated facts into ``skip_indices`` / ``reinforce`` in place.

    Only facts Stage 7 would create are looked up (not skipped, both endpoints resolved),
    in one query for the group. An edge any fact of this batch retires (a Stage 6 verdict in
    ``invalidate``, or a writer-side one in ``also_retiring``) is never a match,
    since absorbing a fact into it would leave the claim with no live edge; a fact judged
    to contradict its own near-copy is a drop-in replacement and is written as one.
    Returns the number of facts moved."""
    todo = [
        i
        for i, f in enumerate(facts)
        if i not in skip_indices and uuid_map.get(f.source) and uuid_map.get(f.target)
    ]
    if not todo:
        return 0
    retiring = {u for uuids in invalidate.values() for u in uuids} | set(also_retiring)
    hits = kg.nearest_live_edges(
        [fact_embeddings[i] for i in todo],
        group_id,
        max_distance=_RECHECK_MAX_DIST,
        limit=_RECHECK_NEIGHBOURS,
    )
    moved = 0
    for i, neighbours in zip(todo, hits, strict=True):
        dup = restated_edge(facts[i].fact, [n for n in neighbours if n["uuid"] not in retiring])
        if dup is None:
            continue
        skip_indices.add(i)
        reinforce[i] = [dup]
        moved += 1
    return moved
