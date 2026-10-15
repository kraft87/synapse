"""Contextual Retrieval for web_chunks: prepend an LLM-generated context blurb to each chunk
before it is embedded.

Anthropic's recipe (https://www.anthropic.com/news/contextual-retrieval):
  - For each chunk, generate 50-100 tokens of context that situate it in
    the parent doc
  - Embed `context + content` instead of `content` alone
  - Reported: 49% recall failure reduction alone, 67% with rerank

All chunks of one page share its text, so they go to the LLM together (in batches of
:data:`_MAX_CHUNKS_PER_CALL`) to pay for the page's tokens once. The poller's maintenance
thread calls :func:`contextualize_pending` every cycle, on a cap, for the pages whose
chunks have no prefix yet (``web_chunks.context_prefix IS NULL``); the embed step vectors
a chunk only once it has one. A single-chunk page needs none (prefix ''), and so does a
batch the model keeps answering badly (a warning, then '': the chunks are embedded bare
rather than retried at every cycle's cost).

The LLM is whatever SYNAPSE_LLM_PROVIDER selects (create_llm_client), on the CONTEXTUALIZE
stage model (SYNAPSE_CONTEXTUALIZE_MODEL, else SYNAPSE_LLM_MODEL, else Haiku). A cycle stops
at the first LLM outage or quota exhaustion rather than burning retries on every remaining
page. To redo every prefix (a changed prompt), set ``context_prefix`` to NULL on the rows:
the poller refills them a few pages per cycle.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ingestion.db import Database
from ingestion.llm_client import LLM_TRANSPORT_ERRORS, MalformedResponseError, parse_with_retry

logger = logging.getLogger(__name__)

#: Pages one cycle contextualizes at most, and the time after which it starts no more.
ARTIFACTS_PER_CYCLE = 20
CYCLE_BUDGET_S = 180.0


class _ContextItem(BaseModel):
    idx: int
    context: str = Field(min_length=1)


class _ContextsResponse(BaseModel):
    contexts: list[_ContextItem]


# Response contract. It travels in-prompt (the CLI enforces nothing), so the
# reply is validated against _ContextsResponse in parse_contexts.
_CONTEXTS_SCHEMA = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "contexts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer"},
                        "context": {"type": "string"},
                    },
                    "required": ["idx", "context"],
                },
            }
        },
        "required": ["contexts"],
    },
}

# Chunks per LLM call. Above this we split the page into batches so the
# prompt doesn't blow Haiku's context window on giant scraper-blob pages
# (Amazon search results, GitHub README dumps, etc.). 30 chunks * 1500 chars
# = ~45k chars chunks_section + the doc cap of 30k = ~75k char prompt,
# well within Haiku's window.
_MAX_CHUNKS_PER_CALL = 30

# Minimum coverage required to accept a batch (rest of missing chunks padded
# with empty context_prefix). Below this we retry. Empirically Haiku drops
# 0-1 entries on healthy responses; 30%+ missing means a degenerate response
# (e.g. it only described the first few chunks then gave up).
_MIN_COVERAGE = 0.7

PROMPT = """Situate chunks of a web page in the page, for retrieval.

<document>
{document}
</document>

Below are {n_chunks} chunks taken from the document. For each chunk, write a brief context (50-100 tokens) describing where this chunk fits in the document, what specific topic/section it covers, and any disambiguating information (e.g. "this is the pricing section for voyage-4-large" rather than "this discusses pricing").

The context will be prepended to the chunk for embedding-based retrieval. The goal: when someone searches "voyage embeddings pricing", the contextualized chunk should match more reliably than the bare chunk.

<chunks>
{chunks_section}
</chunks>"""


def parse_contexts(text: str, expected_n: int) -> tuple[list[str] | None, str | None]:
    """Parse + validate via Pydantic. Returns (list of expected_n strings, error_reason).

    On success the second element is None. On failure the first is None and the
    second is a short human-readable reason suitable for feeding back to the LLM
    on retry (e.g. "JSONDecodeError at line 3", "missing field 'context'",
    "schema-echo shape detected").

    Rejects partial responses below _MIN_COVERAGE so the retry loop can re-prompt.
    Up to one missing entry at the tail is padded (LLM occasionally drops the
    last item) — that's the only "good enough" slack we extend.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"response was not valid JSON: {e}"
    if not isinstance(obj, dict):
        return None, f"response was a {type(obj).__name__}, expected object"

    # Schema-echo failure mode: Haiku returned the schema definition itself
    # with data nested under properties.contexts.items.
    if "contexts" not in obj and "properties" in obj:
        return None, (
            "you returned the JSON schema shape ({type, properties, ...}) instead of "
            "data matching it. The response should look like "
            '{"contexts": [{"idx": 0, "context": "..."}, ...]}'
        )

    try:
        parsed = _ContextsResponse.model_validate(obj)
    except ValidationError as e:
        return None, f"response did not validate: {e.errors(include_url=False)[:3]}"

    indexed = {item.idx: item.context.strip() for item in parsed.contexts if item.context.strip()}
    if not indexed:
        return None, "contexts array was empty"

    missing = [i for i in range(expected_n) if i not in indexed]
    if len(missing) > max(1, int(expected_n * (1 - _MIN_COVERAGE))):
        sample = missing[:10]
        return None, (
            f"you only returned contexts for {len(indexed)}/{expected_n} chunks. "
            f"Missing indices: {sample}{'...' if len(missing) > 10 else ''}. "
            f"Every chunk needs a context — return exactly {expected_n} items."
        )
    return [indexed.get(i, "") for i in range(expected_n)], None


def _contexts_parser(expected_n: int) -> Callable[[str], list[str]]:
    """parse_contexts as a parse_with_retry parser: a failure raises with the reason,
    which parse_with_retry quotes back to the model on the next attempt."""

    def parse(text: str) -> list[str]:
        contexts, reason = parse_contexts(text, expected_n)
        if contexts is None:
            raise MalformedResponseError(reason or "unparseable response", raw_response=text)
        return contexts

    return parse


def contextualize_artifact(
    db: Database, llm: Any, artifact_id: int, content: str, model: str
) -> int:
    """Generate the context prefix of every chunk of one page; returns the chunks stored.

    A batch the model keeps answering badly after the retries stores '' for its chunks.
    LLM_TRANSPORT_ERRORS (outage, quota) propagate, storing nothing for the page."""
    chunks = db.get_web_chunks(artifact_id)
    if not chunks:
        return 0
    if len(chunks) == 1:
        # Single chunk == whole doc. Context is trivial; skip the LLM call.
        db.set_web_chunk_contexts([(chunks[0]["id"], "")])
        return 1

    # Cap document at ~30k chars; chunker overlap means truncated doc still
    # contextualizes every chunk usefully.
    doc = content[:30_000]
    # Split into _MAX_CHUNKS_PER_CALL batches so oversized aggregator pages
    # (Amazon search, github trees with 200+ chunks) don't blow the context
    # window. Each batch gets its own LLM call.
    contexts: list[str] = []
    for batch_start in range(0, len(chunks), _MAX_CHUNKS_PER_CALL):
        batch = chunks[batch_start : batch_start + _MAX_CHUNKS_PER_CALL]
        # Batch-local indices (0..len(batch)-1) in the prompt keep the LLM's idx field in
        # sync with parse_contexts' coverage check.
        chunks_section = "\n\n".join(f"[{i}]\n{c['content']}" for i, c in enumerate(batch))
        prompt = PROMPT.format(document=doc, n_chunks=len(batch), chunks_section=chunks_section)

        # Replies occasionally come back malformed (schema echo, truncated JSON,
        # dropped chunks): up to three attempts, each failure's reason fed back.
        try:
            contexts += parse_with_retry(
                llm,
                base_prompt=prompt,
                parser=_contexts_parser(len(batch)),
                model=model,
                max_tokens=min(8192, 250 * len(batch) + 500),
                response_format=_CONTEXTS_SCHEMA,
                max_attempts=3,
            )
        except MalformedResponseError as e:
            logger.warning(
                "web artifact %d: no usable context for chunks %d-%d after retries (%s); "
                "storing them without one. last reply: %.200s",
                artifact_id,
                batch_start,
                batch_start + len(batch) - 1,
                e,
                e.raw_response,
            )
            contexts += [""] * len(batch)

    db.set_web_chunk_contexts([(c["id"], ctx) for ctx, c in zip(contexts, chunks, strict=True)])
    return len(chunks)


def contextualize_pending(
    db: Database,
    llm: Any,
    model: str,
    *,
    max_artifacts: int = ARTIFACTS_PER_CYCLE,
    budget_s: float = CYCLE_BUDGET_S,
) -> int:
    """Contextualize up to ``max_artifacts`` pages with chunks still lacking a prefix,
    starting no page after ``budget_s``. Returns the chunks stored. An LLM outage or an
    exhausted quota ends the call; the pages left wait for the next cycle."""
    deadline = time.monotonic() + budget_s
    total = 0
    for page in db.artifacts_to_contextualize(max_artifacts):
        if time.monotonic() >= deadline:
            break
        try:
            total += contextualize_artifact(db, llm, page["id"], page["content_markdown"], model)
        except LLM_TRANSPORT_ERRORS as e:
            logger.warning("Web chunk contexts paused, LLM unavailable: %s", e)
            break
    if total:
        logger.info("Contextualized %d web chunk(s)", total)
    return total
