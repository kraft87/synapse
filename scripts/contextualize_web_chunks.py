#!/usr/bin/env python3
"""
Contextual Retrieval for web_chunks: prepend an LLM-generated context blurb
to each chunk before re-embedding.

Anthropic's recipe (https://www.anthropic.com/news/contextual-retrieval):
  - For each chunk, generate 50-100 tokens of context that situate it in
    the parent doc
  - Embed (and BM25-index) `context + content` instead of `content` alone
  - Reported: 49% recall failure reduction alone, 67% with rerank

This implementation batches all chunks of one doc into a single Haiku call
to amortize the document tokens. ~$3 total for the full 311-doc corpus
(311 calls, doc tokens not paid 4000x3898 times).

After contextualizing, chunks must be re-embedded (is_embedded=false flips
to true after embedding completes). Run scripts/backfill_web_chunks.py
--embed-only afterwards.

Usage:
    uv run python scripts/contextualize_web_chunks.py [--limit N] [--reset]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.llm_client import ClaudeCLIClient


class _ContextItem(BaseModel):
    idx: int
    context: str = Field(min_length=1)


class _ContextsResponse(BaseModel):
    contexts: list[_ContextItem]


# JSON schema enforced by the Agent SDK's structured-output mode. No more
# fragile regex parsing of free-form text — the SDK guarantees the response
# matches this shape (or fails the call).
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

# Chunks per LLM call. Above this we split the artifact into batches so the
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

PROMPT = """<document>
{document}
</document>

Below are {n_chunks} chunks taken from the document. For each chunk, write a brief context (50-100 tokens) describing where this chunk fits in the document, what specific topic/section it covers, and any disambiguating information (e.g. "this is the pricing section for voyage-4-large" rather than "this discusses pricing").

The context will be prepended to the chunk for embedding-based retrieval. The goal: when someone searches "voyage embeddings pricing", the contextualized chunk should match more reliably than the bare chunk.

<chunks>
{chunks_section}
</chunks>"""


def load_env():
    db_url = os.environ.get("SYNAPSE_DB_URL")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists() and not db_url:
        for line in env_path.read_text().splitlines():
            if line.startswith("SYNAPSE_DB_URL="):
                db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not db_url:
        print("error: SYNAPSE_DB_URL not set", file=sys.stderr)
        sys.exit(2)
    return db_url


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


def contextualize_artifact(
    conn: psycopg.Connection, llm: ClaudeCLIClient, artifact_id: int, content: str
) -> tuple[int, str | None]:
    """Generate context_prefix for every chunk of one artifact.

    Returns (chunks_updated, error_or_None).
    """
    chunks = conn.execute(
        "SELECT id, idx, content FROM web_chunks WHERE web_artifact_id = %s ORDER BY idx",
        (artifact_id,),
    ).fetchall()
    if not chunks:
        return 0, None
    if len(chunks) == 1:
        # Single chunk == whole doc. Context is trivial; skip the LLM call.
        conn.execute(
            "UPDATE web_chunks SET context_prefix = %s, is_embedded = false WHERE id = %s",
            ("", chunks[0][0]),
        )
        return 1, None

    # Cap document at ~30k chars; chunker overlap means truncated doc still
    # contextualizes every chunk usefully.
    doc = content[:30_000]
    # Split into _MAX_CHUNKS_PER_CALL batches so oversized aggregator pages
    # (Amazon search, github trees with 200+ chunks) don't blow the context
    # window. Each batch gets its own LLM call.
    contexts: list[str] = []
    for batch_start in range(0, len(chunks), _MAX_CHUNKS_PER_CALL):
        batch = chunks[batch_start : batch_start + _MAX_CHUNKS_PER_CALL]
        # Use batch-local indices (0..len(batch)-1) in the prompt so the LLM's
        # idx field stays in sync with parse_contexts' coverage check. The
        # absolute chunk idx (c[1]) is only used downstream to map context back
        # to the DB row via the same `batch` ordering.
        chunks_section = "\n\n".join(f"[{i}]\n{c[2]}" for i, c in enumerate(batch))
        prompt = PROMPT.format(document=doc, n_chunks=len(batch), chunks_section=chunks_section)

        # Structured-output occasionally returns malformed shapes (schema echo,
        # truncated JSON). Retry up to twice with the prior failed response +
        # validation reason fed back, so the LLM can self-correct.
        batch_contexts: list[str] | None = None
        last_text = ""
        last_err: str | None = None
        current_prompt = prompt
        for _ in range(3):
            try:
                resp = llm.messages.create(
                    messages=[{"role": "user", "content": current_prompt}],
                    max_tokens=min(8192, 250 * len(batch) + 500),
                    response_format=_CONTEXTS_SCHEMA,
                )
                last_text = resp.content[0].text if resp.content else ""
            except Exception as e:
                last_err = f"llm: {e}"
                continue
            batch_contexts, reason = parse_contexts(last_text, len(batch))
            if batch_contexts is not None:
                break
            last_err = reason
            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous response failed validation: {reason}\n\n"
                f"Previous response (truncated):\n{last_text[:1500]}\n\n"
                f"Return ONLY a JSON object matching the schema — no schema echo, "
                f'no markdown. Shape: {{"contexts": [{{"idx": <int>, "context": <string>}}, ...]}}'
            )

        if batch_contexts is None:
            return 0, f"batch {batch_start} after retries: {last_err}; last_text={last_text[:200]}"
        contexts.extend(batch_contexts)

    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE web_chunks SET context_prefix = %s, is_embedded = false WHERE id = %s",
            [(ctx, c[0]) for ctx, c in zip(contexts, chunks, strict=True)],
        )
    return len(chunks), None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process at most N artifacts.")
    parser.add_argument(
        "--reset", action="store_true", help="NULL out all existing context_prefix values first."
    )
    parser.add_argument(
        "--artifact-ids", nargs="+", type=int, help="Limit to specific artifact IDs."
    )
    args = parser.parse_args()

    db_url = load_env()
    llm = ClaudeCLIClient(model="claude-haiku-4-5")

    with psycopg.connect(db_url, autocommit=False) as conn:
        if args.reset:
            n = conn.execute(
                "UPDATE web_chunks SET context_prefix = NULL, is_embedded = false "
                "WHERE context_prefix IS NOT NULL"
            ).rowcount
            conn.commit()
            print(f"[+] reset: cleared {n} context_prefix rows", file=sys.stderr)

        if args.artifact_ids:
            rows = conn.execute(
                "SELECT id, content_markdown FROM web_artifacts WHERE id = ANY(%s)",
                (args.artifact_ids,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT a.id, a.content_markdown
                FROM web_artifacts a
                JOIN web_chunks c ON c.web_artifact_id = a.id
                WHERE a.kind IN ('web_scrape', 'research_brief')
                  AND a.content_markdown IS NOT NULL
                  AND c.context_prefix IS NULL
                ORDER BY a.id
                """
            ).fetchall()
        if args.limit:
            rows = rows[: args.limit]

        print(f"[+] processing {len(rows)} artifacts", file=sys.stderr)
        t0 = time.time()
        ok = 0
        errors = 0
        total_chunks = 0
        for i, (aid, content) in enumerate(rows, 1):
            n, err = contextualize_artifact(conn, llm, aid, content)
            if err:
                errors += 1
                print(f"  [{i}/{len(rows)}] artifact={aid} ERROR: {err}", file=sys.stderr)
            else:
                ok += 1
                total_chunks += n
                if i % 5 == 0 or i == len(rows):
                    elapsed = time.time() - t0
                    print(
                        f"  [{i}/{len(rows)}] ok={ok} err={errors} chunks={total_chunks} ({elapsed:.1f}s)",
                        file=sys.stderr,
                    )
            conn.commit()

        print(
            f"\nContextualized {ok} artifacts ({total_chunks} chunks) in {time.time() - t0:.1f}s. "
            f"{errors} errors."
        )
        print("Next step: rerun `scripts/backfill_web_chunks.py --embed-only` to re-embed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
