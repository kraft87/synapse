#!/usr/bin/env python3
"""
Chunk + embed web_artifacts into web_chunks, the table recall's web bucket reads.

Two phases (both idempotent):
  1. Chunk: for any web_artifact of the selected kinds without chunks, run
     chunk_markdown() and INSERT new web_chunks rows (is_embedded=false).
  2. Embed: for any web_chunks with is_embedded=false, batch through the
     configured embedder and UPDATE the embedding column.

Kinds default to research_brief only. Generic web_scrape capture stopped in
July 2026 and its chunks scored 10 helpful / 125 noise in recall feedback, so
it is not rebuilt; pass --kinds web_scrape to chunk one explicitly.

The embedder comes from ingestion.embedding.create_embedder, the factory recall
builds its query embedder with, and is validated against synapse_meta, so chunk
vectors always match the model and width the web leg queries with.

Re-runs cheaply pick up where they left off via the UNIQUE (web_artifact_id, idx)
and the is_embedded flag.

Usage:
    uv run python scripts/backfill_web_chunks.py [--dry-run] [--chunk-only | --embed-only]
        [--kinds research_brief ...]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.embedding import create_embedder, embed_provider
from ingestion.web_chunker import chunk_markdown

DEFAULT_KINDS = ("research_brief",)


def load_env() -> str:
    """SYNAPSE_DB_URL from the environment, else the repo-root .env. VOYAGE_API_KEY is
    read the same way into the environment, where create_embedder looks for it."""
    db_url = os.environ.get("SYNAPSE_DB_URL")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists() and (not db_url or not os.environ.get("VOYAGE_API_KEY")):
        for line in env_path.read_text().splitlines():
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k == "SYNAPSE_DB_URL" and not db_url:
                db_url = v
            elif k == "VOYAGE_API_KEY" and not os.environ.get("VOYAGE_API_KEY"):
                os.environ["VOYAGE_API_KEY"] = v
    if not db_url:
        print("error: SYNAPSE_DB_URL not set", file=sys.stderr)
        sys.exit(2)
    return db_url


def connect(db_url: str, *, read_only: bool = False) -> psycopg.Connection[Any]:
    """Connection for one run. read_only (the --dry-run path) makes the server refuse
    any write, so a dry run against production can't change it."""
    conn = psycopg.connect(db_url, autocommit=False)
    conn.read_only = read_only
    return conn


def _pending(conn: psycopg.Connection[Any], kinds: Sequence[str]) -> list[Any]:
    return conn.execute(
        """
        SELECT a.id, a.content_markdown, a.fetched_at, a.kind, a.tool_use_id
        FROM web_artifacts a
        WHERE a.kind = ANY(%s)
          AND a.content_markdown IS NOT NULL
          AND LENGTH(a.content_markdown) > 0
          AND NOT EXISTS (SELECT 1 FROM web_chunks c WHERE c.web_artifact_id = a.id)
        ORDER BY a.id
        """,
        (list(kinds),),
    ).fetchall()


def dry_run(conn: psycopg.Connection[Any], kinds: Sequence[str] = DEFAULT_KINDS) -> tuple[int, int]:
    """List the artifacts phase_chunk would chunk and how many chunks each would get.
    Writes nothing. Returns (artifacts, chunks)."""
    artifacts = chunks = 0
    for artifact_id, content, fetched_at, kind, tool_use_id in _pending(conn, kinds):
        n = len(chunk_markdown(content))
        if not n:
            continue
        artifacts += 1
        chunks += n
        print(f"  {artifact_id}  {str(fetched_at)[:10]}  {kind}  {tool_use_id}  {n} chunks")
    pending_embed = conn.execute(
        "SELECT COUNT(*) FROM web_chunks WHERE is_embedded = false"
    ).fetchone()
    print(
        f"dry-run: would chunk {artifacts} artifacts into {chunks} chunks "
        f"(kinds={','.join(kinds)}); {pending_embed[0] if pending_embed else 0} existing "
        "chunks awaiting embedding"
    )
    return artifacts, chunks


def phase_chunk(
    conn: psycopg.Connection[Any], kinds: Sequence[str] = DEFAULT_KINDS
) -> tuple[int, int]:
    """Chunk artifacts that don't yet have any chunks. Returns (artifacts, chunks_written)."""
    rows = _pending(conn, kinds)

    artifacts_done = 0
    chunks_written = 0
    for artifact_id, content, fetched_at, _kind, _tool_use_id in rows:
        chunks = chunk_markdown(content)
        if not chunks:
            continue
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO web_chunks
                    (web_artifact_id, idx, content, char_start, char_end, content_ts)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (web_artifact_id, idx) DO NOTHING
                """,
                [
                    (artifact_id, c.idx, c.content, c.char_start, c.char_end, fetched_at)
                    for c in chunks
                ],
            )
        artifacts_done += 1
        chunks_written += len(chunks)
        if artifacts_done % 25 == 0:
            conn.commit()
            print(
                f"  chunked {artifacts_done}/{len(rows)} artifacts ({chunks_written} chunks)",
                file=sys.stderr,
            )
    conn.commit()
    return artifacts_done, chunks_written


def phase_embed(conn: psycopg.Connection[Any], embedder: Any, batch_size: int = 128) -> int:
    """Embed pending web_chunks (is_embedded=false). Returns count embedded."""
    total = 0
    t0 = time.time()
    while True:
        rows = conn.execute(
            """
            SELECT id, content, COALESCE(context_prefix, '') AS context_prefix
            FROM web_chunks WHERE is_embedded = false
            ORDER BY id LIMIT %s
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        # Prepend Contextual Retrieval prefix if present so the embedding
        # captures "where this chunk fits in the doc" alongside the chunk.
        texts = [(r[2] + "\n\n" + r[1]) if r[2] else r[1] for r in rows]
        embeddings = embedder.embed(texts, task="document")
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE web_chunks SET embedding = %s::vector, is_embedded = true WHERE id = %s",
                [(emb, cid) for cid, emb in zip(ids, embeddings, strict=True)],
            )
        conn.commit()
        total += len(rows)
        rate = total / max(time.time() - t0, 0.1)
        print(
            f"  embedded {total} chunks ({rate:.0f}/s)",
            file=sys.stderr,
        )
    return total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill web_chunks: chunk + embed.")
    parser.add_argument("--chunk-only", action="store_true", help="Run only the chunking phase.")
    parser.add_argument("--embed-only", action="store_true", help="Run only the embedding phase.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the artifacts that would be chunked and their chunk counts, on a "
        "read-only connection. Writes nothing and needs no embedding key.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=list(DEFAULT_KINDS),
        help="web_artifacts.kind values to chunk (default: research_brief).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DELETE the web_chunks of the selected kinds and re-chunk + re-embed them.",
    )
    parser.add_argument("--batch", type=int, default=128, help="Embedding batch size.")
    args = parser.parse_args(argv)

    db_url = load_env()

    if args.dry_run:
        with connect(db_url, read_only=True) as conn:
            dry_run(conn, args.kinds)
        return 0

    embedder = None
    if not args.chunk_only:
        if embed_provider() == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            print("error: VOYAGE_API_KEY not set", file=sys.stderr)
            return 2
        embedder = create_embedder(db_url=db_url)

    with connect(db_url) as conn:
        if args.reset:
            n = conn.execute(
                "DELETE FROM web_chunks c USING web_artifacts a "
                "WHERE a.id = c.web_artifact_id AND a.kind = ANY(%s)",
                (args.kinds,),
            ).rowcount
            conn.commit()
            print(f"[+] reset: deleted {n} web_chunks rows", file=sys.stderr)

        if not args.embed_only:
            print(f"[+] phase 1: chunking {','.join(args.kinds)}", file=sys.stderr)
            t0 = time.time()
            arts, chunks = phase_chunk(conn, args.kinds)
            print(
                f"    chunked {arts} artifacts → {chunks} chunks in {time.time() - t0:.1f}s",
                file=sys.stderr,
            )

        if embedder is not None:
            print("[+] phase 2: embedding pending chunks", file=sys.stderr)
            t0 = time.time()
            n = phase_embed(conn, embedder, batch_size=args.batch)
            print(
                f"    embedded {n} chunks in {time.time() - t0:.1f}s",
                file=sys.stderr,
            )

        # Final summary
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_chunks,
                COUNT(*) FILTER (WHERE is_embedded) AS embedded,
                COUNT(*) FILTER (WHERE NOT is_embedded) AS pending,
                COUNT(DISTINCT web_artifact_id) AS artifacts_covered
            FROM web_chunks
            """
        ).fetchone()
        assert summary is not None
        print(
            f"\nweb_chunks: total={summary[0]:,} embedded={summary[1]:,} pending={summary[2]:,} artifacts={summary[3]:,}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
