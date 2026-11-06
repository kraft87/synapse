#!/usr/bin/env python3
"""
Backfill web_chunks from existing web_artifacts.kind='web_scrape' rows.

Two phases (both idempotent):
  1. Chunk: for any web_artifact without chunks, run chunk_markdown() and
     INSERT new web_chunks rows (is_embedded=false).
  2. Embed: for any web_chunks with is_embedded=false, batch through Voyage-4-large
     and UPDATE the embedding column.

Re-runs cheaply pick up where they left off via the UNIQUE (web_artifact_id, idx)
and the is_embedded flag.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.embedding import VoyageEmbeddingModel
from ingestion.web_chunker import chunk_markdown


def load_env():
    db_url = os.environ.get("SYNAPSE_DB_URL")
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists() and (not db_url or not voyage_key):
        for line in env_path.read_text().splitlines():
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k == "SYNAPSE_DB_URL" and not db_url:
                db_url = v
            elif k == "VOYAGE_API_KEY" and not voyage_key:
                voyage_key = v
    if not db_url:
        print("error: SYNAPSE_DB_URL not set", file=sys.stderr)
        sys.exit(2)
    if not voyage_key:
        print("error: VOYAGE_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    return db_url, voyage_key


def phase_chunk(conn: psycopg.Connection) -> tuple[int, int]:
    """Chunk web_scrapes that don't yet have any chunks. Returns (artifacts, chunks_written)."""
    rows = conn.execute(
        """
        SELECT a.id, a.content_markdown, a.fetched_at
        FROM web_artifacts a
        WHERE a.kind IN ('web_scrape', 'research_brief')
          AND a.content_markdown IS NOT NULL
          AND LENGTH(a.content_markdown) > 0
          AND NOT EXISTS (SELECT 1 FROM web_chunks c WHERE c.web_artifact_id = a.id)
        ORDER BY a.id
        """
    ).fetchall()

    artifacts_done = 0
    chunks_written = 0
    for r in rows:
        artifact_id, content, fetched_at = r
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


def phase_embed(conn: psycopg.Connection, voyage_key: str, batch_size: int = 128) -> int:
    """Embed pending web_chunks (is_embedded=false). Returns count embedded."""
    embedder = VoyageEmbeddingModel(api_key=voyage_key)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill web_chunks: chunk + embed.")
    parser.add_argument("--chunk-only", action="store_true", help="Run only the chunking phase.")
    parser.add_argument("--embed-only", action="store_true", help="Run only the embedding phase.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP all web_chunks rows and re-chunk + re-embed from scratch.",
    )
    parser.add_argument("--batch", type=int, default=128, help="Embedding batch size.")
    args = parser.parse_args()

    db_url, voyage_key = load_env()

    with psycopg.connect(db_url, autocommit=False) as conn:
        if args.reset:
            n = conn.execute("DELETE FROM web_chunks").rowcount
            conn.commit()
            print(f"[+] reset: deleted {n} web_chunks rows", file=sys.stderr)

        if not args.embed_only:
            print("[+] phase 1: chunking web_scrapes", file=sys.stderr)
            t0 = time.time()
            arts, chunks = phase_chunk(conn)
            print(
                f"    chunked {arts} artifacts → {chunks} chunks in {time.time() - t0:.1f}s",
                file=sys.stderr,
            )

        if not args.chunk_only:
            print("[+] phase 2: embedding pending chunks", file=sys.stderr)
            t0 = time.time()
            n = phase_embed(conn, voyage_key, batch_size=args.batch)
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
        print(
            f"\nweb_chunks: total={summary[0]:,} embedded={summary[1]:,} pending={summary[2]:,} artifacts={summary[3]:,}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
