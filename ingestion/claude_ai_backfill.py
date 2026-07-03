"""Backfill claude.ai conversation export into Synapse.

Usage:
    python -m ingestion.claude_ai_backfill <conversations.json> [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from ingestion.chunks import rebuild_chunks
from ingestion.claude_ai_client import parse_export
from ingestion.config import get_settings
from ingestion.db import Database
from ingestion.models import Episode, ExtractionItem

logger = logging.getLogger(__name__)


def _write_session(db: Database, session_id: str, eps: list[Episode]) -> int:
    existing = db.get_session_episodes(session_id)
    next_seq = (max((e["sequence"] for e in existing), default=0)) + 1
    written = 0
    for ep in sorted(eps, key=lambda e: e.sequence):
        if ep.span_id and db.span_id_exists(ep.span_id):
            continue
        ep_to_write = ep.model_copy(update={"sequence": next_seq + written})
        episode_id = db.upsert_episode(ep_to_write)
        written += 1
        if episode_id and ep_to_write.content.strip():
            db.enqueue_extraction(
                ExtractionItem(
                    episode_id=episode_id,
                    session_id=ep_to_write.session_id,
                    content=ep_to_write.content,
                    content_type="episode",
                    project=ep_to_write.project,
                )
            )
    return written


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Backfill claude.ai conversation export into Synapse"
    )
    parser.add_argument("path", type=Path, help="Path to conversations.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        return 2

    logger.info("Parsing %s", args.path)
    eps = parse_export(args.path)

    by_session: dict[str, list[Episode]] = defaultdict(list)
    for ep in eps:
        by_session[ep.session_id].append(ep)

    logger.info("Parsed %d episodes across %d conversations", len(eps), len(by_session))

    if args.dry_run:
        print({"episodes": len(eps), "sessions": len(by_session), "written": 0})
        return 0

    cfg = get_settings()
    db = Database(cfg.db_url)
    written = 0
    try:
        for i, (session_id, session_eps) in enumerate(by_session.items(), start=1):
            written += _write_session(db, session_id, session_eps)
            rebuild_chunks(db, session_id)
            if i % 200 == 0:
                logger.info("Progress: %d / %d conversations", i, len(by_session))
    finally:
        db.close()

    print({"episodes": len(eps), "sessions": len(by_session), "written": written})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
