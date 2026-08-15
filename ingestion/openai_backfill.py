"""Backfill a ChatGPT (OpenAI) data export into Synapse.

Usage:
    python -m ingestion.openai_backfill <export_dir_or_conversations.json>
                                        [--project NAME] [--dry-run]

Accepts the extracted export directory (handles both the single
conversations.json and the sharded conversations-NNN.json layouts) or a
single conversations JSON file. Same shared write path as the claude.ai and
Cursor backfills: span_id dedup, private-session gate, extraction enqueue.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from ingestion.backfill import write_backfill_session
from ingestion.chunks import rebuild_chunks
from ingestion.config import get_settings
from ingestion.db import Database
from ingestion.models import Episode
from ingestion.openai_client import parse_export

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Backfill ChatGPT export into Synapse")
    parser.add_argument(
        "path", type=Path, help="Extracted export directory or conversations JSON file"
    )
    parser.add_argument("--project", default=None, help="Project tag for all episodes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        return 2

    logger.info("Parsing %s", args.path)
    eps, skipped_dnr = parse_export(args.path)
    if args.project:
        eps = [ep.model_copy(update={"project": args.project}) for ep in eps]

    by_session: dict[str, list[Episode]] = defaultdict(list)
    for ep in eps:
        by_session[ep.session_id].append(ep)

    logger.info(
        "Parsed %d episodes across %d conversations (%d do-not-remember conversation(s) skipped)",
        len(eps),
        len(by_session),
        skipped_dnr,
    )

    if args.dry_run:
        print(
            {
                "episodes": len(eps),
                "sessions": len(by_session),
                "skipped_do_not_remember": skipped_dnr,
                "written": 0,
            }
        )
        return 0

    cfg = get_settings()
    db = Database(cfg.db_url)
    written = 0
    try:
        for i, (session_id, session_eps) in enumerate(by_session.items(), start=1):
            written += write_backfill_session(db, session_id, session_eps)
            rebuild_chunks(db, session_id)
            if i % 200 == 0:
                logger.info("Progress: %d / %d conversations", i, len(by_session))
    finally:
        db.close()

    print(
        {
            "episodes": len(eps),
            "sessions": len(by_session),
            "skipped_do_not_remember": skipped_dnr,
            "written": written,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
