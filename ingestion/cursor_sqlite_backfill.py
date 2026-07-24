"""Backfill Cursor SQLite chat history into Synapse.

Usage:
    python -m ingestion.cursor_sqlite_backfill <root_dir> [--dry-run] [--limit N]

Walks <root_dir> for *.vscdb files, parses each via CursorSQLiteParser, and
writes Episodes per composer (one composer = one conversation).
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
from ingestion.cursor_sqlite_client import CursorSQLiteParser
from ingestion.db import Database
from ingestion.models import Episode

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Backfill Cursor SQLite history into Synapse")
    parser.add_argument("root", type=Path, help="Directory containing .vscdb files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N db files")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} does not exist", file=sys.stderr)
        return 2

    files = sorted(args.root.rglob("*.vscdb"))
    if args.limit:
        files = files[: args.limit]

    logger.info("Scanning %d Cursor SQLite files", len(files))

    by_session: dict[str, list[Episode]] = defaultdict(list)
    parsed_files = 0
    parsed_episodes = 0

    for path in files:
        try:
            eps = CursorSQLiteParser.parse_file(path)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", path, e)
            continue
        parsed_files += 1
        if eps:
            parsed_episodes += len(eps)
            for ep in eps:
                by_session[ep.session_id].append(ep)

    logger.info(
        "Parsed %d files into %d episodes across %d composers",
        parsed_files,
        parsed_episodes,
        len(by_session),
    )

    if args.dry_run:
        print(
            {
                "files": parsed_files,
                "episodes": parsed_episodes,
                "composers": len(by_session),
                "written": 0,
            }
        )
        return 0

    cfg = get_settings()
    db = Database(cfg.db_url)
    written = 0
    try:
        for i, (session_id, eps) in enumerate(by_session.items(), start=1):
            written += write_backfill_session(db, session_id, eps)
            rebuild_chunks(db, session_id)
            if i % 50 == 0:
                logger.info("Progress: %d / %d composers", i, len(by_session))
    finally:
        db.close()

    print(
        {
            "files": parsed_files,
            "episodes": parsed_episodes,
            "composers": len(by_session),
            "written": written,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
