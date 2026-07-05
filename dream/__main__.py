"""Dream mode — unified nightly memory consolidation pipeline.

Stages run sequentially (numbering preserved; the old Logfire-review stage 1
was removed when Logfire ingestion was ripped out, and stage 2 — dream docs
over segment summaries — was deleted with the summary layer + FalkorDB
decommission, #67 PR 3):
  3. Memory proposals: mine recent dreams for behavioral patterns → memory_proposals table

Usage:
    python -m dream               # run once immediately
    python -m dream --schedule    # loop, fire at 2am UTC daily
    python -m dream --stage 3     # run only stage 3 (useful for testing)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Config from env / defaults
_DB_URL = os.environ.get("SYNAPSE_DB_URL", "")


def _load_db_url() -> str:
    if _DB_URL:
        return _DB_URL
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("SYNAPSE_DB_URL="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("SYNAPSE_DB_URL not set")


def run_once(stages: set[int] | None = None) -> None:
    # Stage 3 (memory proposals) is RETIRED with summaries (task #63): it reads
    # doc_type='summary', which is no longer generated. It stays defined for a
    # future dream rework on a better substrate, but run_once no longer invokes
    # it by default (pass an explicit stage set to run it deliberately).
    stages = stages or set()
    db_url = _load_db_url()

    logger.info("Dream pipeline starting (stages=%s)", sorted(stages))

    # dream→skills lane. Lazy import + try/except so a lane error can't crash the dream loop,
    # and gated on SKILLS_LANE_ENABLED so the image can ship inert until the cutover flips it on
    # (deploy → verify a manual pass → set the flag → retire the client cron).
    if os.environ.get("SKILLS_LANE_ENABLED") == "1":
        try:
            from dream.skills.nightly import run_lane

            run_lane(limit=int(os.environ.get("SKILLS_LANE_LIMIT", "30")))
        except Exception as e:
            logger.error("dream→skills lane failed: %s", e, exc_info=True)

    # dream→config lane (mines behavioral corrections -> config_proposals). ON by default; the env
    # var is a KILL SWITCH (CONFIG_LANE_ENABLED=0 to disable), not an enable gate. Propose-only and
    # fail-soft, so a lane error can't crash the dream loop — nothing reaches disk without review.
    if os.environ.get("CONFIG_LANE_ENABLED", "1") != "0":
        try:
            from dream.config.nightly import run_lane as run_config_lane

            run_config_lane(limit=int(os.environ.get("CONFIG_LANE_LIMIT", "30")))
        except Exception as e:
            logger.error("dream→config lane failed: %s", e, exc_info=True)

    if 3 in stages:
        _stage3_memory_proposals(db_url)

    logger.info("Dream pipeline complete")


def _stage3_memory_proposals(db_url: str) -> None:
    """Mine behavioral patterns from recent dreams → memory_proposals rows."""
    logger.info("Stage 3: Memory-proposal mining starting")
    try:
        from dream.stage3 import mine_proposals

        n = mine_proposals(db_url)
        logger.info("Stage 3: %d memory proposal(s) inserted", n)
    except Exception as e:
        logger.error("Stage 3 failed: %s", e, exc_info=True)


def _seconds_until_2am() -> float:
    now = datetime.now(UTC)
    target = now.replace(hour=2, minute=3, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta

        target += timedelta(days=1)
    return (target - now).total_seconds()


if __name__ == "__main__":
    from ingestion.schema_check import check_schema_version

    check_schema_version(_load_db_url())

    args = set(sys.argv[1:])

    # Parse --stage N for selective execution
    stages: set[int] | None = None
    if "--stage" in sys.argv:
        idx = sys.argv.index("--stage")
        if idx + 1 < len(sys.argv):
            stages = {int(sys.argv[idx + 1])}

    if "--schedule" in args:
        logger.info("Dream scheduler — will fire at 2:03am UTC daily")
        while True:
            wait = _seconds_until_2am()
            logger.info("Next dream in %.0fs (%.1fh)", wait, wait / 3600)
            time.sleep(wait)
            try:
                run_once(stages)
            except Exception as e:
                logger.error("Dream pipeline failed: %s", e, exc_info=True)
    else:
        run_once(stages)
