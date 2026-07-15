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
from typing import Any

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

    # dream_runs bookkeeping (schema 044): one row per run, feeds the Metrics ops page's
    # "last nightly-dream run" panel + the phase-5 Dream-report page. ALL of it is fail-soft:
    # a bookkeeping error can never crash the pipeline (see _dream_* helpers). run_id is None
    # when the start-insert failed, in which case the finish-update no-ops.
    run_id = _dream_record_start(db_url)
    run_stages: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    errors: list[str] = []
    overall_ok = True

    # dream→skills lane. Lazy import + try/except so a lane error can't crash the dream loop,
    # and gated on SKILLS_LANE_ENABLED so the image can ship inert until the cutover flips it on
    # (deploy → verify a manual pass → set the flag → retire the client cron).
    if os.environ.get("SKILLS_LANE_ENABLED") == "1":
        try:
            from dream.skills.nightly import run_lane

            run_lane(limit=int(os.environ.get("SKILLS_LANE_LIMIT", "30")))
            run_stages["skills"] = {"ran": True, "ok": True}
        except Exception as e:
            logger.error("dream→skills lane failed: %s", e, exc_info=True)
            run_stages["skills"] = {"ran": True, "ok": False}
            errors.append(f"skills lane: {e}")
            overall_ok = False

    # dream→config lane (mines behavioral corrections -> config_proposals). ON by default; the env
    # var is a KILL SWITCH (CONFIG_LANE_ENABLED=0 to disable), not an enable gate. Propose-only and
    # fail-soft, so a lane error can't crash the dream loop — nothing reaches disk without review.
    if os.environ.get("CONFIG_LANE_ENABLED", "1") != "0":
        try:
            from dream.config.nightly import run_lane as run_config_lane

            res = run_config_lane(limit=int(os.environ.get("CONFIG_LANE_LIMIT", "30")))
            run_stages["config"] = {"ran": True, "ok": True}
            # The config lane returns {"sessions", "found", "proposed"} — the ONLY counts a
            # lane exposes cheaply today. Record what it gives; leave the rest absent.
            if isinstance(res, dict):
                if res.get("sessions") is not None:
                    counts["config_sessions_scanned"] = res["sessions"]
                if res.get("found") is not None:
                    counts["config_corrections_found"] = res["found"]
                if res.get("proposed") is not None:
                    counts["config_proposals"] = res["proposed"]
        except Exception as e:
            logger.error("dream→config lane failed: %s", e, exc_info=True)
            run_stages["config"] = {"ran": True, "ok": False}
            errors.append(f"config lane: {e}")
            overall_ok = False

    if 3 in stages:
        _stage3_memory_proposals(db_url)

    # Aggregate the proposals actually raised this run + a few bounded samples (cheap DB reads;
    # fail-soft) so the report panel has drill-in material without re-architecting the lanes.
    samples = _dream_collect_proposals(db_url, run_id, counts)
    _dream_record_finish(db_url, run_id, run_stages, counts, samples, errors, overall_ok)

    logger.info("Dream pipeline complete")


# ---------------------------------------------------------------------------
# dream_runs bookkeeping (schema 044) — every function is fail-soft: a bookkeeping
# error is logged and swallowed so it can NEVER break the pipeline.
# ---------------------------------------------------------------------------

_SAMPLE_CAP = 10  # bound per-count sample arrays (schema 044 header)


def _dream_record_start(db_url: str) -> int | None:
    """INSERT the run row (ok NULL, finished_at NULL) and return its id, or None on any error."""
    try:
        import psycopg

        with psycopg.connect(db_url, autocommit=True) as conn:
            row = conn.execute(
                "INSERT INTO dream_runs (started_at) VALUES (now()) RETURNING id"
            ).fetchone()
            return int(row[0]) if row else None
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs start-insert failed (bookkeeping only): %s", e)
        return None


def _dream_collect_proposals(
    db_url: str, run_id: int | None, counts: dict[str, Any]
) -> dict[str, Any]:
    """Best-effort, cheap: count + sample the proposals raised since this run started, across
    both lanes. Sets counts['proposals_raised'] and returns {"proposals": [...]} (bounded).
    Each sub-query is independently fail-soft — a missing lane schema just yields nothing."""
    samples: dict[str, Any] = {}
    if run_id is None:
        return samples
    try:
        import psycopg

        with psycopg.connect(db_url, autocommit=True, row_factory=None) as conn:
            started = conn.execute(
                "SELECT started_at FROM dream_runs WHERE id = %s", (run_id,)
            ).fetchone()
            if not started:
                return samples
            since = started[0]
            proposals: list[dict[str, Any]] = []
            total = 0
            # skills lane — status 'proposed', created since the run began.
            try:
                for pid, name in conn.execute(
                    "SELECT id, name FROM skills_lane.skill_gap_candidates "
                    "WHERE status = 'proposed' AND created_at >= %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (since, _SAMPLE_CAP),
                ).fetchall():
                    proposals.append({"id": f"skill:{pid}", "kind": "skill", "name": name})
                total += conn.execute(
                    "SELECT count(*) FROM skills_lane.skill_gap_candidates "
                    "WHERE status = 'proposed' AND created_at >= %s",
                    (since,),
                ).fetchone()[0]
            except Exception as e:  # pragma: no cover - lane may be absent
                logger.debug("dream_runs skills sample skipped: %s", e)
            # config lane — status 'proposed', created since the run began.
            try:
                for pid, fkey in conn.execute(
                    "SELECT id, file_key FROM config_lane.config_proposals "
                    "WHERE status = 'proposed' AND created_at >= %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (since, _SAMPLE_CAP),
                ).fetchall():
                    proposals.append({"id": f"config:{pid}", "kind": "config-edit", "name": fkey})
                total += conn.execute(
                    "SELECT count(*) FROM config_lane.config_proposals "
                    "WHERE status = 'proposed' AND created_at >= %s",
                    (since,),
                ).fetchone()[0]
            except Exception as e:  # pragma: no cover - lane may be absent
                logger.debug("dream_runs config sample skipped: %s", e)
            counts["proposals_raised"] = total
            if proposals:
                samples["proposals"] = proposals[:_SAMPLE_CAP]
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs proposal aggregation failed (bookkeeping only): %s", e)
    return samples


def _dream_record_finish(
    db_url: str,
    run_id: int | None,
    run_stages: dict[str, Any],
    counts: dict[str, Any],
    samples: dict[str, Any],
    errors: list[str],
    ok: bool,
) -> None:
    """UPDATE the run row with finished_at + the collected outcome. No-op if run_id is None."""
    if run_id is None:
        return
    try:
        import psycopg
        from psycopg.types.json import Json

        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute(
                "UPDATE dream_runs SET finished_at = now(), stages = %s, counts = %s, "
                "samples = %s, errors = %s, ok = %s WHERE id = %s",
                (Json(run_stages), Json(counts), Json(samples), Json(errors), ok, run_id),
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs finish-update failed (bookkeeping only): %s", e)


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
