"""Dream mode — unified nightly memory consolidation pipeline.

The numbered stages are all gone: stage 1 (Logfire review) left with the
Logfire-ingestion rip-out, stage 2 (dream docs over segment summaries) with the
summary layer + FalkorDB decommission (#67 PR 3), and stage 3 (memory-proposal
mining) was deleted 2026-07-24 — its substrate (summaries) and sink (auto-memory
files) were both retired, and the concept shipped for real as the dream→skills
and dream→config proposal lanes (plus the dream→notes curation lane), which are
what run_once() runs today.

Usage:
    python -m dream               # run once immediately
    python -m dream --schedule    # loop, fire at 2:03am UTC daily
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import logfire
import psycopg
from psycopg.types.json import Json

from dream.skills import config as skills_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _require_db_url() -> str:
    """The Postgres DSN, resolved via the skills-lane config (env, then .env fallback).
    Hard-fails if unset — the pipeline cannot run without a database."""
    url = skills_config.db_url()
    if not url:
        raise RuntimeError("SYNAPSE_DB_URL not set")
    return url


_SAMPLE_CAP = 10  # bound per-count sample arrays (schema 044 header)

# ---------------------------------------------------------------------------
# The lanes. Each runner adapts its lane to ONE result shape:
#   counts   flat {name: number}, merged into dream_runs.counts (the Dream report prints them)
#   samples  a few strings, kept under dream_runs.samples[<lane name>] (optional)
#   errors   stages that failed ("<stage>: <exception>"); any marks the lane, and the run, not ok
# A runner that raises is recorded the same way, as one error for the whole lane.
# ---------------------------------------------------------------------------
LaneResult = dict[str, Any]


class Lane(NamedTuple):
    name: str
    #: The env var that gates the lane. ``=0`` turns it off; anything else, or unset, leaves it on.
    switch: str | None
    run: Callable[[], LaneResult]


def _skills_lane() -> LaneResult:
    """dream→skills: mines past sessions for new-skill / retune / merge proposals and pings
    Discord; nothing changes until a proposal is accepted in review. The lane isolates its
    own stages and lists the ones that failed."""
    from dream.skills.nightly import run_lane

    res = run_lane(limit=int(os.environ.get("SKILLS_LANE_LIMIT", "30")))
    s = res["stages"]
    judge, usage, drafts = s.get("judge", {}), s.get("usage", {}), s.get("drafts", {})
    # One judge call per session judged. "unparsed" counts the replies with no JSON object,
    # the one failure a lane cannot tell from "found nothing" (a model it cannot reach raises).
    counts = {
        "skills_sessions_judged": judge.get("sessions", 0),
        "skills_judge_unparsed": judge.get("unparsed", 0),
        "skills_detector_unparsed": sum(
            s.get(stage, {}).get("judge_failures", 0) for stage in ("struggle_arc", "post_fire")
        ),
        "skills_gaps": judge.get("derive", {}).get("gaps", 0),
        "skills_retune_signals": judge.get("retune", {}).get("under_trigger", 0),
        "skills_fire_events": usage.get("fire_events", 0),
        "skills_dismissals": usage.get("dismissals", 0),
        "skills_drafted": drafts.get("drafted", 0) + drafts.get("redrafted", 0),
        "skills_draft_errors": drafts.get("errors", 0),
        "skills_retired": s.get("decay", {}).get("retired", 0),
    }
    return {"counts": counts, "errors": res["errors"]}


def _config_lane() -> LaneResult:
    """dream→config: mines behavioral corrections into config_proposals. Propose-only, so
    nothing reaches disk without review."""
    from dream.config.nightly import run_lane

    res = run_lane(limit=int(os.environ.get("CONFIG_LANE_LIMIT", "30")))
    counts = {
        "config_sessions_scanned": res["sessions"],  # one judge call each
        "config_corrections_found": res["found"],
        "config_proposals": res["proposed"],
    }
    return {"counts": counts, "errors": []}


def _notes_lane() -> LaneResult:
    """dream→notes curation (merges duplicates, retires corrected notes, re-scopes global
    notes that are really about one project). Unlike the two lanes above it AUTO-APPLIES — no
    proposals, no review queue, by explicit user decision. Its safety model is structural
    instead: every op is reversible (superseded_by / type+project, nothing deleted),
    judgements need a both-orders consensus, per-night caps bound the blast radius, and
    every verdict lands in notes_curation (051)."""
    from dream.notes.nightly import run_lane

    res = run_lane()
    counts = dict(res["counts"])
    samples = list(res["samples"])[:_SAMPLE_CAP]
    # Per-item judge failures are fail-soft by design (the item is simply re-queued
    # tomorrow), so they are reported as a count + samples rather than as stage errors —
    # `errors`/`ok` stay reserved for a stage that blew up.
    if res["errors"]:
        counts["notes_judge_errors"] = len(res["errors"])
        samples += [f"error: {e}" for e in res["errors"][:_SAMPLE_CAP]]
    return {"counts": counts, "samples": samples, "errors": []}


#: recall_metrics gets a row per recall, board serve and remember, and nothing else caps it.
#: The dashboard reads at most 30 days of it.
RECALL_METRICS_KEEP_DAYS = 180


def _prune_telemetry() -> LaneResult:
    with psycopg.connect(_require_db_url(), autocommit=True) as conn:
        pruned = conn.execute(
            "DELETE FROM recall_metrics WHERE created_at < now() - make_interval(days => %s)",
            (RECALL_METRICS_KEEP_DAYS,),
        ).rowcount
    return {"counts": {"recall_metrics_pruned": pruned}, "errors": []}


# Every switch is a KILL SWITCH, not an enable gate: the lanes are ON by default.
LANES = (
    Lane("skills", "SKILLS_LANE_ENABLED", _skills_lane),
    Lane("config", "CONFIG_LANE_ENABLED", _config_lane),
    Lane("notes", "SYNAPSE_NOTES_LANE", _notes_lane),
    Lane("retention", None, _prune_telemetry),
)

#: Any constant no other advisory lock in the database uses.
_RUN_LOCK = 0x44524D00


@contextmanager
def _run_lock(db_url: str) -> Iterator[bool]:
    """Hold the dream run lock for the block; yields whether this process got it. Two runs
    (the scheduled one and a manual `python -m dream`) would race on the lane cursors and
    judge and apply the same notes twice. A session-level advisory lock, so it is released
    when the connection closes, including when the process dies."""
    with psycopg.connect(db_url, autocommit=True) as conn:
        row = conn.execute("SELECT pg_try_advisory_lock(%s)", (_RUN_LOCK,)).fetchone()
        yield bool(row and row[0])


def run_once() -> None:
    db_url = _require_db_url()
    with _run_lock(db_url) as acquired:
        if not acquired:
            logger.warning("Another dream run is in progress (advisory lock held); skipping")
            return
        _run(db_url)


def _run(db_url: str) -> None:
    logger.info("Dream pipeline starting")

    # dream_runs bookkeeping (schema 044): one row per run, feeds the Metrics ops page's
    # "last nightly-dream run" panel + the phase-5 Dream-report page. ALL of it is fail-soft:
    # a bookkeeping error can never crash the pipeline (see _dream_* helpers). run_id is None
    # when the start-insert failed, in which case the finish-update no-ops.
    _dream_close_interrupted(db_url)
    run_id = _dream_record_start(db_url)
    proposed_before = _proposed(db_url)
    run_stages: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    errors: list[str] = []
    overall_ok = True

    # A lane error can't crash the dream loop. The skills and config lanes are propose-only
    # (nothing reaches disk without review) and the notes lane is reversible.
    for lane in LANES:
        if lane.switch and os.environ.get(lane.switch, "1") == "0":
            continue
        try:
            with logfire.span("dream lane {lane}", lane=lane.name):
                res = lane.run()
        except Exception as e:
            logger.error("dream→%s lane failed: %s", lane.name, e, exc_info=True)
            run_stages[lane.name] = {"ran": True, "ok": False}
            errors.append(f"{lane.name} lane: {e}")
            overall_ok = False
            continue
        counts.update(res["counts"])
        if res.get("samples"):
            samples[lane.name] = res["samples"][: _SAMPLE_CAP * 2]
        run_stages[lane.name] = {"ran": True, "ok": not res["errors"]}
        if res["errors"]:
            errors.extend(f"{lane.name} lane {e}" for e in res["errors"])
            overall_ok = False

    # Aggregate the proposals actually raised this run + a few bounded samples (cheap DB reads;
    # fail-soft) so the report panel has drill-in material without re-architecting the lanes.
    samples.update(_dream_collect_proposals(db_url, proposed_before, counts))
    _dream_record_finish(db_url, run_id, run_stages, counts, samples, errors, overall_ok)

    logger.info("Dream pipeline complete")


# ---------------------------------------------------------------------------
# dream_runs bookkeeping (schema 044) — every function is fail-soft: a bookkeeping
# error is logged and swallowed so it can NEVER break the pipeline.
# ---------------------------------------------------------------------------

_INTERRUPTED = "interrupted: the process stopped before the run finished"


def _dream_close_interrupted(db_url: str) -> None:
    """Mark the runs that never finished as failed. Called with the run lock held, so a row
    still open belongs to a process that died (a redeploy in the middle of a run) and not to
    one that is running. finished_at stays NULL: nobody knows when it died."""
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            closed = conn.execute(
                "UPDATE dream_runs SET ok = false, errors = errors || %s::jsonb WHERE ok IS NULL",
                (Json([_INTERRUPTED]),),
            ).rowcount
        if closed:
            logger.warning("Marked %d unfinished dream run(s) as interrupted", closed)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs interrupted-update failed (bookkeeping only): %s", e)


def _dream_record_start(db_url: str) -> int | None:
    """INSERT the run row (ok NULL, finished_at NULL) and return its id, or None on any error."""
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            row = conn.execute(
                "INSERT INTO dream_runs (started_at) VALUES (now()) RETURNING id"
            ).fetchone()
            return int(row[0]) if row else None
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs start-insert failed (bookkeeping only): %s", e)
        return None


#: (id prefix, sample kind, query for the id and name of every proposal awaiting review)
_PROPOSED = (
    (
        "skill",
        "skill",
        "SELECT id, name FROM skills_lane.skill_gap_candidates "
        "WHERE status = 'proposed' ORDER BY id DESC",
    ),
    (
        "config",
        "config-edit",
        "SELECT id, file_key FROM config_lane.config_proposals "
        "WHERE status = 'proposed' ORDER BY id DESC",
    ),
)


def _proposed(db_url: str) -> dict[str, dict[str, str]] | None:
    """Every proposal awaiting review in both lanes, as {"skill:12": {id, kind, name}}, or
    None when the database cannot be read. Each query is independently fail-soft — a
    missing lane schema just yields nothing."""
    found: dict[str, dict[str, str]] = {}
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            for prefix, kind, sql in _PROPOSED:
                try:
                    for pid, name in conn.execute(sql).fetchall():
                        key = f"{prefix}:{pid}"
                        found[key] = {"id": key, "kind": kind, "name": name}
                except Exception as e:  # pragma: no cover - lane may be absent
                    logger.debug("dream_runs %s proposals skipped: %s", prefix, e)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs proposal aggregation failed (bookkeeping only): %s", e)
        return None
    return found


def _dream_collect_proposals(
    db_url: str, before: dict[str, dict[str, str]] | None, counts: dict[str, Any]
) -> dict[str, Any]:
    """The proposals awaiting review now that were not when the run started, across both
    lanes. That includes an older observe row a sighting tonight promoted, which a
    created_at filter misses. Sets counts['proposals_raised'] and returns
    {"proposals": [...]} (bounded)."""
    after = _proposed(db_url)
    if before is None or after is None:
        return {}
    raised = [p for key, p in after.items() if key not in before]
    counts["proposals_raised"] = len(raised)
    return {"proposals": raised[:_SAMPLE_CAP]} if raised else {}


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
        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute(
                "UPDATE dream_runs SET finished_at = now(), stages = %s, counts = %s, "
                "samples = %s, errors = %s, ok = %s WHERE id = %s",
                (Json(run_stages), Json(counts), Json(samples), Json(errors), ok, run_id),
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("dream_runs finish-update failed (bookkeeping only): %s", e)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _last_window(now: datetime) -> datetime:
    """The most recent 02:03 UTC at or before `now`."""
    target = now.replace(hour=2, minute=3, second=0, microsecond=0)
    return target if target <= now else target - timedelta(days=1)


def _seconds_until_2am() -> float:
    now = datetime.now(UTC)
    return (_last_window(now) + timedelta(days=1) - now).total_seconds()


def _missed_window(db_url: str, now: datetime) -> bool:
    """True when the dream has run before but no run since the last 02:03 window finished:
    the container was down at 02:03, or redeployed during the run (watchtower redeploys on
    every merge). A database with no run at all is a new install, which waits for 02:03."""
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM dream_runs) AND NOT EXISTS ("
                "SELECT 1 FROM dream_runs WHERE finished_at IS NOT NULL AND started_at >= %s)",
                (_last_window(now),),
            ).fetchone()
    except Exception as e:
        logger.warning("dream_runs catch-up check failed, waiting for 2:03am: %s", e)
        return False
    return bool(row and row[0])


def _run_scheduled() -> None:
    try:
        run_once()
    except Exception as e:
        logger.error("Dream pipeline failed: %s", e, exc_info=True)


def main(argv: list[str]) -> None:
    from ingestion.schema_check import check_schema_version

    db_url = _require_db_url()
    check_schema_version(db_url)

    # Spans for the lanes' LLM calls and HTTP requests, under LOGFIRE_SERVICE_NAME (compose
    # sets synapse-dream), the way ingestion/__main__ configures the poller.
    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "synapse-dream"),
        send_to_logfire="if-token-present",
    )
    logfire.instrument_claude_agent_sdk()
    logfire.instrument_httpx()

    if "--schedule" not in argv:
        run_once()
        return
    logger.info("Dream scheduler — will fire at 2:03am UTC daily")
    if _missed_window(db_url, datetime.now(UTC)):
        logger.info("No run finished since the last 2:03am window; running now")
        _run_scheduled()
    while True:
        wait = _seconds_until_2am()
        logger.info("Next dream in %.0fs (%.1fh)", wait, wait / 3600)
        time.sleep(wait)
        _run_scheduled()


if __name__ == "__main__":
    main(sys.argv[1:])
