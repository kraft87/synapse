"""Recall history and the metrics aggregates (phase 4): Recall, Ingestion and Corpus ops."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from starlette.requests import Request

from ingestion import pg_pool
from mcp_server.dash.common import _SNIPPET_MAX, _limit, _TtlCache
from mcp_server.http_helpers import _iso, route

_RECALL_HISTORY_DEFAULT = 50
_RECALL_HISTORY_MAX = 200


def _recall_history(db_url: str, limit: int) -> dict[str, Any]:
    """Recent recall() calls from the recall_metrics telemetry log (kind='recall').

    A dedicated slim endpoint for the Recall console's History tab — newest first,
    just the columns the table renders. Deviates from spec §8 (which routed history
    through the phase-4 /metrics/recall aggregate) by shipping now; the aggregate can
    still supersede it later. Turns-mode (kind='episodes') / fetch / remember rows are
    excluded by the kind filter so the console shows only overview recall() calls.
    """
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT id, created_at, query, source, ms_total, est_tokens, rerank_top_score "
            "FROM recall_metrics WHERE kind = 'recall' "
            "ORDER BY created_at DESC, id DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return {
            "items": [
                {
                    "id": r["id"],
                    "created_at": _iso(r["created_at"]),
                    "query": r["query"],
                    "source": r["source"],
                    "ms_total": r["ms_total"],
                    "est_tokens": r["est_tokens"],
                    "rerank_top_score": r["rerank_top_score"],
                }
                for r in rows
            ]
        }


# Three read-only aggregate endpoints over the telemetry + queue + corpus tables. The
# honesty rules (docs/dashboard-contract.md §"Phase 4"):
#   * recall percentiles come straight from recall_metrics (kind='recall') — the SAME
#     numbers the engine already logs; no re-instrumentation.
#   * ingestion series carry ONLY what extraction_queue's real columns support. Historical
#     queue DEPTH is NOT reconstructable (a row's status is overwritten in place), so we
#     never fabricate a depth history — we series enqueue/hour (enqueued_at) and
#     completions/hour (processed_at, the only completion timestamp) and report depth as a
#     LIVE snapshot only.
#   * corpus headline counts use pg_class.reltuples (a fast estimate) for large tables and
#     an exact count(*) for small ones; the estimated flag is surfaced per-table.

_METRICS_WINDOW_MAX_S = 30 * 86400  # window is capped at 30d (contract)
_RECALL_WINDOW_DEFAULT_S = 7 * 86400  # /metrics/recall default 7d
_INGEST_WINDOW_DEFAULT_S = 48 * 3600  # /metrics/ingestion default 48h
# Recall legs timed by the engine (recall_metrics ms_* columns). notes is only populated
# when the notes leg ran — a bucket with none stays absent for that leg. The retired
# timeline/prefs legs (ms_timeline/ms_prefs) are no longer reported.
_RECALL_LEGS = ("embed", "bm25", "vector", "kg", "web", "rerank", "notes")
_SLOWEST_CAP = 10
_SCORE_BINS = 10
_INGEST_FAILURES_CAP = 20
_CORPUS_TTL_S = 3600  # 1h in-process cache (row counts over 40K+ rows are the cost)
# Below this reltuples estimate, count exactly; at/above it, trust the estimate (contract).
_CORPUS_ESTIMATE_FLOOR = 50000


# (table, time_column|None) — the corpus tables to report + which column drives spark_30d.
_CORPUS_TABLES: tuple[tuple[str, str | None], ...] = (
    ("episodes", "created_at"),
    ("kg_entities", "t_created"),
    ("kg_relationships", "t_created"),
    ("timeline_events", "ingested_at"),
    ("preferences", "ingested_at"),
    ("notes", "created_at"),
    ("chunks", "created_at"),
)


def _parse_window(raw: str | None, default_s: int) -> int:
    """A window string ('7d' / '48h' / '90m' / a bare seconds int) → seconds, floored at 1h
    and capped at 30d. Anything unparseable falls back to the endpoint default."""
    if not raw:
        return default_s
    raw = raw.strip().lower()
    try:
        if raw.endswith("d"):
            secs = int(float(raw[:-1]) * 86400)
        elif raw.endswith("h"):
            secs = int(float(raw[:-1]) * 3600)
        elif raw.endswith("m"):
            secs = int(float(raw[:-1]) * 60)
        else:
            secs = int(float(raw))
    except (TypeError, ValueError):
        return default_s
    return max(3600, min(secs, _METRICS_WINDOW_MAX_S))


def _r1(v: Any) -> float | None:
    """Round a numeric to 1 decimal, passing None through."""
    return round(float(v), 1) if v is not None else None


def _metrics_recall(db_url: str, window_s: int) -> dict[str, Any]:
    """Hourly recall latency percentiles + per-leg p50 + slowest queries + rerank-score
    histogram, over recall_metrics (kind='recall') within the window, plus `summary`: the
    window's own p50/p95/tokens (percentiles do not average across hourly buckets).
    percentile_cont is a true continuous percentile; NULL leg timings are ignored by the
    aggregate (a disabled leg simply doesn't contribute)."""
    leg_select = ", ".join(
        f"percentile_cont(0.5) WITHIN GROUP (ORDER BY ms_{leg}::double precision) AS leg_{leg}"
        for leg in _RECALL_LEGS
    )
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        series_rows = conn.execute(
            "SELECT date_trunc('hour', created_at) AS t, count(*) AS calls, "  # nosec B608 — leg_select is built from the _RECALL_LEGS constants
            "  percentile_cont(0.5) WITHIN GROUP (ORDER BY ms_total::double precision) AS p50, "
            "  percentile_cont(0.95) WITHIN GROUP (ORDER BY ms_total::double precision) AS p95, "
            "  percentile_cont(0.5) WITHIN GROUP (ORDER BY est_tokens::double precision) AS tok, "
            f"  {leg_select} "
            "FROM recall_metrics WHERE kind = 'recall' AND created_at >= now() - %s::interval "
            "GROUP BY 1 ORDER BY 1",
            (f"{window_s} seconds",),
        ).fetchall()
        series = []
        for r in series_rows:
            legs = {
                leg: _r1(r[f"leg_{leg}"]) for leg in _RECALL_LEGS if r[f"leg_{leg}"] is not None
            }
            series.append(
                {
                    "t": _iso(r["t"]),
                    "p50": _r1(r["p50"]),
                    "p95": _r1(r["p95"]),
                    "calls": int(r["calls"]),
                    "tokens_p50": int(r["tok"]) if r["tok"] is not None else None,
                    "legs_p50": legs,
                }
            )

        summary_row = (
            conn.execute(
                "SELECT count(*) AS calls, "
                "  percentile_cont(0.5) WITHIN GROUP (ORDER BY ms_total::double precision) AS p50, "
                "  percentile_cont(0.95) WITHIN GROUP (ORDER BY ms_total::double precision) AS p95, "
                "  percentile_cont(0.5) WITHIN GROUP (ORDER BY est_tokens::double precision) AS tok "
                "FROM recall_metrics WHERE kind = 'recall' AND created_at >= now() - %s::interval",
                (f"{window_s} seconds",),
            ).fetchone()
            or {}
        )
        summary = {
            "calls": int(summary_row.get("calls") or 0),
            "p50": _r1(summary_row.get("p50")),
            "p95": _r1(summary_row.get("p95")),
            "tokens_p50": int(summary_row["tok"]) if summary_row.get("tok") is not None else None,
        }

        slow_rows = conn.execute(
            "SELECT query, ms_total, created_at FROM recall_metrics "
            "WHERE kind = 'recall' AND ms_total IS NOT NULL AND created_at >= now() - %s::interval "
            "ORDER BY ms_total DESC, id DESC LIMIT %s",
            (f"{window_s} seconds", _SLOWEST_CAP),
        ).fetchall()
        slowest = [
            {
                "query": r["query"],
                "ms_total": _r1(r["ms_total"]),
                "created_at": _iso(r["created_at"]),
            }
            for r in slow_rows
        ]

        # width_bucket(x, 0, 1, 10): 1..10 for x in [0,1); 0 for x<0; 11 for x>=1. Fold the
        # out-of-range edges into the terminal bins so a 1.0 score lands in the top bin.
        hist_rows = conn.execute(
            "SELECT width_bucket(rerank_top_score, 0, 1, %s) AS b, count(*) AS n "
            "FROM recall_metrics WHERE kind = 'recall' AND rerank_top_score IS NOT NULL "
            "AND created_at >= now() - %s::interval GROUP BY 1",
            (_SCORE_BINS, f"{window_s} seconds"),
        ).fetchall()
        counts_by_bin = [0] * _SCORE_BINS
        for r in hist_rows:
            b = int(r["b"])
            idx = min(max(b - 1, 0), _SCORE_BINS - 1)  # clamp 0→bin0 and 11→bin9
            counts_by_bin[idx] += int(r["n"])
        score_hist = [
            {"lo": round(i / _SCORE_BINS, 2), "hi": round((i + 1) / _SCORE_BINS, 2), "n": n}
            for i, n in enumerate(counts_by_bin)
        ]
        return {"summary": summary, "series": series, "slowest": slowest, "score_hist": score_hist}


def _metrics_ingestion(db_url: str, window_s: int) -> dict[str, Any]:
    """Extraction-queue health. queue_depth is a LIVE snapshot (pending/processing/failed
    right now); throughput series are enqueue/hour (enqueued_at) + completions/hour
    (processed_at of the rows that finished 'done'; a failed row stamps processed_at too and
    is no completion) — the ONLY honest series the columns support. There is no historical
    depth series because a row's status is overwritten in place (contract §Phase 4)."""
    interval = f"{window_s} seconds"
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        depth = conn.execute(
            "SELECT count(*) FILTER (WHERE status = 'pending') AS pending, "
            "  count(*) FILTER (WHERE status = 'processing') AS processing, "
            "  count(*) FILTER (WHERE status = 'failed') AS failed "
            # Finished rows are kept (with their content) and outnumber these by far;
            # naming the live statuses lets extraction_queue_status_idx skip them.
            "FROM extraction_queue WHERE status IN ('pending', 'processing', 'failed')"
        ).fetchone() or {"pending": 0, "processing": 0, "failed": 0}

        enq = conn.execute(
            "SELECT date_trunc('hour', enqueued_at) AS t, count(*) AS n FROM extraction_queue "
            "WHERE enqueued_at >= now() - %s::interval GROUP BY 1 ORDER BY 1",
            (interval,),
        ).fetchall()
        comp = conn.execute(
            "SELECT date_trunc('hour', processed_at) AS t, count(*) AS n FROM extraction_queue "
            "WHERE status = 'done' AND processed_at >= now() - %s::interval "
            "GROUP BY 1 ORDER BY 1",
            (interval,),
        ).fetchall()
        fails = conn.execute(
            "SELECT id, episode_id, error, enqueued_at, processed_at, attempts "
            "FROM extraction_queue WHERE status = 'failed' "
            "AND COALESCE(processed_at, enqueued_at) >= now() - %s::interval "
            "ORDER BY COALESCE(processed_at, enqueued_at) DESC, id DESC LIMIT %s",
            (interval, _INGEST_FAILURES_CAP),
        ).fetchall()

        last = conn.execute(
            "SELECT id, started_at, finished_at, stages, counts, samples, errors, ok, "
            "  EXTRACT(EPOCH FROM (finished_at - started_at)) AS duration_s "
            # A run that never finished (crashed mid-way) must not hide the last real one.
            "FROM dream_runs ORDER BY (finished_at IS NULL), started_at DESC, id DESC LIMIT 1"
        ).fetchone()

        return {
            "queue_depth": int(depth["pending"] or 0),
            "queue": {
                "pending": int(depth["pending"] or 0),
                "processing": int(depth["processing"] or 0),
                "failed": int(depth["failed"] or 0),
            },
            "throughput": {
                "enqueued_per_hour": [{"t": _iso(r["t"]), "n": int(r["n"])} for r in enq],
                "completed_per_hour": [{"t": _iso(r["t"]), "n": int(r["n"])} for r in comp],
            },
            "failures": [
                {
                    "id": r["id"],
                    "episode_id": r["episode_id"],
                    "error": (r["error"] or "")[:_SNIPPET_MAX],
                    "enqueued_at": _iso(r["enqueued_at"]),
                    "processed_at": _iso(r["processed_at"]),
                    "attempts": r["attempts"],
                }
                for r in fails
            ],
            "last_dream": _dream_run_json(last) if last else None,
        }


def _dream_run_json(r: dict[str, Any]) -> dict[str, Any]:
    """Shape a dream_runs row for the wire (used by the ingestion endpoint's last_dream)."""
    return {
        "id": r["id"],
        "started_at": _iso(r["started_at"]),
        "finished_at": _iso(r["finished_at"]),
        "duration_s": _r1(r["duration_s"]) if r.get("duration_s") is not None else None,
        "ok": r["ok"],
        "stages": r["stages"] or {},
        "counts": r["counts"] or {},
        "samples": r["samples"] or {},
        "errors": r["errors"] or [],
    }


def _spark_30d(conn: psycopg.Connection[dict[str, Any]], table: str, tcol: str) -> list[int]:
    """30 daily row counts (oldest→newest) off a table's own time column. table/tcol are from
    the trusted _CORPUS_TABLES constant, never user input (safe to interpolate)."""
    rows = conn.execute(
        f"SELECT (date_trunc('day', {tcol}))::date AS d, count(*) AS n FROM {table} "  # nosec B608
        f"WHERE {tcol} >= (now() - interval '30 days') GROUP BY 1",
        (),
    ).fetchall()
    by_day = {r["d"]: int(r["n"]) for r in rows}
    today = datetime.now(UTC).date()
    return [by_day.get(today - timedelta(days=29 - i), 0) for i in range(30)]


def _metrics_corpus(db_url: str) -> dict[str, Any]:
    """Per-table row counts (+ 30-day sparkline + 30d delta) and the episodes-by-project /
    by-source proportions. Whole response is cached 1h in the route (row counts are the cost).
    Headline counts are pg_class estimates for big tables, exact for small ones."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        tables = []
        for name, tcol in _CORPUS_TABLES:
            present = conn.execute("SELECT to_regclass(%s) AS r", (f"public.{name}",)).fetchone()
            if not present or present["r"] is None:
                continue
            est_row = conn.execute(
                "SELECT reltuples::bigint AS n FROM pg_class WHERE relname = %s AND relkind = 'r'",
                (name,),
            ).fetchone()
            est = est_row["n"] if est_row else None
            if est is None or est < _CORPUS_ESTIMATE_FLOOR:
                exact = conn.execute(f"SELECT count(*) AS c FROM {name}").fetchone()  # nosec B608
                rows_n = int(exact["c"]) if exact else 0
                estimated = False
            else:
                rows_n = int(est)
                estimated = True
            spark = _spark_30d(conn, name, tcol) if tcol else []
            tables.append(
                {
                    "name": name,
                    "rows": rows_n,
                    "rows_estimated": estimated,
                    "spark_30d": spark,
                    "delta_30d": sum(spark),
                }
            )

        by_project = [
            {"name": r["name"], "n": int(r["n"])}
            for r in conn.execute(
                "SELECT COALESCE(NULLIF(project, ''), 'untagged') AS name, count(*) AS n "
                "FROM episodes GROUP BY 1 ORDER BY n DESC, name LIMIT 12"
            ).fetchall()
        ]
        by_source = [
            {"name": r["name"], "n": int(r["n"])}
            for r in conn.execute(
                "SELECT COALESCE(NULLIF(source, ''), 'untagged') AS name, count(*) AS n "
                "FROM episodes GROUP BY 1 ORDER BY n DESC, name LIMIT 12"
            ).fetchall()
        ]
        return {"tables": tables, "by_project": by_project, "by_source": by_source}


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount recall history and the three metrics aggregates (admin gated)."""
    # Contract: corpus metrics are cached 1h in-process.
    corpus_cache = _TtlCache[dict[str, Any]](_CORPUS_TTL_S)

    @mcp.custom_route("/dash/api/recall/history", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    def dash_recall_history(request: Request) -> dict[str, Any]:
        limit = _limit(
            request.query_params.get("limit"), _RECALL_HISTORY_DEFAULT, _RECALL_HISTORY_MAX
        )
        return _recall_history(db_url, limit)

    @mcp.custom_route("/dash/api/metrics/recall", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    def dash_metrics_recall(request: Request) -> dict[str, Any]:
        window_s = _parse_window(request.query_params.get("window"), _RECALL_WINDOW_DEFAULT_S)
        return _metrics_recall(db_url, window_s)

    @mcp.custom_route("/dash/api/metrics/ingestion", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    def dash_metrics_ingestion(request: Request) -> dict[str, Any]:
        window_s = _parse_window(request.query_params.get("window"), _INGEST_WINDOW_DEFAULT_S)
        return _metrics_ingestion(db_url, window_s)

    @mcp.custom_route("/dash/api/metrics/corpus", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    def dash_metrics_corpus(request: Request) -> dict[str, Any]:
        return corpus_cache.get(lambda: _metrics_corpus(db_url))
