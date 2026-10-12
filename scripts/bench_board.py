"""Before/after timing for the board build (mcp_server/board.py ``build_board``).

Loads ``build_board`` as it was at a git ref (default origin/main) and as it is in the
working tree, builds the board with each on the same database as a FULL-TRUST caller,
and reports the median wall and CPU time per build. Exits non-zero if any board's text,
note ids or overflow differ between the two. Read-only: the board is pure SELECTs.

    uv run python scripts/bench_board.py --db-url "$SYNAPSE_DB_URL"
    uv run python scripts/bench_board.py --db-url ... --ref origin/main --reps 20 --project synapse

With no ``--project``, it benchmarks the all-projects board plus every project on the
banner's most-recent list. The two versions alternate build by build, so drift in
database load lands on both. ``--pool`` adds a third column: the working tree's build
again with the server's connection pool (ingestion/pg_pool) open, so it borrows
connections instead of opening one per read. The before and after columns always run
with no pool open.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion import pg_pool  # noqa: E402
from ingestion.surfaces import FULL_TRUST  # noqa: E402
from mcp_server import board as work_tree_board  # noqa: E402


def board_at_ref(ref: str) -> types.ModuleType:
    """mcp_server/board.py as of ``ref``, imported against the working tree's other
    modules (only the board itself is swapped)."""
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:mcp_server/board.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    mod = types.ModuleType("board_at_ref")
    mod.__file__ = f"{ref}:mcp_server/board.py"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)  # our own repo's file, at a ref
    return mod


def median_ms(samples: list[tuple[float, float]]) -> str:
    walls, cpus = zip(*samples, strict=True)
    return f"{statistics.median(walls):7.1f}/{statistics.median(cpus):6.1f}"


def timed_build(mod: Any, db_url: str, project: str | None) -> tuple[dict[str, Any], float, float]:
    w0, c0 = time.perf_counter(), time.thread_time()
    out = mod.build_board(db_url, project, trust=FULL_TRUST)
    return out, (time.perf_counter() - w0) * 1000, (time.thread_time() - c0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--ref", default="origin/main", help="git ref for the 'before' board")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--project", action="append", help="repeatable; default: all + banner")
    ap.add_argument("--pool", action="store_true", help="also time the build with a pool open")
    args = ap.parse_args()

    before = board_at_ref(args.ref)
    if args.project:
        projects: list[str | None] = list(args.project)
    else:
        _, _, names = work_tree_board._banner_stats(args.db_url)
        projects = [None, *names]

    mismatches = 0
    labels = ["before", "after", *(["after+pool"] if args.pool else [])]
    print(f"before = {args.ref}, after = working tree, {args.reps} builds each, median ms")
    print(
        f"{'project':<24} {'notes':>5} {'drop':>5}"
        + "".join(f"  {lb + ' wall/cpu':>20}" for lb in labels)
    )
    for project in projects:
        stats: dict[str, list[tuple[float, float]]] = {lb: [] for lb in labels}
        outs: dict[str, dict[str, Any]] = {}
        for _ in range(args.reps):
            for label, mod in (("before", before), ("after", work_tree_board)):
                outs[label], wall, cpu = timed_build(mod, args.db_url, project)
                stats[label].append((wall, cpu))
        if args.pool:
            pg_pool.open_pool(args.db_url)
            try:
                pg_pool._POOLS[args.db_url].wait(timeout=10)
                for _ in range(args.reps):
                    outs["after+pool"], wall, cpu = timed_build(
                        work_tree_board, args.db_url, project
                    )
                    stats["after+pool"].append((wall, cpu))
            finally:
                pg_pool.close_pool(args.db_url)
        same = all(
            outs[lb][k] == outs["before"][k]
            for lb in labels
            for k in ("text", "note_ids", "overflow")
        )
        mismatches += not same

        after = outs["after"]
        print(
            f"{(project or '(all)')[:24]:<24} {after['n_notes'] + after['overflow']:>5} "
            f"{after['overflow']:>5}"
            + "".join(f"  {median_ms(stats[lb]):>20}" for lb in labels)
            + ("" if same else "  OUTPUT DIFFERS")
        )
    print("identical output on every board" if not mismatches else f"{mismatches} boards differ")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
