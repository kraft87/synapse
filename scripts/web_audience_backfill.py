#!/usr/bin/env python3
"""Propose, review, then apply audience labels for web research stored before schema 063.

Every pre-063 artifact is unlabelled (NULL), which a restricted surface is never served.
That is safe and useless: a work laptop gets no web research until something says which
pages are work-safe. This script is that something, in two deliberately separate passes,
the same shape as ``scripts/audience_backfill.py`` for notes.

  --propose   Run the precedence (ingestion/web_audience.py: own-surface rule, then the
              conservative classifier) over every unlabelled artifact that has content,
              and write a REVIEW FILE (TSV: id, kind, proposed, source, title, detail).
              Writes nothing to the database. Works before 063 is applied, so the labels
              can be measured before anything is stored.
  --apply F   Read the (reviewed) file back and set each artifact's audience, only where
              it is still NULL. A human edit of the ``proposed`` column wins.

The review file holds the title of every researched page, i.e. exactly the content this
feature keeps off other machines. It lands outside the repository by default
(``~/data/synapse/web-audience-review.tsv``) and MUST stay out: this repository is public.

``--propose`` needs the extraction LLM provider (``SYNAPSE_LLM_PROVIDER`` and friends, as
the poller has them); ``--apply`` needs only the database. After 063 is applied, the
poller labels new artifacts by itself (ingestion/poller.py), so this is a one-off.

Usage:
    scripts/web_audience_backfill.py --propose [--out PATH] [--ids FILE] [--workers N]
    scripts/web_audience_backfill.py --apply PATH [--dry-run]

The DSN comes from --db-url, else $SYNAPSE_DB_URL.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.surfaces import AUDIENCES, WEB_AUDIENCE_PROBE
from ingestion.web_audience import (
    SERVED_KINDS,
    AudienceDecision,
    classify_row,
    derive_web_audience,
)

DEFAULT_OUT = Path("~/data/synapse/web-audience-review.tsv").expanduser()
COLUMNS = ("id", "kind", "proposed", "source", "title", "detail")

_CANDIDATES_SQL = """
SELECT id, kind, title, url, session_id, content_markdown
FROM web_artifacts
WHERE kind = ANY(%s) AND content_markdown IS NOT NULL {unlabelled} {only_ids}
ORDER BY id
"""

_APPLY_SQL = "UPDATE web_artifacts SET audience = %s WHERE id = %s AND audience IS NULL"


def _cell(v: Any) -> str:
    """One TSV cell: tabs and newlines would break the row, so they become spaces."""
    return " ".join(str(v if v is not None else "").split())


def candidates(conn: Any, ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Artifacts to label: servable kinds with content, still unlabelled (every row, before
    063 exists), optionally restricted to ``ids``."""
    labelled_col = WEB_AUDIENCE_PROBE.ready(conn, "backfill")
    sql = _CANDIDATES_SQL.format(
        unlabelled="AND audience IS NULL" if labelled_col else "",
        only_ids="AND id = ANY(%s)" if ids else "",
    )
    params: list[Any] = [list(SERVED_KINDS)]
    if ids:
        params.append(ids)
    return list(conn.cursor(row_factory=dict_row).execute(sql, params))


def propose(
    conn: Any,
    make_llm: Any,
    out: Any,
    *,
    ids: list[int] | None = None,
    workers: int = 1,
    model: str | None = None,
) -> dict[str, int]:
    """Decide every candidate and write the review rows to ``out``. Rule 2 (own surface)
    is read on ``conn``; the classifier runs in ``workers`` threads, one LLM client each.
    A transport error on a row is recorded as ``error`` (left unlabelled by --apply)."""
    rows = candidates(conn, ids)
    ruled = {
        r["id"]: derive_web_audience(conn, explicit=None, session_id=r.get("session_id"))
        for r in rows
    }
    local = threading.local()

    def _one(r: dict[str, Any]) -> AudienceDecision:
        d = ruled[r["id"]]
        if d is not None:
            return d
        if not hasattr(local, "llm"):
            local.llm = make_llm()
        try:
            return classify_row(local.llm, r, model=model)
        except Exception as e:  # transport: no label, retried by a later run
            return AudienceDecision("error", "transport", str(e)[:200])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        decisions = list(ex.map(_one, rows))
    counts: dict[str, int] = {}
    out.write("\t".join(COLUMNS) + "\n")
    for r, d in zip(rows, decisions, strict=True):
        out.write(
            "\t".join(
                _cell(v) for v in (r["id"], r["kind"], d.audience, d.source, r["title"], d.detail)
            )
            + "\n"
        )
        counts[d.audience] = counts.get(d.audience, 0) + 1
    out.flush()
    return counts


def read_review(path: Path) -> list[tuple[int, str]]:
    """``(id, audience)`` for every row whose ``proposed`` is a real audience. Rows marked
    ``error`` (or anything else) are skipped, so they stay unlabelled."""
    pairs: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t")[: len(COLUMNS)] != list(COLUMNS):
        raise SystemExit(f"{path}: not a review file (header must be {COLUMNS})")
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        try:
            aid = int(cells[0])
        except (ValueError, IndexError):
            raise SystemExit(f"{path}:{n}: bad id {cells[:1]}") from None
        audience = cells[2].strip() if len(cells) > 2 else ""
        if audience in AUDIENCES:
            pairs.append((aid, audience))
    return pairs


def apply(conn: Any, pairs: list[tuple[int, str]], *, dry_run: bool = False) -> int:
    """Write each label where the row is still NULL. Returns the rows changed."""
    if not WEB_AUDIENCE_PROBE.ready(conn, "backfill"):
        raise SystemExit("web_artifacts.audience is missing: apply schema/063_web_audience.sql")
    changed = 0
    with conn.transaction():
        for aid, audience in pairs:
            changed += conn.execute(_APPLY_SQL, (audience, aid)).rowcount
        if dry_run:
            raise _Rollback(changed)
    return changed


class _Rollback(Exception):
    def __init__(self, changed: int) -> None:
        super().__init__(changed)
        self.changed = changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propose", action="store_true", help="write a review file, no DB writes")
    mode.add_argument("--apply", metavar="PATH", help="apply a reviewed file")
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT), help="review file for --propose ('-' = stdout)"
    )
    ap.add_argument("--ids", help="file of artifact ids to restrict --propose to (one per line)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent classifier calls")
    ap.add_argument("--dry-run", action="store_true", help="--apply: count, then roll back")
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    args = ap.parse_args(argv)
    if not args.db_url:
        print("error: no DSN (--db-url or SYNAPSE_DB_URL)", file=sys.stderr)
        return 2

    with psycopg.connect(args.db_url, autocommit=True) as conn:
        if args.propose:
            from ingestion.llm_client import create_llm_client

            ids = None
            if args.ids:
                ids = [int(x) for x in Path(args.ids).read_text().split() if x.strip()]
            if args.out == "-":
                counts = propose(conn, create_llm_client, sys.stdout, ids=ids, workers=args.workers)
            else:
                out_path = Path(args.out).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    counts = propose(conn, create_llm_client, f, ids=ids, workers=args.workers)
                print(f"wrote {out_path}", file=sys.stderr)
            print(f"proposed: {counts}", file=sys.stderr)
            return 0
        pairs = read_review(Path(args.apply).expanduser())
        try:
            changed = apply(conn, pairs, dry_run=args.dry_run)
        except _Rollback as rb:
            print(f"dry run: would label {rb.changed} of {len(pairs)} rows", file=sys.stderr)
            return 0
        print(f"labelled {changed} of {len(pairs)} rows", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
