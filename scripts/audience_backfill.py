#!/usr/bin/env python3
"""Propose, review, then apply audience tags for the notes written before schema 053.

Every pre-053 note is ``personal`` — the fail-closed default. That is safe and useless:
a restricted board runs empty until someone says which notes are work-safe. This script
is that someone's tooling, in two deliberately separate passes.

  --propose   Read the live notes, guess an audience for each, and write a REVIEW FILE
              (a markdown table) to a path OUTSIDE the repo. Writes nothing to the DB.
  --apply F   Read the (hand-edited) review file back and set each note's audience.

The split is the point. The heuristic — ``type='project'`` on a project some restricted
surface can already read ⇒ work-safe, everything else personal — is a starting draft,
not a classifier. Misclassifying one note leaks it to every work session from then on,
so a human reads the table and edits the column before anything is written. Editing the
file IS the review; there is no separate approval step.

The review file lands outside the repo by default (``~/data/synapse/audience-review.md``)
and MUST stay there. It contains every note hook in the store — the exact content this
whole feature exists to keep off other machines — and this repository is public.

Usage:
    scripts/audience_backfill.py --propose [--out PATH] [--db-url DSN]
    scripts/audience_backfill.py --apply PATH [--dry-run] [--db-url DSN]

The DSN comes from --db-url, else $SYNAPSE_DB_URL. Run --propose, open the file, fix the
`audience` column where the guess is wrong, save, run --apply.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.db import Database
from ingestion.notes import _OWNER
from ingestion.surfaces import (
    AUDIENCES,
    DEFAULT_AUDIENCE,
    RESTRICTED_AUDIENCE,
    restricted_project_union,
)

DEFAULT_OUT = Path("~/data/synapse/audience-review.md").expanduser()

_HEADER = """# Audience review — Synapse notes (schema 053)

Edit the `audience` column: `personal` (default, never leaves a trusted host) or
`work-safe` (served on restricted surfaces). Leave the id and hook columns alone.
Then: `scripts/audience_backfill.py --apply {path}`

Rows below are the LIVE note set at generation time. A note added after this file was
generated is not listed and stays `personal` until the next pass.

DO NOT COMMIT THIS FILE — the Synapse repository is public and these hooks are the
content audience scoping exists to protect.

| id | audience | type | project | hook |
| --- | --- | --- | --- | --- |
"""

# A table row: | 12 | work-safe | project | alpha | Some hook text |
_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([a-z-]+)\s*\|")


def _cell(value: Any) -> str:
    """Markdown-table-safe cell: pipes and newlines would break the row grammar."""
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def _propose_audience(note: dict[str, Any], work_projects: set[str]) -> str:
    """The heuristic draft (spec §rollout 3): a project note filed under a project some
    restricted surface can already read is work-safe; everything else is personal.

    Deliberately conservative. Global user/feedback notes stay personal even when they
    read as technical — 'User prefers tabs' is harmless, 'User is job hunting' is not,
    and nothing in the row distinguishes them. That is exactly the call a human makes
    while reading the table."""
    if note.get("type") == "project" and (note.get("project") or "") in work_projects:
        return RESTRICTED_AUDIENCE
    return DEFAULT_AUDIENCE


def _live_notes(db: Database) -> list[dict[str, Any]]:
    """Every live note, oldest first — including project notes for EVERY project, which
    list_board_notes deliberately won't serve (it scopes to one project at a time)."""
    with db._conn() as conn:
        rows = conn.execute(
            "SELECT id, hook, type, project, audience FROM notes "
            "WHERE owner_id = %s AND superseded_by IS NULL ORDER BY id",
            (_OWNER,),
        ).fetchall()
    return [dict(r) for r in rows]


def propose(db_url: str, out_path: Path) -> int:
    db = Database(db_url)
    try:
        notes = _live_notes(db)
        work_projects = restricted_project_union(db)
    finally:
        db.close()

    if not work_projects:
        print(
            "note: no restricted surface has any allowed_projects yet, so the heuristic "
            "proposes 'personal' for everything. Register the work surface first "
            "(PUT /surfaces/<id>) for a more useful draft.",
            file=sys.stderr,
        )

    lines = [_HEADER.format(path=out_path)]
    n_work = 0
    for n in notes:
        proposed = _propose_audience(n, work_projects)
        n_work += proposed == RESTRICTED_AUDIENCE
        lines.append(
            f"| {n['id']} | {proposed} | {_cell(n['type'])} | {_cell(n['project'])} "
            f"| {_cell(n['hook'])} |\n"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(notes)} notes to {out_path} ({n_work} proposed work-safe)")
    print("review and edit the audience column, then re-run with --apply")
    return 0


def parse_review(text: str) -> list[tuple[int, str]]:
    """(note_id, audience) per table row. Raises ValueError on an unusable audience —
    a typo'd tier must stop the run, never silently fall back to a default."""
    out: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        note_id, audience = int(m.group(1)), m.group(2)
        if audience not in AUDIENCES:
            raise ValueError(f"note {note_id}: invalid audience {audience!r} (row: {line.strip()})")
        out.append((note_id, audience))
    return out


def apply(db_url: str, path: Path, dry_run: bool) -> int:
    rows = parse_review(path.read_text(encoding="utf-8"))
    if not rows:
        print(f"no table rows found in {path}", file=sys.stderr)
        return 1
    db = Database(db_url)
    try:
        current = {n["id"]: n["audience"] for n in _live_notes(db)}
        changes = [(i, a) for i, a in rows if i in current and current[i] != a]
        missing = [i for i, _ in rows if i not in current]
        for note_id in missing:
            print(f"skipping note {note_id}: not in the live set (superseded or deleted)")
        if dry_run:
            for note_id, audience in changes:
                print(f"would set n:{note_id} {current[note_id]} -> {audience}")
            print(f"dry run: {len(changes)} of {len(rows)} rows would change")
            return 0
        with db._conn() as conn:
            for note_id, audience in changes:
                # updated_at is NOT bumped: classifying a note does not restate it, and a
                # bump would re-queue it for the dream lane's judges for no reason.
                conn.execute("UPDATE notes SET audience = %s WHERE id = %s", (audience, note_id))
    finally:
        db.close()
    print(f"applied {len(changes)} audience change(s) from {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propose", action="store_true", help="write the review file")
    mode.add_argument("--apply", metavar="PATH", help="apply a reviewed file")
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"--propose target (default {DEFAULT_OUT})"
    )
    p.add_argument("--dry-run", action="store_true", help="--apply: print changes, write nothing")
    p.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    args = p.parse_args(argv)

    if not args.db_url:
        print("no database URL — pass --db-url or set SYNAPSE_DB_URL", file=sys.stderr)
        return 2
    if args.propose:
        return propose(args.db_url, args.out.expanduser())
    return apply(args.db_url, Path(args.apply).expanduser(), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
