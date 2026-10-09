#!/usr/bin/env python3
"""One-time repair: redact credentials already stored in memory.

Ingest redacts since the Episode validator (ingestion.secret_redact) landed; rows
written before it still hold whatever was pasted into a session, and recall serves
them back verbatim. This applies the same ``redact_secrets`` to every stored text
column that recall, the board or extraction reads.

Usage:
    python scripts/redact_stored_secrets.py [DSN]           # dry-run report
    python scripts/redact_stored_secrets.py [DSN] --apply   # rewrite in one transaction

DSN defaults to $SYNAPSE_DB_URL. Idempotent: a redacted row has nothing left to match.
The report is counts only; it never prints a matched value, because the point of the
script is to get those values out of places they can be read.

Not covered: embeddings computed from the unredacted text stay as they are (a vector
does not give the key back), and database backups taken before the run still hold the
original rows.
"""

from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.secret_redact import redact_secrets

#: table -> text columns. Every table keys on ``id``.
TARGETS: dict[str, tuple[str, ...]] = {
    "episodes": ("content", "human_turn", "assistant_turn"),
    "episodes_quarantine": ("content", "human_turn", "assistant_turn"),
    "chunks": ("content",),
    "extraction_queue": ("content",),
    "synth_documents": ("content",),
    "kg_relationships": ("fact",),
    "kg_entities": ("summary",),
    "notes": ("hook", "body"),
    "timeline_events": ("fact",),
}

#: Cheap server-side prefilter: every redaction rule needs one of these substrings, so a
#: row matching none of them cannot change. Keeps the scan from shipping the whole
#: corpus over the wire.
_PREFILTER = (
    r"sk-|[rs]k_(live|test)_|gh[pousr]_|github_pat_|glpat-|hf_|xox[a-z]-|AKIA|ASIA|AIza|eyJ"
    r"|PRIVATE KEY|bearer|basic|://|password|passwd|passphrase|secret|token|api.?key"
    r"|access.?key|credential|[MN][A-Za-z0-9_-]{23,27}\."
)
_MARK = re.compile(r"\[REDACTED:([a-z-]+)\]")


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()[0]  # type: ignore[index]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    dsn = args[0] if args else os.environ.get("SYNAPSE_DB_URL", "")
    if not dsn:
        sys.exit("usage: redact_stored_secrets.py <DSN> [--apply]  (or set SYNAPSE_DB_URL)")

    with psycopg.connect(dsn) as conn:
        total = 0
        for table, cols in TARGETS.items():
            if not _table_exists(conn, table):
                print(f"{table}: absent, skipped")
                continue
            where = " OR ".join(f"{c} ~* %(pre)s" for c in cols)
            rows = conn.execute(
                f"SELECT id, {', '.join(cols)} FROM {table} WHERE {where}",  # nosec B608 — table/cols are literals from TARGETS
                {"pre": _PREFILTER},
            ).fetchall()
            kinds: collections.Counter[str] = collections.Counter()
            updates = []
            for row_id, *values in rows:
                new = [redact_secrets(v) for v in values]
                if new == values:
                    continue
                for old_v, new_v in zip(values, new, strict=True):
                    if old_v != new_v:
                        kinds.update(_MARK.findall(new_v or ""))
                        kinds.subtract(_MARK.findall(old_v or ""))
                updates.append((*new, row_id))
            kinds = +kinds
            print(f"{table}: {len(rows)} candidate row(s), {len(updates)} to redact {dict(kinds)}")
            if apply and updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"UPDATE {table} SET {', '.join(f'{c} = %s' for c in cols)} WHERE id = %s",  # nosec B608 — literals from TARGETS
                        updates,
                    )
            total += len(updates)
        if not apply:
            conn.rollback()
            print(f"dry run: {total} row(s) would change — pass --apply to rewrite them")
            return
        conn.commit()
        print(f"redacted {total} row(s)")


if __name__ == "__main__":
    main()
