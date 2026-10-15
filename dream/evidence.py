"""What the skills ledger and the config lane's ledger both do with a sighting: compare two
token sets, and fold new evidence entries into a row's JSONB list."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

EVIDENCE_CAP = 60  # bound the JSONB


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Overlap of two token sets; two empty sets are identical (1.0)."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def union_evidence(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append new, dedup by (session_id, signal, class, work date): a scanner seeing the same
    work again (a re-run after a held cursor, the miner's overlapping window) adds nothing,
    while the same session on another day is a fresh sighting. Entries without a date
    (gap_scan, explicit requests, legacy rows) key on scan_night instead, so a later night's
    sighting still counts; entries with neither all key to one bucket (the config lane's, which
    carry only a session, so it counts a session once).

    The cap drops the oldest judge entries. A grounded entry is a human decision or a user
    signal (an accept is what lets a row stand as promoted: grounded_required_to_apply) and
    is never dropped."""
    out, seen = [], set()
    for e in list(old) + list(new):
        k = (
            e.get("session_id"),
            e.get("signal"),
            e.get("class"),
            e.get("date") or e.get("scan_night"),
        )
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    excess = len(out) - EVIDENCE_CAP
    if excess > 0:
        oldest = {id(e) for e in [e for e in out if e.get("class") != "grounded"][:excess]}
        out = [e for e in out if id(e) not in oldest]
    return out
