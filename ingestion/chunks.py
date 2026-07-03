"""Sliding-window chunk construction — shared by the live poller maintenance
loop and the backfill importers.

A chunk is ``_CHUNK_WINDOW`` consecutive episodes, advanced ``_CHUNK_STEP`` at a
time, so adjacent chunks overlap by ``_CHUNK_WINDOW - _CHUNK_STEP`` episodes.
Chunks are the verbatim retrieval layer between single episodes and the
(paraphrased, coarser) segment summaries.

``Database.upsert_chunk`` is idempotent on ``(session_id, start_sequence,
end_sequence)`` (``ON CONFLICT DO NOTHING``), so rebuilding a whole session is
safe: only genuinely-new windows are inserted, and the trailing window that grew
as new episodes arrived re-upserts under a new ``end_sequence``.

This logic used to live (duplicated) in each backfill importer and, for live
ingestion, in the now-removed Logfire poll path. It is consolidated here so the
poller maintenance loop and every backfill build chunks identically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ingestion.db import Database

_CHUNK_WINDOW = 4
_CHUNK_STEP = 3


def rebuild_chunks(
    db: Database,
    session_id: str,
    on_new: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Build any missing COMPLETE chunk windows for one session. Idempotent.

    Returns the number of NEW chunks inserted. Only full ``_CHUNK_WINDOW`` windows
    are emitted: a complete window never grows, so it is written once and never
    superseded. A *partial* trailing window, by contrast, would be re-emitted at a
    longer ``end_sequence`` every time the session gains an episode, leaving stale
    same-start prefix chunks (the bug Phase 1 shipped). The trailing <window
    episodes wait until enough arrive to form a full window; they stay covered by
    the episode layer meanwhile. Existing windows are skipped — no wasted upsert,
    no re-embedding.

    ``on_new`` (optional) is invoked once per genuinely-new chunk with
    ``{session_id, start_sequence, end_sequence, episode_ids, content, project}``.
    The live poller uses it to enqueue the chunk for KG fact extraction (task #63)
    exactly once, at birth — so each chunk is extracted once and never re-enqueued.
    Backfills pass ``None`` (they bulk-enqueue separately).
    """
    eps = db.get_session_episodes(session_id)
    n = len(eps)
    if n < _CHUNK_WINDOW:
        return 0
    existing = db.get_chunk_ranges(session_id)
    project = eps[0].get("project")
    built = 0
    # Complete windows only: i + _CHUNK_WINDOW <= n (range stop is exclusive).
    for i in range(0, n - _CHUNK_WINDOW + 1, _CHUNK_STEP):
        window = eps[i : i + _CHUNK_WINDOW]
        start_seq = window[0]["sequence"]
        end_seq = window[-1]["sequence"]
        if (start_seq, end_seq) in existing:
            continue
        content = "\n\n---\n\n".join(ep["content"] for ep in window if ep.get("content"))
        if not content.strip():
            continue
        episode_ids = [ep["id"] for ep in window]
        db.upsert_chunk(
            session_id=session_id,
            start_sequence=start_seq,
            end_sequence=end_seq,
            episode_ids=episode_ids,
            content=content,
            project=project,
        )
        built += 1
        if on_new is not None:
            on_new(
                {
                    "session_id": session_id,
                    "start_sequence": start_seq,
                    "end_sequence": end_seq,
                    "episode_ids": episode_ids,
                    "content": content,
                    "project": project,
                }
            )
    return built
