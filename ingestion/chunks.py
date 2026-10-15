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
from datetime import UTC, datetime, timedelta
from typing import Any

from ingestion.db import Database

_CHUNK_WINDOW = 4
_CHUNK_STEP = 3
#: A session this long without a new turn is over: its tail is chunked as it stands.
#: ``Database.sessions_with_pending_chunks`` mirrors the 2 hours in SQL.
_IDLE_AFTER = timedelta(hours=2)


def rebuild_chunks(
    db: Database,
    session_id: str,
    on_new: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Build any missing chunk windows for one session. Idempotent.

    Returns the number of NEW chunks inserted. A live session gets only full
    ``_CHUNK_WINDOW`` windows: a complete window never grows, so it is written once and
    never superseded. A *partial* trailing window, by contrast, would be re-emitted at a
    longer ``end_sequence`` every time the session gains an episode, leaving stale
    same-start prefix chunks (the bug Phase 1 shipped). The trailing <window episodes
    wait until enough arrive to form a full window; they stay covered by the episode
    layer meanwhile. Existing windows are skipped — no wasted upsert, no re-embedding.

    A session idle for ``_IDLE_AFTER`` is over, so waiting any longer would leave its tail
    out of KG extraction for good (every turn of a session shorter than a window, the
    last 1-3 turns of the rest, where the conclusion usually is). It gets one last window
    from the next grid position, whose first turn is the overlap turn a regular window
    would start from. That window is written once; if the session resumes, the regular
    windows continue and the small overlap is absorbed by the duplicate checks.

    ``on_new`` (optional) is invoked once per genuinely-new chunk with
    ``{chunk_id, session_id, start_sequence, end_sequence, episode_ids, content, project,
    tail}`` (``tail``: it is a flushed tail, catch-up work rather than live ingest).
    The live poller uses it to enqueue the chunk for KG fact extraction (task #63)
    exactly once, at birth — so each chunk is extracted once and never re-enqueued.
    Backfills pass ``None`` (they bulk-enqueue separately).
    """
    eps = db.get_session_episodes(session_id)
    if not eps:
        return 0
    existing = db.get_chunk_ranges(session_id)
    project = eps[0].get("project")
    # Complete windows only: i + _CHUNK_WINDOW <= n (range stop is exclusive).
    windows = [
        eps[i : i + _CHUNK_WINDOW] for i in range(0, len(eps) - _CHUNK_WINDOW + 1, _CHUNK_STEP)
    ]
    full = len(windows)
    if datetime.now(UTC) - max(ep["created_at"] for ep in eps) > _IDLE_AFTER:
        tail = eps[len(windows) * _CHUNK_STEP :]
        # With a window before it the tail's first turn is the overlap, already covered;
        # a remember() archive among the rest is no reason for a chunk (see below).
        fresh = tail[1:] if windows else tail
        if any(ep.get("source") != "manual" for ep in fresh):
            windows.append(tail)
    built = 0
    for n, window in enumerate(windows):
        start_seq = window[0]["sequence"]
        end_seq = window[-1]["sequence"]
        if (start_seq, end_seq) in existing:
            continue
        # A remember() archive episode is already a note, and the turn that prompted it is
        # in the window anyway: chunking it would extract the same statement again.
        turns = [ep for ep in window if ep.get("source") != "manual"]
        content = "\n\n---\n\n".join(ep["content"] for ep in turns if ep.get("content"))
        if not content.strip():
            continue
        episode_ids = [ep["id"] for ep in turns]
        chunk_id = db.upsert_chunk(
            session_id=session_id,
            start_sequence=start_seq,
            end_sequence=end_seq,
            episode_ids=episode_ids,
            content=content,
            project=project,
        )
        if chunk_id is None:
            continue  # another worker wrote this window first, and enqueues it
        built += 1
        if on_new is not None:
            on_new(
                {
                    "chunk_id": chunk_id,
                    "session_id": session_id,
                    "start_sequence": start_seq,
                    "end_sequence": end_seq,
                    "episode_ids": episode_ids,
                    "content": content,
                    "project": project,
                    "tail": n >= full,
                }
            )
    return built
