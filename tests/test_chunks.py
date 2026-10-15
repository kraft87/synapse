"""Unit tests for ingestion.chunks.rebuild_chunks — complete-window logic, no DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ingestion.chunks import rebuild_chunks


class _FakeDB:
    """Records upsert_chunk calls; serves a fixed episode list + existing ranges."""

    def __init__(
        self, eps: list[dict[str, Any]], existing: set[tuple[int, int]] | None = None
    ) -> None:
        self._eps = eps
        self._existing = set(existing or set())
        self.chunks: list[dict[str, Any]] = []

    def get_session_episodes(self, session_id: str) -> list[dict[str, Any]]:
        return self._eps

    def get_chunk_ranges(self, session_id: str) -> set[tuple[int, int]]:
        return set(self._existing)

    def upsert_chunk(
        self,
        *,
        session_id: str,
        start_sequence: int,
        end_sequence: int,
        episode_ids: list[int],
        content: str,
        project: str | None,
    ) -> int | None:
        self.chunks.append(
            {
                "start": start_sequence,
                "end": end_sequence,
                "episode_ids": episode_ids,
                "content": content,
                "project": project,
            }
        )
        return 100 + len(self.chunks)


def _eps(n: int, idle: bool = False) -> list[dict[str, Any]]:
    """``n`` turns; the last one was ``idle`` (3 h ago) or just now."""
    last = datetime.now(UTC) - (timedelta(hours=3) if idle else timedelta(minutes=1))
    return [
        {
            "id": i,
            "sequence": i,
            "content": f"turn {i}",
            "project": "synapse",
            "created_at": last - timedelta(minutes=n - i),
        }
        for i in range(1, n + 1)
    ]


def test_complete_windows_only_no_partials() -> None:
    db = _FakeDB(_eps(10))
    built = rebuild_chunks(db, "s")  # type: ignore[arg-type]
    # window=4, step=3 -> complete windows at positions 0,3,6 (no grid-9 partial)
    assert built == len(db.chunks) == 3
    assert [(c["start"], c["end"]) for c in db.chunks] == [(1, 4), (4, 7), (7, 10)]
    # every chunk is a full 4-episode window — no partials to later be superseded
    assert all(len(c["episode_ids"]) == 4 for c in db.chunks)
    # adjacent windows overlap by exactly one episode
    assert db.chunks[0]["end"] == db.chunks[1]["start"] == 4
    assert db.chunks[0]["project"] == "synapse"


def test_too_short_for_a_window() -> None:
    # fewer than a full window -> nothing (no partial chunk)
    for n in (1, 2, 3):
        db = _FakeDB(_eps(n))
        assert rebuild_chunks(db, "s") == 0  # type: ignore[arg-type]
    # exactly one window
    db = _FakeDB(_eps(4))
    assert rebuild_chunks(db, "s") == 1  # type: ignore[arg-type]
    assert (db.chunks[0]["start"], db.chunks[0]["end"]) == (1, 4)


def test_trailing_incomplete_window_waits() -> None:
    # 11 episodes: complete windows cover up to ep 10; ep 11 waits for a full window
    db = _FakeDB(_eps(11))
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert [(c["start"], c["end"]) for c in db.chunks] == [(1, 4), (4, 7), (7, 10)]


def test_skips_existing_windows() -> None:
    # (1,4) and (4,7) already exist -> only (7,10) is inserted
    db = _FakeDB(_eps(10), existing={(1, 4), (4, 7)})
    built = rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert built == 1
    assert [(c["start"], c["end"]) for c in db.chunks] == [(7, 10)]


def test_blank_content_skipped() -> None:
    eps = [{**ep, "content": "", "project": None} for ep in _eps(4)]
    db = _FakeDB(eps)
    assert rebuild_chunks(db, "s") == 0  # type: ignore[arg-type]


def test_on_new_fires_once_per_new_chunk() -> None:
    # on_new is the live poller's enqueue hook (task #63): fires once per genuinely-new
    # complete window, carrying the chunk's identity + source episode_ids for the backlink.
    db = _FakeDB(_eps(10))
    seen: list[tuple[int, int]] = []
    payloads: list[dict[str, object]] = []

    def on_new(chunk: dict[str, object]) -> None:
        seen.append((chunk["start_sequence"], chunk["end_sequence"]))  # type: ignore[arg-type]
        payloads.append(chunk)

    built = rebuild_chunks(db, "s", on_new=on_new)  # type: ignore[arg-type]
    assert built == 3
    assert seen == [(1, 4), (4, 7), (7, 10)]
    # payload carries the source episode_ids (full window) + project for enqueue
    assert payloads[0]["episode_ids"] == [1, 2, 3, 4]
    assert payloads[0]["project"] == "synapse"
    assert payloads[0]["session_id"] == "s"
    assert "---" in payloads[0]["content"]  # type: ignore[operator]


def test_on_new_skips_existing_windows() -> None:
    # Existing chunks must NOT re-fire on_new — each chunk is enqueued exactly once, at birth.
    db = _FakeDB(_eps(10), existing={(1, 4), (4, 7)})
    fired: list[tuple[int, int]] = []
    rebuild_chunks(
        db,  # type: ignore[arg-type]
        "s",
        on_new=lambda c: fired.append((c["start_sequence"], c["end_sequence"])),
    )
    assert fired == [(7, 10)]


def test_remember_archive_episodes_stay_out_of_chunk_text() -> None:
    """A remember() archive is already a note, and the turn that prompted it is in the
    window: chunking it too would extract the same statement twice. The window itself
    (its sequence range) is unchanged, so existing chunks still line up."""
    eps = _eps(4)
    eps[2]["source"] = "manual"
    db = _FakeDB(eps)
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    (chunk,) = db.chunks
    assert (chunk["start"], chunk["end"]) == (1, 4)
    assert chunk["episode_ids"] == [1, 2, 4]
    assert "turn 3" not in chunk["content"] and "turn 4" in chunk["content"]


def test_on_new_carries_the_chunk_id() -> None:
    # The queue row names the chunk it extracts, so the worker reads it by key.
    db = _FakeDB(_eps(7))
    ids: list[object] = []
    rebuild_chunks(db, "s", on_new=lambda c: ids.append(c["chunk_id"]))  # type: ignore[arg-type]
    assert ids == [101, 102]


def test_a_chunk_that_already_exists_is_not_announced() -> None:
    # upsert_chunk answers None for a window another worker wrote first: that worker
    # enqueues it, and this call does not count it.
    class _Lost(_FakeDB):
        def upsert_chunk(self, **kw: Any) -> int | None:
            super().upsert_chunk(**kw)
            return None

    db = _Lost(_eps(4))
    fired: list[object] = []
    assert rebuild_chunks(db, "s", on_new=fired.append) == 0  # type: ignore[arg-type]
    assert fired == []


# --- the tail of a session that has gone quiet -----------------------------------------
#
# Complete windows only leave the last 1-3 turns of most sessions, and every turn of a
# session shorter than a window, without a chunk, so no KG fact is ever extracted from
# them. Once a session has been idle long enough it is over, and its tail is chunked once.


def _spans(db: _FakeDB) -> list[tuple[int, int]]:
    return [(c["start"], c["end"]) for c in db.chunks]


def test_a_short_idle_session_is_chunked_whole() -> None:
    for n in (1, 2, 3):
        db = _FakeDB(_eps(n, idle=True))
        assert rebuild_chunks(db, "s") == 1  # type: ignore[arg-type]
        assert _spans(db) == [(1, n)]


def test_an_idle_tail_becomes_one_last_window_starting_at_the_overlap_turn() -> None:
    # n=5: (1,4) is complete; turn 5 is the tail, with turn 4 (the overlap a regular
    # next window would start from) as its context.
    db = _FakeDB(_eps(5, idle=True))
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert _spans(db) == [(1, 4), (4, 5)]
    # n=9: complete windows end at 7; turns 8 and 9 are the tail
    db = _FakeDB(_eps(9, idle=True))
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert _spans(db) == [(1, 4), (4, 7), (7, 9)]


def test_an_idle_session_with_no_tail_adds_nothing() -> None:
    # n=7: the last complete window ends on the last turn
    db = _FakeDB(_eps(7, idle=True))
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert _spans(db) == [(1, 4), (4, 7)]


def test_a_live_session_keeps_its_tail_for_the_next_window() -> None:
    db = _FakeDB(_eps(5))
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert _spans(db) == [(1, 4)]


def test_the_flushed_tail_is_written_once() -> None:
    db = _FakeDB(_eps(5, idle=True), existing={(1, 4), (4, 5)})
    assert rebuild_chunks(db, "s") == 0  # type: ignore[arg-type]
    assert db.chunks == []


def test_a_tail_of_remember_archives_alone_is_not_chunked() -> None:
    # The archive episode is a note already; a window of nothing else has no text.
    eps = _eps(2, idle=True)
    for ep in eps:
        ep["source"] = "manual"
    db = _FakeDB(eps)
    assert rebuild_chunks(db, "s") == 0  # type: ignore[arg-type]


def test_an_overlap_turn_and_a_remember_archive_are_not_a_new_chunk() -> None:
    # n=5, the last turn an archive: the tail is turn 4 (already in (1,4)) + the archive.
    eps = _eps(5, idle=True)
    eps[4]["source"] = "manual"
    db = _FakeDB(eps)
    rebuild_chunks(db, "s")  # type: ignore[arg-type]
    assert _spans(db) == [(1, 4)]


def test_a_flushed_tail_is_marked_so_the_queue_can_put_it_behind_live_ingest() -> None:
    db = _FakeDB(_eps(9, idle=True))
    flags: list[tuple[int, bool]] = []
    rebuild_chunks(db, "s", on_new=lambda c: flags.append((c["end_sequence"], c["tail"])))  # type: ignore[arg-type]
    assert flags == [(4, False), (7, False), (9, True)]
