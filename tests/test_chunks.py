"""Unit tests for ingestion.chunks.rebuild_chunks — complete-window logic, no DB."""

from __future__ import annotations

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
    ) -> None:
        self.chunks.append(
            {
                "start": start_sequence,
                "end": end_sequence,
                "episode_ids": episode_ids,
                "content": content,
                "project": project,
            }
        )


def _eps(n: int) -> list[dict[str, Any]]:
    return [
        {"id": i, "sequence": i, "content": f"turn {i}", "project": "synapse"}
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
    eps = [{"id": i, "sequence": i, "content": "", "project": None} for i in range(1, 5)]
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
