"""parse_records is the seam shared by the disk sweep and the /ingest push.

The load-bearing invariant: feeding the SAME transcript records through
``parse_file`` (sweep) and ``parse_records`` (push) must yield byte-identical
episodes — same (session_id, sequence) and span_id — so the two paths converge
idempotently against ``upsert_episode``'s ON CONFLICT (session_id, sequence)
and the unique span_id index. If this drifts, a momentary-outage backfill would
double-insert instead of no-op'ing.
"""

from __future__ import annotations

import orjson

from ingestion.jsonl_client import JSONLParser


def _transcript() -> list[dict]:
    """Two-turn synthetic Claude Code transcript (raw JSONL record shape)."""
    sid = "11111111-2222-3333-4444-555555555555"
    return [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": sid,
            "cwd": "/home/user/services/synapse",
            "timestamp": "2026-05-26T12:00:00Z",
            "message": {"role": "user", "content": "why FalkorDB over Neo4j?"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": sid,
            "cwd": "/home/user/services/synapse",
            "timestamp": "2026-05-26T12:00:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Lightweight; fits the homelab box."}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": sid,
            "cwd": "/home/user/services/synapse",
            "timestamp": "2026-05-26T12:01:00Z",
            "message": {"role": "user", "content": "and the embedding dims?"},
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": sid,
            "cwd": "/home/user/services/synapse",
            "timestamp": "2026-05-26T12:01:03Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Voyage-4-large at 2048 dims."}],
            },
        },
    ]


def test_push_matches_sweep(tmp_path):
    """parse_records (push) == parse_file (sweep) on the same records."""
    records = _transcript()
    parser = JSONLParser()

    # sweep path: write the records to a .jsonl file and parse_file it
    f = tmp_path / "session.jsonl"
    f.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    sweep = parser.parse_file(f)

    # push path: hand the raw records straight to parse_records
    push = parser.parse_records(records, "hook", None)

    assert len(sweep) == len(push) > 0
    key = lambda eps: [(e.session_id, e.sequence, e.span_id) for e in eps]  # noqa: E731
    assert key(sweep) == key(push)
    assert [e.content for e in sweep] == [e.content for e in push]


def test_span_ids_stable_and_unique():
    """Each turn gets a span_id derived from its transcript UUID (idempotency key)."""
    eps = JSONLParser().parse_records(_transcript(), "hook", None)
    span_ids = [e.span_id for e in eps]
    assert all(s and s.startswith("jsonl:") for s in span_ids)
    assert len(span_ids) == len(set(span_ids))  # unique per turn


def test_repeated_uuids_dedupe_to_one_episode_each():
    """Compaction/resume physically repeats records (same uuid up to ~5x in one file).
    Each repeated turn must collapse to a single episode so span_ids stay unique and the
    /ingest push doesn't collide on the partial-unique span_id index (was a 500)."""
    records = _transcript()
    # Re-dump the whole transcript a second time, as Claude Code does on compaction:
    # identical records, identical uuids.
    doubled = records + [dict(r) for r in records]
    eps = JSONLParser().parse_records(doubled, "hook", None)
    once = JSONLParser().parse_records(records, "hook", None)
    # Doubling the records must not double the episodes.
    assert len(eps) == len(once) > 0
    span_ids = [e.span_id for e in eps]
    assert len(span_ids) == len(set(span_ids))  # no duplicate span_id survives


def test_project_override_wins():
    eps = JSONLParser().parse_records(_transcript(), "hook", project_override="synapse")
    assert eps and all(e.project == "synapse" for e in eps)


def test_unfiltered_records_are_filtered():
    """The push path passes raw records; sidechain/machinery must still be dropped."""
    records = _transcript()
    records.append(
        {
            "type": "assistant",
            "uuid": "side1",
            "sessionId": records[0]["sessionId"],
            "isSidechain": True,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "sidechain"}]},
        }
    )
    eps = JSONLParser().parse_records(records, "hook", None)
    # sidechain turn must not produce an episode
    assert all("sidechain" not in (e.content or "") for e in eps)
