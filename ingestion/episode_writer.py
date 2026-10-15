"""The one place a parsed turn becomes a stored episode.

``POST /ingest`` (the Stop hook's push) and every backfill (the Claude Code disk sweep and
the Codex, Cursor, claude.ai and ChatGPT importers) end here, so a turn meets the same guards
whichever way it arrives. Dropping a turn here means chunks, KG extraction, dream and the
timeline never see it; nothing downstream needs to know why.

Guards, in order, for each turn:

* already stored (its span_id is in the session's span index) or no span_id to key it by:
  skipped. A push ships a bounded tail whose positional sequence is meaningless, so identity
  is the span_id (a turn's last record uuid), stable across full and tail parses;
* transcribe_ai deposition payloads and Synapse's own extraction/judge calls
  (``ingestion.contamination``): third-party PII must not enter memory, and the eval must not
  eat itself;
* throwaway smoke-test workspaces (``ingestion.scratch_projects``);
* private sessions (schema 050, ``ingestion.private_sessions``);
* tombstones (``deleted_turns``): the turn was deleted from memory and must not return;
* replays: the span_id is stored under another session, or byte-identical content is already
  stored in this project or working directory (schema 036: retried sessions re-ship the same
  turns under fresh session ids AND span ids, 4.6% of episodes on a real corpus).

A new turn is appended at ``max(sequence) + 1`` and queued for extraction. The whole call is
one transaction on one connection, so a failure at any turn rolls back every write of the
call (the client's retry skips by span_id what did land, and nothing is stored without its
queue row).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, fields

from ingestion.contamination import is_harness_call, is_transcript_contamination
from ingestion.db import Database
from ingestion.models import Episode, ExtractionItem
from ingestion.private_sessions import PrivateSessions
from ingestion.scratch_projects import is_scratch_turn

logger = logging.getLogger(__name__)


@dataclass
class StoreStats:
    """What a :func:`store_episodes` call did with the turns it was given."""

    written: int = 0
    #: transcribe_ai payloads, Synapse's own harness calls and scratch-workspace turns
    dropped: int = 0
    private: int = 0
    tombstoned: int = 0
    #: span_id stored under another session, or identical content already stored
    replays: int = 0
    #: no span_id, so no way to tell a re-ship from a new turn
    unkeyed: int = 0

    def add(self, other: StoreStats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def log_skipped(self) -> None:
        """One line naming each kind of turn the call refused (nothing when none were)."""
        skipped = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "written"}
        skipped = {name: n for name, n in skipped.items() if n}
        if skipped:
            logger.info("episode writer skipped turns: %s", skipped)


def store_episodes(
    db: Database, eps: Iterable[Episode], project_default: str | None = None
) -> StoreStats:
    """Store ``eps`` (any number of sessions) through the guards in the module docstring.

    ``project_default`` labels the turns the parser left without a project."""
    stats = StoreStats()
    private = PrivateSessions(db)  # per-call memo: one SELECT per session id
    # Per session, the stored span_ids and the high-water sequence, loaded on first sight.
    seen: dict[str, set[str]] = {}
    top_seq: dict[str, int] = {}
    with db.atomic():
        for ep in sorted(eps, key=lambda e: e.sequence):
            if ep.session_id not in seen:
                seen[ep.session_id], top_seq[ep.session_id] = db.get_session_span_index(
                    ep.session_id
                )
            span_id = ep.span_id
            if not span_id:
                stats.unkeyed += 1
                continue
            if span_id in seen[ep.session_id]:
                continue  # idempotent no-op; also a tail's truncated leading fragment
            if not ep.project and project_default:
                ep = ep.model_copy(update={"project": project_default})
            if is_transcript_contamination(ep.content) or is_harness_call(ep.content):
                stats.dropped += 1
                continue
            if is_scratch_turn(ep.project, ep.metadata):
                stats.dropped += 1
                continue
            # The plugin's hook already stops posting a private session while its local
            # marker exists; this is the durable half, so a catch-up sweep or a backfill
            # months later can't ingest what the hook skipped.
            if private.is_private(ep.session_id):
                stats.private += 1
                continue
            cwd = (ep.metadata or {}).get("cwd")
            if db.is_tombstoned(span_id, ep.project, ep.content, cwd):
                stats.tombstoned += 1
                continue
            if db.span_id_exists(span_id) or db.content_dup_exists(ep.project, ep.content, cwd):
                stats.replays += 1
                continue
            top_seq[ep.session_id] += 1
            seen[ep.session_id].add(span_id)
            ep = ep.model_copy(update={"sequence": top_seq[ep.session_id]})
            episode_id = db.upsert_episode(ep)
            if ep.content.strip():
                db.enqueue_extraction(
                    ExtractionItem(
                        episode_id=episode_id,
                        session_id=ep.session_id,
                        content=ep.content,
                        content_type="episode",
                        project=ep.project,
                    )
                )
            stats.written += 1
    return stats
