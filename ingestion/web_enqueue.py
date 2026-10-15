"""
Enqueue web_chunks for KG extraction (task #68).

Web chunks ride the same extraction_queue / claim / drain machinery as episode
chunks, discriminated by content_type='web_chunk' and keyed by web_chunk_id
(partial UNIQUE index in schema 018 makes enqueue idempotent).

Gates — the lane refuses substrate that would pollute the graph:
  - link-density: a chunk that is mostly markdown link syntax is page chrome
    (nav blocks, footers, "related articles" lists), not content.
  - minimum size: sub-200-char fragments carry no extractable relations.
  - near-dup pages: when several artifacts share a content_hash (same article
    scraped from multiple URLs / sessions), only the lowest-id artifact's
    chunks are enqueued.
  - opt-in: only an artifact marked metadata.kg_extract = true is extracted. Server-side
    capture (ingestion.capture) marks the pages of a rule that trusts its source; a generic
    capture is searchable, not extracted; scripts/ingest_research_brief.py marks its briefs.
    An artifact with no mark (the backlog from before capture) is never extracted by this
    lane, so turning it on does not spend on old pages.
  - readiness: only chunks that finished the contextualize+embed lane
    (context_prefix IS NOT NULL AND is_embedded) — guarantees the contextual
    prefix exists before extraction reads the chunk, whichever order the
    poller's maintenance stages ran in.

The queued `content` is `context_prefix + chunk` when a non-empty prefix
exists: the prefix situates the chunk in its parent doc, which measurably
helps extraction the same way it helps retrieval (Anthropic contextual
retrieval, schema 013).

The poller's maintenance thread calls enqueue_web_chunks every cycle
(``Maintenance.enqueue_web_extraction``), after it chunked, contextualized and embedded the
pages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

GATE_MIN_CHARS = 200
GATE_MAX_LINK_DENSITY = 0.60

# Web extraction is a background-research lane: below fresh conversation
# ingest (priority 0), above bulk historical backfills (priority 10).
WEB_EXTRACTION_PRIORITY = 5

# Inline markdown link or image: [text](url) / ![alt](url). Reference-style
# links are rare in scraper output; the inline form is what chrome looks like.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def link_density(text: str) -> float:
    """Fraction of characters inside markdown link/image syntax (0.0-1.0)."""
    if not text:
        return 0.0
    link_chars = sum(m.end() - m.start() for m in _LINK_RE.finditer(text))
    return link_chars / len(text)


@dataclass
class EnqueueStats:
    candidates: int = 0
    enqueued: int = 0
    skipped_link_density: int = 0
    skipped_too_small: int = 0
    skipped_conflict: int = 0  # already queued (race with a concurrent run)

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "enqueued": self.enqueued,
            "skipped_link_density": self.skipped_link_density,
            "skipped_too_small": self.skipped_too_small,
            "skipped_conflict": self.skipped_conflict,
        }


_CANDIDATES_SQL = """
SELECT c.id AS web_chunk_id,
       c.content,
       c.context_prefix,
       a.id AS web_artifact_id,
       a.session_id
FROM web_chunks c
JOIN web_artifacts a ON a.id = c.web_artifact_id
WHERE a.kind IN ('web_scrape', 'research_brief')
  AND COALESCE((a.metadata->>'kg_extract')::boolean, FALSE)
  AND c.context_prefix IS NOT NULL
  AND c.is_embedded
  AND NOT EXISTS (
      SELECT 1 FROM extraction_queue q WHERE q.web_chunk_id = c.id
  )
  -- near-dup collapse: one canonical artifact per content_hash
  AND (a.content_hash IS NULL OR a.id = (
      SELECT min(a2.id) FROM web_artifacts a2
      WHERE a2.content_hash = a.content_hash
  ))
ORDER BY a.id, c.idx
"""

_INSERT_SQL = """
INSERT INTO extraction_queue
    (web_chunk_id, session_id, content, content_type, project, priority)
VALUES (%s, %s, %s, 'web_chunk', NULL, %s)
ON CONFLICT (web_chunk_id) WHERE web_chunk_id IS NOT NULL DO NOTHING
"""


def enqueue_web_chunks(conn: Any) -> EnqueueStats:
    """Enqueue gate-passing, not-yet-queued web chunks for KG extraction.

    Idempotent: re-runs are no-ops via the partial UNIQUE index. `project`
    stays NULL — web research has no project tag; the per-entity group
    classifier in the extractor routes personal-domain entities regardless.

    Every call reads all the candidates, including the chunks the gates refuse (they are never
    queued): a LIMIT would let those fill it. The opt-in keeps the set small.
    """
    stats = EnqueueStats()
    rows = conn.execute(_CANDIDATES_SQL).fetchall()

    for row in rows:
        # dict_row vs tuple-row tolerance
        if isinstance(row, dict):
            chunk_id, content = row["web_chunk_id"], row["content"]
            prefix, session_id = row["context_prefix"], row["session_id"]
        else:
            chunk_id, content, prefix, _artifact_id, session_id = row
        stats.candidates += 1

        if len(content) < GATE_MIN_CHARS:
            stats.skipped_too_small += 1
            continue
        if link_density(content) > GATE_MAX_LINK_DENSITY:
            stats.skipped_link_density += 1
            continue

        queued_content = f"{prefix}\n\n{content}" if prefix else content
        cur = conn.execute(
            _INSERT_SQL, (chunk_id, session_id, queued_content, WEB_EXTRACTION_PRIORITY)
        )
        if cur.rowcount:
            stats.enqueued += 1
        else:
            stats.skipped_conflict += 1

    conn.commit()
    if stats.enqueued:
        logger.info("web_enqueue: %s", stats.as_dict())
    return stats


# Gate check exposed for the retro-clean script (Phase 3 of the spec): chunks
# already in the DB that fail the gates today.
def gate_failures(content: str) -> list[str]:
    """Which gates this content fails (empty list = clean)."""
    reasons = []
    if len(content) < GATE_MIN_CHARS:
        reasons.append("too_small")
    if link_density(content) > GATE_MAX_LINK_DENSITY:
        reasons.append("link_density")
    return reasons
