"""Typed per-operation snapshots of retrieval tuning knobs."""

from dataclasses import dataclass

# Fixed signature defaults shared by the public APIs and candidate searches.
_EPISODE_FETCH = 100
_EPISODE_RERANK_POOL = 100
_EPISODE_LIMIT = 5
_SUP_LIMIT = 3


@dataclass(frozen=True)
class RecallSettings:
    """Snapshot of the engine facade's live tuning knobs."""

    _RERANK_RECENCY_FLOOR: float
    _RERANK_WINDOW: bool
    _ECHO_MIN_QUERY_LEN: int
    _RECALL_FACT_FLOOR: float
    _RECALL_PASSAGE_CAND: int
    _EPISODE_CUTOFF_TAU: float
    _EPISODE_CUTOFF_MIN_K: int
    _EPISODE_CUTOFF_MAX_K: int
    _WEB_FETCH: int
    _WEB_LIMIT: int
    _NOTES_FLOOR: float
    _EPISODE_FETCH: int
    _NOTES_IN_RECALL: bool
    _SUPERSEDED_LIMIT: int
    _RECALL_SELF_EXCLUDE: int
    _RECALL_BM25_FUSE: bool
    _RECALL_FLOOR_ENFORCE: bool
    _RERANK_RECENCY: bool
    _SUPPRESS_QUERY_ECHO: bool
    _WEB_RERANK_POOL: int
    _WEB_FLOOR: float
    _NOTES_LIMIT: int
    _NOTES_BODY_CAP: int
    _EPISODE_RERANK_POOL: int
    _FACT_LIMIT: int
    _RECALL_EPISODE_LIMIT: int
    _RECALL_PASSAGE_SRC_K: int
    _RECALL_FLOOR: float
    _KG_OWNER: str
    _RERANK_DOC_CAP: int
    _ECHO_CONTENT_CAP: int
    _RECALL_PASSAGE_N: int
    _RECALL_FLOOR_KEEP_MIN: int
    _SUP_MAX_DIST: float
    _NOTES_FETCH: int
    _RERANK_RECENCY_HALF_LIFE_DAYS: int
    _SUP_CANDIDATES: int
    _EMBED_DIMS: int
