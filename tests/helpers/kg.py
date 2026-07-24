"""Shared KG test scaffolding.

Two things the KG tests all reach for:

* ``_axis_list`` / ``DIM`` — a one-hot unit vector (cosine-orthogonal set) used
  to seed deterministic embeddings without a real embedder.
* ``make_edge_row`` — the edge-row factory mirroring the writer's row contract
  (``ingestion.kg_pg_write.create_edges`` / ``KGClient.create_edges_batch``
  destructure these dicts). Each call site passes the keys that differ for its
  case via kwargs, and drops keys its contract omits via ``drop=(...)``.
"""

from __future__ import annotations

DIM = 2048


def _axis_list(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def make_edge_row(uuid: str, *, drop: tuple[str, ...] = (), **over) -> dict:
    """Build an edge row for the writer. ``over`` overrides values (extra keys
    are added verbatim); ``drop`` removes keys this call site's contract omits."""
    row = {
        "src": "e-s",
        "tgt": "e-t",
        "edge_uuid": uuid,
        "name": "USES",
        "fact": f"fact for {uuid}",
        "episodes": [1, 2],
        "created_at": "2026-06-01T00:00:00+00:00",
        "t_created": "2026-06-01T00:00:00+00:00",
        "valid_at": "2026-06-01T00:00:00+00:00",
        "t_valid": "2026-06-01T00:00:00+00:00",
        "emb": _axis_list(0),
    }
    row.update(over)
    for key in drop:
        row.pop(key, None)
    return row
