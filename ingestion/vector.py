"""The pgvector text literal: one formatter for every write and every query.

Six decimals per component. That is lossless for what the stores keep (the ANN indexes are
``halfvec``, about three significant digits) and less than half the bytes of ``str(float)``,
which matters because a statement carries the literal as a bound parameter and Postgres
parses all of it again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload


class QueryVector(list[float]):
    """A query embedding that carries its own literal.

    One recall binds the same vector into every leg (episodes, chunks, facts, entity seeds,
    supersessions, web, notes); the literal is formatted here once instead of once per leg.
    """

    literal: str

    def __init__(self, emb: Sequence[float]) -> None:
        super().__init__(emb)
        self.literal = _format(self)


def _format(emb: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"


@overload
def vec_literal(emb: Sequence[float]) -> str: ...


@overload
def vec_literal(emb: None) -> None: ...


def vec_literal(emb: Sequence[float] | None) -> str | None:
    """``emb`` as pgvector text input (``[a,b,c]``); ``None`` stays ``None`` (a NULL column)."""
    if emb is None:
        return None
    if isinstance(emb, QueryVector):
        return emb.literal
    return _format(emb)
