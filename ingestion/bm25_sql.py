"""The filter of a BM25 statement, written so pg_search checks it once per hit.

pg_search (ParadeDB) builds one Tantivy ``heap_filter`` clause for every top-level WHERE
conjunct the BM25 index does not cover, and each clause re-runs the text match and checks
the heap for every hit. Cost and shared buffers grow with the number of clauses, and
``paradedb.score()`` is multiplied by it (the same factor on every row, so the order holds).
A WHERE-level OR is worse: it becomes boolean "should" sub-queries over
``indexed_query: "all"``, scanning far more of the index.

Folding every such conjunct into one ``CASE WHEN (...) THEN 1 END = 1`` leaves one clause:
the same rows in the same order. On prod (pg_search 0.23.2), real questions: the KG fact leg
261 -> 120ms at full trust and 467 -> 128ms restricted, and a restricted device's episode leg
(whose own-surface rule is an OR) 209 -> 108ms. pg_search 0.24.0 plans it the same way.

Shapes pg_search rejects with "Unsupported query shape", on 0.23.2 and 0.24.0 alike:
``coalesce(<boolean>, false)`` and ``CASE ... THEN <boolean> END``. Keep the comparison
to an integer outside the CASE.
"""

from __future__ import annotations

from collections.abc import Sequence


def bm25_filter(conjuncts: Sequence[str]) -> str:
    """``conjuncts`` as one heap filter, to append after the ``@@@`` match.

    Returns ``""`` for none, else ``" AND CASE WHEN ((c1) AND ... AND (cn)) THEN 1 END = 1"``.
    The ``%s`` placeholders keep their order, so the caller's parameters do not change. A
    conjunct that is NULL or false makes the CASE NULL, and NULL = 1 filters the row, as the
    plain WHERE would. Every conjunct goes in, a lone OR included: one OR on its own still
    expands into "match everything" sub-queries.
    """
    if not conjuncts:
        return ""
    return " AND CASE WHEN (" + " AND ".join(f"({c})" for c in conjuncts) + ") THEN 1 END = 1"
