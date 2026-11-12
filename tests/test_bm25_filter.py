"""ingestion.bm25_sql: a BM25 statement's filter as one pg_search heap filter.

The plans and rows are held by tests/test_bm25_filter_db.py; this file holds the text and the
rule that every BM25 statement in the source builds its filter here.
"""

from __future__ import annotations

import re
from pathlib import Path

from ingestion.bm25_sql import bm25_filter

_ROOT = Path(__file__).resolve().parent.parent

#: Modules that send a BM25 match without bm25_filter, and why.
_EXEMPT = {
    # Its one filter is on the joined web_artifacts row, which the web_chunks scan never
    # sees: the join checks it, not pg_search.
    "mcp_server/recall_web.py",
}
#: The modules that send BM25 statements today, so the scan below cannot pass by finding none.
_KNOWN = {
    "ingestion/kg_pg_read.py",
    "mcp_server/dashboard_routes.py",
    "mcp_server/kg_pg.py",
    "mcp_server/recall_episode_search.py",
    "mcp_server/recall_web.py",
}
_SKIP_DIRS = {"tests", ".venv", "node_modules", "web", ".git"}

#: The right-hand side of a BM25 match: a paradedb query function, a string or a parameter.
_MATCH = re.compile(r"@@@\s*(?:paradedb\.\w+\([^)]*\)|'[^']*'|%s)")
#: Where the statement's WHERE ends in the source: its ORDER BY, or the end of the string.
_WHERE_END = re.compile(r'ORDER BY|LIMIT|",|"""|"\)')
_INTERPOLATION = re.compile(r"\{([^{}]*)\}")
_EMPTY_INIT = re.compile(r'(?:""|\[\])(?:\s*,\s*(?:""|\[\]))*')


def test_no_filter_appends_nothing():
    assert bm25_filter([]) == ""


def test_the_filters_are_one_case_with_their_placeholders_in_order():
    assert bm25_filter(["owner_id = %s", "project = ANY(%s)", "t_invalid IS NULL"]) == (
        " AND CASE WHEN ((owner_id = %s) AND (project = ANY(%s)) AND (t_invalid IS NULL))"
        " THEN 1 END = 1"
    )


def test_an_or_stays_inside_its_own_parentheses():
    got = bm25_filter(["project = ANY(%s) OR surface_id = %s", "session_id = %s"])
    assert got == (
        " AND CASE WHEN ((project = ANY(%s) OR surface_id = %s) AND (session_id = %s))"
        " THEN 1 END = 1"
    )
    # The comparison is to an integer outside the CASE: pg_search rejects a CASE that
    # yields a boolean, and coalesce(<boolean>, false), as an unsupported query shape.
    assert got.endswith("END = 1") and "THEN true" not in got and "coalesce" not in got


def _bm25_statements() -> list[tuple[str, str, str]]:
    """(module, the source after the match up to the end of its WHERE, the module's text)
    for every BM25 match in the source tree."""
    out = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT)
        if rel.parts[0] in _SKIP_DIRS:
            continue
        text = path.read_text()
        for m in _MATCH.finditer(text):
            tail = text[m.end() :]
            end = _WHERE_END.search(tail)
            out.append((rel.as_posix(), tail[: end.start()] if end else tail, text))
    return out


def _from_bm25_filter(expr: str, module: str) -> bool:
    """``expr`` is a bm25_filter call, or a name the module assigns only from one (besides
    an empty initial value)."""
    if "bm25_filter(" in expr:
        return True
    name = re.escape(expr.strip())
    assigned = re.findall(rf"^\s*(?:\w+\s*,\s*)*{name}\b(?:\s*,\s*\w+)*\s*=\s*(.+)$", module, re.M)
    sources = [a for a in assigned if not _EMPTY_INIT.fullmatch(a.strip())]
    return bool(sources) and all(a.startswith("bm25_filter(") for a in sources)


def test_every_bm25_statement_builds_its_filter_with_bm25_filter():
    """Whatever follows a match in its WHERE is string glue and interpolations of
    bm25_filter: no AND or OR written out, no fragment built another way."""
    statements = _bm25_statements()
    assert {mod for mod, _, _ in statements} == _KNOWN
    bad = []
    for mod, where, text in statements:
        if mod in _EXEMPT:
            continue
        if re.search(r"\b(AND|OR)\b", _INTERPOLATION.sub("", where), re.I):
            bad.append((mod, where))
        bad += [(mod, e) for e in _INTERPOLATION.findall(where) if not _from_bm25_filter(e, text)]
    assert bad == []
