"""No SQL in the application names the legacy ``group_id`` column.

The knowledge graph is one memory per owner. Schema 069 drops ``group_id`` from
``kg_entities``, ``kg_relationships``, ``notes`` and ``preferences``, and it is applied by
hand after the release, so the same image runs on a database with the column and on one
without. A statement that names it would fail on the second, and on the first a write that
names it would quietly bring the two-pile split back. This test scans ``ingestion/`` and
``mcp_server/`` and fails on the first such statement, naming the file and line.

How it tells SQL from prose. The sources are parsed with ``ast``, so comments are never
seen. A docstring (the first statement of a module, class or function) is prose: the
modules explain the legacy column there, and those are skipped. Every other string
literal counts as possible SQL, including the constant parts of an f-string and the pieces
of ``"..." + "..."``, because a statement is often built in fragments
(``" AND group_id = %s"``) that name no table on their own. So the rule is the plain one:
no string literal outside a docstring may name ``group_id``, whichever table it is about.
That covers the four tables above and any other (``recall_metrics`` carries the column
too). ``timeline_events.domain`` is a different column and never matches.

A literal that is not SQL but has to say ``group_id`` goes in :data:`ALLOWED`, with the
reason. An entry names the file and the exact literal, so a second use in the same file
still fails, and an entry that no longer matches anything fails too, so the list cannot
outlive what it excuses.

This is a tripwire for the accident of writing such a statement, not a defence against
building one from a variable.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("ingestion", "mcp_server")
_GROUP_ID = re.compile(r"\bgroup_id\b", re.IGNORECASE)

#: (file relative to the repository root, the literal) -> why it is not SQL.
ALLOWED: dict[tuple[str, str], str] = {
    ("mcp_server/hidden_tools.py", "group_id"): (
        "RETIRED_ARGS: the name of a tool argument an older client may still send, which the "
        "middleware drops before validation. It is a dictionary key, not a column."
    ),
}


def _docstrings(tree: ast.AST) -> set[int]:
    """ids of the string nodes that are docstrings: the first statement of a module, class
    or function, when it is a bare string."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _literal_text(node: ast.AST) -> str | None:
    """The text of a string expression made only of literals: a string, an f-string (each
    interpolation counts as ``{}``) or a ``+`` of those. None for anything else."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            text = _literal_text(part)
            parts.append("{}" if text is None else text)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_text(node.left), _literal_text(node.right)
        return None if left is None or right is None else left + right
    return None


def group_id_literals(source: str) -> list[tuple[int, str]]:
    """(line, text) of every string literal in ``source`` that names ``group_id``, outside
    docstrings."""
    tree = ast.parse(source)
    skip = _docstrings(tree)
    hits: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        if id(node) in skip:
            return
        text = _literal_text(node)
        if text is not None:
            if _GROUP_ID.search(text):
                hits.append((getattr(node, "lineno", 0), text))
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return hits


def _scan() -> list[tuple[str, int, str]]:
    """(file, line, text) for every hit in the scanned directories."""
    out: list[tuple[str, int, str]] = []
    for directory in SCANNED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            out += [(rel, line, text) for line, text in group_id_literals(path.read_text())]
    return out


def test_no_statement_in_the_application_names_group_id():
    findings = [
        f"{rel}:{line}: a string literal names group_id: {text.strip()[:100]!r}"
        for rel, line, text in _scan()
        if (rel, text) not in ALLOWED
    ]
    assert not findings, (
        "group_id is a legacy column the application must not read or write (schema 069 drops it). "
        "If a literal is not SQL, add it to ALLOWED with the reason.\n" + "\n".join(findings)
    )


def test_every_allowed_literal_is_still_there():
    present = {(rel, text) for rel, _line, text in _scan()}
    stale = sorted(set(ALLOWED) - present)
    assert not stale, f"ALLOWED entries that match nothing any more, remove them: {stale}"


def test_every_allowed_literal_gives_a_reason():
    assert all(reason.strip() for reason in ALLOWED.values())


# ---------------------------------------------------------------------------------------
# The scanner itself
# ---------------------------------------------------------------------------------------

SQL_SHAPES = {
    "plain": 'q = "SELECT uuid FROM kg_entities WHERE group_id = %s"\n',
    "upper case column": 'q = "SELECT uuid FROM notes WHERE GROUP_ID = %s"\n',
    "insert column list": 'q = "INSERT INTO kg_relationships (uuid, group_id) VALUES (%s, %s)"\n',
    "update": "q = \"UPDATE preferences SET group_id = 'x'\"\n",
    "qualified": 'q = "SELECT e.group_id FROM kg_entities e"\n',
    "f-string": 'q = f"SELECT uuid FROM {table} WHERE group_id = %s"\n',
    "fragment": 'q = base\nq += " AND group_id = %s"\n',
    "implicit concatenation": 'q = ("SELECT uuid FROM kg_entities "\n     "WHERE group_id = %s")\n',
    "plus": 'q = "SELECT uuid FROM kg_entities WHERE " + "group_id = %s"\n',
    "split name": 'q = "SELECT uuid FROM kg_entities WHERE group_" + "id = %s"\n',
    "inside a function": 'def f():\n    return execute("DELETE FROM notes WHERE group_id = 1")\n',
    "inside a class": 'class A:\n    SQL = "SELECT group_id FROM preferences"\n',
    "second statement of a function": (
        'def f():\n    """Does a thing."""\n    return "SELECT group_id FROM notes"\n'
    ),
    "string after a docstring-like expression": 'x = 1\n"SELECT group_id FROM notes"\n',
}


@pytest.mark.parametrize("source", SQL_SHAPES.values(), ids=SQL_SHAPES.keys())
def test_the_scanner_finds_a_statement_that_names_group_id(source):
    assert group_id_literals(source)


PROSE_SHAPES = {
    "comment": "# the legacy group_id column is unused\nx = 1\n",
    "trailing comment": 'q = "SELECT uuid FROM kg_entities"  # no group_id here\n',
    "module docstring": '"""Rows carry a legacy group_id."""\nx = 1\n',
    "function docstring": 'def f():\n    """Never filters on group_id."""\n    return 1\n',
    "class docstring": 'class A:\n    """Has a group_id column."""\n',
    "async function docstring": 'async def f():\n    """About group_id."""\n    return 1\n',
    "timeline domain": 'q = "SELECT domain FROM timeline_events WHERE domain = %s"\n',
    "another name": 'q = "SELECT kg_group_id, group_ids FROM kg_entities"\n',
    "no literal": "group = 1\nq = group\n",
}


@pytest.mark.parametrize("source", PROSE_SHAPES.values(), ids=PROSE_SHAPES.keys())
def test_the_scanner_ignores_prose_and_unrelated_columns(source):
    assert group_id_literals(source) == []


def test_the_scanner_reports_the_line():
    source = 'a = 1\n\nq = (\n    "SELECT uuid"\n    " FROM kg_entities WHERE group_id = %s"\n)\n'
    assert [line for line, _ in group_id_literals(source)] == [4]
