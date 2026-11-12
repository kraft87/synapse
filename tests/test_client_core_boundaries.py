"""The client core's hard rules, enforced by reading its source.

plugin/synapse_client/ is shipped to every client machine and imported by harnesses that
bring their own interpreter (Python 3.11 and up) and their own import path:

- standard library only, so it runs wherever the harness runs; the one exception is a
  harness adapter's module that subclasses its host's base class (``_HOST_IMPORTS``);
- the core never imports a harness adapter, and an adapter imports only the core;
- relative imports inside the package, never ``import synapse_client``, so a host that
  vendors it under another name still loads it;
- no work at import time and no module-level mutable state, so a long-lived host that
  serves several sessions can hold several settings and import costs nothing;
- Python 3.11 syntax, and no module over 600 lines.

Each rule is an AST check; the self-tests at the bottom prove each check fails on a
violation, so a passing run means something.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "plugin" / "synapse_client"
_MODULES = sorted(_PACKAGE.rglob("*.py"))
_MAX_LINES = 600
_HARNESSES = _PACKAGE / "harnesses"
# The host modules an adapter module may import, by module: Hermes loads its provider
# as a subclass of its own base class, so that one module needs it.
_HOST_IMPORTS = {
    _HARNESSES / "hermes" / "provider.py": frozenset({"agent.memory_provider"}),
}
# Data files an adapter reads from the package (the installer's templates).
_DATA_SUFFIXES = (".py", ".json", ".tmpl")

# Calls allowed at import time (module and class level, decorators): each builds an
# immutable value or declares a type, and touches nothing outside the process.
_IMPORT_TIME_CALLS = frozenset(
    {
        "MappingProxyType",
        "types.MappingProxyType",
        "frozenset",
        "re.compile",
        "dataclass",
        "dataclasses.dataclass",
        "field",
        "dataclasses.field",
        "TypeVar",
        "typing.TypeVar",
    }
)
_DECORATORS = frozenset(
    {"property", "classmethod", "staticmethod", "dataclass", "runtime_checkable"}
)
# Reading a module constant (``KEYS.items()``) to derive another is not work.
_VIEWS = (".items", ".keys", ".values")
_TYPE_CHECKING = ("TYPE_CHECKING", "typing.TYPE_CHECKING")
_MUTABLE = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _import_violations(tree: ast.Module, host: frozenset[str] = frozenset()) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".", 1)[0]
            if top == "synapse_client":
                out.append(f"line {node.lineno}: absolute import of {name}; use a relative one")
            elif top not in sys.stdlib_module_names and name not in host:
                out.append(f"line {node.lineno}: {name} is not in the standard library")
    return out


def _relative_targets(path: Path, tree: ast.Module) -> list[tuple[int, str]]:
    """``(line, package-relative path)`` of what each relative import in ``path`` names:
    ``harnesses/hermes/messages`` for ``from .messages import x`` in harnesses/hermes/,
    ``harnesses`` for ``from .. import harnesses`` in a core module."""
    out = []
    here = path.parent.relative_to(_PACKAGE).parts
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level):
            continue
        base = here[: len(here) - (node.level - 1)] if node.level > 1 else here
        module = tuple((node.module or "").split(".")) if node.module else ()
        if node.module:
            out.append((node.lineno, "/".join(base + module)))
        else:
            out += [(node.lineno, "/".join((*base, alias.name))) for alias in node.names]
    return out


def _in_harnesses(target: str) -> bool:
    return target == "harnesses" or target.startswith("harnesses/")


def _allowed_call(call: ast.Call) -> bool:
    name = _dotted(call.func)
    if name in _IMPORT_TIME_CALLS:
        return True
    return name.endswith(_VIEWS) and name.count(".") == 1 and not call.args


def _import_time_value(node: ast.expr | None, where: str) -> list[str]:
    """Violations in an expression evaluated at import: a mutable literal or a call
    outside the allowlist, anywhere inside it."""
    if node is None:
        return []
    # A literal handed straight to MappingProxyType/frozenset is frozen there, and one in
    # a subscript is a type expression (``Callable[[str], None]``), not a value.
    exempt = {
        id(arg)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and _dotted(sub.func) in _IMPORT_TIME_CALLS
        for arg in sub.args
    }
    exempt |= {
        id(inner)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Subscript)
        for inner in ast.walk(sub.slice)
        if isinstance(inner, ast.List)
    }
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, _MUTABLE):
            if id(sub) not in exempt:
                out.append(f"line {sub.lineno}: mutable {type(sub).__name__} {where}")
        elif isinstance(sub, ast.Call) and not _allowed_call(sub):
            out.append(f"line {sub.lineno}: call to {_dotted(sub.func) or '?'} {where}")
    return out


def _body_violations(body: list[ast.stmt], where: str) -> list[str]:
    out = []
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(stmt, ast.Expr):
            docstring = isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
            if not (docstring or isinstance(stmt.value, ast.Constant)):
                out.append(f"line {stmt.lineno}: statement runs {where}")
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in stmt.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if _dotted(target) not in _DECORATORS | _IMPORT_TIME_CALLS:
                    out.append(f"line {deco.lineno}: decorator {_dotted(target)} {where}")
            args = stmt.args
            for default in [*args.defaults, *(d for d in args.kw_defaults if d is not None)]:
                out += _import_time_value(default, f"in a default of {stmt.name}")
            continue
        if isinstance(stmt, ast.ClassDef):
            for deco in stmt.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if _dotted(target) not in _DECORATORS | _IMPORT_TIME_CALLS:
                    out.append(f"line {deco.lineno}: decorator {_dotted(target)} {where}")
                if isinstance(deco, ast.Call):
                    out += _import_time_value(deco, where)
            out += _body_violations(stmt.body, f"in class {stmt.name}")
            continue
        if isinstance(stmt, ast.Assign):
            out += _import_time_value(stmt.value, where)
            continue
        if isinstance(stmt, (ast.AnnAssign, ast.TypeAlias)):
            out += _import_time_value(stmt.value, where)
            continue
        if isinstance(stmt, ast.If) and _dotted(stmt.test) in _TYPE_CHECKING:
            continue
        out.append(f"line {stmt.lineno}: {type(stmt).__name__} runs {where}")
    return out


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 11))


def _ids(path: Path) -> str:
    return str(path.relative_to(_ROOT))


def test_the_package_is_where_the_rules_look():
    assert (_PACKAGE / "__init__.py").exists()
    assert len(_MODULES) >= 2


@pytest.mark.parametrize("path", _MODULES, ids=_ids)
def test_standard_library_and_relative_imports_only(path: Path) -> None:
    assert _import_violations(_parse(path), _HOST_IMPORTS.get(path, frozenset())) == []


def test_the_host_exception_is_one_module():
    """Hermes's base class is allowed in its provider module and nowhere else: every
    other module is held to the plain rule above."""
    assert list(_HOST_IMPORTS) == [_HARNESSES / "hermes" / "provider.py"]
    assert all(path in _MODULES for path in _HOST_IMPORTS)
    assert _import_violations(ast.parse("from agent.memory_provider import MemoryProvider\n"))


@pytest.mark.parametrize("path", _MODULES, ids=_ids)
def test_the_core_imports_no_harness_and_an_adapter_only_the_core(path: Path) -> None:
    targets = _relative_targets(path, _parse(path))
    rel = path.relative_to(_PACKAGE)
    if rel.parts[0] != "harnesses":
        bad = [t for t in targets if _in_harnesses(t[1])]
        assert bad == [], f"{_ids(path)} imports a harness: {bad}"
    elif len(rel.parts) > 2:
        own = "/".join(rel.parts[:2])
        bad = [
            t
            for t in targets
            if _in_harnesses(t[1]) and not (t[1] == own or t[1].startswith(own + "/"))
        ]
        assert bad == [], f"{_ids(path)} imports another harness: {bad}"


def test_the_harness_rule_reads_relative_imports():
    core = _PACKAGE / "context.py"
    hermes = _HARNESSES / "hermes" / "provider.py"
    assert _relative_targets(core, ast.parse("from .harnesses import registry\n")) == [
        (1, "harnesses")
    ]
    assert _relative_targets(core, ast.parse("from . import harnesses, api\n")) == [
        (1, "harnesses"),
        (1, "api"),
    ]
    assert _relative_targets(hermes, ast.parse("from ...api import Client\n")) == [(1, "api")]
    assert _relative_targets(hermes, ast.parse("from . import messages\n")) == [
        (1, "harnesses/hermes/messages")
    ]
    assert _relative_targets(hermes, ast.parse("from ..codex import x\n")) == [
        (1, "harnesses/codex")
    ]


@pytest.mark.parametrize("path", _MODULES, ids=_ids)
def test_no_work_at_import_and_no_module_level_mutable_state(path: Path) -> None:
    assert _body_violations(_parse(path).body, "at import") == []


def _too_long(path: Path) -> int:
    """The physical line count when over the ceiling, else 0. Documentation counts, so
    the ceiling cannot be met by hiding code behind comments or strings."""
    lines = len(path.read_text(encoding="utf-8").splitlines())
    return lines if lines > _MAX_LINES else 0


@pytest.mark.parametrize("path", _MODULES, ids=_ids)
def test_module_size(path: Path) -> None:
    lines = _too_long(path)
    assert not lines, f"{_ids(path)} has {lines} lines; split by responsibility"


@pytest.mark.parametrize("path", _MODULES, ids=_ids)
def test_parses_as_python_3_11(path: Path) -> None:
    _parse(path)  # raises SyntaxError on newer-only syntax (PEP 695 type aliases, ...)


def test_data_files_ship_beside_the_code():
    """Non-Python files the package reads must sit inside it, so the bundle carries them."""
    for path in _PACKAGE.rglob("*"):
        if path.is_file() and path.suffix not in _DATA_SUFFIXES:
            assert path.name == "py.typed" or "__pycache__" in path.parts, path


# ---------------------------------------------------------------------------
# The checks themselves catch what they claim to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import synapse_client\n",
        "from synapse_client.settings import Settings\n",
        "import requests\n",
        "from httpx import Client\n",
        "def f():\n    import yaml\n",
    ],
)
def test_the_import_rule_rejects(source):
    assert _import_violations(ast.parse(source))


def test_the_import_rule_accepts_relative_and_stdlib():
    source = "from __future__ import annotations\nimport json\nfrom . import x\nfrom .a import b\n"
    assert _import_violations(ast.parse(source)) == []


@pytest.mark.parametrize(
    "source",
    [
        "CACHE = {}\n",
        "SEEN: set[str] = set()\n",
        "ITEMS = [1, 2]\n",
        "URL = os.environ.get('SYNAPSE_URL')\n",
        "DATA = json.loads(open('x').read())\n",
        "print('hi')\n",
        "for i in range(3):\n    pass\n",
        "class C:\n    registry = []\n",
        "def f(x=[]):\n    pass\n",
        "def f(x=time.time()):\n    pass\n",
        "@functools.lru_cache\ndef f():\n    pass\n",
    ],
)
def test_the_import_time_rule_rejects(source):
    assert _body_violations(ast.parse(source).body, "at import")


def test_the_import_time_rule_accepts_frozen_constants():
    source = (
        '"""doc"""\n'
        "import types\n"
        "KEYS = types.MappingProxyType({'a': 'b'})\n"
        "NAMES = frozenset({'x'})\n"
        "PAIR = ('a', 'b')\n"
        "RX = re.compile(r'x')\n"
        "Origin = tuple[str, str]\n"
        "Log = Callable[[str], None]\n"
        "@dataclass(frozen=True)\n"
        "class S:\n"
        "    a: int = field(default=0, repr=False)\n"
        "    def f(self, x: int = 3, *, y: str = '') -> None:\n"
        "        global_state = {}\n"
    )
    assert _body_violations(ast.parse(source).body, "at import") == []


def test_the_size_rule_counts_physical_lines(tmp_path):
    big = tmp_path / "big.py"
    big.write_text('"""\n' + "x\n" * (_MAX_LINES - 2) + '"""\n')
    assert not _too_long(big)
    big.write_text('"""\n' + "x\n" * (_MAX_LINES - 1) + '"""\n')
    assert _too_long(big) == _MAX_LINES + 1


def test_the_syntax_rule_rejects_newer_syntax():
    with pytest.raises(SyntaxError):
        ast.parse("type Alias = int\n", feature_version=(3, 11))
