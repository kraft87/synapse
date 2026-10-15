"""Capture rules: what a deployment adds to the generic detector, as files.

The generic detector (``ingestion.capture.detect``) recognises web research by shape. A
rule teaches it a source the shapes cannot see: a shell command's output (``bird thread``,
``yt-dlp``), a tool that returns a page in a format of its own, a source worth keeping cold.

Rules are TOML files read with the stdlib ``tomllib``, from two directories: the ones shipped
in the package (``ingestion/capture/rules/``) and an operator directory,
``SYNAPSE_CAPTURE_DIR`` (mount it into the container). A file is named for its rule; an
operator file overrides the shipped one of the same name, and ``enabled = false`` in it turns
that rule off. The first rule that matches a call (in name order) decides what happens to it,
ahead of the generic detector. One file is not a rule: ``deny.toml``, the list of connector
families the generic detector never captures. An operator ``deny.toml`` replaces the shipped
one, so copy it and edit.

A rule::

    description = "what it captures"          # optional
    destination = "archive"                   # "searchable" or "archive" (see below)
    min_chars = 200                           # shortest body kept (default 200)
    kind = "x_thread"                         # archive only: the kind column (default: the name)
    kg_extract = false                        # searchable only: let the knowledge graph read it

    [match]                                   # at least one of the two; both must match
    tool = ["mcp__acme__*"]                   # globs on the tool name
    command = 'bird\\s+thread\\s+(?P<id>\\d+)'    # regex searched in the shell command

    [fields]                                  # where each value comes from
    url = "https://x.com/i/status/{id}"
    source_id = "thread:x:{id}"               # archive only: what makes two captures one
    title = "Thread {id}"                     # optional
    author = ["@{json.0.author.username}"]    # optional; a list is alternatives, first that fills
    body = "{output}"

A value is a template; ``{...}`` is a regex group of ``command`` (``{id}``), a field of the
tool input (``{input.url}``), the tool output (``{output}``) or a path into the output parsed
as JSON (``{json.0.author.username}``; numbers index lists). A template with a reference that
resolves to nothing does not fill. A rule whose ``url`` or ``body`` does not fill is not
decided: the call goes on to the next rule and the generic detector.

``destination = "searchable"`` stores a page in ``web_artifacts``: chunked, contextualised and
embedded by the poller, served by recall's web bucket, and read by the knowledge graph only if
the rule says ``kg_extract = true`` (name a source you trust). ``"archive"`` stores the body
verbatim in the cold ``research_archive`` table, which recall never reads.

Where ``[fields]`` cannot say it, ``hook = "file.py:function"`` replaces it: the file sits next
to the rule file, the function takes ``(tool_input, output)`` and returns a dict of those
fields (or None for "not mine"). **A hook is Python the server imports and runs**: operator
rules are code, so mount the directory read-only and write only what you trust.

``SYNAPSE_CAPTURE_DIR`` is read once per process; edit, then restart the MCP server. A bad
file fails the load with the file and field named, and the server refuses to start with one.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, cast

SHIPPED_DIR = Path(__file__).resolve().parent / "rules"
DENY_FILE = "deny.toml"

DESTINATIONS = ("searchable", "archive")
DEFAULT_MIN_CHARS = 200

_ALLOWED_FIELDS = {
    "searchable": frozenset({"url", "title", "body", "author"}),
    "archive": frozenset({"url", "title", "body", "author", "source_id"}),
}
_REQUIRED_FIELDS = {
    "searchable": frozenset({"url", "body"}),
    "archive": frozenset({"url", "body", "source_id"}),
}
_RULE_KEYS = frozenset(
    {"description", "enabled", "destination", "kind", "kg_extract", "min_chars", "match"}
    | {"fields", "hook"}
)
_RESERVED_GROUPS = frozenset({"output", "input", "json"})
_REF = re.compile(r"\{([^{}]+)\}")

Hook = Callable[[dict[str, Any], str], dict[str, Any] | None]


class RuleError(ValueError):
    """A rule file or ``SYNAPSE_CAPTURE_DIR`` that cannot be loaded; names the file and field."""


def _fail(path: Path, where: str, problem: str) -> RuleError:
    return RuleError(f"{path}: {where}: {problem}")


@dataclass(frozen=True)
class Rule:
    name: str
    path: Path
    destination: str
    kind: str
    kg_extract: bool
    min_chars: int
    tools: tuple[str, ...]
    command: re.Pattern[str] | None
    #: field -> templates; the first that fills wins
    templates: dict[str, tuple[str, ...]]
    hook: Hook | None

    def extract(self, tool_name: str, tool_input: dict[str, Any], output: str) -> dict[str, str]:
        """The fields this rule takes from the call, or {} when it does not match or its
        required fields do not fill."""
        if self.tools and not any(fnmatchcase(tool_name, glob) for glob in self.tools):
            return {}
        groups: dict[str, str] = {}
        if self.command:
            command = command_of(tool_input)
            found = self.command.search(command) if command else None
            if not found:
                return {}
            groups = {k: v for k, v in found.groupdict().items() if v}
        if self.hook:
            values = self._hooked(tool_input, output)
        else:
            source = _Source(tool_input, output, groups)
            values = {}
            for field, alternatives in self.templates.items():
                if value := next((v for t in alternatives if (v := source.fill(t))), None):
                    values[field] = value
        return values if _REQUIRED_FIELDS[self.destination] <= values.keys() else {}

    def _hooked(self, tool_input: dict[str, Any], output: str) -> dict[str, str]:
        assert self.hook is not None
        returned = self.hook(tool_input, output)
        if not returned:
            return {}
        bad = set(returned) - _ALLOWED_FIELDS[self.destination]
        if bad or not all(isinstance(v, str) for v in returned.values()):
            raise RuleError(
                f"{self.path}: hook returned {sorted(returned)}; expected string values for "
                f"{sorted(_ALLOWED_FIELDS[self.destination])}"
            )
        return {k: v.strip() for k, v in returned.items() if v.strip()}


def command_of(tool_input: dict[str, Any]) -> str | None:
    """The shell command of a call: Claude's ``command``, Codex's ``cmd`` or an argv list."""
    command = tool_input.get("command") or tool_input.get("cmd")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    return command if isinstance(command, str) else None


class _Source:
    """What a template can reference for one call. The output is parsed as JSON once, on
    the first ``{json...}``."""

    _UNPARSED = object()

    def __init__(self, tool_input: dict[str, Any], output: str, groups: dict[str, str]) -> None:
        self._input, self._output, self._groups = tool_input, output, groups
        self._json: Any = self._UNPARSED

    def fill(self, template: str) -> str | None:
        unresolved = False

        def value(match: re.Match[str]) -> str:
            nonlocal unresolved
            found = self._lookup(match.group(1))
            if found is None:
                unresolved = True
                return ""
            return found

        text = _REF.sub(value, template).strip()
        return None if unresolved or not text else text

    def _lookup(self, ref: str) -> str | None:
        if ref == "output":
            return self._output
        if ref.startswith("input."):
            return _leaf(_walk(self._input, ref[len("input.") :]))
        if ref.startswith("json."):
            if self._json is self._UNPARSED:
                try:
                    self._json = json.loads(self._output)
                except ValueError:
                    self._json = None
            return _leaf(_walk(self._json, ref[len("json.") :]))
        return self._groups.get(ref)


def _walk(node: Any, path: str) -> Any:
    for key in path.split("."):
        if isinstance(node, dict):
            node = node.get(key)
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


def _leaf(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return None


# ---------- Loading ----------


def _only(path: Path, table: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """Refuse a key the table does not know: a typo must not silently turn a rule into
    something else."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise _fail(
            path, where, f"unknown key {unknown[0]!r} (known: {', '.join(sorted(allowed))})"
        )


def _strings(path: Path, where: str, value: Any) -> tuple[str, ...]:
    """A string or a non-empty list of non-empty strings, as a tuple."""
    items = [value] if isinstance(value, str) else value
    if (
        not isinstance(items, list)
        or not items
        or not all(isinstance(i, str) and i.strip() for i in items)
    ):
        raise _fail(path, where, "expected a string or a list of strings")
    return tuple(items)


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise _fail(path, "file", str(e)) from e


def _load_deny(path: Path) -> frozenset[str]:
    data = _read(path)
    _only(path, data, frozenset({"words"}), "deny list")
    words = _strings(path, "words", data.get("words"))
    return frozenset(w.lower() for w in words)


def _load_rule(path: Path) -> Rule | None:
    """The rule in ``path``, or None when it says ``enabled = false``."""
    data = _read(path)
    name = path.stem
    _only(path, data, _RULE_KEYS, "rule")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise _fail(path, "enabled", "expected true or false")
    if not enabled:
        return None
    destination = data.get("destination")
    if destination not in DESTINATIONS:
        raise _fail(path, "destination", f"must be one of {', '.join(DESTINATIONS)}")
    kind = data.get("kind", name)
    if "kind" in data and destination != "archive":
        raise _fail(path, "kind", "only an archive rule has a kind")
    if not isinstance(kind, str) or not kind.strip():
        raise _fail(path, "kind", "expected a non-empty string")
    kg_extract = data.get("kg_extract", False)
    if not isinstance(kg_extract, bool):
        raise _fail(path, "kg_extract", "expected true or false")
    if kg_extract and destination != "searchable":
        raise _fail(path, "kg_extract", "only a searchable rule can be read by the graph")
    min_chars = data.get("min_chars", DEFAULT_MIN_CHARS)
    if not isinstance(min_chars, int) or isinstance(min_chars, bool) or min_chars < 1:
        raise _fail(path, "min_chars", "expected a positive integer")

    match = data.get("match")
    if not isinstance(match, dict):
        raise _fail(path, "[match]", "missing: a rule needs a tool glob or a command regex")
    _only(path, match, frozenset({"tool", "command"}), "[match]")
    if not match:
        raise _fail(path, "[match]", "needs a tool glob or a command regex")
    tools = _strings(path, "[match] tool", match["tool"]) if "tool" in match else ()
    command = None
    if "command" in match:
        pattern = _strings(path, "[match] command", match["command"])
        if len(pattern) != 1:
            raise _fail(path, "[match] command", "expected one regex")
        try:
            command = re.compile(pattern[0])
        except re.error as e:
            raise _fail(path, "[match] command", f"not a regex: {e}") from e
        for group in command.groupindex:
            if group in _RESERVED_GROUPS:
                raise _fail(path, "[match] command", f"group name {group!r} is reserved")

    templates: dict[str, tuple[str, ...]] = {}
    hook = None
    if "hook" in data:
        if "fields" in data:
            raise _fail(path, "hook", "a rule has [fields] or a hook, not both")
        hook = _load_hook(path, data["hook"])
    else:
        fields = data.get("fields")
        if not isinstance(fields, dict):
            raise _fail(path, "[fields]", "missing: say where the values come from, or name a hook")
        _only(path, fields, _ALLOWED_FIELDS["archive"], "[fields]")
        for field in sorted(fields):
            if field not in _ALLOWED_FIELDS[destination]:
                raise _fail(path, f"[fields] {field}", f"not used by a {destination} rule")
            templates[field] = _strings(path, f"[fields] {field}", fields[field])
            for template in templates[field]:
                _check_refs(path, f"[fields] {field}", template, command)
        for field in sorted(_REQUIRED_FIELDS[destination] - templates.keys()):
            raise _fail(path, f"[fields] {field}", f"required by a {destination} rule")
    return Rule(
        name,
        path,
        destination,
        kind.strip(),
        kg_extract,
        min_chars,
        tools,
        command,
        templates,
        hook,
    )


def _check_refs(path: Path, where: str, template: str, command: re.Pattern[str] | None) -> None:
    for ref in _REF.findall(template):
        if ref == "output":
            continue
        if ref.startswith(("input.", "json.")):
            if ref.endswith("."):
                raise _fail(path, where, f"{{{ref}}} names no path")
            continue
        if command is None or ref not in command.groupindex:
            raise _fail(
                path,
                where,
                f"{{{ref}}} is not a group of the [match] command regex, nor output, "
                "input.<field> or json.<path>",
            )


def _load_hook(path: Path, spec: Any) -> Hook:
    where = "hook"
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise _fail(path, where, "expected 'file.py:function'")
    file, _, function = spec.partition(":")
    if not file.endswith(".py") or "/" in file or "\\" in file or not function.isidentifier():
        raise _fail(path, where, "expected 'file.py:function', the file next to the rule")
    source = path.parent / file
    if not source.is_file():
        raise _fail(path, where, f"{source} does not exist")
    module_name = f"synapse_capture_hook_{source.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise _fail(path, where, f"{source} cannot be imported")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise _fail(path, where, f"importing {file} failed: {e!r}") from e
    func = getattr(module, function, None)
    if not callable(func):
        raise _fail(path, where, f"{file} has no function {function!r}")
    return cast(Hook, func)


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[Rule, ...]
    #: lower-case words of connector names the generic detector never captures
    deny_words: frozenset[str]

    def match(
        self, tool_name: str, tool_input: dict[str, Any], output: str
    ) -> tuple[Rule, dict[str, str]] | None:
        """The first rule whose match and required fields fill for this call, and its fields."""
        for rule in self.rules:
            if fields := rule.extract(tool_name, tool_input, output):
                return rule, fields
        return None


@functools.cache
def load_rules(operator_dir: str | None = None) -> RuleSet:
    """The shipped rules and deny list, overridden by name from ``operator_dir``."""
    rules: dict[str, Rule | None] = {}
    deny: frozenset[str] | None = None
    directories = [SHIPPED_DIR] + ([Path(operator_dir)] if operator_dir else [])
    for directory in directories:
        if not directory.is_dir():
            raise RuleError(f"SYNAPSE_CAPTURE_DIR={directory} is not a directory")
        for path in sorted(directory.glob("*.toml")):
            if path.name == DENY_FILE:
                deny = _load_deny(path)
            else:
                rules[path.stem] = _load_rule(path)
    assert deny is not None, f"{SHIPPED_DIR / DENY_FILE} is missing"
    return RuleSet(tuple(r for _, r in sorted(rules.items()) if r is not None), deny)


def active_rules() -> RuleSet:
    """The rule set of this process: the shipped rules, plus ``SYNAPSE_CAPTURE_DIR``."""
    return load_rules(os.environ.get("SYNAPSE_CAPTURE_DIR", "").strip() or None)
