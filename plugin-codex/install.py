#!/usr/bin/env python3
"""Install the Synapse hooks + MCP server into a local Codex CLI.

The hooks run from ~/.codex/config.toml rather than from a Codex plugin, because a plugin
install would bring the Claude plugin's hooks (hooks/hooks.json) instead of these.
This installer wires the two halves directly:

  1. ``[[hooks.<Event>]]`` blocks for the six hook scripts.
  2. ``[mcp_servers.synapse]`` with a saved-credential HTTP headers helper.

Every command it writes has one portable shape: a bare interpreter token, then each path
in double quotes with forward slashes, and no other quoting. Codex runs hooks through the
user's shell (PowerShell by default on Windows, else cmd or sh) and the headers helper
through ``cmd /C`` on Windows and ``sh -c`` elsewhere, and that shape means the same thing
in all of them. The final file is parsed with tomllib before anything is written.

Under the managed client (``curl <server>/install | python3 -``) the commands call the
fixed launcher ``~/.synapse/bin/synapse_run.py`` with a path relative to the client, never
a versioned directory: Codex trusts a hook by a hash of its command text, so a command
naming ``lib/<version>/`` would switch itself off on every update until re-trusted in
/hooks. Run from a git checkout, they name that checkout's scripts directly.

Idempotent: a Synapse hook already present is rewritten in place only when its command
differs from what this run would write (an unreadable one, like the backslash paths older
versions wrote on Windows, is repaired), legacy matchers and MCP authentication are
upgraded, and unrelated configuration and hook trust records are preserved.

Usage:
    python3 plugin-codex/install.py [--synapse-url URL] [--dry-run]

``synapse install codex`` runs the same code. After installing, run ``codex`` once and
trust the new hooks via /hooks — Codex refuses to run unreviewed hooks.
"""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(os.path.expanduser("~/.codex/config.toml"))
CLIENT_ROOT = Path(__file__).resolve().parent.parent  # holds plugin/ and plugin-codex/
HOOKS_DIR = CLIENT_ROOT / "plugin-codex" / "hooks"
SCRIPTS_DIR = CLIENT_ROOT / "plugin-codex" / "scripts"
HEADERS_HELPER = "plugin/scripts/mcp_headers.py"

# Characters no one quoting style protects in sh, cmd and PowerShell alike: " ends the
# quotes, $ and ` expand inside double quotes in sh and PowerShell, % in cmd.
_UNQUOTABLE = re.compile(r"[\"$`%\x00-\x1f\x7f]")


def _quoted(value: str) -> str:
    text = value.replace("\\", "/")
    if _UNQUOTABLE.search(text):
        raise ValueError(
            f"cannot write a portable command for {text!r}: it contains a double quote, $, `, % or a control character"
        )
    return f'"{text}"'


class Launcher:
    """How a command line reaches a client script: through the managed launcher
    (``runner``) or by the script's own path under ``root``."""

    def __init__(
        self,
        python: str,
        *,
        root: Path = CLIENT_ROOT,
        runner: Path | None = None,
        helper_python: str | None = None,
    ):
        self.python, self.root, self.runner = python, root, runner
        self.helper_python = helper_python or python

    def rest(self, script: str, *args: str) -> str:
        """The command without its interpreter token."""
        head = (
            [_quoted(str(self.runner)), script]
            if self.runner
            else [_quoted(str(self.root / script))]
        )
        return " ".join([*head, *args])

    def command(self, script: str, *args: str, helper: bool = False) -> str:
        """The full command line; ``helper`` picks the interpreter for the headers helper
        (see ``helper_interpreter``)."""
        python = self.helper_python if helper else self.python
        return f"{python} {self.rest(script, *args)}"


def python_token() -> str:
    """The interpreter token for a command line. ``python3`` on POSIX. On Windows it is
    often absent or the Microsoft Store stub, so the first of ``py -3``, ``python``,
    ``python3`` that actually runs a 3.11+ interpreter."""
    if os.name != "nt":
        return "python3"
    probe = "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    for token in ("py -3", "python", "python3"):
        try:
            done = subprocess.run(
                [*token.split(), "-c", probe], capture_output=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return token
    return "python"


def helper_interpreter(token: str) -> str:
    """The interpreter for the headers helper. On Windows Codex runs it through cmd.exe
    with the session's directory (the repository the user opened) as cwd, and cmd looks
    there for a bare ``py``/``python`` before PATH: a py.bat in a cloned repository would
    run and could read the token. So there it is the interpreter's absolute path, quoted
    (cmd strips Codex's outer quotes and keeps these). Hooks keep the bare token: they
    run through PowerShell, which never runs a program from the current directory by its
    bare name, and a quoted first token is a string to PowerShell, not a command."""
    if os.name != "nt":
        return token
    first, _, rest = token.partition(" ")
    # shutil.which tries the current directory first too, unless this is set (3.12+);
    # and only an absolute hit means the same interpreter from every session directory.
    saved = os.environ.get("NoDefaultCurrentDirectoryInExePath")
    os.environ["NoDefaultCurrentDirectoryInExePath"] = "1"
    try:
        path = shutil.which(first)
    finally:
        if saved is None:
            os.environ.pop("NoDefaultCurrentDirectoryInExePath", None)
        else:
            os.environ["NoDefaultCurrentDirectoryInExePath"] = saved
    if not path or not ntpath.isabs(path) or _UNQUOTABLE.search(path.replace("\\", "/")):
        return token
    return f'"{path}"' + (f" {rest}" if rest else "")


def default_launcher() -> Launcher:
    """The managed launcher when this file runs from the managed client
    (<SYNAPSE_HOME>/lib/<version>/), else this checkout's own scripts."""
    home = CLIENT_ROOT.parent.parent
    runner = home / "bin" / "synapse_run.py"
    token = python_token()
    if CLIENT_ROOT.parent.name == "lib" and runner.is_file():
        return Launcher(token, runner=runner, helper_python=helper_interpreter(token))
    return Launcher(token, helper_python=helper_interpreter(token))


LAUNCHER: Launcher | None = None  # set by the caller; resolved lazily otherwise


def _launcher() -> Launcher:
    global LAUNCHER
    if LAUNCHER is None:
        LAUNCHER = default_launcher()
    return LAUNCHER


# One entry per hook block: (marker, event, matcher, script, args, timeout). The marker
# (the script filename plus its argument) identifies a Synapse hook already present.
_HOOKS: list[tuple[str, str, str | None, str, tuple[str, ...], int]] = [
    ("synapse_stop_hook.py", "Stop", None, "plugin-codex/hooks/synapse_stop_hook.py", (), 30),
    ("session_start.py", "SessionStart", None, "plugin-codex/hooks/session_start.py", (), 20),
    (
        "user_prompt_submit.py",
        "UserPromptSubmit",
        None,
        "plugin-codex/hooks/user_prompt_submit.py",
        (),
        5,
    ),
    (
        "pre_tool_use.py",
        "PreToolUse",
        "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$",
        "plugin-codex/hooks/pre_tool_use.py",
        (),
        5,
    ),
    (
        "post_tool_use.py",
        "PostToolUse",
        "mcp__.*__(recall|recall_full_turns)$",
        "plugin-codex/hooks/post_tool_use.py",
        (),
        5,
    ),
    (
        "private_mode.py --session-end",
        "SessionEnd",
        None,
        "plugin-codex/scripts/private_mode.py",
        ("--session-end",),
        3,
    ),
]


def _toml_str(value: str) -> str:
    """A TOML basic string. json.dumps escapes exactly what TOML requires, provided it
    leaves non-ASCII alone (ensure_ascii would split astral characters into surrogate
    escapes, which TOML rejects)."""
    return json.dumps(value, ensure_ascii=False)


def _block(event: str, matcher: str | None, command: str, timeout: int) -> str:
    lines = [f"[[hooks.{event}]]"]
    if matcher:
        lines.append(f"matcher = {_toml_str(matcher)}")
    lines += [
        f"[[hooks.{event}.hooks]]",
        'type = "command"',
        f"command = {_toml_str(command)}",
        f"timeout = {timeout}",
    ]
    return "\n".join(lines) + "\n"


def hook_blocks(launcher: Launcher | None = None) -> list[tuple[str, str]]:
    """(marker, TOML block) for every Synapse hook, as this install would write it."""
    launcher = launcher or _launcher()
    return [
        (marker, _block(event, matcher, launcher.command(script, *args), timeout))
        for marker, event, matcher, script, args, timeout in _HOOKS
    ]


def _marker_of(text: str) -> tuple | None:
    """The _HOOKS entry a ``command =`` line runs, if it is a Synapse hook: it names one of
    the scripts (and its argument) AND a Synapse location (this plugin's directory or the
    managed launcher), so a user's own hook that happens to share a filename is never
    taken for ours."""
    if "plugin-codex" not in text and "synapse_run.py" not in text:
        return None
    for entry in _HOOKS:
        name, _, arg = entry[0].partition(" ")
        if name in text and (not arg or arg in text):
            return entry
    return None


_COMMAND_LINE = re.compile(r"^([ \t]*command[ \t]*=[ \t]*)(.*?)[ \t]*$")


def _repoint(text: str, launcher: Launcher) -> tuple[str, set[str], int]:
    """Rewrite each Synapse hook ``command =`` line to the command this install writes.
    Returns (text, markers found, lines rewritten). A line TOML can't read (the
    backslash paths older versions wrote on Windows) is replaced too."""
    out, found, changed = [], set(), 0
    for line in text.splitlines(keepends=True):
        m = _COMMAND_LINE.match(line.rstrip("\r\n"))
        entry = _marker_of(line) if m else None
        if not m or not entry:
            out.append(line)
            continue
        marker, _event, _matcher, script, args, _timeout = entry
        found.add(marker)
        try:
            current = tomllib.loads("v = " + m.group(2))["v"]
        except (tomllib.TOMLDecodeError, KeyError):
            current = None
        rest = launcher.rest(script, *args)
        if isinstance(current, str) and (current == rest or current.endswith(" " + rest)):
            out.append(line)  # same target, interpreter aside: keep (its trust survives)
            continue
        ending = line[len(line.rstrip("\r\n")) :]
        out.append(m.group(1) + _toml_str(launcher.command(script, *args)) + ending)
        changed += 1
    return "".join(out), found, changed


_OLD_MATCHER = 'matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session)$"'
_PRE_TOOL_BLOCKS = re.compile(
    r"(?ms)^\[\[hooks\.PreToolUse\]\].*?(?=^\[(?!\[hooks\.PreToolUse\.hooks\]\])|\Z)"
)


def _hook_text(existing: str, launcher: Launcher, verb) -> str:
    """``existing`` with every Synapse hook present and pointing where this install
    points. ``verb(done, pending)`` picks the tense of the progress lines."""
    new_matcher = _OLD_MATCHER.replace("fetch_session)", "fetch_session|remember)")
    upgraded = _PRE_TOOL_BLOCKS.sub(
        lambda match: (
            match[0].replace(_OLD_MATCHER, new_matcher)
            if "pre_tool_use.py" in match[0]
            else match[0]
        ),
        existing,
    )
    if upgraded != existing:
        print(
            f"hooks: {verb('upgraded', 'would upgrade')} the Synapse session-id matcher to include remember"
        )
    updated, found, changed = _repoint(upgraded, launcher)
    if changed:
        print(
            f"hooks: {verb('re-pointed', 'would re-point')} {changed} Synapse hook command(s); "
            "Codex asks you to trust them again in /hooks"
        )
    missing = [block for marker, block in hook_blocks(launcher) if marker not in found]
    if missing:
        appended = "\n# Synapse hooks (installed by synapse plugin-codex/install.py)\n" + "\n".join(
            missing
        )
        updated = updated.rstrip("\n") + ("\n" if updated.strip() else "") + appended
        print(
            f"hooks: {verb('appended', 'would append')} {len(missing)} hook block(s) to {CONFIG_PATH}"
        )
    if updated == existing:
        print(f"hooks: all {len(_HOOKS)} blocks already present in {CONFIG_PATH} — skipping")
    return updated


def _synapse_table(text: str) -> dict:
    """The parsed [mcp_servers.synapse] table of ``text``. When the whole file doesn't
    parse (older Windows hook lines install() hasn't repaired yet), that table alone must:
    a custom connection there is never overwritten blind. Raises ValueError."""
    try:
        return tomllib.loads(text).get("mcp_servers", {}).get("synapse", {})
    except tomllib.TOMLDecodeError:
        pass
    table = re.search(r"(?m)^\[mcp_servers\.synapse\][ \t]*(?:#.*)?$", text)
    if not table:
        return {}
    next_table = re.search(r"(?m)^\[", text[table.end() :])
    end = table.end() + (next_table.start() if next_table else len(text) - table.end())
    try:
        return tomllib.loads(text[table.start() : end]).get("mcp_servers", {}).get("synapse", {})
    except tomllib.TOMLDecodeError:
        raise ValueError(
            f"[mcp_servers.synapse] in {CONFIG_PATH} is unreadable; left unchanged"
        ) from None


def _mcp_text(existing: str, synapse_url: str, launcher: Launcher) -> str:
    """``existing`` with [mcp_servers.synapse] pointing at ``synapse_url`` through the
    saved-credential helper. Raises ValueError on a custom connection it must not touch."""
    current = _synapse_table(existing)
    if current.get("command"):
        raise ValueError("Refusing to replace a custom Synapse stdio connection")
    if current.get("http_headers", {}).get("Authorization") or current.get(
        "env_http_headers", {}
    ).get("Authorization"):
        raise ValueError("Remove the custom Synapse Authorization override before migrating")
    url = synapse_url.rstrip("/")
    if not url.endswith("/mcp"):
        url += "/mcp"
    helper = launcher.command(HEADERS_HELPER, "--url", _quoted(url), helper=True)
    fields = f"url = {_toml_str(url)}\nhttp_headers_helper = {_toml_str(helper)}\n"
    table = re.search(r"(?m)^\[mcp_servers\.synapse\][ \t]*(?:#.*)?$", existing)
    if table:
        next_table = re.search(r"(?m)^\[", existing[table.end() :])
        end = table.end() + next_table.start() if next_table else len(existing)
        body = existing[table.end() : end]
        body = re.sub(
            r"(?m)^[ \t]*(?:url|bearer_token_env_var|http_headers_helper)[ \t]*=.*\n?",
            "",
            body,
        )
        return existing[: table.end()] + "\n" + fields + body.lstrip("\n") + existing[end:]
    if current:
        raise ValueError("Synapse config uses an unsupported table layout; left unchanged")
    sep = "\n\n" if existing.strip() else ""
    return existing.rstrip() + sep + "[mcp_servers.synapse]\n" + fields


def _commit(existing: str, updated: str, dry_run: bool) -> bool:
    """Validate and write; returns whether anything changed."""
    if updated == existing:
        return False
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{CONFIG_PATH} would not be valid TOML ({e}); left unchanged") from None
    if not dry_run:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(updated, encoding="utf-8")
    return True


def _verb(dry_run: bool):
    return lambda done, pending: pending if dry_run else done


def _read() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""


def install_hook(dry_run: bool, launcher: Launcher | None = None) -> None:
    existing = _read()
    _commit(existing, _hook_text(existing, launcher or _launcher(), _verb(dry_run)), dry_run)


def install_mcp(synapse_url: str, dry_run: bool, launcher: Launcher | None = None) -> None:
    existing = _read()
    changed = _commit(existing, _mcp_text(existing, synapse_url, launcher or _launcher()), dry_run)
    if not changed:
        print("mcp: saved-credential helper already configured")
    else:
        print(
            "mcp: would configure the saved-credential helper (no token copied into config)"
            if dry_run
            else "mcp: configured saved-credential helper"
        )


def install(synapse_url: str, dry_run: bool = False, launcher: Launcher | None = None) -> bool:
    """MCP entry and hooks in one validated write. Returns whether anything changed."""
    launcher = launcher or _launcher()
    existing = _read()
    # Hooks first: repairing older unreadable lines lets the MCP step read the table it
    # is about to edit, so its refusal to touch a custom connection holds either way.
    with_hooks = _hook_text(existing, launcher, _verb(dry_run))
    updated = _mcp_text(with_hooks, synapse_url, launcher)
    print(
        "mcp: saved-credential helper already configured"
        if updated == with_hooks
        else f"mcp: {'would configure' if dry_run else 'configured'} the saved-credential helper "
        "(no token copied into config)"
    )
    return _commit(existing, updated, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Synapse into Codex CLI")
    parser.add_argument(
        "--synapse-url",
        default=None,
        help="Synapse base URL (default: existing MCP URL, then the saved Synapse config)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (HOOKS_DIR / "synapse_stop_hook.py").exists():
        print(f"error: hook scripts not found under {HOOKS_DIR}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(SCRIPTS_DIR))
    from common import BASE_URL

    try:
        # The table alone: a file older Windows hook lines broke still has its URL.
        existing = _synapse_table(_read()).get("url")
    except ValueError:
        existing = None  # install() refuses the unreadable table itself
    url = args.synapse_url or existing or BASE_URL
    try:
        install(url, args.dry_run)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(
        "\nDone. Next: start `codex`, run /hooks, and trust the Synapse hooks.\n"
        "MCP and hooks share the saved Synapse device credential (~/.synapse/config.json);\n"
        "no token is copied into config.toml. Requires Codex with http_headers_helper support."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
