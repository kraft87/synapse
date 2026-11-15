"""How the Claude Code plugin starts Python: plugin/hooks/hooks.json, the MCP headers
helper in plugin.json and plugin/bin/synapse-python (with its cmd twin).

Claude Code runs a plugin's shell-form hooks with `sh -c` on Linux and macOS and with
Git Bash on Windows, and its headers helper with cmd on Windows. On Windows `python3`
is often the Microsoft Store stand-in that only prints "Python was not found", so:

  * every hook command is `[ "$OS" = Windows_NT ] && . synapse-python ...; exec python3 ...`:
    off Windows that is the old `python3 <script>` exec'd, nothing else started
  * synapse-python takes the first of python3, python and `py -3` that runs Python 3.11
    or newer, remembers it, and execs it once: stdin, stdout and the exit status are
    the script's own, a failing script is never run again under another interpreter
  * with no usable Python, one line names the fix (the board hook), the other hooks
    exit 0 silently (--quiet), and --check says what `synapse status` reports
  * the headers helper and the bin/ shims go through the same launcher on Windows

Fake interpreters stand in for the Store stand-in and for an older Python; the real one
is this test's own interpreter. The Windows branch runs under bash, as under Git Bash.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "plugin"
_LAUNCHER = _PLUGIN / "bin" / "synapse-python"


def _shells() -> tuple[str | None, str | None, str]:
    """(bash, sh, the folder of dirname). On Windows those of Git for Windows: its bash is
    what Claude Code runs hooks with, and System32's bash.exe is WSL's."""
    git = shutil.which("git")
    if os.name == "nt":
        if not git:
            return None, None, ""
        root = Path(git).resolve().parents[1]
        bash, sh = root / "bin" / "bash.exe", root / "bin" / "sh.exe"
        return (
            str(bash) if bash.is_file() else None,
            str(sh) if sh.is_file() else None,
            str(root / "usr" / "bin"),
        )
    return shutil.which("bash"), shutil.which("sh"), os.path.dirname(shutil.which("dirname") or "")


_BASH, _SH, _TOOLS = _shells()
#: Windows Python does not start without these.
_SYSTEM = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ}
_NO_PYTHON = "Synapse: no Python 3.11 or newer found (tried python3, python and py -3)."

needs_bash = pytest.mark.skipif(_BASH is None or _SH is None, reason="needs bash and sh")


def _hooks() -> list[tuple[str, dict]]:
    config = json.loads((_PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    return [
        (event, hook)
        for event, groups in config["hooks"].items()
        for group in groups
        for hook in group["hooks"]
        if hook["type"] == "command"
    ]


_SHAPE = re.compile(
    r'\[ "\$OS" = Windows_NT \] && \. "\$CLAUDE_PLUGIN_ROOT/bin/synapse-python" '
    r'(?P<quiet>--quiet )?(?P<run>"\$CLAUDE_PLUGIN_ROOT/scripts/[a-z_]+\.py"[^;]*); '
    r"exec python3 (?P<posix>.*)"
)


def test_every_hook_runs_once_under_python3_or_the_windows_launcher():
    hooks = _hooks()
    assert len(hooks) == 12
    for event, hook in hooks:
        # Claude Code has no per-platform command field: commandWindows was never read.
        assert set(hook) == {"type", "command", "timeout"}, event
        m = _SHAPE.fullmatch(hook["command"])
        assert m, hook["command"]
        assert m["run"] == m["posix"], hook["command"]
        board = "board_block.py" in m["run"]
        # Only the board hook says "no Python"; the rest stay quiet, so a session
        # shows that line once.
        assert (m["quiet"] is None) == board, hook["command"]


def test_the_mcp_headers_helper_goes_through_the_launcher():
    plugin = json.loads((_PLUGIN / ".claude-plugin" / "plugin.json").read_text("utf-8"))
    helper = plugin["mcpServers"]["synapse"]["headersHelper"]
    # cwd is the plugin root; sh runs bin/synapse-python, cmd finds bin/synapse-python.cmd.
    assert helper == '"bin/synapse-python" scripts/mcp_headers.py'
    assert "||" not in helper


def test_the_launchers_keep_their_line_endings_and_mode():
    assert b"\r" not in _LAUNCHER.read_bytes()
    cmd = (_PLUGIN / "bin" / "synapse-python.cmd").read_bytes()
    assert cmd.count(b"\n") == cmd.count(b"\r\n") and cmd.isascii()
    if os.name != "nt":
        assert os.access(_LAUNCHER, os.X_OK)


def test_slash_commands_never_name_python3():
    for md in (_PLUGIN / "commands").glob("*.md"):
        text = md.read_text(encoding="utf-8")
        assert 'python3 "${CLAUDE_PLUGIN_ROOT}' not in text, md.name


# ---------------------------------------------------------------------------
# Running it: a scratch plugin root, fake interpreters on a PATH of their own
# ---------------------------------------------------------------------------

_SCRIPT = """\
import json, os, sys
with open(os.environ["RUNS"], "a") as f:
    f.write("run\\n")
print(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read()}))
sys.exit(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0)
"""


def _exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def root(tmp_path):
    plugin = tmp_path / "plugin"
    (plugin / "bin").mkdir(parents=True)
    (plugin / "scripts").mkdir()
    shutil.copy2(_LAUNCHER, plugin / "bin" / "synapse-python")
    (plugin / "scripts" / "ingest_hook.py").write_text(_SCRIPT, encoding="utf-8")
    (plugin / "scripts" / "board_block.py").write_text(_SCRIPT, encoding="utf-8")
    return plugin


def _fakes(tmp_path: Path, *, stub: bool = True, old: bool = True, py: bool = True) -> Path:
    """python3 = the Store stand-in, python = Python 3.10 (prints nothing for the
    probe), py -3 = the real interpreter."""
    d = tmp_path / "fakebin"
    d.mkdir(exist_ok=True)
    if stub:
        _exe(d / "python3", 'echo "Python was not found; run without arguments" >&2\nexit 9009\n')
    if old:
        _exe(d / "python", "exit 0\n")
    if py:
        _exe(d / "py", f'[ "$1" = -3 ] && shift\nexec "{sys.executable}" "$@"\n')
    return d


def _hook_command(script: str) -> str:
    for _event, hook in _hooks():
        if f"/scripts/{script}" in hook["command"]:
            return hook["command"]
    raise AssertionError(script)


def _run(shell, command, root, tmp_path, *, path, os_name=None, stdin="{}", args=()):
    env = {
        **_SYSTEM,
        "PATH": str(path),
        "CLAUDE_PLUGIN_ROOT": str(root),
        "RUNS": str(tmp_path / "runs"),
        "TMP": str(tmp_path),
        "HOME": str(tmp_path),
    }
    if os_name:
        env["OS"] = os_name
    return subprocess.run(
        [shell, "-c", command, "hook", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def _runs(tmp_path: Path) -> int:
    runs = tmp_path / "runs"
    return len(runs.read_text().splitlines()) if runs.exists() else 0


@needs_bash
@pytest.mark.skipif(os.name == "nt", reason="the POSIX branch: sh with a real python3")
def test_off_windows_the_hook_is_python3_with_its_stdin_stdout_and_exit(root, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    _exe(real / "python3", f'exec "{sys.executable}" "$@"\n')
    command = _hook_command("ingest_hook.py").replace(
        '/scripts/ingest_hook.py"', '/scripts/ingest_hook.py" 3'
    )
    done = _run(_SH, command, root, tmp_path, path=real, stdin='{"event": "Stop"}')
    assert done.returncode == 3
    assert json.loads(done.stdout) == {"argv": ["3"], "stdin": '{"event": "Stop"}'}
    assert _runs(tmp_path) == 1


@needs_bash
def test_on_windows_the_store_stand_in_and_an_old_python_are_skipped(root, tmp_path):
    command = _hook_command("ingest_hook.py").replace(
        '/scripts/ingest_hook.py"', '/scripts/ingest_hook.py" 4'
    )
    path = _fakes(tmp_path)
    done = _run(_BASH, command, root, tmp_path, path=path, os_name="Windows_NT", stdin="in")
    assert done.returncode == 4, done.stderr
    assert json.loads(done.stdout) == {"argv": ["4"], "stdin": "in"}
    assert done.stderr == ""
    # Exactly once, though the script failed: no second interpreter is tried.
    assert _runs(tmp_path) == 1
    # Remembered: the next hook starts it directly, with nothing left on PATH.
    assert (tmp_path / "synapse-python").read_text().strip() == sys.executable
    again = _run(_BASH, command, root, tmp_path, path=tmp_path / "nothing", os_name="Windows_NT")
    assert again.returncode == 4 and _runs(tmp_path) == 2


@needs_bash
def test_a_remembered_python_that_is_gone_is_looked_up_again(root, tmp_path):
    (tmp_path / "synapse-python").write_text(str(tmp_path / "gone" / "python.exe") + "\n")
    done = _run(
        _BASH,
        _hook_command("ingest_hook.py"),
        root,
        tmp_path,
        path=_fakes(tmp_path),
        os_name="Windows_NT",
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "synapse-python").read_text().strip() == sys.executable


@needs_bash
def test_with_no_python_only_the_board_hook_says_so(root, tmp_path):
    path = _fakes(tmp_path, py=False)
    quiet = _run(
        _BASH, _hook_command("ingest_hook.py"), root, tmp_path, path=path, os_name="Windows_NT"
    )
    assert (quiet.returncode, quiet.stdout, quiet.stderr) == (0, "", "")
    board = _run(
        _BASH, _hook_command("board_block.py"), root, tmp_path, path=path, os_name="Windows_NT"
    )
    assert board.returncode == 1 and board.stdout == ""
    assert board.stderr.splitlines() == [
        _NO_PYTHON + " Install Python from python.org, then start a new session."
    ]
    assert "Python was not found" not in board.stderr
    assert _runs(tmp_path) == 0
    assert not (tmp_path / "synapse-python").exists()


@needs_bash
def test_check_prints_the_interpreter_or_the_problem(tmp_path):
    def check(path):
        env = {**_SYSTEM, "PATH": str(path), "TMP": str(tmp_path), "OS": "Windows_NT"}
        return subprocess.run(
            [_BASH, str(_LAUNCHER), "--check"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    none = check(_fakes(tmp_path, py=False))
    assert none.returncode == 1 and none.stderr.startswith(_NO_PYTHON)
    found = check(_fakes(tmp_path))
    assert found.returncode == 0 and found.stdout.strip() == sys.executable


@needs_bash
def test_the_bin_shims_use_the_launcher_on_windows(tmp_path):
    plugin = tmp_path / "plugin"
    shutil.copytree(_PLUGIN / "bin", plugin / "bin")
    (plugin / "scripts").mkdir()
    (plugin / "scripts" / "synapse_cli.py").write_text(_SCRIPT, encoding="utf-8")
    env = {
        **_SYSTEM,
        "PATH": str(_fakes(tmp_path)) + os.pathsep + _TOOLS,
        "TMP": str(tmp_path),
        "OS": "Windows_NT",
        "RUNS": str(tmp_path / "runs"),
    }
    done = subprocess.run(
        [_BASH, str(plugin / "bin" / "synapse-login"), "5"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 5, done.stderr
    assert json.loads(done.stdout)["argv"] == ["login", "5"]


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe")
def test_the_cmd_launcher_runs_a_script_once_with_its_exit_status(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    done = subprocess.run(
        ["cmd", "/d", "/c", str(_PLUGIN / "bin" / "synapse-python.cmd"), str(script), "6"],
        input="x",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "RUNS": str(tmp_path / "runs")},
    )
    assert done.returncode == 6, done.stderr
    assert json.loads(done.stdout) == {"argv": ["6"], "stdin": "x"}
    assert _runs(tmp_path) == 1
