"""plugin-codex/install.py writes commands every shell Codex uses reads the same way, and
repairs the ones older versions wrote (#222).

Codex runs hooks through the user's shell (PowerShell on Windows, else cmd or sh) and the
MCP headers helper through ``cmd /C`` or ``sh -c``. The 0.19 installer pasted the raw
checkout path into a TOML basic string, so on Windows ``python3 C:\\Users\\...`` held an
invalid ``\\U`` escape and Codex could not read config.toml at all, and its shlex-quoted
helper used single quotes that cmd does not understand. Every command now has one shape: a
bare interpreter token, then each path in double quotes with forward slashes.

  * Windows paths render to that shape and the file parses with tomllib
  * on POSIX the generated commands really run, through sh and bash, from paths with
    spaces, both directly and through the managed launcher (~/.synapse/bin/synapse_run.py)
  * the old unreadable Windows lines are repaired in place, with no duplicate blocks
  * hooks naming an old checkout are re-pointed once and trust records survive; a command
    that differs only in its interpreter is kept, because Codex trusts a hook by the hash
    of its command text
  * a user's own hooks, unquotable paths, dry runs and invalid results leave the file alone
  * default_launcher() and python_token() pick the managed form and a working interpreter

Stdlib + pytest. install.py is loaded by path with CONFIG_PATH on a scratch file; the only
subprocesses are the generated commands themselves, run against fake client scripts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PureWindowsPath
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_INSTALL_PY = _REPO / "plugin-codex" / "install.py"
_RUNNER_PY = _REPO / "plugin" / "scripts" / "synapse_run.py"
_MODULES = ("config", "enroll", "synapse_login", "harness", "synapse_cli", "synapse_home", "common")

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_MACHINE_ROLE",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_SKILLS_SYNC",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_SYNC",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_PATHS",
)

_URL = "https://memory.example/"  # trailing slash: the helper still gets exactly one /mcp
_MCP_URL = "https://memory.example/mcp"
_WIN_ROOT = PureWindowsPath(r"C:\Users\x y\.synapse\lib\0.20.0")
_WIN_RUNNER = PureWindowsPath(r"C:\Users\x y\.synapse\bin\synapse_run.py")

# Trust records Codex writes once the user approves hooks in /hooks.
_TRUST = (
    '\n[hooks.state."stop:0:0"]\ntrusted_hash = "sha256:1111"\n'
    '\n[hooks.state."session_start:0:0"]\ntrusted_hash = "sha256:2222"\n'
)

_FAKE_SCRIPT = (
    "import json, os, sys\n"
    "print(json.dumps({'argv': sys.argv, 'home': os.environ.get('SYNAPSE_HOME')}))\n"
)

posix_only = pytest.mark.skipif(os.name == "nt", reason="runs the commands through sh/bash")
_SHELLS = [
    pytest.param(name, marks=pytest.mark.skipif(shutil.which(name) is None, reason=f"no {name}"))
    for name in ("sh", "bash")
]


@pytest.fixture(autouse=True)
def _forget_modules():
    for mod in _MODULES:
        sys.modules.pop(mod, None)
    yield
    for mod in _MODULES:
        sys.modules.pop(mod, None)


@pytest.fixture
def installer(monkeypatch, tmp_path) -> ModuleType:
    """install.py loaded fresh, with HOME, cwd and every Synapse setting pointed at scratch."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("synapse_codex_install_portable", _INSTALL_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG_PATH = tmp_path / "codex" / "config.toml"
    return module


def _hook_commands(config: dict, event: str) -> list[str]:
    return [
        hook["command"]
        for group in config.get("hooks", {}).get(event, [])
        for hook in group["hooks"]
    ]


def _helper(config: dict) -> str:
    return config["mcp_servers"]["synapse"]["http_headers_helper"]


def _command_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("command =")]


def _old_install(hooks_dir: str, scripts_dir: str, helper: str, *, sep: str = "/") -> str:
    """config.toml as the 0.19 installer left it: the MCP table, then each hook block with
    the raw directory pasted into a basic string (backslashes and all on Windows)."""
    blocks = [
        ("Stop", None, f"{hooks_dir}{sep}synapse_stop_hook.py", 30),
        ("SessionStart", None, f"{hooks_dir}{sep}session_start.py", 20),
        ("UserPromptSubmit", None, f"{hooks_dir}{sep}user_prompt_submit.py", 5),
        (
            "PreToolUse",
            "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$",
            f"{hooks_dir}{sep}pre_tool_use.py",
            5,
        ),
        (
            "PostToolUse",
            "mcp__.*__(recall|recall_full_turns)$",
            f"{hooks_dir}{sep}post_tool_use.py",
            5,
        ),
        ("SessionEnd", None, f"{scripts_dir}{sep}private_mode.py --session-end", 3),
    ]
    text = (
        f'[mcp_servers.synapse]\nurl = "{_MCP_URL}"\n'
        f"http_headers_helper = {json.dumps(helper)}\n"
        "\n# Synapse hooks (installed by synapse plugin-codex/install.py)\n"
    )
    for event, matcher, target, timeout in blocks:
        text += f"[[hooks.{event}]]\n"
        if matcher:
            text += f'matcher = "{matcher}"\n'
        text += (
            f'[[hooks.{event}.hooks]]\ntype = "command"\n'
            f'command = "python3 {target}"\ntimeout = {timeout}\n\n'
        )
    return text


# --- Windows rendering ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("launcher_kwargs", "head"),
    [
        ({"root": _WIN_ROOT}, lambda script: [f"C:/Users/x y/.synapse/lib/0.20.0/{script}"]),
        (
            {"runner": _WIN_RUNNER},
            lambda script: ["C:/Users/x y/.synapse/bin/synapse_run.py", script],
        ),
    ],
    ids=["direct", "managed"],
)
def test_windows_paths_render_as_a_bare_interpreter_and_double_quoted_forward_slash_paths(
    installer, launcher_kwargs, head
):
    launcher = installer.Launcher("py -3", **launcher_kwargs)
    assert installer.install(_URL, launcher=launcher) is True
    text = installer.CONFIG_PATH.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    # The only backslashes left in the file escape the double quotes, so no \U can return.
    assert "\\" not in text.replace('\\"', "")

    expected = {}
    for _marker, event, _matcher, script, args, _timeout in installer._HOOKS:
        assert len(_hook_commands(config, event)) == 1
        expected[_hook_commands(config, event)[0]] = ["py", "-3", *head(script), *args]
    expected[_helper(config)] = [
        "py",
        "-3",
        *head(installer.HEADERS_HELPER),
        "--url",
        _MCP_URL,
    ]
    for command, argv in expected.items():
        assert "'" not in command and "\\" not in command
        # A bare first token: cmd /C strips a leading quote pair, and PowerShell would
        # take a leading quoted string for a value rather than a command to run.
        assert command.split(' "', 1)[0] == "py -3"
        assert f'"{argv[2]}"' in command
        # Without backslashes or single quotes, sh, cmd and PowerShell split alike.
        assert shlex.split(command) == argv
    assert _helper(config).endswith(f'--url "{_MCP_URL}"')
    assert config["mcp_servers"]["synapse"]["url"] == _MCP_URL


def test_the_managed_form_never_names_a_versioned_directory(installer):
    installer.install(_URL, launcher=installer.Launcher("py -3", runner=_WIN_RUNNER))
    config = tomllib.loads(installer.CONFIG_PATH.read_text(encoding="utf-8"))
    commands = [c for _, e, *_ in installer._HOOKS for c in _hook_commands(config, e)]
    for command in [*commands, _helper(config)]:
        assert "lib/" not in command and "0.20.0" not in command


# --- POSIX execution -----------------------------------------------------------------


def _fake_client(root: Path, installer: ModuleType) -> None:
    for script in [entry[3] for entry in installer._HOOKS] + [installer.HEADERS_HELPER]:
        path = root / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_FAKE_SCRIPT, encoding="utf-8")


@pytest.fixture
def shell_env(tmp_path) -> dict[str, str]:
    """An environment whose ``python3`` is this interpreter, as python_token() names it on
    POSIX, and with SYNAPSE_HOME unset so the launcher's own default shows."""
    bin_dir = tmp_path / "shim-bin"
    bin_dir.mkdir()
    shim = bin_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "SYNAPSE_HOME"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(shell: str, command: str, env: dict[str, str], cwd: Path) -> dict:
    done = subprocess.run(
        [shell, "-c", command], capture_output=True, text=True, env=env, cwd=cwd, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    return json.loads(done.stdout)


def _generated(installer: ModuleType) -> list[tuple[str, str, tuple[str, ...]]]:
    """(command, client-relative script, args) for every command install() wrote."""
    config = tomllib.loads(installer.CONFIG_PATH.read_text(encoding="utf-8"))
    out = []
    for _marker, event, _matcher, script, args, _timeout in installer._HOOKS:
        (command,) = _hook_commands(config, event)
        out.append((command, script, args))
    out.append((_helper(config), installer.HEADERS_HELPER, ("--url", _MCP_URL)))
    return out


@posix_only
@pytest.mark.parametrize("shell", _SHELLS)
def test_generated_commands_run_from_a_checkout_path_with_spaces(
    installer, tmp_path, shell_env, shell
):
    root = tmp_path / "my client (x86) & co"
    _fake_client(root, installer)
    installer.install(_URL, launcher=installer.Launcher("python3", root=root))

    for command, script, args in _generated(installer):
        assert command.startswith('python3 "')
        ran = _run(shell, command, shell_env, tmp_path)
        assert ran["argv"] == [str(root / script), *args]


@posix_only
@pytest.mark.parametrize("shell", _SHELLS)
def test_generated_commands_run_through_the_managed_launcher(installer, tmp_path, shell_env, shell):
    home = tmp_path / "x y (1)" / ".synapse"
    (home / "bin").mkdir(parents=True)
    shutil.copy(_RUNNER_PY, home / "bin" / "synapse_run.py")
    client = home / "lib" / "0.20.0"
    _fake_client(client, installer)
    (home / "lib" / "CURRENT").write_text("0.20.0\n", encoding="utf-8")
    runner = home / "bin" / "synapse_run.py"
    installer.install(_URL, launcher=installer.Launcher("python3", runner=runner))

    for command, script, args in _generated(installer):
        assert command.startswith(f'python3 "{runner}" {script}')
        ran = _run(shell, command, shell_env, tmp_path)
        assert ran["argv"] == [str((client / script).resolve()), *args]
        # Codex empties the helper's environment; the launcher names its own home.
        assert ran["home"] == str(home.resolve())


# --- Repair --------------------------------------------------------------------------

_OLD_WINDOWS_HELPER = (
    r"python3 'C:\Users\x\synapse\plugin-codex\scripts\mcp_headers.py' --url " + _MCP_URL
)
_UNRELATED = (
    'model = "gpt-5-codex"\n\n'
    "[projects.'C:\\Users\\x\\proj']\n"
    'trust_level = "trusted"\n\n'
    "[mcp_servers.other]\n"
    'command = "other-tool"\n\n'
)


@pytest.mark.parametrize("sep", ["/", "\\"], ids=["as-0.19-wrote-it", "all-backslashes"])
def test_old_unreadable_windows_lines_are_repaired_in_place(installer, capsys, sep):
    old = _UNRELATED + _old_install(
        r"C:\Users\x\synapse\plugin-codex\hooks",
        r"C:\Users\x\synapse\plugin-codex\scripts",
        _OLD_WINDOWS_HELPER,
        sep=sep,
    )
    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(old)
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(old, encoding="utf-8")
    launcher = installer.Launcher("py -3", runner=_WIN_RUNNER)

    assert installer.install(_URL, launcher=launcher) is True
    out = capsys.readouterr().out
    assert "re-pointed 6 Synapse hook command(s)" in out
    assert "appended" not in out
    text = installer.CONFIG_PATH.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    for _marker, event, _matcher, script, args, _timeout in installer._HOOKS:
        assert _hook_commands(config, event) == [launcher.command(script, *args)]
        assert text.count(f"[[hooks.{event}]]") == 1
    assert text.count("# Synapse hooks") == 1
    assert "'" not in _helper(config)
    assert _helper(config) == launcher.command(installer.HEADERS_HELPER, "--url", f'"{_MCP_URL}"')
    assert text.startswith(_UNRELATED)
    assert config["model"] == "gpt-5-codex"
    assert config["projects"] == {r"C:\Users\x\proj": {"trust_level": "trusted"}}
    assert config["mcp_servers"]["other"] == {"command": "other-tool"}

    assert installer.install(_URL, launcher=launcher) is False
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == text


def test_an_old_stop_only_install_is_repaired_and_completed(installer, capsys):
    """The first Codex installer wrote only the Stop hook."""
    old = (
        "[[hooks.Stop]]\n[[hooks.Stop.hooks]]\n"
        'type = "command"\n'
        'command = "python3 C:\\Users\\x\\synapse\\plugin-codex\\hooks/synapse_stop_hook.py"\n'
        "timeout = 30\n"
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(old, encoding="utf-8")
    launcher = installer.Launcher("py -3", runner=_WIN_RUNNER)

    installer.install(_URL, launcher=launcher)
    out = capsys.readouterr().out
    assert "re-pointed 1 Synapse hook command(s)" in out
    assert "appended 5 hook block(s)" in out
    config = tomllib.loads(installer.CONFIG_PATH.read_text(encoding="utf-8"))
    for _marker, event, _matcher, script, args, _timeout in installer._HOOKS:
        assert _hook_commands(config, event) == [launcher.command(script, *args)]


# --- Re-point ------------------------------------------------------------------------


def test_hooks_naming_an_old_checkout_move_to_the_managed_launcher_once(
    installer, tmp_path, capsys
):
    old = _old_install(
        "/home/me/src/synapse/plugin-codex/hooks",
        "/home/me/src/synapse/plugin-codex/scripts",
        f"python3 /home/me/src/synapse/plugin-codex/scripts/mcp_headers.py --url {_MCP_URL}",
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(old + _TRUST, encoding="utf-8")
    runner = tmp_path / "home" / ".synapse" / "bin" / "synapse_run.py"
    launcher = installer.Launcher("python3", runner=runner)

    assert installer.install(_URL, launcher=launcher) is True
    assert "re-pointed 6 Synapse hook command(s)" in capsys.readouterr().out
    text = installer.CONFIG_PATH.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    for _marker, event, _matcher, script, args, _timeout in installer._HOOKS:
        assert _hook_commands(config, event) == [launcher.command(script, *args)]
    assert "/home/me/src/synapse" not in text
    assert text.endswith(_TRUST)
    assert config["hooks"]["state"] == {
        "stop:0:0": {"trusted_hash": "sha256:1111"},
        "session_start:0:0": {"trusted_hash": "sha256:2222"},
    }

    assert installer.install(_URL, launcher=launcher) is False
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == text
    assert "already present" in capsys.readouterr().out


@pytest.mark.parametrize(
    "where",
    [{"root": _WIN_ROOT}, {"runner": _WIN_RUNNER}],
    ids=["direct", "managed"],
)
def test_a_hook_differing_only_in_its_interpreter_is_kept(installer, capsys, where):
    installer.install(_URL, launcher=installer.Launcher("python3", **where))
    installer.CONFIG_PATH.write_text(
        installer.CONFIG_PATH.read_text(encoding="utf-8") + _TRUST, encoding="utf-8"
    )
    before = installer.CONFIG_PATH.read_text(encoding="utf-8")
    capsys.readouterr()

    installer.install(_URL, launcher=installer.Launcher("py -3", **where))
    after = installer.CONFIG_PATH.read_text(encoding="utf-8")
    assert _command_lines(after) == _command_lines(before)
    assert all(line.startswith('command = "python3 ') for line in _command_lines(after))
    assert "re-point" not in capsys.readouterr().out
    assert after.endswith(_TRUST)
    tomllib.loads(after)


# --- Safety --------------------------------------------------------------------------


def test_a_users_own_hook_sharing_a_script_name_is_left_alone(installer, capsys):
    own = (
        "[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\n"
        'type = "command"\n'
        'command = "python3 /home/me/synapse-tools/session_start.py"\n'
        "timeout = 10\n\n"
        "[[hooks.SessionEnd]]\n[[hooks.SessionEnd.hooks]]\n"
        'type = "command"\n'
        'command = "python3 /home/me/bin/private_mode.py --session-end"\n'
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(own, encoding="utf-8")
    launcher = installer.Launcher("py -3", runner=_WIN_RUNNER)

    installer.install(_URL, launcher=launcher)
    out = capsys.readouterr().out
    assert "re-point" not in out
    assert "appended 6 hook block(s)" in out
    text = installer.CONFIG_PATH.read_text(encoding="utf-8")
    assert text.startswith(own)
    config = tomllib.loads(text)
    assert _hook_commands(config, "SessionStart") == [
        "python3 /home/me/synapse-tools/session_start.py",
        launcher.command("plugin-codex/hooks/session_start.py"),
    ]
    assert _hook_commands(config, "SessionEnd") == [
        "python3 /home/me/bin/private_mode.py --session-end",
        launcher.command("plugin-codex/scripts/private_mode.py", "--session-end"),
    ]


def test_quoted_writes_backslashes_as_forward_slashes(installer):
    assert installer._quoted(r"C:\Users\x y\a.py") == '"C:/Users/x y/a.py"'


@pytest.mark.parametrize(
    "char",
    ['"', "$", "`", "%", "\n", "\t", "\x00", "\x1b", "\x7f"],
    ids=["dquote", "dollar", "backtick", "percent", "lf", "tab", "nul", "esc", "del"],
)
def test_quoted_rejects_what_no_one_quoting_style_protects(installer, char):
    with pytest.raises(ValueError, match="cannot write a portable command"):
        installer._quoted(f"/home/a{char}b/script.py")


def test_an_apostrophe_is_literal_inside_double_quotes_everywhere(installer):
    """sh, PowerShell and cmd all take ' literally inside "...": a profile folder like
    C:\\Users\\Mary O'Brien must still get Codex wired."""
    quoted = installer._quoted("C:\\Users\\Mary O'Brien\\x.py")
    assert quoted == '"C:/Users/Mary O\'Brien/x.py"'
    assert shlex.split(quoted) == ["C:/Users/Mary O'Brien/x.py"]


@pytest.mark.parametrize(
    "where",
    [
        {"root": Path("/home/$USER/synapse")},
        {"root": PureWindowsPath(r"C:\Users\100%\synapse")},
        {"runner": Path("/home/`id`/.synapse/bin/synapse_run.py")},
        {"runner": PureWindowsPath(r'C:\Users\x"y\.synapse\bin\synapse_run.py')},
    ],
    ids=["root-dollar", "root-percent", "runner-backtick", "runner-dquote"],
)
def test_an_unquotable_path_raises_and_leaves_the_file_unchanged(installer, where):
    original = _UNRELATED + "[mcp_servers.synapse]\n" + f'url = "{_MCP_URL}"\n'
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="portable command"):
        installer.install(_URL, launcher=installer.Launcher("python3", **where))
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == original


def test_an_unquotable_path_creates_no_file(installer):
    with pytest.raises(ValueError):
        installer.install(_URL, launcher=installer.Launcher("python3", root=Path("/a$b")))
    assert not installer.CONFIG_PATH.exists()


def test_dry_run_writes_nothing(installer, capsys):
    launcher = installer.Launcher("py -3", runner=_WIN_RUNNER)
    assert installer.install(_URL, dry_run=True, launcher=launcher) is True
    assert not installer.CONFIG_PATH.exists()
    assert "would append 6 hook block(s)" in capsys.readouterr().out

    old = _old_install(
        r"C:\Users\x\synapse\plugin-codex\hooks",
        r"C:\Users\x\synapse\plugin-codex\scripts",
        _OLD_WINDOWS_HELPER,
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(old, encoding="utf-8")
    assert installer.install(_URL, dry_run=True, launcher=launcher) is True
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == old
    out = capsys.readouterr().out
    assert "would re-point 6" in out and "would configure" in out


def test_an_invalid_result_raises_and_leaves_the_file_unchanged(installer):
    # Unreadable for a reason install() does not own: it must not write over it.
    original = '[profiles.win]\ncwd = "C:\\Users\\x"\n'
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="would not be valid TOML"):
        installer.install(_URL, launcher=installer.Launcher("py -3", runner=_WIN_RUNNER))
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == original


def test_the_old_pre_tool_use_matcher_is_still_upgraded(installer, tmp_path, capsys):
    old_matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session)$"
    old = (
        f'[[hooks.PreToolUse]]\nmatcher = "{old_matcher}"\n'
        "[[hooks.PreToolUse.hooks]]\n"
        'type = "command"\n'
        'command = "python3 /home/me/src/synapse/plugin-codex/hooks/pre_tool_use.py"\n'
        "timeout = 5\n\n"
        f'[[hooks.PreToolUse]]\nmatcher = "{old_matcher}"\n'
        "[[hooks.PreToolUse.hooks]]\n"
        'type = "command"\n'
        'command = "other-hook"\n'
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(old, encoding="utf-8")
    launcher = installer.Launcher("python3", runner=tmp_path / "bin" / "synapse_run.py")

    installer.install(_URL, launcher=launcher)
    assert "upgraded the Synapse session-id matcher" in capsys.readouterr().out
    groups = tomllib.loads(installer.CONFIG_PATH.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]
    assert [(g["matcher"], g["hooks"][0]["command"]) for g in groups] == [
        (
            "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$",
            launcher.command("plugin-codex/hooks/pre_tool_use.py"),
        ),
        (old_matcher, "other-hook"),
    ]


@pytest.mark.parametrize(
    "custom",
    ['command = "my-stdio-server"', 'http_headers = { Authorization = "Bearer custom" }'],
    ids=["stdio", "authorization"],
)
def test_a_custom_synapse_connection_is_refused_even_beside_unreadable_old_hook_lines(
    installer, custom
):
    original = (
        f"[mcp_servers.synapse]\n{custom}\n\n"
        "[[hooks.Stop]]\n[[hooks.Stop.hooks]]\n"
        'type = "command"\n'
        'command = "python3 C:\\Users\\x\\synapse\\plugin-codex\\hooks/synapse_stop_hook.py"\n'
    )
    installer.CONFIG_PATH.parent.mkdir(parents=True)
    installer.CONFIG_PATH.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        installer.install(_URL, launcher=installer.Launcher("py -3", runner=_WIN_RUNNER))
    assert installer.CONFIG_PATH.read_text(encoding="utf-8") == original


# --- default_launcher() --------------------------------------------------------------


@pytest.mark.parametrize(
    ("parent", "runner_kind", "managed"),
    [
        ("lib", "file", True),
        ("lib", None, False),
        ("lib", "dir", False),
        ("src", "file", False),
    ],
    ids=["managed-client", "no-runner", "runner-not-a-file", "not-under-lib"],
)
def test_default_launcher_uses_the_runner_only_from_a_managed_client(
    installer, monkeypatch, tmp_path, parent, runner_kind, managed
):
    home = tmp_path / "x y" / ".synapse"
    client = home / parent / "0.20.0"
    client.mkdir(parents=True)
    runner = home / "bin" / "synapse_run.py"
    if runner_kind == "file":
        runner.parent.mkdir(parents=True)
        shutil.copy(_RUNNER_PY, runner)
    elif runner_kind == "dir":
        runner.mkdir(parents=True)
    monkeypatch.setattr(installer, "CLIENT_ROOT", client)
    monkeypatch.setattr(installer, "python_token", lambda: "python3")

    launcher = installer.default_launcher()
    assert launcher.python == "python3"
    command = launcher.command("plugin-codex/hooks/session_start.py")
    if managed:
        assert launcher.runner == runner
        assert command == f'python3 "{runner}" plugin-codex/hooks/session_start.py'
    else:
        assert launcher.runner is None
        assert "synapse_run.py" not in command
        assert command.endswith('/plugin-codex/hooks/session_start.py"')


# --- python_token() ------------------------------------------------------------------


def _fake_subprocess(outcomes: dict, calls: list) -> SimpleNamespace:
    """A stand-in for install.py's subprocess module: each candidate token's probe exits
    with (or raises) what ``outcomes`` says; unknown tokens exit 1."""

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        outcome = outcomes.get(" ".join(argv[:-2]), 1)
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(argv, outcome)

    return SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError)


def test_python_token_is_python3_on_posix_without_probing(installer, monkeypatch):
    calls: list = []
    monkeypatch.setattr(installer, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(installer, "subprocess", _fake_subprocess({}, calls))
    assert installer.python_token() == "python3"
    assert calls == []


@pytest.mark.parametrize(
    ("outcomes", "expected", "probed"),
    [
        ({"py -3": 0}, "py -3", ["py -3"]),
        (
            # 9009: the Microsoft Store stub that stands in for an absent python.
            {"py -3": FileNotFoundError(), "python": 9009, "python3": 0},
            "python3",
            ["py -3", "python", "python3"],
        ),
        ({"py -3": 1, "python": 0, "python3": 0}, "python", ["py -3", "python"]),
        (
            {
                "py -3": subprocess.TimeoutExpired("py", 15),
                "python": OSError(),
                "python3": 1,
            },
            "python",
            ["py -3", "python", "python3"],
        ),
    ],
    ids=["py-launcher", "store-stub-then-python3", "py-too-old", "none-runs"],
)
def test_python_token_on_windows_takes_the_first_candidate_that_runs(
    installer, monkeypatch, outcomes, expected, probed
):
    calls: list = []
    # A stand-in module, not the real os.name: with that patched, every pathlib.Path()
    # built anywhere in the process would try to be a WindowsPath.
    monkeypatch.setattr(installer, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(installer, "subprocess", _fake_subprocess(outcomes, calls))

    assert installer.python_token() == expected
    assert [" ".join(argv[:-2]) for argv, _ in calls] == probed
    for argv, kwargs in calls:
        assert argv[-2] == "-c"
        assert kwargs.get("timeout")  # a hanging stub must not hang the install
    probe = calls[0][0][-1]
    assert "(3, 11)" in probe
    assert subprocess.run([sys.executable, "-c", probe], timeout=60).returncode == 0
