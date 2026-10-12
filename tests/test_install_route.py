"""GET /install hands a machine the server's own client, and the installer it serves sets
that machine up from nothing (#222).

Until now the Claude plugin was the only way to get the client, so a Codex-only or
CLI-only machine had none. Now ``curl -sSL <server>/install | python3 -`` fetches
mcp_server/install_script.py with plugin/ and plugin-codex/ zipped inside and unpacks
them into ~/.synapse/lib/<version>/ behind a stable launcher (bin/synapse_run.py), so a
harness config never names a versioned path and an update never breaks Codex's hook trust.

  * the route serves an ASCII, parseable, uncached script naming the origin the request
    came in on (forwarded headers honoured, garbage ones ignored), and a JSON 503 when
    the server has no client tree
  * bundle() is byte-for-byte deterministic, carries both trees with their exec bits and
    none of the bytecode, and refuses an unusable version
  * the installer unpacks side by side, writes CURRENT and the shims (LF for sh, CRLF for
    cmd), records the URL only when none is set, is idempotent, and refuses a damaged
    download or a bundle member outside the client without installing anything
  * pruning keeps the two newest versions plus CURRENT, compared numerically
  * the installed `synapse` shim runs the real CLI; synapse_run.py follows CURRENT, falls
    back to the newest version, refuses paths outside the client and sets SYNAPSE_HOME
    only when the caller has not

Stdlib + pytest + starlette's TestClient. The installer and the launcher run as real
subprocesses against scratch homes; nothing touches the network or the real HOME.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from starlette.testclient import TestClient

from mcp_server import install_routes

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_RUNNER_PY = _SCRIPTS / "synapse_run.py"
_REAL_VERSION = json.loads(
    (_REPO / "plugin" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
)["version"]
_MODULES = (
    "config",
    "enroll",
    "synapse_login",
    "harness",
    "synapse_cli",
    "synapse_home",
    "synapse_run",
)

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

_SERVER = "https://syn.example.net"

posix_only = pytest.mark.skipif(os.name == "nt", reason="runs the sh shim and checks modes")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Scratch HOME, Claude config dir, data dir and cwd; no Synapse settings in the
    environment; a fresh bundle cache, so one test's tree never answers for another's."""
    monkeypatch.setenv("HOME", str(tmp_path / "home-dir"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(install_routes, "_cache", {})
    for mod in _MODULES:
        sys.modules.pop(mod, None)
    yield
    for mod in _MODULES:
        sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tree(root: Path, version: str = "1.0.0", *, extra: dict[str, str] | None = None) -> Path:
    """A minimal client tree: the two scripts the installer itself needs (the launcher
    and synapse_home.py), an executable launcher, a Codex file, and bytecode litter."""
    files = {
        "plugin/.claude-plugin/plugin.json": json.dumps({"name": "synapse", "version": version}),
        "plugin/scripts/synapse_run.py": _RUNNER_PY.read_text(encoding="utf-8"),
        "plugin/scripts/synapse_home.py": (_SCRIPTS / "synapse_home.py").read_text("utf-8"),
        "plugin/bin/synapse": "#!/bin/sh\necho synapse\n",
        "plugin-codex/install.py": "print('codex')\n",
        "plugin/scripts/__pycache__/synapse_home.cpython-312.pyc": "bytecode",
        "plugin-codex/stray.pyc": "bytecode",
        **(extra or {}),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    os.chmod(root / "plugin" / "bin" / "synapse", 0o755)
    os.chmod(root / "plugin" / ".claude-plugin" / "plugin.json", 0o644)
    return root


def _render(path: Path, server: str = _SERVER, root: Path | None = None) -> Path:
    path.write_text(install_routes.render(server, root), encoding="utf-8")
    return path


def _env(home: Path | None, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SYNAPSE_HOME", None)
    if home is not None:
        env["SYNAPSE_HOME"] = str(home)
    env["SHELL"] = "/bin/bash"  # a known shell, so the PATH advice is predictable
    # The installer defers to an older Claude plugin's saved URL: never the real one's.
    env["CLAUDE_CONFIG_DIR"] = str((home or Path(env.get("HOME", "/nonexistent"))) / "no-claude")
    env.update(extra)
    return env


def _install(
    script: Path, home: Path | None, *args: str, stdin: bool = False, **env: str
) -> subprocess.CompletedProcess:
    """Run a rendered installer the way a user does: as a file, or piped into `python -`."""
    if stdin:
        cmd = [sys.executable, "-", *args]
        text = script.read_text(encoding="utf-8")
    else:
        cmd = [sys.executable, str(script), *args]
        text = None
    return subprocess.run(
        cmd,
        input=text,
        env=_env(home, **env),
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _constants(text: str) -> dict[str, object]:
    """The script's top-level literal assignments (SERVER, VERSION, SHA256, PAYLOAD)."""
    out = {}
    for node in ast.parse(text).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            out[node.targets[0].id] = node.value.value
    return out


def _zip_names(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.namelist()


def _zip_mode(data: bytes, name: str) -> int:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return (zf.getinfo(name).external_attr >> 16) & 0o777


def _client():
    from fastmcp import FastMCP

    mcp = FastMCP("test-install")
    install_routes.register(mcp)
    return TestClient(mcp.http_app())


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("synapse_run", _RUNNER_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------


def test_install_serves_an_ascii_python_script_naming_the_origin_it_was_fetched_from():
    with _client() as client:
        r = client.get("/install")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/x-python")
    # A fresh fetch must reach the server: `synapse update` relies on it, and the
    # script names the origin of this one request.
    assert r.headers["cache-control"] == "no-store"
    # Windows PowerShell 5 re-encodes what it pipes; ASCII survives that.
    text = r.content.decode("ascii")
    ast.parse(text)
    assert 'SERVER = "http://testserver"\n' in text
    assert _constants(text)["SERVER"] == "http://testserver"


def test_the_served_script_carries_this_servers_client_and_its_checksum():
    with _client() as client:
        values = _constants(client.get("/install").text)
    data = base64.b64decode(values["PAYLOAD"])
    assert values["VERSION"] == _REAL_VERSION
    assert values["SHA256"] == hashlib.sha256(data).hexdigest()
    assert (values["VERSION"], data, values["SHA256"]) == install_routes.bundle()


@pytest.mark.parametrize(
    ("headers", "server"),
    [
        ({"x-forwarded-proto": "https", "x-forwarded-host": "mem.example"}, "https://mem.example"),
        (
            {"x-forwarded-proto": "https, http", "x-forwarded-host": "mem.example, app.internal"},
            "https://mem.example",
        ),
        (
            {"x-forwarded-proto": "https", "x-forwarded-host": "mem.example:8443"},
            "https://mem.example:8443",
        ),
        ({"x-forwarded-proto": "https", "x-forwarded-host": "[::1]:8765"}, "https://[::1]:8765"),
        ({"x-forwarded-proto": "https"}, "https://testserver"),
        ({"x-forwarded-host": "mem.example"}, "http://mem.example"),
    ],
)
def test_a_reverse_proxys_forwarded_origin_is_the_one_written_in(headers, server):
    with _client() as client:
        r = client.get("/install", headers=headers)
    assert r.status_code == 200
    assert _constants(r.text)["SERVER"] == server


@pytest.mark.parametrize(
    "headers",
    [
        {"x-forwarded-proto": "https", "x-forwarded-host": "mem.example/evil"},
        {"x-forwarded-proto": "https", "x-forwarded-host": "user@mem.example"},
        {"x-forwarded-proto": "https", "x-forwarded-host": "mem.example:123456"},
        {"x-forwarded-proto": "https", "x-forwarded-host": 'm"+__import__("os").system("PWNED")'},
        {"x-forwarded-proto": "javascript", "x-forwarded-host": "mem.example"},
        {"x-forwarded-proto": "ftp", "x-forwarded-host": "mem.example"},
    ],
)
def test_a_garbage_forwarded_header_falls_back_to_the_request_base_url(headers):
    with _client() as client:
        r = client.get("/install", headers=headers)
    assert r.status_code == 200
    assert _constants(r.text)["SERVER"] == "http://testserver"
    assert "PWNED" not in r.text


def test_a_server_without_a_client_tree_answers_503_json_not_a_broken_script(monkeypatch, tmp_path):
    empty = tmp_path / "no-client"
    empty.mkdir()
    monkeypatch.setattr(install_routes, "_CLIENT_ROOT", empty)
    with _client() as client:
        r = client.get("/install")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["status"] == "error"


def test_render_keeps_any_server_value_a_plain_string_literal(tmp_path):
    """Whatever origin a request claims, it reaches the script only as data."""
    tree = _tree(tmp_path / "client")
    hostile = 'http://x"\nimport os; os.system("PWNED")  # \\'
    text = install_routes.render(hostile, tree)
    values = _constants(text)
    assert values["SERVER"] == hostile
    assert not any(line.startswith("import os; os.system") for line in text.splitlines())


def test_a_template_missing_a_placeholder_is_refused_rather_than_served_half_filled(
    monkeypatch, tmp_path
):
    template = tmp_path / "install_script.py"
    original = install_routes._TEMPLATE.read_text(encoding="utf-8")
    template.write_text(original.replace('SHA256 = ""\n', ""), encoding="utf-8")
    monkeypatch.setattr(install_routes, "_TEMPLATE", template)
    with pytest.raises(ValueError, match="SHA256"):
        install_routes.render(_SERVER, _tree(tmp_path / "client"))


# ---------------------------------------------------------------------------
# bundle()
# ---------------------------------------------------------------------------


def test_bundle_is_byte_identical_across_builds_of_the_same_tree(monkeypatch, tmp_path):
    tree = _tree(tmp_path / "client")
    first = install_routes.bundle(tree)
    assert install_routes.bundle(tree) is first  # built once per process
    # A rebuild after every mtime moved (a fresh image layer) still gives the same bytes.
    monkeypatch.setattr(install_routes, "_cache", {})
    for path in tree.rglob("*"):
        os.utime(path, (1_900_000_000, 1_900_000_000))
    second = install_routes.bundle(tree)
    assert second is not first
    assert second == first


def test_bundle_carries_both_real_client_trees():
    version, data, sha = install_routes.bundle()
    names = _zip_names(data)
    assert version == _REAL_VERSION
    assert sha == hashlib.sha256(data).hexdigest()
    for required in (
        "plugin/.claude-plugin/plugin.json",
        "plugin/bin/synapse",
        "plugin/scripts/synapse_run.py",
        "plugin/scripts/synapse_cli.py",
        "plugin/scripts/synapse_home.py",
        "plugin-codex/install.py",
        "plugin-codex/scripts/common.py",
    ):
        assert required in names
    assert all(n.startswith(("plugin/", "plugin-codex/")) for n in names)
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]


def test_bundle_leaves_out_bytecode_and_pycache(tmp_path):
    _, data, _ = install_routes.bundle(_tree(tmp_path / "client"))
    names = _zip_names(data)
    assert "plugin/scripts/synapse_home.py" in names
    assert "plugin/scripts/__pycache__/synapse_home.cpython-312.pyc" not in names
    assert "plugin-codex/stray.pyc" not in names


@posix_only
def test_bundle_keeps_each_files_permission_bits(tmp_path):
    _, data, _ = install_routes.bundle(_tree(tmp_path / "client"))
    assert _zip_mode(data, "plugin/bin/synapse") == 0o755
    assert _zip_mode(data, "plugin/.claude-plugin/plugin.json") == 0o644
    # The real launcher is executable in git, and must stay so in the bundle.
    _, real, _ = install_routes.bundle()
    assert _zip_mode(real, "plugin/bin/synapse") & 0o111


def test_bundle_takes_its_version_from_plugin_json(tmp_path):
    version, _, _ = install_routes.bundle(_tree(tmp_path / "client", "1.2.3+build-7"))
    assert version == "1.2.3+build-7"


@pytest.mark.parametrize("version", ["1.0/../../x", "1.0 beta", "", "1.0\n", "1.0\\x"])
def test_an_unusable_version_is_refused(tmp_path, version):
    with pytest.raises(ValueError, match="unusable client version"):
        install_routes.bundle(_tree(tmp_path / "client", version))


@pytest.mark.parametrize("version", [".", ".."])
def test_dot_and_dotdot_are_not_usable_versions(tmp_path, version):
    with pytest.raises(ValueError, match="unusable client version"):
        install_routes.bundle(_tree(tmp_path / "client", version))


# ---------------------------------------------------------------------------
# the installer, end to end
# ---------------------------------------------------------------------------


@posix_only
def test_a_fresh_install_unpacks_the_client_and_writes_the_shims(tmp_path):
    tree = _tree(tmp_path / "client", "1.0.0")
    script = _render(tmp_path / "install.py", root=tree)
    home = tmp_path / "home"
    r = _install(script, home)
    assert r.returncode == 0, r.stdout + r.stderr

    lib, bin_dir = home / "lib", home / "bin"
    client = lib / "1.0.0"
    assert (client / "plugin" / "scripts" / "synapse_run.py").is_file()
    assert (client / "plugin-codex" / "install.py").is_file()
    # The bundle's bytecode litter stayed behind (recording the URL imports
    # synapse_home.py from the client, so a real __pycache__ may appear there).
    assert not (client / "plugin-codex" / "stray.pyc").exists()
    assert not [p for p in client.rglob("*.pyc") if p.read_bytes() == b"bytecode"]
    assert os.access(client / "plugin" / "bin" / "synapse", os.X_OK)
    assert (lib / "CURRENT").read_text(encoding="utf-8") == "1.0.0\n"
    assert (client / ".bundle-sha256").read_text().strip() == install_routes.bundle(tree)[2]

    sh = (bin_dir / "synapse").read_bytes()
    assert sh.startswith(b"#!/bin/sh\n")
    assert b"\r" not in sh  # a CR in the shebang line breaks it on Linux/macOS
    assert os.access(bin_dir / "synapse", os.X_OK)
    cmd = (bin_dir / "synapse.cmd").read_bytes()
    assert cmd.startswith(b"@echo off\r\n")
    assert cmd.count(b"\n") == cmd.count(b"\r\n") > 1
    assert (bin_dir / "synapse_run.py").read_bytes() == _RUNNER_PY.read_bytes()

    assert json.loads((home / "config.json").read_text(encoding="utf-8")) == {"url": _SERVER}
    assert stat.S_IMODE((home / "config.json").stat().st_mode) == 0o600

    out = r.stdout
    assert "Installed Synapse client 1.0.0" in out
    assert f"Recorded the server {_SERVER}" in out
    assert f'export PATH="{bin_dir}:$PATH"' in out and "~/.bashrc" in out
    assert out.rstrip().splitlines()[-1] == "Now run: synapse login"


def test_an_existing_different_url_is_kept_and_the_switch_command_is_printed(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    before = {"url": "https://old.example", "token": "tk-existing"}
    (home / "config.json").write_text(json.dumps(before), encoding="utf-8")
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    r = _install(script, home)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((home / "config.json").read_text(encoding="utf-8")) == before
    assert "Recorded" not in r.stdout
    assert f"synapse login --url {_SERVER} --reenroll" in r.stdout


def test_a_url_an_older_claude_plugin_saved_is_left_for_synapse_login_to_copy(tmp_path):
    """Two spellings of one server (localhost vs 127.0.0.1) would read as a move to
    another server at login; the machine enrolled with the plugin's, so that one wins."""
    home = tmp_path / "home"
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "settings.json").write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@synapse": {"options": {"SYNAPSE_URL": "http://127.0.0.1:8765"}}
                }
            }
        ),
        encoding="utf-8",
    )
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    r = _install(script, home, CLAUDE_CONFIG_DIR=str(claude))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (home / "config.json").exists()
    assert "Recorded" not in r.stdout


def test_the_windows_shim_is_ascii_and_reports_pythons_exit_code(tmp_path):
    """cmd reads a batch file in the console code page, so nothing in it may need UTF-8:
    the runner is found relative to the shim. And cmd expands %errorlevel% in a
    parenthesised block before the block runs, so the call and its exit are top level."""
    home = tmp_path / "home"
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    assert _install(script, home).returncode == 0
    raw = (home / "bin" / "synapse.cmd").read_bytes()
    assert raw.isascii()
    lines = raw.decode("ascii").split("\r\n")
    call = next(i for i, line in enumerate(lines) if "synapse_cli.py" in line)
    # The jump rides on the call's own line: `synapse update` rewrites this file while
    # cmd is still reading it, and cmd resumes at a byte offset, not a line.
    assert lines[call] == (
        '"%PY%" "%~dp0synapse_run.py" plugin/scripts/synapse_cli.py %* & goto done'
    )
    assert lines[-3:] == [":done", "exit /b %errorlevel%", ""]
    assert not any(line.lstrip().startswith(("(", ")")) or line.endswith("(") for line in lines)
    assert str(home) not in raw.decode("ascii")


def test_the_exit_label_is_written_nowhere_but_its_own_line():
    """After `synapse update` rewrites the running shim, cmd searches for the label from a
    stale byte offset: a comment spelling it could be taken for it."""
    shim = _installer_module()._CMD_SHIM
    assert shim.count(":done") == 1 and "\r\n:done\r\n" in shim


def test_the_windows_shim_falls_back_to_path_only_with_the_current_directory_off(tmp_path):
    """cmd looks for a bare python in the current directory before PATH, unless
    NoDefaultCurrentDirectoryInExePath is set."""
    home = tmp_path / "home"
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    assert _install(script, home).returncode == 0
    lines = (home / "bin" / "synapse.cmd").read_text(encoding="ascii").splitlines()
    bare = lines.index('set "PY=python"')
    assert 'set "NoDefaultCurrentDirectoryInExePath=1"' in lines[:bare]
    assert "py -3" not in "\n".join(lines)


def _installer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "synapse_installer_t", _REPO / "mcp_server" / "install_script.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_WIN_ENV = {
    "USERPROFILE": "C:\\Users\\Jos\u00e9",
    "LOCALAPPDATA": "C:\\Users\\Jos\u00e9\\AppData\\Local",
    "APPDATA": "C:\\Users\\Jos\u00e9\\AppData\\Roaming",
    "ProgramFiles": "C:\\Program Files",
    "ProgramW6432": "C:\\Program Files",
    "ProgramFiles(x86)": "C:\\Program Files (x86)",
}
# What a 32-bit process sees on 64-bit Windows (WOW64).
_WOW64_ENV = {**_WIN_ENV, "ProgramFiles": "C:\\Program Files (x86)"}


@pytest.mark.parametrize(
    ("python", "expected"),
    [
        # A per-user python.org install under a non-ASCII profile: the longest prefix wins.
        (
            "C:\\Users\\Jos\u00e9\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
            "%LOCALAPPDATA%\\Programs\\Python\\Python312\\python.exe",
        ),
        ("c:\\users\\JOS\u00c9\\.pyenv\\python.exe", "%USERPROFILE%\\.pyenv\\python.exe"),
        (
            "C:\\Program Files (x86)\\Python311\\python.exe",
            "%ProgramFiles(x86)%\\Python311\\python.exe",
        ),
        ("C:\\Python312\\python.exe", "C:\\Python312\\python.exe"),
        # Not a prefix at a folder boundary.
        ("C:\\Users\\Jos\u00e9X\\python.exe", ""),
        # Nothing ASCII can stand for it, or cmd would expand or end the string.
        ("D:\\Pyth\u00f6n\\python.exe", ""),
        ("C:\\50%\\python.exe", ""),
        ('C:\\a"b\\python.exe', ""),
        ("", ""),
    ],
)
def test_the_cmd_shim_interpreter_is_ascii_or_left_to_path(python, expected):
    assert _installer_module()._cmd_interpreter(python, _WIN_ENV) == expected


@pytest.mark.parametrize(
    ("env", "python", "expected"),
    [
        # A 32-bit Python: its ProgramFiles is Program Files (x86), which the 64-bit cmd
        # the user types `synapse` into expands to Program Files.
        (
            _WOW64_ENV,
            "C:\\Program Files (x86)\\Python312-32\\python.exe",
            "%ProgramFiles(x86)%\\Python312-32\\python.exe",
        ),
        (
            _WIN_ENV,
            "C:\\Program Files\\Python312\\python.exe",
            "%ProgramW6432%\\Python312\\python.exe",
        ),
        # 32-bit Windows has only ProgramFiles.
        (
            {"ProgramFiles": "C:\\Program Files"},
            "C:\\Program Files\\Python312\\python.exe",
            "%ProgramFiles%\\Python312\\python.exe",
        ),
    ],
)
def test_the_program_files_variable_does_not_depend_on_the_installers_bitness(
    env, python, expected
):
    assert _installer_module()._cmd_interpreter(python, env) == expected


def test_a_failed_swap_keeps_the_working_client(tmp_path, monkeypatch):
    """Reinstalling a version whose bundle changed renames the old copy aside first; if
    the new one can't take its place (a file held open on Windows), the old one goes back."""
    mod = _installer_module()
    lib = tmp_path / "lib"
    (lib / "1.0.0").mkdir(parents=True)
    (lib / "1.0.0" / "keep.txt").write_text("working client")
    tree = _tree(tmp_path / "client")
    _version, zipped, sha = install_routes.bundle(tree)
    monkeypatch.setattr(mod, "VERSION", "1.0.0")
    monkeypatch.setattr(mod, "SHA256", sha)
    real = os.replace

    def flaky(src, dst):
        if Path(dst).name == "1.0.0" and ".old" not in Path(src).name:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(mod.os, "replace", flaky)
    with pytest.raises(PermissionError):
        mod._unpack(lib, zipped)
    assert (lib / "1.0.0" / "keep.txt").read_text() == "working client"
    assert [p.name for p in lib.iterdir()] == ["1.0.0"]


def test_the_path_hint_names_the_file_a_macos_login_bash_reads(tmp_path, monkeypatch, capsys):
    mod = _installer_module()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    mod._ensure_path(tmp_path / "bin")
    assert "~/.bash_profile" in capsys.readouterr().out
    # Login bash reads only the first that exists: creating ~/.bash_profile next to a
    # ~/.profile (rustup writes one) would hide it.
    (home / ".profile").write_text('. "$HOME/.cargo/env"\n')
    mod._ensure_path(tmp_path / "bin")
    assert "Add this line to ~/.profile" in capsys.readouterr().out
    monkeypatch.setattr(mod.sys, "platform", "linux")
    mod._ensure_path(tmp_path / "bin")
    assert "~/.bashrc" in capsys.readouterr().out


def test_the_same_url_with_a_trailing_slash_is_left_alone_without_a_note(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"url": _SERVER + "/"}), encoding="utf-8")
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    r = _install(script, home)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((home / "config.json").read_text(encoding="utf-8")) == {"url": _SERVER + "/"}
    assert "Recorded" not in r.stdout and "synapse login --url" not in r.stdout


def test_rerunning_the_same_installer_is_idempotent(tmp_path):
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    home = tmp_path / "home"
    assert _install(script, home).returncode == 0
    # Something only the first unpack's directory holds: proves it was not replaced.
    marker = home / "lib" / "1.0.0" / "plugin" / "marker"
    marker.write_text("still here", encoding="utf-8")
    config_before = (home / "config.json").read_text(encoding="utf-8")

    r = _install(script, home)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Already installed: Synapse client 1.0.0" in r.stdout
    assert marker.is_file()
    assert sorted(p.name for p in (home / "lib").iterdir()) == ["1.0.0", "CURRENT"]
    assert (home / "config.json").read_text(encoding="utf-8") == config_before
    assert "Recorded" not in r.stdout and "synapse login --url" not in r.stdout


def test_a_different_bundle_under_the_same_version_replaces_the_old_one(tmp_path):
    first = _tree(tmp_path / "a", "1.0.0")
    second = _tree(tmp_path / "b", "1.0.0", extra={"plugin/scripts/new.py": "NEW = 1\n"})
    home = tmp_path / "home"
    assert _install(_render(tmp_path / "a.py", root=first), home).returncode == 0
    marker = home / "lib" / "1.0.0" / "plugin" / "marker"
    marker.write_text("old", encoding="utf-8")

    r = _install(_render(tmp_path / "b.py", root=second), home)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Installed Synapse client 1.0.0" in r.stdout
    client = home / "lib" / "1.0.0"
    assert not marker.exists()
    assert (client / "plugin" / "scripts" / "new.py").is_file()
    assert (client / ".bundle-sha256").read_text().strip() == install_routes.bundle(second)[2]
    # No staging or .old directory left behind.
    assert sorted(p.name for p in (home / "lib").iterdir()) == ["1.0.0", "CURRENT"]


def test_an_argv_url_overrides_the_server_baked_into_the_script(tmp_path):
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    home = tmp_path / "home"
    r = _install(script, home, "https://other.example/", stdin=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((home / "config.json").read_text(encoding="utf-8")) == {
        "url": "https://other.example"
    }
    assert _SERVER not in r.stdout


def test_without_synapse_home_it_installs_under_home_dot_synapse(tmp_path):
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    r = _install(script, None, HOME=str(tmp_path / "home-dir"))
    assert r.returncode == 0, r.stdout + r.stderr
    home = tmp_path / "home-dir" / ".synapse"
    assert (home / "lib" / "CURRENT").is_file()
    assert (home / "bin" / "synapse_run.py").is_file()
    assert json.loads((home / "config.json").read_text(encoding="utf-8")) == {"url": _SERVER}


def test_no_path_advice_when_the_bin_dir_is_already_on_path(tmp_path):
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    home = tmp_path / "home"
    path = os.pathsep.join([str(home / "bin"), os.environ.get("PATH", "")])
    r = _install(script, home, PATH=path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not on your PATH" not in r.stdout
    assert "Now run: synapse login" in r.stdout


def test_a_tampered_payload_is_refused_and_nothing_is_installed(tmp_path):
    script = _render(tmp_path / "install.py", root=_tree(tmp_path / "client"))
    text = script.read_text(encoding="utf-8")
    start = text.index('PAYLOAD = """\n') + len('PAYLOAD = """\n')
    swapped = "A" if text[start] != "A" else "B"
    script.write_text(text[:start] + swapped + text[start + 1 :], encoding="utf-8")
    home = tmp_path / "home"
    r = _install(script, home)
    assert r.returncode == 1
    assert "damaged" in r.stdout
    assert not home.exists()


def _malicious_bundle(evil_name: str, version: str = "6.6.6") -> tuple[str, bytes, str]:
    """A bundle that looks like a client but carries one member aimed outside it, last,
    so the members before it have already been unpacked when it is reached."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("plugin/.claude-plugin/plugin.json", json.dumps({"version": version}))
        zf.writestr("plugin/scripts/synapse_run.py", "# not the real launcher\n")
        zf.writestr("plugin/scripts/synapse_home.py", (_SCRIPTS / "synapse_home.py").read_text())
        zf.writestr(evil_name, "escaped\n")
    data = buf.getvalue()
    return version, data, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("evil", ["../../escape.txt", "plugin/../../../escape.txt", "ABSOLUTE"])
def test_a_bundle_member_outside_the_client_is_refused(monkeypatch, tmp_path, evil):
    home = tmp_path / "home"
    good = _render(tmp_path / "good.py", root=_tree(tmp_path / "client", "1.0.0"))
    assert _install(good, home).returncode == 0
    runner_before = (home / "bin" / "synapse_run.py").read_bytes()

    if evil == "ABSOLUTE":
        evil = str(tmp_path / "escape.txt")
    malicious = _malicious_bundle(evil)
    monkeypatch.setattr(install_routes, "bundle", lambda root=None: malicious)
    r = _install(_render(tmp_path / "evil.py"), home)

    assert r.returncode == 1
    assert "outside the client" in r.stdout
    assert not list(tmp_path.rglob("escape.txt"))
    # The previous client is still the one installed, and no staging dir is left.
    assert sorted(p.name for p in (home / "lib").iterdir()) == ["1.0.0", "CURRENT"]
    assert (home / "lib" / "CURRENT").read_text(encoding="utf-8") == "1.0.0\n"
    assert (home / "bin" / "synapse_run.py").read_bytes() == runner_before


def test_a_failed_install_leaves_current_on_the_previous_version(tmp_path):
    home = tmp_path / "home"
    old = _render(tmp_path / "old.py", root=_tree(tmp_path / "v1", "1.0.0"))
    new = _render(tmp_path / "new.py", root=_tree(tmp_path / "v2", "2.0.0"))
    assert _install(old, home).returncode == 0
    # Something in the way of the sh shim, so writing it fails (root ignores chmod).
    (home / "bin" / "synapse").unlink()
    (home / "bin" / "synapse" / "in-the-way").mkdir(parents=True)

    r = _install(new, home)
    assert r.returncode == 1
    assert "the previous client (if any) is unchanged" in r.stdout
    assert (home / "lib" / "CURRENT").read_text(encoding="utf-8") == "1.0.0\n"


def _install_versions(tmp_path: Path, home: Path, *versions: str) -> None:
    for version in versions:
        script = _render(
            tmp_path / f"install-{version}.py", root=_tree(tmp_path / version, version)
        )
        r = _install(script, home)
        assert r.returncode == 0, r.stdout + r.stderr


def _installed(home: Path) -> list[str]:
    return sorted(p.name for p in (home / "lib").iterdir() if p.is_dir())


def test_pruning_keeps_the_two_newest_versions_compared_numerically(tmp_path):
    home = tmp_path / "home"
    # As strings, 1.10.0 sorts below both others and 1.2.0 would survive.
    _install_versions(tmp_path, home, "1.2.0", "1.9.0", "1.10.0")
    assert _installed(home) == ["1.10.0", "1.9.0"]
    assert (home / "lib" / "CURRENT").read_text(encoding="utf-8") == "1.10.0\n"


def test_a_rolled_back_server_keeps_current_on_top_of_the_two_newest(tmp_path):
    home = tmp_path / "home"
    _install_versions(tmp_path, home, "2.0.0", "3.0.0", "1.0.0")
    assert _installed(home) == ["1.0.0", "2.0.0", "3.0.0"]
    assert (home / "lib" / "CURRENT").read_text(encoding="utf-8") == "1.0.0\n"
    _install_versions(tmp_path, home, "0.9.0")
    assert _installed(home) == ["0.9.0", "2.0.0", "3.0.0"]


@posix_only
def test_the_real_client_installs_and_its_shim_prints_the_version_from_a_path_with_spaces(
    tmp_path,
):
    script = _render(tmp_path / "install.py")
    home = tmp_path / "my home" / ".synapse"
    r = _install(script, home)
    assert r.returncode == 0, r.stdout + r.stderr
    client = home / "lib" / _REAL_VERSION
    assert (client / "plugin" / "scripts" / "synapse_cli.py").is_file()
    assert (client / "plugin-codex" / "install.py").is_file()
    assert os.access(client / "plugin" / "bin" / "synapse", os.X_OK)

    version = subprocess.run(
        [str(home / "bin" / "synapse"), "version"],
        env=_env(home),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert version.returncode == 0, version.stdout + version.stderr
    assert version.stdout.strip() == _REAL_VERSION


# ---------------------------------------------------------------------------
# synapse_run.py: which client, and running its scripts
# ---------------------------------------------------------------------------


def _lib(home: Path, *versions: str, current: str | None = None) -> Path:
    for version in versions:
        (home / "lib" / version / "plugin").mkdir(parents=True)
    (home / "lib").mkdir(parents=True, exist_ok=True)
    if current is not None:
        (home / "lib" / "CURRENT").write_text(current + "\n", encoding="utf-8")
    return home / "lib"


def test_client_root_follows_current_even_when_a_newer_version_is_unpacked(tmp_path):
    runner = _load_runner()
    lib = _lib(tmp_path / "home", "1.0.0", "2.0.0", current="1.0.0")
    assert runner.client_root(tmp_path / "home") == lib / "1.0.0"


@pytest.mark.parametrize("current", [None, "", "9.9.9", "../1.9.0", "1.9.0/plugin", "2.0.0"])
def test_client_root_falls_back_to_the_newest_version_when_current_is_missing_or_bogus(
    tmp_path, current
):
    runner = _load_runner()
    lib = _lib(tmp_path / "home", "1.2.0", "1.9.0", "1.10.0", current=current)
    (lib / "2.0.0").mkdir()  # a version dir with no plugin/ is not a client
    assert runner.client_root(tmp_path / "home") == lib / "1.10.0"


def test_a_leftover_staging_dir_never_outranks_a_real_version(tmp_path):
    runner = _load_runner()
    lib = _lib(tmp_path / "home", "1.0.0", ".1.10.0.k2j3h4")
    assert runner.client_root(tmp_path / "home") == lib / "1.0.0"


@pytest.mark.parametrize("layout", ["no lib", "empty lib", "no client in lib"])
def test_client_root_is_none_when_nothing_is_installed(tmp_path, layout):
    runner = _load_runner()
    home = tmp_path / "home"
    home.mkdir()
    if layout != "no lib":
        (home / "lib").mkdir()
    if layout == "no client in lib":
        (home / "lib" / "1.0.0").mkdir()
        (home / "lib" / "CURRENT").write_text("1.0.0\n", encoding="utf-8")
    assert runner.client_root(home) is None


_PROBE = """\
import json, os, sys
import probe_sibling  # importable only with this script's own dir on sys.path
print(json.dumps({
    "argv": sys.argv,
    "path0": sys.path[0],
    "name": __name__,
    "home": os.environ.get("SYNAPSE_HOME"),
    "sibling": probe_sibling.VALUE,
}))
raise SystemExit(int(os.environ.get("PROBE_EXIT", "0")))
"""

_ESCAPED = "print('ESCAPED')\n"


def _launcher_home(tmp_path: Path) -> Path:
    """A home as the installer leaves it: the launcher in bin/, CURRENT on 1.0.0 and a
    newer 2.0.0 beside it, each with a probe script; plus scripts outside the client."""
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    shutil.copy(_RUNNER_PY, home / "bin" / "synapse_run.py")
    for version in ("1.0.0", "2.0.0"):
        scripts = home / "lib" / version / "plugin" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "probe.py").write_text(_PROBE, encoding="utf-8")
        (scripts / "probe_sibling.py").write_text(f"VALUE = {version!r}\n", encoding="utf-8")
    (home / "lib" / "CURRENT").write_text("1.0.0\n", encoding="utf-8")
    (home / "lib" / "outside.py").write_text(_ESCAPED, encoding="utf-8")
    (home / "outside.py").write_text(_ESCAPED, encoding="utf-8")
    return home


def _launch(home: Path, *args: str, synapse_home: str | None = None, **env: str):
    environ = _env(None, **env)
    if synapse_home is not None:
        environ["SYNAPSE_HOME"] = synapse_home
    return subprocess.run(
        [sys.executable, str(home / "bin" / "synapse_run.py"), *args],
        env=environ,
        cwd=home.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_launcher_runs_a_current_client_script_with_its_args_and_dir_on_sys_path(
    tmp_path,
):
    home = _launcher_home(tmp_path)
    r = _launch(home, "plugin/scripts/probe.py", "a", "--b c")
    assert r.returncode == 0, r.stdout + r.stderr
    seen = json.loads(r.stdout)
    script = (home / "lib" / "1.0.0" / "plugin" / "scripts" / "probe.py").resolve()
    assert seen["argv"] == [str(script), "a", "--b c"]
    assert seen["path0"] == str(script.parent)
    assert seen["name"] == "__main__"
    assert seen["sibling"] == "1.0.0"


def test_the_scripts_exit_status_passes_through_the_launcher(tmp_path):
    home = _launcher_home(tmp_path)
    r = _launch(home, "plugin/scripts/probe.py", PROBE_EXIT="3")
    assert r.returncode == 3


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, "LAUNCHER_HOME"), ("/somewhere/else", "/somewhere/else")],
)
def test_synapse_home_is_set_to_the_launchers_own_home_only_when_unset(tmp_path, given, expected):
    home = _launcher_home(tmp_path)
    r = _launch(home, "plugin/scripts/probe.py", synapse_home=given)
    assert r.returncode == 0, r.stdout + r.stderr
    want = str(home.resolve()) if expected == "LAUNCHER_HOME" else expected
    assert json.loads(r.stdout)["home"] == want


@pytest.mark.parametrize(
    "target",
    [
        "../outside.py",
        "../../outside.py",
        "plugin/../../outside.py",
        "ABSOLUTE",
        "plugin/scripts/missing.py",
        "plugin/scripts",
        "",
    ],
)
def test_the_launcher_refuses_anything_but_a_script_inside_the_client(tmp_path, target):
    home = _launcher_home(tmp_path)
    if target == "ABSOLUTE":
        target = str(home / "outside.py")
    r = _launch(home, target)
    assert r.returncode == 1
    assert f"no such client script: {target}" in r.stderr
    assert "ESCAPED" not in r.stdout


def test_the_launcher_without_arguments_prints_its_usage(tmp_path):
    r = _launch(_launcher_home(tmp_path))
    assert r.returncode == 1
    assert "usage: synapse_run.py" in r.stderr


def test_the_launcher_with_no_client_installed_says_how_to_install_one(tmp_path):
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    shutil.copy(_RUNNER_PY, home / "bin" / "synapse_run.py")
    r = _launch(home, "plugin/scripts/synapse_cli.py", "version")
    assert r.returncode == 1
    assert "Synapse client not installed" in r.stderr
    assert "/install" in r.stderr
