#!/usr/bin/env python3
"""Synapse client installer. GET /install serves this file with the client inside.

    curl -sSL https://<your-synapse>/install | python3 -
    irm https://<your-synapse>/install | python -          # Windows PowerShell
    ... | python3 - https://<url-to-record>                # record a different server URL

It unpacks the client matching the server's version into ~/.synapse/lib/<version>/
(plugin/ and plugin-codex/, side by side as in the repository), points
~/.synapse/lib/CURRENT at it, writes the launcher ~/.synapse/bin/synapse_run.py and the
`synapse` shims next to it, records the server URL in ~/.synapse/config.json when none
is set, and makes sure ~/.synapse/bin is on PATH. Then: `synapse login`.

Re-running it (or `synapse update`) installs whatever version the server now serves;
the two most recent versions are kept, so a hook that is running mid-update still finds
its files. Nothing else on the machine is touched: harness configs are `synapse login`'s
and `synapse install`'s job.

Stdlib only, and ASCII only: Windows PowerShell 5 re-encodes text it pipes to a program.
SYNAPSE_HOME overrides ~/.synapse.
"""

import base64
import hashlib
import importlib.util
import io
import os
import shlex
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

# Filled in by the server (mcp_server/install_routes.py), one value per line.
SERVER = ""
VERSION = ""
SHA256 = ""
PAYLOAD = ""

_KEEP_VERSIONS = 2

_SH_SHIM = """#!/bin/sh
# synapse: the Synapse client CLI. Written by the Synapse installer; rewritten on update.
PY={python}
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)
exec "$PY" {runner} plugin/scripts/synapse_cli.py "$@"
"""

_CMD_SHIM = """@echo off\r
REM synapse: the Synapse client CLI. Written by the Synapse installer; rewritten on update.\r
REM ASCII only: cmd reads a batch file in the console code page, so a path that may hold\r
REM other characters comes from a variable (%~dp0, %LOCALAPPDATA%, ...) expanded at run time.\r
setlocal\r
set "PY={python}"\r
if defined PY if exist "%PY%" goto run\r
REM The interpreter the installer ran under is gone: the python on PATH, never one in the\r
REM current directory.\r
set "NoDefaultCurrentDirectoryInExePath=1"\r
set "PY=python"\r
:run\r
REM The call and its jump on one line: cmd reads a running batch file again from a saved\r
REM byte offset after each command, and `synapse update` rewrites this file meanwhile.\r
REM goto finds its label by name; that label and the exit after it stay last in every\r
REM version, and its name is written nowhere else (a stale offset could land on it).\r
"%PY%" "%~dp0synapse_run.py" plugin/scripts/synapse_cli.py %* & goto done\r
:done\r
exit /b %errorlevel%\r
"""

# Prefixes of the interpreter path the cmd shim writes as variables: a profile folder
# may hold characters the console code page can't, and a Python is usually under it.
# The names whose value doesn't depend on the bitness of the process come first, so
# they win a tie: a 32-bit Python sees ProgramFiles as "Program Files (x86)", which a
# 64-bit cmd expands to "Program Files".
_CMD_VARS = (
    "LOCALAPPDATA",
    "APPDATA",
    "USERPROFILE",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "ProgramFiles",
)


def _cmd_interpreter(python, environ):
    """``python`` as the cmd shim can embed it: ASCII, with the longest known folder
    prefix written as its variable; "" when that can't be done (the shim then runs the
    python on PATH)."""
    head, rest = "", python
    for var in _CMD_VARS:
        value = (environ.get(var) or "").replace("/", "\\").rstrip("\\")
        if (
            value
            and len(value) > len(python) - len(rest)
            and python.lower().startswith(value.lower() + "\\")
        ):
            head, rest = "%" + var + "%", python[len(value) :]
    if not rest.isascii() or any(c in rest for c in '%"') or any(ord(c) < 32 for c in rest):
        return ""
    return head + rest


def _say(text):
    print(text, flush=True)


def _home():
    return Path(os.path.expanduser(os.environ.get("SYNAPSE_HOME") or "~/.synapse"))


def _extract(data, dest):
    """Unpack the bundle into ``dest``, refusing any member that would land outside it."""
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError("bundle member outside the client: " + info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            mode = (info.external_attr >> 16) & 0o777
            if mode and os.name != "nt":
                os.chmod(target, mode | stat.S_IRUSR | stat.S_IWUSR)


def _write_atomic(path, text, newline="\n", mode=None, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as f:
            f.write(text)
        if mode is not None and os.name != "nt":
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _unpack(lib, data):
    """lib/VERSION holding exactly this bundle; reused when it already does."""
    target = lib / VERSION
    stamp = target / ".bundle-sha256"
    try:
        if stamp.read_text(encoding="ascii").strip() == SHA256:
            return target, False
    except OSError:
        pass
    lib.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + VERSION + ".", dir=str(lib)))
    try:
        _extract(data, staging)
        (staging / ".bundle-sha256").write_text(SHA256 + "\n", encoding="ascii")
        if not target.exists():
            os.replace(staging, target)
        else:
            old = lib / ("." + VERSION + ".old")
            shutil.rmtree(old, ignore_errors=True)
            os.replace(target, old)
            try:
                os.replace(staging, target)
            except OSError:
                os.replace(old, target)  # a file held open (Windows): keep the working copy
                raise
            shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target, True


def _version_key(name):
    out = []
    for part in name.replace("-", ".").replace("+", ".").split("."):
        out.append(int(part) if part.isdigit() else -1)
    return tuple(out)


def _prune(lib):
    """Keep the newest versions (CURRENT among them), drop older ones."""
    versions = sorted(
        (d for d in lib.iterdir() if d.is_dir() and not d.name.startswith(".")),
        key=lambda d: _version_key(d.name),
        reverse=True,
    )
    keep = {VERSION} | {d.name for d in versions[:_KEEP_VERSIONS]}
    for d in versions:
        if d.name not in keep:
            shutil.rmtree(d, ignore_errors=True)


def _write_bin(home, client):
    bin_dir = home / "bin"
    runner = bin_dir / "synapse_run.py"
    _write_atomic(runner, (client / "plugin" / "scripts" / "synapse_run.py").read_text("utf-8"))
    python = (sys.executable or "python3").replace("\\", "/")
    where = str(runner).replace("\\", "/")
    sh = _SH_SHIM.format(python=shlex.quote(python), runner=shlex.quote(where))
    native = os.path.normpath(sys.executable) if sys.executable else ""
    cmd = _CMD_SHIM.format(python=_cmd_interpreter(native, os.environ)).replace("\r\n", "\n")
    _write_atomic(bin_dir / "synapse", sh, mode=0o755)
    _write_atomic(bin_dir / "synapse.cmd", cmd, "\r\n", encoding="ascii")
    return bin_dir


def _record_server(client, server):
    """Store the server URL in config.json unless one is set (then only say so), or an
    older Claude plugin install has one: `synapse login` copies that one, exactly as the
    machine enrolled with it, and a second spelling of the same server would read as a
    switch to another."""
    spec = importlib.util.spec_from_file_location(
        "synapse_home", client / "plugin" / "scripts" / "synapse_home.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    claude = Path(os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"))
    if mod.plugin_options([claude / "settings.json"]).get("SYNAPSE_URL"):
        return
    current = mod.load().get("url")
    if not current:
        mod.update({"url": server})
        _say("Recorded the server " + server + " in " + str(mod.config_file()) + ".")
    elif current.rstrip("/") != server:
        _say(
            "Note: "
            + str(mod.config_file())
            + " points at "
            + current
            + "; to use "
            + server
            + " instead, run: synapse login --url "
            + server
            + " --reenroll"
        )


def _on_path(bin_dir):
    want = os.path.normcase(os.path.abspath(str(bin_dir)))
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and os.path.normcase(os.path.abspath(os.path.expanduser(entry))) == want:
            return True
    return False


def _ensure_path(bin_dir):
    if _on_path(bin_dir):
        return
    if os.name == "nt":
        if _add_to_windows_path(str(bin_dir)):
            _say("Added " + str(bin_dir) + " to your user PATH; open a new terminal to use it.")
            return
        _say("Add " + str(bin_dir) + " to your PATH to run `synapse`.")
        return
    shell = os.path.basename(os.environ.get("SHELL", ""))
    # macOS terminals start login shells: a login bash never reads ~/.bashrc, and reads
    # only the first of these that exists (creating ~/.bash_profile would hide ~/.profile).
    bash_rc = "~/.bashrc"
    if sys.platform == "darwin":
        bash_rc = next(
            (
                p
                for p in ("~/.bash_profile", "~/.bash_login", "~/.profile")
                if os.path.exists(os.path.expanduser(p))
            ),
            "~/.bash_profile",
        )
    rc = {"zsh": "~/.zshrc", "bash": bash_rc, "fish": "~/.config/fish/config.fish"}.get(
        shell, "your shell's startup file"
    )
    line = (
        "fish_add_path " + str(bin_dir)
        if shell == "fish"
        else 'export PATH="' + str(bin_dir) + ':$PATH"'
    )
    _say(str(bin_dir) + " is not on your PATH. Add this line to " + rc + ":\n    " + line)


def _add_to_windows_path(entry):
    """Append ``entry`` to the user PATH (HKCU\\Environment) and tell running programs."""
    try:
        import ctypes
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE
        ) as key:
            try:
                value, kind = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                value, kind = "", winreg.REG_EXPAND_SZ
            parts = [p for p in value.split(";") if p]
            if any(os.path.normcase(p) == os.path.normcase(entry) for p in parts):
                return True
            winreg.SetValueEx(key, "Path", 0, kind, ";".join([*parts, entry]))
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 2, 5000, None)
        return True
    except Exception:
        return False


def main(argv):
    # It runs under whatever python3 the machine has, not the server's 3.12.
    if sys.version_info < (3, 11):  # noqa: UP036
        _say("Synapse needs Python 3.11 or newer; this is " + sys.version.split()[0] + ".")
        return 1
    if not PAYLOAD:
        _say("This is the installer template; fetch it from your server's /install instead.")
        return 1
    server = (argv[0] if argv else SERVER).strip().rstrip("/")
    data = base64.b64decode(PAYLOAD)
    if hashlib.sha256(data).hexdigest() != SHA256:
        _say("The download is damaged (checksum mismatch); nothing was changed.")
        return 1
    home = _home()
    lib = home / "lib"
    try:
        client, fresh = _unpack(lib, data)
        bin_dir = _write_bin(home, client)
        # Last: until CURRENT names it, every hook and shim keeps running the old version.
        _write_atomic(lib / "CURRENT", VERSION + "\n")
    except (OSError, ValueError) as e:
        _say("Install failed, the previous client (if any) is unchanged: " + str(e))
        return 1
    _prune(lib)
    _say(
        ("Installed" if fresh else "Already installed:")
        + " Synapse client "
        + VERSION
        + " in "
        + str(client)
        + "."
    )
    if server:
        _record_server(client, server)
    _ensure_path(bin_dir)
    _say("Now run: synapse login")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
