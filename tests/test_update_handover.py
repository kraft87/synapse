"""`synapse update`, run for real: the client it installs refreshes the harnesses.

Two real processes on a scratch user: this checkout's CLI, and then the installed
client's CLI through the launcher. A local server's ``/install`` makes another client
version current. Hermes is the witness: its plugin folder is a copy of the core of
whichever client built it, so it shows whose code did the second half.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.helpers.client_core import PLUGIN, REPO

_INSTALLER = """\
import os, sys
from pathlib import Path
Path(os.environ["SYNAPSE_HOME"], "lib", "CURRENT").write_text("{version}\\n", encoding="utf-8")
print("installer: {version} is current for", sys.argv[1])
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = self.server.install if self.path == "/install" else b""  # type: ignore[attr-defined]
        self.send_response(200 if body else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def server() -> Iterator[ThreadingHTTPServer]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _client(lib: Path, version: str, *, rewire: bool = True) -> None:
    """This checkout's client unpacked as ``lib/<version>``, calling itself ``version``.
    Without ``rewire``, its CLI is one from before the ``update --rewire`` step: it
    refuses the flag, and its ``update`` says it ran."""
    root = lib / version
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PLUGIN, root / "plugin", ignore=skip)
    shutil.copytree(REPO / "plugin-codex", root / "plugin-codex", ignore=skip)
    manifest = root / "plugin" / ".claude-plugin" / "plugin.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ours = data["version"]
    manifest.write_text(json.dumps({**data, "version": version}), encoding="utf-8")
    core = root / "plugin" / "synapse_client" / "__init__.py"
    text = core.read_text(encoding="utf-8")
    assert text.count(f'"{ours}"') == 1
    core.write_text(text.replace(f'"{ours}"', f'"{version}"'), encoding="utf-8")
    if not rewire:
        (root / "plugin" / "scripts" / "synapse_cli.py").write_text(
            "import sys\n"
            "if sys.argv[1:] != ['update']:\n"
            "    sys.exit('unrecognized arguments: ' + ' '.join(sys.argv[1:]))\n"
            f"print('{version}: a whole update')\n"
            "sys.exit(7)\n",
            encoding="utf-8",
        )


def _update(tmp_path: Path, server: ThreadingHTTPServer, version: str, *, rewire: bool = True):
    """Run ``synapse update`` from this checkout for a scratch user whose server installs
    ``version``. Returns the finished process and the scratch Hermes home."""
    home = tmp_path / "user"
    synapse = home / ".synapse"
    lib = synapse / "lib"
    (synapse / "bin").mkdir(parents=True)
    shutil.copy(PLUGIN / "scripts" / "synapse_run.py", synapse / "bin" / "synapse_run.py")
    # What was current before: a client with nothing in it. A hand-over made before the
    # installer has finished would find no CLI to run.
    (lib / "0.0.1" / "plugin").mkdir(parents=True)
    (lib / "CURRENT").write_text("0.0.1\n", encoding="utf-8")
    _client(lib, version, rewire=rewire)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    (synapse / "config.json").write_text(json.dumps({"url": url}), encoding="utf-8")
    server.install = _INSTALLER.format(version=version).encode()  # type: ignore[attr-defined]

    # Hermes, wired by some earlier client.
    hermes = tmp_path / "hermes"
    folder = hermes / "plugins" / "synapse"
    (folder / "synapse_client" / "harnesses" / "hermes").mkdir(parents=True)
    (folder / "plugin.yaml").write_text("name: synapse\nversion: 0.0.1\n", encoding="utf-8")
    (folder / "synapse_client" / "harnesses" / "hermes" / "provider.py").write_text("")

    empty = tmp_path / "no-programs"  # no claude, codex or hermes to find
    empty.mkdir()
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "SYNAPSE_HOME": str(synapse),
        "HERMES_HOME": str(hermes),
        "CODEX_HOME": str(home / ".codex"),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "PATH": str(empty),
        "PYTHONDONTWRITEBYTECODE": "1",
        **{k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "TEMP", "TMP", "TMPDIR")},
    }
    done = subprocess.run(
        [sys.executable, str(PLUGIN / "scripts" / "synapse_cli.py"), "update"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done, hermes


def test_the_installed_client_rebuilds_the_hermes_folder_from_its_own_core(tmp_path, server):
    done, hermes = _update(tmp_path, server, "9.9.9")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("installer: 9.9.9 is current for http://127.0.0.1:")
    assert "Hermes: memory provider updated to 9.9.9" in lines[1:]
    folder = hermes / "plugins" / "synapse"
    assert 'version: "9.9.9"' in (folder / "plugin.yaml").read_text(encoding="utf-8")
    core = (folder / "synapse_client" / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "9.9.9"' in core


def test_an_older_installed_client_gets_a_whole_update_and_its_exit_code(tmp_path, server):
    """The server serves a client from before the step: it is asked for `update`, which
    it has had since the launcher existed, and this checkout's code builds nothing."""
    done, hermes = _update(tmp_path, server, "0.24.8", rewire=False)
    assert done.returncode == 7, done.stderr
    assert done.stdout.splitlines()[1:] == ["0.24.8: a whole update"]
    folder = hermes / "plugins" / "synapse"
    assert (folder / "plugin.yaml").read_text(encoding="utf-8") == "name: synapse\nversion: 0.0.1\n"
