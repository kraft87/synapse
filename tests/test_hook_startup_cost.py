"""The hooks that run on every prompt, turn and recall must not pay for imports they skip.

Each Claude/Codex hook is a fresh interpreter, so every module a hook imports is time the
user waits: UserPromptSubmit blocks the prompt, Stop blocks the end of the turn. These
tests run the real scripts in a subprocess and look at what ended up in ``sys.modules``,
which is stable where wall-clock timing is not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX_HOOKS = _REPO / "plugin-codex" / "hooks"

# What a hook that never talks to the server must not load: the HTTP stack is most of
# `import config`'s old cost.
_HTTP = ("urllib.request", "http.client", "ssl")


def _run(script: Path, tmp_path: Path, stdin: str = "{}", *args: str, watch: tuple[str, ...]):
    """Run ``script`` as __main__ in a fresh interpreter. Popen is stubbed, so a Stop parent's
    detached child is recorded rather than started. Returns (stdout, spawned argv lists,
    the ``watch`` modules that got imported)."""
    code = (
        "import json, runpy, subprocess, sys\n"
        "spawned = []\n"
        "subprocess.Popen = lambda argv, **kw: spawned.append(argv)\n"
        "script, watch = sys.argv[1], json.loads(sys.argv[2])\n"
        "sys.argv = [script, *sys.argv[3:]]\n"
        "try:\n"
        "    runpy.run_path(script, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "sys.stdout.flush()\n"
        "print('\\x00' + json.dumps({'spawned': spawned, "
        "'loaded': [m for m in watch if m in sys.modules]}))\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SYNAPSE_HOME": str(tmp_path / "home"),
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
    }
    done = subprocess.run(
        [sys.executable, "-c", code, str(script), json.dumps(list(watch)), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out, _, tail = done.stdout.partition("\x00")
    return out, json.loads(tail)["spawned"], json.loads(tail)["loaded"]


def test_importing_config_does_not_load_the_http_stack():
    """config.py is imported by every hook; urllib.request alone was ~20 ms of it."""
    code = (
        f"import sys; sys.path.insert(0, {str(_SCRIPTS)!r}); import config; "
        f"print([m for m in {_HTTP!r} if m in sys.modules])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_config_requests_still_work_with_the_deferred_import(tmp_path):
    """The deferred import is inside the helper that needs it: a real round trip."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    seen: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers["User-Agent"]))
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        code = (
            f"import sys; sys.path.insert(0, {str(_SCRIPTS)!r}); import config; "
            "print(config.get_json('/ping', {'a': 1}))"
        )
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "SYNAPSE_HOME": str(tmp_path / "home"),
            "SYNAPSE_URL": f"http://127.0.0.1:{httpd.server_port}",
        }
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
    finally:
        httpd.shutdown()
    assert done.stdout.strip() == "{'ok': True}"
    assert seen and seen[0][0] == "/ping?a=1"
    assert seen[0][1].startswith("synapse-plugin/") and not seen[0][1].endswith("/unknown")


@pytest.mark.parametrize("script", ["recall_nudge.py", "recall_feedback_nudge.py"])
def test_the_nudge_hooks_do_not_load_the_http_stack(script, tmp_path):
    _out, _, loaded = _run(_SCRIPTS / script, tmp_path, watch=_HTTP)
    assert loaded == []


def test_the_claude_stop_hook_spawns_before_importing_the_client(tmp_path):
    transcript = tmp_path / "s.jsonl"
    transcript.write_text("{}\n")
    _out, spawned, loaded = _run(
        _SCRIPTS / "ingest_hook.py",
        tmp_path,
        json.dumps({"transcript_path": str(transcript)}),
        watch=("config", "synapse_ingest_retry", *_HTTP),
    )
    assert loaded == []
    (argv,) = spawned
    assert argv[2:] == ["--ship", str(transcript)]


def test_the_codex_stop_hook_spawns_before_importing_the_client(tmp_path):
    payload = {"session_id": "s1", "transcript_path": "/x/rollout.jsonl", "cwd": "/w"}
    _out, spawned, loaded = _run(
        _CODEX_HOOKS / "synapse_stop_hook.py",
        tmp_path,
        json.dumps({**payload, "last_assistant_message": "A" * 10_000}),
        watch=("common", "config", "synapse_ingest_retry", *_HTTP),
    )
    assert loaded == []
    (argv,) = spawned
    assert argv[2] == "--ship" and json.loads(argv[3]) == payload
