"""The Claude Code hooks on the client core (plugin/scripts over plugin/synapse_client).

config.py resolves every setting on the core and sends through its Transport; the hook
scripts print what the core builds, and the Stop hook uploads through the Transport and
asks the core about private mode. What a Claude Code user sees is pinned here
literally: the resolution ranking, the token rules, each notice byte for byte, and each
hook's output for a normal reply, a 401, a restricted caller, a 500 and its kill switch.

Every script is loaded by path, or run as Claude Code runs it, with HOME, SYNAPSE_HOME and
every other file it reads at import under a temp dir (scratch_user), and talks to a local
stdlib server on 127.0.0.1. No test reaches a configured Synapse server or reads a real
credential.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import PLUGIN, SCRIPTS, drop_scripts, load_script, scratch_user

_TOKEN = "tkCLAUDEcore5Vn8Qx"
_DEVICE = "tkDEVICEclaude7Rq2"
_STALE = "tkSTALEclaude9Wm4"
_SURFACE_ID = "dev-claude-test"
_SERVER = "https://syn.example.net"


# ---------------------------------------------------------------------------
# A local Synapse stand-in
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # silence
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        self.server.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body) if body else None,
            }
        )
        status, reply, delay = self.server.routes.get(path, (200, {}, 0.0))
        if delay:
            self.server.release.wait(delay)
        if 300 <= status < 400:  # a redirect: the reply is the Location
            self.send_response(status)
            self.send_header("Location", reply)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        raw = json.dumps(reply).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:  # the client gave up waiting
            pass

    do_GET = do_POST = do_PUT = do_DELETE = _handle


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[dict[str, Any]] = []
        #: path -> (status, JSON reply, seconds to stall before answering)
        self.routes: dict[str, tuple[int, Any, float]] = {}
        self.release = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def answer(self, path: str, reply: Any, status: int = 200, delay: float = 0.0) -> None:
        self.routes[path] = (status, reply, delay)


@pytest.fixture()
def server() -> Iterator[_Server]:
    srv = _Server()
    threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
    yield srv
    srv.release.set()
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def scratch(monkeypatch, tmp_path) -> Iterator[Path]:
    """A scratch user; sys.path and the hook modules put back after."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield scratch_user(monkeypatch, tmp_path)
    drop_scripts()


def _write_config(home: Path, data: dict[str, Any]) -> None:
    path = home / ".synapse" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_settings(home: Path, data: dict[str, Any]) -> None:
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _options(values: dict[str, Any]) -> dict[str, Any]:
    return {"pluginConfigs": {"synapse@synapse": {"options": values}}}


def _record(token: str = _DEVICE, server: str = _SERVER) -> dict[str, Any]:
    return {
        "surface_id": _SURFACE_ID,
        "server": server,
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    }


def _user_agent() -> str:
    version = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())["version"]
    return f"synapse-client/{version} claude-code"


# ---------------------------------------------------------------------------
# config.py: the read path is the core's
# ---------------------------------------------------------------------------


def _words(path: Path) -> tuple[set[str], set[str]]:
    """The modules ``path`` imports, and its identifiers and string literals."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    words: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            words.add(node.value)
    return imported, words


def test_config_has_no_http_or_token_resolution_of_its_own(scratch, monkeypatch):
    imported, words = _words(SCRIPTS / "config.py")
    assert not imported & {"urllib.parse", "urllib.error", "http.client", "hashlib"}
    assert not words & {"Authorization", "sha256", "hexdigest", "build_opener"}
    assert not any(word.startswith("Bearer") for word in words)

    monkeypatch.setenv("SYNAPSE_URL", _SERVER + "/mcp")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    config = load_script("config")
    from synapse_client.settings import Settings

    assert isinstance(config.SETTINGS, Settings)
    assert Path(sys.modules["synapse_client"].__file__).parents[1] == PLUGIN
    assert config.BASE_URL == config.SETTINGS.base_url == _SERVER
    assert config.INGEST_URL == config.SETTINGS.ingest_url == _SERVER + "/ingest"
    assert config.INGEST_TOKEN == config.SETTINGS.token == _TOKEN


_URLS: dict[str, tuple[dict[str, str], dict[str, Any], dict[str, Any], tuple[str, str, str]]] = {
    # case: (env, config.json, plugin options in settings.json) -> (base, ingest, mcp)
    "nothing-is-localhost": ({}, {}, {}, ("http://localhost:8765", "", "")),
    "env-beats-config-json": (
        {"SYNAPSE_URL": "https://env.example/"},
        {"url": "https://cfg.example"},
        {"SYNAPSE_URL": "https://opt.example"},
        ("https://env.example", "", ""),
    ),
    "config-json-beats-the-options": (
        {"CLAUDE_PLUGIN_OPTION_SYNAPSE_URL": "https://exported.example"},
        {"url": "https://cfg.example"},
        {"SYNAPSE_URL": "https://opt.example"},
        ("https://cfg.example", "", ""),
    ),
    "an-exported-option-beats-settings-json": (
        {"CLAUDE_PLUGIN_OPTION_SYNAPSE_URL": "https://exported.example/ingest"},
        {},
        {"SYNAPSE_URL": "https://opt.example"},
        ("https://exported.example", "", ""),
    ),
    "settings-json-last": (
        {},
        {},
        {"SYNAPSE_URL": "https://opt.example/mcp"},
        ("https://opt.example", "", ""),
    ),
    "a-legacy-ingest-url-names-the-server": (
        {"SYNAPSE_INGEST_URL": "https://ing.example/ingest"},
        {},
        {},
        ("https://ing.example", "https://ing.example/ingest", ""),
    ),
    "the-legacy-endpoint-keys-still-win": (
        {"SYNAPSE_URL": "https://env.example"},
        {},
        {
            "SYNAPSE_INGEST_URL": "https://ing.example/in",
            "SYNAPSE_MCP_URL": "https://m.example/mcp",
        },
        ("https://env.example", "https://ing.example/in", "https://m.example/mcp"),
    ),
    "an-empty-env-value-is-unset": (
        {"SYNAPSE_URL": ""},
        {"url": "https://cfg.example/"},
        {},
        ("https://cfg.example", "", ""),
    ),
    "a-url-of-only-slashes-leaves-relative-endpoints": (
        {"SYNAPSE_URL": "//"},
        {},
        {},
        ("", "/ingest", "/mcp"),
    ),
    "a-falsy-json-option-reads-as-its-string": (
        {},
        {},
        {"SYNAPSE_URL": 0},
        ("0", "0/ingest", "0/mcp"),
    ),
}


@pytest.mark.parametrize("case", sorted(_URLS))
def test_url_resolution_ranking(scratch, monkeypatch, case):
    env, home_json, options, (base, ingest, mcp) = _URLS[case]
    _write_config(scratch, home_json)
    _write_settings(scratch, _options(options))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = load_script("config")
    assert config.BASE_URL == base
    assert config.INGEST_URL == (ingest or base + "/ingest")
    assert config.MCP_URL == (mcp or base + "/mcp")
    assert config.SKILLS_URL == base + "/skills"


def test_flags_rank_env_then_config_json_then_the_options_and_read_live(scratch, monkeypatch):
    _write_config(scratch, {"role": "work", "config_paths": []})
    _write_settings(scratch, _options({"SYNAPSE_BOARD": "0", "SYNAPSE_CONFIG_PATHS": "~/stale/*"}))
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_SURFACE", "exported")
    config = load_script("config")
    assert config.MACHINE_ROLE == "work" and config.requested_trust() == "restricted"
    assert config._cfg("SYNAPSE_BOARD", "1") == "0"
    assert config.CONFIG_PATHS == []  # a cleared list does not fall through
    assert config.SURFACE == "exported"
    monkeypatch.setenv("SYNAPSE_BOARD", "1")
    monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", "Personal ")
    assert config._cfg("SYNAPSE_BOARD", "x") == "1"
    assert config.machine_role() == "personal"
    monkeypatch.delenv("SYNAPSE_MACHINE_ROLE")
    config.set_home_values({"role": "personal"})
    assert config.machine_role() == "personal"


# The token rules: (env, config.json, settings.json) -> (token sent, note, notice).
_NOTICE_ENV_SHELL = (
    "[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
    "enrolled as dev-claude-test. Remove it from your shell profile."
)
_NOTICE_ENV_BLOCK = (
    "[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
    'enrolled as dev-claude-test. Remove it from the "env" block of ~/.claude/settings.json.'
)
_NOTICE_PLUGIN_OPTION = (
    "[Synapse] The Synapse token Claude Code loaded for this session is ignored: this "
    "machine is enrolled as dev-claude-test, and hooks use its device token. Run "
    "/reload-plugins (or restart) so recall uses it too."
)
_NOTICE_CONFIG_FILE = (
    "[Synapse] The token in ~/.synapse/config.json is ignored: this machine is enrolled "
    "as dev-claude-test with a different one. Run `synapse login --reenroll` to replace "
    "the enrollment, or `synapse config token --unset`."
)


def _withheld(url: str) -> str:
    return (
        f"[Synapse] {url} is not the server this machine enrolled with ({_SERVER}), so its "
        f"device token is not sent there. The same server at a new address: `synapse "
        f"config url {url} --same-server`. Another server: `synapse login --url {url} "
        "--reenroll`."
    )


_ENROLLED = {"url": _SERVER, "token": _DEVICE, "device": _record()}
_TOKENS: dict[str, tuple[dict, dict, dict, str, dict, str]] = {
    "unenrolled-env-token-is-sent": (
        {"SYNAPSE_INGEST_TOKEN": _TOKEN},
        {"url": _SERVER},
        {},
        _TOKEN,
        {},
        "",
    ),
    "unenrolled-config-token": ({}, {"url": _SERVER, "token": _TOKEN}, {}, _TOKEN, {}, ""),
    "unenrolled-options-token": (
        {},
        {"url": _SERVER},
        _options({"SYNAPSE_INGEST_TOKEN": _TOKEN}),
        _TOKEN,
        {},
        "",
    ),
    "the-device-token-beats-an-env-token": (
        {"SYNAPSE_INGEST_TOKEN": _STALE},
        _ENROLLED,
        {},
        _DEVICE,
        {"surface_id": _SURFACE_ID, "sources": ["env"]},
        _NOTICE_ENV_SHELL,
    ),
    "the-notice-names-the-settings-env-block": (
        {"SYNAPSE_INGEST_TOKEN": _STALE},
        _ENROLLED,
        {"env": {"SYNAPSE_INGEST_TOKEN": _STALE}},
        _DEVICE,
        {"surface_id": _SURFACE_ID, "sources": ["env"]},
        _NOTICE_ENV_BLOCK,
    ),
    "a-device-token-in-settings-json-beats-the-rest": (
        {"SYNAPSE_INGEST_TOKEN": _STALE, "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN": "tkOPT1"},
        {"url": _SERVER, "token": "tkCFG2", "device": _record()},
        _options({"SYNAPSE_INGEST_TOKEN": _DEVICE}),
        _DEVICE,
        {"surface_id": _SURFACE_ID, "sources": ["env", "config_file", "plugin_option"]},
        "\n".join([_NOTICE_ENV_SHELL, _NOTICE_CONFIG_FILE, _NOTICE_PLUGIN_OPTION]),
    ),
    "a-stale-config-token-with-no-device-token-anywhere-is-sent": (
        {},
        {"url": _SERVER, "token": _STALE, "device": _record()},
        {},
        _STALE,
        {},
        "",
    ),
    "the-device-token-is-withheld-from-another-server": (
        {"SYNAPSE_URL": "https://other.example.org"},
        _ENROLLED,
        {},
        "",
        {
            "surface_id": _SURFACE_ID,
            "sources": [],
            "withheld": "https://other.example.org",
            "server": _SERVER,
        },
        _withheld("https://other.example.org"),
    ),
    "and-from-the-localhost-default": (
        {},
        {"token": _DEVICE, "device": _record()},
        {},
        "",
        {
            "surface_id": _SURFACE_ID,
            "sources": [],
            "withheld": "http://localhost:8765",
            "server": _SERVER,
        },
        _withheld("http://localhost:8765"),
    ),
    "a-falsy-json-token-in-settings-json-is-no-token": (
        {},
        {"url": _SERVER},
        _options({"SYNAPSE_INGEST_TOKEN": 0}),
        "",
        {},
        "",
    ),
    "a-falsy-json-url-in-settings-json-is-no-address-for-the-device-token": (
        {},
        {"device": _record()},
        _options({"SYNAPSE_INGEST_TOKEN": _DEVICE, "SYNAPSE_URL": {}}),
        "",
        {"surface_id": _SURFACE_ID, "sources": [], "withheld": "://", "server": _SERVER},
        _withheld("://"),
    ),
    "a-url-of-only-slashes-is-no-address-for-the-device-token": (
        {"SYNAPSE_URL": "/"},
        _ENROLLED,
        {},
        "",
        {"surface_id": _SURFACE_ID, "sources": [], "withheld": "://", "server": _SERVER},
        _withheld("://"),
    ),
}


@pytest.mark.parametrize("case", sorted(_TOKENS))
def test_token_rules_and_notices(scratch, monkeypatch, case):
    env, home_json, settings, token, note, notice = _TOKENS[case]
    _write_config(scratch, home_json)
    _write_settings(scratch, settings)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = load_script("config")
    assert config.INGEST_TOKEN == token
    assert config.TOKEN_OVERRIDE == note
    assert config.token_override_notice() == notice
    for secret in (_TOKEN, _DEVICE, _STALE):
        assert secret not in notice


def test_device_origins_read_the_files_as_they_are_now(scratch):
    _write_config(scratch, {"url": "http://srv-a:8765"})
    config = load_script("config")
    assert config.device_origins() == set()
    config.set_home_values({"device": _record()})
    assert config.device_origins() == {("https", "syn.example.net")}
    assert config.read_device_state() == _record()


def test_config_requests_carry_the_client_user_agent_and_the_bearer(scratch, monkeypatch, server):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    config = load_script("config")
    server.answer("/context", {"status": "ok"})
    assert config.get_json("/context", {"project": "p"}) == {"status": "ok"}
    config.post_json("/skills/list", {"a": 1})
    config.request_json("PUT", "/private-sessions/s-1", timeout=5)
    assert [(r["method"], r["path"]) for r in server.seen] == [
        ("GET", "/context?project=p"),
        ("POST", "/skills/list"),
        ("PUT", "/private-sessions/s-1"),
    ]
    for req in server.seen:
        assert req["headers"]["user-agent"] == _user_agent()
        assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert server.seen[1]["body"] == {"a": 1}


@pytest.mark.parametrize("key", ["SYNAPSE_INGEST_URL", "SYNAPSE_MCP_URL"])
def test_the_bearer_goes_to_the_ingest_and_mcp_origins(scratch, monkeypatch, server, key):
    monkeypatch.setenv("SYNAPSE_URL", "http://localhost:9")
    monkeypatch.setenv(key, server.url + "/x")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    config = load_script("config")
    config.post_json(server.url + "/elsewhere", {})
    assert server.seen[0]["headers"]["authorization"] == f"Bearer {_TOKEN}"


def test_config_keeps_the_bearer_off_another_origin(scratch, monkeypatch, server):
    """Intended with the move to the core's Transport: an absolute URL at an origin none
    of this machine's URLs name is sent without the bearer."""
    monkeypatch.setenv("SYNAPSE_URL", "http://localhost:9")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    config = load_script("config")
    config.post_json(server.url + "/elsewhere", {})
    config.get_json(server.url + "/elsewhere")
    assert len(server.seen) == 2
    assert all("authorization" not in r["headers"] for r in server.seen)


def test_act_gates_on_the_expected_status(scratch, monkeypatch, server, capsys):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    config = load_script("config")
    server.answer("/skills/proposals/act", {"status": "applied"})
    assert config.act("/skills", 7, "accept", expect="applied", reason="r") == {"status": "applied"}
    assert server.seen[0]["body"] == {"id": 7, "action": "accept", "reason": "r"}
    server.answer("/skills/proposals/act", {"status": "x", "detail": "nope"})
    assert config.act("/skills", 7, "accept", expect="applied") is None
    assert capsys.readouterr().out == "nope\n"


def test_hooks_json_is_unchanged():
    """A change to the core changes no hook registration: the file is byte-identical to
    the one last changed on purpose (the preferences hook went when board_block.py began
    to print both blocks from one request)."""
    raw = (PLUGIN / "hooks" / "hooks.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "542bd1149033d81d9e83ac3804d6c64fac4f17fef5a42f6642765186f811f06b"
    )


# ---------------------------------------------------------------------------
# Session start: board_block.py prints what the core builds, from one request
# ---------------------------------------------------------------------------

_NOT_ENROLLED = (
    "[Synapse — this machine is not enrolled]\n"
    "No memory is being served here: since schema 054 every machine authenticates with its "
    "own device token.\n"
    "Run `! synapse-login` in this session (or `synapse-login` in a terminal) and approve the "
    "sign-in on any device."
)
_RESTRICTED = (
    "[Synapse — this machine is served nothing]\n"
    "The server answered, but this caller is `restricted` with no projects: it holds no "
    "device credential of its own (schema 054), so the board and recall come back empty.\n"
    "Fix it either way:\n"
    "  - Server with a sign-in configured: run `! synapse-login` in this session.\n"
    "  - Local server, no sign-in: run\n"
    '        docker compose exec mcp-server synapse-admin bootstrap "<label>"\n'
    "    on the server, then run `synapse login --token` here (or paste it into\n"
    "    /plugin > synapse > Synapse token)."
)
_BOARD = "[Synapse board]\n- a note"
_PREFS_TEXT = "[Synapse preferences]\n  - (likes) Terse answers\n  - (rule) Full-time roles only"
_BOTH_PARTS = "parts=board%2Cpreferences"


def _session_start(text: str, notice: str = "") -> str:
    """The SessionStart envelope Claude Code reads, as the hooks print it."""
    if not text:
        return ""
    out = '{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": '
    out += json.dumps(text) + "}"
    if notice:
        out += ', "systemMessage": ' + json.dumps(notice)
    return out + "}\n"


def _home(scratch: Path, server: _Server, *, enrolled: bool) -> None:
    data: dict[str, Any] = {"url": server.url}
    if enrolled:
        data |= {"token": _DEVICE, "device": _record(server=server.url)}
    _write_config(scratch, data)


def _run_hook(monkeypatch, capsys, name: str, stdin: str = "{}") -> str:
    hook = load_script(name)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    hook.main()
    return capsys.readouterr().out


_BOARD_REPLIES: dict[str, tuple[int, Any, bool, str]] = {
    # name -> (status, reply, enrolled, the context printed). A reply's ``preferences``
    # is what this caller is served of them: none, in these.
    "normal": (
        200,
        {"status": "ok", "text": _BOARD, "trust": "full", "preferences": ""},
        True,
        _BOARD,
    ),
    "401-not-enrolled": (401, {"detail": "no"}, False, _NOT_ENROLLED),
    "401-enrolled": (401, {"detail": "no"}, True, ""),
    "restricted-without-a-device-token": (
        200,
        {"status": "ok", "text": _BOARD, "trust": "restricted", "preferences": ""},
        False,
        _RESTRICTED + "\n\n" + _BOARD,
    ),
    "restricted-with-a-device-token": (
        200,
        {"status": "ok", "text": _BOARD, "trust": "restricted", "preferences": ""},
        True,
        _BOARD,
    ),
    "restricted-empty": (
        200,
        {"status": "ok", "text": "", "trust": "restricted", "preferences": ""},
        False,
        _RESTRICTED,
    ),
    "500": (500, {"detail": "boom"}, True, ""),
    "not-ok": (200, {"status": "error", "text": _BOARD, "preferences": ""}, True, ""),
}


@pytest.mark.parametrize("case", sorted(_BOARD_REPLIES))
def test_board_block_prints_the_board(scratch, monkeypatch, capsys, server, case):
    status, reply, enrolled, printed = _BOARD_REPLIES[case]
    _home(scratch, server, enrolled=enrolled)
    server.answer("/context", reply, status=status)
    assert _run_hook(monkeypatch, capsys, "board_block") == _session_start(printed)
    # One request, whatever the answer: an error status is not followed by another try.
    assert [r["path"] for r in server.seen] == [
        f"/context?project={scratch.parent.name}&{_BOTH_PARTS}"
    ]


def test_board_block_envelope_bytes(scratch, monkeypatch, capsys, server):
    _home(scratch, server, enrolled=True)
    server.answer("/context", {"status": "ok", "text": "[board]", "preferences": "[prefs]"})
    out = _run_hook(monkeypatch, capsys, "board_block", json.dumps({"cwd": str(scratch)}))
    assert out == (
        '{"hookSpecificOutput": {"hookEventName": "SessionStart", '
        '"additionalContext": "[board]\\n\\n[prefs]"}}\n'
    )
    assert len(server.seen) == 1
    assert server.seen[0]["headers"]["authorization"] == f"Bearer {_DEVICE}"
    assert server.seen[0]["headers"]["user-agent"] == _user_agent()


def test_board_block_prints_the_preferences_the_server_rendered(
    scratch, monkeypatch, capsys, server
):
    """The block comes with the board, in one request, and is printed after it as it
    came: the server renders it, so every harness shows the same lines."""
    _home(scratch, server, enrolled=True)
    server.answer("/context", {"status": "ok", "text": _BOARD, "preferences": _PREFS_TEXT})
    out = _run_hook(monkeypatch, capsys, "board_block")
    assert out == _session_start(_BOARD + "\n\n" + _PREFS_TEXT)
    assert [r["path"] for r in server.seen] == [
        f"/context?project={scratch.parent.name}&{_BOTH_PARTS}"
    ]


def test_board_block_shows_the_preferences_of_a_restricted_machine_after_the_explainer(
    scratch, monkeypatch, capsys, server
):
    _home(scratch, server, enrolled=False)
    reply = {"status": "ok", "text": "", "trust": "restricted", "preferences": _PREFS_TEXT}
    server.answer("/context", reply)
    out = _run_hook(monkeypatch, capsys, "board_block")
    assert out == _session_start(_RESTRICTED + "\n\n" + _PREFS_TEXT)


@pytest.mark.parametrize("status", [200, 500])
def test_board_block_leads_with_the_credential_notice(scratch, monkeypatch, capsys, server, status):
    """The notice is local: it is printed, to the user too, whatever the server says."""
    _home(scratch, server, enrolled=True)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    reply = {"status": "ok", "text": _BOARD, "preferences": _PREFS_TEXT}
    server.answer("/context", reply, status=status)
    out = _run_hook(monkeypatch, capsys, "board_block")
    served = "\n\n" + _BOARD + "\n\n" + _PREFS_TEXT if status == 200 else ""
    assert out == _session_start(_NOTICE_ENV_SHELL + served, notice=_NOTICE_ENV_SHELL)


@pytest.mark.parametrize("value", ["0", " 0"])
def test_board_block_kill_switch(scratch, monkeypatch, capsys, server, value):
    """SYNAPSE_BOARD=0 asks for no board, and for no project (the payload is not read);
    the credential notice and the preferences still show. Only exactly "0" switches it
    off: " 0" does not, as it never has."""
    _home(scratch, server, enrolled=True)
    monkeypatch.setenv("SYNAPSE_BOARD", value)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    server.answer("/context", {"status": "ok", "text": _BOARD, "preferences": _PREFS_TEXT})
    out = _run_hook(monkeypatch, capsys, "board_block")
    if value == "0":
        text = _NOTICE_ENV_SHELL + "\n\n" + _PREFS_TEXT
        assert out == _session_start(text, notice=_NOTICE_ENV_SHELL)
        assert [r["path"] for r in server.seen] == ["/context?parts=preferences"]
    else:
        text = _NOTICE_ENV_SHELL + "\n\n" + _BOARD + "\n\n" + _PREFS_TEXT
        assert out == _session_start(text, notice=_NOTICE_ENV_SHELL)
    # The payload names the folder, which only the board is scoped by.
    assert (sys.stdin.tell() == 0) is (value == "0")


def test_preferences_kill_switch(scratch, monkeypatch, capsys, server):
    _home(scratch, server, enrolled=True)
    monkeypatch.setenv("SYNAPSE_PREFS_BLOCK", "0")
    server.answer("/context", {"status": "ok", "text": _BOARD, "preferences": _PREFS_TEXT})
    assert _run_hook(monkeypatch, capsys, "board_block") == _session_start(_BOARD)
    assert [r["path"] for r in server.seen] == [
        f"/context?project={scratch.parent.name}&parts=board"
    ]


def test_both_switched_off_sends_nothing(scratch, monkeypatch, capsys, server):
    _home(scratch, server, enrolled=True)
    monkeypatch.setenv("SYNAPSE_BOARD", "0")
    monkeypatch.setenv("SYNAPSE_PREFS_BLOCK", "0")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    out = _run_hook(monkeypatch, capsys, "board_block")
    assert out == _session_start(_NOTICE_ENV_SHELL, notice=_NOTICE_ENV_SHELL)
    assert server.seen == []


def test_board_block_with_the_server_down_prints_nothing(scratch, monkeypatch, capsys):
    _write_config(scratch, {"url": "http://127.0.0.1:9", "token": _DEVICE})
    assert _run_hook(monkeypatch, capsys, "board_block") == ""


_PREFERENCES = {
    "items": [
        {"pref": "Terse answers", "polarity": "like"},
        {"pref": "Emoji", "polarity": "dislike"},
        {"pref": "Full-time roles only", "polarity": "rule"},
        {"pref": "", "polarity": "like"},
        {"pref": "Plain text", "polarity": "neutral"},
    ]
}


# A server from before /context carried the preferences block (kept for one release)


def test_a_server_from_before_the_block_is_asked_for_its_top_preferences(
    scratch, monkeypatch, capsys, server
):
    """Its /context reply has no ``preferences``: the items come from /preferences/top
    and the client renders them, to the same text."""
    _home(scratch, server, enrolled=True)
    server.answer("/context", {"status": "ok", "text": _BOARD, "trust": "full"})
    server.answer("/preferences/top", _PREFERENCES)
    assert _run_hook(monkeypatch, capsys, "board_block") == _session_start(
        _BOARD + "\n\n[Synapse preferences]\n  - (likes) Terse answers\n  - (dislikes) Emoji\n"
        "  - (rule) Full-time roles only\n  - (neutral) Plain text"
    )
    assert [r["path"] for r in server.seen] == [
        f"/context?project={scratch.parent.name}&{_BOTH_PARTS}",
        "/preferences/top?limit=7",
    ]
    assert server.seen[1]["headers"]["authorization"] == f"Bearer {_DEVICE}"


@pytest.mark.parametrize(
    ("status", "reply"),
    [(401, {"detail": "no"}), (500, {"detail": "boom"}), (200, {"items": []}), (200, {})],
    ids=["401", "500", "none", "no-items"],
)
def test_an_older_server_with_no_preferences_to_show_still_shows_its_board(
    scratch, monkeypatch, capsys, server, status, reply
):
    _home(scratch, server, enrolled=True)
    server.answer("/context", {"status": "ok", "text": _BOARD})
    server.answer("/preferences/top", reply, status=status)
    assert _run_hook(monkeypatch, capsys, "board_block") == _session_start(_BOARD)
    assert len(server.seen) == 2


def test_an_older_server_is_not_asked_for_preferences_that_are_switched_off(
    scratch, monkeypatch, capsys, server
):
    _home(scratch, server, enrolled=True)
    monkeypatch.setenv("SYNAPSE_PREFS_BLOCK", "0")
    server.answer("/context", {"status": "ok", "text": _BOARD})
    server.answer("/preferences/top", _PREFERENCES)
    assert _run_hook(monkeypatch, capsys, "board_block") == _session_start(_BOARD)
    assert len(server.seen) == 1


def test_no_preferences_hook_is_left(scratch):
    """One session-start hook prints both: a second process and a second request were
    the cost of rendering the preferences on the client."""
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    commands = [
        hook["command"] for entry in hooks["hooks"]["SessionStart"] for hook in entry["hooks"]
    ]
    assert sum("board_block.py" in command for command in commands) == 1
    assert not any("preferences_block" in command for command in commands)
    assert not (SCRIPTS / "preferences_block.py").exists()


def test_the_enrollment_explainers_are_the_claude_wording(scratch):
    """enroll.py builds both from the core with this plugin's sign-in and token slot."""
    enroll = load_script("enroll")
    assert enroll.not_enrolled_block() == _NOT_ENROLLED
    assert enroll.restricted_block() == _RESTRICTED


# ---------------------------------------------------------------------------
# Per prompt: recall_nudge.py prints the core's prompt_nudge
# ---------------------------------------------------------------------------

#: The timestamp line, as the hook has always printed it ("[Wed 2026-08-26 15:55:01 EDT]").
_CLOCK_LINE = re.compile(
    r"\[(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [^\]\"]*\]"
)
_PROMPT_NUDGE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable \\u2014 assume it "
    "has context on any past work, project, device, or person they name. BEFORE answering "
    "anything that references the past: synapse:recall (skip only if the answer is "
    "already in this conversation). About to say 'noted' or 'I'll remember'? Call "
    "synapse:remember FIRST, then reply."
)
_PROMPT = '{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": '
_BOTH = _PROMPT + '"[TS]\\n' + _PROMPT_NUDGE + '"}}\n'


@pytest.mark.parametrize(
    ("env", "printed"),
    [
        ({}, _BOTH),
        ({"SYNAPSE_RECALL_NUDGE": "0"}, _PROMPT + '"[TS]"}}\n'),
        ({"SYNAPSE_PROMPT_TIMESTAMP": "0"}, _PROMPT + '"' + _PROMPT_NUDGE + '"}}\n'),
        ({"SYNAPSE_RECALL_NUDGE": "0", "SYNAPSE_PROMPT_TIMESTAMP": "0"}, ""),
        ({"CLAUDE_PLUGIN_OPTION_SYNAPSE_RECALL_NUDGE": "0"}, _PROMPT + '"[TS]"}}\n'),
        ({"SYNAPSE_RECALL_NUDGE": "", "SYNAPSE_PROMPT_TIMESTAMP": "off"}, _BOTH),
    ],
    ids=["defaults", "no-nudge", "no-timestamp", "both-off", "plugin-option", "only-0-is-off"],
)
def test_recall_nudge_prints_the_time_and_the_nudge(scratch, monkeypatch, capsys, env, printed):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    out = _run_hook(monkeypatch, capsys, "recall_nudge")
    masked, stamps = _CLOCK_LINE.subn("[TS]", out)
    assert masked == printed
    assert stamps == printed.count("[TS]")


# ---------------------------------------------------------------------------
# Stop: ingest_hook.py posts through the core's Transport, private mode is the core's
# ---------------------------------------------------------------------------

_SID = "s-claude-1"


def _records(cwd: Path, sid: str = _SID) -> list[dict[str, Any]]:
    return [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": sid,
            "cwd": str(cwd),
            "message": {"role": "user", "content": "hello"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": sid,
            "cwd": str(cwd),
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        },
    ]


def _ship(scratch: Path, *, sid: str = _SID, record_sid: str = _SID):
    """Ship one transcript as the detached Stop child does (``--ship``): returns the hook
    module, ``_ship``'s (posts, records) and the records written."""
    cwd = scratch.parent
    records = _records(cwd, record_sid)
    path = cwd / "projects" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hook = load_script("ingest_hook")
    return hook, hook._ship(str(path)), records


def _logged(scratch: Path) -> list[str]:
    """The ingest log's lines without their timestamps."""
    log = scratch.parent / "ingest.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [line.split(" ", 1)[1] for line in lines]


@pytest.mark.parametrize("suffix", ["/ingest", "/ingest/", "/ingest//", "/ingest?x=1"])
def test_ingest_posts_to_the_ingest_url_exactly_as_configured(scratch, monkeypatch, server, suffix):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    monkeypatch.setenv("SYNAPSE_INGEST_URL", server.url + suffix)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    server.answer(suffix.split("?")[0], {"status": "ok", "inserted": 2})
    _hook, shipped, records = _ship(scratch)
    assert shipped == (1, 2)
    (req,) = server.seen
    assert (req["method"], req["path"]) == ("POST", suffix)
    assert req["headers"]["user-agent"] == _user_agent()
    assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert req["headers"]["content-type"] == "application/json"
    assert req["body"] == {"records": records, "source": "hook", "project": scratch.parent.name}
    assert _logged(scratch) == [
        f'OK {_SID}.jsonl stop 2 recs cur=0 -> {{"status": "ok", "inserted": 2}}'
    ]


def test_ingest_sends_the_bearer_to_the_ingest_origin(scratch, monkeypatch, server):
    """The ingest URL may name another host than SYNAPSE_URL: the bearer goes to it, as
    it always has."""
    monkeypatch.setenv("SYNAPSE_URL", "http://localhost:9")
    monkeypatch.setenv("SYNAPSE_INGEST_URL", server.url + "/ingest")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    _ship(scratch)
    assert server.seen[0]["headers"]["authorization"] == f"Bearer {_TOKEN}"


def test_ingest_keeps_the_bearer_off_another_origin(scratch, monkeypatch, server):
    """A redirect to another origin is followed without the bearer."""
    other = _Server()
    threading.Thread(target=other.serve_forever, args=(0.02,), daemon=True).start()
    try:
        monkeypatch.setenv("SYNAPSE_URL", server.url)
        monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
        server.answer("/ingest", other.url + "/ingest", status=302)
        other.answer("/ingest", {"status": "ok"})
        _hook, shipped, _records_sent = _ship(scratch)
        assert shipped == (1, 2)
        assert server.seen[0]["headers"]["authorization"] == f"Bearer {_TOKEN}"
        assert "authorization" not in other.seen[0]["headers"]
    finally:
        other.shutdown()
        other.server_close()


@pytest.mark.parametrize(
    ("status", "line", "reason"),
    [
        (401, "HTTPError: HTTP Error 401: Unauthorized", "HTTP 401 Unauthorized"),
        (500, "HTTPError: HTTP Error 500: Internal Server Error", "HTTP 500 Internal Server Error"),
    ],
)
def test_ingest_failure_is_logged_and_recorded(scratch, monkeypatch, server, status, line, reason):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/ingest", {"detail": "no"}, status=status)
    hook, shipped, _records_sent = _ship(scratch)
    path = str(scratch.parent / "projects" / f"{_SID}.jsonl")
    assert shipped == (0, 0)
    assert _logged(scratch) == [f"ERR stop {path}: {line}"]
    state = json.loads(Path(hook.CURSOR_PATH).read_text(encoding="utf-8"))
    assert (state[path]["offset"], state[path]["fails"]) == (0, 1)
    assert state[path]["last_error"] == reason


def _marker(scratch: Path, sid: str, hours_old: float) -> Path:
    marker = scratch.parent / "private" / sid
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    then = time.time() - hours_old * 3600
    os.utime(marker, (then, then))
    return marker


@pytest.mark.parametrize("record_sid", [_SID, "s-resumed"])
def test_ingest_ships_nothing_for_a_private_session(scratch, monkeypatch, server, record_sid):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    marker = _marker(scratch, record_sid, 1)
    _hook, shipped, _records_sent = _ship(scratch, record_sid=record_sid)
    assert shipped == (0, 0) and server.seen == []
    suffix = "" if record_sid == _SID else " (record sid)"
    assert _logged(scratch) == [f"PRIVATE skip stop {_SID}.jsonl{suffix}"]
    assert marker.exists()


@pytest.mark.parametrize(("ttl", "hours_old"), [(None, 13), ("2", 3)])
def test_ingest_removes_a_stale_marker_and_ships(scratch, monkeypatch, server, ttl, hours_old):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    if ttl:
        monkeypatch.setenv("SYNAPSE_PRIVATE_TTL_HOURS", ttl)
    marker = _marker(scratch, _SID, hours_old)
    _hook, shipped, _records_sent = _ship(scratch)
    assert shipped == (1, 2) and not marker.exists()
    first = _logged(scratch)[0]
    assert re.fullmatch(rf"PRIVATE stale marker removed {_SID} \(age \d+s\)", first)


def test_ingest_reads_a_record_id_that_is_a_path_as_not_private(scratch, monkeypatch, server):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    outside = _marker(scratch, "../outside", 40)
    _hook, shipped, _records_sent = _ship(scratch, record_sid="../outside")
    assert shipped == (1, 2) and outside.exists()
    assert _logged(scratch)[0] == (
        "PRIVATE ignored a session id that is not a file name: '../outside'"
    )


# ---------------------------------------------------------------------------
# harness.py: the registry comes from the core beside the script
# ---------------------------------------------------------------------------


def test_the_registry_is_the_one_of_the_core_config_imported(scratch, tmp_path):
    """A process runs one client version: a managed client of another version on the
    machine does not put a second ``synapse_client`` into this one. `synapse update` hands
    its second half to that client's own CLI instead (test_synapse_cli.py)."""
    home = Path(os.environ["SYNAPSE_HOME"])
    other = home / "lib" / "9.9.9" / "plugin" / "synapse_client" / "harnesses"
    other.mkdir(parents=True)
    (other.parent / "__init__.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    (other / "__init__.py").write_text("def registry():\n    return ['other']\n")
    (home / "lib" / "CURRENT").write_text("9.9.9\n", encoding="utf-8")
    # A managed client as the installer leaves one (the launcher, the Codex installer):
    # the registry used to be taken from there when there was one.
    (home / "bin").mkdir(parents=True)
    shutil.copy(PLUGIN / "scripts" / "synapse_run.py", home / "bin" / "synapse_run.py")
    (home / "lib" / "9.9.9" / "plugin-codex").mkdir()
    shutil.copy(
        PLUGIN.parent / "plugin-codex" / "install.py",
        home / "lib" / "9.9.9" / "plugin-codex" / "install.py",
    )
    harness = load_script("harness")
    assert harness.managed_root() == (home / "lib" / "9.9.9").resolve()
    loaded = sys.modules["synapse_client"]
    assert [h.name for h in harness._core_registry()] == ["hermes"]
    assert sys.modules["synapse_client"] is loaded
    assert Path(loaded.__file__).parents[1] == PLUGIN
    assert not hasattr(harness, "_forget_other_core")
