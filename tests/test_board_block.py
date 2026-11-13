"""SessionStart board block (plugin/scripts/board_block.py) — thin-client hook.

The hook must (a) print the server-rendered board text verbatim (stdout lands in the
session's context), (b) scope it to the project derived from the hook payload's cwd
the same way the ingest path labels episodes (basename of cwd; process cwd fallback),
(c) honor the SYNAPSE_BOARD=0 kill switch without touching the network, (d) be
fail-open: server down, timeout, or a non-ok payload prints nothing and exits 0 — a
broken board must never break a session start — and (e) since schema 054, explain a 401
on an UNENROLLED machine instead of failing silently, because a machine with no
credential of its own is served nothing and an unexplained empty session start is worse
than a restricted one.

It no longer sends a ``surface`` param: the bearer identifies the caller now, and a
host name the client asserts is evidence of nothing. Enrollment itself is interactive
(it prints a sign-in code and waits for a human), so the hook never runs it.

The same hook prints the preferences block, which comes in the same reply. These tests
are about the board: the preferences are switched off here (SYNAPSE_PREFS_BLOCK=0), so
the hook asks for ``parts=board``. tests/test_claude_on_core.py covers the two together.

Stdlib-only script loaded by path (it lives outside the package). No live server: the
client core's ``urlopen``, which every request the hook makes goes through, is replaced
(tests.helpers.client_core.fake_urlopen). Whether the machine is enrolled is a device
record in a scratch ~/.synapse/config.json, as the hook reads it.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers.client_core import fake_urlopen, requested

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "plugin" / "scripts" / "board_block.py"

# Every env var the plugin config layer (or this hook) consults — cleared so the
# tests are hermetic on any machine (dev box or CI).
_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_BOARD",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_BOARD",
)

_BOARD_TEXT = (
    "[Synapse board — project: synapse]\n"
    "42 episodes across 2 projects (most recent: synapse, scripts).\n"
    "\n"
    "## Rules & feedback\n"
    "- verify before shipping (n:3, upd 07-01)"
)


_SURFACE = "test-board-block-host"


def _isolated_env(monkeypatch, tmp_path, *, enrolled: bool = True) -> None:
    """Point the plugin config layer at a scratch config dir: no real env vars,
    no settings.json options, no credentials file, no project .claude, and a scratch
    ~/.synapse whose config.json holds a device record when ``enrolled``. Enrollment is a
    separate concern with its own tests; the default pins it "already enrolled" so the
    assertions stay about the board fetch."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    synapse_home = tmp_path / "synapse-home"
    synapse_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SYNAPSE_HOME", str(synapse_home))
    if enrolled:
        record = {"device": {"surface_id": "dev-board-block-test"}}
        (synapse_home / "config.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    # Pin the display label so nothing depends on the hostname of whatever machine
    # runs the suite. It is a label now, not an identity.
    monkeypatch.setenv("SYNAPSE_SURFACE", _SURFACE)
    monkeypatch.chdir(tmp_path)
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SYNAPSE_PREFS_BLOCK", "0")  # the board alone (module docstring)


def _load_hook() -> ModuleType:
    """Load the hook fresh. `config` is popped around the load so it re-resolves
    from the current env rather than a module cached by an earlier test."""
    for name in ("config", "enroll", "board_block"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("board_block", _SCRIPT)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPT.parent))
    for name in ("config", "enroll", "board_block"):
        sys.modules.pop(name, None)
    return mod


def _run(monkeypatch, mod: ModuleType, stdin: str, reply) -> list[tuple[str, dict]]:
    """Run main() with the hook payload on stdin and the network faked; return the
    (path, params) of each request made. `reply` is a payload dict or an exception to
    raise."""
    seen = fake_urlopen(monkeypatch, reply)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    mod.main()
    return [requested(req) for req in seen]


def _context(capsys) -> str:
    output = capsys.readouterr().out
    if not output:
        return ""
    payload = json.loads(output)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    return payload["hookSpecificOutput"]["additionalContext"]


def test_board_text_printed_verbatim(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    payload = {"status": "ok", "text": _BOARD_TEXT, "n_notes": 1, "overflow": 0}
    calls = _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), payload)
    assert _context(capsys) == _BOARD_TEXT
    # No `surface`: the token names the caller (schema 054), and the server decides
    # what that credential may see.
    assert calls == [("/context", {"project": "synapse", "parts": "board"})]


def test_kill_switch_no_output_no_http(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SYNAPSE_BOARD", "0")
    mod = _load_hook()
    payload = {"status": "ok", "text": _BOARD_TEXT}
    calls = _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), payload)
    assert _context(capsys) == ""
    assert calls == []


@pytest.mark.parametrize(
    "reply",
    [
        urllib.error.URLError("server down"),
        TimeoutError("timed out"),
        urllib.error.HTTPError("http://x/context", 404, "not found", None, None),  # older server
    ],
    ids=["conn-refused", "timeout", "http-404"],
)
def test_server_errors_are_silent(monkeypatch, tmp_path, capsys, reply):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), reply)
    assert _context(capsys) == ""  # fail-open: no block, no noise, exit 0


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "error", "detail": "board build failed"},
        {"status": "ok"},  # ok but no text
        {"status": "ok", "text": ""},
        {},
    ],
    ids=["error-status", "missing-text", "empty-text", "empty-payload"],
)
def test_non_ok_payload_is_silent(monkeypatch, tmp_path, capsys, reply):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/services/synapse"}), reply)
    assert _context(capsys) == ""


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/home/user/services/synapse", "synapse"),
        ("/home/user/scripts/", "scripts"),  # trailing slash — same as the ingest labeler
    ],
)
def test_project_derived_from_hook_cwd(monkeypatch, tmp_path, capsys, cwd, expected):
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    calls = _run(monkeypatch, mod, json.dumps({"cwd": cwd}), {"status": "ok", "text": "b"})
    assert calls == [("/context", {"project": expected, "parts": "board"})]
    assert _context(capsys) == "b"


def test_unprintable_board_is_silent(monkeypatch, tmp_path):
    """Fail-open covers the print, not just the fetch: a stdout that rejects a character
    (a Windows cp1252 console meeting the board's `→`) must not turn a SessionStart hook
    into a traceback. config re-wraps stdout as UTF-8 so this shouldn't happen — this is
    the belt to that suspenders."""
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()

    class _RejectingStdout(io.StringIO):
        def write(self, s: str) -> int:
            raise UnicodeEncodeError("charmap", s, 0, 1, "character maps to <undefined>")

    monkeypatch.setattr(sys, "stdout", _RejectingStdout())
    _run(monkeypatch, mod, json.dumps({"cwd": "/home/user/synapse"}), {"status": "ok", "text": "→"})


@pytest.mark.parametrize(
    "stdin",
    ["", "not json {", json.dumps({}), json.dumps({"cwd": "/"})],
    ids=["empty-stdin", "bad-json", "no-cwd", "root-cwd"],
)
def test_project_falls_back_to_process_cwd(monkeypatch, tmp_path, stdin):
    _isolated_env(monkeypatch, tmp_path)  # chdirs to tmp_path
    mod = _load_hook()
    calls = _run(monkeypatch, mod, stdin, {"status": "ok", "text": "b"})
    assert calls == [("/context", {"project": tmp_path.name, "parts": "board"})]


def test_a_401_on_an_unenrolled_machine_explains_itself(monkeypatch, tmp_path, capsys):
    """The one error worth a block: this machine holds no credential of its own, so it
    will go on being served nothing until someone signs in. Silence would read as
    "memory is empty"."""
    _isolated_env(monkeypatch, tmp_path, enrolled=False)
    mod = _load_hook()
    unauthorized = urllib.error.HTTPError("http://x/context", 401, "unauthorized", None, None)
    _run(monkeypatch, mod, "{}", unauthorized)
    out = _context(capsys)
    assert "not enrolled" in out and "synapse-login" in out


def test_a_401_on_an_enrolled_machine_stays_silent(monkeypatch, tmp_path, capsys):
    """A revoked or briefly-broken credential on a machine that DID enroll is not the
    user's cue to re-run login — it is an operator problem, and a session start is the
    wrong place to guess at it."""
    _isolated_env(monkeypatch, tmp_path, enrolled=True)
    mod = _load_hook()
    unauthorized = urllib.error.HTTPError("http://x/context", 401, "unauthorized", None, None)
    _run(monkeypatch, mod, "{}", unauthorized)
    assert _context(capsys) == ""


# --- the SILENT case: 200 + restricted + no device credential -----------------------


def test_a_restricted_unenrolled_caller_explains_itself(monkeypatch, tmp_path, capsys):
    """The failure a fresh install actually hits. An open server, or a caller holding
    the shared root token, is served a perfectly successful empty board — same outcome
    as the 401 above, none of the signal. Both remedies must appear, because which one
    applies depends on whether the server has a sign-in at all."""
    _isolated_env(monkeypatch, tmp_path, enrolled=False)
    mod = _load_hook()
    _run(monkeypatch, mod, "{}", {"status": "ok", "text": "", "trust": "restricted", "n_notes": 0})
    out = _context(capsys)
    assert "served nothing" in out
    assert "synapse-login" in out  # remedy 1: server with an IdP
    assert "synapse-admin bootstrap" in out  # remedy 2: local server, no IdP
    assert "Synapse token" in out  # ...and where the minted token goes


def test_restricted_but_enrolled_stays_silent(monkeypatch, tmp_path, capsys):
    """A restricted WORK machine is a deliberate grant, not a misconfiguration. It got
    the credential it was meant to get; nagging it every session would train the block
    to be ignored on the machine that needs it."""
    _isolated_env(monkeypatch, tmp_path)
    mod = _load_hook()
    calls = _run(
        monkeypatch,
        mod,
        "{}",
        {"status": "ok", "text": _BOARD_TEXT, "trust": "restricted"},
    )
    assert calls  # the fetch happened
    assert _context(capsys) == _BOARD_TEXT  # board only, no explainer


def test_full_trust_prints_no_explainer(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path, enrolled=False)
    mod = _load_hook()
    _run(monkeypatch, mod, "{}", {"status": "ok", "text": _BOARD_TEXT, "trust": "full"})
    assert _context(capsys) == _BOARD_TEXT
