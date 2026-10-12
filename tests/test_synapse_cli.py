"""`synapse` (plugin/scripts/synapse_cli.py) and the harness wiring under it (harness.py).

`synapse login` is the one setup step on a machine: it moves the old Claude install-prompt
answers into ~/.synapse/config.json, asks this machine's questions once, signs in (one
approval) unless the machine is already enrolled, and wires every harness it finds. Each
of those steps can cost a user their device token or leave a harness pointing at the
wrong server, so each is pinned here:

  * login: the URL is normalized and saved; a bad or missing URL stops before anything
    is written; a terminal is asked the URL, the role (re-asked on a bad answer) and
    "Sync skills to this machine? [Y/n]", a non-terminal nothing; flags skip questions;
    the role is asked only when this run enrolls; --token stores a token, drops the old
    record and never signs in; an enrolled machine never signs in again unless its
    server's origin changed (then it re-enrolls); a failed sign-in wires nothing;
    --no-install wires nothing; legacy answers are adopted first and named; the modules
    are reloaded so enrollment reaches the URL and role saved in the same run
  * harness.adopt_legacy copies only what config.json lacks, adopts the old device.json
    (pinning the adopted token), drops exactly the four retired plugin options, and is
    idempotent
  * harness.install_claude drives `claude plugin` (marketplace add only when missing,
    an existing plugin id/scope reused) and falls back to a settings.json bootstrap that
    never carries a token
  * harness.normalize_url accepts only what every shell and both config formats carry
    unquoted; codex_installer prefers the managed client's fixed launcher
  * `synapse config` never prints the token, validates every value, drops the enrollment
    record with a replaced token and re-wires only the harnesses already wired
  * `synapse install`, `update`, `skill retire` and the passthrough commands

Stdlib-only scripts loaded by path. HOME, SYNAPSE_HOME, CLAUDE_CONFIG_DIR,
SYNAPSE_DATA_DIR and the cwd all point into tmp_path. No network, no `claude` or `codex`
process: HTTP, subprocesses, harness detection and sign-in are fakes.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_MODULES = ("config", "enroll", "synapse_login", "harness", "synapse_cli", "synapse_home")

_KEYS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_RECALL_URL",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
)
_ENV_VARS = _KEYS + tuple(f"CLAUDE_PLUGIN_OPTION_{k}" for k in _KEYS)

# Distinctive values, so "no token material in the output" can check prefixes too.
_DEVICE = "tkDEV6r3Hq9WmZ2b"
_NEW = "tkNEW8c1Vx4PjL7s"
_URL = "https://synapse.example.net"
_OLD_URL = "https://old-synapse.example.org"
_SURFACE_ID = "dev-abc123"
_NEW_SURFACE_ID = "dev-new456"
_CLAUDE_EXE = "/opt/fake/bin/claude"

_URL_PROMPT = "Synapse server URL"
_ROLE_PROMPT = "Is this a personal or a work machine?"
_SKILLS_PROMPT = "Sync skills to this machine? [Y/n]: "


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Everything the client reads or writes lives under tmp_path, no setting arrives
    through the real environment, and the sys.path entries the scripts add for
    themselves are undone with the modules."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNAPSE_HOME", str(home / ".synapse"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-laptop")
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _forget_modules()
    yield
    _forget_modules()


def _forget_modules() -> None:
    for mod in _MODULES:
        sys.modules.pop(mod, None)


def _load(name: str, scripts: Path = _SCRIPTS) -> ModuleType:
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(scripts))
    return mod


def _config_json() -> Path:
    return Path(os.environ["SYNAPSE_HOME"]) / "config.json"


def _write_home(data: dict) -> None:
    path = _config_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_home() -> dict:
    return json.loads(_config_json().read_text(encoding="utf-8"))


def _settings_path() -> Path:
    return Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json"


def _write_settings(data: dict) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_settings() -> dict:
    return json.loads(_settings_path().read_text(encoding="utf-8"))


def _legacy_device_file() -> Path:
    return Path(os.environ["SYNAPSE_DATA_DIR"]) / "device.json"


def _write_legacy_record(record: dict) -> None:
    path = _legacy_device_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _record(token: str = _DEVICE, *, server: str = _URL) -> dict:
    return {
        "surface_id": _SURFACE_ID,
        "trust": "full",
        "token_sha256": _sha(token),
        "server": server,
    }


def _enrolled_home(url: str = _URL, server: str = _URL) -> dict:
    return {"url": url, "token": _DEVICE, "device": _record(server=server)}


def _no_token_material(text: str) -> None:
    for tok in (_DEVICE, _NEW):
        assert tok not in text
        assert tok[:6] not in text and tok[-6:] not in text


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Cli:
    """synapse_cli as a fresh `synapse` process loads it, with every outside effect
    replaced by a recorder: the terminal, harness detection and wiring, and the sign-in
    (enroll.enroll; synapse_login.main itself is real, so "already enrolled" is decided
    by the real code). Loading waits for the first run() or load(): the modules read
    config.json and settings.json at import, so a test arranges those first.

    The CLI reloads its modules in place after it changes the config; that would put
    the real functions back over these fakes, so _reload is a no-op here, and one test
    below runs the real reload end to end."""

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        self.mod: ModuleType | None = None
        self.answers: list[str] | None = None  # None: stdin is not a terminal
        self.prompts: list[str] = []
        self.found = {"claude": False, "codex": False}
        self.wired: list[tuple] = []
        self.wire_errors: dict[str, Exception] = {}
        self.signin_ok = True
        self.signins: list[dict] = []
        self.login_argv: list[list[str]] = []
        self.plugin_id: str | None = None
        self.codex_is_wired = False
        self.claude_update: str | None = None

    def terminal(self, *answers: str) -> None:
        self.answers = list(answers)

    def load(self) -> ModuleType:
        if self.mod is not None:
            return self.mod
        mod = _load("synapse_cli")
        enroll = _load("enroll")
        login = _load("synapse_login")
        harness = mod.harness
        mp = self.mp
        mp.setattr(mod, "_reload", lambda: None)
        mp.setattr(mod, "_interactive", lambda: self.answers is not None)
        mp.setattr(builtins, "input", self._input)
        mp.setattr(harness, "detect", lambda: dict(self.found))
        mp.setattr(harness, "install_claude", self._install_claude)
        mp.setattr(harness, "install_codex", self._install_codex)
        mp.setattr(harness, "claude_plugin_id", lambda runner=None: self.plugin_id)
        mp.setattr(harness, "codex_wired", lambda: self.codex_is_wired)
        mp.setattr(harness, "update_claude", lambda runner=None: self.claude_update)
        mp.setattr(enroll, "enroll", self._enroll)
        real_main = login.main

        def main(argv=None):
            self.login_argv.append(list(argv))
            return real_main(argv)

        mp.setattr(login, "main", main)
        self.mod = mod
        return mod

    def run(self, *argv: str) -> int:
        return self.load().main(list(argv))

    def _input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        assert self.answers, f"unexpected question: {prompt!r}"
        return self.answers.pop(0)

    def _install_claude(self, url, runner=None):
        self.wired.append(("claude", url))
        if "claude" in self.wire_errors:
            raise self.wire_errors["claude"]
        return f"Claude Code: plugin installed, SYNAPSE_URL={url}"

    def _install_codex(self, url, dry_run=False):
        self.wired.append(("codex", url, dry_run))
        if "codex" in self.wire_errors:
            raise self.wire_errors["codex"]
        return "Codex: wired"

    def _enroll(self, interactive=True, replace=False, open_browser=False, on_missing_route=None):
        """What a successful approval leaves behind: the new token and its record, with
        the server it was minted by."""
        cfg = self.mod.config
        self.signins.append({"replace": replace, "role": cfg.machine_role()})
        if not self.signin_ok:
            return {}
        record = {
            "surface_id": _NEW_SURFACE_ID,
            "trust": "full",
            "token_sha256": _sha(_NEW),
            # Resolved the way enroll's config.BASE_URL is: a server being switched to is
            # in the process environment only, until this enrollment succeeds.
            "server": cfg._cfg("SYNAPSE_URL"),
        }
        if replace:
            cfg.clear_device_state()
        cfg.store_token(_NEW, record)
        return record


@pytest.fixture()
def cli(monkeypatch) -> _Cli:
    return _Cli(monkeypatch)


# ---------------------------------------------------------------------------
# login: the server URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    [
        "https://synapse.example.net/",
        "https://synapse.example.net/mcp",
        "https://synapse.example.net/mcp/",
        "  https://synapse.example.net//  ",
    ],
)
def test_login_saves_the_url_normalized(cli, given):
    assert cli.run("login", "--url", given) == 0
    assert _read_home()["url"] == _URL
    assert cli.signins[0]["replace"] is False


@pytest.mark.parametrize("given", ["ftp://synapse.example.net", "https://synapse.example.net/a b"])
def test_login_with_an_invalid_url_fails_before_writing_anything(cli, capsys, given):
    assert cli.run("login", "--url", given) == 1
    assert "not a plain http(s) base URL" in capsys.readouterr().err
    assert not _config_json().exists()
    assert cli.signins == [] and cli.login_argv == []


def test_login_without_a_url_or_a_terminal_fails_with_a_message(cli, capsys):
    assert cli.run("login") == 1
    assert "no server URL: pass --url https://<your-synapse>" in capsys.readouterr().err
    assert cli.prompts == [] and cli.signins == []
    assert not _config_json().exists()


# ---------------------------------------------------------------------------
# login: this machine's questions
# ---------------------------------------------------------------------------


def test_a_terminal_login_asks_the_url_the_role_and_skills_sync_and_saves_the_answers(cli, capsys):
    cli.terminal("https://synapse.example.net/", "boss", "Work", "")
    assert cli.run("login") == 0

    assert len(cli.prompts) == 4
    assert cli.prompts[0].startswith(_URL_PROMPT)
    assert cli.prompts[1].startswith(_ROLE_PROMPT) and cli.prompts[2] == cli.prompts[1]
    assert cli.prompts[3] == _SKILLS_PROMPT
    assert cli.answers == []
    assert "Please answer personal or work." in capsys.readouterr().out
    home = _read_home()
    assert (home["url"], home["role"], home["skills_sync"]) == (_URL, "work", True)
    # The answer is in effect when the machine enrolls: it declares that role.
    assert cli.signins == [{"replace": False, "role": "work"}]


@pytest.mark.parametrize(
    ("answers", "role"),
    [([""], "personal"), (["work"], "work"), (["dunno", "PERSONAL"], "personal")],
)
def test_the_role_question_defaults_to_personal_and_re_asks_on_anything_else(cli, answers, role):
    cli.terminal(*answers, "y")
    assert cli.run("login", "--url", _URL) == 0
    assert _read_home()["role"] == role
    assert sum(p.startswith(_ROLE_PROMPT) for p in cli.prompts) == len(answers)


@pytest.mark.parametrize(
    ("answers", "sync"), [([""], True), (["n"], False), (["maybe", "YES"], True), (["off"], False)]
)
def test_the_skills_question_defaults_to_yes_and_re_asks_on_anything_else(
    cli, capsys, answers, sync
):
    cli.terminal(*answers)
    assert cli.run("login", "--url", _URL, "--role", "personal") == 0
    assert _read_home()["skills_sync"] is sync
    assert cli.prompts == [_SKILLS_PROMPT] * len(answers)
    assert ("Please answer y or n." in capsys.readouterr().out) is (len(answers) > 1)


def test_without_a_terminal_nothing_is_asked_and_role_and_skills_sync_stay_unset(cli):
    assert cli.run("login", "--url", _URL) == 0
    home = _read_home()
    assert "role" not in home and "skills_sync" not in home
    assert cli.prompts == []
    assert len(cli.signins) == 1


def test_role_and_skills_sync_flags_skip_their_questions(cli):
    cli.terminal()  # a terminal, but any question fails the test
    assert cli.run("login", "--url", _URL, "--role", "work", "--skills-sync", "no") == 0
    home = _read_home()
    assert (home["role"], home["skills_sync"]) == ("work", False)
    assert cli.prompts == []


def test_a_skills_sync_flag_that_is_not_yes_or_no_is_a_usage_error(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.run("login", "--url", _URL, "--skills-sync", "maybe")
    assert exc.value.code == 2
    assert "expected yes or no" in capsys.readouterr().err


def test_saved_answers_are_not_asked_again(cli):
    _write_home({"url": _URL, "role": "work", "skills_sync": False})
    cli.terminal()
    assert cli.run("login") == 0
    assert cli.prompts == []
    assert _read_home()["skills_sync"] is False


def test_the_role_is_not_asked_on_an_already_enrolled_machine(cli):
    """The role travels once, at enrollment; asking it on an enrolled machine would
    suggest the answer changes something when it does not."""
    _write_home(_enrolled_home())
    cli.terminal("")
    assert cli.run("login") == 0
    assert cli.prompts == [_SKILLS_PROMPT]
    home = _read_home()
    assert "role" not in home and home["skills_sync"] is True


# ---------------------------------------------------------------------------
# login: signing in, or not
# ---------------------------------------------------------------------------


def test_login_with_a_token_stores_it_drops_the_old_record_and_never_signs_in(cli, capsys):
    _write_home(_enrolled_home())
    _write_legacy_record(_record())
    assert cli.run("login", "--url", _URL, "--token", f"  {_NEW}  ") == 0

    home = _read_home()
    assert home["token"] == _NEW
    assert "device" not in home
    assert not _legacy_device_file().exists()
    assert cli.login_argv == [] and cli.signins == []
    out = capsys.readouterr().out
    assert f"Saved the token to {_config_json()}." in out
    _no_token_material(out)


def test_a_blank_token_is_refused_and_leaves_the_enrollment_alone(cli):
    _write_home(_enrolled_home())
    assert cli.run("login", "--url", _URL, "--token", "   ") != 0
    assert _read_home() == _enrolled_home()


def test_an_enrolled_machine_does_not_sign_in_again_but_is_still_wired(cli, capsys):
    _write_home(_enrolled_home())
    cli.found = {"claude": True, "codex": True}
    assert cli.run("login") == 0

    assert cli.login_argv == [[]]
    assert cli.signins == []
    assert f"already enrolled as {_SURFACE_ID}" in capsys.readouterr().out
    assert cli.wired == [("claude", _URL), ("codex", _URL, False)]
    assert _read_home() == _enrolled_home()


@pytest.mark.parametrize("server", [_URL, "https://SYNAPSE.example.net"])
def test_the_same_server_origin_does_not_re_enroll(cli, server):
    _write_home(_enrolled_home(server=server))
    assert cli.run("login", "--url", _URL + "/") == 0
    assert cli.login_argv == [[]] and cli.signins == []


def test_logging_in_to_another_server_needs_reenroll_and_changes_nothing_without_it(cli, capsys):
    """Moving a machine to another server is never implied by a URL alone: the old
    server's enrollment stays as it was, and nothing is sent to the new one."""
    before = _enrolled_home(url=_OLD_URL, server=_OLD_URL)
    _write_home(before)
    assert cli.run("login", "--url", _URL) == 1

    assert "--reenroll" in capsys.readouterr().err
    assert cli.signins == [] and cli.login_argv == []
    assert _read_home() == before


def test_an_enrolled_machine_logging_in_to_another_server_re_enrolls(cli, capsys):
    _write_home(_enrolled_home(url=_OLD_URL, server=_OLD_URL))
    cli.terminal("work", "n")  # re-enrolling is enrolling: the role is asked again
    assert cli.run("login", "--url", _URL, "--reenroll") == 0

    assert cli.login_argv == [["--reenroll"]]
    assert cli.signins == [{"replace": True, "role": "work"}]
    home = _read_home()
    assert home["url"] == _URL
    assert home["token"] == _NEW
    assert home["device"]["server"] == _URL
    assert home["device"]["surface_id"] == _NEW_SURFACE_ID


@pytest.mark.parametrize(
    "home",
    [
        {"url": _OLD_URL, "token": _DEVICE},  # set up with --token
        {"url": _OLD_URL, "device": {"surface_id": _SURFACE_ID, "trust": "full"}},
    ],
)
def test_a_machine_without_a_recorded_server_moves_only_through_a_sign_in(cli, capsys, home):
    """A pasted token, or a record too old to name its server, belongs to the server the
    saved URL points at: `login --url` elsewhere is a move like any other."""
    _write_home(home)
    assert cli.run("login", "--url", _URL, "--no-install") == 1
    assert "--reenroll" in capsys.readouterr().err
    assert _read_home() == home and cli.signins == []


def test_a_failed_move_keeps_the_old_url_with_its_token(cli, capsys):
    home = {"url": _OLD_URL, "token": _DEVICE}
    _write_home(home)
    cli.signin_ok = False
    assert cli.run("login", "--url", _URL, "--reenroll", "--no-install") != 0
    assert len(cli.signins) == 1
    assert _read_home()["url"] == _OLD_URL and _read_home()["token"] == _DEVICE
    assert f"still points at {_OLD_URL}" in capsys.readouterr().out


def test_a_token_only_machine_moved_by_a_sign_in_gets_the_new_servers_token(cli):
    _write_home({"url": _OLD_URL, "token": _DEVICE})
    assert cli.run("login", "--url", _URL, "--reenroll", "--no-install") == 0
    home = _read_home()
    assert home["url"] == _URL and home["token"] == _NEW
    assert home["device"]["server"] == _URL


def test_a_token_for_the_new_server_moves_the_machine_without_a_sign_in(cli):
    _write_home({"url": _OLD_URL, "token": _DEVICE})
    assert cli.run("login", "--url", _URL, "--token", _NEW, "--no-install") == 0
    home = _read_home()
    assert home["url"] == _URL and home["token"] == _NEW and cli.signins == []


def test_reenroll_flag_signs_in_again_on_the_same_server(cli):
    _write_home(_enrolled_home())
    assert cli.run("login", "--reenroll") == 0
    assert cli.login_argv == [["--reenroll"]]
    assert cli.signins[0]["replace"] is True


def test_a_failed_sign_in_wires_nothing_and_exits_nonzero(cli):
    cli.found = {"claude": True, "codex": True}
    cli.signin_ok = False
    assert cli.run("login", "--url", _URL) == 1
    assert len(cli.signins) == 1
    assert cli.wired == []


def test_no_install_skips_the_wiring(cli):
    cli.found = {"claude": True, "codex": True}
    assert cli.run("login", "--url", _URL, "--no-install") == 0
    assert len(cli.signins) == 1
    assert cli.wired == []


def test_login_wires_only_the_harnesses_it_finds(cli, capsys):
    cli.found = {"claude": False, "codex": True}
    assert cli.run("login", "--url", _URL) == 0
    assert cli.wired == [("codex", _URL, False)]

    cli.found = {"claude": False, "codex": False}
    cli.wired.clear()
    assert cli.run("login") == 0
    assert cli.wired == []
    assert "No Claude Code or Codex found on this machine" in capsys.readouterr().out


def test_login_reports_a_harness_it_could_not_wire_and_exits_nonzero(cli, capsys):
    cli.found = {"claude": True, "codex": True}
    cli.wire_errors["codex"] = RuntimeError("no Codex client files here")
    assert cli.run("login", "--url", _URL) == 1
    assert cli.wired == [("claude", _URL), ("codex", _URL, False)]
    assert "Codex: not wired: no Codex client files here" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# login: an install that predates config.json
# ---------------------------------------------------------------------------


def test_login_adopts_the_legacy_answers_first_and_names_them(cli, capsys):
    _write_settings(
        {
            "pluginConfigs": {
                "synapse@synapse": {
                    "options": {
                        "SYNAPSE_URL": _URL,
                        "SYNAPSE_MACHINE_ROLE": "work",
                        "SYNAPSE_SKILLS_SYNC": "1",
                    }
                }
            }
        }
    )
    cli.terminal()  # the adopted answers must keep both questions from being asked
    assert cli.run("login") == 0

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        f"Copied role, skills_sync, url from the Claude plugin's settings into {_config_json()}."
    )
    home = _read_home()
    assert (home["url"], home["role"], home["skills_sync"]) == (_URL, "work", True)
    assert cli.signins == [{"replace": False, "role": "work"}]
    assert cli.prompts == []
    # A copy: a Claude plugin older than 0.20 still reads these until it updates.
    assert set(_read_settings()["pluginConfigs"]["synapse@synapse"]["options"]) == {
        "SYNAPSE_URL",
        "SYNAPSE_MACHINE_ROLE",
        "SYNAPSE_SKILLS_SYNC",
    }


def test_an_enrolled_machine_whose_token_only_claude_code_holds_is_told_to_re_enroll(cli, capsys):
    """Pre-0.20 install: the token went into Claude Code's secure storage (a sensitive
    option), so neither settings.json nor config.json has it."""
    _write_settings({"pluginConfigs": {"synapse@synapse": {"options": {"SYNAPSE_URL": _URL}}}})
    _write_legacy_record({"surface_id": _SURFACE_ID, "trust": "full", "server": _URL})
    assert cli.run("login") == 0

    out = capsys.readouterr().out
    assert "Claude Code's secure storage" in out
    assert "synapse login --reenroll" in out
    assert cli.signins == []
    assert _read_home()["device"]["surface_id"] == _SURFACE_ID
    assert _legacy_device_file().exists()  # copied, not moved


def test_login_reloads_the_client_so_enrollment_reaches_the_url_and_role_it_just_saved(
    monkeypatch,
):
    """The real _reload and the real device flow, faked only at the HTTP layer: the
    modules read the URL at import, so without the reload the enrollment would go to the
    default http://localhost:8765 and declare no role."""
    seen: list[tuple[str, dict]] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data or b"{}")
        seen.append((req.full_url, body))
        if req.full_url.endswith("/device/code"):
            reply = {
                "device_code": "dc-1",
                "user_code": "ABCD-1234",
                "verification_uri": "https://idp.example.com/device",
                "interval": 1,
                "expires_in": 60,
            }
        else:
            reply = {
                "status": "ok",
                "token": _NEW,
                "login": "owner",
                "surface": {"surface_id": _NEW_SURFACE_ID, "trust": "restricted"},
            }
        return _Resp(json.dumps(reply).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(webbrowser, "open", lambda *_a, **_k: True)
    mod = _load("synapse_cli")
    monkeypatch.setattr(mod, "_interactive", lambda: False)

    assert mod.main(["login", "--url", _URL + "/mcp", "--role", "work", "--no-install"]) == 0
    assert [url for url, _ in seen] == [f"{_URL}/device/code", f"{_URL}/surfaces/enroll"]
    assert seen[1][1]["trust"] == "restricted"
    home = _read_home()
    assert (home["url"], home["role"], home["token"]) == (_URL, "work", _NEW)
    assert home["device"]["server"] == _URL
    assert home["device"]["token_sha256"] == _sha(_NEW)


# ---------------------------------------------------------------------------
# harness.adopt_legacy
# ---------------------------------------------------------------------------


_LEGACY_OPTIONS = {
    "SYNAPSE_URL": "https://legacy.example.net",
    "SYNAPSE_INGEST_TOKEN": _DEVICE,
    "SYNAPSE_MACHINE_ROLE": "Work",
    "SYNAPSE_SKILLS_SYNC": "0",
    "SYNAPSE_CONFIG_SYNC": "1",
    "SYNAPSE_CONFIG_PATHS": "output-styles/*.md, agents/*.md",
}


def _legacy_settings(options: dict | None = None) -> dict:
    return {
        "theme": "dark",
        "env": {"FOO": "bar"},
        "enabledPlugins": {"synapse@synapse": True, "other@market": True},
        "pluginConfigs": {
            "synapse@synapse": {"options": dict(options or _LEGACY_OPTIONS)},
            # Another plugin's option of the same name is not ours to drop.
            "other@market": {"options": {"SYNAPSE_MACHINE_ROLE": "keep-me"}},
        },
    }


def test_adopt_legacy_copies_only_the_answers_config_json_lacks():
    _write_settings(_legacy_settings())
    _write_home({"url": _URL})
    adopted = _load("harness").adopt_legacy()

    assert adopted == ["config_paths", "config_sync", "role", "skills_sync", "token"]
    assert _read_home() == {
        "url": _URL,  # config.json's own value is never overwritten
        "token": _DEVICE,
        "role": "work",
        "skills_sync": False,
        "config_sync": True,
        "config_paths": ["output-styles/*.md", "agents/*.md"],
    }


@pytest.mark.parametrize(("text", "value"), [("0", False), ("1", True)])
def test_adopt_legacy_reads_the_on_off_options_as_the_plugin_did(text, value):
    _write_settings(_legacy_settings({"SYNAPSE_SKILLS_SYNC": text, "SYNAPSE_CONFIG_SYNC": text}))
    _load("harness").adopt_legacy()
    home = _read_home()
    assert (home["skills_sync"], home["config_sync"]) == (value, value)


def test_adopt_legacy_keeps_an_explicit_off_already_in_config_json():
    _write_settings(_legacy_settings({"SYNAPSE_SKILLS_SYNC": "1"}))
    _write_home({"skills_sync": False})
    assert _load("harness").adopt_legacy() == []
    assert _read_home() == {"skills_sync": False}


def test_adopt_legacy_copies_the_device_json_record_and_pins_the_adopted_token():
    record = {"surface_id": _SURFACE_ID, "trust": "full", "server": _URL}
    _write_legacy_record(record)
    _write_settings(_legacy_settings({"SYNAPSE_INGEST_TOKEN": _DEVICE}))
    adopted = _load("harness").adopt_legacy()

    assert adopted == ["device", "token"]
    assert _read_home()["device"] == {**record, "token_sha256": _sha(_DEVICE)}
    assert _legacy_device_file().exists()  # an older plugin still reads it


def test_adopt_legacy_pins_a_token_config_json_already_had():
    record = {"surface_id": _SURFACE_ID, "trust": "full", "server": _URL}
    _write_legacy_record(record)
    _write_home({"token": _DEVICE})
    assert _load("harness").adopt_legacy() == ["device"]
    assert _read_home()["device"]["token_sha256"] == _sha(_DEVICE)


def test_adopt_legacy_keeps_a_record_s_own_hash_and_adds_none_without_a_token():
    hashed = _record(_NEW)
    _write_legacy_record(hashed)
    _write_settings(_legacy_settings({"SYNAPSE_INGEST_TOKEN": _DEVICE}))
    _load("harness").adopt_legacy()
    assert _read_home()["device"] == hashed

    _config_json().unlink()
    bare = {"surface_id": _SURFACE_ID, "trust": "full"}
    _write_legacy_record(bare)
    _write_settings(_legacy_settings({"SYNAPSE_URL": _URL}))
    _forget_modules()
    _load("harness").adopt_legacy()
    assert _read_home()["device"] == bare


def test_adopt_legacy_leaves_device_json_alone_when_config_json_has_a_record():
    _write_home({"device": _record()})
    _write_legacy_record({"surface_id": "dev-old", "trust": "restricted"})
    assert _load("harness").adopt_legacy() == []
    assert _read_home() == {"device": _record()}


def test_adopt_legacy_leaves_settings_json_alone():
    _write_settings(_legacy_settings())
    before = _settings_path().read_bytes()
    _load("harness").adopt_legacy()
    assert _settings_path().read_bytes() == before


def test_drop_options_drops_exactly_the_four_retired_options():
    _write_settings(_legacy_settings())
    harness = _load("harness")
    harness.drop_options(harness.DROPPED_OPTIONS)

    expected = _legacy_settings()
    expected["pluginConfigs"]["synapse@synapse"]["options"] = {
        "SYNAPSE_URL": "https://legacy.example.net",
        "SYNAPSE_INGEST_TOKEN": _DEVICE,
    }
    assert _read_settings() == expected


def test_drop_options_works_under_any_marketplace_name():
    settings = _legacy_settings()
    settings["pluginConfigs"]["synapse@my-fork"] = settings["pluginConfigs"].pop("synapse@synapse")
    _write_settings(settings)
    harness = _load("harness")
    harness.drop_options(harness.DROPPED_OPTIONS)
    assert set(_read_settings()["pluginConfigs"]["synapse@my-fork"]["options"]) == {
        "SYNAPSE_URL",
        "SYNAPSE_INGEST_TOKEN",
    }


def test_adopt_legacy_is_idempotent():
    _write_settings(_legacy_settings())
    _write_legacy_record({"surface_id": _SURFACE_ID, "trust": "full", "server": _URL})
    harness = _load("harness")
    assert harness.adopt_legacy()
    home, settings = _config_json().read_bytes(), _settings_path().read_bytes()

    assert harness.adopt_legacy() == []
    assert _config_json().read_bytes() == home
    assert _settings_path().read_bytes() == settings


def test_adopt_legacy_on_a_machine_with_nothing_to_adopt_writes_nothing():
    assert _load("harness").adopt_legacy() == []
    assert not _config_json().exists()
    assert not _settings_path().exists()


# ---------------------------------------------------------------------------
# harness.install_claude
# ---------------------------------------------------------------------------


class _ClaudeCli:
    """A scripted `claude` executable: what its list commands print, which steps fail,
    and every command line it was given (without the executable)."""

    def __init__(self, marketplaces=(), plugins=(), fail=()):
        self.marketplaces = list(marketplaces)
        self.plugins = list(plugins)
        self.fail = set(fail)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == _CLAUDE_EXE
        assert kwargs.get("check") is False and kwargs.get("timeout")
        args = list(cmd[1:])
        self.calls.append(args)
        step = f"marketplace {args[2]}" if args[1] == "marketplace" else f"plugin {args[1]}"
        if step in self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", f"error: {step} refused")
        stdout = ""
        if step == "marketplace list":
            stdout = json.dumps(self.marketplaces)
        elif step == "plugin list":
            stdout = json.dumps(self.plugins)
        return subprocess.CompletedProcess(cmd, 0, stdout, "")


_LIST_MARKETPLACES = ["plugin", "marketplace", "list", "--json"]
_ADD_MARKETPLACE = ["plugin", "marketplace", "add", "kraft87/synapse", "--scope", "user"]
_LIST_PLUGINS = ["plugin", "list", "--json"]


def _install_cmd(pid: str = "synapse@synapse", scope: str = "user") -> list[str]:
    return ["plugin", "install", pid, "--scope", scope, "--config", f"SYNAPSE_URL={_URL}"]


@pytest.fixture()
def claude_harness(monkeypatch) -> ModuleType:
    """harness with a `claude` executable on PATH (and no other)."""
    harness = _load("harness")
    monkeypatch.setattr(
        harness.shutil, "which", lambda name: _CLAUDE_EXE if name == "claude" else None
    )
    return harness


def test_install_claude_adds_the_marketplace_then_installs_with_the_url(claude_harness):
    claude = _ClaudeCli()
    report = claude_harness.install_claude(_URL, runner=claude)
    assert claude.calls == [
        _LIST_MARKETPLACES,
        _ADD_MARKETPLACE,
        _LIST_PLUGINS,
        _install_cmd(),
        _LIST_PLUGINS,  # the installed version decides whether the old options go
    ]
    assert report == f"Claude Code: plugin installed, SYNAPSE_URL={_URL}"
    assert not _settings_path().exists()


def test_install_claude_does_not_add_a_marketplace_that_is_already_there(claude_harness):
    claude = _ClaudeCli(marketplaces=[{"name": "other"}, {"name": "synapse"}])
    claude_harness.install_claude(_URL, runner=claude)
    assert claude.calls == [_LIST_MARKETPLACES, _LIST_PLUGINS, _install_cmd(), _LIST_PLUGINS]


@pytest.mark.parametrize(
    ("installed", "pid", "scope"),
    [
        (
            {"id": "synapse@kraft87-synapse", "scope": "project", "projectPath": "."},
            "synapse@kraft87-synapse",
            "project",
        ),
        # A local or project install made in another directory: reinstalling at its scope
        # here would enable the plugin in this directory only. User scope instead.
        (
            {"id": "synapse@kraft87-synapse", "scope": "local", "projectPath": "/elsewhere"},
            "synapse@kraft87-synapse",
            "user",
        ),
        ({"id": "synapse@my-fork"}, "synapse@my-fork", "user"),
    ],
)
def test_install_claude_reuses_the_id_and_scope_of_an_installed_plugin(
    claude_harness, installed, pid, scope
):
    if installed.get("projectPath") == ".":
        installed = {**installed, "projectPath": os.getcwd()}
    claude = _ClaudeCli(
        marketplaces=[{"name": "synapse"}],
        plugins=[{"id": "synapse-tools@elsewhere", "scope": "user"}, installed],
    )
    claude_harness.install_claude(_URL, runner=claude)
    # Installed already: the marketplace and the plugin are updated first, so a cached
    # older plugin doesn't go on running without the headers helper.
    assert claude.calls[2:5] == [
        ["plugin", "marketplace", "update", pid.split("@", 1)[1]],
        ["plugin", "update", pid],
        _install_cmd(pid, scope),
    ]


def test_install_claude_prefers_the_user_scope_install_over_a_local_one(claude_harness):
    claude = _ClaudeCli(
        marketplaces=[{"name": "synapse"}],
        plugins=[
            {"id": "synapse@synapse", "scope": "local", "projectPath": "/elsewhere"},
            {"id": "synapse@synapse", "scope": "user"},
        ],
    )
    claude_harness.install_claude(_URL, runner=claude)
    assert _install_cmd("synapse@synapse", "user") in claude.calls


def test_the_retired_options_are_copied_to_config_json_before_they_are_dropped(claude_harness):
    """`synapse install` and `synapse config url` reach the drop without a login."""
    _write_settings(_legacy_settings())
    claude = _ClaudeCli(
        marketplaces=[{"name": "synapse"}],
        plugins=[{"id": "synapse@synapse", "scope": "user", "version": "0.20.0"}],
    )
    claude_harness.install_claude(_URL, runner=claude)
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert "SYNAPSE_SKILLS_SYNC" not in options
    home = _read_home()
    for key, name in (
        ("role", "SYNAPSE_MACHINE_ROLE"),
        ("skills_sync", "SYNAPSE_SKILLS_SYNC"),
        ("config_sync", "SYNAPSE_CONFIG_SYNC"),
        ("config_paths", "SYNAPSE_CONFIG_PATHS"),
    ):
        assert key in home, (key, name)


def test_the_retired_options_stay_when_config_json_cannot_take_them(claude_harness, monkeypatch):
    _write_settings(_legacy_settings())
    claude = _ClaudeCli(
        marketplaces=[{"name": "synapse"}],
        plugins=[{"id": "synapse@synapse", "scope": "user", "version": "0.20.0"}],
    )

    def refuse():
        raise OSError("read-only")

    monkeypatch.setattr(claude_harness, "adopt_legacy", refuse)
    claude_harness.install_claude(_URL, runner=claude)
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert "SYNAPSE_SKILLS_SYNC" in options


@pytest.mark.parametrize(("version", "dropped"), [("0.20.0", True), ("0.19.6", False)])
def test_the_retired_options_go_only_once_the_installed_plugin_no_longer_reads_them(
    claude_harness, version, dropped
):
    _write_settings(_legacy_settings())
    claude = _ClaudeCli(
        marketplaces=[{"name": "synapse"}],
        plugins=[{"id": "synapse@synapse", "scope": "user", "version": version}],
    )
    claude_harness.install_claude(_URL, runner=claude)
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert ("SYNAPSE_SKILLS_SYNC" not in options) is dropped
    assert "SYNAPSE_URL" in options


def test_the_settings_bootstrap_refuses_to_rewrite_an_unreadable_settings_json(monkeypatch):
    _settings_path().parent.mkdir(parents=True, exist_ok=True)
    _settings_path().write_text('{"theme": "dark",}', encoding="utf-8")  # trailing comma
    harness = _load("harness")
    monkeypatch.setattr(harness.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="isn't valid JSON"):
        harness.install_claude(_URL)
    assert _settings_path().read_text(encoding="utf-8") == '{"theme": "dark",}'


def _assert_bootstrapped(settings: dict, key: str = "synapse@synapse") -> None:
    assert settings["extraKnownMarketplaces"]["synapse"] == {
        "source": {"source": "github", "repo": "kraft87/synapse"}
    }
    assert settings["enabledPlugins"][key] is True
    assert settings["pluginConfigs"][key]["options"] == {"SYNAPSE_URL": _URL}


@pytest.mark.parametrize("failing", ["marketplace list", "marketplace add", "plugin install"])
def test_a_failing_claude_cli_falls_back_to_the_settings_bootstrap_without_a_token(
    claude_harness, capsys, failing
):
    _write_home({"url": _URL, "token": _DEVICE})
    report = claude_harness.install_claude(_URL, runner=_ClaudeCli(fail=[failing]))

    assert f"`claude plugin` failed ({failing}" in capsys.readouterr().out
    assert "plugin enabled in settings.json" in report
    _assert_bootstrapped(_read_settings())
    text = _settings_path().read_text(encoding="utf-8")
    assert "SYNAPSE_INGEST_TOKEN" not in text
    _no_token_material(text)


def test_a_claude_cli_that_cannot_run_falls_back_to_the_settings_bootstrap(claude_harness):
    def broken(cmd, **_kw):
        raise OSError("exec format error")

    claude_harness.install_claude(_URL, runner=broken)
    _assert_bootstrapped(_read_settings())


def test_without_the_claude_cli_settings_json_is_bootstrapped_directly(monkeypatch):
    _write_home({"url": _URL, "token": _DEVICE})
    _write_settings(
        {
            "theme": "dark",
            "enabledPlugins": {"synapse@my-fork": False, "other@market": True},
            "pluginConfigs": {"synapse@my-fork": {"options": {"SYNAPSE_URL": _OLD_URL}}},
        }
    )
    harness = _load("harness")
    monkeypatch.setattr(harness.shutil, "which", lambda _name: None)
    claude = _ClaudeCli()

    report = harness.install_claude(_URL, runner=claude)
    assert claude.calls == []
    assert "restart Claude Code twice" in report
    settings = _read_settings()
    _assert_bootstrapped(settings, key="synapse@my-fork")  # the existing entry is reused
    assert "synapse@synapse" not in settings["enabledPlugins"]
    assert settings["theme"] == "dark"
    assert settings["enabledPlugins"]["other@market"] is True
    _no_token_material(_settings_path().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# harness.normalize_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (_URL, _URL),
        (_URL + "/", _URL),
        ("  http://localhost:8765/mcp/ ", "http://localhost:8765"),
        ("http://192.168.1.20:8765/ingest", "http://192.168.1.20:8765"),
        ("http://[::1]:8765", "http://[::1]:8765"),
        ("http://[fe80::1]/install", "http://[fe80::1]"),
        ("https://example.net/synapse/base", "https://example.net/synapse/base"),
        ("https://example.net/synapse/recall", "https://example.net/synapse"),
        ("https://example.net/v1.2/a_b~c-d/skills", "https://example.net/v1.2/a_b~c-d"),
    ],
)
def test_normalize_url_accepts_plain_http_base_urls(given, expected):
    assert _load("harness").normalize_url(given) == expected


@pytest.mark.parametrize(
    "given",
    [
        'https://synapse.example.net/"x',
        "https://synapse.example.net'",
        "https://synapse.example.net/a b",
        "https://synapse.example.net/a%20b",
        "https://synapse.example.net/$HOME",
        "https://synapse.example.net/a&b",
        "https://synapse.example.net/?q=1",
        "https://user:pw@synapse.example.net",
        "https://synapse.example.net:123456",
        "ftp://synapse.example.net",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "synapse.example.net",
        "https://",
        "",
    ],
)
def test_normalize_url_rejects_anything_a_shell_or_config_file_would_need_quoted(given):
    with pytest.raises(ValueError, match="not a plain http"):
        _load("harness").normalize_url(given)


def test_normalize_url_keeps_a_host_named_like_an_endpoint():
    assert _load("harness").normalize_url("http://mcp") == "http://mcp"


# ---------------------------------------------------------------------------
# harness.codex_installer
# ---------------------------------------------------------------------------


def _managed_client(*, with_codex: bool = True) -> Path:
    """~/.synapse as `curl <server>/install | python3 -` leaves it: the fixed launcher in
    bin/, lib/CURRENT naming the version, the client unpacked under lib/<version>/."""
    home = Path(os.environ["SYNAPSE_HOME"])
    (home / "bin").mkdir(parents=True)
    shutil.copy(_SCRIPTS / "synapse_run.py", home / "bin" / "synapse_run.py")
    lib = home / "lib"
    for version in ("0.19.6", "0.20.0"):
        (lib / version / "plugin").mkdir(parents=True)
        if with_codex:
            (lib / version / "plugin-codex").mkdir()
            shutil.copy(
                _REPO / "plugin-codex" / "install.py", lib / version / "plugin-codex" / "install.py"
            )
    (lib / "CURRENT").write_text("0.19.6\n", encoding="utf-8")
    return lib / "0.19.6"


def test_codex_installer_prefers_the_managed_client_through_its_fixed_launcher():
    root = _managed_client()
    installer = _load("harness").codex_installer()

    assert Path(installer.__file__).resolve() == (root / "plugin-codex" / "install.py").resolve()
    runner = Path(os.environ["SYNAPSE_HOME"]).resolve() / "bin" / "synapse_run.py"
    assert installer.LAUNCHER.runner == runner
    assert installer.LAUNCHER.root == root
    # Codex trusts a hook by its command text: it must not name the version directory.
    command = installer.LAUNCHER.command("plugin-codex/hooks/stop.py")
    assert str(runner.as_posix()) in command
    assert "0.19.6" not in command


def test_the_runner_is_spelled_resolved_through_a_symlinked_home(monkeypatch, tmp_path):
    """The managed shim's synapse_run.py exports a resolved SYNAPSE_HOME; the Claude
    plugin's own shims don't set it. Codex trusts hooks by their command text, so both
    entry points must write the same spelling."""
    root = _managed_client()
    real = Path(os.environ["SYNAPSE_HOME"])
    link = tmp_path / "linked-home"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("SYNAPSE_HOME", str(link))
    installer = _load("harness").codex_installer()
    assert installer.LAUNCHER.runner == real.resolve() / "bin" / "synapse_run.py"
    assert installer.LAUNCHER.root.resolve() == root.resolve()
    assert str(link) not in installer.LAUNCHER.command("plugin-codex/hooks/stop.py")


def test_codex_installer_falls_back_to_the_checkout_with_a_direct_launcher():
    installer = _load("harness").codex_installer()
    assert Path(installer.__file__).resolve() == (_REPO / "plugin-codex" / "install.py").resolve()
    assert installer.LAUNCHER.runner is None
    assert installer.LAUNCHER.root == _REPO
    assert str(_REPO.as_posix()) in installer.LAUNCHER.command("plugin-codex/hooks/stop.py")


def test_a_managed_client_without_codex_files_falls_back_to_the_checkout():
    _managed_client(with_codex=False)
    installer = _load("harness").codex_installer()
    assert installer.LAUNCHER.runner is None
    assert installer.LAUNCHER.root == _REPO


def test_the_claude_plugins_own_copy_has_no_codex_installer(tmp_path):
    """The Claude plugin is installed as plugin/ alone: no plugin-codex beside it."""
    scripts = tmp_path / "plugin-cache" / "plugin" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("harness.py", "config.py", "synapse_home.py"):
        shutil.copy(_SCRIPTS / name, scripts / name)
    harness = _load("harness", scripts)

    assert harness.codex_installer() is None
    with pytest.raises(RuntimeError, match=f"curl -sSL {_URL}/install \\| python3 -"):
        harness.install_codex(_URL)


# ---------------------------------------------------------------------------
# synapse config
# ---------------------------------------------------------------------------


def _full_home() -> dict:
    return {
        "url": _URL,
        "token": _DEVICE,
        "role": "work",
        "skills_sync": True,
        "config_sync": False,
        "config_paths": ["output-styles/*.md", "agents/*.md"],
        "device": _record(),
    }


def test_config_lists_every_key_and_never_prints_the_token(cli, capsys):
    _write_home(_full_home())
    assert cli.run("config") == 0
    out = capsys.readouterr().out

    _no_token_material(out)
    lines = out.splitlines()
    assert lines[0] == f"# {_config_json()}"
    shown = {line.split()[0]: line for line in lines[1:7]}
    assert list(shown) == ["url", "token", "role", "skills_sync", "config_sync", "config_paths"]
    assert "(set; not shown)" in shown["token"]
    assert f" {_URL} " in shown["url"]
    assert " work " in shown["role"]
    assert " on " in shown["skills_sync"] and " off " in shown["config_sync"]
    assert "output-styles/*.md, agents/*.md" in shown["config_paths"]
    assert lines[7].split() == ["(enrolled)", _SURFACE_ID, "(full)"]


@pytest.mark.parametrize(
    ("key", "shown"),
    [
        ("url", _URL),
        ("token", "(set; not shown)"),
        ("role", "work"),
        ("skills_sync", "on"),
        ("config_sync", "off"),
        ("config_paths", "output-styles/*.md, agents/*.md"),
    ],
)
def test_config_key_prints_its_value(cli, capsys, key, shown):
    _write_home(_full_home())
    assert cli.run("config", key) == 0
    assert capsys.readouterr().out == shown + "\n"


def test_config_key_prints_unset_for_a_missing_value(cli, capsys):
    assert cli.run("config", "role") == 0
    assert capsys.readouterr().out == "(unset)\n"


@pytest.mark.parametrize(
    ("key", "text", "stored"),
    [
        ("role", " Work ", "work"),
        ("role", "personal", "personal"),
        ("skills_sync", "off", False),
        ("skills_sync", "YES", True),
        ("config_sync", "1", True),
        ("config_sync", "no", False),
        (
            "config_paths",
            "output-styles/*.md, agents/*.md  x.md",
            ["output-styles/*.md", "agents/*.md", "x.md"],
        ),
        ("url", "https://synapse.example.net/mcp/", _URL),
    ],
)
def test_config_key_value_validates_and_saves(cli, key, text, stored):
    _write_home({"url": _OLD_URL})
    assert cli.run("config", key, text) == 0
    assert _read_home()[key] == stored


@pytest.mark.parametrize(
    ("key", "text", "error"),
    [
        ("role", "admin", "role must be personal or work"),
        ("skills_sync", "maybe", "skills_sync takes on/off"),
        ("config_sync", "2", "config_sync takes on/off"),
        ("url", "ftp://synapse.example.net", "not a plain http(s) base URL"),
        ("url", "https://synapse.example.net/a b", "not a plain http(s) base URL"),
        ("token", "   ", "token can't be empty; use --unset"),
    ],
)
def test_config_rejects_an_invalid_value_and_changes_nothing(cli, capsys, key, text, error):
    _write_home(_full_home())
    before = _config_json().read_bytes()
    assert cli.run("config", key, text) == 1
    assert error in capsys.readouterr().err
    assert _config_json().read_bytes() == before
    assert cli.wired == []


def test_config_unset_removes_only_that_key(cli, capsys):
    _write_home(_full_home())
    assert cli.run("config", "role", "--unset") == 0
    assert capsys.readouterr().out == "role unset\n"
    expected = _full_home()
    del expected["role"]
    assert _read_home() == expected


def test_config_unset_token_also_forgets_the_enrollment_record(cli):
    _write_home(_full_home())
    _write_legacy_record(_record())
    assert cli.run("config", "token", "--unset") == 0
    home = _read_home()
    assert "token" not in home and "device" not in home
    assert not _legacy_device_file().exists()


def test_unsetting_a_stray_config_json_token_keeps_the_enrollment_it_does_not_describe(cli):
    """The session-start notice for a config.json token the record does not describe
    offers two fixes: `synapse login --reenroll` (replace the enrollment) or
    `synapse config token --unset` (drop the stray token, keep the enrollment)."""
    options = {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}
    _write_settings({"pluginConfigs": {"synapse@synapse": {"options": options}}})
    _write_home({"url": _URL, "token": _NEW, "device": _record(_DEVICE)})
    assert "`synapse config token --unset`" in cli.load().config.token_override_notice()

    assert cli.run("config", "token", "--unset") == 0
    home = _read_home()
    assert "token" not in home
    assert home.get("device") == _record(_DEVICE)


def test_config_token_replaces_the_token_and_drops_the_record_that_described_the_old_one(
    cli, capsys
):
    _write_home(_full_home())
    _write_legacy_record(_record())
    assert cli.run("config", "token", _NEW) == 0

    home = _read_home()
    assert home["token"] == _NEW
    assert "device" not in home
    assert not _legacy_device_file().exists()
    out = capsys.readouterr().out
    assert out == "token = (set; not shown)\n"
    _no_token_material(out)


def test_config_role_on_an_enrolled_machine_says_when_it_takes_effect(cli, capsys):
    _write_home(_full_home())
    assert cli.run("config", "role", "personal") == 0
    assert "synapse login --reenroll" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("plugin_id", "codex", "wired"),
    [
        ("synapse@synapse", False, [("claude", _URL)]),
        (None, True, [("codex", _URL, False)]),
        ("synapse@my-fork", True, [("claude", _URL), ("codex", _URL, False)]),
        (None, False, []),
    ],
)
def test_config_url_rewires_only_the_harnesses_already_wired(cli, plugin_id, codex, wired):
    _write_home({"url": _OLD_URL})
    cli.found = {"claude": True, "codex": True}  # installed is not the same as wired
    cli.plugin_id, cli.codex_is_wired = plugin_id, codex
    assert cli.run("config", "url", _URL + "/") == 0
    assert cli.wired == wired


def test_config_url_reports_a_harness_it_could_not_rewire(cli, capsys):
    cli.codex_is_wired = True
    cli.wire_errors["codex"] = ValueError("config.toml left unchanged")
    assert cli.run("config", "url", _URL) == 1
    assert "Codex: not wired: config.toml left unchanged" in capsys.readouterr().out
    assert _read_home()["url"] == _URL


@pytest.mark.parametrize(
    "home",
    [
        _enrolled_home(url=_OLD_URL, server=_OLD_URL),
        # A pasted token (`synapse login --token`): no record, but a credential all the same.
        {"url": _OLD_URL, "token": _DEVICE},
        # A record from before `server` was kept.
        {"url": _OLD_URL, "device": {"surface_id": _SURFACE_ID, "trust": "full"}},
    ],
)
def test_config_url_refuses_to_carry_a_credential_to_another_server(cli, capsys, home):
    """The old server's token must never be sent to the new one: a move is a sign-in."""
    _write_home(home)
    cli.codex_is_wired = True
    assert cli.run("config", "url", _URL) == 1
    err = capsys.readouterr().err
    assert f"synapse login --url {_URL} --reenroll" in err and "--same-server" in err
    assert _read_home() == home and cli.wired == []


def test_config_url_same_server_moves_the_credential_with_the_address(cli):
    _write_home(_enrolled_home(url=_OLD_URL, server=_OLD_URL))
    assert cli.run("config", "url", _URL, "--same-server") == 0
    home = _read_home()
    assert home["url"] == _URL and home["token"] == _DEVICE
    assert home["device"]["server"] == _URL
    assert cli.mod.config.read_device_state()["server"] == _URL


def test_config_url_on_the_same_origin_needs_no_flag(cli):
    _write_home(_enrolled_home(url=_URL, server=_URL))
    assert cli.run("config", "url", _URL + "/mcp") == 0
    assert _read_home()["url"] == _URL


def _plugin_settings(url: str, token: str | None = None) -> dict:
    options = {"SYNAPSE_URL": url, **({"SYNAPSE_INGEST_TOKEN": token} if token else {})}
    return {"pluginConfigs": {"synapse@synapse": {"options": options}}}


def test_install_refuses_a_hand_edited_url_the_credential_does_not_belong_to(cli, capsys):
    """Claude Code falls back to the plugin's static token header when the helper
    refuses: re-pointing the plugin at another origin would send it the old token."""
    home = {"url": _URL, "token": _DEVICE, "device": _record(server=_OLD_URL)}
    _write_home(home)
    _write_settings(_plugin_settings(_OLD_URL, _DEVICE))
    cli.found = {"claude": True, "codex": True}
    assert cli.run("install") == 1
    err = capsys.readouterr().err
    assert f"synapse login --url {_URL} --reenroll" in err and "--same-server" in err
    assert cli.wired == []
    _no_token_material(err)


def test_install_refuses_an_environment_url_and_says_where_it_came_from(cli, capsys, monkeypatch):
    _write_home(_enrolled_home())
    monkeypatch.setenv("SYNAPSE_URL", _OLD_URL)
    cli.found = {"claude": True, "codex": False}
    assert cli.run("install", "claude") == 1
    assert "SYNAPSE_URL in your environment" in capsys.readouterr().err
    assert cli.wired == []


def test_config_url_sees_a_token_claude_code_keeps_in_its_credentials_file(cli, capsys):
    """A 0.19 install dialog stored a pasted token as a plugin secret: no record, no
    plaintext copy, and still a credential for the plugin's URL."""
    _write_settings(_plugin_settings(_OLD_URL))
    creds = Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".credentials.json"
    creds.write_text(
        json.dumps({"pluginSecrets": {"synapse@synapse": {"SYNAPSE_INGEST_TOKEN": _DEVICE}}})
    )
    assert cli.run("config", "url", _URL) == 1
    assert "--reenroll" in capsys.readouterr().err
    assert _read_home().get("url") == _OLD_URL  # adopted, as login would; not moved


def test_on_macos_a_configured_plugin_may_hold_a_keychain_token(cli, capsys, monkeypatch):
    _write_settings(_plugin_settings(_OLD_URL))
    mod = cli.load()
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    assert cli.run("config", "url", _URL) == 1
    assert cli.run("config", "url", _URL, "--same-server") == 0
    monkeypatch.setattr(mod.sys, "platform", "linux")
    assert cli.run("config", "url", "https://third.example.org") == 0  # nothing held there


def test_a_failed_sign_in_elsewhere_keeps_the_url_even_with_no_credential_in_sight(cli, capsys):
    """The token may be in the macOS keychain: the URL a sign-in moves to is saved only
    once that sign-in succeeds."""
    _write_settings(_plugin_settings(_OLD_URL))
    cli.signin_ok = False
    assert cli.run("login", "--url", _URL, "--no-install") != 0
    assert len(cli.signins) == 1
    assert _read_home().get("url") == _OLD_URL
    assert f"still points at {_OLD_URL}" in capsys.readouterr().out


def test_a_plugin_url_holding_the_device_token_is_not_a_move(cli):
    """0.19: Claude's own dialog moved the plugin to the same server's new address; its
    config holds the device token and its MCP header sends it there already."""
    _write_home({"url": _URL, "token": _DEVICE, "device": _record(server=_OLD_URL)})
    _write_settings(_plugin_settings(_URL, _DEVICE))
    cli.found = {"claude": True, "codex": False}
    assert cli.run("login") == 0
    assert cli.signins == [] and cli.wired == [("claude", _URL)]


def test_same_server_moves_a_record_too_old_to_name_its_server(cli):
    """Enrolled on 0.19.0-0.19.2: the token in the plugin config, a record with no
    server. `--same-server` stamps the new address into the record and re-wires the
    plugin, so the hooks follow."""
    _write_settings(_plugin_settings(_OLD_URL, _DEVICE))
    _write_legacy_record({"surface_id": _SURFACE_ID, "trust": "full"})
    cli.plugin_id = "synapse@synapse"
    assert cli.run("config", "url", _URL) == 1  # still a move without the flag
    assert cli.run("config", "url", _URL, "--same-server") == 0
    assert cli.wired == [("claude", _URL)]
    assert _read_home()["device"]["server"] == _URL
    # Adopted first, as login would: the record pins its token, so it is still
    # recognised once the plaintext copy goes.
    assert _read_home()["device"]["token_sha256"] == _sha(_DEVICE)
    _write_settings(_plugin_settings(_URL, _DEVICE))  # what the re-wire leaves behind
    _forget_modules()
    cfg = _load("config")
    assert cfg.BASE_URL == _URL and cfg.INGEST_TOKEN == _DEVICE


def test_same_server_moves_a_token_only_the_plugin_holds(cli):
    _write_settings(_plugin_settings(_OLD_URL, _DEVICE))
    cli.plugin_id = "synapse@synapse"
    assert cli.run("config", "url", _URL, "--same-server") == 0
    assert cli.wired == [("claude", _URL)] and _read_home()["url"] == _URL


def test_on_macos_a_dialog_changed_plugin_url_is_still_a_move_for_the_device_token(
    cli, capsys, monkeypatch
):
    """The keychain guess may widen what is refused, never where the device token goes:
    the hooks go by the record, so the record must move with `--same-server`."""
    _write_home({"url": _OLD_URL, "token": _DEVICE, "device": _record(server=_OLD_URL)})
    _write_settings(_plugin_settings(_URL))  # changed in Claude's /plugin dialog
    mod = cli.load()
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    assert cli.run("login", "--url", _URL, "--no-install") == 1
    capsys.readouterr()
    assert cli.run("config", "url", _URL) == 1
    err = capsys.readouterr().err
    assert "--same-server" in err
    # Names the server the credential is for, never the one refused.
    assert f"credential is for {_OLD_URL}, and {_URL} must not" in err
    assert cli.run("config", "url", _URL, "--same-server") == 0
    assert _read_home()["device"]["server"] == _URL
    _forget_modules()
    cfg = _load("config")
    assert cfg.BASE_URL == _URL and cfg.INGEST_TOKEN == _DEVICE


@pytest.mark.parametrize("how", ["token", "reenroll"])
def test_a_move_without_wiring_takes_the_token_out_of_the_old_plugins_static_slot(cli, how):
    """--no-install leaves the plugin pointing at the old server, and its static header
    goes there whenever the helper refuses: it must not carry the new server's token."""
    _write_home({"url": _OLD_URL, "token": _DEVICE, "device": _record(server=_OLD_URL)})
    _write_settings(_plugin_settings(_OLD_URL, _DEVICE))
    argv = ["--token", _NEW] if how == "token" else ["--reenroll"]
    assert cli.run("login", "--url", _URL, *argv, "--no-install") == 0
    assert _read_home()["token"] == _NEW
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert options["SYNAPSE_URL"] == _OLD_URL and "SYNAPSE_INGEST_TOKEN" not in options


def test_the_static_slot_follows_a_new_token_for_the_server_it_points_at(cli):
    _write_home({"url": _URL, "token": _DEVICE})
    _write_settings(_plugin_settings(_URL, _DEVICE))
    assert cli.run("login", "--token", _NEW, "--no-install") == 0
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert options["SYNAPSE_INGEST_TOKEN"] == _NEW


@pytest.mark.parametrize("where", ["keychain", "credentials-file"])
def test_an_old_record_whose_token_only_claude_code_holds_still_refuses_a_move(
    cli, capsys, monkeypatch, where
):
    """Enrolled before 0.19.3, the token re-saved through the /plugin dialog into Claude
    Code's secure storage, `config url` before any login: nothing on disk names where the
    token goes but the plugin's URL."""
    _write_settings(_plugin_settings(_OLD_URL))
    _write_legacy_record({"surface_id": _SURFACE_ID, "trust": "full"})
    mod = cli.load()
    if where == "keychain":
        monkeypatch.setattr(mod.sys, "platform", "darwin")
    else:
        creds = Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".credentials.json"
        creds.write_text(
            json.dumps({"pluginSecrets": {"synapse@synapse": {"SYNAPSE_INGEST_TOKEN": _DEVICE}}})
        )
    cli.plugin_id = "synapse@synapse"
    assert cli.run("config", "url", _URL) == 1
    assert cli.wired == [] and _read_home().get("url") != _URL
    assert cli.run("config", "url", _URL, "--same-server") == 0
    assert _read_home()["device"]["server"] == _URL


def test_a_wired_move_keeps_the_static_slot_in_step_with_the_new_token(cli):
    """The copy comes out while the plugin still points at the old server, and goes back
    once login has pointed the plugin at the new one."""
    _write_home({"url": _OLD_URL, "token": _DEVICE, "device": _record(server=_OLD_URL)})
    _write_settings(_plugin_settings(_OLD_URL, _DEVICE))
    cli.found = {"claude": True, "codex": False}

    def install_claude(url, runner=None):
        cli.wired.append(("claude", url))
        settings = _read_settings()
        settings["pluginConfigs"]["synapse@synapse"]["options"]["SYNAPSE_URL"] = url
        _write_settings(settings)
        return f"Claude Code: plugin installed, SYNAPSE_URL={url}"

    cli.load()
    cli.mp.setattr(cli.mod.harness, "install_claude", install_claude)
    assert cli.run("login", "--url", _URL, "--token", _NEW) == 0
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert options == {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _NEW}


def test_a_sign_in_goes_to_the_url_login_was_given_whatever_the_shell_exports(cli, monkeypatch):
    """`SYNAPSE_URL=E synapse login --url B --reenroll`: enrolling with E would put E's
    token into config.json for B, and into the plugin's static slot at B."""
    _write_home(_enrolled_home())
    _write_settings(_plugin_settings(_URL, _DEVICE))
    monkeypatch.setenv("SYNAPSE_URL", "https://elsewhere.example.org")
    assert cli.run("login", "--url", _URL, "--reenroll", "--no-install") == 0
    home = _read_home()
    assert home["device"]["server"] == _URL and home["token"] == _NEW
    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert options == {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _NEW}
    assert os.environ["SYNAPSE_URL"] == "https://elsewhere.example.org"  # put back


def test_the_restored_slot_never_takes_a_token_minted_by_another_server(cli):
    """The record says the token came from another server: whatever the plugin points
    at, its static slot doesn't get it."""
    _write_home({"url": _URL, "token": _NEW, "device": _record(_NEW, server=_OLD_URL)})
    _write_settings(_plugin_settings(_URL))
    cli.mod = None
    mod = cli.load()
    mod._restore_static_slot(_URL)
    assert (
        "SYNAPSE_INGEST_TOKEN"
        not in _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    )


def test_the_static_slot_is_written_to_the_enabled_plugins_block(cli):
    """A fork's leftover block next to the installed plugin's must not get the token."""
    settings = {
        "enabledPlugins": {"synapse@my-fork": False, "synapse@synapse": True},
        "pluginConfigs": {
            "synapse@my-fork": {"options": {"SYNAPSE_URL": "https://old-fork.example.net"}},
            "synapse@synapse": {"options": {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}},
        },
    }
    _write_settings(settings)
    _write_home({"url": _URL, "token": _DEVICE})
    assert cli.run("login", "--token", _NEW, "--no-install") == 0
    configs = _read_settings()["pluginConfigs"]
    assert configs["synapse@my-fork"]["options"] == {"SYNAPSE_URL": "https://old-fork.example.net"}
    assert configs["synapse@synapse"]["options"]["SYNAPSE_INGEST_TOKEN"] == _NEW


def test_config_unknown_key_fails(cli, capsys):
    assert cli.run("config", "colour", "blue") == 1
    assert "unknown key 'colour'" in capsys.readouterr().err
    assert not _config_json().exists()


# ---------------------------------------------------------------------------
# synapse install
# ---------------------------------------------------------------------------


def test_install_needs_a_configured_url(cli, capsys):
    cli.found = {"claude": True, "codex": True}
    assert cli.run("install") == 1
    assert "no server URL yet" in capsys.readouterr().err
    assert cli.wired == []


@pytest.mark.parametrize(
    ("argv", "wired"),
    [
        (["claude"], [("claude", _URL)]),
        (["codex"], [("codex", _URL, False)]),
        (["codex", "--dry-run"], [("codex", _URL, True)]),
    ],
)
def test_install_wires_the_named_harness_even_when_none_was_detected(cli, argv, wired):
    _write_home({"url": _URL})
    assert cli.run("install", *argv) == 0
    assert cli.wired == wired


def test_install_without_a_name_wires_every_detected_harness(cli):
    _write_home({"url": _URL})
    cli.found = {"claude": True, "codex": False}
    assert cli.run("install") == 0
    assert cli.wired == [("claude", _URL)]


def test_install_dry_run_passes_through_and_never_touches_claude(cli, capsys):
    _write_home({"url": _URL})
    cli.found = {"claude": True, "codex": True}
    assert cli.run("install", "--dry-run") == 0
    assert cli.wired == [("codex", _URL, True)]
    assert f"Claude Code: would install the plugin with SYNAPSE_URL={_URL}" in (
        capsys.readouterr().out
    )


def test_install_takes_the_url_from_the_legacy_plugin_settings(cli):
    _write_settings(
        {"pluginConfigs": {"synapse@synapse": {"options": {"SYNAPSE_URL": _URL + "/"}}}}
    )
    assert cli.run("install", "codex") == 0
    assert cli.wired == [("codex", _URL, False)]


# ---------------------------------------------------------------------------
# synapse skill retire
# ---------------------------------------------------------------------------


def _fake_server(cli: _Cli, reply) -> list[tuple]:
    calls: list[tuple] = []

    def post_json(path, payload, timeout=30.0):
        calls.append((path, payload))
        if isinstance(reply, Exception):
            raise reply
        return reply

    cli.mp.setattr(cli.load().config, "post_json", post_json)
    return calls


@pytest.mark.parametrize(
    ("reply", "code", "stream", "text"),
    [
        (
            {"status": "retired", "name": "pdf-tools", "scope": None},
            0,
            "out",
            "Retired pdf-tools (global).",
        ),
        (
            {"status": "retired", "name": "pdf-tools", "scope": "project:synapse"},
            0,
            "out",
            "Retired pdf-tools (project:synapse).",
        ),
        (
            {"status": "already_retired", "retired_at": "2026-10-01T00:00:00Z"},
            0,
            "out",
            "pdf-tools was already retired (2026-10-01T00:00:00Z).",
        ),
        ({"status": "not_found"}, 1, "err", "no skill named 'pdf-tools' in Synapse"),
        (
            {"status": "error", "detail": "schema 060 is not applied"},
            1,
            "err",
            "synapse: schema 060 is not applied",
        ),
        ({"status": "surprise"}, 1, "err", "retire failed: {'status': 'surprise'}"),
    ],
)
def test_skill_retire_maps_the_server_status_to_output_and_exit_code(
    cli, capsys, reply, code, stream, text
):
    calls = _fake_server(cli, reply)
    assert cli.run("skill", "retire", "pdf-tools") == code
    assert calls == [("/skills/retire", {"name": "pdf-tools"})]
    assert text in getattr(capsys.readouterr(), stream)


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (urllib.error.HTTPError(_URL, 403, "Forbidden", None, None), "retire failed: HTTP 403"),
        (urllib.error.URLError("connection refused"), "retire failed: <urlopen error"),
    ],
)
def test_skill_retire_fails_on_an_http_or_network_error(cli, capsys, error, text):
    _fake_server(cli, error)
    assert cli.run("skill", "retire", "pdf-tools") == 1
    assert text in capsys.readouterr().err


# ---------------------------------------------------------------------------
# synapse update
# ---------------------------------------------------------------------------


_INSTALLER = b"print('the server's installer')\n"


def _fake_update(cli: _Cli, *, fetch_error=None, returncode: int = 0) -> dict:
    """Fake <url>/install and the interpreter it is piped into."""
    seen: dict = {"requests": [], "runs": []}

    def urlopen(req, timeout=None):
        seen["requests"].append(req)
        if fetch_error:
            raise fetch_error
        return _Resp(_INSTALLER)

    def run(cmd, input=None, check=None, **_kw):
        seen["runs"].append((cmd, input))
        return subprocess.CompletedProcess(cmd, returncode)

    mod = cli.load()
    cli.mp.setattr(mod.urllib.request, "urlopen", urlopen)
    cli.mp.setattr(mod.subprocess, "run", run)
    return seen


def test_update_runs_the_servers_installer_then_updates_claude_and_rewires_codex(cli, capsys):
    _write_home({"url": _URL})
    cli.claude_update = "Claude Code: plugin updated"
    cli.codex_is_wired = True
    seen = _fake_update(cli)

    assert cli.run("update") == 0
    (req,) = seen["requests"]
    assert req.full_url == f"{_URL}/install"
    assert req.get_header("User-agent") == "synapse-cli"
    assert seen["runs"] == [([sys.executable, "-", _URL], _INSTALLER)]
    assert cli.wired == [("codex", _URL, False)]
    out = capsys.readouterr().out
    assert "Claude Code: plugin updated" in out and "Codex: wired" in out


def test_update_leaves_codex_alone_when_it_was_never_wired(cli):
    _write_home({"url": _URL})
    _fake_update(cli)
    assert cli.run("update") == 0
    assert cli.wired == []


def test_a_failed_installer_stops_the_update(cli, capsys):
    _write_home({"url": _URL})
    cli.codex_is_wired = True
    seen = _fake_update(cli, returncode=1)

    assert cli.run("update") == 1
    assert "the installer failed; the previous client is still in place" in (
        capsys.readouterr().err
    )
    assert len(seen["runs"]) == 1
    assert cli.wired == []


def test_an_installer_that_cannot_be_fetched_is_never_run(cli, capsys):
    _write_home({"url": _URL})
    seen = _fake_update(cli, fetch_error=urllib.error.URLError("timed out"))
    assert cli.run("update") == 1
    assert f"could not fetch {_URL}/install" in capsys.readouterr().err
    assert seen["runs"] == []


def test_update_needs_a_configured_url(cli, capsys):
    seen = _fake_update(cli)
    assert cli.run("update") == 1
    assert "no server URL yet" in capsys.readouterr().err
    assert seen["requests"] == []


def test_a_codex_rewire_failure_after_an_update_is_reported_not_fatal(cli, capsys):
    _write_home({"url": _URL})
    cli.codex_is_wired = True
    cli.wire_errors["codex"] = RuntimeError("no Codex client files here")
    _fake_update(cli)
    assert cli.run("update") == 0
    assert "Codex: not re-wired: no Codex client files here" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# passthrough commands and version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "module"),
    [("devices", "device_admin"), ("private", "private_mode"), ("import", "import_history")],
)
def test_passthrough_commands_hand_their_arguments_to_the_tool(cli, monkeypatch, command, module):
    calls: list[list[str]] = []
    fake = types.ModuleType(module)
    fake.main = lambda argv: calls.append(argv)
    monkeypatch.setitem(sys.modules, module, fake)

    assert cli.run(command, "list", "--json") == 0
    assert calls == [["list", "--json"]]


def test_a_passthrough_tools_exit_code_is_the_commands(cli, monkeypatch):
    fake = types.ModuleType("device_admin")
    fake.main = lambda _argv: 3
    monkeypatch.setitem(sys.modules, "device_admin", fake)
    assert cli.run("devices", "revoke", "dev-x") == 3


def test_version_prints_the_plugin_manifest_version(cli, capsys):
    manifest = json.loads((_REPO / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert cli.run("version") == 0
    assert capsys.readouterr().out == f"{manifest['version']}\n"


# ---------------------------------------------------------------------------
# Hardening from review: moving servers, token-only machines, secrets
# ---------------------------------------------------------------------------


def test_enrollment_requests_carry_no_bearer(monkeypatch):
    """/device/code and /surfaces/enroll take none, and enrolling is how a machine moves
    to another server: that server must never be handed the old server's token."""
    _write_home(_enrolled_home())
    enroll = _load("enroll")
    seen: list[str | None] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.get_header("Authorization"))
        return _Resp(json.dumps({"error": "access_denied"}).encode())

    monkeypatch.setattr(enroll.config.urllib.request, "urlopen", fake_urlopen)
    enroll._post("/device/code", {})
    assert seen == [None]
    assert enroll.config.INGEST_TOKEN == _DEVICE  # the process's own token is back


def test_a_failed_move_to_another_server_leaves_the_old_one_in_place(cli, capsys):
    before = _enrolled_home(url=_OLD_URL, server=_OLD_URL)
    _write_home(before)
    cli.signin_ok = False
    assert cli.run("login", "--url", _URL, "--reenroll", "--role", "personal") == 1

    assert _read_home() == {**before, "role": "personal"}
    assert "SYNAPSE_URL" not in os.environ
    assert f"still points at {_OLD_URL}" in capsys.readouterr().out


def test_a_machine_set_up_with_a_token_is_not_signed_in_again(cli, capsys):
    """`synapse login --token` stores a working credential and no record. Re-running
    login (to wire a harness installed since) must not ask for an approval, which on a
    server with no sign-in would just fail before wiring."""
    _write_home({"url": _URL, "token": _DEVICE})
    cli.found = {"claude": False, "codex": True}
    assert cli.run("login") == 0

    assert cli.signins == [] and cli.login_argv == []
    assert cli.wired == [("codex", _URL, False)]
    assert "never enrolled by signing in" in capsys.readouterr().out


@pytest.mark.parametrize("how", ["prompt", "stdin"])
def test_the_token_can_be_given_without_putting_it_in_argv(cli, monkeypatch, how):
    if how == "prompt":
        cli.terminal()
        monkeypatch.setattr(cli.load().getpass, "getpass", lambda _prompt: f"  {_NEW}  ")
        argv = ("login", "--url", _URL, "--skills-sync", "no", "--token", "--no-install")
    else:
        monkeypatch.setattr(sys, "stdin", io.StringIO(_NEW + "\n"))
        argv = ("login", "--url", _URL, "--token", "-", "--no-install")
    assert cli.run(*argv) == 0
    assert _read_home()["token"] == _NEW


def test_adopt_legacy_reads_a_token_claude_code_kept_as_a_plugin_secret():
    """Typed into the old install dialog, the token is a sensitive option: Claude Code
    keeps it in .credentials.json (Linux, Windows), not in settings.json."""
    _write_settings(_legacy_settings({"SYNAPSE_URL": _URL}))
    (_settings_path().parent / ".credentials.json").write_text(
        json.dumps({"pluginSecrets": {"synapse@synapse": {"SYNAPSE_INGEST_TOKEN": _DEVICE}}})
    )
    assert "token" in _load("harness").adopt_legacy()
    assert _read_home()["token"] == _DEVICE


def test_unsetting_the_token_also_clears_the_legacy_plaintext_copy(cli, capsys):
    _write_home({"url": _URL, "token": _DEVICE})
    _write_settings(_legacy_settings({"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}))
    assert cli.run("config", "token", "--unset") == 0

    options = _read_settings()["pluginConfigs"]["synapse@synapse"]["options"]
    assert "SYNAPSE_INGEST_TOKEN" not in options and options["SYNAPSE_URL"] == _URL
    assert "token" not in _read_home()
    assert "secure storage" in capsys.readouterr().out


def test_config_listing_says_where_an_unset_value_comes_from(cli, capsys):
    _write_settings(_legacy_settings({"SYNAPSE_URL": _OLD_URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}))
    _write_home({})
    assert cli.run("config") == 0
    out = capsys.readouterr().out
    assert f"{_OLD_URL} (from the Claude plugin's settings)" in out
    assert "(set; not shown, from the Claude plugin's settings)" in out
    _no_token_material(out)


def test_the_windows_headers_helper_names_its_interpreter_by_absolute_path(monkeypatch):
    """cmd.exe runs Codex's helper with the opened repository as cwd and looks there for
    a bare `py` first; an absolute, quoted path can't be shadowed by a planted py.bat."""
    installer = _load_installer()
    monkeypatch.setattr(installer.os, "name", "nt")
    monkeypatch.setattr(
        installer.shutil, "which", lambda name: r"C:\Windows\py.exe" if name == "py" else None
    )
    assert installer.helper_interpreter("py -3") == '"C:\\Windows\\py.exe" -3'
    assert installer.helper_interpreter("python3") == "python3"  # not found: unchanged
    # A hit in the current directory (Python 3.11's which, or one that ignores the
    # variable) is relative: never written.
    monkeypatch.setattr(installer.shutil, "which", lambda name: ".\\py.BAT")
    assert installer.helper_interpreter("py -3") == "py -3"
    seen = []
    monkeypatch.setattr(
        installer.shutil,
        "which",
        lambda name: seen.append(os.environ.get("NoDefaultCurrentDirectoryInExePath")),
    )
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    installer.helper_interpreter("py -3")
    assert seen == ["1"] and "NoDefaultCurrentDirectoryInExePath" not in os.environ
    launcher = installer.Launcher("py -3", helper_python='"C:\\Windows\\py.exe" -3')
    assert launcher.command("plugin/scripts/mcp_headers.py", helper=True).startswith(
        '"C:\\Windows\\py.exe" -3 "'
    )
    assert launcher.command("plugin-codex/hooks/session_start.py").startswith('py -3 "')


def _load_installer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "synapse_install_hardening", _SCRIPTS.parents[1] / "plugin-codex" / "install.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
