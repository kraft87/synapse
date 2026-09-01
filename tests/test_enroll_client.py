"""Device enrollment on the client side (plugin/scripts/enroll.py, schema 054).

The script has one job: run the IdP device flow and turn the result into a credential
that belongs to THIS machine. The tests are about the parts that are easy to get subtly
wrong and hard to notice:

  * the minted token is persisted BEFORE the local record, because a machine with a
    token and no record re-enrolls harmlessly while a machine with a record and no token
    is locked out of a credential it can never obtain again;
  * the poll loop keeps waiting on ``authorization_pending`` (HTTP 202) and stops on a
    refusal (403), which is the difference between "the human hasn't clicked yet" and
    "the human said no";
  * the install prompt's personal/work answer becomes the trust the machine enrolls at,
    and an unrecognised answer states nothing rather than guessing.

Stdlib-only script loaded by path. No live server: config.post_json is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_SCRIPT = _SCRIPTS / "enroll.py"

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MACHINE_ROLE",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_MACHINE_ROLE",
)

_START = {
    "device_code": "dc-1",
    "user_code": "ABCD-1234",
    "verification_uri": "https://idp.example.net/device",
    "interval": 0,  # no real sleeping in tests
    "expires_in": 5,
}
_MINTED = {
    "status": "ok",
    "token": "device-token-xyz",
    "login": "owner",
    "surface": {
        "surface_id": "dev-abc123",
        "trust": "restricted",
        "allowed_projects": ["work-a"],
        "status": "approved",
    },
}


def _load() -> ModuleType:
    """Load enroll.py (and the `config` it imports) fresh, so both re-read the env."""
    for name in ("config", "enroll"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("enroll", _SCRIPT)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPTS))
    return mod


def _isolate(monkeypatch, tmp_path) -> None:
    """Point the plugin config layer at a scratch config dir: no real env vars, no
    settings.json options, no credentials file, no project .claude."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-laptop")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def _reload_hook(monkeypatch, tmp_path) -> ModuleType:
    """Re-import after an env change — MACHINE_ROLE resolves at module import."""
    _isolate(monkeypatch, tmp_path)
    return _load()


@pytest.fixture()
def hook(monkeypatch, tmp_path) -> ModuleType:
    _isolate(monkeypatch, tmp_path)
    yield _load()
    for name in ("config", "enroll"):
        sys.modules.pop(name, None)


def _replies(monkeypatch, mod, enroll_replies) -> list[tuple[str, dict]]:
    """Script the two endpoints. ``enroll_replies`` is consumed one per poll; the last
    entry repeats, so a test can say "pending forever" with one item.

    The poll interval is also neutralized here: the loop's floor of 1s exists to stop a
    hostile server hammering the IdP, and honouring it would make this file a minute
    long for nothing."""
    calls: list[tuple[str, dict]] = []
    queue = list(enroll_replies)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, dict(payload)))
        if path == "/device/code":
            return dict(_START)
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(mod.config, "post_json", fake_post)
    return calls


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_enrollment_starts_the_device_flow_and_prints_the_code(monkeypatch, hook, capsys):
    _replies(monkeypatch, hook, [_MINTED])
    hook.enroll()
    err = capsys.readouterr().err
    assert "ABCD-1234" in err and "https://idp.example.net/device" in err


def test_enrollment_sends_the_device_code_hostname_and_role(monkeypatch, hook):
    """The hostname is a display label server-side; the role is authoritative, because
    the person who set it is the person about to authenticate."""
    calls = _replies(monkeypatch, hook, [_MINTED])
    hook.enroll(interactive=False)
    assert calls[0][0] == "/device/code"
    assert calls[1] == (
        "/surfaces/enroll",
        {"device_code": "dc-1", "label": "test-laptop", "trust": "full"},
    )


def test_the_minted_token_lands_in_the_config_slot_the_mcp_header_reads(
    monkeypatch, hook, tmp_path
):
    """settings.json's SYNAPSE_INGEST_TOKEN — so plugin.json's
    `Bearer ${user_config.SYNAPSE_INGEST_TOKEN}` picks the device token up with no
    change on either side. The client never manages two credentials."""
    _replies(monkeypatch, hook, [_MINTED])
    hook.enroll(interactive=False)
    settings = json.loads((tmp_path / "claude" / "settings.json").read_text())
    opts = next(iter(settings["pluginConfigs"].values()))["options"]
    assert opts["SYNAPSE_INGEST_TOKEN"] == "device-token-xyz"


def test_the_local_record_captures_what_was_granted(monkeypatch, hook):
    _replies(monkeypatch, hook, [_MINTED])
    state = hook.enroll(interactive=False)
    assert state == {
        "surface_id": "dev-abc123",
        "trust": "restricted",
        "allowed_projects": ["work-a"],
        "label": "test-laptop",
        "login": "owner",
    }
    assert hook.config.read_device_state() == state
    assert hook.is_enrolled()


def test_the_credential_is_persisted_before_the_local_record(monkeypatch, hook):
    """Ordering matters on a crash: a machine with a token and no record re-enrolls;
    a machine with a record and no token is locked out."""
    order: list[str] = []
    monkeypatch.setattr(hook.config, "write_user_config", lambda k, v: order.append(f"config:{k}"))
    monkeypatch.setattr(hook.config, "write_device_state", lambda s: order.append("state"))
    _replies(monkeypatch, hook, [_MINTED])
    hook.enroll(interactive=False)
    assert order == ["config:SYNAPSE_INGEST_TOKEN", "state"]


def test_an_already_enrolled_machine_does_not_re_enroll(monkeypatch, hook):
    calls = _replies(monkeypatch, hook, [_MINTED])
    hook.enroll(interactive=False)
    n = len(calls)
    assert hook.enroll(interactive=False)
    assert len(calls) == n  # no second device flow


# ---------------------------------------------------------------------------
# The poll loop: waiting vs refused
# ---------------------------------------------------------------------------


def test_the_loop_keeps_waiting_while_the_human_has_not_approved(monkeypatch, hook):
    """202 authorization_pending is the normal state for most of the flow. It is a 2xx
    on purpose, so urllib hands it back as an ordinary body rather than an exception."""
    pending = {"status": "pending", "error": "authorization_pending"}
    calls = _replies(monkeypatch, hook, [pending, pending, _MINTED])
    state = hook.enroll(interactive=False)
    assert state["surface_id"] == "dev-abc123"
    assert sum(1 for path, _ in calls if path == "/surfaces/enroll") == 3


def test_a_refusal_stops_the_loop_instead_of_spinning(monkeypatch, hook, capsys):
    denied = urllib.error.HTTPError(
        "http://x/surfaces/enroll",
        403,
        "denied",
        None,
        _Body({"status": "pending", "error": "access_denied"}),
    )
    calls = _replies(monkeypatch, hook, [denied])
    assert hook.enroll(interactive=False) == {}
    assert sum(1 for path, _ in calls if path == "/surfaces/enroll") == 1
    assert "access_denied" in capsys.readouterr().err
    assert hook.config.read_device_state() == {}


def test_an_allowlist_refusal_is_reported_not_retried(monkeypatch, hook, capsys):
    refused = urllib.error.HTTPError(
        "http://x/surfaces/enroll",
        403,
        "forbidden",
        None,
        _Body({"status": "error", "detail": "fake user 'stranger' not in allowlist"}),
    )
    _replies(monkeypatch, hook, [refused])
    assert hook.enroll(interactive=False) == {}
    assert "not in allowlist" in capsys.readouterr().err


def test_a_server_without_the_device_flow_says_so_once(monkeypatch, hook, capsys):
    def fake_post(path, payload, timeout=30.0):
        return {"error": "device_flow_disabled"}

    monkeypatch.setattr(hook.config, "post_json", fake_post)
    assert hook.enroll(interactive=False) == {}
    assert "device_flow_disabled" in capsys.readouterr().err
    assert hook.config.read_device_state() == {}


def test_an_unreachable_server_leaves_no_record(monkeypatch, hook):
    def fake_post(path, payload, timeout=30.0):
        raise urllib.error.URLError("server down")

    monkeypatch.setattr(hook.config, "post_json", fake_post)
    assert hook.enroll(interactive=False) == {}
    assert hook.config.read_device_state() == {} and not hook.is_enrolled()


# ---------------------------------------------------------------------------
# The install-prompt role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "declared"),
    [
        ("personal", "full"),
        ("work", "restricted"),
        ("PERSONAL", "full"),  # case-insensitive: it comes from a hand-typed prompt
        (None, "full"),  # unset -> the documented prompt default
        ("kiosk", "restricted"),  # unrecognised -> state nothing, take the narrow side
    ],
)
def test_the_install_role_becomes_the_declared_trust(monkeypatch, tmp_path, role, declared):
    if role is not None:
        monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", role)
    mod = _reload_hook(monkeypatch, tmp_path)
    if role is not None:  # _isolate cleared it; set it again for the reloaded module
        monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", role)
        mod = _load()
    calls = _replies(monkeypatch, mod, [_MINTED])
    mod.enroll(interactive=False)
    assert calls[1][1]["trust"] == declared


# ---------------------------------------------------------------------------
# The not-enrolled notice the SessionStart hook prints
# ---------------------------------------------------------------------------


def test_the_not_enrolled_block_names_the_reason_and_the_fix(hook):
    """An empty session start with no explanation is the one outcome worse than a
    restricted one."""
    block = hook.not_enrolled_block()
    assert "not enrolled" in block and "synapse-login" in block


class _Body:
    """Minimal file-like for HTTPError's fp argument. `read` is what the code under
    test calls; `close` is what HTTPError's own finalizer calls, and omitting it turns
    every one of these into an unraisable-exception warning at GC time."""

    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode()

    def read(self, *_a: object) -> bytes:
        return self._data

    def close(self) -> None:
        pass
