"""Device enrollment on the client side (plugin/scripts/enroll.py, schema 054).

The hook that runs this has one job and two hard constraints: swap the shared
enrollment credential for a token minted for THIS machine, and never break a session
start doing it. So the tests are about the ordering and the silence:

  * the minted token is persisted BEFORE the local record, because a machine with a
    token and no record re-enrolls harmlessly while a machine with a record and no
    token is locked out;
  * a pending device gets a pairing block carrying its code — the same code the trusted
    machine's board shows, since an approval only one side can see cannot be checked;
  * every failure (no credential, server down, pre-054 server, malformed reply) leaves
    NO record, so the next session retries.

Stdlib-only script loaded by path. No live server: config.post_json is monkeypatched.
"""

from __future__ import annotations

import importlib.util
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
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


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
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-laptop")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", "enrollment-credential")
    monkeypatch.chdir(tmp_path)


def _reload_hook(monkeypatch, tmp_path) -> ModuleType:
    """Re-import after an env change — MACHINE_ROLE is resolved at module import."""
    _isolate(monkeypatch, tmp_path)
    return _load()


@pytest.fixture()
def hook(monkeypatch, tmp_path) -> ModuleType:
    """A freshly loaded enroll.py pointed at a scratch config dir and data dir."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("SYNAPSE_MACHINE_ROLE", raising=False)
    yield _load()
    for name in ("config", "enroll"):
        sys.modules.pop(name, None)


def _reply(monkeypatch, hook, reply) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def fake_post(path, payload, timeout=30.0):
        calls.append((path, dict(payload)))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(hook.config, "post_json", fake_post)
    return calls


_PENDING = {
    "status": "pending",
    "pair_code": "AB12CD",
    "token": "device-token-xyz",
    "surface": {"surface_id": "dev-abc123", "status": "pending", "pair_code": "AB12CD"},
}
_APPROVED = {
    "status": "approved",
    "pair_code": None,
    "token": "device-token-first",
    "surface": {"surface_id": "dev-first", "status": "approved", "trust": "full"},
}


def test_enrollment_sends_the_hostname_as_a_display_label_only(monkeypatch, hook):
    """The hostname is the ONLY machine-supplied string in the request, and it is
    display-only server-side: nothing resolves a row by it."""
    calls = _reply(monkeypatch, hook, _PENDING)
    hook.ensure_enrolled()
    # No role set on this fixture, so no request rides along.
    assert calls == [("/surfaces/enroll", {"label": "test-laptop", "requested_trust": "full"})]


def test_the_minted_token_replaces_the_enrollment_credential(monkeypatch, hook, tmp_path):
    """Same config slot — settings.json's ``SYNAPSE_INGEST_TOKEN`` — so plugin.json's
    ``Bearer ${user_config.SYNAPSE_INGEST_TOKEN}`` header picks the device token up with
    no change on either side. The client never has to manage two credentials.

    (An explicit env var still outranks settings.json, which is why this reads the file
    rather than ``_cfg``: this fixture sets the env var to stand in for the pasted
    install-prompt credential.)"""
    import json

    _reply(monkeypatch, hook, _PENDING)
    hook.ensure_enrolled()
    settings = json.loads((tmp_path / "claude" / "settings.json").read_text())
    opts = next(iter(settings["pluginConfigs"].values()))["options"]
    assert opts["SYNAPSE_INGEST_TOKEN"] == "device-token-xyz"


def test_the_credential_is_persisted_before_the_local_record(monkeypatch, hook):
    """Ordering matters on a crash: a machine with a token and no record re-enrolls
    (one stray pending row, harmless); a machine with a record and no token is locked
    out with a credential it can never obtain again."""
    order: list[str] = []
    monkeypatch.setattr(hook.config, "write_user_config", lambda k, v: order.append(f"config:{k}"))
    monkeypatch.setattr(hook.config, "write_device_state", lambda s: order.append("state"))
    _reply(monkeypatch, hook, _PENDING)
    hook.ensure_enrolled()
    assert order == ["config:SYNAPSE_INGEST_TOKEN", "state"]


def test_a_pending_enrollment_records_its_pair_code(monkeypatch, hook):
    _reply(monkeypatch, hook, _PENDING)
    state = hook.ensure_enrolled()
    assert state == {
        "surface_id": "dev-abc123",
        "status": "pending",
        "pair_code": "AB12CD",
        "label": "test-laptop",
        "role": "personal",
    }
    assert hook.config.read_device_state() == state


def test_the_pairing_block_shows_the_code_and_says_why_memory_is_empty(monkeypatch, hook):
    _reply(monkeypatch, hook, _PENDING)
    block = hook.pairing_block(hook.ensure_enrolled())
    assert "AB12CD" in block
    assert "test-laptop" in block
    assert "not approved yet" in block


def test_a_bootstrap_enrollment_is_approved_with_no_code(monkeypatch, hook):
    _reply(monkeypatch, hook, _APPROVED)
    state = hook.ensure_enrolled()
    assert state["status"] == "approved" and state["pair_code"] is None


def test_enrollment_happens_exactly_once(monkeypatch, hook):
    calls = _reply(monkeypatch, hook, _PENDING)
    hook.ensure_enrolled()
    hook.ensure_enrolled()
    assert len(calls) == 1


def test_mark_approved_clears_the_pairing_state(monkeypatch, hook):
    """The client is never TOLD it was approved — it finds out by being served, so a
    successful board fetch is the signal that stops the pairing block printing."""
    _reply(monkeypatch, hook, _PENDING)
    hook.ensure_enrolled()
    hook.mark_approved()
    state = hook.config.read_device_state()
    assert state["status"] == "approved" and "pair_code" not in state


@pytest.mark.parametrize(
    "reply",
    [
        urllib.error.URLError("server down"),
        urllib.error.HTTPError("http://x/surfaces/enroll", 404, "nope", None, None),  # pre-054
        TimeoutError("timed out"),
        {},  # malformed / empty
        {"status": "pending", "token": "t"},  # no surface id
    ],
    ids=["conn-refused", "pre-054-server", "timeout", "empty", "no-surface-id"],
)
def test_every_failure_leaves_no_record_so_the_next_session_retries(monkeypatch, hook, reply):
    _reply(monkeypatch, hook, reply)
    assert hook.ensure_enrolled() == {}
    assert hook.config.read_device_state() == {}


def test_no_credential_means_no_call_at_all(monkeypatch, tmp_path):
    """An open/local server needs no credential and a gated one needs `synapse login`
    first. Either way, silence — not a doomed request."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for name in ("config", "enroll"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("enroll", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPTS))

    calls = _reply(monkeypatch, mod, _PENDING)
    assert mod.ensure_enrolled() == {} and calls == []
    for name in ("config", "enroll"):
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# The install-prompt machine role, sent once as a request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "asked"),
    [
        ("personal", "full"),
        ("work", "restricted"),
        ("PERSONAL", "full"),  # case-insensitive: it comes from a hand-typed prompt
        (None, "full"),  # unset -> the documented default
    ],
)
def test_the_install_role_becomes_the_requested_trust(monkeypatch, hook, role, asked, tmp_path):
    if role is None:
        monkeypatch.delenv("SYNAPSE_MACHINE_ROLE", raising=False)
    else:
        monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", role)
    mod = _reload_hook(monkeypatch, tmp_path)
    calls = _reply(monkeypatch, mod, _PENDING)
    mod.ensure_enrolled()
    assert calls == [
        ("/surfaces/enroll", {"label": "test-laptop", "requested_trust": asked}),
    ]


def test_an_unrecognised_role_asks_for_nothing(monkeypatch, hook, tmp_path):
    """Better to ask for nothing than to guess: an unstated request falls back to the
    server's narrow default, while a wrong guess could ask for full trust."""
    monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", "kiosk")
    mod = _reload_hook(monkeypatch, tmp_path)
    calls = _reply(monkeypatch, mod, _PENDING)
    mod.ensure_enrolled()
    assert calls == [("/surfaces/enroll", {"label": "test-laptop"})]


def test_the_pairing_block_names_the_role_this_machine_declared(monkeypatch, hook, tmp_path):
    monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", "work")
    mod = _reload_hook(monkeypatch, tmp_path)
    _reply(monkeypatch, mod, _PENDING)
    block = mod.pairing_block(mod.ensure_enrolled())
    assert "enrolled as a work machine" in block
