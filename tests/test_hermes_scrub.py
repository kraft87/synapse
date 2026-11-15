"""Secrets: what Hermes masks for a memory provider never leaves through Synapse.

Hermes scrubs every string it hands a memory provider (agent/memory_manager.py
``_redact_for_provider``, even with ``security.redact_secrets`` off); its session store
keeps the rows unscrubbed. The fixtures are one run as Hermes v0.23.1 and upstream main
wrote it to the store (``raw``) and as each handed it to a provider (``handed``),
produced by their own ``_redact_messages_for_egress``
(followups/hermes-turn-cut/review-round3/h_redact.py).

The live provider sends what it was handed. ``synapse import hermes`` reads the store
and has Hermes scrub it (harnesses/hermes/scrub.py); the last test runs that against a
real Hermes when ``SYNAPSE_HERMES_PYTHON`` (Hermes's interpreter) and
``SYNAPSE_HERMES_SOURCE`` (a Hermes checkout) are set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tests import test_hermes_provider as hp
from tests.helpers.hermes_store import HermesStore

hermes_stub = hp.hermes_stub
user = hp.user
plugin = hp.plugin
server = hp.server
_new, _start = hp._new, hp._start

from synapse_client.harnesses.hermes import scrub  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SID = "20261115_101500_aa11bb"
VERSIONS = ["hermes_scrub_v0.23.1.json", "hermes_scrub_main.json"]


def _secrets(data: dict) -> list[str]:
    return [data["key"], data["gh"], "wJalrXUtnFEMI"]


@pytest.mark.parametrize("name", VERSIONS)
def test_the_live_path_sends_what_hermes_handed_over_not_the_store(plugin, server, tmp_path, name):
    """review-round3 probe4: the store branch sent an Anthropic key, a GitHub token and an
    AWS secret the released client never sent. The rows now come from the list."""
    data = json.loads((FIXTURES / name).read_text("utf-8"))
    home = tmp_path / "hermes-a"
    store = HermesStore(home)
    store.session(SID)
    for raw in data["raw"]:
        store.add(
            SID,
            raw["role"],
            raw["content"],
            uid=raw["message_uid"],
            tool_calls=raw.get("tool_calls"),
            tool_name=raw.get("tool_name"),
            tool_call_id=raw.get("tool_call_id"),
        )
    provider = _start(_new(plugin), home, SID, platform="tui")
    provider.sync_turn("q", "a", session_id=SID, messages=data["handed"])
    wire = json.dumps([b["records"] for b in server.posts()])
    assert server.posts() and "SECRETTEST" in wire
    assert not [s for s in _secrets(data) if s in wire]


def test_the_interpreter_is_found_beside_the_hermes_program(tmp_path):
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "hermes").write_text("#!/x/python\n", encoding="utf-8")
    (venv / "python").write_text("", encoding="utf-8")
    assert scrub.interpreter(lambda _: str(venv / "hermes")) == str(venv / "python")
    assert scrub.interpreter(lambda _: None) is None
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "hermes").write_text(f"#!{sys.executable}\n", encoding="utf-8")
    assert scrub.interpreter(lambda _: str(lone / "hermes")) == sys.executable


def test_a_scrub_that_cannot_run_answers_none(tmp_path):
    assert scrub.rows({"s": [{"role": "user", "content": "x"}]}, None) is None
    assert scrub.rows({"s": [{"role": "user", "content": "x"}]}, sys.executable) is None


def _evil(root: Path) -> Path:
    """A directory whose own ``agent/memory_manager.py`` hands back the text as it is."""
    pkg = root / "agent"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "memory_manager.py").write_text(
        "def _redact_messages_for_egress(rows):\n    return list(rows)\n", encoding="utf-8"
    )
    return root


def test_code_in_the_working_directory_or_pythonpath_is_never_the_scrub(tmp_path, monkeypatch):
    """review-round4 s1_scrub.py: the child ran with the caller's working directory and
    environment, so a directory's own ``agent/memory_manager.py`` (or one on PYTHONPATH)
    was taken for Hermes's scrub and the key went on the wire. The child now runs
    isolated in an empty directory with no PYTHONPATH, and an answer whose scrub file is
    not under the interpreter's own install is refused."""
    rows = {
        "s": [{"role": "user", "content": "key sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]
    }
    evil = _evil(tmp_path / "evil")
    monkeypatch.chdir(evil)
    assert scrub.rows(rows, sys.executable) is None
    monkeypatch.setenv("PYTHONPATH", str(evil))
    assert scrub.rows(rows, sys.executable) is None


def test_an_answer_from_outside_the_install_is_refused(tmp_path):
    roots = [tmp_path / "venv"]
    (tmp_path / "venv" / "lib").mkdir(parents=True)
    inside = tmp_path / "venv" / "lib" / "agent.py"
    inside.write_text("", encoding="utf-8")
    assert scrub._within(inside, roots)
    assert not scrub._within(tmp_path / "elsewhere" / "agent.py", roots)


HERMES = [p for p in os.environ.get("SYNAPSE_HERMES_PYTHONS", "").split(os.pathsep) if p]


@pytest.mark.skipif(not HERMES, reason="needs Hermes interpreters (SYNAPSE_HERMES_PYTHONS)")
@pytest.mark.parametrize("python", HERMES)
@pytest.mark.parametrize("name", VERSIONS)
def test_the_import_has_hermes_scrub_the_stored_rows(tmp_path, monkeypatch, python, name):
    """Against a real Hermes install (v0.23.1 and main were run): the scrub is found with
    the child isolated, masks the secrets, and a PYTHONPATH or working directory holding
    another ``agent`` package changes nothing."""
    data = json.loads((FIXTURES / name).read_text("utf-8"))
    evil = _evil(tmp_path / "evil")
    monkeypatch.chdir(evil)
    monkeypatch.setenv("PYTHONPATH", str(evil))
    done = scrub.rows({"s": data["raw"]}, python)
    assert done is not None
    wire = json.dumps(done)
    assert "SECRETTEST" in wire and not [s for s in _secrets(data) if s in wire]
