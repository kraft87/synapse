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


@pytest.mark.skipif(
    not (os.environ.get("SYNAPSE_HERMES_PYTHON") and os.environ.get("SYNAPSE_HERMES_SOURCE")),
    reason="needs a Hermes interpreter and checkout (SYNAPSE_HERMES_PYTHON, SYNAPSE_HERMES_SOURCE)",
)
@pytest.mark.parametrize("name", VERSIONS)
def test_the_import_has_hermes_scrub_the_stored_rows(monkeypatch, name):
    data = json.loads((FIXTURES / name).read_text("utf-8"))
    monkeypatch.setenv("PYTHONPATH", os.environ["SYNAPSE_HERMES_SOURCE"])
    done = scrub.rows({"s": data["raw"]}, os.environ["SYNAPSE_HERMES_PYTHON"])
    assert done is not None
    wire = json.dumps(done)
    assert "SECRETTEST" in wire and not [s for s in _secrets(data) if s in wire]
