"""The server's whole-skill hash must equal the client sync's, byte for byte.

/skills/list reports each retired skill with the hash of the content it was retired with,
and a client compares that against its own copy to tell the copy Synapse delivered (safe
to remove) from a local edit (kept). If the two formulas drift, every copy looks locally
edited and a retirement never reaches any machine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from mcp_server.skill_sync_routes import whole_skill_hash

_SCRIPTS = Path(__file__).resolve().parents[1] / "plugin" / "scripts"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def client_hash(monkeypatch):
    monkeypatch.syspath_prepend(str(_SCRIPTS))  # the engine imports the plugin's config
    monkeypatch.delitem(sys.modules, "config", raising=False)
    spec = importlib.util.spec_from_file_location(
        "synapse_plugin_skills_sync_hash", _SCRIPTS / "skills_sync.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._whole_hash


@pytest.mark.parametrize(
    "body,files",
    [
        pytest.param("", [], id="empty"),
        pytest.param("---\nname: a\n---\nbody\n", [], id="no-files"),
        pytest.param(
            "# héllo — ✓\n",
            [("scripts/ünï.sh", _sha(b"x")), ("资料/说明.md", _sha(b"y"))],
            id="unicode",
        ),
        pytest.param(
            "body",
            [("z.txt", _sha(b"z")), ("a.txt", _sha(b"a")), ("m/n.txt", _sha(b"n"))],
            id="unsorted",
        ),
    ],
)
def test_server_hash_matches_the_client_sync(client_hash, body, files):
    assert whole_skill_hash(body, files) == client_hash(body, files)
    assert whole_skill_hash(body, list(reversed(files))) == client_hash(body, files)


def test_hash_without_files_is_the_body_digest():
    assert whole_skill_hash("abc", []) == _sha(b"abc")
    assert whole_skill_hash("abc", []) != whole_skill_hash("abc", [("a", _sha(b""))])
