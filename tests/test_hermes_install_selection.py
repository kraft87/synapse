"""How the Hermes installer reads the current ``memory.provider``
(plugin/synapse_client/harnesses/hermes/install.py).

``hermes config get memory.provider`` prints ``null`` both for a None value and for a
provider named "null"; with ``--json`` (what Hermes's own usage line documents) a None
is ``null`` and the name is ``"null"``. Hermes's core names are
agent/memory_provider.py's CORE_MEMORY_PROVIDER_SENTINELS, which "null" is not. The
outputs below are the real CLI's for those three values.

Without the ``hermes`` program, a config.yaml that already selects the provider is
reported as selected instead of naming the command that would select it.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_install as base
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import install as hi  # noqa: E402

_COMMAND = "hermes config set memory.provider synapse"


class _JsonHermes(base._Hermes):
    """``config get ... --json`` prints the value as JSON, as Hermes does."""

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        done = super().__call__(argv, **kwargs)
        if list(argv)[1:3] == ["config", "get"] and "--json" in argv and done.returncode == 0:
            done = subprocess.CompletedProcess(list(argv), 0, json.dumps(self.value) + "\n", "")
        return done


def _install(home: Path, hermes: Any, **kwargs: Any) -> str:
    return base._harness(home, hermes, "/bin/hermes").install(
        "http://unused", hermes_home=home, **kwargs
    )


def _without_program(home: Path) -> str:
    return base._harness(home).install("http://unused", hermes_home=home)


# ---------------------------------------------------------------------------
# With the program
# ---------------------------------------------------------------------------


def test_the_value_is_read_as_json(tmp_path):
    home = tmp_path / "hermes"
    base._with_config(home)
    hermes = _JsonHermes("synapse")
    assert _install(home, hermes).endswith("; already selected")
    (get,) = [argv for argv, _ in hermes.calls if argv[2] == "get"]
    assert get == ["/bin/hermes", "config", "get", "memory.provider", "--json"]


class _NoneHermes(_JsonHermes):
    """``memory.provider`` present but None: ``get --json`` prints ``null``, rc 0."""

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if list(argv)[1:3] == ["config", "get"]:
            self.calls.append((list(argv), kwargs))
            return subprocess.CompletedProcess(list(argv), 0, "null\n", "")
        return super().__call__(argv, **kwargs)


def test_a_none_value_is_no_provider(tmp_path):
    home = tmp_path / "hermes"
    base._with_config(home, "memory:\n  provider:\n")
    hermes = _NoneHermes("")
    assert _install(home, hermes).endswith("; selected")
    assert hermes.sets == [["/bin/hermes", "config", "set", "memory.provider", "synapse"]]


def test_a_provider_named_null_is_another_external_provider(tmp_path):
    home = tmp_path / "hermes"
    base._with_config(home)
    hermes = _JsonHermes("null")
    report = _install(home, hermes)
    assert "left memory.provider at 'null'" in report
    assert hermes.sets == []
    report = _install(home, hermes, replace=True)
    assert report.endswith("; selected (was 'null')")


@pytest.mark.parametrize("value", ["", "builtin", "Default", "none", "built-in"])
def test_hermes_core_names_are_replaced(tmp_path, value):
    home = tmp_path / "hermes"
    base._with_config(home)
    hermes = _JsonHermes(value)
    assert _install(home, hermes).endswith("; selected")


def test_a_plain_answer_is_still_read(tmp_path):
    """A program that prints the bare value (an older Hermes, a wrapper)."""
    home = tmp_path / "hermes"
    base._with_config(home)
    assert _install(home, base._Hermes("synapse")).endswith("; already selected")
    assert "left memory.provider at 'honcho'" in _install(home, base._Hermes("honcho"))


def test_an_unreadable_answer_falls_back_to_the_file(tmp_path):
    home = tmp_path / "hermes"
    base._with_config(home, "memory:\n  provider: synapse\n")
    assert _install(home, base._Hermes("", get_rc=2)).endswith("; already selected")
    base._with_config(home, "memory:\n  provider: honcho\n")
    report = _install(home, base._Hermes("", get_rc=2))
    assert report.endswith(f"could not read memory.provider; select it with `{_COMMAND}`")


# ---------------------------------------------------------------------------
# Without the program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "memory:\n  provider: synapse\n",
        "model: x\nmemory:\n  enabled: true\n  provider: synapse  # Synapse\n",
        "memory:\n    provider: 'synapse'\nterminal:\n    backend: local\n",
        '# comment\nmemory:\n\n  # which one\n  provider: "synapse"\n',
    ],
)
def test_a_config_that_already_selects_it_says_so(tmp_path, text):
    home = tmp_path / "hermes"
    config = base._with_config(home, text)
    assert _without_program(home).endswith("; already selected")
    assert config.read_text(encoding="utf-8") == text


@pytest.mark.parametrize(
    "text",
    [
        "memory:\n  provider: honcho\n",
        "memory:\n  provider:\n",
        "memory:\n  extra:\n    provider: synapse\n",
        "terminal:\n  provider: synapse\n",
        "memory: {provider: synapse}\n",
        "memory:\n  provider: synapse-old\n",
        "memory:\n  provider: synapse\nmemory:\n  provider: honcho\n",
    ],
)
def test_any_other_config_names_the_command(tmp_path, text):
    home = tmp_path / "hermes"
    base._with_config(home, text)
    assert _without_program(home).endswith(f"select it with `{_COMMAND}`")


def test_an_unreadable_config_names_the_command(tmp_path):
    home = tmp_path / "hermes"
    (home / "config.yaml").mkdir(parents=True)
    assert not hi.selected_in(home / "config.yaml")
    (home / "config.yaml").rmdir()
    base._with_config(home).write_bytes(b"memory:\n  provider: \xff\n")
    assert not hi.selected_in(home / "config.yaml")
