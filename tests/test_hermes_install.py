"""The Hermes installer (plugin/synapse_client/harnesses/hermes/install.py).

Every Hermes home here is a temp dir and the ``hermes`` program is a fake runner: the
real one sets a new home up on first launch, a large download.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import PLUGIN, use_core

use_core()
from synapse_client import __version__, harnesses  # noqa: E402
from synapse_client.harnesses.hermes import install as hi  # noqa: E402

_CORE = PLUGIN / "synapse_client"


class _Hermes:
    """A fake ``hermes config get/set``: answers ``get`` from ``value`` (None: the key is
    not set) and records every call."""

    def __init__(self, value: str | None = "", *, get_rc: int = 0, set_rc: int = 0) -> None:
        self.value = value
        self.get_rc = get_rc
        self.set_rc = set_rc
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append((argv, kwargs))
        if argv[1:3] == ["config", "get"]:
            if self.get_rc:
                return subprocess.CompletedProcess(argv, self.get_rc, "", "usage: hermes ...")
            if self.value is None:
                return subprocess.CompletedProcess(argv, 1, "", "Config key not set: x\n")
            return subprocess.CompletedProcess(argv, 0, f"{self.value}\n", "")
        if argv[1:3] == ["config", "set"]:
            if not self.set_rc:
                self.value = argv[4]
            return subprocess.CompletedProcess(argv, self.set_rc, "", "")
        raise AssertionError(argv)

    @property
    def sets(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls if argv[2] == "set"]


def _harness(home: Path, runner: Any = None, program: str | None = None) -> hi.HermesHarness:
    return hi.HermesHarness(
        runner=runner or _Hermes(),
        which=lambda name: program,
        env={"HERMES_HOME": str(home)},
    )


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# The folder
# ---------------------------------------------------------------------------


def test_the_folder_holds_the_manifest_the_entry_module_and_the_core(tmp_path):
    home = tmp_path / "hermes"
    _harness(home).install("http://unused", hermes_home=home)
    folder = home / "plugins" / "synapse"

    expected = {"plugin.yaml", "__init__.py"} | {
        f"synapse_client/{rel}"
        for rel in _files(_CORE)
        if "__pycache__" not in rel and not rel.endswith((".pyc", ".pyo"))
        if not rel.startswith("harnesses/")
        or rel == "harnesses/__init__.py"
        or rel.startswith("harnesses/hermes/")
    }
    assert _files(folder) == expected
    for name in ("__init__", "messages", "provider", "install"):
        assert f"synapse_client/harnesses/hermes/{name}.py" in expected
    assert "synapse_client/harnesses/hermes/plugin.yaml.tmpl" in expected
    assert "synapse_client/harnesses/hermes/plugin_init.py.tmpl" in expected
    assert "synapse_client/tool_schemas.json" in expected
    assert not list(folder.rglob("__pycache__"))
    # Hermes imports the folder's top-level modules: there is exactly one.
    assert [p.name for p in folder.glob("*.py")] == ["__init__.py"]
    assert {p.name for p in (home / "plugins").iterdir()} == {"synapse"}


def test_no_other_harness_is_copied():
    names = ["__init__.py", "hermes", "codex", "__pycache__", "x.pyc"]
    assert hi._ignore(str(_CORE / "harnesses"), names) == {"codex", "__pycache__", "x.pyc"}
    assert hi._ignore(str(_CORE), ["api.py", "__pycache__"]) == {"__pycache__"}


def test_the_entry_module_is_what_hermes_discovers_and_loads(tmp_path):
    home = tmp_path / "hermes"
    _harness(home).install("http://unused", hermes_home=home)
    folder = home / "plugins" / "synapse"
    init = (folder / "__init__.py").read_text(encoding="utf-8")
    # plugins/memory/__init__.py: the marker must be in the first 8 KB.
    assert "register_memory_provider" in init[:8192]
    assert "from .synapse_client.harnesses.hermes.provider import SynapseMemoryProvider" in init
    assert "$version" not in init and __version__ in init
    compile(init, "__init__.py", "exec")
    manifest = (folder / "plugin.yaml").read_text(encoding="utf-8")
    assert manifest.splitlines()[0] == "name: synapse"
    assert "kind: exclusive" in manifest
    assert hi.installed_version(home) == __version__


def test_a_reinstall_replaces_the_folder_whole(tmp_path):
    home = tmp_path / "hermes"
    harness = _harness(home)
    harness.install("http://unused", hermes_home=home)
    stray = home / "plugins" / "synapse" / "synapse_client" / "stale.py"
    stray.write_text("x = 1\n", encoding="utf-8")
    harness.install("http://unused", hermes_home=home)
    assert not stray.exists()
    assert {p.name for p in (home / "plugins").iterdir()} == {"synapse"}


def test_a_failed_build_leaves_the_old_folder(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    harness = _harness(home)
    harness.install("http://unused", hermes_home=home)
    marker = home / "plugins" / "synapse" / "plugin.yaml"
    before = marker.read_text(encoding="utf-8")

    def broken(target: Path) -> None:
        target.mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(hi, "assemble", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        harness.install("http://unused", hermes_home=home)
    assert marker.read_text(encoding="utf-8") == before
    assert {p.name for p in (home / "plugins").iterdir()} == {"synapse"}


def test_wired_and_update(tmp_path):
    home = tmp_path / "hermes"
    harness = _harness(home)
    assert not harness.wired()
    assert harness.update() is None

    harness.install("http://unused", hermes_home=home)
    assert harness.wired()
    assert harness.update() == f"Hermes: memory provider already at {__version__}"

    manifest = home / "plugins" / "synapse" / "plugin.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(__version__, "0.0.1"), encoding="utf-8"
    )
    assert hi.installed_version(home) == "0.0.1"
    assert harness.update() == f"Hermes: memory provider updated to {__version__}"
    assert hi.installed_version(home) == __version__

    (
        home / "plugins" / "synapse" / "synapse_client" / "harnesses" / "hermes" / "provider.py"
    ).unlink()
    assert not harness.wired()


def test_detect(tmp_path):
    home = tmp_path / "hermes"
    assert not _harness(home).detect()
    assert _harness(home, program="/usr/bin/hermes").detect()
    home.mkdir()
    assert _harness(home).detect()


def test_install_without_hermes_and_without_a_home_raises(tmp_path):
    home = tmp_path / "hermes"
    with pytest.raises(RuntimeError, match="Hermes was not found"):
        _harness(home).install("http://unused")
    assert not home.exists()


def test_the_default_home_follows_hermes(tmp_path, monkeypatch):
    assert hi.default_home({"HERMES_HOME": str(tmp_path / "h")}) == tmp_path / "h"
    monkeypatch.setenv("HOME", str(tmp_path))
    if hi.sys.platform != "win32":
        assert hi.default_home({}) == tmp_path / ".hermes"
        assert hi.default_home({"HERMES_DATA_DIR_SUFFIX": "-dev"}) == tmp_path / ".hermes-dev"


def test_the_registry_offers_hermes():
    (hermes,) = harnesses.registry()
    assert (hermes.name, hermes.label) == ("hermes", "Hermes")
    assert isinstance(hermes, hi.HermesHarness)


# ---------------------------------------------------------------------------
# Selecting the provider
# ---------------------------------------------------------------------------


def _with_config(home: Path, text: str = "model: x\n") -> Path:
    home.mkdir(parents=True, exist_ok=True)
    config = home / "config.yaml"
    config.write_text(text, encoding="utf-8")
    return config


@pytest.mark.parametrize("value", [None, "", "null", "builtin", "Default", "none"])
def test_an_empty_or_built_in_provider_is_replaced(tmp_path, value):
    home = tmp_path / "hermes"
    _with_config(home)
    hermes = _Hermes(value)
    report = _harness(home, hermes, "/bin/hermes").install("http://unused", hermes_home=home)
    assert report.endswith("; selected")
    assert hermes.sets == [["/bin/hermes", "config", "set", "memory.provider", "synapse"]]
    for _, kwargs in hermes.calls:
        assert kwargs["env"]["HERMES_HOME"] == str(home)
        assert kwargs["timeout"] and kwargs["capture_output"]


def test_an_existing_selection_is_left(tmp_path):
    home = tmp_path / "hermes"
    _with_config(home)
    hermes = _Hermes("synapse")
    report = _harness(home, hermes, "/bin/hermes").install("http://unused", hermes_home=home)
    assert report.endswith("; already selected")
    assert hermes.sets == []


def test_another_external_provider_stays_unless_replace(tmp_path):
    home = tmp_path / "hermes"
    _with_config(home)
    hermes = _Hermes("honcho")
    harness = _harness(home, hermes, "/bin/hermes")
    report = harness.install("http://unused", hermes_home=home)
    assert "left memory.provider at 'honcho'" in report
    assert "synapse install hermes --replace" in report
    assert "hermes config set memory.provider synapse" in report
    assert hermes.sets == []
    assert hi.installed_version(home) == __version__  # the folder is in place either way

    report = harness.install("http://unused", hermes_home=home, replace=True)
    assert report.endswith("; selected (was 'honcho')")
    assert len(hermes.sets) == 1


def test_an_unreadable_or_failed_selection_names_the_command(tmp_path):
    home = tmp_path / "hermes"
    _with_config(home)
    hermes = _Hermes(get_rc=2)
    report = _harness(home, hermes, "/bin/hermes").install("http://unused", hermes_home=home)
    assert "could not read memory.provider" in report
    assert hermes.sets == []

    hermes = _Hermes("", set_rc=1)
    report = _harness(home, hermes, "/bin/hermes").install("http://unused", hermes_home=home)
    assert "could not select it; run `" in report


def test_the_program_is_not_run_for_a_home_without_config(tmp_path):
    home = tmp_path / "hermes"
    hermes = _Hermes("")
    report = _harness(home, hermes, "/bin/hermes").install("http://unused", hermes_home=home)
    assert hermes.calls == []
    assert (home / "config.yaml").read_text(encoding="utf-8") == "memory:\n  provider: synapse\n"
    assert report.endswith(f"selected in {home / 'config.yaml'}")


def test_without_the_program_a_new_config_is_written(tmp_path):
    home = tmp_path / "hermes"
    report = _harness(home).install("http://unused", hermes_home=home)
    assert (home / "config.yaml").read_text(encoding="utf-8") == "memory:\n  provider: synapse\n"
    assert "selected in" in report


def test_without_the_program_an_existing_config_is_left(tmp_path):
    home = tmp_path / "hermes"
    config = _with_config(home, "model: x\nmemory:\n  provider: honcho\n")
    elsewhere = hi.HermesHarness(runner=_Hermes(), which=lambda name: None, env={})
    report = elsewhere.install("http://unused", hermes_home=home)
    assert config.read_text(encoding="utf-8") == "model: x\nmemory:\n  provider: honcho\n"
    assert report.endswith(
        f"select it with `HERMES_HOME={home} hermes config set memory.provider synapse`"
    )
    # In the default home the command needs no prefix.
    report = _harness(home).install("http://unused", hermes_home=home)
    assert report.endswith("select it with `hermes config set memory.provider synapse`")


def test_dry_run_changes_nothing(tmp_path):
    home = tmp_path / "hermes"
    report = _harness(home).install("http://unused", dry_run=True, hermes_home=home)
    assert report.startswith("Hermes: would install the memory provider in ")
    assert "would write" in report
    assert not home.exists()

    _with_config(home)
    hermes = _Hermes("")
    report = _harness(home, hermes, "/bin/hermes").install(
        "http://unused", dry_run=True, hermes_home=home
    )
    assert report.endswith("would select it")
    assert hermes.sets == []
    assert not (home / "plugins").exists()
