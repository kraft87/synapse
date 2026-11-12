"""A mistyped number in the environment does not break a hook.

The hooks read their timeouts, windows and caps from the environment at import, with
float() or int(); a value that did not parse raised there, and the hook died before it
did anything (the Stop hook shipped nothing, the spool flushed nothing). Each now falls
back to its default. git_feeder reads its window in main(), where the same mistake
raised past the hook's fail-open loop.

Every file a hook reads at import is under a temp dir (scratch_user).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers.client_core import REPO, drop_scripts, load_script, scratch_user


def _codex_stop_hook() -> ModuleType:
    sys.modules.pop("common", None)
    spec = importlib.util.spec_from_file_location(
        "codex_stop_hook_env_test", REPO / "plugin-codex" / "hooks" / "synapse_stop_hook.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script(name: str) -> Callable[[], ModuleType]:
    def load() -> ModuleType:
        sys.modules.pop(name, None)
        return load_script(name)

    load.__name__ = name
    return load


#: (loader, environment variable, module attribute, default)
CASES = [
    (_script("ingest_hook"), "SYNAPSE_INGEST_TIMEOUT", "TIMEOUT", 30.0),
    (_script("ingest_hook"), "SYNAPSE_INGEST_TAIL", "TAIL_RECORDS", 400),
    (_script("ingest_hook"), "SYNAPSE_INGEST_CATCHUP_DAYS", "CATCHUP_DAYS", 3.0),
    (_script("ingest_hook"), "SYNAPSE_INGEST_ACTIVE_GRACE", "ACTIVE_GRACE", 300.0),
    (_script("ingest_hook"), "SYNAPSE_INGEST_CATCHUP_MAX", "CATCHUP_MAX_FILES", 20),
    (_script("ingest_hook"), "SYNAPSE_PRIVATE_TTL_HOURS", "_PRIVATE_TTL_HOURS", 12.0),
    (_codex_stop_hook, "SYNAPSE_INGEST_TIMEOUT", "TIMEOUT", 30.0),
    (_codex_stop_hook, "SYNAPSE_CODEX_CATCHUP_DAYS", "CATCHUP_DAYS", 3.0),
    (_codex_stop_hook, "SYNAPSE_CODEX_CATCHUP_MAX", "CATCHUP_MAX_FILES", 20),
    (_script("remember_spool"), "SYNAPSE_SPOOL_TIMEOUT", "TIMEOUT", 10.0),
    (_script("remember_spool"), "SYNAPSE_SPOOL_PROBE_TIMEOUT", "PROBE_TIMEOUT", 4.0),
    (_script("remember_spool"), "SYNAPSE_SPOOL_BUDGET", "BUDGET", 12.0),
    (_script("remember_spool"), "SYNAPSE_SPOOL_MAX", "MAX_PER_FLUSH", 25),
]
_IDS = [f"{loader.__name__.lstrip('_')}:{var}" for loader, var, _, _ in CASES]


@pytest.fixture()
def scratch(monkeypatch, tmp_path) -> Iterator[Path]:
    home = scratch_user(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the hooks extend it; put it back
    saved = {name: sys.modules.get(name) for name in ("common", "remember_spool", "git_feeder")}
    yield home
    drop_scripts()
    for name, mod in saved.items():
        sys.modules.pop(name, None)
        if mod is not None:
            sys.modules[name] = mod


@pytest.mark.parametrize(("load", "var", "attr", "default"), CASES, ids=_IDS)
@pytest.mark.parametrize("garbage", ["soon", "", "1,5", "2.5x"])
def test_a_value_that_does_not_parse_falls_back_to_the_default(
    scratch, monkeypatch, load, var, attr, default, garbage
):
    monkeypatch.setenv(var, garbage)
    value = getattr(load(), attr)
    assert value == default and type(value) is type(default)


@pytest.mark.parametrize(("load", "var", "attr", "default"), CASES, ids=_IDS)
def test_a_value_that_parses_is_still_used(scratch, monkeypatch, load, var, attr, default):
    monkeypatch.setenv(var, "7")
    value = getattr(load(), attr)
    assert value == 7 and type(value) is type(default)


def test_an_int_setting_does_not_take_a_fraction(scratch, monkeypatch):
    """int() as before: "2.5" is not a count, so the default stands."""
    monkeypatch.setenv("SYNAPSE_INGEST_TAIL", "2.5")
    assert _script("ingest_hook")().TAIL_RECORDS == 400


def test_the_ingest_hook_ships_with_a_mistyped_timeout(scratch, monkeypatch, tmp_path):
    """The hook imports, and its POST goes out with the default timeout."""
    monkeypatch.setenv("SYNAPSE_INGEST_TIMEOUT", "thirty")
    hook = _script("ingest_hook")()
    seen: list[float] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status":"ok"}'

    def fake_urlopen(req, timeout):
        seen.append(timeout)
        return _Resp()

    monkeypatch.setattr(hook.config, "_urlopen", fake_urlopen)
    hook._post_records([])
    assert seen == [30.0]


def test_git_feeder_falls_back_to_30_days(scratch, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("SYNAPSE_TIMELINE_REPOS", str(repo))
    monkeypatch.setenv("SYNAPSE_TIMELINE_SINCE_DAYS", "a month")
    feeder = _script("git_feeder")()
    pushed: list[tuple[Path, int]] = []
    monkeypatch.setattr(feeder, "_push_repo", lambda r, days: pushed.append((r, days)))
    feeder.main()
    assert pushed == [(repo, 30)]
    monkeypatch.setenv("SYNAPSE_TIMELINE_SINCE_DAYS", "0")  # 0 still means no bound
    feeder.main()
    assert pushed[-1] == (repo, 0)
