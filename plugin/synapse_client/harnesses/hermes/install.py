"""Wire Synapse into Hermes: its plugin folder, and the config line that selects it.

``synapse install hermes`` builds ``<hermes home>/plugins/synapse/``:

    plugin.yaml       the manifest Hermes lists the provider by (plugin.yaml.tmpl)
    __init__.py       hands Hermes a SynapseMemoryProvider (plugin_init.py.tmpl)
    synapse_client/   this client's core, with harnesses/hermes/ and no other harness

The new folder is assembled beside the old one under a dot-name (Hermes skips those)
and swapped in with two renames, so Hermes never loads half of it. The package inside
is imported relatively, under whatever name Hermes gives the folder, so it cannot meet
another copy of synapse_client on the same interpreter.

Then the provider is selected, ``memory.provider: synapse`` in the home's config.yaml:

- with the ``hermes`` program on PATH and a config.yaml already there, through
  ``hermes config get`` / ``hermes config set``. An empty value or one of Hermes's
  built-in names is replaced; another external provider is left alone unless
  ``replace`` is given, since Hermes runs one at a time. The program is not run for a
  home without config.yaml: a first launch sets up that home, a large download.
- otherwise a home with no config.yaml gets one of two lines; one that has a config.yaml
  is left as it is and the report names the command that selects the provider.

The Hermes home is the ``hermes_home`` option, else ``$HERMES_HOME``, else Hermes's
platform default (hermes_constants.get_hermes_home). The URL is not stored here: the
provider reads it from ~/.synapse/config.json, as every other harness does.

Importable without Hermes: nothing here imports ``provider``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ... import __version__

PLUGIN = "synapse"
#: What Hermes treats as "no external provider" (agent/memory_provider.py's
#: CORE_MEMORY_PROVIDER_SENTINELS), and how `hermes config get` prints a None.
_BUILT_IN = frozenset({"", "default", "builtin", "built-in", "none", "null"})
_CONFIG_LINES = "memory:\n  provider: synapse\n"
_TIMEOUT = 60

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _here() -> Path:
    return Path(__file__).resolve().parent


def _core() -> Path:
    return _here().parents[1]


def default_home(env: Mapping[str, str] | None = None) -> Path:
    """``$HERMES_HOME`` (expanded), else Hermes's platform default."""
    environ = os.environ if env is None else env
    value = (environ.get("HERMES_HOME") or "").strip()
    if value:
        return Path(os.path.expanduser(os.path.expandvars(value)))
    suffix = environ.get("HERMES_DATA_DIR_SUFFIX", "")
    if sys.platform == "win32":
        local = (environ.get("LOCALAPPDATA") or "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / ("hermes" + suffix)
    return Path.home() / (".hermes" + suffix)


def _render(template: str) -> str:
    text = (_here() / template).read_text(encoding="utf-8")
    return string.Template(text).safe_substitute(version=__version__)


def _ignore(directory: str, names: list[str]) -> set[str]:
    """What the copy of the core leaves out: caches, and every harness but Hermes."""
    skip = {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}
    if Path(directory).resolve() == _core() / "harnesses":
        skip |= {name for name in names if name not in ("__init__.py", "hermes")}
    return skip


def assemble(target: Path) -> None:
    """Build the plugin folder at ``target`` (which must not exist)."""
    target.mkdir()
    (target / "plugin.yaml").write_text(_render("plugin.yaml.tmpl"), encoding="utf-8")
    (target / "__init__.py").write_text(_render("plugin_init.py.tmpl"), encoding="utf-8")
    shutil.copytree(_core(), target / "synapse_client", ignore=_ignore)


def swap_in(plugins: Path) -> Path:
    """Assemble a fresh ``plugins/synapse`` and swap it for the old one. Raises OSError,
    with the old folder back in place, when either step fails."""
    plugins.mkdir(parents=True, exist_ok=True)
    final = plugins / PLUGIN
    stage = Path(tempfile.mkdtemp(prefix=f".{PLUGIN}.new-", dir=plugins))
    old: Path | None = None
    try:
        built = stage / PLUGIN
        assemble(built)
        if final.exists():
            old = Path(tempfile.mkdtemp(prefix=f".{PLUGIN}.old-", dir=plugins)) / PLUGIN
            final.rename(old)
        try:
            built.rename(final)
        except OSError:
            if old is not None:
                old.rename(final)
                old = None
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if old is not None:
            shutil.rmtree(old.parent, ignore_errors=True)
    return final


def installed_version(home: Path) -> str:
    """The client version a home's plugin folder was built from; "" when unknown."""
    try:
        text = (home / "plugins" / PLUGIN / "plugin.yaml").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "version":
            return value.strip().strip("\"'")
    return ""


class HermesHarness:
    """The Hermes harness. ``runner`` (subprocess.run's shape), ``which`` and ``env``
    are seams for tests."""

    name = "hermes"
    label = "Hermes"

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        which: Callable[[str], str | None] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._runner: Runner = runner or subprocess.run
        self._which = which or shutil.which
        self._env = env

    def _home(self, explicit: Any = None) -> Path:
        if explicit:
            return Path(os.path.expanduser(str(explicit)))
        return default_home(self._env)

    def _program(self) -> str | None:
        try:
            return self._which("hermes")
        except Exception:
            return None

    # -- the Harness protocol ---------------------------------------------------------

    def detect(self) -> bool:
        """``hermes`` on PATH, or a Hermes home on disk."""
        try:
            return bool(self._program()) or self._home().is_dir()
        except Exception:
            return False

    def wired(self) -> bool:
        """The plugin folder is there and carries this client's Hermes provider."""
        try:
            folder = self._home() / "plugins" / PLUGIN
            return (folder / "plugin.yaml").is_file() and (
                folder / "synapse_client" / "harnesses" / "hermes" / "provider.py"
            ).is_file()
        except Exception:
            return False

    def install(self, url: str, *, dry_run: bool = False, **options: Any) -> str:
        """Put the plugin folder in place and select the provider. ``options``:
        ``hermes_home`` (a path) and ``replace`` (take over from another external
        provider). ``url`` is not stored: the provider reads Synapse's own config.

        Raises RuntimeError when Hermes is not on this machine and no home was named,
        or when the folder could not be written."""
        explicit = options.get("hermes_home")
        replace = bool(options.get("replace"))
        if not explicit and not self.detect():
            raise RuntimeError(
                "Hermes was not found on this machine (no `hermes` program and no "
                f"{self._home()}); install it, or name its home with --hermes-home"
            )
        home = self._home(explicit)
        if dry_run:
            folder = f"would install the memory provider in {home / 'plugins' / PLUGIN}"
            return f"Hermes: {folder}; {self._select(home, dry_run=True, replace=replace)}"
        try:
            swap_in(home / "plugins")
        except OSError as err:
            raise RuntimeError(f"could not write {home / 'plugins' / PLUGIN}: {err}") from err
        selected = self._select(home, dry_run=False, replace=replace)
        return f"Hermes: memory provider {__version__} installed in {home}; {selected}"

    def update(self) -> str | None:
        """Rebuild the plugin folder when it was built by another client version."""
        if not self.wired():
            return None
        home = self._home()
        if installed_version(home) == __version__:
            return f"Hermes: memory provider already at {__version__}"
        try:
            swap_in(home / "plugins")
        except OSError as err:
            return f"Hermes: not updated: {err}"
        return f"Hermes: memory provider updated to {__version__}"

    # -- selecting the provider -------------------------------------------------------

    def _command(self, home: Path) -> str:
        prefix = "" if home == default_home(self._env) else f"HERMES_HOME={shlex.quote(str(home))} "
        return f"{prefix}hermes config set memory.provider {PLUGIN}"

    def _select(self, home: Path, *, dry_run: bool, replace: bool) -> str:
        config = home / "config.yaml"
        program = self._program()
        if program and config.is_file():
            return self._select_with(program, home, dry_run=dry_run, replace=replace)
        if config.exists():
            return f"select it with `{self._command(home)}`"
        if dry_run:
            return f"would write {config} selecting it"
        try:
            config.write_text(_CONFIG_LINES, encoding="utf-8")
        except OSError as err:
            return f"could not write {config} ({err}); select it with `{self._command(home)}`"
        return f"selected in {config}"

    def _run(
        self, program: str, home: Path, args: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "HERMES_HOME": str(home)}
        return self._runner(
            [program, "config", *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=env,
            check=False,
        )

    def _current(self, program: str, home: Path) -> str | None:
        """``memory.provider`` as Hermes reads it ("" when unset), or None when the
        program could not tell."""
        try:
            done = self._run(program, home, ("get", "memory.provider"))
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode == 0:
            lines = (done.stdout or "").splitlines()
            return lines[0].strip() if lines else ""
        if "not set" in (done.stderr or "") + (done.stdout or ""):
            return ""
        return None

    def _select_with(self, program: str, home: Path, *, dry_run: bool, replace: bool) -> str:
        current = self._current(program, home)
        command = self._command(home)
        if current is None:
            return f"could not read memory.provider; select it with `{command}`"
        if current == PLUGIN:
            return "already selected"
        if current.lower() not in _BUILT_IN and not replace:
            return (
                f"left memory.provider at {current!r}, Hermes runs one external provider; "
                f"run `synapse install hermes --replace` or `{command}` to switch"
            )
        was = f" (was {current!r})" if current.lower() not in _BUILT_IN else ""
        if dry_run:
            return f"would select it{was}"
        try:
            done = self._run(program, home, ("set", "memory.provider", PLUGIN))
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is None or done.returncode != 0:
            return f"could not select it; run `{command}`"
        return f"selected{was}"
