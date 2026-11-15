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
  is left as it is and the report names the command that selects the provider, unless
  the file already selects it (``selected_in``).

The Hermes home is the ``hermes_home`` option, else ``$HERMES_HOME``, else Hermes's
platform default (hermes_constants.get_hermes_home). The URL is not stored here: the
provider reads it from ~/.synapse/config.json, as every other harness does.

Importable without Hermes: nothing here imports ``provider``.
"""

from __future__ import annotations

import json
import os
import re
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
#: CORE_MEMORY_PROVIDER_SENTINELS). A None value reads as "" (``config get --json``
#: prints null); the string "null" is a provider's name to Hermes.
_BUILT_IN = frozenset({"", "default", "builtin", "built-in", "none"})
_CONFIG_LINES = "memory:\n  provider: synapse\n"
_SELECTED = "already selected"
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


def _windows(windows: bool | None) -> bool:
    return sys.platform == "win32" if windows is None else windows


def _quoted(home: Path, windows: bool | None) -> str:
    """``home`` as one word for the user's shell: PowerShell on Windows, else POSIX."""
    if _windows(windows):
        return "'" + str(home).replace("'", "''") + "'"
    return shlex.quote(str(home))


def home_prefix(home: Path, windows: bool | None = None) -> str:
    """What sets ``HERMES_HOME`` for the one command after it: a POSIX shell's
    ``VAR=value``, or PowerShell's ``$env:`` on Windows."""
    if _windows(windows):
        return f"$env:HERMES_HOME = {_quoted(home, True)}; "
    return f"HERMES_HOME={_quoted(home, False)} "


def home_option(home: Path, windows: bool | None = None) -> str:
    """`` --hermes-home <home>`` for a ``synapse install hermes`` command, quoted for the
    user's shell."""
    return f" --hermes-home {_quoted(home, windows)}"


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


def provider_in(config: Path) -> str | None:
    """``memory.provider`` as config.yaml names it, read without a YAML parser: only the
    block layout Hermes and this installer write (``memory:``, then ``provider: <name>``
    one level in) is read. "" when the file sets none; None when it cannot be read or
    sets ``memory`` in another layout."""
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    found, inside, indent = "", False, ""
    for line in lines:
        body = line.lstrip()
        if not body or body.startswith("#"):
            continue
        if body == line:  # a top-level key
            key = line.split("#", 1)[0].rstrip()
            if key.startswith("memory:") and key != "memory:":
                return None  # memory: {provider: ...}, or another layout
            inside, indent = key == "memory:", ""
            continue
        here = line[: len(line) - len(body)]
        indent = indent or here
        match = re.fullmatch(r"provider:\s*([^#]*?)\s*(#.*)?", body)
        if inside and here == indent and match:
            found = match.group(1).strip("'\"")
    return found


def selected_in(config: Path) -> bool:
    """config.yaml names this provider as ``memory.provider`` (``provider_in``). False
    for any other layout or an unreadable file, so it can only spare the user a command,
    never select anything."""
    return provider_in(config) == PLUGIN


def _fallback_note(home: Path) -> str:
    """What ``synapse status`` adds when the last upload read Hermes's live list, not its
    session store (provider.py ``_fell_back``). Never raises."""
    try:
        data = json.loads((home / "synapse" / "store_fallback.json").read_text("utf-8"))
    except (OSError, ValueError):
        return ""
    reason = data.get("reason") if isinstance(data, dict) else None
    if not isinstance(reason, str) or not reason:
        return ""
    return f"; uploads read the live list, not state.db ({reason})"


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
        """The plugin folder is there and carries this client's Hermes provider. Whether
        Hermes runs it is ``memory.provider`` (``status``)."""
        try:
            folder = self._home() / "plugins" / PLUGIN
            return (folder / "plugin.yaml").is_file() and (
                folder / "synapse_client" / "harnesses" / "hermes" / "provider.py"
            ).is_file()
        except Exception:
            return False

    def status(self) -> str:
        """What ``synapse status`` says of Hermes: not installed, installed and selected
        ("wired"), or installed while Hermes runs another provider or none, with the
        command that selects this one. Never raises."""
        try:
            if not self.wired():
                return "found, not wired" if self.detect() else "not found"
            selected = self._selected(self._home())
        except Exception:
            return "installed; whether Hermes runs it could not be read"
        if selected != _SELECTED:
            return f"installed, {selected}"
        return "wired" + _fallback_note(self._home())

    def _selected(self, home: Path) -> str:
        """Whether ``home`` selects this provider: _SELECTED, else what it selects and
        the command that switches, as ``install`` words it. config.yaml is read first;
        the program is asked only when the file does not say (it takes seconds to
        start)."""
        config = home / "config.yaml"
        program = self._program()
        current = provider_in(config) if config.is_file() else ""
        if current is None and program:
            current = self._current(program, home)
        if current == PLUGIN:
            return _SELECTED
        command = self._command(home)
        if current is None:
            return f"not selected as far as can be read; select it with `{command}`"
        if current.lower() in _BUILT_IN:
            return f"not selected; run `{self._install_command(home)}`"
        if not program:
            return f"not selected: memory.provider is {current!r}; select it with `{command}`"
        return f"not selected: memory.provider is {current!r}, {self._switch(home)}"

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
            return f"Hermes: memory provider already at {__version__}{self._unselected(home)}"
        try:
            swap_in(home / "plugins")
        except OSError as err:
            return f"Hermes: not updated: {err}"
        return f"Hermes: memory provider updated to {__version__}{self._unselected(home)}"

    def _unselected(self, home: Path) -> str:
        """ "; not selected..." when Hermes does not run this provider, else ""."""
        selected = self._selected(home)
        return "" if selected == _SELECTED else f"; {selected}"

    # -- selecting the provider -------------------------------------------------------

    def _command(self, home: Path) -> str:
        prefix = "" if home == default_home(self._env) else home_prefix(home)
        return f"{prefix}hermes config set memory.provider {PLUGIN}"

    def _install_command(self, home: Path, replace: bool = False) -> str:
        """``synapse install hermes`` for ``home``: the home is named unless it is the
        default one, which the command finds by itself."""
        where = "" if home == default_home(self._env) else home_option(home)
        return f"synapse install hermes{' --replace' if replace else ''}{where}"

    def _switch(self, home: Path) -> str:
        return (
            "Hermes runs one external provider; "
            f"run `{self._install_command(home, replace=True)}` or `{self._command(home)}` "
            "to switch"
        )

    def _select(self, home: Path, *, dry_run: bool, replace: bool) -> str:
        config = home / "config.yaml"
        program = self._program()
        if program and config.is_file():
            return self._select_with(program, home, dry_run=dry_run, replace=replace)
        if config.exists():
            if selected_in(config):
                return _SELECTED
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
        """``memory.provider`` as Hermes reads it ("" when unset or None), or None when
        the program could not tell. Read as JSON, since the plain form prints a None and
        the string "null" alike."""
        try:
            done = self._run(program, home, ("get", "memory.provider", "--json"))
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode == 0:
            lines = (done.stdout or "").splitlines()
            line = lines[0].strip() if lines else ""
            try:
                value = json.loads(line)
            except ValueError:
                return line
            if value is None:
                return ""
            return value.strip() if isinstance(value, str) else None
        if "not set" in (done.stderr or "") + (done.stdout or ""):
            return ""
        return None

    def _select_with(self, program: str, home: Path, *, dry_run: bool, replace: bool) -> str:
        current = self._current(program, home)
        command = self._command(home)
        if current is None:
            if selected_in(home / "config.yaml"):
                return _SELECTED
            return f"could not read memory.provider; select it with `{command}`"
        if current == PLUGIN:
            return _SELECTED
        if current.lower() not in _BUILT_IN and not replace:
            return f"left memory.provider at {current!r}, {self._switch(home)}"
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
