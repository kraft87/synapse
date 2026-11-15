"""Hermes's own secret scrub, for rows read from its session store.

Hermes scrubs every string it hands a memory provider (agent/memory_manager.py
``_redact_messages_for_egress``, built on agent/redact.py ``redact_for_egress``): API
keys, tokens and the like, and then every other place the same token appears. Its
session store holds the rows as they were, unscrubbed. The live provider sends what
Hermes handed it, so it never needs this. ``synapse import hermes`` reads the store, so
it asks Hermes to scrub what it read, with Hermes's own code, run by Hermes's own
interpreter: the client runs on another Python and imports nothing of Hermes. This
module is the whole of that: ``rows`` is the one call.

``interpreter`` finds that interpreter beside the ``hermes`` program (a console script
of Hermes's virtual environment: ``bin/python`` on POSIX, ``Scripts\\python.exe`` on
Windows, or the script's ``#!`` line).

The child must run Hermes's code and nothing else: it runs isolated (``-I``: no
``PYTHON*`` variables, no user site, neither the script's nor the working directory on
``sys.path``), with an environment that has no ``PYTHONPATH``, in a new empty
directory. It names the file it took the scrub from, and the answer is used only when
that file belongs to the interpreter's own install: under its prefix, or under a
source tree that install's site-packages points to (an editable install's ``.pth`` or
finder). Anything else, a failure or a timeout, answers None: the caller decides what
that means (the import stops, unless told to send the stored text as it is).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404: runs Hermes's own interpreter, with no shell
import tempfile
from pathlib import Path
from typing import Any

#: Seconds Hermes's interpreter gets to import its scrub and run it.
TIMEOUT = 300.0
#: Environment variables the child does not get: they could put other code first.
_DROPPED = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONSAFEPATH")

_CODE = r"""
import json, site, sys
data = json.load(sys.stdin)
try:
    import agent.memory_manager as mm
    scrub = mm._redact_messages_for_egress
except Exception:
    sys.exit(3)
flat = [row for rows in data.values() for row in rows]
done = scrub(flat)
if not isinstance(done, list) or len(done) != len(flat):
    sys.exit(4)
out, i = {}, 0
for sid, rows in data.items():
    out[sid] = done[i : i + len(rows)]
    i += len(rows)
json.dump({"rows": out, "file": mm.__file__, "prefix": sys.prefix,
           "isolated": sys.flags.isolated, "sites": site.getsitepackages()}, sys.stdout)
"""

_PATH = re.compile(r"""["']?(/[^"'\s,)]+|[A-Za-z]:\\[^"'\s,)]+)["']?""")


def interpreter(which: Any = shutil.which) -> str | None:
    """The Python that runs Hermes, found from the ``hermes`` program, or None."""
    program = which("hermes")
    if not program:
        return None
    path = Path(program).resolve()
    for name in ("python", "python3", "python.exe"):
        beside = path.parent / name
        if beside.is_file():
            return str(beside)
    try:
        with open(path, "rb") as f:
            first = f.readline(512)
    except OSError:
        return None
    if first.startswith(b"#!"):
        line = first[2:].decode("utf-8", "replace").strip().split()
        if line and Path(line[0]).name.startswith("python") and Path(line[0]).is_file():
            return line[0]
    return None


def _roots(prefix: str, sites: list[str]) -> list[Path]:
    """The interpreter's prefix, and every directory its site-packages names in a
    ``.pth`` file or an editable install's finder."""
    roots = [Path(prefix)]
    for site_dir in sites:
        directory = Path(site_dir)
        try:
            names = [p for p in directory.iterdir() if p.suffix == ".pth" or "finder" in p.name]
        except OSError:
            continue
        for file in names:
            try:
                text = file.read_text("utf-8", "replace")
            except OSError:
                continue
            for found in _PATH.findall(text):
                candidate = Path(found)
                if candidate.is_dir():
                    roots.append(candidate)
    return roots


def _within(file: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            file.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def rows(
    sessions: dict[str, list[dict[str, Any]]], python: str | None, home: Any = None
) -> dict[str, list[dict[str, Any]]] | None:
    """``sessions`` (rows by session id) as Hermes scrubs them for a provider, or None
    when that could not be done, or not provably by Hermes. Never raises."""
    if not python:
        return None
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED}
    if home:
        env["HERMES_HOME"] = str(home)
    try:
        with tempfile.TemporaryDirectory(prefix="synapse-scrub-") as empty:
            done = subprocess.run(  # nosec B603: a fixed program and a fixed script, no shell
                [python, "-I", "-c", _CODE],
                input=json.dumps(sessions, default=str),
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                env=env,
                cwd=empty,
                check=False,
            )
            if done.returncode != 0:
                return None
            answer = json.loads(done.stdout)
            if not isinstance(answer, dict) or answer.get("isolated") != 1:
                return None
            file = Path(str(answer.get("file") or "")).resolve()
            if not file.is_file() or _within(file, [Path(empty)]):
                return None
            if not _within(
                file, _roots(str(answer.get("prefix")), list(answer.get("sites") or []))
            ):
                return None
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None
    out = answer.get("rows")
    if not isinstance(out, dict) or set(out) != set(sessions):
        return None
    if any(len(out[sid]) != len(sessions[sid]) for sid in sessions):
        return None
    return out
