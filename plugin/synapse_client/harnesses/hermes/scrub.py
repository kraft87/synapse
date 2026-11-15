"""Hermes's own secret scrub, for rows read from its session store.

Hermes scrubs every string it hands a memory provider (agent/memory_manager.py
``_redact_messages_for_egress``, built on agent/redact.py ``redact_for_egress``): API
keys, tokens and the like, and then every other place the same token appears. Its
session store holds the rows as they were, unscrubbed. The live provider sends what
Hermes handed it, so it never needs this. ``synapse import hermes`` reads the store, so
it asks Hermes to scrub what it read, with Hermes's own code, run by Hermes's own
interpreter: the client runs on another Python and imports nothing of Hermes.

``interpreter`` finds that interpreter beside the ``hermes`` program (a console script
of Hermes's virtual environment: ``bin/python`` on POSIX, ``Scripts\\python.exe`` on
Windows, or the script's ``#!`` line). ``rows`` runs it once for every session read, so
a token Hermes masks in one row is masked in all. It answers None when Hermes is not
found, its scrub cannot be imported, or anything else fails: the caller decides what
that means (the import stops, unless told to send the stored text as it is).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404: runs Hermes's own interpreter, with no shell
from pathlib import Path
from typing import Any

#: Seconds Hermes's interpreter gets to import its scrub and run it.
TIMEOUT = 300.0

_CODE = r"""
import json, sys
data = json.load(sys.stdin)
try:
    from agent.memory_manager import _redact_messages_for_egress as scrub
except Exception:
    try:
        from agent.memory_manager import _redact_for_provider
    except Exception:
        sys.exit(3)
    scrub = lambda rows: list(_redact_for_provider(rows))
flat = [row for rows in data.values() for row in rows]
done = scrub(flat)
if not isinstance(done, list) or len(done) != len(flat):
    sys.exit(4)
out, i = {}, 0
for sid, rows in data.items():
    out[sid] = done[i : i + len(rows)]
    i += len(rows)
json.dump(out, sys.stdout)
"""


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


def rows(
    sessions: dict[str, list[dict[str, Any]]], python: str | None, home: Any = None
) -> dict[str, list[dict[str, Any]]] | None:
    """``sessions`` (rows by session id) as Hermes scrubs them for a provider, or None
    when that could not be done. Never raises."""
    if not python:
        return None
    env = dict(os.environ)
    if home:
        env["HERMES_HOME"] = str(home)
    try:
        done = subprocess.run(  # nosec B603: a fixed program and a fixed script, no shell
            [python, "-c", _CODE],
            input=json.dumps(sessions, default=str),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0:
        return None
    try:
        out = json.loads(done.stdout)
    except ValueError:
        return None
    if not isinstance(out, dict) or set(out) != set(sessions):
        return None
    if any(len(out[sid]) != len(sessions[sid]) for sid in sessions):
        return None
    return out
