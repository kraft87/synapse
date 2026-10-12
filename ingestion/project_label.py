"""Server-side project labels: the fallback for transcripts that arrive without one.

Clients resolve the label themselves (``plugin/scripts/synapse_project.py``: a
``.synapse-project`` marker, else the git toplevel's name, else the folder's name) and
send it with every /ingest body. A body without one — a catch-up sweep of a transcript
whose folder is gone, an older plugin, a backfill — is labelled here from the
transcript's ``cwd``, with the same normalisation the client applies to whatever it
found: either path separator, trailing separators dropped, the last component kept
with its case. Before this, the parsers split on "/" only, so every Windows session
was labelled with its whole path.

The client copy and this one must agree on :func:`label_from_path` for any string;
``tests/test_project_label.py`` checks them against each other.
"""

from __future__ import annotations

import re

_MAX_LABEL = 200
_DRIVE = re.compile(r"^[A-Za-z]:$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def label_from_path(path: str | None) -> str | None:
    """The last component of ``path`` as a label, either separator, case kept.

    None for an empty path, a bare drive (``C:``), ``.`` or ``..``."""
    if not path:
        return None
    name = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()
    if not name or name in (".", "..") or _DRIVE.match(name):
        return None
    return name[:_MAX_LABEL]


def clean_label(label: object) -> str | None:
    """A client-sent ``project`` as a label, or None when it isn't usable as one.

    A label never carries a separator (skill scopes are ``project:<label>`` and the
    allowlists hold bare names), so a path an older or hand-rolled client sends is cut
    to its last component, exactly as a ``cwd`` would be."""
    if not isinstance(label, str):
        return None
    return label_from_path(_CONTROL.sub("", label))
