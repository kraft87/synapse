"""The project resolver's old name, kept for the hooks that import it.

The resolver lives in the client core, ``plugin/synapse_client/project.py``. The Claude
hooks here and the Codex plugin's hooks import ``synapse_project`` off their own
sys.path (Codex's skills sync loads this file by path), so this name re-exports the
public functions. ``plugin/`` goes first on sys.path, so a different ``synapse_client``
installed on the host cannot stand in for this checkout's. The other folders there
(``bin/``, ``commands/``, ``hooks/``, ``scripts/``) have no ``__init__.py``: at most they
become namespace packages, and those never hide a module of the same name found later
on the path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse_client.project import (
    MARKER,
    git_exclude,
    label_from_path,
    project_label,
    resolve,
)

__all__ = ["MARKER", "git_exclude", "label_from_path", "project_label", "resolve"]
