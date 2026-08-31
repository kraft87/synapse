"""Cross-platform exclusive file locks for the plugin scripts.

POSIX has fcntl.flock; Windows has msvcrt.locking (byte-range, so we lock the
first byte — every caller locks a dedicated lock file, never a data file, so a
1-byte range is equivalent to a whole-file lock). Callers treat OSError from a
non-blocking acquire as "someone else holds it", which both backends raise.

Named ``synapse_filelock`` rather than ``filelock``: the hooks run by inserting
this directory onto sys.path, so a module here named after a real PyPI package
SHADOWS it process-wide once imported. As ``filelock`` it broke huggingface_hub
(``from filelock import BaseFileLock``) in any process that had loaded a plugin
script first — which in the test suite is whichever worker happened to draw both
files. Keep plugin-script module names prefixed for that reason.
"""

from __future__ import annotations

from typing import IO

try:
    import fcntl

    def lock_exclusive(f: IO, *, blocking: bool = True) -> None:
        fcntl.flock(f, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)

except ImportError:  # Windows
    import msvcrt

    def lock_exclusive(f: IO, *, blocking: bool = True) -> None:
        # LK_LOCK retries for ~10s then raises OSError; that beats blocking a
        # 15s-budget hook forever, so it stands in for a true blocking lock.
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
