"""Cross-platform exclusive file locks for the plugin scripts.

POSIX has fcntl.flock; Windows has msvcrt.locking (byte-range, so we lock the
first byte — every caller locks a dedicated lock file, never a data file, so a
1-byte range is equivalent to a whole-file lock). Callers treat OSError from a
non-blocking acquire as "someone else holds it", which both backends raise.
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
