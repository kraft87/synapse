"""CI has to test what production runs."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PARADEDB_IMAGE = re.compile(r"^\s+image:\s*(paradedb/\S+)\s*$", re.MULTILINE)


def _image(path: str) -> str:
    m = _PARADEDB_IMAGE.search((_ROOT / path).read_text())
    assert m, f"{path} names no paradedb image"
    return m.group(1)


def test_ci_tests_against_the_database_image_compose_runs():
    """A service container's image cannot read a shared variable, so ci.yml carries a
    copy of compose.yml's digest-pinned image. Bump the two together."""
    ci = _image(".github/workflows/ci.yml")
    assert "@sha256:" in ci, "CI's database image must be pinned by digest, as compose.yml's is"
    assert ci == _image("compose.yml")
