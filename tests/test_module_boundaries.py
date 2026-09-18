"""Keep the decomposed retrieval, storage, extraction, and server modules bounded."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PATTERNS = (
    "ingestion/db*.py",
    "ingestion/extract*.py",
    "mcp_server/recall*.py",
    "mcp_server/server*.py",
    "mcp_server/caller_trust.py",
    "mcp_server/http_auth.py",
    "mcp_server/auxiliary_routes.py",
    "mcp_server/retrieval_tools.py",
    "mcp_server/remember_tool.py",
    "mcp_server/feedback_tools.py",
    "mcp_server/ingest_route.py",
)
_MODULES = sorted({path for pattern in _PATTERNS for path in _ROOT.glob(pattern)})


@pytest.mark.parametrize("path", _MODULES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_refactored_module_size(path: Path) -> None:
    # Count physical lines, including documentation, so the ceiling cannot be
    # satisfied by hiding a large module behind comments or string literals.
    lines = len(path.read_text().splitlines())
    assert lines <= 600, f"{path.relative_to(_ROOT)} has {lines} lines; split by responsibility"
