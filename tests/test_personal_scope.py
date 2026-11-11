"""The personal-scope switch (SYNAPSE_PERSONAL_SCOPE, ingestion/scope.py).

The recall tool's group_id reads the switch. It is pinned here in both positions,
because "off" has to be off everywhere or a work-only deployment still asks for a
graph it does not keep.

Pure unit tests: no DB, no LLM. Every case sets the env var with monkeypatch, so
the switch must be read per call, not frozen into a module-level constant.
"""

from __future__ import annotations

import pytest

from ingestion.scope import active_groups, coerce_group, personal_scope_enabled

# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_scope_defaults_on(monkeypatch):
    monkeypatch.delenv("SYNAPSE_PERSONAL_SCOPE", raising=False)
    assert personal_scope_enabled() is True
    assert active_groups() == ("technical", "personal")


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off "])
def test_scope_off_values(monkeypatch, raw):
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", raw)
    assert personal_scope_enabled() is False
    assert active_groups() == ("technical",)


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "", "  ", "banana"])
def test_scope_on_values(monkeypatch, raw):
    """Anything that is not a recognized off value keeps the split: a typo must
    not silently collapse the two graphs into one."""
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", raw)
    assert personal_scope_enabled() is True


def test_coerce_group(monkeypatch):
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert coerce_group("personal") == "personal"
    assert coerce_group("technical") == "technical"
    assert coerce_group(None) is None

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert coerce_group("personal") == "technical"
    assert coerce_group(" Personal ") == "technical"
    assert coerce_group("technical") == "technical"
    assert coerce_group(None) is None
    # Unknown values pass through; validating them belongs to the caller.
    assert coerce_group("something-else") == "something-else"


# ---------------------------------------------------------------------------
# Read side: the recall tool
# ---------------------------------------------------------------------------


def _recall_tool(monkeypatch):
    """Call the MCP recall tool with the retrieval layer stubbed; return the
    group_id the server passed down."""
    import mcp_server.server as srv

    captured: dict[str, str] = {}

    class _Stub:
        def recall(self, **kw):
            captured["group_id"] = kw["group_id"]
            return {"results": []}

    monkeypatch.setattr(srv, "_get_recall", lambda: _Stub())
    monkeypatch.setattr(srv, "_caller_trust", lambda: None)
    srv.recall(query="what did we decide", group_id="personal")
    return captured["group_id"]


def test_recall_tool_keeps_personal_when_scope_on(monkeypatch):
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _recall_tool(monkeypatch) == "personal"


def test_recall_tool_coerces_personal_when_scope_off(monkeypatch):
    """A model that asks for the retired scope gets the graph that was actually
    written, not an empty one."""
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert _recall_tool(monkeypatch) == "technical"


def test_recall_tool_docstring_hides_personal_scope(monkeypatch):
    """The tool description is the model's only view of the scope; with the
    scope off it must not mention a graph this deployment does not keep."""
    import mcp_server.server as srv

    doc = srv.recall.__doc__ or ""
    assert srv._GROUP_ID_DOC_SPLIT in doc

    def _fn():
        pass

    _fn.__doc__ = doc
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert srv._GROUP_ID_DOC_SPLIT in (srv._scope_doc(_fn).__doc__ or "")

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    stripped = srv._scope_doc(_fn).__doc__ or ""
    assert srv._GROUP_ID_DOC_SPLIT not in stripped
    assert "personal" not in stripped
