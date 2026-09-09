"""The personal-scope switch (SYNAPSE_PERSONAL_SCOPE, ingestion/scope.py).

Four points read the switch: the extractor's group routing, the timeline gate's
prompt + stored domain, the recall tool's group_id, and the timeline ingest
route. Each is pinned here in both positions, because "off" has to be off
everywhere or a work-only deployment still ends up with writes in a graph its
reads never touch.

Pure unit tests: no DB, no LLM. Every case sets the env var with monkeypatch, so
the switch must be read per call, not frozen into a module-level constant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult
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
# Extractor routing
# ---------------------------------------------------------------------------


def test_default_group_for_project_respects_switch(monkeypatch):
    from ingestion.extractor import _default_group_for_project

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _default_group_for_project("jobs") == "personal"
    assert _default_group_for_project("synapse") == "technical"

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert _default_group_for_project("jobs") == "technical"
    assert _default_group_for_project("personal") == "technical"
    assert _default_group_for_project(None) == "technical"


def test_classify_entity_group_respects_switch(monkeypatch):
    from ingestion.extractor import _classify_entity_group

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _classify_entity_group("dentist appointment", None, "technical") == "personal"
    assert _classify_entity_group("Postgres", None, "personal") == "technical"

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    # The personal regex never runs: "dentist appointment" would otherwise land
    # in a graph the default group_id="technical" search never reads.
    assert _classify_entity_group("dentist appointment", None, "technical") == "technical"
    assert _classify_entity_group("Postgres", None, "personal") == "technical"


def test_notes_group_routing_follows_switch(monkeypatch):
    """The notes/preferences lanes derive their group from the same helper."""
    from ingestion.preferences_gate import _group_for

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _group_for("jobs") == "personal"
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert _group_for("jobs") == "technical"


def _entity(name: str) -> ExtractedEntity:
    return ExtractedEntity(name=name, type="Tool", summary="")


def _pipeline():
    from ingestion.extractor import ExtractionPipeline

    embedder = MagicMock()
    embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
    db = MagicMock()
    db.get_session_episodes.return_value = []
    db.get_synth_document_source_ids.return_value = []
    kg = MagicMock()
    kg.find_similar_nodes.return_value = []
    return ExtractionPipeline(db=db, llm_client=MagicMock(), embedder=embedder, kg_client=kg)


def _run_item(pipe, project: str):
    """Drive process_item with one personal-flavored fact; return the groups used."""
    entities = [_entity("dentist appointment"), _entity("calendar")]
    facts = [
        ExtractedFact(
            source="dentist appointment",
            target="calendar",
            relationship="BOOKED_IN",
            fact="the dentist appointment was booked in the calendar",
        )
    ]
    pipe._stage2_deterministic = MagicMock(return_value=[])
    pipe._stage3_llm = MagicMock(return_value=ExtractionResult(entities=entities, facts=facts))
    pipe._stage4_resolve = MagicMock(
        side_effect=lambda ents, group_id, deduper=None: {e.name: f"new:{e.name}" for e in ents}
    )
    pipe._stage5_write_nodes = MagicMock()
    pipe._process_facts_for_group = MagicMock()
    pipe._deduper_for = MagicMock(return_value=MagicMock())

    pipe.process_item(
        {
            "content_type": "summary",
            "content": "booked the dentist appointment",
            "project": project,
            "session_id": "s1",
        }
    )
    resolve_groups = {c.args[1] for c in pipe._stage4_resolve.call_args_list}
    fact_groups = {c.args[3] for c in pipe._process_facts_for_group.call_args_list}
    return resolve_groups, fact_groups


def test_process_item_splits_groups_when_scope_on(monkeypatch):
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    resolve_groups, fact_groups = _run_item(_pipeline(), project="jobs")
    assert resolve_groups == {"personal"}
    assert fact_groups == {"personal"}


def test_process_item_routes_everything_technical_when_scope_off(monkeypatch):
    """Same item, scope off: the personal bucket is empty and nothing is written
    to a graph the default read never opens."""
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    resolve_groups, fact_groups = _run_item(_pipeline(), project="jobs")
    assert resolve_groups == {"technical"}
    assert fact_groups == {"technical"}


# ---------------------------------------------------------------------------
# Timeline gate: prompt + stored domain
# ---------------------------------------------------------------------------


def test_gate_prompt_asks_for_domain_when_scope_on(monkeypatch):
    from ingestion.timeline_gate import GATE_PROMPT, gate_prompt

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    p = gate_prompt()
    assert p == GATE_PROMPT
    assert 'domain: "personal"' in p
    assert '"domain": "personal"|"technical"' in p


def test_gate_prompt_drops_domain_when_scope_off(monkeypatch):
    from ingestion.timeline_gate import gate_prompt

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    p = gate_prompt()
    assert "domain" not in p
    # The rest of the prompt is untouched: same framing, same output contract.
    assert "QUOTED MATERIAL" in p
    assert 'Output ONLY JSON: {"events"' in p
    assert '"event_type": "decision"|"action"|"finding"|"milestone", "date"' in p


class _RecordingDb:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    def get_episodes_valid_at(self, ids):
        return "2026-09-04T10:00:00+00:00"

    def timeline_ident_exists(self, *a, **k):
        return False

    def insert_timeline_event(self, **kw):
        self.inserted.append(kw)
        return 1


class _StubEmb:
    model_name = "stub"

    def embed(self, texts, task=None):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _gate_with(monkeypatch, domain):
    import ingestion.timeline_gate as tg
    from ingestion.llm_schemas import TimelineGateEvents

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    monkeypatch.setenv("SYNAPSE_TIMELINE_DEDUP", "0")
    db = _RecordingDb()
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "structured_call",
        lambda *a, **k: TimelineGateEvents.model_validate(
            {
                "events": [
                    {
                        "event": "booked a dentist appointment",
                        "salience": 1,
                        "event_type": "action",
                        "domain": domain,
                        "date": None,
                    }
                ]
            }
        ),
    )
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "neuron"})
    return db.inserted


def test_gate_stores_model_domain_when_scope_on(monkeypatch):
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _gate_with(monkeypatch, "personal")[0]["domain"] == "personal"


def test_gate_stores_technical_when_scope_off(monkeypatch):
    """Deterministic stored value: even a model that volunteers "personal"
    (an old prompt cached, a stray few-shot) writes technical."""
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert _gate_with(monkeypatch, "personal")[0]["domain"] == "technical"
    assert _gate_with(monkeypatch, None)[0]["domain"] == "technical"


# ---------------------------------------------------------------------------
# Read side: recall + the timeline ingest route
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
    monkeypatch.setattr(srv, "_caller_trust", lambda surface: None)
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


def test_timeline_ingest_domain(monkeypatch):
    """The push route's own git-vs-explicit logic is unchanged; only a personal
    label collapses when the scope is off."""
    from mcp_server.timeline_routes import _event_domain

    git = {"source": "git:synapse", "domain": None}
    personal = {"source": "chat", "domain": "personal"}
    unlabeled = {"source": "chat", "domain": None}

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
    assert _event_domain(git) == "technical"
    assert _event_domain(personal) == "personal"
    assert _event_domain(unlabeled) is None

    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert _event_domain(git) == "technical"
    assert _event_domain(personal) == "technical"
    # Unlabeled still fails open at read: the switch does not invent a label.
    assert _event_domain(unlabeled) is None
