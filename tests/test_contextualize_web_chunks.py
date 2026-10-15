"""scripts/contextualize_web_chunks.py: the retry loop is parse_with_retry, and an LLM
outage or exhausted quota stops the run instead of failing every remaining artifact."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.llm_client import LLMUnavailableError, UsageLimitError

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "contextualize_web_chunks.py"
_spec = importlib.util.spec_from_file_location("contextualize_web_chunks", _SCRIPT)
assert _spec and _spec.loader
cwc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cwc  # pydantic resolves the script's string annotations here
_spec.loader.exec_module(cwc)

_CHUNKS = [(10, 0, "first chunk"), (11, 1, "second chunk")]


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, params):
        self._conn.updates.extend(params)


class _Conn:
    """Just enough of a psycopg connection for contextualize_artifact."""

    def __init__(self):
        self.updates: list[tuple[str, int]] = []

    def execute(self, sql, params=None):
        return SimpleNamespace(fetchall=lambda: _CHUNKS)

    def cursor(self):
        return _Cursor(self)


class _LLM:
    """Replies with the scripted texts in order; an Exception in the script is raised."""

    def __init__(self, script):
        self._script = list(script)
        self.prompts: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        reply = self._script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])


def _reply(*pairs):
    return json.dumps({"contexts": [{"idx": i, "context": c} for i, c in pairs]})


def test_invalid_reply_is_fed_back_and_retried():
    conn = _Conn()
    llm = _LLM(["```not json```", _reply((0, "about a"), (1, "about b"))])
    n, err = cwc.contextualize_artifact(conn, llm, 1, "doc", "m")
    assert (n, err) == (2, None)
    assert conn.updates == [("about a", 10), ("about b", 11)]
    assert len(llm.prompts) == 2
    assert "not valid JSON" in llm.prompts[1]


def test_unusable_replies_after_the_retries_are_the_artifacts_error():
    conn = _Conn()
    llm = _LLM(["not json"] * 3)
    n, err = cwc.contextualize_artifact(conn, llm, 1, "doc", "m")
    assert n == 0
    assert err and "after retries" in err
    assert len(llm.prompts) == 3
    assert conn.updates == []


@pytest.mark.parametrize("exc", [UsageLimitError("quota"), LLMUnavailableError("down")])
def test_transport_failure_propagates_after_one_attempt(exc):
    llm = _LLM([exc])
    with pytest.raises(type(exc)):
        cwc.contextualize_artifact(_Conn(), llm, 1, "doc", "m")
    assert len(llm.prompts) == 1
