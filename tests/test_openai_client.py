"""Tests for the ChatGPT export parser (pure unit tests — no I/O beyond tmp files)."""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.openai_client import OpenAIChatParser, parse_export

CONV_ID = "67e1e894-0000-4000-8000-000000000001"


def _msg(mid: str, role: str, text: str, create_time: float, ctype: str = "text") -> dict:
    return {
        "id": mid,
        "author": {"role": role},
        "create_time": create_time,
        "content": {"content_type": ctype, "parts": [text]},
        "metadata": {"model_slug": "gpt-5" if role == "assistant" else None},
    }


def _node(nid: str, msg: dict | None, parent: str | None, children: list[str]) -> dict:
    return {"id": nid, "message": msg, "parent": parent, "children": children}


def _conversation(**overrides) -> dict:
    """root -> u1 -> a1-old (abandoned) / a1 -> u2 -> thoughts -> a2"""
    mapping = {
        "root": _node("root", None, None, ["n-u1"]),
        "n-u1": _node(
            "n-u1",
            _msg("m-u1", "user", "how do I sort a list?", 1750000000.0),
            "root",
            ["n-a1-old", "n-a1"],
        ),
        "n-a1-old": _node(
            "n-a1-old", _msg("m-a1-old", "assistant", "ABANDONED BRANCH", 1750000010.0), "n-u1", []
        ),
        "n-a1": _node(
            "n-a1", _msg("m-a1", "assistant", "Use sorted(xs).", 1750000020.0), "n-u1", ["n-u2"]
        ),
        "n-u2": _node(
            "n-u2", _msg("m-u2", "user", "and descending?", 1750000100.0), "n-a1", ["n-th"]
        ),
        "n-th": _node(
            "n-th",
            _msg("m-th", "assistant", "hidden reasoning", 1750000110.0, ctype="thoughts"),
            "n-u2",
            ["n-a2"],
        ),
        "n-a2": _node(
            "n-a2", _msg("m-a2", "assistant", "sorted(xs, reverse=True).", 1750000120.0), "n-th", []
        ),
    }
    convo = {
        "conversation_id": CONV_ID,
        "id": CONV_ID,
        "title": "sorting help",
        "create_time": 1750000000.0,
        "current_node": "n-a2",
        "default_model_slug": "gpt-4o",
        "is_do_not_remember": False,
        "mapping": mapping,
    }
    convo.update(overrides)
    return convo


class TestParseConversation:
    def test_two_turns_canonical_branch(self):
        eps = OpenAIChatParser.parse_conversation(_conversation())
        assert len(eps) == 2
        first, second = eps
        assert first.session_id == CONV_ID
        assert first.human_turn == "how do I sort a list?"
        assert first.assistant_turn == "Use sorted(xs)."
        assert first.span_id == "openai:m-a1"
        assert first.source == "openai"
        assert first.platform == "chatgpt"
        assert first.model == "gpt-5"
        assert first.content.startswith("[title] sorting help")
        assert first.created_at is not None
        assert first.created_at.year == 2025

        assert second.human_turn == "and descending?"
        assert second.span_id == "openai:m-a2"
        assert second.content.startswith("[context] Use sorted(xs).")

    def test_abandoned_branch_excluded(self):
        eps = OpenAIChatParser.parse_conversation(_conversation())
        assert "ABANDONED BRANCH" not in "\n".join(e.content for e in eps)

    def test_thoughts_excluded(self):
        eps = OpenAIChatParser.parse_conversation(_conversation())
        assert "hidden reasoning" not in "\n".join(e.content for e in eps)

    def test_do_not_remember_skipped(self):
        assert OpenAIChatParser.parse_conversation(_conversation(is_do_not_remember=True)) == []

    def test_broken_chain_falls_back_to_time_order(self):
        eps = OpenAIChatParser.parse_conversation(_conversation(current_node="missing-node"))
        # fallback takes ALL message nodes by create_time, including the
        # abandoned sibling — degraded but complete
        assert len(eps) == 2
        assert eps[0].human_turn == "how do I sort a list?"

    def test_event_time_from_message_epoch(self):
        eps = OpenAIChatParser.parse_conversation(_conversation())
        assert eps[0].created_at.isoformat().startswith("2025-06-15")

    def test_default_model_fallback(self):
        convo = _conversation()
        for node in convo["mapping"].values():
            if node["message"]:
                node["message"]["metadata"] = {}
        eps = OpenAIChatParser.parse_conversation(convo)
        assert eps[0].model == "gpt-4o"

    def test_multimodal_image_part(self):
        convo = _conversation()
        convo["mapping"]["n-u1"]["message"]["content"] = {
            "content_type": "multimodal_text",
            "parts": [{"asset_pointer": "file-abc"}, "what is this?"],
        }
        eps = OpenAIChatParser.parse_conversation(convo)
        assert eps[0].human_turn == "[image]\n\nwhat is this?"


class TestParseExport:
    def test_sharded_layout(self, tmp_path: Path):
        (tmp_path / "conversations-000.json").write_text(json.dumps([_conversation()]))
        second = _conversation(
            conversation_id="67e1e894-0000-4000-8000-000000000002", is_do_not_remember=True
        )
        (tmp_path / "conversations-001.json").write_text(json.dumps([second]))
        eps, skipped = parse_export(tmp_path)
        assert len(eps) == 2
        assert skipped == 1

    def test_single_file_layout(self, tmp_path: Path):
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([_conversation()]))
        eps, skipped = parse_export(tmp_path)
        assert len(eps) == 2
        assert skipped == 0

    def test_direct_file_path(self, tmp_path: Path):
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([_conversation()]))
        eps, _ = parse_export(p)
        assert len(eps) == 2
