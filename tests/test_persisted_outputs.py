"""plugin/scripts/persisted_outputs.py: a large tool result Claude Code saved to a file is read
back into the record a hook ships, when the server could capture the call.

The server cannot open client files, so without this the call is skipped as
``persisted_output``. The guards: only research-shaped calls, only files under the projects
directory, only files that exist, a cap per result and per POST, and the caller's records are
never changed.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from ingestion.capture import calls, detect, guards, shapes
from ingestion.models import Episode

_REPO = Path(__file__).resolve().parents[1]
_PAGE = "# Release notes\n\n" + "The measured latency was low and the design held up. " * 20


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "persisted_outputs", _REPO / "plugin" / "scripts" / "persisted_outputs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


po = _load()


def redirect(path: Path) -> str:
    return (
        "<persisted-output>\n"
        f"Output too large (31.5KB). Full output saved to: {path}\n\n"
        "Preview (first 2KB):\n# Release notes\n...\n</persisted-output>"
    )


def pair(name: str, tool_input: dict, text: str, tag: str = "a", is_error: bool = False):
    use = {"type": "tool_use", "id": f"toolu_{tag}", "name": name, "input": tool_input}
    res = {"type": "tool_result", "tool_use_id": f"toolu_{tag}", "content": text}
    if is_error:
        res["is_error"] = True
    return [
        {"type": "assistant", "uuid": f"a-{tag}", "message": {"content": [use]}},
        {"type": "user", "uuid": f"r-{tag}", "message": {"content": [res]}},
    ]


def result_of(records: list[dict]) -> str:
    return records[1]["message"]["content"][0]["content"]


@pytest.fixture()
def root(tmp_path):
    d = tmp_path / "projects" / "-work-app" / "sess" / "tool-results"
    d.mkdir(parents=True)
    return tmp_path / "projects"


def saved(root: Path, text: str = _PAGE, name: str = "toolu_a.txt") -> Path:
    path = root / "-work-app" / "sess" / "tool-results" / name
    path.write_text(text, encoding="utf-8")
    return path


FETCH = ("mcp__acme__fetch", {"url": "https://example.com/post"})


def test_a_redirect_of_a_research_call_is_replaced_by_the_file(root):
    recs = pair(*FETCH, redirect(saved(root)))
    out = po.inline(recs, root)
    assert result_of(out) == _PAGE
    assert "persisted-output" in result_of(recs)  # the caller's records are as they were read


def test_a_search_with_a_query_counts_and_a_json_file_of_text_blocks_is_joined(root):
    path = saved(
        root, json.dumps([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])
    )
    path = path.rename(path.with_suffix(".json"))
    out = po.inline(pair("web_search", {"query": "x"}, redirect(path)), root)
    assert result_of(out) == "one\ntwo"


def test_the_error_form_and_a_path_with_spaces(root):
    d = root / "-work-app" / "my project"
    d.mkdir()
    path = d / "out.txt"
    path.write_text(_PAGE, encoding="utf-8")
    text = f"Error: result (54,120 characters) exceeds maximum allowed tokens. Output has been saved to {path}\nFormat: text"
    assert result_of(po.inline(pair(*FETCH, text), root)) == _PAGE


def test_a_redirect_stays_when_the_file_cannot_be_trusted_or_read(root, tmp_path):
    outside = tmp_path / "elsewhere.txt"
    outside.write_text(_PAGE, encoding="utf-8")
    link = root / "-work-app" / "sess" / "tool-results" / "link.txt"
    link.symlink_to(outside)  # a link out of the projects directory
    big = saved(root, "x" * (po.MAX_RESULT_BYTES + 1), "big.txt")
    for path in (outside, link, root / "gone.txt", big, root / "-work-app"):
        recs = pair(*FETCH, redirect(path))
        assert result_of(po.inline(recs, root)) == result_of(recs), path


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("Bash", {"command": "curl https://example.com/post"}),  # a shell
        ("mcp__claude_ai_Gmail__search_threads", {"query": "invoice"}),  # personal data
        ("mcp__playwright__browser_navigate", {"url": "https://example.com"}),  # a browser
        ("mcp__synapse__recall", {"query": "x"}),  # Synapse itself
        ("Read", {"file_path": "/etc/hosts"}),  # nothing to look up
    ],
)
def test_only_calls_the_server_could_capture_are_inlined(root, name, tool_input):
    recs = pair(name, tool_input, redirect(saved(root)))
    assert result_of(po.inline(recs, root)) == result_of(recs)


def test_an_error_result_or_a_result_without_its_call_is_left_alone(root):
    errored = pair(*FETCH, redirect(saved(root)), is_error=True)
    assert result_of(po.inline(errored, root)) == result_of(errored)
    orphan = pair(*FETCH, redirect(saved(root)))[1:]  # the tool_use is in an earlier batch
    assert po.inline(orphan, root) == orphan


def test_a_post_carries_at_most_its_budget(root, monkeypatch):
    monkeypatch.setattr(po, "MAX_POST_BYTES", len(_PAGE.encode()) + 10)
    recs = pair(*FETCH, redirect(saved(root)), "a") + pair(
        "mcp__acme__fetch",
        {"url": "https://example.com/two"},
        redirect(saved(root, name="toolu_b.txt")),
        "b",
    )
    out = po.inline(recs, root)
    assert [result_of(out[i : i + 2]) == _PAGE for i in (0, 2)] == [True, False]


def test_it_never_raises(root):
    assert po.inline([{"message": {"content": [{"type": "tool_result", "content": 3}]}}], root)
    assert po.inline(pair(*FETCH, redirect(saved(root))), None) is not None  # a bad root


def test_the_server_captures_what_the_hook_inlined(root):
    """Both halves: the redirect is a skip, the inlined result is a page."""
    recs = pair(*FETCH, redirect(saved(root)))

    def decision(records):
        ep = Episode(
            session_id="s", sequence=1, content="x", platform="claude_code", records=records
        )
        ((call,),) = [calls.episode_tool_calls(ep)]
        return detect.decide(call)

    assert decision(recs) == detect.Skip("persisted_output")
    assert isinstance(decision(po.inline(recs, root)), detect.Capture)


def test_the_hooks_never_list_mirrors_the_servers():
    deny = set(tomllib.loads((_REPO / "ingestion/capture/rules/deny.toml").read_text())["words"])
    assert po._NEVER_WORDS == deny | guards._BROWSER_WORDS | {"synapse"}
    assert po._RESEARCH_KEYS == shapes._URL_KEYS + shapes._QUERY_KEYS
    assert po._REDIRECT.pattern == detect._PERSISTED_REDIRECT.pattern
    assert po._SHELL_KEYS == ("command", "cmd")


def test_the_hooks_post_body_carries_the_inlined_record(root, monkeypatch, tmp_path):
    from tests.test_ingest_hook_tail import _PLUGIN_HOOK, _isolated_plugin_env, _load_hook

    _isolated_plugin_env(monkeypatch, tmp_path)
    hook = _load_hook(_PLUGIN_HOOK)
    monkeypatch.setattr(hook.config, "PROJECTS_DIR", root)
    recs = pair(*FETCH, redirect(saved(root)))
    body = hook._payload(recs, "hook")
    assert result_of(body["records"]) == _PAGE and body["source"] == "hook"
    assert "persisted-output" in result_of(recs)
