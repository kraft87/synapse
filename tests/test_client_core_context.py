"""The core's session-start block, per-prompt nudge and private-mode check
(plugin/synapse_client/context.py, private.py).

The wording is the hooks' wording, byte for byte: each renderer is compared with the
hook it ports (recall_nudge.py, board_block.py, enroll.py and
ingest_hook._is_private), loaded by path over a scratch home with its server call
replaced. The one intended difference, the Claude-only paste hint in the restricted
block, is spelled out where it is compared. The preferences block is the server's
(mcp_server/preferences_routes.render_block), and the core's copy for an older server
is compared with that.
"""

from __future__ import annotations

import email.message
import hashlib
import json
import os
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers.client_core import (
    drop_scripts,
    fake_urlopen,
    load_script,
    requested,
    scratch_user,
    use_core,
)

use_core()
from synapse_client import context  # noqa: E402
from synapse_client.private import is_private  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402

_CLAUDE_HINT = "run `! synapse-login` in this session (or `synapse-login` in a terminal)"
_CLAUDE_RESTRICTED_HINT = "run `! synapse-login` in this session"
_PASTE_HINT = " (or paste it into\n    /plugin > synapse > Synapse token)"
_NOW = 1_790_000_000.0  # a fixed moment for the timestamp line


@pytest.fixture()
def scratch(monkeypatch, tmp_path):
    yield scratch_user(monkeypatch, tmp_path)
    drop_scripts()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://s.example/context", code, "x", email.message.Message(), None
    )


class _FakeClient:
    """Answers context() and preferences() from canned values; raises an exception
    value instead of returning it. ``board`` is the whole ``/context`` reply: with a
    ``preferences`` key it is a server that renders the block, without one a server
    from before that, whose items are ``prefs``."""

    def __init__(self, board: Any = None, prefs: Any = None) -> None:
        self.board, self.prefs = board, prefs
        self.calls: list[tuple[Any, ...]] = []
        self.timeouts: list[float] = []

    def context(self, project=None, timeout=10, *, parts=()):
        self.calls.append(("context", project, tuple(parts)))
        if isinstance(self.board, BaseException):
            raise self.board
        return self.board

    def preferences(self, limit=7, timeout=10):
        self.calls.append(("preferences", limit))
        self.timeouts.append(timeout)
        if isinstance(self.prefs, BaseException):
            raise self.prefs
        return self.prefs


def _settings(tmp_path: Path, *, enrolled: bool = False, env: dict | None = None) -> Settings:
    home = tmp_path / "core-home"
    home.mkdir(exist_ok=True)
    cfg: dict = {"url": "https://s.example"}
    if enrolled:
        cfg |= {
            "token": "tkDEVICE0123456",
            "device": {"surface_id": "dev-1", "server": "https://s.example"},
        }
    (home / "config.json").write_text(json.dumps(cfg))
    return Settings.load(env=env or {}, home=home)


def _hook_context(capsys) -> str:
    out = capsys.readouterr().out
    if not out.strip():
        return ""
    return json.loads(out)["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# prompt_nudge == recall_nudge.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"SYNAPSE_RECALL_NUDGE": "0"},
        {"SYNAPSE_PROMPT_TIMESTAMP": "0"},
        {"SYNAPSE_RECALL_NUDGE": "0", "SYNAPSE_PROMPT_TIMESTAMP": "0"},
        {"SYNAPSE_RECALL_NUDGE": "", "SYNAPSE_PROMPT_TIMESTAMP": "1"},
    ],
    ids=["defaults", "no-nudge", "no-timestamp", "both-off", "empty-is-unset"],
)
def test_prompt_nudge_is_byte_identical_to_the_hook(scratch, monkeypatch, capsys, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    hook = load_script("recall_nudge")
    # The hook asks the core for the current time; this test's clock is _NOW.
    monkeypatch.setattr(
        context,
        "time",
        SimpleNamespace(strftime=time.strftime, localtime=lambda now=None: time.localtime(_NOW)),
    )
    hook.main()
    expected = context.prompt_nudge(Settings.load(), now=_NOW)
    assert expected == _hook_context(capsys)


def test_prompt_nudge_names_the_tools_as_the_harness_does(tmp_path):
    text = context.prompt_nudge(
        _settings(tmp_path), recall="synapse_recall", remember="synapse_remember", now=_NOW
    )
    assert "BEFORE answering anything that references the past: synapse_recall (" in text
    assert "Call synapse_remember FIRST" in text
    assert "synapse:" not in text
    assert text.splitlines()[0] == time.strftime("[%a %Y-%m-%d %H:%M:%S %Z]", time.localtime(_NOW))


def test_prompt_nudge_honours_config_json_flags(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.json").write_text("{}")
    s = Settings.load(env={}, home=home, extra_candidates=[("x", {"SYNAPSE_RECALL_NUDGE": "0"})])
    assert context.prompt_nudge(s, now=_NOW).count("\n") == 0
    assert "[Synapse]" not in context.prompt_nudge(s, now=_NOW)


# ---------------------------------------------------------------------------
# preferences_block == the block the server renders
# ---------------------------------------------------------------------------

_PREF_CASES = {
    "typical": [
        {"pref": "Terse answers", "polarity": "like"},
        {"pref": "Emoji", "polarity": "dislike"},
        {"pref": "Full-time roles only", "polarity": "rule"},
    ],
    "more-than-seven": [{"pref": f"p{i}", "polarity": "like"} for i in range(12)],
    "skips-empty-prefs-within-the-first-seven": [
        {"pref": "", "polarity": "like"},
        {"polarity": "rule"},
        *({"pref": f"p{i}", "polarity": "dislike"} for i in range(8)),
    ],
    "unknown-and-missing-polarity": [
        {"pref": "a", "polarity": "neutral"},
        {"pref": "b"},
        {"pref": "c", "polarity": None},
    ],
    "nothing-renders": [{"pref": ""}, {"polarity": "like"}],
    "empty": [],
}


def test_the_preferences_block_wording():
    assert context.preferences_block(_PREF_CASES["typical"]) == (
        "[Synapse preferences]\n"
        "  - (likes) Terse answers\n"
        "  - (dislikes) Emoji\n"
        "  - (rule) Full-time roles only"
    )
    assert context.preferences_block(_PREF_CASES["more-than-seven"]).count("\n") == 7
    assert context.preferences_block(_PREF_CASES["unknown-and-missing-polarity"]) == (
        "[Synapse preferences]\n  - (neutral) a\n  - () b\n  - () c"
    )
    assert context.preferences_block(_PREF_CASES["nothing-renders"]) == ""
    assert context.preferences_block([]) == ""


@pytest.mark.parametrize("case", sorted(_PREF_CASES))
def test_preferences_block_is_byte_identical_to_the_servers(case):
    """A client on a server from before ``/context`` carried the block renders the items
    itself: the same text, so the block does not change with the server's version."""
    from mcp_server.preferences_routes import BLOCK_LINES, render_block

    items = _PREF_CASES[case]
    assert context.preferences_block(items) == render_block(items)
    assert BLOCK_LINES == context._PREFS_MAX_LINES


# ---------------------------------------------------------------------------
# The enrollment explainers == enroll.py
# ---------------------------------------------------------------------------


def test_the_explainers_match_enroll_with_the_claude_hints(scratch):
    enroll = load_script("enroll")
    assert context.not_enrolled_block(_CLAUDE_HINT) == enroll.not_enrolled_block()
    # The one difference: the core does not point at the Claude plugin's token slot.
    assert context.restricted_block(_CLAUDE_RESTRICTED_HINT) == enroll.restricted_block().replace(
        _PASTE_HINT, ""
    )
    assert (
        context.restricted_block(_CLAUDE_RESTRICTED_HINT, paste_aside=_PASTE_HINT)
        == enroll.restricted_block()
    )
    assert context.BOOTSTRAP_CMD == enroll.BOOTSTRAP_CMD
    assert context.PASTE_CMD == enroll.PASTE_CMD


def test_the_paste_aside_closes_the_restricted_explainer():
    plain = context.restricted_block()
    assert plain.endswith(f"    on the server, then run `{context.PASTE_CMD}` here.")
    assert context.restricted_block(paste_aside="") == plain
    aside = context.restricted_block(paste_aside=" (or in the token slot)")
    assert aside == plain[:-1] + " (or in the token slot)."


def test_the_default_hint_is_harness_neutral():
    text = context.not_enrolled_block()
    assert text.endswith("Run `synapse login` in a terminal and approve the sign-in on any device.")
    restricted = context.restricted_block()
    assert (
        "  - Server with a sign-in configured: run `synapse login` in a terminal.\n" in restricted
    )
    for block in (text, restricted):
        assert "/plugin" not in block and "! synapse-login" not in block


# ---------------------------------------------------------------------------
# The board part == what board_block.py prints for the board
# ---------------------------------------------------------------------------

_BOARD_CASES = {
    "ok": ({"status": "ok", "text": "BOARD", "trust": "full"}, False),
    "ok-empty-text": ({"status": "ok", "text": ""}, False),
    "not-ok": ({"status": "error", "text": "ignored"}, False),
    "restricted-unenrolled": ({"status": "ok", "text": "BOARD", "trust": "restricted"}, False),
    "restricted-enrolled": ({"status": "ok", "text": "BOARD", "trust": "restricted"}, True),
    "401-unenrolled": (401, False),
    "401-enrolled": (401, True),
    "403": (403, False),
    "500": (500, False),
}
_BOARD_ONLY = {"SYNAPSE_PREFS_BLOCK": "0"}


@pytest.mark.parametrize("case", sorted(_BOARD_CASES))
def test_the_board_part_matches_the_hook(scratch, monkeypatch, capsys, tmp_path, case):
    reply, enrolled = _BOARD_CASES[case]
    answer = _http_error(reply) if isinstance(reply, int) else reply
    if enrolled:  # the hook reads enrollment from the device record, as it does at start
        (scratch / ".synapse").mkdir(parents=True, exist_ok=True)
        record = {"device": {"surface_id": "dev-1"}}
        (scratch / ".synapse" / "config.json").write_text(json.dumps(record))
    monkeypatch.setenv("SYNAPSE_PREFS_BLOCK", "0")  # the board alone

    hook = load_script("board_block")
    seen = fake_urlopen(monkeypatch, answer)
    monkeypatch.setattr(hook, "_project_label", lambda: "proj")
    hook.main()
    assert [requested(req) for req in seen] == [("/context", {"project": "proj", "parts": "board"})]
    # The hook prints the parts joined by blank lines (no credential notice applies here).
    expected = _hook_context(capsys).replace(_PASTE_HINT, "")

    client = _FakeClient(board=answer)
    settings = _settings(tmp_path, enrolled=enrolled, env=_BOARD_ONLY)
    got, prefs = context.server_parts(client, "proj", settings, login_hint=_CLAUDE_HINT)
    # The hook's restricted block names its own hint; compare the rest of the wording.
    got = got.replace(f"configured: {_CLAUDE_HINT}.", f"configured: {_CLAUDE_RESTRICTED_HINT}.")
    assert (got, prefs) == (expected, "")
    assert client.calls == [("context", "proj", ("board",))]


# ---------------------------------------------------------------------------
# server_parts: one request for the board and the preferences
# ---------------------------------------------------------------------------

_PREFS = [{"pref": "Terse", "polarity": "like"}]
_PREFS_TEXT = "[Synapse preferences]\n  - (likes) Terse"
_BOTH = ("board", "preferences")


def _reply(text: str = "BOARD", **more: Any) -> dict[str, Any]:
    """A ``/context`` reply. Pass ``preferences=`` for a server that renders the block."""
    return {"status": "ok", "text": text, **more}


def test_the_board_and_the_preferences_are_one_request(tmp_path):
    client = _FakeClient(board=_reply(preferences=_PREFS_TEXT))
    assert context.server_parts(client, "p", _settings(tmp_path)) == ("BOARD", _PREFS_TEXT)
    assert client.calls == [("context", "p", _BOTH)]


def test_the_servers_block_is_shown_as_it_came(tmp_path):
    """The server renders the block: the client adds no header and drops no line."""
    served = "[Synapse preferences]\n" + "\n".join(f"  - (rule) r{i}" for i in range(9))
    client = _FakeClient(board=_reply(preferences=served))
    assert context.server_parts(client, None, _settings(tmp_path))[1] == served


@pytest.mark.parametrize("served", ["", None, ["x"], {"text": "x"}, 7])
def test_a_reply_with_no_block_to_show_says_nothing(tmp_path, served):
    client = _FakeClient(board=_reply(preferences=served), prefs=_PREFS)
    assert context.server_parts(client, None, _settings(tmp_path)) == ("BOARD", "")
    assert client.calls == [("context", None, _BOTH)]  # the server answered: no second request


def test_a_server_from_before_the_block_is_asked_for_its_top_preferences(tmp_path):
    client = _FakeClient(board=_reply(), prefs=_PREFS)
    assert context.server_parts(client, "p", _settings(tmp_path)) == ("BOARD", _PREFS_TEXT)
    assert client.calls == [("context", "p", _BOTH), ("preferences", 7)]


def test_a_board_that_was_not_asked_for_is_not_shown(tmp_path):
    """A server from before ``parts`` answers with the board whatever is asked."""
    client = _FakeClient(board=_reply(trust="restricted"), prefs=_PREFS)
    settings = _settings(tmp_path, env={"SYNAPSE_BOARD": "0"})
    assert context.server_parts(client, "p", settings) == ("", _PREFS_TEXT)
    assert client.calls == [("context", None, ("preferences",)), ("preferences", 7)]


def test_the_second_request_has_what_is_left_of_the_timeout(tmp_path, monkeypatch):
    clock = iter([100.0, 104.0, 200.0, 212.0])
    monkeypatch.setattr(context, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    client = _FakeClient(board=_reply(), prefs=_PREFS)
    assert context.server_parts(client, None, _settings(tmp_path), timeout=10)[1] == _PREFS_TEXT
    assert client.timeouts == [6.0]
    # Nothing left: the board is shown and the preferences are not asked for.
    assert context.server_parts(client, None, _settings(tmp_path), timeout=10) == ("BOARD", "")
    assert client.timeouts == [6.0]


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("refused"), TimeoutError(), ValueError("bad json"), _http_error(500)],
)
def test_a_second_request_that_fails_keeps_the_board(tmp_path, failure):
    client = _FakeClient(board=_reply(), prefs=failure)
    assert context.server_parts(client, None, _settings(tmp_path)) == ("BOARD", "")


def test_a_malformed_preference_item_fails_open(tmp_path):
    client = _FakeClient(board=_reply("B"), prefs=[{"pref": "x", "polarity": ["?"]}])
    assert context.session_block(client, None, _settings(tmp_path)) == "B"


@pytest.mark.parametrize(
    "reply", [None, [], "text", {"status": "ok", "text": ["B"], "preferences": _PREFS_TEXT}]
)
def test_a_reply_that_is_not_the_routes_never_raises(tmp_path, reply):
    board, prefs = context.server_parts(_FakeClient(board=reply), None, _settings(tmp_path))
    assert isinstance(board, str) and isinstance(prefs, str)


@pytest.mark.parametrize(
    ("flags", "asked", "expected"),
    [
        ({"SYNAPSE_BOARD": "0"}, ("preferences",), ("", _PREFS_TEXT)),
        ({"SYNAPSE_PREFS_BLOCK": "0"}, ("board",), ("BOARD", "")),
        ({"SYNAPSE_BOARD": "0", "SYNAPSE_PREFS_BLOCK": "0"}, None, ("", "")),
    ],
    ids=["no-board", "no-preferences", "neither"],
)
def test_a_piece_switched_off_is_not_asked_for(tmp_path, flags, asked, expected):
    client = _FakeClient(board=_reply(preferences=_PREFS_TEXT), prefs=_PREFS)
    s = _settings(tmp_path, env=flags)
    assert context.server_parts(client, "p", s) == expected
    # The project is the board's: it is not sent when the board is not asked for.
    project = "p" if asked and "board" in asked else None
    assert client.calls == ([("context", project, asked)] if asked else [])
    block = context.session_block(client, "p", s)
    assert block == context.assemble(expected)


@pytest.mark.parametrize(
    ("reply", "enrolled", "expected"),
    [
        (401, False, "NOT ENROLLED"),
        ({"status": "ok", "text": "B", "trust": "restricted"}, False, "RESTRICTED\n\nB"),
        (401, True, ""),
        ({"status": "ok", "text": "B", "trust": "restricted"}, True, "B"),
        ({"status": "ok", "text": "B", "trust": "full"}, False, "B"),
    ],
    ids=["401", "restricted", "401-enrolled", "restricted-enrolled", "full"],
)
def test_a_harness_may_word_the_explainers_itself(tmp_path, reply, enrolled, expected):
    """``not_enrolled`` and ``restricted`` replace the explainers rendered from the hint,
    used exactly as given, under the same conditions; the hint is then not used."""
    board = _http_error(reply) if isinstance(reply, int) else reply
    settings = _settings(tmp_path, enrolled=enrolled)
    words = {"not_enrolled": "NOT ENROLLED", "restricted": "RESTRICTED"}
    got = context.server_parts(_FakeClient(board=board, prefs=[]), None, settings, **words)
    assert got == (expected, "")
    hinted = context.server_parts(
        _FakeClient(board=board, prefs=[]), None, settings, login_hint="x", **words
    )
    assert hinted == got


def test_an_explainer_is_part_of_the_board(tmp_path):
    """With the board switched off, a machine is not told why it is served none."""
    settings = _settings(tmp_path, env={"SYNAPSE_BOARD": "0"})
    for board in (_http_error(401), _reply(trust="restricted", preferences="")):
        assert context.server_parts(_FakeClient(board=board), None, settings) == ("", "")


# ---------------------------------------------------------------------------
# session_block: order, switches, fail-open
# ---------------------------------------------------------------------------


def _noted_settings(tmp_path: Path, **env: str) -> Settings:
    """Enrolled, with a different env token: the credential notice applies."""
    home = tmp_path / "noted"
    home.mkdir()
    device = "tkDEVICEabcdef1234"
    record = {
        "surface_id": "dev-9",
        "server": "https://s.example",
        "token_sha256": hashlib.sha256(device.encode()).hexdigest(),
    }
    (home / "config.json").write_text(
        json.dumps({"url": "https://s.example", "token": device, "device": record})
    )
    return Settings.load(env={"SYNAPSE_INGEST_TOKEN": "tkSTALEzyxw98765", **env}, home=home)


def test_session_block_order_is_notice_warning_board_preferences(tmp_path):
    s = _noted_settings(tmp_path)
    client = _FakeClient(board=_reply(preferences=_PREFS_TEXT))
    block = context.session_block(
        client, "p", s, upload_warning="UPLOADS FAILING", env_location="~/.x/.env"
    )
    parts = block.split("\n\n")
    assert parts[0].startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")
    assert parts[0].endswith("Remove it from ~/.x/.env.")
    assert parts[1:] == ["UPLOADS FAILING", "BOARD", _PREFS_TEXT]
    assert "tkSTALE" not in block and "tkDEVICE" not in block
    assert client.calls == [("context", "p", _BOTH)]


def test_the_pieces_compose_into_the_block(tmp_path):
    s = _noted_settings(tmp_path)
    client = _FakeClient(board=_reply(preferences=_PREFS_TEXT))
    pieces = [context.notice_part(s), "W", *context.server_parts(client, "p", s)]
    assert context.assemble(pieces) == context.session_block(client, "p", s, upload_warning="W")


def test_notice_part_passes_the_harness_wording_through(tmp_path):
    s = _noted_settings(tmp_path)
    own = {"env": "[Synapse] env token ignored for {surface}."}
    assert context.notice_part(s, source_lines=own) == "[Synapse] env token ignored for dev-9."
    assert context.notice_part(s, source_lines={"env": ""}) == ""
    assert context.notice_part(s, env_location="~/a b/.env").endswith("Remove it from ~/a b/.env.")


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("refused"), TimeoutError(), ValueError("bad json"), AttributeError()],
)
def test_each_piece_fails_open(tmp_path, failure):
    s = _noted_settings(tmp_path)
    client = _FakeClient(board=failure, prefs=failure)
    block = context.session_block(client, "p", s, upload_warning="W")
    parts = block.split("\n\n")
    assert len(parts) == 2 and parts[1] == "W"  # the local lines survive
    assert client.calls == [("context", "p", _BOTH)]  # no answer: no second request


def test_an_unenrolled_401_explains_itself_with_the_harness_hint(tmp_path):
    client = _FakeClient(board=_http_error(401), prefs=_http_error(401))
    block = context.session_block(client, None, _settings(tmp_path), login_hint="run `hermes x`")
    assert block == context.not_enrolled_block("run `hermes x`")
    assert client.calls == [("context", None, _BOTH)]


def test_nothing_to_say_is_empty(tmp_path):
    client = _FakeClient(board=_reply("", preferences=""))
    assert context.session_block(client, None, _settings(tmp_path)) == ""


# ---------------------------------------------------------------------------
# is_private == ingest_hook._is_private, minus the path escape
# ---------------------------------------------------------------------------


def _marker(private_dir: Path, sid: str, age_hours: float) -> Path:
    private_dir.mkdir(parents=True, exist_ok=True)
    path = private_dir / sid
    path.write_text("")
    then = time.time() - age_hours * 3600
    os.utime(path, (then, then))
    return path


_PRIVATE_CASES = {
    "no-marker": None,
    "fresh-marker": 1.0,
    "stale-marker": 13.0,
    "empty-session-id": "",
    "unreadable-dir": "file",
}


@pytest.mark.parametrize("case", sorted(_PRIVATE_CASES))
def test_is_private_matches_the_hook(scratch, tmp_path, case):
    setup = _PRIVATE_CASES[case]
    private_dir = tmp_path / "private"
    sid = "" if setup == "" else "sess-1"

    def arrange() -> None:
        if private_dir.is_dir():
            for p in private_dir.iterdir():
                p.unlink()
            private_dir.rmdir()
        elif private_dir.exists():
            private_dir.unlink()
        if isinstance(setup, float):
            _marker(private_dir, sid, setup)
        elif setup == "file":
            private_dir.write_text("not a directory")

    hook = load_script("ingest_hook")
    assert str(hook.PRIVATE_DIR) == str(private_dir)
    arrange()
    expected = hook._is_private(sid)
    expected_left = private_dir.is_dir() and (private_dir / sid).exists() if sid else False
    arrange()
    lines: list[str] = []
    s = Settings.load()
    assert s.private_dir == private_dir
    assert is_private(s, sid, lines.append) is expected
    assert (
        private_dir.is_dir() and (private_dir / sid).exists() if sid else False
    ) == expected_left
    if case == "stale-marker":
        assert lines and lines[0].startswith("PRIVATE stale marker removed sess-1 (age ")
    if case == "unreadable-dir":
        assert lines == ["PRIVATE unreadable sess-1: NotADirectoryError — skipping ingestion"]


def test_the_ttl_comes_from_the_environment(tmp_path):
    private_dir = tmp_path / "p"
    _marker(private_dir, "s", 2.0)
    env = {"SYNAPSE_PRIVATE_DIR": str(private_dir), "SYNAPSE_PRIVATE_TTL_HOURS": "3"}
    assert is_private(Settings.load(env=env, home=tmp_path / "h"), "s") is True
    env["SYNAPSE_PRIVATE_TTL_HOURS"] = "1"
    assert is_private(Settings.load(env=env, home=tmp_path / "h"), "s") is False
    assert not (private_dir / "s").exists()
    _marker(private_dir, "s", 2.0)
    env["SYNAPSE_PRIVATE_TTL_HOURS"] = "soon"  # unparseable: the 12h default
    assert is_private(Settings.load(env=env, home=tmp_path / "h"), "s") is True


@pytest.mark.parametrize("sid", ["../victim", "sub/../../victim", "VICTIM_ABS", ".", ".."])
def test_a_session_id_that_is_a_path_touches_nothing(tmp_path, sid):
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    old = time.time() - 48 * 3600
    os.utime(victim, (old, old))
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    sid = str(victim) if sid == "VICTIM_ABS" else sid
    s = Settings.load(env={"SYNAPSE_PRIVATE_DIR": str(private_dir)}, home=tmp_path / "h")
    lines: list[str] = []
    assert is_private(s, sid, lines.append) is False
    assert victim.read_text() == "keep me"
    assert lines and lines[0].startswith("PRIVATE ignored a session id")


def test_a_raising_log_does_not_escape(tmp_path):
    private_dir = tmp_path / "p"
    _marker(private_dir, "s", 20.0)

    def bad_log(line: str) -> None:
        raise RuntimeError(line)

    s = Settings.load(env={"SYNAPSE_PRIVATE_DIR": str(private_dir)}, home=tmp_path / "h")
    assert is_private(s, "s", bad_log) is False
