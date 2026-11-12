"""The core's session-start block, per-prompt nudge and private-mode check
(plugin/synapse_client/context.py, private.py).

The wording is the hooks' wording, byte for byte: each renderer is compared with the
hook it ports (recall_nudge.py, preferences_block.py, board_block.py, enroll.py and
ingest_hook._is_private), loaded by path over a scratch home with its server call
replaced. The one intended difference, the Claude-only paste hint in the restricted
block, is spelled out where it is compared.
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
    value instead of returning it."""

    def __init__(self, board: Any = None, prefs: Any = None) -> None:
        self.board, self.prefs = board, prefs
        self.calls: list[tuple[str, Any]] = []

    def context(self, project=None, timeout=10):
        self.calls.append(("context", project))
        if isinstance(self.board, BaseException):
            raise self.board
        return self.board

    def preferences(self, limit=8, timeout=10):
        self.calls.append(("preferences", limit))
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
# preferences_block == preferences_block.py
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


@pytest.mark.parametrize("case", sorted(_PREF_CASES))
def test_preferences_block_is_byte_identical_to_the_hook(scratch, monkeypatch, capsys, case):
    items = _PREF_CASES[case]
    hook = load_script("preferences_block")
    seen = fake_urlopen(monkeypatch, {"items": items})
    hook.main()
    assert context.preferences_block(items) == _hook_context(capsys)
    assert [requested(req) for req in seen] == [("/preferences/top", {"limit": "8"})]


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
# board_parts == what board_block.py prints
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


@pytest.mark.parametrize("case", sorted(_BOARD_CASES))
def test_board_parts_match_the_hook(scratch, monkeypatch, capsys, tmp_path, case):
    reply, enrolled = _BOARD_CASES[case]
    answer = _http_error(reply) if isinstance(reply, int) else reply
    if enrolled:  # the hook reads enrollment from the device record, as it does at start
        (scratch / ".synapse").mkdir(parents=True, exist_ok=True)
        record = {"device": {"surface_id": "dev-1"}}
        (scratch / ".synapse" / "config.json").write_text(json.dumps(record))

    hook = load_script("board_block")
    seen = fake_urlopen(monkeypatch, answer)
    monkeypatch.setattr(hook, "_project_label", lambda: "proj")
    hook.main()
    assert [requested(req) for req in seen] == [("/context", {"project": "proj"})]
    # The hook prints the parts joined by blank lines (no credential notice applies here).
    expected = _hook_context(capsys).replace(_PASTE_HINT, "")

    client = _FakeClient(board=answer)
    settings = _settings(tmp_path, enrolled=enrolled)
    got = context.board_parts(client, "proj", settings, login_hint=_CLAUDE_HINT)
    # The hook's restricted block names its own hint; compare the rest of the wording.
    got = [
        p.replace(f"configured: {_CLAUDE_HINT}.", f"configured: {_CLAUDE_RESTRICTED_HINT}.")
        for p in got
    ]
    assert "\n\n".join(got) == expected
    assert client.calls == [("context", "proj")]


def test_board_parts_raise_what_is_not_an_http_status(tmp_path):
    client = _FakeClient(board=urllib.error.URLError("refused"))
    with pytest.raises(urllib.error.URLError):
        context.board_parts(client, None, _settings(tmp_path))


# ---------------------------------------------------------------------------
# session_block: order, switches, fail-open
# ---------------------------------------------------------------------------

_PREFS = [{"pref": "Terse", "polarity": "like"}]


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
    client = _FakeClient(board={"status": "ok", "text": "BOARD"}, prefs=_PREFS)
    block = context.session_block(
        client, "p", s, upload_warning="UPLOADS FAILING", env_location="~/.x/.env"
    )
    parts = block.split("\n\n")
    assert parts[0].startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")
    assert parts[0].endswith("Remove it from ~/.x/.env.")
    assert parts[1:] == ["UPLOADS FAILING", "BOARD", "[Synapse preferences]\n  - (likes) Terse"]
    assert "tkSTALE" not in block and "tkDEVICE" not in block


def test_the_pieces_compose_into_the_block(tmp_path):
    s = _noted_settings(tmp_path)
    client = _FakeClient(board={"status": "ok", "text": "BOARD"}, prefs=_PREFS)
    pieces = [
        context.notice_part(s),
        "W",
        context.board_part(client, "p", s),
        context.preferences_part(client, s),
    ]
    assert context.assemble(pieces) == context.session_block(client, "p", s, upload_warning="W")


def test_notice_part_passes_the_harness_wording_through(tmp_path):
    s = _noted_settings(tmp_path)
    own = {"env": "[Synapse] env token ignored for {surface}."}
    assert context.notice_part(s, source_lines=own) == "[Synapse] env token ignored for dev-9."
    assert context.notice_part(s, source_lines={"env": ""}) == ""
    assert context.notice_part(s, env_location="~/a b/.env").endswith("Remove it from ~/a b/.env.")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"SYNAPSE_BOARD": "0"}, ["[Synapse preferences]\n  - (likes) Terse"]),
        ({"SYNAPSE_PREFS_BLOCK": "0"}, ["BOARD"]),
        ({"SYNAPSE_BOARD": "0", "SYNAPSE_PREFS_BLOCK": "0"}, []),
    ],
)
def test_session_block_honours_the_switches(tmp_path, flags, expected):
    client = _FakeClient(board={"status": "ok", "text": "BOARD"}, prefs=_PREFS)
    s = _settings(tmp_path, env=flags)
    block = context.session_block(client, None, s)
    assert (block.split("\n\n") if block else []) == expected
    called = {name for name, _ in client.calls}
    assert ("context" in called) == ("SYNAPSE_BOARD" not in flags)
    assert ("preferences" in called) == ("SYNAPSE_PREFS_BLOCK" not in flags)


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


def test_a_malformed_preference_item_fails_open(tmp_path):
    client = _FakeClient(
        board={"status": "ok", "text": "B"}, prefs=[{"pref": "x", "polarity": ["?"]}]
    )
    assert context.session_block(client, None, _settings(tmp_path)) == "B"


def test_an_unenrolled_401_explains_itself_with_the_harness_hint(tmp_path):
    client = _FakeClient(board=_http_error(401), prefs=_http_error(401))
    block = context.session_block(client, None, _settings(tmp_path), login_hint="run `hermes x`")
    assert block == context.not_enrolled_block("run `hermes x`")


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
    got = context.board_part(_FakeClient(board=board), None, settings, login_hint="x", **words)
    assert got == expected
    parts = context.board_parts(_FakeClient(board=board), None, settings, **words)
    assert "\n\n".join(parts) == expected


def test_nothing_to_say_is_empty(tmp_path):
    client = _FakeClient(board={"status": "ok", "text": ""}, prefs=[])
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
