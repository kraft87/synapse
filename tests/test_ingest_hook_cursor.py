"""The Stop hook ships from a durable per-file cursor; SessionStart sweeps gaps.

``_plan_from_cursor`` must (a) split POSTs only at turn boundaries so a chunk
never ends mid-turn (that would mint a bogus span_id for a turn a later chunk
completes), (b) park the cursor at the START of the last turn — never EOF — so
a turn still growing at Stop time is re-shipped once complete, and (c) hold the
cursor still when the window has no boundary.

The cursor store must advance monotonically (concurrent Stop/sweep children
can't rewind each other), reset on truncation, and only record the
everything-shipped ``size`` on the final chunk so a crash mid-backlog can't
fake the skip condition. ``_catchup_candidates`` picks exactly the transcripts
a dead session or an unreachable server left behind.

Cursor logic lives in the plugin hook (``plugin/scripts/ingest_hook.py``), loaded by
path like test_ingest_hook_tail.py — the hook runs under the CLI's bare Python.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN_HOOK = _REPO / "plugin" / "scripts" / "ingest_hook.py"

_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


@pytest.fixture()
def hook(monkeypatch, tmp_path) -> ModuleType:
    """A fresh plugin-hook module whose config/state all live under tmp_path."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location("ingest_hook_cursor_test", _PLUGIN_HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


def _u(uid: str, text: str = "q") -> dict:
    return {"type": "user", "uuid": uid, "sessionId": "s1", "message": {"content": text}}


def _a(uid: str, text: str = "r") -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "sessionId": "s1",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _turns3() -> list[dict]:
    # 3 turns, 6 records: u1 a1 | u2 a2 | u3 a3
    return [_u("u1"), _a("a1"), _u("u2"), _a("a2"), _u("u3"), _a("a3")]


def _write_jsonl(path: Path, recs: list[dict]) -> list[int]:
    """Write records one per line; return each line's byte offset."""
    offsets, pos = [], 0
    with open(path, "wb") as f:
        for r in recs:
            offsets.append(pos)
            line = json.dumps(r).encode() + b"\n"
            f.write(line)
            pos += len(line)
    return offsets


def _offset_recs(recs: list[dict]) -> list[tuple[int, dict]]:
    """Synthetic (offset, record) pairs: 100 bytes apart, offsets stable."""
    return [(i * 100, r) for i, r in enumerate(recs)]


# ---------------------------------------------------------------------------
# _plan_from_cursor (pure)
# ---------------------------------------------------------------------------


def test_plan_from_zero_ships_all_parks_cursor_at_last_turn(hook):
    plans = hook._plan_from_cursor(_offset_recs(_turns3()), 0, chunk_records=400)
    assert len(plans) == 1
    records, cursor_after = plans[0]
    assert [r["uuid"] for r in records] == ["u1", "a1", "u2", "a2", "u3", "a3"]
    assert cursor_after == 400  # offset of u3, the last turn-start — not EOF


def test_plan_resume_reships_only_from_cursor(hook):
    # cursor parked at u3 (offset 400) → only the last turn ships again
    plans = hook._plan_from_cursor(_offset_recs(_turns3()), 400, chunk_records=400)
    assert len(plans) == 1
    records, cursor_after = plans[0]
    assert [r["uuid"] for r in records] == ["u3", "a3"]
    assert cursor_after == 400


def test_plan_nothing_past_cursor(hook):
    assert hook._plan_from_cursor(_offset_recs(_turns3()), 10_000, 400) == []


def test_plan_chunks_split_at_turn_boundaries(hook):
    # chunk budget of 3 records: turns are 2 records each, so chunks hold one
    # turn... except chunk 1, which also glues nothing (no prefix here).
    plans = hook._plan_from_cursor(_offset_recs(_turns3()), 0, chunk_records=3)
    assert [[r["uuid"] for r in recs] for recs, _ in plans] == [
        ["u1", "a1"],
        ["u2", "a2"],
        ["u3", "a3"],
    ]
    # intermediate cursors land on the NEXT turn's start; final on the last turn
    assert [c for _, c in plans] == [200, 400, 400]


def test_plan_glues_leading_prefix_onto_first_turn(hook):
    recs = [{"type": "summary", "uuid": "meta"}, *_turns3()]
    plans = hook._plan_from_cursor(_offset_recs(recs), 0, chunk_records=400)
    assert plans[0][0][0]["uuid"] == "meta"


def test_plan_no_boundary_keeps_cursor(hook):
    # only assistant records past the cursor → ship them, cursor stays put
    plans = hook._plan_from_cursor(_offset_recs([_a("a1"), _a("a2")]), 0, 400)
    assert len(plans) == 1
    records, cursor_after = plans[0]
    assert [r["uuid"] for r in records] == ["a1", "a2"]
    assert cursor_after == 0


def test_oversized_single_turn_ships_as_one_chunk(hook):
    recs = [_u("u1"), *(_a(f"a{i}") for i in range(10))]
    plans = hook._plan_from_cursor(_offset_recs(recs), 0, chunk_records=3)
    assert len(plans) == 1
    assert len(plans[0][0]) == 11


# ---------------------------------------------------------------------------
# Cursor store
# ---------------------------------------------------------------------------


def test_cursor_monotonic_unless_forced(hook, tmp_path):
    p = str(tmp_path / "t.jsonl")
    Path(p).touch()
    hook._advance_cursor(p, 500, 900)
    hook._advance_cursor(p, 300, 900)  # a stale racer must not rewind
    assert hook._load_state()[p]["offset"] == 500
    hook._advance_cursor(p, 0, 100, force=True)  # truncation reset may
    assert hook._load_state()[p]["offset"] == 0


def test_cursor_prunes_deleted_files(hook, tmp_path):
    gone = str(tmp_path / "gone.jsonl")
    kept = str(tmp_path / "kept.jsonl")
    Path(gone).touch()
    Path(kept).touch()
    hook._advance_cursor(gone, 10, 20)
    os.unlink(gone)
    hook._advance_cursor(kept, 10, 20)  # any write sweeps dead entries
    state = hook._load_state()
    assert kept in state and gone not in state


# ---------------------------------------------------------------------------
# _ship end-to-end against a fake POST
# ---------------------------------------------------------------------------


def test_ship_advances_cursor_and_skips_when_unchanged(hook, tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    offsets = _write_jsonl(path, _turns3())
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )

    posts, shipped = hook._ship(str(path), mode="catchup")
    assert (posts, shipped) == (1, 6)
    ent = hook._load_state()[str(path)]
    assert ent["offset"] == offsets[4]  # start of the u3 turn
    assert ent["size"] == path.stat().st_size

    # unchanged file → skip without a POST
    assert hook._ship(str(path), mode="catchup") == (0, 0)
    assert len(posted) == 1


def test_ship_resumes_after_failed_post(hook, tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, _turns3())
    calls = {"n": 0}

    def flaky(recs, source="hook"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("server unreachable")
        return "ok"

    monkeypatch.setattr(hook, "_post_records", flaky)
    assert hook._ship(str(path), mode="catchup") == (0, 0)
    ent = hook._load_state()[str(path)]
    assert ent["offset"] == 0 and ent["size"] == -1  # nothing advanced...
    assert ent["pending"]  # ...but the failure is on record, so catch-up retries it
    posts, shipped = hook._ship(str(path), mode="catchup")  # the retry ships it all
    assert (posts, shipped) == (1, 6)
    assert "pending" not in hook._load_state()[str(path)]


def test_ship_crash_mid_backlog_does_not_fake_completion(hook, tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, _turns3())
    monkeypatch.setattr(hook, "TAIL_RECORDS", 3)  # 3 chunks of one turn each
    calls = {"n": 0}

    def two_then_die(recs, source="hook"):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("died on the final chunk")
        return "ok"

    monkeypatch.setattr(hook, "_post_records", two_then_die)
    posts, _ = hook._ship(str(path), mode="catchup")
    assert posts == 2
    ent = hook._load_state()[str(path)]
    assert ent["size"] == -1  # intermediate marker: NOT everything-shipped
    # so the next run must not skip — it resumes and ships the last turn
    posts, shipped = hook._ship(str(path), mode="catchup")
    assert (posts, shipped) == (1, 2)
    assert hook._load_state()[str(path)]["size"] == path.stat().st_size


def test_ship_truncated_file_resets_and_reships(hook, tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, _turns3())
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": "ok")
    hook._ship(str(path), mode="catchup")
    _write_jsonl(path, _turns3()[:2])  # rewritten shorter than the cursor
    posts, shipped = hook._ship(str(path), mode="catchup")
    assert (posts, shipped) == (1, 2)
    assert hook._load_state()[str(path)]["offset"] == 0  # u1 is the only turn


def test_ship_stop_mode_seeds_cursor_from_tail(hook, tmp_path, monkeypatch):
    """First cursored run on a long pre-existing session must not re-parse the
    whole file — earlier hook fires already shipped it turn-by-turn."""
    path = tmp_path / "s.jsonl"
    offsets = _write_jsonl(path, _turns3())
    monkeypatch.setattr(hook, "TAIL_RECORDS", 3)
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )
    hook._ship(str(path), mode="stop")
    # tail window = [a2, u3, a3] → seed at u3; only the last turn ships
    assert [r["uuid"] for r in posted[0]] == ["u3", "a3"]
    assert hook._load_state()[str(path)]["offset"] == offsets[4]


def _spy_reads(hook, monkeypatch) -> list[int]:
    """Record the start offset of every transcript read `_ship` makes."""
    starts: list[int] = []
    real = hook._read_offset_records

    def spy(path, start=0):
        starts.append(start)
        return real(path, start)

    monkeypatch.setattr(hook, "_read_offset_records", spy)
    return starts


def test_ship_does_not_open_an_unchanged_transcript(hook, tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, _turns3())
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": "ok")
    hook._ship(str(path), mode="catchup")
    starts = _spy_reads(hook, monkeypatch)
    assert hook._ship(str(path), mode="stop") == (0, 0)
    assert starts == []  # the size check is a stat, not a parse of the whole file


def test_ship_reads_only_from_the_cursor(hook, tmp_path, monkeypatch):
    """Every Stop used to json.loads the whole transcript to find its few new records
    (O(turns^2) over a session). The cursor is a line start, so the read begins there."""
    path = tmp_path / "s.jsonl"
    recs = [_u("u1", "old"), _a("a1", "old"), _u("u2", "old"), _a("a2", "old"), _u("u3"), _a("a3")]
    offsets = _write_jsonl(path, recs)
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )
    hook._ship(str(path), mode="catchup")
    assert hook._load_state()[str(path)]["offset"] == offsets[4]  # cursor at u3

    with open(path, "ab") as f:
        f.write(json.dumps(_u("u4")).encode() + b"\n")
        f.write(json.dumps(_a("a4")).encode() + b"\n")
    starts = _spy_reads(hook, monkeypatch)
    posted.clear()
    assert hook._ship(str(path), mode="stop") == (1, 4)
    assert starts == [offsets[4]]
    # the last turn re-ships with the new one (the transcript may lag a Stop)
    assert [r["uuid"] for r in posted[0]] == ["u3", "a3", "u4", "a4"]


def test_ship_cursor_inside_a_line_means_the_file_was_rewritten(hook, tmp_path, monkeypatch):
    """A cursor that is within the file but not at a line start belongs to a transcript
    replaced by a longer one: reship it from the top rather than parse a line's tail."""
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, _turns3())
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )
    hook._ship(str(path), mode="catchup")
    cursor = hook._load_state()[str(path)]["offset"]
    # a different transcript, longer than the old one, with a line spanning the cursor
    fresh = [_u(f"n{i}", "x" * 60) if i % 2 == 0 else _a(f"n{i}", "y" * 60) for i in range(10)]
    offsets = _write_jsonl(path, fresh)
    assert all(o != cursor for o in offsets) and cursor < path.stat().st_size
    posted.clear()

    posts, shipped = hook._ship(str(path), mode="stop")
    assert posts == 1 and shipped == 10
    assert [r["uuid"] for r in posted[0]] == [r["uuid"] for r in fresh]
    assert hook._load_state()[str(path)]["offset"] == offsets[8]


# ---------------------------------------------------------------------------
# Catch-up candidate selection
# ---------------------------------------------------------------------------


def test_catchup_candidates_pick_only_lagging_idle_transcripts(hook, tmp_path):
    root = tmp_path / "projects"
    proj = root / "-home-user-dev"
    proj.mkdir(parents=True)
    now = time.time()

    def mk(name: str, age_s: float) -> str:
        p = proj / name
        _write_jsonl(p, _turns3())
        os.utime(p, (now - age_s, now - age_s))
        return str(p)

    lagging = mk("lagging.jsonl", age_s=3600)
    too_old = mk("too_old.jsonl", age_s=hook.CATCHUP_DAYS * 86400 + 3600)
    active = mk("active.jsonl", age_s=10)  # its own Stop hooks own it
    current = mk("current.jsonl", age_s=3600)
    shipped = mk("shipped.jsonl", age_s=3600)
    state = {shipped: {"offset": 0, "size": os.path.getsize(shipped), "ts": now}}

    got = hook._catchup_candidates(str(root), current, state, now)
    assert got == [lagging]
    assert too_old not in got and active not in got and shipped not in got


def test_catchup_orders_oldest_first(hook, tmp_path):
    root = tmp_path / "projects"
    proj = root / "p"
    proj.mkdir(parents=True)
    now = time.time()
    newer, older = proj / "newer.jsonl", proj / "older.jsonl"
    for p, age in ((newer, 3600), (older, 7200)):
        _write_jsonl(p, _turns3())
        os.utime(p, (now - age, now - age))
    got = hook._catchup_candidates(str(root), "", {}, now)
    assert got == [str(older), str(newer)]


def test_post_records_sends_a_real_user_agent(hook, monkeypatch):
    """Cloudflare 403s urllib's default ``Python-urllib/x`` UA ("error code: 1010"), and
    the hook fails soft, so a missing UA silently drops every ingest. It must send the
    same UA the rest of the plugin does."""
    seen: dict[str, str] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status":"ok"}'

    def fake_urlopen(req, timeout=None):
        seen.update({k.lower(): v for k, v in req.header_items()})
        return _Resp()

    monkeypatch.setattr(hook.urllib.request, "urlopen", fake_urlopen)
    hook._post_records([_u("u1")])
    assert seen.get("user-agent") == hook.config.USER_AGENT
    assert not seen["user-agent"].startswith("Python-urllib")


# ---------------------------------------------------------------------------
# Web capture: sub-agent transcripts and saved tool outputs go to a server that captures
# ---------------------------------------------------------------------------


class _Server:
    """urlopen, replaced: records every body and answers with ``reply``."""

    def __init__(self, hook, monkeypatch, reply: dict | str | None = None) -> None:
        self.bodies: list[dict] = []
        self.reply = {"status": "ok", "ingested": 1} if reply is None else reply
        monkeypatch.setattr(hook.urllib.request, "urlopen", self._open)

    def _open(self, req, timeout=None):
        import io

        self.bodies.append(json.loads(req.data))
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return io.BytesIO(text.encode())


def _subagent_records(sid: str) -> list[dict]:
    """A sub-agent's transcript: its prompt (the only turn start), a tool call and its result."""
    use = {"type": "tool_use", "id": "ts", "name": "mcp__acme__fetch", "input": {"url": "https://e.com"}}  # fmt: skip
    res = {"type": "tool_result", "tool_use_id": "ts", "content": "page"}
    recs = [
        _u("s1", "research this"),
        {**_a("s2"), "message": {"content": [use]}},
        {**_u("s3"), "message": {"content": [res]}},
        _a("s4", "done"),
    ]
    return [{**r, "sessionId": sid, "isSidechain": True} for r in recs]


def _session(root: Path, sid: str = "sess1") -> tuple[Path, Path]:
    """<root>/-proj/<sid>.jsonl and its sub-agent transcript beside it, as Claude Code writes
    them (the records carry the parent's sessionId)."""
    main = root / "-proj" / f"{sid}.jsonl"
    main.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(main, [{**r, "sessionId": sid} for r in (_u("u1"), _a("a1"))])
    side = main.parent / sid / "subagents" / "agent-a1.jsonl"
    side.parent.mkdir(parents=True)
    _write_jsonl(side, _subagent_records(sid))
    return main, side


def _redirect_records(saved: Path) -> list[dict]:
    use = {"type": "tool_use", "id": "t1", "name": "mcp__acme__fetch", "input": {"url": "https://e.com"}}  # fmt: skip
    redirect = f"Output too large (2KB). Full output saved to: {saved}"
    res = {"type": "tool_result", "tool_use_id": "t1", "content": redirect}
    return [
        _u("u1"),
        {**_a("a1"), "message": {"content": [use]}},
        {**_u("r1"), "message": {"content": [res]}},
    ]


def test_saved_outputs_are_read_in_only_for_a_server_that_captures(hook, tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    saved = projects / "-proj" / "s" / "tool-results" / "t1.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("the whole page " * 50)
    monkeypatch.setattr(hook.config, "PROJECTS_DIR", projects)
    server = _Server(hook, monkeypatch)

    def result_sent() -> str:
        hook._post_records(_redirect_records(saved))
        return server.bodies[-1]["records"][2]["message"]["content"][0]["content"]

    assert "saved to:" in result_sent()  # nothing known yet: the redirect goes as it is
    server.reply = {"status": "ok", "ingested": 1, "capture": True}
    assert "saved to:" in result_sent()  # still sent without the file; its reply is the news
    assert result_sent() == "the whole page " * 50  # the server said it captures
    server.reply = {"status": "ok", "ingested": 1}  # the switch was turned off, or an old server
    assert result_sent() == "the whole page " * 50  # this POST went out on the old answer
    assert "saved to:" in result_sent()  # ...and its reply ended it


def test_a_reply_that_is_not_the_servers_leaves_what_was_known(hook, monkeypatch):
    server = _Server(hook, monkeypatch, {"status": "ok", "capture": True})
    hook._post_records([_u("u1")])
    assert hook._server_captures()
    for junk in ("<html>502 Bad Gateway</html>", "[1, 2]", ""):
        server.reply = junk
        hook._post_records([_u("u1")])
        assert hook._server_captures()
    server.reply = {"status": "ok", "capture": False}
    hook._post_records([_u("u1")])
    assert not hook._server_captures()


def test_the_stop_hook_ships_the_sessions_subagent_transcripts(hook, tmp_path, monkeypatch):
    main, side = _session(tmp_path / "projects")
    server = _Server(hook, monkeypatch, {"status": "ok", "capture": True})
    hook._ship_session(str(main))
    sent = [[r["uuid"] for r in b["records"]] for b in server.bodies]
    # the transcript first (its reply says capture is on), then the sub-agent's whole file
    assert sent == [["u1", "a1"], ["s1", "s2", "s3", "s4"]]
    assert {r.get("isSidechain") for r in server.bodies[1]["records"]} == {True}
    state = hook._load_state()
    assert state[str(side)]["offset"] == 0 and state[str(side)]["size"] == side.stat().st_size
    hook._ship_session(str(main))  # nothing changed: nothing is sent again
    assert len(server.bodies) == 2


def test_a_subagent_transcript_grown_since_is_shipped_again(hook, tmp_path, monkeypatch):
    main, side = _session(tmp_path / "projects")
    server = _Server(hook, monkeypatch, {"status": "ok", "capture": True})
    hook._ship_session(str(main))
    with open(side, "ab") as f:
        f.write(json.dumps({**_a("s5", "more"), "isSidechain": True}).encode() + b"\n")
    hook._ship_session(str(main))
    assert [r["uuid"] for r in server.bodies[-1]["records"]] == ["s1", "s2", "s3", "s4", "s5"]


def test_no_subagent_transcript_goes_to_a_server_that_does_not_capture(hook, tmp_path, monkeypatch):
    main, side = _session(tmp_path / "projects")
    server = _Server(hook, monkeypatch)  # an older server: no `capture` in the reply
    hook._ship_session(str(main))
    assert len(server.bodies) == 1 and str(side) not in hook._load_state()


def test_a_private_sessions_subagent_transcripts_are_not_shipped(hook, tmp_path, monkeypatch):
    main, side = _session(tmp_path / "projects")
    monkeypatch.setattr(hook, "PRIVATE_DIR", tmp_path / "private")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "sess1").write_text("")
    hook.CAPTURE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    hook.CAPTURE_MARKER.touch()  # the server captures...
    server = _Server(hook, monkeypatch, {"status": "ok", "capture": True})
    hook._ship_session(str(main))
    assert server.bodies == []  # ...but neither file goes: the parent session is private
    assert hook._ship(str(side), mode="catchup") == (0, 0) and server.bodies == []


def test_the_sweep_takes_subagent_transcripts_only_for_a_server_that_captures(hook, tmp_path):
    root = tmp_path / "projects"
    main, side = _session(root)
    old = time.time() - 3600
    for p in (main, side):
        os.utime(p, (old, old))
    assert hook._catchup_candidates(str(root), "", {}, time.time()) == [str(main)]
    got = hook._catchup_candidates(str(root), "", {}, time.time(), subagents=True)
    assert sorted(got) == sorted([str(main), str(side)])


def test_the_sweep_ships_a_subagent_transcript_from_byte_zero(hook, tmp_path, monkeypatch):
    root = tmp_path / "projects"
    main, side = _session(root)
    old = time.time() - 3600
    for p in (main, side):
        os.utime(p, (old, old))
    server = _Server(hook, monkeypatch, {"status": "ok", "capture": True})
    hook.CAPTURE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    hook.CAPTURE_MARKER.touch()
    hook._catchup(str(root), skip_path="")
    assert len(server.bodies) == 2
    assert any(r.get("isSidechain") for b in server.bodies for r in b["records"])
