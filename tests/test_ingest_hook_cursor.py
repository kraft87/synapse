"""The Stop hook ships from a durable per-file cursor; SessionStart sweeps gaps.

``_plan_from_cursor`` must (a) split POSTs only at turn boundaries so a chunk
never ends mid-turn (that would mint a bogus span_id for a turn a later chunk
completes), (b) park the cursor at the START of the last turn — never EOF — so
a turn still growing at Stop time is re-shipped and lands as a span_id no-op
once complete, and (c) hold the cursor still when the window has no boundary.

The cursor store must advance monotonically (concurrent Stop/sweep children
can't rewind each other), reset on truncation, and only record the
everything-shipped ``size`` on the final chunk so a crash mid-backlog can't
fake the skip condition. ``_catchup_candidates`` picks exactly the transcripts
a dead session or an unreachable server left behind.

Cursor logic lives only in the plugin hook (``plugin/scripts/ingest_hook.py``);
the repo-root hook keeps the legacy tail behavior. Loaded by path like
test_ingest_hook_tail.py — the hook runs under the CLI's bare Python.
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
    monkeypatch.setenv("SYNAPSE_INGEST_LOG", str(tmp_path / "hook.log"))
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
    assert str(path) not in hook._load_state()  # nothing advanced
    posts, shipped = hook._ship(str(path), mode="catchup")  # the retry ships it all
    assert (posts, shipped) == (1, 6)


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
