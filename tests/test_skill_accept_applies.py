"""Accept applies a skill proposal — DB-backed tests (migration 055 + skill_sync_routes).

The review accept writes the proposal into skills_lane.skill_registry through the same
upsert /skills/publish uses and marks the candidate 'promoted' in one transaction; the
client's two-way sync then delivers it. Runs against the shared Postgres test DB (tagged
into the `db` xdist group via conftest). Every skill, project and session is synthetic
(acc- prefix) and cleaned up around each test.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from psycopg.types.json import Json

import dream.skills.skill_derive as SD
import dream.skills.skill_draft as DR
import mcp_server.skill_sync_routes as SSR
from dream.skills.skill_doc import body_hash
from ingestion import pg_pool
from mcp_server.skill_sync_routes import (
    _fetch_skill,
    _list_skills,
    _proposal_act,
    _proposal_detail,
    _publish_skill,
    _record_overwrite,
    _routing_eval,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "plugin" / "scripts"
_OLD_TIME = "2026-01-01T00:00:00+00:00"


def _skill(name: str, desc: str, step: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\n1. {step}\n"


V1 = _skill("acc-backup", "Back up the demo database.", "Dump the database.")
V2 = _skill("acc-backup", "Back up the demo database.", "Dump the database, then verify it.")
RETUNED = _skill("acc-backup", "Back up, dump or copy the demo database.", "Dump the database.")


@pytest.fixture()
def acc(conn):
    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'acc-%%'")
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'acc-%%'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE 'acc-%%'")
        conn.execute("DELETE FROM episodes WHERE session_id LIKE 'acc-%%'")

    _clean()
    yield conn
    _clean()


def _publish(db_url, name, body, scope="global", when=_OLD_TIME, description=""):
    _publish_skill(
        db_url,
        {
            "name": name,
            "scope": scope,
            "body": body,
            "description": description,
            "content_modified_at": when,
            "files": [],
        },
    )


def _candidate(
    conn,
    *,
    kind="retune",
    name="acc-backup",
    direction="fix",
    target=("acc-backup",),
    body=None,
    base=None,
    evidence=None,
    status="proposed",
):
    ev = evidence if evidence is not None else [{"session_id": "acc-s1", "class": "judge"}]
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates "
        "(kind, name, direction, target_skills, status, summary, evidence, proposal_body, "
        " base_body_hash, judge_sessions, judge_weight) "
        "VALUES (%s,%s,%s,%s,%s,'synthetic',%s,%s,%s,1,1.0) RETURNING id",
        (kind, name, direction, list(target), status, Json(ev), body, base),
    ).fetchone()[0]


def _episode(conn, session, project, cwd):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, platform, content, metadata) "
        "VALUES (%s, 1, %s, 'claude_code', 'synthetic', %s)",
        (session, project, Json({"cwd": cwd})),
    )


def _accept(db_url, cid, **kw):
    return _proposal_act(db_url, cid, "accept", None, lambda *_: "routing-eval: stub", **kw)


def _registry(conn, name):
    return conn.execute(
        "SELECT body, scope, description, content_modified_at, status "
        "FROM skills_lane.skill_registry WHERE name=%s",
        (name,),
    ).fetchone()


def _cand(conn, cid):
    return conn.execute(
        "SELECT status, grounded_weight, evidence, proposal_body "
        "FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (cid,),
    ).fetchone()


# ------------------------------------------------------------------ accept applies


def test_accept_writes_registry_records_history_and_promotes(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))

    r = _accept(db_url, cid)

    assert r["status"] == "promoted" and r["applied"] is True
    assert (r["skill"], r["scope"], r["created"]) == ("acc-backup", "global", False)
    assert r["routing_eval"] == "routing-eval: stub"
    body, scope, desc, cmod, status = _registry(acc, "acc-backup")
    assert body == V2 and scope == "global" and status == "active"
    assert desc == "Back up the demo database."  # parsed from frontmatter, like a publish
    assert cmod > datetime.fromisoformat(_OLD_TIME)  # now(): newer than every disk copy
    hist = acc.execute(
        "SELECT body, op FROM skills_lane.skill_history WHERE name='acc-backup'"
    ).fetchall()
    assert hist == [(V1, "superseded")]  # the trigger kept the old body recoverable
    status, gw, ev, applied = _cand(acc, cid)
    assert status == "promoted" and gw > 0 and applied == V2
    assert ev[-1] == {"session_id": None, "class": "grounded", "signal": "accept"}


def test_stale_base_hash_is_refused_and_force_overrides(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    edited = V1.replace("Dump the database.", "Dump the database with compression.")
    _publish(db_url, "acc-backup", edited)  # the owner edited the skill after drafting

    r = _accept(db_url, cid)
    assert r["status"] == "refused" and r["reason"] == "stale"
    assert "changed since this proposal was drafted" in r["detail"]
    assert _registry(acc, "acc-backup")[0] == edited  # nothing written
    assert _cand(acc, cid)[0] == "proposed"

    r = _accept(db_url, cid, force=True)
    assert r["status"] == "promoted" and r["forced"] == ["stale"]
    assert _registry(acc, "acc-backup")[0] == V2


def test_body_file_overrides_the_draft(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    mine = _skill("acc-backup", "Back up or snapshot the demo database.", "Snapshot it.")

    r = _accept(db_url, cid, body=mine)

    assert r["status"] == "promoted"
    assert _registry(acc, "acc-backup")[0] == mine
    assert _cand(acc, cid)[3] == mine  # the row records what was actually applied


def test_retune_keeps_its_scope_and_scope_flag_overrides(acc, db_url):
    _publish(db_url, "acc-backup", V1, scope="project:acc-proj")
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    assert _accept(db_url, cid)["scope"] == "project:acc-proj"
    assert _registry(acc, "acc-backup")[1] == "project:acc-proj"

    _publish(db_url, "acc-other", V1.replace("acc-backup", "acc-other"), scope="project:acc-proj")
    v2 = V2.replace("acc-backup", "acc-other")
    cid2 = _candidate(acc, name="acc-other", target=("acc-other",), body=v2)
    assert _accept(db_url, cid2, scope="global")["scope"] == "global"
    assert _registry(acc, "acc-other")[1] == "global"


_NEW = _skill("acc-new-skill", "Rotate the demo service logs.", "Rotate the logs.")


@pytest.mark.parametrize(
    "sessions,expected",
    [
        (
            [("acc-d1", "acc-proj", "/work/acc-proj"), ("acc-d2", "acc-proj", "/work/acc-proj")],
            "project:acc-proj",
        ),
        (
            [("acc-d1", "acc-proj", "/work/acc-proj"), ("acc-d2", "acc-other", "/work/acc-other")],
            "global",
        ),
        # a session started in a home dir: the client never syncs project:<home>
        ([("acc-d1", "someone", "/home/someone")], "global"),
        ([], "global"),  # no session evidence at all
    ],
)
def test_derive_scope_follows_its_evidence(acc, db_url, sessions, expected):
    for sid, project, cwd in sessions:
        _episode(acc, sid, project, cwd)
    ev = [{"session_id": sid, "class": "judge"} for sid, _, _ in sessions]
    cid = _candidate(
        acc, kind="derive", name="acc-new", direction=None, target=(), body=_NEW, evidence=ev
    )
    r = _accept(db_url, cid)
    assert r["status"] == "promoted" and r["created"] is True
    assert (r["skill"], r["scope"]) == ("acc-new-skill", expected)  # name from frontmatter
    assert _registry(acc, "acc-new-skill")[:2] == (_NEW, expected)


def test_derive_scope_flag_overrides_the_evidence(acc, db_url):
    _episode(acc, "acc-d1", "acc-proj", "/work/acc-proj")
    cid = _candidate(
        acc, kind="derive", name="acc-new", direction=None, target=(), body=_NEW,
        evidence=[{"session_id": "acc-d1", "class": "judge"}],
    )  # fmt: skip
    assert _accept(db_url, cid, scope="global")["scope"] == "global"


def test_derive_onto_a_taken_name_is_refused_unless_forced(acc, db_url):
    _publish(db_url, "acc-new-skill", V1.replace("acc-backup", "acc-new-skill"), scope="global")
    cid = _candidate(acc, kind="derive", name="acc-new", direction=None, target=(), body=_NEW)
    r = _accept(db_url, cid)
    assert r["status"] == "refused" and r["reason"] == "exists"
    r = _accept(db_url, cid, force=True)
    assert r["status"] == "promoted" and r["scope"] == "global"  # keeps the existing scope


def test_invalid_scope_and_renaming_retune_are_refused(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    assert _accept(db_url, cid, scope="project:")["reason"] == "bad_scope"
    renamed = V2.replace("name: acc-backup", "name: acc-renamed")
    r = _accept(db_url, cid, body=renamed, force=True)  # --force never overrides a rename
    assert r["status"] == "refused" and r["reason"] == "renamed"
    assert _registry(acc, "acc-backup")[0] == V1


def test_retune_of_a_registry_row_without_a_body_takes_a_body_file(acc, db_url):
    acc.execute(
        "INSERT INTO skills_lane.skill_registry (name, description) "
        "VALUES ('acc-backup', 'legacy row, no body')"
    )
    cid = _candidate(acc, body=None)
    assert _accept(db_url, cid)["reason"] == "no_draft"
    other = V2.replace("name: acc-backup", "name: acc-other")
    assert _accept(db_url, cid, body=other)["reason"] == "renamed"
    assert _accept(db_url, cid, body=V2)["status"] == "promoted"
    assert _registry(acc, "acc-backup")[0] == V2


def test_legacy_row_without_a_body_is_refused_cleanly(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, direction="widen", body=None)

    r = _accept(db_url, cid)

    assert r["status"] == "refused" and r["reason"] == "no_draft"
    assert r["detail"].startswith("no draft yet; re-drafts on the next nightly run")
    assert "routing_eval" not in r  # no advisory LLM call for a guaranteed refusal
    assert _registry(acc, "acc-backup")[0] == V1 and _cand(acc, cid)[0] == "proposed"
    # the reviewer can still apply it by supplying the body
    assert _accept(db_url, cid, body=V2)["status"] == "promoted"


def test_promote_is_a_noop(acc, db_url):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    r = _proposal_act(db_url, cid, "promote", None, None)
    assert r["status"] == "refused" and "no separate promote step" in r["detail"]

    _accept(db_url, cid)
    r = _proposal_act(db_url, cid, "promote", None, None)
    assert r["status"] == "promoted" and r["noop"] is True
    assert "applied at accept" in r["detail"]
    assert _registry(acc, "acc-backup")[0] == V2

    again = _accept(db_url, cid)
    assert again["status"] == "refused" and again["reason"] == "already_applied"


def test_consolidate_accept_still_only_records(acc, db_url):
    cid = _candidate(
        acc, kind="consolidate", name="acc-a+acc-b", direction=None, target=("acc-a", "acc-b")
    )
    assert _accept(db_url, cid)["status"] == "accepted"
    assert _proposal_act(db_url, cid, "promote", None, None)["status"] == "promoted"


def test_detail_reports_target_scope_and_registry_body(acc, db_url):
    _publish(db_url, "acc-backup", V1, scope="project:acc-proj")
    cid = _candidate(acc, body=V2, base=body_hash(V1))
    d = _proposal_detail(db_url, cid)
    assert d["apply_to"] == {"skill": "acc-backup", "scope": "project:acc-proj", "exists": True}
    assert d["registry_body"] == V1 and d["proposal_body"] == V2
    assert d["base_body_hash"] == body_hash(V1) and d["apply_issues"] == []


# ------------------------------------------------------------------ the routing check
# Once a retune is applied, a model is asked where the phrasings the skill missed route now.
# These run the real _routing_eval, on the connection _proposal_act hands it; only the
# model is a stand-in.


class _Model:
    """Stands in for the LLM client: keeps each prompt, answers with `reply` (text, or a
    function of the prompt)."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []
        self.messages = self

    def create(self, *, model, max_tokens, messages):
        self.prompts.append(messages[0]["content"])
        text = self.reply(self.prompts[-1]) if callable(self.reply) else self.reply
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


@pytest.fixture()
def model(monkeypatch):
    def _install(reply):
        fake = _Model(reply)
        monkeypatch.setattr("ingestion.llm_client.create_llm_client", lambda *a, **k: fake)
        return fake

    return _install


def _missed(*phrasings):
    return [
        {"session_id": f"acc-s{i}", "class": "judge", "signal": "under_trigger", "phrasing": p}
        for i, p in enumerate(phrasings, 1)
    ]


def _retune_with_misses(acc, db_url, *phrasings, **candidate):
    _publish(db_url, "acc-backup", V1, description="Back up the demo database.")
    candidate = {"body": RETUNED, "base": body_hash(V1), **candidate}
    return _candidate(acc, evidence=_missed(*phrasings), **candidate)


def test_a_retune_accept_asks_where_the_missed_phrasings_route(acc, db_url, model):
    _publish(db_url, "acc-restore", "# acc-restore\n", description="Restore the demo database.")
    _publish(db_url, "acc-gone", "# acc-gone\n", description="A retired skill.")
    acc.execute("UPDATE skills_lane.skill_registry SET status='retired' WHERE name='acc-gone'")
    cid = _retune_with_misses(acc, db_url, "dump the db for me", "save a copy of the database")
    fake = model('Here it is:\n{"matches": ["acc-backup", "none"]}')

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert r["status"] == "promoted"
    assert r["routing_eval"] == (
        "routing-eval: 1/2 missed phrasings map to 'acc-backup' after the retune"
    )
    (prompt,) = fake.prompts
    # the catalog is the active skills, each with the description it has after the retune
    assert "- acc-backup: Back up, dump or copy the demo database.\n" in prompt
    assert "Back up the demo database." not in prompt
    assert "- acc-restore: Restore the demo database.\n" in prompt
    assert "acc-gone" not in prompt
    assert "- name: description" not in prompt
    assert "- dump the db for me\n- save a copy of the database\n" in prompt
    # and the model is not told which skill is hoped for: its catalog line is the one mention
    assert prompt.count("acc-backup") == 1


def test_the_routing_check_reads_the_description_the_reviewer_supplied(acc, db_url, model):
    cid = _retune_with_misses(acc, db_url, "dump the db for me")
    edited = _skill("acc-backup", "Dump the demo database to a file.", "Dump the database.")
    fake = model('{"matches": ["acc-backup"]}')

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval, body=edited)

    assert r["status"] == "promoted"
    (prompt,) = fake.prompts
    assert "- acc-backup: Dump the demo database to a file.\n" in prompt
    assert "Back up, dump or copy" not in prompt


def test_the_routing_check_asks_about_the_skill_the_retune_was_written_to(acc, db_url, model):
    cid = _retune_with_misses(acc, db_url, "dump the db for me", name="acc-backup-widen")
    model('{"matches": ["acc-backup"]}')

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert (r["status"], r["skill"]) == ("promoted", "acc-backup")
    assert r["routing_eval"] == (
        "routing-eval: 1/1 missed phrasings map to 'acc-backup' after the retune"
    )


def test_a_refused_retune_asks_nothing(acc, db_url, model):
    cid = _retune_with_misses(acc, db_url, "dump the db for me", base=body_hash(V2))  # stale
    fake = model('{"matches": ["acc-backup"]}')

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert (r["status"], r["reason"]) == ("refused", "stale")
    assert "routing_eval" not in r and fake.prompts == []
    assert _registry(acc, "acc-backup")[0] == V1


def test_a_retune_with_no_missed_phrasings_asks_nothing(acc, db_url, model):
    _publish(db_url, "acc-backup", V1)
    cid = _candidate(acc, body=V2, base=body_hash(V1))  # judge evidence, no under-trigger
    fake = model('{"matches": []}')

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert r["status"] == "promoted"
    assert r["routing_eval"] == "routing-eval: no phrasings to eval"
    assert fake.prompts == []


@pytest.mark.parametrize(
    ("reply", "note"),
    [
        ('{"matches": ["acc-backup"]}', "routing-eval skipped: 1 answers for 2 phrasings"),
        ("I cannot say.", "routing-eval skipped: the answer is not JSON"),
    ],
)
def test_an_answer_that_does_not_fit_the_phrasings_is_not_counted(acc, db_url, model, reply, note):
    cid = _retune_with_misses(acc, db_url, "dump the db for me", "save a copy of the database")
    model(reply)

    r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert r["status"] == "promoted" and r["routing_eval"] == note


def test_a_failing_model_does_not_stop_the_accept(acc, db_url, model, caplog):
    cid = _retune_with_misses(acc, db_url, "dump the db for me")

    def _boom(_prompt):
        raise RuntimeError("no login")

    model(_boom)
    with caplog.at_level("WARNING", logger="mcp_server.skill_sync_routes"):
        r = _proposal_act(db_url, cid, "accept", None, _routing_eval)

    assert r["status"] == "promoted"
    assert r["routing_eval"] == "routing-eval skipped: RuntimeError: no login"
    assert f"routing-eval for skill candidate {cid} failed" in caplog.text


def test_a_slow_model_is_not_waited_for(acc, db_url, model, monkeypatch):
    cid = _retune_with_misses(acc, db_url, "dump the db for me")
    release = threading.Event()

    def _slow(_prompt):
        release.wait(30)
        return '{"matches": ["acc-backup"]}'

    model(_slow)
    monkeypatch.setattr(SSR, "_ROUTING_EVAL_BUDGET_S", 0.2)
    started = time.monotonic()
    try:
        r = _proposal_act(db_url, cid, "accept", None, _routing_eval)
        waited = time.monotonic() - started
    finally:
        release.set()

    assert r["status"] == "promoted"
    assert r["routing_eval"] == "routing-eval skipped: no answer within 0.2s"
    assert waited < 10  # the accept came back while the model was still thinking


def test_no_transaction_is_open_while_the_model_is_asked(acc, db_url, model):
    cid = _retune_with_misses(acc, db_url, "dump the db for me")
    seen = []
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        c.execute("SELECT 1").fetchone()  # a read is open on the connection it is handed

        def _answer(_prompt):
            seen.append(c.info.transaction_status)
            return '{"matches": ["acc-backup"]}'

        model(_answer)
        note = _routing_eval(c, cid, "acc-backup")
        c.commit()

    assert note == "routing-eval: 1/1 missed phrasings map to 'acc-backup' after the retune"
    assert seen == [TransactionStatus.IDLE]


# ------------------------------------------------------------------ sync delivers it


def _sync_engine(monkeypatch):
    monkeypatch.syspath_prepend(str(_SCRIPTS))  # the engine imports the plugin's config
    spec = importlib.util.spec_from_file_location("acc_skills_sync", _SCRIPTS / "skills_sync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RouteClient:
    """The plugin sync's HTTP client, answered by the real route functions + JSON round-trip."""

    def __init__(self, db_url):
        self.db_url = db_url

    def post_json(self, path, payload, timeout=30.0):
        route = {
            "/skills/list": lambda p: _list_skills(self.db_url, p["scope"]),
            "/skills/fetch": lambda p: _fetch_skill(self.db_url, p["name"]),
            "/skills/publish": lambda p: _publish_skill(self.db_url, p),
            "/skills/overwrite": lambda p: _record_overwrite(self.db_url, p),
        }[path]
        return json.loads(json.dumps(route(payload)))


@pytest.mark.parametrize("scope", ["global", "project:acc-proj"])
def test_accepted_change_is_pulled_by_the_owning_machine(acc, db_url, tmp_path, monkeypatch, scope):
    engine, client = _sync_engine(monkeypatch), _RouteClient(db_url)
    _publish(db_url, "acc-backup", V1, scope=scope)
    folder = tmp_path / "skills"
    engine._sync(scope, folder, client=client)  # the owner's copy, in sync with the registry
    md = folder / "acc-backup" / "SKILL.md"
    assert md.read_text() == V1
    assert engine._sync(scope, folder, client=client) == (0, 0, 0)  # steady state

    cid = _candidate(acc, body=V2, base=body_hash(V1))
    assert _accept(db_url, cid)["status"] == "promoted"

    assert engine._sync(scope, folder, client=client) == (1, 0, 0)  # next session start: a pull
    assert md.read_text() == V2
    cmod = _registry(acc, "acc-backup")[3]
    assert abs(md.stat().st_mtime - cmod.timestamp()) < 1  # disk clock aligned to the edit
    assert engine._sync(scope, folder, client=client) == (0, 0, 0)  # converged, no push-back


def test_accepted_new_skill_is_materialized_in_its_project(acc, db_url, tmp_path, monkeypatch):
    engine, client = _sync_engine(monkeypatch), _RouteClient(db_url)
    _episode(acc, "acc-d1", "acc-proj", "/work/acc-proj")
    cid = _candidate(
        acc, kind="derive", name="acc-new", direction=None, target=(), body=_NEW,
        evidence=[{"session_id": "acc-d1", "class": "judge"}],
    )  # fmt: skip
    assert _accept(db_url, cid)["scope"] == "project:acc-proj"
    folder = tmp_path / "acc-proj" / ".claude" / "skills"
    pulled, pushed, _removed = engine._sync("project:acc-proj", folder, client=client)
    assert pulled >= 1 and pushed == 0
    assert (folder / "acc-new-skill" / "SKILL.md").read_text() == _NEW


# ------------------------------------------------------------------ nightly draft pass


def test_draft_pending_drafts_bodyless_rows_and_redrafts_stale_ones(acc, db_url, monkeypatch):
    _publish(db_url, "acc-backup", V1)
    ev = [
        {
            "session_id": "acc-s1",
            "class": "judge",
            "signal": "under_trigger",
            "missing_phrasing": "snapshot the db",
        }
    ]
    widen = _candidate(acc, direction="widen", body=None, evidence=ev)
    fix = _candidate(acc, direction="fix", body=V1 + "old draft\n", base="0" * 64)  # stale
    derive = _candidate(acc, kind="derive", name="acc-new", direction=None, target=(), body=None)
    merge = _candidate(
        acc, kind="consolidate", name="acc-a+acc-b", direction=None, target=("acc-a", "acc-b")
    )

    def fake_llm(prompt):
        if '{"description"' in prompt:
            return json.dumps({"description": "Back up or snapshot the demo database."})
        if "CURRENT SKILL.md" in prompt:
            return V2
        return _NEW

    monkeypatch.setattr(SD, "_draft_call", fake_llm)

    stats = DR.draft_pending(acc)

    assert stats["drafted"] >= 2 and stats["redrafted"] >= 1 and stats["errors"] == 0
    row = lambda cid: acc.execute(  # noqa: E731
        "SELECT proposal_body, base_body_hash FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (cid,),
    ).fetchone()
    body, base = row(widen)
    assert base == body_hash(V1)
    assert body == V1.replace(
        "description: Back up the demo database.",
        "description: Back up or snapshot the demo database.",
    )
    assert row(fix) == (V2, body_hash(V1))
    assert row(derive) == (_NEW, None)
    assert row(merge) == (None, None)
    # drafted and current: a second pass leaves them alone
    monkeypatch.setattr(SD, "_draft_call", lambda prompt: pytest.fail("re-drafted"))
    ours = {widen, fix, derive}
    monkeypatch.setattr(DR, "needs_draft", lambda r, _f=DR.needs_draft: r["id"] in ours and _f(r))
    assert DR.draft_pending(acc)["pending"] == 0
