"""dream/skills/procedure_miner.py — weekly deterministic procedure detector.

Pure-Python: the SQL seam (_fetch_rows), registry seam (_load_registry), LLM seam
(_confirm_llm) and the ledger writer (skill_ledger.merge_candidate, pinned v2 contract)
are stubbed. All transcript text is synthetic.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from dream.skills import procedure_miner as pm

TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- helpers
def norm(cmd: str) -> list[str]:
    return pm.normalize_command(cmd)


def fps_only(content: str) -> list[str]:
    return [fp for fp, _ in pm.content_fingerprints(content)]


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))

    def fetchone(self):
        return self._conn.fetchone_result

    def fetchall(self):
        return []


class FakeConn:
    def __init__(self, fetchone_result=None):
        self.executed = []
        self.commits = 0
        self.fetchone_result = fetchone_result

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def episode(sid, seq, content, d=TODAY):
    return (sid, seq, d, content)


CONFIRM_OK = json.dumps(
    {
        "procedure": True,
        "name": "service-health-sweep",
        "summary": "1. ssh nas 'docker ps'\n2. run health-report.sh",
        "varies": "which host is checked first",
        "disposition": "skill",
        "covered_by": None,
        "extends": None,
    }
)

SWEEP_SIGNATURE = "script:health-report.sh ssh:nas>docker:ps"  # sorted fingerprint set


def wire(monkeypatch, rows, registry=None, confirm_raw=CONFIRM_OK):
    """Stub every external seam; return the merge_candidate call recorder."""
    calls = []

    def fake_merge(conn, kind, name, evidence_entries, **kw):
        calls.append({"kind": kind, "name": name, "evidence": evidence_entries, **kw})
        return {"id": len(calls), "status": "observe", "score": 0.5, "merged": False}

    monkeypatch.setattr(pm, "_fetch_rows", lambda conn, wd: rows)
    monkeypatch.setattr(pm, "_load_registry", lambda conn: dict(registry or {}))
    monkeypatch.setattr(pm, "_confirm_llm", lambda prompt, model: confirm_raw)
    monkeypatch.setattr(pm.skill_ledger, "merge_candidate", fake_merge)
    return calls


def health_sweep_rows(n_sessions, dup_chunk_for=()):
    """Synthetic corpus: each session runs the same 2-step procedure. Sessions listed in
    dup_chunk_for also re-serve the same commands via an overlapping chunk row."""
    content = (
        '[tool:Bash] ssh nas "docker ps"\n[result] ok\n[tool:Bash] bash ~/scripts/health-report.sh'
    )
    rows = [episode(f"s{i}", 3, content) for i in range(n_sessions)]
    rows += [episode(sid, 2, content) for sid in dup_chunk_for]  # chunk overlapping seq 2-4
    return rows


# ---------------------------------------------------------------------- normalization
class TestNormalization:
    def test_ssh_target_plus_remote_head(self):
        assert norm('ssh gateway "echo ok"') == ["ssh:gateway>echo"]

    def test_ssh_recurses_into_remote_docker_exec(self):
        assert norm('ssh nas "docker exec -i app-postgres-1 psql -U app -d app"') == [
            "ssh:nas>docker-exec:app-postgres-1:psql"
        ]

    def test_ssh_remote_docker_verb(self):
        assert norm('ssh nas "docker ps"') == ["ssh:nas>docker:ps"]

    def test_docker_exec_keeps_subcommand_drops_flags(self):
        assert norm("docker exec app-web python -m job") == ["docker-exec:app-web:python"]
        assert norm("docker exec -it app-web sh -c 'ls'") == ["docker-exec:app-web:sh"]
        assert norm("docker exec app-web supervisorctl restart worker") == [
            "docker-exec:app-web:supervisorctl restart"
        ]

    def test_docker_verb_and_target(self):
        assert norm("docker logs app-worker --tail 25") == ["docker:logs:app-worker"]
        assert norm("docker compose up -d") == ["docker:compose up"]

    def test_docker_target_stops_at_shell_operator(self):
        assert norm("docker compose pull -q") == ["docker:compose pull"]

    def test_script_basename(self):
        assert norm('bash ~/scripts/notify-send.sh infra "done"') == ["script:notify-send.sh"]
        assert norm("/home/alice/scripts/dbdump.py --full") == ["script:dbdump.py"]
        assert norm("$HOME/scripts/rotate-logs.sh") == ["script:rotate-logs.sh"]

    def test_git_subcommand(self):
        assert norm("git status") == ["git:status"]
        assert norm("git -C /tmp/repo log --oneline -5") == ["git:log"]
        assert norm("git frobnicate") == []  # unknown subcommand: no fingerprint

    def test_gh_subcommand(self):
        assert norm("gh pr create --title x") == ["bin:gh:pr"]
        assert norm("gh run list --limit 5") == ["bin:gh:run"]

    def test_himalaya_subcommand(self):
        assert norm("himalaya envelope list -a personal --folder INBOX") == ["himalaya:envelope"]

    def test_systemctl_verb_and_unit(self):
        assert norm("systemctl --user restart chat-bot") == ["systemctl:restart:chat-bot"]
        assert norm("sudo systemctl status nginx") == ["systemctl:status:nginx"]

    def test_journalctl_collapses_to_one_fingerprint(self):
        assert norm('journalctl --user -u chat-bot --since "30 min ago"') == ["journalctl"]
        assert norm("sudo journalctl -xe") == ["journalctl"]

    def test_crontab(self):
        assert norm("crontab -l") == ["crontab:-l"]

    def test_venv_use(self):
        assert norm("source ~/venv/bin/activate") == ["venv:use"]
        assert norm("~/venv/bin/python -m pytest") == ["venv:use"]

    def test_runtime_arg_basename(self):
        assert norm("python3 -c 'print(1)'") == ["python3:-c"]
        assert norm("python3 /some/dir/train.py --epochs 3") == ["python3:train.py"]

    def test_curl_url_normalized(self):
        assert norm("curl -sf https://api.example.com/v1/runs/42815?full=1") == [
            "curl:https://api.example.com/v1/runs/N"
        ]

    def test_psql_direct_and_piped(self):
        assert norm("psql -U app -d app -c 'select 1'") == ["psql:direct"]
        assert norm("cat /tmp/q.sql | psql -U app") == ["psql:direct"]

    def test_compound_atoms_split(self):
        text = 'git add -A && git commit -m "x"'
        assert [fp for atom in pm.split_atoms(text) for fp in norm(atom)] == [
            "git:add",
            "git:commit",
        ]

    def test_ssh_line_not_split_on_operators(self):
        assert pm.split_atoms('ssh nas "docker compose pull -q && docker compose up -d"') == [
            'ssh nas "docker compose pull -q && docker compose up -d"'
        ]

    def test_mcp_tool_name(self):
        content = "[tool:mcp__acme_tools__run_query] {'q': 'x'}"
        assert fps_only(content) == ["tool:mcp:acme_tools:run_query"]

    def test_marker_capture_stops_at_next_bracket_line(self):
        content = "[tool:Bash] git status\ngit log\n[result] clean tree"
        assert fps_only(content) == ["git:status", "git:log"]

    def test_unknown_commands_yield_nothing(self):
        assert norm("ls -la") == []
        assert norm("cd /tmp") == []


# ------------------------------------------------------------------------- exclusions
class TestExclusions:
    def test_mcp_native_excluded(self):
        assert not pm.eligible("tool:mcp:acme_tools:recall")
        assert not pm.eligible("tool:Agent")

    def test_generic_dev_workflow_excluded(self):
        for fp in ("git:status", "git:push", "bin:gh:pr", "venv:use", "python3:-c", "crontab:-l"):
            assert not pm.eligible(fp), fp

    def test_real_procedure_fingerprints_eligible(self):
        for fp in (
            "ssh:nas>docker:ps",
            "script:health-report.sh",
            "docker-exec:app-postgres-1:psql",
            "himalaya:envelope",
            "systemctl:restart:chat-bot",
            "journalctl",
        ):
            assert pm.eligible(fp), fp


# ---------------------------------------------------------------- union + clustering
class TestUnionAndClustering:
    def test_chunk_episode_union_dedupes_sessions(self):
        rows = health_sweep_rows(2, dup_chunk_for=("s0", "s1"))
        sess_fp_pos, _, _ = pm._occurrences(rows)
        fp_sessions = pm._fp_sessions(sess_fp_pos)
        assert fp_sessions["ssh:nas>docker:ps"] == {"s0", "s1"}  # 4 rows, 2 sessions

    def test_cluster_forms_at_floor(self):
        sess_fp_pos, _, _ = pm._occurrences(health_sweep_rows(4))
        clusters = pm._cluster(pm._pairs(sess_fp_pos), pm._fp_sessions(sess_fp_pos), 4)
        assert len(clusters) == 1
        assert set(clusters[0]["steps"]) == {"ssh:nas>docker:ps", "script:health-report.sh"}
        assert len(clusters[0]["sessions"]) == 4

    def test_below_session_floor_dropped(self):
        sess_fp_pos, _, _ = pm._occurrences(health_sweep_rows(3))
        assert pm._cluster(pm._pairs(sess_fp_pos), pm._fp_sessions(sess_fp_pos), 4) == []

    def test_far_apart_fingerprints_do_not_pair(self):
        rows = []
        for i in range(5):
            rows.append(episode(f"s{i}", 1, '[tool:Bash] ssh nas "docker ps"'))
            rows.append(episode(f"s{i}", 60, "[tool:Bash] bash ~/scripts/health-report.sh"))
        sess_fp_pos, _, _ = pm._occurrences(rows)
        assert pm._pairs(sess_fp_pos) == {}

    def test_single_fingerprint_never_clusters(self):
        rows = [
            episode(f"s{i}", 1, "[tool:Bash] bash ~/scripts/notify-send.sh hi") for i in range(10)
        ]
        sess_fp_pos, _, _ = pm._occurrences(rows)
        assert pm._cluster(pm._pairs(sess_fp_pos), pm._fp_sessions(sess_fp_pos), 4) == []

    def test_generic_fingerprints_do_not_join_clusters(self):
        content = "[tool:Bash] git status\n[tool:Bash] bash ~/scripts/notify-send.sh hi"
        rows = [episode(f"s{i}", 1, content) for i in range(6)]
        sess_fp_pos, _, _ = pm._occurrences(rows)
        assert pm._pairs(sess_fp_pos) == {}  # git:status ineligible, script alone can't pair


# ------------------------------------------------------------------ registry coverage
class TestRegistryCoverage:
    def test_covered_cluster_detected(self):
        registry = {"flutter-tester": "Use when creating or reviewing tests in a Flutter project"}
        assert (
            pm._covered_by_registry(["flutter:analyze", "flutter:test"], registry)
            == "flutter-tester"
        )

    def test_uncovered_cluster_passes(self):
        registry = {"flutter-tester": "Use when creating or reviewing tests in a Flutter project"}
        steps = ["ssh:nas>docker-exec:app-postgres-1:psql", "psql:direct"]
        assert pm._covered_by_registry(steps, registry) is None

    def test_empty_registry(self):
        assert pm._covered_by_registry(["script:notify-send.sh", "journalctl"], {}) is None


# --------------------------------------------------------------------------- run()
class TestRun:
    def test_emits_derive_candidate_with_v2_contract_fields(self, monkeypatch):
        calls = wire(monkeypatch, health_sweep_rows(5))
        conn = FakeConn()
        stats = pm.run(conn, window_days=35, min_sessions=4, model=None)

        assert stats["derived"] == 1 and stats["llm_calls"] == 1
        (call,) = calls
        assert call["kind"] == "derive"
        assert call["name"] == "service-health-sweep"
        assert call["source_detector"] == "procedure_miner"
        assert call["salience"] == 3  # 5 sessions < 8
        # stable identity: signature = sorted fingerprint set, tools = the fingerprint list
        assert call["signature"] == SWEEP_SIGNATURE
        assert call["tools"] == ["script:health-report.sh", "ssh:nas>docker:ps"]
        assert "health-report" in call["summary"] or "1." in call["summary"]
        assert not call["summary"].startswith("[script-candidate]")
        ev = call["evidence"]
        assert len(ev) == 5  # one per distinct session
        for e in ev:
            assert e["class"] == "judge"
            assert e["signal"] == "procedure_freq"
            assert e["scan_night"] == TODAY
            assert e["date"] == TODAY
            assert "ssh:nas>docker:ps" in e["quote"]  # the fingerprint IS the quote

    def test_salience_4_at_eight_sessions(self, monkeypatch):
        calls = wire(monkeypatch, health_sweep_rows(8))
        pm.run(FakeConn(), min_sessions=4)
        assert calls[0]["salience"] == 4

    def test_evidence_capped_at_ten_sessions(self, monkeypatch):
        calls = wire(monkeypatch, health_sweep_rows(13))
        pm.run(FakeConn(), min_sessions=4)
        assert len(calls[0]["evidence"]) == pm.EVIDENCE_SESSION_CAP

    def test_registry_prescreen_skips_without_llm(self, monkeypatch):
        registry = {"nas-health": "ssh nas docker ps health report sweep for the lab"}
        calls = wire(monkeypatch, health_sweep_rows(5), registry=registry)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["covered"] == 1 and stats["llm_calls"] == 0
        assert calls == []

    def test_llm_covered_by_skips_candidate(self, monkeypatch):
        registry = {"ops-check": "Daily operations checklist for the homelab"}
        raw = json.dumps(
            {"procedure": True, "name": "x", "summary": "s", "covered_by": "ops-check"}
        )
        calls = wire(monkeypatch, health_sweep_rows(5), registry=registry, confirm_raw=raw)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["covered"] == 1 and stats["llm_calls"] == 1
        assert calls == []

    def test_extends_routes_to_retune(self, monkeypatch):
        registry = {"deploy-verify": "Deploy the stack and verify it came up healthy"}
        raw = json.dumps(
            {
                "procedure": True,
                "name": "deploy-verify-extra",
                "summary": "1. pull\n2. up\n3. health poll",
                "extends": "deploy-verify",
            }
        )
        calls = wire(monkeypatch, health_sweep_rows(5), registry=registry, confirm_raw=raw)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["retuned"] == 1 and stats["derived"] == 0
        (call,) = calls
        assert call["kind"] == "retune"
        assert call["name"] == "deploy-verify"
        assert call["direction"] == "extend"
        assert call["target_skills"] == ["deploy-verify"]
        assert call["do_embed"] is False
        assert call["signature"] == SWEEP_SIGNATURE
        assert call["evidence"][0]["signal"] == "procedure_freq"

    def test_script_disposition_flags_summary_and_stats(self, monkeypatch):
        raw = json.dumps(
            {
                "procedure": True,
                "name": "morning-status-readout",
                "summary": "1. ssh nas 'docker ps'\n2. run health-report.sh",
                "varies": "nothing",
                "disposition": "script",
            }
        )
        calls = wire(monkeypatch, health_sweep_rows(5), confirm_raw=raw)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["script_candidates"] == 1 and stats["derived"] == 1
        assert calls[0]["summary"].startswith("[script-candidate] ")
        assert calls[0]["signature"] == SWEEP_SIGNATURE  # still a normal derive otherwise

    def test_missing_disposition_defaults_to_skill(self, monkeypatch):
        raw = json.dumps({"procedure": True, "name": "x-y", "summary": "1. a\n2. b"})
        calls = wire(monkeypatch, health_sweep_rows(5), confirm_raw=raw)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["script_candidates"] == 0
        assert not calls[0]["summary"].startswith("[script-candidate]")

    def test_name_collision_with_registry_becomes_retune(self, monkeypatch):
        registry = {"service-health-sweep": "Existing skill with this exact name"}
        calls = wire(monkeypatch, health_sweep_rows(5), registry=registry)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["retuned"] == 1
        assert calls[0]["kind"] == "retune"

    def test_llm_rejection_emits_nothing(self, monkeypatch):
        raw = json.dumps({"procedure": False, "name": "", "summary": ""})
        calls = wire(monkeypatch, health_sweep_rows(5), confirm_raw=raw)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["rejected"] == 1
        assert calls == []

    def test_llm_budget_capped(self, monkeypatch):
        rows = []
        for i in range(10):  # 10 distinct 2-step procedures, far apart, in 4 sessions each
            content = (
                f"[tool:Bash] bash ~/scripts/proc{i}a.sh\n[tool:Bash] bash ~/scripts/proc{i}b.sh"
            )
            rows += [episode(f"s{j}", i * 100, content) for j in range(4)]
        calls = wire(monkeypatch, rows)
        stats = pm.run(FakeConn(), min_sessions=4)
        assert stats["clusters"] == 10
        assert stats["llm_calls"] == pm.MAX_CONFIRM
        assert len(calls) == pm.MAX_CONFIRM

    def test_rerun_produces_identical_evidence(self, monkeypatch):
        """Idempotency: same week, same corpus -> byte-identical evidence entries, so the
        ledger's (session_id, signal, class, scan_night) dedup collapses the re-run."""
        calls = wire(monkeypatch, health_sweep_rows(6))
        pm.run(FakeConn(), min_sessions=4)
        pm.run(FakeConn(), min_sessions=4)
        assert len(calls) == 2
        assert calls[0]["evidence"] == calls[1]["evidence"]
        assert calls[0]["name"] == calls[1]["name"]

    def test_watermark_stamped_via_config_merge(self, monkeypatch):
        wire(monkeypatch, health_sweep_rows(5))
        conn = FakeConn()
        pm.run(conn, min_sessions=4)
        stamps = [
            (sql, params)
            for sql, params in conn.executed
            if "skill_scan_cursor" in sql and "procedure_miner" in sql
        ]
        assert len(stamps) == 1
        sql, params = stamps[0]
        assert "||" in sql and "last_scan_at" not in sql  # merge config, never the scan watermark
        assert params == (TODAY,)
        assert conn.commits == 1


# ----------------------------------------------------------------- cadence helpers
class TestCadence:
    def test_due_when_never_ran(self):
        assert pm.due(FakeConn(fetchone_result=(None,))) is True
        assert pm.due(FakeConn(fetchone_result=None)) is True

    def test_not_due_within_week(self):
        recent = (date.today() - timedelta(days=3)).isoformat()
        assert pm.due(FakeConn(fetchone_result=(recent,))) is False

    def test_due_after_week(self):
        old = (date.today() - timedelta(days=8)).isoformat()
        assert pm.due(FakeConn(fetchone_result=(old,))) is True

    def test_run_signature_is_pinned(self):
        import inspect

        sig = inspect.signature(pm.run)
        params = list(sig.parameters.values())
        assert [p.name for p in params] == ["conn", "window_days", "min_sessions", "model"]
        assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:])
        assert params[1].default == 35 and params[2].default == 4 and params[3].default is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
